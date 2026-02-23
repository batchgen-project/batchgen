# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 specific wrappers for BatchGen execution.

Standalone wrappers for GLM-5 FP8 model — no cross-model imports.
- GLM5ExpertWrapper: Expert wrapper with FP8 dequantization
- GLM5AttnWrapper: Attention wrapper with FP8 dequant + DSA integration

Key differences from DeepSeek:
- kv_a_proj_with_mqa naming (same)
- DSA indexer integration in attention wrapper
- Different MLA dimensions (qk_nope=192, v_head=256)
- rope_interleave=True
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase


def glm5_fp8_dequantization(
    weight_data_fp8: torch.Tensor,
    weight_scale_inv_fp32: torch.Tensor,
    block_size=(128, 128),
) -> torch.Tensor:
    """Blockwise FP8 dequantization (standalone, no cross-model import)."""
    rows, cols = weight_data_fp8.shape
    block_rows, block_cols = block_size
    n_block_rows = rows // block_rows
    n_block_cols = cols // block_cols
    weight_4d = weight_data_fp8.reshape(
        n_block_rows, block_rows, n_block_cols, block_cols
    ).to(torch.float32)
    scale_4d = weight_scale_inv_fp32.unsqueeze(1).unsqueeze(-1)
    dequantized_4d = weight_4d * scale_4d
    return dequantized_4d.reshape(rows, cols).to(torch.bfloat16)


class GLM5ExpertWrapper(ExpertWrapperBase):
    """Expert wrapper for GLM-5 models.

    Supports both variants:
    - GLM-5 (BF16): standard nn.Linear forward
    - GLM-5-FP8: FP8 deepgemm forward with w8a16_gemm
    Controlled by `is_fp8` flag set during PSM configuration.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
        is_fp8: bool = False,
    ):
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}
        self.is_fp8 = is_fp8
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not self.is_fp8:
            return weights_dict
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = glm5_fp8_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def _register_fp8_weights(self):
        self.cached_gate = self.module.gate_proj.weight.data
        self.cached_up = self.module.up_proj.weight.data
        self.cached_down = self.module.down_proj.weight.data

    def _unregister_fp8_weights(self):
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.is_fp8:
            return self.module.deepgemm_forward(hidden_states, self.weight_dequant_scale)
        return self.module(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.persistent:
            weights = self.load_weights(self.module_key)
            for name, param in self.module.named_parameters():
                param.data = weights[name]
        else:
            self.module.gate_proj.weight.data = self.cached_gate
            self.module.up_proj.weight.data = self.cached_up
            self.module.down_proj.weight.data = self.cached_down

        result = self.micro_batch_forward(hidden_states, "expert")

        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        return result


class GLM5AttnWrapper(AttnWrapperBase):
    """Attention wrapper with FP8 dequant + DSA for GLM-5.

    Key differences from DeepSeek:
    - kv_a_proj_with_mqa (same naming)
    - DSA indexer integration: prefill populates auxiliary cache,
      decode uses indexer scoring for sparse attention
    - MLA dims: qk_nope=192, v_head=256, q_lora_rank=2048
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )
        # FP8 weight caching for GLM-5 MLA
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None

    def _register_fp8_weights(self):
        """Cache FP8 attention weights. GLM-5 uses kv_a_proj_with_mqa."""
        self.fp8_q_a_proj = self.module.q_a_proj.weight.data
        self.fp8_q_b_proj = self.module.q_b_proj.weight.data
        self.fp8_kv_a_proj = self.module.kv_a_proj_with_mqa.weight.data
        self.fp8_kv_b_proj = self.module.kv_b_proj.weight.data
        self.fp8_o_proj = self.module.o_proj.weight.data

    def _unregister_fp8_weights(self):
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return FP8 weights unchanged — deepgemm handles FP8 directly."""
        return weights_dict

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward with DSA auxiliary cache population.

        1. Standard MLA prefill via FA3 (full attention)
        2. Compute indexer K and write to auxiliary cache
        """
        if self.prepack_mode:
            hidden_states_2d = hidden_states.squeeze(0)
            attn_output, offload_kv = self.module.prefill_attn_w8a16_prepacked(
                hidden_states_2d,
                self.position_ids.to(hidden_states_2d.device),
                self.prepack_cu_seqlens.to(hidden_states_2d.device),
                self.prepack_max_seqlen,
                self.prepack_num_sequences,
                self.weight_dequant_scale
            )

            # DSA: compute indexer K and offload to auxiliary cache
            gpu_paged_kv_manager_aux = AttnWrapperBase.gpu_paged_kv_manager_aux
            if gpu_paged_kv_manager_aux is not None and hasattr(self.module, 'indexer'):
                indexer_kv = self.module.indexer.compute_indexer_kv(
                    hidden_states_2d.unsqueeze(0),
                    positions=self.position_ids.to(hidden_states_2d.device),
                )
                # indexer_kv: [1, total_tokens, 1, index_dim]
                self._offload_prepacked_indexer_kv(indexer_kv.squeeze(0))

            self._offload_prepacked_kv(offload_kv)
            attn_output = attn_output.unsqueeze(0)
            return (attn_output, None, None)
        else:
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)
            attn_output, offload_kv = self.module.prefill_attn_w8a16(
                hidden_states, attention_mask, position_ids,
                self.weight_dequant_scale
            )
            return (attn_output, None, offload_kv)

    def _offload_prepacked_indexer_kv(self, offload_kv: torch.Tensor):
        """Offload indexer KV cache per-sequence to auxiliary host memory."""
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx
            seq_kv = offload_kv[start_idx:end_idx].unsqueeze(0).unsqueeze(2)
            seq_global_id = [global_sequence_ids[seq_idx]]
            AttnWrapperBase.host_paged_kv_worker_view_aux.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload KV cache per-sequence to host memory."""
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx
            seq_kv = offload_kv[start_idx:end_idx].unsqueeze(0).unsqueeze(2)
            seq_global_id = [global_sequence_ids[seq_idx]]
            self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Decode forward with DSA sparse attention.

        For DSA models:
        1. Compute MLA compressed KV, write to primary cache
        2. Compute indexer K for new token, write to auxiliary cache
        3. Score all cached tokens, select top-K
        4. Gather MLA KV at top-K positions from primary cache
        5. Compute absorbed Q, run sparse FlashMLA
        6. out_absorb → o_proj

        Falls back to standard full-cache FlashMLA when DSA is not active.
        """
        past_key_states = AttnWrapperBase.past_key_states
        position_ids = AttnWrapperBase.position_ids
        cache_seqlens = AttnWrapperBase.cache_seqlens
        max_seqlen = AttnWrapperBase.max_seqlen
        gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        gpu_paged_kv_manager_aux = AttnWrapperBase.gpu_paged_kv_manager_aux

        if gpu_paged_kv_manager is not None:
            dsa_active = (
                gpu_paged_kv_manager_aux is not None
                and hasattr(self.module, 'indexer')
            )

            if dsa_active:
                attn_output = self._forward_decode_dsa(
                    hidden_states, position_ids, cache_seqlens, max_seqlen,
                    gpu_paged_kv_manager, gpu_paged_kv_manager_aux,
                )
                return (attn_output, None, None)

            # Standard BF16 paged KV path (full-cache attention, no DSA)
            attn_output, k_tensor = self.module.decoding_attn_mode_3_bf16(
                hidden_states,
                position_ids,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale,
                gpu_paged_kv_manager,
                self.layer_idx,
                None
            )
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)
            return (attn_output, None, None)
        else:
            # FP8 KV cache with tensor references
            attention_mask = AttnWrapperBase.attention_mask
            scale = AttnWrapperBase.scale
            layer_past_key = past_key_states[self.layer_idx] if past_key_states else None
            layer_scale = scale[self.layer_idx] if scale else None
            attn_output, updated_past_key, updated_scale = self.module.decoding_attn_mode_3_fp8(
                hidden_states,
                layer_past_key,
                None,
                attention_mask,
                position_ids,
                layer_scale,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale
            )
            return (attn_output, updated_past_key, updated_scale)

    def _forward_decode_dsa(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen: int,
        gpu_paged_kv_manager,
        gpu_paged_kv_manager_aux,
    ) -> torch.Tensor:
        """DSA sparse attention decode path.

        Computes MLA KV and writes to primary cache first, then runs indexer
        scoring on aux cache, gathers sparse MLA KV, and runs sparse FlashMLA.
        """
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.attention.mla.fused_rmsnorm_rope import fused_rmsnorm_rope_with_q
        from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization
        from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
        from batchgen.attention.dsa.sparse_decode_mla import sparse_flash_mla_decode
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm

        weight_scale = self.weight_dequant_scale
        attn = self.module
        indexer = attn.indexer
        bsz = hidden_states.shape[0]

        if not hasattr(GLM5AttnWrapper, '_dsa_logged'):
            GLM5AttnWrapper._dsa_logged = True
            logging.info(
                f"[DSA] _forward_decode_dsa invoked: layer={self.layer_idx}, "
                f"bsz={bsz}, cache_seqlens={cache_seqlens.tolist()[:4]}, "
                f"index_topk={indexer.index_topk}, "
                f"index_dim={indexer.index_head_dim}"
            )

        # --- Shared FP8 activation quantization ---
        hidden_flat = hidden_states.squeeze(1)  # [batch, hidden_size]
        hidden_fp8, hidden_scale = act_quant(hidden_flat)

        # --- Q path: q_a_proj → layernorm → q_b_proj → split → RoPE ---
        q_a = w8a8_deepgemm(
            hidden_fp8, hidden_scale,
            attn.q_a_proj.weight, weight_scale["q_a_proj.weight_scale_inv"],
        )
        q_a_normed = attn.q_a_layernorm(q_a)
        q_a_fp8, q_a_scale = act_quant(q_a_normed)
        q = w8a8_deepgemm(
            q_a_fp8, q_a_scale,
            attn.q_b_proj.weight, weight_scale["q_b_proj.weight_scale_inv"],
        )
        q = q.view(bsz, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1
        )
        q_pe = q_pe.contiguous()

        # --- KV path: kv_a_proj → fused_rmsnorm_rope → compressed KV ---
        new_compressed_kv = w8a8_deepgemm(
            hidden_fp8, hidden_scale,
            attn.kv_a_proj_with_mqa.weight,
            weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
        ).view(bsz, 1, -1)

        cos, sin = attn.rotary_emb(q_pe, seq_len=max_seqlen)
        offload_kv = fused_rmsnorm_rope_with_q(
            new_compressed_kv, q_pe, cos, sin, position_ids,
            attn.kv_a_layernorm.weight,
            attn.kv_lora_rank, attn.qk_rope_head_dim,
        )

        # --- Step 1: Write new MLA KV to primary cache ---
        manager_device = gpu_paged_kv_manager.device
        k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1)).to(manager_device)
        seq_lengths_i32 = position_ids.squeeze(-1).to(dtype=torch.int32, device=manager_device)
        gpu_paged_kv_manager.update_layer_decode_new_token(
            k_tensor=k_tensor,
            v_tensor=None,
            sequence_lengths=seq_lengths_i32,
            layer_idx=self.layer_idx,
        )
        if AttnWrapperBase.kv_append_callback is not None:
            AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)

        # --- Step 2: Write indexer K to auxiliary cache ---
        # position_ids = cache_seqlens - 1 = 0-based position of the new token
        # Must match primary cache write position (line 400 uses position_ids)
        new_token_pos = position_ids.squeeze(-1)  # [batch]
        indexer_kv = indexer.compute_indexer_kv(hidden_states, positions=new_token_pos)
        indexer_k_tensor = indexer_kv  # [batch, 1, 1, index_dim]
        seq_lengths_i32_aux = new_token_pos.to(dtype=torch.int32, device=gpu_paged_kv_manager_aux.device)
        gpu_paged_kv_manager_aux.update_layer_decode_new_token(
            k_tensor=indexer_k_tensor,
            v_tensor=None,
            sequence_lengths=seq_lengths_i32_aux,
            layer_idx=self.layer_idx,
        )
        # Offload indexer K to auxiliary host cache
        if AttnWrapperBase.kv_append_callback_aux is not None:
            AttnWrapperBase.kv_append_callback_aux(self.layer_idx, indexer_k_tensor, None)

        # --- Step 3: Score all cached tokens (including new), select top-K ---
        # cache_seqlens already includes the new token (pre-incremented in worker)
        updated_seqlens = cache_seqlens

        if torch.all(updated_seqlens <= indexer.index_topk):
            # Short-circuit: all sequences fit within topk — use full range
            max_len = int(updated_seqlens.max())
            top_k_indices = torch.arange(
                max_len, device=hidden_states.device, dtype=torch.long,
            ).unsqueeze(0).expand(bsz, -1)
        else:
            # Full indexer scoring path
            q_a_for_indexer = q_a_normed.unsqueeze(1)  # [batch, 1, q_lora_rank]
            indexer_blocked_k, _, idx_block_table = \
                gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(self.layer_idx)
            aux_page_size = gpu_paged_kv_manager_aux.config.page_size_tokens
            top_k_indices = indexer.score_and_select_paged(
                q_a_for_indexer, hidden_states,
                indexer_blocked_k, idx_block_table,
                updated_seqlens, aux_page_size,
                positions=new_token_pos,
            )

        if not hasattr(GLM5AttnWrapper, '_dsa_topk_logged'):
            GLM5AttnWrapper._dsa_topk_logged = True
            logging.info(
                f"[DSA] Indexer top-K: layer={self.layer_idx}, "
                f"top_k_indices.shape={top_k_indices.shape}, "
                f"updated_seqlens={updated_seqlens.tolist()[:4]}, "
                f"sample indices[0]={top_k_indices[0, :10].tolist()}"
            )

        # --- Step 4: Sparse gather MLA KV at top-K positions ---
        mla_blocked_k, _, mla_block_table = \
            gpu_paged_kv_manager.get_layer_kv_with_page_table(self.layer_idx)
        mla_page_size = gpu_paged_kv_manager.config.page_size_tokens
        sparse_mla_kv = sparse_gather_from_paged_kv(
            mla_blocked_k, mla_block_table, top_k_indices, mla_page_size,
        )
        # sparse_mla_kv: [batch, topk, 1, 576]

        # --- Step 5: Absorbed Q → sparse FlashMLA ---
        kv_b_proj = deepseek_v3_dequantization(
            attn.kv_b_proj.weight.data,
            weight_scale["kv_b_proj.weight_scale_inv"],
        ).view(attn.num_heads, -1, attn.kv_lora_rank)
        q_absorb = kv_b_proj[:, :attn.qk_nope_head_dim, :]
        out_absorb = kv_b_proj[:, attn.qk_nope_head_dim:, :]

        qk_head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
        query_states = torch.empty(
            bsz, attn.num_heads, 1, qk_head_dim,
            dtype=sparse_mla_kv.dtype, device=sparse_mla_kv.device,
        )
        q_nope_squeezed = q_nope.squeeze(2)
        query_states[:, :, :, :attn.kv_lora_rank] = torch.einsum(
            "bhd,hdc->bhc", q_nope_squeezed, q_absorb,
        ).view(bsz, attn.num_heads, 1, attn.kv_lora_rank)
        query_states[:, :, :, attn.kv_lora_rank:] = q_pe
        query_states = query_states.view(bsz, 1, attn.num_heads, qk_head_dim)

        # Sparse seqlens: min(topk, actual cache length)
        topk = top_k_indices.shape[1]
        sparse_seqlens = torch.clamp(updated_seqlens, max=topk)

        attn_out = sparse_flash_mla_decode(
            query_states, sparse_mla_kv, sparse_seqlens,
            attn.num_heads, attn.softmax_scale,
            head_dim_v=attn.kv_lora_rank,
            page_size=mla_page_size,
        )

        # --- Step 6: out_absorb → o_proj ---
        attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, attn.num_heads * attn.v_head_dim)
        attn_output_fp8, attn_output_scale = act_quant(attn_output)
        attn_output = w8a8_deepgemm(
            attn_output_fp8, attn_output_scale,
            attn.o_proj.weight, weight_scale["o_proj.weight_scale_inv"],
        )
        attn_output = attn_output.view(bsz, 1, -1)
        return attn_output
