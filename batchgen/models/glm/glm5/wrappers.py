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
import os
from contextlib import nullcontext as _nullctx
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# DSA fused kernel enable flags — all default ON after L2 validation
# confirmed accuracy parity with the PyTorch fallback path (79.69% vs
# 79.69%) and throughput gains (prefill +83 %, decode +24 %). Flip any
# flag to 0 (e.g. BATCHGEN_GLM5_WP2=0) to force the PyTorch fallback
# for that work-package — useful for bisecting any future regression.
_GLM5_WP2_ENABLED = os.environ.get("BATCHGEN_GLM5_WP2", "1") == "1"
_GLM5_WP4_ENABLED = os.environ.get("BATCHGEN_GLM5_WP4", "1") == "1"
_GLM5_WP5_ENABLED = os.environ.get("BATCHGEN_GLM5_WP5", "1") == "1"
import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.timing import init_decode_timer

# Try importing FP8 absorb kernels (WP5)
try:
    from batchgen_kernels.attention.dsa.fp8_absorb import (
        FP8AbsorbWeights, fp8_q_absorb, fp8_out_absorb,
    )
    _HAS_FP8_ABSORB = True
except Exception as _e:
    _HAS_FP8_ABSORB = False
    logging.debug(f"[WP5] FP8 absorb import failed: {_e}")

# Try importing fused indexer KV proj (WP2)
try:
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        build_module as _build_indexer_module,
        FP8IndexerWeightsCUDA,
    )
    _HAS_FUSED_INDEXER_KV = True
except Exception as _e:
    _HAS_FUSED_INDEXER_KV = False
    logging.debug(f"[WP2] Fused indexer KV proj import failed: {_e}")

# Try importing fused scoring pipeline (WP4)
try:
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        FP8WqbWeightsCUDA,
        fused_score_pipeline,
    )
    _HAS_FUSED_SCORE = True
except Exception as _e:
    _HAS_FUSED_SCORE = False
    logging.debug(f"[WP4] Fused scoring import failed: {_e}")

# Initialize GLM-5 decode timer (activated by BATCHGEN_DECODE_TIMING=1)
_GLM5_ATTN_CATEGORIES = [
    "act_quant", "q_proj", "kv_proj", "kv_write",
    "indexer_k", "indexer_score", "sparse_gather",
    "q_absorb", "sparse_attn", "o_proj",
]
_GLM5_MOE_CATEGORIES = [
    "allgather", "routing", "dispatch", "grouped_gemm",
    "expert_loop", "scatter_reduce", "allreduce",
]
_glm5_decode_timer = init_decode_timer(
    "GLM-5", _GLM5_ATTN_CATEGORIES + _GLM5_MOE_CATEGORIES
)


# DSA decode-iter diagnostic. Gated by BATCHGEN_GLM5_DSA_DIAG=1.
# Logs which DSA branch (dense short-circuit ≤ index_topk vs sparse scoring
# > index_topk) fires on each decode iter at layer 0 — proves the sparse
# path engages once context exceeds 2048.
import os as _os_dsa_diag
_DSA_DIAG_SEEN: dict = {}
def _dsa_diag_log(branch: str, max_seqlen: int, batch_size: int) -> None:
    # Bucket max_seqlen by power of two so each shape logs ~once.
    key = (branch, max_seqlen.bit_length())
    if key in _DSA_DIAG_SEEN:
        return
    _DSA_DIAG_SEEN[key] = True
    logging.info(f"[DSA L0] branch={branch} max_seqlen={max_seqlen} batch={batch_size}")


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
        """Cache weight pointers for persistent experts.

        Supports both:
        - Placeholders with flat attrs (routed experts, set by _load_local_routed_experts)
        - Real Glm5Expert modules (shared experts, weights in gate_proj.weight.data)
        """
        if hasattr(self.module, 'fp8_gate'):
            # Placeholder with flat attrs
            self.cached_gate = self.module.fp8_gate
            self.cached_up = self.module.fp8_up
            self.cached_down = self.module.fp8_down
        else:
            # Real nn.Module (shared experts)
            self.cached_gate = self.module.gate_proj.weight.data
            self.cached_up = self.module.up_proj.weight.data
            self.cached_down = self.module.down_proj.weight.data

    def _unregister_fp8_weights(self):
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP8 forward using cached weight tensors directly (no nn.Module delegation)."""
        from batchgen.attention.mla.fa3_backend import w8a16_gemm
        gate = w8a16_gemm(
            self.cached_gate, self.weight_dequant_scale.get('gate_proj.weight_scale_inv'), hidden_states
        )
        up = w8a16_gemm(
            self.cached_up, self.weight_dequant_scale.get('up_proj.weight_scale_inv'), hidden_states
        )
        intermediate = F.silu(gate) * up
        return w8a16_gemm(
            self.cached_down, self.weight_dequant_scale.get('down_proj.weight_scale_inv'), intermediate
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.persistent:
            weights = self.load_weights(self.module_key)
            self.cached_gate = weights["gate_proj.weight"]
            self.cached_up = weights["up_proj.weight"]
            self.cached_down = weights["down_proj.weight"]

        result = self._forward_impl(hidden_states)

        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.cached_gate = self.cached_up = self.cached_down = None

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
        # Cached absorbed projections (Fix 1: avoid 78× FP8 dequant per step)
        self._cached_q_absorb = None
        self._cached_out_absorb = None
        # WP5: FP8 absorb weights (pre-quantized once at init)
        self._fp8_absorb_weights = None
        # WP2: Fused indexer KV proj (CUDA WGMMA)
        self._indexer_cuda_module = None
        self._indexer_cuda_weights = None
        # WP4: Fused scoring pipeline
        self._fused_wqb_weights = None
        self._fused_score_module = None
        # Fallback warning guards
        self._warned_fp8_absorb_fallback = False
        self._warned_indexer_kv_fallback = False

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

    def initialize_decode_absorb(self):
        """Pre-compute absorbed projections from FP8 kv_b_proj weight.

        Dequantizes the static FP8 weight once and caches the BF16
        q_absorb and out_absorb matrices. Eliminates 78× per-step
        dequantization in _forward_decode_dsa().
        """
        from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization
        attn = self.module
        weight_scale = self.weight_dequant_scale
        if weight_scale is None or "kv_b_proj.weight_scale_inv" not in weight_scale:
            return
        kv_b_proj = deepseek_v3_dequantization(
            attn.kv_b_proj.weight.data,
            weight_scale["kv_b_proj.weight_scale_inv"],
        ).view(attn.num_heads, -1, attn.kv_lora_rank)
        self._cached_q_absorb = kv_b_proj[:, :attn.qk_nope_head_dim, :].contiguous()
        self._cached_out_absorb = kv_b_proj[:, attn.qk_nope_head_dim:, :].contiguous()

        # WP5: Pre-quantize absorb weights for FP8 WGMMA kernel
        if _HAS_FP8_ABSORB and _GLM5_WP5_ENABLED:
            try:
                self._fp8_absorb_weights = FP8AbsorbWeights(
                    self._cached_q_absorb,   # [H, 192, 512]
                    self._cached_out_absorb,  # [H, 256, 512]
                )
                logging.debug(
                    f"[layer {self.layer_idx}] FP8 absorb weights initialized"
                )
            except Exception as e:
                logging.warning(
                    f"[layer {self.layer_idx}] FP8 absorb init failed: {e}"
                )
                self._fp8_absorb_weights = None

        # WP2/WP4 init moved to initialize_fused_kernels() — must run after set_device

    def initialize_fused_kernels(self):
        """Initialize TMA-based CUDA kernels (WP2/WP4).

        Must be called AFTER torch.cuda.set_device(local_rank) — TMA descriptors
        contain physical GPU addresses and are not portable across devices —
        AND after _setup_fp8_scales has attached indexer.wk_scale / wq_b_scale.
        """
        attn = self.module
        if not hasattr(attn, "indexer"):
            return
        indexer = attn.indexer

        # Load-bearing ordering contract: WP2/WP4 both need the FP8 dequant scales
        # attached by PSM._setup_fp8_scales. If this fires, somebody moved
        # _init_fused_kernels back in front of _setup_fp8_scales (see commit
        # d3b99222).
        assert hasattr(indexer, 'wk_scale'), (
            f"[layer {self.layer_idx}] initialize_fused_kernels called before "
            "_setup_fp8_scales attached indexer.wk_scale"
        )

        # WP2: Fused indexer KV proj (GEMM-only path, LayerNorm stays in PyTorch)
        if _HAS_FUSED_INDEXER_KV and _GLM5_WP2_ENABLED:
            try:
                self._indexer_cuda_module = _build_indexer_module()
                # Dequantize FP8 wk weight to BF16 (kernel re-quantizes internally for TMA)
                wk_bf16 = glm5_fp8_dequantization(
                    indexer.wk.weight.data, indexer.wk_scale,
                )
                self._indexer_cuda_weights = FP8IndexerWeightsCUDA(
                    wk_bf16, self._indexer_cuda_module,
                )
                logging.debug(
                    f"[layer {self.layer_idx}] WP2 fused indexer KV proj initialized"
                )
            except Exception as e:
                logging.warning(
                    f"[layer {self.layer_idx}] WP2 init failed: {e}"
                )
                self._indexer_cuda_module = None
                self._indexer_cuda_weights = None

        # WP4: Fused scoring pipeline (CUDA WGMMA wq_b + RoPE + Hadamard + scoring + topk)
        if _HAS_FUSED_SCORE and _GLM5_WP4_ENABLED and self._indexer_cuda_module is not None and hasattr(indexer, 'wq_b_scale'):
            try:
                wq_b_bf16 = glm5_fp8_dequantization(
                    indexer.wq_b.weight.data, indexer.wq_b_scale,
                )
                self._fused_wqb_weights = FP8WqbWeightsCUDA(
                    wq_b_bf16, self._indexer_cuda_module,
                )
                # Attach to indexer so score_and_select_paged can use it
                indexer._fused_score_weights = self._fused_wqb_weights
                indexer._fused_score_module = self._indexer_cuda_module
                indexer._warned_fused_score_fallback = False
                logging.debug(
                    f"[layer {self.layer_idx}] WP4 fused scoring pipeline initialized"
                )
            except Exception as e:
                logging.warning(
                    f"[layer {self.layer_idx}] WP4 init failed: {e}"
                )
                self._fused_wqb_weights = None

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
        # Early-return if auxiliary view is unavailable (e.g.,
        # BATCHGEN_GLM5_DISABLE_DUAL_KV=1 forces single primary view).
        if AttnWrapperBase.host_paged_kv_worker_view_aux is None:
            return
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx
            seq_kv = offload_kv[start_idx:end_idx].unsqueeze(0).unsqueeze(2)
            seq_global_id = [global_sequence_ids[seq_idx]]
            task = AttnWrapperBase.host_paged_kv_worker_view_aux.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )
            # Capture the future. CPU thread issues cudaMemcpyAsync inside; if
            # we discard the task, decode may start reading host KV before the
            # offload's CPU lambda has even queued its copies.
            if task is not None:
                AttnWrapperBase.pending_prefill_offload_tasks.append(task)

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
            task = self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )
            if task is not None:
                AttnWrapperBase.pending_prefill_offload_tasks.append(task)

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

            # Dense-MLA path (use_dense_mla=True, indexer not constructed).
            # Uses absorbed MLA with top-K = arange(max_seqlen); WP2/WP4/WP5
            # never fire; no aux KV cache. Mirrors Kimi-K2.5's decode shape
            # but uses sparse_flash_mla_decode (which supports head_dim_v=
            # kv_lora_rank=512) instead of flash_mla_with_kvcache (requires
            # head_size_v=576, incompatible with GLM-5's absorbed layout).
            attn_output = self._forward_decode_dense(
                hidden_states, position_ids, cache_seqlens, max_seqlen,
                gpu_paged_kv_manager,
            )
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

    def _forward_decode_dense(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen: int,
        gpu_paged_kv_manager,
    ) -> torch.Tensor:
        """Dense absorbed-MLA decode (no DSA, no WP5).

        Shape matches Kimi-K2.5's decode (absorbed MLA over full KV cache)
        but uses sparse_flash_mla_decode with top-K = arange(max_seqlen) so
        we can pass head_dim_v=kv_lora_rank=512 — flash_mla_with_kvcache
        enforces head_size_v=576 which GLM-5's absorbed layout can't satisfy.

        WP5 (fp8_q_absorb / fp8_out_absorb) is intentionally NOT used here;
        absorbed Q and out are pure torch.einsum for maximum clarity on this
        diagnostic path. WP2/WP4 are irrelevant because no indexer is
        constructed when use_dense_mla=True.
        """
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.attention.mla.fused_rmsnorm_rope import fused_rmsnorm_rope_with_q
        from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization
        from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
        from batchgen.attention.dsa.sparse_decode_mla import sparse_flash_mla_decode
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
        from batchgen.timing import get_decode_timer

        weight_scale = self.weight_dequant_scale
        attn = self.module
        bsz = hidden_states.shape[0]
        dt = get_decode_timer()
        li = self.layer_idx

        if bsz == 0:
            return hidden_states.new_empty(0, 1, attn.hidden_size)

        with (dt.timed("act_quant", li) if dt else _nullctx()):
            hidden_flat = hidden_states.squeeze(1)
            hidden_fp8, hidden_scale = act_quant(hidden_flat)

        # --- Q path ---
        with (dt.timed("q_proj", li) if dt else _nullctx()):
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

        # --- KV path ---
        with (dt.timed("kv_proj", li) if dt else _nullctx()):
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
                eps=attn.kv_a_layernorm.eps,
            )

        new_token_pos = position_ids.squeeze(-1)
        manager_device = gpu_paged_kv_manager.device
        seq_lengths_i32 = new_token_pos.to(dtype=torch.int32, device=manager_device)

        # Step 1: write new MLA KV to primary cache
        with (dt.timed("kv_write", li) if dt else _nullctx()):
            k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1))
            if k_tensor.device != manager_device:
                k_tensor = k_tensor.to(manager_device)
            gpu_paged_kv_manager.update_layer_decode_new_token(
                k_tensor=k_tensor,
                v_tensor=None,
                sequence_lengths=seq_lengths_i32,
                layer_idx=li,
            )
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(li, k_tensor, None)

        # Step 3 (replaced): dense top-K = arange over the full cache.
        updated_seqlens = cache_seqlens
        top_k_indices = torch.arange(
            max_seqlen, device=hidden_states.device, dtype=torch.long,
        ).unsqueeze(0).expand(bsz, -1)

        # Step 4: gather from primary cache (structurally dense)
        with (dt.timed("sparse_gather", li) if dt else _nullctx()):
            mla_blocked_k, _, mla_block_table = \
                gpu_paged_kv_manager.get_layer_kv_with_page_table(li)
            mla_page_size = gpu_paged_kv_manager.config.page_size_tokens
            sparse_mla_kv = sparse_gather_from_paged_kv(
                mla_blocked_k, mla_block_table, top_k_indices, mla_page_size,
            )

        # Step 5: absorbed Q via einsum (WP5 off), sparse_flash_mla_decode
        with (dt.timed("q_absorb", li) if dt else _nullctx()):
            if self._cached_q_absorb is not None:
                q_absorb = self._cached_q_absorb
                out_absorb = self._cached_out_absorb
            else:
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
            # Pure einsum absorbed Q (matches Kimi wrappers.py flashmla_backend:1548-1550).
            query_states[:, :, :, :attn.kv_lora_rank] = torch.einsum(
                "bhd,hdc->bhc", q_nope_squeezed, q_absorb,
            ).view(bsz, attn.num_heads, 1, attn.kv_lora_rank)
            query_states[:, :, :, attn.kv_lora_rank:] = q_pe
            query_states = query_states.view(bsz, 1, attn.num_heads, qk_head_dim)

        with (dt.timed("sparse_attn", li) if dt else _nullctx()):
            topk = top_k_indices.shape[1]
            sparse_seqlens = torch.clamp(updated_seqlens, max=topk)
            attn_out = sparse_flash_mla_decode(
                query_states, sparse_mla_kv, sparse_seqlens,
                attn.num_heads, attn.softmax_scale,
                head_dim_v=attn.kv_lora_rank,
                page_size=mla_page_size,
            )

        # Step 6: out-absorb via einsum → FP8 quant → o_proj
        with (dt.timed("o_proj", li) if dt else _nullctx()):
            attn_output = torch.einsum('bqhc,hdc->bqhd', attn_out, out_absorb)
            attn_output = attn_output.reshape(bsz, attn.num_heads * attn.v_head_dim)
            attn_output_fp8, attn_output_scale = act_quant(attn_output)
            attn_output = w8a8_deepgemm(
                attn_output_fp8, attn_output_scale,
                attn.o_proj.weight, weight_scale["o_proj.weight_scale_inv"],
            )
            attn_output = attn_output.view(bsz, 1, -1)

        return attn_output

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
        from batchgen.timing import get_decode_timer

        weight_scale = self.weight_dequant_scale
        attn = self.module
        indexer = attn.indexer
        bsz = hidden_states.shape[0]
        dt = get_decode_timer()
        li = self.layer_idx

        # Handle empty batch (some DP ranks have 0 sequences at late decode stages)
        if bsz == 0:
            return hidden_states.new_empty(0, 1, attn.hidden_size)

        # --- Shared FP8 activation quantization ---
        with (dt.timed("act_quant", li) if dt else _nullctx()):
            hidden_flat = hidden_states.squeeze(1)  # [batch, hidden_size]
            hidden_fp8, hidden_scale = act_quant(hidden_flat)

        # --- Q path: q_a_proj → layernorm → q_b_proj → split → RoPE ---
        with (dt.timed("q_proj", li) if dt else _nullctx()):
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
        with (dt.timed("kv_proj", li) if dt else _nullctx()):
            new_compressed_kv = w8a8_deepgemm(
                hidden_fp8, hidden_scale,
                attn.kv_a_proj_with_mqa.weight,
                weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
            ).view(bsz, 1, -1)

            cos, sin = attn.rotary_emb(q_pe, seq_len=max_seqlen)
            # Explicitly pass the module's RMSNorm eps (config.rms_norm_eps=1e-5
            # for GLM-5). The kernel default is 1e-6; without this override the
            # decode-side new-KV gets normalized with a 10× tighter regularizer
            # than the prefill-cached KV, creating a slow drift that compounds
            # across 78 layers × decode steps and manifests as ngram loops /
            # off-topic generation after ~hundreds of tokens.
            offload_kv = fused_rmsnorm_rope_with_q(
                new_compressed_kv, q_pe, cos, sin, position_ids,
                attn.kv_a_layernorm.weight,
                attn.kv_lora_rank, attn.qk_rope_head_dim,
                eps=attn.kv_a_layernorm.eps,
            )

        # Pre-compute seq_lengths_i32 once (shared by kv_write and indexer_k)
        new_token_pos = position_ids.squeeze(-1)  # [batch]
        manager_device = gpu_paged_kv_manager.device
        seq_lengths_i32 = new_token_pos.to(dtype=torch.int32, device=manager_device)

        # --- Step 1: Write new MLA KV to primary cache ---
        with (dt.timed("kv_write", li) if dt else _nullctx()):
            k_tensor = offload_kv.view(bsz, 1, 1, offload_kv.size(-1))
            if k_tensor.device != manager_device:
                k_tensor = k_tensor.to(manager_device)
            gpu_paged_kv_manager.update_layer_decode_new_token(
                k_tensor=k_tensor,
                v_tensor=None,
                sequence_lengths=seq_lengths_i32,
                layer_idx=li,
            )
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(li, k_tensor, None)

        # --- Step 2: Write indexer K to auxiliary cache ---
        # Skip entirely when aux is disabled (BATCHGEN_GLM5_DISABLE_DUAL_KV).
        # Combine with BATCHGEN_GLM5_FORCE_DENSE_MLA so scoring also bypasses aux.
        if gpu_paged_kv_manager_aux is not None:
            with (dt.timed("indexer_k", li) if dt else _nullctx()):
                # WP2: Fused CUDA WGMMA wk_proj (GEMM only) + PyTorch LayerNorm + RoPE + Hadamard
                if self._indexer_cuda_weights is not None:
                    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import cuda_wk_proj_gemm_only
                    # CUDA kernel: hidden_states → FP8 act_quant → WGMMA wk_proj
                    k_raw = cuda_wk_proj_gemm_only(
                        hidden_flat,  # [B, hidden_size] — already squeezed above
                        self._indexer_cuda_weights,
                        self._indexer_cuda_module,
                    )  # [B, index_head_dim=128]
                    # LayerNorm (model uses nn.LayerNorm with bias, not RMSNorm)
                    k_normed = indexer.k_norm(k_raw)
                    # Apply RoPE + Hadamard (reuse indexer's fused op)
                    k_normed_3d = k_normed.unsqueeze(1)  # [B, 1, 128]
                    indexer_kv = indexer._fused_rope_hadamard_or_fallback(
                        k_normed_3d, new_token_pos, max_seqlen=max_seqlen,
                    ).unsqueeze(2)  # [B, 1, 1, 128]
                else:
                    if not self._warned_indexer_kv_fallback:
                        self._warned_indexer_kv_fallback = True
                        logging.warning(
                            f"[layer {self.layer_idx}] WP2 fused indexer KV proj unavailable, "
                            "falling back to PyTorch w8a16_gemm — check batchgen_kernels import"
                        )
                    indexer_kv = indexer.compute_indexer_kv(
                        hidden_states, positions=new_token_pos, max_seqlen=max_seqlen,
                    )
                indexer_k_tensor = indexer_kv  # [batch, 1, 1, index_dim]
                aux_device = gpu_paged_kv_manager_aux.device
                seq_lengths_i32_aux = seq_lengths_i32 if aux_device == manager_device else new_token_pos.to(dtype=torch.int32, device=aux_device)
                gpu_paged_kv_manager_aux.update_layer_decode_new_token(
                    k_tensor=indexer_k_tensor,
                    v_tensor=None,
                    sequence_lengths=seq_lengths_i32_aux,
                    layer_idx=li,
                )
                # Offload indexer K to auxiliary host cache
                if AttnWrapperBase.kv_append_callback_aux is not None:
                    AttnWrapperBase.kv_append_callback_aux(li, indexer_k_tensor, None)

        # --- Step 3: Score all cached tokens (including new), select top-K ---
        with (dt.timed("indexer_score", li) if dt else _nullctx()):
            # cache_seqlens already includes the new token (pre-incremented in worker)
            updated_seqlens = cache_seqlens

            # BATCHGEN_GLM5_FORCE_DENSE_MLA=1: bypass DSA indexer scoring entirely
            # and run dense MLA over ALL cached tokens, even past index_topk.
            # Used to bisect whether the indexer.score_and_select_paged path is
            # the source of L4 LongBench gibberish. Read per-iter — hot-pluggable.
            _force_dense = _os_dsa_diag.environ.get("BATCHGEN_GLM5_FORCE_DENSE_MLA", "0") == "1"
            if _force_dense or max_seqlen <= indexer.index_topk:
                # Short-circuit: all sequences fit within topk — use full range
                max_len = max_seqlen
                top_k_indices = torch.arange(
                    max_len, device=hidden_states.device, dtype=torch.long,
                ).unsqueeze(0).expand(bsz, -1)
                if li == 0 and _os_dsa_diag.environ.get("BATCHGEN_GLM5_DSA_DIAG", "0") == "1":
                    branch = "dense-forced" if _force_dense else "dense"
                    _dsa_diag_log(branch, max_seqlen, bsz)
            else:
                if li == 0 and _os_dsa_diag.environ.get("BATCHGEN_GLM5_DSA_DIAG", "0") == "1":
                    _dsa_diag_log("sparse", max_seqlen, bsz)
                # Full indexer scoring path
                q_a_for_indexer = q_a_normed.unsqueeze(1)  # [batch, 1, q_lora_rank]
                indexer_blocked_k, _, idx_block_table = \
                    gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(li)
                aux_page_size = gpu_paged_kv_manager_aux.config.page_size_tokens
                top_k_indices = indexer.score_and_select_paged(
                    q_a_for_indexer, hidden_states,
                    indexer_blocked_k, idx_block_table,
                    updated_seqlens, aux_page_size,
                    positions=new_token_pos,
                    max_seqlen=max_seqlen,
                )

        # --- Step 4: Sparse gather MLA KV at top-K positions ---
        with (dt.timed("sparse_gather", li) if dt else _nullctx()):
            mla_blocked_k, _, mla_block_table = \
                gpu_paged_kv_manager.get_layer_kv_with_page_table(li)
            mla_page_size = gpu_paged_kv_manager.config.page_size_tokens
            sparse_mla_kv = sparse_gather_from_paged_kv(
                mla_blocked_k, mla_block_table, top_k_indices, mla_page_size,
            )
            # sparse_mla_kv: [batch, topk, 1, 576]

        # --- Step 5: Absorbed Q → sparse FlashMLA ---
        with (dt.timed("q_absorb", li) if dt else _nullctx()):
            if self._cached_q_absorb is not None:
                q_absorb = self._cached_q_absorb
                out_absorb = self._cached_out_absorb
            else:
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
            q_nope_squeezed = q_nope.squeeze(2)  # [B, H, qk_nope_head_dim]

            # WP5: FP8 absorb kernel or fallback to torch.einsum
            if self._fp8_absorb_weights is not None:
                absorbed_q = fp8_q_absorb(q_nope_squeezed, self._fp8_absorb_weights)
                query_states[:, :, :, :attn.kv_lora_rank] = absorbed_q.view(
                    bsz, attn.num_heads, 1, attn.kv_lora_rank
                )
            else:
                if not self._warned_fp8_absorb_fallback:
                    self._warned_fp8_absorb_fallback = True
                    logging.warning(
                        f"[layer {self.layer_idx}] WP5 FP8 absorb unavailable, "
                        "falling back to torch.einsum — check batchgen_kernels import"
                    )
                query_states[:, :, :, :attn.kv_lora_rank] = torch.einsum(
                    "bhd,hdc->bhc", q_nope_squeezed, q_absorb,
                ).view(bsz, attn.num_heads, 1, attn.kv_lora_rank)

            query_states[:, :, :, attn.kv_lora_rank:] = q_pe
            query_states = query_states.view(bsz, 1, attn.num_heads, qk_head_dim)

        with (dt.timed("sparse_attn", li) if dt else _nullctx()):
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
        with (dt.timed("o_proj", li) if dt else _nullctx()):
            # WP5: FP8 out_absorb kernel or fallback to torch.einsum
            if self._fp8_absorb_weights is not None:
                attn_output = fp8_out_absorb(attn_out, self._fp8_absorb_weights)
            else:
                attn_output = torch.einsum('bqhc,hdc->bqhd', attn_out, out_absorb)
            attn_output = attn_output.reshape(bsz, attn.num_heads * attn.v_head_dim)
            attn_output_fp8, attn_output_scale = act_quant(attn_output)
            attn_output = w8a8_deepgemm(
                attn_output_fp8, attn_output_scale,
                attn.o_proj.weight, weight_scale["o_proj.weight_scale_inv"],
            )
            attn_output = attn_output.view(bsz, 1, -1)

        return attn_output
