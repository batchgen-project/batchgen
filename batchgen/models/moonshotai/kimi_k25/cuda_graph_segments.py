"""CUDA Graph capturable segments for Kimi K2.5 decode.

Segments:
  K25AttnSegment: Per-layer MLA attention (inlined, static page_table).
    Captured per-layer, replayed inside KimiK25DecoderLayer.forward().
    MoE stays eager with async shared expert overlap preserved.

MLA forward is INLINED (not delegated to decoding_attn_mode_3_bf16) because:
  - CUDA graph requires static tensor addresses — the gpu_paged_kv_manager's internal
    block_table may be reallocated. We use the static page_table input instead.
  - Same approach as GPT-OSS FullAttnSegment (see gpt_oss_120b/cuda_graph_segments.py).
  - Zero overhead: same kernels, same number of launches, just different page_table pointer.
"""

import logging
from typing import Dict

import torch
import torch.nn.functional as F

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers.attention import AttnWrapperBase

logger = logging.getLogger(__name__)


class K25AttnSegment:
    """MLA attention block as a single CUDA-graph-capturable segment.

    Covers: input_layernorm → Q projections → KV norm+RoPE → KV write →
            Q absorb → FlashMLA → out absorb → O_proj → residual+post_attn_norm

    Inputs:  hidden_states [B, 1, H], cache_seqlens [B], page_table [B, max_pages], slot_indices [B]
    Outputs: normed [B, 1, H] (MoE input), residual [B, 1, H], k_tensor [B, 1, 1, kv_dim]
    """

    def __init__(self, decoder_layer, attn_wrapper, layer_idx: int,
                 max_seq_len: int, max_pages_per_seq: int, page_size_tokens: int):
        # Get the actual attention module (unwrap if needed)
        self.attn_mod = attn_wrapper.module if hasattr(attn_wrapper, 'module') else attn_wrapper
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len
        self.max_pages_per_seq = max_pages_per_seq
        self.page_size_tokens = page_size_tokens

        # Layer norms
        self.input_ln_weight = decoder_layer.input_layernorm.weight
        self.input_ln_eps = decoder_layer.input_layernorm.variance_epsilon
        self.post_ln_weight = decoder_layer.post_attention_layernorm.weight
        self.post_ln_eps = decoder_layer.post_attention_layernorm.variance_epsilon

        # MLA dimensions
        attn = self.attn_mod
        self.hidden_size = attn.hidden_size
        self.num_heads = attn.num_heads              # 64
        self.q_lora_rank = attn.q_lora_rank          # 1536
        self.kv_lora_rank = attn.kv_lora_rank        # 512
        self.qk_nope_head_dim = attn.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = attn.qk_rope_head_dim  # 64
        self.v_head_dim = attn.v_head_dim            # 128
        self.q_head_dim = attn.q_head_dim            # 192
        self.kv_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576
        self.softmax_scale = attn.softmax_scale

        # Cache fused functions (avoid repeated lookups)
        self._fused_add_rmsnorm = decoder_layer._get_fused_add_rmsnorm_fn()
        from batchgen.models.moonshotai.kimi_k25.model import RMSNorm
        self._fused_rmsnorm = RMSNorm._get_fused_fn()

        # Pre-compute q_absorb and out_absorb from kv_b_proj (fixed tensors)
        kv_b_proj = attn.kv_b_proj.weight.data.view(self.num_heads, -1, self.kv_lora_rank)
        self.q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]   # [64, 128, 512]
        self.out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]  # [64, 128, 512]

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=1
            ),
            "page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            "slot_indices": TensorSpec(
                ("batch_size",), torch.int32, fill_value=-1
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "normed": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "residual": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.kv_dim), torch.bfloat16
            ),
        }

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply RMSNorm using cached fused function or PyTorch fallback."""
        if self._fused_rmsnorm is not None:
            return self._fused_rmsnorm(x, weight, eps)
        h = x.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        return (weight * (h * torch.rsqrt(variance + eps))).to(x.dtype)

    def compute_shared_decode_ctx(
        self, cache_seqlens: torch.Tensor, dtype_ref: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Layer-invariant decode quantities, computed once per forward.

        RoPE cos/sin, FlashMLA tile-scheduler metadata, and the position/token
        ids all derive only from the shared ``cache_seqlens`` (and the shared
        rotary instance + fixed ``max_seq_len``), so they are bit-identical
        across all 61 layers. The whole-model graph computes these once and
        threads them into every ``K25AttnSegment.forward`` (``shared_ctx=``),
        replacing ~61x redundant metadata/cos-sin/elementwise launches per step.
        ``dtype_ref`` only supplies dtype/device for cos/sin (rotary reads
        neither shape nor values when seq_len is passed explicitly).
        """
        from flash_mla import get_mla_metadata
        q_position_ids = (cache_seqlens - 1).clamp(min=0).unsqueeze(1).to(torch.int64)
        token_indices = q_position_ids.squeeze(-1).to(torch.int32)
        cos, sin = self.attn_mod.rotary_emb(dtype_ref, seq_len=self.max_seq_len)
        tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 128, 1)
        return {
            "q_position_ids": q_position_ids,
            "token_indices": token_indices,
            "cos": cos,
            "sin": sin,
            "tile_scheduler_metadata": tile_scheduler_metadata,
            "num_splits": num_splits,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
        shared_ctx: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inlined MLA attention with static page_table for CUDA graph compatibility.

        All ops are graph-safe:
        - get_mla_metadata: pure CUDA kernel (1 block, 32 threads)
        - flash_mla_with_kvcache: CUDA kernel, internal allocs handled by graph pool
        - fused_rmsnorm_rope_with_q: Triton kernel, internal alloc handled by graph pool
        - run_paged_kv_token_update_fused: Triton kernel
        - F.linear: cuBLAS
        - RMSNorm: CUDA kernel
        """
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata
        from batchgen_kernels.triton.fused_rmsnorm_rope import fused_rmsnorm_rope_with_q
        from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused

        B = hidden_states.shape[0]
        attn = self.attn_mod
        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager

        # Position IDs: cache_seqlens = current length AFTER this token is written,
        # so position = cache_seqlens - 1 (0-indexed). When the whole-model graph
        # supplies shared_ctx these are computed once per forward (layer-invariant);
        # otherwise derive here (standalone per-layer-graph path).
        if shared_ctx is not None:
            q_position_ids = shared_ctx["q_position_ids"]
            token_indices = shared_ctx["token_indices"]
        else:
            q_position_ids = (cache_seqlens - 1).clamp(min=0).unsqueeze(1).to(torch.int64)
            token_indices = q_position_ids.squeeze(-1).to(torch.int32)

        # === Pre-attn RMSNorm ===
        residual = hidden_states
        normed = self._rmsnorm(hidden_states, self.input_ln_weight, self.input_ln_eps)

        # === Q + KV projections ===
        normed_sq = normed.squeeze(1)  # [B, H]
        q = F.linear(normed_sq, attn.q_a_proj.weight)
        new_compressed_kv = F.linear(normed_sq, attn.kv_a_proj_with_mqa.weight).view(B, 1, -1)
        q = attn.q_a_layernorm(q)
        q = F.linear(q, attn.q_b_proj.weight)

        # === Q reshape + split ===
        q = q.view(B, 1, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = q_pe.contiguous()

        # === RoPE cos/sin (pre-extended to max_seq_len during init) ===
        if shared_ctx is not None:
            cos, sin = shared_ctx["cos"], shared_ctx["sin"]
        else:
            cos, sin = attn.rotary_emb(q_pe, seq_len=self.max_seq_len)

        # === Fused KV norm + RoPE on both KV and Q ===
        offload_kv = fused_rmsnorm_rope_with_q(
            new_compressed_kv, q_pe, cos, sin,
            q_position_ids, attn.kv_a_layernorm.weight,
            self.kv_lora_rank, self.qk_rope_head_dim,
        )

        # === KV tensor for host offload ===
        k_tensor = offload_kv.view(B, 1, 1, offload_kv.size(-1))

        # === KV write — use STATIC page_table + slot_indices ===
        # get_layer_kv_with_page_table returns k_cache at fixed GPU address.
        # We discard its block_table and use our static page_table input instead.
        blocked_k, _, _ = gpu_kv_manager.get_layer_kv_with_page_table(self.layer_idx)
        run_paged_kv_token_update_fused(
            k_cache=blocked_k,
            k_tokens=k_tensor.view(B, -1),
            page_table=page_table,
            slot_indices=slot_indices,
            token_indices=token_indices,
            page_size_tokens=self.page_size_tokens,
        )

        # === Q absorb + query states construction ===
        qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        query_states = torch.empty(
            B, self.num_heads, 1, qk_head_dim,
            dtype=blocked_k.dtype, device=hidden_states.device,
        )
        q_nope_sq = q_nope.squeeze(2)
        query_states[:, :, :, :self.kv_lora_rank] = torch.einsum(
            "bhd,hdc->bhc", q_nope_sq, self.q_absorb
        ).view(B, self.num_heads, 1, self.kv_lora_rank)
        query_states[:, :, :, self.kv_lora_rank:] = q_pe
        query_states = query_states.view(B, 1, self.num_heads, qk_head_dim)

        # === FlashMLA — use STATIC page_table ===
        # get_mla_metadata is a pure CUDA kernel (flash_fwd_mla_metadata.cu) — graph-safe.
        # flash_mla_with_kvcache internal allocs are handled by the CUDA graph memory pool.
        if shared_ctx is not None:
            tile_scheduler_metadata = shared_ctx["tile_scheduler_metadata"]
            num_splits = shared_ctx["num_splits"]
        else:
            tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 128, 1)
        attn_out, _ = flash_mla_with_kvcache(
            query_states, blocked_k,
            page_table,
            cache_seqlens,
            self.kv_lora_rank,  # head_dim_v = 512
            tile_scheduler_metadata, num_splits,
            self.softmax_scale, True,
        )

        # === Output absorb + O_proj ===
        attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, self.out_absorb)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, self.num_heads * self.v_head_dim)
        attn_output = F.linear(attn_output, attn.o_proj.weight)
        attn_output = attn_output.view(B, 1, -1)

        # === Post-attn: residual add + RMSNorm ===
        if self._fused_add_rmsnorm is not None:
            normed_out, residual_out = self._fused_add_rmsnorm(
                residual, attn_output,
                self.post_ln_weight, self.post_ln_eps,
            )
        else:
            combined = residual + attn_output
            residual_out = combined
            normed_out = self._rmsnorm(combined, self.post_ln_weight, self.post_ln_eps)

        return {"normed": normed_out, "residual": residual_out, "k_tensor": k_tensor}
