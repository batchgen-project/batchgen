"""CUDA Graph capturable segments for GPT-OSS-120B decode.

Segments:
  FullAttnSegment: RMSNorm → QKV → RoPE → KV write → FA → O_proj → post-attn norm
  MoESegment: AllGather → router → grouped WGMMA MoE → ReduceScatter

SharedMoEBufferPool allocates max-bucket-sized buffers ONCE and creates
per-bucket views (slices sharing the same data_ptr). All 36 MoE layers
share a single pool instance — they execute sequentially so no conflicts.

Dynamic metadata (cache_seqlens, page_table) are static-address input buffers
whose contents are updated before each replay. KV cache and cos/sin tables
are at fixed GPU addresses. NCCL collectives use PyNccl (ctypes) for graph
compatibility.
"""

import logging
from typing import Dict, List, Optional

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers.attention import AttnWrapperBase

logger = logging.getLogger(__name__)


class FullAttnSegment:
    """Full attention block as a single CUDA-graph-capturable segment.

    Inputs:  hidden_states [B, 1, hidden_size], cache_seqlens [B] int32,
             page_table [B, max_pages_per_seq] int32, slot_indices [B] int32
    Outputs: normed [B, 1, hidden_size] (MoE input), residual [B, 1, hidden_size]
    """

    def __init__(self, decoder_layer, attn_wrapper, layer_idx: int, max_seq_len: int,
                 max_pages_per_seq: int, page_size_tokens: int):
        # Pre-attn: RMSNorm
        self.ln_weight = decoder_layer.input_layernorm.weight
        self.ln_eps = decoder_layer.input_layernorm.eps

        # Pre-attn: QKV proj
        self.qkv_proj = attn_wrapper.module.qkv_proj

        # Dimensions
        self.hidden_size = decoder_layer.hidden_size
        self.q_size = attn_wrapper.q_size
        self.kv_size = attn_wrapper.kv_size
        self.num_heads = attn_wrapper.num_heads
        self.num_kv_heads = attn_wrapper.num_kv_heads
        self.head_dim = attn_wrapper.head_dim

        # Mid: RoPE
        self.rotary_emb = attn_wrapper.module.rotary_emb
        self.max_seq_len = max_seq_len

        # Mid: attention
        self.o_proj = attn_wrapper.module.o_proj
        self.scale = attn_wrapper.scale
        self.sliding_window = attn_wrapper.sliding_window
        self.sinks = attn_wrapper.sinks
        self.layer_idx = layer_idx

        # Post-attn: RMSNorm
        self.post_ln_weight = decoder_layer.post_attention_layernorm.weight
        self.post_ln_eps = decoder_layer.post_attention_layernorm.eps

        # Page table dimensions for static buffer
        self.max_pages_per_seq = max_pages_per_seq
        self.page_size_tokens = page_size_tokens

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
                ("batch_size",), torch.int32, fill_value=0
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
        }

    def forward(
        self, hidden_states: torch.Tensor, cache_seqlens: torch.Tensor,
        page_table: torch.Tensor, slot_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        from batchgen.attention.fused_kernels import (
            cuda_rmsnorm, cuda_qkv_split, cuda_rope, cuda_add_rmsnorm,
        )
        from batchgen.attention.gqa.fa_decode import gqa_decode_fa
        from batchgen.kv_cache.gpu_kv_kernels import run_paged_kv_token_update_fused

        B = hidden_states.shape[0]

        # === Pre-attn: RMSNorm → QKV proj → split → reshape ===
        normed = cuda_rmsnorm(hidden_states, self.ln_weight, self.ln_eps)
        qkv = self.qkv_proj(normed)
        query, key, value = cuda_qkv_split(qkv, self.q_size, self.kv_size)
        query = query.view(B, 1, self.num_heads, self.head_dim)
        key = key.view(B, 1, self.num_kv_heads, self.head_dim)
        value = value.view(B, 1, self.num_kv_heads, self.head_dim)

        # === RoPE ===
        current_pos = cache_seqlens - 1
        cos, sin = self.rotary_emb(value.transpose(1, 2), seq_len=self.max_seq_len)
        cos = cos[current_pos].unsqueeze(1)
        sin = sin[current_pos].unsqueeze(1)
        query, key = cuda_rope(query, key, cos, sin)

        # === KV write (direct kernel call with static buffers) ===
        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        k_cache, v_cache, _ = gpu_kv_manager.get_layer_kv_with_page_table(
            self.layer_idx
        )
        run_paged_kv_token_update_fused(
            k_cache=k_cache,
            k_tokens=key.view(B, -1),
            page_table=page_table,
            slot_indices=slot_indices,
            token_indices=current_pos,
            page_size_tokens=self.page_size_tokens,
            v_cache=v_cache,
            v_tokens=value.view(B, -1),
        )
        attn_out, _ = gqa_decode_fa(
            q=query, k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=cache_seqlens, block_table=page_table,
            sinks=self.sinks, softmax_scale=self.scale,
            sliding_window=self.sliding_window,
        )

        # === O_proj ===
        attn_out = attn_out.view(B, 1, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)

        # === Post-attn: residual add + RMSNorm ===
        normed, residual = cuda_add_rmsnorm(
            hidden_states, attn_out, self.post_ln_weight, self.post_ln_eps
        )
        return {"normed": normed, "residual": residual}


# ---------------------------------------------------------------------------
# Shared MoE buffer pool
# ---------------------------------------------------------------------------

class SharedMoEBufferPool:
    """Single buffer pool shared across all MoE layers for CUDA graph capture.

    Allocates max-bucket-sized buffers once. For each bucket size, creates
    correctly-sized views (same data_ptr, different shape) and TMA descriptors.

    Memory usage: ~207MB for max_bucket=256, W=8, H=2880, E=128, K=4.
    Previous per-layer approach: 36 layers × 16 buckets × ~20MB+ = OOM.
    """

    _BLOCK_M = 64

    def __init__(
        self,
        world_size: int,
        hidden_size: int,
        total_experts: int,
        num_experts_per_tok: int,
        num_local_experts: int,
        N_intermediate: int,
        device: torch.device,
    ):
        self.W = world_size
        self.H = hidden_size
        self.E = total_experts
        self.K = num_experts_per_tok
        self.E_local = num_local_experts
        self.N_inter = N_intermediate
        self.device = device
        self._base = {}
        self._views = {}
        self._tma_descs = {}

    def setup(self, bucket_sizes: List[int]) -> None:
        """Allocate base buffers for max bucket, create per-bucket views + TMA descs."""
        max_B = max(bucket_sizes)
        WB_max = self.W * max_B
        NK_max = WB_max * self.K
        disp_max = max(NK_max, self._BLOCK_M)

        b = self._base
        d = self.device
        H = self.H

        # Base allocations at max size
        b["padded"]          = torch.zeros(max_B, H, dtype=torch.bfloat16, device=d)
        b["all_tokens"]      = torch.zeros(WB_max, H, dtype=torch.bfloat16, device=d)
        b["router_logits"]   = torch.empty(WB_max, self.E, dtype=torch.bfloat16, device=d)
        b["router_f32"]      = torch.empty(WB_max, self.E, dtype=torch.float32, device=d)
        b["topk_indices"]    = torch.empty(WB_max, self.K, dtype=torch.int32, device=d)
        b["topk_weights"]    = torch.empty(WB_max, self.K, dtype=torch.float32, device=d)
        b["expert_counts"]   = torch.zeros(self.E_local, dtype=torch.int32, device=d)
        b["expert_offsets"]  = torch.empty(self.E_local + 1, dtype=torch.int32, device=d)
        b["expert_counters"] = torch.zeros(self.E_local, dtype=torch.int32, device=d)
        b["topk_pos"]        = torch.full((NK_max,), -1, dtype=torch.int32, device=d)
        b["dispatched_x"]    = torch.zeros(disp_max, H, dtype=torch.bfloat16, device=d)
        b["intermediate"]    = torch.zeros(disp_max, self.N_inter, dtype=torch.bfloat16, device=d)
        b["sorted_output"]   = torch.zeros(disp_max, H, dtype=torch.bfloat16, device=d)
        b["moe_output"]      = torch.zeros(WB_max, H, dtype=torch.bfloat16, device=d)
        b["local_output"]    = torch.zeros(max_B, H, dtype=torch.bfloat16, device=d)

        total_bytes = sum(t.nelement() * t.element_size() for t in b.values())
        logger.info(
            f"SharedMoEBufferPool: allocated {total_bytes / 1024**2:.1f}MB base buffers "
            f"(max_bucket={max_B}, W={self.W}, H={H})"
        )

        for B in bucket_sizes:
            self._create_views(B)

    def _create_views(self, B: int) -> None:
        from batchgen.moe.fused_wgmma_grouped import create_tma_descriptor

        WB = self.W * B
        NK = WB * self.K
        disp = max(NK, self._BLOCK_M)
        b = self._base

        v = {}
        v["padded"]          = b["padded"][:B]
        v["all_tokens"]      = b["all_tokens"][:WB]
        v["router_logits"]   = b["router_logits"][:WB]
        v["router_f32"]      = b["router_f32"][:WB]
        v["topk_indices"]    = b["topk_indices"][:WB]
        v["topk_weights"]    = b["topk_weights"][:WB]
        v["expert_counts"]   = b["expert_counts"]        # always E_local
        v["expert_offsets"]  = b["expert_offsets"]        # always E_local+1
        v["expert_counters"] = b["expert_counters"]       # always E_local
        v["topk_pos"]        = b["topk_pos"][:NK]
        v["dispatched_x"]    = b["dispatched_x"][:disp]
        v["intermediate"]    = b["intermediate"][:disp]
        v["sorted_output"]   = b["sorted_output"][:disp]
        v["moe_output"]      = b["moe_output"][:WB]
        v["local_output"]    = b["local_output"][:B]
        self._views[B] = v

        # TMA descriptors for WGMMA kernels — must be created before graph capture
        self._tma_descs[B] = {
            "dispatched": create_tma_descriptor(v["dispatched_x"]),
            "intermediate": create_tma_descriptor(v["intermediate"]),
        }

    def get(self, bucket_size: int):
        """Return (views_dict, tma_descs_dict) for this bucket."""
        return self._views[bucket_size], self._tma_descs[bucket_size]


# ---------------------------------------------------------------------------
# MoE segment (lightweight — references shared pool)
# ---------------------------------------------------------------------------

class MoESegment:
    """MoE block as a CUDA-graph-capturable segment (EP, persistent experts only).

    Lightweight wrapper: holds only per-layer model weights and references
    the SharedMoEBufferPool for all intermediate buffers. No buffer ownership.

    Pipeline: AllGather → router (bf16 matmul + f32 cast) → gate_topk_softmax →
    dispatch → WGMMA stage1 → WGMMA stage2 → reduce_weighted_scatter →
    reduce_scatter

    Inputs:  hidden_states [B, H] bf16
    Outputs: moe_output [B, H] bf16
    """

    def __init__(self, moe_decode, pool: SharedMoEBufferPool, comm,
                 world_size: int, rank: int, device: torch.device):
        import torch.distributed as dist
        self.dist = dist

        self.pool = pool
        self.comm = comm
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.hidden_size = moe_decode.hidden_size
        self.num_experts_per_tok = moe_decode.num_experts_per_tok
        self.total_experts = moe_decode.total_experts
        self.expert_start = moe_decode.expert_start
        self.num_local_experts = len(moe_decode.persistent_expert_indices)

        # Router weight: store bf16 transpose for graph-compatible matmul
        # Router matmul: all_tokens [WB, H] bf16 @ router_weight_t [H, E] bf16 → router_logits [WB, E] bf16
        # Then cast the small [WB, E] result to f32 for gate kernel
        self.router_weight_bf16_t = moe_decode.router.weight.T.contiguous()  # [H, E] bf16
        _bias = getattr(moe_decode.router, 'bias', None)
        self.router_bias_bf16 = _bias.to(torch.bfloat16) if _bias is not None else None

        # Weight pointer arrays (at fixed GPU addresses)
        self.gate_ptrs = moe_decode.gate_ptrs
        self.gate_scale_ptrs = moe_decode.gate_scale_ptrs
        self.up_ptrs = moe_decode.up_ptrs
        self.up_scale_ptrs = moe_decode.up_scale_ptrs
        self.down_ptrs = moe_decode.down_ptrs
        self.down_scale_ptrs = moe_decode.down_scale_ptrs
        self.gate_weight_ref = moe_decode.gate_weight_ref
        self.gate_scale_ref = moe_decode.gate_scale_ref
        self.down_weight_ref = moe_decode.down_weight_ref
        self.down_scale_ref = moe_decode.down_scale_ref
        self.gate_bias_ptrs = getattr(moe_decode, 'gate_bias_ptrs', None)
        self.up_bias_ptrs = getattr(moe_decode, 'up_bias_ptrs', None)
        self.down_bias_ptrs = getattr(moe_decode, 'down_bias_ptrs', None)

        self.N_intermediate = self.gate_weight_ref.shape[0]
        self.s1_stride_weight_n = self.gate_weight_ref.shape[1]
        self.s1_stride_scale_n = self.gate_scale_ref.shape[1]
        self.s2_stride_weight_n = self.down_weight_ref.shape[1]
        self.s2_stride_scale_n = self.down_scale_ref.shape[1]

    def setup_static_buffers(self, bucket_size: int) -> None:
        """No-op: buffers are managed by SharedMoEBufferPool."""
        pass

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", self.hidden_size), torch.bfloat16
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "moe_output": TensorSpec(
                ("batch_size", self.hidden_size), torch.bfloat16
            ),
        }

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """CUDA-graph-compatible forward: no allocations, no CPU driver calls."""
        from batchgen.moe.routing import (
            gate_topk_softmax_cuda,
            dispatch_count_gather_cuda,
            reduce_weighted_scatter_cuda,
        )
        from batchgen.moe.fused_wgmma_grouped import (
            fused_mxfp4_grouped_stage1_inplace,
            fused_mxfp4_grouped_stage2_inplace,
        )

        B = hidden_states.shape[0]
        H = self.hidden_size
        W = self.world_size
        WB = W * B
        bufs, tma = self.pool.get(B)

        # 1. Copy input to padded buffer
        bufs["padded"].copy_(hidden_states)

        # 2. AllGather (PyNccl, graph-compatible)
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                bufs["all_tokens"], bufs["padded"],
                stream=torch.cuda.current_stream(self.device),
            )

        # 3. Router matmul in bf16 + bias in bf16 (matching nn.Linear), then cast to f32
        torch.mm(bufs["all_tokens"], self.router_weight_bf16_t, out=bufs["router_logits"])
        if self.router_bias_bf16 is not None:
            bufs["router_logits"].add_(self.router_bias_bf16)
        bufs["router_f32"].copy_(bufs["router_logits"])

        # 4. Gate top-k softmax (f32 input, pre-allocated int32/f32 outputs)
        gate_topk_softmax_cuda(
            bufs["router_f32"],
            topk_indices=bufs["topk_indices"],
            topk_weights=bufs["topk_weights"],
            k=self.num_experts_per_tok,
        )

        # 5. Dispatch: count + prefix_sum + gather into pre-allocated buffers
        bufs["dispatched_x"].zero_()
        dispatch_count_gather_cuda(
            bufs["all_tokens"], bufs["topk_indices"],
            self.expert_start, self.num_local_experts,
            expert_counts=bufs["expert_counts"],
            expert_offsets=bufs["expert_offsets"],
            expert_counters=bufs["expert_counters"],
            dispatched_x=bufs["dispatched_x"],
            topk_pos=bufs["topk_pos"],
        )

        # 6. WGMMA Stage 1 (gate + up + SwiGLU)
        fused_mxfp4_grouped_stage1_inplace(
            bufs["dispatched_x"], bufs["intermediate"],
            tma["dispatched"], bufs["expert_offsets"],
            self.gate_ptrs, self.gate_scale_ptrs,
            self.up_ptrs, self.up_scale_ptrs,
            self.N_intermediate,
            self.s1_stride_weight_n, self.s1_stride_scale_n,
            gate_bias_ptrs=self.gate_bias_ptrs,
            up_bias_ptrs=self.up_bias_ptrs,
        )

        # 7. WGMMA Stage 2 (down projection)
        fused_mxfp4_grouped_stage2_inplace(
            bufs["intermediate"], bufs["sorted_output"],
            tma["intermediate"], bufs["expert_offsets"],
            self.down_ptrs, self.down_scale_ptrs,
            H,
            self.s2_stride_weight_n, self.s2_stride_scale_n,
            down_bias_ptrs=self.down_bias_ptrs,
        )

        # 8. Reduce: weighted scatter-add back to original token order
        reduce_weighted_scatter_cuda(
            bufs["sorted_output"], bufs["topk_pos"], bufs["topk_weights"],
            WB, H, self.num_experts_per_tok,
            output=bufs["moe_output"],
        )

        # 9. AllReduce in-place (proven graph-compatible)
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                bufs["moe_output"], op=self.dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        # 10. Extract local rank's slice
        start = self.rank * B
        return {"moe_output": bufs["moe_output"][start:start + B]}
