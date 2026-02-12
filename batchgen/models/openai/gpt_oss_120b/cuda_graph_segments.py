"""CUDA Graph capturable segments for GPT-OSS-120B decode.

Segments:
  FullAttnSegment: RMSNorm → QKV → RoPE → KV write → FA → O_proj → post-attn norm
  MoESegment: AllGather → router → grouped WGMMA MoE → AllReduce

Dynamic metadata (cache_seqlens, page_table) are static-address input buffers
whose contents are updated before each replay. KV cache and cos/sin tables
are at fixed GPU addresses. NCCL collectives use PyNccl (ctypes) for graph
compatibility.
"""

import logging
from typing import Dict

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


class MoESegment:
    """MoE block as a CUDA-graph-capturable segment (EP, persistent experts only).

    Bypasses GptOssMoEDecode._forward_ep to use per-bucket-sized buffers and
    call kernels directly. NCCL via PyNccl for graph compatibility.

    All intermediate buffers and TMA descriptors are pre-allocated per bucket
    via setup_static_buffers() before CUDA graph capture. The forward() method
    issues only kernel launches on static-address buffers — no allocations,
    no CPU-side driver API calls.

    Inputs:  hidden_states [B, H] bf16
    Outputs: moe_output [B, H] bf16
    """

    _BLOCK_M = 64  # TMA tile height

    def __init__(self, moe_decode, comm, world_size: int, rank: int, device):
        import torch.distributed as dist
        self.dist = dist

        self.router = moe_decode.router
        self.router_weight = moe_decode.router.weight  # [E, H] for linear
        # Pre-compute float32 weight transpose for graph-compatible router matmul
        self.router_weight_f32_t = moe_decode.router.weight.float().T.contiguous()  # [H, E]
        _bias = getattr(moe_decode.router, 'bias', None)
        self.router_bias_f32 = _bias.float() if _bias is not None else None
        self.comm = comm
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.hidden_size = moe_decode.hidden_size
        self.num_experts_per_tok = moe_decode.num_experts_per_tok
        self.total_experts = moe_decode.total_experts
        self.expert_start = moe_decode.expert_start
        self.num_local_experts = len(moe_decode.persistent_expert_indices)

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

        # Intermediate dimension from gate weight shape
        self.N_intermediate = self.gate_weight_ref.shape[0]

        # Strides from reference weights
        self.s1_stride_weight_n = self.gate_weight_ref.shape[1]   # K // 2
        self.s1_stride_scale_n = self.gate_scale_ref.shape[1]     # K // 32
        self.s2_stride_weight_n = self.down_weight_ref.shape[1]   # N // 2
        self.s2_stride_scale_n = self.down_scale_ref.shape[1]     # N // 32

        # Per-bucket static buffers: populated by setup_static_buffers()
        self._static_bufs = {}

    def setup_static_buffers(self, bucket_size: int) -> None:
        """Pre-allocate all intermediate buffers for a given bucket size.

        Must be called once per bucket BEFORE CUDA graph capture. Creates
        static-address buffers and TMA descriptors that remain valid during
        graph replay.
        """
        from batchgen.moe.fused_wgmma_grouped import create_tma_descriptor

        B = bucket_size
        W = self.world_size
        H = self.hidden_size
        K_topk = self.num_experts_per_tok
        E_local = self.num_local_experts
        WB = W * B
        NK = WB * K_topk
        # Dispatched tokens buffer must be >= BLOCK_M for TMA descriptor validity
        disp_rows = max(NK, self._BLOCK_M)

        bufs = {}

        # Communication buffers
        bufs["padded"] = torch.zeros(B, H, dtype=torch.bfloat16, device=self.device)
        bufs["all_tokens"] = torch.zeros(WB, H, dtype=torch.bfloat16, device=self.device)
        bufs["global_results"] = torch.zeros(WB, H, dtype=torch.bfloat16, device=self.device)

        # Router output (avoid nn.Linear allocation)
        bufs["all_tokens_f32"] = torch.zeros(WB, H, dtype=torch.float32, device=self.device)
        bufs["router_logits"] = torch.empty(WB, self.total_experts, dtype=torch.float32, device=self.device)

        # Gate/dispatch buffers
        bufs["topk_indices"] = torch.empty(WB, K_topk, dtype=torch.int32, device=self.device)
        bufs["topk_weights"] = torch.empty(WB, K_topk, dtype=torch.float32, device=self.device)
        bufs["expert_counts"] = torch.zeros(E_local, dtype=torch.int32, device=self.device)
        bufs["expert_offsets"] = torch.empty(E_local + 1, dtype=torch.int32, device=self.device)
        bufs["expert_counters"] = torch.zeros(E_local, dtype=torch.int32, device=self.device)
        bufs["topk_pos"] = torch.full((NK,), -1, dtype=torch.int32, device=self.device)
        bufs["dispatched_x"] = torch.zeros(disp_rows, H, dtype=torch.bfloat16, device=self.device)

        # WGMMA intermediate and output
        bufs["intermediate"] = torch.zeros(disp_rows, self.N_intermediate, dtype=torch.bfloat16, device=self.device)
        bufs["sorted_output"] = torch.zeros(disp_rows, H, dtype=torch.bfloat16, device=self.device)
        bufs["moe_output"] = torch.zeros(WB, H, dtype=torch.bfloat16, device=self.device)

        # Pre-build TMA descriptors (CPU-side driver API, not capturable)
        bufs["tma_desc_dispatched"] = create_tma_descriptor(bufs["dispatched_x"])
        bufs["tma_desc_intermediate"] = create_tma_descriptor(bufs["intermediate"])

        self._static_bufs[B] = bufs
        logger.info(f"MoESegment: pre-allocated static buffers for bucket_size={B} "
                     f"(disp_rows={disp_rows}, WB={WB})")

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

        B, H = hidden_states.shape
        W = self.world_size
        WB = W * B
        bufs = self._static_bufs[B]

        # 1. Copy input to padded buffer
        bufs["padded"].copy_(hidden_states)

        # 2. AllGather (PyNccl, graph-compatible)
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                bufs["all_tokens"], bufs["padded"],
                stream=torch.cuda.current_stream(self.device),
            )

        # 3. Router: manual matmul into pre-allocated buffer (avoid nn.Linear alloc)
        # router_weight is [E, H], we compute all_tokens @ router_weight.T
        # Use pre-allocated f32 buffer to avoid .float() allocation
        bufs["all_tokens_f32"].copy_(bufs["all_tokens"])
        torch.mm(bufs["all_tokens_f32"], self.router_weight_f32_t, out=bufs["router_logits"])
        if self.router_bias_f32 is not None:
            bufs["router_logits"].add_(self.router_bias_f32)

        # 4. Gate top-k softmax into pre-allocated buffers
        gate_topk_softmax_cuda(
            bufs["router_logits"],
            topk_indices=bufs["topk_indices"],
            topk_weights=bufs["topk_weights"],
            k=self.num_experts_per_tok,
        )

        # 5. Dispatch (count + prefix_sum + gather) into pre-allocated buffers
        # Note: dispatch_count_gather_cuda zeros expert_counts/counters internally.
        # We zero dispatched_x to ensure padded rows beyond actual tokens are zero
        # (important for TMA correctness when dispatched < BLOCK_M).
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

        # 6. WGMMA Stage 1 (gate + up + SwiGLU) — inplace with pre-built TMA desc
        fused_mxfp4_grouped_stage1_inplace(
            bufs["dispatched_x"], bufs["intermediate"],
            bufs["tma_desc_dispatched"], bufs["expert_offsets"],
            self.gate_ptrs, self.gate_scale_ptrs,
            self.up_ptrs, self.up_scale_ptrs,
            self.N_intermediate,
            self.s1_stride_weight_n, self.s1_stride_scale_n,
            gate_bias_ptrs=self.gate_bias_ptrs,
            up_bias_ptrs=self.up_bias_ptrs,
        )

        # 7. WGMMA Stage 2 (down projection) — inplace with pre-built TMA desc
        fused_mxfp4_grouped_stage2_inplace(
            bufs["intermediate"], bufs["sorted_output"],
            bufs["tma_desc_intermediate"], bufs["expert_offsets"],
            self.down_ptrs, self.down_scale_ptrs,
            H,
            self.s2_stride_weight_n, self.s2_stride_scale_n,
            down_bias_ptrs=self.down_bias_ptrs,
        )

        # 8. Reduce (weighted scatter-add back to original order)
        reduce_weighted_scatter_cuda(
            bufs["sorted_output"], bufs["topk_pos"], bufs["topk_weights"],
            WB, H, self.num_experts_per_tok,
            output=bufs["moe_output"],
        )

        # 9. Copy MoE result into global_results for AllReduce
        bufs["global_results"].copy_(bufs["moe_output"])

        # 10. AllReduce (PyNccl, graph-compatible)
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                bufs["global_results"], op=self.dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        # 11. Extract local rank's slice
        start = self.rank * B
        return {"moe_output": bufs["global_results"][start:start + B]}
