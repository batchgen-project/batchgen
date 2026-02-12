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

    Inputs:  hidden_states [B, H] bf16
    Outputs: moe_output [B, H] bf16
    """

    def __init__(self, moe_decode, comm, world_size: int, rank: int, device):
        import torch.distributed as dist
        self.dist = dist

        self.router = moe_decode.router
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
        from batchgen.moe.routing import gate_topk_softmax_cuda
        from batchgen.moe.fused_wgmma_grouped import (
            fused_mxfp4_grouped_moe_forward_cuda_routing,
        )

        B, H = hidden_states.shape
        W = self.world_size

        # Per-bucket buffers — graph captures allocations at fixed addresses
        padded = torch.zeros(B, H, dtype=torch.bfloat16, device=self.device)
        all_tokens = torch.zeros(W * B, H, dtype=torch.bfloat16, device=self.device)
        global_results = torch.zeros(
            W * B, H, dtype=torch.bfloat16, device=self.device
        )

        padded.copy_(hidden_states)

        # AllGather (PyNccl, graph-compatible)
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded,
                stream=torch.cuda.current_stream(self.device),
            )

        # Route
        router_logits = self.router(all_tokens)
        topk_indices, topk_weights = gate_topk_softmax_cuda(
            router_logits, k=self.num_experts_per_tok,
        )

        # Grouped MoE (persistent experts only, 4 kernel launches)
        global_results[:W * B] = fused_mxfp4_grouped_moe_forward_cuda_routing(
            all_tokens, topk_indices, topk_weights,
            self.gate_ptrs, self.gate_scale_ptrs,
            self.up_ptrs, self.up_scale_ptrs,
            self.down_ptrs, self.down_scale_ptrs,
            self.gate_weight_ref, self.gate_scale_ref,
            self.down_weight_ref, self.down_scale_ref,
            num_experts=self.total_experts,
            expert_start=self.expert_start,
            num_local_experts=self.num_local_experts,
            gate_bias_ptrs=self.gate_bias_ptrs,
            up_bias_ptrs=self.up_bias_ptrs,
            down_bias_ptrs=self.down_bias_ptrs,
        )

        # AllReduce (PyNccl, graph-compatible)
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results, op=self.dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        # Extract local rank's slice
        start = self.rank * B
        return {"moe_output": global_results[start:start + B]}
