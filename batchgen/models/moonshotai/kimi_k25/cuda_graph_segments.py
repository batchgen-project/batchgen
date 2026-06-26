"""CUDA Graph capturable segments for Kimi K2.5 decode.

Segments:
  K25AttnSegment: Per-layer MLA attention (inlined, static page_table).
    Captured per-layer, replayed inside KimiK25DecoderLayer.forward().
    MoE stays eager with async shared expert overlap preserved.

  K25TpMoEGraphSegment: Per-layer TP-MoE decode (BATCHGEN_KIMI_TP_MOE=1) as a
    single capturable region — AllGather -> gate -> SGLang fused_marlin_moe ->
    AllReduce -> local slice + shared expert. Captured per-layer, replayed inside
    KimiK25MoE.forward() so the whole decode MoE step replays in ~1 launch. The
    kernel sequence is reused verbatim from the validated eager `_forward_decode_tp`;
    the only differences are the capture-safe transforms (collectives on the capture
    stream, pre-sized static buffers, shared expert serialized inline). EP MoE stays
    eager (no precedent / no marlin), matching the existing per-layer K2.5 design.

MLA forward is INLINED (not delegated to decoding_attn_mode_3_bf16) because:
  - CUDA graph requires static tensor addresses — the gpu_paged_kv_manager's internal
    block_table may be reallocated. We use the static page_table input instead.
  - Same approach as GPT-OSS FullAttnSegment (see gpt_oss_120b/cuda_graph_segments.py).
  - Zero overhead: same kernels, same number of launches, just different page_table pointer.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
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
        # so position = cache_seqlens - 1 (0-indexed).
        q_position_ids = (cache_seqlens - 1).clamp(min=0).unsqueeze(1).to(torch.int64)

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
            token_indices=q_position_ids.squeeze(-1).to(torch.int32),
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


# ============================================================================
# TP-MoE decode graph (BATCHGEN_KIMI_TP_MOE=1)
# ============================================================================

@dataclass
class _K25TpMoEBuffers:
    """Static per-bucket buffers for one rank, shared across all TP-MoE layers."""
    all_tokens: torch.Tensor   # [world_size * bucket, H] — AllGather destination
    moe_output: torch.Tensor   # [bucket, H] — this rank's combined MoE output


class K25TpMoEGraphBufferPool:
    """Shared static buffers for all K2.5 TP-MoE graph segments on one rank.

    Sizes the AllGather buffer to ``world_size * max_bucket`` up front so the
    captured graph never reallocates (the eager path's ``resize_if_needed`` is
    bypassed entirely here). One pool instance is shared across all 60 MoE layers.
    """

    def __init__(
        self,
        *,
        world_size: int,
        hidden_size: int,
        device: torch.device,
        bucket_sizes: List[int],
    ) -> None:
        if not bucket_sizes:
            raise ValueError("K2.5 TP-MoE graph requires at least one bucket size")
        self.world_size = int(world_size)
        self.hidden_size = int(hidden_size)
        self.device = device
        self.bucket_sizes = sorted({int(b) for b in bucket_sizes})
        self._base: Dict[str, torch.Tensor] = {}
        self._views: Dict[int, _K25TpMoEBuffers] = {}

    def setup(self) -> None:
        if self._base:
            return
        max_bucket = max(self.bucket_sizes)
        d = self.device
        h = self.hidden_size
        b = self._base
        # AllGather output is contiguous [world_size * sendcount, H]; sub-max
        # buckets slice the leading rows so per-rank blocks stay packed at the
        # bucket stride (rank r at rows [r*bucket, (r+1)*bucket)).
        b["all_tokens"] = torch.zeros(
            self.world_size * max_bucket, h, dtype=torch.bfloat16, device=d
        )
        b["moe_output"] = torch.zeros(max_bucket, h, dtype=torch.bfloat16, device=d)

        total_bytes = sum(t.nelement() * t.element_size() for t in b.values())
        logger.info(
            "K25TpMoEGraphBufferPool: allocated %.2f MiB (max_bucket=%d, world_size=%d)",
            total_bytes / (1024 ** 2),
            max_bucket,
            self.world_size,
        )

        for bucket in self.bucket_sizes:
            self._views[bucket] = _K25TpMoEBuffers(
                all_tokens=b["all_tokens"][: self.world_size * bucket],
                moe_output=b["moe_output"][:bucket],
            )

    def get(self, bucket_size: int) -> _K25TpMoEBuffers:
        self.setup()
        return self._views[int(bucket_size)]

    def release(self) -> None:
        self._views.clear()
        self._base.clear()


class K25TpMoEGraphSegment:
    """Graph-capturable TP-MoE decode module segment for Kimi K2.5.

    Captured region (numerically identical to ``KimiK25MoE._forward_decode_tp``):
        AllGather(local zero-padded tokens) -> CUDA gate -> SGLang GPTQ-Marlin int4
        fused_marlin_moe over all experts (this rank's TP slice) -> AllReduce(SUM)
        -> slice this rank's tokens + shared expert.

    Capture-safety transforms vs the eager path (the ONLY differences):
      - NCCL collectives submitted on ``current_stream`` (the capture stream), not
        ``default_stream`` (the eager legacy stream), so they record into the graph.
      - AllGather destination is a pre-sized static pool buffer (no resize under capture).
      - Shared expert runs serial-inline on the capture stream (no side-stream overlap;
        forked side streams have no capture precedent in BatchGen — GLM-5 serializes too).
      - No logging inside the captured region.
    Padded rows are exact zeros (the replay harness zero-fills the static input tail),
    and bias-free INT4 experts map 0 -> 0, so padding never needs an explicit mask.
    """

    def __init__(
        self,
        moe,
        pool: K25TpMoEGraphBufferPool,
        comm,
        *,
        world_size: int,
        rank: int,
        device: torch.device,
    ) -> None:
        self.moe = moe
        self.pool = pool
        self.comm = comm
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.device = device
        self.hidden_size = int(moe.hidden_size)
        self.num_experts = int(moe.num_experts)

        if comm is None:
            raise RuntimeError(
                f"Layer {getattr(moe, '_layer_idx', '?')}: K2.5 TP-MoE graph requires "
                "an EP communicator"
            )
        if getattr(moe, "_tp_w13_marlin", None) is None:
            raise RuntimeError(
                f"Layer {getattr(moe, '_layer_idx', '?')}: K2.5 TP-MoE graph requires "
                "the Marlin int4 TP weights (_tp_w13_marlin)"
            )

    def setup_static_buffers(self, bucket_size: int) -> None:
        if hasattr(self.comm, "disabled"):
            self.comm.disabled = False
        self.pool.setup()
        # Pre-warm the SGLang kernel import/server-args seed and the module-level
        # Marlin workspace BEFORE capture so the captured forward allocates nothing
        # host-side and the workspace pointer is baked into the graph.
        from batchgen.models.moonshotai.kimi_k25.model import (
            _get_marlin_workspace,
            _load_fused_marlin_moe,
        )
        _load_fused_marlin_moe()
        _get_marlin_workspace(self.device)

    def release_static_buffers(self, bucket_size: int) -> None:
        self.pool.release()

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        # Local (this rank's) decode tokens, zero-padded to the bucket. fill_value
        # 0.0 means the replay harness zero-fills the [num_tokens:bucket] tail.
        return {"padded": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16)}

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {"moe_output": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16)}

    def forward(self, *, padded: torch.Tensor) -> Dict[str, torch.Tensor]:
        import torch.distributed as dist

        from batchgen.models.moonshotai.kimi_k25.model import (
            _get_marlin_workspace,
            _load_fused_marlin_moe,
        )

        fused_marlin_moe = _load_fused_marlin_moe()
        moe = self.moe
        bucket_size = padded.shape[0]
        bufs = self.pool.get(bucket_size)
        all_tokens = bufs.all_tokens
        num_global = self.world_size * bucket_size
        h = self.hidden_size

        # 1) AllGather local (zero-padded) tokens across ranks — capture stream.
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded,
                stream=torch.cuda.current_stream(self.device),
            )

        # 2) CUDA gate on the gathered buffer (global expert ids, scaled weights).
        topk_idx, topk_weight = moe.gate(all_tokens.view(num_global, 1, h))

        # 3) One SGLang GPTQ-Marlin int4 MoE call over all experts (this rank's TP slice).
        workspace = _get_marlin_workspace(self.device)
        tp_out = fused_marlin_moe(
            hidden_states=all_tokens,
            w1=moe._tp_w13_marlin,
            w2=moe._tp_w2_marlin,
            w1_scale=moe._tp_w13_scale_marlin,
            w2_scale=moe._tp_w2_scale_marlin,
            gating_output=topk_weight,
            topk_weights=topk_weight,
            topk_ids=topk_idx.to(torch.int32),
            global_num_experts=self.num_experts,
            expert_map=None,
            g_idx1=None, g_idx2=None,
            sort_indices1=None, sort_indices2=None,
            w1_zeros=None, w2_zeros=None,
            workspace=workspace,
            num_bits=4,
            is_k_full=True,
            inplace=False,
            routed_scaling_factor=None,
        )

        # 4) AllReduce(SUM) → full MoE output for all gathered tokens — capture stream.
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                tp_out, op=dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        # 5) Slice this rank's tokens + shared expert (serialized inline on capture stream).
        start = self.rank * bucket_size
        moe_output = bufs.moe_output
        moe_output.copy_(tp_out[start:start + bucket_size])
        moe_output.add_(moe.shared_experts(padded))
        return {"moe_output": moe_output}
