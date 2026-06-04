"""CUDA-graph capturable Kimi-K2.5 MoE decode segment.

Mirrors ``glm5/moe_cuda_graph_segments.py`` but for K2.5's INT4-W4A16 Marlin
experts (GLM-5 uses FP8 blockwise). The graph captures the full decode MoE
module boundary:

    padded local tokens -> all_gather -> router -> rank padding mask
      -> dispatch_scatter_3d -> Marlin 3-stage (fused S1 + S3) -> reduce
      -> all_reduce -> local slice + shared expert

POIS decision: the shared expert runs **inline** in the captured forward (no
async side-stream); serializing it is accepted.

The graph-hostile ops of the eager ``KimiK25MoE._forward_decode`` are *designed
out* here, not refactored — the eager path stays as the fallback / compare
reference:
  * no ``buf.resize_if_needed`` — the pool is pre-sized for the max bucket;
  * Marlin C-ptrs are computed once at ``setup`` (no per-step ``data_ptr``
    compare, cf. ``model.py`` s3_C_ptrs recompute);
  * padding mask is applied with ``torch.where`` into static buffers (no
    ``.any()`` host sync, cf. ``model.py`` padding mask).

The decoder-layer residual add stays with the caller (layer segment / eager
``KimiK25DecoderLayer.forward``), matching GLM-5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.moe.dispatch_scatter_3d import dispatch_scatter_3d, reduce_weighted_scatter
from batchgen.moe.routing import gate_sigmoid_topk_cuda

logger = logging.getLogger(__name__)

_MTP_BLOCK = 64  # matches model._BLOCK_M (TMA constraint: global M >= BLOCK_M)


def make_k25_moe_graph_segment_name(layer_idx: int) -> str:
    return f"k25_layer_{layer_idx}_moe"


@dataclass
class _K25MoEGraphBuffers:
    padded: torch.Tensor
    all_tokens: torch.Tensor
    router_logits_bf16: torch.Tensor
    router_logits: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    topk_masked_indices: torch.Tensor
    topk_masked_weights: torch.Tensor
    topk_negative_ones: torch.Tensor
    topk_zero_weights: torch.Tensor
    rank_ids: torch.Tensor
    local_pos: torch.Tensor
    expert_counts: torch.Tensor
    expert_counters: torch.Tensor
    topk_pos: torch.Tensor
    dispatched_x: torch.Tensor
    intermediate: torch.Tensor
    expert_out: torch.Tensor
    routed_global_output: torch.Tensor
    local_moe_output: torch.Tensor
    # Marlin per-mtp tables (computed against this pool's buffers)
    expert_starts: torch.Tensor
    s1_fused_C_ptrs: torch.Tensor
    s3_C_ptrs: torch.Tensor
    s1_workspace: torch.Tensor
    s3_workspace: torch.Tensor
    max_tokens_padded: int
    max_marlin_m_tiles: int


class K25MoEGraphBufferPool:
    """Shared static buffers for all K2.5 MoE graph segments on one rank.

    Owns its own activation/routing buffers (decoupled from the eager
    ``KimiK25MoEBufferManager``) so capture never aliases the eager path. Only
    the per-mtp Marlin *output* tables (C-ptrs/workspaces) are pool-local; the
    Marlin *weight* pointers are taken from the model unchanged.
    """

    def __init__(
        self,
        *,
        world_size: int,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        num_local_experts: int,
        intermediate_size: int,
        device: torch.device,
        bucket_sizes: List[int],
    ) -> None:
        if not bucket_sizes:
            raise ValueError("K2.5 MoE graph requires at least one bucket size")
        self.world_size = int(world_size)
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_local_experts = int(num_local_experts)
        self.intermediate_size = int(intermediate_size)
        self.device = device
        self.bucket_sizes = sorted({int(b) for b in bucket_sizes})
        self._base: Dict[str, torch.Tensor] = {}
        self._views: Dict[int, _K25MoEGraphBuffers] = {}

    def setup(self) -> None:
        if self._base:
            return

        max_bucket = max(self.bucket_sizes)
        max_global = self.world_size * max_bucket
        mtp = self._round_up(max_global, _MTP_BLOCK)
        rows = self.num_local_experts * mtp
        nk = max_global * self.num_experts_per_tok
        d = self.device
        h = self.hidden_size
        n = self.intermediate_size
        e = self.num_local_experts
        k = self.num_experts_per_tok

        b = self._base
        b["padded"] = torch.zeros(max_bucket, h, dtype=torch.bfloat16, device=d)
        b["all_tokens"] = torch.zeros(max_global, h, dtype=torch.bfloat16, device=d)
        b["router_logits_bf16"] = torch.empty(max_global, self.num_experts, dtype=torch.bfloat16, device=d)
        b["router_logits"] = torch.empty(max_global, self.num_experts, dtype=torch.float32, device=d)
        b["topk_indices"] = torch.empty(max_global, k, dtype=torch.int32, device=d)
        b["topk_weights"] = torch.empty(max_global, k, dtype=torch.float32, device=d)
        b["topk_masked_indices"] = torch.empty(max_global, k, dtype=torch.int32, device=d)
        b["topk_masked_weights"] = torch.empty(max_global, k, dtype=torch.float32, device=d)
        b["topk_negative_ones"] = torch.full((max_global, k), -1, dtype=torch.int32, device=d)
        b["topk_zero_weights"] = torch.zeros(max_global, k, dtype=torch.float32, device=d)
        b["expert_counts"] = torch.zeros(e, dtype=torch.int32, device=d)
        b["expert_counters"] = torch.zeros(e, dtype=torch.int32, device=d)
        b["topk_pos"] = torch.full((nk,), -1, dtype=torch.int32, device=d)
        b["dispatched_x"] = torch.zeros(rows, h, dtype=torch.bfloat16, device=d)
        b["intermediate"] = torch.zeros(rows, n, dtype=torch.bfloat16, device=d)
        b["expert_out"] = torch.zeros(rows, h, dtype=torch.bfloat16, device=d)
        b["routed_global_output"] = torch.empty(max_global, h, dtype=torch.bfloat16, device=d)
        b["local_moe_output"] = torch.empty(max_bucket, h, dtype=torch.bfloat16, device=d)

        # Marlin per-mtp tables, computed against this pool's buffers.
        n_tiles_s1 = n // 256
        n_tiles_s3 = h // 256
        bpr_n = n * 2  # bytes per row of intermediate (bf16)
        bpr_h = h * 2  # bytes per row of expert_out (bf16)
        b["expert_starts"] = torch.arange(e, dtype=torch.int32, device=d) * mtp
        b["s1_fused_C_ptrs"] = torch.tensor(
            [b["intermediate"].data_ptr() + i * mtp * bpr_n for i in range(e)],
            dtype=torch.int64, device=d,
        )
        b["s3_C_ptrs"] = torch.tensor(
            [b["expert_out"].data_ptr() + i * mtp * bpr_h for i in range(e)],
            dtype=torch.int64, device=d,
        )
        b["s1_workspace"] = torch.zeros(e * (n_tiles_s1 + 17), dtype=torch.int32, device=d)
        b["s3_workspace"] = torch.zeros(e * (n_tiles_s3 + 17), dtype=torch.int32, device=d)
        self._mtp = mtp

        total_bytes = sum(t.nelement() * t.element_size() for t in b.values())
        logger.info(
            "K25MoEGraphBufferPool: allocated %.2f GiB "
            "(max_bucket=%d, world_size=%d, mtp=%d, rows=%d)",
            total_bytes / (1024**3), max_bucket, self.world_size, mtp, rows,
        )

        for bucket_size in self.bucket_sizes:
            self._create_view(bucket_size, mtp)

    def get(self, bucket_size: int) -> _K25MoEGraphBuffers:
        self.setup()
        return self._views[int(bucket_size)]

    def _create_view(self, bucket_size: int, mtp: int) -> None:
        global_rows = self.world_size * bucket_size
        nk = global_rows * self.num_experts_per_tok
        rows = self.num_local_experts * mtp
        b = self._base
        positions = torch.arange(global_rows, dtype=torch.int64, device=self.device)
        # Pigeonhole bound on Marlin M-tiles (16-row tiles), fixed per bucket.
        max_marlin_m_tiles = (min(global_rows, mtp) + 15) // 16
        self._views[bucket_size] = _K25MoEGraphBuffers(
            padded=b["padded"][:bucket_size],
            all_tokens=b["all_tokens"][:global_rows],
            router_logits_bf16=b["router_logits_bf16"][:global_rows],
            router_logits=b["router_logits"][:global_rows],
            topk_indices=b["topk_indices"][:global_rows],
            topk_weights=b["topk_weights"][:global_rows],
            topk_masked_indices=b["topk_masked_indices"][:global_rows],
            topk_masked_weights=b["topk_masked_weights"][:global_rows],
            topk_negative_ones=b["topk_negative_ones"][:global_rows],
            topk_zero_weights=b["topk_zero_weights"][:global_rows],
            rank_ids=positions // bucket_size,
            local_pos=positions % bucket_size,
            expert_counts=b["expert_counts"],
            expert_counters=b["expert_counters"],
            topk_pos=b["topk_pos"][:nk],
            dispatched_x=b["dispatched_x"][:rows],
            intermediate=b["intermediate"][:rows],
            expert_out=b["expert_out"][:rows],
            routed_global_output=b["routed_global_output"][:global_rows],
            local_moe_output=b["local_moe_output"][:bucket_size],
            expert_starts=b["expert_starts"],
            s1_fused_C_ptrs=b["s1_fused_C_ptrs"],
            s3_C_ptrs=b["s3_C_ptrs"],
            s1_workspace=b["s1_workspace"],
            s3_workspace=b["s3_workspace"],
            max_tokens_padded=mtp,
            max_marlin_m_tiles=max_marlin_m_tiles,
        )

    @staticmethod
    def _round_up(value: int, block: int) -> int:
        return ((value + block - 1) // block) * block

    def release(self) -> None:
        self._views.clear()
        self._base.clear()


class K25MoEGraphSegment:
    """Graph-capturable full K2.5 MoE decode module segment."""

    def __init__(
        self,
        moe,
        pool: K25MoEGraphBufferPool,
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
        self.num_experts_per_tok = int(moe.top_k)
        self.num_local_experts = int(moe.experts_per_rank)
        self.expert_start = int(moe.routed_expert_start_idx)
        self.intermediate_size = int(moe.moe_intermediate_size)
        self.routed_scaling_factor = float(moe.gate.routed_scaling_factor)

        if comm is None:
            raise RuntimeError(
                f"Layer {getattr(moe, '_layer_idx', '?')}: K2.5 MoE graph requires EP communicator"
            )
        if not getattr(moe, "_use_marlin_decode", False):
            raise RuntimeError(
                f"Layer {getattr(moe, '_layer_idx', '?')}: K2.5 MoE graph requires Marlin decode weights"
            )

        # Static gate inputs (weights are address-stable).
        self.gate_weight_t = moe.gate.weight.detach().to(torch.bfloat16).t().contiguous()
        self.gate_bias_fp32 = moe.gate.e_score_correction_bias.detach().float().contiguous()

        # Marlin weight pointer tables (weight-only; reused from the model).
        mw = moe._marlin_weights
        self._gate_B_ptrs = mw["gate_B_ptrs"]
        self._gate_scales_ptrs = mw["gate_scales_ptrs"]
        self._up_B_ptrs = mw["up_B_ptrs"]
        self._up_scales_ptrs = mw["up_scales_ptrs"]
        self._s3_B_ptrs = mw["s3_B_ptrs"]
        self._s3_scales_ptrs = mw["s3_scales_ptrs"]
        self._N = int(mw["N"])
        self._K = int(mw["K"])

        from batchgen.moe.marlin_grouped_moe import _load_module
        self._marlin_mod = _load_module()

    def setup_static_buffers(self, bucket_size: int) -> None:
        if hasattr(self.comm, "disabled"):
            self.comm.disabled = False
        self.pool.setup()

    def release_static_buffers(self, bucket_size: int) -> None:
        self.pool.release()

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "padded": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "moe_output": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16),
        }

    def forward(
        self,
        *,
        padded: torch.Tensor,
        rank_token_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bucket_size = padded.shape[0]
        bufs = self.pool.get(bucket_size)
        global_rows = self.world_size * bucket_size
        e = self.num_local_experts
        N = self._N  # moe_intermediate_size (2048)
        K = self._K  # hidden_size (7168)

        # 1) AllGather local tokens -> global token buffer (in-graph NCCL).
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                bufs.all_tokens,
                padded,
                stream=torch.cuda.current_stream(self.device),
            )

        # 2) Router: logits (static-output mm + cast) -> sigmoid/top-k.
        torch.mm(bufs.all_tokens, self.gate_weight_t, out=bufs.router_logits_bf16)
        bufs.router_logits.copy_(bufs.router_logits_bf16)
        gate_sigmoid_topk_cuda(
            bufs.router_logits,
            self.gate_bias_fp32,
            k=self.num_experts_per_tok,
            routed_scaling_factor=self.routed_scaling_factor,
            topk_indices=bufs.topk_indices,
            topk_weights=bufs.topk_weights,
        )

        # 3) Rank padding mask (graph-safe; no .any() host sync).
        valid_per_row = rank_token_counts[bufs.rank_ids]
        padding_mask = bufs.local_pos >= valid_per_row
        padding_mask_2d = padding_mask.unsqueeze(1).expand_as(bufs.topk_indices)
        torch.where(
            padding_mask_2d, bufs.topk_negative_ones, bufs.topk_indices,
            out=bufs.topk_masked_indices,
        )
        torch.where(
            padding_mask_2d, bufs.topk_zero_weights, bufs.topk_weights,
            out=bufs.topk_masked_weights,
        )

        # 4) 3D dispatch scatter into strided expert buffer.
        bufs.dispatched_x.zero_()
        expert_counts, topk_pos = dispatch_scatter_3d(
            bufs.all_tokens,
            bufs.topk_masked_indices,
            bufs.dispatched_x,
            self.expert_start,
            e,
            bufs.max_tokens_padded,
            bufs.expert_counts,
            bufs.expert_counters,
            bufs.topk_pos,
        )

        # 5) Marlin 3-stage expert compute (fused gate+up+SiLU, then down).
        #    S1: (E, N, K) fused gate+up+SiLU; S3: (E, K, N) down. Arg order
        #    matches eager _forward_decode (model.py).
        mod_m = self._marlin_mod
        n_tiles_s1 = N // 256
        n_tiles_s3 = K // 256
        mod_m.grouped_marlin_gemm_m16_s1(
            bufs.dispatched_x,
            self._gate_B_ptrs, self._up_B_ptrs, bufs.s1_fused_C_ptrs,
            self._gate_scales_ptrs, self._up_scales_ptrs,
            bufs.expert_starts, expert_counts,
            e, N, K, bufs.s1_workspace, n_tiles_s1, bufs.max_marlin_m_tiles,
        )
        mod_m.grouped_marlin_gemm_m16(
            bufs.intermediate, self._s3_B_ptrs, bufs.s3_C_ptrs,
            self._s3_scales_ptrs, bufs.expert_starts, expert_counts,
            e, K, N, bufs.s3_workspace,
            e, n_tiles_s3, bufs.max_marlin_m_tiles,
        )

        # 6) Reduce weighted scatter -> flat global output.
        bufs.routed_global_output.zero_()
        routed_global_output = reduce_weighted_scatter(
            bufs.expert_out,
            topk_pos,
            bufs.topk_masked_weights,
            global_rows,
            self.hidden_size,
            self.num_experts_per_tok,
            output=bufs.routed_global_output,
        )

        # 7) AllReduce routed outputs across ranks (in-graph NCCL).
        import torch.distributed as dist

        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                routed_global_output,
                op=dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        # 8) Local slice + inline shared expert (serialized, per POIS).
        start = self.rank * bucket_size
        bufs.local_moe_output.copy_(routed_global_output[start:start + bucket_size])
        bufs.local_moe_output.add_(self.moe.shared_experts(padded))
        return {"moe_output": bufs.local_moe_output}
