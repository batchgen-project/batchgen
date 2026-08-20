"""CUDA-graph capturable GLM-5 MoE decode segments.

The GLM-5 MoE graph captures the full decode MoE module boundary:

    padded local tokens -> all_gather
      -> E=1 CuTe shared expert
      -> router -> rank padding mask -> dispatch_scatter_ragged
         -> FP8 blockwise S1/S3 -> reduce_weighted_scatter -> reduce_scatter
         on the graph capture stream
      -> add shared expert

All MoE work stays on the graph capture stream. Reusing a side stream across
MoE layers invalidates a whole-model graph when the second layer is added to
the same capture on CUDA 12.9 / NCCL 2.27. Eager decode keeps its existing
shared/routed overlap; only this whole-graph path is serialized.

The decoder-layer residual add remains eager in the caller because it is owned
by ``Glm5DecoderLayer.forward()``, not by ``Glm5MoE.forward()``.

Buffers use the compact ragged layout (``moe_ragged.py``): one row space of
``ragged_row_capacity(...)`` rows instead of a per-expert ``mtp`` slot table.
The capture-time shapes stay static per bucket; the only per-step dynamic data
is the contents of ``expert_counts`` / ``cu_seqlens``, which are device tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields as dataclass_fields
from typing import Dict, List

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.moe.dispatch_scatter_3d import reduce_weighted_scatter
from batchgen.moe.grouped_fp8_blockwise_moe import (
    grouped_fp8_blockwise_fused_s1,
    grouped_fp8_blockwise_s3,
)
from batchgen.moe.routing import FusedGateContext, gate_sigmoid_topk_cuda

from .moe_ragged import (
    GEMM_TILEM_AVG as _GLM5_MOE_GEMM_TILEM_AVG,
    act_quant_ragged,
    dispatch_scatter_ragged,
    make_quant_buffers,
    ragged_row_capacity,
)

logger = logging.getLogger(__name__)


def make_glm5_moe_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{layer_idx}_moe"


GLM5_MOE_ALL_GATHER_GRAPH_SEGMENT = "glm5_moe_all_gather"
GLM5_MOE_REDUCE_SCATTER_GRAPH_SEGMENT = "glm5_moe_reduce_scatter"


def pack_glm5_shared_scale_for_grouped_gemm(scale: torch.Tensor) -> torch.Tensor:
    """Pack one shared-expert block scale matrix as grouped-GEMM ``E=1``."""
    if scale.dim() != 2:
        raise ValueError(
            f"GLM-5 shared scale must be 2D [N/128, K/128], got {tuple(scale.shape)}"
        )
    n_blocks, k_blocks = scale.shape
    k_blocks_pad4 = (k_blocks + 3) // 4 * 4
    packed = torch.zeros(
        1,
        n_blocks,
        k_blocks_pad4,
        dtype=torch.float32,
        device=scale.device,
    )
    packed[0, :, :k_blocks].copy_(scale)
    return packed


@dataclass
class _Glm5MoEGraphBuffers:
    padded: torch.Tensor
    all_tokens: torch.Tensor
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
    x_fp8: torch.Tensor
    x_scale: torch.Tensor
    inter_fp8: torch.Tensor
    inter_scale: torch.Tensor
    routed_global_output: torch.Tensor
    local_moe_output: torch.Tensor
    shared_output: torch.Tensor
    cu_seqlens: torch.Tensor
    shared_seqlens: torch.Tensor
    shared_cu_seqlens: torch.Tensor
    shared_x_scale: torch.Tensor
    shared_inter_scale: torch.Tensor
    capacity: int


# ---------------------------------------------------------------------------
# Per-bucket buffer VIEWS (allocate once at the largest bucket). Same contract
# as the DSA statics (cuda_graph_segments.py): slices share storage with the
# base tensor, capture happens AFTER setup, and buckets replay one at a time.
#
# Three disjoint sets, and every dataclass field must be in exactly one of them
# (enforced by _assert_moe_buffer_field_coverage below).
#
#   * _BUCKET_DIM_FIELDS  — dim 0 scales with the bucket; take a leading slice.
#     Every initializer used for them is slice-invariant (empty / zeros / full).
#   * _SHARED_FIELDS      — bucket-independent (per-expert metadata); one tensor
#     handed to every bucket.
#   * _REBUILT_FIELDS     — must be built per bucket:
#       rank_ids/local_pos   values depend on bucket_size, not just length;
#       x_scale/inter_scale  are TRANSPOSED [K/128, capacity], so the bucket
#                            dimension is dim 1 — a dim-1 slice is not
#                            contiguous and the grouped GEMM requires a
#                            contiguous [K/128, m_pad] scale tensor. They are
#                            also zero-init-once buffers (see make_quant_buffers).
#       capacity             a plain int.
_MOE_BUCKET_DIM_FIELDS = (
    "padded",
    "all_tokens",
    "router_logits",
    "topk_indices",
    "topk_weights",
    "topk_masked_indices",
    "topk_masked_weights",
    "topk_negative_ones",
    "topk_zero_weights",
    "topk_pos",
    "dispatched_x",
    "intermediate",
    "expert_out",
    "x_fp8",
    "inter_fp8",
    "routed_global_output",
    "local_moe_output",
    "shared_output",
)

_MOE_SHARED_FIELDS = (
    "expert_counts",
    "expert_counters",
    "cu_seqlens",
)

_MOE_REBUILT_FIELDS = (
    "rank_ids",
    "local_pos",
    "x_scale",
    "inter_scale",
    "shared_seqlens",
    "shared_cu_seqlens",
    "shared_x_scale",
    "shared_inter_scale",
    "capacity",
)


def _assert_moe_buffer_field_coverage(covered, where: str) -> None:
    """Fail loudly if the buffer dataclass grew a field the view path ignores."""
    expected = {f.name for f in dataclass_fields(_Glm5MoEGraphBuffers)}
    covered = set(covered)
    missing = sorted(expected - covered)
    unknown = sorted(covered - expected)
    if missing or unknown:
        raise RuntimeError(
            f"{where}: _Glm5MoEGraphBuffers field coverage is stale — "
            f"missing={missing} unknown={unknown}. Add each new field to the "
            "sliced, shared, or rebuilt set."
        )


_assert_moe_buffer_field_coverage(
    _MOE_BUCKET_DIM_FIELDS + _MOE_SHARED_FIELDS + _MOE_REBUILT_FIELDS,
    "moe_cuda_graph_segments",
)


class Glm5MoEGraphBufferPool:
    """Shared static buffers for all GLM-5 MoE graph segments on one rank."""

    def __init__(
        self,
        *,
        world_size: int,
        hidden_size: int,
        num_experts_per_tok: int,
        num_local_experts: int,
        intermediate_size: int,
        device: torch.device,
        bucket_sizes: List[int],
        base_mtp: int = 0,
    ) -> None:
        """``base_mtp`` is DEAD as of M1a-2 (compact ragged layout).

        The dispatch/result buffers are no longer sized by a per-expert stride,
        so the value is ignored. The parameter only survives because
        ``batchgen_worker.py`` still forwards ``base_mtp=_GLM5_3D_MTP``; drop
        both together in the follow-up that may touch the worker.
        """
        if not bucket_sizes:
            raise ValueError("GLM-5 MoE graph requires at least one bucket size")
        self.world_size = int(world_size)
        self.hidden_size = int(hidden_size)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_local_experts = int(num_local_experts)
        self.intermediate_size = int(intermediate_size)
        self.device = device
        self.bucket_sizes = sorted({int(b) for b in bucket_sizes})
        self._base: Dict[str, torch.Tensor] = {}
        self._views: Dict[int, _Glm5MoEGraphBuffers] = {}

    def _capacity_for(self, bucket_size: int) -> int:
        return ragged_row_capacity(
            self.world_size * int(bucket_size),
            self.num_experts_per_tok,
            self.num_local_experts,
        )

    def setup(self) -> None:
        if self._base:
            return

        max_bucket = max(self.bucket_sizes)
        max_global = self.world_size * max_bucket
        capacity = self._capacity_for(max_bucket)
        nk = max_global * self.num_experts_per_tok
        d = self.device
        h = self.hidden_size
        n = self.intermediate_size
        k = self.num_experts_per_tok

        b = self._base
        b["padded"] = torch.zeros(max_bucket, h, dtype=torch.bfloat16, device=d)
        b["all_tokens"] = torch.zeros(max_global, h, dtype=torch.bfloat16, device=d)
        b["router_logits"] = torch.empty(max_global, 256, dtype=torch.float32, device=d)
        b["topk_indices"] = torch.empty(max_global, k, dtype=torch.int32, device=d)
        b["topk_weights"] = torch.empty(max_global, k, dtype=torch.float32, device=d)
        b["topk_masked_indices"] = torch.empty(max_global, k, dtype=torch.int32, device=d)
        b["topk_masked_weights"] = torch.empty(max_global, k, dtype=torch.float32, device=d)
        b["topk_negative_ones"] = torch.full((max_global, k), -1, dtype=torch.int32, device=d)
        b["topk_zero_weights"] = torch.zeros(max_global, k, dtype=torch.float32, device=d)
        b["expert_counts"] = torch.zeros(self.num_local_experts, dtype=torch.int32, device=d)
        b["expert_counters"] = torch.zeros(self.num_local_experts, dtype=torch.int32, device=d)
        # Written on device by dispatch_scatter_ragged every step (64-aligned
        # segment starts); no longer the constant arange of the mtp layout.
        b["cu_seqlens"] = torch.zeros(self.num_local_experts + 1, dtype=torch.int32, device=d)
        b["topk_pos"] = torch.full((nk,), -1, dtype=torch.int32, device=d)
        b["dispatched_x"] = torch.zeros(capacity, h, dtype=torch.bfloat16, device=d)
        b["intermediate"] = torch.empty(capacity, n, dtype=torch.bfloat16, device=d)
        b["expert_out"] = torch.empty(capacity, h, dtype=torch.bfloat16, device=d)
        b["x_fp8"] = torch.empty(capacity, h, dtype=torch.uint8, device=d)
        b["inter_fp8"] = torch.empty(capacity, n, dtype=torch.uint8, device=d)
        b["routed_global_output"] = torch.empty(max_global, h, dtype=torch.bfloat16, device=d)
        b["local_moe_output"] = torch.empty(max_bucket, h, dtype=torch.bfloat16, device=d)
        b["shared_output"] = torch.empty(max_bucket, h, dtype=torch.bfloat16, device=d)

        for bucket_size in self.bucket_sizes:
            self._create_view(bucket_size)

        total_bytes = sum(t.nelement() * t.element_size() for t in b.values())
        scale_bytes = sum(
            v.x_scale.nelement() * v.x_scale.element_size()
            + v.inter_scale.nelement() * v.inter_scale.element_size()
            + v.shared_x_scale.nelement() * v.shared_x_scale.element_size()
            + v.shared_inter_scale.nelement() * v.shared_inter_scale.element_size()
            for v in self._views.values()
        )
        logger.info(
            "Glm5MoEGraphBufferPool: allocated %.2f GiB base + %.2f GiB per-bucket "
            "scales (max_bucket=%d, world_size=%d, capacity=%d rows)",
            total_bytes / (1024**3),
            scale_bytes / (1024**3),
            max_bucket,
            self.world_size,
            capacity,
        )

    def get(self, bucket_size: int) -> _Glm5MoEGraphBuffers:
        self.setup()
        return self._views[int(bucket_size)]

    def _create_view(self, bucket_size: int) -> None:
        global_rows = self.world_size * bucket_size
        nk = global_rows * self.num_experts_per_tok
        capacity = self._capacity_for(bucket_size)
        b = self._base
        base_capacity = b["dispatched_x"].shape[0]
        if capacity > base_capacity:
            raise RuntimeError(
                f"GLM-5 MoE pool: bucket {bucket_size} needs {capacity} ragged rows "
                f"but the base allocation has {base_capacity}; setup() must run on "
                "the largest bucket first."
            )
        positions = torch.arange(global_rows, dtype=torch.int64, device=self.device)
        sliced = {
            "padded": b["padded"][:bucket_size],
            "all_tokens": b["all_tokens"][:global_rows],
            "router_logits": b["router_logits"][:global_rows],
            "topk_indices": b["topk_indices"][:global_rows],
            "topk_weights": b["topk_weights"][:global_rows],
            "topk_masked_indices": b["topk_masked_indices"][:global_rows],
            "topk_masked_weights": b["topk_masked_weights"][:global_rows],
            "topk_negative_ones": b["topk_negative_ones"][:global_rows],
            "topk_zero_weights": b["topk_zero_weights"][:global_rows],
            "topk_pos": b["topk_pos"][:nk],
            "dispatched_x": b["dispatched_x"][:capacity],
            "intermediate": b["intermediate"][:capacity],
            "expert_out": b["expert_out"][:capacity],
            "x_fp8": b["x_fp8"][:capacity],
            "inter_fp8": b["inter_fp8"][:capacity],
            "routed_global_output": b["routed_global_output"][:global_rows],
            "local_moe_output": b["local_moe_output"][:bucket_size],
            "shared_output": b["shared_output"][:bucket_size],
        }
        if set(sliced) != set(_MOE_BUCKET_DIM_FIELDS):
            raise RuntimeError(
                "GLM-5 MoE pool: sliced view set drifted from _MOE_BUCKET_DIM_FIELDS "
                f"(extra={sorted(set(sliced) - set(_MOE_BUCKET_DIM_FIELDS))}, "
                f"missing={sorted(set(_MOE_BUCKET_DIM_FIELDS) - set(sliced))})"
            )
        # x_fp8/inter_fp8 borrow the base storage; only the two transposed scale
        # tensors are per-bucket (dim 1 is the bucket dim — a dim-1 slice is not
        # contiguous, and the GEMM needs a contiguous [K/128, m_pad]).
        _, x_scale = make_quant_buffers(capacity, self.hidden_size, self.device)
        _, inter_scale = make_quant_buffers(capacity, self.intermediate_size, self.device)
        shared_x_scale = torch.zeros(
            self.hidden_size // 128,
            bucket_size,
            dtype=torch.float32,
            device=self.device,
        )
        shared_inter_scale = torch.zeros(
            self.intermediate_size // 128,
            bucket_size,
            dtype=torch.float32,
            device=self.device,
        )
        self._views[bucket_size] = _Glm5MoEGraphBuffers(
            **sliced,
            rank_ids=positions // bucket_size,
            local_pos=positions % bucket_size,
            expert_counts=b["expert_counts"],
            expert_counters=b["expert_counters"],
            cu_seqlens=b["cu_seqlens"],
            x_scale=x_scale,
            inter_scale=inter_scale,
            shared_seqlens=torch.tensor(
                [bucket_size],
                dtype=torch.int32,
                device=self.device,
            ),
            shared_cu_seqlens=torch.tensor(
                [0, bucket_size],
                dtype=torch.int32,
                device=self.device,
            ),
            shared_x_scale=shared_x_scale,
            shared_inter_scale=shared_inter_scale,
            capacity=capacity,
        )

    def release(self) -> None:
        self._views.clear()
        self._base.clear()


class Glm5MoEGraphSegment:
    """Graph-capturable full GLM-5 MoE decode module segment."""

    def __init__(
        self,
        moe,
        pool: Glm5MoEGraphBufferPool,
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
        self.num_experts_per_tok = int(moe.num_experts_per_tok)
        self.num_local_experts = int(moe.experts_per_rank)
        self.expert_start = int(moe.routed_expert_start_idx)
        self.intermediate_size = int(moe.config.moe_intermediate_size)
        self.routed_scaling_factor = float(moe.gate.routed_scaling_factor)

        if not getattr(moe, "_fp8_blockwise_ready", False):
            raise RuntimeError(
                f"Layer {moe.layer_idx}: GLM-5 MoE graph requires FP8 blockwise weights"
            )
        if comm is None:
            raise RuntimeError(f"Layer {moe.layer_idx}: GLM-5 MoE graph requires EP communicator")
        self._validate_shared_expert_graph_safe()

        self.gate_weight_bf16 = moe.gate.weight.detach().to(torch.bfloat16).contiguous()
        self.gate_bias_fp32 = moe.gate.e_score_correction_bias.detach().float().contiguous()
        self.router_context = FusedGateContext(
            self.gate_weight_bf16,
            router_bias=None,
            topk=self.num_experts_per_tok,
        )
        shared = self.moe.shared_experts
        self.shared_gate_weight = shared.cached_gate.unsqueeze(0).contiguous()
        self.shared_up_weight = shared.cached_up.unsqueeze(0).contiguous()
        self.shared_down_weight = shared.cached_down.unsqueeze(0).contiguous()
        self.shared_gate_scale = pack_glm5_shared_scale_for_grouped_gemm(
            shared.weight_dequant_scale["gate_proj.weight_scale_inv"]
        )
        self.shared_up_scale = pack_glm5_shared_scale_for_grouped_gemm(
            shared.weight_dequant_scale["up_proj.weight_scale_inv"]
        )
        self.shared_down_scale = pack_glm5_shared_scale_for_grouped_gemm(
            shared.weight_dequant_scale["down_proj.weight_scale_inv"]
        )
        self.routed_stream = type(moe)._routed_moe_stream
        if self.routed_stream is None:
            raise RuntimeError(
                f"Layer {moe.layer_idx}: GLM-5 MoE graph requires a CUDA side stream"
            )

    def _validate_shared_expert_graph_safe(self) -> None:
        shared = getattr(self.moe, "shared_experts", None)
        if shared is None:
            raise RuntimeError(
                f"Layer {self.moe.layer_idx}: GLM-5 MoE graph requires a shared expert module"
            )
        if getattr(shared, "persistent", True) is False:
            raise RuntimeError(
                f"Layer {self.moe.layer_idx}: GLM-5 MoE graph requires persistent "
                "shared expert weights; non-persistent shared expert forward loads "
                "and frees weights inside forward"
            )
        if getattr(shared, "is_fp8", False):
            missing = [
                name
                for name in ("cached_gate", "cached_up", "cached_down")
                if getattr(shared, name, None) is None
            ]
            if missing:
                raise RuntimeError(
                    f"Layer {self.moe.layer_idx}: GLM-5 MoE graph requires cached "
                    f"shared expert FP8 weights; missing {missing}"
                )

    def setup_static_buffers(self, bucket_size: int) -> None:
        if hasattr(self.comm, "disabled"):
            self.comm.disabled = False
        self.pool.setup()
        self.router_context.warmup(self.pool._base["all_tokens"])

    def release_static_buffers(self, bucket_size: int) -> None:
        self.pool.release()

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "padded": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "moe_output": TensorSpec(
                ("batch_size", self.hidden_size),
                torch.bfloat16,
            ),
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

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                bufs.all_tokens,
                padded,
                stream=torch.cuda.current_stream(self.device),
            )

        current_stream = torch.cuda.current_stream(self.device)
        shared_output = self._shared_expert_forward(bufs, padded)

        with torch.cuda.stream(current_stream):
            routed_global_output = self._routed_local_forward(
                bufs,
                bufs.all_tokens,
                rank_token_counts,
            )

            with self.comm.change_state(enable=True):
                self.comm.reduce_scatter(
                    bufs.local_moe_output,
                    routed_global_output,
                    stream=torch.cuda.current_stream(self.device),
                )

        bufs.local_moe_output.add_(shared_output)
        return {"moe_output": bufs.local_moe_output}

    def _routed_local_forward(
        self,
        bufs: _Glm5MoEGraphBuffers,
        all_tokens: torch.Tensor,
        rank_token_counts: torch.Tensor,
    ) -> torch.Tensor:
        global_rows = all_tokens.shape[0]
        if 192 <= global_rows <= 512:
            from batchgen_kernels.triton.glm5_router_gemm import (
                glm5_router_gemm,
            )

            glm5_router_gemm(
                all_tokens,
                self.gate_weight_bf16,
                bufs.router_logits,
            )
        else:
            self.router_context.router_forward(
                all_tokens,
                logits=bufs.router_logits,
            )
        gate_sigmoid_topk_cuda(
            bufs.router_logits,
            self.gate_bias_fp32,
            k=self.num_experts_per_tok,
            routed_scaling_factor=self.routed_scaling_factor,
            topk_indices=bufs.topk_indices,
            topk_weights=bufs.topk_weights,
        )

        valid_per_row = rank_token_counts[bufs.rank_ids]
        padding_mask = bufs.local_pos >= valid_per_row
        padding_mask_2d = padding_mask.unsqueeze(1).expand_as(bufs.topk_indices)
        torch.where(
            padding_mask_2d,
            bufs.topk_negative_ones,
            bufs.topk_indices,
            out=bufs.topk_masked_indices,
        )
        torch.where(
            padding_mask_2d,
            bufs.topk_zero_weights,
            bufs.topk_weights,
            out=bufs.topk_masked_weights,
        )

        # No memset (see eager path): TMA extents clip to seqlens[e] and
        # act_quant_ragged skips every row that is not a live token.
        expert_counts, cu_seqlens, topk_pos = dispatch_scatter_ragged(
            all_tokens,
            bufs.topk_masked_indices,
            bufs.dispatched_x,
            self.expert_start,
            self.num_local_experts,
            bufs.expert_counts,
            bufs.expert_counters,
            bufs.cu_seqlens,
            bufs.topk_pos,
        )

        self._fp8_blockwise_gemm_3d(bufs, expert_counts, cu_seqlens)

        return reduce_weighted_scatter(
            bufs.expert_out,
            topk_pos,
            bufs.topk_masked_weights,
            global_rows,
            self.hidden_size,
            self.num_experts_per_tok,
            output=bufs.routed_global_output,
        )

    def _shared_expert_forward(
        self,
        bufs: _Glm5MoEGraphBuffers,
        padded: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-only E=1 CuTe shared expert.

        DeepGEMM is numerically valid but its SM90 FP8 runtime cannot follow
        the routed post-S1 quantizer across repeated layers in one CUDA graph.
        Reuse the graph-stable grouped CuTe path with one synthetic expert.
        The routed FP8/intermediate scratch is safe to borrow here because the
        routed path overwrites it only after ``shared_output`` is complete.
        """
        bucket_size = padded.shape[0]
        act_quant_ragged(
            padded,
            bufs.shared_seqlens,
            bufs.shared_cu_seqlens,
            bufs.x_fp8[:bucket_size],
            bufs.shared_x_scale,
        )
        grouped_fp8_blockwise_fused_s1(
            bufs.x_fp8[:bucket_size].view(torch.float8_e4m3fn),
            bufs.shared_x_scale,
            self.shared_gate_weight.view(torch.float8_e4m3fn),
            self.shared_up_weight.view(torch.float8_e4m3fn),
            self.shared_gate_scale,
            self.shared_up_scale,
            bufs.shared_seqlens,
            bufs.shared_cu_seqlens,
            _GLM5_MOE_GEMM_TILEM_AVG,
            output=bufs.intermediate[:bucket_size],
        )
        act_quant_ragged(
            bufs.intermediate[:bucket_size],
            bufs.shared_seqlens,
            bufs.shared_cu_seqlens,
            bufs.inter_fp8[:bucket_size],
            bufs.shared_inter_scale,
        )
        grouped_fp8_blockwise_s3(
            bufs.inter_fp8[:bucket_size].view(torch.float8_e4m3fn),
            bufs.shared_inter_scale,
            self.shared_down_weight.view(torch.float8_e4m3fn),
            self.shared_down_scale,
            bufs.shared_seqlens,
            bufs.shared_cu_seqlens,
            _GLM5_MOE_GEMM_TILEM_AVG,
            output=bufs.shared_output,
        )
        return bufs.shared_output

    def _fp8_blockwise_gemm_3d(
        self,
        bufs: _Glm5MoEGraphBuffers,
        expert_counts: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> None:
        """Compact ragged S1/S3. Allocates nothing — every staging tensor is a
        pool buffer, so no block of this size is minted inside graph capture,
        and the scale tensors are already in the grouped GEMM's transposed
        layout (the 3D path needed a `.t().contiguous()` after each quant).
        """
        e = self.num_local_experts
        seqlens = expert_counts[:e]
        avg = _GLM5_MOE_GEMM_TILEM_AVG

        act_quant_ragged(bufs.dispatched_x, seqlens, cu_seqlens, bufs.x_fp8, bufs.x_scale)

        s1_result = grouped_fp8_blockwise_fused_s1(
            bufs.x_fp8.view(torch.float8_e4m3fn),
            bufs.x_scale,
            self.moe.fp8_gate_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_up_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_gate_ws3d,
            self.moe.fp8_up_ws3d,
            seqlens,
            cu_seqlens,
            avg,
            output=bufs.intermediate,
        )
        act_quant_ragged(s1_result, seqlens, cu_seqlens, bufs.inter_fp8, bufs.inter_scale)

        grouped_fp8_blockwise_s3(
            bufs.inter_fp8.view(torch.float8_e4m3fn),
            bufs.inter_scale,
            self.moe.fp8_down_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_down_ws3d,
            seqlens,
            cu_seqlens,
            avg,
            output=bufs.expert_out,
        )


class Glm5MoELocalGraphSegment(Glm5MoEGraphSegment):
    """Graph local MoE work with separately captured reusable collectives.

    The all-gather and reduce-scatter each have one graph shared by all 75
    layers. This avoids the H200 ceiling hit by 75 independent NCCL-bearing
    graphs while retaining graph replay for the complete MoE boundary.
    """

    separate_collective_graphs = True

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "routed_global_output": TensorSpec(
                (self.world_size * bucket_size, self.hidden_size),
                torch.bfloat16,
            ),
            "shared_output": TensorSpec(
                ("batch_size", self.hidden_size),
                torch.bfloat16,
            ),
        }

    def initialize_static_inputs(
        self,
        static_inputs: Dict[str, torch.Tensor],
        bucket_size: int,
    ) -> None:
        bufs = self.pool.get(bucket_size)
        bufs.all_tokens.zero_()
        bufs.padded.zero_()
        static_inputs["rank_token_counts"].fill_(bucket_size)

    def forward(
        self,
        *,
        rank_token_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bucket_size = max(self.pool.bucket_sizes)
        bufs = self.pool.get(bucket_size)
        shared_output = self._shared_expert_forward(bufs, bufs.padded)
        routed_global_output = self._routed_local_forward(
            bufs,
            bufs.all_tokens,
            rank_token_counts,
        )
        return {
            "routed_global_output": routed_global_output,
            "shared_output": shared_output,
        }


class Glm5MoEAllGatherGraphSegment:
    """One reusable all-gather graph shared by every GLM-5.2 MoE layer."""

    def __init__(self, pool: Glm5MoEGraphBufferPool, comm, device: torch.device):
        self.pool = pool
        self.comm = comm
        self.device = device

    def setup_static_buffers(self, bucket_size: int) -> None:
        if hasattr(self.comm, "disabled"):
            self.comm.disabled = False
        bufs = self.pool.get(bucket_size)
        bufs.padded.zero_()
        bufs.all_tokens.zero_()

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {}

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        bufs = self.pool.get(bucket_size)
        return {
            "all_tokens": TensorSpec(tuple(bufs.all_tokens.shape), torch.bfloat16),
        }

    def forward(self) -> Dict[str, torch.Tensor]:
        bucket_size = max(self.pool.bucket_sizes)
        bufs = self.pool.get(bucket_size)
        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                bufs.all_tokens,
                bufs.padded,
                stream=torch.cuda.current_stream(self.device),
            )
        return {"all_tokens": bufs.all_tokens}


class Glm5MoEReduceScatterGraphSegment:
    """One reusable reduce-scatter graph shared by every GLM-5.2 MoE layer."""

    def __init__(self, pool: Glm5MoEGraphBufferPool, comm, device: torch.device):
        self.pool = pool
        self.comm = comm
        self.device = device

    def setup_static_buffers(self, bucket_size: int) -> None:
        if hasattr(self.comm, "disabled"):
            self.comm.disabled = False
        bufs = self.pool.get(bucket_size)
        bufs.routed_global_output.zero_()
        bufs.local_moe_output.zero_()

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {}

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        bufs = self.pool.get(bucket_size)
        return {
            "local_moe_output": TensorSpec(
                tuple(bufs.local_moe_output.shape),
                torch.bfloat16,
            ),
        }

    def forward(self) -> Dict[str, torch.Tensor]:
        bucket_size = max(self.pool.bucket_sizes)
        bufs = self.pool.get(bucket_size)
        with self.comm.change_state(enable=True):
            self.comm.reduce_scatter(
                bufs.local_moe_output,
                bufs.routed_global_output,
                stream=torch.cuda.current_stream(self.device),
            )
        return {"local_moe_output": bufs.local_moe_output}
