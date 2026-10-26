"""CUDA-graph capturable GLM-5 MoE decode segments.

The GLM-5 MoE graph captures the full decode MoE module boundary:

    padded local tokens -> all_gather -> router -> rank padding mask
      -> dispatch_scatter_ragged -> FP8 blockwise S1/S3 -> reduce_weighted_scatter
      -> reduce_scatter -> local slice + shared expert

The decoder-layer residual add remains eager in the caller because it is owned
by ``Glm5DecoderLayer.forward()``, not by ``Glm5MoE.forward()``.

Buffers use the compact ragged layout (``moe_ragged.py``): one row space of
``ragged_row_capacity(...)`` rows instead of a per-expert ``mtp`` slot table.
The capture-time shapes stay static per bucket; the only per-step dynamic data
is the contents of ``expert_counts`` / ``cu_seqlens``, which are device tensors.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, fields as dataclass_fields
from typing import Dict, List

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.moe.dispatch_scatter_3d import reduce_weighted_scatter
from batchgen.moe.grouped_fp8_blockwise_moe import (
    grouped_fp8_blockwise_fused_s1,
    grouped_fp8_blockwise_s3,
    prepare_grouped_fp8_s1_s3_tma,
)
from batchgen.moe.routing import (
    GATE_SIGMOID_TOPK_WARP_MAX_EXPERTS,
    FusedGateContext,
    gate_sigmoid_topk_cuda,
    gate_sigmoid_topk_warp_cuda,
)

from .moe_ragged import (
    ROW_ALIGN,
    gemm_tilem_avg_effective as _glm5_gemm_tilem_avg,
    act_quant_ragged,
    deepep_ll_scale_repack,
    dispatch_scatter_ragged,
    make_quant_buffers,
    ragged_row_capacity,
)

logger = logging.getLogger(__name__)


def make_glm5_moe_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{layer_idx}_moe"


def _graphed_router_split_k_kernel(rows: int):
    """The split-K router GEMM for ``rows``, or ``None`` outside its range.

    Both MoE decode paths route a row count that is static per captured bucket,
    so which router kernel runs is resolved once at capture time. Mid-size row
    counts belong to the Triton small-M split-K router (block_m=16); outside
    that range the WGMMA router is the better shape, and the split-K pair would
    be launching more programs than there is work for.

    The range is the kernel policy's own
    ``ROUTER_SPLIT_K_MIN_ROWS``/``ROUTER_SPLIT_K_MAX_ROWS``, read here rather
    than restated as literals, so the model layer and the kernel cannot
    disagree about which shapes the split-K path serves. Both are imported
    lazily because ``batchgen_kernels.triton`` imports Triton at package
    import and this module stays importable without it; a missing Triton
    therefore raises from this call rather than silently selecting the other
    kernel.
    """
    from batchgen_kernels.triton.glm5_router_gemm_config import (
        ROUTER_SPLIT_K_MAX_ROWS,
        ROUTER_SPLIT_K_MIN_ROWS,
    )

    if not ROUTER_SPLIT_K_MIN_ROWS <= rows <= ROUTER_SPLIT_K_MAX_ROWS:
        return None

    from batchgen_kernels.triton.glm5_router_gemm import glm5_router_gemm

    return glm5_router_gemm


# ---------------------------------------------------------------------------
# Shared-expert side-stream overlap (DeepEP decode path only).
#
# The DeepEP low-latency dispatch is a communication kernel that spends most of
# its duration waiting on peers, so the SMs it occupies do no arithmetic. The
# shared expert is the one piece of per-layer compute that depends on nothing
# the exchange produces -- it reads ``padded`` and the shared-expert weights
# only -- so issuing it on a second stream between the gate and the accumulate
# turns critical-path work into work hidden behind the dispatch.
#
# The stream and the events are PROCESS-lifetime and are created here, never
# inside a capture: a stream or event minted during ``torch.cuda.graph`` would
# be a capture-time resource that replay cannot reproduce. The ``wait_stream`` /
# ``record`` / ``wait_event`` calls, by contrast, are issued inside the capture
# on purpose -- there they are recorded as dependency edges of the captured DAG,
# and that is what makes the concurrency survive into every replay.
_SIDE_STREAMS: Dict[int, torch.cuda.Stream] = {}


def _glm5_moe_side_stream(device: torch.device) -> torch.cuda.Stream:
    """The one shared-expert side stream of this process, per CUDA device.

    Process-lifetime by construction: the graph pool is dropped and rebuilt
    around a mid-run prefill wave (``Glm5MoEGraphBufferPool.release``), and a
    stream minted per pool would leak one handle per rebuild while the point of
    the cache is that every MoE layer forks onto the SAME stream. Layers are
    strictly sequential and each joins before the next forks, so one stream is
    enough and two would only add fan-out the replay cannot use.
    """
    if device.type != "cuda":
        raise RuntimeError(
            "GLM-5 MoE shared-expert overlap requires a CUDA device, got "
            f"{device!r}")
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    stream = _SIDE_STREAMS.get(index)
    if stream is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GLM-5 MoE shared-expert side stream must exist before CUDA "
                "graph capture; it is being created inside a capture, which "
                "means a segment was constructed under torch.cuda.graph")
        stream = torch.cuda.Stream(device=index)
        _SIDE_STREAMS[index] = stream
    return stream


class _SharedExpertOverlap:
    """One MoE layer's fork/join pair for the shared expert.

    One event per layer rather than one per process: inside a single capture
    every layer records and waits once, so each event carries exactly one
    record/wait pair and the captured DAG needs no reasoning about re-recording
    a live event. Both the stream and the event are allocated here, at segment
    construction, which the graph manager runs before warmup and capture.
    """

    def __init__(self, device: torch.device, layer_idx: int) -> None:
        self.stream = _glm5_moe_side_stream(device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Layer {layer_idx}: GLM-5 MoE shared-expert overlap event "
                "must be created before CUDA graph capture")
        self.event = torch.cuda.Event()
        self.layer_idx = int(layer_idx)

    @contextlib.contextmanager
    def fork(self):
        main = torch.cuda.current_stream()
        # One ``wait_stream`` covers both orderings the side stream needs:
        #   * ``padded`` and the gate's writes were issued on the main stream;
        #   * the PREVIOUS layer's accumulate READ the shared-output buffer on
        #     the main stream.
        # The second is what makes a single process-wide output buffer correct
        # instead of one per layer: the next write to it cannot start before
        # the previous read of it has retired.
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            yield
        self.event.record(self.stream)

    def join(self) -> None:
        torch.cuda.current_stream().wait_event(self.event)


class _InlineSharedExpert:
    """No-stream stand-in: the shared expert runs in place, on the one stream.

    Reachable only where there is no CUDA stream to fork onto, i.e. the CPU
    contract tests. The arithmetic, the buffers and the order of the writes are
    the side-stream path's; only the stream topology is absent, so the two
    produce bitwise-identical output by construction.
    """

    @contextlib.contextmanager
    def fork(self):
        yield

    def join(self) -> None:
        return


@dataclass
class _Glm5MoEGraphBuffers:
    padded: torch.Tensor
    all_tokens: torch.Tensor
    router_logits: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
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
    cu_seqlens: torch.Tensor
    capacity: int
    # DeepEP low-latency mode only (``Glm5MoEGraphBufferPool.configure_deepep``).
    # ``ll_topk_idx`` holds the int64 expert ids the exchange requires, written
    # by the gate directly (``topk_indices`` is then an empty tensor);
    # ``ll_x_scale`` is the repacked activation scale the grouped GEMM
    # requires; ``ll_cu_seqlens`` is the constant uniform-stride segment table.
    # All three are empty tensors on the NCCL path.
    ll_topk_idx: torch.Tensor
    ll_x_scale: torch.Tensor
    ll_cu_seqlens: torch.Tensor
    # DeepEP low-latency mode only: the per-expert X/Y TMA descriptors of the
    # fused S1 and of the S3 GEMM ([2 * E_local, 128] bytes each) and their one
    # shared tile table, all written by ONE prepare launch per layer
    # (prepare_grouped_fp8_s1_s3_tma) and read by both GEMMs later on the same
    # stream. One set for every MoE layer and bucket: layers run in stream
    # order and each layer's S3 has read them before the next layer's prepare
    # rewrites them. Empty on the NCCL path.
    ll_s1_tma: torch.Tensor
    ll_s3_tma: torch.Tensor
    ll_tiles: torch.Tensor
    ll_cu_tiles: torch.Tensor
    # DeepEP low-latency mode only: the shared expert's output, produced on the
    # side stream and consumed by the accumulate on the main stream. A pool
    # buffer rather than a side-stream temporary because the consumer is a
    # DIFFERENT stream from the producer: a tensor minted on the side stream and
    # freed by the main stream is a cross-stream lifetime the caching allocator
    # has to be told about, while a process-lifetime buffer cannot be recycled
    # under either stream at all. Empty on the NCCL path.
    shared_out: torch.Tensor


# ---------------------------------------------------------------------------
# Per-bucket buffer VIEWS (allocate once at the largest bucket). Same contract
# as the DSA statics (cuda_graph_segments.py): slices share storage with the
# base tensors, and buckets replay one at a time with every non-constant field
# fully rewritten within its replay.
#
# Three disjoint sets; every dataclass field must be in exactly one of them
# (enforced by _assert_moe_buffer_field_coverage below).
#
#   * _MOE_BUCKET_DIM_FIELDS — dim 0 scales with the bucket; take a leading
#     slice. Every initializer used for them is slice-invariant.
#   * _MOE_SHARED_FIELDS     — bucket-independent per-expert metadata; one
#     tensor handed to every bucket.
#   * _MOE_REBUILT_FIELDS    — built per bucket:
#       x_scale/inter_scale  are TRANSPOSED [dim/128, capacity], so the bucket
#                            dimension is dim 1 — a dim-1 slice is not
#                            contiguous and the grouped GEMM requires a
#                            contiguous [dim/128, m_pad] scale tensor. They are
#                            also zero-init-once buffers (make_quant_buffers).
#       capacity             a plain int.
_MOE_BUCKET_DIM_FIELDS = (
    "padded",
    "all_tokens",
    "router_logits",
    "topk_indices",
    "topk_weights",
    "topk_pos",
    "dispatched_x",
    "intermediate",
    "expert_out",
    "x_fp8",
    "inter_fp8",
    "routed_global_output",
    "local_moe_output",
    "ll_topk_idx",
    "shared_out",
)

_MOE_SHARED_FIELDS = (
    "expert_counts",
    "expert_counters",
    "cu_seqlens",
    # DeepEP mode: the receive row space is E_local * num_ranks * tokens/rank,
    # which does not scale with the bucket, so both are bucket-independent.
    "ll_x_scale",
    "ll_cu_seqlens",
    "ll_s1_tma",
    "ll_s3_tma",
    "ll_tiles",
    "ll_cu_tiles",
)

_MOE_REBUILT_FIELDS = (
    "x_scale",
    "inter_scale",
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
        """``base_mtp`` is DEAD as of the compact ragged layout.

        The dispatch/result buffers are no longer sized by a per-expert
        stride, so the value is ignored. The parameter only survives because
        ``batchgen_worker.py`` still forwards ``base_mtp=resolve_glm5_3d_mtp(...)``;
        drop both together in a follow-up that may touch the worker.
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
        # DeepEP low-latency mode, installed by configure_deepep() before
        # setup(). ``deepep`` is the process-lifetime exchange; ``ll_stride``
        # is its per-local-expert row stride (num_ranks * tokens/rank).
        self.deepep = None
        self.ll_stride = 0
        self._deepep_request = None

    def configure_deepep(
        self,
        group,
        *,
        max_tokens_per_rank: int,
        num_experts_global: int,
    ) -> None:
        """Select the DeepEP low-latency exchange for this pool's buffers.

        Must run before ``setup()``: the receive layout is expert-major with a
        uniform per-expert stride, which is a different row space from the
        compact ragged layout, so the base allocation depends on it. Calling it
        twice with a different request is a bug, not a reconfiguration — the
        NVSHMEM buffer is process-lifetime and cannot be resized.
        """
        request = (int(max_tokens_per_rank), int(num_experts_global))
        if self._base:
            raise RuntimeError(
                "Glm5MoEGraphBufferPool.configure_deepep must run before "
                "setup(); the base buffers are already allocated")
        if self._deepep_request is not None and self._deepep_request != request:
            raise RuntimeError(
                "Glm5MoEGraphBufferPool: DeepEP was already configured for "
                f"{self._deepep_request}, cannot switch to {request}")
        if max(self.bucket_sizes) > int(max_tokens_per_rank):
            raise ValueError(
                f"DeepEP max_tokens_per_rank {max_tokens_per_rank} is below "
                f"the largest decode bucket {max(self.bucket_sizes)}")
        if int(num_experts_global) != self.world_size * self.num_local_experts:
            raise ValueError(
                f"DeepEP num_experts_global {num_experts_global} != "
                f"world_size {self.world_size} * local experts "
                f"{self.num_local_experts}")
        self._deepep_request = request
        self._deepep_group = group

    @property
    def deepep_enabled(self) -> bool:
        return self._deepep_request is not None

    def _capacity_for(self, bucket_size: int) -> int:
        return ragged_row_capacity(
            self.world_size * int(bucket_size),
            self.num_experts_per_tok,
            self.num_local_experts,
        )

    def setup(self) -> None:
        if self._base:
            return
        if self.deepep_enabled:
            self._setup_deepep()
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
        # DeepEP-only fields: empty on the NCCL path so the dataclass stays
        # total and every consumer fails on a shape, never on a None.
        b["ll_topk_idx"] = torch.empty(0, k, dtype=torch.int64, device=d)
        b["ll_x_scale"] = torch.empty(h // 128, 0, dtype=torch.float32, device=d)
        b["ll_cu_seqlens"] = torch.empty(0, dtype=torch.int32, device=d)
        b["ll_s1_tma"] = torch.empty(0, 128, dtype=torch.uint8, device=d)
        b["ll_s3_tma"] = torch.empty(0, 128, dtype=torch.uint8, device=d)
        b["ll_tiles"] = torch.empty(0, dtype=torch.int32, device=d)
        b["ll_cu_tiles"] = torch.empty(0, dtype=torch.int32, device=d)
        # The shared expert stays on the one stream here, so it keeps writing
        # the tensor it allocates and this buffer is never read.
        b["shared_out"] = torch.empty(0, h, dtype=torch.bfloat16, device=d)

        for bucket_size in self.bucket_sizes:
            self._create_view(bucket_size)

        total_bytes = sum(t.nelement() * t.element_size() for t in b.values())
        scale_bytes = sum(
            v.x_scale.nelement() * v.x_scale.element_size()
            + v.inter_scale.nelement() * v.inter_scale.element_size()
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

    def _setup_deepep(self) -> None:
        """Allocate the expert-major DeepEP receive layout.

        DeepEP LL packs each local expert's received tokens at rows
        ``[e * stride, e * stride + recv_count[e])`` of a
        ``[E_local, stride, hidden]`` buffer with ``stride = num_ranks *
        tokens/rank`` -- exactly the uniform mtp-stride activation layout the
        grouped FP8 blockwise GEMM already documents (``cu_seqlens[e] =
        e * stride``), so no compaction pass is needed. ``stride`` must stay
        TileM-aligned for the GEMM's per-expert scale-tile addressing; 64 is
        the largest TileM the dispatch maps.

        Dropped against the NCCL path: ``all_tokens`` (the EP all-gather is
        gone -- the router runs on local rows only), ``dispatched_x`` and
        ``x_fp8`` (the exchange delivers FP8 rows directly), ``topk_pos`` and
        ``routed_global_output`` (the LL combine applies the top-k weights and
        returns this rank's rows, replacing reduce_weighted_scatter +
        reduce_scatter).
        """
        from batchgen.moe.deepep_ll import get_low_latency_exchange

        max_tokens_per_rank, num_experts_global = self._deepep_request
        e = self.num_local_experts
        d = self.device
        h = self.hidden_size
        n = self.intermediate_size
        k = self.num_experts_per_tok

        self.deepep = get_low_latency_exchange(
            self._deepep_group,
            max_tokens_per_rank=max_tokens_per_rank,
            hidden=h,
            num_experts=num_experts_global,
            num_local_experts=e,
        )
        stride = int(self.deepep.expert_stride)
        if stride % ROW_ALIGN != 0:
            raise RuntimeError(
                f"DeepEP expert stride {stride} (num_ranks "
                f"{self.deepep.num_ranks} x {max_tokens_per_rank} tokens/rank) "
                f"is not a multiple of {ROW_ALIGN}; the grouped FP8 GEMM "
                "requires TileM-aligned segment starts")
        rows = e * stride
        self.ll_stride = stride

        b = self._base
        max_bucket = max(self.bucket_sizes)
        b["padded"] = torch.zeros(max_bucket, h, dtype=torch.bfloat16, device=d)
        # Router on LOCAL rows only: world_size x less gate work than the
        # all-gathered row space the NCCL path routes.
        b["router_logits"] = torch.empty(max_bucket, 256, dtype=torch.float32, device=d)
        # The gate writes DeepEP's int64 ids directly (``ll_topk_idx``); there
        # is no int32 index buffer on this path.
        b["topk_weights"] = torch.empty(max_bucket, k, dtype=torch.float32, device=d)
        b["ll_topk_idx"] = torch.empty(max_bucket, k, dtype=torch.int64, device=d)
        b["intermediate"] = torch.empty(rows, n, dtype=torch.bfloat16, device=d)
        b["expert_out"] = torch.empty(rows, h, dtype=torch.bfloat16, device=d)
        b["inter_fp8"] = torch.empty(rows, n, dtype=torch.uint8, device=d)
        b["local_moe_output"] = torch.empty(max_bucket, h, dtype=torch.bfloat16, device=d)
        # Shared-expert output, written on the side stream and read by the
        # accumulate on the main stream. ONE buffer for all MoE layers: they run
        # sequentially and each layer's join precedes the next layer's fork, so
        # no two layers hold it live at once (a per-layer buffer would cost
        # num_layers x this and displace KV pages for nothing).
        b["shared_out"] = torch.empty(max_bucket, h, dtype=torch.bfloat16, device=d)
        b["expert_counts"] = torch.zeros(e, dtype=torch.int32, device=d)
        b["expert_counters"] = torch.zeros(e, dtype=torch.int32, device=d)
        # Constant uniform-stride segment table; the GEMMs take their live row
        # counts from DeepEP's recv_count instead.
        b["ll_cu_seqlens"] = (
            torch.arange(e + 1, dtype=torch.int32, device=d) * stride
        ).contiguous()
        b["cu_seqlens"] = b["ll_cu_seqlens"]
        # Activation scales: DeepEP returns one contiguous [h/128, stride]
        # block per local expert, the GEMM wants a single [h/128, rows]
        # tensor, so the dispatch output is repacked into this buffer -- per
        # expert only the columns the GEMM's tiles can load
        # (deepep_ll_scale_repack); the rest keep whatever they held.
        _, ll_x_scale = make_quant_buffers(rows, h, d)
        b["ll_x_scale"] = ll_x_scale
        # Both GEMMs' descriptors + their shared tile table, written by one
        # prepare launch per layer (see _Glm5MoEGraphBuffers.ll_s1_tma).
        b["ll_s1_tma"] = torch.empty(2 * e, 128, dtype=torch.uint8, device=d)
        b["ll_s3_tma"] = torch.empty(2 * e, 128, dtype=torch.uint8, device=d)
        b["ll_tiles"] = torch.empty(e, dtype=torch.int32, device=d)
        b["ll_cu_tiles"] = torch.empty(e + 1, dtype=torch.int32, device=d)
        _, inter_scale = make_quant_buffers(rows, n, d)
        b["inter_scale"] = inter_scale
        empty_h = torch.empty(0, h, dtype=torch.bfloat16, device=d)
        b["_unused_bf16_h"] = empty_h
        b["_unused_u8_h"] = torch.empty(0, h, dtype=torch.uint8, device=d)
        b["_unused_i32"] = torch.empty(0, dtype=torch.int32, device=d)
        b["_unused_i32_k"] = torch.empty(0, k, dtype=torch.int32, device=d)

        for bucket_size in self.bucket_sizes:
            self._create_view_deepep(bucket_size, rows)

        total_bytes = sum(t.nelement() * t.element_size() for t in b.values())
        logger.info(
            "Glm5MoEGraphBufferPool[DeepEP]: allocated %.2f GiB "
            "(max_bucket=%d, world_size=%d, expert_stride=%d, rows=%d, "
            "rdma buffer %.0f MiB)",
            total_bytes / (1024**3),
            max_bucket,
            self.world_size,
            stride,
            rows,
            self.deepep.rdma_bytes / (1024**2),
        )

    def _create_view_deepep(self, bucket_size: int, rows: int) -> None:
        b = self._base
        self._views[int(bucket_size)] = _Glm5MoEGraphBuffers(
            padded=b["padded"][:bucket_size],
            # The router reads the local rows; there is no all-gathered space.
            all_tokens=b["padded"][:bucket_size],
            router_logits=b["router_logits"][:bucket_size],
            topk_indices=b["_unused_i32_k"],
            topk_weights=b["topk_weights"][:bucket_size],
            topk_pos=b["_unused_i32"],
            dispatched_x=b["_unused_bf16_h"],
            intermediate=b["intermediate"],
            expert_out=b["expert_out"],
            x_fp8=b["_unused_u8_h"],
            inter_fp8=b["inter_fp8"],
            routed_global_output=b["_unused_bf16_h"],
            local_moe_output=b["local_moe_output"][:bucket_size],
            ll_topk_idx=b["ll_topk_idx"][:bucket_size],
            shared_out=b["shared_out"][:bucket_size],
            expert_counts=b["expert_counts"],
            expert_counters=b["expert_counters"],
            cu_seqlens=b["ll_cu_seqlens"],
            ll_cu_seqlens=b["ll_cu_seqlens"],
            ll_x_scale=b["ll_x_scale"],
            ll_s1_tma=b["ll_s1_tma"],
            ll_s3_tma=b["ll_s3_tma"],
            ll_tiles=b["ll_tiles"],
            ll_cu_tiles=b["ll_cu_tiles"],
            x_scale=b["ll_x_scale"],
            inter_scale=b["inter_scale"],
            capacity=rows,
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
        sliced = {
            "padded": b["padded"][:bucket_size],
            "all_tokens": b["all_tokens"][:global_rows],
            "router_logits": b["router_logits"][:global_rows],
            "topk_indices": b["topk_indices"][:global_rows],
            "topk_weights": b["topk_weights"][:global_rows],
            "topk_pos": b["topk_pos"][:nk],
            "dispatched_x": b["dispatched_x"][:capacity],
            "intermediate": b["intermediate"][:capacity],
            "expert_out": b["expert_out"][:capacity],
            "x_fp8": b["x_fp8"][:capacity],
            "inter_fp8": b["inter_fp8"][:capacity],
            "routed_global_output": b["routed_global_output"][:global_rows],
            "local_moe_output": b["local_moe_output"][:bucket_size],
            # NCCL path: the exchange buffers and the side-stream staging
            # buffer do not exist, so both stay the empty base tensor.
            "ll_topk_idx": b["ll_topk_idx"],
            "shared_out": b["shared_out"],
        }
        if set(sliced) != set(_MOE_BUCKET_DIM_FIELDS):
            raise RuntimeError(
                "GLM-5 MoE pool: sliced view set drifted from _MOE_BUCKET_DIM_FIELDS "
                f"(extra={sorted(set(sliced) - set(_MOE_BUCKET_DIM_FIELDS))}, "
                f"missing={sorted(set(_MOE_BUCKET_DIM_FIELDS) - set(sliced))})"
            )
        # x_fp8/inter_fp8 borrow the base storage; only the two transposed scale
        # tensors are per-bucket (dim 1 is the bucket dim — a dim-1 slice is not
        # contiguous, and the GEMM needs a contiguous [dim/128, m_pad]).
        _, x_scale = make_quant_buffers(capacity, self.hidden_size, self.device)
        _, inter_scale = make_quant_buffers(capacity, self.intermediate_size, self.device)
        self._views[bucket_size] = _Glm5MoEGraphBuffers(
            **sliced,
            expert_counts=b["expert_counts"],
            expert_counters=b["expert_counters"],
            cu_seqlens=b["cu_seqlens"],
            ll_x_scale=b["ll_x_scale"],
            ll_cu_seqlens=b["ll_cu_seqlens"],
            ll_s1_tma=b["ll_s1_tma"],
            ll_s3_tma=b["ll_s3_tma"],
            ll_tiles=b["ll_tiles"],
            ll_cu_tiles=b["ll_cu_tiles"],
            x_scale=x_scale,
            inter_scale=inter_scale,
            capacity=capacity,
        )

    def release(self) -> None:
        """Drop the graph buffers. ``self.deepep`` is NOT dropped: the NVSHMEM
        buffer is process-lifetime (see ``DeepEPLowLatencyExchange``) and a
        rebuilt pool reuses the same exchange."""
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
        # ``glm5_deepep_ll`` is set on the MoE module by the GLM-5 initializer
        # from the server's --enable-deepep; absent or False keeps the NCCL
        # all_gather / reduce_scatter path, which stays the default.
        self.deepep_ll = bool(getattr(moe, "glm5_deepep_ll", False))
        if self.deepep_ll:
            self._validate_deepep_graph_safe()

        # Shared-expert fork/join, DeepEP path only: the NCCL path is the
        # default and the merged baseline, and a side stream there would be a
        # second unvalidated variable in the same measurement.
        #
        # Deliberately NOT capture-gated. The warmup iterations the graph
        # manager runs before each capture execute this same forked code, which
        # is what forces every lazy DeepGEMM/cuBLAS workspace to be allocated on
        # the side stream BEFORE capture; a capture-only fork would first touch
        # the side stream inside the capture. SGLang gates only its two
        # attention-internal forks on capture mode -- the shared-expert fork
        # this mirrors is ungated there as well.
        self._shared_overlap = (
            _SharedExpertOverlap(device, moe.layer_idx)
            if self.deepep_ll and device.type == "cuda"
            else _InlineSharedExpert()
        )

        self.gate_weight_bf16 = moe.gate.weight.detach().to(torch.bfloat16).contiguous()
        self.gate_bias_fp32 = moe.gate.e_score_correction_bias.detach().float().contiguous()
        self.router_context = FusedGateContext(
            self.gate_weight_bf16,
            router_bias=None,
            topk=self.num_experts_per_tok,
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
            # The DeepEP path hands the shared expert its staging buffer as
            # ``out=``, which only the fused gate/up chain accepts. Checked
            # here, before warmup and capture, instead of failing inside them.
            if getattr(shared, "cached_gate_up", None) is None:
                raise RuntimeError(
                    f"Layer {self.moe.layer_idx}: GLM-5 MoE graph requires the "
                    "shared expert's fused gate/up FP8 weight binding "
                    "(GLM5ExpertWrapper._bind_fused_gate_up)"
                )

    def _validate_deepep_graph_safe(self) -> None:
        """Fail loudly before capture for every assumption the DeepEP decode
        path makes. No silent NCCL fallback: --enable-deepep is explicit."""
        from batchgen.moe.deepep_ll import deepep_unavailable_reason

        moe = self.moe
        num_experts_global = self.world_size * self.num_local_experts
        max_tokens_per_rank = max(self.pool.bucket_sizes)
        reason = deepep_unavailable_reason(
            hidden=self.hidden_size,
            topk=self.num_experts_per_tok,
            num_experts=num_experts_global,
            num_ranks=self.world_size,
            max_tokens_per_rank=max_tokens_per_rank,
            use_fp8=True,
        )
        if reason is not None:
            raise RuntimeError(
                f"Layer {moe.layer_idx}: --enable-deepep requested for GLM-5 "
                f"decode MoE but {reason}")
        # DeepEP owns local experts [rank * E_local, (rank + 1) * E_local) and
        # returns recv_count in that order; GLM's own expert window must agree
        # or every routed row would land on the wrong weight block.
        expected_start = self.rank * self.num_local_experts
        if self.expert_start != expected_start:
            raise RuntimeError(
                f"Layer {moe.layer_idx}: DeepEP expects local experts to start "
                f"at rank * experts_per_rank = {expected_start}, but this rank "
                f"owns experts from {self.expert_start}")
        # O5: the offloading/mixed-compute path loads and frees expert weights
        # inside forward, so the expert-major receive layout has no stable
        # weight block to GEMM against.
        if bool(getattr(moe, "enable_ep_offloading", False)) or bool(
                getattr(moe, "enable_ep_with_offloading", False)):
            raise RuntimeError(
                f"Layer {moe.layer_idx}: --enable-deepep and EP expert "
                "offloading are mutually exclusive for GLM-5 decode")
        # The DeepEP path routes with the one-warp-per-row gate, which holds a
        # row's experts in registers.
        if num_experts_global > GATE_SIGMOID_TOPK_WARP_MAX_EXPERTS:
            raise RuntimeError(
                f"Layer {moe.layer_idx}: the DeepEP decode gate handles at most "
                f"{GATE_SIGMOID_TOPK_WARP_MAX_EXPERTS} routed experts, this model "
                f"has {num_experts_global}")
        if self.num_experts_per_tok not in (2, 4, 8, 16):
            raise RuntimeError(
                f"Layer {moe.layer_idx}: the DeepEP decode gate supports top-k "
                f"2, 4, 8 or 16, got {self.num_experts_per_tok}")
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                f"Layer {moe.layer_idx}: --enable-deepep requires an "
                "initialized torch.distributed process group")
        if dist.get_world_size() != self.world_size:
            raise RuntimeError(
                f"Layer {moe.layer_idx}: --enable-deepep requires the "
                f"torch.distributed world ({dist.get_world_size()}) to equal "
                f"the EP world ({self.world_size})")

    def setup_static_buffers(self, bucket_size: int) -> None:
        if hasattr(self.comm, "disabled"):
            self.comm.disabled = False
        if self.deepep_ll and not self.pool.deepep_enabled:
            import torch.distributed as dist

            self.pool.configure_deepep(
                dist.group.WORLD,
                max_tokens_per_rank=max(self.pool.bucket_sizes),
                num_experts_global=self.world_size * self.num_local_experts,
            )
        self.pool.setup()
        # DeepEP routes on local rows, so the router's largest input is the
        # bucket itself; the NCCL path routes the all-gathered row space.
        self.router_context.warmup(
            self.pool._base["padded"] if self.deepep_ll
            else self.pool._base["all_tokens"])

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

        if self.deepep_ll:
            return self._forward_deepep(
                padded=padded,
                rank_token_counts=rank_token_counts,
                bufs=bufs,
                bucket_size=bucket_size,
                global_rows=global_rows,
            )

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                bufs.all_tokens,
                padded,
                stream=torch.cuda.current_stream(self.device),
            )

        # This path routes the ALL-GATHERED row space, so the row count the
        # dispatch is resolved on is global_rows, static per captured bucket.
        split_k_router = _graphed_router_split_k_kernel(global_rows)
        if split_k_router is not None:
            split_k_router(
                bufs.all_tokens,
                self.gate_weight_bf16,
                bufs.router_logits,
            )
        else:
            self.router_context.router_forward(
                bufs.all_tokens,
                logits=bufs.router_logits,
            )
        # The rank padding mask is the gate's own epilogue. The all-gathered row
        # space is world_size contiguous blocks of bucket_size rows and block r
        # is live only up to rank_token_counts[r], so padding rows come out as
        # expert id -1 / weight 0 directly. This replaces four elementwise
        # launches per routed layer (the rank_token_counts gather, the local_pos
        # compare, and one torch.where per output) with the same values written
        # by the gate call that was already running.
        gate_sigmoid_topk_cuda(
            bufs.router_logits,
            self.gate_bias_fp32,
            k=self.num_experts_per_tok,
            routed_scaling_factor=self.routed_scaling_factor,
            topk_indices=bufs.topk_indices,
            topk_weights=bufs.topk_weights,
            rank_token_counts=rank_token_counts,
            rows_per_rank=bucket_size,
        )

        # No memset (see the eager path): TMA extents clip to seqlens[e] and
        # act_quant_ragged skips every row that is not a live token.
        expert_counts, cu_seqlens, topk_pos = dispatch_scatter_ragged(
            bufs.all_tokens,
            bufs.topk_indices,
            bufs.dispatched_x,
            self.expert_start,
            self.num_local_experts,
            bufs.expert_counts,
            bufs.expert_counters,
            bufs.cu_seqlens,
            bufs.topk_pos,
        )

        self._fp8_blockwise_gemm_3d(bufs, expert_counts, cu_seqlens)

        routed_global_output = reduce_weighted_scatter(
            bufs.expert_out,
            topk_pos,
            bufs.topk_weights,
            global_rows,
            self.hidden_size,
            self.num_experts_per_tok,
            output=bufs.routed_global_output,
        )

        import torch.distributed as dist

        with self.comm.change_state(enable=True):
            self.comm.reduce_scatter(
                bufs.local_moe_output,
                routed_global_output,
                op=dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        bufs.local_moe_output.add_(self.moe.shared_expert_forward(padded))
        return {"moe_output": bufs.local_moe_output}

    def _forward_deepep(
        self,
        *,
        padded: torch.Tensor,
        rank_token_counts: torch.Tensor,
        bufs: _Glm5MoEGraphBuffers,
        bucket_size: int,
        global_rows: int,
    ) -> Dict[str, torch.Tensor]:
        """Route-based decode MoE over the DeepEP low-latency exchange.

        Replaces, per routed layer, the NCCL all_gather (dispatch) and
        reduce_scatter (combine) plus the three expert-adjacent kernels they
        forced:

        * no all_gather    -- the router runs on this rank's ``bucket_size``
          rows instead of ``world_size * bucket_size``;
        * no dispatch_scatter_ragged -- the exchange delivers expert-major
          rows directly in the grouped GEMM's uniform-stride layout;
        * no act_quant_ragged on the activations -- the per-token 128-block
          FP8 cast is folded into the dispatch (``use_fp8=True``), which also
          halves the bytes on the wire;
        * no reduce_weighted_scatter and no reduce_scatter -- the LL combine
          performs the fp32-accumulated top-k weighted sum on the owning rank
          and writes this rank's rows.

        The shared expert is additionally forked onto a side stream for the
        span of the routed chain: fork right after the router GEMM (before the
        top-k), join before the accumulate, after the combine. It reads only
        ``padded``, so the exchange and the grouped GEMMs never wait on it.
        Dispatch and combine each launch as a send kernel and a separate
        receive kernel on the main stream (``split_phases=True``), so the
        shared expert, issued first, runs against the top-k, the id copy and
        the dispatch send instead of contending with one exchange kernel that
        spans the whole round trip. The shared expert itself is the fused
        four-launch chain (one gate/up GEMM, SiLU*up fused into the down
        input's quantization, no scale transposes) writing straight into the
        staging buffer, so it fits inside the dispatch window. The operand
        values and the order of writes are unchanged; only the stream the
        shared expert is issued on, its kernel decomposition and the
        exchange's kernel decomposition move, so the output is expected
        bitwise-identical to the in-line version.

        M-independence (padded rows stay inert): DeepEP leaves rows at or
        beyond ``recv_count[e]`` holding whatever the previous step left in the
        receive buffer. Both grouped GEMMs take their per-expert row count from
        ``recv_count``, so they neither read nor write those rows; the combine
        reads only the rows named by the dispatch handle's ``src_info`` /
        ``layout_range``. Stale rows therefore cannot reach the output, and the
        captured graph's result depends only on the live row count, never on
        the padded extent. The x-scale matrix follows the same rule at tile
        granularity: S1 loads, per expert, the scale columns of its
        ``ceil(recv_count[e] / TileM)`` row tiles (the partial last tile's
        extra columns only scale rows whose X loads are zero-filled and whose
        Y stores are clipped), the repack copies exactly those columns, and
        every column past them keeps a stale value no tile addresses.

        The side-stream staging buffer does not weaken that: it is one process
        buffer sliced to the bucket, the shared expert rewrites every row of
        the slice each step, and the accumulate reads that slice only -- rows
        beyond it keep a larger bucket's values and are unreachable. Within the
        slice, padded rows carry exactly the garbage they carried when the
        shared expert ran in line, because the rows it reads and writes are the
        same ones.
        """
        exchange = self.pool.deepep
        if exchange is None:
            raise RuntimeError(
                f"Layer {self.moe.layer_idx}: DeepEP decode path selected but "
                "the pool holds no exchange")
        stride = self.pool.ll_stride
        e = self.num_local_experts

        # Router on local rows only. The padding mask is the gate's epilogue,
        # as on the NCCL path, but the geometry is a SINGLE rank block: this
        # rank's row r is live only while r < rank_token_counts[rank].
        #
        # The kernel dispatch is the NCCL path's, resolved on the row count
        # THIS path routes: dropping the all_gather divides the router's rows
        # by the world size, so a bucket whose all-gathered space would have
        # taken the split-K router must still take it on its local rows alone.
        # Without this gate the whole decode range falls to the WGMMA router,
        # whose tile shape is the wrong one at a single rank's row count.
        split_k_router = _graphed_router_split_k_kernel(bucket_size)
        if split_k_router is not None:
            split_k_router(padded, self.gate_weight_bf16, bufs.router_logits)
        else:
            self.router_context.router_forward(
                padded, logits=bufs.router_logits
            )

        # FORK, right after the router GEMM and before the top-k: the shared
        # expert consumes ``padded`` and its own weights only, so from here it
        # is independent of everything routed. Issued this early it runs
        # against the top-k, the id copy and the dispatch SEND kernel; the
        # dispatch RECEIVE kernel is a separate launch (``split_phases``
        # below), so the routed chain does not hold the SMs across the whole
        # exchange round trip while the shared expert is in flight.
        with self._shared_overlap.fork():
            # The down projection writes straight into the pool's staging
            # buffer (no staging copy): the chain is quantize, ONE gate/up
            # GEMM, fused SiLU*up + quantize, down GEMM -- four launches on
            # the side stream. The bytes the accumulate reads are the ones the
            # GEMM produced, i.e. exactly what a separate bf16 -> bf16 copy of
            # the same shape would have delivered.
            shared_out = self.moe.shared_expert_forward(
                padded, out=bufs.shared_out)
            if shared_out is not bufs.shared_out:
                raise RuntimeError(
                    f"Layer {self.moe.layer_idx}: shared expert returned a "
                    f"tensor other than the staging buffer it was given "
                    f"({tuple(shared_out.shape)}/{shared_out.dtype} vs "
                    f"{tuple(bufs.shared_out.shape)}/{bufs.shared_out.dtype}); "
                    "the accumulate would read a buffer the side stream never "
                    "wrote.")

        # The top-k runs on this stream while the shared expert's x quant and
        # gate/up GEMM run on the side stream. A GEMM CTA is not placed on an
        # SM that a top-k CTA still occupies, so the top-k has to be FINISHED
        # by the time the x quant releases the GEMM. The one-warp-per-row
        # gate is built for that: one 32-thread CTA per row, one memory round
        # trip, a programmatic dependent launch right behind the router GEMM,
        # and the int64 ids DeepEP reads written straight into the exchange's
        # buffer (no int32 -> int64 copy launch). Same bytes as the
        # block-per-row gate; -1 padding rows stay unrouted.
        gate_sigmoid_topk_warp_cuda(
            bufs.router_logits,
            self.gate_bias_fp32,
            k=self.num_experts_per_tok,
            routed_scaling_factor=self.routed_scaling_factor,
            topk_indices=bufs.ll_topk_idx,
            topk_weights=bufs.topk_weights,
            rank_token_counts=rank_token_counts[self.rank : self.rank + 1],
            rows_per_rank=bucket_size,
        )

        # Dispatch and combine each run as a SEND kernel followed by a separate
        # RECEIVE kernel, both on this stream (``split_phases``; DeepEP's
        # ``return_recv_hook=True`` with the hook issued at once). Same bytes,
        # same reduction order as the single-kernel exchange.
        recv_x, recv_scales, recv_count, handle = exchange.dispatch_fp8(
            padded, bufs.ll_topk_idx, split_phases=True
        )
        # recv_x [e, stride, h] fp8 -> the uniform-stride activation the
        # grouped GEMM takes, expert i's rows at [i * stride, ...).
        x_fp8 = recv_x.view(e * stride, self.hidden_size)
        inter_fp8 = bufs.inter_fp8.view(torch.float8_e4m3fn)

        avg = _glm5_gemm_tilem_avg(
            global_rows,
            self.num_experts_per_tok,
            self.num_local_experts * self.world_size,
        )
        # ONE launch prepares both grouped GEMMs of this layer: their only
        # per-step input is recv_count (as the per-expert row extent of the
        # X/Y descriptors and as the tile counts), which S1 and S3 share, as
        # they share cu_seqlens and avg (=> TileM). S3 reads its descriptors
        # three launches later on this same stream.
        tma_plan = prepare_grouped_fp8_s1_s3_tma(
            x_fp8,
            bufs.intermediate,
            inter_fp8,
            bufs.expert_out,
            recv_count,
            bufs.ll_cu_seqlens,
            avg,
            s1_tma_desc=bufs.ll_s1_tma,
            s3_tma_desc=bufs.ll_s3_tma,
            tiles=bufs.ll_tiles,
            cu_tiles=bufs.ll_cu_tiles,
        )
        # recv_scales is logically [e, stride, h/128] but its last two
        # dimensions are a transposed view of a contiguous [e, h/128, stride]
        # allocation (csrc/deep_ep.cpp packs scales column-major for TMA).
        # The GEMM wants ONE contiguous [h/128, e * stride]. Its x-scale tiles
        # load, per expert, columns [0, ceil(recv_count[e] / TileM) * TileM)
        # of that expert's block and nothing else, so exactly that span is
        # repacked; columns past it are never addressed and keep stale values.
        deepep_ll_scale_repack(
            recv_scales, recv_count, bufs.ll_x_scale, tma_plan.tile_m)

        s1_result = grouped_fp8_blockwise_fused_s1(
            x_fp8,
            bufs.ll_x_scale,
            self.moe.fp8_gate_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_up_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_gate_ws3d,
            self.moe.fp8_up_ws3d,
            recv_count,
            bufs.ll_cu_seqlens,
            avg,
            output=bufs.intermediate,
            tma_plan=tma_plan,
        )
        act_quant_ragged(
            s1_result, recv_count, bufs.ll_cu_seqlens,
            bufs.inter_fp8, bufs.inter_scale,
        )
        grouped_fp8_blockwise_s3(
            inter_fp8,
            bufs.inter_scale,
            self.moe.fp8_down_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_down_ws3d,
            recv_count,
            bufs.ll_cu_seqlens,
            avg,
            output=bufs.expert_out,
            tma_plan=tma_plan,
        )

        exchange.combine(
            bufs.expert_out.view(e, stride, self.hidden_size),
            bufs.ll_topk_idx,
            bufs.topk_weights,
            handle,
            out=bufs.local_moe_output,
            split_phases=True,
        )

        # JOIN, at the first point that reads the shared output -- the combine
        # above is the last main-stream work the fork can hide behind. There
        # is no earlier cross-stream edge: the shared expert and the routed
        # GEMMs are ordered only by when they are issued.
        self._shared_overlap.join()
        bufs.local_moe_output.add_(bufs.shared_out)
        return {"moe_output": bufs.local_moe_output}

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
        # Static per captured bucket: routed_global_output is sized to the
        # bucket's global rows, so this resolves to a host int at capture.
        avg = _glm5_gemm_tilem_avg(
            bufs.routed_global_output.shape[0],
            self.num_experts_per_tok,
            self.num_local_experts * self.world_size,
        )

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
