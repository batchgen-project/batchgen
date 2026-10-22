"""Compact ragged workspace helpers for GLM-5 grouped prefill and MoE decode."""

from __future__ import annotations

import torch

from batchgen.moe.dispatch_scatter_3d import (
    dispatch_scatter_ragged as _dispatch_scatter_ragged,
)


ROW_ALIGN = 64
QUANT_BLOCK = 128
_CAPACITY_BLOCK = 128
# Average-M tile hint handed to the grouped GEMM's persistent scheduler
# (`num_seq_per_group_avg`, which the compiled dispatch maps to TileM
# 16/32/64: <=16 -> 16, <=32 -> 32, else 64). The former pinned constant 64
# came from the 3D layout's mtp // E_local and oversized the M-tile for
# every compact-layout decode bucket; the real average rows per expert
# segment is global_rows * topk / num_global_experts, which is static per
# captured bucket (a host-side int at capture time, never a device
# readback), so a per-bucket value stays CUDA-graph safe. ROW_ALIGN = 64 is
# a multiple of every mapped TileM, so segment starts remain tile-aligned.
def gemm_tilem_avg(global_rows: int, topk: int, num_global_experts: int) -> int:
    """Average rows per expert segment for the grouped GEMM TileM selector."""
    if num_global_experts <= 0:
        raise ValueError("num_global_experts must be positive")
    return max(1, (int(global_rows) * int(topk)) // int(num_global_experts))


def _tilem_avg_override():
    """Batch-level sweep override `batchgen_debug.glm5_moe_tilem_avg` (int).

    Graphs are re-captured every decode round, so the override takes effect
    on the next round's capture; the eager path reads it per call. Absent or
    invalid means the per-bucket formula applies.
    """
    try:
        from batchgen.models.wrappers.attention import AttnWrapperBase
    except ImportError:
        return None
    debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
    if not isinstance(debug, dict):
        return None
    try:
        parsed = int(debug.get("glm5_moe_tilem_avg"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def recv_count_histogram_width(stride: int, block_m: int) -> int:
    """Accumulator width for :func:`accumulate_recv_count_histogram`.

    Slot 0 counts accumulated layer-steps, slot 1 the routed rows, and slots
    ``2 + t`` the expert-slots that needed ``t`` m-tiles. ``t`` cannot exceed
    ``ceil(stride / block_m)`` because an expert's rows are bounded by the
    receive stride.
    """
    if int(stride) <= 0 or int(block_m) <= 0:
        raise ValueError(
            f"recv-count histogram needs positive stride/block_m; got "
            f"{stride}/{block_m}"
        )
    return 2 + (int(stride) + int(block_m) - 1) // int(block_m) + 1


def recv_count_histogram_enabled() -> bool:
    """Batch-level diagnostic `batchgen_debug.glm52_recv_count_hist` (bool).

    Read on the host where the decode graph is built, exactly like
    :func:`_tilem_avg_override`: graphs are re-captured every decode round, so
    setting it takes effect on the next round's capture. It is read ONCE per
    capture and never per replay, so a disabled histogram costs nothing at all
    -- no buffer traffic and no extra launch on the measured path.
    """
    try:
        from batchgen.models.wrappers.attention import AttnWrapperBase
    except ImportError:
        return False
    debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
    if not isinstance(debug, dict):
        return False
    return bool(debug.get("glm52_recv_count_hist", False))


def accumulate_recv_count_histogram(hist, recv_count, block_m: int) -> None:
    """Fold one layer's ``recv_count`` into ``hist``, on device, no sync.

    Capture-safe: static shapes, integer arithmetic, and an ``index_add_``;
    nothing reads a device value on the host. Called once per routed layer
    while the diagnostic is on, so the accumulation is cumulative over every
    replay of the captured graph -- which is what makes the readback a
    distribution rather than a single step's snapshot.
    """
    import torch

    if hist.dtype != torch.int64:
        raise RuntimeError(
            f"recv-count histogram must be int64; got {hist.dtype}"
        )
    if recv_count.dim() != 1:
        raise RuntimeError(
            "recv_count must be the per-expert [E] row counts; got "
            f"{tuple(recv_count.shape)}"
        )
    counts = recv_count.to(torch.int64)
    tiles = torch.div(
        counts + (int(block_m) - 1), int(block_m), rounding_mode="floor"
    )
    tiles = torch.clamp(tiles, 0, hist.numel() - 3) + 2
    hist.index_add_(0, tiles, torch.ones_like(tiles))
    hist[0] += 1
    hist[1] += counts.sum()


def format_recv_count_histogram(hist, block_m: int) -> str:
    """One host readback of the cumulative histogram, for a log line.

    The only host-visible synchronization in this diagnostic. ``hist.tolist()``
    is a device-to-host copy, so it must run OUTSIDE every CUDA graph capture
    region: issuing it during a capture fails the capture outright
    (``cudaErrorStreamCaptureUnsupported``), and it would fail only with the
    diagnostic ON, i.e. exactly in the one configuration that uses it.

    The caller is ``Glm5MoEGraphSegment.setup_static_buffers``, which the graph
    manager runs before each round's warmup and capture. The accumulate half
    (:func:`accumulate_recv_count_histogram`) is the opposite: it is pure
    device work and belongs inside the capture.

    The capture check below locks that split into the contract rather than
    leaving it to the call site, in the same style as this package's
    side-stream and event guards.
    """
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "format_recv_count_histogram reads the histogram back to the host "
            "and must not be called inside a CUDA graph capture; call it from "
            "a pre-capture host path (setup_static_buffers) and leave only "
            "accumulate_recv_count_histogram inside the captured forward"
        )
    # gpu-sync: allow diagnostic readback, proven outside any capture region by
    # the guard above (batch-level glm52_recv_count_hist only; never on the
    # measured replay path)
    vals = [int(v) for v in hist.tolist()]
    layer_steps, rows = vals[0], vals[1]
    buckets = vals[2:]
    slots = sum(buckets)
    if slots == 0:
        return "recv-count histogram: no accumulation yet"
    over = sum(buckets[2:])
    parts = " ".join(
        f"{t}t={c}({100.0 * c / slots:.2f}%)"
        for t, c in enumerate(buckets)
        if c
    )
    return (
        f"recv-count histogram (block_m={block_m}): layer-steps={layer_steps} "
        f"expert-slots={slots} rows={rows} "
        f"mean_rows_per_slot={rows / slots:.2f} "
        f"over_one_tile={over}({100.0 * over / slots:.3f}%) | {parts}"
    )


def gemm_tilem_avg_effective(
    global_rows: int, topk: int, num_global_experts: int
) -> int:
    """The TileM average actually handed to the grouped GEMM: the batch-level
    debug override when present, else the per-bucket formula."""
    override = _tilem_avg_override()
    if override is not None:
        return override
    return gemm_tilem_avg(global_rows, topk, num_global_experts)

_ops_module = None


def _require_dispatch_module():
    from batchgen.moe.dispatch_scatter_3d import (
        require_dispatch_scatter_3d_kernels,
    )

    module = require_dispatch_scatter_3d_kernels()
    for symbol in (
        "dispatch_scatter_ragged",
        "reduce_weighted_scatter_bf16_ordered",
    ):
        if not hasattr(module, symbol):
            raise RuntimeError(
                "batchgen_kernels.moe._C_dispatch_scatter_3d has no "
                f"{symbol}; rebuild batchgen_kernels"
            )
    return module


def _require_ops_module():
    global _ops_module
    if _ops_module is not None:
        return _ops_module

    import batchgen_kernels

    module = batchgen_kernels.load_extension(
        "batchgen_kernels.moe._C_fp8_blockwise_ops"
    )
    if not hasattr(module, "act_quant_ragged"):
        raise RuntimeError(
            "batchgen_kernels.moe._C_fp8_blockwise_ops has no "
            "act_quant_ragged; rebuild batchgen_kernels"
        )
    _ops_module = module
    return module


def require_ragged_kernels():
    """Load every compact-ragged kernel required by grouped prefill."""
    return _require_dispatch_module(), _require_ops_module()


def ragged_row_capacity(
    max_global_tokens: int,
    topk: int,
    num_local_experts: int,
) -> int:
    """Return the static row capacity for 64-row-aligned expert segments."""
    live_rows = int(max_global_tokens) * int(topk)
    raw_rows = live_rows + int(num_local_experts) * (ROW_ALIGN - 1)
    return (
        (raw_rows + _CAPACITY_BLOCK - 1) // _CAPACITY_BLOCK
    ) * _CAPACITY_BLOCK


def make_quant_buffers(rows: int, dim: int, device: torch.device):
    """Allocate persistent FP8 activation and grouped-GEMM scale buffers."""
    if dim % QUANT_BLOCK != 0:
        raise ValueError(
            f"ragged quant dim must be a multiple of {QUANT_BLOCK}, got {dim}"
        )
    quantized = torch.empty(rows, dim, dtype=torch.uint8, device=device)
    scale = torch.zeros(
        dim // QUANT_BLOCK,
        rows,
        dtype=torch.float32,
        device=device,
    )
    return quantized, scale


def dispatch_scatter_ragged(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    act_buffer: torch.Tensor,
    expert_start: int,
    num_local_experts: int,
    expert_counts: torch.Tensor,
    expert_counters: torch.Tensor,
    cu_seqlens: torch.Tensor,
    topk_pos: torch.Tensor,
):
    """Dispatch through the current compact-ragged wrapper API."""
    return _dispatch_scatter_ragged(
        x,
        topk_indices,
        act_buffer,
        expert_start,
        num_local_experts,
        expert_counts,
        expert_counters,
        topk_pos,
        cu_seqlens,
    )


def act_quant_ragged(
    x: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Quantize live ragged rows into caller-owned output buffers."""
    _require_ops_module().act_quant_ragged(
        x,
        seqlens,
        cu_seqlens,
        output,
        scale,
    )


def deepep_ll_scale_repack(
    recv_scales: torch.Tensor,
    recv_count: torch.Tensor,
    out: torch.Tensor,
    span_align: int,
) -> None:
    """Repack DeepEP LL activation scales into the grouped GEMM's scale matrix.

    ``recv_scales`` is the exchange's [E, stride, hidden/128] view of its
    contiguous [E, hidden/128, stride] block; ``out`` is the [hidden/128,
    E * stride] matrix the grouped GEMM reads, expert e at column e * stride.
    Per expert only columns [0, ceil(recv_count[e] / span_align) * span_align)
    are copied -- with ``span_align`` = the GEMM's TileM that is exactly the
    set of columns its tiles load -- and the rest of ``out`` is left as is.
    """
    kernel = getattr(_require_ops_module(), "deepep_ll_scale_repack", None)
    if kernel is None:
        raise RuntimeError(
            "batchgen_kernels.moe._C_fp8_blockwise_ops has no "
            "deepep_ll_scale_repack; rebuild batchgen_kernels"
        )
    kernel(recv_scales, recv_count, out, int(span_align))


def reduce_weighted_scatter_bf16_ordered(
    expert_output: torch.Tensor,
    topk_pos: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    topk: int,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Call the ordered BF16 reducer without a module-import dependency."""
    dispatch, _ = require_ragged_kernels()
    return dispatch.reduce_weighted_scatter_bf16_ordered(
        expert_output,
        topk_pos,
        topk_indices,
        topk_weights,
        num_tokens,
        hidden_size,
        topk,
        output,
    )
