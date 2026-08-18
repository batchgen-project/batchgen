"""Structural tests for the GLM-5 compact ragged MoE layout (M1a-2).

CPU-only. These pin the *index arithmetic* that the CUDA kernels implement:

* ``dispatch_scatter_ragged``  — 64-aligned exclusive scan of the expert counts
* ``fp8_blockwise_gemm_kernel``/``fp8_blockwise_s1_kernel`` — x_scale tile index
  ``cu_seqlens[igroup] / kTileM + itile_m``
* ``act_quant_ragged``         — the row -> expert binary search
* ``Glm5MoEGraphBufferPool``   — per-bucket views of one max-bucket base

plus the backward-compatibility claim that makes the single CUDA index edit
safe: for the padded ``[E, mtp, dim]`` layout the new expression reproduces the
old ``igroup * mtp_tiles + itile_m`` exactly.

GPU numerics (quant+GEMM parity vs the 3D path) are validated separately on the
node; nothing here touches CUDA.
"""

import random

import pytest
import torch

from batchgen.models.glm.glm5.moe_ragged import (
    QUANT_BLOCK,
    ROW_ALIGN,
    ragged_row_capacity,
)

TILE_MS = (16, 32, 64)


# ---------------------------------------------------------------------------
# Python models of the device-side arithmetic
# ---------------------------------------------------------------------------

def _cu_seqlens(counts, align=ROW_ALIGN):
    """Mirror of count_tokens_ragged_kernel's single-thread exclusive scan."""
    out, acc = [], 0
    for c in counts:
        out.append(acc)
        acc += ((c + align - 1) // align) * align
    out.append(acc)
    return out


def _row_to_expert(cu, num_experts, row):
    """Mirror of act_quant_ragged_kernel's binary search (largest e with cu[e] <= row)."""
    lo, hi = 0, num_experts
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if cu[mid] <= row:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _random_counts(num_experts, nk, rng):
    """A routing outcome: non-negative counts summing to at most nk."""
    counts = [0] * num_experts
    for _ in range(nk):
        if rng.random() < 0.35:      # some routed pairs land on other ranks
            continue
        counts[rng.randrange(num_experts)] += 1
    return counts


# ---------------------------------------------------------------------------
# Capacity bound (spec fact 4: hard bound, never a silent regrow)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_experts", [1, 8, 32, 64])
@pytest.mark.parametrize("max_global,topk", [(32, 8), (256, 8), (2048, 8)])
def test_capacity_bounds_every_routing_outcome(num_experts, max_global, topk):
    capacity = ragged_row_capacity(max_global, topk, num_experts)
    nk = max_global * topk
    assert capacity % 128 == 0, "capacity must keep the x_scale row stride 16B aligned"
    assert capacity >= nk + num_experts * (ROW_ALIGN - 1)

    rng = random.Random(1234 + num_experts + max_global)
    for _ in range(50):
        counts = _random_counts(num_experts, nk, rng)
        cu = _cu_seqlens(counts)
        assert cu[-1] <= capacity, (counts, cu[-1], capacity)

    # The adversarial case the bound exists for: every routed pair local, and
    # every expert one row past an alignment boundary.
    worst = [0] * num_experts
    worst[0] = nk - (num_experts - 1)
    for e in range(1, num_experts):
        worst[e] = 1
    assert sum(worst) == nk
    assert _cu_seqlens(worst)[-1] <= capacity


def test_capacity_is_monotone_in_bucket():
    caps = [ragged_row_capacity(g, 8, 32) for g in (32, 64, 256, 1024, 2048)]
    assert caps == sorted(caps)


# ---------------------------------------------------------------------------
# The single CUDA edit: x_scale tile index
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_experts", [1, 8, 32, 64])
@pytest.mark.parametrize("mtp", [64, 128, 2048, 4096])
@pytest.mark.parametrize("tile_m", TILE_MS)
def test_padded_layout_scale_index_is_unchanged(num_experts, mtp, tile_m):
    """`cu_seqlens[e] / kTileM` == the old `e * mtp_tiles` for the padded layout.

    This is what keeps the edit safe for the existing padded callers
    (minimax_m25 MoE and its packed-QKV E=1 GEMM), which are not being ported.
    Old kernel: mtp_tiles = m_pad / (kTileM * num_group) with m_pad = E * mtp.
    """
    m_pad = num_experts * mtp
    mtp_tiles = m_pad // (tile_m * num_experts)
    cu_seqlens = [e * mtp for e in range(num_experts + 1)]
    for e in range(num_experts):
        assert cu_seqlens[e] % tile_m == 0
        assert cu_seqlens[e] // tile_m == e * mtp_tiles


@pytest.mark.parametrize("tile_m", TILE_MS)
def test_compact_segment_starts_are_tile_aligned(tile_m):
    rng = random.Random(7)
    for _ in range(200):
        counts = _random_counts(32, 512, rng)
        for start in _cu_seqlens(counts):
            assert start % ROW_ALIGN == 0
            assert start % tile_m == 0, "ROW_ALIGN must divide every supported TileM"


@pytest.mark.parametrize("tile_m", TILE_MS)
def test_scale_tiles_never_cross_into_the_next_expert(tile_m):
    """Every M-tile the GEMM reads for expert e lands inside e's own segment.

    The GEMM walks itile_m in [0, ceil(count/TileM)) and reads scale tile
    `cu_seqlens[e] // TileM + itile_m`. If that could reach
    `cu_seqlens[e+1] // TileM` it would silently consume the next expert's
    scales.
    """
    rng = random.Random(99)
    num_experts = 32
    for _ in range(200):
        counts = _random_counts(num_experts, 1024, rng)
        cu = _cu_seqlens(counts)
        for e, count in enumerate(counts):
            base = cu[e] // tile_m
            limit = cu[e + 1] // tile_m
            n_tiles = (count + tile_m - 1) // tile_m
            assert base + n_tiles <= limit, (e, count, base, n_tiles, limit)
            # ... and the rows those tiles cover stay inside the segment.
            assert cu[e] + n_tiles * tile_m <= cu[e + 1]


def test_scale_tile_reads_stay_in_capacity():
    num_experts, max_global, topk = 32, 2048, 8
    capacity = ragged_row_capacity(max_global, topk, num_experts)
    rng = random.Random(5)
    for _ in range(100):
        counts = _random_counts(num_experts, max_global * topk, rng)
        cu = _cu_seqlens(counts)
        for tile_m in TILE_MS:
            top_tile = max(
                cu[e] // tile_m + (counts[e] + tile_m - 1) // tile_m
                for e in range(num_experts)
            )
            assert top_tile * tile_m <= capacity


# ---------------------------------------------------------------------------
# act_quant_ragged row -> expert search
# ---------------------------------------------------------------------------

def test_row_to_expert_search_matches_segments():
    rng = random.Random(31337)
    num_experts = 17  # deliberately not a power of two
    for _ in range(60):
        counts = _random_counts(num_experts, 400, rng)
        cu = _cu_seqlens(counts)
        live, pad = 0, 0
        for row in range(cu[-1]):
            e = _row_to_expert(cu, num_experts, row)
            assert cu[e] <= row < cu[e + 1]
            local = row - cu[e]
            if local < counts[e]:
                live += 1
            else:
                pad += 1
        assert live == sum(counts)
        assert live + pad == cu[-1]


# ---------------------------------------------------------------------------
# Scatter: one compact row per routed local pair, no collisions
# ---------------------------------------------------------------------------

def test_scatter_rows_are_unique_and_inside_their_segment():
    rng = random.Random(2026)
    num_experts, expert_start, topk, n_tok = 32, 64, 8, 96
    nk = n_tok * topk
    topk_indices = [rng.randrange(0, 256) for _ in range(nk)]

    counts = [0] * num_experts
    for eid in topk_indices:
        loc = eid - expert_start
        if 0 <= loc < num_experts:
            counts[loc] += 1
    cu = _cu_seqlens(counts)

    counters = [0] * num_experts
    topk_pos = [-1] * nk
    for i, eid in enumerate(topk_indices):
        loc = eid - expert_start
        if not (0 <= loc < num_experts):
            continue
        topk_pos[i] = cu[loc] + counters[loc]
        counters[loc] += 1

    written = [p for p in topk_pos if p >= 0]
    assert len(written) == len(set(written)), "two routed slots collided on one row"
    assert len(written) == sum(counts)
    for i, p in enumerate(topk_pos):
        if p < 0:
            continue
        loc = topk_indices[i] - expert_start
        assert cu[loc] <= p < cu[loc] + counts[loc]
    assert counters == counts


# ---------------------------------------------------------------------------
# Graph buffer pool (C4)
# ---------------------------------------------------------------------------

def _make_pool(bucket_sizes):
    from batchgen.models.glm.glm5.moe_cuda_graph_segments import Glm5MoEGraphBufferPool

    return Glm5MoEGraphBufferPool(
        world_size=8,
        hidden_size=256,
        num_experts_per_tok=8,
        num_local_experts=32,
        intermediate_size=128,
        device=torch.device("cpu"),
        bucket_sizes=bucket_sizes,
        base_mtp=4096,  # dead parameter; must be accepted and ignored
    )


def test_pool_views_share_one_base_allocation():
    from batchgen.models.glm.glm5.moe_cuda_graph_segments import _MOE_BUCKET_DIM_FIELDS

    buckets = [8, 16, 32]
    pool = _make_pool(buckets)
    pool.setup()
    biggest = pool.get(max(buckets))
    for bucket in buckets:
        view = pool.get(bucket)
        for name in _MOE_BUCKET_DIM_FIELDS:
            small, big = getattr(view, name), getattr(biggest, name)
            assert small.data_ptr() == big.data_ptr(), f"{name} is not a view of the base"
            assert small.shape[0] <= big.shape[0]
            assert small.is_contiguous()


def test_pool_shared_fields_are_one_tensor():
    from batchgen.models.glm.glm5.moe_cuda_graph_segments import _MOE_SHARED_FIELDS

    buckets = [8, 32]
    pool = _make_pool(buckets)
    pool.setup()
    a, b = pool.get(8), pool.get(32)
    for name in _MOE_SHARED_FIELDS:
        assert getattr(a, name) is getattr(b, name), f"{name} must be shared across buckets"
    # cu_seqlens is now written per step, not a constant arange.
    assert a.cu_seqlens.shape == (33,)
    assert a.cu_seqlens.dtype == torch.int32
    assert int(a.cu_seqlens.abs().sum()) == 0


def test_pool_scale_buffers_are_per_bucket_transposed_and_zeroed():
    buckets = [8, 32]
    pool = _make_pool(buckets)
    pool.setup()
    for bucket in buckets:
        v = pool.get(bucket)
        cap = v.capacity
        assert cap == ragged_row_capacity(8 * bucket, 8, 32)
        assert v.dispatched_x.shape == (cap, 256)
        assert v.x_fp8.shape == (cap, 256)
        assert v.inter_fp8.shape == (cap, 128)
        # Transposed, GEMM-ready: [dim/128, capacity]
        assert v.x_scale.shape == (256 // QUANT_BLOCK, cap)
        assert v.inter_scale.shape == (128 // QUANT_BLOCK, cap)
        assert v.x_scale.is_contiguous() and v.inter_scale.is_contiguous()
        assert float(v.x_scale.abs().sum()) == 0.0
        assert float(v.inter_scale.abs().sum()) == 0.0
    assert pool.get(8).x_scale.data_ptr() != pool.get(32).x_scale.data_ptr()


def test_pool_field_coverage_guard_catches_a_stale_set():
    from batchgen.models.glm.glm5.moe_cuda_graph_segments import (
        _MOE_BUCKET_DIM_FIELDS,
        _MOE_REBUILT_FIELDS,
        _MOE_SHARED_FIELDS,
        _assert_moe_buffer_field_coverage,
    )

    full = _MOE_BUCKET_DIM_FIELDS + _MOE_SHARED_FIELDS + _MOE_REBUILT_FIELDS
    _assert_moe_buffer_field_coverage(full, "self-check")  # the real set passes

    with pytest.raises(RuntimeError, match="missing="):
        _assert_moe_buffer_field_coverage(full[:-1], "dropped-a-field")
    with pytest.raises(RuntimeError, match="unknown="):
        _assert_moe_buffer_field_coverage(full + ("not_a_field",), "typo")


def test_pool_rejects_a_bucket_larger_than_the_base():
    pool = _make_pool([8, 32])
    pool.setup()
    with pytest.raises(RuntimeError, match="largest bucket first"):
        pool._create_view(4096)


# ---------------------------------------------------------------------------
# Eager buffer manager
# ---------------------------------------------------------------------------

def test_eager_buffers_are_compact_and_regrow_only_at_job_boundaries():
    pytest.importorskip("triton")  # glm5/model.py imports triton at module scope
    from batchgen.models.glm.glm5.model import Glm5MoE3DBuffers

    E, H, N, topk, max_global = 32, 256, 128, 8, 256
    buf = Glm5MoE3DBuffers(
        E_local=E,
        max_global_bsz=max_global,
        H=H,
        N_inter=N,
        topk=topk,
        num_tokens_per_rank=32,
        device=torch.device("cpu"),
    )
    cap = ragged_row_capacity(max_global, topk, E)
    assert buf.capacity == cap
    assert not hasattr(buf, "max_tokens_padded"), "the per-expert stride must be gone"
    assert buf.dispatched_x.shape == (cap, H)
    assert buf.expert_out.shape == (cap, H)
    assert buf.intermediate.shape == (cap, N)
    assert buf.x_fp8.shape == (cap, H) and buf.x_fp8.dtype == torch.uint8
    assert buf.x_scale.shape == (H // QUANT_BLOCK, cap)
    assert buf.inter_scale.shape == (N // QUANT_BLOCK, cap)
    assert float(buf.x_scale.abs().sum()) == 0.0
    assert buf.cu_seqlens.shape == (E + 1,)
    assert buf.local_result_buffer.shape == (32, H)

    buf.resize_if_needed(max_global)          # no growth -> no-op
    assert buf.dispatched_x.shape == (cap, H)

    # n32 -> n2048 style job growth: the compact row space regrows with the
    # comm buffers (see resize_if_needed docstring re: m1a2_spec.md C5).
    bigger = max_global * 4
    buf.resize_if_needed(bigger)
    new_cap = ragged_row_capacity(bigger, topk, E)
    assert new_cap > cap
    assert buf.capacity == new_cap
    assert buf.dispatched_x.shape == (new_cap, H)
    assert buf.expert_out.shape == (new_cap, H)
    assert buf.intermediate.shape == (new_cap, N)
    assert buf.x_fp8.shape == (new_cap, H)
    assert buf.x_scale.shape == (H // QUANT_BLOCK, new_cap)
    assert float(buf.x_scale.abs().sum()) == 0.0
    assert buf.all_tokens.shape == (bigger, H)
    assert buf.topk_pos.shape == (bigger * topk,)


def test_eager_buffers_regrow_only_the_send_buffer():
    pytest.importorskip("triton")  # glm5/model.py imports triton at module scope
    from batchgen.models.glm.glm5.model import Glm5MoE3DBuffers

    buf = Glm5MoE3DBuffers(
        E_local=8,
        max_global_bsz=64,
        H=256,
        N_inter=128,
        topk=8,
        num_tokens_per_rank=4,
        device=torch.device("cpu"),
    )
    buf.resize_if_needed(64, num_tokens_per_rank=16)
    assert buf.padded.shape == (16, 256)
    assert buf.local_result_buffer.shape == (16, 256)
