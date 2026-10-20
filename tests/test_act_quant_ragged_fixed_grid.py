"""act_quant_ragged on a grid that is not sized by the padded capacity.

The kernel used to launch one 128-thread block per row of the padded buffer
(``grid = max_rows``); it now launches one resident wave (or one block per 16
rows of a capacity larger than 16 waves) and walks the experts' 64-aligned
spans. These tests pin the contract that change
must keep, against a reference of the one-block-per-row behaviour:

* the write set is unchanged -- row ``r`` is written iff ``r`` is in
  ``[0, max_rows)`` and ``[cu[0], cu[E])`` and ``r - cu[o] < align64(seqlens[o])``
  for ``o`` = the binary-search owner of ``r`` (rightmost ``e`` with
  ``cu[e] <= r``); written rows carry act_quant_3d bytes/scales on valid rows
  and the zero-input encoding on pad rows;
* every other byte of ``y`` and ``scale`` keeps its sentinel (the kernel never
  pre-zeroes), and no unwritten row of ``x`` is read (those rows hold inf/NaN);
* the grid is not sized by ``max_rows`` (one resident wave up to 16 waves of
  capacity, one block per 16 rows beyond), and a captured graph replays
  correctly when the device-side seqlens change after capture.

Comparisons are byte-for-byte (``y`` as uint8, ``scale`` as int32 bits).
GPU only; every test skips cleanly without CUDA or the extension.
"""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest
import torch

ALIGN = 64
BLOCK = 128
Y_SENTINEL = 0xAB
# A NaN with a payload: a float compare could never match it, a bit compare does.
SCALE_SENTINEL_BITS = 0x7FC0DEAD

# Real decode shapes of the GLM-5.2 MoE path (config.py: hidden_size 6144,
# moe_intermediate_size 2048; 256 routed experts over 8 EP ranks = 32 local).
HIDDEN = 6144
INTERMEDIATE = 2048
LOCAL_EXPERTS = 32
# DeepEP LL uniform-stride receive layout: cu[e] = e * stride with
# stride = num_ranks * max_tokens_per_rank = 8 * 128.
DEEPEP_STRIDE = 1024


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.fixture(scope="module")
def ops_mod():
    _require_cuda()
    name = "batchgen_kernels.moe._C_fp8_blockwise_ops"
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any load failure means unavailable
        pytest.skip(f"{name} unavailable: {exc}")
    if not hasattr(mod, "act_quant_ragged"):
        pytest.skip(f"{name} has no act_quant_ragged")
    return mod


def _align64(v: int) -> int:
    return (v + ALIGN - 1) // ALIGN * ALIGN


def _compact_cu(counts):
    cu = [0]
    for c in counts:
        cu.append(cu[-1] + _align64(c))
    return cu


# ──────────────────────────────────────────────────────────────────────────────
# Reference: the one-block-per-row kernel, evaluated for every row at once
# ──────────────────────────────────────────────────────────────────────────────

def _reference_write_set(seqlens, cu, max_rows):
    """(written, is_valid) per row, exactly as the old grid=max_rows kernel decides."""
    dev = seqlens.device
    e = seqlens.numel()
    cu64 = cu.long()
    row = torch.arange(max_rows, dtype=torch.long, device=dev)
    in_range = (row >= cu64[0]) & (row < cu64[e])
    lo = torch.zeros_like(row)
    hi = torch.full_like(row, e - 1)
    while True:
        active = lo < hi
        if not bool(active.any()):
            break
        mid = torch.div(lo + hi + 1, 2, rounding_mode="floor")
        take = cu64[mid] <= row
        lo = torch.where(active & take, mid, lo)
        hi = torch.where(active & ~take, mid - 1, hi)
    rel = row - cu64[lo]
    valid = seqlens.long()[lo]
    # C++ integer division truncates toward zero (matters for negative counts).
    span = torch.div(valid + ALIGN - 1, ALIGN, rounding_mode="trunc") * ALIGN
    written = in_range & (rel < span)
    return written, written & (rel < valid)


def _reference_outputs(ops_mod, x, seqlens, cu, y_init, scale_init):
    max_rows, k = x.shape
    nkb = (k + BLOCK - 1) // BLOCK
    written, is_valid = _reference_write_set(seqlens, cu, max_rows)
    y = y_init.clone()
    scale = scale_init.clone()

    valid_rows = is_valid.nonzero().flatten()
    if valid_rows.numel():
        xv = x[valid_rows].unsqueeze(0).contiguous()
        yv, sv = ops_mod.act_quant_3d(
            xv, torch.tensor([valid_rows.numel()], dtype=torch.int32, device=x.device))
        y[valid_rows] = yv[0]
        scale[:, valid_rows] = sv[0].t()

    pad_rows = (written & ~is_valid).nonzero().flatten()
    if pad_rows.numel():
        zero = torch.zeros(1, 1, k, dtype=torch.bfloat16, device=x.device)
        _, zs = ops_mod.act_quant_3d(zero, torch.ones(1, dtype=torch.int32, device=x.device))
        y[pad_rows] = 0
        scale[:, pad_rows] = zs[0, 0].view(nkb, 1)
    return y, scale, written, is_valid


def _offset_view(rows, cols, dtype, offset, dev):
    """A contiguous [rows, cols] tensor whose data starts `offset` elements into its storage."""
    flat = torch.empty(rows * cols + offset, dtype=dtype, device=dev)
    return flat[offset:].view(rows, cols)


def _make_inputs(seqlens_list, cu_list, max_rows, k, seed, x_offset=0, y_offset=0):
    """x has finite data on the rows the reference reads and inf/NaN elsewhere.

    x_offset / y_offset (in elements) misalign the buffers' base pointers, which
    sends the kernel down its scalar-access path.
    """
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)
    seqlens = torch.tensor(seqlens_list, dtype=torch.int32, device=dev)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=dev)
    gain = torch.empty(max_rows, 1).uniform_(0.05, 20.0, generator=gen)
    x = _offset_view(max_rows, k, torch.bfloat16, x_offset, dev)
    x.copy_((torch.randn(max_rows, k, generator=gen) * gain).to(torch.bfloat16))
    _, is_valid = _reference_write_set(seqlens, cu, max_rows)
    poison = torch.full((k,), float("inf"), dtype=torch.bfloat16, device=dev)
    poison[1::2] = float("nan")
    x[~is_valid] = poison
    nkb = (k + BLOCK - 1) // BLOCK
    y0 = _offset_view(max_rows, k, torch.uint8, y_offset, dev)
    y0.fill_(Y_SENTINEL)
    s0 = torch.full((nkb, max_rows), SCALE_SENTINEL_BITS, dtype=torch.int32,
                    device=dev).view(torch.float32)
    return x, seqlens, cu, y0, s0


def _check(ops_mod, seqlens_list, cu_list, max_rows, k, seed=0, y_dtype=torch.uint8,
           x_offset=0, y_offset=0):
    x, seqlens, cu, y0, s0 = _make_inputs(
        seqlens_list, cu_list, max_rows, k, seed, x_offset, y_offset)
    y_ref, s_ref, written, _ = _reference_outputs(ops_mod, x, seqlens, cu, y0, s0)

    y = _offset_view(max_rows, k, torch.uint8, y_offset, x.device)
    y.copy_(y0)
    if y_dtype != torch.uint8:
        y = y.view(y_dtype)
    scale = s0.clone()
    ret = ops_mod.act_quant_ragged(x, seqlens, cu, y, scale)
    torch.cuda.synchronize()
    assert ret[0].data_ptr() == y.data_ptr() and ret[1].data_ptr() == scale.data_ptr()

    y_u8 = y.view(torch.uint8)
    assert torch.equal(y_u8, y_ref), "y bytes differ from the one-block-per-row reference"
    assert torch.equal(scale.view(torch.int32), s_ref.view(torch.int32)), \
        "scale bits differ from the one-block-per-row reference"
    # Untouched regions are explicitly sentinel (redundant with the equality
    # above, kept so a failure names the region).
    assert bool((y_u8[~written] == Y_SENTINEL).all())
    assert bool((scale.view(torch.int32)[:, ~written] == SCALE_SENTINEL_BITS).all())
    return int(written.sum())


# ──────────────────────────────────────────────────────────────────────────────
# Layout cases
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [256, 300, INTERMEDIATE, HIDDEN])
def test_compact_layout_with_zero_and_64_multiple_experts(ops_mod, k):
    counts = [0, 1, 63, 64, 65, 128, 0, 2, 0]
    cu = _compact_cu(counts)
    written = _check(ops_mod, counts, cu, cu[-1] + ALIGN + 5, k, seed=1)
    assert written == cu[-1]


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_compact_layout_random_counts_with_zeros(ops_mod, seed):
    gen = torch.Generator().manual_seed(seed)
    e = LOCAL_EXPERTS
    counts = torch.randint(0, 150, (e,), generator=gen)
    counts[torch.randperm(e, generator=gen)[: e // 4]] = 0
    counts = counts.tolist()
    cu = _compact_cu(counts)
    _check(ops_mod, counts, cu, cu[-1] + 3 * ALIGN, INTERMEDIATE, seed=seed)


@pytest.mark.parametrize("k", [300, INTERMEDIATE])
def test_all_experts_full_compact(ops_mod, k):
    counts = [128, 64, 192, 64]
    cu = _compact_cu(counts)  # cu[E] == max_rows: no tail rows at all
    written = _check(ops_mod, counts, cu, cu[-1], k, seed=5)
    assert written == cu[-1]


def test_all_experts_empty(ops_mod):
    counts = [0] * LOCAL_EXPERTS
    cu = [0] * (LOCAL_EXPERTS + 1)
    assert _check(ops_mod, counts, cu, 4 * ALIGN, INTERMEDIATE, seed=6) == 0


def test_all_experts_empty_uniform_stride(ops_mod):
    # DeepEP layout with nothing received: every row is past its expert's span.
    e, stride = 4, 128
    cu = [i * stride for i in range(e + 1)]
    assert _check(ops_mod, [0] * e, cu, e * stride, 256, seed=7) == 0


@pytest.mark.parametrize("count,max_rows", [(0, 64), (1, 64), (37, 100), (64, 64), (64, 200), (65, 256)])
def test_single_expert(ops_mod, count, max_rows):
    cu = [0, _align64(count)]
    _check(ops_mod, [count], cu, max_rows, 300, seed=8)


# ──────────────────────────────────────────────────────────────────────────────
# Real decode shapes
# ──────────────────────────────────────────────────────────────────────────────

def _decode_counts(rows_this_rank, e, gen):
    """Multinomial expert loads, as top-k routing of decode tokens produces."""
    probs = torch.ones(e) / e
    idx = torch.multinomial(probs, rows_this_rank, replacement=True, generator=gen)
    return torch.bincount(idx, minlength=e).tolist()


@pytest.mark.parametrize("rows_this_rank", [8, 512, 4096])
def test_deepep_uniform_stride_decode_shape(ops_mod, rows_this_rank):
    """The measured path: S1 output (K=2048) in the [32 x 1024]-row DeepEP buffer."""
    gen = torch.Generator().manual_seed(rows_this_rank)
    e, stride = LOCAL_EXPERTS, DEEPEP_STRIDE
    counts = [min(c, stride) for c in _decode_counts(rows_this_rank, e, gen)]
    cu = [i * stride for i in range(e + 1)]
    written = _check(ops_mod, counts, cu, e * stride, INTERMEDIATE, seed=9)
    assert written == sum(_align64(c) for c in counts)


def test_deepep_uniform_stride_full_capacity(ops_mod):
    """Every row live: 32768 rows on a ~2k-block grid, so each block strides."""
    e, stride = LOCAL_EXPERTS, DEEPEP_STRIDE
    cu = [i * stride for i in range(e + 1)]
    written = _check(ops_mod, [stride] * e, cu, e * stride, INTERMEDIATE, seed=10)
    assert written == e * stride


@pytest.mark.parametrize("k", [HIDDEN, INTERMEDIATE])
def test_nccl_compact_decode_shape(ops_mod, k):
    """NCCL-path compact layout at g512: 512 tokens x top-8 = 4096 routed rows
    over 256 experts, ~512 of them on this rank's 32 local experts; capacity is
    the pool's ragged capacity for 512 tokens (dispatch's 64-row padding bound)."""
    gen = torch.Generator().manual_seed(k)
    counts = _decode_counts(512, LOCAL_EXPERTS, gen)
    cu = _compact_cu(counts)
    nk = 512 * 8
    capacity = (nk + 63 * min(LOCAL_EXPERTS, nk)) // 64 * 64
    assert cu[-1] <= capacity
    _check(ops_mod, counts, cu, capacity, k, seed=11)


@pytest.mark.parametrize("k,x_offset,y_offset", [
    (302, 0, 0),     # K % 4 != 0
    (301, 0, 0),
    (2048, 1, 0),    # x base not 8-byte aligned
    (2048, 2, 0),
    (2048, 0, 1),    # y base not 4-byte aligned
    (2048, 0, 2),
])
def test_scalar_access_path(ops_mod, k, x_offset, y_offset):
    counts = [0, 1, 63, 64, 65, 2]
    cu = _compact_cu(counts)
    _check(ops_mod, counts, cu, cu[-1] + 9, k, seed=15, x_offset=x_offset, y_offset=y_offset)


@pytest.mark.parametrize("e", [129, 256, 300])
def test_more_experts_than_threads(ops_mod, e):
    """The kernel scans experts in chunks of 128; E > 128 crosses chunk boundaries."""
    gen = torch.Generator().manual_seed(e)
    counts = torch.randint(0, 90, (e,), generator=gen)
    counts[torch.randperm(e, generator=gen)[: e // 5]] = 0
    counts = counts.tolist()
    cu = _compact_cu(counts)
    _check(ops_mod, counts, cu, cu[-1] + 77, 256, seed=16)


def test_grouped_prefill_shape(ops_mod):
    """Grouped prefill: all 256 routed experts, compact layout, 4096 tokens x top-8."""
    gen = torch.Generator().manual_seed(17)
    e, tokens, topk = 256, 4096, 8
    idx = torch.multinomial(torch.ones(e) / e, tokens * topk, replacement=True, generator=gen)
    counts = torch.bincount(idx, minlength=e).tolist()
    cu = _compact_cu(counts)
    capacity = (tokens * topk + e * 63 + 127) // 128 * 128
    _check(ops_mod, counts, cu, capacity, INTERMEDIATE, seed=18)


def test_float8_output_dtype(ops_mod):
    counts = [3, 0, 70]
    cu = _compact_cu(counts)
    _check(ops_mod, counts, cu, cu[-1] + 10, 256, seed=12, y_dtype=torch.float8_e4m3fn)


# ──────────────────────────────────────────────────────────────────────────────
# Device values the host never checks: the write set must still be the old one
# ──────────────────────────────────────────────────────────────────────────────

_MALFORMED = {
    # cu[E] beyond max_rows: rows >= max_rows were never launched.
    "cu_end_past_max_rows": ([10, 70], [0, 64, 192], 150),
    # cu[0] > 0: rows below cu[0] are skipped even if a span would cover them.
    "cu_start_offset": ([5, 9], [64, 128, 192], 256),
    # Negative cu[0]: the old grid simply had no negative rows.
    "cu_start_negative": ([90, 4], [-32, 64, 128], 160),
    # Overlapping spans: expert 0's span runs into expert 1's rows.
    "overlapping_spans": ([150, 3], [0, 64, 192], 256),
    # Non-monotonic cu: the binary search's owner, not the walk order, decides.
    "non_monotonic_cu": ([20, 40, 10], [0, 128, 64, 256], 256),
    # Negative counts: truncating division gives span 0 or a negative span.
    "negative_counts": ([-10, -100, 5], [0, 64, 128, 192], 192),
    # cu[0] > cu[E]: empty row range, nothing written.
    "reversed_range": ([4, 4], [128, 64, 0], 192),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_unchecked_device_values_keep_the_old_write_set(ops_mod, case):
    counts, cu, max_rows = _MALFORMED[case]
    _check(ops_mod, counts, cu, max_rows, 300, seed=13)


# ──────────────────────────────────────────────────────────────────────────────
# Launch shape and CUDA graph
# ──────────────────────────────────────────────────────────────────────────────

def _profiled_grids(ops_mod, max_rows):
    dev = torch.device("cuda")
    e = LOCAL_EXPERTS
    stride = max_rows // e
    seqlens = torch.full((e,), 3, dtype=torch.int32, device=dev)
    cu = (torch.arange(e + 1, dtype=torch.int32, device=dev) * stride).contiguous()
    x = torch.zeros(max_rows, 256, dtype=torch.bfloat16, device=dev)
    y = torch.empty(max_rows, 256, dtype=torch.uint8, device=dev)
    scale = torch.empty(2, max_rows, dtype=torch.float32, device=dev)
    ops_mod.act_quant_ragged(x, seqlens, cu, y, scale)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        ops_mod.act_quant_ragged(x, seqlens, cu, y, scale)
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as fh:
            events = json.load(fh)["traceEvents"]
    return [ev["args"]["grid"] for ev in events
            if "act_quant_ragged_kernel" in ev.get("name", "") and "grid" in ev.get("args", {})]


def test_grid_is_not_sized_by_capacity(ops_mod):
    """At most one block per 16 rows of capacity, and one resident wave whenever
    the capacity fits 16 waves (the DeepEP decode buffer is 32 x 1024 rows)."""
    props = torch.cuda.get_device_properties(0)
    wave_cap = props.multi_processor_count * 16  # 2048 threads per SM / 128 per block
    grids = {}
    for max_rows in (16 * 1024, LOCAL_EXPERTS * DEEPEP_STRIDE, 147200):
        g = _profiled_grids(ops_mod, max_rows)
        assert len(g) == 1, (max_rows, g)
        gx, gy, gz = g[0]
        assert (gy, gz) == (1, 1)
        grids[max_rows] = gx
    small, decode, prefill = grids[16 * 1024], grids[LOCAL_EXPERTS * DEEPEP_STRIDE], grids[147200]
    assert small == decode, grids          # same launch shape for both decode-sized buffers
    assert 0 < decode <= wave_cap, grids   # a single resident wave
    assert decode < LOCAL_EXPERTS * DEEPEP_STRIDE // 8, grids
    assert prefill <= max(wave_cap, -(-147200 // 16)), grids


_GRAPH_SCRIPT = textwrap.dedent(
    """
    import torch, batchgen_kernels.moe._C_fp8_blockwise_ops as ops
    dev = torch.device("cuda")
    e, stride, k = 8, 128, 2048
    rows = e * stride
    gen = torch.Generator().manual_seed(14)
    x = (torch.randn(rows, k, generator=gen) * 3).to(torch.bfloat16).to(dev)
    seqlens = torch.tensor([5, 0, 64, 1, 0, 128, 33, 2], dtype=torch.int32, device=dev)
    cu = (torch.arange(e + 1, dtype=torch.int32, device=dev) * stride).contiguous()
    y = torch.full((rows, k), 0xAB, dtype=torch.uint8, device=dev)
    scale = torch.full((k // 128, rows), 0x7FC0DEAD, dtype=torch.int32, device=dev).view(torch.float32)
    # The very first act_quant_ragged call of the process happens INSIDE capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.act_quant_ragged(x, seqlens, cu, y, scale)
    # Change the live counts on device after capture, then replay.
    seqlens.copy_(torch.tensor([128, 7, 0, 65, 64, 1, 0, 100], dtype=torch.int32))
    y.fill_(0xAB); scale.view(torch.int32).fill_(0x7FC0DEAD)
    graph.replay()
    torch.cuda.synchronize()
    y_e = torch.full_like(y, 0xAB)
    s_e = torch.full_like(scale.view(torch.int32), 0x7FC0DEAD).view(torch.float32)
    ops.act_quant_ragged(x, seqlens, cu, y_e, s_e)
    torch.cuda.synchronize()
    assert torch.equal(y, y_e), "replay y differs from eager"
    assert torch.equal(scale.view(torch.int32), s_e.view(torch.int32)), "replay scale differs"
    print("GRAPH_REPLAY_MATCHES_EAGER")
    """
)


def test_graph_captured_first_call_replays_new_device_counts(ops_mod):
    proc = subprocess.run(
        [sys.executable, "-c", _GRAPH_SCRIPT],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GRAPH_REPLAY_MATCHES_EAGER" in proc.stdout, proc.stdout + proc.stderr
