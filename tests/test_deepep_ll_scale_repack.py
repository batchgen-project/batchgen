"""deepep_ll_scale_repack: the recv_count-bounded DeepEP LL scale repack.

DeepEP's low-latency FP8 dispatch hands back the activation scales as one
contiguous [E, hidden/128, S] block per call (exposed as the (1, 2)-transposed
[E, S, hidden/128] view); the grouped FP8 GEMM reads one [hidden/128, E * S]
matrix with expert e at column e * S. The kernel copies, per expert, only the
columns [0, ceil(recv_count[e] / span_align) * span_align) -- with span_align =
the GEMM's TileM, exactly the columns its x-scale tiles load.

Contract pinned here (GPU only; skips cleanly without CUDA or the extension):

* every copied column is bit-identical to the full torch repack it replaces
  (``out.view(KB, E, S).copy_(scales.transpose(1, 2).permute(1, 0, 2))``);
* every column past the span is left untouched;
* raw bit patterns (NaN payloads, infinities, signed zeros, subnormals) move
  unchanged;
* the counts are read on device only: a captured graph replays new counts;
* the fused S1 GEMM produces the same bytes from the bounded repack as from the
  full one even when every uncopied column is NaN, at TileM 16 / 32 / 64 --
  i.e. no tile ever loads a column past the span;
* malformed inputs are rejected on the host, an impossible count traps.
"""

import importlib
import subprocess
import sys
import textwrap

import pytest
import torch

E, S, KB = 32, 1024, 48            # GLM-5.2 decode: 32 local experts, 8 x 128 rows, 6144/128
COUNTS = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 1023,
          1024, 2, 3, 0, 5, 48, 47, 49, 96, 100, 200, 300, 400, 500, 600, 7]
SENTINEL_BITS = 0x7FC0DEAD          # a quiet NaN no copy can produce by accident


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _ops():
    _require_cuda()
    try:
        mod = importlib.import_module("batchgen_kernels.moe._C_fp8_blockwise_ops")
    except Exception as exc:  # noqa: BLE001 - any load failure means unavailable
        pytest.skip(f"_C_fp8_blockwise_ops unavailable: {exc}")
    if not hasattr(mod, "deepep_ll_scale_repack"):
        pytest.skip("_C_fp8_blockwise_ops has no deepep_ll_scale_repack")
    return mod


def _gemm():
    try:
        mod = importlib.import_module("batchgen_kernels.moe._C_fp8_blockwise_gemm")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"_C_fp8_blockwise_gemm unavailable: {exc}")
    return mod


def _deepep_scales(e=E, s=S, kb=KB, seed=0, raw_bits=False):
    """DeepEP's allocation [e, kb, s] contiguous and the [e, s, kb] view it
    returns."""
    g = torch.Generator().manual_seed(seed)
    if raw_bits:
        bits = torch.randint(-2**31, 2**31 - 1, (e, kb, s), generator=g,
                             dtype=torch.int64).to(torch.int32)
        flat = bits.view(-1)
        specials = torch.tensor([0x7F800000, -0x00800000, 0x7FC00001, -0x003FFFFF,
                                 0x00000000, -0x80000000, 0x00000001, 0x007FFFFF],
                                dtype=torch.int64).to(torch.int32)
        flat[: specials.numel()] = specials
        base = bits.view(torch.float32)
    else:
        base = torch.rand(e, kb, s, generator=g) * 1e-2 + 1e-4
    base = base.cuda()
    return base, base.transpose(1, 2)


def _sentinel_out(kb=KB, cols=E * S):
    out = torch.empty(kb, cols, dtype=torch.int32, device="cuda")
    out.fill_(SENTINEL_BITS)
    return out.view(torch.float32)


def _full_repack(view, e=E, s=S, kb=KB):
    """The torch copy the kernel replaces (moe_cuda_graph_segments, 9a50798d)."""
    out = _sentinel_out(kb, e * s)
    out.view(kb, e, s).copy_(view.transpose(1, 2).permute(1, 0, 2))
    return out


def _span(c, align):
    return -(-int(c) // align) * align


def _check(out, ref, counts, align, s=S):
    o = out.view(torch.int32).cpu()
    r = ref.view(torch.int32).cpu()
    for e, c in enumerate(counts):
        lo, sp = e * s, _span(c, align)
        assert torch.equal(o[:, lo:lo + sp], r[:, lo:lo + sp]), (
            f"expert {e} count {c} align {align}: copied span differs")
        assert torch.all(o[:, lo + sp:lo + s] == SENTINEL_BITS), (
            f"expert {e} count {c} align {align}: wrote past the span")


@pytest.mark.parametrize("align", [16, 32, 64])
def test_span_is_bit_identical_to_the_full_repack_and_the_rest_untouched(align):
    ops = _ops()
    _base, view = _deepep_scales()
    counts = torch.tensor(COUNTS, dtype=torch.int32, device="cuda")
    out = _sentinel_out()
    ops.deepep_ll_scale_repack(view, counts, out, align)
    torch.cuda.synchronize()
    _check(out, _full_repack(view), COUNTS, align)


@pytest.mark.parametrize("counts", [[0] * E, [S] * E, [1] * E])
def test_empty_full_and_single_row_loads(counts):
    ops = _ops()
    _base, view = _deepep_scales(seed=3)
    out = _sentinel_out()
    ops.deepep_ll_scale_repack(
        view, torch.tensor(counts, dtype=torch.int32, device="cuda"), out, 16)
    torch.cuda.synchronize()
    _check(out, _full_repack(view), counts, 16)


def test_raw_bit_patterns_move_unchanged():
    ops = _ops()
    base, view = _deepep_scales(seed=5, raw_bits=True)
    out = _sentinel_out()
    ops.deepep_ll_scale_repack(
        view, torch.full((E,), S, dtype=torch.int32, device="cuda"), out, 64)
    torch.cuda.synchronize()
    want = base.view(torch.int32).permute(1, 0, 2).reshape(KB, E * S).cpu()
    assert torch.equal(out.view(torch.int32).cpu(), want)


def test_counts_are_read_on_device_under_graph_replay():
    ops = _ops()
    _base, view = _deepep_scales(seed=7)
    counts = torch.zeros(E, dtype=torch.int32, device="cuda")
    out = _sentinel_out()
    ops.deepep_ll_scale_repack(view, counts, out, 16)       # warm outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.deepep_ll_scale_repack(view, counts, out, 16)
    ref = _full_repack(view)
    g = torch.Generator().manual_seed(11)
    for _ in range(4):
        new = torch.randint(0, S + 1, (E,), generator=g, dtype=torch.int32)
        counts.copy_(new.cuda())
        out.view(torch.int32).fill_(SENTINEL_BITS)
        graph.replay()
        torch.cuda.synchronize()
        _check(out, ref, new.tolist(), 16)


def _fp8_rows(rows, k, g):
    b = torch.randint(0, 256, (rows, k), generator=g, dtype=torch.int32).to(torch.uint8)
    b[(b & 0x7F) == 0x7F] = 0                      # no e4m3 NaN encodings
    return b


@pytest.mark.parametrize("avg, tile_m", [(8, 16), (20, 32), (40, 64)])
def test_the_gemm_reads_no_column_past_the_span(avg, tile_m):
    """Fused S1 over the bounded repack (uncopied columns = NaN) vs over the
    full repack: the whole output buffer is bit-identical, so no tile loaded a
    column past ceil(count / TileM) * TileM."""
    ops = _ops()
    gemm = _gemm()
    e, s, k, n = 8, 256, 6144, 2048
    kb = k // 128
    g = torch.Generator().manual_seed(100 + avg)
    counts_l = [0, 1, tile_m - 1, tile_m, tile_m + 1, 2 * tile_m + 3, s - 1, s]
    counts = torch.tensor(counts_l, dtype=torch.int32, device="cuda")
    cu = (torch.arange(e + 1, dtype=torch.int32) * s).cuda()
    x = _fp8_rows(e * s, k, g).cuda().view(torch.float8_e4m3fn)
    base = (torch.rand(e, kb, s, generator=g) * 1e-2 + 1e-4).cuda()
    view = base.transpose(1, 2)
    w_gate = _fp8_rows(e * n, k, g).view(e, n, k).cuda().view(torch.float8_e4m3fn)
    w_up = _fp8_rows(e * n, k, g).view(e, n, k).cuda().view(torch.float8_e4m3fn)
    kb4 = (kb + 3) // 4 * 4
    ws_gate = (torch.rand(e, n // 128, kb4, generator=g) + 0.5).cuda()
    ws_up = (torch.rand(e, n // 128, kb4, generator=g) + 0.5).cuda()

    full = _full_repack(view, e, s, kb)
    bounded = _sentinel_out(kb, e * s)            # NaN everywhere ...
    ops.deepep_ll_scale_repack(view, counts, bounded, tile_m)   # ... but the span
    outs = []
    for xs in (full, bounded):
        y = torch.full((e * s, n), -3.0, dtype=torch.bfloat16, device="cuda")
        gemm.fp8_blockwise_fused_s1(x, w_gate, w_up, counts, cu, xs,
                                    ws_gate, ws_up, avg, y)
        outs.append(y)
    torch.cuda.synchronize()
    live = outs[1][3 * s:3 * s + counts_l[3]]      # expert 3: one full tile
    assert torch.isfinite(live.float()).all()
    assert not torch.equal(live, torch.full_like(live, -3.0))
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


def test_host_checks_reject_malformed_inputs():
    ops = _ops()
    base, view = _deepep_scales(e=4, s=64, kb=2)
    good = dict(recv_scales=view,
                recv_count=torch.zeros(4, dtype=torch.int32, device="cuda"),
                out=torch.zeros(2, 4 * 64, device="cuda"), span_align=16)
    cases = [
        (dict(recv_scales=view.contiguous()), "transposed view"),
        (dict(recv_count=torch.zeros(4, dtype=torch.int64, device="cuda")), "recv_count"),
        (dict(recv_count=torch.zeros(3, dtype=torch.int32, device="cuda")), "recv_count"),
        (dict(out=torch.zeros(2, 4 * 64 - 1, device="cuda")), "out must be"),
        (dict(out=torch.zeros(2, 4 * 64, device="cuda", dtype=torch.float64)), "out must be"),
        (dict(span_align=24), "span_align"),
        (dict(span_align=6), "span_align"),
        (dict(span_align=0), "span_align"),
        (dict(out=base.view(2, 4 * 64)), "must not overlap"),
    ]
    for over, match in cases:
        kw = dict(good, **over)
        with pytest.raises(RuntimeError, match=match):
            ops.deepep_ll_scale_repack(kw["recv_scales"], kw["recv_count"],
                                       kw["out"], kw["span_align"])
    ops.deepep_ll_scale_repack(good["recv_scales"], good["recv_count"],
                               good["out"], good["span_align"])
    torch.cuda.synchronize()


def test_a_count_the_layout_cannot_hold_traps():
    _ops()
    code = textwrap.dedent("""
        import torch, importlib
        ops = importlib.import_module("batchgen_kernels.moe._C_fp8_blockwise_ops")
        base = torch.zeros(2, 2, 64, device="cuda")
        out = torch.zeros(2, 128, device="cuda")
        ops.deepep_ll_scale_repack(base.transpose(1, 2),
                                   torch.tensor([65, 0], dtype=torch.int32, device="cuda"),
                                   out, 16)
        torch.cuda.synchronize()
        print("NO_TRAP")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode != 0 and "NO_TRAP" not in proc.stdout, (
        proc.returncode, proc.stdout[-400:], proc.stderr[-400:])
