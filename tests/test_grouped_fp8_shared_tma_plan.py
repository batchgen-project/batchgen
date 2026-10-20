"""fp8_blockwise_prepare_s1_s3_tma: one descriptor-preparation launch for the
fused S1 and the grouped S3 GEMM of a MoE layer.

Each grouped GEMM normally runs its own ``update_expert_tma`` launch first. A
fused S1 + S3 pair that shares seqlens / cu_seqlens / TileM can instead be
prepared once and run with ``update_tma=False``. Contract pinned here (GPU
only; skips cleanly without CUDA or the extension):

* the descriptors and tile tables the one launch writes are byte-identical to
  the ones each GEMM's own preparation writes (captured through the GEMMs'
  ``tma_desc`` / ``tiles`` / ``cu_tiles`` workspaces), at TileM 16 / 32 / 64;
* S1 and S3 run on the prepared plan write bit-identical output buffers to S1
  and S3 run with their own preparation;
* the counts are read on device only: prepare + S1 + quant + S3 captured in one
  CUDA graph replays new counts and matches the eager two-launch chain;
* ``update_tma=False`` without the prepared workspaces, and prepare inputs that
  do not share one row space or overlap, are rejected on the host;
* through the Python wrapper, both GEMMs accept the plan and match.
"""

import importlib

import pytest
import torch

E, S = 8, 256                  # local experts x uniform per-expert stride
H, N = 6144, 2048              # GLM-5.2 hidden, moe intermediate
ROWS = E * S


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _mod(name, symbol):
    _require_cuda()
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any load failure means unavailable
        pytest.skip(f"{name} unavailable: {exc}")
    if not hasattr(mod, symbol):
        pytest.skip(f"{name} has no {symbol}")
    return mod


def _gemm():
    return _mod("batchgen_kernels.moe._C_fp8_blockwise_gemm",
                "fp8_blockwise_prepare_s1_s3_tma")


def _ops():
    return _mod("batchgen_kernels.moe._C_fp8_blockwise_ops", "act_quant_ragged")


def _fp8(shape, g):
    b = torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(torch.uint8)
    b[(b & 0x7F) == 0x7F] = 0                      # no e4m3 NaN encodings
    return b.cuda().view(torch.float8_e4m3fn)


def _pad4(v):
    return (v + 3) // 4 * 4


def _layer(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        x=_fp8((ROWS, H), g),
        xs=(torch.rand(H // 128, ROWS, generator=g) * 1e-2 + 1e-4).cuda(),
        w_gate=_fp8((E, N, H), g), w_up=_fp8((E, N, H), g),
        ws_gate=(torch.rand(E, N // 128, _pad4(H // 128), generator=g) + .5).cuda(),
        ws_up=(torch.rand(E, N // 128, _pad4(H // 128), generator=g) + .5).cuda(),
        w_down=_fp8((E, H, N), g),
        ws_down=(torch.rand(E, H // 128, _pad4(N // 128), generator=g) + .5).cuda(),
        cu=(torch.arange(E + 1, dtype=torch.int32) * S).cuda(),
    )


def _bufs():
    return dict(
        inter=torch.full((ROWS, N), -2.0, dtype=torch.bfloat16, device="cuda"),
        inter_fp8=torch.zeros(ROWS, N, dtype=torch.uint8, device="cuda"),
        inter_scale=torch.zeros(N // 128, ROWS, device="cuda"),
        out=torch.full((ROWS, H), -5.0, dtype=torch.bfloat16, device="cuda"),
    )


def _ws():
    return dict(
        s1=torch.zeros(2 * E, 128, dtype=torch.uint8, device="cuda"),
        s3=torch.zeros(2 * E, 128, dtype=torch.uint8, device="cuda"),
        tiles=torch.full((E,), -1, dtype=torch.int32, device="cuda"),
        cu_tiles=torch.full((E + 1,), -1, dtype=torch.int32, device="cuda"),
    )


def _counts(tile_m):
    return [0, 1, tile_m - 1, tile_m, tile_m + 1, 3 * tile_m + 5, S - 1, S]


def _chain(gemm, ops, L, counts, B, avg, ws=None, plan=False):
    """S1 -> act_quant_ragged -> S3. plan=False: each GEMM prepares itself
    (writing into ws when given, to expose its descriptor bytes); plan=True:
    one prepare launch, both GEMMs with update_tma=False."""
    if plan:
        gemm.fp8_blockwise_prepare_s1_s3_tma(
            L["x"], B["inter"], B["inter_fp8"].view(torch.float8_e4m3fn), B["out"],
            counts, L["cu"], avg, ws["s1"], ws["s3"], ws["tiles"], ws["cu_tiles"])
    s1_ws = (ws["s1"], ws["tiles"], ws["cu_tiles"]) if ws is not None else (None,) * 3
    gemm.fp8_blockwise_fused_s1(
        L["x"], L["w_gate"], L["w_up"], counts, L["cu"], L["xs"],
        L["ws_gate"], L["ws_up"], avg, B["inter"], *s1_ws, not plan)
    s1_tiles = None if ws is None else (ws["tiles"].clone(), ws["cu_tiles"].clone())
    ops.act_quant_ragged(B["inter"], counts, L["cu"], B["inter_fp8"], B["inter_scale"])
    s3_ws = (ws["s3"], ws["tiles"], ws["cu_tiles"]) if ws is not None else (None,) * 3
    gemm.fp8_blockwise_grouped_gemm(
        B["inter_fp8"].view(torch.float8_e4m3fn), L["w_down"], counts, L["cu"],
        B["inter_scale"], L["ws_down"], avg, B["out"], *s3_ws, not plan)
    return s1_tiles


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("avg, tile_m", [(8, 16), (20, 32), (40, 64)])
def test_one_launch_writes_the_bytes_both_gemms_would_write(avg, tile_m):
    gemm, ops = _gemm(), _ops()
    L = _layer(seed=avg)
    counts = torch.tensor(_counts(tile_m), dtype=torch.int32, device="cuda")

    own_ws, own = _ws(), _bufs()
    s1_tiles = _chain(gemm, ops, L, counts, own, avg, ws=own_ws, plan=False)
    torch.cuda.synchronize()
    tiles = [-(-c // tile_m) for c in _counts(tile_m)]
    assert s1_tiles[0].tolist() == tiles
    assert s1_tiles[1].tolist() == [sum(tiles[:i]) for i in range(E + 1)]

    plan_ws, planned = _ws(), _bufs()
    _chain(gemm, ops, L, counts, planned, avg, ws=plan_ws, plan=True)
    torch.cuda.synchronize()

    # Descriptors encode buffer addresses, and the two runs used different
    # output buffers, so the one-launch bytes are compared against the
    # self-prepared run's own buffers: prepare once more for exactly those.
    redo = _ws()
    gemm.fp8_blockwise_prepare_s1_s3_tma(
        L["x"], own["inter"], own["inter_fp8"].view(torch.float8_e4m3fn), own["out"],
        counts, L["cu"], avg, redo["s1"], redo["s3"], redo["tiles"], redo["cu_tiles"])
    torch.cuda.synchronize()
    assert _same(redo["s1"], own_ws["s1"]), "S1 descriptors differ"
    assert _same(redo["s3"], own_ws["s3"]), "S3 descriptors differ"
    assert torch.equal(redo["tiles"], s1_tiles[0])
    assert torch.equal(redo["cu_tiles"], s1_tiles[1])
    assert torch.equal(own_ws["tiles"], s1_tiles[0]), "S3's own tiles differ from S1's"
    assert torch.equal(own_ws["cu_tiles"], s1_tiles[1])

    for name in ("inter", "inter_fp8", "inter_scale", "out"):
        assert _same(own[name], planned[name]), name
    live = planned["out"][3 * S:3 * S + tile_m]
    assert not torch.equal(live, torch.full_like(live, -5.0))


def test_graph_replay_of_the_planned_chain_matches_the_eager_two_launch_chain():
    gemm, ops = _gemm(), _ops()
    L = _layer(seed=77)
    avg = 8
    counts = torch.tensor(_counts(16), dtype=torch.int32, device="cuda")
    ws, B = _ws(), _bufs()
    _chain(gemm, ops, L, counts, B, avg, ws=ws, plan=True)      # warm
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _chain(gemm, ops, L, counts, B, avg, ws=ws, plan=True)
    g = torch.Generator().manual_seed(5)
    for step in range(4):
        new = torch.randint(0, 3 * 16 + 1, (E,), generator=g, dtype=torch.int32)
        new[step % E] = S
        counts.copy_(new.cuda())
        for t, v in ((B["inter"], -2.0), (B["out"], -5.0)):
            t.fill_(v)
        B["inter_fp8"].zero_()
        B["inter_scale"].zero_()
        graph.replay()
        ref = _bufs()
        _chain(gemm, ops, L, counts, ref, avg)
        torch.cuda.synchronize()
        for name in ("inter", "inter_fp8", "inter_scale", "out"):
            assert _same(B[name], ref[name]), (step, name)


def test_update_tma_false_requires_the_prepared_workspaces():
    gemm = _gemm()
    L, B, ws = _layer(seed=1), _bufs(), _ws()
    counts = torch.zeros(E, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="update_tma=false requires"):
        gemm.fp8_blockwise_fused_s1(L["x"], L["w_gate"], L["w_up"], counts, L["cu"],
                                    L["xs"], L["ws_gate"], L["ws_up"], 8, B["inter"],
                                    ws["s1"], None, ws["cu_tiles"], False)
    with pytest.raises(RuntimeError, match="update_tma=false requires"):
        gemm.fp8_blockwise_grouped_gemm(
            B["inter_fp8"].view(torch.float8_e4m3fn), L["w_down"], counts, L["cu"],
            B["inter_scale"], L["ws_down"], 8, B["out"], None, ws["tiles"],
            ws["cu_tiles"], False)
    with pytest.raises(RuntimeError, match=r"\[2 \* num_group, 128\]"):
        gemm.fp8_blockwise_fused_s1(L["x"], L["w_gate"], L["w_up"], counts, L["cu"],
                                    L["xs"], L["ws_gate"], L["ws_up"], 8, B["inter"],
                                    ws["s1"][:-1], ws["tiles"], ws["cu_tiles"], False)


def test_prepare_rejects_inputs_that_are_not_one_pair():
    gemm = _gemm()
    L, B, ws = _layer(seed=2), _bufs(), _ws()
    counts = torch.zeros(E, dtype=torch.int32, device="cuda")
    s3_x = B["inter_fp8"].view(torch.float8_e4m3fn)

    def call(**over):
        kw = dict(s1_x=L["x"], s1_output=B["inter"], s3_x=s3_x, s3_output=B["out"],
                  seqlens=counts, cu_seqlens=L["cu"], avg=8, s1=ws["s1"], s3=ws["s3"],
                  tiles=ws["tiles"], cu_tiles=ws["cu_tiles"])
        kw.update(over)
        gemm.fp8_blockwise_prepare_s1_s3_tma(
            kw["s1_x"], kw["s1_output"], kw["s3_x"], kw["s3_output"], kw["seqlens"],
            kw["cu_seqlens"], kw["avg"], kw["s1"], kw["s3"], kw["tiles"], kw["cu_tiles"])

    with pytest.raises(RuntimeError, match="share the row space"):
        call(s3_output=B["out"][:-64])
    with pytest.raises(RuntimeError, match="must not overlap"):
        call(s3=ws["s1"])
    with pytest.raises(RuntimeError, match="cu_seqlens"):
        call(cu_seqlens=L["cu"][:-1])
    with pytest.raises(RuntimeError, match="tiles must be"):
        call(tiles=ws["tiles"][:-1])
    with pytest.raises(RuntimeError, match="s1_x must be"):
        call(s1_x=B["inter"])
    call()
    torch.cuda.synchronize()


def test_the_python_wrapper_runs_both_gemms_on_one_plan():
    _gemm()
    ops = _ops()
    try:
        from batchgen.moe import grouped_fp8_blockwise_moe as w
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"batchgen wrapper unavailable: {exc}")
    if not hasattr(w, "prepare_grouped_fp8_s1_s3_tma"):
        pytest.skip(f"{w.__file__} predates the shared S1/S3 TMA plan")
    L = _layer(seed=9)
    avg = 20
    counts = torch.tensor(_counts(32), dtype=torch.int32, device="cuda")
    ws = _ws()

    def run(use_plan):
        B = _bufs()
        s3_x = B["inter_fp8"].view(torch.float8_e4m3fn)
        plan = None
        if use_plan:
            plan = w.prepare_grouped_fp8_s1_s3_tma(
                L["x"], B["inter"], s3_x, B["out"], counts, L["cu"], avg,
                s1_tma_desc=ws["s1"], s3_tma_desc=ws["s3"], tiles=ws["tiles"],
                cu_tiles=ws["cu_tiles"])
            assert plan.tile_m == 32
        w.grouped_fp8_blockwise_fused_s1(
            L["x"], L["xs"], L["w_gate"], L["w_up"], L["ws_gate"], L["ws_up"],
            counts, L["cu"], avg, output=B["inter"], tma_plan=plan)
        ops.act_quant_ragged(B["inter"], counts, L["cu"], B["inter_fp8"],
                             B["inter_scale"])
        w.grouped_fp8_blockwise_s3(
            s3_x, B["inter_scale"], L["w_down"], L["ws_down"], counts, L["cu"],
            avg, output=B["out"], tma_plan=plan)
        torch.cuda.synchronize()
        return B

    a, b = run(False), run(True)
    for name in ("inter", "inter_fp8", "inter_scale", "out"):
        assert _same(a[name], b[name]), name
