"""Kimi-K3 resident handoff: admission waves after the first decode phase
prefill through the resident EP shard instead of releasing it.

Measured on H200 (2026-09-06, runs r21/v1): every mid-run wave paid
`deep_free` (9 s) + streamed-SP8 reconfigure (5-9 s) + the shard's
"one-time repack" from the node store on the way back — 3.7 s on rank 0
but 22-139 s on the other seven ranks (8 x 84 GiB through the page cache),
with the slowest rank gating the decode barrier: ~2-2.5 min per wave, 18
waves on the 512-request contract. The handoff keeps the model and shard,
releases only the decode graphs (they bake KV-pool addresses; the pool is
rebuilt), and runs the wave's prefill through resident EP in chunks of
`k3_resident_prefill_token_cap` tokens.

The PSM and worker import the whole engine, so the method bodies are
exec'd / inspected in isolation.
"""
import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PSM = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "Parallel_Strategy_Manager.py"
WORKER = ROOT / "batchgen" / "batchgen_worker.py"


def _class_method_source(path, class_name, method_name):
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == method_name)
    lines = source.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[fn.lineno - 1:fn.end_lineno])), fn


def _psm_stub(**attrs):
    ns = {}
    for name in ("set_prefill_moe_mode", "resident_ep_prefill_available",
                 "release_decode_graph", "prefill_uses_resident_ep"):
        src, _ = _class_method_source(PSM, "KimiLinearParallelStrategyManager", name)
        exec(src, ns)

    class _PSM:
        pass

    obj = _PSM()
    for name in ("set_prefill_moe_mode", "resident_ep_prefill_available",
                 "release_decode_graph", "prefill_uses_resident_ep"):
        setattr(_PSM, name, ns[name])
    obj._is_k3 = True
    obj._distributed_weight_sharded = True
    obj._resident_ep_built = False
    obj._attn_tp_size = 8
    obj._stream_all_modules = False
    obj._prefill_moe_mode = "streamed_sp8"
    obj._decode_graph = None
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def test_distributed_weights_accept_resident_ep_only_after_the_shard_exists():
    import pytest
    psm = _psm_stub()
    with pytest.raises(ValueError):
        psm.set_prefill_moe_mode("resident_ep")
    psm.set_prefill_moe_mode("streamed_sp8")
    assert psm._prefill_moe_mode == "streamed_sp8"
    psm._resident_ep_built = True
    psm.set_prefill_moe_mode("resident_ep")
    assert psm.prefill_uses_resident_ep()
    # the legacy replicated path is still refused under distributed weights
    with pytest.raises(ValueError):
        psm.set_prefill_moe_mode("streamed")


def test_resident_prefill_availability_needs_built_shard_and_tp_attention():
    assert not _psm_stub().resident_ep_prefill_available()
    assert _psm_stub(_resident_ep_built=True).resident_ep_prefill_available()
    assert not _psm_stub(_resident_ep_built=True, _attn_tp_size=1).resident_ep_prefill_available()
    assert not _psm_stub(_resident_ep_built=True, _stream_all_modules=True).resident_ep_prefill_available()
    assert not _psm_stub(_resident_ep_built=True, _is_k3=False).resident_ep_prefill_available()


def test_release_decode_graph_drops_the_captured_graphs():
    class _Graph:
        released = False

        def release(self):
            self.released = True

    g = _Graph()
    psm = _psm_stub(_decode_graph=g)
    psm.release_decode_graph()
    assert g.released and psm._decode_graph is None
    psm.release_decode_graph()  # idempotent


def test_worker_handoff_keeps_the_model_and_releases_only_the_graphs():
    _, fn = _class_method_source(WORKER, "BatchGenWorker", "_config_prefill_for_batch")
    # find `if reuse_startup_prefill: ... elif resident_handoff: ... else: deep_free`
    branch = None
    for node in ast.walk(fn):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "reuse_startup_prefill"
                and node.orelse and isinstance(node.orelse[0], ast.If)
                and isinstance(node.orelse[0].test, ast.Name)
                and node.orelse[0].test.id == "resident_handoff"):
            branch = node.orelse[0]
            break
    assert branch is not None, "resident_handoff branch missing"

    def _calls(nodes):
        out = set()
        for n in nodes:
            for c in ast.walk(n):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute):
                    out.add(c.func.attr)
        return out

    handoff_calls = _calls(branch.body)
    assert "release_decode_graph" in handoff_calls
    assert "deep_free_model_memory" not in handoff_calls
    assert "_destroy_gpu_paged_kv_cache" not in handoff_calls  # done once, after the branch
    # the fallback branch still deep-frees
    assert "deep_free_model_memory" in _calls(branch.orelse)
    # the handoff sets the resident token cap on the batching config
    caps = [c for n in branch.body for c in ast.walk(n)
            if isinstance(c, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == "prefill_micro_batch_token_cap"
                for t in c.targets)]
    assert caps, "handoff must install k3_resident_prefill_token_cap as the micro-batch cap"


def test_worker_handoff_decision_yields_to_explicit_mode_and_zero_cap():
    src, _ = _class_method_source(WORKER, "BatchGenWorker", "_config_prefill_for_batch")
    decision = src[src.index("resident_handoff = ("):src.index("if resident_handoff:")]
    assert "not reuse_startup_prefill" in decision
    assert '"k3_prefill_moe_mode" not in prefill_debug' in decision
    assert "resident_token_cap > 0" in decision
    assert "resident_ep_prefill_available()" in decision


def test_collective_only_pass_runs_every_resident_moe_layer_with_zero_rows():
    """r23 died with `resident-EP prefill requires the same microbatch count
    on all ranks; got [1, 1, ...]`: D512 waves are asymmetric across nodes.
    A rank short of the global pass count must still join each MoE layer's
    EP-world collectives, with zero local rows, in layer order."""
    import types
    import pytest
    torch = pytest.importorskip("torch")
    src, _ = _class_method_source(
        PSM, "KimiLinearParallelStrategyManager",
        "run_resident_ep_collective_only_prefill")
    ns = {"torch": torch}
    exec(src, ns)

    calls = []

    class _Resident:
        def __init__(self, idx, tp):
            self.idx = idx
            self.latent_tp_size = tp

        def forward(self, x, gate, x_group=None):
            calls.append((self.idx, tuple(x.shape), x.dtype, gate,
                          None if x_group is None else tuple(x_group.shape)))

    def _moe_layer(idx, tp):
        moe = types.SimpleNamespace(gate=f"gate{idx}", _resident_ep_moe=_Resident(idx, tp))
        return types.SimpleNamespace(block_sparse_moe=moe)

    dense = types.SimpleNamespace(block_sparse_moe=None)
    # layer 1 unsharded, layer 2 TP8-sharded latent projections: r26 died in
    # the padding pass with "sharded latent projections need the
    # group-replicated rows" because the (empty) group rows were not passed
    psm = types.SimpleNamespace(
        _resident_ep_built=True,
        model=types.SimpleNamespace(model=types.SimpleNamespace(
            layers=[dense, _moe_layer(1, 1), _moe_layer(2, 8)])),
        loaded_model_config=types.SimpleNamespace(hidden_size=16),
        engine_config=types.SimpleNamespace(Basic_Config=types.SimpleNamespace(
            device_torch=torch.device("cpu"))),
    )
    ns["run_resident_ep_collective_only_prefill"](psm)
    assert calls == [
        (1, (0, 16), torch.bfloat16, "gate1", None),
        (2, (0, 16), torch.bfloat16, "gate2", (0, 16)),
    ]
    psm._resident_ep_built = False
    with pytest.raises(RuntimeError):
        ns["run_resident_ep_collective_only_prefill"](psm)


def test_worker_pads_resident_prefill_passes_instead_of_raising():
    src, _ = _class_method_source(WORKER, "BatchGenWorker", "prefill_prepacked")
    assert "requires the same microbatch count" not in src
    # no world collective may live inside prefill_prepacked's resident branch:
    # ranks with no local rows never call it
    resident_branch = src[src.index("prefill_uses_resident_ep()"):src.index("Prepacked prefill:")]
    assert "all_gather" not in resident_branch and "all_reduce" not in resident_branch
    gen, _ = _class_method_source(WORKER, "BatchGenWorker", "generate")
    pad = gen[gen.index("if transport_only_passes:"):gen.index("prefill_time += time.perf_counter() - transport_start")]
    assert "run_resident_ep_collective_only_prefill" in pad
    # the rank-count sync precedes the collective-only pass, as in a real pass
    assert pad.index("_sync_prefill_moe_rank_counts") < pad.index("run_resident_ep_collective_only_prefill")
    align, _ = _class_method_source(WORKER, "BatchGenWorker", "_streamed_sp8_prefill_pass_alignment")
    assert "prefill_uses_resident_ep" in align


def test_decode_profile_flag_captures_once_after_warmup():
    """batchgen_debug.k3_decode_profile_steps wraps N decode steps of rank 0 in
    torch.profiler after 3 warm-up steps, once per server, and is inert
    without the flag."""
    import types
    import pytest
    torch = pytest.importorskip("torch")
    src, _ = _class_method_source(WORKER, "BatchGenWorker", "_k3_decode_profile_step")
    logs = []
    # the dev box may have no CUDA: the method's only device call is the
    # synchronize before closing the capture
    torch_shim = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda d=None: None))
    ns = {"torch": torch_shim, "logging": types.SimpleNamespace(info=lambda *a, **k: logs.append(a))}
    exec(src, ns)
    fn = ns["_k3_decode_profile_step"]

    class _Prof:
        entered = 0
        exited = 0

        def __enter__(self):
            _Prof.entered += 1

        def __exit__(self, *a):
            _Prof.exited += 1

        def key_averages(self):
            return types.SimpleNamespace(table=lambda **k: "TABLE")

    import torch.profiler as tp
    real_profile = tp.profile
    tp.profile = lambda **k: _Prof()
    try:
        w = types.SimpleNamespace(rank=0, torch_device=torch.device("cpu"))
        for _ in range(10):
            fn(w, None, 8)
        assert _Prof.entered == 0
        for i in range(1, 8):
            fn(w, {"k3_decode_profile_steps": 2}, 8)
            if i == 4:
                assert _Prof.entered == 1 and _Prof.exited == 0
        assert _Prof.entered == 1 and _Prof.exited == 1
        assert any("TABLE" in str(a) for a in logs)
        assert w._k3_decode_profile_done
        for _ in range(10):
            fn(w, {"k3_decode_profile_steps": 2}, 8)
        assert _Prof.entered == 1
        r1 = types.SimpleNamespace(rank=1, torch_device=torch.device("cpu"))
        for _ in range(10):
            fn(r1, {"k3_decode_profile_steps": 2}, 8)
        assert _Prof.entered == 1
    finally:
        tp.profile = real_profile
