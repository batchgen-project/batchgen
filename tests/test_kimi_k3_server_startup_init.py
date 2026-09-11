"""Kimi-K3 pre-readiness startup. CPU only: no weights, no GPU, no CUDA.

The HTTP server used to signal worker readiness right after BatchGenWorker(args)
and then do every one-time build — model, resident EP decode shards,
streamed-SP8 buffers, H2D weight schedule — on the first admission. These gates
pin the replacement contract: the workload-independent work happens before the
readiness event, only for the exact Kimi-K3 model id, exactly once, and the
first real prefill inherits it instead of rebuilding it.
"""

import ast
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
MAIN_LOOP = ROOT / "batchgen" / "server_worker_main_loop.py"
KIMI_LINEAR_SERVING = (
    ROOT
    / "batchgen"
    / "models"
    / "moonshotai"
    / "kimi_linear"
    / "serving_modules.py"
)
KIMI_LINEAR_GRAPH = KIMI_LINEAR_SERVING.with_name("cuda_graph_segments.py")


def _function(path, function_name, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    return next(
        node
        for node in body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _isolated_method(path, class_name, function_name, globals_=None):
    """Compile ONE method out of its module, with no module-level imports."""
    method = copy.deepcopy(_function(path, function_name, class_name))
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Isolated",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return getattr(namespace["Isolated"], function_name)


def _module_constant(path, name):
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )
    return ast.literal_eval(node.value)


def _body_source(function):
    """``function`` unparsed without its docstring."""
    body = copy.deepcopy(function).body
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _calls(function):
    """``{called_name: [lineno, ...]}`` for every call in ``function``."""
    found = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            found.setdefault(node.func.attr, []).append(node.lineno)
        elif isinstance(node.func, ast.Name):
            found.setdefault(node.func.id, []).append(node.lineno)
    return found


def _startup_guard():
    """The ``if`` statement that gates the K3 startup call in the main loop."""
    impl = _function(MAIN_LOOP, "_server_worker_main_impl")
    return next(
        node
        for node in ast.walk(impl)
        if isinstance(node, ast.If)
        and "prepare_kimi_k3_startup" in _calls(node)
    )


# --------------------------------------------------------------------------- #
#  (4) ordering: the startup work precedes the barrier and the ready event
# --------------------------------------------------------------------------- #


def test_kimi_k3_startup_runs_before_the_all_worker_barrier_and_ready_signal():
    impl = _function(MAIN_LOOP, "_server_worker_main_impl")
    calls = _calls(impl)

    startup = calls["prepare_kimi_k3_startup"]
    ready = calls["_signal_local_worker_manager_ready"]
    assert len(startup) == 1 and len(ready) == 1

    # The watchdog must already be attached: the model build below is the
    # longest single stretch of work the worker ever does unsupervised.
    assert max(calls["set_watchdog"]) < startup[0]

    # ...and the readiness event, plus the all-worker barrier that guards it,
    # must both trail it, so no rank reports ready with the build outstanding.
    barrier_before_ready = max(line for line in calls["barrier"] if line < ready[0])
    assert startup[0] < barrier_before_ready < ready[0]

    # The worker is instantiated before any of this, and nothing responds to
    # requests until the main loop, which starts after the ready signal.
    assert max(calls["BatchGenWorker"]) < startup[0]


def test_kimi_k3_startup_failure_is_not_caught_and_so_blocks_readiness():
    guard = _startup_guard()
    # A try/except around the call would let a half-built worker go on to
    # signal ready. server_worker_main's own handler logs and _exit(1)s.
    assert not any(isinstance(node, ast.Try) for node in ast.walk(guard))


def test_decode_graph_reserves_kda_scratch_before_sequence_admission():
    """The graph scratch consumes one pool item during startup, not decode."""

    class SlotManager:
        def __init__(self, capacity):
            self.free = list(range(capacity))
            self.owners = {}

        def alloc(self, seq_id):
            if seq_id in self.owners:
                return self.owners[seq_id]
            if not self.free:
                raise RuntimeError("Insufficient free KDA state items")
            slot = self.free.pop()
            self.owners[seq_id] = slot
            return slot

    # Four user sequences plus one dedicated graph-scratch item.
    slots = SlotManager(capacity=5)
    wrapper = SimpleNamespace(slot_manager=slots)
    original_model_forward = object()
    original_layer_forward = object()
    inner = SimpleNamespace(
        forward=original_model_forward,
        layers=[SimpleNamespace(forward=original_layer_forward)],
    )
    graph = SimpleNamespace(
        _installed=False,
        _scratch_slot=None,
        _uses_block_residual=False,
        model=SimpleNamespace(model=inner),
        _orig_model_forward=None,
        _orig_new_block_residual=None,
        _orig_layer_forwards={},
        _make_model_forward=lambda orig: ("model", orig),
        _make_layer_forward=lambda layer_idx, orig: ("layer", layer_idx, orig),
        rank=1,
        mode="eager",
        bucketing=SimpleNamespace(bucket_sizes=[1, 2, 4]),
        compare_every=64,
    )
    install = _isolated_method(
        KIMI_LINEAR_GRAPH,
        "KimiLinearDecodeGraph",
        "install",
        {
            "KimiLinearKDAWrapper": wrapper,
            "GRAPH_SCRATCH_SEQ_ID": -1_000_001,
            "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
        },
    ).__get__(graph)

    install()

    assert graph._scratch_slot == slots.owners[-1_000_001]
    assert len(slots.free) == 4
    for sequence_id in range(4):
        slots.alloc(sequence_id)
    with pytest.raises(RuntimeError, match="Insufficient free KDA state items"):
        slots.alloc(4)

    # Idempotent re-installation must not consume another state item.
    install()
    assert len(slots.owners) == 5

    ensure_built = _function(
        KIMI_LINEAR_GRAPH,
        "_ensure_built",
        class_name="KimiLinearDecodeGraph",
    )
    assert "alloc" not in _calls(ensure_built), (
        "KDA scratch allocation must not return to first-decode lazy init"
    )


# --------------------------------------------------------------------------- #
#  (4) gating: the exact served model id, never a family match
# --------------------------------------------------------------------------- #


def test_kimi_k3_startup_is_gated_on_the_exact_model_id():
    model_id = _module_constant(WORKER, "KIMI_K3_MODEL_ID")
    assert model_id == "moonshotai/Kimi-K3"

    test = _startup_guard().test

    def matches(name):
        return eval(
            compile(
                ast.Expression(ast.fix_missing_locations(copy.deepcopy(test))),
                "<guard>",
                "eval",
            ),
            {"KIMI_K3_MODEL_ID": model_id, "args": SimpleNamespace(model_name=name)},
        )

    assert matches("moonshotai/Kimi-K3")
    # Every sibling that shares the Kimi-Linear architecture, the bare name and
    # a case variant must all miss: they do not share this lifecycle.
    for other in (
        "moonshotai/Kimi-K2.5",
        "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "moonshotai/Kimi-K3-Instruct",
        "Kimi-K3",
        "moonshotai/kimi-k3",
        "zai-org/GLM-5",
    ):
        assert not matches(other), other

    # One equality against the served model id — no substring, prefix or
    # case-folded family match anywhere in the predicate.
    assert ast.unparse(test) == "args.model_name == KIMI_K3_MODEL_ID"


# --------------------------------------------------------------------------- #
#  (2) the startup pass itself: order, arguments, idempotence
# --------------------------------------------------------------------------- #


def _startup_worker(trace, *, default_mode="streamed_sp8"):
    """A worker whose every collaborator records instead of running."""
    manager = SimpleNamespace(
        set_comm=lambda comm: trace.append(("set_comm", comm)),
        default_prefill_moe_mode=lambda: default_mode,
        set_prefill_moe_mode=lambda mode: trace.append(("set_prefill_moe_mode", mode)),
        prepare_resident_ep_prefill_output=lambda n: trace.append(("prefill_output", n)),
    )

    def configure_prefill():
        trace.append(("configure_prefill",))
        return "prefill-model", {"routed_expert": ["e0"]}

    manager.configure_prefill = configure_prefill
    worker = SimpleNamespace(
        rank=0,
        comm="pynccl",
        model=None,
        weight_copy_task=None,
        parallel_manager=manager,
        _k3_startup_completed=False,
        _k3_startup_prefill_ready=False,
    )
    worker.Init = lambda *args, **kwargs: trace.append(("Init", args, kwargs))
    worker._preload_kimi_k3_runtime_extensions = lambda: trace.append(
        ("preload_runtime_extensions",)
    )
    worker._ensure_pynccl_communicator = lambda: trace.append(("ensure_pynccl",))
    worker._load_decode_model = lambda bsz, comm=None: trace.append(
        ("load_decode_model", bsz, comm)
    )
    worker.set_phase = lambda phase: trace.append(("set_phase", phase))
    worker._install_prefill_weight_copy_pipeline = lambda k3_prefill_profile: trace.append(
        ("install_h2d", k3_prefill_profile)
    )
    return worker


def _bind_startup(worker):
    return _isolated_method(
        WORKER,
        "BatchGenWorker",
        "prepare_kimi_k3_startup",
        {
            "logging": SimpleNamespace(info=lambda *a, **k: None),
            "_K3_STARTUP_MAX_DECODING_LENGTH": _module_constant(
                WORKER, "_K3_STARTUP_MAX_DECODING_LENGTH"
            ),
            "_MAX_DECODE_RANK_BSZ": 128,
        },
    ).__get__(worker)


def test_kimi_k3_startup_builds_decode_shards_then_the_prefill_phase():
    trace = []
    worker = _startup_worker(trace)

    _bind_startup(worker)()

    assert trace == [
        # Core components first — everything below needs the parallel manager.
        ("Init", (None, 4096, 0), {}),
        # Compile/import the exact kernels otherwise first reached by KDA and
        # resident grouped MoE on the first admitted prefill.
        ("preload_runtime_extensions",),
        ("ensure_pynccl",),
        # The resident-EP decode MoE collectives run on this communicator, so
        # the manager holds it before configure_decoding, not at first decode.
        ("set_comm", "pynccl"),
        # Resident decode materializes the stacked MXFP4 EP shard, sized for
        # the largest per-rank batch the runtime decode path can admit.
        ("load_decode_model", 128, "pynccl"),
        # Then the topology's own prefill mode, so the first admission finds
        # the streamed-SP8 buffers already attached.
        ("set_prefill_moe_mode", "streamed_sp8"),
        ("configure_prefill",),
        ("set_phase", "prefill"),
        ("install_h2d", False),
    ]
    assert worker.model == "prefill-model"
    assert worker.weight_copy_task == {"routed_expert": ["e0"]}
    assert worker._k3_startup_completed
    assert worker._k3_startup_prefill_ready
    assert worker._k3_startup_prefill_mode == "streamed_sp8"


class _FlashMLAStub(ModuleType):
    """``flash_mla`` that records which symbols the preload actually reads."""

    def __init__(self, trace, symbols):
        super().__init__("flash_mla")
        self._trace = trace
        self._symbols = symbols

    def __getattr__(self, name):
        if name not in self._symbols:
            raise AttributeError(name)
        self._trace.append(f"flash_mla.{name}")
        return self._symbols[name]


def _bind_preload(monkeypatch, trace, rank, flash_mla):
    """Bind the preload alone. ``flash_mla=None`` makes its import fail."""
    conv1d = ModuleType("batchgen_kernels.conv1d")
    conv1d._get_ext = lambda: trace.append("causal_conv1d") or object()
    kda_fused_decode = ModuleType("batchgen_kernels.attention.kda_fused_decode")
    kda_fused_decode._get_ext = lambda: trace.append("kda_fused_decode") or object()
    dispatch = ModuleType("batchgen.moe.dispatch_scatter_3d")
    dispatch._load_dispatch_reduce_module = (
        lambda: trace.append("dispatch_scatter_3d") or object()
    )
    monkeypatch.setitem(sys.modules, conv1d.__name__, conv1d)
    monkeypatch.setitem(sys.modules, kda_fused_decode.__name__, kda_fused_decode)
    monkeypatch.setitem(sys.modules, dispatch.__name__, dispatch)
    # A None entry makes ``import flash_mla`` raise regardless of what is
    # installed on the host running the test.
    monkeypatch.setitem(sys.modules, "flash_mla", flash_mla)

    worker = SimpleNamespace(
        rank=rank,
        _warmup_kimi_k3_flash_mla=lambda module: trace.append("flash_mla.warmup"),
    )
    return _isolated_method(
        WORKER,
        "BatchGenWorker",
        "_preload_kimi_k3_runtime_extensions",
        {"logging": SimpleNamespace(info=lambda *a, **k: None)},
    ).__get__(worker)


def test_kimi_k3_startup_preloads_the_first_forward_extensions_and_flashmla(
    monkeypatch,
):
    trace = []
    flash_mla = _FlashMLAStub(
        trace,
        {"flash_mla_with_kvcache": lambda: None, "get_mla_metadata": lambda: None},
    )
    _bind_preload(monkeypatch, trace, 7, flash_mla)()

    # FlashMLA is validated last, but still inside the one pre-readiness pass:
    # both symbols the K3 NoPE decode consumes are resolved before the barrier.
    assert trace == [
        "causal_conv1d",
        "dispatch_scatter_3d",
        "kda_fused_decode",
        "flash_mla.flash_mla_with_kvcache",
        "flash_mla.get_mla_metadata",
        "flash_mla.warmup",
    ]


def test_kimi_k3_startup_fails_closed_when_flash_mla_is_not_importable(monkeypatch):
    preload = _bind_preload(monkeypatch, [], 3, None)

    # This is the exact first-decode ModuleNotFoundError, moved ahead of
    # readiness: an unimportable flash_mla must never reach an HTTP-ready rank.
    with pytest.raises(
        RuntimeError,
        match="Rank 3: required Kimi-K3 extension flash_mla failed to load",
    ):
        preload()


@pytest.mark.parametrize(
    "symbols",
    [
        # Symbol absent from the module entirely...
        {"flash_mla_with_kvcache": lambda: None},
        {"get_mla_metadata": lambda: None},
        # ...and present but not callable, i.e. a half-built extension whose
        # binding never resolved.
        {"flash_mla_with_kvcache": lambda: None, "get_mla_metadata": None},
    ],
)
def test_kimi_k3_startup_fails_closed_on_an_invalid_flash_mla(monkeypatch, symbols):
    trace = []
    preload = _bind_preload(monkeypatch, trace, 9, _FlashMLAStub(trace, symbols))

    with pytest.raises(
        RuntimeError,
        match=r"Rank 9: required Kimi-K3 extension flash_mla is missing callable "
        r"(flash_mla_with_kvcache|get_mla_metadata)",
    ):
        preload()


@pytest.mark.parametrize(
    ("path", "function_name"),
    [
        (KIMI_LINEAR_SERVING, "mla_decoding_nope_with_pagekv"),
        (KIMI_LINEAR_GRAPH, "_mla_decode_graph_safe"),
    ],
)
def test_kimi_k3_decode_imports_only_the_validated_flashmla_symbols(
    path, function_name
):
    decode = _function(path, function_name)
    imported = {
        alias.name
        for node in ast.walk(decode)
        if isinstance(node, ast.ImportFrom) and node.module == "flash_mla"
        for alias in node.names
    }
    assert imported == {"flash_mla_with_kvcache", "get_mla_metadata"}

    # The pure-BF16 consumers must not pull in the legacy FP8/FA3 backend:
    # that module eagerly imports DeepGEMM although these K3 forwards do not
    # call it.
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "batchgen.attention.mla.flashmla_backend"
        for node in ast.walk(decode)
    )

    # Whatever the real K3 consumers import is what startup must prove exists;
    # otherwise the readiness check can drift away from the failing path.
    source = _body_source(
        _function(WORKER, "_preload_kimi_k3_runtime_extensions", "BatchGenWorker")
    )
    for symbol in imported:
        assert symbol in source


def test_kimi_k3_graph_mla_preserves_the_tp_output_reduce():
    graph_decode = _function(KIMI_LINEAR_GRAPH, "_mla_decode_graph_safe")
    calls = [
        node
        for node in ast.walk(graph_decode)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_reduce_mla_tp_output"
    ]
    assert len(calls) == 1

    source = _body_source(graph_decode)
    assert source.index("F.linear(attn_output, attn.o_proj.weight)") < source.index(
        "_reduce_mla_tp_output(attn, attn_output)"
    )


def test_kimi_k3_flashmla_warmup_launches_the_real_decode_kernel_and_syncs():
    trace = []

    class FakeTorch:
        int32 = "int32"
        bfloat16 = "bfloat16"

        class cuda:
            @staticmethod
            def current_device():
                trace.append("current_device")
                return 3

            @staticmethod
            def synchronize(device):
                trace.append(("synchronize", device))

        @staticmethod
        def device(kind, index):
            return f"{kind}:{index}"

        @staticmethod
        def ones(shape, *, dtype, device):
            value = ("ones", shape, dtype, device)
            trace.append(value)
            return value

        @staticmethod
        def zeros(shape, *, dtype, device):
            value = ("zeros", shape, dtype, device)
            trace.append(value)
            return value

    class FakeFlashMLA:
        @staticmethod
        def get_mla_metadata(cache_seqlens, num_q_tokens, num_kv_heads):
            trace.append(("metadata", cache_seqlens, num_q_tokens, num_kv_heads))
            return "metadata", "splits"

        @staticmethod
        def flash_mla_with_kvcache(*args, **kwargs):
            trace.append(("decode", args, kwargs))

    warmup = _isolated_method(
        WORKER,
        "BatchGenWorker",
        "_warmup_kimi_k3_flash_mla",
        {"torch": FakeTorch},
    ).__get__(SimpleNamespace(rank=11))
    warmup(FakeFlashMLA)

    assert trace[0] == "current_device"
    assert ("zeros", (1, 1, 12, 576), "bfloat16", "cuda:3") in trace
    assert ("zeros", (1, 64, 1, 576), "bfloat16", "cuda:3") in trace
    metadata = next(item for item in trace if item[0] == "metadata")
    assert metadata[2:] == (12, 1)
    decode = next(item for item in trace if item[0] == "decode")
    assert decode[1][4] == 512
    assert decode[2] == {"causal": True}
    assert trace[-1] == ("synchronize", "cuda:3")


def test_kimi_k3_startup_fails_closed_when_a_runtime_extension_is_missing(
    monkeypatch,
):
    conv1d = ModuleType("batchgen_kernels.conv1d")
    conv1d._get_ext = lambda: object()
    dispatch = ModuleType("batchgen.moe.dispatch_scatter_3d")
    dispatch._load_dispatch_reduce_module = lambda: None
    monkeypatch.setitem(sys.modules, conv1d.__name__, conv1d)
    monkeypatch.setitem(sys.modules, dispatch.__name__, dispatch)

    worker = SimpleNamespace(rank=14)
    preload = _isolated_method(
        WORKER,
        "BatchGenWorker",
        "_preload_kimi_k3_runtime_extensions",
        {"logging": SimpleNamespace(info=lambda *a, **k: None)},
    ).__get__(worker)

    with pytest.raises(
        RuntimeError,
        match="Rank 14: required Kimi-K3 extension dispatch_scatter_3d failed to load",
    ):
        preload()


def test_kimi_k3_startup_uses_the_runtime_decode_batch_cap():
    # 128 is the cap the decode path applies to its own per-rank estimate, so
    # the padded MoE buffers built here already fit the worst admitted batch.
    assign = next(
        node
        for node in ast.parse(WORKER.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_MAX_DECODE_RANK_BSZ"
            for t in node.targets
        )
    )
    assert "128" in ast.unparse(assign)
    assert "_MAX_DECODE_RANK_BSZ" in ast.unparse(
        _function(WORKER, "prepare_kimi_k3_startup", "BatchGenWorker")
    )


def test_kimi_k3_startup_is_idempotent():
    trace = []
    worker = _startup_worker(trace)
    startup = _bind_startup(worker)

    startup()
    first = list(trace)
    startup()
    startup()

    # A second call must not re-Init, re-enter configure_prefill or reinstall
    # the H2D queue — the weight daemon walks one schedule for the process.
    assert trace == first


def test_kimi_k3_startup_follows_the_topology_default_prefill_mode():
    trace = []
    worker = _startup_worker(trace, default_mode="streamed")
    _bind_startup(worker)()

    assert ("set_prefill_moe_mode", "streamed") in trace
    # The mode comes from the manager, never from a literal in the worker.
    source = _body_source(
        _function(WORKER, "prepare_kimi_k3_startup", "BatchGenWorker")
    )
    assert "default_prefill_moe_mode()" in source
    for literal in ("'streamed_sp8'", "'resident_ep'", "'streamed'"):
        assert literal not in source


def test_kimi_k3_startup_leaves_request_shaped_allocations_to_admission():
    trace = []
    worker = _startup_worker(trace)
    _bind_startup(worker)()

    # The QueryBook buffer pool needs tokenized prompt/decode widths and the
    # resident prefill output needs the admitted micro-batch token count;
    # neither is known before a request arrives.
    called = {entry[0] for entry in trace}
    assert "prefill_output" not in called
    source = _body_source(
        _function(WORKER, "prepare_kimi_k3_startup", "BatchGenWorker")
    )
    assert "_ensure_buffer_pool" not in source
    assert "prepare_resident_ep_prefill_output" not in source


def test_kimi_decode_graph_is_prewarmed_before_continuous_decode():
    """The first measured decode forward must not capture graphs lazily."""
    generate = _function(WORKER, "generate", "BatchGenWorker")
    calls = _calls(generate)
    assert len(calls["prewarm_decode_graphs"]) == 1
    assert len(calls["decoding_continuous"]) == 1
    assert calls["prewarm_decode_graphs"][0] < calls["decoding_continuous"][0]

    source = _body_source(generate)
    assert "self._get_cuda_graph_gpu_manager()" in source


# --------------------------------------------------------------------------- #
#  (3) the H2D queue installation is shared and happens exactly once
# --------------------------------------------------------------------------- #


class _RecordingCoreEngine:
    def __init__(self, trace):
        self._trace = trace

    def __getattr__(self, name):
        def call(*args):
            self._trace.append((name,) + args)

        return call


def _bind_install(worker, trace, monkeypatch):
    expert_module = ModuleType(
        "batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert"
    )
    expert_module.KimiK3MXFP4ExpertWrapper = SimpleNamespace(
        reset_prefill_profile=lambda flag: trace.append(("expert_profile", flag))
    )
    sp8_module = ModuleType("batchgen.moe.streamed_sp8_mxfp4")
    sp8_module.StreamedSP8MXFP4MoELayer = SimpleNamespace(
        reset_prefill_profile=lambda flag: trace.append(("sp8_profile", flag))
    )
    monkeypatch.setitem(sys.modules, expert_module.__name__, expert_module)
    monkeypatch.setitem(sys.modules, sp8_module.__name__, sp8_module)
    worker._weight_copy_task_fingerprint = _isolated_method(
        WORKER, "BatchGenWorker", "_weight_copy_task_fingerprint"
    )
    return _isolated_method(
        WORKER, "BatchGenWorker", "_install_prefill_weight_copy_pipeline"
    ).__get__(worker)


def _install_worker(
    trace, *, streamed_sp8=True, reseed_reentry=False, task=None
):
    return SimpleNamespace(
        rank=0,
        core_engine=_RecordingCoreEngine(trace),
        parallel_manager=SimpleNamespace(
            prefill_uses_streamed_sp8=lambda: streamed_sp8,
            streamed_sp8_reseeds_h2d_on_reentry=lambda: reseed_reentry,
        ),
        weight_copy_task=task if task is not None else {"routed_expert": ["a", "b"]},
        _streamed_sp8_h2d_installed=False,
        _streamed_sp8_weight_copy_fingerprint=None,
    )


def test_host_rdma_streamed_sp8_queue_is_seeded_once_and_resumed(monkeypatch):
    trace = []
    worker = _install_worker(trace)
    install = _bind_install(worker, trace, monkeypatch)

    install(k3_prefill_profile=False)

    # First install: seed the queue and start the copy engine.
    assert trace == [
        ("stop_h2d_worker",),
        ("clear_weight_copy_queue",),
        ("reset_prefill_buffer",),
        ("reset_weight_stream_profile", False),
        ("expert_profile", False),
        ("sp8_profile", False),
        ("set_weight_copy_queue", {"routed_expert": ["a", "b"]}),
        ("start_h2d_worker",),
    ]
    assert worker._streamed_sp8_h2d_installed
    fingerprint = worker._streamed_sp8_weight_copy_fingerprint
    assert fingerprint == (("routed_expert", ("a", "b")),)

    # Re-entry (the first real prefill after startup, and every prefill after a
    # decode phase): the daemon has already erased keys it released, so the
    # cursor must NOT be rewound and the prefill buffer must NOT be reset.
    trace.clear()
    install(k3_prefill_profile=False)
    assert trace == [
        ("stop_h2d_worker",),
        ("reset_weight_stream_profile", False),
        ("expert_profile", False),
        ("sp8_profile", False),
        ("start_h2d_worker",),
    ]
    assert worker._streamed_sp8_weight_copy_fingerprint == fingerprint


def test_hierarchical_gdr_reentry_reseeds_queue_and_ring(monkeypatch):
    trace = []
    worker = _install_worker(trace, reseed_reentry=True)
    install = _bind_install(worker, trace, monkeypatch)

    install(k3_prefill_profile=False)
    first = list(trace)
    trace.clear()
    install(k3_prefill_profile=False)

    # Hierarchical GDR has no remote-host acquire/release generations. A new
    # admission therefore returns both the task queue and GPU leases to the
    # layer-zero boundary instead of inheriting a partial ring from decode.
    assert trace == first


def test_streamed_sp8_reentry_rejects_a_changed_weight_schedule(monkeypatch):
    trace = []
    worker = _install_worker(trace)
    install = _bind_install(worker, trace, monkeypatch)
    install(k3_prefill_profile=False)

    # Same experts, different order: the copy engine drains each list front to
    # front, so the preserved cursor would stream the wrong expert.
    worker.weight_copy_task = {"routed_expert": ["b", "a"]}
    with pytest.raises(RuntimeError, match="weight-copy schedule changed"):
        install(k3_prefill_profile=False)


def test_non_streamed_prefill_keeps_reseeding_the_queue(monkeypatch):
    trace = []
    worker = _install_worker(trace, streamed_sp8=False)
    install = _bind_install(worker, trace, monkeypatch)

    install(k3_prefill_profile=False)
    install(k3_prefill_profile=False)

    # No preserved cursor outside streamed-SP8: both passes reseed, and the
    # fingerprint stays cleared so a later install cannot claim re-entry.
    assert [entry[0] for entry in trace] == [
        "stop_h2d_worker",
        "clear_weight_copy_queue",
        "reset_prefill_buffer",
        "reset_weight_stream_profile",
        "set_weight_copy_queue",
        "start_h2d_worker",
    ] * 2
    assert worker._streamed_sp8_h2d_installed is False
    assert worker._streamed_sp8_weight_copy_fingerprint is None


def test_startup_and_per_batch_prefill_share_one_installer():
    startup = _function(WORKER, "prepare_kimi_k3_startup", "BatchGenWorker")
    config = _function(WORKER, "_config_prefill_for_batch", "BatchGenWorker")

    for function in (startup, config):
        assert "_install_prefill_weight_copy_pipeline" in _calls(function)
        # The install invariants live in one place only.
        source = _body_source(function)
        assert "set_weight_copy_queue" not in source
        assert "_streamed_sp8_h2d_installed" not in source


# --------------------------------------------------------------------------- #
#  (5) the first real prefill inherits the startup-prepared phase
# --------------------------------------------------------------------------- #


def test_first_prefill_after_startup_does_not_free_the_prepared_model():
    config = _function(WORKER, "_config_prefill_for_batch", "BatchGenWorker")
    guard = next(
        node
        for node in ast.walk(config)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "reuse_startup_prefill"
    )

    # Startup-prepared phase: keep model, streamed-SP8 buffers and the daemon
    # cursor; configure_prefill below re-enters idempotently.
    assert "deep_free_model_memory" not in _calls(guard.test)
    kept = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert "deep_free_model_memory" not in kept
    # ...and the flag is consumed here, exactly once.
    assert "self._k3_startup_prefill_ready = False" in kept

    # Every later prefill follows a decode phase and frees normally.
    freed = ast.unparse(ast.Module(body=guard.orelse, type_ignores=[]))
    assert "self.deep_free_model_memory()" in freed

    # That guarded call is the only deep free in the whole method. The normal
    # handoff also skips the otherwise-idempotent configure/H2D reinstall so
    # no workload-independent phase work leaks back onto first admission.
    assert len(_calls(config)["deep_free_model_memory"]) == 1
    source = _body_source(config)
    assert "if not reuse_startup_prefill:" in source
    for call in ("configure_prefill", "_install_prefill_weight_copy_pipeline"):
        assert len(_calls(config)[call]) == 1
        call_line = _calls(config)[call][0]
        enclosing = [
            node for node in ast.walk(config)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "not reuse_startup_prefill"
            and node.lineno < call_line <= max(
                getattr(
                    child,
                    "end_lineno",
                    getattr(child, "lineno", node.lineno),
                )
                or getattr(child, "lineno", node.lineno)
                for child in ast.walk(node)
            )
        ]
        assert enclosing, call


def test_startup_prefill_handoff_flag_starts_false():
    init = _function(WORKER, "__init__", "BatchGenWorker")
    source = ast.unparse(init)
    assert "self._k3_startup_completed = False" in source
    assert "self._k3_startup_prefill_ready = False" in source
    assert "self._k3_startup_prefill_mode = None" in source


# --------------------------------------------------------------------------- #
#  (1) generate() reuses the extracted communicator helper
# --------------------------------------------------------------------------- #


def test_generate_reuses_the_pynccl_communicator_helper():
    generate = _function(WORKER, "generate", "BatchGenWorker")
    calls = _calls(generate)

    assert len(calls["_ensure_pynccl_communicator"]) == 1
    # The creation block moved wholesale: generate() no longer builds either
    # half of the communicator itself.
    source = _body_source(generate)
    assert "PyNcclCommunicator(" not in source
    assert "StatelessProcessGroup.create" not in source
    assert "COMM_MASTER_ADDR" not in source

    # ...and the periodic health check still runs first, unchanged.
    assert calls["_check_and_reinit_pynccl"][0] < calls[
        "_ensure_pynccl_communicator"
    ][0]


def test_pynccl_helper_keeps_the_collective_creation_contract():
    helper = _function(WORKER, "_ensure_pynccl_communicator", "BatchGenWorker")
    source = _body_source(helper)
    calls = _calls(helper)

    # The all-ranks vote, the port broadcast and the pre-TCPStore barrier are
    # collectives: dropping any of them on the idempotent path would hang the
    # ranks that still need to create a communicator.
    assert "all_reduce" in calls and "broadcast" in calls and "barrier" in calls
    assert "any_rank_needs_init" in source
    assert "StatelessProcessGroup.create" in source
    assert "PyNcclCommunicator(" in source
    # Single GPU and the all-to-all build both opt out, as before.
    assert "BATCHGEN_ENABLE_ALL_TO_ALL" in source
    assert "self.world_size == 1" in source


def test_startup_uses_the_same_helper_as_generate():
    startup = _function(WORKER, "prepare_kimi_k3_startup", "BatchGenWorker")
    assert "_ensure_pynccl_communicator" in _calls(startup)
