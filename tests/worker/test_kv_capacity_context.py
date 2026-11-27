"""GPU pool operations run from frozen values and explicit collaborators."""

import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from batchgen.worker.kv_manager import PageTableCapacityRequest
from tests.kv_capacity_source import METHODS, load_kv_capacity, method_node
from tests.worker_method_source import WORKER_SOURCE


GIB = 1024 ** 3


def runtime(*, free=12, total=20, override=None, reduced=None):
    events = []

    def memory(rank):
        events.append(("memory", rank))
        return free * GIB, total * GIB

    def reduce(tensor, op):
        events.append(("reduce", op))
        if reduced is not None:
            tensor.fill_(reduced)

    ns = load_kv_capacity({
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda device: events.append("sync"),
                empty_cache=lambda: events.append("empty"),
                mem_get_info=memory, is_available=lambda: True,
            ),
            tensor=torch.tensor, float32=torch.float32, float64=torch.float64,
        ),
        "dist": SimpleNamespace(
            broadcast=lambda tensor, src: events.append(("broadcast", src)),
            all_reduce=reduce, ReduceOp=SimpleNamespace(MIN="MIN"),
        ),
        "gc": SimpleNamespace(collect=lambda: events.append("gc")),
        "_GPU_KV_CACHE_SIZE_OVERRIDE": override,
        "DualKVCacheCoordinator": type("Dual", (), {}),
    })
    return ns, ns["KVCapacityEngine"](), events


def manager(*, initialized=True, mapped=True, fixed=True, pages=1024):
    return SimpleNamespace(
        is_initialized=initialized, pool_is_mapped=mapped,
        uses_fixed_address_pool=fixed,
        get_stats=lambda: SimpleNamespace(num_total_pages=pages),
    )


@pytest.mark.parametrize("name", sorted(METHODS))
def test_converted_methods_have_no_worker_or_rebinding(name):
    node = method_node(name)
    assert [arg.arg for arg in node.args.args[:2]] == ["self", "ctx"]
    assert not any(isinstance(n, ast.Name) and n.id in {"self", "worker"}
                   for stmt in node.body for n in ast.walk(stmt))
    assert not any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.GeneratorExp))
                   for n in ast.walk(node))
    for arg in node.args.kwonlyargs:
        assert "Callable" in ast.unparse(arg.annotation)


def test_frozen_contexts_and_exact_outcome_fields():
    ns, _, _ = runtime()
    expected = {
        "GPUKVSizeOutcome": {"gpu_kv_cache_size_gb"},
        "GPUKVBindOutcome": {"gpu_paged_kv_cache_manager", "_gpu_kv_pool_created"},
        "GPUKVInitOutcome": {
            "gpu_kv_cache_size_gb", "_gpu_kv_sizing_provisional", "_gpu_kv_cached_pool_gb",
        },
        "GPUKVFinalizeOutcome": {"_gpu_kv_sizing_provisional", "_gpu_kv_cached_pool_gb"},
    }
    for name, cls in ns.items():
        if not inspect.isclass(cls) or not hasattr(cls, "__dataclass_fields__"):
            continue
        if name == "PageTableCapacityRequest":
            continue
        members = fields(cls)
        value = cls(**{field.name: None for field in members})
        assert not any("Callable" in str(field.type) for field in members)
        assert not any(callable(getattr(value, field.name)) for field in members)
        with pytest.raises(FrozenInstanceError):
            setattr(value, members[0].name, object())
        if name in expected:
            assert {field.name for field in members} == expected[name]


@pytest.mark.parametrize("rank,override,expected", [(0, None, 7), (0, "4.5", 4.5), (1, "4.5", 4.5)])
def test_memory_calculation_and_override(rank, override, expected):
    ns, engine, events = runtime(override=override)
    ctx = ns["GPUKVMemoryContext"](rank, 0, "cpu", 0.75)
    assert engine._calculate_gpu_kv_cache_size(ctx) == expected
    assert events == ([] if override else [("memory", 0), ("broadcast", 0)])


def test_nonzero_rank_only_broadcasts_and_nonpositive_budget_keeps_floor():
    ns, engine, events = runtime(free=1)
    assert engine._calculate_gpu_kv_cache_size(ns["GPUKVMemoryContext"](0, 0, "cpu", 0.75)) == 1
    events.clear()
    assert engine._calculate_gpu_kv_cache_size(ns["GPUKVMemoryContext"](1, 0, "cpu", 0.75)) == 0
    assert events == [("broadcast", 0)]


@pytest.mark.parametrize("device,available,expected", [(0, True, True), ("cpu", True, False), (0, False, False)])
def test_fixed_address_availability(device, available, expected):
    ns, engine, _ = runtime()
    ns["torch"].cuda.is_available = lambda: available
    assert engine._kv_fixed_address_pool_available(ns["GPUKVDeviceContext"](device)) is expected


@pytest.mark.parametrize("bound", [None, 0, 4096])
def test_deployment_bound(bound):
    ns, engine, _ = runtime()
    assert engine._deployment_context_bound_tokens(ns["DeploymentContext"](bound)) == (bound or 0)


@pytest.mark.parametrize("configured", [False, True])
def test_page_table_request_uses_deployment_and_only_reads_configured_ladder(configured):
    ns, engine, events = runtime()
    config = SimpleNamespace(
        Basic_Config=SimpleNamespace(num_queries=32, decode_graph_buckets=[1, 160]),
        Module_Batching_Config=SimpleNamespace(global_batch_size=64, attn_decoding_micro_batch_size=8),
    ) if configured else None
    ctx = ns["PageTableContext"](
        config, SimpleNamespace(max_position_embeddings=8192),
        SimpleNamespace(cuda_graph_max_bucket_size=128),
    )

    def bound():
        events.append("bound")
        return 4096

    def bucket():
        events.append("bucket")
        return 192

    request = engine._make_page_table_capacity_request(
        ctx, _deployment_context_bound_tokens=bound, _decode_graph_top_bucket=bucket,
    )
    assert request.model_max_position_embeddings == 4096
    assert request.decode_graph_top_bucket == (192 if configured else None)
    assert request.engine_basic_decode_graph_max_bucket == (160 if configured else None)
    assert events == (["bound", "bucket"] if configured else ["bound"])


def test_page_table_config_is_adjusted_from_the_resolved_request():
    ns, engine, _ = runtime()
    request = PageTableCapacityRequest(
        engine_module_global_batch_size=1, engine_module_attn_decoding_micro_batch_size=1,
        engine_basic_num_queries=1, model_max_position_embeddings=4096,
        args_cuda_graph_max_bucket_size=128, decode_graph_top_bucket=160,
    )
    seen = []
    config = object()
    ns["KVCacheManager"] = SimpleNamespace(apply_page_table_capacity=lambda req, cfg: seen.append((req, cfg)) or cfg)
    assert engine._with_cuda_graph_page_table_capacity(request, config) is config
    assert seen == [(request, config)]


@pytest.mark.parametrize("dual", [False, True])
def test_bind_returns_worker_fields_and_binds_core_slots(dual):
    ns, engine, _ = runtime()
    pool = ns["DualKVCacheCoordinator"]() if dual else manager()
    if dual:
        pool.primary, pool.auxiliary = object(), object()
    core = SimpleNamespace(gpu_paged_kv_manager=None, gpu_paged_kv_manager_aux=None)
    outcome = engine._bind_gpu_paged_kv_manager(ns["GPUKVBindContext"](core), pool)
    assert outcome.gpu_paged_kv_cache_manager is pool
    assert outcome._gpu_kv_pool_created is True
    assert core.gpu_paged_kv_manager is (pool.primary if dual else pool)
    assert core.gpu_paged_kv_manager_aux is (pool.auxiliary if dual else None)


@pytest.mark.parametrize("kind", ["missing", "single", "dual"])
def test_graph_manager_handle_does_not_change_graph_ownership(kind):
    ns, engine, _ = runtime()
    pool = None if kind == "missing" else ns["DualKVCacheCoordinator"]() if kind == "dual" else manager()
    fallback = object()
    ctx = ns["GPUKVHandleContext"](pool, SimpleNamespace(gpu_paged_kv_manager=fallback))
    assert engine._get_cuda_graph_gpu_manager(ctx) is (fallback if pool is None else pool)
    assert {field.name for field in fields(ctx)} == {"gpu_paged_kv_cache_manager", "core_engine"}


@pytest.mark.parametrize("dual,initial_size", [(False, None), (False, 2.0), (True, None), (True, 2.0)])
def test_fixed_size_creation_preserves_config_initialize_bind_order(monkeypatch, dual, initial_size):
    ns, engine, events = runtime()
    profile = SimpleNamespace(
        bytes_per_page=lambda: 1024, num_layers=2, page_size=64,
        num_k_heads=1, k_head_dim=4, num_v_heads=0, v_head_dim=0,
        kv_dtype="float16", logical_to_physical_layer=[0, -1, 1],
    )
    config_module = ModuleType("batchgen.kv_cache.host_kv_mananger_config")
    config_module.build_gpu_kv_config_fixed_size = lambda **kw: SimpleNamespace(num_pages=512, **kw)
    config_module.is_dsa_model = lambda name: dual
    config_module._resolve_profile = lambda name: profile
    config_module._resolve_indexer_profile = lambda name: profile
    config_module._torch_dtype_from_string = lambda value: value
    gpu_module = ModuleType("batchgen.kv_cache.gpu_paged_kv_manager")
    gpu_module.GPUPagedKVConfig = SimpleNamespace
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, gpu_module.__name__, gpu_module)

    class Pool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            events.append("construct")

        def initialize(self):
            events.append("initialize")

    class Dual:
        def __init__(self, primary, auxiliary):
            self.primary, self.auxiliary = primary, auxiliary
            events.append("coordinate")

    def capacity(config):
        events.append("capacity")
        return config

    ns["GPUPagedKVCacheManager"], ns["DualKVCacheCoordinator"] = Pool, Dual
    ctx = ns["GPUKVFixedSizeContext"](0, 0, "model", initial_size)
    pool, outcome = engine._initialize_gpu_kv_manager_fixed_size(
        ctx, _calculate_gpu_kv_cache_size=lambda: events.append("calculate") or 2.0,
        _with_cuda_graph_page_table_capacity=capacity,
        _kv_fixed_address_pool_available=lambda: events.append("fixed") or True,
        _bind_gpu_paged_kv_manager=lambda pool: events.append("bind"),
    )
    prefix = ["calculate"] if initial_size is None else []
    assert events == prefix + (
        ["capacity", "capacity", "fixed", "construct", "initialize", "construct", "initialize", "coordinate", "bind"]
        if dual else ["capacity", "fixed", "construct", "initialize", "bind"]
    )
    assert outcome.gpu_kv_cache_size_gb == (2.0 if initial_size is None else ns["GPU_KV_UNCHANGED"])
    if dual:
        assert pool.primary.config.num_pages == 2 * GIB // 4096
        assert pool.auxiliary.config.logical_to_physical_layer == [0, -1, 1]
    assert ctx.gpu_kv_cache_size_gb == initial_size


def init_context(ns, **changes):
    values = dict(rank=0, local_rank=0, torch_device="cpu", gpu_memory_frac=0.75,
                  gpu_paged_kv_cache_manager=None, _phase="decode", _gpu_kv_cached_pool_gb=None)
    return ns["GPUKVInitContext"](**(values | changes))


@pytest.mark.parametrize("cached", [None, 6.0])
def test_actual_size_uses_minimum_rank_or_cached_budget_and_returns_binds(cached):
    ns, engine, events = runtime(reduced=6)
    ctx = init_context(ns, _gpu_kv_cached_pool_gb=cached)

    def allocate(*, gpu_kv_cache_size_gb):
        events.append(("allocate", gpu_kv_cache_size_gb))
        return manager()

    outcome = engine._init_gpu_kv_with_actual_size(
        ctx, _remap_reused_kv_pool=lambda: pytest.fail("unexpected remap"),
        _initialize_gpu_kv_manager_fixed_size=allocate,
        _verify_deployment_max_model_len=lambda size: events.append(("verify", size)),
    )
    assert outcome.gpu_kv_cache_size_gb == 6
    if cached is None:
        assert events == ["sync", "empty", ("memory", 0), ("verify", 7), ("reduce", "MIN"), ("allocate", 6)]
        assert outcome._gpu_kv_cached_pool_gb == 6
        assert outcome._gpu_kv_sizing_provisional is False
    else:
        assert events == ["sync", "gc", "empty", ("memory", 0), ("allocate", 6)]
        assert outcome._gpu_kv_cached_pool_gb is ns["GPU_KV_UNCHANGED"]
        assert outcome._gpu_kv_sizing_provisional is ns["GPU_KV_UNCHANGED"]


@pytest.mark.parametrize("state", ["ready", "remap", "wrong_phase", "shortfall", "verify_error"])
def test_actual_size_early_and_fatal_paths(state):
    ns, engine, events = runtime(free=1 if state == "shortfall" else 12)
    ctx = init_context(ns)
    if state in {"ready", "remap"}:
        ctx = replace(ctx, gpu_paged_kv_cache_manager=manager(initialized=state == "ready"))
    if state == "wrong_phase":
        ctx = replace(ctx, _phase="prefill")
    if state == "shortfall":
        ctx = replace(ctx, _gpu_kv_cached_pool_gb=6)

    def refuse(size):
        raise RuntimeError("verification refused")

    kwargs = dict(
        _remap_reused_kv_pool=lambda: events.append("remap"),
        _initialize_gpu_kv_manager_fixed_size=lambda **kw: pytest.fail("unexpected allocation"),
        _verify_deployment_max_model_len=refuse,
    )
    if state in {"ready", "remap"}:
        assert engine._init_gpu_kv_with_actual_size(ctx, **kwargs) is None
        assert events == (["remap"] if state == "remap" else [])
    else:
        with pytest.raises(RuntimeError):
            engine._init_gpu_kv_with_actual_size(ctx, **kwargs)
    assert ctx._gpu_kv_cached_pool_gb == (6 if state == "shortfall" else None)


def test_finalize_returns_cache_keeps_live_pool_and_refuses_floor():
    ns, engine, events = runtime(free=5)
    ctx = ns["GPUKVFinalizeContext"](0, 0, "cpu", 0.75, manager(), 4, True,
                                       SimpleNamespace(bucket_sizes=[1, 128]))
    outcome = engine._finalize_gpu_kv_pool_after_capture(ctx, ctx.gpu_paged_kv_cache_manager)
    assert outcome._gpu_kv_cached_pool_gb == 7
    assert outcome._gpu_kv_sizing_provisional is False
    assert ctx._gpu_kv_sizing_provisional is True
    assert ctx.gpu_kv_cache_size_gb == 4
    assert events == ["sync", "empty", ("memory", 0), ("broadcast", 0)]
    events.clear()
    assert engine._finalize_gpu_kv_pool_after_capture(replace(ctx, _gpu_kv_sizing_provisional=False), None) is None
    assert events == []
    with pytest.raises(RuntimeError, match="below floor"):
        engine._finalize_gpu_kv_pool_after_capture(replace(ctx, whole_model_bucketing=SimpleNamespace(bucket_sizes=[4096])), None)
    assert ctx._gpu_kv_sizing_provisional is True


def test_release_asserts_before_destroy_and_remap_initializes_once():
    ns, engine, events = runtime()
    pool = manager(initialized=False, mapped=False)
    pool.initialize = lambda: events.append("initialize")
    ctx = ns["GPUKVPoolContext"](0, pool, True, 4)
    engine._release_suspended_decode_kv_pool(
        ctx, _assert_kv_pool_addresses_are_fixed=lambda phase: events.append(phase),
        _destroy_gpu_paged_kv_cache=lambda: events.append("destroy"),
    )
    assert events == ["decode->prefill", "destroy"]
    events.clear()
    engine._remap_reused_kv_pool(ctx, _assert_kv_pool_addresses_are_fixed=lambda phase: events.append(phase))
    assert events == ["prefill->decode", "initialize"]
    events.clear()
    pool.pool_is_mapped = True
    engine._remap_reused_kv_pool(ctx, _assert_kv_pool_addresses_are_fixed=lambda phase: events.append(phase))
    assert events == ["prefill->decode"]
    empty = ns["GPUKVPoolContext"](0, None)
    assert engine._remap_reused_kv_pool(empty, _assert_kv_pool_addresses_are_fixed=lambda _: pytest.fail("unexpected assertion")) is None
    with pytest.raises(RuntimeError, match="already created"):
        engine._remap_reused_kv_pool(replace(empty, _gpu_kv_pool_created=True), _assert_kv_pool_addresses_are_fixed=lambda _: None)


@pytest.mark.parametrize("name,state,error", [
    ("_assert_kv_pool_addresses_are_fixed", "none", None),
    ("_assert_kv_pool_addresses_are_fixed", "ready", None),
    ("_assert_kv_pool_addresses_are_fixed", "moving", "not held at fixed addresses"),
    ("_assert_kv_region_mapped", "none", None),
    ("_assert_kv_region_mapped", "missing", "unmapped"),
    ("_assert_kv_region_mapped", "uninitialized", "unmapped"),
    ("_assert_kv_region_mapped", "released", "physical pages unmapped"),
    ("_assert_kv_region_mapped", "ready", None),
    ("_assert_kv_region_released", "none", None),
    ("_assert_kv_region_released", "released", None),
    ("_assert_kv_region_released", "ready", "mapped"),
])
def test_region_assertions(name, state, error):
    ns, engine, _ = runtime()
    pool = None if state in {"none", "missing"} else manager(
        initialized=state != "uninitialized", mapped=state != "released", fixed=state != "moving",
    )
    ctx = ns["GPUKVPoolContext"](0, pool, state != "none")
    if error:
        with pytest.raises(RuntimeError, match=error):
            getattr(engine, name)(ctx, "phase")
    else:
        assert getattr(engine, name)(ctx, "phase") is None


def load_delegators(monkeypatch, ns):
    """Compile the actual worker class with only the delegators under test."""
    module = ModuleType("batchgen.worker.kv_capacity_engine")
    module.__dict__.update(ns)
    monkeypatch.setitem(sys.modules, "batchgen.worker.kv_capacity_engine", module)
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    cls.body = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    cls.bases, cls.decorator_list = [], []
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(code), str(WORKER_SOURCE), "exec"), namespace)
    return namespace["BatchGenWorker"]()


@pytest.mark.parametrize("cached", [None, 6.0])
def test_delegators_pass_new_size_and_bind_nested_outcomes(monkeypatch, cached):
    ns, engine, events = runtime(reduced=6)
    worker = load_delegators(monkeypatch, ns)
    worker._kv_capacity_engine = engine
    worker.rank = worker.local_rank = 0
    worker.torch_device = "cpu"
    worker.gpu_memory_frac = 0.75
    worker.huggingface_ckpt_name = "model"
    worker._phase = "decode"
    worker._gpu_kv_cached_pool_gb = cached
    worker.gpu_kv_cache_size_gb = None
    worker._gpu_kv_sizing_provisional = True
    worker.gpu_paged_kv_cache_manager = None
    worker._gpu_kv_pool_created = False
    worker._held_decode_phase_context = None
    worker.core_engine = SimpleNamespace(gpu_paged_kv_manager=None)
    worker._verify_deployment_max_model_len = lambda size: events.append(("verify", size))
    pool = manager()

    def fixed(ctx, **helpers):
        assert ctx.gpu_kv_cache_size_gb == 6
        assert worker.gpu_kv_cache_size_gb is None  # binding waits for the outer outcome
        helpers["_bind_gpu_paged_kv_manager"](pool)
        assert worker.gpu_paged_kv_cache_manager is pool
        assert worker._gpu_kv_pool_created is True
        assert worker.core_engine.gpu_paged_kv_manager is pool
        return pool, ns["GPUKVSizeOutcome"]()

    engine._initialize_gpu_kv_manager_fixed_size = fixed
    assert worker._init_gpu_kv_with_actual_size() is None
    assert worker.gpu_kv_cache_size_gb == worker._gpu_kv_cached_pool_gb == 6
    assert worker._gpu_kv_sizing_provisional is (cached is not None)
    assert worker._kv_capacity_engine is engine
    before = list(events)
    assert worker._init_gpu_kv_with_actual_size() is None
    assert events == before


def test_delegator_binds_calculated_size_and_finalize_outcomes(monkeypatch):
    ns, engine, _ = runtime(free=5)
    worker = load_delegators(monkeypatch, ns)
    worker._kv_capacity_engine = engine
    worker.rank = worker.local_rank = 0
    worker.torch_device = "cpu"
    worker.gpu_memory_frac = 0.75
    worker.huggingface_ckpt_name = "model"
    worker.gpu_kv_cache_size_gb = None
    worker.gpu_paged_kv_cache_manager = manager()
    pool = worker.gpu_paged_kv_cache_manager
    engine._initialize_gpu_kv_manager_fixed_size = lambda ctx, **helpers: (pool, ns["GPUKVSizeOutcome"](4))
    assert worker._initialize_gpu_kv_manager_fixed_size() is pool
    assert worker.gpu_kv_cache_size_gb == 4
    worker._gpu_kv_sizing_provisional = True
    worker._gpu_kv_cached_pool_gb = None
    graph = SimpleNamespace(whole_model_bucketing=SimpleNamespace(bucket_sizes=[1, 128]))
    worker._graph = graph
    assert worker._finalize_gpu_kv_pool_after_capture(pool) is None
    assert worker._gpu_kv_sizing_provisional is False
    assert worker._gpu_kv_cached_pool_gb == 7
    assert worker.gpu_kv_cache_size_gb == 4
    assert worker._graph is graph


def test_bind_guard_and_fatal_core_assignment_leave_worker_binds_unapplied(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._kv_capacity_engine = engine
    worker.rank = 0
    old_pool, new_pool = object(), object()
    worker.gpu_paged_kv_cache_manager = old_pool
    worker._gpu_kv_pool_created = False
    worker._held_decode_phase_context = object()
    with pytest.raises(RuntimeError, match="context is held"):
        worker._bind_gpu_paged_kv_manager(new_pool)

    class Core:
        @property
        def gpu_paged_kv_manager(self):
            return old_pool

        @gpu_paged_kv_manager.setter
        def gpu_paged_kv_manager(self, value):
            raise RuntimeError("core bind failed")

    worker.core_engine = Core()
    worker._held_decode_phase_context = None
    with pytest.raises(RuntimeError, match="core bind failed"):
        worker._bind_gpu_paged_kv_manager(new_pool)
    assert worker.gpu_paged_kv_cache_manager is old_pool
    assert worker._gpu_kv_pool_created is False
