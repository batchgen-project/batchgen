"""GPU pool operations run from frozen values and explicit collaborators."""

import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from batchgen.config.max_model_len import MaxModelLenCapacityError
from batchgen.sequence import SequenceStatus
from batchgen.worker.admission.context import (
    DecodePhaseContext,
    RankFacts,
    SchedulingState,
)
from batchgen.worker.kv_manager import (
    KVCacheManager,
    KVStats,
    PageTableCapacityRequest,
)
from batchgen.worker.kv_migration_types import MigrationOp
from tests.kv_capacity_source import (
    D6_3_METHODS,
    D6_4_METHODS,
    METHODS,
    engine_method_names,
    load_kv_capacity,
    method_node,
)
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
    assert not any(isinstance(n, ast.arg) and n.arg == "worker"
                   for n in ast.walk(node))
    assert not any(isinstance(n, ast.Name) and n.id == "worker"
                   for stmt in node.body for n in ast.walk(stmt))
    # ``self`` is the stateless engine, never a rebound worker: it is a store
    # target nowhere, and it is loaded only to call a sibling engine method
    # (the stats reads call the manager builder, as the free-page reads do).
    engine_calls = {
        id(call.func.value) for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr in engine_method_names()
    }
    for stmt in node.body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name) and n.id == "self":
                assert isinstance(n.ctx, ast.Load), ast.unparse(stmt)
                assert id(n) in engine_calls, ast.unparse(stmt)
    # The seam is a plain function: it never yields, and it never hands a
    # lazy generator across the boundary. A generator expression consumed on
    # the spot by a reduction builtin is arithmetic the verbatim move carried
    # with it (`sum(m.pages for m in migrations)`), not seam plumbing, so it
    # is allowed exactly there and nowhere else. A generator exhausted by the
    # `for` statement that owns it (`for a, b in (f(e) for e in entries)`) is
    # the same case: it is created, consumed and dropped inside one statement
    # of this body and is never a value the boundary passes.
    assert not any(isinstance(n, (ast.Yield, ast.YieldFrom))
                   for n in ast.walk(node))
    reduced = {
        id(arg) for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in {"sum", "any", "all", "min", "max", "sorted",
                             "tuple", "list", "set", "dict", "frozenset"}
        for arg in call.args
    } | {
        id(loop.iter) for loop in ast.walk(node)
        if isinstance(loop, (ast.For, ast.AsyncFor))
    }
    assert not [ast.unparse(n) for n in ast.walk(node)
                if isinstance(n, ast.GeneratorExp) and id(n) not in reduced]
    for arg in node.args.kwonlyargs:
        assert "Callable" in ast.unparse(arg.annotation)


def test_the_guard_covers_every_method_this_stage_converted():
    """D6.3's and D6.4's names are in the guarded set, and still define this engine."""
    assert D6_3_METHODS <= METHODS
    assert D6_3_METHODS <= engine_method_names()
    assert D6_4_METHODS <= METHODS
    assert D6_4_METHODS <= engine_method_names()
    # The pending-append wait shares this file section but arrived
    # context-shaped before Phase D, so D6.4 left it alone: it is out of the
    # guarded set, and a later stage claiming it must add it deliberately.
    assert "_wait_pending_kv_append_tasks" not in METHODS
    assert "_wait_pending_kv_append_tasks" in engine_method_names()


def test_frozen_contexts_and_exact_outcome_fields():
    ns, _, _ = runtime()
    expected = {
        "GPUKVSizeOutcome": {"gpu_kv_cache_size_gb"},
        "GPUKVBindOutcome": {"gpu_paged_kv_cache_manager", "_gpu_kv_pool_created"},
        "GPUKVInitOutcome": {
            "gpu_kv_cache_size_gb", "_gpu_kv_sizing_provisional", "_gpu_kv_cached_pool_gb",
        },
        "GPUKVFinalizeOutcome": {"_gpu_kv_sizing_provisional", "_gpu_kv_cached_pool_gb"},
        "DeploymentVerifyOutcome": {
            "model_context_length", "engine_config_max_model_len",
            "_deployment_length_verified",
        },
        "WatermarkTriggerOutcome": {
            "_host_kv_page_stats", "_watermark_check_counter",
        },
        "MigrationPlanOutcome": {"_logged_migration_skip_tp"},
        "KVMigrationReceiveOutcome": {
            "_migrated_sequences", "_pending_migrated_query_book",
        },
        "DeferredKVFlushContext": {
            "deferred_entries", "deferred_entries_aux", "deferred_worker_view",
            "deferred_batch", "deferred_worker_view_aux",
        },
        "DeferredKVFlushOutcome": {
            "_deferred_kv_entries", "_deferred_kv_entries_aux",
            "_deferred_kv_batch", "_deferred_kv_worker_view",
            "_deferred_kv_worker_view_aux",
        },
        "AuxLayerSlotsContext": {
            "aux_coordinator", "_aux_layers_with_slots_cache",
        },
        "AuxLayerSlotsOutcome": {"_aux_layers_with_slots_cache"},
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


def load_delegators(monkeypatch, ns, extra=()):
    """Compile the actual worker class with only the delegators under test.

    ``extra`` names worker-owned helpers a delegator calls (its context
    builders); they are the worker's own methods, not delegated bodies.
    """
    module = ModuleType("batchgen.worker.kv_capacity_engine")
    module.__dict__.update(ns)
    monkeypatch.setitem(sys.modules, "batchgen.worker.kv_capacity_engine", module)
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    wanted = METHODS | set(extra)
    cls.body = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in wanted]
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


# ---- D6.2: KV stats, host views, the decode cap and the deployment length ---


class _DualHostKV:
    """Two host-KV regions behind one coordinator, as a DSA deployment has."""

    def __init__(self, primary, auxiliary):
        self.primary, self._auxiliary = primary, auxiliary
        self.purposes = []

    def require_auxiliary(self, purpose):
        self.purposes.append(purpose)
        return self._auxiliary


def view(*, free=4, used=6, total=10, audited="audit"):
    return SimpleNamespace(
        get_stats=lambda: SimpleNamespace(
            num_free_pages=free, num_used_pages=used, num_total_pages=total,
        ),
        get_audited_stats=lambda: audited,
    )


def pool(*, prefilled=0, onhold=0, in_decode=0):
    counts = {
        SequenceStatus.PREFILLED: prefilled,
        SequenceStatus.ON_HOLD: onhold,
        SequenceStatus.IN_DECODE: in_decode,
    }
    return SimpleNamespace(
        get_sequences_for_rank_with_status=(
            lambda rank, status: [(rank, status, n) for n in range(counts[status])]
        ),
    )


def phase_context(*, host_view=None, sequences=None, rank=0, world_size=1,
                  local_rank=0, local_world_size=1, node_id=0,
                  native=False, compare=False):
    """The real decode phase context the free-page reads already take."""
    return DecodePhaseContext(
        facts=RankFacts(
            rank=rank, world_size=world_size, local_rank=local_rank,
            local_world_size=local_world_size,
            rank_ctx=SimpleNamespace(node_id=node_id, num_nodes=1),
            attn_tp_size=1, device="cpu", page_size=64, decision_interval=1,
            enable_prefix_cache=False, enable_host_kv_eviction=False,
            host_kv_chunk_size=1, kv_stats_native=native, kv_stats_compare=compare,
        ),
        sched=SchedulingState(pool=sequences, seq_table=None, trajectories=None),
        gpu_kv=None, host_view=host_view, core_engine=None, gpu_resident=None,
        prefix_coordinator=None, prefix_runtime=None, chunk_sizer=None,
    )


def d62(**extra):
    """The engine with the collaborators the D6.2 bodies name at module scope."""
    events = []
    namespace = {
        "logging": SimpleNamespace(
            info=lambda line: events.append(("info", line)),
            debug=lambda line: events.append(("debug", line)),
        ),
        "DualHostKVCoordinator": _DualHostKV,
        "format_host_kv_health": lambda name, audited: f"{name}={audited}",
    }
    namespace.update(extra)
    ns = load_kv_capacity(namespace)
    return ns, ns["KVCapacityEngine"](), events


def test_utilization_request_is_the_phase_context_identity():
    _, engine, _ = d62()
    sequences = pool()
    ctx = phase_context(sequences=sequences, rank=3, world_size=8,
                        local_rank=1, local_world_size=4)
    request = engine._make_kv_utilization_request(ctx)
    assert (request.rank, request.world_size) == (3, 8)
    assert (request.local_rank, request.num_gpus_per_node) == (1, 4)
    assert request.global_batch is sequences


def test_legacy_host_utilization_counts_every_rank_on_the_node():
    _, engine, events = d62()
    ctx = phase_context(
        host_view=view(free=4, used=6, total=10),
        sequences=pool(prefilled=2, onhold=1, in_decode=3),
        world_size=4, local_world_size=2, node_id=1, rank=2,
    )
    assert engine._legacy_get_host_kv_utilization(ctx) == {
        "rank": 2, "node_id": 1, "num_free_pages": 4, "num_total_pages": 10,
        "num_used_pages": 6, "free_percent": 40, "num_in_decode": 6,
        "num_onhold": 2, "num_prefilled": 4, "num_valid_sequences": 12,
    }
    # The node leader logs the C++ stats line; a follower stays silent.
    assert [kind for kind, _ in events] == ["debug"]
    events.clear()
    follower = replace(ctx, facts=replace(ctx.facts, local_rank=1))
    assert engine._legacy_get_host_kv_utilization(follower)["rank"] == 2
    assert events == []


def test_legacy_host_utilization_of_an_empty_pool_reports_full_free():
    _, engine, _ = d62()
    ctx = phase_context(host_view=view(free=0, used=0, total=0), sequences=pool())
    result = engine._legacy_get_host_kv_utilization(ctx)
    assert (result["free_percent"], result["num_valid_sequences"]) == (100, 0)


@pytest.mark.parametrize("native,compare", [(False, False), (True, False),
                                            (False, True), (True, True)])
def test_host_utilization_native_and_legacy_paths_agree(native, compare):
    _, engine, _ = d62()
    ctx = phase_context(
        host_view=view(free=4, used=6, total=10),
        sequences=pool(prefilled=2, onhold=1, in_decode=3),
        native=native, compare=compare,
    )
    built = []

    def manager(context):
        built.append(context)
        stats = context.host_view.get_stats()
        return KVCacheManager(backend=SimpleNamespace(
            get_host_stats=lambda: KVStats(
                num_free_pages=stats.num_free_pages,
                num_used_pages=stats.num_used_pages,
                num_total_pages=stats.num_total_pages,
            ),
            get_gpu_stats=lambda: None,
        ))

    engine._make_kv_cache_manager = manager
    assert engine._get_host_kv_utilization(ctx) == (
        engine._legacy_get_host_kv_utilization(ctx)
    )
    # The manager is built only on a path that reads the native backend, and
    # from the SAME context the delegator passed in.
    assert built == [ctx] * (1 if native or compare else 0)


def test_host_utilization_compare_mismatch_refuses():
    _, engine, _ = d62()
    ctx = phase_context(host_view=view(), sequences=pool(), compare=True)
    engine._make_kv_cache_manager = lambda context: KVCacheManager(
        backend=SimpleNamespace(
            get_host_stats=lambda: KVStats(
                num_free_pages=1, num_used_pages=1, num_total_pages=2,
            ),
            get_gpu_stats=lambda: None,
        ),
    )
    with pytest.raises(AssertionError, match="kv_stats compare mismatch"):
        engine._get_host_kv_utilization(ctx)


def test_total_pages_reads_the_declared_view():
    ns, engine, _ = d62()
    ctx = ns["HostKVStatsContext"](host_view=view(total=2048))
    assert engine._get_host_kv_total_pages(ctx) == 2048
    assert {field.name for field in fields(ctx)} == {"host_view"}
    with pytest.raises(AttributeError):
        engine._get_host_kv_total_pages(ns["HostKVStatsContext"](host_view=None))


@pytest.mark.parametrize("dual", [False, True])
def test_health_line_per_region_carries_the_node_id(dual):
    ns, engine, events = d62()
    primary, aux = view(audited="p"), view(audited="a")
    region = _DualHostKV(primary, aux) if dual else primary
    ctx = ns["HostKVHealthContext"](host_view=region, node_id=3)
    assert engine._log_host_kv_health(ctx) is None
    assert events == ([("info", "[HOST_KV] node=3 primary=p"),
                       ("info", "[HOST_KV] node=3 aux=a")] if dual else
                      [("info", "[HOST_KV] node=3 host=p")])
    if dual:
        assert region.purposes == ["host-KV health"]


def test_health_line_without_a_region_returns_before_reading_anything():
    ns, engine, events = d62()
    ctx = ns["HostKVHealthContext"](host_view=None, node_id=None)
    assert engine._log_host_kv_health(ctx) is None
    assert events == []


def decode_cap_context(ns, *, planned=0, max_bucket=None, args=True):
    return ns["DecodeRankBszContext"](
        engine_config=SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            MoE_decoding_micro_batch_size=planned,
        )),
        args=SimpleNamespace(cuda_graph_max_bucket_size=max_bucket) if args else None,
    )


def decode_cap_helpers(events, *, attn_tp_size=1, whole_model=False):
    return dict(
        _decode_attn_tp_size=lambda: events.append("attn_tp") or attn_tp_size,
        _whole_model_decode_graphs_requested=(
            lambda: events.append("whole_model") or whole_model
        ),
    )


def test_decode_cap_refuses_diverging_env_names_before_any_collaborator(monkeypatch):
    ns, engine, events = d62()
    monkeypatch.setenv("BATCHGEN_MAX_DECODE_RANK_BSZ", "128")
    monkeypatch.setenv("BATCHGEN_MAX_RANK_BSZ", "160")
    with pytest.raises(RuntimeError, match="must match when both are set"):
        engine._max_decode_rank_bsz(
            decode_cap_context(ns), **decode_cap_helpers(events),
        )
    assert events == []


@pytest.mark.parametrize("attn_tp_size,planned,expected", [
    (8, 320, 320),   # TP group plans wider than the guard
    (8, 96, 128),    # planned below the guard
    (1, 320, 128),   # pure DP: the planned value does not lift the cap
])
def test_decode_cap_takes_a_wider_tp_plan(monkeypatch, attn_tp_size, planned, expected):
    ns, engine, events = d62()
    monkeypatch.delenv("BATCHGEN_MAX_DECODE_RANK_BSZ", raising=False)
    monkeypatch.delenv("BATCHGEN_MAX_RANK_BSZ", raising=False)
    ctx = decode_cap_context(ns, planned=planned, max_bucket=0)
    helpers = decode_cap_helpers(events, attn_tp_size=attn_tp_size)
    assert engine._max_decode_rank_bsz(ctx, **helpers) == expected
    # The graph predicate is consulted only inside the flag-bucket branch,
    # which a zero bucket never enters.
    assert events == ["attn_tp"]


@pytest.mark.parametrize("max_bucket,whole_model,expected", [
    (256, True, 256),    # the bucket flag is the width knob on graph deployments
    (256, False, 128),   # eager deployment: the flag does not widen decode
    (64, True, 128),     # a bucket below the guard never lowers it
])
def test_decode_cap_follows_the_graph_bucket_flag(monkeypatch, max_bucket,
                                                  whole_model, expected):
    ns, engine, events = d62()
    monkeypatch.delenv("BATCHGEN_MAX_DECODE_RANK_BSZ", raising=False)
    monkeypatch.delenv("BATCHGEN_MAX_RANK_BSZ", raising=False)
    ctx = decode_cap_context(ns, max_bucket=max_bucket)
    helpers = decode_cap_helpers(events, whole_model=whole_model)
    assert engine._max_decode_rank_bsz(ctx, **helpers) == expected
    assert events == (["attn_tp", "whole_model"] if max_bucket > 128
                      else ["attn_tp"])


def test_decode_cap_with_an_explicit_env_skips_the_bucket_branch(monkeypatch):
    ns, engine, events = d62()
    monkeypatch.setenv("BATCHGEN_MAX_DECODE_RANK_BSZ", "128")
    monkeypatch.delenv("BATCHGEN_MAX_RANK_BSZ", raising=False)
    ctx = decode_cap_context(ns, max_bucket=256)
    assert engine._max_decode_rank_bsz(ctx, **decode_cap_helpers(
        events, whole_model=True)) == 128
    assert events == ["attn_tp"]


def test_decode_cap_without_launch_args_keeps_the_guard(monkeypatch):
    ns, engine, events = d62()
    monkeypatch.delenv("BATCHGEN_MAX_DECODE_RANK_BSZ", raising=False)
    monkeypatch.delenv("BATCHGEN_MAX_RANK_BSZ", raising=False)
    ctx = decode_cap_context(ns, args=False)
    assert engine._max_decode_rank_bsz(ctx, **decode_cap_helpers(events)) == 128
    assert events == ["attn_tp"]


def deployment_profiles(monkeypatch, *, dsa):
    profile = SimpleNamespace(page_size=64, num_layers=2, bytes_per_page=lambda: 512)
    module = ModuleType("batchgen.kv_cache.host_kv_mananger_config")
    module._resolve_profile = lambda name: profile
    module._resolve_indexer_profile = lambda name: profile
    module.is_dsa_model = lambda name: dsa
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return profile


def verify_context(ns, **changes):
    values = dict(
        rank=0, local_rank=0, world_size=1, local_world_size=1,
        huggingface_ckpt_name="test-model", model_context_length=1024,
        args=SimpleNamespace(max_model_len=-1), traj=SimpleNamespace(
            trajectory_book=SimpleNamespace(page_tokens=64, page_count=100),
        ),
        _deployment_length_verified=False,
    )
    return ns["DeploymentVerifyContext"](**(values | changes))


def test_deployment_length_outcome_is_exactly_the_three_old_effects(monkeypatch):
    deployment_profiles(monkeypatch, dsa=True)
    gathered = []
    ns, engine, events = d62(dist=SimpleNamespace(
        all_gather_object=lambda out, local: (gathered.extend(local),
                                              out.__setitem__(0, local)),
    ))
    ctx = verify_context(ns)
    outcome = engine._verify_deployment_max_model_len(
        ctx, 100 * 2048 / (1024 ** 3),
        _decode_attn_tp_size=lambda: events.append("attn_tp") or 2,
        # Two host regions of four pages each, mirrored on both copies of the
        # attention group: 2 pages x 64 tokens is the limiting capacity, so
        # auto-fit lowers the promised 1024 to 128.
        _host_views_by_group=lambda: events.append("views") or {
            region: view(total=4) for region in (0, 1)
        },
    )
    assert {field.name for field in fields(outcome)} == {
        "model_context_length", "engine_config_max_model_len",
        "_deployment_length_verified",
    }
    assert outcome.model_context_length == 128
    assert outcome.engine_config_max_model_len == 128
    assert outcome._deployment_length_verified is True
    # The launch config is NOT in the context: the engine writes nothing into
    # an object the worker owns, it returns the length the worker publishes.
    assert "engine_config" not in {field.name for field in fields(ctx)}
    assert events == ["views", "attn_tp", "attn_tp"]
    assert [(row.resource, row.region) for row in gathered] == [
        ("GPU KV", "0"), ("host KV", "0"), ("GPU KV", "1"), ("host KV", "1"),
        ("input IDs", "trajectory"),
    ]
    assert all(row.copies == 2 for row in gathered if row.resource == "host KV")


def test_deployment_length_follower_reports_only_its_device_regions(monkeypatch):
    deployment_profiles(monkeypatch, dsa=False)
    gathered = []
    ns, engine, _ = d62(dist=SimpleNamespace(
        all_gather_object=lambda out, local: (gathered.extend(local),
                                              out.__setitem__(0, local)),
    ))
    ctx = verify_context(ns, local_rank=1, args=SimpleNamespace(max_model_len=1024))
    outcome = engine._verify_deployment_max_model_len(
        ctx, 100 * 1024 / (1024 ** 3),
        _decode_attn_tp_size=lambda: pytest.fail("no host row on a follower"),
        _host_views_by_group=lambda: {},
    )
    assert [row.resource for row in gathered] == ["GPU KV"]
    assert outcome.model_context_length == 1024


@pytest.mark.parametrize("max_model_len,verified", [
    (1024, False),   # a launch that did not opt into auto-fit
    (-1, True),      # auto-fit cannot silently lower a published length again
])
def test_deployment_length_refusal_propagates_and_publishes_nothing(
    monkeypatch, max_model_len, verified,
):
    deployment_profiles(monkeypatch, dsa=False)
    ns, engine, _ = d62(dist=SimpleNamespace(
        all_gather_object=lambda out, local: out.__setitem__(0, local),
    ))
    # One page of host KV cannot hold the requested 1024 tokens.
    ctx = verify_context(
        ns, _deployment_length_verified=verified,
        args=SimpleNamespace(max_model_len=max_model_len),
    )
    with pytest.raises(MaxModelLenCapacityError):
        engine._verify_deployment_max_model_len(
            ctx, 1024 / (1024 ** 3),
            _decode_attn_tp_size=lambda: 1,
            _host_views_by_group=lambda: {0: view(total=1)},
        )
    assert ctx._deployment_length_verified is verified


def test_delegator_publishes_the_deployment_length_in_the_old_order(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._kv_capacity_engine = engine
    worker.rank = worker.local_rank = 0
    worker.world_size = worker.local_world_size = 1
    worker.huggingface_ckpt_name = "test-model"
    worker.model_context_length = 1024
    worker.args = SimpleNamespace(max_model_len=-1)
    worker._traj = SimpleNamespace(trajectory_book=None)
    worker.engine_config = SimpleNamespace(Basic_Config=SimpleNamespace())
    worker._decode_attn_tp_size = lambda: 1
    worker._host_views_by_group = lambda: {}
    seen = []

    def verify(ctx, pool_gb, **helpers):
        seen.append((ctx, pool_gb, sorted(helpers)))
        assert worker.model_context_length == 1024  # the bind waits for the outcome
        return ns["DeploymentVerifyOutcome"](
            model_context_length=128, engine_config_max_model_len=128,
            _deployment_length_verified=True,
        )

    engine._verify_deployment_max_model_len = verify
    assert worker._verify_deployment_max_model_len(4.0) is None
    ctx, pool_gb, helpers = seen[0]
    assert pool_gb == 4.0
    assert helpers == ["_decode_attn_tp_size", "_host_views_by_group"]
    assert ctx.huggingface_ckpt_name == "test-model"
    assert ctx._deployment_length_verified is False
    assert worker.model_context_length == 128
    assert worker.engine_config.Basic_Config.max_model_len == 128
    assert worker._deployment_length_verified is True


# ---- D6.3: the host-KV watermark, host capacity and KV migration -----------


def d63(**extra):
    """The engine with the collaborators the D6.3 bodies name at module scope."""
    events = []
    namespace = {
        "logging": SimpleNamespace(
            info=lambda line: events.append(("info", line)),
            debug=lambda line: events.append(("debug", line)),
            error=lambda line: events.append(("error", line)),
        ),
        # The migration transfer tests run the single-region branch; the dual
        # coordinator's own split is covered by the D6.2 health-line tests.
        "DualHostKVCoordinator": type("NoDual", (), {}),
    }
    namespace.update(extra)
    ns = load_kv_capacity(namespace)
    return ns, ns["KVCapacityEngine"](), events


def recording_pool(*, queued=False, evicted=False, by_status=()):
    """A sequence pool that records which question it was asked."""
    asked = []
    return SimpleNamespace(
        asked=asked,
        has_queueing=lambda: asked.append("queueing") or queued,
        has_evicted=lambda: asked.append("evicted") or evicted,
        get_sequences_by_status=(
            lambda status: asked.append(("status", status)) or list(by_status)
        ),
    )


def watermark_context(ns, **changes):
    values = dict(
        rank=0, local_rank=0, world_size=1, enable_decode_preemption=True,
        enable_host_kv_eviction=False, host_kv_watermark=70,
        pool=recording_pool(), _watermark_check_counter=None,
    )
    return ns["HostKVWatermarkContext"](**(values | changes))


def test_watermark_request_is_the_declared_snapshot_and_stays_lazy():
    ns, engine, _ = d63()
    pool = recording_pool(queued=True, evicted=True)
    ctx = watermark_context(ns, pool=pool, host_kv_watermark=65)
    request = engine._make_watermark_trigger_request(ctx, [{"node_id": 0}])
    assert request.node_stats == ({"node_id": 0},)
    assert request.host_kv_watermark == 65
    assert request.has_queued is True
    # The launch flag is the left operand, so a deployment without eviction
    # never asks the pool whether anything was evicted.
    assert request.has_evicted is False
    assert pool.asked == ["queueing"]
    enabled = watermark_context(
        ns, pool=recording_pool(queued=True, evicted=True),
        enable_host_kv_eviction=True,
    )
    assert engine._make_watermark_trigger_request(enabled, []).has_evicted is True


def credit_context(*, prefix=True, coordinator=None, queued=False,
                   evicted=False, eviction=False):
    pool = recording_pool(queued=queued, evicted=evicted)
    return DecodePhaseContext(
        facts=RankFacts(
            rank=0, world_size=1, local_rank=0, local_world_size=1,
            rank_ctx=SimpleNamespace(node_id=0), attn_tp_size=1, device="cpu",
            page_size=64, decision_interval=1, enable_prefix_cache=prefix,
            enable_host_kv_eviction=eviction, host_kv_chunk_size=1,
        ),
        sched=SchedulingState(pool=pool, seq_table=None, trajectories=None),
        gpu_kv=None, host_view=None, core_engine=None, gpu_resident=None,
        prefix_coordinator=coordinator, prefix_runtime=None, chunk_sizer=None,
    ), pool


def coordinator_with(pages):
    asked = []
    return SimpleNamespace(
        asked=asked,
        count_releasable_pages=lambda group: asked.append(group) or pages,
    )


RAW_STATS = {
    "num_total_pages": 10, "num_used_pages": 8, "num_free_pages": 2,
    "free_percent": 20,
}


@pytest.mark.parametrize("prefix,pages,queued,eviction,evicted,credited", [
    (False, 4, True, False, False, False),    # prefix cache off
    (True, None, True, False, False, False),  # no coordinator on this rank
    (True, 4, False, False, False, False),    # nothing waiting
    (True, 4, True, False, False, True),      # queued work waiting
    (True, 4, False, True, True, True),       # evicted work waiting
])
def test_reclaimable_credit_only_when_work_is_waiting(
    prefix, pages, queued, eviction, evicted, credited,
):
    _, engine, _ = d63()
    coordinator = None if pages is None else coordinator_with(pages)
    ctx, pool = credit_context(
        prefix=prefix, coordinator=coordinator, queued=queued,
        eviction=eviction, evicted=evicted,
    )
    result = engine._credit_reclaimable_prefix_pages(ctx, RAW_STATS)
    if not credited:
        assert result is RAW_STATS
        assert coordinator is None or coordinator.asked == []
    else:
        assert result["num_free_pages"] == 6
        assert result["num_used_pages"] == 4
        assert result["num_reclaimable_prefix_pages"] == 4
        assert result["free_percent"] == 60
        assert coordinator.asked == [0]
    # The eviction flag is the right operand of the waiting test, so the pool
    # is asked about eviction only on a deployment that evicts.
    if prefix and coordinator is not None:
        assert ("evicted" in pool.asked) is eviction


def due_context(rows, sequences, *, world_size=1):
    return DecodePhaseContext(
        facts=RankFacts(
            rank=0, world_size=world_size, local_rank=0, local_world_size=1,
            rank_ctx=SimpleNamespace(node_id=0), attn_tp_size=1, device="cpu",
            page_size=64, decision_interval=1, enable_prefix_cache=False,
            enable_host_kv_eviction=False, host_kv_chunk_size=1,
        ),
        sched=SchedulingState(
            pool=SimpleNamespace(get_sequence=sequences.get),
            seq_table=SimpleNamespace(local_to_uuid=rows),
            trajectories=None,
        ),
        gpu_kv=None, host_view=None, core_engine=None, gpu_resident=None,
        prefix_coordinator=None, prefix_runtime=None, chunk_sizer=None,
    )


def row(*, capacity, context):
    return SimpleNamespace(
        host_token_capacity=capacity, current_context_length=context,
    )


def test_host_capacity_due_local_reads_the_rows_of_the_declared_table():
    _, engine, _ = d63()
    seen = []
    sequences = {"a": row(capacity=128, context=64), "b": row(capacity=64, context=64)}
    ctx = due_context({0: "a", 2: "b"}, sequences)
    completed = lambda seq: seen.append(seq) or False
    # Row 1 is unbound and row 3 names a sequence the pool dropped: neither
    # reaches the completion predicate or the capacity test.
    assert engine._host_capacity_due_local(
        ctx, [1, 0], _is_sequence_completed=completed,
    ) is False
    assert seen == [sequences["a"]]
    assert engine._host_capacity_due_local(
        ctx, [0, 2], _is_sequence_completed=completed,
    ) is True
    assert engine._host_capacity_due_local(
        ctx, [2], _is_sequence_completed=lambda seq: True,
    ) is False


def test_host_capacity_due_all_ranks_reduces_the_local_verdict_with_max():
    collectives = []
    ns, engine, _ = d63(dist=SimpleNamespace(
        all_reduce=lambda tensor, op: collectives.append(
            ("all_reduce", op, int(tensor[0].item()))
        ) or tensor.fill_(1),
        ReduceOp=SimpleNamespace(MAX="MAX"),
    ), torch=torch)
    sequences = {"a": row(capacity=128, context=64)}
    single = due_context({0: "a"}, sequences, world_size=1)
    assert engine._host_capacity_due_all_ranks(
        single, [0], _is_sequence_completed=lambda seq: False,
    ) is False
    # A single-rank deployment answers locally: no collective at all.
    assert collectives == []
    many = due_context({0: "a"}, sequences, world_size=4)
    assert engine._host_capacity_due_all_ranks(
        many, [0], _is_sequence_completed=lambda seq: False,
    ) is True
    assert collectives == [("all_reduce", "MAX", 0)]


def gathering_dist(payloads):
    """``all_gather_object`` that fills the out list with ``payloads``."""
    calls = []

    def all_gather_object(out, local, group=None):
        calls.append(local)
        for index, value in enumerate(payloads):
            out[index] = value

    return SimpleNamespace(all_gather_object=all_gather_object), calls


NODE_STATS = {
    "node_id": 0, "free_percent": 90, "num_used_pages": 1,
    "num_total_pages": 10, "num_free_pages": 9,
}


def test_watermark_trigger_without_preemption_reaches_no_collective():
    ns, engine, events = d63(dist=SimpleNamespace(
        all_gather_object=lambda out, local: pytest.fail("unexpected gather"),
    ))
    ctx = watermark_context(ns, enable_decode_preemption=False)
    assert engine._check_host_kv_watermark_trigger(
        ctx,
        _get_host_kv_utilization=lambda: pytest.fail("unexpected stats read"),
        _credit_reclaimable_prefix_pages=lambda stats: stats,
    ) == (False, ns["WatermarkTriggerOutcome"]())
    assert events == []


def test_watermark_trigger_on_rank_zero_returns_only_the_page_stats():
    distributed, gathered = gathering_dist([NODE_STATS])
    ns, engine, events = d63(dist=distributed)
    pool = recording_pool(queued=True, by_status=("q1", "q2"))
    ctx = watermark_context(ns, pool=pool)
    order = []
    should_trigger, outcome = engine._check_host_kv_watermark_trigger(
        ctx,
        _get_host_kv_utilization=lambda: order.append("stats") or dict(NODE_STATS),
        _credit_reclaimable_prefix_pages=(
            lambda stats: order.append("credit") or stats
        ),
    )
    # The node leader reads its stats and credits them BEFORE the gather.
    assert order == ["stats", "credit"]
    assert gathered == [NODE_STATS]
    assert should_trigger is True
    assert outcome._host_kv_page_stats == {
        'used': 1, 'total': 10, 'free_percent': 90, 'num_nodes': 1,
    }
    # Rank 0 never touches the follower counter.
    assert outcome._watermark_check_counter is ns["GPU_KV_UNCHANGED"]
    assert [kind for kind, _ in events] == ["info", "info"]
    assert ("status", SequenceStatus.QUEUEING) in pool.asked


def test_watermark_trigger_on_a_follower_only_advances_its_counter():
    distributed, _ = gathering_dist([NODE_STATS])
    ns, engine, events = d63(dist=distributed)
    ctx = watermark_context(ns, rank=1, local_rank=1, pool=recording_pool(queued=True))
    _, outcome = engine._check_host_kv_watermark_trigger(
        ctx,
        _get_host_kv_utilization=lambda: pytest.fail("a follower reports nothing"),
        _credit_reclaimable_prefix_pages=lambda stats: stats,
    )
    # An absent counter starts at one, exactly as the pre-move hasattr did.
    assert outcome._watermark_check_counter == 1
    assert outcome._host_kv_page_stats is ns["GPU_KV_UNCHANGED"]
    assert events == []
    ninth = replace(ctx, _watermark_check_counter=9)
    _, outcome = engine._check_host_kv_watermark_trigger(
        ninth,
        _get_host_kv_utilization=lambda: pytest.fail("a follower reports nothing"),
        _credit_reclaimable_prefix_pages=lambda stats: stats,
    )
    assert outcome._watermark_check_counter == 10
    assert [kind for kind, _ in events] == ["debug"]


def test_watermark_trigger_with_no_node_reporting_writes_nothing():
    distributed, _ = gathering_dist([None])
    ns, engine, _ = d63(dist=distributed)
    ctx = watermark_context(ns, local_rank=1)
    assert engine._check_host_kv_watermark_trigger(
        ctx,
        _get_host_kv_utilization=lambda: pytest.fail("a follower reports nothing"),
        _credit_reclaimable_prefix_pages=lambda stats: stats,
    ) == (False, ns["WatermarkTriggerOutcome"]())


def migration_context(*, rank=0, local_rank=0, world_size=4,
                      local_world_size=2, sequences=None, by_status=None,
                      seq_table=None, trajectories=None, gpu_resident=None,
                      host_view=None, host_view_aux=None, core_engine=None):
    sequences = sequences or {}
    by_status = by_status or {}
    pool = SimpleNamespace(
        get_sequence=sequences.get,
        get_sequences_for_rank_with_status=(
            lambda rank, status: list(by_status.get((rank, status), ()))
        ),
        assign_rank=lambda uuid, new_rank: sequences[uuid].__setattr__(
            "assigned_rank", new_rank
        ),
    )
    return DecodePhaseContext(
        facts=RankFacts(
            rank=rank, world_size=world_size, local_rank=local_rank,
            local_world_size=local_world_size,
            rank_ctx=SimpleNamespace(node_id=rank // max(1, local_world_size)),
            attn_tp_size=1, device="cpu", page_size=64, decision_interval=1,
            enable_prefix_cache=False, enable_host_kv_eviction=False,
            host_kv_chunk_size=1,
        ),
        sched=SchedulingState(
            pool=pool, seq_table=seq_table, trajectories=trajectories,
        ),
        gpu_kv=None, host_view=host_view, host_view_aux=host_view_aux,
        core_engine=core_engine, gpu_resident=gpu_resident,
        prefix_coordinator=None, prefix_runtime=None, chunk_sizer=None,
    )


def candidate(uuid, *, rank, idx, budget=512, pages=4):
    return SimpleNamespace(
        uuid=uuid, assigned_rank=rank, global_idx=idx, kv_token_budget=budget,
        host_pages_allocated=pages, prompt_length=2, decoded_length=1,
    )


def test_migration_plan_request_enumerates_both_statuses_of_every_rank():
    _, engine, _ = d63()
    sequences = {
        "a": candidate("a", rank=0, idx=0), "b": candidate("b", rank=1, idx=1),
        "gone": candidate("gone", rank=1, idx=9),
    }
    del sequences["gone"]
    ctx = migration_context(
        world_size=2, local_world_size=2, sequences=sequences,
        by_status={
            (0, SequenceStatus.PREFILLED): ["a"],
            (1, SequenceStatus.ON_HOLD): ["b", "gone"],
        },
    )
    request = engine._make_migration_plan_request(ctx, {0: NODE_STATS})
    assert request.node_stats == {0: NODE_STATS}
    assert [c.uuid for c in request.candidates] == ["a", "b"]
    assert (request.world_size, request.num_gpus_per_node) == (2, 2)


def test_plan_kv_migration_skips_a_tp_group_and_latches_the_log_once():
    ns, engine, events = d63(dist=SimpleNamespace(
        all_gather_object=lambda out, local: pytest.fail("unexpected gather"),
    ))
    ctx = migration_context()
    helpers = dict(
        _decode_attn_tp_size=lambda: 8,
        _get_host_kv_utilization=lambda: pytest.fail("unexpected stats read"),
    )
    migrations, outcome = engine._plan_kv_migration(ctx, False, **helpers)
    assert migrations == []
    assert outcome._logged_migration_skip_tp is True
    assert [kind for kind, _ in events] == ["info"]
    assert "attn_tp_size=8" in events[0][1]
    # Already latched, or not rank zero: the same safe plan, no second line.
    for latched, rank in ((True, 0), (False, 1)):
        again = migration_context(rank=rank)
        migrations, outcome = engine._plan_kv_migration(again, latched, **helpers)
        assert migrations == []
        assert outcome._logged_migration_skip_tp is ns["GPU_KV_UNCHANGED"]
    assert len(events) == 1


def test_plan_kv_migration_on_one_node_plans_nothing_and_latches_nothing():
    distributed, gathered = gathering_dist([dict(NODE_STATS), None])
    ns, engine, events = d63(dist=distributed)
    ctx = migration_context(world_size=2)
    migrations, outcome = engine._plan_kv_migration(
        ctx, False,
        _decode_attn_tp_size=lambda: 1,
        _get_host_kv_utilization=lambda: dict(NODE_STATS),
    )
    assert migrations == []
    assert outcome._logged_migration_skip_tp is ns["GPU_KV_UNCHANGED"]
    assert gathered == [NODE_STATS]
    assert events == [("info", "MIGRATION: Single node detected, skipping rebalancing")]


def test_plan_kv_migration_returns_the_planner_decision_verbatim():
    second = {**NODE_STATS, "node_id": 1}
    distributed, gathered = gathering_dist([dict(NODE_STATS), second])
    planned = [MigrationOp("a" * 12, from_rank=0, to_rank=2, pages=3, host_pages=3)]
    seen = []
    ns, engine, events = d63(
        dist=distributed,
        KVCacheManager=SimpleNamespace(
            plan_kv_migration=lambda req: seen.append(req) or planned,
        ),
    )
    ctx = migration_context(world_size=2, local_world_size=1)
    migrations, outcome = engine._plan_kv_migration(
        ctx, False,
        _decode_attn_tp_size=lambda: 1,
        _get_host_kv_utilization=lambda: dict(NODE_STATS),
    )
    assert migrations is planned
    assert outcome._logged_migration_skip_tp is ns["GPU_KV_UNCHANGED"]
    # The request is built from the SAME context the delegator passed in.
    assert seen[0].node_stats == {0: NODE_STATS, 1: second}
    assert gathered == [NODE_STATS]
    assert [line for kind, line in events if "Planned" in line]


def recording_dist(**extra):
    """A distributed stub that records barriers in call order."""
    order = []
    stub = SimpleNamespace(barrier=lambda *a, **k: order.append("barrier"))
    for name, value in extra.items():
        setattr(stub, name, value)
    return stub, order


def test_execute_migrations_keeps_the_group_barrier_round_order():
    distributed, order = recording_dist()
    ns, engine, _ = d63(dist=distributed)
    first = MigrationOp("aaaaaaaaaaaa", from_rank=0, to_rank=1, pages=1, host_pages=1)
    second = MigrationOp("bbbbbbbbbbbb", from_rank=0, to_rank=2, pages=1, host_pages=1)
    sequences = {
        "aaaaaaaaaaaa": candidate("aaaaaaaaaaaa", rank=0, idx=0),
        "bbbbbbbbbbbb": candidate("bbbbbbbbbbbb", rank=0, idx=1),
    }
    for seq in sequences.values():
        seq.status = "PREFILLED"
    ctx = migration_context(sequences=sequences)
    helpers = dict(
        _get_or_create_gloo_group=lambda: order.append("group"),
        _execute_single_kv_migration=(
            lambda *, uuid, from_rank, to_rank: order.append(("migrate", uuid[:4]))
        ),
    )
    # An empty plan is not a collective: no group, no barrier.
    assert engine._execute_kv_migrations_parallel(ctx, [], **helpers) is None
    assert order == []
    assert engine._execute_kv_migrations_parallel(
        ctx, [first, second], **helpers,
    ) is None
    # The group comes first, then one barrier, then one barrier per round, and
    # rank 0 shares a rank with both migrations so the rounds are serial.
    assert order == [
        "group", "barrier",
        ("migrate", "aaaa"), "barrier",
        ("migrate", "bbbb"), "barrier",
    ]


def test_execute_migrations_still_barriers_when_the_sequence_is_gone():
    distributed, order = recording_dist()
    ns, engine, events = d63(dist=distributed)
    op = MigrationOp("cccccccccccc", from_rank=0, to_rank=1, pages=1, host_pages=1)
    ctx = migration_context(sequences={})
    assert engine._execute_kv_migrations_parallel(
        ctx, [op],
        _get_or_create_gloo_group=lambda: order.append("group"),
        _execute_single_kv_migration=lambda **kw: pytest.fail("no sequence"),
    ) is None
    assert order == ["group", "barrier", "barrier"]
    assert [kind for kind, _ in events if kind == "error"] == ["error"]


def test_a_rank_outside_the_round_reaches_the_same_barriers():
    distributed, order = recording_dist()
    _, engine, _ = d63(dist=distributed)
    op = MigrationOp("dddddddddddd", from_rank=1, to_rank=2, pages=1, host_pages=1)
    ctx = migration_context(rank=3, local_rank=1, sequences={})
    assert engine._execute_kv_migrations_parallel(
        ctx, [op],
        _get_or_create_gloo_group=lambda: order.append("group"),
        _execute_single_kv_migration=lambda **kw: pytest.fail("not this rank"),
    ) is None
    assert order == ["group", "barrier", "barrier"]


def test_grouping_uses_the_declared_node_width():
    _, engine, _ = d63()
    ops = [
        MigrationOp("a", from_rank=0, to_rank=2, pages=1, host_pages=1),
        MigrationOp("b", from_rank=1, to_rank=3, pages=1, host_pages=1),
        MigrationOp("c", from_rank=0, to_rank=3, pages=1, host_pages=1),
    ]
    # The declared node width decides which sources share a staging buffer:
    # at two GPUs per node ranks 0 and 1 are one source node and serialize,
    # at one GPU per node they are two and go in the same round.
    assert [[op.uuid for op in group] for group in
            engine._group_migrations_for_parallel_execution(
                migration_context(local_world_size=2), ops)] == [["a"], ["b"], ["c"]]
    rounds = engine._group_migrations_for_parallel_execution(
        migration_context(local_world_size=1), ops,
    )
    assert [[op.uuid for op in group] for group in rounds] == [["a", "b"], ["c"]]
    assert sum(len(group) for group in rounds) == len(ops)


def migration_views(*, pages=1, tokens=4):
    """One host-KV region holding a page of placeholder KV."""
    written = []
    released = []
    registered = []
    return SimpleNamespace(
        written=written, released=released, registered=registered,
        read_sequence_kv_to_cpu=lambda idx: (
            torch.zeros((pages, tokens), dtype=torch.float32), torch.empty(0)
        ),
        write_sequence_kv_from_cpu=(
            lambda idx, k, v: written.append((idx, tuple(k.shape), v))
        ),
        register_sequences=registered.append,
        allocate_pages_for_sequences=registered.append,
        release_sequence_pages=released.append,
    )


def migration_book(uuid, *, slot=0, present=True):
    restored = []
    return SimpleNamespace(
        restored=restored,
        has_sequence=lambda name: present and name == uuid,
        slot_for=lambda name: slot,
        turns_payload=lambda s: torch.zeros((0, 6), dtype=torch.int64),
        restore=lambda s, tokens, **kw: restored.append((s, tokens.tolist(), kw)),
    )


def test_single_migration_on_the_source_sends_then_releases_and_writes_nothing():
    order = []
    distributed = SimpleNamespace(
        send=lambda tensor, dst, group=None: order.append(("send", dst, tensor.numel())),
        recv=lambda tensor, src, group=None: pytest.fail("the source never receives"),
    )
    ns, engine, _ = d63(dist=distributed, torch=torch)
    uuid = "eeeeeeeeeeee"
    seq = candidate(uuid, rank=0, idx=5, pages=1)
    view = migration_views()
    book = migration_book(uuid)
    ctx = migration_context(
        sequences={uuid: seq}, host_view=view,
        trajectories=SimpleNamespace(trajectory_book=book),
    )
    engine._release_prefix_cache_attachments = (
        lambda context, ids: order.append(("detach", list(ids)))
    )
    assert engine._execute_single_kv_migration(
        ctx, uuid, 0, 1,
        _trajectory_tokens=lambda sequence, dtype: torch.zeros(3, dtype=dtype),
        _trajectory_slot=lambda sequence: 0,
        _get_or_create_gloo_group=lambda: order.append("group") or "gloo",
    ) is None
    # KV first, then the header, the trajectory and the turns; the pages are
    # released only after the whole payload left, and the attachments after.
    assert order == [
        "group", ("send", 1, 4), ("send", 1, 5), ("send", 1, 3), ("send", 1, 0),
        ("detach", [5]),
    ]
    assert view.released == [[5]]


def test_single_migration_on_the_destination_returns_both_staging_writes():
    order = []
    header = [1, 3, 2, 1, 0]

    def recv(tensor, src, group=None):
        order.append(("recv", src, tuple(tensor.shape)))
        if tensor.dtype == torch.int64 and tensor.numel() == 5:
            tensor.copy_(torch.tensor(header, dtype=torch.int64))
        elif tensor.dtype == torch.int32:
            tensor.copy_(torch.tensor([7, 8, 9], dtype=torch.int32))

    ns, engine, _ = d63(
        dist=SimpleNamespace(
            send=lambda tensor, dst, group=None: pytest.fail("the dest never sends"),
            recv=recv,
        ),
        torch=torch,
    )
    uuid = "ffffffffffff"
    seq = candidate(uuid, rank=0, idx=6, pages=1)
    seq._buffer_slot = -1
    view = migration_views()
    book = migration_book(uuid, slot=4)
    ctx = migration_context(
        rank=1, sequences={uuid: seq}, host_view=view,
        trajectories=SimpleNamespace(trajectory_book=book),
    )
    outcome = engine._execute_single_kv_migration(
        ctx, uuid, 0, 1,
        _trajectory_tokens=lambda sequence, dtype: pytest.fail("dest reads nothing"),
        _trajectory_slot=lambda sequence: pytest.fail("dest reads nothing"),
        _get_or_create_gloo_group=lambda: order.append("group") or "gloo",
    )
    assert view.registered == [[6], [(6, 64)]]
    assert book.restored == [(4, [7, 8, 9], {
        "prompt_length": 2, "decoded_length": 1, "turns": [],
    })]
    assert seq._buffer_slot == 4
    # The outcome carries exactly what each write added, nothing more.
    assert outcome._migrated_sequences == uuid
    assert outcome._pending_migrated_query_book["slot"] == 4
    assert outcome._pending_migrated_query_book["restored"] is True
    assert outcome._pending_migrated_query_book["prompt_length"] == 2
    assert outcome._pending_migrated_query_book["trajectory"].tolist() == [7, 8, 9]


def test_single_migration_refuses_a_destination_without_a_reservation():
    def recv(tensor, src, group=None):
        if tensor.dtype == torch.int64 and tensor.numel() == 5:
            tensor.copy_(torch.tensor([1, 0, 0, 0, 0], dtype=torch.int64))

    ns, engine, _ = d63(
        dist=SimpleNamespace(send=lambda **kw: None, recv=recv), torch=torch,
    )
    uuid = "gggggggggggg"
    seq = candidate(uuid, rank=0, idx=7, pages=1)
    ctx = migration_context(
        rank=1, sequences={uuid: seq}, host_view=migration_views(),
        trajectories=SimpleNamespace(
            trajectory_book=migration_book(uuid, present=False),
        ),
    )
    with pytest.raises(Exception, match="no admission reservation"):
        engine._execute_single_kv_migration(
            ctx, uuid, 0, 1,
            _trajectory_tokens=lambda sequence, dtype: None,
            _trajectory_slot=lambda sequence: 0,
            _get_or_create_gloo_group=lambda: "gloo",
        )


@pytest.mark.parametrize("pages,message", [
    (0, "no host pages allocated"),
])
def test_single_migration_refuses_an_unallocated_sequence(pages, message):
    ns, engine, events = d63()
    uuid = "hhhhhhhhhhhh"
    ctx = migration_context(sequences={uuid: candidate(uuid, rank=0, idx=8, pages=pages)})
    assert engine._execute_single_kv_migration(
        ctx, uuid, 0, 1,
        _trajectory_tokens=lambda sequence, dtype: pytest.fail("no transfer"),
        _trajectory_slot=lambda sequence: pytest.fail("no transfer"),
        _get_or_create_gloo_group=lambda: pytest.fail("no group"),
    ) is None
    assert [line for kind, line in events if message in line]
    missing = migration_context(sequences={})
    assert engine._execute_single_kv_migration(
        missing, uuid, 0, 1,
        _trajectory_tokens=lambda sequence, dtype: pytest.fail("no transfer"),
        _trajectory_slot=lambda sequence: pytest.fail("no transfer"),
        _get_or_create_gloo_group=lambda: pytest.fail("no group"),
    ) is None
    assert [line for kind, line in events if "sequence not found" in line]


def rebalance_sequence(uuid, *, rank):
    commits = []
    return SimpleNamespace(
        uuid=uuid, assigned_rank=rank, global_idx=0, kv_token_budget=512,
        host_pages_allocated=1, commits=commits, prompt_length=1,
        input_ids=None, decoded_tokens=None,
        log_event=lambda *args, **kwargs: None,
        commit_host_reservation=(
            lambda pages, tokens, owned: commits.append((pages, tokens, owned))
        ),
        rebase_decoded_length=lambda value: None,
    )


def test_rebalance_without_preemption_plans_nothing():
    ns, engine, events = d63()
    assert engine._rebalance_host_kv(
        migration_context(), False, {}, None,
        _plan_kv_migration=lambda: pytest.fail("no plan without preemption"),
        _execute_kv_migrations_parallel=lambda ops: pytest.fail("no transfer"),
        _sync_sequence_metadata=lambda uuids: pytest.fail("no sync"),
        _bump_decode_row_epoch=lambda: pytest.fail("no epoch bump"),
        _get_host_kv_utilization=lambda: pytest.fail("no stats read"),
    ) is None
    assert events == []


def test_rebalance_with_an_empty_plan_reaches_no_barrier():
    distributed, order = recording_dist()
    ns, engine, _ = d63(dist=distributed)
    assert engine._rebalance_host_kv(
        migration_context(), True, {}, None,
        _plan_kv_migration=lambda: order.append("plan") or [],
        _execute_kv_migrations_parallel=lambda ops: pytest.fail("no transfer"),
        _sync_sequence_metadata=lambda uuids: pytest.fail("no sync"),
        _bump_decode_row_epoch=lambda: pytest.fail("no epoch bump"),
        _get_host_kv_utilization=lambda: pytest.fail("no stats read"),
    ) is None
    assert order == ["plan"]


def test_rebalance_keeps_the_plan_execute_sync_barrier_order_on_the_source():
    distributed, order = recording_dist()
    ns, engine, _ = d63(dist=distributed)
    uuid = "iiiiiiiiiiii"
    seq = rebalance_sequence(uuid, rank=0)
    op = MigrationOp(uuid, from_rank=0, to_rank=1, pages=2, host_pages=2)
    table = SimpleNamespace(
        unbind=lambda name: order.append(("unbind", name)) or 5,
        local_of=lambda name: 5,
        bind=lambda name: pytest.fail("the source never binds"),
    )
    resident = SimpleNamespace(drop=lambda name: order.append(("drop", name)))
    book = {5: "entry"}
    ctx = migration_context(
        rank=0, sequences={uuid: seq}, seq_table=table, gpu_resident=resident,
        trajectories=SimpleNamespace(trajectory_book=None),
    )
    assert engine._rebalance_host_kv(
        ctx, True, book, None,
        _plan_kv_migration=lambda: order.append("plan") or [op],
        _execute_kv_migrations_parallel=lambda ops: order.append(("execute", len(ops))),
        _sync_sequence_metadata=lambda uuids: order.append(("sync", list(uuids))),
        _bump_decode_row_epoch=lambda: order.append("epoch"),
        _get_host_kv_utilization=lambda: order.append("stats") or dict(NODE_STATS),
    ) is None
    assert order == [
        "plan", ("execute", 1),
        "barrier", ("sync", [uuid]), "barrier",
        ("unbind", uuid), "epoch", ("drop", uuid),
        "barrier", "stats",
    ]
    # Ownership and the host reservation are committed before the barriers.
    assert seq.assigned_rank == 1
    assert seq.commits == [(2, 2 * 64, 2)]
    assert book == {}


def test_rebalance_on_the_destination_consumes_the_staging_entry():
    distributed, order = recording_dist()
    ns, engine, events = d63(dist=distributed)
    uuid = "jjjjjjjjjjjj"
    seq = rebalance_sequence(uuid, rank=0)
    op = MigrationOp(uuid, from_rank=0, to_rank=1, pages=1, host_pages=1)
    table = SimpleNamespace(
        unbind=lambda name: pytest.fail("the destination never unbinds"),
        local_of=lambda name: 3,
        bind=lambda name: order.append(("bind", name)) or 3,
    )
    book = {}
    pending = {uuid: {
        'trajectory': torch.tensor([1, 2, 3], dtype=torch.int32),
        'prompt_length': 2, 'decoded_length': 1, 'turns': (),
        'restored': True, 'slot': 3, 'kv_token_budget': 512,
    }}
    trajectory_book = migration_book(uuid, slot=3)
    ctx = migration_context(
        rank=1, sequences={uuid: seq}, seq_table=table,
        gpu_resident=SimpleNamespace(drop=lambda name: pytest.fail("no drop")),
        trajectories=SimpleNamespace(trajectory_book=trajectory_book),
    )
    assert engine._rebalance_host_kv(
        ctx, True, book, pending,
        _plan_kv_migration=lambda: [op],
        _execute_kv_migrations_parallel=lambda ops: order.append("execute"),
        _sync_sequence_metadata=lambda uuids: order.append("sync"),
        _bump_decode_row_epoch=lambda: order.append("epoch"),
        _get_host_kv_utilization=lambda: pytest.fail("a follower logs nothing"),
    ) is None
    assert order == [
        "execute", "barrier", "sync", "barrier",
        ("bind", uuid), "epoch", "barrier",
    ]
    # The staged record is consumed exactly once, and an already-restored
    # payload is not restored a second time.
    assert pending == {}
    assert trajectory_book.restored == []
    assert book[3]
    missing = [line for kind, line in events if "No pending query_book" in line]
    assert missing == []


def test_rebalance_without_a_staging_map_logs_the_missing_payload():
    distributed, _ = recording_dist()
    ns, engine, events = d63(dist=distributed)
    uuid = "kkkkkkkkkkkk"
    seq = rebalance_sequence(uuid, rank=0)
    op = MigrationOp(uuid, from_rank=0, to_rank=1, pages=1, host_pages=1)
    ctx = migration_context(
        rank=1, sequences={uuid: seq},
        seq_table=SimpleNamespace(bind=lambda name: 3, local_of=lambda name: 3),
        trajectories=SimpleNamespace(trajectory_book=None),
    )
    assert engine._rebalance_host_kv(
        ctx, True, {}, None,
        _plan_kv_migration=lambda: [op],
        _execute_kv_migrations_parallel=lambda ops: None,
        _sync_sequence_metadata=lambda uuids: None,
        _bump_decode_row_epoch=lambda: None,
        _get_host_kv_utilization=lambda: pytest.fail("a follower logs nothing"),
    ) is None
    assert [line for kind, line in events if "No pending query_book" in line]


# ---- D6.3 delegators -------------------------------------------------------


WATERMARK_HELPERS = ("_host_kv_watermark_context", "_decode_phase_context")


def test_delegator_applies_the_watermark_writes_in_the_old_order(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, WATERMARK_HELPERS)
    worker._kv_capacity_engine = engine
    worker.rank = worker.local_rank = 0
    worker.world_size = 1
    worker.enable_decode_preemption = True
    worker.enable_host_kv_eviction = False
    worker.host_kv_watermark = 70
    worker.global_batch = recording_pool()
    worker._get_host_kv_utilization = lambda: dict(NODE_STATS)
    worker._credit_reclaimable_prefix_pages = lambda stats: stats
    seen = []

    def check(ctx, **helpers):
        seen.append((ctx, sorted(helpers)))
        assert not hasattr(worker, "_host_kv_page_stats")
        return True, ns["WatermarkTriggerOutcome"](
            _host_kv_page_stats={"used": 1},
        )

    engine._check_host_kv_watermark_trigger = check
    assert worker._check_host_kv_watermark_trigger() is True
    ctx, helpers = seen[0]
    assert helpers == [
        "_credit_reclaimable_prefix_pages", "_get_host_kv_utilization",
    ]
    # The context carries the launch flags as values and the counter as the
    # hasattr miss the pre-move code saw.
    assert ctx.host_kv_watermark == 70
    assert ctx._watermark_check_counter is None
    assert ctx.pool is worker.global_batch
    assert worker._host_kv_page_stats == {"used": 1}
    # An unchanged field is not bound, so the counter stays absent.
    assert not hasattr(worker, "_watermark_check_counter")
    engine._check_host_kv_watermark_trigger = lambda ctx, **kw: (
        False, ns["WatermarkTriggerOutcome"](_watermark_check_counter=4)
    )
    assert worker._check_host_kv_watermark_trigger() is False
    assert worker._watermark_check_counter == 4
    assert worker._host_kv_page_stats == {"used": 1}


def test_delegator_latches_the_migration_skip_log(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, WATERMARK_HELPERS)
    worker._kv_capacity_engine = engine
    worker._held_decode_phase_context = object()
    worker._decode_attn_tp_size = lambda: 8
    worker._get_host_kv_utilization = lambda: dict(NODE_STATS)
    planned = [MigrationOp("a", from_rank=0, to_rank=1, pages=1, host_pages=1)]
    seen = []

    def plan(ctx, latched, **helpers):
        seen.append((ctx, latched, sorted(helpers)))
        return planned, ns["MigrationPlanOutcome"](_logged_migration_skip_tp=True)

    engine._plan_kv_migration = plan
    assert worker._plan_kv_migration() is planned
    ctx, latched, helpers = seen[0]
    assert ctx is worker._held_decode_phase_context
    assert latched is False
    assert helpers == ["_decode_attn_tp_size", "_get_host_kv_utilization"]
    assert worker._logged_migration_skip_tp is True
    # The latch the delegator now reports is the one it bound.
    engine._plan_kv_migration = lambda ctx, latched, **kw: (
        [], ns["MigrationPlanOutcome"]()
    )
    assert worker._plan_kv_migration() == []
    assert worker._logged_migration_skip_tp is True
    engine._plan_kv_migration = plan
    worker._plan_kv_migration()
    assert seen[-1][1] is True


def test_delegator_adds_the_received_payload_without_replacing_the_stores(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, WATERMARK_HELPERS)
    worker._kv_capacity_engine = engine
    worker._held_decode_phase_context = object()
    worker._trajectory_tokens = lambda seq, dtype: None
    worker._trajectory_slot = lambda seq: 0
    worker._get_or_create_gloo_group = lambda: "gloo"
    seen = []

    def execute(ctx, uuid, from_rank, to_rank, **helpers):
        seen.append((ctx, uuid, from_rank, to_rank, sorted(helpers)))
        assert not hasattr(worker, "_migrated_sequences")
        return ns["KVMigrationReceiveOutcome"](uuid, {"slot": 1})

    engine._execute_single_kv_migration = execute
    assert worker._execute_single_kv_migration("one", 0, 1) is None
    ctx, uuid, from_rank, to_rank, helpers = seen[0]
    assert (ctx, uuid, from_rank, to_rank) == (
        worker._held_decode_phase_context, "one", 0, 1,
    )
    assert helpers == [
        "_get_or_create_gloo_group", "_trajectory_slot", "_trajectory_tokens",
    ]
    assert worker._migrated_sequences == {"one"}
    assert worker._pending_migrated_query_book == {"one": {"slot": 1}}
    # A second receive ADDS to both stores instead of replacing them.
    engine._execute_single_kv_migration = lambda *a, **kw: (
        ns["KVMigrationReceiveOutcome"]("two", {"slot": 2})
    )
    assert worker._execute_single_kv_migration("two", 0, 1) is None
    assert worker._migrated_sequences == {"one", "two"}
    assert worker._pending_migrated_query_book == {
        "one": {"slot": 1}, "two": {"slot": 2},
    }
    # The source branch writes nothing at all.
    engine._execute_single_kv_migration = lambda *a, **kw: None
    assert worker._execute_single_kv_migration("three", 0, 1) is None
    assert worker._migrated_sequences == {"one", "two"}


def test_rebalance_delegator_hands_over_the_worker_owned_containers(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, WATERMARK_HELPERS)
    worker._kv_capacity_engine = engine
    worker._held_decode_phase_context = object()
    worker.enable_decode_preemption = True
    worker.query_book = {7: "entry"}
    worker._sync_sequence_metadata = lambda uuids: None
    worker._bump_decode_row_epoch = lambda: 1
    worker._get_host_kv_utilization = lambda: dict(NODE_STATS)
    worker._decode_attn_tp_size = lambda: 1
    worker._trajectory_tokens = lambda seq, dtype: None
    worker._trajectory_slot = lambda seq: 0
    worker._get_or_create_gloo_group = lambda: "gloo"
    seen = []

    def rebalance(ctx, preemption, book, pending, **helpers):
        seen.append((ctx, preemption, book, pending, sorted(helpers)))
        # The receive stages its payload DURING the rebalance, so the map the
        # engine was handed has to be the live one the receive adds into --
        # resolving it before the transfers would hand over a map that is
        # already stale (and, on a worker's first rebalance, absent).
        engine._execute_single_kv_migration = lambda *a, **kw: (
            ns["KVMigrationReceiveOutcome"]("one", {"slot": 1})
        )
        worker._execute_single_kv_migration("one", 0, 1)
        assert pending == {"one": {"slot": 1}}
        assert pending is worker._pending_migrated_query_book

    engine._rebalance_host_kv = rebalance
    assert worker._rebalance_host_kv() is None
    ctx, preemption, book, pending, helpers = seen[0]
    assert ctx is worker._held_decode_phase_context
    assert preemption is True
    assert book is worker.query_book
    assert helpers == [
        "_bump_decode_row_epoch", "_execute_kv_migrations_parallel",
        "_get_host_kv_utilization", "_plan_kv_migration",
        "_sync_sequence_metadata",
    ]
    # A later rebalance reuses the same container, never a fresh one.
    worker._rebalance_host_kv()
    assert seen[-1][3] is worker._pending_migrated_query_book


def test_due_check_delegators_pass_the_completion_predicate(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, WATERMARK_HELPERS)
    worker._kv_capacity_engine = engine
    worker._held_decode_phase_context = object()
    worker._is_sequence_completed = lambda seq: False
    seen = []
    engine._host_capacity_due_local = (
        lambda ctx, batch, **helpers: seen.append(("local", ctx, batch, helpers))
    )
    engine._host_capacity_due_all_ranks = (
        lambda ctx, batch, **helpers: seen.append(("all", ctx, batch, helpers))
    )
    worker._host_capacity_due_local([1, 2])
    worker._host_capacity_due_all_ranks([3])
    assert [row[0] for row in seen] == ["local", "all"]
    for _, ctx, batch, helpers in seen:
        assert ctx is worker._held_decode_phase_context
        assert helpers == {"_is_sequence_completed": worker._is_sequence_completed}
    assert [row[2] for row in seen] == [[1, 2], [3]]


# ---- D6.4: the decode KV append to host and the host-to-GPU KV load ---------


def d64(events=None, **extra):
    """The engine with the collaborators the D6.4 bodies name at module scope.

    ``torch.cuda`` refuses every attribute unless a test opts in: the launch
    paths must not reach the CPU-side synchronizer, and the two loads that do
    sync have to be seen doing it in the recorded order. A caller that already
    owns an event log passes it, so the collective and the sync land in the
    SAME ordered record as the launches.
    """
    events = [] if events is None else events

    class _ForbiddenCuda:
        def __getattr__(self, name):
            raise AssertionError(f"the engine touched torch.cuda.{name}")

    namespace = {
        "logging": SimpleNamespace(
            info=lambda *a: events.append(("info", a)),
            debug=lambda *a: events.append(("debug", a)),
            warning=lambda *a: events.append(("warning", a)),
            error=lambda *a: events.append(("error", a)),
        ),
        "torch": SimpleNamespace(
            cuda=_ForbiddenCuda(), Tensor=object, tensor=torch.tensor,
            int64=torch.int64,
        ),
        "time": SimpleNamespace(perf_counter=lambda: len(events) * 0.5),
        "dist": SimpleNamespace(
            barrier=lambda: events.append("barrier"),
        ),
        "DualKVCacheCoordinator": type("NoDual", (), {}),
        "DualHostKVCoordinator": type("NoDualHost", (), {}),
    }
    namespace.update(extra)
    ns = load_kv_capacity(namespace)
    return ns, ns["KVCapacityEngine"](), events


class _FakeTensor:
    """A tensor stand-in whose rows and rank the flush actually inspects."""

    def __init__(self, rows=2, dims=3, name="k"):
        self.shape = (rows,) + (1,) * (dims - 1)
        self._dims = dims
        self.name = name
        self.unsqueezed = 0

    def dim(self):
        return self._dims

    def unsqueeze(self, dim):
        assert dim == 2
        self.unsqueezed += 1
        self._dims += 1
        return self

    def __repr__(self):
        return f"<{self.name}>"


def flush_channel(events, *, batched=True):
    """A host view that records every launch into the shared event log."""

    def launch_batched(entries, sequence_ids, sequence_lengths):
        events.append(("launch_batched", tuple(e[0] for e in entries),
                       tuple(sequence_ids), tuple(sequence_lengths)))
        return f"task:{tuple(e[0] for e in entries)}"

    def launch_one(layer_idx, sequence_ids, k_tensor, v_tensor,
                   sequence_lengths, scale_tensor=None):
        events.append(("launch_one", layer_idx, k_tensor.name,
                       None if v_tensor is None else v_tensor.name))
        return f"task:{layer_idx}"

    view = SimpleNamespace(async_append_decode_kv_to_host=launch_one)
    if batched:
        view.async_append_decode_kv_to_host_batched_kernel = launch_batched
    return view


def flush_context(ns, **changes):
    values = dict(
        deferred_entries=[], deferred_entries_aux=[],
        deferred_worker_view=None, deferred_batch=None,
        deferred_worker_view_aux=None,
    )
    return ns["DeferredKVFlushContext"](**(values | changes))


class _RecordingChannel:
    """The worker's append channel, recording the order of every mutation."""

    def __init__(self, events):
        self.events = events
        self.tasks = _RecordingList(events, "task")
        self.tensors = _RecordingList(events, "retain")


class _RecordingList(list):
    def __init__(self, events, label):
        super().__init__()
        self._events = events
        self._label = label

    def append(self, item):
        self._events.append((self._label, getattr(item, "name", item)))
        super().append(item)

    def clear(self):
        self._events.append((self._label, "cleared"))
        super().clear()


def test_deferred_flush_with_an_empty_channel_resets_nothing():
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    outcome = engine._flush_deferred_kv_to_host(
        flush_context(ns), channel.tasks, channel.tensors,
        _ensure_host_kv_append_capacity=lambda ids, lengths: events.append("cap"),
    )
    unchanged = ns["GPU_KV_UNCHANGED"]
    assert all(getattr(outcome, field.name) is unchanged
               for field in fields(outcome))
    assert events == []


def test_deferred_flush_refuses_aux_entries_without_the_aux_view():
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    ctx = flush_context(ns, deferred_entries_aux=[(3, _FakeTensor(), None)])
    with pytest.raises(RuntimeError, match="auxiliary host KV worker view"):
        engine._flush_deferred_kv_to_host(
            ctx, channel.tasks, channel.tensors,
            _ensure_host_kv_append_capacity=lambda ids, lengths: None,
        )
    # The refusal precedes every channel mutation and every launch.
    assert events == []


def test_deferred_flush_without_a_usable_batch_resets_only_the_entry_lists():
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    # Primary entries queued but no batch metadata and no aux view: the
    # pre-move body dropped the entries and left the three view/batch fields
    # exactly as the forward had set them.
    ctx = flush_context(
        ns, deferred_entries=[(0, _FakeTensor(), None)],
        deferred_worker_view=flush_channel(events), deferred_batch=None,
    )
    outcome = engine._flush_deferred_kv_to_host(
        ctx, channel.tasks, channel.tensors,
        _ensure_host_kv_append_capacity=lambda ids, lengths: None,
    )
    unchanged = ns["GPU_KV_UNCHANGED"]
    assert outcome._deferred_kv_entries == []
    assert outcome._deferred_kv_entries_aux == []
    assert outcome._deferred_kv_batch is unchanged
    assert outcome._deferred_kv_worker_view is unchanged
    assert outcome._deferred_kv_worker_view_aux is unchanged
    assert events == []


def test_deferred_flush_checks_capacity_before_any_launch_and_retains_in_order(
    monkeypatch,
):
    """The batched-kernel path: capacity, then retain, then launch, then task.

    This is the stream-order contract. The C++ append orders its own D2H
    stream behind the producer stream with an event, so the body must not
    sync; the tensors have to be retained in the SAME list the later drain
    clears, before the task that reads from them is handed over.
    """
    monkeypatch.setenv("BATCHGEN_KV_OFFLOAD_UVA_KERNEL", "1")
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    k_primary, v_primary = _FakeTensor(name="kp"), _FakeTensor(name="vp")
    k_aux, scale_aux = _FakeTensor(name="ka"), _FakeTensor(name="sa")
    view, aux_view = flush_channel(events), flush_channel(events)
    ctx = flush_context(
        ns,
        deferred_entries=[(1, k_primary, v_primary)],
        deferred_entries_aux=[(2, k_aux, None, scale_aux)],
        deferred_worker_view=view, deferred_worker_view_aux=aux_view,
        deferred_batch=([11, 12], [63, 64]),
    )
    outcome = engine._flush_deferred_kv_to_host(
        ctx, channel.tasks, channel.tensors,
        _ensure_host_kv_append_capacity=(
            lambda ids, lengths: events.append(("capacity", tuple(ids), tuple(lengths)))
        ),
    )
    assert events == [
        ("capacity", (11, 12), (63, 64)),
        ("retain", "kp"), ("retain", "vp"),
        ("launch_batched", (1,), (11, 12), (63, 64)),
        ("task", "task:(1,)"),
        ("retain", "ka"), ("retain", "sa"),
        ("launch_batched", (2,), (11, 12), (63, 64)),
        ("task", "task:(2,)"),
    ]
    # Every append landed in the objects the worker still owns.
    assert channel.tensors == [k_primary, v_primary, k_aux, scale_aux]
    assert channel.tasks == ["task:(1,)", "task:(2,)"]
    # The whole channel is reset on the full path.
    assert outcome._deferred_kv_entries == []
    assert outcome._deferred_kv_entries_aux == []
    assert outcome._deferred_kv_batch is None
    assert outcome._deferred_kv_worker_view is None
    assert outcome._deferred_kv_worker_view_aux is None


def test_deferred_flush_per_layer_path_launches_before_it_retains(monkeypatch):
    """The non-kernel path launches each layer FIRST, then retains its rows.

    The two paths differ in exactly this order and the move kept both: the
    per-layer API takes the tensors itself, so the retention follows its
    return, while the batched kernel is handed a prepared list.
    """
    monkeypatch.setenv("BATCHGEN_KV_OFFLOAD_UVA_KERNEL", "0")
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    k0, v0 = _FakeTensor(name="k0"), _FakeTensor(name="v0")
    k1 = _FakeTensor(name="k1")
    ctx = flush_context(
        ns, deferred_entries=[(0, k0, v0), (1, k1, None)],
        deferred_worker_view=flush_channel(events, batched=False),
        deferred_batch=([5, 6], [7, 8]),
    )
    engine._flush_deferred_kv_to_host(
        ctx, channel.tasks, channel.tensors,
        _ensure_host_kv_append_capacity=lambda ids, lengths: events.append("capacity"),
    )
    assert events == [
        "capacity",
        ("launch_one", 0, "k0", "v0"), ("retain", "k0"), ("retain", "v0"),
        ("task", "task:0"),
        ("launch_one", 1, "k1", None), ("retain", "k1"), ("task", "task:1"),
    ]


def test_deferred_flush_refuses_a_row_mismatch_before_handing_over_the_task(
    monkeypatch,
):
    monkeypatch.setenv("BATCHGEN_KV_OFFLOAD_UVA_KERNEL", "1")
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    ctx = flush_context(
        ns, deferred_entries=[(4, _FakeTensor(rows=3, name="kp"), None)],
        deferred_worker_view=flush_channel(events),
        deferred_batch=([1, 2], [9, 9]),
    )
    with pytest.raises(RuntimeError, match="primary deferred KV row mismatch at layer 4"):
        engine._flush_deferred_kv_to_host(
            ctx, channel.tasks, channel.tensors,
            _ensure_host_kv_append_capacity=lambda ids, lengths: None,
        )
    assert channel.tasks == []


def test_deferred_flush_skips_the_capacity_check_without_batch_metadata(
    monkeypatch,
):
    """Aux-only entries with no batch tuple still launch, with ids ``None``."""
    monkeypatch.setenv("BATCHGEN_KV_OFFLOAD_UVA_KERNEL", "1")
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    aux_view = flush_channel(events)
    ctx = flush_context(
        ns, deferred_entries_aux=[(2, _FakeTensor(name="ka"), None, None)],
        deferred_worker_view_aux=aux_view, deferred_batch=None,
    )
    engine._flush_deferred_kv_to_host(
        ctx, channel.tasks, channel.tensors,
        _ensure_host_kv_append_capacity=lambda ids, lengths: events.append("capacity"),
    )
    # No batch metadata, so no capacity call and no launch (the aux branch
    # needs sequence ids); the channel reset still runs.
    assert events == []


def append_capacity_context(sequences, *, pool=True):
    rows = {int(gid): seq for gid, seq in sequences.items()}
    return DecodePhaseContext(
        facts=RankFacts(
            rank=0, world_size=1, local_rank=0, local_world_size=1,
            rank_ctx=SimpleNamespace(node_id=0), attn_tp_size=1, device="cpu",
            page_size=64, decision_interval=1, enable_prefix_cache=False,
            enable_host_kv_eviction=False, host_kv_chunk_size=1,
        ),
        sched=SchedulingState(
            pool=SimpleNamespace(
                get_sequence_by_global_idx=rows.get,
            ) if pool else None,
            seq_table=None, trajectories=None,
        ),
        gpu_kv=None, host_view=None, core_engine=None, gpu_resident=None,
        prefix_coordinator=None, prefix_runtime=None, chunk_sizer=None,
    )


def append_row(*, capacity=128, budget=4096, context=64, decoded=0, growth=2):
    return SimpleNamespace(
        host_token_capacity=capacity, kv_token_budget=budget,
        current_context_length=context, decoded_length=decoded,
        status=SimpleNamespace(name="DECODING"),
        get_host_growth_pages=lambda tokens: growth,
    )


def test_append_capacity_passes_a_row_inside_its_reservation():
    _, engine, _ = d64()
    ctx = append_capacity_context({7: append_row(capacity=128)})
    assert engine._ensure_host_kv_append_capacity(ctx, [7], [63]) is None


def test_append_capacity_without_a_pool_returns_before_reading_rows():
    _, engine, _ = d64()
    ctx = append_capacity_context({}, pool=False)
    assert engine._ensure_host_kv_append_capacity(ctx, [7], [63]) is None


@pytest.mark.parametrize("ids,lengths,rows,message", [
    ([7], [1, 2], {}, "host KV append metadata mismatch"),
    ([7], [1], {}, "host KV append for unknown global_idx=7"),
    ([7], [-1], {7: append_row()}, "negative write position"),
    ([7], [3], {7: append_row(capacity=0)}, "unallocated gid=7"),
    ([7], [9], {7: append_row(budget=4)}, "exceed token budget"),
    ([7], [200], {7: append_row(capacity=128, budget=4096)},
     "reached without a boundary reservation"),
])
def test_append_capacity_refuses_every_broken_row(ids, lengths, rows, message):
    _, engine, _ = d64()
    ctx = append_capacity_context(rows)
    with pytest.raises(RuntimeError, match=message):
        engine._ensure_host_kv_append_capacity(ctx, ids, lengths)


def aux_slots_context(ns, *, layer_map=(), aux_pool=True, coordinator=True,
                      cache=False):
    """``coordinator=False`` = no GPU KV manager; ``aux_pool=False`` = no aux."""
    coord = None
    if coordinator:
        aux = SimpleNamespace(
            config=SimpleNamespace(logical_to_physical_layer=layer_map)
        ) if aux_pool else None
        coord = SimpleNamespace(auxiliary=aux)
    return ns["AuxLayerSlotsContext"](
        aux_coordinator=coord, _aux_layers_with_slots_cache=cache,
    )


def test_aux_slot_map_resolves_the_mapped_layers_and_caches_them():
    ns, engine, _ = d64()
    ctx = aux_slots_context(ns, layer_map=[0, None, 2, -1, 4])
    result, outcome = engine._aux_layers_with_slots(ctx)
    assert result == frozenset({0, 2, 4})
    assert outcome._aux_layers_with_slots_cache == frozenset({0, 2, 4})


def test_aux_slot_map_without_a_layer_mapping_caches_none():
    ns, engine, _ = d64()
    result, outcome = engine._aux_layers_with_slots(
        aux_slots_context(ns, layer_map=None)
    )
    # ``logical_to_physical_layer=None`` means every layer has a slot; that
    # IS a resolved answer, so it is cached as None.
    assert result is None
    assert outcome._aux_layers_with_slots_cache is None


@pytest.mark.parametrize("coordinator,aux_pool", [
    (False, True),   # no GPU KV coordinator bound yet
    (True, False),   # coordinator bound, auxiliary pool not created
])
def test_aux_slot_map_without_an_aux_pool_returns_none_and_does_not_cache(
    coordinator, aux_pool,
):
    ns, engine, _ = d64()
    ctx = aux_slots_context(ns, coordinator=coordinator, aux_pool=aux_pool)
    result, outcome = engine._aux_layers_with_slots(ctx)
    assert result is None
    assert outcome._aux_layers_with_slots_cache is ns["GPU_KV_UNCHANGED"]


@pytest.mark.parametrize("cache", [None, frozenset({3})])
def test_aux_slot_map_cache_hit_returns_the_memo_and_rewrites_nothing(cache):
    ns, engine, _ = d64()
    # A hit must not even look at the coordinator: ``False`` is the only
    # "unresolved" value, so a cached ``None`` short-circuits too.
    ctx = ns["AuxLayerSlotsContext"](
        aux_coordinator=None, _aux_layers_with_slots_cache=cache,
    )
    result, outcome = engine._aux_layers_with_slots(ctx)
    assert result is cache
    assert outcome._aux_layers_with_slots_cache is ns["GPU_KV_UNCHANGED"]


def aux_append_context(rows, sequences, *, aux_view):
    return DecodePhaseContext(
        facts=RankFacts(
            rank=0, world_size=1, local_rank=0, local_world_size=1,
            rank_ctx=SimpleNamespace(node_id=0), attn_tp_size=1, device="cpu",
            page_size=64, decision_interval=1, enable_prefix_cache=False,
            enable_host_kv_eviction=False, host_kv_chunk_size=1,
        ),
        sched=SchedulingState(
            pool=SimpleNamespace(get_sequence=sequences.get),
            seq_table=SimpleNamespace(local_to_uuid=rows),
            trajectories=None,
        ),
        gpu_kv=None, host_view=None, core_engine=None, gpu_resident=None,
        prefix_coordinator=None, prefix_runtime=None, chunk_sizer=None,
        host_view_aux=aux_view,
    )


@pytest.mark.parametrize("aux_view,batch", [(None, [0]), ("view", [])])
def test_aux_append_without_a_view_or_a_batch_never_touches_the_channel(
    aux_view, batch,
):
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    ctx = aux_append_context({}, {}, aux_view=aux_view)
    engine._append_decode_kv_to_host_aux_async(
        ctx, channel.tasks, channel.tensors, 0, batch, _FakeTensor(),
        _aux_layers_with_slots=lambda: events.append("slots"),
        _wait_pending_kv_append_tasks=lambda: events.append("drain"),
    )
    assert events == []


def test_aux_append_drops_an_unmapped_layer_before_resolving_any_row():
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    ctx = aux_append_context(
        {0: "u0"}, {}, aux_view=flush_channel(events, batched=False),
    )
    engine._append_decode_kv_to_host_aux_async(
        ctx, channel.tasks, channel.tensors, 7, [0], _FakeTensor(),
        _aux_layers_with_slots=lambda: frozenset({1, 2}),
        _wait_pending_kv_append_tasks=lambda: events.append("drain"),
    )
    # The gate returns before the row lookup, the launch and the channel.
    assert events == []
    assert channel.tasks == [] and channel.tensors == []


def test_aux_append_launches_then_retains_then_queues_the_task():
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    k, scale = _FakeTensor(name="ka"), _FakeTensor(name="sa")
    ctx = aux_append_context(
        {0: "u0", 1: "u1"},
        {"u0": SimpleNamespace(global_idx=10, current_context_length=5),
         "u1": SimpleNamespace(global_idx=11, current_context_length=9)},
        aux_view=flush_channel(events, batched=False),
    )
    engine._append_decode_kv_to_host_aux_async(
        ctx, channel.tasks, channel.tensors, 3, [0, 1], k, None, scale,
        _aux_layers_with_slots=lambda: None,
        _wait_pending_kv_append_tasks=lambda: events.append("drain"),
    )
    assert events == [
        ("launch_one", 3, "ka", None),
        ("retain", "ka"), ("retain", "sa"), ("task", "task:3"),
    ]
    # The write position is the row's context length minus one, per row.
    assert k.unsqueezed == 1


def test_aux_append_drains_only_after_the_appends_and_only_at_the_cap():
    """The 256-task cap drains the channel AFTER this round went into it.

    The drain clears both lists, so an append that landed anywhere but in
    those objects would be lost by it -- and an append made after the drain
    would survive a wait that never covered its copy.
    """
    ns, engine, events = d64()
    channel = _RecordingChannel(events)
    channel.tasks.extend(f"old:{i}" for i in range(255))
    ctx = aux_append_context(
        {0: "u0"}, {"u0": SimpleNamespace(global_idx=10, current_context_length=5)},
        aux_view=flush_channel(events, batched=False),
    )

    def drain():
        events.append(("drain", len(channel.tasks)))
        channel.tasks.clear()
        channel.tensors.clear()

    engine._append_decode_kv_to_host_aux_async(
        ctx, channel.tasks, channel.tensors, 3, [0], _FakeTensor(name="ka"),
        _aux_layers_with_slots=lambda: None,
        _wait_pending_kv_append_tasks=drain,
    )
    assert events == [
        ("launch_one", 3, "ka", None), ("retain", "ka"), ("task", "task:3"),
        ("drain", 256), ("task", "cleared"), ("retain", "cleared"),
    ]
    assert channel.tasks == [] and channel.tensors == []
    # One task short of the cap, the body queues and returns without draining.
    events.clear()
    channel.tasks.extend(f"old:{i}" for i in range(254))
    engine._append_decode_kv_to_host_aux_async(
        ctx, channel.tasks, channel.tensors, 3, [0], _FakeTensor(name="kb"),
        _aux_layers_with_slots=lambda: None,
        _wait_pending_kv_append_tasks=drain,
    )
    assert [e for e in events if e[0] == "drain"] == []
    assert len(channel.tasks) == 255


def load_context(ns, *, sequences=None, core_engine=..., worker_view=None,
                 synchronize=None):
    if core_engine is ...:
        core_engine = SimpleNamespace(host_paged_kv_worker_view=worker_view)
    return DecodePhaseContext(
        facts=RankFacts(
            rank=0, world_size=1, local_rank=0, local_world_size=1,
            rank_ctx=SimpleNamespace(node_id=0), attn_tp_size=1,
            device="cuda:0", page_size=64, decision_interval=1,
            enable_prefix_cache=False, enable_host_kv_eviction=False,
            host_kv_chunk_size=1,
        ),
        sched=SchedulingState(
            pool=SimpleNamespace(get_sequence=(sequences or {}).get),
            seq_table=SimpleNamespace(
                uuid_to_local={uuid: i for i, uuid in enumerate(sequences or {})},
            ),
            trajectories=None,
        ),
        gpu_kv=None, host_view=None, core_engine=core_engine,
        gpu_resident=None, prefix_coordinator=None, prefix_runtime=None,
        chunk_sizer=None,
    )


def ordered(events):
    """The recorded sequence without the diagnostic log lines.

    The log calls share the event list so a test CAN pin a message, but the
    order these tests assert is the stream/collective order, which the
    debug-only lines must not be able to shift.
    """
    return [e for e in events
            if not (isinstance(e, tuple) and e[0] in {"debug", "info"})]


def syncing_runtime(events, **extra):
    """``d64`` with the device sync allowed, so its position is observable."""
    return d64(
        events,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda device: events.append(("sync", device)),
            ),
            Tensor=object, tensor=torch.tensor, int64=torch.int64,
        ),
        **extra,
    )


def test_host_to_gpu_load_without_ids_returns_before_reading_the_view():
    _, engine, events = d64()
    ctx = load_context(None, core_engine=None)
    assert engine._load_host_kv_to_gpu(ctx, object(), []) is None
    assert events == []


def test_host_to_gpu_load_refuses_without_a_bound_worker_view():
    _, engine, _ = d64()
    ctx = load_context(None, worker_view=None)
    with pytest.raises(RuntimeError, match="not bound to the core engine"):
        engine._load_host_kv_to_gpu(ctx, object(), [10])


def test_host_to_gpu_load_waits_then_syncs_on_the_declared_device():
    """The primary path: launch, wait, then ONE device sync on ctx's device.

    The sync has to follow the wait -- the task's own completion event covers
    only its copy stream, and the caller is about to hand the pages to the
    model.
    """
    events = []
    ns, engine, _ = syncing_runtime(events)

    def launch(sequence_ids, active_page_counts, k_device_ptrs, v_device_ptrs):
        events.append(("launch", tuple(sequence_ids.tolist()), active_page_counts))
        return SimpleNamespace(wait=lambda: events.append("wait"))

    view = SimpleNamespace(async_load_layer_paged_kv_to_device=launch)
    manager = SimpleNamespace(
        get_padded_3d_page_pointers=lambda: ("k", "v"),
        export_active_sequence_page_counts=lambda: "counts",
    )
    ctx = load_context(ns, worker_view=view)
    assert engine._load_host_kv_to_gpu(ctx, manager, [10, 11]) is None
    assert ordered(events) == [
        ("launch", (10, 11), "counts"), "wait", ("sync", "cuda:0"),
    ]


def test_host_to_gpu_load_dual_path_prepares_launches_waits_then_syncs():
    events = []
    dual = type("Dual", (), {})
    ns, engine, _ = syncing_runtime(events, DualKVCacheCoordinator=dual)
    engine._prepare_dual_kv_load_pointers = (
        lambda ctx, mgr, ids: events.append(("prepare", tuple(ids))) or "ptrs"
    )
    engine._launch_dual_host_kv_load = (
        lambda ctx, ptrs: events.append(("launch", ptrs))
        or SimpleNamespace(wait=lambda: events.append("wait"))
    )
    ctx = load_context(ns, worker_view=object())
    assert engine._load_host_kv_to_gpu(ctx, dual(), [10]) is None
    assert ordered(events) == [
        ("prepare", (10,)), ("launch", "ptrs"), "wait", ("sync", "cuda:0"),
    ]


def test_integrate_waits_syncs_barriers_then_finalizes():
    """The exit protocol's four steps, in the one order every rank runs.

    All ranks reach this from a broadcast decision, so the barrier has to sit
    between the local wait and the finalize: a rank that finalized before the
    barrier would free pages a peer is still copying into.
    """
    events = []
    ns, engine, _ = syncing_runtime(events)
    task = SimpleNamespace(wait=lambda: events.append("wait"))
    ctx = load_context(ns)
    result = engine._integrate_pending_async_load(
        ctx, task, ["u0"], [0], [10], ["u1"], [1], "mgr",
        _finalize_async_load_minimal=(
            lambda *args: events.append(("finalize", args)) or (["u1", "u0"], [1, 0])
        ),
    )
    assert result == (["u1", "u0"], [1, 0])
    assert ordered(events) == [
        "wait", ("sync", "cuda:0"), "barrier",
        ("finalize", (task, ["u0"], [0], [10], ["u1"], [1], "mgr")),
    ]


def test_integrate_without_a_task_still_barriers_before_finalizing():
    events = []
    ns, engine, _ = syncing_runtime(events)
    engine._integrate_pending_async_load(
        load_context(ns), None, [], [], [], ["u1"], [1], "mgr",
        _finalize_async_load_minimal=lambda *args: events.append("finalize") or ([], []),
    )
    # No in-flight task means no wait and no sync, but the collective stays.
    assert ordered(events) == ["barrier", "finalize"]


def allocate_context(ns, *, manager, rows=None, sequences=None, resident=None):
    rows = rows if rows is not None else {}
    return DecodePhaseContext(
        facts=RankFacts(
            rank=0, world_size=1, local_rank=0, local_world_size=1,
            rank_ctx=SimpleNamespace(node_id=0), attn_tp_size=1, device="cpu",
            page_size=64, decision_interval=1, enable_prefix_cache=False,
            enable_host_kv_eviction=False, host_kv_chunk_size=1,
        ),
        sched=SchedulingState(
            pool=SimpleNamespace(get_sequence=(sequences or {}).get),
            seq_table=SimpleNamespace(
                local_to_uuid=rows,
                uuid_to_local={uuid: idx for idx, uuid in rows.items()},
            ),
            trajectories=None,
        ),
        gpu_kv=manager, host_view=None, core_engine=None,
        gpu_resident=resident, prefix_coordinator=None, prefix_runtime=None,
        chunk_sizer=None,
    )


def alloc_row(global_idx, *, pages=2, decoded=0):
    granted = []
    return SimpleNamespace(
        global_idx=global_idx, decoded_length=decoded,
        current_context_length=64, had_initial_gpu_reservation=False,
        granted=granted,
        get_gpu_pages_for_two_page_buffer=lambda: pages,
        grant_gpu_pages=(
            lambda count, initial: granted.append((count, initial))
        ),
    )


class _ResidentSet(set):
    def mark_resident(self, uuid):
        self.add(uuid)


@pytest.mark.parametrize("manager,rows,expected", [
    (None, [1], False),                       # no pool bound
    ("uninitialized", [1], False),            # pool bound but not initialized
    ("ready", [], True),                      # nothing to allocate
])
def test_two_page_allocation_early_exits_without_the_collective(
    manager, rows, expected,
):
    ns, engine, events = d64()
    pool = {
        None: None,
        "uninitialized": SimpleNamespace(is_initialized=False),
        "ready": SimpleNamespace(
            is_initialized=True,
            get_stats=lambda: SimpleNamespace(num_free_pages=9),
        ),
    }[manager]
    ctx = allocate_context(ns, manager=pool)
    engine._collective_gpu_kv_allocation_preflight = (
        lambda c, required, free: events.append(("preflight", required, free))
    )
    assert engine._allocate_gpu_kv_two_page_buffer(ctx, rows) is expected
    # Without the preflight flag no rank joins a collective on these exits.
    assert events == []


@pytest.mark.parametrize("manager,required,free", [
    (None, 0, 0), ("uninitialized", 0, 0), ("ready", 0, 9),
])
def test_two_page_allocation_early_exits_still_join_the_collective(
    manager, required, free,
):
    """A rank that cannot allocate still votes, so peers cannot run ahead."""
    ns, engine, events = d64()
    pool = {
        None: None,
        "uninitialized": SimpleNamespace(is_initialized=False),
        "ready": SimpleNamespace(
            is_initialized=True,
            get_stats=lambda: SimpleNamespace(num_free_pages=9),
        ),
    }[manager]
    ctx = allocate_context(ns, manager=pool)
    engine._collective_gpu_kv_allocation_preflight = (
        lambda c, req, fr: events.append(("preflight", req, fr)) or "voted"
    )
    rows = [] if manager == "ready" else [1]
    result = engine._allocate_gpu_kv_two_page_buffer(
        ctx, rows, collective_preflight=True,
    )
    assert result == "voted"
    assert events == [("preflight", required, free)]


def ready_pool(events, *, free=10):
    return SimpleNamespace(
        is_initialized=True,
        get_stats=lambda: SimpleNamespace(num_free_pages=free),
        count_new_shared_pages=(
            lambda ids, tokens, keys: events.append(("count", tuple(ids))) or 3
        ),
        rebuild_page_table=lambda ids: events.append(("rebuild", tuple(ids))),
        _gpu_page_table_manager=None,
    )


def test_two_page_allocation_runs_keys_allocate_rebuild_load_mark_rebuild():
    """The allocation order the KV identity guard depends on.

    The host-identity keys are previewed BEFORE the allocation, the rows are
    granted only after it succeeds, the host load runs against the page table
    this batch just rebuilt, and the final rebuild covers every resident row.
    """
    ns, engine, events = d64()
    manager = ready_pool(events)
    # ``u9`` is already resident AND bound in the local table, so the final
    # rebuild covers it as well as this batch's two rows.
    rows = {1: "u1", 2: "u2", 9: "u9"}
    sequences = {"u1": alloc_row(10), "u2": alloc_row(11), "u9": alloc_row(9)}
    resident = _ResidentSet({"u9"})
    ctx = allocate_context(
        ns, manager=manager, rows=rows, sequences=sequences, resident=resident,
    )
    engine._host_identity_page_keys = (
        lambda c, mgr, ids, tokens: events.append(("keys", tuple(ids), tuple(tokens))) or "K"
    )
    engine._allocate_gpu_pages_with_host_identity = (
        lambda c, mgr, ids, tokens: events.append(("allocate", tuple(ids)))
    )
    engine._load_host_kv_to_gpu = (
        lambda c, mgr, ids: events.append(("load", tuple(ids)))
    )
    assert engine._allocate_gpu_kv_two_page_buffer(ctx, [1, 2]) is True
    assert ordered(events) == [
        ("keys", (10, 11), (128, 128)),
        ("count", (10, 11)),
        ("allocate", (10, 11)),
        ("rebuild", (10, 11)),
        ("load", (10, 11)),
        ("rebuild", (9, 10, 11)),
    ]
    assert sequences["u1"].granted == [(2, True)]
    assert resident == {"u1", "u2", "u9"}


def test_two_page_allocation_votes_before_it_reraises_a_key_failure():
    ns, engine, events = d64()
    manager = ready_pool(events)
    ctx = allocate_context(
        ns, manager=manager, rows={1: "u1"}, sequences={"u1": alloc_row(10)},
    )
    engine._host_identity_page_keys = (
        lambda *a: events.append("keys") or (_ for _ in ()).throw(ValueError("boom"))
    )
    engine._collective_gpu_kv_allocation_preflight = (
        lambda c, req, fr: events.append(("preflight", req, fr))
    )
    with pytest.raises(ValueError, match="boom"):
        engine._allocate_gpu_kv_two_page_buffer(
            ctx, [1], collective_preflight=True,
        )
    # The failed rank joined the collective as the losing vote, then raised.
    assert events == ["keys", ("preflight", 1, 0)]


def test_two_page_allocation_refuses_a_demand_over_the_free_pages():
    ns, engine, events = d64()
    manager = ready_pool(events, free=1)
    ctx = allocate_context(
        ns, manager=manager, rows={1: "u1"}, sequences={"u1": alloc_row(10)},
    )
    engine._host_identity_page_keys = lambda *a: "K"
    engine._allocate_gpu_pages_with_host_identity = (
        lambda *a: events.append("allocate")
    )
    assert engine._allocate_gpu_kv_two_page_buffer(ctx, [1]) is False
    assert "allocate" not in events


def test_two_page_allocation_reports_an_insufficient_page_runtime_error():
    ns, engine, events = d64()
    manager = ready_pool(events)
    ctx = allocate_context(
        ns, manager=manager, rows={1: "u1"}, sequences={"u1": alloc_row(10)},
    )
    engine._host_identity_page_keys = lambda *a: None
    engine._allocate_gpu_pages_with_host_identity = (
        lambda *a: (_ for _ in ()).throw(RuntimeError("Insufficient free pages"))
    )
    assert engine._allocate_gpu_kv_two_page_buffer(ctx, [1]) is False
    # Any OTHER allocation error is a defect and propagates.
    engine._allocate_gpu_pages_with_host_identity = (
        lambda *a: (_ for _ in ()).throw(RuntimeError("page table corrupt"))
    )
    with pytest.raises(RuntimeError, match="page table corrupt"):
        engine._allocate_gpu_kv_two_page_buffer(ctx, [1])


# ---- D6.4 delegators -------------------------------------------------------


D64_HELPERS = (
    "_decode_phase_context", "_deferred_kv_flush_context",
    "_pending_kv_append_tensor_list",
)


def test_flush_delegator_shares_the_channel_and_resets_in_the_old_order(
    monkeypatch,
):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, D64_HELPERS)
    worker._kv_capacity_engine = engine
    worker._deferred_kv_entries = [("entry",)]
    worker._deferred_kv_batch = ([1], [2])
    worker._deferred_kv_worker_view = "view"
    worker._pending_kv_append_tasks = []
    seen = []

    def flush(ctx, tasks, tensors, **helpers):
        seen.append((ctx, tasks, tensors, sorted(helpers)))
        # The list the engine appends into is the worker's own, created by the
        # delegator so the later drain clears the very same object.
        tensors.append("retained")
        assert tensors is worker._pending_kv_append_tensors
        return ns["DeferredKVFlushOutcome"](
            _deferred_kv_entries=[], _deferred_kv_entries_aux=[],
            _deferred_kv_batch=None, _deferred_kv_worker_view=None,
            _deferred_kv_worker_view_aux=None,
        )

    engine._flush_deferred_kv_to_host = flush
    assert worker._flush_deferred_kv_to_host() is None
    ctx, tasks, tensors, helpers = seen[0]
    assert helpers == ["_ensure_host_kv_append_capacity"]
    assert tasks is worker._pending_kv_append_tasks
    assert worker._pending_kv_append_tensors == ["retained"]
    # The context carries the forward's channel with the pre-move defaults.
    assert ctx.deferred_entries == [("entry",)]
    assert ctx.deferred_batch == ([1], [2])
    assert ctx.deferred_worker_view == "view"
    assert ctx.deferred_entries_aux == []
    assert ctx.deferred_worker_view_aux is None
    # All five resets applied.
    assert worker._deferred_kv_entries == []
    assert worker._deferred_kv_batch is None
    assert worker._deferred_kv_worker_view is None
    assert worker._deferred_kv_worker_view_aux is None
    # A second flush reuses the SAME tensor list, never a fresh one.
    held = worker._pending_kv_append_tensors
    worker._flush_deferred_kv_to_host()
    assert seen[-1][2] is held


def test_flush_delegator_leaves_the_untouched_fields_alone(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, D64_HELPERS)
    worker._kv_capacity_engine = engine
    worker._deferred_kv_batch = ([3], [4])
    worker._deferred_kv_worker_view = "view"
    worker._pending_kv_append_tasks = []
    engine._flush_deferred_kv_to_host = lambda *a, **kw: (
        ns["DeferredKVFlushOutcome"](
            _deferred_kv_entries=[], _deferred_kv_entries_aux=[],
        )
    )
    worker._flush_deferred_kv_to_host()
    assert worker._deferred_kv_entries == []
    # The three sentinel fields are not bound, so the forward's handoff stands.
    assert worker._deferred_kv_batch == ([3], [4])
    assert worker._deferred_kv_worker_view == "view"
    assert not hasattr(worker, "_deferred_kv_worker_view_aux")


def test_aux_slot_delegator_caches_only_what_the_outcome_carries(monkeypatch):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, D64_HELPERS)
    worker._kv_capacity_engine = engine
    worker.gpu_paged_kv_cache_manager = "coord"
    seen = []

    def resolve(ctx):
        seen.append(ctx)
        return frozenset({2}), ns["AuxLayerSlotsOutcome"](
            _aux_layers_with_slots_cache=frozenset({2}),
        )

    engine._aux_layers_with_slots = resolve
    assert worker._aux_layers_with_slots() == frozenset({2})
    # The context is the narrow pair, with the pre-move getattr defaults.
    assert seen[0].aux_coordinator == "coord"
    assert seen[0]._aux_layers_with_slots_cache is False
    assert worker._aux_layers_with_slots_cache == frozenset({2})
    # The non-caching exits leave the memo exactly as it was.
    engine._aux_layers_with_slots = lambda ctx: (
        seen.append(ctx) or (None, ns["AuxLayerSlotsOutcome"]())
    )
    assert worker._aux_layers_with_slots() is None
    assert worker._aux_layers_with_slots_cache == frozenset({2})
    assert seen[-1]._aux_layers_with_slots_cache == frozenset({2})


def test_aux_slot_delegator_works_without_a_coordinator_attribute(monkeypatch):
    """The pre-move body tolerated a worker with no GPU KV manager bound.

    The narrow context keeps that: resolving the full decode phase context
    here would raise on a startup-time caller instead of reporting "no aux".
    """
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, D64_HELPERS)
    worker._kv_capacity_engine = engine
    seen = []
    engine._aux_layers_with_slots = lambda ctx: (
        seen.append(ctx) or (None, ns["AuxLayerSlotsOutcome"]())
    )
    assert worker._aux_layers_with_slots() is None
    assert seen[0].aux_coordinator is None


def test_aux_append_delegator_hands_over_the_channel_and_collaborators(
    monkeypatch,
):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, D64_HELPERS)
    worker._kv_capacity_engine = engine
    worker._held_decode_phase_context = object()
    worker._pending_kv_append_tasks = ["t"]
    # The drain is a worker method of its own, outside this stage's set.
    worker._wait_pending_kv_append_tasks = lambda **kwargs: 0
    seen = []
    engine._append_decode_kv_to_host_aux_async = (
        lambda ctx, tasks, tensors, *args, **helpers:
        seen.append((ctx, tasks, tensors, args, sorted(helpers)))
    )
    worker._append_decode_kv_to_host_aux_async(5, [0], "k", "v", "s")
    ctx, tasks, tensors, args, helpers = seen[0]
    assert ctx is worker._held_decode_phase_context
    assert tasks is worker._pending_kv_append_tasks
    assert tensors is worker._pending_kv_append_tensors
    assert args == (5, [0], "k", "v", "s")
    assert helpers == [
        "_aux_layers_with_slots", "_wait_pending_kv_append_tasks",
    ]


def test_capacity_load_and_allocate_delegators_pass_the_phase_context(
    monkeypatch,
):
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, D64_HELPERS)
    worker._kv_capacity_engine = engine
    worker._held_decode_phase_context = object()
    seen = []
    engine._ensure_host_kv_append_capacity = (
        lambda ctx, ids, lengths: seen.append(("capacity", ctx, ids, lengths))
    )
    engine._load_host_kv_to_gpu = (
        lambda ctx, mgr, ids: seen.append(("load", ctx, mgr, ids))
    )
    engine._allocate_gpu_kv_two_page_buffer = (
        lambda ctx, rows, load, preflight:
        seen.append(("allocate", ctx, rows, load, preflight))
    )
    worker._ensure_host_kv_append_capacity([1], [2])
    worker._load_host_kv_to_gpu("mgr", [10])
    worker._allocate_gpu_kv_two_page_buffer([3], False, True)
    assert [row[0] for row in seen] == ["capacity", "load", "allocate"]
    for row in seen:
        assert row[1] is worker._held_decode_phase_context
    assert seen[0][2:] == ([1], [2])
    assert seen[1][2:] == ("mgr", [10])
    assert seen[2][2:] == ([3], False, True)


def test_integrate_delegator_injects_the_finalize_collaborator(monkeypatch):
    """The finalize arrives as the WORKER's method, not the engine sibling.

    Its delegator carries the load-tensor holder across the call and stamps
    the decode row epoch afterwards, so the engine must reach it through the
    worker or both effects are silently dropped.
    """
    ns, engine, _ = runtime()
    worker = load_delegators(monkeypatch, ns, D64_HELPERS)
    worker._kv_capacity_engine = engine
    worker._held_decode_phase_context = object()
    worker._finalize_async_load_minimal = lambda *args: (["u"], [0])
    seen = []
    engine._integrate_pending_async_load = (
        lambda ctx, *args, **helpers:
        seen.append((ctx, args, helpers)) or (["u"], [0])
    )
    assert worker._integrate_pending_async_load(
        "task", ["u0"], [0], [10], ["u1"], [1], "mgr",
    ) == (["u"], [0])
    ctx, args, helpers = seen[0]
    assert ctx is worker._held_decode_phase_context
    assert args == ("task", ["u0"], [0], [10], ["u1"], [1], "mgr")
    assert helpers == {
        "_finalize_async_load_minimal": worker._finalize_async_load_minimal,
    }
