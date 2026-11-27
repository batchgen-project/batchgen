"""The prefill engine's readers and predicates run from frozen values.

D7.1, the first stage of this engine's conversion: fourteen small operations
stop receiving the worker. Each takes a frozen plain-value context (or, where
it reads no worker state at all, only the launch value its delegator resolves)
plus explicit keyword-only collaborator callables. Nothing here builds a
stand-in worker: every body is driven by the context it declares.

The bodies are compiled out of the engine's own AST
(``tests/prefill_source.py``), because importing the engine pulls in the
compiled core engine and UCX; the delegators are compiled out of the real
worker class the same way the KV capacity tests do it.
"""

import ast
from dataclasses import FrozenInstanceError, fields
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.prefill_source import (
    METHODS,
    STATELESS_METHODS,
    engine_class_node,
    engine_method_names,
    load_prefill,
    method_node,
)
from tests.worker_method_source import WORKER_SOURCE


# The four readers that share the PSM context, and the two probe operations
# that share the probe context: the two coincidences D7.1 collapsed into one
# dataclass each.
PSM_READERS = (
    "_pool_prefill_group_safe", "_prefill_pool_group",
    "_prefill_pool_psm_layout", "_prefill_pool_page_dims",
)
PROBE_READERS = ("_pool_probe_debug", "_pool_pad_poison_enabled")

# Every worker delegator this stage converted, plus the two context builders
# they share.
DELEGATORS = tuple(sorted(METHODS))
DELEGATOR_HELPERS = ("_prefill_pool_parallel_context", "_pool_probe_context")


def runtime(**namespace):
    """The converted bodies and a stateless engine instance to drive them."""
    ns = load_prefill(namespace)
    return ns, ns["PrefillEngine"]()


class Explode:
    """An object whose every attribute read is a test failure.

    Put in a context field a branch must not reach, it turns "this read is
    lazy" into an assertion instead of a comment.
    """

    def __init__(self, name="field"):
        self._name = name

    def __getattr__(self, attr):
        raise AssertionError(f"{self._name}.{attr} was read")


def psm(**declarations):
    """A parallel-strategy manager that declares exactly ``declarations``."""
    return SimpleNamespace(**declarations)


def wave(*, uuids=("a", "b"), segments=(), page_tokens=4, chunks=()):
    return SimpleNamespace(
        uuids=list(uuids),
        page_tokens=page_tokens,
        plan=SimpleNamespace(segments=tuple(segments), chunks=tuple(chunks)),
        probe_debug=None,
        probe_debug_read=False,
        computed_segments=lambda: tuple(segments),
    )


# --------------------------------------------------------------------------- #
#  The seam itself
# --------------------------------------------------------------------------- #
_CONSUMING_CALLS = {
    "sum", "any", "all", "min", "max", "len", "tuple", "list", "set", "dict",
    "sorted", "zip",
}


@pytest.mark.parametrize("name", sorted(METHODS))
def test_converted_methods_have_no_worker_or_rebinding(name):
    node = method_node(name)
    args = [arg.arg for arg in node.args.args]
    if name in STATELESS_METHODS:
        # No seam parameter at all: the body reads no worker state, so an
        # empty context would claim a dependency it does not have.
        assert args[0] == "self"
        assert "ctx" not in args
    else:
        assert args[:2] == ["self", "ctx"]
    assert not any(isinstance(n, ast.arg) and n.arg == "worker"
                   for n in ast.walk(node))
    assert not any(isinstance(n, ast.Name) and n.id == "worker"
                   for stmt in node.body for n in ast.walk(stmt))
    # ``self`` is the stateless engine, never a rebound worker: it is a store
    # target nowhere, and it is loaded only to call a sibling engine method
    # that takes the SAME context.
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
    # The seam is a plain function: it never yields, and no generator crosses
    # it. A generator expression consumed on the spot by a reduction builtin is
    # arithmetic the verbatim move carried with it.
    assert not any(isinstance(n, (ast.Yield, ast.YieldFrom))
                   for n in ast.walk(node))
    consumed = {
        id(arg) for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id in _CONSUMING_CALLS
        for arg in call.args
    }
    for n in ast.walk(node):
        if isinstance(n, ast.GeneratorExp):
            assert id(n) in consumed, ast.unparse(n)


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_collaborator_is_keyword_only(name):
    """A collaborator callable is passed by name, never positionally.

    The convention is the one D6 settled: a parameter named after a worker
    method IS a collaborator, and it is keyword-only so a call site cannot
    silently slide a context into it.
    """
    node = method_node(name)
    worker_shaped = engine_method_names() | {"_debug_flag_enabled"}
    for arg in node.args.args:
        assert arg.arg not in worker_shaped, arg.arg
    for arg in node.args.kwonlyargs:
        if arg.arg.startswith("_"):
            assert arg.arg in worker_shaped, arg.arg


def test_no_context_field_holds_a_collaborator():
    """Callables never go inside a dataclass.

    A field named after a worker method would be a bound method travelling in
    the context, which is the shape this phase exists to remove; a ``Callable``
    annotation would be the same thing under another name.
    """
    ns, _ = runtime()
    engine_methods = engine_method_names() | {"_debug_flag_enabled"}
    for name, value in sorted(ns.items()):
        if not (isinstance(value, type) and name.endswith("Context")):
            continue
        for field in fields(value):
            assert field.name not in engine_methods, (name, field.name)
            assert "Callable" not in str(field.type), (name, field.name)


def test_every_context_is_frozen():
    ns, _ = runtime()
    contexts = [
        value for name, value in sorted(ns.items())
        if isinstance(value, type) and name.endswith("Context")
    ]
    assert len(contexts) == 8
    for context in contexts:
        instance = context(**{field.name: None for field in fields(context)})
        with pytest.raises(FrozenInstanceError):
            setattr(instance, fields(context)[0].name, object())


def test_the_psm_readers_share_one_context_because_they_read_one_object():
    """Four readers, one field: the PSM handle is their whole read set."""
    ns, _ = runtime()
    context = ns["PrefillPoolParallelContext"]
    assert [field.name for field in fields(context)] == ["parallel_manager"]
    for name in PSM_READERS:
        node = method_node(name)
        annotation = node.args.args[1].annotation
        assert ast.unparse(annotation) == "PrefillPoolParallelContext", name


def test_the_probe_pair_shares_one_context_because_their_reads_coincide():
    ns, _ = runtime()
    context = ns["PoolProbeContext"]
    assert {field.name for field in fields(context)} == {
        "rank", "_phase_prefill_debug",
    }
    for name in PROBE_READERS:
        node = method_node(name)
        assert ast.unparse(node.args.args[1].annotation) == "PoolProbeContext"


def test_the_stage_converted_exactly_its_own_methods():
    """The file still holds worker-shaped operations: this is stage one.

    The later stages of this engine convert the rest, so the guard that the
    whole FILE names no worker belongs to the last one, not here.
    """
    converted, worker_shaped = set(), set()
    for node in engine_class_node().body:
        if not isinstance(node, ast.FunctionDef):
            continue
        args = [arg.arg for arg in node.args.args]
        if len(args) >= 2 and args[1] == "worker":
            worker_shaped.add(node.name)
        elif len(args) >= 2 and args[1] == "ctx" or node.name in STATELESS_METHODS:
            converted.add(node.name)
    assert converted == METHODS
    assert not worker_shaped & METHODS
    assert worker_shaped, "nothing left to convert, but D7.1 is not the last stage"


# --------------------------------------------------------------------------- #
#  The PSM readers
# --------------------------------------------------------------------------- #
def psm_context(ns, manager):
    return ns["PrefillPoolParallelContext"](parallel_manager=manager)


@pytest.mark.parametrize("declared,expected", [(True, True), (False, False)])
def test_group_pool_safety_is_what_the_psm_declares(declared, expected):
    ns, engine = runtime()
    assert engine._pool_prefill_group_safe(
        psm_context(ns, psm(supports_group_pool_prefill=lambda: declared))
    ) is expected


def test_a_psm_that_declares_nothing_is_not_group_pool_safe():
    ns, engine = runtime()
    assert engine._pool_prefill_group_safe(psm_context(ns, psm())) is False


def test_the_pool_group_is_none_at_one_rank_per_group():
    ns, engine = runtime()
    # The PSM is not read at all at G==1.
    assert engine._prefill_pool_group(
        psm_context(ns, Explode("parallel_manager")), attn_group_size=1,
    ) is None


def test_the_pool_group_is_the_attention_tp_group_above_one():
    ns, engine = runtime()
    assert engine._prefill_pool_group(
        psm_context(ns, psm(attn_tp_group="pg")), attn_group_size=2,
    ) == "pg"


def test_a_grouped_pool_without_an_attention_group_is_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="needs the attention TP process group"):
        engine._prefill_pool_group(
            psm_context(ns, psm()), attn_group_size=4,
        )


def test_the_psm_layout_override_is_called_once_per_read():
    ns, engine = runtime()
    calls = []
    manager = psm(prefill_pool_layout=lambda: calls.append(1) or {"num_layers": 2})
    assert engine._prefill_pool_psm_layout(psm_context(ns, manager)) == {
        "num_layers": 2,
    }
    assert calls == [1]


def test_a_psm_without_an_override_leaves_the_host_geometry_in_place():
    ns, engine = runtime()
    assert engine._prefill_pool_psm_layout(psm_context(ns, psm())) is None


GEOMETRY = SimpleNamespace(
    num_layers=3, num_k_heads=1, k_head_dim=8, kv_dtype=torch.float32,
)
K3_LAYOUT = {
    "num_layers": 2, "num_kv_heads": 2, "k_head_dim": 6, "v_head_dim": 4,
}


def test_the_page_dims_follow_the_psm_override_through_its_sibling():
    ns, engine = runtime()
    calls = []
    manager = psm(
        prefill_pool_layout=lambda: calls.append(1) or dict(K3_LAYOUT),
    )
    assert engine._prefill_pool_page_dims(
        psm_context(ns, manager), GEOMETRY,
    ) == (2, 2, 6, 4)
    # Reached the engine sibling with the SAME context: one override read.
    assert calls == [1]


def test_the_page_dims_fall_back_to_the_host_geometry():
    """Without an override the V width is the Host's K width, as stored."""
    ns, engine = runtime()
    assert engine._prefill_pool_page_dims(
        psm_context(ns, psm()), GEOMETRY,
    ) == (3, 1, 8, 8)


def test_an_empty_psm_layout_is_not_an_override():
    ns, engine = runtime()
    assert engine._prefill_pool_page_dims(
        psm_context(ns, psm(prefill_pool_layout=lambda: {})), GEOMETRY,
    ) == (3, 1, 8, 8)


# --------------------------------------------------------------------------- #
#  The page-byte arithmetic
# --------------------------------------------------------------------------- #
def page_bytes_context(ns, *, page_tokens=4, ckpt="org/model"):
    return ns["PrefillPoolPageBytesContext"](
        prefix_runtime=SimpleNamespace(
            group_specs=(SimpleNamespace(raw_page_tokens=page_tokens),),
        ),
        huggingface_ckpt_name=ckpt,
    )


def geometry_module(monkeypatch, geometry=GEOMETRY, calls=None):
    """Serve the body's own import of the Host geometry builder."""
    def build(ckpt, page_tokens):
        if calls is not None:
            calls.append((ckpt, tuple(page_tokens)))
        return geometry
    monkeypatch.setitem(
        sys.modules, "batchgen.kv_cache.host_kv_mananger_config",
        SimpleNamespace(build_gpu_kv_config=build),
    )


@pytest.mark.parametrize("with_v", [True, False])
def test_the_page_byte_count_is_the_layout_identity(monkeypatch, with_v):
    ns, engine = runtime()
    calls = []
    geometry_module(monkeypatch, calls=calls)
    assert engine._prefill_pool_page_bytes(
        page_bytes_context(ns, page_tokens=4),
        _prefill_pool_layout=lambda geometry: (with_v, (0, 1), 32),
        _prefill_pool_page_dims=lambda geometry: (2, 2, 6, 4),
    ) == (2 * 4 * 2 * (6 + (4 if with_v else 0)) + 2 * 4 * 32) * 4
    # The ckpt and the page tokens travelled from the context.
    assert calls == [("org/model", (4,))]


def test_the_page_byte_count_asks_both_collaborators_for_this_geometry(
    monkeypatch,
):
    ns, engine = runtime()
    geometry_module(monkeypatch)
    seen = []
    engine._prefill_pool_page_bytes(
        page_bytes_context(ns),
        _prefill_pool_layout=lambda geometry: seen.append(("layout", geometry))
        or (False, (), 0),
        _prefill_pool_page_dims=lambda geometry: seen.append(("dims", geometry))
        or (1, 1, 1, 1),
    )
    assert seen == [("layout", GEOMETRY), ("dims", GEOMETRY)]


# --------------------------------------------------------------------------- #
#  The recurrent-state slots and the pooled-segment ids
# --------------------------------------------------------------------------- #
def test_a_model_with_no_recurrent_state_reports_minus_one():
    ns, engine = runtime()
    context = ns["PrefillPoolStateSlotsContext"](
        traj=SimpleNamespace(kda_state_manager=lambda: None),
    )
    assert engine._prefill_pool_free_state_slots(context) == -1


def test_the_free_state_slots_are_the_managers_own_count():
    ns, engine = runtime()
    manager = SimpleNamespace(
        get_stats=lambda: SimpleNamespace(num_free_state_items=7.0),
    )
    context = ns["PrefillPoolStateSlotsContext"](
        traj=SimpleNamespace(kda_state_manager=lambda: manager),
    )
    slots = engine._prefill_pool_free_state_slots(context)
    assert slots == 7 and isinstance(slots, int)


def test_the_next_segment_id_comes_from_the_prefix_reuse_owner():
    ns, engine = runtime()
    issued = []
    context = ns["PoolSegmentSidContext"](
        prefix=SimpleNamespace(
            next_pool_segment_sid=lambda: issued.append(1) or -17,
        ),
    )
    assert engine._next_pool_segment_sid(context) == -17
    assert issued == [1]


# --------------------------------------------------------------------------- #
#  The two launch-value predicates (no context at all)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,expected", [
    ("packing_v2", True), ("global_idx", False),
])
def test_the_packing_arm_is_the_launch_mode(mode, expected):
    _, engine = runtime()
    assert engine._prefix_rank_packing_mode([], mode=mode) is expected


@pytest.mark.parametrize("mode", ["", "packing", "PACKING_V2", None, True])
def test_a_mode_the_launcher_cannot_produce_is_refused(mode):
    _, engine = runtime()
    with pytest.raises(ValueError, match="prefix-rank-assignment-mode"):
        engine._prefix_rank_packing_mode([], mode=mode)


@pytest.mark.parametrize("value,expected", [("on", False), ("off", True)])
def test_the_cost_profile_arm_is_the_launch_value(value, expected):
    _, engine = runtime()
    assert engine._prefix_rank_cost_profile_disabled([], value=value) is expected


@pytest.mark.parametrize("value", ["", "OFF", " off ", None, 0])
def test_a_cost_profile_value_the_launcher_cannot_produce_is_refused(value):
    _, engine = runtime()
    with pytest.raises(ValueError, match="prefix-rank-cost-profile"):
        engine._prefix_rank_cost_profile_disabled([], value=value)


# --------------------------------------------------------------------------- #
#  Is the in-wave prefix pool on
# --------------------------------------------------------------------------- #
def enabled_context(
    ns, *, workspace=1 << 20, cache=True, integrity=False, manager=None,
    semantics=None,
):
    from batchgen.prefix_reuse.config import PrefixKVGroupSemantic

    specs = tuple(
        SimpleNamespace(semantic=semantic)
        for semantic in (semantics or (PrefixKVGroupSemantic.FULL_KV,))
    )
    return ns["PrefillPrefixPoolEnabledContext"](
        parallel_manager=psm() if manager is None else manager,
        engine_config=SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            prefill_prefix_pool_workspace_bytes=workspace,
        )),
        enable_prefix_cache=cache,
        prefix_runtime=SimpleNamespace(group_specs=specs),
        _prefix_integrity=integrity,
    )


def refuse():
    return lambda: pytest.fail("the group-safety declaration was read")


def test_the_pool_is_off_without_prefix_caching():
    ns, engine = runtime()
    # Decided before the group specs are read: the runtime config may not even
    # be resolved on a server with prefix caching off.
    context = ns["PrefillPrefixPoolEnabledContext"](
        parallel_manager=Explode("parallel_manager"),
        engine_config=SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            prefill_prefix_pool_workspace_bytes=1 << 20,
        )),
        enable_prefix_cache=False,
        prefix_runtime=Explode("prefix_runtime"),
        _prefix_integrity=False,
    )
    assert engine._prefill_prefix_pool_enabled(
        context, attn_group_size=1, _pool_prefill_group_safe=refuse(),
    ) is False


@pytest.mark.parametrize("workspace,integrity", [(0, False), (1 << 20, True)])
def test_the_pool_is_off_without_a_workspace_or_under_the_integrity_check(
    workspace, integrity,
):
    ns, engine = runtime()
    assert engine._prefill_prefix_pool_enabled(
        enabled_context(ns, workspace=workspace, integrity=integrity),
        attn_group_size=1, _pool_prefill_group_safe=refuse(),
    ) is False


@pytest.mark.parametrize("workspace,integrity", [(0, False), (1 << 20, True)])
def test_mla_latent_prefix_reuse_without_the_pool_is_refused(
    workspace, integrity,
):
    """GLM DSA and Kimi-K3 have no second path: this is a hard error."""
    from batchgen.prefix_reuse.config import PrefixKVGroupSemantic

    ns, engine = runtime()
    latent = next(
        semantic for semantic in PrefixKVGroupSemantic
        if semantic is not PrefixKVGroupSemantic.FULL_KV
    )
    with pytest.raises(RuntimeError, match="MLA-latent prefix reuse"):
        engine._prefill_prefix_pool_enabled(
            enabled_context(
                ns, workspace=workspace, integrity=integrity,
                semantics=(PrefixKVGroupSemantic.FULL_KV, latent),
            ),
            attn_group_size=1, _pool_prefill_group_safe=refuse(),
        )


def test_a_resident_ep_prefill_cannot_run_the_pool():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="runs per DP rank"):
        engine._prefill_prefix_pool_enabled(
            enabled_context(
                ns, manager=psm(prefill_uses_resident_ep=lambda: True),
            ),
            attn_group_size=1, _pool_prefill_group_safe=refuse(),
        )


def test_a_pure_dp_prefill_at_one_rank_per_group_runs_the_pool():
    ns, engine = runtime()
    assert engine._prefill_prefix_pool_enabled(
        enabled_context(ns, manager=psm(prefill_uses_resident_ep=lambda: False)),
        attn_group_size=1,
        # Short-circuited: at G==1 the group declaration is never consulted.
        _pool_prefill_group_safe=refuse(),
    ) is True


@pytest.mark.parametrize("safe,raises", [(True, False), (False, True)])
def test_a_grouped_prefill_runs_the_pool_only_where_the_psm_declares_it(
    safe, raises,
):
    ns, engine = runtime()
    asked = []

    def declare():
        asked.append(1)
        return safe

    call = lambda: engine._prefill_prefix_pool_enabled(
        enabled_context(ns), attn_group_size=8,
        _pool_prefill_group_safe=declare,
    )
    if raises:
        with pytest.raises(RuntimeError, match="runs per DP rank"):
            call()
    else:
        assert call() is True
    assert asked == [1]


# --------------------------------------------------------------------------- #
#  The model-pass count
# --------------------------------------------------------------------------- #
def pass_count_context(
    ns, *, pool_wave=None, prepack=True, micro_batch=4, sequences=None,
    prefix_states=None, local_to_uuid=None,
):
    return ns["PrefillModelPassCountContext"](
        _prefill_pool_wave=pool_wave,
        enable_prepack=prepack,
        engine_config=SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            MoE_prefill_micro_batch_size=micro_batch,
        )),
        pool=SimpleNamespace(get_sequence=(sequences or {}).get),
        local_to_uuid=local_to_uuid or {},
        prefix=SimpleNamespace(
            prefix_sequence_states=prefix_states or {},
        ),
    )


def test_an_empty_batch_runs_no_forward_and_reads_nothing_else():
    ns, engine = runtime()
    context = ns["PrefillModelPassCountContext"](
        _prefill_pool_wave=None,
        enable_prepack=Explode("enable_prepack"),
        engine_config=Explode("engine_config"),
        pool=Explode("pool"),
        local_to_uuid=Explode("local_to_uuid"),
        prefix=Explode("prefix"),
    )
    assert engine._prefill_model_pass_count(
        context, [],
        _plan_prefill_micro_batches=lambda *a: pytest.fail("planned"),
    ) == 0


def test_a_pooled_wave_runs_one_forward_per_planned_chunk():
    ns, engine = runtime()
    context = pass_count_context(
        ns, pool_wave=wave(chunks=("c0", "c1", "c2")),
    )
    assert engine._prefill_model_pass_count(
        context, [0, 1, 2, 3, 4],
        _plan_prefill_micro_batches=lambda *a: pytest.fail("planned"),
    ) == 3


@pytest.mark.parametrize("rows,size,expected", [
    (1, 4, 1), (4, 4, 1), (5, 4, 2), (9, 4, 3),
])
def test_without_prepack_the_count_is_the_micro_batch_ceiling(
    rows, size, expected,
):
    ns, engine = runtime()
    context = pass_count_context(ns, prepack=False, micro_batch=size)
    assert engine._prefill_model_pass_count(
        context, list(range(rows)),
        _plan_prefill_micro_batches=lambda *a: pytest.fail("planned"),
    ) == expected
    assert expected == math.ceil(rows / size)


def test_with_prepack_the_planner_is_asked_with_the_suffix_lengths():
    ns, engine = runtime()
    sequences = {
        "a": SimpleNamespace(global_idx=10, prompt_length=100),
        "b": SimpleNamespace(global_idx=11, prompt_length=60),
    }
    context = pass_count_context(
        ns,
        sequences=sequences,
        local_to_uuid={0: "a", 1: "b"},
        prefix_states={10: SimpleNamespace(compute_cached_tokens=40)},
    )
    asked = []

    def plan(computed, full, cached):
        asked.append((list(computed), list(full), list(cached)))
        return ["m0", "m1"], None, None

    assert engine._prefill_model_pass_count(
        context, [0, 1], _plan_prefill_micro_batches=plan,
    ) == 2
    # A row with no prefix state contributes zero cached tokens, and the
    # computed length is the prompt minus what is cached.
    assert asked == [([60, 60], [100, 60], [40, 0])]


# --------------------------------------------------------------------------- #
#  The wave's segment host pages
# --------------------------------------------------------------------------- #
def test_no_wave_allocates_no_segment_pages():
    ns, engine = runtime()
    context = ns["PrefillPoolWaveContext"](_prefill_pool_wave=None)
    assert engine._prefill_pool_segment_host_pages(context) == 0


def test_the_segment_pages_are_the_computed_tokens_in_pages():
    ns, engine = runtime()
    segments = (
        SimpleNamespace(tokens=8), SimpleNamespace(tokens=12),
    )
    context = ns["PrefillPoolWaveContext"](
        _prefill_pool_wave=wave(segments=segments, page_tokens=4),
    )
    assert engine._prefill_pool_segment_host_pages(context) == 5


# --------------------------------------------------------------------------- #
#  The pooled probe seam
# --------------------------------------------------------------------------- #
class CountingDebug:
    """A debug state that counts key lookups, like the production dict."""

    def __init__(self, values):
        self._values = dict(values)
        self.lookups = 0

    def __bool__(self):
        return bool(self._values)

    def get(self, key, default=None):
        self.lookups += 1
        return self._values.get(key, default)


def probe_context(ns, debug=None, *, rank=3):
    return ns["PoolProbeContext"](
        rank=rank, _phase_prefill_debug=CountingDebug(debug or {}),
    )


def test_an_empty_debug_state_arms_nothing_and_looks_no_key_up():
    ns, engine = runtime()
    context = probe_context(ns)
    this_wave = wave()
    assert engine._pool_probe_debug(context, this_wave) is None
    assert context._phase_prefill_debug.lookups == 0
    # The answer is cached on the WAVE, in place.
    assert this_wave.probe_debug is None and this_wave.probe_debug_read is True


def test_only_the_pooled_probe_keys_are_adopted():
    ns, engine = runtime()
    context = probe_context(ns, {
        "prefix_pool_pad_poison": True,
        "prefix_pool_dump_dir": "/d",
        "glm5_dispatch_trace": True,
    })
    assert engine._pool_probe_debug(context, wave()) == {
        "prefix_pool_pad_poison": True, "prefix_pool_dump_dir": "/d",
    }
    assert context._phase_prefill_debug.lookups == len(
        ns["_POOL_PROBE_DEBUG_KEYS"]
    )


def test_a_cached_wave_answers_without_touching_the_debug_state():
    ns, engine = runtime()
    context = probe_context(ns, {"prefix_pool_offload_check": True})
    this_wave = wave()
    first = engine._pool_probe_debug(context, this_wave)
    after = context._phase_prefill_debug.lookups
    assert engine._pool_probe_debug(context, this_wave) is first
    assert context._phase_prefill_debug.lookups == after


def test_pad_poison_is_off_when_no_probe_is_armed():
    ns, engine = runtime()
    assert engine._pool_pad_poison_enabled(
        probe_context(ns), wave(),
        _debug_flag_enabled=lambda value: pytest.fail("the flag was parsed"),
    ) is False


def test_pad_poison_follows_the_flag_reader_on_an_armed_wave():
    from batchgen.prefix_reuse.executor import PAD_POISON_DEBUG_FLAG

    ns, engine = runtime()
    parsed = []

    def flag(value):
        parsed.append(value)
        return value == "on"

    armed = probe_context(ns, {PAD_POISON_DEBUG_FLAG: "on"})
    assert engine._pool_pad_poison_enabled(
        armed, wave(), _debug_flag_enabled=flag,
    ) is True
    off = probe_context(ns, {PAD_POISON_DEBUG_FLAG: "no"})
    assert engine._pool_pad_poison_enabled(
        off, wave(), _debug_flag_enabled=flag,
    ) is False
    assert parsed == ["on", "no"]


def test_pad_poison_reads_the_probe_state_through_its_engine_sibling():
    """One read per wave for both operations, with the same context."""
    from batchgen.prefix_reuse.executor import PAD_POISON_DEBUG_FLAG

    ns, engine = runtime()
    context = probe_context(ns, {PAD_POISON_DEBUG_FLAG: True})
    this_wave = wave()
    assert engine._pool_pad_poison_enabled(
        context, this_wave, _debug_flag_enabled=bool,
    ) is True
    assert this_wave.probe_debug_read is True
    assert context._phase_prefill_debug.lookups == len(
        ns["_POOL_PROBE_DEBUG_KEYS"]
    )


# --------------------------------------------------------------------------- #
#  The worker delegators
# --------------------------------------------------------------------------- #
def load_delegators(monkeypatch, ns, extra=DELEGATOR_HELPERS):
    """Compile the actual worker class with only the delegators under test.

    ``extra`` names the worker's own context builders, which are its methods
    and not delegated bodies.
    """
    module = ModuleType("batchgen.worker.prefill_engine")
    module.__dict__.update(ns)
    monkeypatch.setitem(sys.modules, "batchgen.worker.prefill_engine", module)
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    wanted = set(DELEGATORS) | set(extra)
    cls.body = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in wanted]
    cls.bases, cls.decorator_list = [], []
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")],
                level=0,
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(code), str(WORKER_SOURCE), "exec"),
         namespace)
    worker = namespace["BatchGenWorker"]()
    worker._prefill_engine = ns["PrefillEngine"]()
    return worker


def recording(engine, name, capture):
    """Replace one engine operation with a recorder of its arguments."""
    def record(*args, **kwargs):
        capture.append((args, kwargs))
        return "recorded"
    setattr(engine, name, record)


def test_every_converted_delegator_still_exists_on_the_worker():
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    defined = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
    assert set(DELEGATORS) <= defined
    assert set(DELEGATOR_HELPERS) <= defined


def test_the_psm_delegators_build_the_shared_context_per_call(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.parallel_manager = "first"
    worker._prefill_attn_tp_size = lambda: 2
    seen = []
    for name in PSM_READERS:
        recording(engine, name, seen)
    worker._pool_prefill_group_safe()
    worker._prefill_pool_group()
    worker._prefill_pool_psm_layout()
    worker._prefill_pool_page_dims("geometry")
    # A rebind between calls is seen, because the context is resolved HERE.
    worker.parallel_manager = "second"
    worker._prefill_pool_psm_layout()
    contexts = [args[0] for args, _ in seen]
    assert all(isinstance(c, ns["PrefillPoolParallelContext"]) for c in contexts)
    assert [c.parallel_manager for c in contexts] == [
        "first", "first", "first", "first", "second",
    ]
    # The group width travels as a value beside the context.
    assert seen[1][1] == {"attn_group_size": 2}
    # The geometry stays a positional argument of the page dims.
    assert seen[3][0][1:] == ("geometry",)


def test_the_pool_predicate_delegator_injects_the_group_declaration(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.parallel_manager = "psm"
    worker.engine_config = "config"
    worker.enable_prefix_cache = True
    worker.prefix_cache_runtime_config = "runtime"
    worker._prefix_integrity = True
    worker._prefill_attn_tp_size = lambda: 4
    seen = []
    recording(engine, "_prefill_prefix_pool_enabled", seen)
    assert worker._prefill_prefix_pool_enabled() == "recorded"
    (context,), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPrefixPoolEnabledContext"])
    assert (context.parallel_manager, context.engine_config,
            context.enable_prefix_cache, context.prefix_runtime,
            context._prefix_integrity) == (
        "psm", "config", True, "runtime", True,
    )
    assert kwargs["attn_group_size"] == 4
    assert kwargs["_pool_prefill_group_safe"] == worker._pool_prefill_group_safe


def test_the_pool_predicate_delegator_keeps_the_integrity_default(monkeypatch):
    """The pre-move body read the latch with a getattr default, twice.

    A worker that never armed the integrity check carries no such attribute,
    and resolving it eagerly without that default would turn "absent" into an
    AttributeError -- which is the whole reason this context is narrow.
    """
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.parallel_manager = "psm"
    worker.engine_config = "config"
    worker.enable_prefix_cache = True
    worker.prefix_cache_runtime_config = "runtime"
    worker._prefill_attn_tp_size = lambda: 1
    assert not hasattr(worker, "_prefix_integrity")
    seen = []
    recording(engine, "_prefill_prefix_pool_enabled", seen)
    worker._prefill_prefix_pool_enabled()
    assert seen[0][0][0]._prefix_integrity is False


def test_the_page_byte_delegator_passes_both_sibling_delegators(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.prefix_cache_runtime_config = "runtime"
    worker.huggingface_ckpt_name = "org/model"
    worker._prefill_pool_layout = lambda geometry: None
    seen = []
    recording(engine, "_prefill_pool_page_bytes", seen)
    assert worker._prefill_pool_page_bytes() == "recorded"
    (context,), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPoolPageBytesContext"])
    assert (context.prefix_runtime, context.huggingface_ckpt_name) == (
        "runtime", "org/model",
    )
    assert kwargs == {
        "_prefill_pool_layout": worker._prefill_pool_layout,
        "_prefill_pool_page_dims": worker._prefill_pool_page_dims,
    }


def test_the_state_slot_and_segment_id_delegators_build_their_contexts(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker._traj = "trajectories"
    worker._prefix = "prefix state"
    seen = []
    recording(engine, "_prefill_pool_free_state_slots", seen)
    recording(engine, "_next_pool_segment_sid", seen)
    worker._prefill_pool_free_state_slots()
    worker._next_pool_segment_sid()
    slots, sid = seen[0][0][0], seen[1][0][0]
    assert isinstance(slots, ns["PrefillPoolStateSlotsContext"])
    assert slots.traj == "trajectories"
    assert isinstance(sid, ns["PoolSegmentSidContext"])
    assert sid.prefix == "prefix state"


def test_the_segment_page_delegator_reads_the_live_wave_each_call(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker._prefill_pool_wave = "first"
    seen = []
    recording(engine, "_prefill_pool_segment_host_pages", seen)
    worker._prefill_pool_segment_host_pages()
    worker._prefill_pool_wave = None
    worker._prefill_pool_segment_host_pages()
    assert [args[0]._prefill_pool_wave for args, _ in seen] == ["first", None]
    assert isinstance(seen[0][0][0], ns["PrefillPoolWaveContext"])


def test_the_pass_count_delegator_passes_the_planner_and_the_batch(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker._prefill_pool_wave = None
    worker.enable_prepack = True
    worker.engine_config = "config"
    worker.global_batch = "batch"
    worker._local_to_uuid_map = {0: "a"}
    worker._prefix = "prefix state"
    worker._plan_prefill_micro_batches = lambda *a, **k: ([], None, None)
    seen = []
    recording(engine, "_prefill_model_pass_count", seen)
    assert worker._prefill_model_pass_count([0]) == "recorded"
    (context, batch), kwargs = seen[0]
    assert isinstance(context, ns["PrefillModelPassCountContext"])
    assert (context._prefill_pool_wave, context.enable_prepack,
            context.engine_config, context.pool, context.local_to_uuid,
            context.prefix) == (
        None, True, "config", "batch", {0: "a"}, "prefix state",
    )
    assert batch == [0]
    assert kwargs == {
        "_plan_prefill_micro_batches": worker._plan_prefill_micro_batches,
    }


def test_the_probe_delegators_share_one_context_builder(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 5
    worker._phase_prefill_debug = {"prefix_pool_pad_poison": True}
    worker._debug_flag_enabled = lambda value: bool(value)
    seen = []
    recording(engine, "_pool_probe_debug", seen)
    recording(engine, "_pool_pad_poison_enabled", seen)
    worker._pool_probe_debug("wave")
    worker._pool_pad_poison_enabled("wave")
    for args, _ in seen:
        assert isinstance(args[0], ns["PoolProbeContext"])
        assert args[0].rank == 5
        assert args[0]._phase_prefill_debug == {"prefix_pool_pad_poison": True}
        assert args[1] == "wave"
    assert seen[0][1] == {}
    assert seen[1][1] == {"_debug_flag_enabled": worker._debug_flag_enabled}


def test_the_launch_value_delegators_pass_no_context_at_all(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.args = SimpleNamespace(
        prefix_rank_assignment_mode="global_idx",
        prefix_rank_cost_profile="off",
    )
    seen = []
    recording(engine, "_prefix_rank_packing_mode", seen)
    recording(engine, "_prefix_rank_cost_profile_disabled", seen)
    worker._prefix_rank_packing_mode(["seq"])
    worker._prefix_rank_cost_profile_disabled(["seq"])
    assert seen[0] == ((["seq"],), {"mode": "global_idx"})
    assert seen[1] == ((["seq"],), {"value": "off"})


def test_the_delegators_create_the_engine_once(monkeypatch):
    ns, _ = runtime()
    worker = load_delegators(monkeypatch, ns)
    del worker._prefill_engine
    worker._traj = SimpleNamespace(kda_state_manager=lambda: None)
    worker._prefix = SimpleNamespace(next_pool_segment_sid=lambda: -1)
    assert worker._prefill_pool_free_state_slots() == -1
    engine = worker._prefill_engine
    assert isinstance(engine, ns["PrefillEngine"])
    assert worker._next_pool_segment_sid() == -1
    assert worker._prefill_engine is engine
