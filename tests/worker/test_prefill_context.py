"""The prefill engine's converted operations run from frozen values.

D7.1, the first stage of this engine's conversion: fourteen small operations
stop receiving the worker. Each takes a frozen plain-value context (or, where
it reads no worker state at all, only the launch value its delegator resolves)
plus explicit keyword-only collaborator callables. Nothing here builds a
stand-in worker: every body is driven by the context it declares.

D7.2, the second stage: the five prefix-planning operations. Four of them read
the two owned stores (prefix reuse, trajectories) through their context and
change nothing in them but the one state installation the owner's own method
performs; the fifth, the cost-profile lookup, is this engine's first worker
WRITE handed back in an outcome, and the tests below pin its cache as
create-then-populate, on the delegator, with the second call hitting it.

D7.3, the third stage: pool construction and layout, and the stage with the
COLLECTIVES in it. Nothing is shared here -- no two of the nine read sets are
equal -- and three of the nine hand plain worker values back in outcomes. The
collective pins are the point of this stage's tests: one records the ordered
sequence of ``dist`` calls a run issues and asserts the exact expected order,
another asserts from the AST that each collective stands under the one
condition it already had (the process-group handle) and under nothing else.

D7.4, the fourth stage: wave planning and the pooled instrumentation. It
carries ONE collective, the wave planner's budget reduction, and the pins
below hold it in place three ways -- its enclosing condition, its statement
position, and the ordered sequence a real run issues, which also proves it
still precedes the plan-digest gather one level down. The planner is where
design clause C7 lives, so the budget arithmetic is pinned both as source
identities and by running the body: a wave is bounded by the compute-token
budget, never by how many sequences it holds. Three of the four probe
instruments SHARE D7.1's `PoolProbeContext`, because their read set is
exactly its two fields; the fourth reads the parallel manager instead. The
wave epilogue is the stage's one worker write and it CLEARS the planned wave.

D7.5, the fifth stage: the EXECUTION paths -- the padded forward, the one
shared model pass, the token write-back and the pooled wave driver. It is the
first prefill stage since D7.2 with no collective, no stream and no event in
any of its bodies, and the first with no outcome at all: every attribute store
it makes is an in-place mutation of an object a context carries. What it does
carry is host synchronisation, so the tests below pin each body's ordered
sequence of host-pulling calls and, for both write-backs, that the ONE device
copy stays outside the per-row loop -- a conversion that moved it inside would
fail no other assertion and would simply make prefill slower. Nothing is
shared: the token write-back's three fields are a strict subset of both
drivers', and the rejection is tested, because sharing would make the one call
both drivers end on resolve the model and the weight-copy pipeline.

D7.6, the sixth stage: ONE method, the unpooled prepacked driver. Its context
is wide (eighteen fields) and the tests below prove that is honest rather than
lazy -- the read set of every prefix of the body intersects the read set of
its suffix, so no cut yields two disjoint sets, and the declared fields are
exactly what the body reads. It carries no collective, no outcome, and the
file's ONE explicit stream wait, which is pinned three ways: its statement
position between the two timing samples, its complete absence of enclosing
conditions, and the ordered sequence of the body's seven host-pulling calls
around it. The transitional model-pass call D7.5 documented here is finished,
so the test that pinned the transitional spelling is replaced by one that
pins the converted spelling and the disappearance of the inline context.

The bodies are compiled out of the engine's own AST
(``tests/prefill_source.py``), because importing the engine pulls in the
compiled core engine and UCX; the delegators are compiled out of the real
worker class the same way the KV capacity tests do it.
"""

import ast
from dataclasses import dataclass, FrozenInstanceError, fields
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.prefill_source import (
    COLLECTIVE_METHODS,
    D7_1_METHODS,
    D7_2_METHODS,
    D7_3_METHODS,
    D7_4_METHODS,
    D7_5_METHODS,
    D7_6_METHODS,
    METHODS,
    OUTCOME_METHODS,
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

# Every worker delegator these stages converted, plus the two context builders
# they share.
DELEGATORS = tuple(sorted(METHODS))
DELEGATOR_HELPERS = ("_prefill_pool_parallel_context", "_pool_probe_context")


def worker_method_names():
    """Every method name defined on ``BatchGenWorker``.

    A collaborator callable is the worker's own method of that name, so the
    allowed set of underscore-prefixed keyword-only parameters is "a real
    worker or engine method" rather than a hand-maintained exception list.
    """
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    return {node.name for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


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
    # D7.4: the wave planner picks the first matching hit page with
    # ``next(genexp, None)``, which consumes the generator on the spot just as
    # the reductions above do.
    "next",
}
# A METHOD that likewise consumes its generator argument on the spot. D7.4's
# planner formats its decline list with ``", ".join(genexp)``; recognising the
# consumer by name keeps the invariant ("no generator crosses the seam")
# exactly as strict as a bare-builtin allowlist did, rather than accepting any
# call that happens to take a generator.
_CONSUMING_METHODS = {"join"}


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
        if isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Name)
             and call.func.id in _CONSUMING_CALLS)
            or (isinstance(call.func, ast.Attribute)
                and call.func.attr in _CONSUMING_METHODS)
        )
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
    worker_shaped = engine_method_names() | worker_method_names()
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
    engine_methods = engine_method_names() | worker_method_names()
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
    # Eight from D7.1, four from D7.2, eight from D7.3, four from D7.4 (whose
    # three probe recorders reuse D7.1's probe context instead of declaring),
    # four from D7.5, one per execution path, with no reuse at all, and one
    # from D7.6 for the unpooled prepacked driver.
    assert len(contexts) == 29
    for context in contexts:
        instance = context(**{field.name: None for field in fields(context)})
        with pytest.raises(FrozenInstanceError):
            setattr(instance, fields(context)[0].name, object())


def test_every_outcome_is_frozen_and_declares_no_container():
    """An outcome carries what a write ADDED, never the worker's container."""
    ns, _ = runtime()
    outcomes = [
        value for name, value in sorted(ns.items())
        if isinstance(value, type) and name.endswith("Outcome")
    ]
    # One from D7.2, three from D7.3, one from D7.4.
    assert len(outcomes) == 5
    for outcome in outcomes:
        instance = outcome()
        with pytest.raises(FrozenInstanceError):
            setattr(instance, fields(outcome)[0].name, object())
        # The default IS the engine's own sentinel, so "added nothing" can
        # never be confused with adding ``None`` (a valid cost profile).
        for field in fields(outcome):
            assert getattr(instance, field.name) is ns["PREFILL_UNCHANGED"]


def test_only_the_declared_operation_returns_an_outcome():
    """Every other converted body writes no worker attribute at all.

    The no-worker guard above already proves none of them STORES through the
    seam; this pins the other half, that none of them smuggles a write out in
    an outcome either.
    """
    ns, _ = runtime()
    outcomes = {name for name in ns if name.endswith("Outcome")}
    for name in sorted(METHODS):
        node = method_node(name)
        built = {
            n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        } & outcomes
        assert bool(built) == (name in OUTCOME_METHODS), (name, built)


def test_no_context_carries_a_field_its_operations_do_not_read():
    """Narrowness, measured against the bodies instead of asserted by name.

    A context's declared fields must be exactly what the operations annotated
    with it read through ``ctx`` -- the UNION over those operations, because a
    shared context is shared precisely where the read sets coincide. A field
    nobody reads would make a delegator resolve a worker attribute its
    operation never needed, which is how a narrow read turns into an
    AttributeError.
    """
    reads = {}
    for name in sorted(METHODS):
        if name in STATELESS_METHODS:
            continue
        node = method_node(name)
        annotation = ast.unparse(node.args.args[1].annotation)
        reads.setdefault(annotation, set()).update(
            n.attr for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "ctx"
        )
    ns, _ = runtime()
    # One entry per DISTINCT annotation: 29 contexts, and the probe context is
    # annotated five times (D7.1's pair plus D7.4's three recorders), so the
    # union it is measured against spans all five bodies.
    assert len(reads) == 29
    for context_name, read in sorted(reads.items()):
        declared = {field.name for field in fields(ns[context_name])}
        assert declared == read, (context_name, declared ^ read)


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
    """The file still holds worker-shaped operations: this is stage six of seven.

    D7.7 converts the rest, so the guard that the whole FILE names no worker
    belongs to that stage, not here.
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
    assert worker_shaped, "nothing left to convert, but D7.6 is not the last stage"
    # The six stages are disjoint, and what is left is what D7.7 takes:
    # fourteen converted in D7.1, five in D7.2, nine in D7.3, seven in D7.4,
    # four in D7.5, one here, and the two preparation/config bodies still
    # worker-shaped -- `_prepare_prefill_batch` and `_config_prefill_for_batch`.
    assert D7_1_METHODS.isdisjoint(D7_2_METHODS)
    assert D7_3_METHODS.isdisjoint(D7_1_METHODS | D7_2_METHODS)
    assert D7_4_METHODS.isdisjoint(D7_1_METHODS | D7_2_METHODS | D7_3_METHODS)
    assert D7_5_METHODS.isdisjoint(
        D7_1_METHODS | D7_2_METHODS | D7_3_METHODS | D7_4_METHODS
    )
    assert D7_6_METHODS.isdisjoint(
        D7_1_METHODS | D7_2_METHODS | D7_3_METHODS | D7_4_METHODS
        | D7_5_METHODS
    )
    assert worker_shaped == {
        "_prepare_prefill_batch", "_config_prefill_for_batch",
    }
    assert (
        len(D7_1_METHODS), len(D7_2_METHODS), len(D7_3_METHODS),
        len(D7_4_METHODS), len(D7_5_METHODS), len(D7_6_METHODS),
        len(worker_shaped),
    ) == (14, 5, 9, 7, 4, 1, 2)


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


def recording(engine, name, capture, result="recorded"):
    """Replace one engine operation with a recorder of its arguments.

    ``result`` is what the recorder hands back, because a delegator that
    applies an outcome unpacks the engine's return value: the cost-profile
    delegator needs a ``(profile, outcome)`` pair, not a marker string.
    """
    def record(*args, **kwargs):
        capture.append((args, kwargs))
        return result
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


# --------------------------------------------------------------------------- #
#  D7.2: the cost-profile cache
# --------------------------------------------------------------------------- #
MEM_141G = 141 * 1024 ** 3


@pytest.fixture
def cost_lookup(monkeypatch):
    """The real import seam, with the device and the table recorded.

    The body imports the lookup inside the miss branch and reads device 0, so
    an arm that resolves nothing must leave both of these untouched.
    """
    import batchgen.prefix_reuse.executor_cost as executor_cost

    calls, devices = [], []
    profile = SimpleNamespace(model="kimi", machine="h200", group_size=8)

    def lookup(*args):
        calls.append(args)
        return profile

    def device_properties(index):
        devices.append(index)
        return SimpleNamespace(major=9, minor=0, total_memory=MEM_141G)

    monkeypatch.setattr(executor_cost, "find_executor_cost_profile", lookup)
    monkeypatch.setattr(torch.cuda, "get_device_properties", device_properties,
                        raising=False)
    return SimpleNamespace(calls=calls, devices=devices, profile=profile)


def cost_context(ns, *, cache=None, model="org/Kimi-K3", world=16, rank=3):
    return ns["PrefixRankCostProfileContext"](
        rank=rank,
        world_size=world,
        model_name=model,
        _prefix_rank_cost_profiles=cache,
    )


def test_an_absent_cache_resolves_the_profile_and_reports_the_entry(cost_lookup):
    ns, engine = runtime()
    profile, outcome = engine._cached_prefix_rank_cost_profile(
        # ``None`` is the delegator's pre-move ``hasattr`` miss.
        cost_context(ns), attn_group_size=8,
    )
    assert profile is cost_lookup.profile
    assert outcome.cache_entry == (False, cost_lookup.profile)
    # The group size scores one serve group, and device 0 stands for the node.
    assert cost_lookup.calls == [
        ("org/Kimi-K3", (9, 0), MEM_141G, 16, 8),
    ]
    assert cost_lookup.devices == [0]


def test_a_pure_dp_rank_resolves_a_group_of_one(cost_lookup):
    ns, engine = runtime()
    engine._cached_prefix_rank_cost_profile(cost_context(ns), attn_group_size=1)
    assert cost_lookup.calls[0][-1] == 1


def test_the_off_arm_reports_a_none_entry_without_a_lookup(cost_lookup):
    ns, engine = runtime()
    profile, outcome = engine._cached_prefix_rank_cost_profile(
        cost_context(ns), True, attn_group_size=8,
    )
    assert profile is None
    # ``None`` IS the entry: the token-balance arm caches that answer, which
    # is why the sentinel and not ``None`` means "added nothing".
    assert outcome.cache_entry == (True, None)
    assert outcome.cache_entry[1] is not ns["PREFILL_UNCHANGED"]
    assert cost_lookup.calls == [] and cost_lookup.devices == []


def test_a_cache_hit_adds_nothing_and_looks_nothing_up(cost_lookup):
    ns, engine = runtime()
    cached = object()
    profile, outcome = engine._cached_prefix_rank_cost_profile(
        cost_context(ns, cache={False: cached}), attn_group_size=8,
    )
    assert profile is cached
    assert outcome.cache_entry is ns["PREFILL_UNCHANGED"]
    assert cost_lookup.calls == []


def test_the_other_arms_entry_is_not_this_arms_answer(cost_lookup):
    """The cache is keyed: one process serves both arms."""
    ns, engine = runtime()
    profile, outcome = engine._cached_prefix_rank_cost_profile(
        cost_context(ns, cache={True: None}), attn_group_size=8,
    )
    assert profile is cost_lookup.profile
    assert outcome.cache_entry == (False, cost_lookup.profile)


def test_the_engine_never_writes_the_cache_it_was_handed(cost_lookup):
    """The one worker container this stage writes is written by the delegator.

    The context carries the cache by value for the key test and the tail
    lookup; the add goes back in the outcome, so the object the delegator
    passed in comes back untouched.
    """
    ns, engine = runtime()
    live = {True: None}
    profile, outcome = engine._cached_prefix_rank_cost_profile(
        cost_context(ns, cache=live), attn_group_size=8,
    )
    assert live == {True: None}
    assert outcome.cache_entry == (False, profile)


# --------------------------------------------------------------------------- #
#  D7.2: the wave's trajectory re-entries
# --------------------------------------------------------------------------- #
def frontier_record(pages=((0, (71, 72)),)):
    return SimpleNamespace(page_ids_by_group=dict(pages))


def attachment(*, boundary=8, group_id=0, pages=((0, (71, 72)),)):
    return SimpleNamespace(
        boundary_tokens=boundary, owning_group_id=group_id,
        record=frontier_record(pages),
    )


def seq(global_idx=100, *, prompt_length=32, group=0):
    return SimpleNamespace(
        global_idx=global_idx, prompt_length=prompt_length,
        decode_dp_group=group,
    )


def session_index(entry, *, calls=None):
    """The mirrored session index: replicated, read-only, resolve-by-prompt."""
    def resolve(*, namespace_digest, prompt_token_ids):
        if calls is not None:
            calls.append((namespace_digest, prompt_token_ids))
        return entry
    return SimpleNamespace(resolve=resolve)


def hits_context(
    ns, *, attachments=None, index=None, states=None, digest=(1, 2), rank=4,
):
    return ns["MaterializedPrefixHitsContext"](
        rank=rank,
        prefix_runtime=SimpleNamespace(namespace_digest=digest),
        traj=SimpleNamespace(
            trajectory_attachments=dict(attachments or {}),
            trajectory_session_groups=index,
        ),
        prefix=SimpleNamespace(prefix_sequence_states=dict(states or {})),
    )


def derive(
    engine, context, seqs, *, page_tokens=4, group=None, requested=True,
    tokens=(1, 2, 3), reason=None, owns=False, asked=None,
):
    """Drive the derivation with its four collaborator callables."""
    def frontier(sequence, **kwargs):
        if asked is not None:
            asked.append((sequence.global_idx, kwargs))
        return reason

    return engine._materialized_prefix_hits(
        context, seqs, page_tokens=page_tokens, group=group,
        _trajectory_commit_requested=lambda _seqs: requested,
        _trajectory_tokens=lambda sequence, length: SimpleNamespace(
            tolist=lambda: list(tokens),
        ),
        _unmaterializable_frontier=frontier,
        _owns_host_kv=lambda sequence: owns,
    )


def test_a_flag_off_wave_derives_nothing_and_reads_no_state():
    ns, engine = runtime()
    context = ns["MaterializedPrefixHitsContext"](
        rank=7,
        prefix_runtime=Explode("prefix_runtime"),
        traj=Explode("traj"),
        prefix=Explode("prefix"),
    )
    assert derive(engine, context, [seq()], requested=False) == (
        [], True, {}, {},
    )


def test_its_own_group_reads_its_live_attachment(caplog):
    ns, engine = runtime()
    context = hits_context(
        ns, attachments={100: attachment()},
        states={100: SimpleNamespace(attached_tokens=8)},
    )
    with caplog.at_level("WARNING"):
        hits, confirmed, pages, drops = derive(engine, context, [seq()])
    assert [(hit.index, hit.tokens) for hit in hits] == [(0, 8)]
    assert confirmed is True and drops == {}
    assert pages == {0: {0: [71, 72]}}
    assert caplog.text == ""


def test_a_turn_one_prompt_continues_nothing_and_is_counted_once(caplog):
    ns, engine = runtime()
    with caplog.at_level("INFO"):
        hits, confirmed, pages, drops = derive(
            engine, hits_context(ns), [seq(), seq(101)],
        )
    assert (hits, confirmed, pages, drops) == ([], True, {}, {})
    # One aggregate line, naming where the rank looked.
    assert caplog.text.count("[PREFIX_POOL]") == 1
    assert "2 of 2 flag-on sequence(s)" in caplog.text
    assert "this rank's live attachments" in caplog.text


def test_a_grouped_wave_reads_the_mirrored_index_not_the_registry():
    ns, engine = runtime()
    asked = []
    index = session_index(
        SimpleNamespace(boundary_tokens=8, owning_group_id=0), calls=asked,
    )
    # The live attachment exists and agrees, but the BOUNDARY came from the
    # index: every rank of the group derives the same hit.
    context = hits_context(
        ns, attachments={100: attachment()}, index=index, digest=(9, 9),
        states={100: SimpleNamespace(attached_tokens=8)},
    )
    hits, confirmed, pages, drops = derive(
        engine, context, [seq()], group="pg", tokens=(5, 6),
    )
    assert [(hit.index, hit.tokens) for hit in hits] == [(0, 8)]
    assert (confirmed, drops) == (True, {})
    assert asked == [((9, 9), [5, 6])]


def test_a_grouped_wave_without_an_index_drops_every_row(caplog):
    ns, engine = runtime()
    with caplog.at_level("INFO"):
        hits, confirmed, pages, drops = derive(
            engine, hits_context(ns, index=None), [seq()], group="pg",
        )
    assert (hits, confirmed, pages) == ([], True, {})
    assert drops == {0: "the mirrored session index does not place this session"}
    assert "the mirrored session index" in caplog.text


def test_a_row_the_index_does_not_place_is_dropped_by_name():
    ns, engine = runtime()
    context = hits_context(ns, index=session_index(None))
    hits, confirmed, pages, drops = derive(
        engine, context, [seq()], group="pg",
    )
    assert (hits, confirmed, pages) == ([], True, {})
    assert drops == {0: "the mirrored session index does not place this session"}


def test_an_unmaterializable_frontier_is_dropped_by_its_reason(caplog):
    ns, engine = runtime()
    asked = []
    context = hits_context(ns, attachments={100: attachment(boundary=12)})
    with caplog.at_level("INFO"):
        hits, confirmed, pages, drops = derive(
            engine, context, [seq()], page_tokens=4,
            reason="no_private_token_left", asked=asked,
        )
    assert (hits, confirmed, pages) == ([], True, {})
    assert drops == {
        0: "the pool cannot materialize it (no_private_token_left)",
    }
    # The predicate is asked with the boundary and group the row resolved.
    assert asked == [
        (100, {"boundary": 12, "owning_group_id": 0, "page_tokens": 4}),
    ]
    assert "the pool cannot materialize" in caplog.text


def test_a_leader_without_a_live_record_takes_the_group_back(caplog):
    """The mirror is stale: this rank holds the registry and has no record."""
    ns, engine = runtime()
    context = hits_context(
        ns, index=session_index(
            SimpleNamespace(boundary_tokens=8, owning_group_id=0),
        ),
    )
    with caplog.at_level("WARNING"):
        hits, confirmed, pages, drops = derive(
            engine, context, [seq()], group="pg", owns=True,
        )
    assert [hit.index for hit in hits] == [0]
    assert confirmed is False and pages == {} and drops == {}
    assert "holds no live frontier" in caplog.text


def test_a_follower_without_a_record_has_no_opinion(caplog):
    ns, engine = runtime()
    context = hits_context(
        ns, index=session_index(
            SimpleNamespace(boundary_tokens=8, owning_group_id=0),
        ),
    )
    with caplog.at_level("WARNING"):
        _, confirmed, pages, drops = derive(
            engine, context, [seq()], group="pg", owns=False,
        )
    assert confirmed is True and pages == {} and drops == {}
    assert caplog.text == ""


def test_a_boundary_the_index_disagrees_with_is_dropped_and_vetoed(caplog):
    ns, engine = runtime()
    context = hits_context(
        ns, attachments={100: attachment(boundary=4)},
        index=session_index(
            SimpleNamespace(boundary_tokens=8, owning_group_id=0),
        ),
    )
    with caplog.at_level("WARNING"):
        hits, confirmed, pages, drops = derive(
            engine, context, [seq()], group="pg",
        )
    assert [hit.index for hit in hits] == [0]
    assert confirmed is False and pages == {}
    assert drops == {
        0: "the live frontier is at boundary 4 but the mirrored index says 8",
    }
    assert "the group recomputes this wave" in caplog.text


def test_an_attachment_short_of_the_frontier_is_dropped_and_vetoed(caplog):
    """The resident pages the load reads are kept by this row's attachment."""
    ns, engine = runtime()
    context = hits_context(
        ns, attachments={100: attachment()},
        states={100: SimpleNamespace(attached_tokens=4)},
    )
    with caplog.at_level("WARNING"):
        hits, confirmed, pages, drops = derive(engine, context, [seq()])
    assert [hit.index for hit in hits] == [0]
    assert confirmed is False and pages == {}
    assert drops == {
        0: ("its prefix-cache attachment covers 4 token(s), short of the "
            "8-token frontier it must keep resident"),
    }
    assert "fewer than the" in caplog.text


def test_a_row_with_no_installed_state_counts_as_zero_attached():
    ns, engine = runtime()
    context = hits_context(ns, attachments={100: attachment()}, states={})
    _, confirmed, pages, drops = derive(engine, context, [seq()])
    assert confirmed is False and pages == {}
    assert "covers 0 token(s)" in drops[0]


def test_a_record_naming_no_resident_page_is_dropped_and_vetoed(caplog):
    ns, engine = runtime()
    context = hits_context(
        ns, attachments={100: attachment(pages=())},
        states={100: SimpleNamespace(attached_tokens=8)},
    )
    with caplog.at_level("WARNING"):
        hits, confirmed, pages, drops = derive(engine, context, [seq()])
    assert [hit.index for hit in hits] == [0]
    assert confirmed is False and pages == {}
    assert drops == {0: "its frontier record names no resident Host page"}
    assert "names no resident Host page" in caplog.text


def test_the_wave_index_is_the_position_and_a_hole_is_skipped():
    """``None`` in the wave is a hole: every map is keyed by POSITION."""
    ns, engine = runtime()
    context = hits_context(
        ns,
        attachments={
            100: attachment(),
            102: attachment(boundary=8, pages=((0, (81,)),)),
        },
        states={
            100: SimpleNamespace(attached_tokens=8),
            102: SimpleNamespace(attached_tokens=8),
        },
    )
    hits, confirmed, pages, drops = derive(
        engine, context, [seq(100), None, seq(102)],
    )
    assert [hit.index for hit in hits] == [0, 2]
    assert sorted(pages) == [0, 2]
    assert (confirmed, drops) == (True, {})


def test_the_owned_stores_are_only_read():
    """Prefix-reuse and trajectory state have one owner each: this reads."""
    ns, engine = runtime()
    context = hits_context(
        ns, attachments={100: attachment()},
        states={100: SimpleNamespace(attached_tokens=8)},
    )
    before_attachments = dict(context.traj.trajectory_attachments)
    before_states = dict(context.prefix.prefix_sequence_states)
    derive(engine, context, [seq()])
    assert context.traj.trajectory_attachments == before_attachments
    assert context.prefix.prefix_sequence_states == before_states


# --------------------------------------------------------------------------- #
#  D7.2: why a frontier cannot become a pool segment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("boundary", [0, -4, 3, 5])
def test_a_boundary_off_the_page_grid_is_refused(boundary):
    _, engine = runtime()
    assert engine._unmaterializable_frontier(
        seq(prompt_length=64), boundary=boundary, owning_group_id=0,
        page_tokens=4,
    ) == "boundary_off_the_page_grid"


def test_the_last_prompt_token_must_stay_private():
    """It produces the logit, so the frontier cannot cover the whole prompt."""
    _, engine = runtime()
    assert engine._unmaterializable_frontier(
        seq(prompt_length=8), boundary=8, owning_group_id=0, page_tokens=4,
    ) == "no_private_token_left"
    assert engine._unmaterializable_frontier(
        seq(prompt_length=9), boundary=8, owning_group_id=0, page_tokens=4,
    ) is None


def test_a_sequence_serving_outside_the_owning_group_is_refused():
    _, engine = runtime()
    assert engine._unmaterializable_frontier(
        seq(prompt_length=64, group=1), boundary=8, owning_group_id=0,
        page_tokens=4,
    ) == "serving_group_is_not_the_owning_group"


def test_an_unplaced_sequence_counts_as_group_zero():
    _, engine = runtime()
    assert engine._unmaterializable_frontier(
        seq(prompt_length=64, group=None), boundary=8, owning_group_id=0,
        page_tokens=4,
    ) is None


# --------------------------------------------------------------------------- #
#  D7.2: releasing the cross-wave lookup hits
# --------------------------------------------------------------------------- #
def owner_state():
    """A REAL prefix-reuse owner that records the installs it was asked for."""
    from batchgen.worker.prefix_reuse_state import PrefixReuseState

    class Recording(PrefixReuseState):
        def __init__(self):
            super().__init__(rank=0, world_size=1)
            self.installed = []

        def install_state(self, global_id, state):
            self.installed.append((global_id, state))
            super().install_state(global_id, state)

    return Recording()


def replacement_context(ns, *, released=None, prefix=None, rank=2):
    return ns["PrefixLookupReplacementContext"](
        rank=rank,
        prefix_coordinator=SimpleNamespace(
            release_attachment=(released if released is not None else []).append,
        ),
        prefix=prefix if prefix is not None else owner_state(),
    )


def test_a_lookup_hit_is_released_and_the_prompt_starts_at_its_chain_end():
    ns, engine = runtime()
    released, owner = [], owner_state()
    context = replacement_context(ns, released=released, prefix=owner)
    states = {3: SimpleNamespace(attachment_handle=7)}
    engine._replace_prefix_lookup_hits(
        context, [seq(3)], states, [64],
    )
    assert released == [7]
    installed = states[3]
    assert (installed.attached_tokens, installed.compute_cached_tokens) == (64, 64)
    assert installed.lookup_result is None
    assert installed.commit_attachment_handles == ()
    # Installed through the OWNER's method, which is the only write path.
    assert owner.installed == [(3, installed)]
    assert owner.prefix_sequence_states == {3: installed}


def test_a_retained_hit_keeps_its_attachment_as_a_commit_handle():
    ns, engine = runtime()
    released, owner = [], owner_state()
    engine._replace_prefix_lookup_hits(
        replacement_context(ns, released=released, prefix=owner),
        [seq(3)], {3: SimpleNamespace(attachment_handle=7)}, [64],
        retain={0},
    )
    assert released == []
    assert owner.prefix_sequence_states[3].commit_attachment_handles == (7,)


def test_a_follower_without_a_lookup_releases_nothing():
    ns, engine = runtime()
    released, owner = [], owner_state()
    states = {}
    engine._replace_prefix_lookup_hits(
        replacement_context(ns, released=released, prefix=owner),
        [seq(3)], states, [0],
    )
    assert released == []
    assert states[3].commit_attachment_handles == ()


def test_the_restore_boundary_is_carried_on_the_row_it_was_planned_for():
    ns, engine = runtime()
    owner = owner_state()
    states = {}
    engine._replace_prefix_lookup_hits(
        replacement_context(ns, prefix=owner),
        [seq(3), seq(5)], states, [64, 0], restore_boundaries={0: 64},
    )
    assert states[3].trajectory_restore_boundary == 64
    assert states[5].trajectory_restore_boundary == 0


def test_a_restore_that_does_not_meet_its_compute_start_is_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="Rank 2: sequence 3 is planned"):
        engine._replace_prefix_lookup_hits(
            replacement_context(ns), [seq(3)], {}, [32],
            restore_boundaries={0: 64},
        )


# --------------------------------------------------------------------------- #
#  D7.2: the plan-time restore settlement
# --------------------------------------------------------------------------- #
class Restorer:
    def __init__(self):
        self.declined = []

    def decline(self, attached, *, reason, detail):
        self.declined.append((attached, reason, detail))


def settle_context(
    ns, *, attachments=None, restorer=None, states=None, rank=6, prefix=None,
):
    return ns["TrajectoryRestoreSettleContext"](
        rank=rank,
        traj=SimpleNamespace(
            trajectory_attachments=dict(attachments or {}),
            trajectory_restorer=restorer,
        ),
        prefix=prefix if prefix is not None else SimpleNamespace(
            prefix_sequence_states=dict(states or {}),
        ),
    )


def test_a_follower_with_no_attachment_settles_nothing():
    ns, engine = runtime()
    context = settle_context(ns, prefix=Explode("prefix"))
    assert engine._settle_trajectory_restore_plan(
        context, [seq()], {}, "unused",
    ) is None


def test_a_planned_row_keeps_its_attachment():
    ns, engine = runtime()
    restorer = Restorer()
    held = attachment()
    context = settle_context(
        ns, attachments={100: held}, restorer=restorer,
        states={100: SimpleNamespace(trajectory_restore_boundary=8)},
    )
    engine._settle_trajectory_restore_plan(context, [seq()], {}, "unused")
    assert context.traj.trajectory_attachments == {100: held}
    assert restorer.declined == []


def test_a_plan_that_disagrees_with_the_live_frontier_is_refused():
    ns, engine = runtime()
    context = settle_context(
        ns, attachments={100: attachment(boundary=8)}, restorer=Restorer(),
        states={100: SimpleNamespace(trajectory_restore_boundary=4)},
    )
    with pytest.raises(RuntimeError, match="Rank 6: sequence 100 is planned"):
        engine._settle_trajectory_restore_plan(context, [seq()], {}, "unused")


def test_an_unplanned_attachment_is_consumed_here_with_the_plans_reason():
    ns, engine = runtime()
    restorer = Restorer()
    held = attachment()
    context = settle_context(
        ns, attachments={100: held}, restorer=restorer,
        states={100: SimpleNamespace(trajectory_restore_boundary=0)},
    )
    engine._settle_trajectory_restore_plan(
        context, [seq()], {0: "the group could not confirm it"}, "not planned",
    )
    # Removed from the store's own map, so it never reaches the restore seam.
    assert context.traj.trajectory_attachments == {}
    (declined, reason, detail), = restorer.declined
    assert declined is held
    assert detail == "at plan time: the group could not confirm it"
    from batchgen.prefix_reuse.trajectory_restore import DECLINE_NOT_PLANNED
    assert reason == DECLINE_NOT_PLANNED


def test_a_row_the_plan_dropped_without_a_reason_takes_the_default():
    ns, engine = runtime()
    restorer = Restorer()
    context = settle_context(
        ns, attachments={100: attachment()}, restorer=restorer, states={},
    )
    engine._settle_trajectory_restore_plan(context, [seq()], {}, "no budget")
    assert restorer.declined[0][2] == "at plan time: no budget"


def test_an_unplanned_attachment_without_a_restorer_is_refused():
    ns, engine = runtime()
    context = settle_context(ns, attachments={100: attachment()}, restorer=None)
    with pytest.raises(RuntimeError, match="has no restorer"):
        engine._settle_trajectory_restore_plan(context, [seq()], {}, "unused")


def test_the_settlement_runs_in_wave_order_and_skips_holes():
    ns, engine = runtime()
    restorer = Restorer()
    context = settle_context(
        ns,
        attachments={100: attachment(), 102: attachment()},
        restorer=restorer,
        states={},
    )
    engine._settle_trajectory_restore_plan(
        context, [seq(100), None, seq(102)], {0: "first", 2: "third"}, "unused",
    )
    assert [detail for _, _, detail in restorer.declined] == [
        "at plan time: first", "at plan time: third",
    ]
    assert context.traj.trajectory_attachments == {}


def test_the_settlement_only_reads_the_prefix_states():
    ns, engine = runtime()
    states = {100: SimpleNamespace(trajectory_restore_boundary=0)}
    context = settle_context(
        ns, attachments={100: attachment()}, restorer=Restorer(),
        states=states,
    )
    engine._settle_trajectory_restore_plan(context, [seq()], {}, "unused")
    assert context.prefix.prefix_sequence_states == states


# --------------------------------------------------------------------------- #
#  D7.2: the worker delegators
# --------------------------------------------------------------------------- #
def cost_profile_worker(monkeypatch, ns, engine, *, group=8):
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 3
    worker.world_size = 16
    worker.model_name = "org/Kimi-K3"
    worker._prefill_attn_tp_size = lambda: group
    return worker


def test_the_cost_profile_delegator_creates_the_cache_then_populates_it(
    monkeypatch, cost_lookup,
):
    """The pre-move order, on the worker: create when absent, then add.

    The engine hands the entry back; the delegator is what writes. A second
    call must find the entry and resolve nothing, which is the whole point of
    a cache keyed by the arm.
    """
    ns, engine = runtime()
    worker = cost_profile_worker(monkeypatch, ns, engine)
    assert not hasattr(worker, "_prefix_rank_cost_profiles")
    assert worker._cached_prefix_rank_cost_profile() is cost_lookup.profile
    assert worker._prefix_rank_cost_profiles == {False: cost_lookup.profile}
    # Second call: the cache answers and the lookup is not repeated.
    assert worker._cached_prefix_rank_cost_profile() is cost_lookup.profile
    assert len(cost_lookup.calls) == 1
    assert worker._prefix_rank_cost_profiles == {False: cost_lookup.profile}


def test_the_cost_profile_delegator_keeps_both_arms(monkeypatch, cost_lookup):
    """An outcome carrying the container would drop the other arm's entry."""
    ns, engine = runtime()
    worker = cost_profile_worker(monkeypatch, ns, engine)
    seen = [worker._cached_prefix_rank_cost_profile(disabled)
            for disabled in (False, True, False, True)]
    assert [item is None for item in seen] == [False, True, False, True]
    assert worker._prefix_rank_cost_profiles == {
        False: cost_lookup.profile, True: None,
    }
    assert len(cost_lookup.calls) == 1


def test_the_cost_profile_delegator_writes_nothing_on_a_hit(
    monkeypatch, cost_lookup,
):
    ns, engine = runtime()
    worker = cost_profile_worker(monkeypatch, ns, engine)

    class Watching(dict):
        def __init__(self, *args):
            super().__init__(*args)
            self.writes = []

        def __setitem__(self, key, value):
            self.writes.append(key)
            super().__setitem__(key, value)

    worker._prefix_rank_cost_profiles = Watching({False: cost_lookup.profile})
    assert worker._cached_prefix_rank_cost_profile() is cost_lookup.profile
    assert worker._prefix_rank_cost_profiles.writes == []
    assert cost_lookup.calls == []


def test_the_cost_profile_delegator_builds_its_context(monkeypatch):
    ns, engine = runtime()
    worker = cost_profile_worker(monkeypatch, ns, engine, group=4)
    worker._prefix_rank_cost_profiles = {True: None}
    seen = []
    recording(engine, "_cached_prefix_rank_cost_profile", seen,
              result=("profile", ns["PrefixRankCostProfileCacheOutcome"]()))
    assert worker._cached_prefix_rank_cost_profile(True) == "profile"
    (context, disabled), kwargs = seen[0]
    assert isinstance(context, ns["PrefixRankCostProfileContext"])
    assert (context.rank, context.world_size, context.model_name) == (
        3, 16, "org/Kimi-K3",
    )
    # The cache travels by value, and the live one at that.
    assert context._prefix_rank_cost_profiles == {True: None}
    assert disabled is True
    assert kwargs == {"attn_group_size": 4}


def test_the_cost_profile_delegator_passes_the_hasattr_miss_as_none(monkeypatch):
    ns, engine = runtime()
    worker = cost_profile_worker(monkeypatch, ns, engine)
    assert not hasattr(worker, "_prefix_rank_cost_profiles")
    seen = []
    recording(engine, "_cached_prefix_rank_cost_profile", seen,
              result=(None, ns["PrefixRankCostProfileCacheOutcome"]()))
    worker._cached_prefix_rank_cost_profile()
    assert seen[0][0][0]._prefix_rank_cost_profiles is None
    # The delegator still creates the container, exactly as the pre-move body
    # did before it read the key.
    assert worker._prefix_rank_cost_profiles == {}


def test_the_hits_delegator_builds_its_context_and_its_collaborators(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 4
    worker.prefix_cache_runtime_config = "runtime"
    worker._traj = "trajectories"
    worker._prefix = "prefix state"
    worker._trajectory_commit_requested = lambda seqs: True
    worker._trajectory_tokens = lambda seq, length: seq
    worker._owns_host_kv = lambda seq: False
    seen = []
    recording(engine, "_materialized_prefix_hits", seen)
    assert worker._materialized_prefix_hits(
        ["seq"], page_tokens=64, group="pg",
    ) == "recorded"
    (context, seqs), kwargs = seen[0]
    assert isinstance(context, ns["MaterializedPrefixHitsContext"])
    assert (context.rank, context.prefix_runtime, context.traj,
            context.prefix) == (4, "runtime", "trajectories", "prefix state")
    assert seqs == ["seq"]
    assert kwargs == {
        "page_tokens": 64,
        "group": "pg",
        "_trajectory_commit_requested": worker._trajectory_commit_requested,
        "_trajectory_tokens": worker._trajectory_tokens,
        "_unmaterializable_frontier": worker._unmaterializable_frontier,
        "_owns_host_kv": worker._owns_host_kv,
    }


def test_the_frontier_predicate_delegator_passes_no_context(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    seen = []
    recording(engine, "_unmaterializable_frontier", seen)
    assert worker._unmaterializable_frontier(
        "seq", boundary=8, owning_group_id=0, page_tokens=4,
    ) == "recorded"
    assert seen[0] == (
        ("seq",), {"boundary": 8, "owning_group_id": 0, "page_tokens": 4},
    )


def test_the_lookup_replacement_delegator_builds_its_context(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 2
    worker.prefix_cache_coordinator = "coordinator"
    worker._prefix = "prefix state"
    seen = []
    recording(engine, "_replace_prefix_lookup_hits", seen)
    worker._replace_prefix_lookup_hits(
        ["seq"], {"states": 1}, [64], {0}, {0: 64},
    )
    (context, seqs, states, chain, retain, restores), kwargs = seen[0]
    assert isinstance(context, ns["PrefixLookupReplacementContext"])
    assert (context.rank, context.prefix_coordinator, context.prefix) == (
        2, "coordinator", "prefix state",
    )
    # Every argument of the pre-move call stays positional, in its order.
    assert (seqs, states, chain, retain, restores) == (
        ["seq"], {"states": 1}, [64], {0}, {0: 64},
    )
    assert kwargs == {}


def test_the_settle_delegator_reads_the_live_stores_each_call(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 6
    worker._traj = "first"
    worker._prefix = "prefix state"
    seen = []
    recording(engine, "_settle_trajectory_restore_plan", seen)
    worker._settle_trajectory_restore_plan(["seq"], {0: "why"}, "default")
    worker._traj = "second"
    worker._settle_trajectory_restore_plan(["seq"], {}, "default")
    contexts = [args[0] for args, _ in seen]
    assert all(isinstance(c, ns["TrajectoryRestoreSettleContext"])
               for c in contexts)
    assert [c.traj for c in contexts] == ["first", "second"]
    assert [c.rank for c in contexts] == [6, 6]
    assert seen[0][0][1:] == (["seq"], {0: "why"}, "default")
    assert seen[0][1] == {}


# --------------------------------------------------------------------------- #
#  D7.3: the stage's own seam shape
# --------------------------------------------------------------------------- #
D7_3_CONTEXTS = {
    "_measure_prefill_prefix_kv_budget": "PrefillPrefixKVBudgetContext",
    "_construct_prefill_pool": "PrefillPoolConstructionContext",
    "_check_pool_plan_identical": "PoolPlanIdentityContext",
    "_prefill_pool_layout": "PrefillPoolLayoutContext",
    "_allocate_prefill_pool_host_pages": "PrefillPoolHostPageContext",
    "_materialized_segment_loader": "MaterializedSegmentStagingContext",
    "_materialized_pool_index_key_layers": "MaterializedIndexKeyLayersContext",
    "_load_materialized_segments": "MaterializedSegmentLoadContext",
}


@pytest.mark.parametrize("name", sorted(D7_3_CONTEXTS))
def test_each_construction_operation_declares_its_own_context(name):
    """No sharing in this stage, asserted per operation.

    Eight contexts for eight operations, because no two of their read sets are
    equal; the ninth reads no worker state and takes none. The narrowness
    itself is proved by ``test_no_context_carries_a_field_its_operations_do
    _not_read`` above, which derives each context's fields from the bodies.
    """
    node = method_node(name)
    assert ast.unparse(node.args.args[1].annotation) == D7_3_CONTEXTS[name]


def test_no_earlier_stages_context_was_widened_for_this_stage():
    """The D7.1/D7.2 contexts keep exactly the fields they were declared with.

    The tempting share was the wave context: the host-page allocation reads
    the planned wave, which is that context's whole read set, but it also
    reads the prefix-reuse owner. Widening the wave context would make the
    segment-page reader's delegator resolve an owner it never touches, so the
    allocation got its own.
    """
    ns, _ = runtime()
    assert [f.name for f in fields(ns["PrefillPoolWaveContext"])] == [
        "_prefill_pool_wave",
    ]
    assert [f.name for f in fields(ns["PrefillPoolHostPageContext"])] == [
        "_prefill_pool_wave", "prefix",
    ]
    assert {f.name for f in fields(ns["PrefillPoolParallelContext"])} == {
        "parallel_manager",
    }
    for name, context in sorted(D7_3_CONTEXTS.items()):
        del name
        assert context not in {
            ast.unparse(method_node(other).args.args[1].annotation)
            for other in sorted((D7_1_METHODS | D7_2_METHODS)
                                - STATELESS_METHODS)
        }


def test_the_stage_declares_exactly_three_outcomes():
    ns, _ = runtime()
    assert OUTCOME_METHODS & D7_3_METHODS == {
        "_measure_prefill_prefix_kv_budget", "_construct_prefill_pool",
        "_prefill_pool_layout",
    }
    for outcome, declared in (
        ("PrefillPrefixKVBudgetOutcome",
         ["_prefill_prefix_kv_budget_bytes", "_prefill_pool_page_budget"]),
        ("PrefillPoolBaseOutcome", ["_prefill_pool_base"]),
        ("PrefillPoolLayoutOutcome", ["_prefill_pool_layout_cache"]),
    ):
        assert [f.name for f in fields(ns[outcome])] == declared


# --------------------------------------------------------------------------- #
#  D7.3: the collectives keep their position and their one condition
# --------------------------------------------------------------------------- #
def _parents(node):
    parents = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _collective_calls(node, attr):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "dist"]


@pytest.mark.parametrize("name", sorted(COLLECTIVE_METHODS))
def test_the_collective_stands_under_its_group_handle_and_nothing_else(name):
    """Every rank of a serve group must issue this, here, exactly once.

    The pre-move condition is the process-group handle alone: the reduction
    runs under ``group is not None`` and the gather past an early
    ``group is None`` return. A condition that named the rank, the host-KV
    leadership or the local rank would make one rank of a group skip a
    collective its peers issue, which deadlocks the group.
    """
    node = method_node(name)
    calls = _collective_calls(node, COLLECTIVE_METHODS[name])
    assert len(calls) == 1, name
    parents, current, tests = _parents(node), calls[0], []
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.If, ast.IfExp)):
            tests.append(ast.unparse(current.test))
    for condition in tests:
        for rank_dependent in (
            "ctx.rank", "self.rank", "ctx.local_rank", "local_rank",
            "_owns_host_kv", "ctx._prefill_pool_page_budget",
        ):
            assert rank_dependent not in condition, (name, condition)
        assert "group" in condition, (name, condition)


def test_the_reduction_sits_between_the_tensor_and_the_reduced_read():
    """Statement order around the collective, not just its presence."""
    node = method_node("_construct_prefill_pool")
    branch = next(n for n in ast.walk(node)
                  if isinstance(n, ast.If) and _collective_calls(n, "all_reduce"))
    assert [ast.unparse(stmt).split("=")[0].strip() for stmt in branch.body
            if isinstance(stmt, ast.Assign)] == ["reduced", "budget"]
    assert ast.unparse(branch.body[1]).startswith(
        "dist.all_reduce(reduced, op=dist.ReduceOp.MIN, group=group)"
    )
    assert "reduced.item()" in ast.unparse(branch.body[2])


def test_the_gather_sits_between_the_digest_and_the_comparison():
    node = method_node("_check_pool_plan_identical")
    body = [stmt for stmt in node.body if not isinstance(stmt, ast.Expr)]
    gather = next(i for i, stmt in enumerate(node.body)
                  if _collective_calls(stmt, "all_gather"))
    del body
    assert ast.unparse(node.body[gather - 2]).startswith("digest = torch.tensor")
    assert ast.unparse(node.body[gather - 1]).startswith("gathered = [")
    assert isinstance(node.body[gather + 1], ast.If)
    assert "gathered" in ast.unparse(node.body[gather + 1].test)


# --------------------------------------------------------------------------- #
#  D7.3: fakes for the construction and layout bodies
# --------------------------------------------------------------------------- #
class RecordingDist:
    """``torch.distributed``, recording the ordered sequence of collectives.

    ``reduce_to`` is what a peer group agrees on, so a MIN below this rank's
    own budget is observable; ``gather_to`` is what the peers' digests are.
    """

    class ReduceOp:
        MIN = "MIN"

    def __init__(self, *, log, reduce_to=None, gather_to=None):
        self.log = log
        self._reduce_to = reduce_to
        self._gather_to = gather_to

    def all_reduce(self, tensor, *, op, group):
        self.log.append(("all_reduce", op, group, tensor.value))
        if self._reduce_to is not None:
            tensor.value = self._reduce_to

    def all_gather(self, out, tensor, *, group):
        self.log.append(("all_gather", group, tensor.value, len(out)))
        peers = self._gather_to
        if peers is None:
            peers = [tensor.value] * len(out)
        for slot, value in zip(out, peers):
            slot.value = value


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCuda:
    """The memory accounting the budget and the construction read, recorded."""

    def __init__(self, *, log, free=100, total=400, reserved=7, allocated=3):
        self.log = log
        self.free, self.total = free, total
        self.reserved, self.allocated = reserved, allocated

    def synchronize(self, device):
        self.log.append(("synchronize", device))

    def empty_cache(self):
        self.log.append(("empty_cache",))

    def mem_get_info(self, index):
        self.log.append(("mem_get_info", index))
        return self.free, self.total

    def memory_reserved(self, index):
        self.log.append(("memory_reserved", index))
        return self.reserved

    def memory_allocated(self, index):
        self.log.append(("memory_allocated", index))
        return self.allocated

    def reset_peak_memory_stats(self, index):
        self.log.append(("reset_peak", index))


class FakeTorch:
    """Only what these nine bodies touch of the torch module."""

    int64 = "int64"
    float8_e4m3fn = "float8_e4m3fn"

    def __init__(self, cuda):
        self.cuda = cuda

    @staticmethod
    def tensor(values, *, dtype, device):
        return FakeTensor(values[0])

    @staticmethod
    def zeros_like(tensor):
        return FakeTensor(0)


def construction_runtime(log, **extra):
    cuda = FakeCuda(log=log, **{k: v for k, v in extra.items()
                                if k in {"free", "total", "reserved",
                                         "allocated"}})
    dist = RecordingDist(
        log=log,
        reduce_to=extra.get("reduce_to"),
        gather_to=extra.get("gather_to"),
    )
    return runtime(torch=FakeTorch(cuda), dist=dist)


def stub_module(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def pool_wave(*, total_pages=20, peak=8, scratch=2, page_tokens=4,
              segments=(1,), prompts=("p",)):
    return SimpleNamespace(
        total_pages=total_pages,
        page_tokens=page_tokens,
        scratch_pages=scratch,
        prompts=list(prompts),
        plan=SimpleNamespace(peak_pool_pages=peak, segments=tuple(segments)),
        pool=None,
        executor=None,
    )


class FakePool:
    """``PrefillPrefixPool`` as the construction builds it, by keyword.

    Every argument is kept in ``kwargs`` -- the pool's page count is the one
    the budget reduction settled on, which is what the tests read back.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.page_bytes = kwargs.pop("page_bytes", 64)
        self.num_pages = kwargs.get("num_pages")
        self.kernel_dump = None


# --------------------------------------------------------------------------- #
#  The prefill-phase prefix KV budget
# --------------------------------------------------------------------------- #
def budget_context(ns, *, workspace, kv_budget=None, page_budget=None,
                   rank=3, local_rank=1):
    return ns["PrefillPrefixKVBudgetContext"](
        rank=rank,
        local_rank=local_rank,
        torch_device="cuda:1",
        engine_config=SimpleNamespace(
            Module_Batching_Config=SimpleNamespace(
                prefill_prefix_pool_workspace_bytes=workspace,
            )
        ),
        _prefill_prefix_kv_budget_bytes=kv_budget,
        _prefill_pool_page_budget=page_budget,
    )


def measure(ns, engine, ctx, *, pool_enabled, page_bytes=1):
    return engine._measure_prefill_prefix_kv_budget(
        ctx,
        _prefill_prefix_pool_enabled=lambda: pool_enabled,
        _prefill_pool_page_bytes=lambda: page_bytes,
    )


@pytest.fixture
def pool_budget_module(monkeypatch):
    """The real import seam of the page-budget arithmetic, recorded."""
    calls = []

    def pool_pages_from_budget(free_bytes, workspace, page_bytes):
        calls.append((free_bytes, workspace, page_bytes))
        return max(0, free_bytes - workspace) // page_bytes

    stub_module(monkeypatch, "batchgen.prefix_reuse.pool",
                pool_pages_from_budget=pool_pages_from_budget)
    return calls


def test_the_kv_budget_is_free_memory_beside_the_workspace(pool_budget_module):
    log = []
    ns, engine = construction_runtime(log, free=100, total=400)
    outcome = measure(
        ns, engine, budget_context(ns, workspace=16), pool_enabled=False,
    )
    assert outcome._prefill_prefix_kv_budget_bytes == 84
    # The early exit writes no page budget at all, which is NOT the same as
    # writing None: None is what the KV budget takes without a workspace.
    assert outcome._prefill_pool_page_budget is ns["PREFILL_UNCHANGED"]
    assert not pool_budget_module


def test_a_model_without_a_workspace_has_no_kv_budget(pool_budget_module):
    log = []
    ns, engine = construction_runtime(log, free=100)
    outcome = measure(
        ns, engine, budget_context(ns, workspace=0), pool_enabled=False,
    )
    assert outcome._prefill_prefix_kv_budget_bytes is None
    assert outcome._prefill_pool_page_budget is ns["PREFILL_UNCHANGED"]
    assert not pool_budget_module


def test_the_measurement_synchronizes_before_it_reads_free_memory(
    pool_budget_module,
):
    """The ordered sequence of device operations, the synchronize included.

    The free-memory read is only true once every queued kernel has retired, so
    the synchronize and the cache release stay immediately ahead of it, in
    this order, on the device the context names.
    """
    log = []
    ns, engine = construction_runtime(log, free=100)
    measure(ns, engine, budget_context(ns, workspace=16), pool_enabled=False)
    assert log == [
        ("synchronize", "cuda:1"), ("empty_cache",), ("mem_get_info", 1),
    ]


def test_an_enabled_pool_also_measures_its_page_budget(pool_budget_module):
    log = []
    ns, engine = construction_runtime(log, free=100)
    outcome = measure(
        ns, engine, budget_context(ns, workspace=16), pool_enabled=True,
        page_bytes=4,
    )
    assert outcome._prefill_prefix_kv_budget_bytes == 84
    assert outcome._prefill_pool_page_budget == 21
    # The arithmetic is the pool module's own, over the measured free bytes,
    # the planner's workspace and the engine's page size.
    assert pool_budget_module == [(100, 16, 4)]


def test_the_pool_predicate_is_asked_before_the_device_is_touched(
    pool_budget_module,
):
    """Short-circuit order: a predicate that raises leaves the device alone."""
    log = []
    ns, engine = construction_runtime(log)

    def refuse():
        raise RuntimeError("the pool is not available")

    with pytest.raises(RuntimeError, match="not available"):
        engine._measure_prefill_prefix_kv_budget(
            budget_context(ns, workspace=16),
            _prefill_prefix_pool_enabled=refuse,
            _prefill_pool_page_bytes=lambda: 1,
        )
    assert log == []


def test_an_unchanged_budget_is_not_logged_again(pool_budget_module, caplog):
    log = []
    ns, engine = construction_runtime(log, free=100)
    with caplog.at_level("INFO"):
        measure(
            ns, engine,
            budget_context(ns, workspace=16, kv_budget=84, page_budget=21),
            pool_enabled=True, page_bytes=4,
        )
    assert "[PREFIX_KV]" not in caplog.text
    assert "[PREFIX_POOL]" not in caplog.text


# --------------------------------------------------------------------------- #
#  Building the planned wave's pool
# --------------------------------------------------------------------------- #
def construction_context(ns, *, page_budget, rank=2, local_rank=1):
    return ns["PrefillPoolConstructionContext"](
        rank=rank,
        local_rank=local_rank,
        torch_device="cuda:1",
        huggingface_ckpt_name="org/model",
        _prefill_pool_page_budget=page_budget,
    )


@pytest.fixture
def construction_modules(monkeypatch):
    """The three real import seams the construction reaches, recorded."""
    built = {}

    class Executor:
        def __init__(self, plan, prompts, pool, *, pad_poison):
            self.plan, self.prompts, self.pool = plan, prompts, pool
            self.pad_poison = pad_poison
            built["executor"] = self

    def pool_class(**kwargs):
        pool = FakePool(**kwargs)
        built["pool"] = pool
        return pool

    def build_gpu_kv_config(ckpt, page_tokens):
        built.setdefault("geometry_calls", []).append((ckpt, page_tokens))
        return SimpleNamespace(kv_dtype="bf16", num_layers=4)

    stub_module(monkeypatch, "batchgen.prefix_reuse.executor",
                PoolChunkExecutor=Executor)
    stub_module(monkeypatch, "batchgen.prefix_reuse.pool",
                PrefillPrefixPool=pool_class,
                pool_pages_from_budget=lambda *a: 0)
    stub_module(monkeypatch, "batchgen.kv_cache.host_kv_mananger_config",
                build_gpu_kv_config=build_gpu_kv_config)
    return built


def construct(ns, engine, ctx, wave, *, group=None, page_bytes=64,
              layout=(True, (), 0), dims=(4, 2, 8, 8), dump="dump",
              poison=False, seen=None):
    def record(name, result):
        def call(*args, **kwargs):
            if seen is not None:
                seen.append((name, args, kwargs))
            return result
        return call

    return engine._construct_prefill_pool(
        ctx, wave,
        _prefill_pool_group=record("group", group),
        _prefill_pool_page_bytes=record("page_bytes", page_bytes),
        _prefill_pool_layout=record("layout", layout),
        _prefill_pool_page_dims=record("dims", dims),
        _new_pool_kernel_dump_recorder=record("dump", dump),
        _pool_pad_poison_enabled=record("poison", poison),
    )


def test_one_rank_per_group_builds_the_pool_with_no_collective(
    construction_modules,
):
    log = []
    ns, engine = construction_runtime(log)
    wave = pool_wave()
    outcome = construct(ns, engine, construction_context(ns, page_budget=20),
                        wave, group=None)
    assert [entry[0] for entry in log if entry[0] in
            {"all_reduce", "all_gather"}] == []
    assert wave.pool is construction_modules["pool"]
    assert wave.executor is construction_modules["executor"]
    assert construction_modules["pool"].kwargs["num_pages"] == 20
    assert outcome._prefill_pool_base == (3, 7, 400 - 100 - 7)


def test_a_serve_group_reduces_its_budget_to_the_tightest_rank(
    construction_modules,
):
    """The group MIN, and the only condition the collective has."""
    log = []
    ns, engine = construction_runtime(log, reduce_to=12)
    wave = pool_wave(total_pages=20, peak=8, scratch=2)
    construct(ns, engine, construction_context(ns, page_budget=20), wave,
              group="serve-group")
    assert [entry for entry in log if entry[0] == "all_reduce"] == [
        ("all_reduce", "MIN", "serve-group", 20),
    ]
    # The reduced budget is below the plan, and above peak+scratch, so the
    # pool shrinks to it rather than refusing.
    assert construction_modules["pool"].kwargs["num_pages"] == 12


def test_a_rank_with_no_budget_reduces_as_minus_one(construction_modules):
    """``None`` has to survive the integer reduction, as -1."""
    log = []
    ns, engine = construction_runtime(log, reduce_to=-1)
    wave = pool_wave(total_pages=20)
    construct(ns, engine, construction_context(ns, page_budget=None), wave,
              group="serve-group")
    assert log[0] == ("all_reduce", "MIN", "serve-group", -1)
    # A group whose tightest rank has no budget plans the wave's own pages.
    assert construction_modules["pool"].kwargs["num_pages"] == 20


def test_a_budget_below_the_required_peak_is_refused(construction_modules):
    log = []
    ns, engine = construction_runtime(log, reduce_to=5)
    wave = pool_wave(total_pages=20, peak=8, scratch=2)
    with pytest.raises(RuntimeError, match="Rank 2: prefill pool budget shrank"):
        construct(ns, engine, construction_context(ns, page_budget=20), wave,
                  group="serve-group")
    assert "pool" not in construction_modules


def test_a_pool_page_that_disagrees_with_the_budget_is_refused(
    construction_modules,
):
    log = []
    ns, engine = construction_runtime(log)
    with pytest.raises(RuntimeError, match="prefix pool page is 64 B"):
        construct(ns, engine, construction_context(ns, page_budget=20),
                  pool_wave(), page_bytes=99)


def test_the_construction_asks_every_collaborator_in_its_pre_move_order(
    construction_modules,
):
    log, seen = [], []
    ns, engine = construction_runtime(log)
    wave = pool_wave()
    construct(ns, engine, construction_context(ns, page_budget=20), wave,
              poison=True, seen=seen)
    assert [name for name, _, _ in seen] == [
        "group", "page_bytes", "layout", "dims", "dump", "poison",
    ]
    # The geometry both readers are asked about is the one built from the
    # checkpoint the context names and this wave's page size.
    assert construction_modules["geometry_calls"] == [("org/model", [4])]
    assert construction_modules["pool"].kernel_dump == "dump"
    assert construction_modules["executor"].pad_poison is True


def test_the_memory_baseline_is_read_after_the_pool_exists(
    construction_modules,
):
    """Device-memory order: the pool is allocated, then the baseline taken."""
    log = []
    ns, engine = construction_runtime(log)
    construct(ns, engine, construction_context(ns, page_budget=20),
              pool_wave())
    assert log == [
        ("mem_get_info", 1), ("memory_reserved", 1), ("memory_allocated", 1),
        ("reset_peak", 1),
    ]


# --------------------------------------------------------------------------- #
#  The plan-digest comparison
# --------------------------------------------------------------------------- #
def plan(*chunks):
    return SimpleNamespace(
        chunks=[
            SimpleNamespace(
                release_after=release,
                items=[SimpleNamespace(kind=k, ref=r, token_start=s,
                                       token_end=e, prefix_segments=p)
                       for k, r, s, e, p in items],
            )
            for release, items in chunks
        ],
        segments=(1,),
    )


ONE_CHUNK = ((3, [("segment", 7, 0, 64, (1,))]),)


def identity_context(ns, rank=4):
    return ns["PoolPlanIdentityContext"](rank=rank, torch_device="cuda:1")


def test_one_rank_per_group_compares_nothing(monkeypatch):
    log = []
    ns, engine = construction_runtime(log)
    assert engine._check_pool_plan_identical(
        identity_context(ns), plan(*ONE_CHUNK), None, attn_group_size=1,
    ) is None
    assert log == []


def test_a_group_that_agrees_passes_and_gathers_once():
    log = []
    ns, engine = construction_runtime(log)
    engine._check_pool_plan_identical(
        identity_context(ns), plan(*ONE_CHUNK), "serve-group",
        attn_group_size=4,
    )
    assert [entry[0] for entry in log] == ["all_gather"]
    # One slot per rank of the serve group, over this rank's own digest.
    assert log[0][1] == "serve-group"
    assert log[0][3] == 4


def test_a_group_whose_plans_diverge_is_refused():
    log = []
    ns, engine = construction_runtime(log, gather_to=[11, 22])
    with pytest.raises(RuntimeError, match="Rank 4: the prefill prefix pool "
                                           "plan diverged"):
        engine._check_pool_plan_identical(
            identity_context(ns), plan(*ONE_CHUNK), "serve-group",
            attn_group_size=2,
        )


def test_the_digest_is_the_real_hash_over_the_chunk_items():
    """The digest is computed, not restated: two different plans differ."""
    log = []
    ns, engine = construction_runtime(log)
    for plan_chunks in (
        ONE_CHUNK,
        ((3, [("segment", 7, 0, 65, (1,))]),),
        ((4, [("segment", 7, 0, 64, (1,))]),),
    ):
        engine._check_pool_plan_identical(
            identity_context(ns), plan(*plan_chunks), "g", attn_group_size=1,
        )
    digests = [entry[2] for entry in log]
    assert len(set(digests)) == 3, digests


# --------------------------------------------------------------------------- #
#  The pool row layout
# --------------------------------------------------------------------------- #
def layout_context(ns, *, model=None, cached=None):
    return ns["PrefillPoolLayoutContext"](
        model=model, _prefill_pool_layout_cache=cached,
    )


def resolve_layout(engine, ctx, *, dual=True, psm=None):
    return engine._prefill_pool_layout(
        ctx, SimpleNamespace(num_layers=2, logical_to_physical_layer=None),
        _prefill_host_view_is_dual=lambda: dual,
        _prefill_pool_psm_layout=lambda: psm,
    )


def dsa_model(*, indexer_dims):
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(module=SimpleNamespace(
            indexer=None if dim is None
            else SimpleNamespace(index_head_dim=dim))))
        for dim in indexer_dims
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_a_non_dsa_model_pools_kv_and_caches_nothing():
    ns, engine = runtime()
    layout, outcome = resolve_layout(
        engine, layout_context(ns, model=Explode("model")), dual=False,
    )
    assert layout == (True, (), 0)
    assert outcome._prefill_pool_layout_cache is ns["PREFILL_UNCHANGED"]


def test_a_deep_freed_model_answers_from_the_cache_and_rewrites_nothing():
    ns, engine = runtime()
    layout, outcome = resolve_layout(
        engine, layout_context(ns, model=None, cached=(True, (0, 1), 64)),
        psm={"v_head_dim": 128},
    )
    assert layout == (True, (0, 1), 64)
    assert outcome._prefill_pool_layout_cache is ns["PREFILL_UNCHANGED"]


def test_a_deep_freed_model_with_no_cache_is_refused():
    """The pre-move read took the cache with a default, so absent is False,
    not an AttributeError: the refusal below is what must be reached."""
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="requested before the first"):
        resolve_layout(
            engine, layout_context(ns, model=None, cached=None),
            psm={"v_head_dim": 128},
        )


def test_the_dsa_walk_reports_its_layout_in_the_outcome():
    ns, engine = runtime()
    layout, outcome = resolve_layout(
        engine,
        layout_context(ns, model=dsa_model(indexer_dims=[64, None])),
        psm={"v_head_dim": 128},
    )
    assert layout == (True, (0,), 64)
    assert outcome._prefill_pool_layout_cache == layout


def test_the_pool_carries_v_only_where_the_psm_declares_it():
    ns, engine = runtime()
    layout, _ = resolve_layout(
        engine,
        layout_context(ns, model=dsa_model(indexer_dims=[64, None])),
        psm={"v_head_dim": 0},
    )
    assert layout[0] is False


def test_mixed_indexer_head_dims_are_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="mixed indexer head dims"):
        resolve_layout(
            engine,
            layout_context(ns, model=dsa_model(indexer_dims=[64, 32])),
            psm={"v_head_dim": 128},
        )


def test_a_dsa_model_whose_depth_disagrees_is_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="do not match the KV geometry"):
        resolve_layout(
            engine,
            layout_context(ns, model=dsa_model(indexer_dims=[64, 64, 64])),
            psm={"v_head_dim": 128},
        )


# --------------------------------------------------------------------------- #
#  The wave's segment Host pages
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakePrefixState:
    planned_shared_page_ids: tuple = ()


class RecordingPrefix:
    """``PrefixReuseState``: the owner whose own methods take every write."""

    def __init__(self):
        self.installed = []
        self.refs = []

    def install_state(self, global_id, state):
        self.installed.append((global_id, state))

    def add_pool_member_refs(self, global_id, sids):
        self.refs.append((global_id, sids))


def host_page_wave(*, chains, materialized=(), segment_sids=None):
    return SimpleNamespace(
        global_ids=[100 + index for index in range(len(chains))],
        chains=list(chains),
        materialized=set(materialized),
        segment_sids=segment_sids or {},
        chain_host_pages=lambda index: {0: (71, 72)},
    )


@pytest.fixture
def allocation_module(monkeypatch):
    calls = []

    def allocate_segment_host_pages(wave, host_views, next_sid):
        calls.append((wave, host_views, next_sid))
        return 2

    stub_module(monkeypatch, "batchgen.prefix_reuse.wave_runtime",
                allocate_segment_host_pages=allocate_segment_host_pages)
    return calls


def allocate(ns, engine, wave, prefix, states, *, allocator, planned=2):
    return engine._allocate_prefill_pool_host_pages(
        ns["PrefillPoolHostPageContext"](
            _prefill_pool_wave=wave, prefix=prefix,
        ),
        {"views": 1},
        states,
        _next_pool_segment_sid=allocator,
        _prefill_pool_segment_host_pages=lambda: planned,
    )


def test_the_segment_id_allocator_is_handed_on_as_a_callable(
    allocation_module,
):
    """The helper calls it once per segment, so the body passes it, uncalled."""
    ns, engine = runtime()
    prefix, sentinel = RecordingPrefix(), object()
    allocate(ns, engine, host_page_wave(chains=[()]), prefix,
             {100: FakePrefixState()}, allocator=sentinel)
    (wave, views, handed), = allocation_module
    assert handed is sentinel
    assert views == {"views": 1}


def test_an_allocation_that_disagrees_with_the_plan_is_refused(
    allocation_module,
):
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="got 2 Host pages, planned 5"):
        allocate(ns, engine, host_page_wave(chains=[()]), RecordingPrefix(),
                 {}, allocator=object(), planned=5)


def test_each_member_installs_its_chain_through_the_owner(allocation_module):
    ns, engine = runtime()
    prefix = RecordingPrefix()
    states = {100: FakePrefixState(), 101: FakePrefixState()}
    wave = host_page_wave(chains=[(7, 8), ()], segment_sids={7: -3, 8: -5})
    allocate(ns, engine, wave, prefix, states, allocator=object())
    # Only the member with a chain is touched, and its new state is both in
    # the caller's map and installed through the owner.
    assert [global_id for global_id, _ in prefix.installed] == [100]
    assert states[100].planned_shared_page_ids == {0: (71, 72)}
    assert prefix.installed[0][1] is states[100]
    assert prefix.refs == [(100, (-3, -5))]


def test_a_materialized_segment_holds_no_pseudo_sequence_reference(
    allocation_module,
):
    """Its pages are the prefix cache's, kept by the member's own attachment."""
    ns, engine = runtime()
    prefix = RecordingPrefix()
    wave = host_page_wave(chains=[(7, 8)], materialized={8},
                          segment_sids={7: -3, 8: -5})
    allocate(ns, engine, wave, prefix, {100: FakePrefixState()},
             allocator=object())
    assert prefix.refs == [(100, (-3,))]


# --------------------------------------------------------------------------- #
#  The load staging
# --------------------------------------------------------------------------- #
@pytest.fixture
def staging_modules(monkeypatch):
    built = {}

    class Loader:
        staging_bytes = 128
        slab_pages = 2

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            built.setdefault("primary", []).append(self)

    class KeyLoader:
        staging_bytes = 64

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            built.setdefault("aux", []).append(self)

    def build_gpu_kv_config(ckpt, page_tokens):
        built.setdefault("primary_geometry", []).append((ckpt, page_tokens))
        return SimpleNamespace(num_layers=4, num_k_heads=2, k_head_dim=8,
                               kv_dtype="bf16")

    def build_gpu_kv_config_aux(ckpt, page_tokens):
        built.setdefault("aux_geometry", []).append((ckpt, page_tokens))
        return built.get("aux_profile", SimpleNamespace(
            num_layers=4, k_head_dim=72, kv_dtype="float8_e4m3fn"))

    stub_module(monkeypatch, "batchgen.kv_cache.host_kv_mananger_config",
                build_gpu_kv_config=build_gpu_kv_config,
                build_gpu_kv_config_aux=build_gpu_kv_config_aux)
    stub_module(monkeypatch, "batchgen.prefix_reuse.pool_materialize",
                DEFAULT_SLAB_BYTES=4096, MaterializedSegmentLoader=Loader,
                PageSplitIndexKeyLoader=KeyLoader)
    return built


def staging_context(ns, rank=5):
    return ns["MaterializedSegmentStagingContext"](
        rank=rank, torch_device="cuda:1", huggingface_ckpt_name="org/model",
    )


def stage(ns, engine, *, views, materialized=(3,)):
    log = []
    return engine._materialized_segment_loader(
        staging_context(ns),
        SimpleNamespace(materialized=set(materialized), page_tokens=4),
        _host_views_by_group=lambda: views,
    ), log


def test_a_wave_that_materializes_nothing_stages_nothing(staging_modules):
    log = []
    ns, engine = construction_runtime(log)
    loaders, _ = stage(ns, engine, views={0: "primary"}, materialized=())
    assert loaders is None
    # Nothing is imported or built when there is nothing to load.
    assert staging_modules == {}


def test_a_single_region_model_stages_the_primary_latent(staging_modules):
    log = []
    ns, engine = construction_runtime(log)
    loaders, _ = stage(ns, engine, views={0: "primary"})
    assert sorted(loaders) == [0]
    assert staging_modules["primary_geometry"] == [("org/model", [4])]
    assert loaders[0].kwargs == {
        "worker_view": "primary", "num_layers": 4, "page_tokens": 4,
        "latent_dim": 16, "dtype": "bf16", "device": "cuda:1",
        "max_slab_bytes": 4096,
    }
    assert "aux" not in staging_modules


def test_a_model_with_an_unexpected_region_set_is_refused(staging_modules):
    log = []
    ns, engine = construction_runtime(log)
    with pytest.raises(RuntimeError, match=r"this model has groups \[0, 1, 2\]"):
        stage(ns, engine, views={0: "a", 1: "b", 2: "c"})


def test_a_dsa_model_adds_the_page_split_index_key_loader(staging_modules):
    log = []
    ns, engine = construction_runtime(log)
    loaders, _ = stage(ns, engine, views={0: "primary", 1: "aux"})
    assert sorted(loaders) == [0, 1]
    # The carrier width comes from the engine's own stateless sibling, which
    # reads it off the auxiliary profile.
    assert loaders[1].kwargs["carrier_bytes"] == 72
    assert loaders[1].kwargs["slab_pages"] == 2
    assert staging_modules["aux_geometry"] == [("org/model", [4])]


def test_a_dsa_model_with_no_indexer_profile_is_refused(staging_modules):
    log = []
    ns, engine = construction_runtime(log)
    staging_modules["aux_profile"] = None
    with pytest.raises(RuntimeError, match="no DSA indexer profile"):
        stage(ns, engine, views={0: "primary", 1: "aux"})


def test_an_indexer_region_of_another_depth_is_refused(staging_modules):
    log = []
    ns, engine = construction_runtime(log)
    staging_modules["aux_profile"] = SimpleNamespace(
        num_layers=3, k_head_dim=72, kv_dtype="float8_e4m3fn")
    with pytest.raises(RuntimeError, match="keeps 3 row"):
        stage(ns, engine, views={0: "primary", 1: "aux"})


# --------------------------------------------------------------------------- #
#  The index-key carrier (no context at all)
# --------------------------------------------------------------------------- #
def test_the_carrier_is_the_page_split_byte_width():
    ns, engine = runtime()
    assert engine._prefix_pool_index_key_carrier(
        SimpleNamespace(kv_dtype=torch.float8_e4m3fn, k_head_dim=72)
    ) == 72


def test_an_aos_indexer_region_is_refused():
    """It declares ELEMENTS and carries no scale: reading it as page-split
    bytes would reinterpret every key."""
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="needs the page-split"):
        engine._prefix_pool_index_key_carrier(
            SimpleNamespace(kv_dtype=torch.bfloat16, k_head_dim=64)
        )


def test_the_carrier_takes_no_seam_parameter_at_all():
    node = method_node("_prefix_pool_index_key_carrier")
    assert [arg.arg for arg in node.args.args] == ["self", "aux_geometry"]
    assert not node.args.kwonlyargs


# --------------------------------------------------------------------------- #
#  The index-key layer walk
# --------------------------------------------------------------------------- #
class FakeIndexKeyLayer:
    def __init__(self):
        self.calls = []

    def materialize_pool_segment_index_keys(self, *, index_keys, slots):
        self.calls.append((index_keys, slots))


class FakeLatentLayer:
    def __init__(self):
        self.calls = []

    def materialize_pool_segment(self, *, latent, slots):
        self.calls.append((latent, slots))


def key_layer_model(flags):
    """One decoder layer per flag; True = this layer materializes keys."""
    return SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=FakeIndexKeyLayer() if flag
                        else FakeLatentLayer())
        for flag in flags
    ]))


def index_key_layers(ns, engine, model, pool):
    return engine._materialized_pool_index_key_layers(
        ns["MaterializedIndexKeyLayersContext"](model=model), pool,
    )


def test_a_pool_without_indexer_rows_cannot_take_index_keys():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="needs the pool's indexer rows"):
        index_key_layers(ns, engine, Explode("model"),
                         SimpleNamespace(indexer_row={}))
    with pytest.raises(RuntimeError, match="needs the pool's indexer rows"):
        index_key_layers(ns, engine, Explode("model"), None)


def test_the_walk_returns_the_key_layers_in_pool_row_order():
    ns, engine = runtime()
    model = key_layer_model([True, False, True])
    layers = index_key_layers(
        ns, engine, model, SimpleNamespace(indexer_row={0: 0, 2: 1}),
    )
    assert layers == [model.model.layers[0].self_attn,
                      model.model.layers[2].self_attn]


def test_a_key_layer_with_no_pool_row_is_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="no prefix-pool indexer row"):
        index_key_layers(ns, engine, key_layer_model([True, True]),
                         SimpleNamespace(indexer_row={0: 0}))


def test_a_pool_row_out_of_engine_order_is_refused():
    """The n-th key-materializing layer MUST be indexer row n."""
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="is pool indexer row 1 but the 0th"):
        index_key_layers(ns, engine, key_layer_model([True, True]),
                         SimpleNamespace(indexer_row={0: 1, 1: 0}))


def test_more_pool_rows_than_key_layers_is_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match=r"1 layer\(s\) materialize index "
                                          r"keys against 2 pool"):
        index_key_layers(ns, engine, key_layer_model([True, False]),
                         SimpleNamespace(indexer_row={0: 0, 1: 1}))


# --------------------------------------------------------------------------- #
#  Filling a chunk's loaded segments
# --------------------------------------------------------------------------- #
class Slab:
    def __init__(self, token_start, tokens, *, rows):
        self.token_start, self.tokens = token_start, tokens
        self.latent = rows
        self.keys = rows


class SlabLoader:
    """A loader whose ``slabs`` is a generator, like the real one.

    The real generator issues the H2D and WAITS on it before it yields, so the
    order these are created in and the lockstep in which they are drained is
    what the test records.
    """

    def __init__(self, name, log, slabs):
        self.name, self.log, self._slabs = name, log, slabs

    def slabs(self, *, host_page_ids, tokens):
        self.log.append(("create", self.name, tuple(host_page_ids), tokens))

        def stream():
            for slab in self._slabs:
                self.log.append(("wait", self.name, slab.token_start))
                yield slab
        return stream()


def load_rows(*, materialized):
    return SimpleNamespace(materialized=list(materialized))


def segment(segment_id=3, *, tokens=8, slots=None, pages=(71, 72)):
    return SimpleNamespace(segment_id=segment_id, tokens=tokens,
                           slots=list(slots if slots is not None
                                      else range(tokens)),
                           pages=list(pages))


def load(ns, engine, *, rows, loaders, host_pages, layers=(), key_layers=(),
         pool="pool"):
    return engine._load_materialized_segments(
        ns["MaterializedSegmentLoadContext"](rank=5),
        SimpleNamespace(segment_host_pages=host_pages, pool=pool),
        rows, loaders,
        _materialized_pool_layers=lambda: list(layers),
        _materialized_pool_index_key_layers=lambda p: list(key_layers),
    )


def test_a_chunk_with_no_loaded_segment_loads_nothing():
    ns, engine = runtime()
    assert load(ns, engine, rows=load_rows(materialized=()), loaders=None,
                host_pages=Explode("host pages")) == 0


def test_a_chunk_that_loads_without_staging_is_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="Rank 5: this chunk loads 1 "):
        load(ns, engine, rows=load_rows(materialized=[segment()]),
             loaders=None, host_pages={})


def test_a_segment_with_no_host_page_is_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match="names no Host page to load from"):
        load(ns, engine, rows=load_rows(materialized=[segment()]),
             loaders={0: "primary"}, host_pages={})


def test_a_segment_whose_regions_do_not_match_the_loaders_is_refused():
    ns, engine = runtime()
    with pytest.raises(RuntimeError, match=r"names Host pages in groups \[0\], "
                                          r"this model loads groups \[0, 1\]"):
        load(ns, engine, rows=load_rows(materialized=[segment()]),
             loaders={0: "a", 1: "b"}, host_pages={3: {0: (71, 72)}})


def test_one_region_loads_each_slab_into_the_window_its_tokens_name():
    log = []
    ns, engine = runtime()
    latent = [FakeLatentLayer(), FakeLatentLayer()]
    loader = SlabLoader("primary", log, [
        Slab(0, 4, rows=["row0-a", "row1-a"]),
        Slab(4, 4, rows=["row0-b", "row1-b"]),
    ])
    loaded = load(
        ns, engine, rows=load_rows(materialized=[segment(tokens=8)]),
        loaders={0: loader}, host_pages={3: {0: (71, 72)}}, layers=latent,
    )
    assert loaded == 8
    assert [slots for _, slots in latent[0].calls] == [[0, 1, 2, 3],
                                                       [4, 5, 6, 7]]
    assert [keys for keys, _ in latent[1].calls] == ["row1-a", "row1-b"]
    assert log == [
        ("create", "primary", (71, 72), 8),
        ("wait", "primary", 0), ("wait", "primary", 4),
    ]


def test_both_regions_are_created_in_group_order_and_drained_in_lockstep():
    """The ordering invariant of the load: one zip over both slab streams.

    Each stream's own H2D-and-wait happens inside ``slabs``, so the order the
    streams are created in and the lockstep they are consumed in is what pairs
    a latent slab with its index keys.
    """
    log = []
    ns, engine = runtime()
    latent, keys = [FakeLatentLayer()], [FakeIndexKeyLayer()]
    loaders = {
        0: SlabLoader("primary", log, [Slab(0, 4, rows=["L0"]),
                                       Slab(4, 4, rows=["L1"])]),
        1: SlabLoader("aux", log, [Slab(0, 4, rows=["K0"]),
                                   Slab(4, 4, rows=["K1"])]),
    }
    loaded = load(
        ns, engine, rows=load_rows(materialized=[segment(tokens=8)]),
        loaders=loaders, host_pages={3: {0: (71, 72), 1: (71, 72)}},
        layers=latent, key_layers=keys,
    )
    assert loaded == 8
    assert log == [
        ("create", "primary", (71, 72), 8),
        ("create", "aux", (71, 72), 8),
        ("wait", "primary", 0), ("wait", "aux", 0),
        ("wait", "primary", 4), ("wait", "aux", 4),
    ]
    # Both regions wrote the SAME window, in the same iteration.
    assert [slots for _, slots in latent[0].calls] == [
        [0, 1, 2, 3], [4, 5, 6, 7],
    ]
    assert [slots for _, slots in keys[0].calls] == [
        [0, 1, 2, 3], [4, 5, 6, 7],
    ]


def test_an_index_key_slab_that_does_not_pair_with_its_latent_is_refused():
    log = []
    ns, engine = runtime()
    loaders = {
        0: SlabLoader("primary", log, [Slab(0, 4, rows=["L0"])]),
        1: SlabLoader("aux", log, [Slab(0, 2, rows=["K0"])]),
    }
    with pytest.raises(RuntimeError, match="loaded index keys for"):
        load(ns, engine, rows=load_rows(materialized=[segment(tokens=4)]),
             loaders=loaders, host_pages={3: {0: (71,), 1: (71,)}},
             layers=[FakeLatentLayer()], key_layers=[FakeIndexKeyLayer()])


def test_a_single_region_load_never_asks_for_the_index_key_layers():
    """The walk is consulted only when the auxiliary region is loaded."""
    log = []
    ns, engine = runtime()

    def refuse(pool):
        raise AssertionError("the index-key walk ran on a single-region load")

    assert engine._load_materialized_segments(
        ns["MaterializedSegmentLoadContext"](rank=5),
        SimpleNamespace(segment_host_pages={3: {0: (71,)}}, pool="pool"),
        load_rows(materialized=[segment(tokens=4)]),
        {0: SlabLoader("primary", log, [Slab(0, 4, rows=[])])},
        _materialized_pool_layers=lambda: [],
        _materialized_pool_index_key_layers=refuse,
    ) == 4


# --------------------------------------------------------------------------- #
#  D7.3: the nine worker delegators
# --------------------------------------------------------------------------- #
def test_the_budget_delegator_applies_both_writes_in_the_pre_move_order(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank, worker.local_rank = 3, 1
    worker.torch_device = "cuda:1"
    worker.engine_config = "config"
    worker._prefill_prefix_kv_budget_bytes = None
    worker._prefill_pool_page_budget = None
    seen = []
    recording(engine, "_measure_prefill_prefix_kv_budget", seen,
              result=ns["PrefillPrefixKVBudgetOutcome"](84, 21))
    worker._measure_prefill_prefix_kv_budget()
    (context,), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPrefixKVBudgetContext"])
    assert (context.rank, context.local_rank, context.torch_device,
            context.engine_config,
            context._prefill_prefix_kv_budget_bytes,
            context._prefill_pool_page_budget) == (
        3, 1, "cuda:1", "config", None, None,
    )
    assert kwargs == {
        "_prefill_prefix_pool_enabled": worker._prefill_prefix_pool_enabled,
        "_prefill_pool_page_bytes": worker._prefill_pool_page_bytes,
    }
    assert worker._prefill_prefix_kv_budget_bytes == 84
    assert worker._prefill_pool_page_budget == 21


def test_the_budget_delegator_leaves_the_page_budget_on_the_early_exit(
    monkeypatch,
):
    """The sentinel means "this path wrote nothing", not "wrote None"."""
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank, worker.local_rank = 0, 0
    worker.torch_device = "cuda:0"
    worker.engine_config = "config"
    worker._prefill_prefix_kv_budget_bytes = 1
    worker._prefill_pool_page_budget = 17
    recording(engine, "_measure_prefill_prefix_kv_budget", [],
              result=ns["PrefillPrefixKVBudgetOutcome"](None))
    worker._measure_prefill_prefix_kv_budget()
    assert worker._prefill_prefix_kv_budget_bytes is None
    assert worker._prefill_pool_page_budget == 17


def test_the_budget_delegator_reads_both_budgets_live_each_call(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank, worker.local_rank = 0, 0
    worker.torch_device = "cuda:0"
    worker.engine_config = "config"
    worker._prefill_prefix_kv_budget_bytes = None
    worker._prefill_pool_page_budget = None
    seen = []
    recording(engine, "_measure_prefill_prefix_kv_budget", seen,
              result=ns["PrefillPrefixKVBudgetOutcome"](84, 21))
    worker._measure_prefill_prefix_kv_budget()
    worker._measure_prefill_prefix_kv_budget()
    assert [args[0]._prefill_prefix_kv_budget_bytes for args, _ in seen] == [
        None, 84,
    ]
    assert [args[0]._prefill_pool_page_budget for args, _ in seen] == [
        None, 21,
    ]


def test_the_construction_delegator_passes_six_collaborators(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank, worker.local_rank = 2, 1
    worker.torch_device = "cuda:1"
    worker.huggingface_ckpt_name = "org/model"
    worker._prefill_pool_page_budget = 20
    worker._prefill_pool_base = (0, 0, 0)
    worker._new_pool_kernel_dump_recorder = lambda wave: None
    seen = []
    recording(engine, "_construct_prefill_pool", seen,
              result=ns["PrefillPoolBaseOutcome"]((1, 2, 3)))
    worker._construct_prefill_pool("wave")
    (context, wave), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPoolConstructionContext"])
    assert (context.rank, context.local_rank, context.torch_device,
            context.huggingface_ckpt_name,
            context._prefill_pool_page_budget) == (
        2, 1, "cuda:1", "org/model", 20,
    )
    assert wave == "wave"
    assert kwargs == {
        "_prefill_pool_group": worker._prefill_pool_group,
        "_prefill_pool_page_bytes": worker._prefill_pool_page_bytes,
        "_prefill_pool_layout": worker._prefill_pool_layout,
        "_prefill_pool_page_dims": worker._prefill_pool_page_dims,
        "_new_pool_kernel_dump_recorder": worker._new_pool_kernel_dump_recorder,
        "_pool_pad_poison_enabled": worker._pool_pad_poison_enabled,
    }
    assert worker._prefill_pool_base == (1, 2, 3)


def test_the_construction_delegator_leaves_the_baseline_on_the_sentinel(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank, worker.local_rank = 0, 0
    worker.torch_device = "cuda:0"
    worker.huggingface_ckpt_name = "org/model"
    worker._prefill_pool_page_budget = None
    worker._prefill_pool_base = (9, 9, 9)
    worker._new_pool_kernel_dump_recorder = lambda wave: None
    recording(engine, "_construct_prefill_pool", [],
              result=ns["PrefillPoolBaseOutcome"]())
    worker._construct_prefill_pool("wave")
    assert worker._prefill_pool_base == (9, 9, 9)


def test_the_plan_identity_delegator_builds_its_narrow_context(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 4
    worker.torch_device = "cuda:1"
    worker._prefill_attn_tp_size = lambda: 8
    seen = []
    recording(engine, "_check_pool_plan_identical", seen)
    worker._check_pool_plan_identical("plan", "group")
    (context, plan_arg, group), kwargs = seen[0]
    assert isinstance(context, ns["PoolPlanIdentityContext"])
    assert (context.rank, context.torch_device) == (4, "cuda:1")
    assert (plan_arg, group) == ("plan", "group")
    # The group WIDTH travels as a value beside the context, as it did.
    assert kwargs == {"attn_group_size": 8}


def test_the_layout_delegator_keeps_the_cache_default_and_applies_the_write(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.model = "model"
    worker._prefill_host_view_is_dual = lambda: True
    assert not hasattr(worker, "_prefill_pool_layout_cache")
    seen = []
    recording(engine, "_prefill_pool_layout", seen,
              result=((True, (0,), 64),
                      ns["PrefillPoolLayoutOutcome"]((True, (0,), 64))))
    assert worker._prefill_pool_layout("geometry") == (True, (0,), 64)
    (context, geometry), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPoolLayoutContext"])
    # Absent stays absent: the pre-move read took it with this default.
    assert (context.model, context._prefill_pool_layout_cache) == (
        "model", None,
    )
    assert geometry == "geometry"
    assert kwargs == {
        "_prefill_host_view_is_dual": worker._prefill_host_view_is_dual,
        "_prefill_pool_psm_layout": worker._prefill_pool_psm_layout,
    }
    assert worker._prefill_pool_layout_cache == (True, (0,), 64)


def test_the_layout_delegator_creates_no_cache_on_the_early_exits(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.model = None
    worker._prefill_host_view_is_dual = lambda: False
    recording(engine, "_prefill_pool_layout", [],
              result=((True, (), 0), ns["PrefillPoolLayoutOutcome"]()))
    assert worker._prefill_pool_layout("geometry") == (True, (), 0)
    assert not hasattr(worker, "_prefill_pool_layout_cache")


def test_the_host_page_delegator_passes_the_allocator_uncalled(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker._prefill_pool_wave = "wave"
    worker._prefix = "prefix state"
    seen = []
    recording(engine, "_allocate_prefill_pool_host_pages", seen)
    worker._allocate_prefill_pool_host_pages({"views": 1}, {"states": 2})
    (context, views, states), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPoolHostPageContext"])
    assert (context._prefill_pool_wave, context.prefix) == (
        "wave", "prefix state",
    )
    assert (views, states) == ({"views": 1}, {"states": 2})
    assert kwargs == {
        "_next_pool_segment_sid": worker._next_pool_segment_sid,
        "_prefill_pool_segment_host_pages":
            worker._prefill_pool_segment_host_pages,
    }


def test_the_host_page_delegator_reads_the_live_wave_each_call(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker._prefix = "prefix state"
    seen = []
    recording(engine, "_allocate_prefill_pool_host_pages", seen)
    worker._prefill_pool_wave = "first"
    worker._allocate_prefill_pool_host_pages({}, {})
    worker._prefill_pool_wave = "second"
    worker._allocate_prefill_pool_host_pages({}, {})
    assert [args[0]._prefill_pool_wave for args, _ in seen] == [
        "first", "second",
    ]


def test_the_staging_delegator_passes_the_host_view_delegator(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 5
    worker.torch_device = "cuda:1"
    worker.huggingface_ckpt_name = "org/model"
    worker._host_views_by_group = lambda: {0: "primary"}
    seen = []
    recording(engine, "_materialized_segment_loader", seen)
    assert worker._materialized_segment_loader("wave") == "recorded"
    (context, wave), kwargs = seen[0]
    assert isinstance(context, ns["MaterializedSegmentStagingContext"])
    assert (context.rank, context.torch_device,
            context.huggingface_ckpt_name) == (5, "cuda:1", "org/model")
    assert wave == "wave"
    assert kwargs == {"_host_views_by_group": worker._host_views_by_group}


def test_the_carrier_delegator_passes_no_context_at_all(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    seen = []
    recording(engine, "_prefix_pool_index_key_carrier", seen)
    assert worker._prefix_pool_index_key_carrier("aux") == "recorded"
    assert seen[0] == (("aux",), {})


def test_the_index_key_layer_delegator_reads_the_live_model(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    seen = []
    recording(engine, "_materialized_pool_index_key_layers", seen)
    worker.model = "first"
    worker._materialized_pool_index_key_layers("pool")
    worker.model = "second"
    worker._materialized_pool_index_key_layers("pool")
    contexts = [args[0] for args, _ in seen]
    assert all(isinstance(c, ns["MaterializedIndexKeyLayersContext"])
               for c in contexts)
    assert [c.model for c in contexts] == ["first", "second"]
    assert [args[1] for args, _ in seen] == ["pool", "pool"]
    assert seen[0][1] == {}


def test_the_chunk_load_delegator_passes_both_layer_walks(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 5
    worker._materialized_pool_layers = lambda: []
    seen = []
    recording(engine, "_load_materialized_segments", seen)
    assert worker._load_materialized_segments("wave", "rows", "loaders") == (
        "recorded"
    )
    (context, wave, rows, loaders), kwargs = seen[0]
    assert isinstance(context, ns["MaterializedSegmentLoadContext"])
    assert context.rank == 5
    assert (wave, rows, loaders) == ("wave", "rows", "loaders")
    assert kwargs == {
        "_materialized_pool_layers": worker._materialized_pool_layers,
        "_materialized_pool_index_key_layers":
            worker._materialized_pool_index_key_layers,
    }


def test_every_d7_3_delegator_names_no_worker_in_its_engine_call(monkeypatch):
    """The delegator hands over VALUES and CALLABLES, never itself.

    One assertion over the whole stage: no engine call in any of the nine
    delegators passes a bare ``self``, which is the shape this phase removes.
    """
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in D7_3_METHODS:
            continue
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "engine"):
                continue
            assert not any(isinstance(arg, ast.Name) and arg.id == "self"
                           for arg in call.args), node.name
            assert not any(isinstance(kw.value, ast.Name)
                           and kw.value.id == "self"
                           for kw in call.keywords), node.name


# --------------------------------------------------------------------------- #
#  D7.4: the stage's own seam shape
# --------------------------------------------------------------------------- #
D7_4_CONTEXTS = {
    "_plan_prefill_micro_batches": "PrefillMicroBatchPlanContext",
    "_plan_prefill_pool_wave": "PrefillPoolWavePlanContext",
    "_finish_prefill_pool_wave": "PrefillPoolWaveFinishContext",
    "_record_unsharded_epilogue_gather": "UnshardedEpilogueGatherContext",
    # The three probe recorders SHARE D7.1's probe context.
    "_new_pool_epilogue_tracer": "PoolProbeContext",
    "_new_pool_kernel_dump_recorder": "PoolProbeContext",
    "_new_pool_offload_check_recorder": "PoolProbeContext",
}
# The four pooled instruments. The three recorders coincide on the probe
# context; the gather record does not.
PROBE_RECORDERS = (
    "_new_pool_epilogue_tracer",
    "_new_pool_kernel_dump_recorder",
    "_new_pool_offload_check_recorder",
)


@pytest.mark.parametrize("name", sorted(D7_4_CONTEXTS))
def test_each_wave_operation_declares_the_context_it_reads(name):
    node = method_node(name)
    assert ast.unparse(node.args.args[1].annotation) == D7_4_CONTEXTS[name]


def test_the_three_probe_recorders_share_the_probe_context_d71_declared():
    """Reuse because the read sets coincide, not because the family does.

    D7.1 wrote ``PoolProbeContext`` for the probe read and the pad-poison
    predicate and said the recorders read the same two fields. They do: each
    reads ``ctx.rank`` and reaches the probe state only by handing this same
    context to the probe read, so the union over all five bodies is still
    exactly the two declared fields -- which
    ``test_no_context_carries_a_field_its_operations_do_not_read`` checks
    mechanically.
    """
    ns, _ = runtime()
    assert {field.name for field in fields(ns["PoolProbeContext"])} == {
        "rank", "_phase_prefill_debug",
    }
    for name in PROBE_RECORDERS:
        node = method_node(name)
        assert ast.unparse(node.args.args[1].annotation) == "PoolProbeContext"
        # It reaches the probe state ONLY through the sibling read, with this
        # same context -- it never names the field itself.
        reads = {
            n.attr for n in ast.walk(node)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "ctx"
        }
        assert reads == {"rank"}, (name, reads)
        sibling = [
            call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_pool_probe_debug"
        ]
        assert len(sibling) == 1, name
        assert [ast.unparse(arg) for arg in sibling[0].args] == ["ctx", "wave"]


def test_the_unsharded_gather_record_does_not_share_the_probe_context():
    """The fourth instrument reads the manager and never the probe state."""
    ns, _ = runtime()
    node = method_node("_record_unsharded_epilogue_gather")
    assert ast.unparse(node.args.args[1].annotation) == (
        "UnshardedEpilogueGatherContext"
    )
    assert {f.name for f in fields(ns["UnshardedEpilogueGatherContext"])} == {
        "rank", "parallel_manager",
    }
    assert not any(
        isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id == "ctx" and n.attr == "_phase_prefill_debug"
        for n in ast.walk(node)
    )
    # And it never consults the shared probe read either.
    assert not any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and call.func.attr == "_pool_probe_debug"
        for call in ast.walk(node)
    )


def test_no_earlier_stages_context_was_widened_for_the_wave_stage():
    """D7.1-D7.3's contexts keep exactly the fields they were declared with.

    The one reuse of this stage adds no field, and the three tempting shares
    were all rejected: the micro-batch planner reads the PSM handle, which is
    ``PrefillPoolParallelContext``'s whole read set, but it also reads the
    prefix KV budget; the wave planner reads the pool page budget, which the
    construction context carries, but nothing else of it; the wave epilogue
    reads the planned wave and the prefix owner, which is
    ``PrefillPoolHostPageContext``, but also the pool, the coordinator and
    the runtime config.
    """
    ns, _ = runtime()
    assert {f.name for f in fields(ns["PrefillPoolParallelContext"])} == {
        "parallel_manager",
    }
    assert [f.name for f in fields(ns["PrefillPoolWaveContext"])] == [
        "_prefill_pool_wave",
    ]
    assert [f.name for f in fields(ns["PrefillPoolHostPageContext"])] == [
        "_prefill_pool_wave", "prefix",
    ]
    assert [f.name for f in fields(ns["PrefillPoolConstructionContext"])] == [
        "rank", "local_rank", "torch_device", "huggingface_ckpt_name",
        "_prefill_pool_page_budget",
    ]
    # The three contexts this stage declares are annotated nowhere earlier.
    earlier = {
        ast.unparse(method_node(other).args.args[1].annotation)
        for other in sorted(
            (D7_1_METHODS | D7_2_METHODS | D7_3_METHODS) - STATELESS_METHODS
        )
    }
    for name in (
        "PrefillMicroBatchPlanContext", "PrefillPoolWavePlanContext",
        "PrefillPoolWaveFinishContext", "UnshardedEpilogueGatherContext",
    ):
        assert name not in earlier, name


def test_the_wave_stage_declares_exactly_one_outcome():
    ns, _ = runtime()
    assert OUTCOME_METHODS & D7_4_METHODS == {"_finish_prefill_pool_wave"}
    assert [f.name for f in fields(ns["PrefillPoolWaveFinishOutcome"])] == [
        "_prefill_pool_wave",
    ]


def test_the_wave_planner_writes_nothing_and_returns_the_wave():
    """The planned wave is the planner's RETURN, not a worker write.

    The no-worker guard already proves the body stores through no seam; this
    pins the positive half, that the object the caller binds comes back as the
    return value of the one pooled path.
    """
    node = method_node("_plan_prefill_pool_wave")
    returns = [
        stmt for stmt in ast.walk(node) if isinstance(stmt, ast.Return)
    ]
    returns.sort(key=lambda stmt: stmt.lineno)
    assert [ast.unparse(stmt.value) for stmt in returns] == [
        "None", "None", "None", "wave",
    ]
    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id.endswith("Outcome")
        for n in ast.walk(node)
    )


# --------------------------------------------------------------------------- #
#  D7.4: the ONE collective keeps its position and its one condition
# --------------------------------------------------------------------------- #
def test_the_wave_stage_carries_exactly_one_collective():
    """Six of the seven bodies issue none, and the seventh issues one.

    Scanned over every ``dist.*`` call, not just the reductions, and over
    every stream/event/synchronize/wait the engine could name, because a
    stage that silently acquired one of those would be as bad as a moved
    collective.
    """
    found = {}
    for name in sorted(D7_4_METHODS):
        node = method_node(name)
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            text = ast.unparse(call.func)
            if text.startswith("dist.") or any(
                part in text for part in (
                    "all_reduce", "all_gather", "broadcast", "barrier",
                    "synchronize", "Stream", "Event", "wait_stream",
                    "wait_event", "record_event", "current_stream",
                )
            ):
                found.setdefault(name, []).append(text)
    assert found == {"_plan_prefill_pool_wave": ["dist.all_reduce"]}
    assert COLLECTIVE_METHODS["_plan_prefill_pool_wave"] == "all_reduce"


def test_the_wave_reduction_sits_between_its_tensor_and_the_reduced_reads():
    """Statement order around the collective, not just its presence."""
    node = method_node("_plan_prefill_pool_wave")
    branch = next(n for n in ast.walk(node)
                  if isinstance(n, ast.If) and _collective_calls(n, "all_reduce"))
    assert ast.unparse(branch.test) == "group is not None"
    assert [ast.unparse(stmt).split("=")[0].strip() for stmt in branch.body
            if isinstance(stmt, ast.Assign)] == [
        "budget", "total_pages", "state_slots", "hits_confirmed",
    ]
    assert ast.unparse(branch.body[1]) == (
        "dist.all_reduce(budget, op=dist.ReduceOp.MIN, group=group)"
    )
    # The three reduced values are read immediately after it, in order, and
    # nothing stands between the reduce and the first of them.
    assert [ast.unparse(stmt) for stmt in branch.body[2:5]] == [
        "total_pages = int(budget[0].item())",
        "state_slots = int(budget[1].item())",
        "hits_confirmed = bool(int(budget[2].item()))",
    ]
    assert len(branch.body) == 5


def test_the_reduction_precedes_the_plan_identity_call_in_the_source():
    """The pair's ORDER: reduce first, then the digest gather one level down.

    ``_check_pool_plan_identical`` is where the ``all_gather`` lives (D7.3);
    the planner now reaches it as a collaborator callable, and that call must
    stay after the reduction or two ranks of a serve group would pair a
    reduce with a gather.
    """
    node = method_node("_plan_prefill_pool_wave")
    reduce_line = _collective_calls(node, "all_reduce")[0].lineno
    identity = [
        call for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == "_check_pool_plan_identical"
    ]
    assert len(identity) == 1
    assert reduce_line < identity[0].lineno


# --------------------------------------------------------------------------- #
#  D7.4: fakes for the wave planner and its epilogue
# --------------------------------------------------------------------------- #
class WaveTensor:
    """The int64 vector the budget reduction carries.

    Indexable like the real tensor (``budget[0].item()``) and usable as the
    one-element digest tensor of the plan-identity check, so one fake serves
    both collectives of a pooled wave.
    """

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return WaveTensor([self.values[index]])

    def item(self):
        (value,) = self.values
        return value


class WavePlanDist:
    """``torch.distributed`` for the planner, recording its ordered calls.

    ``peers`` is what the rest of the serve group contributes to the MIN, so a
    tighter peer is observable; ``peer_digest`` is what the peers' plan
    digests are, so a divergence is too.
    """

    class ReduceOp:
        MIN = "MIN"

    def __init__(self, *, log, peers=None, peer_digest=None):
        self.log = log
        self._peers = peers
        self._peer_digest = peer_digest

    def all_reduce(self, tensor, *, op, group):
        self.log.append(("all_reduce", op, group, list(tensor.values)))
        if self._peers is not None:
            tensor.values = [
                min(mine, peer) for mine, peer in zip(tensor.values, self._peers)
            ]

    def all_gather(self, out, tensor, *, group):
        self.log.append(("all_gather", group, tensor.item(), len(out)))
        for slot in out:
            slot.values = [
                tensor.item() if self._peer_digest is None else self._peer_digest
            ]


class WaveTorch:
    """Only what the planner and the plan-identity check touch of torch."""

    int64 = "int64"

    @staticmethod
    def tensor(values, *, dtype, device):
        return WaveTensor(values)

    @staticmethod
    def zeros_like(tensor):
        return WaveTensor([0])


def wave_runtime(log, **extra):
    return runtime(torch=WaveTorch(), dist=WavePlanDist(log=log, **extra))


def plan_segments(*ids, members=None):
    """``plan.segments``: a mapping, read both by id and by length."""
    return {
        segment_id: SimpleNamespace(
            members=(members or {}).get(segment_id, (index,))
        )
        for index, segment_id in enumerate(ids)
    }


def wave_plan(*, segments=(1,), chunks=(), materialized_segments=(),
              declined=(), members=None):
    return SimpleNamespace(
        segments=plan_segments(*segments, members=members),
        chunks=tuple(chunks),
        materialized_segments=tuple(materialized_segments),
        declined_materialized=tuple(declined),
        threshold=2,
        computed_tokens=7,
        prompt_tokens=11,
        same_chunk_ancestors=True,
    )


class PlannedWave:
    """``PoolWave`` as the planner builds it, by keyword."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.plan = kwargs["plan"]
        self.prompts = kwargs["prompts"]
        self.uuids = kwargs["uuids"]
        self.global_ids = kwargs["global_ids"]
        self.page_tokens = kwargs["page_tokens"]
        self.total_pages = kwargs["total_pages"]
        self.scratch_pages = kwargs["scratch_pages"]
        self.materialized = tuple(self.plan.materialized_segments)
        self.bound = []

    def bind_materialized_host_pages(self, segment_id, source):
        self.bound.append((segment_id, source))

    def chain_tokens(self, index):
        return 10 * (index + 1)


def wave_plan_context(ns, *, page_budget, prompts, page_tokens=4, rank=0,
                      psm=None, ckpt="org/model"):
    sequences = {
        f"u{index}": SimpleNamespace(
            global_idx=100 + index, prompt_length=len(prompt),
        )
        for index, prompt in enumerate(prompts)
    }
    return ns["PrefillPoolWavePlanContext"](
        rank=rank,
        torch_device="cuda:0",
        parallel_manager=psm if psm is not None else SimpleNamespace(),
        huggingface_ckpt_name=ckpt,
        pool=SimpleNamespace(get_sequence=sequences.__getitem__),
        prefix_runtime=SimpleNamespace(
            group_specs=[SimpleNamespace(raw_page_tokens=page_tokens)]
        ),
        _prefill_pool_page_budget=page_budget,
    )


def planner_modules(monkeypatch, *, scratch=2, plan=None, calls=None):
    """Stub every module the planner imports, recording the plan's kwargs."""
    calls = calls if calls is not None else {}
    built = plan if plan is not None else wave_plan()

    def plan_wave_prefix_sharing(prompts, *, materialized=(), **kwargs):
        calls["plan"] = (list(prompts), tuple(materialized), dict(kwargs))
        return built

    def scratch_pages_bound(chunk_tokens, longest, count, page_tokens):
        calls["scratch"] = (chunk_tokens, longest, count, page_tokens)
        return scratch

    def build_gpu_kv_config(ckpt, page_tokens):
        calls["geometry"] = (ckpt, list(page_tokens))
        return SimpleNamespace(name="geometry")

    def order_chunk_items_by_level(value):
        calls["level_wise"] = True
        return value

    def validate_wave_prefix_plan(value, prompts):
        calls["validated"] = (value, list(prompts))

    stub_module(monkeypatch, "batchgen.kv_cache.host_kv_mananger_config",
                build_gpu_kv_config=build_gpu_kv_config)
    stub_module(monkeypatch, "batchgen.prefix_reuse.levels",
                order_chunk_items_by_level=order_chunk_items_by_level)
    stub_module(monkeypatch, "batchgen.prefix_reuse.pool",
                scratch_pages_bound=scratch_pages_bound)
    stub_module(monkeypatch, "batchgen.prefix_reuse.wave_plan",
                plan_wave_prefix_sharing=plan_wave_prefix_sharing,
                validate_wave_prefix_plan=validate_wave_prefix_plan)
    stub_module(monkeypatch, "batchgen.prefix_reuse.wave_runtime",
                PoolWave=PlannedWave)
    return calls


def tokens_of(seq, length=None, **_):
    return SimpleNamespace(tolist=lambda: list(range(length or 0)))


def run_plan(ns, engine, ctx, *, uuids, token_cap=64, group=None,
             state_slots=32, psm_layout=None, layout=(True, (), 0),
             hits=(), hits_confirmed=True, hit_pages=None, drops=None,
             identity=None, settled=None, replaced=None):
    """Drive the planner from its context, with every collaborator recorded."""
    settled = settled if settled is not None else []
    replaced = replaced if replaced is not None else []
    return engine._plan_prefill_pool_wave(
        ctx,
        list(uuids),
        {},
        token_cap=token_cap,
        _trajectory_tokens=tokens_of,
        _prefill_pool_group=lambda: group,
        _prefill_pool_layout=lambda geometry: layout,
        _prefill_pool_free_state_slots=lambda: state_slots,
        _prefill_pool_psm_layout=lambda: psm_layout,
        _materialized_prefix_hits=lambda seqs, *, page_tokens, group: (
            list(hits), hits_confirmed, dict(hit_pages or {}),
            dict(drops or {}),
        ),
        _check_pool_plan_identical=(
            identity if identity is not None else lambda plan, group: None
        ),
        _replace_prefix_lookup_hits=lambda *a, **k: replaced.append((a, k)),
        _settle_trajectory_restore_plan=lambda *a: settled.append(a),
    )


# --------------------------------------------------------------------------- #
#  D7.4: the one collective, observed in order at runtime
# --------------------------------------------------------------------------- #
def test_one_rank_per_group_plans_without_issuing_any_collective(monkeypatch):
    log = []
    ns, engine = wave_runtime(log)
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab", "cd"])

    wave = run_plan(ns, engine, ctx, uuids=("u0", "u1"), group=None)

    assert isinstance(wave, PlannedWave)
    assert log == []
    assert calls["plan"][2]["pool_pages"] == 18


def test_a_serve_group_issues_the_reduce_then_the_digest_gather(monkeypatch):
    """The ordered sequence of collectives a pooled wave really issues.

    The plan-identity collaborator is the engine's OWN
    ``_check_pool_plan_identical``, driven with its own context, so the
    ``all_gather`` it issues lands in the same recorder. Reduce first, gather
    second: every rank of a serve group issues exactly this pair, in this
    order.
    """
    log = []
    ns, engine = wave_runtime(log, peers=[16, 8, 1])
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab", "cd"], rank=3)
    identity_ctx = ns["PoolPlanIdentityContext"](rank=3, torch_device="cuda:0")

    wave = run_plan(
        ns, engine, ctx, uuids=("u0", "u1"), group="pg", state_slots=32,
        identity=lambda plan, group: engine._check_pool_plan_identical(
            identity_ctx, plan, group, attn_group_size=2,
        ),
    )

    assert [entry[0] for entry in log] == ["all_reduce", "all_gather"]
    # This rank contributed its own three budgets; the group's MIN won.
    assert log[0] == ("all_reduce", "MIN", "pg", [20, 32, 1])
    assert log[1][1] == "pg" and log[1][3] == 2
    assert wave.total_pages == 16
    assert calls["plan"][2]["pool_pages"] == 14


def test_a_wave_the_budget_declines_issues_the_reduce_and_nothing_else(
    monkeypatch,
):
    """The reduce is NOT conditional on the wave being pooled.

    Every rank of the group reaches it, and the decline that follows is taken
    on the reduced values, so it is group-uniform. Only the gather is skipped,
    because the body returns before the plan exists.
    """
    log = []
    ns, engine = wave_runtime(log, peers=[3, 32, 1])
    planner_modules(monkeypatch, scratch=4)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab", "cd"])
    settled = []

    assert run_plan(
        ns, engine, ctx, uuids=("u0", "u1"), group="pg", settled=settled,
    ) is None
    assert [entry[0] for entry in log] == ["all_reduce"]
    # The decline still settles the wave's restore plan, with its reason.
    assert "pool 3 pages <= scratch 4" in settled[0][2]


def test_an_unmeasured_budget_declines_before_the_group_is_consulted(
    monkeypatch,
):
    """The first wave after startup issues no collective at all."""
    log = []
    ns, engine = wave_runtime(log)
    planner_modules(monkeypatch)
    ctx = wave_plan_context(
        ns, page_budget=None, prompts=["ab"], psm=Explode("parallel_manager"),
    )
    settled = []

    assert run_plan(
        ns, engine, ctx, uuids=("u0",), group="pg", settled=settled,
        identity=lambda plan, group: pytest.fail("compared"),
    ) is None
    assert log == []
    assert settled[0][2] == "the prefill-phase pool budget is not yet measured"


# --------------------------------------------------------------------------- #
#  D7.4: design clause C7 -- the wave is bounded by the token budget
# --------------------------------------------------------------------------- #
def test_the_chunk_bound_is_the_token_cap_however_many_sequences_there_are(
    monkeypatch,
):
    """C7, run: a sequence COUNT must not be what bounds the wave.

    Two sequences and then six, at a page budget wide enough that the pool
    cap is above the server cap: the planner is handed the token cap both
    times, unchanged, and the only thing the count moves is the segment
    allowance and the scratch bound.
    """
    ns, engine = wave_runtime([])
    for count in (2, 6):
        calls = planner_modules(monkeypatch)
        prompts = ["ab"] * count
        ctx = wave_plan_context(ns, page_budget=4096, prompts=prompts)
        run_plan(
            ns, engine, ctx, uuids=tuple(f"u{i}" for i in range(count)),
            token_cap=64, psm_layout={"num_layers": 2}, state_slots=32,
        )
        assert calls["plan"][2]["chunk_tokens"] == 64, count
        assert calls["scratch"][0] == 64, count


def test_the_chunk_bound_falls_to_the_page_identity_when_that_is_tighter(
    monkeypatch,
):
    """The exact identity, not an approximation of it.

    ``max(0, (total_pages - len(prompts)) // 2) * page_tokens``: 16 pages, two
    prompts, 4 tokens a page -> (16 - 2) // 2 * 4 = 28, below the 64-token
    server cap, so 28 is what the planner gets. The integer division
    truncates: 17 pages gives (17 - 2) // 2 = 7, the same 28.
    """
    ns, engine = wave_runtime([])
    for pages, expected in ((16, 28), (17, 28), (18, 32)):
        calls = planner_modules(monkeypatch)
        ctx = wave_plan_context(ns, page_budget=pages, prompts=["ab", "cd"])
        run_plan(
            ns, engine, ctx, uuids=("u0", "u1"), token_cap=64,
            psm_layout={"num_layers": 2},
        )
        assert calls["plan"][2]["chunk_tokens"] == expected, pages
        assert calls["scratch"][0] == expected, pages


def test_a_model_without_a_psm_layout_keeps_the_whole_token_cap(monkeypatch):
    """No pool-page cap at all: the wave is the server cap's, as it was."""
    ns, engine = wave_runtime([])
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(ns, page_budget=16, prompts=["ab", "cd"])

    run_plan(ns, engine, ctx, uuids=("u0", "u1"), token_cap=64,
             psm_layout=None)

    assert calls["plan"][2]["chunk_tokens"] == 64


def test_the_pool_pages_are_the_budget_minus_the_scratch_bound(monkeypatch):
    ns, engine = wave_runtime([])
    calls = planner_modules(monkeypatch, scratch=5)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab"])

    run_plan(ns, engine, ctx, uuids=("u0",))

    assert calls["plan"][2]["pool_pages"] == 15


def test_the_segment_allowance_is_the_spare_slots_and_never_negative(
    monkeypatch,
):
    """``max(0, state_slots - len(prompts))``, and absent at "no such state"."""
    ns, engine = wave_runtime([])
    for slots, expected in ((5, 3), (2, 0), (0, 0)):
        calls = planner_modules(monkeypatch, plan=wave_plan(segments=()))
        ctx = wave_plan_context(ns, page_budget=20, prompts=["ab", "cd"])
        run_plan(ns, engine, ctx, uuids=("u0", "u1"), state_slots=slots)
        assert calls["plan"][2]["max_computed_segments"] == expected, slots
    calls = planner_modules(monkeypatch, plan=wave_plan(segments=()))
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab", "cd"])
    run_plan(ns, engine, ctx, uuids=("u0", "u1"), state_slots=-1)
    assert "max_computed_segments" not in calls["plan"][2]


def test_the_two_refusals_keep_their_comparison_operand_order():
    """C7/K3 as source identities: the arithmetic is an exact expression.

    Pinned textually because a re-association, a flipped operand or a changed
    rounding direction would each change which waves are admitted while every
    behavioural test above still passed on its own numbers.
    """
    node = method_node("_plan_prefill_pool_wave")
    body = ast.unparse(node)
    assert "pool_chunk_tokens = max(0, (total_pages - len(prompts)) // 2) * page_tokens" in body
    assert "if pool_chunk_tokens < chunk_tokens:" in body
    assert "pool_pages=total_pages - scratch" in body
    assert "if total_pages <= scratch:" in body
    assert "plan_kwargs['max_computed_segments'] = max(0, state_slots - len(prompts))" in body
    assert "state_slots_needed = len(plan.segments) - len(plan.materialized_segments) + len(prompts)" in body
    assert "if not plan.segments or 0 <= state_slots < state_slots_needed:" in body
    # The cap itself is the delegator's value, never recomputed here.
    assert "chunk_tokens = token_cap" in body
    assert "_server_prefill_token_cap" not in body


# --------------------------------------------------------------------------- #
#  D7.4: the rest of the wave planner's behaviour, from its context
# --------------------------------------------------------------------------- #
def test_the_geometry_is_built_from_the_contexts_checkpoint_and_page_tokens(
    monkeypatch,
):
    ns, engine = wave_runtime([])
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(
        ns, page_budget=20, prompts=["ab"], page_tokens=64, ckpt="org/k3",
    )

    run_plan(ns, engine, ctx, uuids=("u0",))

    assert calls["geometry"] == ("org/k3", [64])
    assert calls["plan"][2]["block_tokens"] == 64


def test_the_prompts_come_from_the_trajectory_reader_at_each_prompt_length():
    ns, engine = wave_runtime([])
    seen = []
    ctx = wave_plan_context(ns, page_budget=20, prompts=["abc", "de"])

    def reader(seq, length=None, **_):
        seen.append((seq.global_idx, length))
        return SimpleNamespace(tolist=lambda: list(range(length)))

    with pytest.MonkeyPatch.context() as patch:
        calls = planner_modules(patch)
        engine._plan_prefill_pool_wave(
            ctx, ["u0", "u1"], {}, token_cap=64,
            _trajectory_tokens=reader,
            _prefill_pool_group=lambda: None,
            _prefill_pool_layout=lambda geometry: (True, (), 0),
            _prefill_pool_free_state_slots=lambda: 32,
            _prefill_pool_psm_layout=lambda: None,
            _materialized_prefix_hits=lambda seqs, *, page_tokens, group: (
                [], True, {}, {},
            ),
            _check_pool_plan_identical=lambda plan, group: None,
            _replace_prefix_lookup_hits=lambda *a, **k: None,
            _settle_trajectory_restore_plan=lambda *a: None,
        )
    assert seen == [(100, 3), (101, 2)]
    assert calls["plan"][0] == [[0, 1, 2], [0, 1]]


def test_a_level_wise_model_reorders_its_plan_and_shares_its_chunk(
    monkeypatch,
):
    ns, engine = wave_runtime([])
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(
        ns, page_budget=20, prompts=["ab"],
        psm=psm(supports_group_pool_prefill=lambda: True),
    )

    run_plan(ns, engine, ctx, uuids=("u0",), layout=(True, (), 0))

    assert calls["plan"][2]["same_chunk_ancestors"] is True
    assert calls.get("level_wise") is True


def test_a_psm_that_declares_nothing_leaves_the_plan_in_planner_order(
    monkeypatch,
):
    ns, engine = wave_runtime([])
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab"])

    run_plan(ns, engine, ctx, uuids=("u0",), layout=(True, (), 0))

    assert "level_wise" not in calls
    # with_v is True, so a non-level-wise model does not share its chunk and
    # the DSA index-key reservation stays off.
    assert calls["plan"][2]["same_chunk_ancestors"] is False
    assert calls["plan"][2]["chunk_keys"] is None


def test_a_dsa_model_reserves_its_index_key_window(monkeypatch):
    ns, engine = wave_runtime([])
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab"])

    run_plan(ns, engine, ctx, uuids=("u0",), layout=(False, (0,), 64))

    assert calls["plan"][2]["same_chunk_ancestors"] is True
    assert calls["plan"][2]["chunk_keys"] == 8 * 1024 * 1024


def test_a_wave_with_no_segment_declines_and_settles_its_restore_plan(
    monkeypatch,
):
    ns, engine = wave_runtime([])
    planner_modules(monkeypatch, plan=wave_plan(segments=()))
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab"])
    settled, replaced = [], []

    assert run_plan(
        ns, engine, ctx, uuids=("u0",), settled=settled, replaced=replaced,
    ) is None
    # A decline keeps its lookup hits: only the pooled return releases them.
    assert replaced == []
    assert "0 segment(s)" in settled[0][2]


def test_the_pooled_return_releases_the_hits_at_their_chain_ends(monkeypatch):
    ns, engine = wave_runtime([])
    planner_modules(monkeypatch)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab", "cd"])
    settled, replaced = [], []

    wave = run_plan(
        ns, engine, ctx, uuids=("u0", "u1"), settled=settled,
        replaced=replaced,
    )

    assert isinstance(wave, PlannedWave)
    (seqs, states, chains), kwargs = replaced[0]
    assert chains == [10, 20]
    assert kwargs == {"retain": frozenset(), "restore_boundaries": {}}
    assert settled[0][2] == "the wave plan does not materialize its frontier"


def test_a_materialized_segment_binds_the_host_pages_its_leader_recorded(
    monkeypatch,
):
    ns, engine = wave_runtime([])
    planner_modules(
        monkeypatch,
        plan=wave_plan(segments=(7,), materialized_segments=(7,),
                       members={7: (0,)}),
    )
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab", "cd"])
    replaced = []

    wave = run_plan(
        ns, engine, ctx, uuids=("u0", "u1"), hits=(), hit_pages={0: [71, 72]},
        replaced=replaced,
    )

    assert wave.bound == [(7, [71, 72])]
    # The materialized row, and only it, restores at its chain end.
    (_, _, _), kwargs = replaced[0]
    assert kwargs["restore_boundaries"] == {0: 10}
    assert kwargs["retain"] == frozenset({0})


def test_a_follower_without_a_record_binds_nothing_and_still_plans(
    monkeypatch,
):
    ns, engine = wave_runtime([])
    planner_modules(
        monkeypatch,
        plan=wave_plan(segments=(7,), materialized_segments=(7,),
                       members={7: (0,)}),
    )
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab", "cd"])

    wave = run_plan(ns, engine, ctx, uuids=("u0", "u1"), hit_pages={})

    assert wave.bound == []


def test_an_unconfirmed_hit_set_is_dropped_by_name_before_the_plan(
    monkeypatch, caplog,
):
    ns, engine = wave_runtime([])
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab"], rank=4)
    hit = SimpleNamespace(index=0)

    with caplog.at_level("WARNING"):
        run_plan(
            ns, engine, ctx, uuids=("u0",), hits=(hit,),
            hits_confirmed=False,
        )

    assert calls["plan"][1] == ()
    assert "could not confirm the mirrored frontier set" in caplog.text


def test_a_model_whose_pool_cannot_read_its_own_chunk_drops_its_hits(
    monkeypatch, caplog,
):
    """with_v and no level-wise declaration: same_chunk_ancestors is False."""
    ns, engine = wave_runtime([])
    calls = planner_modules(monkeypatch)
    ctx = wave_plan_context(ns, page_budget=20, prompts=["ab"])
    hit = SimpleNamespace(index=0)

    with caplog.at_level("INFO"):
        run_plan(ns, engine, ctx, uuids=("u0",), hits=(hit,),
                 layout=(True, (), 0))

    assert calls["plan"][1] == ()
    assert "does not read a segment resident in its own chunk" in caplog.text


# --------------------------------------------------------------------------- #
#  D7.4: the micro-batch planner
# --------------------------------------------------------------------------- #
def micro_batch_context(ns, *, budget=None, psm_declares=None):
    return ns["PrefillMicroBatchPlanContext"](
        parallel_manager=(
            SimpleNamespace() if psm_declares is None
            else psm(prefill_uses_streamed_sp8=lambda: psm_declares)
        ),
        _prefill_prefix_kv_budget_bytes=budget,
    )


def micro_runtime():
    """The planner's bodies with the real packer replaced by a recorder."""
    log = []

    def build(seq_lengths, token_cap, *, l2_balance, single_sequence_only,
              prefix_hits, kv_pages, kv_page_cap):
        log.append({
            "seq_lengths": list(seq_lengths), "token_cap": token_cap,
            "l2_balance": l2_balance,
            "single_sequence_only": single_sequence_only,
            "prefix_hits": list(prefix_hits), "kv_pages": list(kv_pages),
            "kv_page_cap": kv_page_cap,
        })
        return [(0, len(seq_lengths))], 99

    ns, engine = runtime(build_prefill_micro_batches=build)
    return ns, engine, log


def plan_micro(engine, ctx, *, token_cap=64, attn_group_size=1,
               seq_lengths=(8,), full=(8,), cached=(0,), kv_config=None):
    return engine._plan_prefill_micro_batches(
        ctx, list(seq_lengths), list(full), list(cached),
        token_cap=token_cap,
        attn_group_size=attn_group_size,
        _prefix_materialization_kv_config=(
            kv_config if kv_config is not None
            else lambda lengths: pytest.fail("resolved the KV config")
        ),
    )


def test_a_wave_with_no_prefix_hit_reads_neither_budget_nor_geometry():
    ns, engine, log = micro_runtime()
    ctx = micro_batch_context(ns, budget=Explode("budget"))

    micro_batches, cap, single = plan_micro(
        engine, ctx, seq_lengths=(8, 8), full=(8, 8), cached=(0, 0),
    )

    assert (micro_batches, cap, single) == ([(0, 2)], 99, False)
    assert log[0]["prefix_hits"] == [False, False]
    assert log[0]["kv_pages"] == [0, 0] and log[0]["kv_page_cap"] == 0
    assert log[0]["token_cap"] == 64


def test_a_prefix_hit_without_a_measured_budget_is_refused_by_name():
    ns, engine, _ = micro_runtime()
    ctx = micro_batch_context(ns, budget=None)

    with pytest.raises(RuntimeError, match="no prefill-phase prefix KV budget"):
        plan_micro(engine, ctx, seq_lengths=(8,), full=(1280,), cached=(1272,))


def test_a_prefix_hit_inside_a_serve_group_is_refused_by_name():
    ns, engine, _ = micro_runtime()
    ctx = micro_batch_context(ns, budget=10 ** 9)

    with pytest.raises(RuntimeError, match="rank-local"):
        plan_micro(
            engine, ctx, attn_group_size=4,
            seq_lengths=(8,), full=(1280,), cached=(1272,),
        )


def test_the_page_cap_is_the_budget_over_the_page_bytes(monkeypatch):
    ns, engine, log = micro_runtime()
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.materialization",
        gpu_kv_page_bytes=lambda config: 100,
        temporary_kv_pages=lambda length, page_tokens: length // page_tokens,
    )
    ctx = micro_batch_context(ns, budget=1050)
    seen = []

    plan_micro(
        engine, ctx, seq_lengths=(8, 8), full=(256, 512), cached=(248, 504),
        kv_config=lambda lengths: (
            seen.append(list(lengths)),
            SimpleNamespace(page_size_tokens=128),
        )[1],
    )

    # The config is resolved for ONE sequence, as the pre-move body did.
    assert seen == [[1]]
    assert log[0]["kv_pages"] == [2, 4]
    # Integer division, truncating: 1050 // 100 == 10.
    assert log[0]["kv_page_cap"] == 10
    assert log[0]["prefix_hits"] == [True, True]


@pytest.mark.parametrize("declares,expected", [(True, True), (False, False)])
def test_the_single_sequence_arm_is_what_the_psm_declares(declares, expected):
    ns, engine, log = micro_runtime()
    ctx = micro_batch_context(ns, psm_declares=declares)

    _, _, single = plan_micro(engine, ctx, seq_lengths=(128,), full=(128,),
                              token_cap=64)

    assert single is expected
    assert log[0]["single_sequence_only"] is expected


def test_a_psm_that_declares_nothing_never_runs_single_sequence():
    ns, engine, _ = micro_runtime()
    ctx = micro_batch_context(ns)

    _, _, single = plan_micro(engine, ctx, seq_lengths=(128,), full=(128,),
                              token_cap=64)

    assert single is False


def test_a_cap_no_sequence_exceeds_leaves_the_declaration_unread():
    """Short-circuit order: the length test precedes the PSM question."""
    ns, engine, _ = micro_runtime()
    ctx = ns["PrefillMicroBatchPlanContext"](
        parallel_manager=Explode("parallel_manager"),
        _prefill_prefix_kv_budget_bytes=None,
    )

    _, _, single = plan_micro(engine, ctx, seq_lengths=(8,), full=(8,),
                              token_cap=64)

    assert single is False


# --------------------------------------------------------------------------- #
#  D7.4: the four pooled instruments
# --------------------------------------------------------------------------- #
def probe_wave(*, uuids=("a", "b"), segments=(), page_tokens=64,
               global_ids=(64, 65), debug=None):
    return SimpleNamespace(
        uuids=list(uuids),
        global_ids=list(global_ids),
        page_tokens=page_tokens,
        plan=SimpleNamespace(segments=tuple(segments), chunks=()),
        probe_debug=debug,
        probe_debug_read=debug is not None,
    )


def test_an_unarmed_wave_arms_no_instrument_and_imports_nothing():
    ns, engine = runtime()
    ctx = probe_context(ns, {})
    wave = probe_wave()

    assert engine._new_pool_epilogue_tracer(ctx, wave) is None
    assert engine._new_pool_kernel_dump_recorder(ctx, wave) is None
    assert engine._new_pool_offload_check_recorder(
        ctx, wave, _debug_flag_enabled=lambda value: pytest.fail("read"),
    ) is None


def test_the_epilogue_tracer_is_what_the_helper_built_for_this_rank(
    monkeypatch,
):
    ns, engine = runtime()
    seen = []
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.epilogue_trace",
        tracer_from_debug=lambda debug, *, rank, path: (
            seen.append((dict(debug), rank, path)),
            SimpleNamespace(probe_token="tok"),
        )[1],
    )
    stub_module(monkeypatch, "batchgen.prefix_reuse.probe_guard",
                run_probe=lambda name, call, *, rank: call())
    ctx = probe_context(ns, {"prefix_pool_epilogue_trace": "on"}, rank=7)

    tracer = engine._new_pool_epilogue_tracer(ctx, probe_wave())

    assert seen == [({"prefix_pool_epilogue_trace": "on"}, 7, "pool")]
    assert tracer.probe_token == "tok"


def test_a_tracer_the_guard_refuses_leaves_no_tracer(monkeypatch):
    ns, engine = runtime()
    stub_module(monkeypatch, "batchgen.prefix_reuse.epilogue_trace",
                tracer_from_debug=lambda *a, **k: None)
    stub_module(monkeypatch, "batchgen.prefix_reuse.probe_guard",
                run_probe=lambda name, call, *, rank: None)
    ctx = probe_context(ns, {"prefix_pool_epilogue_trace": "on"})

    assert engine._new_pool_epilogue_tracer(ctx, probe_wave()) is None


def test_the_kernel_dump_recorder_carries_the_rank_mode_dir_and_wave_ids(
    monkeypatch,
):
    ns, engine = runtime()
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.kernel_dump",
        DEBUG_FLAG="prefix_pool_dump_kernel_args",
        DUMP_DIR_FLAG="prefix_pool_dump_dir",
        PoolKernelDumpRecorder=lambda **kwargs: SimpleNamespace(**kwargs),
        dump_directory=lambda value, rank: f"{value}/r{rank}",
        parse_dump_mode=lambda value: value,
    )
    probes = []
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.probe_guard",
        run_probe=lambda name, call, *, rank: (
            probes.append((name, rank)), call(),
        )[1],
    )
    ctx = probe_context(
        ns,
        {"prefix_pool_dump_kernel_args": "sync", "prefix_pool_dump_dir": "/d"},
        rank=5,
    )

    recorder = engine._new_pool_kernel_dump_recorder(
        ctx, probe_wave(global_ids=(9, 10)),
    )

    assert probes == [("kernel_dump.setup", 5)] * 3
    assert (recorder.rank, recorder.mode, recorder.directory) == (
        5, "sync", "/d/r5",
    )
    assert recorder.page_tokens == 64 and recorder.global_ids == [9, 10]


def test_a_dump_mode_the_guard_refuses_stops_before_the_directory(monkeypatch):
    ns, engine = runtime()
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.kernel_dump",
        DEBUG_FLAG="prefix_pool_dump_kernel_args",
        DUMP_DIR_FLAG="prefix_pool_dump_dir",
        PoolKernelDumpRecorder=lambda **kwargs: pytest.fail("built"),
        dump_directory=lambda value, rank: pytest.fail("resolved"),
        parse_dump_mode=lambda value: None,
    )
    stub_module(monkeypatch, "batchgen.prefix_reuse.probe_guard",
                run_probe=lambda name, call, *, rank: call())
    ctx = probe_context(ns, {"prefix_pool_dump_kernel_args": "nonsense"})

    assert engine._new_pool_kernel_dump_recorder(ctx, probe_wave()) is None


def test_the_offload_recorder_follows_the_flag_reader_it_is_handed(
    monkeypatch,
):
    ns, engine = runtime()
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.offload_check",
        DEBUG_FLAG="prefix_pool_offload_check",
        PoolOffloadCheckRecorder=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    stub_module(monkeypatch, "batchgen.prefix_reuse.probe_guard",
                run_probe=lambda name, call, *, rank: call())
    ctx = probe_context(ns, {"prefix_pool_offload_check": "YES"}, rank=2)
    read = []

    recorder = engine._new_pool_offload_check_recorder(
        ctx, probe_wave(),
        _debug_flag_enabled=lambda value: (read.append(value), True)[1],
    )

    assert read == ["YES"]
    assert (recorder.rank, recorder.page_tokens) == (2, 64)


def test_an_offload_flag_the_reader_rejects_arms_no_recorder(monkeypatch):
    ns, engine = runtime()
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.offload_check",
        DEBUG_FLAG="prefix_pool_offload_check",
        PoolOffloadCheckRecorder=lambda **kwargs: pytest.fail("armed"),
    )
    ctx = probe_context(ns, {"prefix_pool_offload_check": "off"})

    assert engine._new_pool_offload_check_recorder(
        ctx, probe_wave(), _debug_flag_enabled=lambda value: False,
    ) is None


def test_the_unsharded_record_asserts_its_premise_against_the_manager(
    monkeypatch,
):
    ns, engine = runtime()
    seen = []
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.epilogue_trace",
        record_unsharded_gather=lambda tracer, selected, *, num_global_rows,
        manager: seen.append((tracer, selected, num_global_rows, manager)),
    )
    probes = []
    stub_module(
        monkeypatch, "batchgen.prefix_reuse.probe_guard",
        run_probe=lambda name, call, *, rank: (
            probes.append((name, rank)), call(),
        )[1],
    )
    ctx = ns["UnshardedEpilogueGatherContext"](rank=6, parallel_manager="psm")

    engine._record_unsharded_epilogue_gather(
        ctx, "tracer", "rows", num_global_rows=12,
    )

    assert seen == [("tracer", "rows", 12, "psm")]
    assert probes == [("epilogue_trace.gather_unsharded", 6)]


# --------------------------------------------------------------------------- #
#  D7.4: the wave epilogue, and the write it hands back
# --------------------------------------------------------------------------- #
def finish_wave(*, uuids=("a", "b"), segment_sids=None, global_ids=(100, 101)):
    return SimpleNamespace(
        uuids=list(uuids),
        global_ids=list(global_ids),
        segment_sids=dict(segment_sids or {7: 900}),
    )


class RecordingPrefixOwner:
    def __init__(self):
        self.handles = []

    def append_commit_handle(self, global_id, handle):
        self.handles.append((global_id, handle))


def finish_context(ns, wave, *, prefix=None, released=None, runtime_cfg=None):
    sequences = {uuid: f"seq-{uuid}" for uuid in (wave.uuids if wave else ())}
    return ns["PrefillPoolWaveFinishContext"](
        _prefill_pool_wave=wave,
        pool=SimpleNamespace(get_sequence=sequences.__getitem__),
        prefix_runtime=runtime_cfg if runtime_cfg is not None else (
            SimpleNamespace(
                namespace_digest="nsd", publish_boundary_tokens=16,
                max_nodes=4,
            )
        ),
        prefix_coordinator=SimpleNamespace(
            release_attachment=(released if released is not None else []).append
        ),
        prefix=prefix if prefix is not None else RecordingPrefixOwner(),
    )


def wave_runtime_module(monkeypatch, *, commit=None, attach=None, calls=None):
    calls = calls if calls is not None else []

    def commit_pool_segments(wave, **kwargs):
        calls.append(("commit", kwargs))
        return commit if commit is not None else ["h1"]

    def attach_pool_members(wave, **kwargs):
        calls.append(("attach", kwargs))
        return attach if attach is not None else {0: "m0"}

    stub_module(monkeypatch, "batchgen.prefix_reuse.wave_runtime",
                attach_pool_members=attach_pool_members,
                commit_pool_segments=commit_pool_segments)
    return calls


def finish(engine, ctx, *, owns=True, views=None, seen=None):
    seen = seen if seen is not None else []
    return engine._finish_prefill_pool_wave(
        ctx,
        _owns_host_kv=lambda seq: (seen.append(seq), owns)[1],
        _host_views_by_group=lambda: (
            views if views is not None else {0: SimpleNamespace(
                unregister_sequences=lambda sids: None,
            )}
        ),
    )


def test_no_wave_in_flight_writes_nothing_and_reads_nothing_else():
    ns, engine = runtime()
    ctx = ns["PrefillPoolWaveFinishContext"](
        _prefill_pool_wave=None,
        pool=Explode("pool"),
        prefix_runtime=Explode("prefix_runtime"),
        prefix_coordinator=Explode("prefix_coordinator"),
        prefix=Explode("prefix"),
    )

    outcome = engine._finish_prefill_pool_wave(
        ctx,
        _owns_host_kv=lambda seq: pytest.fail("asked"),
        _host_views_by_group=lambda: pytest.fail("asked"),
    )

    # The sentinel, not ``None``: this path wrote nothing at all, and ``None``
    # is exactly what the clear assigns.
    assert outcome._prefill_pool_wave is ns["PREFILL_UNCHANGED"]


def test_a_follower_drops_its_mirrored_page_tables_and_clears_the_wave(
    monkeypatch,
):
    ns, engine = runtime()
    wave = finish_wave(segment_sids={7: 900, 8: 901})
    ctx = finish_context(ns, wave)
    dropped, seen = [], []

    outcome = finish(
        engine, ctx, owns=False, seen=seen,
        views={0: SimpleNamespace(unregister_sequences=dropped.append)},
    )

    assert seen == ["seq-a"]
    assert dropped == [[900, 901]]
    assert outcome._prefill_pool_wave is None


def test_the_leader_commits_then_attaches_then_releases_its_segments(
    monkeypatch,
):
    ns, engine = runtime()
    wave = finish_wave()
    prefix, released = RecordingPrefixOwner(), []
    ctx = finish_context(ns, wave, prefix=prefix, released=released)
    calls = wave_runtime_module(
        monkeypatch, commit=["s0", "s1"], attach={0: "m0", 1: "m1"},
    )

    outcome = finish(engine, ctx)

    assert [name for name, _ in calls] == ["commit", "attach"]
    assert calls[0][1]["namespace_digest"] == "nsd"
    assert calls[0][1]["publish_boundary_tokens"] == 16
    assert calls[0][1]["max_scan_nodes"] == 4
    # The segment attachments are released in order, whatever happened above.
    assert released == ["s0", "s1"]
    assert prefix.handles == [(100, "m0"), (101, "m1")]
    assert outcome._prefill_pool_wave is None


def test_a_failed_attach_still_releases_the_segments_and_propagates(
    monkeypatch,
):
    """The clear is lost on the raising path, by construction.

    The engine cannot write the worker, so a body whose publication raises
    leaves this frame before its delegator can apply the clear. The
    docstring's fatal-path argument is what makes that safe; this pins the
    mechanism: the release still runs, and the exception is not swallowed.
    """
    ns, engine = runtime()
    released = []
    ctx = finish_context(ns, finish_wave(), released=released)

    def attach_pool_members(wave, **kwargs):
        raise RuntimeError("attach failed")

    stub_module(monkeypatch, "batchgen.prefix_reuse.wave_runtime",
                attach_pool_members=attach_pool_members,
                commit_pool_segments=lambda wave, **kwargs: ["s0"])

    with pytest.raises(RuntimeError, match="attach failed"):
        finish(engine, ctx)
    assert released == ["s0"]


@pytest.mark.parametrize("owns", [True, False])
def test_the_host_view_map_is_asked_once_on_whichever_path_runs(
    monkeypatch, owns,
):
    """One call per run, exactly as the pre-move body made it.

    The map is rebuilt per phase, so it is resolved where it is used: the
    follower's unregister loop asks once, the leader's commit asks once, and
    the two paths are exclusive. Holding one resolution across both would be a
    mechanism change.
    """
    ns, engine = runtime()
    wave_runtime_module(monkeypatch)
    asked = []
    ctx = finish_context(ns, finish_wave())

    engine._finish_prefill_pool_wave(
        ctx,
        _owns_host_kv=lambda seq: owns,
        _host_views_by_group=lambda: (
            asked.append(1),
            {0: SimpleNamespace(unregister_sequences=lambda sids: None)},
        )[1],
    )

    assert len(asked) == 1


# --------------------------------------------------------------------------- #
#  D7.4: the seven worker delegators
# --------------------------------------------------------------------------- #
def test_the_micro_batch_delegator_builds_its_context_and_the_kv_builder(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.parallel_manager = "psm"
    worker._prefill_prefix_kv_budget_bytes = 84
    worker._server_prefill_token_cap = lambda: 128
    worker._prefill_attn_tp_size = lambda: 1
    worker._prefix_materialization_kv_config = lambda lengths: None
    seen = []
    recording(engine, "_plan_prefill_micro_batches", seen)

    assert worker._plan_prefill_micro_batches([1], [2], [3]) == "recorded"
    (context, suffix, full, cached), kwargs = seen[0]
    assert isinstance(context, ns["PrefillMicroBatchPlanContext"])
    assert (context.parallel_manager,
            context._prefill_prefix_kv_budget_bytes) == ("psm", 84)
    assert (suffix, full, cached) == ([1], [2], [3])
    assert kwargs == {
        "token_cap": 128,
        "attn_group_size": 1,
        "_prefix_materialization_kv_config":
            worker._prefix_materialization_kv_config,
    }


def test_the_micro_batch_delegator_reads_the_live_budget_each_call(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.parallel_manager = "psm"
    worker._server_prefill_token_cap = lambda: 1
    worker._prefill_attn_tp_size = lambda: 1
    worker._prefix_materialization_kv_config = lambda lengths: None
    seen = []
    recording(engine, "_plan_prefill_micro_batches", seen)
    worker._prefill_prefix_kv_budget_bytes = None
    worker._plan_prefill_micro_batches([], [], [])
    worker._prefill_prefix_kv_budget_bytes = 7
    worker._plan_prefill_micro_batches([], [], [])

    assert [args[0]._prefill_prefix_kv_budget_bytes for args, _ in seen] == [
        None, 7,
    ]


def test_the_wave_plan_delegator_passes_nine_collaborators(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 2
    worker.torch_device = "cuda:2"
    worker.parallel_manager = "psm"
    worker.huggingface_ckpt_name = "org/model"
    worker.global_batch = "pool"
    worker.prefix_cache_runtime_config = "runtime"
    worker._prefill_pool_page_budget = 20
    worker._server_prefill_token_cap = lambda: 64
    # The two collaborators the worker owns outside this engine's delegators.
    worker._trajectory_tokens = lambda seq, length=None, **_: None
    worker._materialized_prefix_hits = lambda *a, **k: None
    seen = []
    recording(engine, "_plan_prefill_pool_wave", seen)

    assert worker._plan_prefill_pool_wave(["u0"], {"s": 1}) == "recorded"
    (context, uuids, states), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPoolWavePlanContext"])
    assert (context.rank, context.torch_device, context.parallel_manager,
            context.huggingface_ckpt_name, context.pool,
            context.prefix_runtime, context._prefill_pool_page_budget) == (
        2, "cuda:2", "psm", "org/model", "pool", "runtime", 20,
    )
    assert (uuids, states) == (["u0"], {"s": 1})
    assert kwargs == {
        "token_cap": 64,
        "_trajectory_tokens": worker._trajectory_tokens,
        "_prefill_pool_group": worker._prefill_pool_group,
        "_prefill_pool_layout": worker._prefill_pool_layout,
        "_prefill_pool_free_state_slots":
            worker._prefill_pool_free_state_slots,
        "_prefill_pool_psm_layout": worker._prefill_pool_psm_layout,
        "_materialized_prefix_hits": worker._materialized_prefix_hits,
        "_check_pool_plan_identical": worker._check_pool_plan_identical,
        "_replace_prefix_lookup_hits": worker._replace_prefix_lookup_hits,
        "_settle_trajectory_restore_plan":
            worker._settle_trajectory_restore_plan,
    }


def test_the_wave_plan_delegator_reads_the_live_page_budget_each_call(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 0
    worker.torch_device = "cuda:0"
    worker.parallel_manager = "psm"
    worker.huggingface_ckpt_name = "org/model"
    worker.global_batch = "pool"
    worker.prefix_cache_runtime_config = "runtime"
    worker._server_prefill_token_cap = lambda: 64
    worker._trajectory_tokens = lambda seq, length=None, **_: None
    worker._materialized_prefix_hits = lambda *a, **k: None
    seen = []
    recording(engine, "_plan_prefill_pool_wave", seen)
    worker._prefill_pool_page_budget = None
    worker._plan_prefill_pool_wave([], {})
    worker._prefill_pool_page_budget = 11
    worker._plan_prefill_pool_wave([], {})

    assert [args[0]._prefill_pool_page_budget for args, _ in seen] == [None, 11]


def test_the_three_recorder_delegators_share_the_probe_context_builder(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 9
    worker._phase_prefill_debug = {"prefix_pool_offload_check": True}
    worker._debug_flag_enabled = lambda value: bool(value)
    seen = []
    for name in PROBE_RECORDERS:
        recording(engine, name, seen)

    worker._new_pool_epilogue_tracer("wave")
    worker._new_pool_kernel_dump_recorder("wave")
    worker._new_pool_offload_check_recorder("wave")

    for args, _ in seen:
        assert isinstance(args[0], ns["PoolProbeContext"])
        assert args[0].rank == 9
        assert args[0]._phase_prefill_debug == {
            "prefix_pool_offload_check": True,
        }
        assert args[1] == "wave"
    # Only the offload check takes the flag reader, as the pad-poison
    # predicate does; the other two parse their own values.
    assert seen[0][1] == {} and seen[1][1] == {}
    assert seen[2][1] == {"_debug_flag_enabled": worker._debug_flag_enabled}


def test_the_recorder_delegators_rebuild_the_probe_context_per_call(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 1
    seen = []
    recording(engine, "_new_pool_kernel_dump_recorder", seen)
    worker._phase_prefill_debug = {"a": 1}
    worker._new_pool_kernel_dump_recorder("wave")
    worker._phase_prefill_debug = {"b": 2}
    worker._new_pool_kernel_dump_recorder("wave")

    assert [args[0]._phase_prefill_debug for args, _ in seen] == [
        {"a": 1}, {"b": 2},
    ]


def test_the_gather_record_delegator_builds_its_own_context(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 4
    worker.parallel_manager = "first"
    seen = []
    recording(engine, "_record_unsharded_epilogue_gather", seen)

    worker._record_unsharded_epilogue_gather("t", "rows", num_global_rows=3)
    worker.parallel_manager = "second"
    worker._record_unsharded_epilogue_gather("t", "rows", num_global_rows=3)

    (context, tracer, selected), kwargs = seen[0]
    assert isinstance(context, ns["UnshardedEpilogueGatherContext"])
    assert (context.rank, context.parallel_manager) == (4, "first")
    assert (tracer, selected) == ("t", "rows")
    assert kwargs == {"num_global_rows": 3}
    # Resolved per call: the phase lifecycle rebinds the manager.
    assert [args[0].parallel_manager for args, _ in seen] == [
        "first", "second",
    ]


def test_the_wave_epilogue_delegator_applies_the_clear(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker._prefill_pool_wave = "wave"
    worker.global_batch = "pool"
    worker.prefix_cache_runtime_config = "runtime"
    worker.prefix_cache_coordinator = "coordinator"
    worker._prefix = "prefix state"
    # Collaborators the worker owns outside this engine's delegators.
    worker._owns_host_kv = lambda seq: True
    worker._host_views_by_group = lambda: {}
    seen = []
    recording(engine, "_finish_prefill_pool_wave", seen,
              result=ns["PrefillPoolWaveFinishOutcome"](None))

    worker._finish_prefill_pool_wave()

    (context,), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPoolWaveFinishContext"])
    assert (context._prefill_pool_wave, context.pool, context.prefix_runtime,
            context.prefix_coordinator, context.prefix) == (
        "wave", "pool", "runtime", "coordinator", "prefix state",
    )
    assert kwargs == {
        "_owns_host_kv": worker._owns_host_kv,
        "_host_views_by_group": worker._host_views_by_group,
    }
    assert worker._prefill_pool_wave is None


def test_the_wave_epilogue_delegator_leaves_the_handle_on_the_sentinel(
    monkeypatch,
):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker._prefill_pool_wave = "wave"
    worker.global_batch = "pool"
    worker.prefix_cache_runtime_config = "runtime"
    worker.prefix_cache_coordinator = "coordinator"
    worker._prefix = "prefix state"
    # Collaborators the worker owns outside this engine's delegators.
    worker._owns_host_kv = lambda seq: True
    worker._host_views_by_group = lambda: {}
    recording(engine, "_finish_prefill_pool_wave", [],
              result=ns["PrefillPoolWaveFinishOutcome"]())

    worker._finish_prefill_pool_wave()

    assert worker._prefill_pool_wave == "wave"


def test_the_wave_epilogue_delegator_reads_the_live_wave_each_call(monkeypatch):
    ns, engine = runtime()
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.global_batch = "pool"
    worker.prefix_cache_runtime_config = "runtime"
    worker.prefix_cache_coordinator = "coordinator"
    worker._prefix = "prefix state"
    # Collaborators the worker owns outside this engine's delegators.
    worker._owns_host_kv = lambda seq: True
    worker._host_views_by_group = lambda: {}
    seen = []
    recording(engine, "_finish_prefill_pool_wave", seen,
              result=ns["PrefillPoolWaveFinishOutcome"]())
    worker._prefill_pool_wave = "first"
    worker._finish_prefill_pool_wave()
    worker._prefill_pool_wave = "second"
    worker._finish_prefill_pool_wave()

    assert [args[0]._prefill_pool_wave for args, _ in seen] == [
        "first", "second",
    ]


def test_every_d7_4_delegator_names_no_worker_in_its_engine_call():
    """One assertion over the whole stage: no engine call passes a bare self."""
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    checked = set()
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in D7_4_METHODS:
            continue
        checked.add(node.name)
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "engine"):
                continue
            assert not any(isinstance(arg, ast.Name) and arg.id == "self"
                           for arg in call.args), node.name
            assert not any(isinstance(kw.value, ast.Name)
                           and kw.value.id == "self"
                           for kw in call.keywords), node.name
    assert checked == D7_4_METHODS


# --------------------------------------------------------------------------- #
#  D7.5: the execution paths -- shape, narrowness, and the two things that
#  would break silently (a moved collective, a moved host sync)
# --------------------------------------------------------------------------- #
D7_5_CONTEXTS = {
    "prefill": "PrefillBatchForwardContext",
    "_prefill_model_pass": "PrefillModelPassContext",
    "_write_prefill_tokens": "PrefillTokenWritebackContext",
    "_prefill_prepacked_pool": "PrefillPoolWaveRunContext",
}
# Every `dist` entry point, stream/event handle and device-wide wait a prefill
# body could reach. Spelled out here so the "this stage has none" claim is a
# membership test over the four bodies, not a reading of them.
_ASYNC_NAMES = (
    "all_reduce", "all_gather", "all_gather_into_tensor", "broadcast",
    "barrier", "reduce", "reduce_scatter", "send", "recv", "isend", "irecv",
    "gather", "scatter", "synchronize", "Stream", "Event", "stream",
    "wait_stream", "wait_event", "record_event", "current_stream",
    "default_stream", "set_stream",
)
# The calls the boundary checker counts as host-synchronising, i.e. the ones
# that make the CPU wait for the GPU.
_HOST_SYNC_METHODS = ("item", "tolist", "cpu", "numpy")


def _host_sync_calls(node):
    """``(method, call text)`` for each host-pulling call, in source order."""
    out = []
    for sub in sorted(
        (n for n in ast.walk(node) if isinstance(n, ast.Call)),
        key=lambda n: (n.lineno, n.col_offset),
    ):
        if (isinstance(sub.func, ast.Attribute)
                and sub.func.attr in _HOST_SYNC_METHODS
                and not sub.args and not sub.keywords):
            out.append((sub.func.attr, ast.unparse(sub)))
    return out


@pytest.mark.parametrize("name", sorted(D7_5_CONTEXTS))
def test_each_execution_path_declares_the_context_it_reads(name):
    node = method_node(name)
    assert ast.unparse(node.args.args[1].annotation) == D7_5_CONTEXTS[name]


def test_the_four_execution_contexts_are_four_distinct_read_sets():
    """No share, proved by the read sets rather than by family resemblance."""
    ns, _ = runtime()
    sets = {
        name: frozenset(f.name for f in fields(ns[context]))
        for name, context in D7_5_CONTEXTS.items()
    }
    assert len(set(sets.values())) == 4, sets
    # And none of them equals any context an earlier stage declared, which is
    # the only ground on which this stage could have reused one.
    earlier = {
        ast.unparse(method_node(other).args.args[1].annotation)
        for other in sorted(
            (D7_1_METHODS | D7_2_METHODS | D7_3_METHODS | D7_4_METHODS)
            - STATELESS_METHODS
        )
    }
    declared = {
        name: frozenset(f.name for f in fields(ns[name])) for name in earlier
    }
    for name, read in sets.items():
        assert read not in set(declared.values()), (name, read)


def test_the_token_writeback_does_not_share_a_drivers_context():
    """The tempting share, rejected, with the arithmetic that rejects it.

    ``_write_prefill_tokens`` is the one operation BOTH prefill drivers end
    on, and its three fields are a strict subset of each driver's. A share
    would therefore be legal for the narrowness test only as the UNION, i.e.
    the driver's whole set -- which would make this delegator resolve the
    model, the model config, the launch device and the weight-copy pipeline on
    a call that does nothing but turn sampled rows into sequence state.
    """
    ns, _ = runtime()
    writeback = frozenset(f.name for f in fields(ns["PrefillTokenWritebackContext"]))
    padded = frozenset(f.name for f in fields(ns["PrefillBatchForwardContext"]))
    pooled = frozenset(f.name for f in fields(ns["PrefillPoolWaveRunContext"]))
    assert writeback == {"rank", "pool", "local_to_uuid"}
    assert writeback < padded and writeback < pooled
    # What a share would force each delegator to resolve, and nobody reads.
    assert padded - writeback == {
        "torch_device", "model", "model_config", "engine_config",
        "_weight_passes",
    }
    assert pooled - writeback == {
        "local_rank", "world_size", "engine_config", "parallel_manager",
        "_weight_passes", "_prefill_pool_wave", "_prefill_pool_base",
    }


def test_no_earlier_stages_context_was_widened_for_the_execution_stage():
    """D7.1-D7.4's contexts keep exactly the fields they were declared with."""
    ns, _ = runtime()
    assert [f.name for f in fields(ns["PrefillPoolLayoutContext"])] == [
        "model", "_prefill_pool_layout_cache",
    ]
    assert [f.name for f in fields(ns["MaterializedIndexKeyLayersContext"])] == [
        "model",
    ]
    assert [f.name for f in fields(ns["UnshardedEpilogueGatherContext"])] == [
        "rank", "parallel_manager",
    ]
    assert [f.name for f in fields(ns["PrefillPoolWavePlanContext"])] == [
        "rank", "torch_device", "parallel_manager", "huggingface_ckpt_name",
        "pool", "prefix_runtime", "_prefill_pool_page_budget",
    ]
    assert [f.name for f in fields(ns["PrefillPoolWaveContext"])] == [
        "_prefill_pool_wave",
    ]


def test_the_execution_stage_declares_no_outcome_at_all():
    """Nothing travels back: every store is an in-place mutation.

    The four bodies store through an attribute PATH -- the model wrapper's
    attention-2 flag, a sequence's EOS latch, the pool's dump recorder and its
    three row handles -- and never bind a worker attribute, so there is no
    write for a delegator to apply.
    """
    assert not (OUTCOME_METHODS & D7_5_METHODS)
    stores = {}
    for name in sorted(D7_5_METHODS):
        node = method_node(name)
        for sub in ast.walk(node):
            targets = []
            if isinstance(sub, ast.Assign):
                targets = list(sub.targets)
            elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
                targets = [sub.target]
            for target in targets:
                parts = (target.elts
                         if isinstance(target, (ast.Tuple, ast.List))
                         else [target])
                for part in parts:
                    if isinstance(part, ast.Attribute):
                        stores.setdefault(name, []).append(ast.unparse(part))
    assert stores == {
        # The family branch, kept verbatim for Phase E2: the flag lives on the
        # model the context carries. Then the EOS latch, on the sequence the
        # pool handed back.
        "prefill": [
            "ctx.model.model._use_flash_attention_2", "seq.eos_reached",
        ],
        "_write_prefill_tokens": ["seq.eos_reached"],
        # The pooled wave's teardown, on the pool the wave object holds.
        "_prefill_prepacked_pool": [
            "wave.pool.kernel_dump", "wave.pool.k", "wave.pool.v",
            "wave.pool.indexer",
        ],
    }, stores
    # Each store goes THROUGH an object: either a local the body was handed
    # (``seq``, ``wave``) or an attribute of one the context carries. None is a
    # bare ``ctx.<field> = ...``, which the frozen context would refuse anyway,
    # and none is a worker attribute binding.
    for name, paths in stores.items():
        for path in paths:
            head, _, rest = path.partition(".")
            assert head != "self", (name, path)
            assert rest and "." in rest or head != "ctx", (name, path)


def test_no_execution_body_issues_a_collective_a_stream_or_an_event():
    """The D6.4/D7.3 rule, discharged by absence rather than by position.

    An async or collective call that MOVED would be the one defect this shape
    change could introduce and no runtime test on one rank would see it. There
    is none to move: the four bodies name neither ``dist`` nor any stream,
    event or device-wide wait, so the claim is checked as emptiness.
    """
    for name in sorted(D7_5_METHODS):
        node = method_node(name)
        assert not [
            n for n in ast.walk(node)
            if isinstance(n, ast.Name) and n.id == "dist"
        ], name
        reached = {
            n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
        } | {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name)
        }
        assert not reached & set(_ASYNC_NAMES), (name, reached & set(_ASYNC_NAMES))


def test_each_body_keeps_its_exact_ordered_sequence_of_host_syncs():
    """The call text and the ORDER, per body, as the pre-move bodies had them.

    The file's total is what the boundary checker ratchets; this pins where
    those calls are, which the total cannot. ``_prefill_model_pass`` is listed
    with an empty sequence on purpose: the one body of the stage that pulls
    nothing back must keep pulling nothing back.
    """
    expected = {
        "prefill": [
            ("cpu", "new_tokens.cpu()"),
            ("item", "new_tokens_cpu[i].item()"),
        ],
        "_prefill_model_pass": [],
        "_write_prefill_tokens": [
            ("cpu", "new_tokens.cpu()"),
            ("item", "new_tokens_cpu[i].item()"),
            ("item", "new_tokens_cpu[i].item()"),
        ],
        "_prefill_prepacked_pool": [
            ("tolist", "rows.tail_rows.tolist()"),
            ("tolist", "new_tokens.reshape(-1).tolist()"),
        ],
    }
    for name in sorted(D7_5_METHODS):
        assert _host_sync_calls(method_node(name)) == expected[name], name


@pytest.mark.parametrize("name", ["prefill", "_write_prefill_tokens"])
def test_the_one_device_copy_stays_outside_the_per_row_loop(name):
    """One copy, then a host read per row -- the pattern, not just the count.

    A conversion that pulled ``new_tokens.cpu()`` into the write-back loop
    would fail no other assertion here and would simply make prefill slower:
    one device-to-host copy per sequence instead of one per batch. So the copy
    is pinned OUTSIDE every loop of the body, and both per-row reads inside
    one.
    """
    node = method_node(name)

    def loops_above(target):
        found = []

        def walk(stmt, depth):
            for sub in ast.iter_child_nodes(stmt):
                nxt = depth + isinstance(stmt, (ast.For, ast.While))
                if sub is target:
                    found.append(nxt)
                walk(sub, nxt)

        walk(node, 0)
        return found

    copies = [n for n in ast.walk(node)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "cpu"]
    assert len(copies) == 1
    assert loops_above(copies[0]) == [0], "the device copy moved into a loop"
    reads = [n for n in ast.walk(node)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "item"]
    assert reads
    for read in reads:
        assert loops_above(read) == [1], "a per-row read left its loop"


def test_the_pooled_driver_reaches_the_model_pass_as_a_collaborator():
    """It hands over no worker, because the pass no longer takes one.

    Before this stage both drivers called the pass on the engine instance they
    had captured, with the worker as its first argument. The pass now takes a
    declared context, which a converted body cannot build, so the pooled
    driver receives the worker's own delegator as a keyword-only callable --
    and passes exactly the two positionals it always did.
    """
    node = method_node("_prefill_prepacked_pool")
    assert "_prefill_model_pass" in {a.arg for a in node.args.kwonlyargs}
    calls = [n for n in ast.walk(node)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_prefill_model_pass"]
    assert len(calls) == 1
    call, = calls
    assert [ast.unparse(arg) for arg in call.args] == [
        "rows.input_ids",
        "rows.tail_rows if rows.tail_wave_indices else None",
    ]
    assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords} == {
        "lmhead_fp32": "False", "trace": "epilogue_trace",
    }
    # No `_engine` capture survives in the converted body, and no `self`.
    assert not [n for n in ast.walk(node)
                if isinstance(n, ast.Name) and n.id == "_engine"]


# D7.5 pinned the transitional spelling of the prepacked driver's model-pass
# call -- the direct engine call with the context built inline -- because that
# driver was still worker-shaped and routing the call through the worker's
# delegator would have made the engine file name a worker attribute it had
# never named. That claim is unprovable once the driver is converted: a
# converted body has no `_engine` and no worker attribute to avoid. Its
# replacement is `test_the_prepacked_driver_reaches_the_model_pass_as_a
# _collaborator` in the D7.6 section below, which pins the converted spelling
# AND that the inline construction is gone rather than duplicated.


# --------------------------------------------------------------------------- #
#  D7.5 runtime: the padded forward and the token write-back, driven
# --------------------------------------------------------------------------- #
class WeightPasses:
    """The weight-copy pipeline, recording its scope/begin/end sequence."""

    def __init__(self, log):
        self.log = log

    def scope(self, name):
        log = self.log

        class Scope:
            def __enter__(self):
                log.append(("scope_enter", name))
                return self

            def __exit__(self, *exc):
                log.append(("scope_exit", name))
                return False

        return Scope()

    def begin(self, *, carry):
        self.log.append(("begin", carry))

    def end(self):
        self.log.append(("end",))


class CountingTokens:
    """A token tensor that counts the device copies taken of it."""

    def __init__(self, values):
        self._tensor = torch.tensor(values, dtype=torch.int64)
        self.copies = 0

    def cpu(self):
        self.copies += 1
        return self._tensor

    @property
    def shape(self):
        return self._tensor.shape


def sequence(prompt_length, tokens):
    seq = SimpleNamespace(
        prompt_length=prompt_length, tokens=tokens, advanced=0,
        eos_reached=False,
    )
    seq.advance_one_token = lambda: setattr(seq, "advanced", seq.advanced + 1)
    return seq


def padded_forward(ns, engine, *, seqs, batch, micro=2, log=None, stops=(),
                   model_type="kimi_linear", family="deepseek", inner=None):
    """Drive the converted ``prefill`` over CPU fakes."""
    log = [] if log is None else log
    uuids = {index: f"u{index}" for index in range(len(seqs))}
    inner = inner if inner is not None else SimpleNamespace(
        _use_flash_attention_2=True,
    )

    def forward(ids, attention_mask, use_cache):
        log.append(("forward", tuple(ids.shape), bool(use_cache)))
        return SimpleNamespace(logits=torch.zeros((ids.shape[0], ids.shape[1], 4)))

    # The model is a callable that also carries the wrapper whose attention
    # flag the family branch clears, exactly as the worker's model object does.
    forward.model = inner
    ctx = ns["PrefillBatchForwardContext"](
        rank=0,
        torch_device="cpu",
        model=forward,
        model_config=SimpleNamespace(model_type=model_type),
        engine_config=SimpleNamespace(
            Module_Batching_Config=SimpleNamespace(
                MoE_prefill_micro_batch_size=micro,
            ),
        ),
        pool=SimpleNamespace(get_sequence=lambda uuid: seqs[int(uuid[1:])]),
        local_to_uuid=uuids,
        _weight_passes=WeightPasses(log),
    )
    issued = []

    def select(logits, sequences):
        log.append(("select", logits.shape[0]))
        base = 7 + sum(len(previous) for previous in issued)
        rows = torch.tensor(
            [base + i for i in range(logits.shape[0])], dtype=torch.int64,
        )
        issued.append(rows)
        return rows

    appended = []
    out = engine.prefill(
        ctx, batch,
        _bind_prefill_host_views=lambda: log.append(("bind_host_views",)),
        _prefill_deepseek_family_name=lambda: family,
        _trajectory_tokens=lambda seq, *, length, dtype: torch.tensor(
            seq.tokens[:length], dtype=dtype,
        ),
        feed_watchdog=lambda: log.append(("watchdog",)),
        _bind_prefill_step_metadata=lambda mask: log.append(
            ("bind_metadata", tuple(mask.shape)),
        ),
        _bind_prefill_step_batch=lambda local: log.append(
            ("bind_batch", tuple(local)),
        ),
        _select_tokens=select,
        _append_trajectory_batch=lambda b, cpu: appended.append(
            (tuple(b), cpu.tolist()),
        ),
        _should_stop_at_eos=lambda token, seq: token in stops,
    )
    return out, log, appended, inner


def test_the_padded_forward_binds_then_runs_one_pass_per_micro_batch():
    ns, engine = runtime(tqdm=lambda iterable, **_: iterable)
    seqs = [sequence(3, [1, 2, 3]), sequence(2, [4, 5]), sequence(3, [6, 7, 8])]
    out, log, appended, _ = padded_forward(
        ns, engine, seqs=seqs, batch=[0, 1, 2], micro=2,
    )
    # The host-view binding comes FIRST, before anything reads a sequence.
    assert log[0] == ("bind_host_views",)
    # Two micro-batches (3 sequences, size 2), each watchdog-fed, begun with
    # the carry flag the position implies, bound, forwarded, selected and
    # ended -- all inside ONE weight-pass scope. The padded rows are the
    # batch's own max length (3), not a global one.
    assert log[1:] == [
        ("scope_enter", "prefill"),
        ("watchdog",), ("begin", True),
        ("bind_metadata", (2, 3)), ("bind_batch", (0, 1)),
        ("forward", (2, 3), False), ("select", 2), ("end",),
        ("watchdog",), ("begin", False),
        ("bind_metadata", (1, 3)), ("bind_batch", (2,)),
        ("forward", (1, 3), False), ("select", 1), ("end",),
        ("scope_exit", "prefill"),
    ]
    assert [seq.advanced for seq in seqs] == [1, 1, 1]
    assert not any(seq.eos_reached for seq in seqs)
    # ONE trajectory append for the whole batch, off the one host copy.
    assert appended == [((0, 1, 2), [7, 8, 9])]
    assert out.tolist() == [7, 8, 9]


def test_the_padded_forwards_eos_latch_lands_on_the_sequence_the_pool_gave():
    ns, engine = runtime(tqdm=lambda iterable, **_: iterable)
    seqs = [sequence(2, [1, 2]), sequence(2, [3, 4])]
    padded_forward(ns, engine, seqs=seqs, batch=[0, 1], micro=2, stops={8})
    # `select` hands back 7, 8: the second row stops.
    assert [seq.eos_reached for seq in seqs] == [False, True]


@pytest.mark.parametrize("model_type,cleared", [
    ("deepseek_v3", True), ("glm_moe", False),
])
def test_the_padded_forward_clears_the_attention_flag_only_on_its_family(
    model_type, cleared,
):
    """The model-specific branch, kept verbatim: Phase E2 removes it, not this.

    The flag it clears lives on the model wrapper the CONTEXT carries, so the
    store is an in-place mutation and no outcome is involved -- and a model
    outside the family is still left alone.
    """
    ns, engine = runtime(tqdm=lambda iterable, **_: iterable)
    seqs = [sequence(1, [1])]
    _, _, _, inner = padded_forward(
        ns, engine, seqs=seqs, batch=[0], micro=1, model_type=model_type,
    )
    assert inner._use_flash_attention_2 is not cleared


def writeback(ns, engine, *, seqs, batch, stops=(), tokens=None):
    uuids = {index: f"u{index}" for index in range(len(seqs))}
    ctx = ns["PrefillTokenWritebackContext"](
        rank=3,
        pool=SimpleNamespace(get_sequence=lambda uuid: seqs[int(uuid[1:])]),
        local_to_uuid=uuids,
    )
    appended = []
    new_tokens = tokens if tokens is not None else CountingTokens(
        [11 + i for i in range(len(batch))]
    )
    out = engine._write_prefill_tokens(
        ctx, batch, new_tokens,
        _append_trajectory_token=lambda seq, token: appended.append(
            (seq, token),
        ),
        _should_stop_at_eos=lambda token, seq: token in stops,
    )
    return out, new_tokens, appended


def test_the_writeback_copies_once_and_reads_each_row_on_the_host():
    """The one-copy-then-read pattern, observed rather than read off the AST."""
    ns, engine = runtime()
    seqs = [sequence(1, [1]), sequence(1, [2]), sequence(1, [3])]
    out, new_tokens, appended = writeback(
        ns, engine, seqs=seqs, batch=[0, 1, 2],
    )
    assert new_tokens.copies == 1, "one device copy per batch, not per row"
    assert [token for _, token in appended] == [11, 12, 13]
    assert [seq.advanced for seq in seqs] == [1, 1, 1]
    assert out is new_tokens


def test_the_writeback_appends_before_it_advances_and_latches_eos():
    ns, engine = runtime()
    seqs = [sequence(1, [1]), sequence(1, [2])]
    _, _, appended = writeback(
        ns, engine, seqs=seqs, batch=[0, 1], stops={12},
    )
    assert [seq is got for (got, _), seq in zip(appended, seqs)] == [True, True]
    assert [seq.eos_reached for seq in seqs] == [False, True]


def test_a_writeback_shaped_against_another_batch_is_refused_by_rank():
    ns, engine = runtime()
    seqs = [sequence(1, [1])]
    with pytest.raises(RuntimeError, match=r"Rank 3: prefill writeback shape"):
        writeback(ns, engine, seqs=seqs, batch=[0, 1],
                  tokens=CountingTokens([5]))


# --------------------------------------------------------------------------- #
#  D7.5: the four worker delegators
# --------------------------------------------------------------------------- #
def execution_worker(monkeypatch, ns, engine):
    """A worker shell carrying every value and collaborator the four need."""
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 2
    worker.local_rank = 1
    worker.world_size = 8
    worker.torch_device = "cuda:1"
    worker.model = "model"
    worker.model_config = "model_config"
    worker.engine_config = "engine_config"
    worker.global_batch = "pool"
    worker.parallel_manager = "psm"
    worker._local_to_uuid_map = {0: "u0"}
    worker._weight_passes = "passes"
    worker._prefill_pool_wave = "wave"
    worker._prefill_pool_base = (1, 2, 3)
    # Collaborators the worker owns outside this engine's own delegators.
    for name in (
        "_bind_prefill_host_views", "_prefill_deepseek_family_name",
        "_trajectory_tokens", "feed_watchdog", "_bind_prefill_step_metadata",
        "_bind_prefill_step_batch", "_select_tokens",
        "_append_trajectory_batch", "_should_stop_at_eos",
        "_append_trajectory_token", "_record_unsharded_epilogue_gather",
        "_bind_prefill_pool_segments", "_bind_prefill_pool_layer_state",
        "_install_prefill_pool_rows", "_release_kda_state_slots",
        "_drop_prefill_pool_offload_recorder", "_unbind_prefill_pool_segments",
    ):
        setattr(worker, name, lambda *a, _n=name, **k: _n)
    return worker


def test_the_padded_forward_delegator_builds_its_context(monkeypatch):
    ns, engine = runtime()
    worker = execution_worker(monkeypatch, ns, engine)
    seen = []
    recording(engine, "prefill", seen)

    assert worker.prefill([0, 1]) == "recorded"
    (context, batch), kwargs = seen[0]
    assert isinstance(context, ns["PrefillBatchForwardContext"])
    assert (
        context.rank, context.torch_device, context.model,
        context.model_config, context.engine_config, context.pool,
        context.local_to_uuid, context._weight_passes,
    ) == (2, "cuda:1", "model", "model_config", "engine_config", "pool",
          {0: "u0"}, "passes")
    assert batch == [0, 1]
    assert kwargs == {
        "_bind_prefill_host_views": worker._bind_prefill_host_views,
        "_prefill_deepseek_family_name": worker._prefill_deepseek_family_name,
        "_trajectory_tokens": worker._trajectory_tokens,
        "feed_watchdog": worker.feed_watchdog,
        "_bind_prefill_step_metadata": worker._bind_prefill_step_metadata,
        "_bind_prefill_step_batch": worker._bind_prefill_step_batch,
        "_select_tokens": worker._select_tokens,
        "_append_trajectory_batch": worker._append_trajectory_batch,
        "_should_stop_at_eos": worker._should_stop_at_eos,
    }


def test_the_model_pass_delegator_reads_the_live_model_each_call(monkeypatch):
    """The model is bound and dropped with the phase, so it is read per call."""
    ns, engine = runtime()
    worker = execution_worker(monkeypatch, ns, engine)
    seen = []
    recording(engine, "_prefill_model_pass", seen)

    worker._prefill_model_pass("ids", "rows", lmhead_fp32=True)
    worker.model = "rebound"
    worker.parallel_manager = "psm2"
    worker._prefill_model_pass("ids", None, lmhead_fp32=False, trace="t")

    assert [(args[0].model, args[0].parallel_manager) for args, _ in seen] == [
        ("model", "psm"), ("rebound", "psm2"),
    ]
    assert [args[1:] for args, _ in seen] == [("ids", "rows"), ("ids", None)]
    assert [kwargs for _, kwargs in seen] == [
        {
            "lmhead_fp32": True, "trace": None, "layer_probe": None,
            "_record_unsharded_epilogue_gather":
                worker._record_unsharded_epilogue_gather,
        },
        {
            "lmhead_fp32": False, "trace": "t", "layer_probe": None,
            "_record_unsharded_epilogue_gather":
                worker._record_unsharded_epilogue_gather,
        },
    ]


def test_the_writeback_delegator_builds_its_narrow_context(monkeypatch):
    ns, engine = runtime()
    worker = execution_worker(monkeypatch, ns, engine)
    seen = []
    recording(engine, "_write_prefill_tokens", seen)

    assert worker._write_prefill_tokens([0], "tokens") == "recorded"
    (context, batch, tokens), kwargs = seen[0]
    assert isinstance(context, ns["PrefillTokenWritebackContext"])
    assert (context.rank, context.pool, context.local_to_uuid) == (
        2, "pool", {0: "u0"},
    )
    assert (batch, tokens) == ([0], "tokens")
    # Exactly two collaborators: it resolves no model and no weight pass.
    assert kwargs == {
        "_append_trajectory_token": worker._append_trajectory_token,
        "_should_stop_at_eos": worker._should_stop_at_eos,
    }


def test_the_pool_driver_delegator_passes_fourteen_collaborators(monkeypatch):
    ns, engine = runtime()
    worker = execution_worker(monkeypatch, ns, engine)
    seen = []
    recording(engine, "_prefill_prepacked_pool", seen)

    assert worker._prefill_prepacked_pool([0]) == "recorded"
    (context, batch), kwargs = seen[0]
    assert isinstance(context, ns["PrefillPoolWaveRunContext"])
    assert (
        context.rank, context.local_rank, context.world_size,
        context.engine_config, context.pool, context.local_to_uuid,
        context.parallel_manager, context._weight_passes,
        context._prefill_pool_wave, context._prefill_pool_base,
    ) == (2, 1, 8, "engine_config", "pool", {0: "u0"}, "psm", "passes",
          "wave", (1, 2, 3))
    assert batch == [0]
    # The shared model pass is among them: it no longer takes a worker, so
    # this delegator hands over the worker's own same-name delegator.
    assert set(kwargs) == {
        "_construct_prefill_pool", "_new_pool_epilogue_tracer",
        "_bind_prefill_pool_segments", "_bind_prefill_pool_layer_state",
        "_materialized_segment_loader", "feed_watchdog",
        "_install_prefill_pool_rows", "_load_materialized_segments",
        "_prefill_model_pass", "_select_tokens", "_release_kda_state_slots",
        "_drop_prefill_pool_offload_recorder",
        "_unbind_prefill_pool_segments", "_write_prefill_tokens",
    }
    assert len(kwargs) == 14
    for name, value in kwargs.items():
        assert value == getattr(worker, name), name


def test_the_pool_driver_delegator_reads_the_live_wave_and_baseline(monkeypatch):
    """Both are rebound per wave -- the baseline by the construction this
    body's own first statement may trigger -- so neither may be snapshotted."""
    ns, engine = runtime()
    worker = execution_worker(monkeypatch, ns, engine)
    seen = []
    recording(engine, "_prefill_prepacked_pool", seen)

    worker._prefill_pool_wave = "first"
    worker._prefill_pool_base = (0, 0, 0)
    worker._prefill_prepacked_pool([])
    worker._prefill_pool_wave = "second"
    worker._prefill_pool_base = (9, 9, 9)
    worker._prefill_prepacked_pool([])

    assert [(args[0]._prefill_pool_wave, args[0]._prefill_pool_base)
            for args, _ in seen] == [
        ("first", (0, 0, 0)), ("second", (9, 9, 9)),
    ]


def test_every_d7_5_delegator_names_no_worker_in_its_engine_call():
    """One assertion over the stage: no engine call passes a bare self."""
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    checked = set()
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in D7_5_METHODS:
            continue
        checked.add(node.name)
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "engine"):
                continue
            assert not any(isinstance(arg, ast.Name) and arg.id == "self"
                           for arg in call.args), node.name
            assert not any(isinstance(kw.value, ast.Name)
                           and kw.value.id == "self"
                           for kw in call.keywords), node.name
    assert checked == D7_5_METHODS


# --------------------------------------------------------------------------- #
#  D7.6: the unpooled prepacked driver -- one method, a wide context proved
#  honest, no collective, and the file's one explicit stream wait
# --------------------------------------------------------------------------- #
PREPACKED = "prefill_prepacked"
PREPACKED_CONTEXT = "PrefillPrepackedRunContext"
# The twenty-two worker methods this driver reached through the worker, in the
# order its signature declares them. `_prefill_model_pass` is among them since
# this stage: the pass takes a context a converted body may not build, so it
# arrives as a collaborator and the worker's own delegator builds it -- which
# is also where the unsharded-gather recorder is resolved, so this body stops
# naming that one.
PREPACKED_COLLABORATORS = (
    "_bind_prefill_host_views",
    "_prefill_prepacked_pool",
    "_prefill_deepseek_family_name",
    "_trajectory_tokens",
    "_plan_prefill_micro_batches",
    "_debug_flag_enabled",
    "_k3_prefill_profile_flag_name",
    "_glm5_lmhead_fp32_flag_name",
    "feed_watchdog",
    "_sync_prefill_moe_rank_counts",
    "_prefix_materialization_kv_config",
    "_new_prefill_temporary_gpu_kv_manager",
    "_install_prefill_prepack_metadata",
    "_prefill_model_pass",
    "_select_tokens",
    "_retire_prefill_pending_offloads",
    "_close_prefix_materialization",
    "_clear_prefill_prefix_materialization",
    "_emit_k3_prefill_profile_log",
    "_reset_prefill_prepack_metadata",
    "_log_prefill_timing",
    "_write_prefill_tokens",
)
# The five model-named ones. Phase E2 removes model-specific branches from the
# worker and the engines; this stage keeps every one of them, which is why
# these names survive the conversion instead of being resolved away.
PREPACKED_MODEL_NAMED = (
    "_prefill_deepseek_family_name",
    "_glm5_lmhead_fp32_flag_name",
    "_k3_prefill_profile_flag_name",
    "_emit_k3_prefill_profile_log",
    "_sync_prefill_moe_rank_counts",
)
# Every process-group entry point. NOT the stream/event names D7.5 could check
# for as well: this body has the file's one explicit stream wait, pinned
# separately below by position and conditionality.
_COLLECTIVE_NAMES = (
    "all_reduce", "all_gather", "all_gather_into_tensor", "broadcast",
    "barrier", "reduce", "reduce_scatter", "send", "recv", "isend", "irecv",
    "gather", "scatter",
)


def _parent_map(node):
    out = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            out[id(child)] = parent
    return out


def _enclosing(node, target, parents):
    """The compound statements ``target`` stands under, outermost first."""
    out, cur = [], parents.get(id(target))
    while cur is not None and cur is not node:
        if isinstance(cur, ast.If):
            out.append("if " + ast.unparse(cur.test))
        elif isinstance(cur, ast.For):
            out.append("for " + ast.unparse(cur.target) + " in ...")
        elif isinstance(cur, ast.While):
            out.append("while " + ast.unparse(cur.test))
        elif isinstance(cur, ast.With):
            out.append("with " + ", ".join(
                ast.unparse(item.context_expr) for item in cur.items
            ))
        elif isinstance(cur, ast.Try):
            out.append("try")
        elif isinstance(cur, (ast.FunctionDef, ast.Lambda)):
            out.append("nested " + getattr(cur, "name", "<lambda>"))
        cur = parents.get(id(cur))
    return list(reversed(out))


def _statements(node):
    """The method's top-level statements, docstring dropped."""
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def _ctx_reads(statements):
    return {
        sub.attr
        for stmt in statements for sub in ast.walk(stmt)
        if isinstance(sub, ast.Attribute)
        and isinstance(sub.value, ast.Name) and sub.value.id == "ctx"
    }


def test_the_prepacked_driver_declares_the_context_it_reads():
    node = method_node(PREPACKED)
    assert [arg.arg for arg in node.args.args] == ["self", "ctx", "batch"]
    assert ast.unparse(node.args.args[1].annotation) == PREPACKED_CONTEXT
    # ``token_cap`` stays the delegator-resolved VALUE it already was, and it
    # comes first so the collaborators read as one block.
    assert [arg.arg for arg in node.args.kwonlyargs] == [
        "token_cap", *PREPACKED_COLLABORATORS,
    ]
    assert len(PREPACKED_COLLABORATORS) == 22


def test_the_prepacked_context_is_one_honest_context_not_two():
    """Wide, and measured: no cut through the body splits the reads.

    Eighteen fields is a lot, so the share question has to be answered rather
    than asserted. For EVERY cut between two top-level statements where both
    sides read something at all, the two read sets intersect -- so there is no
    pair of phases with disjoint read sets to split this into, and two
    contexts would each have to carry most of the fields anyway.
    """
    statements = _statements(method_node(PREPACKED))
    assert len(statements) == 51
    split = []
    for cut in range(1, len(statements)):
        before = _ctx_reads(statements[:cut])
        after = _ctx_reads(statements[cut:])
        if before and after and not before & after:
            split.append((cut, sorted(before), sorted(after)))
    assert split == [], split
    # The widest honest split still shares seven of the eighteen fields, which
    # is what makes a two-context version a duplication rather than a
    # narrowing.
    best = max(
        len(_ctx_reads(statements[:cut]) & _ctx_reads(statements[cut:]))
        for cut in range(1, len(statements))
        if _ctx_reads(statements[:cut]) and _ctx_reads(statements[cut:])
    )
    assert best == 7


def test_the_prepacked_context_is_no_earlier_contexts_read_set():
    """The only ground on which this stage could have reused one."""
    ns, _ = runtime()
    mine = frozenset(f.name for f in fields(ns[PREPACKED_CONTEXT]))
    assert len(mine) == 18
    earlier = {
        ast.unparse(method_node(other).args.args[1].annotation)
        for other in sorted(METHODS - STATELESS_METHODS - D7_6_METHODS)
    }
    for name in sorted(earlier):
        assert mine != frozenset(f.name for f in fields(ns[name])), name
    # The two it is closest to, with the exact difference each way: the padded
    # forward's set is a strict SUBSET (sharing it would mean widening it to
    # this one, i.e. no share), and the pooled driver's is neither.
    padded = frozenset(f.name for f in fields(ns["PrefillBatchForwardContext"]))
    pooled = frozenset(f.name for f in fields(ns["PrefillPoolWaveRunContext"]))
    assert padded < mine
    assert mine - padded == {
        "world_size", "parallel_manager", "core_engine", "prefix",
        "prefix_coordinator", "prefix_runtime", "prefix_cache_debug_stats",
        "_prefill_pool_wave", "_prefill_global_passes",
        "_phase_prefill_debug",
    }
    assert not (pooled <= mine) and not (mine <= pooled)
    assert pooled - mine == {"local_rank", "_prefill_pool_base"}


def test_no_earlier_stages_context_was_widened_for_the_prepacked_stage():
    """D7.1-D7.5's contexts keep exactly the fields they were declared with."""
    ns, _ = runtime()
    assert [f.name for f in fields(ns["PrefillBatchForwardContext"])] == [
        "rank", "torch_device", "model", "model_config", "engine_config",
        "pool", "local_to_uuid", "_weight_passes",
    ]
    assert [f.name for f in fields(ns["PrefillModelPassContext"])] == [
        "model", "parallel_manager",
    ]
    assert [f.name for f in fields(ns["PrefillTokenWritebackContext"])] == [
        "rank", "pool", "local_to_uuid",
    ]
    assert [f.name for f in fields(ns["PrefillPoolWaveRunContext"])] == [
        "rank", "local_rank", "world_size", "engine_config", "pool",
        "local_to_uuid", "parallel_manager", "_weight_passes",
        "_prefill_pool_wave", "_prefill_pool_base",
    ]
    assert [f.name for f in fields(ns["PoolProbeContext"])] == [
        "rank", "_phase_prefill_debug",
    ]
    assert [f.name for f in fields(ns["PrefillPoolWaveContext"])] == [
        "_prefill_pool_wave",
    ]


def test_the_prepacked_stage_declares_no_outcome_and_makes_one_store():
    """One store, and it is an in-place mutation of the model the ctx carries.

    The family branch clearing the model wrapper's attention-2 flag is the
    whole of this body's writing, so there is nothing for a delegator to
    apply and the stage needs no outcome, no sentinel and no fatal-path
    reasoning about a write lost mid-function.
    """
    assert not (OUTCOME_METHODS & D7_6_METHODS)
    ns, _ = runtime()
    outcomes = {name for name in ns if name.endswith("Outcome")}
    node = method_node(PREPACKED)
    assert not ({
        sub.func.id for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    } & outcomes)
    stores = []
    for sub in ast.walk(node):
        targets = []
        if isinstance(sub, ast.Assign):
            targets = list(sub.targets)
        elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)):
            targets = [sub.target]
        for target in targets:
            parts = (target.elts
                     if isinstance(target, (ast.Tuple, ast.List))
                     else [target])
            for part in parts:
                if isinstance(part, ast.Attribute):
                    stores.append(ast.unparse(part))
    assert stores == ["ctx.model.model._use_flash_attention_2"]
    # Through the model, never a bare ``ctx.<field> = ...`` (which the frozen
    # context refuses) and never a worker attribute binding.
    assert not [
        sub for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and sub.id == "self"
    ]


def test_the_prepacked_driver_issues_no_collective():
    """Proved by absence: the body names ``dist`` and no collective at all.

    The pass alignment that a resident-EP prefill needs happens BEFORE this
    call -- a rank with no local rows never enters here, so a collective in
    this function would hang the serve group. The converted body must keep
    that property, and it does.
    """
    node = method_node(PREPACKED)
    assert not [
        sub for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and sub.id == "dist"
    ]
    reached = {
        sub.attr for sub in ast.walk(node) if isinstance(sub, ast.Attribute)
    } | {
        sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)
    }
    assert not reached & set(_COLLECTIVE_NAMES)


def test_the_prepacked_driver_keeps_its_ordered_host_syncs():
    """Seven host-pulling calls: the text, the ORDER and the conditions.

    The file's total is what the boundary checker ratchets at 45; this pins
    where this body's share of them is. The debug top-k pair is the site that
    could break silently: both halves of each ``.cpu().tolist()`` must stay
    inside the profile arm, because a half that escaped it would pull a tensor
    back on every production prefill.
    """
    node = method_node(PREPACKED)
    parents = _parent_map(node)
    in_loop = [
        "with torch.inference_mode(), ctx._weight_passes.scope('prefill')",
        "for (batch_idx, (seq_start, seq_end)) in ...",
        "if unpooled_trace is not None",
    ]
    profile_arm = ["if _k3_profile_enabled", "for profile_logits in ..."]
    expected = [
        ("tolist", "last_token_indices.tolist()", in_loop),
        ("tolist", "torch.cat([t.reshape(-1) for t in output_tokens]).tolist()",
         ["if ctx.rank == 0 and output_tokens"]),
        ("tolist", "top_indices.cpu().tolist()", profile_arm),
        ("cpu", "top_indices.cpu()", profile_arm),
        ("tolist", "top_values.cpu().tolist()", profile_arm),
        ("cpu", "top_values.cpu()", profile_arm),
        ("tolist", "output_tokens[0].reshape(-1).tolist()", []),
    ]
    found = [
        (method, text, _enclosing(
            node,
            next(sub for sub in ast.walk(node)
                 if isinstance(sub, ast.Call) and ast.unparse(sub) == text),
            parents,
        ))
        for method, text in _host_sync_calls(node)
    ]
    assert found == expected, found


def test_the_stream_wait_keeps_its_position_and_stands_under_no_condition():
    """The one explicit device wait in the file, pinned the way D7.3/D7.4
    pinned their collectives.

    A host perf-counter sample alone does not prove a non-default stream has
    finished, and the last layer's KV D2H is deliberately off the compute
    stream -- so this wait is what makes the GPU-complete boundary true. It
    must stay BETWEEN the two samples, and it must stay unconditional: a
    conversion that slid it inside the micro-batch loop or under the rank-0
    branch would still pass every other assertion here.
    """
    node = method_node(PREPACKED)
    parents = _parent_map(node)
    waits = [
        sub for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "synchronize"
    ]
    assert len(waits) == 1
    wait, = waits
    assert ast.unparse(wait) == (
        "torch.cuda.current_stream(ctx.torch_device).synchronize()"
    )
    assert _enclosing(node, wait, parents) == []
    statements = _statements(node)
    position = next(
        index for index, stmt in enumerate(statements)
        if any(sub is wait for sub in ast.walk(stmt))
    )
    assert ast.unparse(statements[position]) == ast.unparse(wait)
    assert ast.unparse(statements[position - 1]) == (
        "_prefill_forward_host_s = time.perf_counter() - _prefill_forward_t0"
    )
    assert ast.unparse(statements[position + 1]) == (
        "_prefill_forward_gpu_complete_s = time.perf_counter() "
        "- _prefill_forward_t0"
    )


def test_the_prepacked_driver_reaches_the_model_pass_as_a_collaborator():
    """The call D7.5 left transitional, finished -- and not duplicated.

    D7.5 converted the shared pass and this driver, still worker-shaped, kept
    the direct engine call with the pass's context built INLINE. Now the pass
    is a collaborator like the rest: the two positionals are unchanged, the
    worker's delegator builds the context and supplies the unsharded-gather
    recorder, and the inline construction is GONE from this body rather than
    living on beside the collaborator.
    """
    node = method_node(PREPACKED)
    assert "_prefill_model_pass" in {a.arg for a in node.args.kwonlyargs}
    calls = [sub for sub in ast.walk(node)
             if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
             and sub.func.id == "_prefill_model_pass"]
    assert len(calls) == 1
    call, = calls
    assert [ast.unparse(arg) for arg in call.args] == [
        "batch_input_ids_flat.to(ctx.torch_device, dtype=torch.int64)",
        "last_token_indices",
    ]
    assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords} == {
        "lmhead_fp32": "_lmhead_fp32",
        "trace": "unpooled_trace",
        "layer_probe": "_memprof_probe",
    }
    # No engine capture, no inline context, and the recorder is not named here
    # any more -- the delegator resolves it, as it already did for the pooled
    # driver.
    assert not [sub for sub in ast.walk(node)
                if isinstance(sub, ast.Name) and sub.id == "_engine"]
    assert not [sub for sub in ast.walk(node)
                if isinstance(sub, ast.Name)
                and sub.id == "PrefillModelPassContext"]
    assert not [sub for sub in ast.walk(node)
                if isinstance(sub, ast.Name)
                and sub.id == "_record_unsharded_epilogue_gather"]
    # The pooled driver still builds no context either: the inline
    # construction moved to the worker, not to the sibling driver.
    pool = method_node("_prefill_prepacked_pool")
    assert not [sub for sub in ast.walk(pool)
                if isinstance(sub, ast.Name)
                and sub.id == "PrefillModelPassContext"]


def test_the_prepacked_driver_keeps_the_pooled_delegators_spelling():
    """The pooled hand-off is reached by name, as the pre-move body did.

    ``_prefill_prepacked_pool`` is itself converted (D7.5), but this driver
    called the worker's same-name delegator, not the engine, so it keeps
    receiving that delegator as a collaborator -- which also keeps it from
    having to build the pooled driver's own context.
    """
    node = method_node(PREPACKED)
    calls = [sub for sub in ast.walk(node)
             if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
             and sub.func.id == "_prefill_prepacked_pool"]
    assert len(calls) == 1
    call, = calls
    assert [ast.unparse(arg) for arg in call.args] == ["batch"]
    assert not call.keywords
    # Inside the weight-pass scope, as before, and it is the early return.
    parents = _parent_map(node)
    assert _enclosing(node, call, parents) == [
        "if ctx._prefill_pool_wave is not None",
        "with ctx._weight_passes.scope('prefill')",
    ]


def test_every_model_specific_branch_is_kept_as_a_collaborator():
    """Phase E2's work is not done here, and the count says so.

    Five of the twenty-two collaborators are model-named. They are kept
    because the branches they feed are kept: the family branch that clears the
    attention-2 flag, the two debug-key readers, the K3 profile emitter and
    the resident-EP rank-count sync.
    """
    node = method_node(PREPACKED)
    declared = {a.arg for a in node.args.kwonlyargs}
    for name in PREPACKED_MODEL_NAMED:
        assert name in declared, name
        assert [sub for sub in ast.walk(node)
                if isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == name], name
    assert len(set(PREPACKED_MODEL_NAMED)) == 5


# --------------------------------------------------------------------------- #
#  D7.6: the worker delegator
# --------------------------------------------------------------------------- #
def prepacked_worker(monkeypatch, ns, engine):
    """A worker shell carrying every value and collaborator the driver needs."""
    worker = load_delegators(monkeypatch, ns)
    worker._prefill_engine = engine
    worker.rank = 2
    worker.world_size = 8
    worker.torch_device = "cuda:1"
    worker.model = "model"
    worker.model_config = "model_config"
    worker.engine_config = "engine_config"
    worker.global_batch = "pool"
    worker._local_to_uuid_map = {0: "u0"}
    worker.parallel_manager = "psm"
    worker.core_engine = "core"
    worker._prefix = "prefix"
    worker.prefix_cache_coordinator = "coordinator"
    worker.prefix_cache_runtime_config = "runtime"
    worker.prefix_cache_debug_stats = True
    worker._weight_passes = "passes"
    worker._prefill_pool_wave = "wave"
    worker._phase_prefill_debug = {"k": "v"}
    worker._server_prefill_token_cap = lambda: 4096
    for name in PREPACKED_COLLABORATORS:
        setattr(worker, name, lambda *a, _n=name, **k: _n)
    return worker


def test_the_prepacked_delegator_builds_its_context(monkeypatch):
    ns, engine = runtime()
    worker = prepacked_worker(monkeypatch, ns, engine)
    worker._prefill_global_passes = 3
    seen = []
    recording(engine, PREPACKED, seen)

    assert worker.prefill_prepacked([0, 1]) == "recorded"
    (context, batch), kwargs = seen[0]
    assert isinstance(context, ns[PREPACKED_CONTEXT])
    assert {
        field.name: getattr(context, field.name)
        for field in fields(context)
    } == {
        "rank": 2, "world_size": 8, "torch_device": "cuda:1",
        "model": "model", "model_config": "model_config",
        "engine_config": "engine_config", "pool": "pool",
        "local_to_uuid": {0: "u0"}, "parallel_manager": "psm",
        "core_engine": "core", "prefix": "prefix",
        "prefix_coordinator": "coordinator", "prefix_runtime": "runtime",
        "prefix_cache_debug_stats": True, "_weight_passes": "passes",
        "_prefill_pool_wave": "wave", "_prefill_global_passes": 3,
        "_phase_prefill_debug": {"k": "v"},
    }
    assert batch == [0, 1]
    # The cap is resolved HERE, as it already was, and the twenty-two
    # collaborators are each the worker's own method of that name.
    assert kwargs.pop("token_cap") == 4096
    assert set(kwargs) == set(PREPACKED_COLLABORATORS)
    assert len(kwargs) == 22
    for name, value in kwargs.items():
        assert value == getattr(worker, name), name


def test_the_prepacked_delegator_passes_the_pass_count_miss_as_none(monkeypatch):
    """The ONE field the pre-move body read through a presence guard.

    Only the pass alignment that precedes a resident-EP prefill binds
    ``_prefill_global_passes``, and the pre-move body read it with a
    ``getattr`` default. A frozen context always carries the field, so the
    default lives in the delegator -- a shell that never ran the alignment
    must see ``None``, not an AttributeError.
    """
    ns, engine = runtime()
    worker = prepacked_worker(monkeypatch, ns, engine)
    seen = []
    recording(engine, PREPACKED, seen)

    assert not hasattr(worker, "_prefill_global_passes")
    worker.prefill_prepacked([0])
    worker._prefill_global_passes = 7
    worker.prefill_prepacked([0])

    assert [args[0]._prefill_global_passes for args, _ in seen] == [None, 7]


def test_the_prepacked_delegator_reads_the_live_phase_state(monkeypatch):
    """Every one of these is rebound with the phase or with the batch, so the
    delegator may not snapshot them: the model and its config, the prefix
    cache handles, the weight-copy pipeline, the planned wave and the adopted
    debug state."""
    ns, engine = runtime()
    worker = prepacked_worker(monkeypatch, ns, engine)
    seen = []
    recording(engine, PREPACKED, seen)

    worker.prefill_prepacked([0])
    worker.model = "rebound"
    worker.model_config = "rebound_config"
    worker.prefix_cache_runtime_config = "runtime2"
    worker._weight_passes = "passes2"
    worker._prefill_pool_wave = None
    worker._phase_prefill_debug = {}
    worker.prefill_prepacked([0])

    assert [
        (args[0].model, args[0].model_config, args[0].prefix_runtime,
         args[0]._weight_passes, args[0]._prefill_pool_wave,
         args[0]._phase_prefill_debug)
        for args, _ in seen
    ] == [
        ("model", "model_config", "runtime", "passes", "wave", {"k": "v"}),
        ("rebound", "rebound_config", "runtime2", "passes2", None, {}),
    ]


def test_the_prepacked_delegator_names_no_worker_in_its_engine_call():
    """One assertion over the stage: the engine call passes no bare self."""
    tree = ast.parse(WORKER_SOURCE.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    checked = set()
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in D7_6_METHODS:
            continue
        checked.add(node.name)
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "engine"):
                continue
            assert not any(isinstance(arg, ast.Name) and arg.id == "self"
                           for arg in call.args), node.name
            assert not any(isinstance(kw.value, ast.Name)
                           and kw.value.id == "self"
                           for kw in call.keywords), node.name
    assert checked == D7_6_METHODS
