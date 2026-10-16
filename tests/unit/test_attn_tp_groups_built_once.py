"""The head-parallel NCCL sub-groups are built once per process, not per model.

``_build_attn_tp_group`` runs inside ``_build_model``, which runs again on every
rebuild of a phase instance. Each call created the attention, node-local weight
and cross-node weight families with ``dist.new_group``, and torch keeps every
group it creates for the life of the process (only ``destroy_process_group``
drops one, which no phase switch calls) while each communicator's device
buffers live OUTSIDE the caching allocator. So every cycle added a set of
communicators that nothing held, nothing freed and no torch counter could see:
driver-used grew linearly per switch with torch allocated/reserved flat, until
the largest later allocation — the KV pool's ``cuMemCreate`` — hit the wall.

The group families are compiled out of the PSM's AST onto fakes with a counting
``new_group``, so nothing here needs NCCL, CUDA or a PSM import.
"""

import ast
import functools
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch.distributed as dist


PSM = (
    Path(__file__).resolve().parents[2] / "batchgen" / "models" / "moonshotai"
    / "kimi_linear" / "Parallel_Strategy_Manager.py"
)

WORLD = 16
G = 8


@functools.lru_cache(maxsize=1)
def _psm_class():
    tree = ast.parse(PSM.read_text())
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KimiLinearParallelStrategyManager"
    )


def _method_ast(name):
    return next(
        node for node in _psm_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@functools.lru_cache(maxsize=4)
def _methods(names):
    module = ast.fix_missing_locations(
        ast.Module(body=[_method_ast(name) for name in names], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, str(PSM), "exec"), namespace)
    return {name: namespace[name] for name in names}


@pytest.fixture
def groups(monkeypatch):
    """Count ``new_group`` calls and hand back identifiable group objects."""
    created = []

    def new_group(ranks=None, **kwargs):
        created.append(tuple(ranks or ()))
        return f"group#{len(created)}:{tuple(ranks or ())}"

    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    return created


def _psm(*, attn_tp_size=G, rank=0, built=False):
    psm = SimpleNamespace(
        world_size=WORLD,
        global_rank=rank,
        rank=rank,
        _attn_tp_size=attn_tp_size,
        _attn_tp_rank=rank % attn_tp_size if attn_tp_size else 0,
        _attn_tp_group_id=rank // attn_tp_size if attn_tp_size else 0,
        _attn_tp_group=None,
        _weight_tp_group=None,
        _cross_weight_group=None,
        _cross_weight_root=None,
        _cross_weight_source=None,
        _attn_tp_groups_built=built,
        _attn_tp_group_signature=None,
        # Hierarchical GDR on: the cross-node weight family is built too.
        _hierarchical_gdr=True,
    )
    for name, function in _methods((
        "_build_attn_tp_group", "_build_cross_weight_group",
    )).items():
        setattr(psm, name, MethodType(function, psm))
    return psm


def test_a_rebuild_creates_no_new_communicators(groups):
    psm = _psm()
    psm._build_attn_tp_group()
    first_count = len(groups)
    assert first_count > 0
    first = (psm._attn_tp_group, psm._weight_tp_group, psm._cross_weight_group)

    # What a phase-instance rebuild does: _build_model calls it again.
    for _ in range(4):
        psm._build_attn_tp_group()

    assert len(groups) == first_count, "a rebuild created new NCCL groups"
    assert (psm._attn_tp_group, psm._weight_tp_group,
            psm._cross_weight_group) == first


def test_the_first_build_creates_every_family_once(groups):
    psm = _psm()
    psm._build_attn_tp_group()
    # attention: world/G, node-local weight: world/G, cross-node: G.
    assert len(groups) == WORLD // G + WORLD // G + G
    assert psm._attn_tp_groups_built is True


def test_each_rank_keeps_the_group_it_belongs_to(groups):
    psm = _psm(rank=9)
    psm._build_attn_tp_group()
    assert "(8, 9, 10, 11, 12, 13, 14, 15)" in psm._attn_tp_group
    assert "(8, 9, 10, 11, 12, 13, 14, 15)" in psm._weight_tp_group
    # Cross-node family: local slot 1 on every node.
    assert "(1, 9)" in psm._cross_weight_group


def test_without_hierarchical_gdr_only_the_node_local_families_are_built(groups):
    psm = _psm()
    psm._hierarchical_gdr = False
    psm._build_attn_tp_group()
    assert len(groups) == WORLD // G + WORLD // G
    assert psm._cross_weight_group is None
    psm._build_attn_tp_group()
    assert len(groups) == WORLD // G + WORLD // G


def test_pure_dp_builds_nothing_and_stays_unbuilt(groups):
    psm = _psm(attn_tp_size=1)
    psm._build_attn_tp_group()
    assert groups == []
    assert psm._attn_tp_groups_built is False


def test_the_signature_records_the_topology_the_groups_belong_to(groups):
    psm = _psm()
    assert psm._attn_tp_group_signature is None
    psm._build_attn_tp_group()
    assert psm._attn_tp_group_signature == (WORLD, G)
    # A same-signature rebuild keeps the fast return.
    psm._build_attn_tp_group()
    assert psm._attn_tp_group_signature == (WORLD, G)


@pytest.mark.parametrize("field,value,wanted", [
    ("world_size", 32, (32, G)),
    ("_attn_tp_size", 4, (WORLD, 4)),
])
def test_a_changed_topology_is_refused_by_name(groups, field, value, wanted):
    """Silent reuse would serve groups that no longer describe the world.

    In-process topology is immutable today (both values are fixed at PSM
    construction), so a second build request for a DIFFERENT one is a
    lifecycle error. The groups cannot be rebuilt — torch drops a group only
    at ``destroy_process_group`` and the live ones are baked into captured
    decode graphs — so it is refused loudly, like ``init_state_pools``'
    changed-layout refusal, rather than answered with the old communicators.
    """
    psm = _psm()
    psm._build_attn_tp_group()
    built = len(groups)
    setattr(psm, field, value)

    with pytest.raises(RuntimeError) as excinfo:
        psm._build_attn_tp_group()

    message = str(excinfo.value)
    assert str((WORLD, G)) in message, "the old signature is not named"
    assert str(wanted) in message, "the new signature is not named"
    assert len(groups) == built, "the refused call still created groups"


def test_an_uninitialized_process_group_is_still_refused(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="needs torch.distributed"):
        _psm()._build_attn_tp_group()


def test_the_flag_starts_false_and_is_not_model_scoped():
    init = ast.unparse(_method_ast("__init__"))
    assert "self._attn_tp_groups_built = False" in init
    assert "self._attn_tp_group_signature = None" in init
    # The groups are world-level: a model rebuild must NOT reset them, or the
    # leak returns.
    reset = ast.unparse(_method_ast("_reset_model_attached_state"))
    assert "_attn_tp_groups_built" not in reset


def test_the_guard_precedes_every_new_group_call():
    body = ast.unparse(_method_ast("_build_attn_tp_group"))
    guard = body.index("if self._attn_tp_groups_built:")
    first_new_group = body.index("dist.new_group(")
    mark = body.index("self._attn_tp_groups_built = True")
    assert guard < first_new_group < mark
