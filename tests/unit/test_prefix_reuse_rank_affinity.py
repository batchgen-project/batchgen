import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_torch_stub(monkeypatch):
    torch_stub = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    batchgen_stub = types.ModuleType("batchgen")
    batchgen_stub.__path__ = [str(REPO_ROOT / "batchgen")]
    monkeypatch.setitem(sys.modules, "batchgen", batchgen_stub)


def _rank_affinity_module(monkeypatch):
    _install_torch_stub(monkeypatch)
    return importlib.import_module("batchgen.prefix_reuse.rank_affinity")


def _seq(uuid, prompt_length, assigned_rank=None):
    return SimpleNamespace(
        uuid=uuid,
        prompt_length=prompt_length,
        assigned_rank=assigned_rank,
    )


def test_prefix_rank_hint_is_assigned_before_l2_balance(monkeypatch):
    mod = _rank_affinity_module(monkeypatch)
    sequences = {
        "new-a": _seq("new-a", 100),
        "new-b": _seq("new-b", 200),
        "old": _seq("old", 300, assigned_rank=0),
    }

    result = mod.assign_admitted_ranks(
        uuids=["new-a", "new-b"],
        existing_sequences=sequences.values(),
        get_sequence=sequences.get,
        world_size=2,
        use_l2_balance=True,
        prefix_rank_lookup=lambda seq: 1 if seq.uuid == "new-a" else None,
    )

    assert ("new-a", 1) in result.assignments
    assert result.prefix_assigned_count == 1
    assert result.prefix_assigned_by_rank == [0, 1]


def test_l2_assignment_prefers_lower_existing_prompt_load(monkeypatch):
    mod = _rank_affinity_module(monkeypatch)
    sequences = {
        "new": _seq("new", 100),
        "old-heavy": _seq("old-heavy", 300, assigned_rank=0),
        "old-light": _seq("old-light", 10, assigned_rank=1),
    }

    result = mod.assign_admitted_ranks(
        uuids=["new"],
        existing_sequences=sequences.values(),
        get_sequence=sequences.get,
        world_size=2,
        use_l2_balance=True,
    )

    assert result.assignments == [("new", 1)]


def test_legacy_count_assignment_uses_least_count(monkeypatch):
    mod = _rank_affinity_module(monkeypatch)
    sequences = {
        "new": _seq("new", 100),
        "old-0": _seq("old-0", 10, assigned_rank=0),
        "old-1": _seq("old-1", 10, assigned_rank=0),
    }

    result = mod.assign_admitted_ranks(
        uuids=["new"],
        existing_sequences=sequences.values(),
        get_sequence=sequences.get,
        world_size=2,
        use_l2_balance=False,
    )

    assert result.assignments == [("new", 1)]
