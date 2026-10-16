"""Executor cost profiles are keyed by serve-group size, world size optional.

A profile scores one executor: a pure-DP rank (group size 1) or one serve group
of G ranks. The group size always matches exactly; the world size matches
exactly unless a group-local profile leaves it unset.
"""

import ast
import dataclasses
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from batchgen.prefix_reuse import executor_cost
from batchgen.prefix_reuse.executor_cost import (
    ExecutorCostProfile,
    find_executor_cost_profile,
)


GIB = 1 << 30
WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


def _gptoss():
    """The GPT-OSS entry, by its key: the table holds more than one fit now."""
    return next(
        profile for profile in executor_cost.PROFILES
        if profile.model == "gpt-oss-120b"
    )


def _group_local(**overrides):
    fields = dict(
        model="kimi-k3",
        machine="test",
        compute_capability=(9, 0),
        memory_gib=(130.0, 150.0),
        group_size=8,
        world_size=None,
        fitted_world_size=16,
        intercept=0.0,
        per_1e5_computed_tokens=1.0,
        per_1e9_attention_pairs=0.0,
        per_100_item_rows=0.0,
        per_chunk=0.0,
    )
    fields.update(overrides)
    return ExecutorCostProfile(**fields)


def test_gptoss_profile_keeps_its_world_four_single_rank_key():
    profile = _gptoss()
    assert (profile.group_size, profile.world_size, profile.fitted_world_size) == (1, 4, 4)
    assert find_executor_cost_profile("gpt-oss-120b", (9, 0), 141 * GIB, 4, 1) is profile
    # Its expert shards change with the world size, so it never transfers.
    assert find_executor_cost_profile("gpt-oss-120b", (9, 0), 141 * GIB, 8, 1) is None
    # A serve group is a different executor even on the same world.
    assert find_executor_cost_profile("gpt-oss-120b", (9, 0), 141 * GIB, 4, 4) is None


def test_group_local_profile_matches_any_world_with_its_group_size(monkeypatch):
    profile = _group_local()
    monkeypatch.setattr(executor_cost, "PROFILES", (_gptoss(), profile))
    for world in (16, 32):
        assert find_executor_cost_profile("kimi-k3", (9, 0), 141 * GIB, world, 8) is profile
    # The fitted world is provenance only: matching never reads it.
    assert find_executor_cost_profile("kimi-k3", (9, 0), 141 * GIB, 64, 8) is profile


@pytest.mark.parametrize("world_size", [None, 16])
def test_group_size_mismatch_never_matches(monkeypatch, world_size):
    profile = _group_local(world_size=world_size)
    monkeypatch.setattr(executor_cost, "PROFILES", (profile,))
    for group in (1, 4, 16):
        assert find_executor_cost_profile("kimi-k3", (9, 0), 141 * GIB, 16, group) is None
    assert find_executor_cost_profile("kimi-k3", (9, 0), 141 * GIB, 16, 8) is profile


def test_world_pinned_group_profile_matches_its_world_only(monkeypatch):
    profile = _group_local(world_size=16)
    monkeypatch.setattr(executor_cost, "PROFILES", (profile,))
    assert find_executor_cost_profile("kimi-k3", (9, 0), 141 * GIB, 16, 8) is profile
    assert find_executor_cost_profile("kimi-k3", (9, 0), 141 * GIB, 32, 8) is None


def test_profile_fields_are_required():
    names = {field.name for field in dataclasses.fields(ExecutorCostProfile)}
    assert {"group_size", "world_size", "fitted_world_size"} <= names
    with pytest.raises(TypeError):
        find_executor_cost_profile("gpt-oss-120b", (9, 0), 141 * GIB, 4)


def _cached_profile_method():
    tree = ast.parse(WORKER.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "BatchGenWorker")
    wanted = [node for node in worker.body if isinstance(node, ast.FunctionDef)
              and node.name == "_cached_prefix_rank_cost_profile"]
    assert len(wanted) == 1
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {"torch": torch, "logging": logging}
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace["_cached_prefix_rank_cost_profile"]


@pytest.mark.parametrize("G,expected_group,matched", [(1, 1, False), (8, 8, True)])
def test_worker_lookup_passes_the_serve_group_size(
    monkeypatch, caplog, G, expected_group, matched
):
    calls = []
    devices = []
    profile = _group_local(group_size=expected_group) if matched else None

    def lookup(*args):
        calls.append(args)
        return profile

    def device_properties(index):
        devices.append(index)
        return SimpleNamespace(major=9, minor=0, total_memory=141 * GIB)

    monkeypatch.setattr(executor_cost, "find_executor_cost_profile", lookup)
    monkeypatch.setattr(torch.cuda, "get_device_properties", device_properties)
    worker = SimpleNamespace(
        rank=11, local_rank=3, world_size=16, model_name="moonshotai/Kimi-K3",
        _decode_attn_tp_size=lambda: G,
    )
    worker._cached_prefix_rank_cost_profile = MethodType(_cached_profile_method(), worker)
    with caplog.at_level(logging.INFO):
        assert worker._cached_prefix_rank_cost_profile() is profile
        assert worker._cached_prefix_rank_cost_profile() is profile
        # The default arm uses the profile; the "off" arm is covered in
        # test_prefix_cost_profile_switch.py.
    assert calls == [("moonshotai/Kimi-K3", (9, 0), 141 * GIB, 16, expected_group)]
    # Node-uniform device, not this rank's own local device.
    assert devices == [0]
    # Logged once, at first resolution, with the resolved identity.
    lines = [r.getMessage() for r in caplog.records if "[PREFIX_PACK]" in r.getMessage()]
    resolved = f"kimi-k3/test/g{expected_group}" if matched else "none"
    assert lines == [
        f"[PREFIX_PACK] rank 11 executor cost profile: model=kimi-k3 "
        f"group_size={expected_group} disabled=0 -> {resolved}"
    ]
