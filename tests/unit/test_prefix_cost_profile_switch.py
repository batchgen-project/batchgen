"""The cost-profile arm is a batch-level switch, and the cache follows it.

Scoring the packer's targets by predicted seconds (profile) and balancing their
computed tokens (no profile) are the two arms of the packing comparison, and
the difference between them is a wall time — so both arms have to run inside
ONE process, or the measurement is confounded with cold-start, allocator and
JIT state. That makes two things load-bearing: the arm is resolved from the
batch's ``batchgen_debug`` (replicated through the admitted-batch broadcast,
like ``prefix_rank_assignment_mode``), and the deployment's profile lookup is
cached PER ARM, because a single cached attribute would pin the whole process
to whichever arm arrived first.

Worker methods are compiled out of the module's AST onto fakes, as in
``test_decode_dp_group_packing.py``; the profile table and the lookup are real.
"""

import ast
import functools
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List, Optional

import pytest
import torch

from batchgen.prefix_reuse.executor_cost import (
    PROFILES,
    find_executor_cost_profile,
)


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"

K3_MODEL = "moonshotai/Kimi-Linear-48B-A3B"
SM90 = (9, 0)
MEM_141G = int(141 * 1024 ** 3)
WORLD = 16
G = 8


@functools.lru_cache(maxsize=1)
def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")


def _method_ast(name):
    return next(node for node in _worker_class().body
                if isinstance(node, ast.FunctionDef) and node.name == name)


@functools.lru_cache(maxsize=4)
def _methods(names):
    module = ast.fix_missing_locations(
        ast.Module(body=[_method_ast(name) for name in names], type_ignores=[])
    )
    scope = {"List": List, "Optional": Optional, "torch": torch,
             "logging": SimpleNamespace(info=lambda *a, **k: None)}
    exec(compile(module, str(WORKER), "exec"), scope)
    return {name: scope[name] for name in names}


class _Props:
    major, minor = SM90
    total_memory = MEM_141G


def _worker(*, model=K3_MODEL, world=WORLD, group=G, debug=None,
            device_properties=_Props):
    worker = SimpleNamespace(rank=0, world_size=world, model_name=model)
    worker._decode_attn_tp_size = lambda: group
    worker._active_batchgen_debug_for_sequences = lambda seqs: debug
    for name, function in _methods((
        "_cached_prefix_rank_cost_profile",
        "_prefix_rank_cost_profile_disabled",
    )).items():
        setattr(worker, name, MethodType(function, worker))
    worker._test_properties = device_properties
    return worker


@pytest.fixture(autouse=True)
def _device_properties(monkeypatch):
    """One H200-class device, so the real lookup can match on it."""
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: _Props, raising=False)


# ------------------------------------------------------- the table entry --- #


def test_the_k3_entry_matches_this_deployment():
    profile = find_executor_cost_profile(
        K3_MODEL, SM90, MEM_141G, WORLD, G
    )
    assert profile is not None
    assert (profile.model, profile.group_size, profile.world_size) == (
        "kimi-linear-48b-a3b", G, WORLD
    )


@pytest.mark.parametrize("kwargs,why", [
    (dict(group_size=1), "a pure-DP rank is not an 8-rank serve group"),
    (dict(world_size=8), "the fit saw one world only"),
    (dict(compute_capability=(10, 0)), "another compute capability"),
    (dict(total_memory_bytes=int(96 * 1024 ** 3)), "another memory class"),
    (dict(model_name="moonshotai/Kimi-Linear-96B-A6B"), "another checkpoint"),
])
def test_the_entry_refuses_every_neighbouring_layout(kwargs, why):
    call = dict(model_name=K3_MODEL, compute_capability=SM90,
                total_memory_bytes=MEM_141G, world_size=WORLD, group_size=G)
    call.update(kwargs)
    assert find_executor_cost_profile(**call) is None, why


def test_per_chunk_is_pinned_and_the_rest_are_positive():
    profile = find_executor_cost_profile(K3_MODEL, SM90, MEM_141G, WORLD, G)
    # Pinned, not fitted: no pooled wave in the fit had more than one chunk.
    assert profile.per_chunk == 0.0
    assert profile.per_1e5_computed_tokens > 0
    assert profile.per_1e9_attention_pairs > 0
    assert profile.per_100_item_rows > 0
    # A single-chunk wave therefore costs the intercept plus the three terms,
    # and chunks cannot change a score in this profile's domain.
    base = profile.seconds(10_000, 10 ** 8, 50, 1)
    assert profile.seconds(10_000, 10 ** 8, 50, 4) == base


def test_the_gpt_oss_entry_is_untouched():
    assert any(profile.model == "gpt-oss-120b" and profile.group_size == 1
               for profile in PROFILES)


# ------------------------------------------------------------- the arm --- #


@pytest.mark.parametrize("debug,expected", [
    (None, False),
    ({}, False),
    ({"prefix_rank_assignment_mode": "packing_v2"}, False),
    ({"prefix_rank_cost_profile": "on"}, False),
    ({"prefix_rank_cost_profile": "off"}, True),
    ({"prefix_rank_cost_profile": "OFF"}, True),
    ({"prefix_rank_cost_profile": " off "}, True),
])
def test_the_arm_is_read_from_the_batch(debug, expected):
    assert _worker(debug=debug)._prefix_rank_cost_profile_disabled([]) is expected


@pytest.mark.parametrize("value", ["", "0", "false", "no", True, 1, "disabled"])
def test_a_mistyped_arm_is_refused_not_read_as_on(value):
    worker = _worker(debug={"prefix_rank_cost_profile": value})
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        worker._prefix_rank_cost_profile_disabled([])


# ----------------------------------------------------------- the cache --- #


def test_the_default_arm_resolves_the_profile():
    worker = _worker()
    profile = worker._cached_prefix_rank_cost_profile(False)
    assert profile is not None and profile.model == "kimi-linear-48b-a3b"


def test_the_off_arm_resolves_to_the_token_balance_fallback():
    # None is exactly what "no profile matches" gives, which is the fallback
    # the packer already implements (rank_packing._score).
    assert _worker()._cached_prefix_rank_cost_profile(True) is None


def test_both_arms_alternate_in_one_process():
    worker = _worker()
    seen = [worker._cached_prefix_rank_cost_profile(disabled)
            for disabled in (False, True, False, True, False)]
    assert [item is None for item in seen] == [False, True, False, True, False]
    assert seen[0] is seen[2] is seen[4]      # one lookup, reused
    assert set(worker._prefix_rank_cost_profiles) == {False, True}


def test_the_arms_do_not_share_a_cache_entry(monkeypatch):
    worker = _worker()
    calls = []
    import batchgen.prefix_reuse.executor_cost as module
    real = module.find_executor_cost_profile

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "find_executor_cost_profile", counting)
    worker._cached_prefix_rank_cost_profile(True)
    assert calls == [], "the off arm must not even look a profile up"
    worker._cached_prefix_rank_cost_profile(False)
    worker._cached_prefix_rank_cost_profile(False)
    assert len(calls) == 1, "the on arm looks up once and caches"


def test_a_deployment_without_an_entry_falls_back_on_both_arms():
    worker = _worker(model="some/unfitted-model")
    assert worker._cached_prefix_rank_cost_profile(False) is None
    assert worker._cached_prefix_rank_cost_profile(True) is None


# --------------------------------------------------------- call sites --- #


@pytest.mark.parametrize("method", [
    "_pack_decode_dp_groups_for_sharing",
    "_assign_ranks_for_prefix_sharing",
])
def test_both_packer_call_sites_pass_the_batch_arm(method):
    body = ast.unparse(_method_ast(method))
    assert ("self._cached_prefix_rank_cost_profile("
            "self._prefix_rank_cost_profile_disabled(seqs))") in body


def test_the_cache_is_keyed_and_never_a_single_attribute():
    body = ast.unparse(_method_ast("_cached_prefix_rank_cost_profile"))
    assert "_prefix_rank_cost_profiles" in body
    assert "key = bool(disabled)" in body
    # The old single-attribute cache is what made the arm un-flippable.
    assert "hasattr(self, '_prefix_rank_cost_profile')" not in body
