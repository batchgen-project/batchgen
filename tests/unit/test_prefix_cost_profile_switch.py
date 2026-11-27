"""The cost-profile arm is a launch parameter, and the cache follows it.

Scoring the packer's targets by predicted seconds (profile) and balancing their
computed tokens (no profile) are the two arms of the packing comparison. The
arm is a property of the server (``--prefix-rank-cost-profile``): the worker
passes its launch value to the engine, so every rank resolves the same arm.
The deployment's profile lookup is cached PER ARM, so the cache can never
answer for the other arm.

Worker methods are compiled out of the module's AST onto fakes, as in
``test_decode_dp_group_packing.py``; the profile table and the lookup are real.
"""

import ast
import functools
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from batchgen.prefix_reuse.executor_cost import (
    PROFILES,
    find_executor_cost_profile,
)
from tests.prefill_source import load_prefill


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"
ENGINE = Path(__file__).resolve().parents[2] / "batchgen" / "worker" / "prefill_engine.py"

K3_MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
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


def _engine_method_ast(name):
    tree = ast.parse(ENGINE.read_text())
    klass = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "PrefillEngine")
    return next(node for node in klass.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


@functools.lru_cache(maxsize=4)
def _methods(names):
    """The engine bodies under test, compiled from the engine's own AST.

    ``tests.prefill_source`` is this engine's loader and already knows how to
    compile its converted definitions -- the frozen contexts and the outcome
    sentinel included, which the cost-profile lookup now needs -- so the
    namespace comes from there instead of a second hand-rolled compile.
    """
    namespace = load_prefill({
        "List": List, "Optional": Optional, "torch": torch,
        "logging": SimpleNamespace(info=lambda *a, **k: None),
    })
    engine = namespace["PrefillEngine"]
    return {name: _adapt(name, getattr(engine, name), namespace)
            for name in names}


def _adapt(name, function, namespace):
    """Call the engine method the way its worker delegator does.

    ``None`` always stands for the stateless engine's own ``self``. A method
    that still takes the worker gets the fake as its seam argument; the
    converted cost-profile predicate reads no worker state at all, so it gets
    only the launch value its delegator resolves (the caller passes it); and
    the converted lookup takes a frozen context and hands its cache write
    back in an outcome, so it gets the delegator's full treatment.
    """
    seam = function.__code__.co_varnames[1]
    if seam == "worker":
        return _worker_seam_call(function)
    if seam == "ctx":
        return _cost_profile_delegator(name, function, namespace)
    return _value_seam_call(function)


def _worker_seam_call(function):
    def call(worker, *args, **kwargs):
        return function(None, worker, *args, **kwargs)
    return call


def _value_seam_call(function):
    def call(worker, *args, **kwargs):
        return function(None, *args, **kwargs)
    return call


def _cost_profile_delegator(name, function, namespace):
    """Mirror ``BatchGenWorker._cached_prefix_rank_cost_profile`` exactly.

    The context is resolved per call (``None`` is the pre-move ``hasattr``
    miss) and the outcome is applied in the pre-move order: create the cache
    when it is absent, then add the one entry the lookup resolved. The cache
    the assertions below read is therefore the one the real delegator builds
    on the worker.
    """
    assert name == "_cached_prefix_rank_cost_profile", name
    context = namespace["PrefixRankCostProfileContext"]
    unchanged = namespace["PREFILL_UNCHANGED"]

    def call(worker, *args, **kwargs):
        profile, outcome = function(
            None,
            context(
                rank=worker.rank,
                world_size=worker.world_size,
                model_name=worker.model_name,
                _prefix_rank_cost_profiles=getattr(
                    worker, "_prefix_rank_cost_profiles", None
                ),
            ),
            *args,
            **kwargs,
        )
        if not hasattr(worker, "_prefix_rank_cost_profiles"):
            worker._prefix_rank_cost_profiles = {}
        if outcome.cache_entry is not unchanged:
            key, cached = outcome.cache_entry
            worker._prefix_rank_cost_profiles[key] = cached
        return profile

    return call


class _Props:
    major, minor = SM90
    total_memory = MEM_141G


def _worker(*, model=K3_MODEL, world=WORLD, group=G, cost_profile="on",
            device_properties=_Props):
    worker = SimpleNamespace(rank=0, world_size=world, model_name=model)
    worker._decode_attn_tp_size = lambda: group
    worker._prefill_attn_tp_size = lambda: group
    methods = _methods((
        "_cached_prefix_rank_cost_profile",
        "_prefix_rank_cost_profile_disabled",
    ))
    # The worker's delegator: it hands the engine the prefill group width.
    worker._cached_prefix_rank_cost_profile = lambda disabled=False: methods[
        "_cached_prefix_rank_cost_profile"
    ](worker, disabled, attn_group_size=worker._prefill_attn_tp_size())
    # The worker's delegator: it hands the engine the launch value.
    worker._prefix_rank_cost_profile_disabled = lambda seqs: methods[
        "_prefix_rank_cost_profile_disabled"
    ](worker, seqs, value=cost_profile)
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
        "kimi-linear-48b-a3b-instruct", G, WORLD
    )


@pytest.mark.parametrize("kwargs,why", [
    (dict(group_size=1), "a pure-DP rank is not an 8-rank serve group"),
    (dict(world_size=8), "the fit saw one world only"),
    (dict(compute_capability=(10, 0)), "another compute capability"),
    (dict(total_memory_bytes=int(96 * 1024 ** 3)), "another memory class"),
    (dict(model_name="moonshotai/Kimi-Linear-96B-A6B"), "another checkpoint"),
    (dict(model_name="moonshotai/Kimi-Linear-48B-A3B"),
     "the base name is not the served Instruct checkpoint"),
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


@pytest.mark.parametrize("value,expected", [("on", False), ("off", True)])
def test_the_arm_is_the_launch_value(value, expected):
    assert _worker(cost_profile=value)._prefix_rank_cost_profile_disabled([]) is expected


@pytest.mark.parametrize("value", ["", "0", "OFF", " off ", None, True, 1, "disabled"])
def test_a_value_the_launcher_cannot_produce_is_refused_not_read_as_on(value):
    worker = _worker(cost_profile=value)
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        worker._prefix_rank_cost_profile_disabled([])


# ----------------------------------------------------------- the cache --- #


def test_the_default_arm_resolves_the_profile():
    worker = _worker()
    profile = worker._cached_prefix_rank_cost_profile(False)
    assert profile is not None and profile.model == "kimi-linear-48b-a3b-instruct"


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
def test_both_packer_call_sites_pass_the_server_arm(method):
    body = ast.unparse(_method_ast(method))
    assert ("self._cached_prefix_rank_cost_profile("
            "self._prefix_rank_cost_profile_disabled(seqs))") in body


def test_the_cache_is_keyed_and_never_a_single_attribute():
    body = ast.unparse(_engine_method_ast("_cached_prefix_rank_cost_profile"))
    assert "_prefix_rank_cost_profiles" in body
    assert "key = bool(disabled)" in body
    # The old single-attribute cache is what made the arm un-flippable.
    assert "hasattr(self, '_prefix_rank_cost_profile')" not in body
