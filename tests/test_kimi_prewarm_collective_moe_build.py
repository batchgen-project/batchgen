# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear                                                       #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""K3 decode prewarm issues the same EP-world collectives as the first step (CPU).

``prewarm_decode_graphs`` runs on every rank at decode configure. The K3 MoE
build (exchange pool setup + DeepEP self-check), a whole-model graph capture
and a legacy per-layer MoE graph capture all contain EP-world collectives, so
every rank must issue them in the same sequence. Steady state already
guarantees this: ``_begin_step`` on a rowed rank and
``_begin_empty_whole_step`` on an empty rank capture/replay the same
synchronized whole-model bucket. Prewarm used to diverge from that: an empty
rank returned at ``bsz == 0`` while the rowed ranks entered the collective
build and then a legacy MoE capture that steady state would not take, which
deadlocked the cohort whenever the first decode batch left a group empty.

These tests drive the real ``KimiLinearDecodeGraph`` methods with the graph
builders and captures replaced by recorders, and assert per scenario that:

  * each rank's prewarm records exactly the EP-world calls its first step
    would record from a fresh state (prewarm is a rehearsal of that step);
  * a rowed rank and an empty rank record the same EP-world sequence.

Run: python -m pytest tests/test_kimi_prewarm_collective_moe_build.py -x -q
"""

from __future__ import annotations

import types
from typing import List, Optional

import pytest

import batchgen.models.moonshotai.kimi_linear.cuda_graph_segments as cgs
from batchgen.models.moonshotai.kimi_linear.cuda_graph_segments import (
    KimiLinearDecodeGraph,
)

WHOLE_BUCKET = 8


class _Bucketing:
    bucket_sizes = [1, 2, 4, 8, 16]

    def get_padded_size(self, n: int) -> int:
        for b in self.bucket_sizes:
            if b >= n:
                return b
        raise ValueError(n)


class _KV:
    def ensure_cuda_graph_page_table(self, seq_ids):
        return None


class _Statics:
    def arm_for_capture(self):
        pass


def _make_graph(*, whole_available: bool, all_nonempty: bool):
    """A KimiLinearDecodeGraph whose collective-issuing steps are recorders."""
    g = object.__new__(KimiLinearDecodeGraph)
    ep_calls: List[tuple] = []
    g.ep_calls = ep_calls
    g.rank = 0
    g.mode = "graph"
    g.device = "cpu"
    g.step = 0
    g.compare_every = 1
    g.bucketing = _Bucketing()
    g._kv_signature = "sig"
    g._captured = set()
    g._whole_captured = set()
    g._moe_captured = set()
    g._statics = {}
    g._moe_built_flag = False
    g._compare_step = False

    def ensure_moe_built():
        if not g._moe_built_flag:
            g._moe_built_flag = True
            ep_calls.append(("moe_build",))
        return True

    def bucket_statics(bucket):
        return g._statics.setdefault(bucket, _Statics())

    def capture_whole(bucket):
        ep_calls.append(("capture_whole", bucket))
        g._whole_captured.add(bucket)

    def capture_moe(bucket):
        ep_calls.append(("capture_moe", bucket))
        g._moe_captured.add(bucket)

    g._ensure_built = lambda kv: True
    g._signature = lambda kv: "sig"
    g._drop_graphs = lambda: None
    g._aux_graph_page_table = lambda seq_ids: None
    g._ensure_moe_built = ensure_moe_built
    g._moe_all_ranks_nonempty = lambda: all_nonempty
    g._moe_group_bucket_for_step = lambda: WHOLE_BUCKET
    g._whole_model_bucket_for_step = (
        (lambda: WHOLE_BUCKET) if whole_available else (lambda: None)
    )
    g._ensure_whole_built = lambda: True
    g._bucket_statics = bucket_statics
    g._capture_whole_bucket = capture_whole
    g._capture_moe_bucket = capture_moe
    g._capture_bucket = lambda bucket: g._captured.add(bucket)
    g._refresh_statics = lambda *a, **k: None
    g._fallback = lambda reason: None
    g.moe_segments = {0: types.SimpleNamespace(tp_size=8)}
    return g


@pytest.fixture(autouse=True)
def _decode_phase(monkeypatch):
    monkeypatch.setattr(cgs, "_debug_mode", lambda: None)
    monkeypatch.setattr(cgs.AttnWrapperBase, "phase", "decode", raising=False)
    monkeypatch.setattr(cgs.AttnWrapperBase, "gpu_paged_kv_manager", _KV(), raising=False)
    monkeypatch.setattr(
        cgs.KimiLinearKDAWrapper,
        "state_manager",
        types.SimpleNamespace(prepare_decode_step=lambda ids: None),
        raising=False,
    )


def _calls(run, rows: List[int], **scenario) -> List[tuple]:
    cgs.AttnWrapperBase.cur_batch = rows
    g = _make_graph(**scenario)
    if run == "prewarm":
        g.prewarm_decode_graphs()
    else:
        g._begin_step()
    return list(g.ep_calls)


SCENARIOS = [
    pytest.param(dict(whole_available=True, all_nonempty=False), id="whole-model, a group empty"),
    pytest.param(dict(whole_available=False, all_nonempty=False), id="legacy, a group empty"),
]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_prewarm_rehearses_first_step_on_every_rank(scenario):
    rowed, empty = [101], []
    for rows in (rowed, empty):
        assert _calls("prewarm", rows, **scenario) == _calls("step", rows, **scenario), (
            f"prewarm must issue the same EP-world collectives as the first "
            f"step on a {'rowed' if rows else 'empty'} rank"
        )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_prewarm_rowed_and_empty_ranks_issue_matching_collectives(scenario):
    assert _calls("prewarm", [101], **scenario) == _calls("prewarm", [], **scenario), (
        "a rowed rank and an empty rank must enter the same EP-world "
        "collectives at prewarm, or the cohort deadlocks"
    )


def test_whole_model_prewarm_captures_on_empty_rank():
    # Non-vacuity: the scenario that deadlocked actually exercises a capture.
    calls = _calls("prewarm", [], whole_available=True, all_nonempty=False)
    assert ("capture_whole", WHOLE_BUCKET) in calls


def _graph_attention_functions():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(cgs.__file__).read_text())
    wanted = {"_mla_decode_graph_safe", "_graph_attention"}
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]


def test_graph_attention_reads_cache_without_an_active_page_table():
    # An empty rank captures the whole-model graph before it has ever built
    # an active GPU page table; the graph brings its own static table, so the
    # graph attention may read only the cache tensors.
    import ast

    functions = _graph_attention_functions()
    assert {f.name for f in functions} == {"_mla_decode_graph_safe", "_graph_attention"}
    for function in functions:
        called = {
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "get_layer_kv_with_page_table" not in called, function.name
        assert "get_layer_kv_cache" in called, function.name


def test_legacy_moe_capture_only_when_every_rank_has_rows():
    calls = _calls("prewarm", [101], whole_available=False, all_nonempty=True)
    assert ("capture_moe", WHOLE_BUCKET) in calls
    assert _calls("step", [101], whole_available=False, all_nonempty=True) == calls
