"""The Kimi decode graph mode in force is one answer.

The server debug state's ``kimi_decode_graph_mode`` overrides the planner's
mode for the capture and for every replay. Whether the manager's kept decode
instance owns graphs that persist across a SUSPEND must follow the same
answer: a pool started under ``eager`` captures no ladder, and the startup
capture must not demand one.
"""
import types

import pytest

from batchgen.models.moonshotai.kimi_linear import cuda_graph_segments as cgs
from batchgen.models.moonshotai.kimi_linear.cuda_graph_segments import (
    KimiLinearDecodeGraph,
)
from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
    KimiLinearParallelStrategyManager as PSM,
)


def _graph(planner_mode):
    g = object.__new__(KimiLinearDecodeGraph)
    g.mode = planner_mode
    g._ladder_captured = False
    return g


def _persist(planner_mode, graph, reused=True):
    stub = types.SimpleNamespace(
        _instances_are_reused=reused,
        _decode_graph=graph,
        _decode_graph_mode=lambda: planner_mode,
    )
    return PSM.decode_graphs_persist_across_suspend.fget(stub)


@pytest.mark.parametrize("state_mode, planner_mode, in_force", [
    (None, "graph", "graph"),
    (None, "eager", "eager"),
    ("eager", "graph", "eager"),
    ("compare", "graph", "compare"),
    ("graph", "eager", "graph"),
])
def test_mode_in_force_is_the_state_key_else_the_planner_mode(
    monkeypatch, state_mode, planner_mode, in_force
):
    monkeypatch.setattr(cgs, "_debug_mode", lambda: state_mode)
    g = _graph(planner_mode)
    assert g.capture_mode() == in_force
    assert _persist(planner_mode, g) is (in_force != "eager")


def test_pool_started_under_eager_state_expects_no_ladder(monkeypatch):
    monkeypatch.setattr(cgs, "_debug_mode", lambda: "eager")
    g = _graph("graph")
    assert g.prewarm_decode_graphs() is False
    assert not g.ladder_captured
    assert _persist("graph", g) is False


def test_state_key_is_read_from_the_adopted_debug_values(monkeypatch):
    monkeypatch.setattr(
        cgs.AttnWrapperBase, "batchgen_debug",
        {"kimi_decode_graph_mode": "eager"}, raising=False,
    )
    assert _graph("graph").capture_mode() == "eager"
    monkeypatch.setattr(cgs.AttnWrapperBase, "batchgen_debug", {}, raising=False)
    assert _graph("graph").capture_mode() == "graph"


def test_persist_follows_the_planner_before_the_graph_exists():
    assert _persist("graph", None) is True
    assert _persist("eager", None) is False
    assert _persist("graph", None, reused=False) is False
