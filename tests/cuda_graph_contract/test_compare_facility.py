"""Tests for `batchgen.cuda_graph.compare.compare_decode_outputs`."""

from __future__ import annotations

from typing import Dict, Iterable

import pytest
import torch

from batchgen.cuda_graph.adapter import (
    BatchState,
    GraphDecision,
    GraphMode,
    ModelCudaGraphAdapter,
    SegmentBundle,
)
from batchgen.cuda_graph.compare import (
    CompareFailure,
    CompareReport,
    compare_decode_outputs,
)


class _StubAdapter(ModelCudaGraphAdapter):
    """Minimal real-protocol adapter used by the compare tests.

    NOT a mock — implements every ABC method with a real-ish eager path.
    The compare facility only calls `run_eager_reference`, but we satisfy
    the rest so the adapter actually instantiates.
    """

    def __init__(self, drift: float = 0.0):
        self.drift = drift

    def is_supported(self, engine_config): return True
    def select_buckets(self, engine_config): return [1, 2, 4]
    def advertised_modes(self): return [GraphMode.WHOLE_MODEL]

    def build_segments(self, **kwargs): return SegmentBundle()
    def capture_signature(self, *, bucket, gpu_kv_manager, max_seqlen):
        return (bucket, max_seqlen)
    def capture_inputs_for(self, *, bucket, segment_name, batch_state):
        return {"x": torch.zeros(bucket, 4)}
    def eligibility(self, batch_state):
        return GraphDecision(mode=GraphMode.WHOLE_MODEL, bucket=batch_state.max_rank_bsz)
    def prepare_replay_inputs(self, *, decision, batch_state, segment_name):
        return {"x": torch.zeros(decision.bucket or 0, 4)}
    def stage_post_graph_kv(self, **kwargs): pass

    def run_eager_reference(
        self, *, segment_name, batch_state, captured_inputs, probe_layers=(),
    ) -> Dict[str, torch.Tensor]:
        out = {"logits": captured_inputs["x"].clone() + self.drift}
        for i in probe_layers:
            out[f"hidden_states_layer_{i}"] = captured_inputs["x"].clone() + self.drift * 0.5
        return out


@pytest.fixture
def batch_state() -> BatchState:
    return BatchState(
        local_bsz=2, max_rank_bsz=2, rank_token_counts=None,
        cache_seqlens=None, position_ids=None, max_seqlen=64,
        cur_batch_sequence_ids=(0, 1), gpu_kv_manager=None, decode_iter=0,
    )


@pytest.fixture
def decision() -> GraphDecision:
    return GraphDecision(mode=GraphMode.WHOLE_MODEL, bucket=2)


def test_match_passes(batch_state, decision):
    adapter = _StubAdapter(drift=0.0)
    captured = {"x": torch.zeros(2, 4)}
    graph_out = {"logits": torch.zeros(2, 4)}
    rpt = compare_decode_outputs(
        adapter=adapter, decision=decision, batch_state=batch_state,
        segment_name="ws", captured_inputs=captured, graph_outputs=graph_out,
    )
    assert rpt.passed
    assert rpt.max_abs == 0.0
    assert rpt.mismatched_keys == []


def test_drift_marks_failed_without_raising(batch_state, decision):
    adapter = _StubAdapter(drift=1.0)
    captured = {"x": torch.zeros(2, 4)}
    graph_out = {"logits": torch.zeros(2, 4)}
    rpt = compare_decode_outputs(
        adapter=adapter, decision=decision, batch_state=batch_state,
        segment_name="ws", captured_inputs=captured, graph_outputs=graph_out,
    )
    assert not rpt.passed
    assert "logits" in rpt.mismatched_keys
    assert rpt.max_abs == pytest.approx(1.0)


def test_fail_on_mismatch_raises(batch_state, decision):
    adapter = _StubAdapter(drift=1.0)
    captured = {"x": torch.zeros(2, 4)}
    graph_out = {"logits": torch.zeros(2, 4)}
    with pytest.raises(CompareFailure):
        compare_decode_outputs(
            adapter=adapter, decision=decision, batch_state=batch_state,
            segment_name="ws", captured_inputs=captured, graph_outputs=graph_out,
            fail_on_mismatch=True,
        )


def test_missing_graph_key_reported(batch_state, decision):
    adapter = _StubAdapter(drift=0.0)
    captured = {"x": torch.zeros(2, 4)}
    graph_out = {}  # adapter produces 'logits', graph doesn't
    rpt = compare_decode_outputs(
        adapter=adapter, decision=decision, batch_state=batch_state,
        segment_name="ws", captured_inputs=captured, graph_outputs=graph_out,
    )
    assert not rpt.passed
    assert rpt.missing_graph_keys == ["logits"]


def test_probe_layers_recorded(batch_state, decision):
    adapter = _StubAdapter(drift=0.0)
    captured = {"x": torch.zeros(2, 4)}
    graph_out = {
        "logits": torch.zeros(2, 4),
        "hidden_states_layer_0": torch.zeros(2, 4),
        "hidden_states_layer_12": torch.zeros(2, 4),
    }
    rpt = compare_decode_outputs(
        adapter=adapter, decision=decision, batch_state=batch_state,
        segment_name="ws", captured_inputs=captured, graph_outputs=graph_out,
        probe_layers=[0, 12],
    )
    assert rpt.passed
    assert set(rpt.probe_results.keys()) == {0, 12}
