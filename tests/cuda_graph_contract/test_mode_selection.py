"""Tests for the mode-selection state machine (T4).

Pure-Python verification of the eligibility / fallback walk described in
`batchgen_design/cuda_graph/cuda_graph_contract.md` §B. No CUDA required.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import pytest
import torch

from batchgen.cuda_graph.adapter import (
    BatchState,
    GraphDecision,
    GraphMode,
    ModelCudaGraphAdapter,
    SegmentBundle,
)


def _make_batch_state(max_rank_bsz: int) -> BatchState:
    return BatchState(
        local_bsz=max_rank_bsz, max_rank_bsz=max_rank_bsz,
        rank_token_counts=None, cache_seqlens=None, position_ids=None,
        max_seqlen=64, cur_batch_sequence_ids=tuple(range(max_rank_bsz)),
        gpu_kv_manager=None, decode_iter=0,
    )


class _FallbackAdapter(ModelCudaGraphAdapter):
    """Adapter that advertises WHOLE_MODEL preferred, LAYER as fallback,
    where WHOLE_MODEL preconditions never hold — verifies the fallback walk.
    """

    def __init__(self):
        self.warnings_emitted: List[Tuple[GraphMode, GraphMode, str]] = []

    def is_supported(self, ec): return True
    def select_buckets(self, ec): return [1, 2, 4]
    def advertised_modes(self): return [GraphMode.WHOLE_MODEL, GraphMode.LAYER]

    def build_segments(self, **kwargs): return SegmentBundle()

    def capture_signature(self, *, bucket, gpu_kv_manager, max_seqlen): return (bucket,)
    def capture_inputs_for(self, *, bucket, segment_name, batch_state): return {}
    def prepare_replay_inputs(self, **kwargs): return {}
    def stage_post_graph_kv(self, **kwargs): pass
    def run_eager_reference(self, **kwargs): return {}

    def eligibility(self, batch_state: BatchState) -> GraphDecision:
        # Walk advertised modes in order; WHOLE_MODEL precondition fails
        # ("not_captured"), LAYER precondition succeeds.
        for mode in self.advertised_modes():
            if mode is GraphMode.WHOLE_MODEL:
                continue  # simulate "not captured for this bucket"
            return GraphDecision(mode=mode, bucket=batch_state.max_rank_bsz, reason="ok")
        return GraphDecision(mode=GraphMode.EAGER, bucket=None, reason="no_mode")


def test_eligibility_returns_advertised_mode():
    adapter = _FallbackAdapter()
    bs = _make_batch_state(max_rank_bsz=2)
    d = adapter.eligibility(bs)
    assert d.mode is GraphMode.LAYER
    assert d.bucket == 2


def test_eager_fallback_when_no_mode_advertised():
    class NoModes(_FallbackAdapter):
        def advertised_modes(self): return []
    d = NoModes().eligibility(_make_batch_state(max_rank_bsz=2))
    assert d.mode is GraphMode.EAGER


def test_glm5_adapter_eligibility_without_context_returns_eager():
    from batchgen.models.glm.glm5.cuda_graph_adapter import Glm5CudaGraphAdapter

    a = Glm5CudaGraphAdapter(model_config=None, engine_config=None, world_size=1, rank=0)
    d = a.eligibility(_make_batch_state(max_rank_bsz=2))
    assert d.mode is GraphMode.EAGER
    assert "adapter_not_built" in d.reason
