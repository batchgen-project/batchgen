"""T3: Adapter ABC protocol coverage.

Verifies every model adapter satisfies the `ModelCudaGraphAdapter` contract
type-wise: required methods are present, return types align with the ABC,
and `HasCudaGraphAdapter` accepts the initializer-side hook.

No GPU required.
"""

from __future__ import annotations

import inspect
from typing import Optional

import pytest

from batchgen.cuda_graph.adapter import (
    BatchState,
    GraphDecision,
    GraphMode,
    HasCudaGraphAdapter,
    ModelCudaGraphAdapter,
    SegmentBundle,
)


def test_abc_abstract_method_set():
    abstract = set(ModelCudaGraphAdapter.__abstractmethods__)
    expected = {
        "is_supported",
        "select_buckets",
        "advertised_modes",
        "build_segments",
        "capture_signature",
        "capture_inputs_for",
        "eligibility",
        "prepare_replay_inputs",
        "stage_post_graph_kv",
        "run_eager_reference",
    }
    assert abstract == expected, f"unexpected abstract set: {abstract ^ expected}"


def test_default_methods_concrete():
    # debug_options and release_all have default impls per the contract.
    for name in ("debug_options", "release_all"):
        assert name in ModelCudaGraphAdapter.__dict__
        assert name not in ModelCudaGraphAdapter.__abstractmethods__


def test_graphmode_enum_order():
    # Order is the contract: EAGER < SEGMENTED < LAYER < WHOLE_MODEL.
    modes = list(GraphMode)
    assert modes == [GraphMode.EAGER, GraphMode.SEGMENTED, GraphMode.LAYER, GraphMode.WHOLE_MODEL]


def test_glm5_adapter_satisfies_abc():
    from batchgen.models.glm.glm5.cuda_graph_adapter import Glm5CudaGraphAdapter

    a = Glm5CudaGraphAdapter(model_config=None, engine_config=None, world_size=1, rank=0)
    assert isinstance(a, ModelCudaGraphAdapter)
    assert a.advertised_modes() == [GraphMode.WHOLE_MODEL]


def test_glm5_adapter_select_buckets_returns_list_of_int():
    from batchgen.models.glm.glm5.cuda_graph_adapter import Glm5CudaGraphAdapter

    class FakeEngine:
        max_batch_size = 32
        cuda_graph_num_buckets = 9

    a = Glm5CudaGraphAdapter(model_config=None, engine_config=FakeEngine(), world_size=1, rank=0)
    buckets = a.select_buckets(FakeEngine())
    assert isinstance(buckets, list)
    assert all(isinstance(b, int) for b in buckets)
    assert buckets[0] == 1 and buckets[-1] == 32


def test_initializer_protocol_duck_typed():
    """`HasCudaGraphAdapter` must accept any object with `get_cuda_graph_adapter`
    without requiring inheritance."""

    class FakeInit:
        def get_cuda_graph_adapter(self):
            return None

    class NoAdapter:
        pass

    assert isinstance(FakeInit(), HasCudaGraphAdapter)
    assert not isinstance(NoAdapter(), HasCudaGraphAdapter)


def test_capture_signature_is_hashable():
    from batchgen.models.glm.glm5.cuda_graph_adapter import Glm5CudaGraphAdapter

    a = Glm5CudaGraphAdapter(model_config=None, engine_config=None, world_size=1, rank=0)
    sig = a.capture_signature(bucket=4, gpu_kv_manager=None, max_seqlen=512)
    hash(sig)  # must not raise
