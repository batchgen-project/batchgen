import types

import pytest

from batchgen.cuda_graph.graph_manager import BatchSizeBucketing, CUDAGraphManager


def _make_uninitialized_manager(bucket_sizes, segment_names=("seg",)):
    manager = object.__new__(CUDAGraphManager)
    manager.bucketing = BatchSizeBucketing(list(bucket_sizes))
    manager._graphs = {name: {} for name in segment_names}
    manager._segments = {name: object() for name in segment_names}
    manager._total_capture_time_ms = 0.0
    manager._is_captured = False
    return manager


def test_warmup_and_capture_buckets_captures_only_requested_buckets():
    manager = _make_uninitialized_manager([1, 2, 4])
    captured = []

    def fake_capture_one(self, name, segment, bucket_size):
        captured.append((name, bucket_size))
        self._graphs[name][bucket_size] = object()

    manager._capture_one = types.MethodType(fake_capture_one, manager)
    manager.warmup_and_capture_buckets([1, 4, 4])

    assert captured == [("seg", 1), ("seg", 4)]
    assert manager.has_graph("seg", 1)
    assert not manager.has_graph("seg", 2)
    assert manager.has_graph("seg", 4)
    assert manager.is_captured


def test_power_of_two_bucket_pattern_maps_batches_through_32():
    bucketing = BatchSizeBucketing([1, 2, 4, 8, 16, 32])

    assert [
        bucketing.get_padded_size(batch_size)
        for batch_size in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32]
    ] == [1, 2, 4, 4, 8, 8, 16, 16, 32, 32]
    with pytest.raises(ValueError, match="exceeds max bucket 32"):
        bucketing.get_padded_size(33)


def test_warmup_and_capture_buckets_rejects_unknown_bucket():
    manager = _make_uninitialized_manager([1, 2])
    manager._capture_one = types.MethodType(
        lambda self, name, segment, bucket_size: None,
        manager,
    )

    with pytest.raises(ValueError, match="unknown CUDA graph buckets"):
        manager.warmup_and_capture_buckets([4])


def test_has_bucket_for_all_segments_and_drop_bucket_release_buffers():
    manager = _make_uninitialized_manager([1, 2], segment_names=("a", "b"))

    class Segment:
        def __init__(self):
            self.released = []

        def release_static_buffers(self, bucket_size):
            self.released.append(bucket_size)

    seg_a = Segment()
    seg_b = Segment()
    manager._segments = {"a": seg_a, "b": seg_b}
    manager._graphs = {
        "a": {1: object(), 2: object()},
        "b": {1: object()},
    }
    manager._is_captured = True

    assert manager.has_bucket_for_all_segments(1)
    assert not manager.has_bucket_for_all_segments(2)

    manager.drop_bucket(1)
    assert 1 not in manager._graphs["a"]
    assert 1 not in manager._graphs["b"]
    assert seg_a.released == [1]
    assert seg_b.released == [1]
    assert manager.is_captured

    manager.drop_bucket(2)
    assert not manager.is_captured


def test_capture_one_initializes_static_inputs_before_warmup(monkeypatch):
    import torch

    manager = _make_uninitialized_manager([2])
    manager.device = torch.device("cpu")
    manager._pool = object()
    calls = []

    class Segment:
        def get_static_input_specs(self, bucket_size):
            from batchgen.cuda_graph.graph_manager import TensorSpec

            return {"x": TensorSpec(("batch_size",), torch.int64, fill_value=0)}

        def initialize_static_inputs(self, static_inputs, bucket_size):
            static_inputs["x"].fill_(7)
            calls.append(("init", static_inputs["x"].clone()))

        def forward(self, **inputs):
            calls.append(("forward", inputs["x"].clone()))
            return {"y": inputs["x"]}

    manager._segments = {"seg": Segment()}
    manager._graphs = {"seg": {}}

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: FakeGraph())

    manager._capture_one("seg", manager._segments["seg"], 2)

    assert calls[0][0] == "init"
    assert all(torch.equal(call[1], torch.tensor([7, 7])) for call in calls)
