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
