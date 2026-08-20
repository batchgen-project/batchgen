import types

import pytest
import torch

from batchgen.cuda_graph.graph_manager import BatchSizeBucketing, CapturedGraph, CUDAGraphManager


def _make_uninitialized_manager(bucket_sizes, segment_names=("seg",)):
    manager = object.__new__(CUDAGraphManager)
    manager.bucketing = BatchSizeBucketing(list(bucket_sizes))
    manager._graphs = {name: {} for name in segment_names}
    manager._segments = {name: object() for name in segment_names}
    manager._total_capture_time_ms = 0.0
    manager._is_captured = False
    manager._memory_diag_enabled = False
    manager._capture_memory_stats = []
    return manager


def test_warmup_and_capture_buckets_captures_only_requested_buckets():
    manager = _make_uninitialized_manager([1, 2, 4])
    captured = []

    def fake_capture_one(self, name, segment, bucket_size, warmup_iters=None):
        del segment, warmup_iters
        captured.append((name, bucket_size))
        self._graphs[name][bucket_size] = object()

    manager._capture_one = types.MethodType(fake_capture_one, manager)
    manager.warmup_and_capture_buckets([1, 4, 4])

    assert captured == [("seg", 1), ("seg", 4)]
    assert manager.has_graph("seg", 1)
    assert not manager.has_graph("seg", 2)
    assert manager.has_graph("seg", 4)
    assert manager.is_captured


def test_warmup_and_capture_all_captures_every_configured_bucket_once():
    manager = _make_uninitialized_manager([1, 2, 4, 8])
    captured = []

    def fake_capture_one(self, name, segment, bucket_size, warmup_iters=None):
        del segment, warmup_iters
        captured.append((name, bucket_size))
        self._graphs[name][bucket_size] = object()

    manager._capture_one = types.MethodType(fake_capture_one, manager)
    manager.warmup_and_capture_all()

    assert captured == [("seg", 1), ("seg", 2), ("seg", 4), ("seg", 8)]
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_uses_manager_owned_explicit_stream(monkeypatch):
    device = torch.device("cuda")
    manager = CUDAGraphManager(BatchSizeBucketing([1]), device=device)
    manager.WARMUP_ITERATIONS = 1

    class Segment:
        def __init__(self):
            self.output = torch.empty(1, device=device)

        def get_static_input_specs(self, _bucket_size):
            from batchgen.cuda_graph.graph_manager import TensorSpec

            return {"x": TensorSpec(("batch_size",), torch.float32)}

        def get_static_output_specs(self, _bucket_size):
            from batchgen.cuda_graph.graph_manager import TensorSpec

            return {"output": TensorSpec(("batch_size",), torch.float32)}

        def forward(self, x):
            torch.add(x, 1, out=self.output)
            return {"output": self.output}

    observed_streams = []
    real_graph = torch.cuda.graph

    def recording_graph(graph, *args, **kwargs):
        observed_streams.append(kwargs.get("stream"))
        return real_graph(graph, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "graph", recording_graph)
    manager.register_segment("seg", Segment())
    manager.warmup_and_capture_all()

    assert observed_streams == [manager._capture_stream]
    assert manager._capture_stream != torch.cuda.current_stream(device)


def test_segment_capture_streams_are_stable_and_distinct(monkeypatch):
    manager = _make_uninitialized_manager([1], segment_names=("a", "b"))
    manager.device = torch.device("cpu")
    manager._capture_stream = object()
    manager._separate_capture_streams = False
    manager._segment_capture_streams = {}

    streams = []

    def fake_stream(*, device):
        stream = types.SimpleNamespace(device=device, index=len(streams))
        streams.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", fake_stream)

    assert manager._capture_stream_for("a") is manager._capture_stream
    manager.enable_segment_capture_streams()
    stream_a = manager._capture_stream_for("a")
    stream_b = manager._capture_stream_for("b")

    assert stream_a is manager._capture_stream_for("a")
    assert stream_b is manager._capture_stream_for("b")
    assert stream_a is not stream_b


def test_segment_capture_streams_are_precreated_before_capture(monkeypatch):
    manager = _make_uninitialized_manager([1], segment_names=())
    manager.device = torch.device("cpu")
    manager._capture_stream = object()
    manager._separate_capture_streams = False
    manager._segment_capture_streams = {}
    streams = []

    def fake_stream(*, device):
        stream = types.SimpleNamespace(device=device, index=len(streams))
        streams.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", fake_stream)
    manager.enable_segment_capture_streams()
    manager.register_segment("a", object())
    manager.register_segment("b", object())

    assert tuple(manager._segment_capture_streams) == ("a", "b")
    assert manager._segment_capture_streams["a"] is streams[0]
    assert manager._segment_capture_streams["b"] is streams[1]


def test_segment_capture_barrier_runs_between_registered_segments(monkeypatch):
    manager = _make_uninitialized_manager([1], segment_names=("a", "b"))
    manager.device = torch.device("cpu")
    manager._capture_stream = object()
    manager._separate_capture_streams = False
    manager._segment_capture_streams = {}
    captures = []
    barriers = []

    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda *, device: types.SimpleNamespace(device=device),
    )

    def fake_capture_one(self, name, segment, bucket_size, warmup_iters=None):
        del segment, warmup_iters
        captures.append((name, bucket_size))
        self._graphs[name][bucket_size] = object()

    manager._capture_one = types.MethodType(fake_capture_one, manager)
    manager._is_captured = False
    manager.enable_segment_capture_streams(
        barrier=lambda: barriers.append(tuple(captures))
    )

    manager.warmup_and_capture_buckets([1])

    assert captures == [("a", 1), ("b", 1)]
    assert barriers == [(("a", 1),), (("a", 1), ("b", 1))]


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


def test_replay_auto_populates_num_valid_tokens_static_input():
    import torch

    manager = _make_uninitialized_manager([4])
    static_inputs = {
        "x": torch.full((4,), -9, dtype=torch.int32),
        "num_valid_tokens": torch.full((1,), 4, dtype=torch.int32),
    }
    replay_observations = []

    class FakeGraph:
        def replay(self):
            replay_observations.append(
                (
                    static_inputs["x"].clone(),
                    static_inputs["num_valid_tokens"].clone(),
                )
            )

    manager._graphs["seg"][4] = CapturedGraph(
        bucket_size=4,
        graph=FakeGraph(),
        static_inputs=static_inputs,
        static_outputs={"y": torch.arange(4, dtype=torch.int32)},
        input_fill_values={"x": 0.0, "num_valid_tokens": 4.0},
    )

    result = manager.replay("seg", 2, x=torch.tensor([10, 11], dtype=torch.int32))

    assert torch.equal(static_inputs["x"], torch.tensor([10, 11, 0, 0], dtype=torch.int32))
    assert torch.equal(static_inputs["num_valid_tokens"], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(replay_observations[0][1], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(result["y"], torch.tensor([0, 1], dtype=torch.int32))


def test_capture_one_initializes_static_inputs_before_warmup(monkeypatch):
    import torch

    manager = _make_uninitialized_manager([2])
    manager.device = torch.device("cpu")
    manager._pool = object()
    manager._capture_stream = types.SimpleNamespace(wait_stream=lambda _stream: None)
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
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: manager._capture_stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: None)

    manager._capture_one("seg", manager._segments["seg"], 2)

    assert calls[0][0] == "init"
    assert all(torch.equal(call[1], torch.tensor([7, 7])) for call in calls)


def test_capture_one_records_phase_memory_stats(monkeypatch):
    import torch

    manager = _make_uninitialized_manager([2])
    manager.device = torch.device("cpu")
    manager._pool = object()
    manager._capture_stream = types.SimpleNamespace(wait_stream=lambda _stream: None)
    manager._memory_diag_enabled = True

    snapshots = [
        {
            "free_bytes": 1000,
            "total_bytes": 2000,
            "used_bytes": 1000,
            "allocated_bytes": 100,
            "reserved_bytes": 200,
        },
        {
            "free_bytes": 900,
            "total_bytes": 2000,
            "used_bytes": 1100,
            "allocated_bytes": 140,
            "reserved_bytes": 260,
        },
        {
            "free_bytes": 860,
            "total_bytes": 2000,
            "used_bytes": 1140,
            "allocated_bytes": 150,
            "reserved_bytes": 280,
        },
        {
            "free_bytes": 840,
            "total_bytes": 2000,
            "used_bytes": 1160,
            "allocated_bytes": 155,
            "reserved_bytes": 280,
        },
        {
            "free_bytes": 700,
            "total_bytes": 2000,
            "used_bytes": 1300,
            "allocated_bytes": 180,
            "reserved_bytes": 400,
        },
    ]

    monkeypatch.setattr(manager, "_capture_memory_snapshot", lambda: snapshots.pop(0))

    class Segment:
        def get_static_input_specs(self, bucket_size):
            from batchgen.cuda_graph.graph_manager import TensorSpec

            return {"x": TensorSpec(("batch_size",), torch.float32)}

        def setup_static_buffers(self, bucket_size):
            pass

        def initialize_static_inputs(self, static_inputs, bucket_size):
            static_inputs["x"].fill_(1.0)

        def forward(self, **inputs):
            return {"y": inputs["x"] + 1.0}

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: FakeGraph())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: manager._capture_stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: None)

    manager._segments = {"seg": Segment()}
    manager._graphs = {"seg": {}}
    manager._capture_one("seg", manager._segments["seg"], 2)

    stats = manager.get_capture_stats()["capture_memory_stats"]
    assert len(stats) == 1
    assert stats[0]["segment"] == "seg"
    assert stats[0]["bucket_size"] == 2
    assert [phase["phase"] for phase in stats[0]["phases"]] == [
        "static_input_allocation",
        "segment_setup",
        "warmup",
        "graph_capture",
    ]
    assert stats[0]["phases"][0]["delta"]["used_delta_bytes"] == 100
    assert stats[0]["total_delta"]["used_delta_bytes"] == 300
