"""The engine-bound seam issues each frame's cross-node collective at the
layer's deterministic gate, never ahead of the model thread's TP8 order."""

import pytest

torch = pytest.importorskip("torch")

from batchgen.moe.streamed_sp8_mxfp4 import StreamedSP8LayerBuffer


class _Broadcaster:
    def __init__(self, log):
        self.log = log

    def enqueue(self, layer_idx):
        self.log.append(("enqueue", layer_idx))


def _seam_buffer(log):
    buf = StreamedSP8LayerBuffer.__new__(StreamedSP8LayerBuffer)
    buf.cross_group = None
    buf._acquires_from_host = False
    buf._status_stream = None
    buf._allocate = lambda: None
    return buf


@pytest.fixture
def seam(monkeypatch):
    log = []
    monkeypatch.setattr(
        StreamedSP8LayerBuffer, "seam_broadcaster", _Broadcaster(log)
    )
    return log


def test_frame_launch_waits_for_the_gate_then_hands_order_back(seam):
    buf = _seam_buffer(seam)

    def gate():
        seam.append(("gate",))
        return True

    buf._acquire_local_shard(
        7,
        stream=object(),
        cross_launch_gate=gate,
        cross_launch_callback=lambda: seam.append(("enqueued",)),
    )
    assert seam == [("gate",), ("enqueue", 7), ("enqueued",)]


def test_cancelled_gate_launches_nothing(seam):
    buf = _seam_buffer(seam)
    buf._acquire_local_shard(
        7,
        stream=object(),
        cross_launch_gate=lambda: False,
        cross_launch_callback=lambda: seam.append(("enqueued",)),
    )
    assert seam == []


def test_first_layer_launches_synchronously_on_the_model_thread(seam):
    buf = _seam_buffer(seam)
    buf._acquire_local_shard(3, stream=object())
    assert seam == [("enqueue", 3)]


def test_model_layer_broadcast_surface_is_gone(seam):
    # S4: the replica seam is the ONLY cross-node payload transport. The
    # model-layer broadcast surface must not exist, so no caller can route
    # around the engine-owned per-record release semantics.
    buf = _seam_buffer(seam)
    assert not hasattr(buf, "_broadcast_local_shard")
    assert not hasattr(buf, "_broadcast_source_status")
    assert not hasattr(type(buf), "cross_broadcast_transport")
    buf._acquire_local_shard(5, stream=object())
    assert seam == [("enqueue", 5)]
