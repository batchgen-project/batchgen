"""Self-tests for tests/unit/worker/fakes.py.

Fakes must behave exactly as specified or every downstream test becomes
unreliable. These tests lock down the recording, injection, and accounting
behavior.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit.worker.fakes import (
    FakeClock,
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
    FakeModelExecutor,
    FakeResponseSink,
    FakeTokenizer,
    LifespanEvent,
    RecordingLifespanLogger,
)


class TestFakeCollectiveBackend:
    def test_records_method_name_in_order(self) -> None:
        fake = FakeCollectiveBackend(rank=0, world_size=2)
        fake.barrier()
        fake.all_reduce_sum(torch.zeros(3))
        fake.barrier()
        assert fake.call_names() == ["barrier", "all_reduce_sum", "barrier"]

    def test_all_reduce_max_uses_injected_delta(self) -> None:
        fake = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_reduce_max_deltas=[torch.tensor([5.0, 1.0])],
        )
        t = torch.tensor([2.0, 7.0])
        fake.all_reduce_max(t)
        assert t.tolist() == [5.0, 7.0]

    def test_all_gather_object_default_sets_self_only(self) -> None:
        fake = FakeCollectiveBackend(rank=1, world_size=3)
        obj_list: list[object] = [None, None, None]
        fake.all_gather_object(obj_list, "hello")
        assert obj_list == [None, "hello", None]

    def test_all_gather_object_uses_injected_response(self) -> None:
        fake = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_gather_object_responses=[["self-placeholder", {"u": 42}]],
        )
        obj_list: list[object] = [None, None]
        fake.all_gather_object(obj_list, "local")
        assert obj_list == ["local", {"u": 42}]

    def test_all_gather_object_injected_length_mismatch_raises(self) -> None:
        fake = FakeCollectiveBackend(
            rank=0,
            world_size=2,
            all_gather_object_responses=[[None]],
        )
        with pytest.raises(AssertionError, match="len="):
            fake.all_gather_object([None, None], "x")

    def test_broadcast_object_non_src_requires_injection(self) -> None:
        fake = FakeCollectiveBackend(rank=1, world_size=2)
        with pytest.raises(AssertionError, match="broadcast_object"):
            fake.broadcast_object([None], src=0)

    def test_broadcast_object_src_is_no_op(self) -> None:
        fake = FakeCollectiveBackend(rank=0, world_size=2)
        obj = [{"msg": "x"}]
        fake.broadcast_object(obj, src=0)
        assert obj == [{"msg": "x"}]
        assert fake.call_names() == ["broadcast_object"]

    def test_broadcast_object_non_src_uses_injection(self) -> None:
        fake = FakeCollectiveBackend(
            rank=1,
            world_size=2,
            broadcast_object_responses=[[{"msg": "from-rank0"}]],
        )
        obj: list[object] = [None]
        fake.broadcast_object(obj, src=0)
        assert obj == [{"msg": "from-rank0"}]


class TestFakeGpuKvBackend:
    def test_allocate_decrements_free_and_returns_fresh_ids(self) -> None:
        fake = FakeGpuKvBackend(free_pages=10)
        pages = fake.allocate_pages("u1", 4)
        assert pages == [0, 1, 2, 3]
        assert fake.free_pages() == 6
        assert fake.allocated_pages("u1") == [0, 1, 2, 3]

    def test_allocate_exceeding_free_raises(self) -> None:
        fake = FakeGpuKvBackend(free_pages=2)
        with pytest.raises(RuntimeError, match="exceeds"):
            fake.allocate_pages("u1", 3)

    def test_release_returns_pages_to_free_pool(self) -> None:
        fake = FakeGpuKvBackend(free_pages=10)
        fake.allocate_pages("u1", 4)
        fake.release_pages("u1")
        assert fake.free_pages() == 10
        assert fake.allocated_pages("u1") == []
        assert "u1" not in fake.live_uuids()

    def test_extend_pages_appends_to_existing(self) -> None:
        fake = FakeGpuKvBackend(free_pages=10)
        fake.allocate_pages("u1", 2)
        extra = fake.extend_pages("u1", 3)
        assert extra == [2, 3, 4]
        assert fake.allocated_pages("u1") == [0, 1, 2, 3, 4]
        assert fake.free_pages() == 5

    def test_extend_before_allocate_raises(self) -> None:
        fake = FakeGpuKvBackend(free_pages=10)
        with pytest.raises(RuntimeError, match="before allocate"):
            fake.extend_pages("u1", 2)

    def test_rebuild_page_table_records_uuid_order(self) -> None:
        fake = FakeGpuKvBackend()
        fake.rebuild_page_table(["a", "b", "c"])
        fake.rebuild_page_table(["b", "a"])
        assert fake._rebuilt_page_tables == [["a", "b", "c"], ["b", "a"]]


class TestFakeHostKvBackend:
    def test_allocate_release_roundtrip(self) -> None:
        fake = FakeHostKvBackend(free_pages=50)
        pages = fake.allocate_pages("u1", 7)
        assert len(pages) == 7
        assert fake.free_pages() == 43
        fake.release_pages("u1")
        assert fake.free_pages() == 50

    def test_load_to_gpu_async_returns_monotonic_handle(self) -> None:
        fake = FakeHostKvBackend()
        h1 = fake.load_to_gpu_async("u1", [1, 2])
        h2 = fake.load_to_gpu_async("u2", [3])
        assert h1 < h2
        assert len(fake.recent_handles) == 2
        assert fake.recent_handles[0] == ("u1", [1, 2], h1)


class TestFakeTokenizer:
    def test_encode_returns_ord_codepoints(self) -> None:
        tok = FakeTokenizer()
        assert tok.encode("ab") == [97, 98]

    def test_encode_records_input(self) -> None:
        tok = FakeTokenizer()
        tok.encode("hello")
        tok.encode("world")
        assert tok.encode_calls == ["hello", "world"]

    def test_eos_token_ids_is_plural_set(self) -> None:
        tok = FakeTokenizer(eos_token_ids={1, 2, 3})
        assert tok.eos_token_ids == {1, 2, 3}

    def test_encode_truncates_to_max_len(self) -> None:
        tok = FakeTokenizer(max_len=3)
        assert tok.encode("abcdef") == [97, 98, 99]


class TestFakeModelExecutor:
    def test_prefill_records_batch_and_returns_canned(self) -> None:
        me = FakeModelExecutor(prefill_output="P", decode_output="D")
        out = me.forward_prefill({"uuids": ["u1"]})
        assert out == "P"
        assert me.prefill_batches == [{"uuids": ["u1"]}]

    def test_decode_records_batch_and_returns_canned(self) -> None:
        me = FakeModelExecutor(prefill_output="P", decode_output="D")
        out = me.forward_decode({"uuids": ["u2"]})
        assert out == "D"
        assert me.decode_batches == [{"uuids": ["u2"]}]


class TestRecordingLifespanLogger:
    def test_log_appends_event(self) -> None:
        log = RecordingLifespanLogger()
        log.log("seq1", "EVENT_A", {"k": 1})
        log.log("seq2", "EVENT_B", {"k": 2})
        assert len(log.events) == 2
        assert log.events[0] == LifespanEvent(seq="seq1", event="EVENT_A", detail={"k": 1})

    def test_events_for_filters_by_seq_identity(self) -> None:
        log = RecordingLifespanLogger()
        a = object()
        b = object()
        log.log(a, "E1", {})
        log.log(b, "E2", {})
        log.log(a, "E3", {})
        assert [e.event for e in log.events_for(a)] == ["E1", "E3"]

    def test_detail_is_copied_not_shared(self) -> None:
        log = RecordingLifespanLogger()
        payload = {"x": 1}
        log.log("s", "E", payload)
        payload["x"] = 999
        assert log.events[0].detail == {"x": 1}


class TestFakeClock:
    def test_monotonic_with_step(self) -> None:
        clock = FakeClock(start=10.0, step=0.5)
        assert clock.now() == 10.0
        assert clock.now() == 10.5
        assert clock.now() == 11.0
        assert clock.call_count == 3


class TestFakeResponseSink:
    def test_put_records_uuid_and_payload(self) -> None:
        sink = FakeResponseSink()
        sink.put("u1", {"text": "hi"})
        sink.put("u2", {"text": "bye"})
        assert sink.reported == {
            "u1": {"text": "hi"},
            "u2": {"text": "bye"},
        }
        assert sink.call_order == ["u1", "u2"]

    def test_payload_is_copied_not_shared(self) -> None:
        sink = FakeResponseSink()
        payload = {"text": "hi"}
        sink.put("u1", payload)
        payload["text"] = "MUTATED"
        assert sink.reported["u1"] == {"text": "hi"}
