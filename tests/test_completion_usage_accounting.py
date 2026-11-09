from queue import SimpleQueue
from types import SimpleNamespace

import torch

import batchgen.batchgen_worker as worker_module
from batchgen.batchgen_worker import BatchGenWorker
from batchgen.sequence import SequenceEntry, SequenceStatus
from tests.worker.worker_fakes import (
    attach_identity,
    attach_prefix_state,
    attach_trajectory_state,
)


class _TrajectoryBook:
    """Completion-path view of the unified trajectory book."""

    def __init__(self, generated=None):
        self._generated = dict(generated or {})
        self._slots = {uuid: slot for slot, uuid in enumerate(self._generated)}
        self.released = []

    def has_sequence(self, uuid):
        return uuid in self._slots

    def slot_for(self, uuid):
        return self._slots[uuid]

    def generated_tokens(self, slot):
        uuid = next(u for u, s in self._slots.items() if s == slot)
        return torch.tensor(self._generated[uuid], dtype=torch.int32)

    def release(self, slot):
        self.released.append(slot)


def test_report_completion_keeps_original_prompt_length_after_reentry():
    seq = SequenceEntry(
        "reentered",
        global_idx=7,
        prompt_length=100,
        max_decode_length=512,
    )
    seq.prompt_length = 228
    seq.decoded_length = 128
    seq.status = SequenceStatus.COMPLETED

    response_queue = SimpleQueue()
    worker = SimpleNamespace(
        global_batch=SimpleNamespace(get_sequence=lambda uuid: seq),
        query_book={},
        _response_queue=response_queue,
        _get_finish_reason=lambda _seq: "stop",
        _publish_trajectory_pool_capacity=lambda: None,
        _maybe_emit_prefix_drain_metric=lambda: None,
    )
    attach_identity(worker, rank=0, world_size=1, local_world_size=1)
    attach_trajectory_state(worker, trajectory_book=_TrajectoryBook())
    attach_prefix_state(worker)

    BatchGenWorker._report_completion(worker, seq.uuid, gathered_text="answer")
    result = response_queue.get()

    assert result["prompt_length"] == 100
    assert result["decoded_length"] == 128


def test_gather_completed_tokens_applies_stop_token_trimming(monkeypatch):
    seq = SequenceEntry(
        "stopped",
        global_idx=9,
        prompt_length=10,
        max_decode_length=32,
    )
    seq.decoded_length = 2

    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [42]
            assert skip_special_tokens is True
            return "answer"

    seq.assigned_rank = 0
    worker = object.__new__(BatchGenWorker)
    # Single rank, G=1: it owns the sequence's host KV and so reports its
    # cached-token count (none here: no prefix state).
    attach_identity(worker, rank=0, world_size=1, local_world_size=1,
                    bound={seq.uuid: 0})
    attach_trajectory_state(worker)
    attach_prefix_state(worker)
    worker.global_batch = SimpleNamespace(get_sequence=lambda uuid: seq)
    worker.query_book = {0: SimpleNamespace()}
    worker._traj.trajectory_book = _TrajectoryBook({seq.uuid: [42, 154827]})
    worker.eos_token_ids = {154820, 154827, 154829}
    worker.pad_token_id = 154820
    worker.detokenization_include_special_tokens = False
    worker.tokenizer = Tokenizer()
    worker._prefix.prefix_sequence_states = {}

    def gather(output, local):
        output[0] = local

    monkeypatch.setattr(worker_module.dist, "all_gather_object", gather)

    assert worker._gather_completed_tokens([seq.uuid]) == {seq.uuid: "answer"}


def test_report_completion_fallback_applies_stop_token_trimming():
    seq = SequenceEntry(
        "local-fallback",
        global_idx=11,
        prompt_length=10,
        max_decode_length=32,
    )
    seq.decoded_length = 2
    seq.status = SequenceStatus.COMPLETED

    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [42]
            assert skip_special_tokens is True
            return "answer"

    worker = object.__new__(BatchGenWorker)
    worker.global_batch = SimpleNamespace(get_sequence=lambda uuid: seq)
    attach_identity(worker, rank=0, world_size=1, local_world_size=1)
    attach_trajectory_state(worker)
    attach_prefix_state(worker)
    worker.query_book = {}
    worker._response_queue = SimpleQueue()
    worker.eos_token_ids = {154820, 154827, 154829}
    worker.pad_token_id = 154820
    worker.detokenization_include_special_tokens = False
    worker.tokenizer = Tokenizer()
    worker._get_finish_reason = lambda _seq: "stop"
    worker._traj.trajectory_book = _TrajectoryBook({seq.uuid: [42, 154827]})
    worker._publish_trajectory_pool_capacity = lambda: None
    worker.enable_prefix_cache = False

    worker._report_completion(seq.uuid)

    assert worker._response_queue.get()["text"] == "answer"
    # Completion is the single release point of the trajectory reservation.
    # migrated: worker._trajectory_book -> worker._traj.trajectory_book, same owner object.
    assert worker._traj.trajectory_book.released == [0]
