from queue import SimpleQueue
from types import SimpleNamespace

import torch

import batchgen.batchgen_worker as worker_module
from batchgen.batchgen_worker import BatchGenWorker
from batchgen.sequence import SequenceEntry, SequenceStatus


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
        _uuid_to_local_map={},
        _local_to_uuid_map={},
        query_book={},
        _free_local_indices=set(),
        rank=0,
        _response_queue=response_queue,
        _get_finish_reason=lambda _seq: "stop",
    )

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

    worker = object.__new__(BatchGenWorker)
    worker._uuid_to_local_map = {seq.uuid: 0}
    worker.global_batch = SimpleNamespace(get_sequence=lambda uuid: seq)
    worker.query_book = {
        0: SimpleNamespace(decoded_tokens=torch.tensor([[42, 154827]]))
    }
    worker.world_size = 1
    worker.eos_token_ids = {154820, 154827, 154829}
    worker.pad_token_id = 154820
    worker.detokenization_include_special_tokens = False
    worker.tokenizer = Tokenizer()

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
    seq.decoded_tokens = torch.tensor([[42, 154827]])
    seq.decoded_length = 2
    seq.status = SequenceStatus.COMPLETED

    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids == [42]
            assert skip_special_tokens is True
            return "answer"

    worker = object.__new__(BatchGenWorker)
    worker.global_batch = SimpleNamespace(get_sequence=lambda uuid: seq)
    worker._uuid_to_local_map = {}
    worker._local_to_uuid_map = {}
    worker.query_book = {}
    worker._free_local_indices = set()
    worker.rank = 0
    worker._response_queue = SimpleQueue()
    worker.eos_token_ids = {154820, 154827, 154829}
    worker.pad_token_id = 154820
    worker.detokenization_include_special_tokens = False
    worker.tokenizer = Tokenizer()
    worker._get_finish_reason = lambda _seq: "stop"

    worker._report_completion(seq.uuid)

    assert worker._response_queue.get()["text"] == "answer"
