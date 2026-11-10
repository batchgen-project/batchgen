"""Pool admission binds the prompt token ids the admission message carries.

The server builds every prompt's ids; each rank receives the same int32 array
in the admission message and writes it into the trajectory book unchanged. No
rank tokenizes and no collective runs at admission. Over-context prompts still
fail alone with ``context_length_exceeded``.

batchgen_worker imports the whole engine, so the methods are compiled from the
real source against a fake worker (tests/worker_method_source.py).
"""

import ast
import logging
import pickle
import types
from typing import List

import numpy as np
import pytest
import torch

from batchgen.query_book import QueryBook
from tests.worker_method_source import (
    WORKER_SOURCE,
    load_worker_functions,
    load_worker_methods,
)

_ADMITTED_PROMPT_IDS = load_worker_functions(
    ["_admitted_prompt_ids"], {"np": np, "torch": torch},
)["_admitted_prompt_ids"]
_BIND = load_worker_methods(
    ["_bind_admitted_prompts"],
    {"List": List, "logging": logging, "torch": torch},
)["_bind_admitted_prompts"]


class _Sequence:
    def __init__(self, uuid, max_decode_length):
        self.uuid = uuid
        self.batch_id = None
        self.prompt_length = 0
        self.max_decode_length = max_decode_length
        self.original_max_decode_length = max_decode_length
        self.kv_token_budget = max_decode_length
        self._buffer_slot = -1
        self.original_prompt_length = 0
        self.current_context_length = 0
        self.input_ids = "stale"
        self.decoded_tokens = "stale"

    def clamp_decode_to_context(self, prompt_len, context_len):
        max_decode = max(0, min(self.original_max_decode_length, context_len - prompt_len))
        self.max_decode_length = max_decode
        self.original_max_decode_length = max_decode
        self.kv_token_budget = prompt_len + max_decode
        return self.kv_token_budget


class _Batch:
    def __init__(self, sequences):
        self.sequences = {seq.uuid: seq for seq in sequences}

    def get_sequence(self, uuid):
        return self.sequences.get(uuid)

    def remove_sequence(self, uuid):
        self.sequences.pop(uuid, None)


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _entry(uuid, token_ids):
    return {"request_id": uuid, "input_ids": np.asarray(token_ids, dtype=np.int32)}


def _worker(prompts, *, model_context_length=64, max_decode_length=8):
    sequences = [_Sequence(f"req-{i}", max_decode_length) for i in range(len(prompts))]
    worker = types.SimpleNamespace(
        rank=0,
        model_context_length=model_context_length,
        _response_queue=_Queue(),
        global_batch=_Batch(sequences),
        _trajectory_book=QueryBook(
            capacity_bytes=len(prompts) * model_context_length * 4,
            page_tokens=model_context_length,
        ),
        _publish_trajectory_pool_capacity=lambda: None,
        _prefill_prefix_pool_enabled=lambda: False,
    )
    uuids = [seq.uuid for seq in sequences]
    ids = [_ADMITTED_PROMPT_IDS(_entry(u, p)) for u, p in zip(uuids, prompts)]
    return worker, uuids, ids


def test_received_ids_are_written_to_the_book_unchanged():
    prompts = [[5, 9, 2], [163586, 7], [1]]
    worker, uuids, ids = _worker(prompts)

    _BIND(worker, uuids, ids)

    for uuid, expected in zip(uuids, prompts):
        seq = worker.global_batch.get_sequence(uuid)
        slot = worker._trajectory_book.slot_for(uuid)
        assert worker._trajectory_book.tokens(slot).tolist() == expected
        assert seq._buffer_slot == slot
        assert seq.prompt_length == len(expected)
        assert seq.original_prompt_length == len(expected)
        assert seq.current_context_length == len(expected)
        assert seq.kv_token_budget == len(expected) + seq.max_decode_length
        assert seq.input_ids is None and seq.decoded_tokens is None
    assert worker._response_queue.items == []


def test_over_context_prompt_fails_alone_with_context_length_exceeded():
    prompts = [[1, 2, 3], list(range(8)), [4, 5]]
    worker, uuids, ids = _worker(prompts, model_context_length=8, max_decode_length=4)

    _BIND(worker, uuids, ids)

    assert worker.global_batch.get_sequence(uuids[1]) is None
    (error,) = worker._response_queue.items
    assert error["request_id"] == uuids[1]
    assert error["error"]["code"] == "context_length_exceeded"
    assert error["error"]["message"] == "Prompt length 8 exceeds model context 8"
    for uuid, expected in ((uuids[0], prompts[0]), (uuids[2], prompts[2])):
        slot = worker._trajectory_book.slot_for(uuid)
        assert worker._trajectory_book.tokens(slot).tolist() == expected


def test_binding_neither_tokenizes_nor_runs_a_collective():
    """Every rank already holds identical ids; admission is rank-local."""
    tree = ast.parse(WORKER_SOURCE.read_text())
    (method,) = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_bind_admitted_prompts"
    ]
    names = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(method)
        if isinstance(node, (ast.Attribute, ast.Name))
    }
    assert not names & {"tokenizer", "dist", "all_gather_object", "broadcast_object_list"}


def test_entry_ids_are_shared_not_copied():
    entry = _entry("r", [3, 1, 4])
    tensor = _ADMITTED_PROMPT_IDS(entry)
    assert tensor.dtype == torch.int32
    assert tensor.data_ptr() == entry["input_ids"].ctypes.data


@pytest.mark.parametrize(
    "bad",
    [
        "a prompt string",
        [1, 2, 3],
        np.asarray([1, 2, 3], dtype=np.int64),
        np.zeros((1, 3), dtype=np.int32),
        None,
    ],
)
def test_anything_but_a_flat_int32_array_is_refused(bad):
    with pytest.raises(TypeError, match="input_ids must be a 1-D int32 numpy array"):
        _ADMITTED_PROMPT_IDS({"request_id": "r", "input_ids": bad})


def test_the_pickled_payload_is_about_four_bytes_per_token():
    """The admission message crosses a process queue and a rank broadcast."""
    token_ids = np.arange(100_000, dtype=np.int32)
    payload = pickle.dumps({"entries": [_entry("r", token_ids)]})
    assert len(payload) < token_ids.nbytes + 1024
    (received,) = pickle.loads(payload)["entries"]
    assert np.array_equal(_ADMITTED_PROMPT_IDS(received).numpy(), token_ids)
