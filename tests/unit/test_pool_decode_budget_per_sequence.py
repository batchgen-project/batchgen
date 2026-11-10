"""A pool-mode sequence is bounded by its own decode budget, never the first batch's.

The server sends the pool ``init`` message once, carrying the first batch's
max output length, so ``Init(max_decoding_length=...)`` describes only that
batch. Host-KV sizing must depend on the admitted sequence alone: a sequence
admitted after a small first batch reserves and grows exactly as it would on a
worker initialised with its own budget, and a short sequence admitted after a
large first batch reserves no more than its own budget.

batchgen_worker imports the whole engine, so the methods are compiled from the
real source against a fake worker (tests/worker_method_source.py).
"""

import logging
import math
import os
import time
from types import SimpleNamespace

import pytest

import batchgen.sequence as _sequence_module
from batchgen.continuous_batching import AdaptiveChunkSizer
from batchgen.query_book import release_local_query_slot
from batchgen.sequence import SequenceBatch, SequenceEntry
from batchgen.worker.prefill import compute_prefill_host_reservation
from tests.worker_method_source import load_worker_methods

_METHODS = load_worker_methods(
    [
        "Init",
        "_get_effective_chunk_size",
        "_prefill_host_reservation_plan",
        "_owns_host_kv",
        "_decode_attn_tp_size",
        "_admit_sequences_from_message",
    ],
    {
        "SequenceEntry": SequenceEntry,
        "compute_prefill_host_reservation": compute_prefill_host_reservation,
        "_sequence_module": _sequence_module,
        "release_local_query_slot": release_local_query_slot,
        "_release_tokenizer_arenas": lambda **_: None,
        "logging": logging,
        "math": math,
        "os": os,
        "time": time,
    },
)

HOST_KV_CHUNK = 8192
PAGE = SequenceEntry.PAGE_SIZE


class _Worker:
    """Owner rank 0 of a one-rank pool worker whose core is already built."""

    DECISION_INTERVAL = _sequence_module.DECISION_FREQUENCY_PAGES * PAGE
    world_size = 1
    rank = 0
    device = "cpu"
    enable_prefix_cache = False
    _trajectory_book = None

    Init = _METHODS["Init"]
    _get_effective_chunk_size = _METHODS["_get_effective_chunk_size"]
    _prefill_host_reservation_plan = _METHODS["_prefill_host_reservation_plan"]
    _owns_host_kv = _METHODS["_owns_host_kv"]
    _decode_attn_tp_size = _METHODS["_decode_attn_tp_size"]
    _admit_sequences_from_message = _METHODS["_admit_sequences_from_message"]

    def __init__(self, adaptive: bool):
        self.args = SimpleNamespace(global_rank=0, world_size=1)
        self.parallel_manager = SimpleNamespace(attn_tp_size=1)
        self._core_initialized = True
        self.host_kv_chunk_size = HOST_KV_CHUNK
        self.adaptive_chunk_sizer = (
            AdaptiveChunkSizer(
                initial_chunk=HOST_KV_CHUNK,
                min_chunk=1024,
                max_chunk=65536,
                ema_alpha=1.0,
                multiplier=1.0,
            )
            if adaptive
            else None
        )
        self.global_batch = SequenceBatch()
        self._prefix_sequence_states = {}
        self._group_uuids = set()
        self.max_input_length = 0

    # Pool re-init only refreshes engine-config hints; no sizing happens there.
    def _update_batch_config(self, num_queries):
        pass

    def _tokenize_admitted_sequences(self, uuids):
        for uuid in uuids:
            seq = self.global_batch.get_sequence(uuid)
            seq.prompt_length = len(seq.text)
            seq.original_prompt_length = seq.prompt_length
            seq.kv_token_budget = seq.prompt_length + seq.max_decode_length
            seq.assigned_rank = 0

    def _update_config_after_tokenization(self):
        pass

    def _build_local_query_book_for_admitted(self, uuids):
        pass

    def _maybe_emit_prefix_drain_metric(self):
        pass


def _pool_worker(first_batch_budget: int, *, adaptive: bool) -> _Worker:
    worker = _Worker(adaptive)
    worker.Init(None, first_batch_budget, 0)
    if adaptive:
        # Completed long sequences drive the EMA chunk past the first budget.
        for _ in range(10):
            worker.adaptive_chunk_sizer.report_completion(20000)
    return worker


def _host_kv_schedule(worker: _Worker, prompt: int, max_tokens: int):
    """Initial Host-KV reservation, then every growth step to the budget."""
    seq = SequenceEntry(
        uuid=f"r{len(worker.global_batch)}",
        global_idx=len(worker.global_batch),
        prompt_length=prompt,
        max_decode_length=max_tokens,
    )
    seq.assigned_rank = 0
    worker.global_batch.add_sequence(seq)
    reservations, owner_requests = worker._prefill_host_reservation_plan([seq.uuid])
    pages, tokens = reservations[seq.uuid]
    assert owner_requests == [(seq.uuid, tokens)]
    seq.host_token_capacity = tokens
    chunk = worker._get_effective_chunk_size()
    growth = []
    while (step := seq.get_host_growth_pages(chunk)) > 0:
        growth.append(step)
        seq.host_token_capacity += step * PAGE
    return tokens, growth, seq.host_token_capacity, seq.kv_token_budget


@pytest.mark.parametrize("adaptive", [False, True])
def test_long_request_after_short_first_batch_reserves_like_its_own_budget(adaptive):
    prompt, long_budget = 1000, 32768
    after_short = _host_kv_schedule(
        _pool_worker(128, adaptive=adaptive), prompt, long_budget
    )
    own_budget = _host_kv_schedule(
        _pool_worker(long_budget, adaptive=adaptive), prompt, long_budget
    )
    assert after_short == own_budget
    initial, growth, final_capacity, budget = after_short
    chunk = 20032 if adaptive else HOST_KV_CHUNK
    # The configured chunk sizes the reservation, not the first batch's 128.
    assert initial == math.ceil((prompt + chunk) / PAGE) * PAGE
    assert growth[0] == math.ceil(min(chunk, budget - initial) / PAGE)
    assert final_capacity == math.ceil(budget / PAGE) * PAGE


@pytest.mark.parametrize("adaptive", [False, True])
def test_short_request_after_long_first_batch_reserves_only_its_budget(adaptive):
    prompt, short_budget = 1000, 16
    after_long = _host_kv_schedule(
        _pool_worker(32768, adaptive=adaptive), prompt, short_budget
    )
    own_budget = _host_kv_schedule(
        _pool_worker(short_budget, adaptive=adaptive), prompt, short_budget
    )
    assert after_long == own_budget
    initial, growth, _, budget = after_long
    assert initial == math.ceil(budget / PAGE) * PAGE
    assert growth == []


def _admit(worker, entries):
    worker._admit_sequences_from_message({"entries": entries})


def test_admission_uses_each_entry_budget():
    worker = _pool_worker(128, adaptive=False)
    _admit(worker, [
        {"request_id": "long", "text": "x" * 40, "max_tokens": 32768, "batch_id": "b"},
        {"request_id": "short", "text": "y" * 40, "max_tokens": 16, "batch_id": "b"},
    ])
    long_seq = worker.global_batch.get_sequence("long")
    short_seq = worker.global_batch.get_sequence("short")
    assert (long_seq.max_decode_length, long_seq.kv_token_budget) == (32768, 40 + 32768)
    assert (short_seq.max_decode_length, short_seq.kv_token_budget) == (16, 40 + 16)


def test_admission_without_a_budget_fails_before_admitting_anything():
    worker = _pool_worker(128, adaptive=False)
    with pytest.raises(ValueError, match="no max_tokens"):
        _admit(worker, [
            {"request_id": "ok", "text": "x", "max_tokens": 8, "batch_id": "b"},
            {"request_id": "bare", "text": "y", "batch_id": "b"},
        ])
    assert len(worker.global_batch) == 0


# Methods that may read the worker-wide value: init-time sizing hints only
# (engine-config inputs and the page-table capacity request, whose maximum
# always includes the deployment context bound). None of them caps a sequence.
_INIT_TIME_READERS = {
    "Init",
    "generate_persistent",
    "_initialize_core_components",
    "_update_batch_config",
    "_make_page_table_capacity_request",
}
