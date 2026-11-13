"""Cross-node migration must not change trajectory-book reservations.

Every rank keeps its own QueryBook allocator over one token store shared by
the ranks of its node. Admission binds every admitted sequence on every rank
of every node, and completion or rejection releases it on every rank, so the
allocators of all ranks evolve identically. Migration moves token content
only. A migration that released the reservation on the source node and bound
it again on the receiving rank alone left the ranks of one node with
different free extents: the next admission then placed one prompt at
different pages per rank inside the shared store, a later re-entry found no
reservation on the source node, and the free capacity of the nodes drifted
apart.

Each simulated rank is a shell object carrying the real worker migration
methods, loaded from the worker source (tests/worker_method_source.py)
because importing the worker module requires GPU kernel packages. The ranks run ``_rebalance_host_kv``
concurrently on threads, with point-to-point messages and barriers provided
by an in-process fake of torch.distributed.
"""

import logging
import threading
import time
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
import torch

from batchgen.lifespan import SeqEvent
from batchgen.migration import MigrationOp
from batchgen.query_book import QueryBook, make_query_book_entry
from batchgen.sequence import SequenceEntry
from batchgen.worker.host_rebalancer import HostKVRebalancer
from tests.worker_method_source import load_worker_methods

WORKER_METHODS = (
    "_rebalance_host_kv",
    "_execute_kv_migrations_parallel",
    "_execute_single_kv_migration",
    "_group_migrations_for_parallel_execution",
    "_get_node_for_rank",
    "_trajectory_slot",
    "_trajectory_tokens",
    "_sync_trajectory_metadata",
)

LOCAL_WORLD_SIZE = 2
NUM_NODES = 2
WORLD_SIZE = LOCAL_WORLD_SIZE * NUM_NODES
PAGE_TOKENS = 4
CONTEXT = 16  # four pages per reservation
PAGE_COUNT = 40
TIMEOUT_S = 30


class _FakeDist:
    """Point-to-point queues and a world barrier for threads acting as ranks."""

    def __init__(self, world_size):
        self._barrier = threading.Barrier(world_size, timeout=TIMEOUT_S)
        self._channels = defaultdict(Queue)
        self._local = threading.local()

    def reset(self):
        self._barrier = threading.Barrier(WORLD_SIZE, timeout=TIMEOUT_S)

    def bind_rank(self, rank):
        self._local.rank = rank

    def abort(self):
        self._barrier.abort()

    def barrier(self, *args, **kwargs):
        self._barrier.wait()

    def send(self, tensor, dst, group=None):
        self._channels[(self._local.rank, dst)].put(tensor.clone())

    def recv(self, tensor, src, group=None):
        tensor.copy_(self._channels[(src, self._local.rank)].get(timeout=TIMEOUT_S))


class _NoDualCoordinator:
    """Stands in for DualHostKVCoordinator; the shell uses one host-KV view."""


class QueryBookPoolCapacityError(RuntimeError):
    """Stands in for the worker's named trajectory-book error."""


def _load_worker_shell(fake_dist):
    """Build a rank class from the worker's real migration methods."""
    methods = load_worker_methods(WORKER_METHODS, {
        "BATCHGEN_CB_DEBUG": False,
        "Dict": Dict,
        "DualHostKVCoordinator": _NoDualCoordinator,
        "HostKVRebalancer": HostKVRebalancer,
        "List": List,
        "MigrationOp": MigrationOp,
        "Optional": Optional,
        "QueryBook": QueryBook,
        "QueryBookPoolCapacityError": QueryBookPoolCapacityError,
        "SeqEvent": SeqEvent,
        "SequenceEntry": SequenceEntry,
        "dist": fake_dist,
        "logging": logging,
        "make_query_book_entry": make_query_book_entry,
        "time": time,
        "torch": torch,
    })
    return type("WorkerShell", (), methods)


class _HostKV:
    """Host-KV lifecycle view that carries one page of placeholder KV."""

    def read_sequence_kv_to_cpu(self, global_idx):
        return torch.zeros((1, 4), dtype=torch.float32), torch.empty(0)

    def write_sequence_kv_from_cpu(self, global_idx, k, v):
        pass

    def register_sequences(self, global_indices):
        pass

    def allocate_pages_for_sequences(self, requests):
        pass

    def release_sequence_pages(self, global_indices):
        pass


class _GlobalBatch:
    def __init__(self):
        self._sequences = {}

    def add(self, seq):
        self._sequences[seq.uuid] = seq

    def get_sequence(self, uuid):
        return self._sequences.get(uuid)

    def assign_rank(self, uuid, rank):
        self._sequences[uuid].assigned_rank = rank


def _node(rank):
    return rank // LOCAL_WORLD_SIZE


def _make_cluster():
    storages = [
        torch.zeros((PAGE_COUNT, PAGE_TOKENS), dtype=torch.int32)
        for _ in range(NUM_NODES)
    ]
    fake = _FakeDist(WORLD_SIZE)
    shell_cls = _load_worker_shell(fake)
    workers = []
    for rank in range(WORLD_SIZE):
        worker = shell_cls()
        worker.fake_dist = fake
        worker.PAGE_SIZE = PAGE_TOKENS
        worker.rank = rank
        worker.local_rank = rank % LOCAL_WORLD_SIZE
        worker.world_size = WORLD_SIZE
        worker.local_world_size = LOCAL_WORLD_SIZE
        worker.model_context_length = CONTEXT
        worker.enable_decode_preemption = True
        worker._trajectory_book = QueryBook(
            capacity_bytes=PAGE_COUNT * PAGE_TOKENS * QueryBook.TOKEN_BYTES,
            page_tokens=PAGE_TOKENS,
            storage=storages[_node(rank)],
        )
        worker.global_batch = _GlobalBatch()
        worker.host_paged_kv_worker_view = _HostKV()
        worker.host_paged_kv_worker_view_aux = None
        worker._uuid_to_local_map = {}
        worker._local_to_uuid_map = {}
        worker.query_book = {}
        worker._free_local_indices = set()
        worker._next_local_idx = 0
        worker._sequences_with_gpu_kv = set()
        worker._release_prefix_cache_attachments = lambda indices: None
        worker._get_or_create_gloo_group = lambda: None
        worker._publish_trajectory_pool_capacity = lambda: None
        worker._sync_sequence_metadata = lambda uuids: None
        worker._bump_decode_row_epoch = lambda: None
        worker._get_host_kv_utilization = lambda: {
            "node_id": 0, "num_used_pages": 0, "num_total_pages": 1, "free_percent": 100,
        }
        workers.append(worker)
    return workers


def _admit(workers, prompts, owners):
    """Replay worker admission: bind_batch + write_prompt on every rank."""
    uuids = list(prompts)
    for worker in workers:
        for global_idx, uuid in enumerate(uuids, start=len(worker.global_batch._sequences)):
            length = len(prompts[uuid])
            worker.global_batch.add(SimpleNamespace(
                uuid=uuid, global_idx=global_idx, assigned_rank=owners[uuid],
                host_pages_allocated=1, host_owned_pages=1, host_token_capacity=PAGE_TOKENS,
                prompt_length=length, original_prompt_length=length,
                current_context_length=length, decoded_length=0,
                text="", input_ids=None, decoded_tokens=None, kv_token_budget=CONTEXT,
                _buffer_slot=-1, log_event=lambda *args, **kwargs: None,
            ))
        slots = worker._trajectory_book.bind_batch(uuids, [CONTEXT] * len(uuids))
        for uuid, slot in zip(uuids, slots):
            worker._trajectory_book.write_prompt(slot, torch.tensor(prompts[uuid]))
            worker.global_batch.get_sequence(uuid)._buffer_slot = slot
    for uuid, owner in owners.items():
        local_idx = workers[owner]._next_local_idx
        workers[owner]._next_local_idx += 1
        workers[owner]._uuid_to_local_map[uuid] = local_idx
        workers[owner]._local_to_uuid_map[local_idx] = uuid


def _decode(workers, uuid, owner, tokens):
    """The owning rank appends generated tokens; every rank syncs the length."""
    book = workers[owner]._trajectory_book
    for token in tokens:
        book.append_token(book.slot_for(uuid), token)
    for worker in workers:
        worker.global_batch.get_sequence(uuid).decoded_length += len(tokens)


def _migrate(workers, migrations):
    """Run the worker's _rebalance_host_kv on every rank concurrently."""
    fake = workers[0].fake_dist
    fake.reset()
    errors = [None] * WORLD_SIZE

    def run(rank):
        fake.bind_rank(rank)
        worker = workers[rank]
        worker._plan_kv_migration = lambda: list(migrations)
        try:
            worker._rebalance_host_kv()
        except BaseException as exc:  # surfaced below with the rank
            errors[rank] = exc
            fake.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(WORLD_SIZE)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2 * TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads), "migration replay hung"
    root = [(rank, exc) for rank, exc in enumerate(errors)
            if exc is not None and not isinstance(exc, threading.BrokenBarrierError)]
    if root:
        rank, exc = root[0]
        raise AssertionError(f"rank {rank} failed during migration: {exc!r}") from exc
    assert all(exc is None for exc in errors), errors


def _page_range(worker, uuid):
    book = worker._trajectory_book
    if not book.has_sequence(uuid):
        return None
    return book._page_ranges[book.slot_for(uuid)]


def _tokens(worker, uuid):
    book = worker._trajectory_book
    return book.tokens(book.slot_for(uuid)).tolist()


def test_migrate_away_and_back_keeps_node_page_ranges_identical():
    workers = _make_cluster()
    prompts = {"s0": [11, 12, 13, 14, 15], "s1": [21, 22, 23]}
    _admit(workers, prompts, owners={"s0": 0, "s1": 3})
    _decode(workers, "s0", owner=0, tokens=[101, 102, 103])
    trajectory = prompts["s0"] + [101, 102, 103]

    _migrate(workers, [MigrationOp("s0", from_rank=0, to_rank=2, pages=1, host_pages=1)])
    _migrate(workers, [MigrationOp("s0", from_rank=2, to_rank=1, pages=1, host_pages=1)])
    late_prompts = {"s2": [31, 32, 33, 34, 35, 36]}
    _admit(workers, late_prompts, owners={"s2": 0})

    for node in range(NUM_NODES):
        ranks = [r for r in range(WORLD_SIZE) if _node(r) == node]
        for uuid in ("s0", "s1", "s2"):
            ranges = {r: _page_range(workers[r], uuid) for r in ranks}
            assert len(set(ranges.values())) == 1, (node, uuid, ranges)
    assert _tokens(workers[1], "s0") == trajectory
    for worker in workers:
        assert _tokens(worker, "s2") == late_prompts["s2"]
        assert _tokens(worker, "s1") == prompts["s1"]


def _evict_and_reenter(workers, uuid, owner):
    """Replay the eviction boundary and the all-rank re-entry validation."""
    owner_book = workers[owner]._trajectory_book
    owner_slot = owner_book.slot_for(uuid)
    reentry_len = owner_book.metadata(owner_slot).token_length
    owner_book.begin_reentry_turn(owner_slot, prompt_length=reentry_len)
    record = owner_book.metadata(owner_slot)
    state = {
        "prompt_length": reentry_len,
        "decoded_length": record.decoded_length,
        "trajectory_token_length": record.token_length,
        "trajectory_decoded_length": record.decoded_length,
    }
    for worker in workers:
        seq = worker.global_batch.get_sequence(uuid)
        seq.prompt_length = reentry_len
        if worker.rank != owner:
            worker._sync_trajectory_metadata(seq, state)
        # Mirrors the re-entry loop in _config_prefill_for_batch, which runs on
        # every rank for every re-entering sequence of the wave.
        book = worker._trajectory_book
        if not book.has_sequence(uuid):
            raise QueryBookPoolCapacityError(
                f"Rank {worker.rank}: re-entry of {uuid} has no trajectory slot"
            )
        slot = worker._trajectory_slot(seq)
        assert book.metadata(slot).token_length >= seq.prompt_length
        book.begin_reentry_turn(slot, prompt_length=seq.prompt_length)


def test_reentry_after_on_hold_migration_finds_a_reservation_on_every_rank():
    workers = _make_cluster()
    prompts = {"s0": [41, 42, 43, 44], "s1": [51, 52]}
    _admit(workers, prompts, owners={"s0": 1, "s1": 2})
    _decode(workers, "s0", owner=1, tokens=[201, 202])

    # s0 is ON_HOLD (host KV only) when the planner moves it to node 1.
    _migrate(workers, [MigrationOp("s0", from_rank=1, to_rank=3, pages=1, host_pages=1)])
    _evict_and_reenter(workers, "s0", owner=3)

    for worker in workers:
        assert worker._trajectory_book.turns(worker._trajectory_book.slot_for("s0"))[-1] \
            .prompt_length == len(prompts["s0"]) + 2
    assert _tokens(workers[3], "s0") == prompts["s0"] + [201, 202]


def test_free_extents_stay_equal_across_nodes_after_migrations():
    workers = _make_cluster()
    prompts = {"s0": [1, 2, 3], "s1": [4, 5], "s2": [6, 7, 8, 9]}
    _admit(workers, prompts, owners={"s0": 0, "s1": 1, "s2": 2})
    _decode(workers, "s1", owner=1, tokens=[61])

    _migrate(workers, [
        MigrationOp("s0", from_rank=0, to_rank=2, pages=1, host_pages=1),
        MigrationOp("s2", from_rank=2, to_rank=1, pages=1, host_pages=1),
    ])
    _migrate(workers, [MigrationOp("s1", from_rank=1, to_rank=3, pages=1, host_pages=1)])

    extents = {rank: list(w._trajectory_book._free_extents) for rank, w in enumerate(workers)}
    assert len({tuple(value) for value in extents.values()}) == 1, extents
    assert {w._trajectory_book.free_page_count for w in workers} == {
        PAGE_COUNT - 3 * (CONTEXT // PAGE_TOKENS)
    }


def test_receiving_rank_without_admission_reservation_fails_by_name():
    workers = _make_cluster()
    _admit(workers, {"s0": [7, 8, 9]}, owners={"s0": 0})
    destination = workers[2]._trajectory_book
    destination.release(destination.slot_for("s0"))

    with pytest.raises(AssertionError, match="no admission reservation"):
        _migrate(workers, [MigrationOp("s0", from_rank=0, to_rank=2, pages=1, host_pages=1)])
