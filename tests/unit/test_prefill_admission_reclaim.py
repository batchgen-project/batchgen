"""Prefill admission against a Host pool full of retained prefixes.

The livelock: with Kimi-K3's persistent KDA-slot limit the admission reclaim in
``_prepare_prefill_batch`` was skipped outright, so retained prefix pages filled
the Host pool and every later wave selected zero sequences forever, with no
eviction and no error. These tests drive the REAL ``_prepare_prefill_batch``
(compiled out of the worker's AST onto fakes, as the other worker-method tests
do), the real ``PrefillScheduler`` and the real trajectory-frontier committer and
registry. Only the native boundaries are faked: the shared prefix-cache
coordinator (a Python mirror of ``EvictUntilReleasablePages``: LRU by last
access, nodes with live attachments are protected and counted), the Host-KV
pool, and the KDA snapshot reader.
"""

import ast
import copy
import dataclasses
import hashlib
import logging
import random
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from batchgen.prefix_reuse.eviction import (
    EvictedPageRecorder,
    retire_node_frontiers,
)
from batchgen.query_book import release_local_query_slot
from batchgen.prefix_reuse.trajectory_commit import (
    SnapshotArena,
    SnapshotLayout,
    TrajectoryFrontierCommitter,
)
from batchgen.prefix_reuse.trajectory_frontier import KDASnapshotProfile
from batchgen.sequence import SequenceStatus
from batchgen.worker.prefill import (
    PREFILL_NO_PROGRESS_LIMIT,
    PREFILL_RECLAIM_TARGET_CANDIDATES,
    PrefillAdmissionNoProgress,
    PrefillAdmissionStalled,
    PrefillAdmissionStateDivergence,
    PrefillCandidate,
    PrefillProgressGuard,
    PrefillScheduler,
    PrefillSelectionRequest,
    describe_prefill_blockers,
    prefill_reclaim_target_pages,
)


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"
PAGE_TOKENS = 64  # SequenceEntry.PAGE_SIZE
HOST_PAGE_BYTES = 2048
NAMESPACE = (1, 2, 3, 4)
METHODS = (
    "_prepare_prefill_batch",
    "_make_prefill_selection_request",
    "_prefill_sequence_limits",
    "_trajectory_frontier_eviction_hook",
    "_describe_prefill_admission_blockers",
    "_get_host_kv_total_pages",
    "_prefill_candidate_digest",
    "_prefill_prefix_pressure",
    "_prefill_admission_guard_state",
    "_reject_unsatisfiable_prefill_requests",
    # Called by _prepare_prefill_batch right after the group stamp; a no-op at
    # the G==1 these fakes run, but compiled from the real source so the call
    # is exercised rather than stubbed away.
    "_check_decode_dp_group_plan_identical",
)


# --------------------------------------------------------------------------- #
#  Worker methods compiled onto fakes
# --------------------------------------------------------------------------- #


def _worker_methods(dist):
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    wanted = [
        copy.deepcopy(node) for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name in METHODS
    ]
    assert {node.name for node in wanted} == set(METHODS)
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {
        "List": List, "Optional": Optional, "Sequence": Sequence,
        "Dict": Dict, "Tuple": Tuple, "hashlib": hashlib,
        "logging": logging, "replace": dataclasses.replace, "torch": torch,
        "dist": dist, "core_engine": FakeCoreEngineModule,
        "PrefillScheduler": PrefillScheduler,
        "PrefillCandidate": PrefillCandidate,
        "PrefillSelectionRequest": PrefillSelectionRequest,
        "PrefillAdmissionStalled": PrefillAdmissionStalled,
        "PrefillAdmissionStateDivergence": PrefillAdmissionStateDivergence,
        "describe_prefill_blockers": describe_prefill_blockers,
        "prefill_reclaim_target_pages": prefill_reclaim_target_pages,
        "release_local_query_slot": release_local_query_slot,
        "SequenceStatus": SequenceStatus,
    }
    exec(compile(module, str(WORKER), "exec"), namespace)
    return {name: namespace[name] for name in METHODS}


class FakeDist:
    """Collectives for one stepped rank; peers contribute scripted values.

    ``peers`` maps a collective's ordinal (0 = first collective of the call) to
    the per-rank contributions of the OTHER ranks, keyed by rank. Inside a
    gathered vector, ``None`` means "this peer reports whatever the stepped rank
    reports", which is how a scripted peer agrees on a digest it cannot know.
    """

    ReduceOp = SimpleNamespace(MIN="min", MAX="max")

    def __init__(self, *, rank=0, world_size=1, peers=None):
        self.rank = rank
        self.world_size = world_size
        self.peers = peers or {}
        self.calls: list[str] = []

    def _peer(self, rank):
        return self.peers.get(len(self.calls) - 1, {}).get(rank)

    def all_gather(self, out, tensor):
        self.calls.append("all_gather")
        for rank in range(self.world_size):
            value = tensor if rank == self.rank else self._peer(rank)
            if isinstance(value, (list, tuple)):
                value = [
                    int(tensor[index].item()) if item is None else item
                    for index, item in enumerate(value)
                ]
            out[rank].copy_(torch.as_tensor(value, dtype=tensor.dtype))

    def all_reduce(self, tensor, op):
        self.calls.append("all_reduce")
        for rank in range(self.world_size):
            if rank == self.rank:
                continue
            peer = torch.as_tensor(self._peer(rank), dtype=tensor.dtype)
            tensor.copy_(torch.maximum(tensor, peer) if op == "max"
                         else torch.minimum(tensor, peer))

    def all_gather_object(self, out, obj):
        self.calls.append("all_gather_object")
        for rank in range(self.world_size):
            out[rank] = obj if rank == self.rank else self._peer(rank)


class FakeCoreEngineModule:
    class GroupPageRequirement:
        def __init__(self):
            self.group_id = -1
            self.min_pages = 0


class FakeHostPool:
    """One node's shared Host-KV region: page ids, free count, release."""

    def __init__(self, total_pages):
        self.total = int(total_pages)
        self.used: set[int] = set()
        self._next = 0

    def take(self, count):
        pages = []
        for _ in range(int(count)):
            while self._next in self.used:
                self._next += 1
            pages.append(self._next)
            self.used.add(self._next)
        if len(self.used) > self.total:
            raise AssertionError("fake Host pool over-allocated")
        return pages

    def release_resident_pages(self, page_ids):
        for page in page_ids:
            self.used.remove(int(page))  # KeyError = double free

    def get_stats(self):
        return SimpleNamespace(
            num_total_pages=self.total,
            num_free_pages=self.total - len(self.used),
        )


class FakePrefixCoordinator:
    """Python mirror of the native ``EvictUntilReleasablePages`` contract."""

    def __init__(self, pool):
        self.pool = pool
        self.nodes: dict[int, dict] = {}
        self.evicted_nodes_total = 0
        self.eviction_protected_skips_total = 0
        self.pending_load_entries = 0
        self.calls = []

    def retain(self, node_id, *, pages, last_access, active_refs=0):
        self.nodes[node_id] = {
            "pages": list(pages), "last_access": last_access,
            "active_refs": active_refs,
        }

    def get_stats(self):
        """The pressure counters the admission state snapshots."""
        return SimpleNamespace(
            resident_nodes=len(self.nodes),
            active_attachments=sum(
                int(node["active_refs"]) for node in self.nodes.values()
            ),
            pending_load_entries=self.pending_load_entries,
            pending_load_refs=0,
        )

    def evict_until_releasable_pages(self, requirements, max_scan_nodes):
        required = {int(r.group_id): int(r.min_pages) for r in requirements}
        self.calls.append((dict(required), int(max_scan_nodes)))
        released: dict[int, list[int]] = {}
        evicted = protected = scanned = 0
        order = sorted(self.nodes, key=lambda n: self.nodes[n]["last_access"])
        for node_id in order:
            if max_scan_nodes and scanned >= max_scan_nodes:
                break
            scanned += 1
            node = self.nodes[node_id]
            if node["active_refs"]:
                protected += 1
                continue
            released.setdefault(0, []).extend(node["pages"])
            del self.nodes[node_id]
            evicted += 1
            if all(len(released.get(g, [])) >= m for g, m in required.items()):
                break
        self.evicted_nodes_total += evicted
        self.eviction_protected_skips_total += protected
        return SimpleNamespace(
            evicted_nodes=evicted,
            protected_nodes=protected,
            evicted_group_pages=[
                SimpleNamespace(group_id=g, pages=list(p))
                for g, p in released.items()
            ],
        )


def _profile():
    return KDASnapshotProfile(
        num_kda_layers=2, num_heads_per_rank=2, head_dim=16,
        conv_dim_per_rank=32, conv_width=4,
        recurrent_element_bytes=4, conv_element_bytes=2,
    )


class _Chunk:
    def __init__(self, nbytes):
        self._nbytes = int(nbytes)

    def numel(self):
        return self._nbytes


class _SnapshotReader:
    def __init__(self, profile):
        self.profile = profile

    def read_slot(self, sequence_id):
        return tuple(
            _Chunk(size) for size in SnapshotLayout(profile=self.profile).chunk_bytes
        )


class PoolArenaAllocator:
    """Snapshot arenas drawn from (and returned to) the same fake Host pool."""

    def __init__(self, pool):
        self.pool = pool
        self.releases = []
        self._next_sid = -1

    def store(self, *, page_count, chunks):
        pages = tuple(self.pool.take(page_count))
        sid, self._next_sid = self._next_sid, self._next_sid - 1
        return SnapshotArena(
            storage_handle=0, pseudo_sid=sid, page_ids_by_group={0: pages},
            pages_per_group=int(page_count),
            capacity_bytes=int(page_count) * HOST_PAGE_BYTES,
            payload_bytes=sum(int(c.numel()) for c in chunks),
        )

    def release(self, arena):
        self.releases.append(arena.page_ids_by_group[0])
        self.pool.release_resident_pages(arena.page_ids_by_group[0])
        return {0: len(arena.page_ids_by_group[0])}


def _committer(pool, rank=0):
    allocator = PoolArenaAllocator(pool)
    committer = TrajectoryFrontierCommitter(
        rank=rank, page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        arena_allocator=allocator, snapshot_reader=_SnapshotReader(_profile()),
    )
    return committer, allocator


def _commit_frontier(committer, *, pages, seed, sequence_id):
    tokens = [seed * 1000 + i for i in range(len(pages) * PAGE_TOKENS)]
    outcome = committer.commit(
        namespace_digest=NAMESPACE, frontier_token_ids=tokens,
        page_ids_by_group={0: list(pages)}, owning_group_id=0,
        sequence_id=sequence_id, prompt_token_count=len(tokens),
    )
    assert outcome is not None
    return outcome


class FakeBatch:
    def __init__(self, sequences):
        self.sequences = {seq.uuid: seq for seq in sequences}

    def get_sequence(self, uuid):
        return self.sequences.get(uuid)

    def get_sequences_by_status(self, status):
        return [u for u, s in self.sequences.items() if s.status == status]

    def remove_sequence(self, uuid):
        return self.sequences.pop(uuid, None)

    def _has(self, status):
        return any(s.status == status for s in self.sequences.values())

    def has_prefilled(self):
        return self._has(SequenceStatus.PREFILLED)

    def has_in_decode(self):
        return self._has(SequenceStatus.IN_DECODE)

    def has_on_hold(self):
        return self._has(SequenceStatus.ON_HOLD)


def _seq(uuid, idx, prompt, status=SequenceStatus.QUEUEING):
    return SimpleNamespace(
        uuid=uuid, global_idx=idx, status=status, assigned_rank=0,
        decode_dp_group=None, total_decoded_before_eviction=0,
        prompt_length=prompt, kv_token_budget=1 << 20, PAGE_SIZE=PAGE_TOKENS,
        prefill_order=None,
    )


class FakeWorker:
    def __init__(
        self, *, pool, coordinator, sequences, slot_limit=None, pool_mode=True,
        committer=None, rank=0, local_rank=0, world_size=1, local_world_size=1,
        dist=None, trajectory_book=None,
    ):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.local_world_size = local_world_size
        self.torch_device = torch.device("cpu")
        self.enable_host_kv_eviction = False
        self.enable_prefix_cache = True
        self.prefix_cache_debug_stats = False
        self.prefix_cache_coordinator = coordinator
        self.prefix_cache_runtime_config = SimpleNamespace(max_nodes=0)
        self.host_paged_kv_worker_view = pool
        self.core_engine = SimpleNamespace(host_paged_kv_worker_view=pool)
        self.global_batch = FakeBatch(sequences)
        self._trajectory_frontier_committer = committer
        self._prefill_admission_state = None
        self._prefill_admission_selection = None
        self._prefill_admission_totals = None
        self._prefill_admission_live = False
        self._prefill_reservation_outcome = None
        # Request-lifecycle surfaces a per-request rejection has to release.
        self._trajectory_book = trajectory_book
        self._uuid_to_local_map = {}
        self._local_to_uuid_map = {}
        self.query_book = {}
        self._free_local_indices = set()
        self._response_queue = FakeResponseQueue()
        self.published_capacity = 0
        self._pool_mode = pool_mode
        limits = {} if slot_limit is None else {
            "max_sequences_per_node": int(slot_limit)
        }
        self.parallel_manager = SimpleNamespace(
            prefill_sequence_limits=lambda: limits
        )
        self.dist = dist or FakeDist(rank=rank, world_size=world_size)
        for name, function in _worker_methods(self.dist).items():
            setattr(self, name, types.MethodType(function, self))

    def _get_num_nodes(self):
        return max(1, self.world_size // self.local_world_size)

    def _get_effective_chunk_size(self):
        return 0

    def _get_host_kv_free_pages(self):
        return self.host_paged_kv_worker_view.get_stats().num_free_pages

    def _assign_decode_dp_groups(self, uuids):
        for uuid in uuids:
            self.global_batch.get_sequence(uuid).decode_dp_group = 0

    def _decode_attn_tp_size(self):
        return 1

    def _prefill_prefix_pool_enabled(self):
        return self._pool_mode

    def _host_views_by_group(self):
        return {0: self.host_paged_kv_worker_view}

    def _publish_trajectory_pool_capacity(self):
        self.published_capacity += 1


class FakeResponseQueue:
    """Collects the per-request completions rank 0 publishes."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class FakeTrajectoryBook:
    """Just the reservation release surface a rejection has to drive."""

    def __init__(self, sequence_ids):
        self.slots = {uuid: index for index, uuid in enumerate(sequence_ids)}
        self.released = []

    def has_sequence(self, uuid):
        return uuid in self.slots

    def slot_for(self, uuid):
        return self.slots[uuid]

    def release(self, slot):
        self.released.append(slot)


def _need(prompt):
    """The selector's own reservation for one prompt (no hard-coded pages)."""
    candidate = PrefillCandidate(
        uuid="x", assigned_rank=0, node_id=0, is_evicted=False, global_idx=0,
        total_decoded_before_eviction=0, prompt_length=prompt,
        kv_token_budget=1 << 20, page_size=PAGE_TOKENS,
    )
    request = PrefillSelectionRequest(
        candidates=(candidate,), per_node_host_free=(0,), chunk_size=0,
        num_nodes=1, gpus_per_node=1, initial_gpu_page_buffer=32,
    )
    return PrefillScheduler.required_host_pages(candidate, request)


PROMPT = 640
NEED = _need(PROMPT)


def _full_pool_with_frontiers(*, total, protect_newest=False):
    """A pool of retained prefixes (each carrying a frontier) and little free.

    Node i retains 4 prefix pages; its frontier arena takes more Host pages
    from the same pool. last_access grows with i, so node 0 is the LRU.
    """
    pool = FakeHostPool(total)
    coordinator = FakePrefixCoordinator(pool)
    committer, allocator = _committer(pool)
    outcomes = []
    node = 0
    while pool.get_stats().num_free_pages >= 4 + 8 + NEED // 2:
        pages = pool.take(4)
        outcomes.append(
            _commit_frontier(committer, pages=pages, seed=node + 1, sequence_id=node)
        )
        coordinator.retain(node, pages=pages, last_access=node)
        node += 1
    if protect_newest:
        coordinator.nodes[node - 1]["active_refs"] = 1
    assert pool.get_stats().num_free_pages < NEED
    return pool, coordinator, committer, allocator, outcomes


# --------------------------------------------------------------------------- #
#  The livelock, reproduced on the K3 (slot-limited) path, and its reclaim
# --------------------------------------------------------------------------- #


def test_slot_limited_admission_reclaims_lru_retained_prefixes_and_admits():
    pool, coordinator, committer, allocator, outcomes = _full_pool_with_frontiers(
        total=400
    )
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, committer=committer, slot_limit=8,
        sequences=[_seq("q0", 0, PROMPT)],
    )
    nodes_before = len(coordinator.nodes)
    free_before = pool.get_stats().num_free_pages

    batch = worker._prepare_prefill_batch()

    # Before the fix this returned [] with the coordinator never called.
    assert batch == ["q0"]
    # One reclaim, sized to the shortfall, then one re-selection.
    assert coordinator.calls == [({0: NEED - free_before}, 0)]
    assert worker.dist.calls == ["all_gather", "all_reduce", "all_gather_object", "all_gather"]
    # LRU-first: exactly the oldest nodes went, in order, and no newer one.
    evicted = nodes_before - len(coordinator.nodes)
    assert evicted == coordinator.evicted_nodes_total > 0
    assert sorted(coordinator.nodes) == list(range(evicted, nodes_before))
    # Their frontier records retired with them, arena pages back in the pool.
    retired = {o.session_key for o in outcomes[:evicted]}
    surviving = {o.session_key for o in outcomes[evicted:]}
    assert set(committer.registry.session_keys) == surviving
    assert not retired & set(committer.registry.session_keys)
    assert len(allocator.releases) == evicted
    for outcome in outcomes[:evicted]:
        assert committer.arena_for(outcome.arena.storage_handle) is None
    committer.registry.check_invariants()
    # The re-selection saw the reclaimed pages: the candidate now fits.
    assert pool.get_stats().num_free_pages >= NEED
    assert worker._prefill_admission_state[0] == (pool.get_stats().num_free_pages,)


def test_reclaim_skips_attached_prefixes_and_their_frontiers_survive():
    pool, coordinator, committer, allocator, outcomes = _full_pool_with_frontiers(
        total=400
    )
    # Pin the two OLDEST nodes the way a live attachment does: the native
    # coordinator protects them, and a restore attachment holds the record.
    coordinator.nodes[0]["active_refs"] = 1
    coordinator.nodes[1]["active_refs"] = 1
    pinned = committer.registry.get(outcomes[0].session_key)
    pinned.acquire()
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, committer=committer, slot_limit=8,
        sequences=[_seq("q0", 0, PROMPT)],
    )

    assert worker._prepare_prefill_batch() == ["q0"]

    assert coordinator.eviction_protected_skips_total == 2
    assert 0 in coordinator.nodes and 1 in coordinator.nodes
    keys = set(committer.registry.session_keys)
    assert outcomes[0].session_key in keys and outcomes[1].session_key in keys
    assert pinned.refcount == 2
    assert committer.arena_for(outcomes[0].arena.storage_handle) is not None
    # The unprotected LRU node after them was the first to go.
    assert 2 not in coordinator.nodes
    committer.registry.check_invariants()


def test_nothing_reclaimable_and_nothing_live_fails_by_name_without_spinning():
    pool, coordinator, committer, _allocator, _ = _full_pool_with_frontiers(
        total=400
    )
    for node in coordinator.nodes.values():
        node["active_refs"] = 1
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, committer=committer, slot_limit=8,
        sequences=[_seq("q0", 0, PROMPT)],
    )

    with pytest.raises(PrefillAdmissionStalled) as excinfo:
        worker._prepare_prefill_batch()

    message = str(excinfo.value)
    assert "no progress after eviction with no live sequence" in message
    assert "seq q0" in message and f"needs {NEED} Host pages" in message
    assert coordinator.evicted_nodes_total == 0
    assert coordinator.eviction_protected_skips_total == len(coordinator.nodes)


def test_nothing_reclaimable_with_live_sequences_waits_for_them():
    pool, coordinator, committer, _allocator, _ = _full_pool_with_frontiers(
        total=400
    )
    for node in coordinator.nodes.values():
        node["active_refs"] = 1
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, committer=committer, slot_limit=8,
        sequences=[
            _seq("q0", 0, PROMPT),
            _seq("live", 1, PROMPT, status=SequenceStatus.IN_DECODE),
        ],
    )
    assert worker._prepare_prefill_batch() == []


def test_a_queued_request_larger_than_the_whole_pool_fails_that_request():
    """One oversized prompt must not take the worker down with it.

    The stall exceptions reach the worker fatal boundary, so an impossible
    QUEUEING request is reported on the per-request error channel and dropped.
    """
    pool = FakeHostPool(NEED - 1)
    coordinator = FakePrefixCoordinator(pool)
    book = FakeTrajectoryBook(["big", "ok"])
    small = _need(PAGE_TOKENS)
    assert small <= pool.total < NEED
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, slot_limit=8, trajectory_book=book,
        sequences=[_seq("big", 0, PROMPT), _seq("ok", 1, PAGE_TOKENS)],
    )

    assert worker._prepare_prefill_batch() == []

    # Only the impossible one went, by name, with its reservation released.
    assert list(worker.global_batch.sequences) == ["ok"]
    assert book.released == [book.slots["big"]]
    assert worker.published_capacity == 1
    (rejection,) = worker._response_queue.items
    assert rejection["request_id"] == "big"
    assert rejection["error"]["code"] == "host_kv_capacity_exceeded"
    assert str(NEED) in rejection["error"]["message"]
    assert coordinator.calls == []
    # No candidate set left to spin on, so the guard streak resets.
    assert worker._prefill_admission_guard_state() is None
    # The survivor is admitted on the next wave.
    assert worker._prepare_prefill_batch() == ["ok"]


def test_an_evicted_sequence_larger_than_the_whole_pool_fails_by_name():
    """A re-entrant sequence that no longer fits is an engine bug, not a request."""
    pool = FakeHostPool(NEED - 1)
    coordinator = FakePrefixCoordinator(pool)
    evicted = _seq("re", 0, PROMPT, status=SequenceStatus.EVICTED)
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, slot_limit=8, sequences=[evicted],
    )
    worker.enable_host_kv_eviction = True

    with pytest.raises(PrefillAdmissionStalled) as excinfo:
        worker._prepare_prefill_batch()

    assert "fits the total Host KV pool" in str(excinfo.value)
    assert "(can never fit)" in str(excinfo.value)
    assert coordinator.calls == []
    assert worker._response_queue.items == []
    assert "re" in worker.global_batch.sequences


def test_full_sequence_slots_reclaim_nothing():
    """Host pages are not the binding limit when every KDA slot is taken."""
    pool, coordinator, committer, allocator, _ = _full_pool_with_frontiers(
        total=400
    )
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, committer=committer, slot_limit=0,
        sequences=[
            _seq("q0", 0, PROMPT),
            _seq("live", 1, PROMPT, status=SequenceStatus.IN_DECODE),
        ],
    )
    assert worker._prepare_prefill_batch() == []
    assert coordinator.calls == []
    assert allocator.releases == []


def test_reclaim_targets_only_slot_admissible_candidates():
    pool, coordinator, committer, _allocator, _ = _full_pool_with_frontiers(
        total=400
    )
    free = pool.get_stats().num_free_pages
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, committer=committer, slot_limit=2,
        sequences=[_seq(f"q{i}", i, PROMPT) for i in range(6)],
    )
    batch = worker._prepare_prefill_batch()
    # Minimal progress: the target covers the head of the priority order, not
    # the whole slot-admissible set (which would be 2 * NEED), and never the
    # six candidates the slots exclude anyway.
    assert coordinator.calls[0][0] == {0: NEED - free}
    assert batch[:1] == ["q0"]


@pytest.mark.parametrize("slot_limit", [None, 4])
def test_reclaim_is_bounded_to_the_head_of_the_priority_order(slot_limit):
    """The target admits the first candidate, not the sum over all of them."""
    pool, coordinator, committer, _allocator, _ = _full_pool_with_frontiers(
        total=400
    )
    free = pool.get_stats().num_free_pages
    retained_before = sum(
        len(node["pages"]) for node in coordinator.nodes.values()
    )
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, committer=committer,
        slot_limit=slot_limit,
        sequences=[_seq(f"q{i}", i, PROMPT) for i in range(3)],
    )

    batch = worker._prepare_prefill_batch()

    assert coordinator.calls[0][0] == {0: NEED - free}
    assert batch, "the reclaim has to admit at least the head candidate"
    retained_after = sum(
        len(node["pages"]) for node in coordinator.nodes.values()
    )
    evicted_prefix_pages = retained_before - retained_after
    # Bounded by what ONE candidate needs (plus the LRU node granularity the
    # coordinator evicts in), nowhere near the 3 * NEED the old sum targeted.
    assert 0 < evicted_prefix_pages < 3 * NEED - free


def test_pool_mode_admits_the_first_candidate_without_evicting_for_the_set():
    """Partial admission must not evict for sequences already in the wave.

    The pool has room for exactly one candidate of three. The old target was
    the SUM over the whole slot-admissible set, so it evicted retained
    prefixes worth two more candidates before a wave that could only take
    one; the minimal-progress target is the next UNADMITTED candidate, which
    the free pages already cover, so nothing is evicted at all.
    """
    pool = FakeHostPool(400)
    coordinator = FakePrefixCoordinator(pool)
    retained = 80
    for node in range(retained):
        coordinator.retain(node, pages=pool.take(4), last_access=node)
    # Non-prefix filler so exactly one candidate's reservation is free.
    pool.take(pool.get_stats().num_free_pages - NEED)
    assert pool.get_stats().num_free_pages == NEED
    retained_pages_before = sum(
        len(node["pages"]) for node in coordinator.nodes.values()
    )
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, slot_limit=3,
        sequences=[_seq(f"q{i}", i, PROMPT) for i in range(3)],
    )

    assert worker._prepare_prefill_batch() == ["q0"]

    # The reclaim ran (the trigger fired on 1 < 3) with a zero page target, so
    # the coordinator was never asked to evict anything.
    assert coordinator.calls == []
    assert coordinator.evicted_nodes_total == 0
    assert len(coordinator.nodes) == retained
    assert sum(
        len(node["pages"]) for node in coordinator.nodes.values()
    ) == retained_pages_before


def test_reclaim_target_is_pure_and_skips_what_can_never_fit():
    assert PREFILL_RECLAIM_TARGET_CANDIDATES == 1
    # One candidate's worth, and only the shortfall against free pages.
    assert prefill_reclaim_target_pages(
        fitting_requirements=[40, 7, 9], free_pages=10
    ) == (40, 30)
    assert prefill_reclaim_target_pages(
        fitting_requirements=[40, 7], free_pages=100
    ) == (40, 0)
    # A configurable small batch of head candidates.
    assert prefill_reclaim_target_pages(
        fitting_requirements=[40, 7, 9], free_pages=10, max_candidates=2
    ) == (47, 37)
    # Nothing fits the pool at all: no target, so no eviction.
    assert prefill_reclaim_target_pages(
        fitting_requirements=[], free_pages=10
    ) == (0, 0)
    with pytest.raises(ValueError, match="max_candidates"):
        prefill_reclaim_target_pages(
            fitting_requirements=[4], free_pages=0, max_candidates=0
        )


def test_reclaim_without_a_committer_releases_prefix_pages_only():
    pool = FakeHostPool(400)
    coordinator = FakePrefixCoordinator(pool)
    for node in range(90):
        coordinator.retain(node, pages=pool.take(4), last_access=node)
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, slot_limit=8,
        sequences=[_seq("q0", 0, PROMPT)],
    )
    assert worker._prepare_prefill_batch() == ["q0"]
    assert coordinator.evicted_nodes_total > 0


# --------------------------------------------------------------------------- #
#  Untouched when the pool has room
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slot_limit", [None, 8])
def test_room_in_the_pool_selects_without_any_reclaim_or_extra_collective(
    slot_limit,
):
    pool = FakeHostPool(10 * NEED)
    coordinator = FakePrefixCoordinator(pool)
    committer, allocator = _committer(pool)
    pages = pool.take(4)
    outcome = _commit_frontier(committer, pages=pages, seed=1, sequence_id=0)
    coordinator.retain(0, pages=pages, last_access=0)
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, committer=committer,
        slot_limit=slot_limit, sequences=[_seq(f"q{i}", i, PROMPT) for i in range(3)],
    )

    assert worker._prepare_prefill_batch() == ["q0", "q1", "q2"]

    assert worker.dist.calls == ["all_gather"]
    assert coordinator.calls == []
    assert allocator.releases == []
    assert committer.registry.get(outcome.session_key) is not None


def _reference_select(req):
    """The pre-change selector, verbatim in behavior: a greedy priority walk."""
    evicted = sorted(
        (c for c in req.candidates if c.is_evicted),
        key=lambda c: (-c.total_decoded_before_eviction, c.global_idx),
    )
    queueing = sorted(
        (c for c in req.candidates if not c.is_evicted),
        key=lambda c: (
            c.global_idx if c.prefill_order is None else c.prefill_order,
            c.global_idx,
        ),
    )
    used = [0] * req.num_nodes
    node_seqs = [0] * req.num_nodes
    rank_seqs: dict[int, int] = {}
    out = []
    for c in evicted + queueing:
        if (req.per_rank_sequence_free is not None
                and rank_seqs.get(c.assigned_rank, 0)
                >= req.per_rank_sequence_free[c.assigned_rank]):
            continue
        if (req.per_node_sequence_free is not None
                and node_seqs[c.node_id] >= req.per_node_sequence_free[c.node_id]):
            continue
        need = PrefillScheduler.required_host_pages(c, req)
        if used[c.node_id] + need <= req.per_node_host_free[c.node_id]:
            out.append(c.uuid)
            used[c.node_id] += need
            rank_seqs[c.assigned_rank] = rank_seqs.get(c.assigned_rank, 0) + 1
            node_seqs[c.node_id] += 1
    return out


def _random_request(rng):
    nodes = rng.randint(1, 2)
    ranks = nodes * 2
    candidates = tuple(
        PrefillCandidate(
            uuid=f"c{i}", assigned_rank=(rank := rng.randrange(ranks)),
            node_id=rank // 2, is_evicted=rng.random() < 0.3, global_idx=i,
            total_decoded_before_eviction=rng.randrange(5),
            prompt_length=rng.randrange(1, 3000), kv_token_budget=4096,
            page_size=PAGE_TOKENS,
            prefill_order=rng.choice([None, rng.randrange(20)]),
        )
        for i in range(rng.randint(1, 12))
    )
    scope = rng.choice(["none", "rank", "node"])
    return PrefillSelectionRequest(
        candidates=candidates,
        per_node_host_free=tuple(rng.randrange(0, 400) for _ in range(nodes)),
        chunk_size=rng.choice([0, 512]), num_nodes=nodes, gpus_per_node=2,
        initial_gpu_page_buffer=32,
        per_rank_sequence_free=(
            tuple(rng.randrange(0, 4) for _ in range(ranks))
            if scope == "rank" else None
        ),
        per_node_sequence_free=(
            tuple(rng.randrange(0, 4) for _ in range(nodes))
            if scope == "node" else None
        ),
    )


def test_select_prefill_batch_is_unchanged_by_the_shared_walk():
    rng = random.Random(20261017)
    for _ in range(2000):
        req = _random_request(rng)
        assert PrefillScheduler.select_prefill_batch(req) == _reference_select(req)


def test_slot_admissible_is_the_selection_with_unbounded_host_pages():
    rng = random.Random(7)
    for _ in range(500):
        req = _random_request(rng)
        unbounded = dataclasses.replace(
            req, per_node_host_free=(1 << 40,) * req.num_nodes
        )
        assert [
            c.uuid for c in PrefillScheduler.slot_admissible_candidates(req)
        ] == PrefillScheduler.select_prefill_batch(unbounded)
        if req.per_rank_sequence_free is None and req.per_node_sequence_free is None:
            assert len(PrefillScheduler.slot_admissible_candidates(req)) == len(
                req.candidates
            )


# --------------------------------------------------------------------------- #
#  The scheduler loop's hard guard
# --------------------------------------------------------------------------- #


def test_progress_guard_trips_after_identical_dead_iterations():
    guard = PrefillProgressGuard(limit=3)
    state = ((5,), None, (8,), 1)
    guard.observe(admitted=0, live=False, state=state)
    guard.observe(admitted=0, live=False, state=state)
    with pytest.raises(PrefillAdmissionNoProgress) as excinfo:
        guard.observe(
            admitted=0, live=False, state=state, describe=lambda: "seq q0: x"
        )
    assert "3 consecutive" in str(excinfo.value)
    assert "seq q0: x" in str(excinfo.value)
    assert isinstance(excinfo.value, PrefillAdmissionStalled)


@pytest.mark.parametrize(
    "progress",
    [
        {"admitted": 1, "live": False, "state": ((5,), None, (8,), 1)},
        {"admitted": 0, "live": True, "state": ((5,), None, (8,), 1)},
        {"admitted": 0, "live": False, "state": ((6,), None, (8,), 1)},
        {"admitted": 0, "live": False, "state": ((5,), None, (8,), 2)},
        {"admitted": 0, "live": False, "state": None},
    ],
)
def test_any_progress_resets_the_guard(progress):
    guard = PrefillProgressGuard(limit=3)
    state = ((5,), None, (8,), 1)
    guard.observe(admitted=0, live=False, state=state)
    guard.observe(admitted=0, live=False, state=state)
    guard.observe(**progress)
    assert guard.streak <= 1
    guard.observe(admitted=0, live=False, state=state)
    guard.observe(admitted=0, live=False, state=state)
    assert guard.streak == 2


def test_the_default_guard_bound_is_finite_and_wired_into_the_loop():
    assert 0 < PREFILL_NO_PROGRESS_LIMIT <= 1024
    source = WORKER.read_text()
    loop = source[source.index("prefill_uuids = self._prepare_prefill_batch()"):]
    reserve = loop.index("self._reserve_prefill_host_kv(prefill_uuids)")
    observe = loop.index("self._prefill_progress_guard.observe(")
    config = loop.index("self._config_prefill_for_batch(prefill_uuids)")
    assert reserve < observe < config


def _observe(worker, guard, batch):
    """One scheduler-loop guard observation, exactly as the loop wires it."""
    guard.observe(
        admitted=len(batch),
        live=worker._prefill_admission_live,
        state=worker._prefill_admission_guard_state(),
        describe=worker._describe_prefill_admission_blockers,
    )


def test_guard_trips_on_a_repeated_slot_starved_dead_wave():
    """Slots all taken and nothing live: the worker loop's guard catches it."""
    pool = FakeHostPool(10 * NEED)
    worker = FakeWorker(
        pool=pool, coordinator=FakePrefixCoordinator(pool), slot_limit=0,
        sequences=[_seq("q0", 0, PROMPT)],
    )
    guard = PrefillProgressGuard(limit=5)
    with pytest.raises(PrefillAdmissionNoProgress) as excinfo:
        for _ in range(5):
            worker.dist.calls.clear()
            _observe(worker, guard, worker._prepare_prefill_batch())
    assert "no free sequence slot on node 0 (free=0)" in str(excinfo.value)


def test_a_changed_candidate_set_resets_the_streak_at_equal_free_pages():
    """A count-only state cannot see one candidate swapped for another.

    Free pages, pool total and slot counts are all identical across the two
    waves, so only the candidate IDENTITIES distinguish them.
    """
    pool = FakeHostPool(10 * NEED)
    worker = FakeWorker(
        pool=pool, coordinator=FakePrefixCoordinator(pool), slot_limit=0,
        sequences=[_seq("q0", 0, PROMPT)],
    )
    guard = PrefillProgressGuard(limit=3)
    for _ in range(2):
        worker.dist.calls.clear()
        _observe(worker, guard, worker._prepare_prefill_batch())
    assert guard.streak == 2
    first = worker._prefill_admission_guard_state()

    # Same count, same prompt length, same free pages -- different request.
    worker.global_batch.sequences = {"q1": _seq("q1", 1, PROMPT)}
    worker.dist.calls.clear()
    _observe(worker, guard, worker._prepare_prefill_batch())

    second = worker._prefill_admission_guard_state()
    assert guard.streak == 1
    assert first[0] == second[0], "free pages were supposed to be unchanged"
    assert first != second


def test_a_released_attachment_resets_the_streak_at_equal_free_pages():
    """Pages become RECLAIMABLE without the free count moving.

    An attachment release leaves the prefix resident -- free pages are
    unchanged -- but the next reclaim can now evict it, so the wave was
    waiting on something, not stuck.
    """
    pool = FakeHostPool(10 * NEED)
    coordinator = FakePrefixCoordinator(pool)
    coordinator.retain(0, pages=pool.take(4), last_access=0, active_refs=1)
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, slot_limit=0,
        sequences=[_seq("q0", 0, PROMPT)],
    )
    guard = PrefillProgressGuard(limit=4)
    for _ in range(2):
        worker.dist.calls.clear()
        _observe(worker, guard, worker._prepare_prefill_batch())
    assert guard.streak == 2
    before = worker._prefill_admission_guard_state()
    free_before = pool.get_stats().num_free_pages

    coordinator.nodes[0]["active_refs"] = 0

    worker.dist.calls.clear()
    _observe(worker, guard, worker._prepare_prefill_batch())

    assert guard.streak == 1
    assert pool.get_stats().num_free_pages == free_before
    after = worker._prefill_admission_guard_state()
    assert before[0] == after[0] and before[1] == after[1]
    assert before[5] != after[5], "prefix pressure is the only changed input"


def test_prefix_pressure_is_not_read_while_the_node_has_live_work():
    """The counters scan the whole shared node table; the hot path skips it."""
    pool = FakeHostPool(10 * NEED)
    coordinator = FakePrefixCoordinator(pool)
    coordinator.retain(0, pages=pool.take(4), last_access=0)
    worker = FakeWorker(
        pool=pool, coordinator=coordinator, slot_limit=0,
        sequences=[
            _seq("q0", 0, PROMPT),
            _seq("live", 1, PROMPT, status=SequenceStatus.IN_DECODE),
        ],
    )

    worker._prepare_prefill_batch()
    assert worker._prefill_admission_live is True
    assert worker._prefill_admission_state[5] == ((-1, -1, -1),)

    worker.global_batch.remove_sequence("live")
    worker.dist.calls.clear()
    worker._prepare_prefill_batch()
    assert worker._prefill_admission_live is False
    assert worker._prefill_admission_state[5] == ((1, 0, 0),)


def test_a_changed_reservation_outcome_resets_the_streak():
    """A wave the capacity transaction rejected is a different stall cause."""
    pool = FakeHostPool(10 * NEED)
    worker = FakeWorker(
        pool=pool, coordinator=FakePrefixCoordinator(pool), slot_limit=0,
        sequences=[_seq("q0", 0, PROMPT)],
    )
    guard = PrefillProgressGuard(limit=4)
    for _ in range(2):
        worker.dist.calls.clear()
        _observe(worker, guard, worker._prepare_prefill_batch())
    assert guard.streak == 2

    # _reserve_prefill_host_kv records this when the fresh snapshot rejects.
    worker.dist.calls.clear()
    batch = worker._prepare_prefill_batch()
    worker._prefill_reservation_outcome = "rejected_by_capacity_snapshot"
    _observe(worker, guard, batch)
    assert guard.streak == 1

    # ... and a rejection that keeps repeating with nothing else changing is
    # deterministic, not contention, so the bound still applies.
    worker.dist.calls.clear()
    batch = worker._prepare_prefill_batch()
    worker._prefill_reservation_outcome = "rejected_by_capacity_snapshot"
    _observe(worker, guard, batch)
    assert guard.streak == 2
    assert worker._prefill_admission_guard_state()[-1] == (
        "rejected_by_capacity_snapshot"
    )


def test_prepare_resets_the_reservation_outcome_each_iteration():
    pool = FakeHostPool(10 * NEED)
    worker = FakeWorker(
        pool=pool, coordinator=FakePrefixCoordinator(pool), slot_limit=0,
        sequences=[_seq("q0", 0, PROMPT)],
    )
    worker._prefill_reservation_outcome = "committed"
    worker._prepare_prefill_batch()
    assert worker._prefill_reservation_outcome is None


def _dead_wave_peer(*, free, total, live, slot_free):
    """Rank 0's report in the admission gather: leader of node 0."""
    return [0, free, 2, slot_free, total, int(live), None, 0, 0, 0]


@pytest.mark.parametrize("peer_live", [True, False])
def test_the_guard_verdict_is_the_live_bit_reduced_across_ranks(peer_live):
    """A rank-local `live` read makes ranks raise on different iterations.

    Rank 1 sees nothing live in its own global_batch. Whether the guard may
    raise has to follow the OR across ranks, so with rank 0 reporting a live
    sequence rank 1 must keep waiting, and only with both ranks dead may it
    fail -- the same iteration rank 0 fails on.
    """
    pool = FakeHostPool(10 * NEED)
    free = pool.get_stats().num_free_pages
    peers = {0: {0: _dead_wave_peer(
        free=free, total=pool.total, live=peer_live, slot_free=0
    )}}
    seq = _seq("q0", 0, PROMPT)
    seq.assigned_rank = 1
    worker = FakeWorker(
        pool=pool, coordinator=FakePrefixCoordinator(pool), slot_limit=0,
        sequences=[seq], rank=1, local_rank=1, world_size=2,
        local_world_size=2, dist=FakeDist(rank=1, world_size=2, peers=peers),
    )
    assert not worker.global_batch.has_in_decode()
    guard = PrefillProgressGuard(limit=2)

    def run_two_waves():
        for _ in range(2):
            worker.dist.calls.clear()
            _observe(worker, guard, worker._prepare_prefill_batch())

    if peer_live:
        run_two_waves()
        assert worker._prefill_admission_live is True
        assert guard.streak == 0
    else:
        with pytest.raises(PrefillAdmissionNoProgress):
            run_two_waves()
        assert worker._prefill_admission_live is False


def test_a_divergent_candidate_digest_raises_by_name():
    """Two ranks with different candidate sets must not split the branch."""
    pool = FakeHostPool(10 * NEED)
    peer = _dead_wave_peer(free=-1, total=-1, live=False, slot_free=8)
    peer[6] = 0xDEAD  # a digest this rank cannot have produced
    worker = FakeWorker(
        pool=pool, coordinator=FakePrefixCoordinator(pool), slot_limit=8,
        sequences=[_seq("q0", 0, PROMPT)], rank=0, local_rank=0, world_size=2,
        local_world_size=2,
        dist=FakeDist(rank=0, world_size=2, peers={0: {1: peer}}),
    )

    with pytest.raises(PrefillAdmissionStateDivergence) as excinfo:
        worker._prepare_prefill_batch()

    assert "diverged across ranks" in str(excinfo.value)
    assert "1 candidates" in str(excinfo.value)
    # Raised on the gather, before any reclaim or second collective.
    assert worker.dist.calls == ["all_gather"]


def test_the_candidate_digest_covers_identity_and_the_reservation_inputs():
    pool = FakeHostPool(10 * NEED)

    def digest(sequences):
        worker = FakeWorker(
            pool=pool, coordinator=FakePrefixCoordinator(pool),
            sequences=sequences,
        )
        return worker._prefill_candidate_digest(
            [seq.uuid for seq in sequences]
        )

    base = digest([_seq("q0", 0, PROMPT), _seq("q1", 1, PROMPT)])
    assert base == digest([_seq("q0", 0, PROMPT), _seq("q1", 1, PROMPT)])
    # Order, identity, and every field the selection request reads.
    assert base != digest([_seq("q1", 1, PROMPT), _seq("q0", 0, PROMPT)])
    assert base != digest([_seq("q0", 0, PROMPT), _seq("q2", 1, PROMPT)])
    assert base != digest([_seq("q0", 0, PROMPT), _seq("q1", 1, PROMPT + 1)])
    for field, value in (
        ("assigned_rank", 1),
        ("decode_dp_group", 0),
        ("prefill_order", 3),
        ("kv_token_budget", 4096),
        ("status", SequenceStatus.EVICTED),
    ):
        changed = [_seq("q0", 0, PROMPT), _seq("q1", 1, PROMPT)]
        setattr(changed[1], field, value)
        assert base != digest(changed), field


# --------------------------------------------------------------------------- #
#  Frontier retirement on every rank of the reclaiming node
# --------------------------------------------------------------------------- #


def test_recorder_records_and_forwards_once():
    seen = []
    recorder = EvictedPageRecorder(seen.append)
    recorder({0: [3, 4]})
    recorder({0: [5], 1: [9]})
    assert recorder.pages_by_group == {0: [3, 4, 5], 1: [9]}
    assert recorder.page_count == 4
    assert seen == [{0: [3, 4]}, {0: [5], 1: [9]}]
    assert EvictedPageRecorder(None)({0: [1]}) is None


def test_retire_node_frontiers_uses_only_its_own_nodes_leader_map():
    seen = []
    gathered = [(0, {0: [1, 2]}), (0, None), (1, {0: [7]}), (1, None)]
    assert retire_node_frontiers(
        gathered=gathered, node_id=1, frontier_hook=seen.append
    ) == 1
    assert seen == [{0: [7]}]
    assert retire_node_frontiers(
        gathered=gathered, node_id=0, frontier_hook=None
    ) == 0
    with pytest.raises(RuntimeError, match="exactly one rank per node"):
        retire_node_frontiers(
            gathered=[(0, {0: [1]}), (0, {0: [2]})], node_id=0,
            frontier_hook=seen.append,
        )


def test_a_non_leader_rank_retires_its_frontiers_on_the_leaders_reclaim():
    """G=1: rank 1 owns a frontier on pages the node leader (rank 0) reclaims."""
    pool = FakeHostPool(400)
    committer, allocator = _committer(pool, rank=1)
    victim_pages = pool.take(4)
    keeper_pages = pool.take(4)
    victim = _commit_frontier(committer, pages=victim_pages, seed=1, sequence_id=0)
    keeper = _commit_frontier(committer, pages=keeper_pages, seed=2, sequence_id=1)
    free = pool.get_stats().num_free_pages
    assert free >= NEED
    peers = {
        # 0: free/slot/total/live/digest/pressure gather -- the leader reports a
        # node pool short of NEED but large enough to hold it, no live
        # sequence, and (None) the same candidate digest as this rank.
        0: {0: [0, NEED - 1, 2, 8, 400, 0, None, 0, 0, 0]},
        # 1: failure/reclaimed-pages MAX -- the leader reclaimed victim_pages.
        1: {0: [0, len(victim_pages)]},
        # 2: the leader's reclaimed map.
        2: {0: (0, {0: list(victim_pages)})},
        # 3: updated free -- the leader's post-reclaim count.
        3: {0: [0, NEED + 3, 0]},
    }
    dist = FakeDist(rank=1, world_size=2, peers=peers)
    seq = _seq("q0", 0, PROMPT)
    seq.assigned_rank = 1
    worker = FakeWorker(
        pool=pool, coordinator=FakePrefixCoordinator(pool), committer=committer,
        slot_limit=8, sequences=[seq], rank=1, local_rank=1, world_size=2,
        local_world_size=2, dist=dist,
    )
    assert worker._prepare_prefill_batch() == ["q0"]

    assert dist.calls == [
        "all_gather", "all_reduce", "all_gather_object", "all_gather",
    ]
    assert committer.registry.get(victim.session_key) is None
    assert committer.registry.get(keeper.session_key) is not None
    assert allocator.releases == [victim.arena.page_ids_by_group[0]]
    committer.registry.check_invariants()
