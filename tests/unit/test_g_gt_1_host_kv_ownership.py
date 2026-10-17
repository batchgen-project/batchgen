"""G>1 host-KV ownership: one leader allocates, its serve group mirrors.

The per-node host paged KV region is shared, so a sequence replicated onto all
G ranks of its serve group must be registered, allocated, grown and released
exactly once — by the group leader (``_owns_host_kv``). The other G-1 ranks
reproduce the leader's page table locally through ``_mirror_group_host_kv``
instead, and drop only that local copy when the row goes away. These tests pin
that split without importing the batchgen package: the worker methods are
compiled straight out of the module's AST onto a namespace of fakes, so nothing
here needs CUDA, NCCL or the compiled core engine.
"""

import ast
import copy
import dataclasses
import logging
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import pytest


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"

PAGE_TOKENS = 64


def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")


def _worker_method_node(name):
    return next(node for node in _worker_class().body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _worker_methods(names, globals_=None):
    """Compile the named worker methods onto one shared namespace of fakes."""
    wanted = [copy.deepcopy(_worker_method_node(name)) for name in names]
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {"List": List, **(globals_ or {})}
    exec(compile(module, str(WORKER), "exec"), namespace)
    return {name: namespace[name] for name in names}


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FakePrefixState:
    """The real ``PrefixCacheSequenceState``'s ``lookup_result=None`` branch.

    A mirrored state has no coordinator lookup result — only the leader holds
    the attachment — so its shared pages come from the planned chain.
    """

    lookup_result: object
    attached_tokens: int
    compute_cached_tokens: int
    planned_shared_page_ids: dict = dataclasses.field(default_factory=dict)
    commit_attachment_handles: tuple = ()

    @property
    def attachment_handle(self):
        return int(getattr(self.lookup_result, "attachment_handle", 0))

    def shared_page_ids_for_group(self, group_id):
        if self.attached_tokens == 0:
            return ()
        return tuple(self.planned_shared_page_ids.get(int(group_id), ()))


class FakeBackend:
    """One host-KV region of one KV group, shared by every rank of the node."""

    def __init__(self, first_page):
        self.pages = {}          # global_idx -> committed page ids
        self.registered = []     # (rank, [global_idx]) per register_sequences
        self.allocated = []      # (rank, [(global_idx, tokens)]) per allocate
        self.grown = []          # (rank, [(global_idx, pages)]) per grow
        self.released = []       # (rank, [global_idx]) per release_sequence_pages
        self.unregistered = []   # (rank, [global_idx]) per unregister_sequences
        self.next_page = first_page


class FakeView:
    """A worker view: a LOCAL page table over a shared backend region."""

    def __init__(self, backend, rank):
        self.backend = backend
        self.rank = rank
        self.table = {}

    def register_sequences(self, global_ids):
        self.backend.registered.append((self.rank, list(global_ids)))
        for global_id in global_ids:
            self.table[global_id] = list(self.backend.pages.get(global_id, ()))

    def attach_shared_prefix_pages(self, global_id, page_ids):
        self.table[global_id] = list(page_ids) + self.table.get(global_id, [])

    def allocate_pages_for_sequences(self, seq_token_pairs):
        pairs = [(int(g), int(t)) for g, t in seq_token_pairs]
        self.backend.allocated.append((self.rank, pairs))
        for global_id, tokens in pairs:
            new = [
                self.backend.next_page + offset
                for offset in range(tokens // PAGE_TOKENS)
            ]
            self.backend.next_page += len(new)
            self.backend.pages.setdefault(global_id, []).extend(new)
            self.table.setdefault(global_id, []).extend(new)

    def grow_pages_for_sequences(self, seq_page_pairs):
        pairs = [(int(g), int(p)) for g, p in seq_page_pairs]
        self.backend.grown.append((self.rank, pairs))
        for global_id, pages in pairs:
            new = [self.backend.next_page + offset for offset in range(pages)]
            self.backend.next_page += pages
            self.backend.pages.setdefault(global_id, []).extend(new)
            self.table.setdefault(global_id, []).extend(new)

    def release_sequence_pages(self, global_ids):
        # The real API frees the shared pages AND unregisters the row.
        self.backend.released.append((self.rank, list(global_ids)))
        for global_id in global_ids:
            self.backend.pages.pop(global_id, None)
            self.table.pop(global_id, None)

    def unregister_sequences(self, global_ids):
        # Local-only: drops this view's page-table record, not the pages.
        self.backend.unregistered.append((self.rank, list(global_ids)))
        for global_id in global_ids:
            self.table.pop(global_id, None)

    def build_page_table(self, global_ids):
        return [list(self.table.get(global_id, ())) for global_id in global_ids]

    def set_sequence_pages(self, global_id, page_ids):
        # Mirrors HostKVPageTable::RegisterOrUpdate: a full overwrite that
        # creates the record when absent, and never touches the backend.
        self.table[global_id] = list(page_ids)


class FakeDist:
    """Ranks stepped sequentially in one process share the gather slots."""

    def __init__(self, world_size):
        self.slots = [None] * world_size
        self.rank = 0
        self.calls = 0

    def all_gather_object(self, out, payload):
        self.calls += 1
        self.slots[self.rank] = payload
        for index, value in enumerate(self.slots):
            out[index] = value


def _stub_prefix_modules(monkeypatch):
    """Serve the two ``batchgen`` imports the worker methods do lazily."""
    monkeypatch.setitem(
        sys.modules, "batchgen.decode_dp_group",
        SimpleNamespace(
            host_kv_owner_rank=lambda group, size: int(group) * size,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "batchgen.prefix_reuse.prefill",
        SimpleNamespace(PrefixCacheSequenceState=FakePrefixState),
    )


def _rank(rank, *, attn_tp_size, world_size, seqs, views, fake_dist,
          local_world_size=8):
    """A fake worker holding only what the mirror path reads."""
    methods = _worker_methods(
        ["_mirror_group_host_kv", "_owns_host_kv", "_decode_attn_tp_size"],
        {"dist": fake_dist},
    )
    worker = SimpleNamespace(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        parallel_manager=SimpleNamespace(attn_tp_size=attn_tp_size),
        global_batch=SimpleNamespace(get_sequence=seqs.get),
        _prefix_sequence_states={},
        _host_views_by_group=lambda: views,
    )
    for name, function in methods.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker


def _two_group_views(backends, rank):
    return {group: FakeView(backend, rank) for group, backend in backends.items()}


def _seqs(group):
    """Two sequences bound to the same serve group, so one leader owns both."""
    return {
        "a": SimpleNamespace(uuid="a", global_idx=100, assigned_rank=0,
                             decode_dp_group=group),
        "b": SimpleNamespace(uuid="b", global_idx=101, assigned_rank=1,
                             decode_dp_group=group),
    }


def _leader_states():
    """``a`` reuses a cached prefix page per KV group; ``b`` shares nothing."""
    return {
        100: FakePrefixState(
            lookup_result=object(),
            attached_tokens=PAGE_TOKENS,
            compute_cached_tokens=PAGE_TOKENS,
            planned_shared_page_ids={0: (900,), 1: (901,)},
        ),
        101: FakePrefixState(
            lookup_result=object(), attached_tokens=0, compute_cached_tokens=0,
        ),
    }


def _run_leader_host_kv_section(worker, uuids, states):
    """The leader-gated body of _config_prefill_for_batch STEP 4, in miniature."""
    global_ids = [worker.global_batch.get_sequence(u).global_idx for u in uuids]
    for group, view in worker._host_views_by_group().items():
        view.register_sequences(global_ids)
        for global_id in global_ids:
            shared = states[global_id].shared_page_ids_for_group(group)
            if shared:
                view.attach_shared_prefix_pages(global_id, list(shared))
        view.allocate_pages_for_sequences(
            [(global_id, 2 * PAGE_TOKENS) for global_id in global_ids]
        )


# --------------------------------------------------------------------------- #
#  (i) G=2: one register/allocate per sequence, identical tables after mirror
# --------------------------------------------------------------------------- #


def test_g2_group_registers_once_and_mirrors_identical_page_tables(monkeypatch):
    _stub_prefix_modules(monkeypatch)
    uuids = ["a", "b"]
    backends = {0: FakeBackend(1000), 1: FakeBackend(2000)}
    fake_dist = FakeDist(world_size=2)
    views_by_rank = {rank: _two_group_views(backends, rank) for rank in (0, 1)}
    workers = {}

    for rank in (0, 1):
        fake_dist.rank = rank
        seqs = _seqs(group=0)
        worker = _rank(rank, attn_tp_size=2, world_size=2, seqs=seqs,
                       views=views_by_rank[rank], fake_dist=fake_dist)
        workers[rank] = worker
        owns_host_kv = all(
            worker._owns_host_kv(seqs[uuid]) for uuid in uuids
        )
        states = _leader_states() if owns_host_kv else {}
        if owns_host_kv:
            _run_leader_host_kv_section(worker, uuids, states)
        worker._mirror_group_host_kv(uuids, states)
        worker.states = states

    # Only rank 0, the group leader, touched the shared region — once per group.
    assert workers[0]._owns_host_kv(_seqs(0)["a"])
    assert not workers[1]._owns_host_kv(_seqs(0)["a"])
    for group, backend in backends.items():
        assert backend.registered == [(0, [100, 101])], group
        assert backend.allocated == [
            (0, [(100, 2 * PAGE_TOKENS), (101, 2 * PAGE_TOKENS)])
        ], group
        assert backend.pages[100] == [backend.next_page - 4, backend.next_page - 3]

    # The follower's local tables now equal the leader's, prefix page included.
    for group in backends:
        leader_table = views_by_rank[0][group].table
        assert views_by_rank[1][group].table == leader_table
    assert views_by_rank[0][0].table[100] == [900, 1000, 1001]
    assert views_by_rank[0][1].table[100] == [901, 2000, 2001]

    # Both ranks hold prefix state; the follower's carries no lookup result.
    assert set(workers[1]._prefix_sequence_states) == {100, 101}
    mirrored = workers[1]._prefix_sequence_states[100]
    assert mirrored.lookup_result is None
    assert mirrored.attached_tokens == PAGE_TOKENS
    assert mirrored.compute_cached_tokens == PAGE_TOKENS
    assert mirrored.shared_page_ids_for_group(0) == (900,)
    assert mirrored.shared_page_ids_for_group(1) == (901,)
    assert workers[1]._prefix_sequence_states[101].shared_page_ids_for_group(0) == ()
    # The mirror also feeds back into the caller's dict, which the follower's
    # rank-local metadata loop reads for its private page count.
    assert workers[1].states[100].attached_tokens == PAGE_TOKENS


def test_follower_without_a_leader_payload_is_named_not_silent(monkeypatch):
    # An empty leader payload would leave the follower with an empty page table
    # and a crash far downstream, so the mirror names it here.
    _stub_prefix_modules(monkeypatch)
    fake_dist = FakeDist(world_size=2)
    fake_dist.rank = 1
    seqs = _seqs(group=0)
    worker = _rank(1, attn_tp_size=2, world_size=2, seqs=seqs,
                   views=_two_group_views({0: FakeBackend(1000)}, 1),
                   fake_dist=fake_dist)
    with pytest.raises(RuntimeError, match="leader 0 mirrored no host KV"):
        worker._mirror_group_host_kv(["a", "b"], {})


# --------------------------------------------------------------------------- #
#  (ii) G=1: the validated pure-DP path runs no collective at all
# --------------------------------------------------------------------------- #


def test_g1_mirror_runs_no_collective_and_touches_nothing(monkeypatch):
    _stub_prefix_modules(monkeypatch)
    fake_dist = FakeDist(world_size=1)
    seqs = _seqs(group=None)
    views = _two_group_views({0: FakeBackend(1000)}, 0)
    worker = _rank(0, attn_tp_size=1, world_size=1, seqs=seqs, views=views,
                   fake_dist=fake_dist)
    states = _leader_states()

    worker._mirror_group_host_kv(["a", "b"], states)

    assert fake_dist.calls == 0
    assert views[0].table == {}
    assert worker._prefix_sequence_states == {}
    assert states[100].lookup_result is not None  # untouched


# --------------------------------------------------------------------------- #
#  (iii) The shared-region calls in STEP 4 sit under an _owns_host_kv gate
# --------------------------------------------------------------------------- #

GATED_CALLS = (
    "register_sequences",
    "allocate_pages_for_sequences",
    "lookup_prefix_cache_for_prefill",
    "evict_prefix_pages_for_host_allocation",
)


def _gate_names(method):
    """Locals bound from an expression that reads ``self._owns_host_kv``."""
    names = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and any(
            isinstance(inner, ast.Attribute) and inner.attr == "_owns_host_kv"
            for inner in ast.walk(node.value)
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _reads_gate(test, names):
    for node in ast.walk(test):
        if isinstance(node, ast.Attribute) and node.attr == "_owns_host_kv":
            return True
        if isinstance(node, ast.Name) and node.id in names:
            return True
    return False


def _called_names(node):
    called = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            func = inner.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
            elif isinstance(func, ast.Name):
                called.add(func.id)
    return called


def test_shared_host_region_calls_are_gated_on_owns_host_kv():
    # Host-region prefill calls live in the admission reservation now: the
    # reservation plan selects owners with _owns_host_kv, and the shared
    # register/allocate calls in _reserve_prefill_host_kv run only for
    # owner-derived requests. The coordinator lookup is leader-gated inside
    # the attach helper.
    plan = _worker_method_node("_prefill_host_reservation_plan")
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "_owns_host_kv"
        for node in ast.walk(plan)
    ), "_prefill_host_reservation_plan must select owners with _owns_host_kv"

    method = _worker_method_node("_reserve_prefill_host_kv")
    names = {"owner_requests", "registered"}
    parents = {}
    for node in ast.walk(method):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    reservation_calls = ("register_sequences", "allocate_pages_for_sequences")
    for call in [n for n in ast.walk(method) if isinstance(n, ast.Call)]:
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in reservation_calls:
            continue
        node = call
        gated = False
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If) and _reads_gate(node.test, names):
                gated = True
                break
        assert gated, f"{name} at line {call.lineno} is not under an owner gate"
    called = _called_names(method)
    assert set(reservation_calls) <= called

    attach = _worker_method_node("_attach_prefix_cache_for_prefill")
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "_owns_host_kv"
        for node in ast.walk(attach)
    )
    assert "lookup_prefix_cache_for_prefill" in _called_names(attach)


def test_mirror_is_reached_outside_the_leader_gate():
    """The collective must not sit in a branch only the leader enters."""
    method = _worker_method_node("_config_prefill_for_batch")
    names = _gate_names(method)
    parents = {}
    for node in ast.walk(method):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_mirror_group_host_kv"
    ]
    assert len(calls) == 1
    node = calls[0]
    while node in parents:
        node = parents[node]
        assert not (isinstance(node, ast.If) and _reads_gate(node.test, names)), (
            "_mirror_group_host_kv is called inside the _owns_host_kv gate"
        )

    # The mirror itself keeps the gather outside its own leader branch.
    mirror = _worker_method_node("_mirror_group_host_kv")
    mirror_parents = {}
    for parent in ast.walk(mirror):
        for child in ast.iter_child_nodes(parent):
            mirror_parents[child] = parent
    gather = next(
        n for n in ast.walk(mirror)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "all_gather_object"
    )
    node = gather
    while node in mirror_parents:
        node = mirror_parents[node]
        assert not (
            isinstance(node, ast.If)
            and _reads_gate(node.test, _gate_names(mirror))
        ), "the mirror's gather sits inside its leader-only branch"


# --------------------------------------------------------------------------- #
#  (iv) The node allocator sees one host reservation per sequence, not G
# --------------------------------------------------------------------------- #


def _selection_request(attn_tp_size, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "batchgen.sequence",
        SimpleNamespace(INITIAL_GPU_PAGE_BUFFER=2),
    )
    monkeypatch.setitem(
        sys.modules, "batchgen.decode_dp_group",
        SimpleNamespace(host_kv_owner_rank=lambda group, size: int(group) * size),
    )
    methods = _worker_methods(
        ["_make_prefill_selection_request", "_decode_attn_tp_size"],
        {
            "PrefillCandidate": lambda **kw: SimpleNamespace(**kw),
            "PrefillSelectionRequest": lambda **kw: SimpleNamespace(**kw),
            "SequenceStatus": SimpleNamespace(EVICTED="EVICTED"),
        },
    )
    seq = SimpleNamespace(
        uuid="a", assigned_rank=0, global_idx=100, status="QUEUEING",
        decode_dp_group=0, prompt_length=1024, kv_token_budget=2048,
        PAGE_SIZE=PAGE_TOKENS, total_decoded_before_eviction=0,
        prefill_order=None,
    )
    worker = SimpleNamespace(
        rank=0, local_world_size=8,
        parallel_manager=SimpleNamespace(attn_tp_size=attn_tp_size),
        global_batch=SimpleNamespace(get_sequence=lambda uuid: seq),
    )
    for name, function in methods.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker._make_prefill_selection_request(
        ["a"], [1024], num_nodes=1, chunk_size=PAGE_TOKENS,
    )


@pytest.mark.parametrize("attn_tp_size", [1, 8])
def test_host_reservation_is_never_replicated_across_the_serve_group(
    attn_tp_size, monkeypatch,
):
    # Only the group leader allocates in the shared region, so one admitted
    # sequence costs the node allocator one copy of its reservation at any G.
    request = _selection_request(attn_tp_size, monkeypatch)
    assert request.candidates[0].host_kv_replication_factor == 1


# --------------------------------------------------------------------------- #
#  (v) Page-boundary growth: the leader extends the shared chain, once
# --------------------------------------------------------------------------- #


def _seed_committed_page(backends, views_by_rank, ranks):
    """Give every rank the leader's page table for one page per sequence."""
    for group, backend in backends.items():
        for global_id in (100, 101):
            backend.pages[global_id] = [backend.next_page]
            for rank in ranks:
                views_by_rank[rank][group].set_sequence_pages(
                    global_id, list(backend.pages[global_id])
                )
            backend.next_page += 1


def _growth_seqs(group):
    """Two grouped sequences already holding one host page each."""
    return {
        uuid: SimpleNamespace(
            uuid=uuid, global_idx=global_idx, assigned_rank=rank,
            decode_dp_group=group, PAGE_SIZE=PAGE_TOKENS,
            host_pages_allocated=1, host_owned_pages=1,
            host_token_capacity=PAGE_TOKENS,
        )
        for uuid, global_idx, rank in (("a", 100, 0), ("b", 101, 1))
    }


def _run_growth_section(worker, uuids, growth_pages):
    """Phase C of _page_boundary_fast, in miniature.

    The metadata loop runs on every rank — the page counts come from the
    broadcast rank-0 plan — while the grow against the shared chain is the
    group leader's alone.
    """
    requests = []
    for uuid, pages in zip(uuids, growth_pages):
        seq = worker.global_batch.get_sequence(uuid)
        seq.host_token_capacity += pages * seq.PAGE_SIZE
        seq.host_pages_allocated += pages
        seq.host_owned_pages += pages
        if worker._owns_host_kv(seq):
            requests.append((seq.global_idx, pages))
    if requests:
        for view in worker._host_views_by_group().values():
            view.grow_pages_for_sequences(requests)


def test_g2_growth_extends_the_shared_chain_once_and_mirrors_it(monkeypatch):
    _stub_prefix_modules(monkeypatch)
    uuids = ["a", "b"]
    growth_pages = [2, 1]
    backends = {0: FakeBackend(1000), 1: FakeBackend(2000)}
    views_by_rank = {rank: _two_group_views(backends, rank) for rank in (0, 1)}
    _seed_committed_page(backends, views_by_rank, (0, 1))
    fake_dist = FakeDist(world_size=2)
    workers = {}
    seqs_by_rank = {}

    for rank in (0, 1):
        fake_dist.rank = rank
        seqs = _growth_seqs(group=0)
        seqs_by_rank[rank] = seqs
        worker = _rank(rank, attn_tp_size=2, world_size=2, seqs=seqs,
                       views=views_by_rank[rank], fake_dist=fake_dist)
        workers[rank] = worker
        _run_growth_section(worker, uuids, growth_pages)
        worker._mirror_group_host_kv(uuids, {})

    # The shared chain is extended exactly once per region, by the leader — the
    # plan charged three pages and three pages came out of the pool.
    for group, backend in backends.items():
        assert backend.grown == [(0, [(100, 2), (101, 1)])], group
        assert len(backend.pages[100]) == 3
        assert len(backend.pages[101]) == 2

    # Every rank appends through its OWN view, so the follower's table must now
    # carry the pages the leader just acquired.
    for group in backends:
        assert views_by_rank[1][group].table == views_by_rank[0][group].table
    assert views_by_rank[0][0].table[100] == [1000, 1002, 1003]
    assert views_by_rank[0][0].table[101] == [1001, 1004]

    # Scalar metadata stays identical on both ranks of the group.
    for uuid in uuids:
        leader_seq = seqs_by_rank[0][uuid]
        follower_seq = seqs_by_rank[1][uuid]
        assert follower_seq.host_pages_allocated == leader_seq.host_pages_allocated
        assert follower_seq.host_token_capacity == leader_seq.host_token_capacity
        assert leader_seq.host_token_capacity == (
            leader_seq.host_pages_allocated * PAGE_TOKENS
        )

    # Growth mirrors page tables only; the prefix attachment stays the leader's.
    assert workers[0]._prefix_sequence_states == {}
    assert workers[1]._prefix_sequence_states == {}
    assert fake_dist.calls == 2


# --------------------------------------------------------------------------- #
#  (vi) Release: the follower drops its mirror, the leader frees the pages
# --------------------------------------------------------------------------- #


class RecordingCoordinator:
    def __init__(self):
        self.released = []

    def release_attachment(self, handle):
        self.released.append(int(handle))


def _release_worker(rank, *, attn_tp_size, views, states, coordinator):
    """A fake worker holding only what the real release path reads."""
    methods = _worker_methods(
        [
            "_release_host_kv_pages_for_batch",
            "_release_prefix_cache_attachments",
            "_release_pool_segment_refs",
            "_owns_host_kv",
            "_decode_attn_tp_size",
        ],
        {
            "logging": logging,
            "Sequence": Sequence,
            "SequenceStatus": SimpleNamespace(IN_DECODE="IN_DECODE"),
            # The aux release is guarded on the dual coordinator type; these
            # single-coordinator fakes are never instances of it.
            "DualHostKVCoordinator": type("DualHostKVCoordinator", (), {}),
        },
    )
    seqs = _seqs(group=0)
    worker = SimpleNamespace(
        rank=rank,
        parallel_manager=SimpleNamespace(attn_tp_size=attn_tp_size),
        global_batch=SimpleNamespace(
            get_sequence=seqs.get,
            get_sequences_by_status=lambda status: [],
        ),
        _uuid_to_local_map={"a": 0, "b": 1},
        core_engine=SimpleNamespace(host_paged_kv_worker_view=views[0]),
        host_paged_kv_worker_view_aux=views[1],
        _prefix_integrity=False,
        enable_prefix_cache=True,
        _prefix_sequence_states=states,
        _pool_member_sids={},
        _pool_sid_members={},
        prefix_cache_coordinator=coordinator,
        gpu_paged_kv_cache_manager=None,
        # K3 semantics: ONE prefix group (primary); the indexer-key aux region
        # lives outside the groups and is handled by the explicit aux path.
        _host_views_by_group=lambda: {0: views[0]},
    )
    for name, function in methods.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker


def test_g2_follower_release_drops_its_mirror_without_freeing_shared_pages(
    monkeypatch,
):
    _stub_prefix_modules(monkeypatch)
    backends = {0: FakeBackend(1000), 1: FakeBackend(2000)}
    views_by_rank = {rank: _two_group_views(backends, rank) for rank in (0, 1)}
    _seed_committed_page(backends, views_by_rank, (0, 1))
    # The leader holds the coordinator attachment for ``a``; the follower's
    # mirrored state has lookup_result=None, hence no handle at all.
    states = {
        0: {
            100: FakePrefixState(
                lookup_result=SimpleNamespace(attachment_handle=7),
                attached_tokens=PAGE_TOKENS, compute_cached_tokens=PAGE_TOKENS,
            ),
            101: FakePrefixState(
                lookup_result=None, attached_tokens=0, compute_cached_tokens=0,
            ),
        },
        1: {
            global_id: FakePrefixState(
                lookup_result=None, attached_tokens=0, compute_cached_tokens=0,
            )
            for global_id in (100, 101)
        },
    }
    coordinators = {rank: RecordingCoordinator() for rank in (0, 1)}

    for rank in (0, 1):
        worker = _release_worker(
            rank, attn_tp_size=2, views=views_by_rank[rank],
            states=states[rank], coordinator=coordinators[rank],
        )
        worker._release_host_kv_pages_for_batch(["a", "b"])

    for group, backend in backends.items():
        # The shared pages are freed exactly once, by the group leader, and the
        # follower only unregisters its local record — in EVERY host region.
        assert backend.released == [(0, [100, 101])], group
        assert backend.unregistered == [(1, [100, 101])], group
        assert backend.pages == {}, group

    # Both ranks end with no local page table and no prefix state for the rows.
    for rank in (0, 1):
        for view in views_by_rank[rank].values():
            assert view.table == {}
        assert states[rank] == {}
    # Only the leader owns an attachment, so only the leader calls the
    # coordinator; a follower call would double-release the shared node.
    assert coordinators[0].released == [7]
    assert coordinators[1].released == []


# --------------------------------------------------------------------------- #
#  (vii) Append-time growth is off-plan at G>1, so it must fail loudly
# --------------------------------------------------------------------------- #


def _append_worker(attn_tp_size, view):
    methods = _worker_methods(
        ["_ensure_host_kv_append_capacity", "_decode_attn_tp_size"],
        {"logging": logging, "math": math},
    )
    seq = SimpleNamespace(
        global_idx=100, PAGE_SIZE=PAGE_TOKENS,
        host_pages_allocated=2, host_owned_pages=2,
        host_token_capacity=2 * PAGE_TOKENS,
        kv_token_budget=64 * PAGE_TOKENS,
        current_context_length=2 * PAGE_TOKENS, decoded_length=64,
        status=SimpleNamespace(name="IN_DECODE"),
    )
    worker = SimpleNamespace(
        rank=1,
        parallel_manager=SimpleNamespace(attn_tp_size=attn_tp_size),
        # The append path resolves the row through SequenceBatch's maintained
        # global_idx index, as the other fakes here expose get_sequence.
        global_batch=SimpleNamespace(
            get_sequence_by_global_idx={seq.global_idx: seq}.get,
        ),
        host_paged_kv_worker_view=view,
        _wait_pending_kv_append_tasks=lambda defer_errors: 0,
    )
    for name, function in methods.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker, seq


@pytest.mark.parametrize("attn_tp_size", [1, 2])
def test_append_capacity_raises_instead_of_growing_off_plan(attn_tp_size):
    # Host pages are allocated only by the rank-0 boundary planner: an append
    # that reaches here without a covering reservation is a scheduling bug at
    # EVERY G. At G>1 an off-plan grow would additionally desync the
    # followers' mirrored tables or double-extend the shared chain.
    backend = FakeBackend(1000)
    worker, seq = _append_worker(
        attn_tp_size=attn_tp_size,
        view=FakeView(backend, attn_tp_size - 1),
    )

    with pytest.raises(RuntimeError, match="without a boundary reservation"):
        worker._ensure_host_kv_append_capacity([100], [2 * PAGE_TOKENS])

    assert backend.grown == []
    assert seq.host_token_capacity == 2 * PAGE_TOKENS


def test_append_capacity_within_reservation_never_grows():
    backend = FakeBackend(1000)
    worker, seq = _append_worker(attn_tp_size=2, view=FakeView(backend, 1))

    worker._ensure_host_kv_append_capacity([100], [2 * PAGE_TOKENS - 1])

    assert backend.grown == []
    assert seq.host_token_capacity == 2 * PAGE_TOKENS


# --------------------------------------------------------------------------- #
#  (viii) The boundary grow sits under the leader gate, its mirror does not
# --------------------------------------------------------------------------- #


def _parents(method):
    parents = {}
    for node in ast.walk(method):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def test_boundary_host_growth_is_gated_on_owns_host_kv():
    method = _worker_method_node("_page_boundary_fast")
    parents = _parents(method)

    appends = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "host_grow_requests"
    ]
    assert len(appends) == 1
    node = appends[0]
    gated = False
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If) and _reads_gate(node.test, set()):
            gated = True
            break
    assert gated, "the boundary host-KV grow request is not gated on _owns_host_kv"

    # The grow itself must be fed by that gated list and nothing else, so the
    # gate cannot be bypassed by a second request list.
    grows = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "grow_pages_for_sequences"
    ]
    assert grows
    for call in grows:
        assert [ast.unparse(arg) for arg in call.args] == ["host_grow_requests"]


def test_boundary_growth_mirror_is_reached_under_a_rank_uniform_condition():
    method = _worker_method_node("_page_boundary_fast")
    parents = _parents(method)
    calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_mirror_group_host_kv"
    ]
    assert len(calls) == 1
    node = calls[0]
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            for rank_dependent in ("_owns_host_kv", "self.rank", "_uuid_to_local_map"):
                assert rank_dependent not in test, (
                    f"the growth mirror's gather sits under {test!r}"
                )


def test_mirror_refuses_a_serve_group_spanning_nodes():
    """G=16 on 8-GPU nodes would install the leader's page ids into another
    node's host region; the mirror must refuse before any collective."""
    fake_dist = FakeDist(world_size=16)
    worker = _rank(0, attn_tp_size=16, world_size=16, seqs={}, views={},
                   fake_dist=fake_dist, local_world_size=8)
    with pytest.raises(RuntimeError, match="must divide local_world_size"):
        worker._mirror_group_host_kv([], {})
