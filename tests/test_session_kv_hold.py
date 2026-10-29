"""Agent-session KV hold: decisions, page bookkeeping and worker messages.

The pure decisions (LCP, page boundary, metrics) are tested directly from
``batchgen.worker.session_hold``. The worker hooks are compiled from the real
source against fakes (tests/worker_method_source.py): a fake Host KV view that
tracks page ownership the way the backend does (sequence-owned vs resident)
and a fake node-shared trajectory book.
"""

import logging
import queue
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from batchgen.continuous_batching import select_sequences_for_eviction
from batchgen.prefix_reuse.prefill import (
    PrefixCacheSequenceState,
    SessionHeldPrefixPages,
)
from batchgen.query_book import release_local_query_slot
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker import session_hold as session_hold_lib
from batchgen.worker.session_hold import (
    HeldSession,
    SessionHoldBook,
    SessionHoldInvariantError,
    longest_common_prefix,
    plan_resume,
    split_turn_pages_for_hold,
)
from tests.worker_method_source import load_worker_methods

PAGE = 64


# --------------------------------------------------------------------------- #
#  Pure decisions
# --------------------------------------------------------------------------- #


def _plan(lcp, kv, prompt):
    return plan_resume(
        session_id="s", lcp_tokens=lcp, kv_tokens=kv, prompt_tokens=prompt,
        page_tokens=PAGE,
    )


def test_lcp_counts_leading_agreement():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    assert longest_common_prefix(a, torch.tensor([1, 2, 3, 4, 5])) == 4
    assert longest_common_prefix(a, torch.tensor([1, 2, 9, 4])) == 2
    assert longest_common_prefix(a, torch.tensor([9])) == 0
    assert longest_common_prefix(a, torch.tensor([], dtype=torch.int32)) == 0


@pytest.mark.parametrize(
    "lcp, kv, prompt, cached, keep, release",
    [
        # One page exactly: reused, one page kept.
        (64, 200, 300, 64, 1, 3),
        # One token past a page: the partial second page is kept, not dropped.
        (65, 200, 300, 65, 2, 2),
        # One token short of two pages: two pages, the second partial.
        (127, 200, 300, 127, 2, 2),
        # LCP past the held KV is capped by it (kv=129 -> 3 pages, all kept).
        (500, 129, 1000, 129, 3, 0),
        # The new prompt equals the held tokens: the last prompt token is
        # recomputed so prefill can emit the first output token.
        (200, 200, 200, 199, 4, 0),
        # Held KV ends one token into a page and the LCP stops at the page
        # boundary before it: that last held page is released.
        (128, 129, 300, 128, 2, 1),
    ],
)
def test_resume_plan_page_boundaries(lcp, kv, prompt, cached, keep, release):
    plan = _plan(lcp, kv, prompt)
    assert plan.resumes
    assert (plan.cached_tokens, plan.keep_pages, plan.release_pages) == (
        cached, keep, release)
    # The kept pages cover exactly the reused tokens, partial last page included.
    assert (keep - 1) * PAGE < cached <= keep * PAGE


@pytest.mark.parametrize("lcp, kv, prompt", [(63, 200, 300), (0, 200, 300),
                                             (200, 200, 64)])
def test_resume_below_one_page_drops_the_hold(lcp, kv, prompt):
    plan = _plan(lcp, kv, prompt)
    assert not plan.resumes
    assert plan.drop_reason == session_hold_lib.DROP_LCP_BELOW_PAGE
    assert (plan.cached_tokens, plan.keep_pages) == (0, 0)
    assert plan.release_pages == -(-kv // PAGE)


def test_split_pages_for_hold_keeps_attached_and_covering_own_pages():
    table = [10, 11, 20, 21, 22, 23]  # 2 attached (held) pages + 4 own
    held, own = split_turn_pages_for_hold(
        page_table=table, attached_pages=[10, 11], kv_tokens=3 * PAGE + 1,
        page_tokens=PAGE,
    )
    assert held == [10, 11, 20, 21] and own == [20, 21]
    # KV exactly on a page boundary takes no extra page.
    held, own = split_turn_pages_for_hold(
        page_table=table, attached_pages=[], kv_tokens=2 * PAGE, page_tokens=PAGE)
    assert held == [10, 11] and own == [10, 11]


def test_split_pages_for_hold_refuses_an_incomplete_host_copy():
    with pytest.raises(SessionHoldInvariantError, match="incomplete"):
        split_turn_pages_for_hold(
            page_table=[1, 2], attached_pages=[], kv_tokens=2 * PAGE + 1,
            page_tokens=PAGE)
    with pytest.raises(SessionHoldInvariantError, match="does not start"):
        split_turn_pages_for_hold(
            page_table=[1, 2, 3], attached_pages=[2], kv_tokens=PAGE,
            page_tokens=PAGE)


def test_book_metrics_count_holds_resumes_drops_and_releases():
    book = SessionHoldBook(page_tokens=PAGE)
    book.add(HeldSession("a", "t", "r1", 0, 5, 130, PAGE))
    book.add(HeldSession("b", "t", "r2", 1, 6, 64, PAGE))
    assert (book.held_kv_tokens, book.held_pages) == (194, 4)
    book.note_resume(_plan(100, 130, 300))
    book.note_resume(_plan(10, 64, 300))
    book.note_release("closed")
    book.note_reload_seconds(0.25)
    snap = book.snapshot()
    assert snap["held_sessions"] == 2 and snap["holds"] == 2
    assert snap["resumes"] == 1 and snap["reload_tokens"] == 100
    assert snap["reload_ms"] == 250.0
    assert snap["drops"] == {"lcp_below_page": 1}
    assert snap["releases"] == {"closed": 1}
    assert snap["lcp_histogram"]["<64"] == 1 and snap["lcp_histogram"]["<1024"] == 1


def test_session_prefix_state_reads_held_pages_without_a_coordinator():
    state = PrefixCacheSequenceState(
        lookup_result=SessionHeldPrefixPages("s", (7, 8), 100),
        attached_tokens=2 * PAGE, compute_cached_tokens=100,
    )
    assert state.is_session_hold and state.attachment_handle == 0
    assert state.shared_page_ids == (7, 8)
    assert state.shared_page_ids_for_group(1) == ()


def test_materialization_reads_held_pages_and_takes_no_attachment():
    from batchgen.prefix_reuse.materialization import materialize_gpt_oss_prefixes

    seen = {}

    class _Gpu:
        config = SimpleNamespace(page_size_tokens=PAGE)

        def allocate_pages_for_sequences(self, ids, lens):
            seen["alloc"] = (list(ids), list(lens))

        def rebuild_page_table(self, ids):
            pass

        def prepare_prefill_suffix_append(self, **kwargs):
            seen["append"] = kwargs
            return "plan"

        def get_padded_3d_page_pointers(self):
            ptrs = torch.zeros((1, 2, 4), dtype=torch.int64)
            return ptrs, ptrs.clone()

    class _View:
        def async_load_prefix_pages_to_device(self, **kwargs):
            seen["load"] = kwargs
            return SimpleNamespace(done=lambda: True, wait=lambda: None)

    result = materialize_gpt_oss_prefixes(
        gpu_manager=_Gpu(), host_worker_view=_View(), coordinator=None,
        lookup_results=[SessionHeldPrefixPages("s", (5, 6), 100), None],
        sequence_ids=[11, 12], prompt_lengths=[150, 40],
        compute_cached_tokens=[100, 0], raw_page_tokens=PAGE,
    )
    assert result.attachment_handles == ()
    assert seen["load"]["host_page_ids"].tolist() == [[5, 6], [0, 0]]
    assert seen["load"]["active_page_counts"].tolist() == [2, 0]
    assert seen["append"]["prefix_lens"] == [100, 0]
    assert seen["append"]["suffix_lens"] == [50, 40]


# --------------------------------------------------------------------------- #
#  Worker hooks on fakes
# --------------------------------------------------------------------------- #


class _HostView:
    """Host pages are sequence-owned (in a chain) or resident (owned by none)."""

    def __init__(self, free_pages=range(100, 200)):
        self.free = list(free_pages)
        self.tables: Dict[int, List[int]] = {}
        self.owner: Dict[int, object] = {}

    def allocate(self, gid, n, *, shared=()):
        pages = [self.free.pop(0) for _ in range(n)]
        for page in pages:
            self.owner[page] = gid
        self.tables[gid] = list(shared) + pages
        return pages

    def build_page_table(self, gids):
        return [list(self.tables[g]) for g in gids]

    def retain_sequence_pages(self, gid, pages):
        for page in pages:
            assert self.owner[page] == gid, "only sequence-owned pages retain"
            self.owner[page] = "resident"
        return list(pages)

    def release_resident_pages(self, pages):
        for page in pages:
            assert self.owner[page] == "resident", f"page {page} not resident"
            del self.owner[page]
            self.free.append(page)

    def release_sequence_pages(self, gids):
        for gid in gids:
            for page in self.tables.pop(gid):
                if self.owner.get(page) == gid:
                    del self.owner[page]
                    self.free.append(page)

    def resident(self):
        return sorted(p for p, o in self.owner.items() if o == "resident")


class _Book:
    """Node-shared trajectory book: slot -> tokens; uuid -> slot."""

    def __init__(self):
        self.tokens: Dict[int, List[int]] = {}
        self.by_uuid: Dict[str, int] = {}
        self.released: List[int] = []
        self._next = 0

    def bind(self, uuid, tokens):
        slot = self._next
        self._next += 1
        self.tokens[slot] = list(tokens)
        self.by_uuid[uuid] = slot
        return slot

    def has_sequence(self, uuid):
        return uuid in self.by_uuid

    def slot_for(self, uuid):
        return self.by_uuid[uuid]

    def copy_span_to(self, slot, out, *, start, length):
        out.copy_(torch.tensor(self.tokens[slot][start:start + length],
                               dtype=out.dtype))

    def release(self, slot):
        assert slot in self.tokens, f"slot {slot} released twice"
        del self.tokens[slot]
        self.by_uuid = {u: s for u, s in self.by_uuid.items() if s != slot}
        self.released.append(slot)


_METHOD_NAMES = [
    "_session_hold_eligible", "_session_host_view", "_retain_session_pages_for_hold",
    "_record_session_hold", "_free_held_session", "_apply_session_control",
    "_cancel_request", "_pin_session_hold_ranks", "_plan_session_resumes",
    "_session_resume_plan",
    "_commit_session_resumes", "_release_session_prefix_states",
    "_publish_session_hold_metrics", "_note_session_evictions", "_owns_host_kv",
    "_decode_attn_tp_size", "_drop_unstarted_request", "_prefix_states_in_use",
    "_select_sequences_for_onhold", "_apply_worker_messages", "_poll_admissions",
    "_sync_session_hold_owner_errors", "_session_hold_ttl_message",
    "_sessions_with_live_turns", "drop_lru_session_holds",
]


class _Dist:
    """One-process stand-in: broadcasts are identity, rank 0 is the source.

    ``peer_flag`` is what the other ranks contribute to a MAX all-reduce.
    """

    peer_flag = 0
    all_reduces = 0
    ReduceOp = SimpleNamespace(MAX="max")

    @staticmethod
    def broadcast(tensor, src):
        pass

    @staticmethod
    def broadcast_object_list(container, src):
        pass

    @classmethod
    def all_reduce(cls, tensor, op):
        assert op == cls.ReduceOp.MAX
        cls.all_reduces += 1
        tensor.clamp_(min=cls.peer_flag)


_METHODS = load_worker_methods(_METHOD_NAMES, {
    "session_hold_lib": session_hold_lib,
    "logging": logging,
    "torch": torch,
    "time": time,
    "dist": _Dist,
    "SequenceStatus": SequenceStatus,
    "SequenceEntry": SequenceEntry,
    "release_local_query_slot": release_local_query_slot,
    "_NO_MESSAGE": object(),
    "List": List, "Optional": Optional, "Sequence": Sequence,
    "Dict": Dict, "Tuple": Tuple,
})


class _Worker:
    world_size = 2
    torch_device = "cpu"
    _shutdown_requested = False

    def __init__(self, rank=0, host=None, book=None):
        self.rank = rank
        self.parallel_manager = SimpleNamespace(attn_tp_size=1)
        self.global_batch = SequenceBatch()
        self.host = host if host is not None else _HostView()
        self._trajectory_book = book if book is not None else _Book()
        self._session_hold = SessionHoldBook(page_tokens=PAGE)
        self.enable_prefix_cache = False
        self._prefix_sequence_states = {}
        self._session_attached_pages = {}
        self._session_retained_pages = {}
        self._session_resumed_uuids = set()
        self._cancelled_request_ids = set()
        self._session_hold_owner_errors = []
        self._session_hold_ttl_checked_at = 0.0
        self._session_resume_wave = {}
        self._session_resume_plans = {}
        self._session_reload_seconds = {}
        self._uuid_to_local_map = {}
        self._local_to_uuid_map = {}
        self.query_book = {}
        self._free_local_indices = set()
        self._response_queue = queue.Queue()
        self._admission_queue = None
        self.admitted_messages = []

    def __getattr__(self, name):
        if name in _METHODS:
            return _METHODS[name].__get__(self)
        raise AttributeError(name)

    def _host_views_by_group(self):
        return {0: self.host}

    def _publish_trajectory_pool_capacity(self):
        pass

    def _admit_sequences_from_message(self, msg):
        self.admitted_messages.append(msg)

    def _trajectory_tokens(self, seq, *, start=0, length=None, dtype=torch.int32):
        slot = self._trajectory_book.slot_for(seq.uuid)
        tokens = self._trajectory_book.tokens[slot]
        length = len(tokens) - start if length is None else length
        return torch.tensor(tokens[start:start + length], dtype=dtype)

    # -- scenario helpers -------------------------------------------------- #

    def add_seq(self, uuid, gid, tokens, *, rank=0, session="s1", status=None):
        seq = SequenceEntry(uuid=uuid, global_idx=gid, prompt_length=len(tokens),
                            max_decode_length=32)
        seq.session_id = session
        seq.traj_id = "t1"
        seq.assigned_rank = rank
        self.global_batch.add_sequence(seq)
        self._trajectory_book.bind(uuid, tokens)
        for target in status or ():
            self.global_batch.update_status(uuid, target)
        return seq


def _finish_turn(worker, uuid, gid, prompt, decoded, *, own_pages, shared=()):
    """A turn that decoded ``decoded`` tokens after ``prompt`` and completed."""
    seq = worker.add_seq(uuid, gid, list(prompt) + list(decoded), status=(
        SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
        SequenceStatus.IN_DECODE, SequenceStatus.COMPLETED))
    seq.prompt_length = len(prompt)
    seq.original_prompt_length = len(prompt)
    seq.decoded_length = len(decoded)
    seq.current_context_length = len(prompt) + len(decoded)
    if worker.rank == 0:
        worker.host.allocate(gid, own_pages, shared=shared)
    return seq


def _hold(worker, seq):
    """The completion order of the worker: retain (owner), release, record."""
    worker._retain_session_pages_for_hold([seq.uuid] if worker.rank == 0 else [])
    if worker.rank == 0:
        worker.host.release_sequence_pages([seq.global_idx])
    slot = worker._trajectory_book.slot_for(seq.uuid)
    return worker._record_session_hold(seq, slot)


def test_hold_keeps_exactly_the_pages_covering_the_kv_and_the_slot():
    worker = _Worker()
    prompt, decoded = list(range(1, 101)), [500 + i for i in range(30)]
    seq = _finish_turn(worker, "turn1", 7, prompt, decoded, own_pages=5)
    info = _hold(worker, seq)
    # KV covers prompt + decoded - 1 = 129 tokens -> 3 pages; 2 headroom freed.
    assert info == {"held": True, "kv_tokens": 129, "pages": 3}
    record = worker._session_hold.get("s1")
    assert record.page_ids == (100, 101, 102) and record.rank == 0
    assert worker.host.resident() == [100, 101, 102]
    assert {103, 104} <= set(worker.host.free)
    assert worker._trajectory_book.has_sequence("turn1")  # slot kept
    assert worker._session_hold.snapshot()["held_kv_pages"] == 3


def test_non_owner_rank_records_the_same_hold_without_pages():
    worker = _Worker(rank=1)
    seq = _finish_turn(worker, "turn1", 7, list(range(100)), [1] * 30, own_pages=5)
    info = _hold(worker, seq)
    assert info["kv_tokens"] == 129 and info["pages"] == 3
    assert worker._session_hold.get("s1").page_ids is None


def _held_worker(rank=0):
    worker = _Worker(rank=rank)
    prompt, decoded = list(range(1, 101)), [500 + i for i in range(30)]
    seq = _finish_turn(worker, "turn1", 7, prompt, decoded, own_pages=5)
    _hold(worker, seq)
    return worker, prompt, decoded


def test_resume_reuses_the_lcp_and_frees_held_pages_past_it():
    worker, prompt, decoded = _held_worker()
    # Turn 2 re-renders history: turn-1 prompt + 10 of its decoded tokens,
    # then a new suffix (the template dropped the rest of the output).
    new_prompt = prompt + decoded[:10] + [900 + i for i in range(40)]
    turn2 = worker.add_seq("turn2", 8, new_prompt)
    worker._plan_session_resumes(["turn2"])
    state = worker._prefix_sequence_states[8]
    assert state.is_session_hold
    assert state.compute_cached_tokens == 110          # LCP = 100 + 10
    assert state.attached_tokens == 2 * PAGE           # partial second page kept
    assert state.shared_page_ids == (100, 101)
    # Nothing is freed until the wave's Host reservation commits.
    assert worker.host.resident() == [100, 101, 102]
    worker._commit_session_resumes()
    assert worker.host.resident() == [100, 101]
    assert worker._session_attached_pages == {8: (100, 101)}
    assert worker._session_hold.get("s1") is None
    assert "turn2" in worker._session_resumed_uuids
    snap = worker._session_hold.snapshot()
    assert snap["resumes"] == 1 and snap["reload_tokens"] == 110
    assert snap["held_sessions"] == 0
    plan = worker._session_resume_plans["turn2"]
    assert (plan.lcp_tokens, plan.cached_tokens) == (110, 110)
    assert turn2.uuid == "turn2"


def test_non_owner_reaches_the_same_resume_decision_without_a_state():
    worker, prompt, decoded = _held_worker(rank=1)
    worker.add_seq("turn2", 8, prompt + decoded[:10] + [1] * 40)
    worker._plan_session_resumes(["turn2"])
    assert worker._prefix_sequence_states == {}
    worker._commit_session_resumes()
    assert worker._session_resume_plans["turn2"].cached_tokens == 110
    assert worker._session_hold.snapshot()["reload_tokens"] == 110


def test_resume_with_an_lcp_below_one_page_drops_the_hold():
    worker, prompt, _ = _held_worker()
    worker.add_seq("turn2", 8, prompt[:40] + [7] * 100)
    worker._plan_session_resumes(["turn2"])
    assert worker._prefix_sequence_states == {}
    worker._commit_session_resumes()
    assert worker.host.resident() == []
    assert worker._session_hold.snapshot()["drops"] == {"lcp_below_page": 1}


def test_a_rejected_wave_leaves_the_hold_intact():
    worker, prompt, decoded = _held_worker()
    worker.add_seq("turn2", 8, prompt + decoded + [1] * 5)
    worker._plan_session_resumes(["turn2"])
    # The reservation rejected the wave: its prefix states are released.
    worker._release_session_prefix_states([8])
    assert worker._prefix_sequence_states == {}
    assert worker.host.resident() == [100, 101, 102]
    assert worker._session_hold.get("s1") is not None


def test_a_resumed_turn_that_is_not_held_frees_its_attached_pages():
    worker, prompt, decoded = _held_worker()
    worker.add_seq("turn2", 8, prompt + decoded[:10] + [1] * 40)
    worker._plan_session_resumes(["turn2"])
    worker._commit_session_resumes()
    worker._release_session_prefix_states([8])
    assert worker.host.resident() == []


def test_holding_a_resumed_turn_carries_its_attached_pages_forward():
    worker, prompt, decoded = _held_worker()
    new_prompt = prompt + decoded[:10] + [900 + i for i in range(40)]
    worker.add_seq("turn2", 8, new_prompt)
    worker._plan_session_resumes(["turn2"])
    worker._commit_session_resumes()
    # Turn 2's chain: the 2 attached held pages, then its own pages.
    turn2 = worker.global_batch.get_sequence("turn2")
    for status in (SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
                   SequenceStatus.IN_DECODE, SequenceStatus.COMPLETED):
        worker.global_batch.update_status("turn2", status)
    turn2.decoded_length = 20
    turn2.original_prompt_length = len(new_prompt)
    turn2.current_context_length = len(new_prompt) + 20  # 170 -> KV 169 tokens
    own = worker.host.allocate(8, 4, shared=(100, 101))
    worker._trajectory_book.tokens[worker._trajectory_book.slot_for("turn2")] += [3] * 20
    _hold(worker, turn2)
    record = worker._session_hold.get("s1")
    # 169 tokens = 3 pages: the 2 attached ones and the first own page.
    assert record.kv_tokens == 169 and record.page_ids == (100, 101, own[0])
    assert worker.host.resident() == sorted([100, 101, own[0]])
    assert not set(own[1:]) & set(worker.host.owner)  # headroom freed
    assert worker._session_attached_pages == {}


def test_close_releases_held_pages_and_slot_on_every_rank():
    for rank in (0, 1):
        worker, _, _ = _held_worker(rank=rank)
        slot = worker._session_hold.get("s1").trajectory_slot
        worker._apply_session_control({
            "type": "session_control", "cancel": [],
            "release": [{"session_id": "s1", "reason": "closed"},
                        {"session_id": "never-held", "reason": "closed"}],
        })
        assert worker._session_hold.get("s1") is None
        assert slot in worker._trajectory_book.released
        snap = worker._session_hold.snapshot()
        assert (snap["held_kv_pages"], snap["held_sessions"]) == (0, 0)
        assert snap["releases"] == {"closed": 1}
        if rank == 0:
            assert worker.host.resident() == []
            metrics = [m for m in list(worker._response_queue.queue)
                       if m["type"] == "session_hold_metrics"]
            assert metrics[-1]["held_kv_pages"] == 0


def test_cancel_drops_a_queued_request_and_reports_it():
    worker = _Worker()
    worker.add_seq("turnq", 9, [1, 2, 3])
    worker._apply_session_control(
        {"type": "session_control", "cancel": ["turnq"], "release": [],
         "reason": "abort"})
    assert worker.global_batch.get_sequence("turnq") is None
    assert not worker._trajectory_book.has_sequence("turnq")
    done = [m for m in list(worker._response_queue.queue) if m["type"] == "completion"]
    assert done[0]["request_id"] == "turnq"
    assert done[0]["error"]["code"] == "cancelled"


def test_cancel_finishes_a_decoding_request_and_it_is_never_held():
    worker = _Worker()
    seq = worker.add_seq("turnd", 9, [1, 2, 3], status=(
        SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE))
    worker._apply_session_control(
        {"type": "session_control", "cancel": ["turnd"], "release": []})
    assert seq.eos_reached
    assert not worker._session_hold_eligible(seq)


def test_next_turn_is_pinned_to_the_rank_holding_its_kv():
    worker = _Worker()
    worker._session_hold.add(HeldSession("s1", "t1", "turn1", 3, 0, 100, PAGE))
    a = worker.add_seq("a", 1, [1], session="s1", rank=None)
    worker.add_seq("b", 2, [1], session="other", rank=None)
    worker.add_seq("c", 3, [1], session=None, rank=None)
    assert worker._pin_session_hold_ranks(["a", "b", "c"]) == ["b", "c"]
    assert a.assigned_rank == 3


def test_held_sessions_are_not_victim_candidates():
    """Held KV is resident pages, not a sequence: selectors never see it."""
    worker, _, _ = _held_worker()
    held_uuid = worker._session_hold.get("s1").request_id
    assert worker.global_batch.get_sequence(held_uuid).status == SequenceStatus.COMPLETED
    for uuid, gid in (("d1", 20), ("d2", 21)):
        worker.add_seq(uuid, gid, [1] * 10, session=None, status=(
            SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
            SequenceStatus.IN_DECODE))
        worker._uuid_to_local_map[uuid] = gid
        worker.global_batch.get_sequence(uuid).gpu_pages_allocated = 4
    worker.gpu_paged_kv_cache_manager = SimpleNamespace(
        get_stats=lambda: SimpleNamespace(num_free_pages=0))
    active = worker.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
    on_hold = worker._select_sequences_for_onhold(active, required_free_pages=1000)
    evicted, _ = select_sequences_for_eviction(
        [(u, {"host_pages_allocated": 9, "decoded_length": 0}) for u in active],
        pages_to_free=1000, page_key="host_pages_allocated")
    assert held_uuid not in on_hold and held_uuid not in evicted
    assert set(on_hold) == set(evicted) == {"d1", "d2"}
    # And evicting running sequences never touches the held pages.
    worker._note_session_evictions(["d1"])
    assert worker.host.resident() == [100, 101, 102]
    assert worker._session_hold.snapshot()["evictions_while_held"] == 1


def test_poll_applies_control_messages_without_using_up_the_admit():
    worker = _Worker()
    worker._admission_queue = queue.Queue()
    for msg in (
        {"type": "session_control", "cancel": [], "release": []},
        {"type": "admit", "entries": ["a"]},
        {"type": "session_control", "cancel": [], "release": []},
        {"type": "admit", "entries": ["b"]},
    ):
        worker._admission_queue.put(msg)
    applied = []
    worker._apply_session_control = lambda msg: applied.append(msg)
    assert worker._poll_admissions() is True
    assert [m["entries"] for m in worker.admitted_messages] == [["a"]]
    assert len(applied) == 1
    assert worker._poll_admissions() is True
    assert [m["entries"] for m in worker.admitted_messages] == [["a"], ["b"]]
    assert len(applied) == 2
    # Only control messages: applied, nothing admitted.
    worker._admission_queue.put({"type": "session_control", "cancel": [], "release": []})
    assert worker._poll_admissions() is False
    assert len(applied) == 3


def test_session_hold_model_support_is_decided_by_the_prefix_reuse_config():
    from batchgen.prefix_reuse.config import require_session_hold_model_support

    assert require_session_hold_model_support("openai/gpt-oss-120b")
    with pytest.raises(ValueError, match="GPT-OSS only"):
        require_session_hold_model_support("moonshotai/Kimi-K2-Instruct")


def test_attention_tp_is_refused_at_init_not_at_the_first_held_completion():
    """M1: a TP attention group would raise on every rank at the first held
    completion (and again after each restart); it is refused right after the
    parallel strategy manager is built, before anything is admitted."""
    import ast
    from pathlib import Path

    require = load_worker_methods(
        ["_require_session_hold_pure_dp"],
        {"session_hold_lib": session_hold_lib},
    )["_require_session_hold_pure_dp"]
    worker = _Worker()
    worker.parallel_manager = SimpleNamespace(attn_tp_size=2)
    with pytest.raises(session_hold_lib.SessionHoldUnsupported, match="attn_tp_size=2"):
        require(worker)
    worker.parallel_manager = SimpleNamespace(attn_tp_size=1)
    require(worker)
    # Hold off: nothing to refuse.
    worker.parallel_manager = SimpleNamespace(attn_tp_size=2)
    worker._session_hold = None
    require(worker)

    source = (Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py").read_text()
    init = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_initialize_core_components"
    )
    calls = [ast.unparse(stmt) for stmt in init.body]
    psm = max(i for i, text in enumerate(calls) if "self.parallel_manager = " in text)
    check = calls.index("self._require_session_hold_pure_dp()")
    assert psm < check


def test_pages_are_retained_only_for_completed_sequences():
    """L2: a sequence the completion path does not report (status is not
    COMPLETED, e.g. a stale eos_reached on PREFILLED) is never recorded as a
    hold, so retaining its pages would leak them as resident pages."""
    worker = _Worker()
    seq = worker.add_seq("turnp", 7, list(range(100)), status=(
        SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED))
    seq.current_context_length = 100
    worker.host.allocate(7, 2)
    worker._retain_session_pages_for_hold(["turnp"])
    assert worker._session_retained_pages == {}
    worker.host.release_sequence_pages([7])
    assert worker.host.resident() == []


def test_an_owner_only_kv_append_failure_is_raised_on_every_rank():
    """L3: the owner waits for its in-flight KV appends before a hold; a
    failure there raised on the owner alone, leaving the other ranks blocked
    in their next collective. It is deferred and raised by every rank at the
    hold record's all-reduce."""
    owner = _Worker(rank=0)
    seq = _finish_turn(owner, "turn1", 7, list(range(100)), [1] * 30, own_pages=5)
    owner._pending_kv_append_tasks = ["task"]
    owner._deferred_kv_append_wait_errors = []

    def wait(*, defer_errors=False, **_):
        assert defer_errors, "the owner must not raise alone"
        owner._pending_kv_append_tasks = []
        owner._deferred_kv_append_wait_errors.append("RuntimeError: D2H failed")
        return 1

    owner._wait_pending_kv_append_tasks = wait
    owner._retain_session_pages_for_hold(["turn1"])       # does not raise
    assert owner.host.resident() == []                    # nothing held
    slot = owner._trajectory_book.slot_for("turn1")
    with pytest.raises(SessionHoldInvariantError, match="D2H failed"):
        owner._record_session_hold(seq, slot)

    peer = _Worker(rank=1)
    peer_seq = _finish_turn(peer, "turn1", 7, list(range(100)), [1] * 30, own_pages=5)
    _Dist.peer_flag = 1                                   # the owner failed
    try:
        with pytest.raises(SessionHoldInvariantError, match="owner rank"):
            peer._record_session_hold(peer_seq, peer._trajectory_book.slot_for("turn1"))
    finally:
        _Dist.peer_flag = 0
    assert peer._session_hold.get("s1") is None


def test_a_hold_record_costs_one_flag_all_reduce_when_nothing_failed():
    before = _Dist.all_reduces
    worker, _, _ = _held_worker()
    assert _Dist.all_reduces == before + 1
    assert worker._session_hold.get("s1") is not None


# --------------------------------------------------------------------------- #
#  Idle TTL and the LRU drop hook
# --------------------------------------------------------------------------- #


def _record(session, seq_no, pages, held_at=0.0):
    record = HeldSession(session, "t", f"r-{session}", 0, seq_no, pages * PAGE, PAGE)
    record.hold_seq, record.held_at = seq_no, held_at
    return record


def test_expired_holds_are_idle_past_the_ttl_and_not_busy():
    sessions = {s: _record(s, i, 1, held_at=t) for i, (s, t) in
                enumerate([("old", 0.0), ("busy", 0.0), ("new", 950.0), ("mid", 50.0)])}
    expired = session_hold_lib.expired_holds(
        sessions, now=1000.0, idle_ttl_s=900.0, busy_sessions={"busy"})
    assert expired == [("old", 1000.0), ("mid", 950.0)]
    assert session_hold_lib.expired_holds(
        sessions, now=1e9, idle_ttl_s=0, busy_sessions=set()) == []


def test_lru_victims_cover_the_need_oldest_hold_first():
    sessions = {"a": _record("a", 3, 4), "b": _record("b", 1, 2),
                "c": _record("c", 2, 5), "live": _record("live", 0, 9)}
    victims = session_hold_lib.lru_hold_victims
    assert victims(sessions, pages_needed=6, exclude={"live"}) == ["b", "c"]
    assert victims(sessions, pages_needed=2, exclude={"live"}) == ["b"]
    assert victims(sessions, pages_needed=100, exclude={"live"}) == ["b", "c", "a"]
    assert victims(sessions, pages_needed=0) == []


def test_book_numbers_holds_in_order_and_reports_the_ttl():
    book = SessionHoldBook(page_tokens=PAGE, idle_ttl_s=60.0)
    first, second = _record("a", 0, 1), _record("b", 0, 1)
    book.add(first)
    book.add(second)
    assert (first.hold_seq, second.hold_seq) == (1, 2)
    assert book.snapshot()["idle_ttl_s"] == 60.0
    assert session_hold_lib.DEFAULT_IDLE_TTL_S == 900.0


def test_an_idle_hold_past_the_ttl_is_released_and_counted(caplog):
    worker, prompt, _ = _held_worker()
    worker._session_hold.idle_ttl_s = 900.0
    record = worker._session_hold.get("s1")
    record.held_at = time.monotonic() - 901.0
    # A turn of another session in the worker does not keep s1 busy.
    worker.add_seq("other", 30, [1, 2, 3], session="s2")
    msg = worker._session_hold_ttl_message()
    assert msg["type"] == "session_control" and msg["cancel"] == []
    assert [item["session_id"] for item in msg["release"]] == ["s1"]
    assert msg["release"][0]["reason"] == "idle_ttl"
    assert msg["release"][0]["idle_s"] >= 901.0
    # Checked at most once a second.
    assert worker._session_hold_ttl_message() is None
    with caplog.at_level(logging.WARNING):
        worker._apply_session_control(msg)
    assert worker._session_hold.get("s1") is None
    assert worker.host.resident() == []
    snap = worker._session_hold.snapshot()
    assert snap["releases"] == {"idle_ttl": 1} and snap["held_kv_pages"] == 0
    assert "idle TTL expired: released session=s1" in caplog.text


def test_a_hold_whose_next_turn_is_in_the_worker_does_not_expire():
    worker, prompt, _ = _held_worker()
    worker._session_hold.get("s1").held_at = time.monotonic() - 10_000.0
    worker.add_seq("turn2", 8, prompt + [1] * 10)      # queued, same session
    assert worker._session_hold_ttl_message() is None
    worker._session_hold_ttl_checked_at = 0.0
    worker._session_hold.idle_ttl_s = 0                 # TTL off
    worker.global_batch.remove_sequence("turn2")
    assert worker._session_hold_ttl_message() is None


def test_only_rank_zero_decides_an_expiry():
    worker, _, _ = _held_worker(rank=1)
    worker._session_hold.get("s1").held_at = time.monotonic() - 10_000.0
    assert worker._session_hold_ttl_message() is None


def test_the_poll_broadcasts_a_ttl_release_with_the_queue_messages():
    worker, _, _ = _held_worker()
    worker._session_hold.get("s1").held_at = time.monotonic() - 10_000.0
    worker._admission_queue = queue.Queue()
    worker._admission_queue.put({"type": "admit", "entries": ["a"]})
    assert worker._poll_admissions() is True
    assert worker._session_hold.get("s1") is None
    assert worker._session_hold.snapshot()["releases"] == {"idle_ttl": 1}
    assert [m["entries"] for m in worker.admitted_messages] == [["a"]]


def test_lru_drop_hook_frees_the_oldest_idle_holds_on_every_rank():
    for rank in (0, 1):
        worker = _Worker(rank=rank)
        for index, (uuid, session) in enumerate(
                [("t1", "s1"), ("t2", "s2"), ("t3", "s3")]):
            seq = worker.add_seq(uuid, 10 + index, list(range(100)),
                                 session=session, status=(
                SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
                SequenceStatus.IN_DECODE, SequenceStatus.COMPLETED))
            seq.current_context_length = 130                 # 129 KV -> 3 pages
            if rank == 0:
                worker.host.allocate(10 + index, 3)
            _hold(worker, seq)
        worker.add_seq("next1", 20, list(range(10)), session="s1")  # s1 is live
        dropped = worker.drop_lru_session_holds(4)
        assert dropped == ["s2", "s3"]
        assert set(worker._session_hold.sessions) == {"s1"}
        assert worker._session_hold.snapshot()["drops"] == {"pressure_lru": 2}
        if rank == 0:
            assert len(worker.host.resident()) == 3


# --------------------------------------------------------------------------- #
#  No-progress guard: held KV resets it only when held KV is the blocker
# --------------------------------------------------------------------------- #


def _selection(*, free, prompt=640, slot_free=None):
    from batchgen.worker.prefill import PrefillCandidate, PrefillSelectionRequest

    candidate = PrefillCandidate(
        uuid="q1", assigned_rank=0, node_id=0, is_evicted=False, global_idx=5,
        total_decoded_before_eviction=0, prompt_length=prompt,
        kv_token_budget=4096, page_size=PAGE,
    )
    return PrefillSelectionRequest(
        candidates=(candidate,), per_node_host_free=(free,), chunk_size=0,
        num_nodes=1, gpus_per_node=8, initial_gpu_page_buffer=0,
        per_node_sequence_free=None if slot_free is None else (slot_free,),
    )


def test_blocked_by_held_kv_only_when_freeing_held_pages_would_admit():
    from batchgen.worker.prefill import PrefillScheduler

    blocked = session_hold_lib.admission_blocked_by_held_kv
    needed = PrefillScheduler.required_host_pages(
        _selection(free=0).candidates[0], _selection(free=0))
    # Short by 3 pages, 5 held: held KV is the blocker.
    assert blocked(_selection(free=needed - 3), held_pages=5)
    # Short by 10 pages, 5 held: it would not fit even without the holds.
    assert not blocked(_selection(free=needed - 10), held_pages=5)
    # It fits: the wave selects it, nothing is blocked.
    assert not blocked(_selection(free=needed), held_pages=5)
    # No sequence slot: a slot limit, not held KV.
    assert not blocked(_selection(free=needed - 3, slot_free=0), held_pages=5)
    assert not blocked(None, held_pages=5)
    assert not blocked(_selection(free=needed - 3), held_pages=0)


def _guard_worker(selection):
    note = load_worker_methods(
        ["_note_admission_wait_on_held_kv"],
        {"session_hold_lib": session_hold_lib, "logging": logging,
         "SequenceStatus": SequenceStatus,
         "time": SimpleNamespace(sleep=lambda seconds: None)},
    )["_note_admission_wait_on_held_kv"]
    worker, prompt, _ = _held_worker()          # one hold of 3 pages
    worker.add_seq("q1", 30, list(range(640)), session="s9")
    worker._prefill_admission_live = False
    worker._prefill_admission_selection = selection
    return lambda: note(worker, 0), worker


def test_guard_state_passes_through_unless_held_kv_blocks_admission():
    """H4: with any session held, every empty wave used to reset the
    no-progress guard (state None), masking unrelated stalls."""
    from batchgen.worker.prefill import PrefillScheduler

    probe = _selection(free=0)
    needed = PrefillScheduler.required_host_pages(probe.candidates[0], probe)
    waits_on_held, worker = _guard_worker(_selection(free=needed - 2))
    assert waits_on_held() is True
    assert worker._session_hold.admission_waits == 1
    # A slot stall while a session is held: the guard must see it.
    stalled, worker = _guard_worker(_selection(free=needed - 2, slot_free=0))
    assert stalled() is False
    assert worker._session_hold.admission_waits == 0
    # A request too large even with the held pages freed: not a held-KV wait.
    too_big, _ = _guard_worker(_selection(free=needed - 50))
    assert too_big() is False
