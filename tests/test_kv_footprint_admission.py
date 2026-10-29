"""Footprint admission (I-A) on a session-hold server.

Pure decisions come from ``batchgen.worker.kv_admission``; the worker glue is
compiled from the real source (tests/worker_method_source.py) against the same
fakes the session-hold tests use. Server side: held scheduling slots and the
margin flag.
"""

import importlib.util
import logging
import sys
import time
import types
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker import kv_admission as ka
from batchgen.worker import session_hold as session_hold_lib
from batchgen.worker.prefill import (
    PrefillCandidate,
    PrefillScheduler,
    PrefillSelectionRequest,
)
from batchgen.worker.session_hold import HeldSession, SessionHoldBook
from tests.worker_method_source import load_worker_methods

PAGE = 64


def _load_server_args():
    """``batchgen.server.server_args`` without the server package's __init__.

    The package __init__ imports the HTTP server and, through it, the CUDA
    kernel stack; the margin resolution needs none of that.
    """
    server = Path(__file__).resolve().parents[1] / "batchgen" / "server"
    saved = {name: sys.modules.get(name) for name in (
        "batchgen.server", "batchgen.server.server_args",
        "batchgen.server.runtime_identity",
    )}
    try:
        package = types.ModuleType("batchgen.server")
        package.__path__ = [str(server)]
        sys.modules["batchgen.server"] = package
        module = None
        for name in ("runtime_identity", "server_args"):
            spec = importlib.util.spec_from_file_location(
                f"batchgen.server.{name}", server / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _cand(uuid, kind, *, host, gpu=None, budget=None, rank=0, idx=0, held=0, decoded=0):
    budget = host if budget is None else budget
    return ka.FootprintCandidate(
        uuid=uuid, kind=kind, node_id=0, rank=rank, global_idx=idx,
        host_pages=host, gpu_pages=budget if gpu is None else gpu,
        budget_pages=budget, held_pages=held,
        total_decoded_before_eviction=decoded,
    )


def _state(*, total=1000, running=0, held=0, margin=100, gpu_total=None, gpu_running=(0,)):
    return ka.FootprintState(
        host_total_by_node=(total,),
        host_running_by_node=(running,),
        host_held_by_node=(held,),
        host_margin_by_node=(margin,),
        gpu_total_by_rank=gpu_total,
        gpu_running_by_rank=tuple(gpu_running),
    )


# --------------------------------------------------------------------------- #
#  Footprint arithmetic
# --------------------------------------------------------------------------- #


def test_free_capacity_is_total_minus_running_held_and_margin():
    state = _state(total=1000, running=300, held=250, margin=100)
    assert state.host_free() == [350]
    plan = ka.plan_footprint_admission(
        [_cand("a", ka.KIND_NEW, host=350), _cand("b", ka.KIND_NEW, host=1, idx=1)],
        state, live=True,
    )
    # "a" takes exactly the free pages; "b" no longer fits.
    assert plan.admitted == ("a",)
    assert plan.deferred == {"b": ka.DEFER_HOST}
    assert plan.host_free_after == (0,)


def test_margin_pages_round_up_and_default_follows_the_eviction_watermark():
    assert ka.margin_pages(910, 10.0) == 91
    assert ka.margin_pages(1001, 10.0) == 101
    assert ka.DEFAULT_ADMISSION_MARGIN_PERCENT == 10.0


@pytest.mark.parametrize("margin, watermark", [(9.9, 10), (4.0, 0), (100.0, 10)])
def test_a_margin_below_the_eviction_band_is_refused(margin, watermark):
    with pytest.raises(ka.KVAdmissionConfigError):
        ka.validate_margin_percent(margin, eviction_watermark_percent=watermark)


def test_margin_resolution_from_server_args():
    args = SimpleNamespace(
        enable_session_kv_hold=True,
        session_kv_admission_margin_percent=None,
        host_kv_eviction_watermark=10,
    )
    resolve_session_kv_admission_margin = (
        _load_server_args().resolve_session_kv_admission_margin
    )
    assert resolve_session_kv_admission_margin(args) == 10.0
    args.host_kv_eviction_watermark = 20
    assert resolve_session_kv_admission_margin(args) == 20.0
    args.session_kv_admission_margin_percent = 15.0
    with pytest.raises(ValueError, match="below"):
        resolve_session_kv_admission_margin(args)
    args.session_kv_admission_margin_percent = 25.0
    assert resolve_session_kv_admission_margin(args) == 25.0
    # An explicit margin without session hold is refused, not ignored.
    off = SimpleNamespace(
        enable_session_kv_hold=False,
        session_kv_admission_margin_percent=12.0,
        host_kv_eviction_watermark=10,
    )
    with pytest.raises(ValueError, match="only with --enable-session-kv-hold"):
        resolve_session_kv_admission_margin(off)
    off.session_kv_admission_margin_percent = None
    assert resolve_session_kv_admission_margin(off) is None


def test_a_resumed_turn_is_charged_its_increment_not_its_budget():
    # The session holds 300 pages; its next turn's budget is 340 pages. Only
    # 50 pages are free: the full budget would never fit, the increment does.
    state = _state(total=1000, running=0, held=850, margin=100)
    assert state.host_free() == [50]
    plan = ka.plan_footprint_admission(
        [_cand("r", ka.KIND_RESUMED, host=40, budget=340, held=300)],
        state, live=False,
    )
    assert plan.admitted == ("r",)
    assert not plan.overcommit


def test_gpu_footprint_counts_the_whole_budget_per_rank():
    state = _state(
        total=10_000, margin=0, gpu_total=(500, 500), gpu_running=(450, 0),
    )
    plan = ka.plan_footprint_admission(
        [
            _cand("on0", ka.KIND_NEW, host=100, rank=0, idx=0),
            _cand("on1", ka.KIND_NEW, host=100, rank=1, idx=1),
        ],
        state, live=True,
    )
    assert plan.admitted == ("on1",)
    assert plan.deferred == {"on0": ka.DEFER_GPU}


def test_an_unknown_gpu_pool_does_not_gate():
    state = _state(total=10_000, margin=0, gpu_total=(-1,), gpu_running=(10**6,))
    plan = ka.plan_footprint_admission(
        [_cand("a", ka.KIND_NEW, host=100)], state, live=True,
    )
    assert plan.admitted == ("a",)


def test_a_request_larger_than_the_usable_pool_never_fits():
    state = _state(total=1000, margin=100, gpu_total=(5000,))
    plan = ka.plan_footprint_admission(
        [
            _cand("huge", ka.KIND_NEW, host=901),
            _cand("ok", ka.KIND_NEW, host=900, idx=1),
            _cand("gpu", ka.KIND_NEW, host=10, gpu=5001, idx=2),
        ],
        state, live=False,
    )
    assert set(plan.never_fits) == {"huge", "gpu"}
    assert "usable Host pool 900" in plan.never_fits["huge"]
    assert plan.admitted == ("ok",)


# --------------------------------------------------------------------------- #
#  Priority ordering
# --------------------------------------------------------------------------- #


def test_order_is_evicted_then_resumed_then_new_by_arrival():
    plan = ka.plan_footprint_admission(
        [
            _cand("new_early", ka.KIND_NEW, host=1, idx=1),
            _cand("resumed_late", ka.KIND_RESUMED, host=1, budget=5, held=4, idx=9),
            _cand("evicted_short", ka.KIND_EVICTED, host=1, idx=5, decoded=10),
            _cand("evicted_long", ka.KIND_EVICTED, host=1, idx=7, decoded=90),
            _cand("resumed_early", ka.KIND_RESUMED, host=1, budget=5, held=4, idx=3),
        ],
        _state(), live=True,
    )
    assert plan.order == (
        "evicted_long", "evicted_short", "resumed_early", "resumed_late", "new_early",
    )
    assert plan.admitted == plan.order


def test_a_waiting_resumed_turn_refuses_new_work_but_not_other_resumed_turns():
    # 100 free pages. The big resumed turn (arrived first) needs 150 and
    # waits; a small resumed turn still goes; new work that would fit is
    # refused so it cannot take pages the waiting resumed turn needs.
    state = _state(total=1000, running=400, held=400, margin=100)
    plan = ka.plan_footprint_admission(
        [
            _cand("big", ka.KIND_RESUMED, host=150, budget=350, held=200, idx=1),
            _cand("small", ka.KIND_RESUMED, host=20, budget=220, held=200, idx=2),
            _cand("new", ka.KIND_NEW, host=10, idx=0),
        ],
        state, live=True,
    )
    assert plan.admitted == ("small",)
    assert plan.deferred == {
        "big": ka.DEFER_HOST, "new": ka.DEFER_RESUMED_FIRST,
    }
    assert plan.blocked_resumed == ("big",)
    assert not plan.overcommit  # something runs: the resumed turn waits for it


def test_new_work_is_admitted_while_no_resumed_turn_waits():
    plan = ka.plan_footprint_admission(
        [_cand("n1", ka.KIND_NEW, host=10), _cand("n2", ka.KIND_NEW, host=10, idx=1)],
        _state(total=1000, running=0, held=880, margin=100), live=False,
    )
    assert plan.admitted == ("n1", "n2")


# --------------------------------------------------------------------------- #
#  Deadlock guard
# --------------------------------------------------------------------------- #


def test_genuine_overcommit_is_flagged_only_with_nothing_running():
    # Every free page is held by other sessions and nothing runs.
    state = _state(total=1000, running=0, held=880, margin=100)
    resumed = _cand("r", ka.KIND_RESUMED, host=40, budget=100, held=60)
    new = _cand("n", ka.KIND_NEW, host=5, idx=1)
    plan = ka.plan_footprint_admission([resumed, new], state, live=False)
    assert plan.overcommit
    assert plan.admitted == ()
    assert plan.deferred == {"r": ka.DEFER_OVERCOMMIT, "n": ka.DEFER_RESUMED_FIRST}
    # The same shortage while something runs is a wait, not an overcommit.
    waiting = ka.plan_footprint_admission([resumed, new], state, live=True)
    assert not waiting.overcommit
    assert waiting.deferred["r"] == ka.DEFER_HOST


def test_overcommit_is_not_flagged_when_another_resumed_turn_progresses():
    state = _state(total=1000, running=0, held=880, margin=100)
    plan = ka.plan_footprint_admission(
        [
            _cand("big", ka.KIND_RESUMED, host=40, budget=100, held=60, idx=0),
            _cand("small", ka.KIND_RESUMED, host=10, budget=70, held=60, idx=1),
        ],
        state, live=False,
    )
    assert plan.admitted == ("small",)
    assert not plan.overcommit


def test_stats_count_deferrals_overcommit_episodes_and_footprint_error():
    stats = ka.KVAdmissionStats(margin_percent=10.0, page_tokens=PAGE)
    state = _state(total=1000, running=0, held=880, margin=100)
    plan = ka.plan_footprint_admission(
        [_cand("r", ka.KIND_RESUMED, host=40, budget=100, held=60),
         _cand("n", ka.KIND_NEW, host=5, idx=1)],
        state, live=False,
    )
    stats.note_plan(plan, {"r": ka.KIND_RESUMED, "n": ka.KIND_NEW})
    assert stats.note_overcommit(True) is True
    assert stats.note_overcommit(True) is False
    stats.note_overcommit(False)
    assert stats.note_overcommit(True) is True
    stats.note_completion(projected_pages=100, actual_pages=70)
    stats.note_completion(projected_pages=50, actual_pages=50)
    stats.note_sample(host_total=1000, host_used=900, held=600, running_projected=350,
                      gpu_total=8000, gpu_projected_max=400)
    snap = stats.snapshot()
    assert snap["deferred"] == {"overcommit": 1, "resumed_turn_waiting": 1}
    assert snap["overcommit_episodes"] == 2 and snap["overcommit_waits"] == 3
    assert snap["footprint_error"]["mean_error_pages"] == 15.0
    assert snap["footprint_error"]["max_error_pages"] == 30
    assert snap["footprint_error"]["min_error_pages"] == 0
    assert snap["peak_host_used_pages"] == 900
    assert snap["peak_held_plus_running_projected_pages"] == 950
    assert snap["peak_held_plus_running_projected_over_capacity"] == 0.95
    assert snap["gpu_total_pages"] == 8000


# --------------------------------------------------------------------------- #
#  Worker glue (real methods against fakes)
# --------------------------------------------------------------------------- #


class _Book:
    def __init__(self):
        self.tokens: Dict[int, List[int]] = {}
        self.by_uuid: Dict[str, int] = {}
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
        out.copy_(torch.tensor(self.tokens[slot][start:start + length], dtype=out.dtype))

    def release(self, slot):
        del self.tokens[slot]


_METHODS = load_worker_methods(
    [
        "_kv_budget_pages", "_kv_running_footprint", "_kv_held_by_node",
        "_kv_footprint_state", "_kv_footprint_candidates", "_apply_footprint_admission",
        "_kv_admission_would_admit", "_note_kv_footprint_sample",
        "_note_kv_footprint_completion", "_publish_kv_admission_metrics",
        "_session_resume_plan", "_note_admission_wait_on_held_kv",
        "_publish_session_hold_metrics", "_release_held_kv_for_admission",
    ],
    {
        "kv_admission_lib": ka,
        "session_hold_lib": session_hold_lib,
        "logging": logging, "torch": torch, "time": time, "replace": replace,
        "SequenceStatus": SequenceStatus, "SequenceEntry": SequenceEntry,
        "List": List, "Optional": Optional, "Sequence": Sequence,
        "Dict": Dict, "Tuple": Tuple,
    },
)


class _Worker:
    world_size = 2
    local_world_size = 2
    enable_host_kv_eviction = True
    _KV_STATUSES = (
        SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
        SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD,
    )

    def __init__(self, *, host_total=1000, gpu_total=(-1, -1)):
        self.rank = 0
        self.global_batch = SequenceBatch()
        self._trajectory_book = _Book()
        self._session_hold = SessionHoldBook(page_tokens=PAGE)
        self._kv_admission = ka.KVAdmissionStats(margin_percent=10.0, page_tokens=PAGE)
        self._kv_admission_totals = ((host_total,), tuple(gpu_total))
        self._kv_admission_plan = None
        self._kv_admission_last_log = None
        self._kv_admission_last_publish = 0.0
        self._kv_resume_keep_cache = {}
        self._prefill_admission_live = False
        self._prefill_admission_selection = None
        self._response_queue = None
        self.dropped = []

    def __getattr__(self, name):
        if name in _METHODS:
            return _METHODS[name].__get__(self)
        raise AttributeError(name)

    def _get_num_nodes(self):
        return 1

    def _trajectory_tokens(self, seq, *, start=0, length=None):
        tokens = self._trajectory_book.tokens[self._trajectory_book.slot_for(seq.uuid)]
        length = len(tokens) - start if length is None else length
        return torch.tensor(tokens[start:start + length], dtype=torch.int32)

    def _drop_unstarted_request(self, seq, code, message):
        self.dropped.append((seq.uuid, code))
        self.global_batch.remove_sequence(seq.uuid)

    def add(self, uuid, idx, tokens, *, max_decode, rank=0, session=None, status=()):
        seq = SequenceEntry(uuid=uuid, global_idx=idx, prompt_length=len(tokens),
                            max_decode_length=max_decode)
        seq.assigned_rank = rank
        seq.session_id = session
        self.global_batch.add_sequence(seq)
        self._trajectory_book.bind(uuid, tokens)
        for target in status:
            self.global_batch.update_status(uuid, target)
        return seq

    def hold(self, session, tokens, *, rank=0, request="turn-prev"):
        slot = self._trajectory_book.bind(request, tokens)
        pages = -(-len(tokens) // PAGE)
        self._session_hold.add(HeldSession(
            session_id=session, traj_id="t", request_id=request, rank=rank,
            trajectory_slot=slot, kv_tokens=len(tokens), page_tokens=PAGE,
            page_ids=tuple(range(pages)),
        ))


def _selection(worker, uuids):
    candidates = []
    for uuid in uuids:
        seq = worker.global_batch.get_sequence(uuid)
        candidates.append(PrefillCandidate(
            uuid=uuid, assigned_rank=seq.assigned_rank, node_id=0,
            is_evicted=seq.status == SequenceStatus.EVICTED,
            global_idx=seq.global_idx, total_decoded_before_eviction=0,
            prompt_length=seq.prompt_length, kv_token_budget=seq.kv_token_budget,
            page_size=PAGE,
        ))
    return PrefillSelectionRequest(
        candidates=tuple(candidates), per_node_host_free=(1000,), chunk_size=8192,
        num_nodes=1, gpus_per_node=2, initial_gpu_page_buffer=64,
    )


def test_worker_counts_running_budgets_and_held_pages():
    worker = _Worker()
    # A decoding turn: budget 640 + 640 tokens = 20 pages on rank 1.
    worker.add("run", 0, [1] * 640, max_decode=640, rank=1, status=(
        SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE))
    worker.add("queued", 1, [1] * 64, max_decode=64)  # no KV yet: not counted
    worker.hold("s1", list(range(200)))  # 4 pages
    host, gpu = worker._kv_running_footprint()
    assert host == [20] and gpu == [0, 20]
    assert worker._kv_held_by_node() == [4]


def test_worker_admits_a_resumed_turn_first_with_its_kept_pages_discounted():
    worker = _Worker(host_total=200)
    held = list(range(1, 641))  # 640 tokens = 10 held pages
    worker.hold("s1", held)
    # The resumed turn's prompt extends the held tokens by 128 (LCP = 640).
    worker.add("resumed", 5, held + [7] * 128, max_decode=256, session="s1")
    worker.add("new", 1, [9] * 2560, max_decode=1024)  # 56 pages
    # 200 total - 20 margin - 10 held = 170 free; the resumed increment is
    # ceil(1024 / 64) - 10 = 6 pages, the new turn 56.
    selection, changed = worker._apply_footprint_admission(
        _selection(worker, ["new", "resumed"]), [190], [200], [-1, -1],
    )
    assert not changed
    assert [c.uuid for c in selection.candidates] == ["resumed", "new"]
    resumed = selection.candidates[0]
    assert resumed.cached_prefix_pages == 10  # the held pages it attaches
    assert resumed.prefill_order == 0
    assert worker._kv_admission.admitted == {"new": 1, "resumed": 1}


def test_worker_fails_a_never_fitting_request_by_name():
    worker = _Worker(host_total=100)
    worker.add("huge", 0, [1] * 6400, max_decode=64)  # 101 pages > 90 usable
    _, changed = worker._apply_footprint_admission(
        _selection(worker, ["huge"]), [100], [100], [-1, -1],
    )
    assert changed
    assert worker.dropped == [("huge", ka.REJECT_NEVER_FITS)]
    assert worker._kv_admission.rejected_never_fits == 1


def test_worker_overcommit_is_counted_and_never_drops_held_kv(caplog):
    worker = _Worker(host_total=100)
    worker.hold("s1", list(range(64 * 45)))   # 45 pages
    worker.hold("s2", list(range(64 * 44)), request="turn-s2")  # 44 pages
    # 100 total - 10 margin - 89 held = 1 free page. s1's next turn needs
    # 50 - 45 = 5 more pages; nothing runs.
    worker.add("resumed", 3, list(range(64 * 45)) + [1] * 64, max_decode=256,
               session="s1")
    worker.add("new", 4, [1] * 64, max_decode=0)
    selection, changed = worker._apply_footprint_admission(
        _selection(worker, ["resumed", "new"]), [11], [100], [-1, -1],
    )
    assert not changed and selection.candidates == ()
    assert worker._kv_admission_plan.overcommit
    worker._prefill_admission_live = False
    with caplog.at_level(logging.ERROR):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(time, "sleep", lambda seconds: None)
            assert worker._note_admission_wait_on_held_kv(0) is True
    assert "OVERCOMMIT" in caplog.text
    assert worker._kv_admission.overcommit_episodes == 1
    assert set(worker._session_hold.sessions) == {"s1", "s2"}  # nothing dropped
    assert worker._kv_admission.deferred == {
        ka.DEFER_OVERCOMMIT: 1, ka.DEFER_RESUMED_FIRST: 1,
    }


def test_decode_breaks_for_prefill_only_when_a_wave_would_admit():
    worker = _Worker(host_total=100)
    worker.add("run", 0, [1] * 64 * 80, max_decode=64, status=(
        SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE))
    worker.add("queued", 1, [1] * 64 * 9, max_decode=64)  # 10 pages; 9 free
    assert worker._kv_admission_would_admit() is False
    worker.global_batch.get_sequence("queued").kv_token_budget = 64 * 9
    assert worker._kv_admission_would_admit() is True


def test_completion_records_projected_against_actual_footprint():
    worker = _Worker()
    seq = worker.add("done", 0, [1] * 640, max_decode=4096)
    seq.current_context_length = 640 + 300
    worker._note_kv_footprint_completion(seq)
    error = worker._kv_admission.snapshot()["footprint_error"]
    assert error["projected_pages_sum"] == 74 and error["actual_pages_sum"] == 15


# --------------------------------------------------------------------------- #
#  A2 audit H4: a resumed turn is charged only what its hold does not cover
# --------------------------------------------------------------------------- #


def test_h4_a_session_holding_more_than_half_the_pool_can_resume():
    # 100-page Host pool; the session holds 60 pages (3840 tokens). Its next
    # turn appends 64 tokens and may decode 64: budget 62 pages. Charged in
    # full (the A2 behaviour) it needs 62 > 40 physically free pages and waits
    # forever behind its own hold; charged its suffix + output it needs 2.
    worker = _Worker(host_total=100)
    held = list(range(1, 64 * 60 + 1))
    worker.hold("s1", held)
    worker.add("resumed", 1, held + [7] * 64, max_decode=64, session="s1")
    selection, changed = worker._apply_footprint_admission(
        _selection(worker, ["resumed"]), [40], [100], [-1, -1],
    )
    assert not changed
    assert [c.uuid for c in selection.candidates] == ["resumed"]
    assert selection.candidates[0].cached_prefix_pages == 60
    physical = replace(selection, per_node_host_free=(40,))
    assert PrefillScheduler.select_prefill_batch(physical) == ["resumed"]
    # Without the discount the same physical selection admits nothing.
    undiscounted = replace(physical, candidates=tuple(
        replace(c, cached_prefix_pages=0) for c in physical.candidates))
    assert PrefillScheduler.select_prefill_batch(undiscounted) == []


def test_h4_a_resumed_turn_is_charged_its_suffix_not_unreused_held_pages():
    # The new prompt diverges from the held tokens after 2 pages: only those
    # 2 pages are reused, so the turn is charged budget - 2 (held pages past
    # the LCP are freed only at commit and are not credited).
    worker = _Worker(host_total=1000)
    held = list(range(1, 64 * 10 + 1))
    worker.hold("s1", held)
    prompt = held[: 64 * 2] + [5] * (64 * 3)
    worker.add("resumed", 1, prompt, max_decode=64, session="s1")
    candidates, kinds, keep = worker._kv_footprint_candidates(["resumed"])
    assert kinds == {"resumed": ka.KIND_RESUMED} and keep == {"resumed": 2}
    assert candidates[0].budget_pages == 6 and candidates[0].host_pages == 4


def test_h4_an_empty_wave_not_caused_by_held_kv_keeps_the_guard_state():
    # Holds exist and nothing runs, but footprint admission deferred nothing:
    # the wave was emptied elsewhere (physical reservation, slot limits). The
    # no-progress guard must keep its state so such a stall fails by name.
    worker = _Worker(host_total=1000)
    worker.hold("s1", list(range(64 * 10)))
    worker.add("new", 1, [1] * 64, max_decode=64)
    worker._apply_footprint_admission(
        _selection(worker, ["new"]), [990], [1000], [-1, -1],
    )
    assert worker._kv_admission_plan.admitted == ("new",)
    assert worker._note_admission_wait_on_held_kv(0) is False
    # No plan at all (admission did not run this pass): not a hold wait.
    worker._kv_admission_plan = None
    assert worker._note_admission_wait_on_held_kv(0) is False


def test_h4_an_empty_wave_deferred_by_held_kv_is_a_paced_wait(monkeypatch):
    worker = _Worker(host_total=100)
    worker.hold("s1", list(range(64 * 85)))  # 85 held + 10 margin: 5 free
    worker.add("new", 1, [1] * 64 * 8, max_decode=64)  # 9 pages
    worker._apply_footprint_admission(
        _selection(worker, ["new"]), [15], [100], [-1, -1],
    )
    assert worker._kv_admission_plan.deferred == {"new": ka.DEFER_HOST}
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    assert worker._note_admission_wait_on_held_kv(0) is True
    assert slept == [0.2]
    assert worker._session_hold.admission_waits == 1
    assert set(worker._session_hold.sessions) == {"s1"}  # nothing dropped

