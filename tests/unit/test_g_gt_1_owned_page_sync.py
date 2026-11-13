"""G>1: the serve-group leader's host_owned_pages is the group's value.

Only the leader of a serve group attaches cached prefix pages and retains
committed prompt pages out of a sequence's Host chain, so only its
``host_owned_pages`` carries those deductions. The post-prefill
``_sync_sequence_metadata`` must hand the leader's counters to the followers
(which hold the row locally) and to the ranks outside the group (which used to
take whichever group rank merged last), and the decode boundary must refuse a
group whose ranks still disagree.

The worker methods are compiled from the real source onto fakes, so nothing
here needs CUDA, NCCL or the compiled core engine.
"""

from __future__ import annotations

import logging
import types
from types import SimpleNamespace

import pytest

from batchgen.continuous_batching import (
    host_kv_owner_reports,
    validate_boundary_payload_alignment,
)
from tests.worker_method_source import WORKER_SOURCE, load_worker_methods

PAGE_TOKENS = 64
HOST_PAGES = 6


class _Captured(Exception):
    def __init__(self, payload):
        super().__init__("captured")
        self.payload = payload


class _CaptureDist:
    """Stops a rank at its gather and hands back what it would publish."""

    def all_gather_object(self, out, payload):
        raise _Captured(payload)


class _GatheredDist:
    """Answers a gather with every rank's already-captured payload."""

    def __init__(self, slots):
        self.slots = slots

    def all_gather_object(self, out, payload):
        for index, value in enumerate(self.slots):
            out[index] = value


class FakeSeq:
    """The fields ``_sync_sequence_metadata`` reads and writes, plus the bound
    checks of ``SequenceEntry.validate_metadata`` on the Host counters."""

    def __init__(self, uuid, group, owned):
        self.uuid = uuid
        self.global_idx = {"a": 100, "b": 101}[uuid]
        self.assigned_rank = 0
        self.decode_dp_group = group
        self.prompt_length = 4 * PAGE_TOKENS
        self.original_prompt_length = self.prompt_length
        self.decoded_length = 1
        self.current_context_length = self.prompt_length + 1
        self.gpu_pages_allocated = 0
        self.eos_reached = False
        self.reentry_decoded_baseline = 0
        self.max_decode_length = 32
        self.original_max_decode_length = 32
        self.host_pages_allocated = HOST_PAGES
        self.host_owned_pages = owned
        self.host_token_capacity = HOST_PAGES * PAGE_TOKENS
        self.total_decoded_before_eviction = 0
        self.validated = []

    def validate_metadata(self, context, require_owner_tensors=True,
                          allow_terminal_capacity_gap=False):
        if not 1 <= self.host_owned_pages <= self.host_pages_allocated:
            raise ValueError(
                f"{context}: host_owned_pages={self.host_owned_pages} outside "
                f"[1, {self.host_pages_allocated}]"
            )
        self.validated.append(context)

    def log_event(self, *args):
        pass


_METHODS = load_worker_methods(
    ["_sync_sequence_metadata", "_owns_host_kv", "_decode_attn_tp_size"],
    {
        "List": list,
        "logging": logging,
        "SeqEvent": SimpleNamespace(CTX_REPAIR="CTX_REPAIR"),
        "host_kv_owner_reports": host_kv_owner_reports,
    },
)


def _worker(rank, *, world_size, attn_tp_size, seqs, held):
    worker = SimpleNamespace(
        rank=rank,
        world_size=world_size,
        parallel_manager=SimpleNamespace(attn_tp_size=attn_tp_size),
        global_batch=SimpleNamespace(get_sequence=seqs.get),
        _uuid_to_local_map={uuid: index for index, uuid in enumerate(held)},
        _trajectory_book=None,
        _sync_trajectory_metadata=lambda seq, state: None,
    )
    for name, function in _METHODS.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker


def _run_sync(workers, uuids):
    """Run one world-wide ``_sync_sequence_metadata`` over sequential fakes."""
    namespace = _METHODS["_sync_sequence_metadata"].__globals__
    slots = []
    namespace["dist"] = _CaptureDist()
    for worker in workers:
        with pytest.raises(_Captured) as captured:
            worker._sync_sequence_metadata(uuids)
        slots.append(captured.value.payload)
    namespace["dist"] = _GatheredDist(slots)
    for worker in workers:
        worker._sync_sequence_metadata(uuids)
    return slots


def _g2_world(*, a_owned, b_owned):
    """World 4, G=2: ``a`` serves on group 0 (ranks 0,1), ``b`` on group 1
    (ranks 2,3). ``a_owned``/``b_owned`` give each rank's pre-sync value."""
    seqs_by_rank = {
        rank: {
            "a": FakeSeq("a", 0, a_owned[rank]),
            "b": FakeSeq("b", 1, b_owned[rank]),
        }
        for rank in range(4)
    }
    workers = [
        _worker(rank, world_size=4, attn_tp_size=2, seqs=seqs_by_rank[rank],
                held=["a"] if rank < 2 else ["b"])
        for rank in range(4)
    ]
    return workers, seqs_by_rank


# --------------------------------------------------------------------------- #
#  (i) the leader's post-retain count reaches every rank through the sync
# --------------------------------------------------------------------------- #


def test_leader_retain_deduction_reaches_followers_and_other_groups():
    # After prefill: the leaders (0 and 2) attached/retained prefix pages and
    # own fewer pages; the followers (1 and 3) still hold the reservation
    # commit's count. Rank 1 reports ``a`` LAST, so a last-writer merge would
    # hand ranks 2 and 3 the follower's 5 -- the overcount under test.
    workers, seqs_by_rank = _g2_world(
        a_owned={0: 3, 1: 5, 2: 5, 3: 5},
        b_owned={0: 6, 1: 6, 2: 4, 3: 6},
    )

    slots = _run_sync(workers, ["a", "b"])

    # The payload names the group, so receivers can find its leader's report.
    assert slots[1]["a"]["decode_dp_group"] == 0
    assert slots[1]["a"]["host_owned_pages"] == 5  # stale follower report
    for rank in range(4):
        assert seqs_by_rank[rank]["a"].host_owned_pages == 3, rank
        assert seqs_by_rank[rank]["b"].host_owned_pages == 4, rank
        assert seqs_by_rank[rank]["a"].host_pages_allocated == HOST_PAGES
    # The follower re-validated the adopted counters; the leader did not move.
    assert "_sync_sequence_metadata/host_kv_owner" in (
        seqs_by_rank[1]["a"].validated[-1]
    )
    assert not any(
        "host_kv_owner" in context for context in seqs_by_rank[0]["a"].validated
    )


def test_synced_group_passes_the_boundary_equality_check():
    workers, seqs_by_rank = _g2_world(
        a_owned={0: 3, 1: 5, 2: 5, 3: 5},
        b_owned={0: 6, 1: 6, 2: 4, 3: 6},
    )
    _run_sync(workers, ["a", "b"])

    payloads = [
        {
            "free_pages": 10,
            "seq_state": {
                uuid: {
                    "decode_dp_group": seqs_by_rank[rank][uuid].decode_dp_group,
                    "host_owned_pages": seqs_by_rank[rank][uuid].host_owned_pages,
                }
                for uuid in (["a"] if rank < 2 else ["b"])
            },
            "candidate_state": {},
        }
        for rank in range(4)
    ]
    validate_boundary_payload_alignment(["a", "b"], payloads, group_size=2)


# --------------------------------------------------------------------------- #
#  (ii) the boundary refuses a group that disagrees, and names it
# --------------------------------------------------------------------------- #


def _group_payloads(owned_by_rank):
    return [
        {
            "free_pages": 10,
            "seq_state": {
                "a": {"decode_dp_group": 0, "host_owned_pages": owned},
            },
            "candidate_state": {},
        }
        for owned in owned_by_rank
    ]


def test_boundary_raises_on_seeded_owned_page_divergence():
    # Exactly the pre-fix state: leader post-retain, follower pre-retain.
    with pytest.raises(RuntimeError) as raised:
        validate_boundary_payload_alignment(
            ["a"], _group_payloads([3, 5]), group_size=2
        )
    message = str(raised.value)
    assert message.startswith("[SCHED_INVARIANT]")
    assert "HOST_OWNED_PAGES diverged across serve-group ranks" in message
    assert "a(group=0, host_owned_pages_by_rank={0: 3, 1: 5})" in message


def test_boundary_accepts_agreeing_group():
    validate_boundary_payload_alignment(
        ["a"], _group_payloads([3, 3]), group_size=2
    )


def test_boundary_g1_has_no_cross_rank_owned_check():
    # G==1: one reporter per sequence, nothing to compare.
    payloads = [
        {"free_pages": 10, "seq_state": {"a": {"assigned_rank": 0,
                                               "host_owned_pages": 3}},
         "candidate_state": {}},
        {"free_pages": 10, "seq_state": {"b": {"assigned_rank": 1,
                                               "host_owned_pages": 5}},
         "candidate_state": {}},
    ]
    validate_boundary_payload_alignment(["a", "b"], payloads, group_size=1)


# --------------------------------------------------------------------------- #
#  (iii) the canonical-report helper
# --------------------------------------------------------------------------- #


def test_owner_reports_pick_the_group_leader_not_the_last_writer():
    states = [
        {"a": {"decode_dp_group": 0, "host_owned_pages": 3}},
        {"a": {"decode_dp_group": 0, "host_owned_pages": 5}},
        {"b": {"decode_dp_group": 1, "host_owned_pages": 4}},
        {"b": {"decode_dp_group": 1, "host_owned_pages": 6}},
    ]
    reports = host_kv_owner_reports(states, group_size=2)
    assert reports["a"]["host_owned_pages"] == 3
    assert reports["b"]["host_owned_pages"] == 4


def test_owner_reports_raise_when_the_leader_did_not_report():
    states = [
        None,
        {"a": {"decode_dp_group": 0, "host_owned_pages": 5}},
    ]
    with pytest.raises(RuntimeError, match=r"HOST_KV_OWNER report missing.*"
                                           r"a\(group=0, leader_rank=0\)"):
        host_kv_owner_reports(states, group_size=2)


def test_owner_reports_ungrouped_row_keeps_its_report():
    # Evicted/completed rows drop their group after every rank zeroed them.
    states = [
        {"a": {"decode_dp_group": None, "host_owned_pages": 0}},
        {"a": {"decode_dp_group": None, "host_owned_pages": 0}},
    ]
    assert host_kv_owner_reports(states, group_size=2)["a"][
        "host_owned_pages"
    ] == 0


def test_owner_reports_g1_is_the_single_reporter():
    states = [
        {"a": {"assigned_rank": 0, "host_owned_pages": 3}},
        {"b": {"assigned_rank": 1, "host_owned_pages": 5}},
    ]
    reports = host_kv_owner_reports(states, group_size=1)
    assert reports == {"a": states[0]["a"], "b": states[1]["b"]}


# --------------------------------------------------------------------------- #
#  (iv) G==1 merge is unchanged
# --------------------------------------------------------------------------- #


def test_g1_sync_owner_keeps_and_peer_adopts():
    seqs_by_rank = {rank: {"a": FakeSeq("a", None, 6 if rank else 3)}
                    for rank in range(2)}
    workers = [
        _worker(rank, world_size=2, attn_tp_size=1, seqs=seqs_by_rank[rank],
                held=["a"] if rank == 0 else [])
        for rank in range(2)
    ]
    _run_sync(workers, ["a"])
    assert seqs_by_rank[0]["a"].host_owned_pages == 3
    assert seqs_by_rank[1]["a"].host_owned_pages == 3
    assert not any(
        "host_kv_owner" in context for context in seqs_by_rank[0]["a"].validated
    )


# --------------------------------------------------------------------------- #
#  (v) ordering: the post-prefill sync runs after the leader-only commit
# --------------------------------------------------------------------------- #


def test_post_prefill_sync_follows_the_prefix_commit():
    source = WORKER_SOURCE.read_text()
    generate = source[source.index("\tdef generate(self):"):]
    generate = generate[: generate.index("\n\tdef ", 1)]
    commit = generate.index("self._commit_prefix_cache_for_prefill(")
    finish = generate.index(
        "self._finish_prefill_completed_sequences(prefill_uuids)"
    )
    assert commit < finish

    finish_src = source[
        source.index("\tdef _finish_prefill_completed_sequences("):
    ]
    finish_src = finish_src[: finish_src.index("\n\tdef ", 1)]
    body = finish_src[finish_src.index('"""', finish_src.index('"""') + 3) + 3:]
    # The sync is the first collective, ahead of any branch on its result.
    assert body.index("self._sync_sequence_metadata(prefill_uuids)") < body.index(
        "completed_uuids = []"
    )
