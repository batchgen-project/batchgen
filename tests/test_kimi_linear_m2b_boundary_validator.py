# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear                                                       #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""M2b Tier-1 (CPU): group-aware boundary-payload validator (Option 1, A10).

``validate_boundary_payload_alignment`` gates the decode-boundary all-gather.
G==1 keeps the validated single-owner checks verbatim; G>1 (Option 1 unified
resident TP) replicates a sequence onto ALL G ranks of its ``decode_dp_group``,
so a UUID must be reported by EXACTLY the G contiguous ranks [g*G,(g+1)*G) and
ownership is group membership (rank//G == g), not assigned_rank==rank.

Each G>1 acceptance test carries a NON-VACUITY control: the same replicated
payload FAILS under the old single-owner rule (group_size=1), and a corrupted
group report FAILS under G>1.

Run: python -m pytest tests/test_kimi_linear_m2b_boundary_validator.py -x -q -rA
"""

from __future__ import annotations

import pytest

from batchgen.continuous_batching import validate_boundary_payload_alignment


def _pl(seq_state, candidate_state=None):
    return {
        "free_pages": 100,
        "seq_state": seq_state,
        "candidate_state": candidate_state or {},
    }


# --------------------------------------------------------------------------- #
#  G==1 : validated single-owner path unchanged                               #
# --------------------------------------------------------------------------- #

def test_g1_single_owner_ok():
    payloads = [
        _pl({"a": {"assigned_rank": 0}}),
        _pl({"b": {"assigned_rank": 1}}),
    ]
    validate_boundary_payload_alignment(["a", "b"], payloads, group_size=1)
    # default group_size is 1
    validate_boundary_payload_alignment(["a", "b"], payloads)


def test_g1_duplicate_active_fails():
    payloads = [
        _pl({"a": {"assigned_rank": 0}}),
        _pl({"a": {"assigned_rank": 0}}),   # reported by two ranks
    ]
    with pytest.raises(RuntimeError, match="multiple ranks"):
        validate_boundary_payload_alignment(["a"], payloads, group_size=1)


def test_g1_wrong_owner_fails():
    payloads = [
        _pl({"a": {"assigned_rank": 1}}),   # rank 0 reports owner=1
        _pl({}),
    ]
    with pytest.raises(RuntimeError, match="owner/rank mismatch"):
        validate_boundary_payload_alignment(["a"], payloads, group_size=1)


def test_g1_missing_active_fails():
    payloads = [_pl({}), _pl({})]
    with pytest.raises(RuntimeError, match="missing from gathered"):
        validate_boundary_payload_alignment(["a"], payloads, group_size=1)


# --------------------------------------------------------------------------- #
#  G>1 : Option 1 replicated group ownership                                   #
# --------------------------------------------------------------------------- #

def _replicated_single_group(uuids, world=8, G=8):
    """Gate config: one serve-group; all `world` ranks report all uuids, g=0."""
    payloads = []
    for r in range(world):
        seq_state = {u: {"assigned_rank": 0, "decode_dp_group": 0} for u in uuids}
        payloads.append(_pl(seq_state))
    return payloads


def test_g8_replicated_single_group_ok():
    """world=8, G=8, one group: all 8 ranks report all 4 seqs -> OK under G=8."""
    uuids = ["a", "b", "c", "d"]
    payloads = _replicated_single_group(uuids)
    validate_boundary_payload_alignment(uuids, payloads, group_size=8)


def test_g8_replicated_fails_under_old_single_owner_rule():
    """NON-VACUITY: the very same replicated payload is REJECTED by the old
    single-owner rule (group_size=1) -- i.e. group_size genuinely changes the
    check, it is not vacuously passing."""
    uuids = ["a", "b", "c", "d"]
    payloads = _replicated_single_group(uuids)
    with pytest.raises(RuntimeError):
        validate_boundary_payload_alignment(uuids, payloads, group_size=1)


def test_g8_missing_group_member_fails():
    """A seq reported by only 7 of its group's 8 ranks is caught."""
    uuids = ["a"]
    payloads = _replicated_single_group(uuids)      # 8 ranks report "a"
    payloads[7]["seq_state"] = {}                   # rank 7 drops it
    with pytest.raises(RuntimeError, match="not reported by exactly their decode group"):
        validate_boundary_payload_alignment(uuids, payloads, group_size=8)


def test_g8_cross_group_report_fails():
    """A rank OUTSIDE the seq's group reporting it is a membership violation."""
    world, G = 16, 8
    uuids = ["a"]
    payloads = []
    for r in range(world):
        # "a" belongs to group 0 (ranks 0..7); rank 8 (group 1) wrongly reports it
        if r < 8 or r == 8:
            payloads.append(_pl({"a": {"assigned_rank": 0, "decode_dp_group": 0}}))
        else:
            payloads.append(_pl({}))
    with pytest.raises(RuntimeError):
        validate_boundary_payload_alignment(uuids, payloads, group_size=G)


def test_g8_two_groups_ok():
    """world=16, G=8, num_dp=2: group0 seqs on ranks 0..7, group1 seqs on 8..15."""
    world, G = 16, 8
    uuids = ["a", "b"]
    payloads = []
    for r in range(world):
        seq_state = {}
        if r // G == 0:
            seq_state["a"] = {"assigned_rank": 0, "decode_dp_group": 0}
        if r // G == 1:
            seq_state["b"] = {"assigned_rank": 8, "decode_dp_group": 1}
        payloads.append(_pl(seq_state))
    validate_boundary_payload_alignment(uuids, payloads, group_size=G)


def test_g8_wrong_group_id_fails():
    """A seq stamped with the wrong group id (reported by the wrong rank band)."""
    world, G = 16, 8
    uuids = ["b"]
    payloads = []
    for r in range(world):
        # "b" claims group 1 but is reported by group-0 ranks 0..7
        if r // G == 0:
            payloads.append(_pl({"b": {"assigned_rank": 0, "decode_dp_group": 1}}))
        else:
            payloads.append(_pl({}))
    with pytest.raises(RuntimeError):
        validate_boundary_payload_alignment(uuids, payloads, group_size=G)


def test_g8_candidate_group_ok_and_overlap_fails():
    """Load candidates follow the same group cardinality; an active/candidate
    overlap is still fatal."""
    world, G = 8, 8
    # candidate "c" reported by all 8 ranks of group 0 with a valid load status
    ok_payloads = []
    for r in range(world):
        ok_payloads.append(_pl(
            {"a": {"assigned_rank": 0, "decode_dp_group": 0}},
            {"c": {"assigned_rank": 0, "decode_dp_group": 0, "status": "PREFILLED"}},
        ))
    validate_boundary_payload_alignment(["a"], ok_payloads, group_size=G)

    # now "a" is ALSO a candidate -> overlap
    bad_payloads = []
    for r in range(world):
        bad_payloads.append(_pl(
            {"a": {"assigned_rank": 0, "decode_dp_group": 0}},
            {"a": {"assigned_rank": 0, "decode_dp_group": 0, "status": "PREFILLED"}},
        ))
    with pytest.raises(RuntimeError, match="also reported as load candidates"):
        validate_boundary_payload_alignment(["a"], bad_payloads, group_size=G)


def test_g8_candidate_bad_status_fails():
    world, G = 8, 8
    payloads = []
    for r in range(world):
        payloads.append(_pl(
            {"a": {"assigned_rank": 0, "decode_dp_group": 0}},
            {"c": {"assigned_rank": 0, "decode_dp_group": 0, "status": "IN_DECODE"}},
        ))
    with pytest.raises(RuntimeError, match="invalid status"):
        validate_boundary_payload_alignment(["a"], payloads, group_size=G)
