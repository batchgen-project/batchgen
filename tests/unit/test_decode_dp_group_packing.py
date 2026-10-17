"""Sharing-aware serve-group assignment for G>1 ``packing_v2`` batches.

At G>1 a group prefills its members through its own in-wave prefix pool, so a
prefix family split across groups recomputes its shared prefix once per group.
``_assign_decode_dp_groups`` therefore packs ``packing_v2`` batches onto groups
with the wave packer, and keeps the L^2 balancer for everything else. The worker
methods are compiled straight out of the module's AST onto a namespace of fakes,
so nothing here needs CUDA; the packer and the L^2 balancer are the real ones.
"""

import ast
import functools
import itertools
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List

import pytest
import torch

from batchgen.decode_dp_group import (
    assign_decode_dp_groups,
    evict_from_decode_dp_group,
)
from batchgen.prefix_reuse.rank_packing import (
    _feature_score,
    pack_groups_for_sharing,
    pack_wave_ranks,
    unit_features,
)
from batchgen.prefix_reuse.wave_plan import _build_tree


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"

B = 16  # block tokens
CHUNK = 131_072
SequenceStatus = SimpleNamespace(
    QUEUEING="QUEUEING", IN_PREFILL="IN_PREFILL", EVICTED="EVICTED",
    IN_DECODE="IN_DECODE", COMPLETED="COMPLETED",
)


@functools.lru_cache(maxsize=1)
def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "BatchGenWorker")


def _worker_methods(*names):
    wanted = [node for node in _worker_class().body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(wanted) == len(names), [node.name for node in wanted]
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {"List": List, "torch": torch, "SequenceStatus": SequenceStatus}
    exec(compile(module, str(WORKER), "exec"), namespace)
    return {name: namespace[name] for name in names}


_METHODS = (
    "_assign_decode_dp_groups",
    "_pack_decode_dp_groups_for_sharing",
    "_prefix_rank_packing_mode",
    "_cached_prefix_rank_cost_profile",
    # The A/B arm the packer call sites resolve per batch; these fakes carry no
    # batchgen_debug, so it is always the default (profile) arm.
    "_prefix_rank_cost_profile_disabled",
)


class _Batch:
    """Insertion-ordered sequence table, iterated identically on every rank."""

    def __init__(self, seqs):
        self._seqs = {seq.uuid: seq for seq in seqs}

    def __iter__(self):
        return iter(list(self._seqs.values()))

    def get_sequence(self, uuid):
        return self._seqs.get(uuid)


def _seq(uuid, tokens, *, status=SequenceStatus.QUEUEING, group=None):
    return SimpleNamespace(
        uuid=uuid, tokens=list(tokens), prompt_length=len(tokens), status=status,
        decode_dp_group=group, evicted_token_ids=None, evicted_from_dp_group=None,
    )


def _evicted(uuid, tokens, *, decoded_on):
    """A sequence evicted while decoding on group ``decoded_on``."""
    seq = _seq(uuid, tokens, status=SequenceStatus.IN_DECODE, group=decoded_on)
    evict_from_decode_dp_group(seq)
    seq.status = SequenceStatus.EVICTED
    return seq


def _reader_refusing(*uuids):
    """A trajectory reader that fails the test if it reads ``uuids``."""
    def read(seq, length=None, **_):
        if seq.uuid in uuids:
            raise AssertionError(f"read the prompt of {seq.uuid}")
        return torch.tensor(seq.tokens[:length], dtype=torch.int32)
    return read


def _worker(seqs, *, world_size=16, G=8, packing=True, pool=True, profile=None):
    worker = SimpleNamespace(
        world_size=world_size,
        global_batch=_Batch(seqs),
        # Preset: the cached lookup never reaches the CUDA device query.
        # Pre-seeded cache (keyed by the A/B arm, False = profile arm),
        # so no fake needs a CUDA device query.
        _prefix_rank_cost_profiles={False: profile},
        prefix_cache_runtime_config=SimpleNamespace(
            group_specs=[SimpleNamespace(raw_page_tokens=B)]),
        engine_config=SimpleNamespace(
            Module_Batching_Config=SimpleNamespace(prefill_micro_batch_token_cap=CHUNK)),
        _active_batchgen_debug_for_sequences=lambda batch_sequences: (
            {"prefix_rank_assignment_mode": "packing_v2"} if packing else None
        ),
        _decode_attn_tp_size=lambda: G,
        _prefill_prefix_pool_enabled=lambda: pool,
        _trajectory_tokens=lambda seq, length=None, **_: torch.tensor(
            seq.tokens[:length], dtype=torch.int32
        ),
    )
    for name, method in _worker_methods(*_METHODS).items():
        setattr(worker, name, MethodType(method, worker))
    return worker


def _family(uid, members, shared_blocks, tail_blocks):
    """``members`` prompts sharing ``shared_blocks`` blocks, private tails after."""
    shared = [uid * 100_000 + i for i in range(shared_blocks * B)]
    return [
        shared + [10_000_000 + uid * 100_000 + m * 1_000 + k
                  for k in range(tail_blocks * B + 3)]
        for m in range(members)
    ]


def _singleton(uid, blocks):
    return [50_000_000 + uid * 100_000 + k for k in range(blocks * B + 3)]


def _make(prefix, prompts, **kwargs):
    return [_seq(f"{prefix}{i}", tokens, **kwargs) for i, tokens in enumerate(prompts)]


def _span(prompts, groups, num_groups, profile=None):
    """Exact busiest-group score of an assignment, as the packer scores it."""
    tree = _build_tree(prompts, B)
    lengths = [len(tokens) for tokens in prompts]
    scores = []
    for g in range(num_groups):
        members = [i for i, owner in enumerate(groups) if owner == g]
        if not members:
            scores.append(0.0)
            continue
        feats = unit_features(tree, members, lengths, B)
        scores.append(_feature_score(profile, feats, CHUNK))
    return max(scores)


def _assigned(seqs):
    return [seq.decode_dp_group for seq in seqs]


# --------------------------------------------------------------------------- #
#  The pure wrapper                                                           #
# --------------------------------------------------------------------------- #

def test_group_packer_is_the_wave_packer_over_groups():
    prompts = _family(1, 5, 8, 1) + _family(2, 4, 6, 2) + [_singleton(3, 2)]
    fixed = ((1, 40.0),)
    expected = pack_wave_ranks(
        prompts, world_size=3, block_tokens=B, chunk_tokens=CHUNK,
        profile=None, fixed=fixed,
    ).ranks
    got = pack_groups_for_sharing(
        prompts, num_groups=3, block_tokens=B, chunk_tokens=CHUNK,
        profile=None, fixed=fixed,
    )
    assert got == expected
    with pytest.raises(ValueError):
        pack_groups_for_sharing(
            prompts, num_groups=0, block_tokens=B, chunk_tokens=CHUNK, profile=None
        )


# --------------------------------------------------------------------------- #
#  Worker integration                                                         #
# --------------------------------------------------------------------------- #

def test_two_equal_families_colocate_one_per_group():
    fam_a = _make("a", _family(1, 6, 20, 1))
    fam_b = _make("b", _family(2, 6, 20, 1))
    singles = _make("s", [_singleton(1, 1), _singleton(2, 1)])
    seqs = fam_a + fam_b + singles
    worker = _worker(seqs)
    worker._assign_decode_dp_groups([seq.uuid for seq in seqs])

    groups_a = {seq.decode_dp_group for seq in fam_a}
    groups_b = {seq.decode_dp_group for seq in fam_b}
    assert len(groups_a) == 1 and len(groups_b) == 1
    assert groups_a != groups_b
    assert all(seq.decode_dp_group in (0, 1) for seq in singles)

    # Non-vacuity: the L^2 balancer alternates equal lengths over the groups,
    # so it splits both families.
    lengths = [seq.prompt_length for seq in seqs]
    l2 = assign_decode_dp_groups(lengths, 2, prior_load=[0.0, 0.0])
    l2_by_uuid = dict(zip((seq.uuid for seq in seqs), l2))
    assert len({l2_by_uuid[seq.uuid] for seq in fam_a}) == 2
    assert len({l2_by_uuid[seq.uuid] for seq in fam_b}) == 2


@pytest.mark.parametrize(
    "shared_blocks,tail_blocks,others,expect_split",
    [
        # Private tails dominate and the other group is nearly idle.
        (2, 30, [1, 2, 1], True),
        # One unrelated prompt as heavy as the family occupies the other
        # group: splitting would stack half the family on top of it.
        (40, 1, [50], False),
    ],
    ids=["light-tail", "loaded-other-group"],
)
def test_dominant_family_splits_only_when_the_busiest_group_drops(
    shared_blocks, tail_blocks, others, expect_split
):
    family = _make("f", _family(1, 10, shared_blocks, tail_blocks))
    tail = _make("t", [_singleton(i, blocks) for i, blocks in enumerate(others)])
    seqs = family + tail
    worker = _worker(seqs)
    worker._assign_decode_dp_groups([seq.uuid for seq in seqs])

    prompts = [seq.tokens for seq in seqs]
    got = _assigned(seqs)
    n = len(family)
    # Best co-located alternative: the family on one group, the rest placed
    # however suits it best.
    together = min(
        _span(prompts, [0] * n + list(placement), 2)
        for placement in itertools.product((0, 1), repeat=len(tail))
    )
    split = len(set(got[:n])) == 2
    if split:
        assert _span(prompts, got, 2) < together
    else:
        assert _span(prompts, got, 2) <= together
    assert split == expect_split


def test_grouped_reentry_stays_pinned_and_repels_new_placement():
    pinned = _seq("p", _singleton(9, 40), status=SequenceStatus.EVICTED, group=0)
    # Already decoding on group 1: it owes no prefill, so it loads nothing,
    # although its prompt is longer still.
    decoding = _seq("d", _singleton(8, 80), status=SequenceStatus.IN_DECODE, group=1)
    fresh = _make("n", [_singleton(1, 10), _singleton(2, 10)])
    seqs = [decoding, pinned] + fresh
    worker = _worker(seqs)
    worker._assign_decode_dp_groups(["p", "n0", "n1"])

    assert pinned.decode_dp_group == 0
    assert decoding.decode_dp_group == 1
    assert _assigned(fresh) == [1, 1]

    # Non-vacuity: without the pinned load both groups start empty, and two
    # equal unrelated prompts spread over them.
    pinned_off = _seq("p", _singleton(9, 40), status=SequenceStatus.COMPLETED, group=0)
    fresh_again = _make("n", [_singleton(1, 10), _singleton(2, 10)])
    worker = _worker([decoding, pinned_off] + fresh_again)
    worker._assign_decode_dp_groups(["n0", "n1"])
    assert sorted(_assigned(fresh_again)) == [0, 1]


def test_evicted_reentry_returns_to_its_group_and_repels_new_placement():
    # Decoded on group 1, so its suffix is only in node 1's trajectory book.
    evicted = _evicted("e", _singleton(9, 40), decoded_on=1)
    assert (evicted.decode_dp_group, evicted.evicted_from_dp_group) == (None, 1)
    fresh = _make("n", [_singleton(1, 10), _singleton(2, 10)])
    worker = _worker([evicted] + fresh)
    worker._trajectory_tokens = _reader_refusing("e")
    worker._assign_decode_dp_groups(["e", "n0", "n1"])

    assert evicted.decode_dp_group == 1
    assert evicted.evicted_from_dp_group is None
    # Its pinned length-only load pushes both fresh prompts to group 0.
    assert _assigned(fresh) == [0, 0]

    # Non-vacuity: alone, two equal unrelated prompts spread over the groups.
    fresh_alone = _make("n", [_singleton(1, 10), _singleton(2, 10)])
    _worker(fresh_alone)._assign_decode_dp_groups(["n0", "n1"])
    assert sorted(_assigned(fresh_alone)) == [0, 1]


def test_reentry_without_a_recorded_group_is_refused():
    evicted = _seq("e", _singleton(1, 4), status=SequenceStatus.EVICTED)
    fresh = _make("n", [_singleton(2, 4)])
    worker = _worker([evicted] + fresh)
    worker._trajectory_tokens = _reader_refusing("e")
    with pytest.raises(RuntimeError, match="re-entering sequence e has no recorded"):
        worker._assign_decode_dp_groups(["e", "n0"])
    assert evicted.decode_dp_group is None and fresh[0].decode_dp_group is None


def test_packing_double_eviction_round_trip():
    seq = _evicted("e", _singleton(1, 6), decoded_on=1)
    other = _make("n", [_singleton(2, 6)])
    worker = _worker([seq] + other)
    worker._trajectory_tokens = _reader_refusing("e")
    worker._assign_decode_dp_groups(["e", "n0"])
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (1, None)

    # Decoding again on group 1, then evicted a second time.
    seq.status = SequenceStatus.IN_DECODE
    evict_from_decode_dp_group(seq)
    seq.status = SequenceStatus.EVICTED
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (None, 1)
    worker._assign_decode_dp_groups(["e"])
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (1, None)


def test_sequence_entry_carries_the_eviction_record():
    from batchgen.sequence import SequenceEntry

    assert "evicted_from_dp_group" in SequenceEntry.__slots__
    seq = SequenceEntry("u", 0, 8, 4)
    assert seq.evicted_from_dp_group is None
    seq.decode_dp_group = 3
    evict_from_decode_dp_group(seq)
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (None, 3)


def _l2_expected(seqs, uuids, num_dp):
    """What ``_assign_decode_dp_groups`` computed before packing existed."""
    to_assign = set(uuids)
    prior = [0.0] * num_dp
    for seq in seqs:
        if seq.uuid in to_assign or seq.decode_dp_group is None:
            continue
        prior[seq.decode_dp_group] += float(seq.prompt_length) ** 2
    new = [seq for seq in seqs if seq.uuid in to_assign and seq.decode_dp_group is None]
    return assign_decode_dp_groups(
        [seq.prompt_length for seq in new], num_dp, prior_load=prior
    )


def _l2_wave(num_dp):
    # The decoding sequence sits on the last group: the L^2 prior counts it.
    return (
        [_seq("d", _singleton(8, 30), status=SequenceStatus.IN_DECODE,
              group=num_dp - 1)]
        + _make("a", _family(1, 4, 20, 1))
        + _make("b", _family(2, 4, 20, 1))
    )


@pytest.mark.parametrize(
    "gate",
    [
        dict(packing=False),
        dict(pool=False),
        dict(world_size=8, G=8),  # num_dp == 1
    ],
    ids=["no-packing-mode", "pool-off", "single-group"],
)
def test_gate_off_keeps_the_l2_assignment(monkeypatch, gate):
    def must_not_pack(*args, **kwargs):
        raise AssertionError("the gate is off: the group packer must not run")

    monkeypatch.setattr(
        sys.modules["batchgen.prefix_reuse.rank_packing"],
        "pack_groups_for_sharing", must_not_pack,
    )
    num_dp = gate.get("world_size", 16) // gate.get("G", 8)
    seqs = _l2_wave(num_dp)
    uuids = [seq.uuid for seq in seqs[1:]]
    worker = _worker(seqs, **gate)
    worker._trajectory_tokens = must_not_pack
    expected = _l2_expected(seqs, uuids, num_dp)
    worker._assign_decode_dp_groups(uuids)
    assert _assigned(seqs[1:]) == expected


def test_gate_on_departs_from_l2_and_g1_assigns_nothing():
    seqs = _l2_wave(2)
    uuids = [seq.uuid for seq in seqs[1:]]
    expected = _l2_expected(seqs, uuids, 2)
    _worker(seqs)._assign_decode_dp_groups(uuids)
    assert _assigned(seqs[1:]) != expected

    seqs = _l2_wave(8)
    _worker(seqs, world_size=8, G=1)._assign_decode_dp_groups(uuids)
    assert _assigned(seqs[1:]) == [None] * len(uuids)


def test_assignment_is_deterministic():
    def run():
        seqs = (
            _make("a", _family(1, 7, 12, 3)) + _make("b", _family(2, 5, 30, 1))
            + _make("s", [_singleton(i, i) for i in range(1, 6)])
        )
        seqs.append(_seq("p", _singleton(9, 25), status=SequenceStatus.IN_PREFILL, group=2))
        worker = _worker(seqs, world_size=32, G=8)
        worker._assign_decode_dp_groups([seq.uuid for seq in seqs])
        return _assigned(seqs)

    first = run()
    assert all(g in range(4) for g in first)
    assert run() == first


# --------------------------------------------------------------------------- #
#  L^2 path: an evicted re-entry returns to its node                          #
# --------------------------------------------------------------------------- #

def test_l2_restamps_evicted_reentry_and_counts_it_as_prior_load():
    evicted = _evicted("e", _singleton(9, 40), decoded_on=0)
    decoding = _seq("d", _singleton(8, 5), status=SequenceStatus.IN_DECODE, group=1)
    fresh = _make("n", [_singleton(1, 10), _singleton(2, 10)])
    worker = _worker([decoding, evicted] + fresh, packing=False)
    worker._trajectory_tokens = _reader_refusing("e", "n0", "n1", "d")
    worker._assign_decode_dp_groups(["e", "n0", "n1"])

    assert (evicted.decode_dp_group, evicted.evicted_from_dp_group) == (0, None)
    # Only the fresh sequences reach the balancer, against a prior that
    # charges group 0 for the re-entry.
    prior = [float(evicted.prompt_length) ** 2, float(decoding.prompt_length) ** 2]
    expected = assign_decode_dp_groups(
        [seq.prompt_length for seq in fresh], 2, prior_load=prior
    )
    assert _assigned(fresh) == expected == [1, 1]

    # A lone re-entry recorded on group 1 returns there, although the
    # balancer would put it on group 0 of two empty groups.
    moved = _evicted("e", _singleton(9, 40), decoded_on=1)
    worker = _worker([moved], packing=False)
    worker._assign_decode_dp_groups(["e"])
    assert moved.decode_dp_group == 1


def test_l2_double_eviction_round_trip():
    seq = _evicted("e", _singleton(1, 6), decoded_on=1)
    worker = _worker([seq], packing=False)
    worker._assign_decode_dp_groups(["e"])
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (1, None)

    seq.status = SequenceStatus.IN_DECODE
    evict_from_decode_dp_group(seq)
    seq.status = SequenceStatus.EVICTED
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (None, 1)
    worker._assign_decode_dp_groups(["e"])
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (1, None)



# --------------------------------------------------------------------------- #
#  Saved-group lifecycle                                                      #
# --------------------------------------------------------------------------- #

def test_redundant_eviction_keeps_the_saved_group():
    seq = _evicted("e", _singleton(1, 6), decoded_on=1)
    # Already EVICTED and never re-grouped: evicted again, it holds no group.
    evict_from_decode_dp_group(seq)
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (None, 1)

    worker = _worker([seq])
    worker._trajectory_tokens = _reader_refusing("e")
    worker._assign_decode_dp_groups(["e"])
    assert (seq.decode_dp_group, seq.evicted_from_dp_group) == (1, None)


def _completion_cleanup_loops():
    """Every all-ranks scalar cleanup loop over completed sequences."""
    loops = []
    for node in ast.walk(_worker_class()):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id in ("completed_uuids", "completed_list")
            and "seq.host_token_capacity = 0" in ast.unparse(node)
        ):
            loops.append(node)
    return loops


def test_completion_cleanup_clears_the_saved_group():
    loops = _completion_cleanup_loops()
    # Prefill-time completions, the decode loop, and the page boundary.
    assert len(loops) == 3
    for loop in loops:
        seq = _evicted("c", _singleton(1, 6), decoded_on=1)
        seq.status = SequenceStatus.COMPLETED
        seq.gpu_pages_allocated = seq.host_pages_allocated = 4
        seq.host_owned_pages = seq.host_token_capacity = 4
        fake = SimpleNamespace(
            global_batch=_Batch([seq]), _sequences_with_gpu_kv={"c"},
        )
        module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
        exec(
            compile(module, str(WORKER), "exec"),
            {"self": fake, loop.iter.id: ["c"]},
        )
        assert seq.evicted_from_dp_group is None
        assert seq.host_token_capacity == 0
        assert fake._sequences_with_gpu_kv == set()
