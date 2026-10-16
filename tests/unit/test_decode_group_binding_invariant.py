"""The serve-group map, the local rows and the KDA state must agree (G>1).

``rank_in_decode_group`` resolves ownership from each rank's OWN copy of the
``decode_dp_group`` map, while ``_get_local_indices_for_uuids`` resolves a wave
from ``_uuid_to_local_map`` alone and silently skips what it cannot find. So
two independent failures used to surface only as ``KeyError: Sequence <n> has
no KDA state item`` from ``prepare_decode_step``, on a rank that never ran the
sequence's prefill: a map that diverged across ranks, and a local row that
disagrees with the map. Both are checked at the seam that creates them here.

Worker methods are compiled straight out of the module's AST onto a namespace
of fakes (same approach as ``test_decode_dp_group_packing.py``), so nothing
here needs CUDA; the digest and the disagreement locator are the real ones.
"""

import ast
import functools
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List

import pytest
import torch

from batchgen.decode_dp_group import (
    decode_group_plan_digest,
    first_group_disagreement,
)


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


@functools.lru_cache(maxsize=1)
def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "BatchGenWorker")


def _worker_methods(*names, namespace=None):
    wanted = [node for node in _worker_class().body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(wanted) == len(names), [node.name for node in wanted]
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    scope = {"List": List, "torch": torch, "Optional": object}
    scope.update(namespace or {})
    exec(compile(module, str(WORKER), "exec"), scope)
    return {name: scope[name] for name in names}


class _Dist:
    """``dist.all_gather`` replaced by a scripted per-round answer."""

    def __init__(self, rounds):
        self.rounds = [list(values) for values in rounds]
        self.calls = 0

    def all_gather(self, out, tensor):
        if self.calls >= len(self.rounds):
            raise AssertionError("unexpected extra all_gather")
        values = self.rounds[self.calls]
        self.calls += 1
        assert len(out) == len(values), (len(out), len(values))
        for slot, value in zip(out, values):
            slot.copy_(torch.tensor(value, dtype=slot.dtype))


class _Batch:
    def __init__(self, seqs):
        self._by_uuid = {seq.uuid: seq for seq in seqs}

    def get_sequence(self, uuid):
        return self._by_uuid.get(uuid)

    def __iter__(self):
        return iter(self._by_uuid.values())


def _seq(uuid, global_idx, group, *, assigned_rank=0):
    return SimpleNamespace(
        uuid=uuid, global_idx=global_idx, decode_dp_group=group,
        assigned_rank=assigned_rank, prompt_length=128,
    )


def _worker(seqs, *, rank, world_size, G, dist_obj=None, bound=()):
    methods = _worker_methods(
        "_check_decode_dp_group_plan_identical",
        "_check_local_binding_matches_group",
        "_owns_local_sequence",
        namespace={"dist": dist_obj} if dist_obj is not None else None,
    )
    worker = SimpleNamespace(
        rank=rank, world_size=world_size, torch_device=torch.device("cpu"),
        global_batch=_Batch(seqs),
        _uuid_to_local_map={uuid: index for index, uuid in enumerate(bound)},
    )
    worker._decode_attn_tp_size = lambda: G
    for name, function in methods.items():
        setattr(worker, name, MethodType(function, worker))
    return worker


# --------------------------------------------------------------- digest --- #


def test_digest_depends_on_the_map_not_the_enumeration_order():
    forward = [(0, 0), (1, 1), (2, 0), (3, 1)]
    shuffled = [(2, 0), (0, 0), (3, 1), (1, 1)]
    assert decode_group_plan_digest(forward) == decode_group_plan_digest(shuffled)


def test_digest_separates_one_moved_sequence():
    base = [(0, 0), (1, 1), (2, 0)]
    moved = [(0, 0), (1, 0), (2, 0)]
    assert decode_group_plan_digest(base) != decode_group_plan_digest(moved)


def test_digest_separates_unstamped_from_group_zero():
    assert (decode_group_plan_digest([(7, None)])
            != decode_group_plan_digest([(7, 0)]))


def test_digest_separates_a_missing_sequence():
    assert (decode_group_plan_digest([(0, 0), (1, 0)])
            != decode_group_plan_digest([(0, 0)]))


def test_first_disagreement_is_none_when_every_rank_agrees():
    assert first_group_disagreement([0, 1, 2], [[0, 1, 0], [0, 1, 0]]) is None


def test_first_disagreement_names_the_lowest_diverging_sequence():
    named = first_group_disagreement(
        [4, 5, 6], [[0, 1, 1], [0, 0, 1], [0, 1, 1]]
    )
    assert named == (5, (1, 0, 1))


def test_first_disagreement_refuses_ragged_views():
    with pytest.raises(ValueError, match="entries for 3 sequences"):
        first_group_disagreement([0, 1, 2], [[0, 1, 0], [0, 1]])


# ------------------------------------------------- map agreement check --- #


def _two_group_seqs():
    return [_seq(f"u{i}", i, i % 2) for i in range(4)]


def test_identical_maps_pass_with_one_collective():
    seqs = _two_group_seqs()
    digest = decode_group_plan_digest(
        [(seq.global_idx, seq.decode_dp_group) for seq in seqs]
    )
    fake = _Dist([[[digest, 4], [digest, 4]]])
    worker = _worker(seqs, rank=0, world_size=2, G=8, dist_obj=fake)
    worker._check_decode_dp_group_plan_identical([seq.uuid for seq in seqs])
    assert fake.calls == 1


def test_diverged_map_names_the_first_disagreeing_sequence():
    seqs = _two_group_seqs()
    local = decode_group_plan_digest(
        [(seq.global_idx, seq.decode_dp_group) for seq in seqs]
    )
    # Same sequence count on both ranks, so the second round compares the
    # group vectors column by column: rank 1 put global_idx 1 on group 0.
    fake = _Dist([
        [[local, 4], [local + 1, 4]],
        [[0, 1, 0, 1], [0, 0, 0, 1]],
    ])
    worker = _worker(seqs, rank=0, world_size=2, G=8, dist_obj=fake)
    with pytest.raises(RuntimeError) as excinfo:
        worker._check_decode_dp_group_plan_identical([seq.uuid for seq in seqs])
    message = str(excinfo.value)
    assert "diverged across ranks" in message
    assert "first disagreement global_idx=1" in message
    # This rank's own view leads, then the gathered ones: it put global_idx 1
    # on group 1, rank 1 put it on group 0.
    assert "groups_by_rank=(1, 1, 0)" in message
    assert fake.calls == 2


def test_diverged_counts_report_without_a_second_collective():
    seqs = _two_group_seqs()
    local = decode_group_plan_digest(
        [(seq.global_idx, seq.decode_dp_group) for seq in seqs]
    )
    fake = _Dist([[[local, 4], [local + 1, 3]]])
    worker = _worker(seqs, rank=0, world_size=2, G=8, dist_obj=fake)
    with pytest.raises(RuntimeError, match=r"sequence counts \[3, 4\]"):
        worker._check_decode_dp_group_plan_identical([seq.uuid for seq in seqs])
    assert fake.calls == 1


def test_pure_dp_and_single_rank_run_no_collective():
    seqs = _two_group_seqs()
    uuids = [seq.uuid for seq in seqs]
    fake = _Dist([])
    _worker(seqs, rank=0, world_size=2, G=1,
            dist_obj=fake)._check_decode_dp_group_plan_identical(uuids)
    _worker(seqs, rank=0, world_size=1, G=8,
            dist_obj=fake)._check_decode_dp_group_plan_identical(uuids)
    assert fake.calls == 0


def test_unstamped_sequence_still_enters_the_digest():
    seqs = [_seq("u0", 0, 0), _seq("u1", 1, None)]
    stamped = decode_group_plan_digest([(0, 0), (1, 1)])
    fake = _Dist([
        [[stamped, 2], [stamped, 2]],
        [[0, 1], [0, 1]],
    ])
    worker = _worker(seqs, rank=0, world_size=2, G=8, dist_obj=fake)
    # Both peers reported the stamped map; this rank's own copy still has
    # global_idx 1 unstamped, so its local digest differs and is refused.
    with pytest.raises(RuntimeError) as excinfo:
        worker._check_decode_dp_group_plan_identical(["u0", "u1"])
    assert "diverged across ranks" in str(excinfo.value)
    # The local group vector reports -1 for the unstamped sequence, which is
    # how the column comparison sees it against the peers' group 1.
    assert "first disagreement global_idx=1" in str(excinfo.value)


# ---------------------------------------------- binding/group equivalence --- #


def test_binding_matching_the_group_passes():
    # G=8, rank 3 -> group 0: it must hold exactly the group-0 sequences.
    seqs = [_seq("u0", 0, 0), _seq("u1", 1, 1), _seq("u2", 2, 0)]
    worker = _worker(seqs, rank=3, world_size=16, G=8, bound=("u0", "u2"))
    worker._check_local_binding_matches_group(
        [seq.uuid for seq in seqs], "unit"
    )


def test_row_without_group_ownership_is_refused():
    # The crash shape: rank 3 (group 0) holds a row for a group-1 sequence,
    # whose KDA state item only group 1 ever created.
    seqs = [_seq("u0", 0, 0), _seq("u1", 1, 1)]
    worker = _worker(seqs, rank=3, world_size=16, G=8, bound=("u0", "u1"))
    with pytest.raises(RuntimeError) as excinfo:
        worker._check_local_binding_matches_group(["u0", "u1"], "decode entry")
    message = str(excinfo.value)
    assert "[SCHED_INVARIANT] decode entry" in message
    assert "global_idx=1 group=1 bound=1 owns=0" in message
    assert "G=8, rank//G=0" in message


def test_owned_sequence_without_a_row_is_refused():
    seqs = [_seq("u0", 0, 1), _seq("u1", 1, 1)]
    worker = _worker(seqs, rank=9, world_size=16, G=8, bound=("u0",))
    with pytest.raises(RuntimeError,
                       match="global_idx=1 group=1 bound=0 owns=1"):
        worker._check_local_binding_matches_group(["u0", "u1"], "prefill config")


def test_unstamped_group_with_a_row_is_refused():
    # An unstamped sequence is owned by nobody, so a row for it is a stale
    # binding from before a re-grouping.
    seqs = [_seq("u0", 0, None)]
    worker = _worker(seqs, rank=0, world_size=16, G=8, bound=("u0",))
    with pytest.raises(RuntimeError,
                       match="global_idx=0 group=None bound=1 owns=0"):
        worker._check_local_binding_matches_group(["u0"], "decode entry")


def test_pure_dp_binding_check_is_a_noop():
    # G==1 binds on assigned_rank, which this check does not model.
    seqs = [_seq("u0", 0, None, assigned_rank=5)]
    worker = _worker(seqs, rank=0, world_size=16, G=1, bound=("u0",))
    worker._check_local_binding_matches_group(["u0"], "decode entry")


def test_missing_sequences_are_skipped_not_refused():
    worker = _worker([_seq("u0", 0, 0)], rank=0, world_size=16, G=8,
                     bound=("u0",))
    worker._check_local_binding_matches_group(["u0", "gone"], "decode entry")


# --------------------------------------------------------- call sites --- #


def _method(name):
    return next(node for node in _worker_class().body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(node):
    return [
        ast.unparse(item.func) for item in ast.walk(node)
        if isinstance(item, ast.Call) and not isinstance(item.func, ast.Subscript)
    ]


def test_wave_selection_checks_the_map_before_it_derives_anything():
    body = ast.unparse(_method("_prepare_prefill_batch"))
    stamp = body.index("self._assign_decode_dp_groups(all_candidates)")
    check = body.index("self._check_decode_dp_group_plan_identical(all_candidates)")
    request = body.index("self._make_prefill_selection_request(")
    assert stamp < check < request


def test_prefill_config_checks_bindings_after_it_creates_them():
    body = ast.unparse(_method("_config_prefill_for_batch"))
    bind = body.index("self._bind_local_sequence_to_query_book(uuid)")
    check = body.index("self._check_local_binding_matches_group(prefill_uuids")
    assert bind < check


def test_decode_entry_checks_bindings_before_resolving_local_rows():
    source = WORKER.read_text()
    check = source.index(
        "self._check_local_binding_matches_group(\n\t\t\t\t\tdecode_uuids"
    )
    resolve = source.index(
        "local_decode_indices = self._get_local_indices_for_uuids(decode_uuids)"
    )
    assert check < resolve


def test_host_eviction_releases_the_kda_state_before_dropping_the_group():
    source = WORKER.read_text()
    release = source.index("self._release_kda_state_slots([int(seq.global_idx)])")
    drop = source.index("evict_from_decode_dp_group(seq)")
    assert release < drop
