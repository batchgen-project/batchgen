"""Per-group decode rows are stated once per decode phase, never per step.

At G>1 the decode graph bucket follows the fullest serve group, so two batches
of the same size can decode on different buckets. Three rank-0 records state
the per-group rows: the plan stamp (``[DECODE_GROUP_PLAN]``), the decode-phase
entry sync (``[DECODE_GROUPS]``, again only when an in-phase sync changes the
per-rank unit) and the batch summary (``Decode groups:``). None of them may
change placement or the synchronized unit, and the G==1 output stays as it was.

The worker methods are compiled out of the module AST onto fakes and the row
count collective is replaced by a scripted vector, so nothing needs CUDA.
"""

import ast
import copy
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List

import pytest
import torch

import batchgen.worker.rank_sync_engine as rank_sync_engine
from batchgen.decode_dp_group import (
    DECODE_IN_PHASE_SYNCS,
    DECODE_OUT_OF_PHASE_SYNCS,
    DECODE_PHASE_ENTRY_SYNC,
    assign_decode_dp_groups,
    count_rows_per_group,
    decode_group_rows,
    decode_groups_log_due,
    format_decode_group_plan_line,
    format_decode_group_summary,
    format_decode_groups_line,
    format_group_rows,
)
from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.rank_sync_engine import RankSyncEngine
from batchgen.worker.admission.context import DecodePhaseContext, RankFacts, SchedulingState

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
DECODE_ENGINE = ROOT / "batchgen" / "worker" / "decode_engine.py"
SYNC_METHOD = "_sync_decode_moe_rank_counts"


# ---- formatting helpers -----------------------------------------------------


def test_group_rows_are_the_per_group_max_in_group_order():
    counts = [3, 3, 5, 5, 2, 2, 4, 4]
    assert decode_group_rows(counts, 2) == [3, 5, 2, 4]
    # A group whose ranks disagree reports the max the unit is sized from.
    assert decode_group_rows([3, 4, 1, 1], 2) == [4, 1]
    assert decode_group_rows([7, 0, 2], 1) == [7, 0, 2]
    with pytest.raises(ValueError, match="not divisible"):
        decode_group_rows([1, 2, 3], 2)
    with pytest.raises(ValueError, match="group_size"):
        decode_group_rows([1, 2], 0)


def test_rows_per_group_skip_unstamped_and_refuse_foreign_groups():
    assert count_rows_per_group([0, 2, None, 2, 3], 4) == [1, 0, 2, 1]
    assert count_rows_per_group([], 2) == [0, 0]
    with pytest.raises(ValueError, match="outside 0..1"):
        count_rows_per_group([2], 2)
    with pytest.raises(ValueError, match="num_dp"):
        count_rows_per_group([], 0)


def test_line_formats_are_stable():
    assert format_group_rows([28, 24, 34, 42]) == "[28,24,34,42]"
    assert format_decode_groups_line("decode_entry", [28, 24, 34, 42], 8, 6) == (
        "[DECODE_GROUPS] sync=decode_entry rows=128 groups=[28,24,34,42] "
        "tp=8 ntp=6"
    )
    assert format_decode_group_plan_line("packer", [14, 17, 22, 11],
                                         [14, 17, 22, 11]) == (
        "[DECODE_GROUP_PLAN] placement=packer admitted=64 "
        "admitted_groups=[14,17,22,11] groups=[14,17,22,11]"
    )
    assert format_decode_group_summary([28, 24, 34, 42], 8) == (
        "  Decode groups: [28,24,34,42] (tp=8, ntp=6)"
    )


def test_log_due_once_at_entry_and_on_unit_change_only():
    assert decode_groups_log_due(DECODE_PHASE_ENTRY_SYNC, 4, 4)
    for reason in DECODE_IN_PHASE_SYNCS:
        assert not decode_groups_log_due(reason, 4, 4)
        assert decode_groups_log_due(reason, 3, 4)
    for reason in DECODE_OUT_OF_PHASE_SYNCS:
        assert not decode_groups_log_due(reason, 3, 4)
    with pytest.raises(ValueError, match="unclassified"):
        decode_groups_log_due("somewhere_new", 3, 4)


# ---- every sync call site is classified; entry is once per phase -----------


def _sync_reasons(path):
    reasons = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == SYNC_METHOD):
            continue
        reason = next(kw.value for kw in node.keywords if kw.arg == "reason")
        if isinstance(reason, ast.Name) and reason.id == "reason":
            continue  # the worker's delegator forwards its caller's reason
        assert isinstance(reason, ast.Constant), ast.unparse(node)
        reasons.append(reason.value)
    return reasons


def test_every_sync_reason_is_classified_exactly_once():
    sets = ({DECODE_PHASE_ENTRY_SYNC}, DECODE_IN_PHASE_SYNCS,
            DECODE_OUT_OF_PHASE_SYNCS)
    reasons = set()
    for path in (ROOT / "batchgen").rglob("*.py"):
        if SYNC_METHOD in path.read_text():
            reasons.update(_sync_reasons(path))
    assert reasons, "no decode row-count sync call sites found"
    for reason in reasons:
        assert sum(reason in members for members in sets) == 1, reason


def test_phase_entry_sync_runs_once_per_decode_phase():
    tree = ast.parse(DECODE_ENGINE.read_text())
    run = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    entry_calls = [
        node for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == SYNC_METHOD
        and any(kw.arg == "reason" and getattr(kw.value, "value", None)
                == DECODE_PHASE_ENTRY_SYNC for kw in node.keywords)
    ]
    assert len(entry_calls) == 1
    # Straight-line in the phase body, before the step loop: not in any loop.
    loops = [node for node in ast.walk(run)
             if isinstance(node, (ast.For, ast.While))]
    for loop in loops:
        assert entry_calls[0] not in list(ast.walk(loop))
    # And nowhere else in the package.
    for path in (ROOT / "batchgen").rglob("*.py"):
        if path == DECODE_ENGINE or SYNC_METHOD not in path.read_text():
            continue
        assert DECODE_PHASE_ENTRY_SYNC not in _sync_reasons(path), path


# ---- the sync seam -----------------------------------------------------------


class _ParallelManager:
    def __init__(self, G):
        self.attn_tp_size = G
        self.ntp_set = []

    def set_num_tokens_per_rank(self, ntp):
        self.ntp_set.append(int(ntp))

    def set_rank_token_counts(self, counts):
        pass


def _sync_worker(world_size, G, rank=0):
    worker = SimpleNamespace(
        rank=rank,
        world_size=world_size,
        torch_device=torch.device("cpu"),
        parallel_manager=_ParallelManager(G),
        _decode_local_count_tensor=None,
        _decode_all_rank_counts=None,
    )
    engine = RankSyncEngine()
    context = DecodePhaseContext(
        facts=RankFacts(
            rank=rank,
            world_size=world_size,
            local_rank=0,
            local_world_size=world_size,
            rank_ctx=None,
            attn_tp_size=G,
            device=torch.device("cpu"),
            page_size=1,
            decision_interval=1,
            enable_prefix_cache=False,
            enable_host_kv_eviction=False,
            host_kv_chunk_size=1,
        ),
        sched=SchedulingState(pool=None, seq_table=None, trajectories=None),
        gpu_kv=None,
        host_view=None,
        core_engine=None,
        gpu_resident=None,
        prefix_coordinator=None,
        prefix_runtime=None,
        chunk_sizer=None,
        parallel_manager=worker.parallel_manager,
    )

    def sync(batch, reason):
        outcome = engine._sync_decode_moe_rank_counts(
            context,
            batch,
            reason=reason,
            decode_local_count_tensor=worker._decode_local_count_tensor,
            decode_all_rank_counts=worker._decode_all_rank_counts,
            previous_max=worker.__dict__.get("_current_decode_max_rank_batch_size", 0),
        )
        worker._decode_local_count_tensor = outcome.local_count_tensor
        worker._decode_all_rank_counts = outcome.all_rank_counts
        worker._current_decode_local_batch_size = outcome.current_local_batch_size
        worker._current_decode_max_rank_batch_size = outcome.max_batch_size
        worker._current_decode_rank_token_counts = outcome.rank_token_counts
        return outcome.max_batch_size

    worker.sync = sync
    return worker


@pytest.fixture
def scripted_counts(monkeypatch):
    """The row-count all_gather returns the vector the test scripts next."""
    script = {}

    def all_gather_into_tensor(out, local):
        out.copy_(torch.tensor(script["counts"], dtype=torch.int64))

    monkeypatch.setattr(rank_sync_engine.dist, "all_gather_into_tensor",
                        all_gather_into_tensor)
    return script


def _drive(worker, script, steps):
    for reason, counts in steps:
        script["counts"] = counts
        worker.sync(list(range(counts[0])), reason)


def _group_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("[DECODE_GROUPS]")]


# Two decode phases on 8 ranks in serve groups of 2 (four groups). Inside the
# first phase the unit steps down once (ntp 3 -> 2); the other in-phase syncs
# keep it. Out-of-phase syncs never log, whatever the unit does.
_TWO_PHASES = [
    ("pre_decode_warmup", [3, 3, 5, 5, 2, 2, 4, 4]),
    ("decode_entry", [3, 3, 5, 5, 2, 2, 4, 4]),
    ("decode_step_batch_change", [3, 3, 5, 5, 2, 2, 4, 4]),
    ("page_boundary", [3, 3, 5, 5, 2, 2, 4, 4]),
    ("page_boundary", [3, 3, 4, 4, 2, 2, 4, 4]),
    ("post_pending_load_finalize", [1, 1, 4, 4, 2, 2, 4, 4]),
    ("page_boundary", [1, 1, 3, 3, 2, 2, 4, 4]),
    ("pre_decode_warmup", [6, 6, 1, 1, 1, 1, 1, 1]),
    ("decode_entry", [6, 6, 1, 1, 1, 1, 1, 1]),
    ("page_boundary", [6, 6, 1, 1, 1, 1, 1, 1]),
]


def test_rank0_states_groups_once_per_phase_entry_and_on_unit_change(
    scripted_counts, caplog,
):
    worker = _sync_worker(world_size=8, G=2)
    with caplog.at_level(logging.INFO):
        _drive(worker, scripted_counts, _TWO_PHASES)
    assert _group_lines(caplog) == [
        "[DECODE_GROUPS] sync=decode_entry rows=14 groups=[3,5,2,4] tp=2 ntp=3",
        "[DECODE_GROUPS] sync=page_boundary rows=13 groups=[3,4,2,4] tp=2 ntp=2",
        "[DECODE_GROUPS] sync=decode_entry rows=9 groups=[6,1,1,1] tp=2 ntp=3",
    ]
    entries = [line for line in _group_lines(caplog) if "sync=decode_entry" in line]
    assert len(entries) == sum(r == "decode_entry" for r, _ in _TWO_PHASES)
    # Observability only: the unit handed to the model is unchanged.
    assert worker.parallel_manager.ntp_set == [3, 3, 3, 3, 2, 2, 2, 3, 3, 3]


def test_other_ranks_and_pure_dp_state_nothing(scripted_counts, caplog):
    with caplog.at_level(logging.INFO):
        _drive(_sync_worker(world_size=8, G=2, rank=1), scripted_counts,
               _TWO_PHASES)
        _drive(_sync_worker(world_size=8, G=1), scripted_counts, _TWO_PHASES)
    assert _group_lines(caplog) == []


# ---- the plan stamp ----------------------------------------------------------


def _worker_methods(*names):
    tree = ast.parse(WORKER.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "BatchGenWorker")
    wanted = [copy.deepcopy(node) for node in cls.body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(wanted) == len(names)
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {"List": List, "logging": logging,
                 "SequenceStatus": SequenceStatus}
    exec(compile(module, str(WORKER), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _Batch:
    def __init__(self, seqs):
        self._seqs = {seq.uuid: seq for seq in seqs}

    def __iter__(self):
        return iter(list(self._seqs.values()))

    def get_sequence(self, uuid):
        return self._seqs.get(uuid)


def _seq(uuid, length, *, status=SequenceStatus.QUEUEING, group=None):
    seq = SimpleNamespace(uuid=uuid, prompt_length=length, status=status,
                          decode_dp_group=group, evicted_from_dp_group=None)
    # The group stamp is owned by SequenceEntry; bind the real method (never
    # re-implemented here) so the planner under test can call it.
    seq.assign_decode_group = MethodType(SequenceEntry.assign_decode_group, seq)
    return seq


def _plan_worker(seqs, *, packer_groups=None, rank=0, G=2, world_size=8):
    worker = SimpleNamespace(
        rank=rank,
        world_size=world_size,
        global_batch=_Batch(seqs),
        _decode_attn_tp_size=lambda: G,
        _trajectory_session_residency=lambda uuids, num_groups: {},
        _prefill_prefix_pool_enabled=lambda: packer_groups is not None,
        _prefix_rank_packing_mode=lambda seqs: packer_groups is not None,
        _pack_decode_dp_groups_for_sharing=(
            lambda seqs, num_dp, residency: list(packer_groups)
        ),
    )
    for name, method in _worker_methods("_assign_decode_dp_groups").items():
        setattr(worker, name, MethodType(method, worker))
    return worker


def _standing():
    # One row still decoding on group 3; one finished row on group 0 that no
    # longer decodes and so is not counted.
    return [
        _seq("standing", 100, status=SequenceStatus.IN_DECODE, group=3),
        _seq("finished", 100, status=SequenceStatus.COMPLETED, group=0),
    ]


def _plan_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("[DECODE_GROUP_PLAN]")]


def test_plan_stamp_states_rows_on_the_packer_path(caplog):
    admitted = [_seq(f"a{i}", 10 + i) for i in range(5)]
    worker = _plan_worker(_standing() + admitted, packer_groups=[1, 1, 2, 0, 1])
    with caplog.at_level(logging.INFO):
        worker._assign_decode_dp_groups([seq.uuid for seq in admitted])
    assert [seq.decode_dp_group for seq in admitted] == [1, 1, 2, 0, 1]
    assert _plan_lines(caplog) == [
        "[DECODE_GROUP_PLAN] placement=packer admitted=5 "
        "admitted_groups=[1,3,1,0] groups=[1,3,1,1]"
    ]


def test_plan_stamp_states_rows_on_the_l2_path_without_moving_rows(caplog):
    admitted = [_seq(f"a{i}", length) for i, length in
                enumerate([400, 300, 200, 100, 50])]
    worker = _plan_worker(_standing() + admitted)
    with caplog.at_level(logging.INFO):
        worker._assign_decode_dp_groups([seq.uuid for seq in admitted])
    # The L^2 prior charges every grouped row, finished ones included; the
    # stated live rows count only the rows still to decode.
    expected = assign_decode_dp_groups(
        [400, 300, 200, 100, 50], 4,
        prior_load=[100.0 * 100.0, 0.0, 0.0, 100.0 * 100.0],
    )
    assert [seq.decode_dp_group for seq in admitted] == expected
    admitted_rows = count_rows_per_group(expected, 4)
    live_rows = list(admitted_rows)
    live_rows[3] += 1
    assert _plan_lines(caplog) == [
        "[DECODE_GROUP_PLAN] placement=l2 admitted=5 "
        f"admitted_groups={format_group_rows(admitted_rows)} "
        f"groups={format_group_rows(live_rows)}"
    ]


def test_plan_stamp_is_rank0_only_and_absent_without_new_rows(caplog):
    with caplog.at_level(logging.INFO):
        _plan_worker(_standing() + [_seq("a0", 10)],
                     rank=3)._assign_decode_dp_groups(["a0"])
        worker = _plan_worker(_standing() + [_seq("a0", 10)])
        worker._assign_decode_dp_groups(["a0"])
        # Idempotent re-entry: nothing stamped, nothing stated.
        worker._assign_decode_dp_groups(["a0"])
        _plan_worker([_seq("b0", 10)], G=1)._assign_decode_dp_groups(["b0"])
    assert len(_plan_lines(caplog)) == 1


# ---- the batch summary -------------------------------------------------------


def _summary_worker(seqs, G, world_size=8):
    worker = SimpleNamespace(
        world_size=world_size,
        global_batch=_Batch(seqs),
        _group_uuids={seq.uuid for seq in seqs} | {"rejected"},
        _decode_attn_tp_size=lambda: G,
    )
    for name, method in _worker_methods("_group_decode_rows_summary").items():
        setattr(worker, name, MethodType(method, worker))
    return worker


def test_summary_line_is_empty_for_pure_dp_and_states_groups_otherwise():
    seqs = [_seq(f"s{i}", 10, status=SequenceStatus.COMPLETED, group=g)
            for i, g in enumerate([0, 1, 1, 1, 3])]
    assert _summary_worker(seqs, G=1)._group_decode_rows_summary() == ""
    assert _summary_worker(seqs, G=2)._group_decode_rows_summary() == (
        "\n  Decode groups: [1,3,0,1] (tp=2, ntp=2)"
    )


def test_batch_summary_appends_the_group_line_last():
    tree = ast.parse(WORKER.read_text())
    generate = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "generate")
    summaries = [
        node for node in ast.walk(generate)
        if isinstance(node, ast.JoinedStr)
        and any(isinstance(v, ast.Constant)
                and "Pool batch group completed" in str(v.value)
                for v in node.values)
    ]
    assert len(summaries) == 1
    last = summaries[0].values[-1]
    assert isinstance(last, ast.FormattedValue)
    assert ast.unparse(last.value) == "self._group_decode_rows_summary()"
