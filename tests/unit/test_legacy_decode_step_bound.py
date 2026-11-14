"""The legacy decode loop runs on the decode set's own budgets, never the first batch's.

In pool mode the worker is initialised once with the first batch's max output
length, so the worker-wide ``max_decoding_length`` describes only that batch.
The legacy attention modes (0/1/2) used it as the step bound of every decode
call: a later, longer batch had each call cut at the first batch's budget, and
a first batch of ``max_tokens=1`` left later batches with no decode step at all.

The loop is driven through its deprecated mode-2 branch, which only advances
the step counter, so the steps it runs are exactly the bound under test.
batchgen_worker imports the whole engine, so the methods are compiled from the
real source against a fake worker (tests/worker_method_source.py).
"""

from types import SimpleNamespace
from typing import List

import pytest
import torch

from batchgen.config.decode_strategy import DecodeStrategy
from batchgen.sequence import SequenceBatch, SequenceEntry
from tests.worker_method_source import load_worker_methods


class _StepLog:
    """Counts the mode-2 branch's per-step deprecation warning."""

    def __init__(self):
        self.steps = 0

    def info(self, *args, **kwargs):
        pass

    def warning(self, message, *args, **kwargs):
        if "RUNTIME_ATTN_MODE 2" in message:
            self.steps += 1


def _methods(step_log, barrier_log):
    return load_worker_methods(
        ["_legacy_decode_step_bound", "_decoding_legacy_modes"],
        {
            "List": List,
            "DecodeStrategy": DecodeStrategy,
            "torch": torch,
            "logging": step_log,
            "dist": SimpleNamespace(barrier=lambda: barrier_log.append(1)),
        },
    )


class _Worker:
    rank = 1  # not rank 0: keeps the per-step info log out of the way

    def __init__(self, first_batch_budget, decision_interval, joiners=()):
        self.max_decoding_length = first_batch_budget
        self.DECISION_INTERVAL = decision_interval
        self.engine_config = SimpleNamespace(
            Basic_Config=SimpleNamespace(attn_mode=2)
        )
        self.global_batch = SequenceBatch()
        self._joiners = list(joiners)

    def add(self, uuid, budget):
        seq = SequenceEntry(
            uuid=uuid,
            global_idx=len(self.global_batch),
            prompt_length=8,
            max_decode_length=budget,
        )
        seq.decoded_length = 1  # the prefill token
        self.global_batch.add_sequence(seq)
        return uuid

    def _check_and_handle_completions(self, decode_uuids, batch, new_token_idx):
        return decode_uuids, batch, []

    def _try_load_new_sequences(self, decode_uuids, batch):
        if not self._joiners:
            return decode_uuids, batch
        uuid, budget = self._joiners.pop(0)
        self.add(uuid, budget)
        return decode_uuids + [uuid], batch + [len(batch)]


def _run(worker, decode_uuids):
    step_log, barriers = _StepLog(), []
    methods = _methods(step_log, barriers)
    worker._legacy_decode_step_bound = (
        lambda uuids: methods["_legacy_decode_step_bound"](worker, uuids)
    )
    methods["_decoding_legacy_modes"](
        worker,
        torch.zeros((len(decode_uuids), 1), dtype=torch.int64),
        list(decode_uuids),
        list(range(len(decode_uuids))),
        1,
        DecodeStrategy.HYBRID_CPU_GPU,
    )
    return step_log.steps


@pytest.mark.parametrize("first_batch_budget", [1, 16, 32768])
def test_call_runs_to_the_largest_budget_of_its_decode_set(first_batch_budget):
    # No boundary inside the call, so only the step bound ends it.
    worker = _Worker(first_batch_budget, decision_interval=1 << 20)
    uuids = [worker.add("long", 64), worker.add("short", 16)]
    # Steps 1..63: every row's own budget fits, whatever the first batch was.
    assert _run(worker, uuids) == 63


def test_bound_matches_a_worker_initialised_with_the_batch_budget():
    after_short = _Worker(1, decision_interval=1 << 20)
    own = _Worker(40, decision_interval=1 << 20)
    assert _run(after_short, [after_short.add("r", 40)]) == _run(own, [own.add("r", 40)])


def test_rows_loaded_at_a_boundary_extend_the_bound():
    # A short row decodes alone until the first boundary, where a longer
    # prefilled row joins the call; the call then covers the joiner's budget.
    worker = _Worker(1, decision_interval=16, joiners=[("joiner", 100)])
    assert _run(worker, [worker.add("short", 20)]) == 99


def test_empty_decode_set_runs_no_step():
    worker = _Worker(4096, decision_interval=1 << 20)
    assert _run(worker, []) == 0


def test_step_bound_reads_only_the_decode_set():
    worker = _Worker(1, decision_interval=1 << 20)
    worker.add("live", 300)
    worker.add("not_decoding", 9000)
    methods = _methods(_StepLog(), [])
    assert methods["_legacy_decode_step_bound"](worker, ["live"]) == 300
