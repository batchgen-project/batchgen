"""Regression: the "Pool batch group completed" summary must cover one group.

The pool worker restarts its prefill/decode/wall timers when an idle
admission starts a new batch group, but global_batch keeps every completed
sequence. The summary used to sum tokens over the whole global_batch, so after
the second Batch every tok/s figure divided cumulative tokens by one group's
time (a two-Batch run reported both Batches' tokens and ~2x decode tok/s).

batchgen_worker imports the whole engine, so the admission and totals methods
are compiled in isolation and generate() is checked statically.
"""
import ast
import copy
import logging
from pathlib import Path

from batchgen.sequence import SequenceBatch, SequenceEntry

WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
TREE = ast.parse(WORKER.read_text())


def _method(name):
    return next(n for n in ast.walk(TREE)
                if isinstance(n, ast.FunctionDef) and n.name == name)


class _Worker:
    rank = 1
    max_decoding_length = 64
    max_input_length = 1 << 20

    def __init__(self):
        self.global_batch = SequenceBatch()
        self._group_uuids = set()

    def _tokenize_admitted_sequences(self, uuids):
        for u in uuids:
            seq = self.global_batch.get_sequence(u)
            seq.prompt_length = len(seq.text)

    def _update_config_after_tokenization(self):
        pass

    def _assign_admitted_sequences_to_ranks(self, uuids):
        pass

    def _assign_decode_dp_groups(self, uuids):
        pass

    def _build_local_query_book_for_admitted(self, uuids):
        pass


namespace = {"SequenceEntry": SequenceEntry, "logging": logging}
exec(compile(ast.fix_missing_locations(ast.Module(
    body=[copy.deepcopy(_method("_admit_sequences_from_message")),
          copy.deepcopy(_method("_group_token_totals"))],
    type_ignores=[])), str(WORKER), "exec"), namespace)
_Worker._admit_sequences_from_message = namespace["_admit_sequences_from_message"]
_Worker._group_token_totals = namespace["_group_token_totals"]


def _admit(worker, batch_id, prompts):
    worker._admit_sequences_from_message({"entries": [
        {"request_id": f"{batch_id}-{i}", "text": text, "batch_id": batch_id}
        for i, text in enumerate(prompts)]})


def _decode(worker, decoded):
    for seq in worker.global_batch:
        if seq.decoded_length == 0:
            seq.decoded_length = decoded


def test_totals_cover_only_the_current_batch_group():
    worker = _Worker()
    _admit(worker, "b1", ["x" * 100, "x" * 200, "x" * 300])
    _decode(worker, decoded=50)
    assert worker._group_token_totals() == (3, 600, 150)

    # Idle admission starts a new group; b1 stays in global_batch.
    worker._group_uuids = set()
    _admit(worker, "b2", ["x" * 10, "x" * 20])
    # A mid-flight admission joins the running group.
    _admit(worker, "b3", ["x" * 40])
    _decode(worker, decoded=7)
    assert len(worker.global_batch) == 6
    assert worker._group_token_totals() == (3, 70, 21)


def test_rejected_sequences_are_not_counted():
    worker = _Worker()
    _admit(worker, "b1", ["x" * 10, "x" * 20])
    worker.global_batch.remove_sequence("b1-1")
    assert worker._group_token_totals() == (1, 10, 0)


def test_generate_starts_a_new_group_before_each_idle_admission():
    fn = _method("generate")
    admits = 0
    for node in ast.walk(fn):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for prev, stmt in zip(body, body[1:]):
            call = getattr(stmt, "value", None)
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_admit_sequences_from_message"):
                continue
            admits += 1
            assert ast.unparse(prev) == "self._group_uuids = set()", ast.unparse(prev)
    assert admits == 2  # rank-0 and non-rank-0 idle-admission branches


def test_summaries_use_group_totals():
    src = ast.unparse(_method("generate"))
    assert src.count("self._group_token_totals()") == 2
    assert "sum(s.prompt_length for s in self.global_batch)" not in src
