"""The decode step's host bookkeeping must stay batched and skip retired rows.

Every item here was per-step host work that produced nothing new:

* a ``global_idx -> entry`` dict rebuilt over the whole batch inside
  ``_ensure_host_kv_append_capacity``, now a lookup on SequenceBatch's
  maintained gid index;
* one ``.item()`` per row twice over plus one pybind trajectory append per row,
  now a single ``tolist()`` and one batched append;
* a second resolution of the same rows through ``_local_to_uuid_map`` +
  ``get_sequence`` to build the host-KV append metadata;
* a per-row scan for per-request debug flags that returns None whenever no
  sequence carries any.

The trap in the batched append is the completed-row skip: a retired row must
not receive a token in the trajectory book, so the skip has to be explicit at
the call site rather than inherited from the batched helper.

batchgen_worker imports the whole engine (triton, CUDA extensions), so these
are static checks on the method bodies, like the host-KV append drain test
next to this one.
"""
import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
SOURCE = WORKER.read_text()
TREE = ast.parse(SOURCE)


def _method(name):
    return next(n for n in ast.walk(TREE)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node, func_src):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == func_src]


def _names(node, name):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Name) and n.id == name]


def _attributes(node, attr):
    return [n for n in ast.walk(node) if isinstance(n, ast.Attribute)
            and n.attr == attr]


def _assign_to(node, target):
    return [n for n in ast.walk(node) if isinstance(n, ast.Assign)
            and [ast.unparse(t) for t in n.targets] == [target]]


# ---------------------------------------------------------------- gid index

def test_host_kv_append_capacity_uses_the_maintained_gid_index():
    fn = _method("_ensure_host_kv_append_capacity")
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.DictComp)], (
        "the per-step gid dict rebuild is back"
    )
    assert len(_calls(fn, "self.global_batch.get_sequence_by_global_idx")) == 1


def test_decode_step_resolves_host_kv_metadata_from_batch_sequences():
    # batch_sequences is resolved once per step for the forward; re-resolving
    # the same rows for the host-KV append metadata was pure duplication.
    fn = _method("decoding_continuous")
    assert len(_assign_to(fn, "_kv_seq_ids")) == 1
    loop = next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.For) and any(_names(n, "_kv_seq_ids"))
    )
    assert ast.unparse(loop.iter) == "batch_sequences"
    assert ast.unparse(loop.target) == "seq"
    assert not _attributes(loop, "_local_to_uuid_map")
    assert not _calls(loop, "self.global_batch.get_sequence")


# -------------------------------------------------- batched trajectory append

def test_decode_step_reads_the_sampled_tokens_once_per_step():
    fn = _method("decoding_continuous")
    assert len(_calls(fn, "_new_tokens_pinned[:bs].reshape(-1).tolist")) == 1
    # No per-row readback of the sampled tokens survives in the step loop.
    assert not _calls(fn, "new_tokens_cpu[i].item")
    assert not _names(fn, "new_tokens_cpu")


def test_decode_step_appends_trajectory_tokens_in_one_batched_call():
    fn = _method("decoding_continuous")
    batched = _calls(fn, "self._append_trajectory_tokens")
    assert len(batched) == 1, "the decode step must append once per step"
    assert not _calls(fn, "self._append_trajectory_token"), (
        "a per-row trajectory append is back in the decode step"
    )


def test_decode_step_decides_each_row_once_and_keeps_the_completed_row_skip():
    fn = _method("decoding_continuous")
    completed = _calls(fn, "self._is_sequence_completed")
    assert len(completed) == 1, (
        "the completion predicate must be evaluated once per row, in the "
        "row-active mask"
    )

    mask = _assign_to(fn, "_row_active")
    assert len(mask) == 1
    value = mask[0].value
    assert isinstance(value, ast.ListComp)
    assert isinstance(value.elt, ast.UnaryOp)
    assert isinstance(value.elt.op, ast.Not)
    assert ast.unparse(value.elt.operand) == "self._is_sequence_completed(seq)"
    assert ast.unparse(value.generators[0].iter) == "batch_sequences"


def test_the_batched_append_receives_only_the_live_rows():
    fn = _method("decoding_continuous")
    call = _calls(fn, "self._append_trajectory_tokens")[0]
    mask = _assign_to(fn, "_row_active")[0]
    assert mask.lineno < call.lineno, (
        "rows must be classified before any token is appended"
    )
    assert len(call.args) == 2
    for argument in call.args:
        assert isinstance(argument, ast.ListComp), (
            "the append must be filtered at the call site, not inside the "
            "helper"
        )
        condition = argument.generators[0].ifs
        assert [ast.unparse(test) for test in condition] == ["live"]
        assert "_row_active" in ast.unparse(argument.generators[0].iter)


def test_the_per_row_update_loop_still_skips_retired_rows():
    fn = _method("decoding_continuous")
    loop = next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and ast.unparse(n.iter) == "enumerate(zip(batch, batch_sequences))"
    )
    guard = loop.body[0]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "not _row_active[i]"
    assert isinstance(guard.body[0], ast.Continue)


def test_the_batched_helper_does_not_decide_which_rows_are_live():
    # The helper takes the rows it must append to; if it ever grew a
    # completion check of its own, the skip would live in two places.
    fn = _method("_append_trajectory_tokens")
    assert not _calls(fn, "self._is_sequence_completed")
    assert len(_calls(fn, "self._trajectory_book.append_tokens")) == 1


# ------------------------------------------------------------- debug scanning

def test_debug_flag_scan_exits_before_scanning_when_no_sequence_carries_flags():
    fn = _method("_active_batchgen_debug_for_sequences")
    guard = next(
        n for n in fn.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not self._any_sequence_batchgen_debug"
    )
    assert isinstance(guard.body[0], ast.Return)
    assert ast.unparse(guard.body[0].value) == "None"
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For))
    assert guard.lineno < loop.lineno


def test_the_debug_flag_is_raised_where_sequences_receive_their_flags():
    fn = _method("_admit_sequences_from_message")
    raised = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
              and [ast.unparse(t) for t in n.targets]
              == ["self._any_sequence_batchgen_debug"]]
    assert len(raised) == 1
    assert ast.unparse(raised[0].value) == "True"


# ----------------------------------------------- tracked dead per-step write

def test_the_dsa_step_state_write_stays_and_names_its_tracking_record():
    # The index_topk branch is unreachable today only because the engine
    # config projection drops the field. Deleting it here would delete the
    # surface that record's fix lands on, so both the write and the pointer
    # to the record must survive.
    fn = _method("_reset_glm5_dsa_step_state")
    body = ast.get_source_segment(SOURCE, fn)
    assert "batchgen-glm52-index-topk-config-projection-loss" in body
    guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If))
    assert ast.unparse(guard.test) == "index_topk is not None"
    assert _attributes(guard, "_dsa_short_count")
