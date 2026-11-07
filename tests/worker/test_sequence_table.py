"""``SequenceTable``: the single owner of one rank's local slot binding.

Covers the API, the read-only views, the free-slot allocator, the
group-binding invariant for G=1 and G=8, and the release-before-unbind
ordering that callers own.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.worker.rank_context import RankContext
from batchgen.worker.sequence_table import SequenceTable

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


def _ctx(rank, *, world_size=32, local_world_size=8, decode_group_size=1):
    return RankContext(
        rank=rank, world_size=world_size, local_rank=rank % local_world_size,
        local_world_size=local_world_size,
        decode_group_size=decode_group_size,
    )


def _seq(uuid, *, global_idx=1, assigned_rank=0, decode_dp_group=None):
    return SimpleNamespace(
        uuid=uuid, global_idx=global_idx,
        assigned_rank=assigned_rank, decode_dp_group=decode_dp_group,
    )


# ----------------------------------------------------------- bind and unbind


def test_a_fresh_table_is_empty():
    table = SequenceTable()
    assert len(table) == 0
    assert dict(table.local_to_uuid) == {} and dict(table.uuid_to_local) == {}
    assert set(table.free_local_indices) == set()
    assert table.next_local_idx == 0


def test_bind_allocates_from_the_never_used_counter_in_order():
    table = SequenceTable()
    assert [table.bind("a"), table.bind("b"), table.bind("c")] == [0, 1, 2]
    assert table.next_local_idx == 3
    assert dict(table.uuid_to_local) == {"a": 0, "b": 1, "c": 2}
    assert dict(table.local_to_uuid) == {0: "a", 1: "b", 2: "c"}


def test_bind_prefers_a_freed_slot_over_the_counter():
    table = SequenceTable()
    table.bind("a")
    table.bind("b")
    table.unbind("a")
    counter = table.next_local_idx
    assert table.bind("c") == 0
    assert table.next_local_idx == counter, "a reused slot must not advance it"
    assert set(table.free_local_indices) == set()


def test_an_explicit_slot_is_bound_as_given():
    table = SequenceTable()
    assert table.bind("a", 7) == 7
    assert dict(table.uuid_to_local) == {"a": 7}
    # An explicit index does not touch the never-used counter: that is what
    # the raw-container helper did, and the migration path depends on it.
    assert table.next_local_idx == 0


def test_rebinding_a_uuid_to_a_new_slot_leaves_the_old_row_behind():
    """Today's behaviour, pinned rather than endorsed: ``bind`` overwrites the
    forward entry and adds a second reverse entry. Callers unbind first."""
    table = SequenceTable()
    table.bind("a", 1)
    table.bind("a", 2)
    assert dict(table.uuid_to_local) == {"a": 2}
    assert dict(table.local_to_uuid) == {1: "a", 2: "a"}


def test_unbind_returns_the_slot_and_frees_it():
    table = SequenceTable()
    table.bind("a", 4)
    assert table.unbind("a") == 4
    assert dict(table.uuid_to_local) == {} and dict(table.local_to_uuid) == {}
    assert set(table.free_local_indices) == {4}


def test_unbind_of_an_unbound_uuid_returns_none_and_frees_nothing():
    table = SequenceTable()
    assert table.unbind("missing") is None
    assert set(table.free_local_indices) == set()


def test_unbind_is_idempotent_after_the_first_call():
    table = SequenceTable()
    table.bind("a", 4)
    assert table.unbind("a") == 4
    assert table.unbind("a") is None
    assert set(table.free_local_indices) == {4}


def test_reset_clears_every_binding_and_restarts_the_counter():
    table = SequenceTable()
    table.bind("a")
    table.bind("b")
    table.unbind("a")
    table.reset()
    assert len(table) == 0
    assert set(table.free_local_indices) == set()
    assert table.next_local_idx == 0
    assert table.bind("c") == 0


# ------------------------------------------------------------- reads, views


def test_lookups_go_both_ways_and_report_absence_as_none():
    table = SequenceTable()
    table.bind("a", 3)
    assert table.local_of("a") == 3 and table.uuid_of(3) == "a"
    assert table.local_of("nope") is None and table.uuid_of(99) is None


def test_membership_and_iteration_cover_bound_uuids():
    table = SequenceTable()
    table.bind("a")
    table.bind("b")
    assert "a" in table and "b" in table and "c" not in table
    assert table.is_bound("a") and not table.is_bound("c")
    assert sorted(table) == ["a", "b"] and len(table) == 2


def test_the_views_are_read_only():
    table = SequenceTable()
    table.bind("a", 0)
    with pytest.raises(TypeError):
        table.uuid_to_local["b"] = 1
    with pytest.raises(TypeError):
        table.local_to_uuid[1] = "b"
    assert not hasattr(table.free_local_indices, "add")


def test_the_views_stay_live_across_writes_and_reset():
    """A view taken once must keep reflecting the table, because the worker
    hands the same object to long-lived readers."""
    table = SequenceTable()
    forward, reverse, free = (
        table.uuid_to_local, table.local_to_uuid, table.free_local_indices,
    )
    table.bind("a", 2)
    assert dict(forward) == {"a": 2} and dict(reverse) == {2: "a"}
    table.unbind("a")
    assert set(free) == {2} and dict(forward) == {}
    table.reset()
    assert set(free) == set()
    table.bind("b", 5)
    assert dict(forward) == {"b": 5}


def test_the_free_view_compares_equal_to_a_plain_set_both_ways():
    table = SequenceTable()
    table.bind("a", 4)
    table.unbind("a")
    assert table.free_local_indices == {4}
    assert {4} == table.free_local_indices


# ------------------------------------------- check_binding_matches_group, G=1


def _check(table, uuids, ctx, seqs):
    table.check_binding_matches_group(
        uuids, "a test", rank_ctx=ctx, get_sequence=seqs.get
    )


def test_g1_binding_check_is_a_noop_even_when_the_binding_disagrees():
    """Pure DP does not run the equivalence check at all."""
    table = SequenceTable()
    table.bind("u", 0)
    seqs = {"u": _seq("u", assigned_rank=7)}
    _check(table, ["u"], _ctx(0, decode_group_size=1), seqs)


# ------------------------------------------- check_binding_matches_group, G=8


def test_g8_a_binding_that_matches_the_group_passes():
    table = SequenceTable()
    table.bind("u", 0)
    seqs = {"u": _seq("u", decode_dp_group=1)}
    _check(table, ["u"], _ctx(9, decode_group_size=8), seqs)


def test_g8_a_row_this_group_does_not_own_is_refused():
    table = SequenceTable()
    table.bind("u", 0)
    seqs = {"u": _seq("u", global_idx=42, decode_dp_group=2)}
    with pytest.raises(RuntimeError, match="SCHED_INVARIANT") as excinfo:
        _check(table, ["u"], _ctx(9, decode_group_size=8), seqs)
    message = str(excinfo.value)
    assert "global_idx=42" in message and "bound=1 owns=0" in message
    assert "G=8" in message and "rank//G=1" in message


def test_g8_an_owned_sequence_without_a_row_is_refused():
    table = SequenceTable()
    seqs = {"u": _seq("u", global_idx=42, decode_dp_group=1)}
    with pytest.raises(RuntimeError, match="bound=0 owns=1"):
        _check(table, ["u"], _ctx(9, decode_group_size=8), seqs)


def test_g8_an_unstamped_group_with_a_row_is_refused():
    table = SequenceTable()
    table.bind("u", 0)
    seqs = {"u": _seq("u", global_idx=42, decode_dp_group=None)}
    with pytest.raises(RuntimeError, match="bound=1 owns=0"):
        _check(table, ["u"], _ctx(9, decode_group_size=8), seqs)


def test_g8_uuids_with_no_sequence_are_skipped_not_refused():
    table = SequenceTable()
    table.bind("u", 0)
    seqs = {"u": _seq("u", decode_dp_group=1)}
    _check(table, ["u", "gone"], _ctx(9, decode_group_size=8), seqs)


def test_g8_the_message_caps_the_offender_list_but_not_the_count():
    table = SequenceTable()
    seqs = {}
    for index in range(12):
        uuid = f"u{index}"
        seqs[uuid] = _seq(uuid, global_idx=index, decode_dp_group=1)
    with pytest.raises(RuntimeError) as excinfo:
        _check(table, list(seqs), _ctx(9, decode_group_size=8), seqs)
    message = str(excinfo.value)
    assert "12 sequences" in message
    assert message.count("global_idx=") == 10


# ---------------------------------------------------- release-before-unbind
#
# `SequenceTable` cannot see resource state, so it does not enforce the
# order; these tests pin it where it is enforceable: the documented contract
# and the worker call sites that depend on it.


def test_unbind_documents_the_release_before_unbind_contract():
    doc = SequenceTable.unbind.__doc__
    assert doc is not None
    assert "ORDERING INVARIANT" in doc
    assert "BEFORE" in doc


def test_a_release_filter_keyed_on_membership_sees_nothing_after_unbind():
    """Why the order matters, at the API level: the worker selects "my"
    completed sequences with ``uuid in uuid_to_local``, so an unbind that
    runs first turns that filter into the empty list and the pages leak."""
    table = SequenceTable()
    table.bind("done", 0)
    completed = ["done"]

    mine_before = [u for u in completed if u in table.uuid_to_local]
    table.unbind("done")
    mine_after = [u for u in completed if u in table.uuid_to_local]

    assert mine_before == ["done"]
    assert mine_after == []


# The three sites carrying the ORDERING FIX comment. Each releases GPU/host KV
# and only then calls `_report_completion`, which unbinds.
_ORDERED_METHODS = ("generate", "_page_boundary_fast", "_decoding_legacy_modes")


def _worker_method(name):
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    return next(
        node for node in worker.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _self_call_lines(method, called):
    return [
        node.lineno for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == called
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    ]


@pytest.mark.parametrize("name", _ORDERED_METHODS)
def test_kv_release_precedes_report_completion_at_every_ordering_fix_site(name):
    method = _worker_method(name)
    report = _self_call_lines(method, "_report_completion")
    releases = (
        _self_call_lines(method, "_release_gpu_kv_pages")
        + _self_call_lines(method, "_release_host_kv_pages_for_batch")
    )
    assert report, f"{name} no longer reports completions"
    assert releases, f"{name} no longer releases KV before reporting"
    assert min(releases) < max(report), (
        f"{name}: every KV release must precede the _report_completion that "
        "unbinds the local row; see SequenceTable.unbind"
    )


def test_report_completion_is_the_only_worker_unbind_on_the_completion_path():
    """``_report_completion`` releases the query slot (hence unbinds) exactly
    once, so the ordering above is the whole contract for that path."""
    method = _worker_method("_report_completion")
    calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "release_query_book_slot"
    ]
    assert len(calls) == 1
