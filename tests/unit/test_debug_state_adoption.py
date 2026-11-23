"""A worker runs under one debug-state generation and changes it only at a phase entry.

The server publishes a state; every rank keeps it as pending and adopts it at
the start of a prefill wave or at decode entry, never inside a wave or a decode
step. A change of a key that is baked into captured graphs or into a phase
instance must not reach a running pool this way: the server re-warms the pool
and the new pool starts under the state.

The worker methods are compiled from source (``load_worker_methods``), so these
run without the GPU stack.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from batchgen.config.model_registry import DEBUG_KEY_REGISTRY
from batchgen.debug_state import DebugState, format_log_line
from tests.worker_method_source import load_worker_methods

ROOT = Path(__file__).resolve().parents[2]
WORKER_SOURCE = ROOT / "batchgen" / "batchgen_worker.py"
MAIN_LOOP_SOURCE = ROOT / "batchgen" / "server_worker_main_loop.py"

_METHODS = ("_debug_values", "receive_debug_state", "_adopt_debug_state")


def _worker(*, local_rank=0, node=0):
    methods = load_worker_methods(_METHODS, {
        "Optional": Optional,
        "logging": logging,
        "DebugState": DebugState,
        "DEBUG_KEY_REGISTRY": DEBUG_KEY_REGISTRY,
        "format_debug_state_log_line": format_log_line,
    })
    worker = SimpleNamespace(
        rank=8 * node + local_rank,
        local_rank=local_rank,
        args=SimpleNamespace(nnode_rank=node),
        _debug_state=DEBUG_KEY_REGISTRY.empty,
        _pending_debug_state=None,
        _debug_state_values=None,
        _debug_state_logged_generation=None,
    )
    for name, function in methods.items():
        setattr(worker, name, function.__get__(worker))
    return worker


def _wire(generation, **values):
    state = DEBUG_KEY_REGISTRY.empty
    for _ in range(generation):
        state = state.changed(set=values)
    return state.to_wire()


def test_a_received_state_waits_for_the_next_phase_entry():
    worker = _worker()
    worker.receive_debug_state(_wire(1, glm5_graph_path_log=True))
    # Published, not adopted: a wave or a decode step in flight keeps reading
    # the generation it started under.
    assert worker._debug_values() is None
    assert worker._debug_state.generation == 0

    worker._adopt_debug_state("prefill")

    assert worker._debug_state.generation == 1
    assert worker._debug_values() == {"glm5_graph_path_log": True}
    assert worker._pending_debug_state is None


def test_the_adopted_values_are_plain_and_the_same_object_until_the_next_adoption():
    worker = _worker()
    worker.receive_debug_state(_wire(
        1, weight_pass_end_delay={"rank": 1, "seconds": 2}, decode_profile_ranks=[0, 8]
    ))
    worker._adopt_debug_state("decode")
    values = worker._debug_values()
    # Readers test ``isinstance(value, dict)`` and ``rank in value``.
    assert values == {
        "weight_pass_end_delay": {"rank": 1, "seconds": 2.0},
        "decode_profile_ranks": [0, 8],
    }
    assert type(values["weight_pass_end_delay"]) is dict
    assert worker._debug_values() is values
    worker._adopt_debug_state("prefill")
    assert worker._debug_values() is values


def test_clearing_every_key_returns_the_worker_to_no_debug_values():
    worker = _worker()
    worker.receive_debug_state(_wire(1, triton_strict=True))
    worker._adopt_debug_state("prefill")
    cleared = worker._debug_state.changed(clear=["triton_strict"])
    worker.receive_debug_state(cleared.to_wire())
    worker._adopt_debug_state("decode")
    assert worker._debug_values() is None and worker._debug_state.generation == 2


def test_the_pool_starts_under_the_state_of_its_init_message():
    worker = _worker()
    worker.receive_debug_state(_wire(3, kimi_decode_graph_mode="eager"), at_start=True)
    # In force before anything is captured: no phase entry is needed, and a
    # re-warm key is legal here.
    assert worker._debug_state.generation == 3
    assert worker._debug_values() == {"kimi_decode_graph_mode": "eager"}
    assert worker._pending_debug_state is None


def test_a_pool_started_without_a_state_runs_under_the_empty_one():
    worker = _worker()
    worker.receive_debug_state(None, at_start=True)
    assert worker._debug_state is DEBUG_KEY_REGISTRY.empty
    assert worker._debug_values() is None


@pytest.mark.parametrize("key,value", [
    ("kimi_decode_graph_mode", "eager"),
    ("k3_prefill_profile", True),
    ("glm52_recv_count_hist", True),
])
def test_a_rewarm_key_cannot_change_on_a_running_pool(key, value):
    worker = _worker()
    worker.receive_debug_state(_wire(1, **{key: value}))
    with pytest.raises(RuntimeError, match=f"{key} on a running worker pool"):
        worker._adopt_debug_state("decode")
    # Nothing was adopted.
    assert worker._debug_state.generation == 0 and worker._debug_values() is None


def test_a_non_empty_state_is_logged_at_every_phase_entry_by_the_first_rank_of_each_node(
    caplog,
):
    lines = {}
    for local_rank, node in ((0, 0), (3, 0), (0, 1)):
        worker = _worker(local_rank=local_rank, node=node)
        worker.receive_debug_state(_wire(1, triton_strict=True))
        with caplog.at_level(logging.INFO):
            caplog.clear()
            worker._adopt_debug_state("prefill")
            worker._adopt_debug_state("decode")
            lines[(local_rank, node)] = [
                record.getMessage() for record in caplog.records
                if record.getMessage().startswith("[DEBUG_STATE]")
            ]
    # Every phase served under debug switches says so, so a slice of the log
    # that starts after the adoption still shows the state.
    assert lines[(0, 0)] == [
        "[DEBUG_STATE] generation=1 node=0 keys=triton_strict=True phase=prefill",
        "[DEBUG_STATE] generation=1 node=0 keys=triton_strict=True phase=decode",
    ]
    assert lines[(3, 0)] == []
    assert lines[(0, 1)] == [
        "[DEBUG_STATE] generation=1 node=1 keys=triton_strict=True phase=prefill",
        "[DEBUG_STATE] generation=1 node=1 keys=triton_strict=True phase=decode",
    ]


def test_the_empty_state_is_logged_once_per_generation(caplog):
    worker = _worker()
    with caplog.at_level(logging.INFO):
        worker._adopt_debug_state("prefill")
        worker._adopt_debug_state("decode")
        worker._adopt_debug_state("prefill")
    # The official path stays quiet: one line, then nothing per phase.
    assert [record.getMessage() for record in caplog.records] == [
        "[DEBUG_STATE] generation=0 node=0 keys=none phase=prefill"
    ]


# ------------------------------------------------------------- source pins

def _worker_method(name):
    tree = ast.parse(WORKER_SOURCE.read_text())
    klass = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    return next(node for node in klass.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(function, attr):
    return [node for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr]


def test_adoption_happens_at_exactly_the_two_phase_entries():
    tree = ast.parse(WORKER_SOURCE.read_text())
    sites = [ast.unparse(node.args[0]) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "_adopt_debug_state"]
    assert sorted(sites) == ["'decode'", "'prefill'"]
    assert _calls(_worker_method("_begin_prefill_wave"), "_adopt_debug_state")


def test_a_published_state_is_broadcast_like_an_admission_and_admits_nothing():
    poll = _worker_method("_poll_admissions")
    source = ast.unparse(poll)
    assert "msg.get('type') in ('admit', 'debug_state')" in source
    branch = next(
        node for node in ast.walk(poll)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "msg_data.get('type') == 'debug_state'"
    )
    assert _calls(branch, "receive_debug_state")
    assert isinstance(branch.body[-1], ast.Return)
    assert ast.unparse(branch.body[-1].value) == "False"
    # The broadcast precedes the branch: every rank receives the same message.
    broadcast = next(node for node in ast.walk(poll)
                     if isinstance(node, ast.Call)
                     and ast.unparse(node.func) == "dist.broadcast_object_list")
    assert broadcast.lineno < branch.lineno


def test_no_worker_code_reads_debug_flags_from_a_sequence():
    source = WORKER_SOURCE.read_text()
    assert "_active_batchgen_debug_for_sequences" not in source
    assert "seq.batchgen_debug" not in source
    assert "_any_sequence_batchgen_debug" not in source
    sequence = (ROOT / "batchgen" / "sequence.py").read_text()
    assert "batchgen_debug" not in sequence


def test_the_main_loop_takes_a_state_published_before_init_and_the_init_state():
    source = MAIN_LOOP_SOURCE.read_text()
    assert 'task_data.get("type") == "debug_state"' in source
    assert 'worker.receive_debug_state(task_data.get("state"), at_start=True)' in source
    assert 'worker.receive_debug_state(task_data.get("debug_state"), at_start=True)' in source
    early = source.index('task_data.get("type") == "debug_state"')
    init = source.index('task_data.get("type") == "init"')
    assert early < init
