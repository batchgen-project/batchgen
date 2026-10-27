"""Every streamed model forward of the worker runs inside one weight pass.

The core engine leases a weight only inside a pass (begin_weight_pass /
end_weight_pass), so the worker arms one per forward, carries the next one
while another forward follows, and closes every pass it armed before a loop
exits or a phase installs its weight-copy lists. A forward that raises stops
the producer and aborts its passes instead of draining them.

batchgen_worker imports the whole engine, so ``_WeightPasses`` is compiled in
isolation against a fake core engine and the wiring is checked statically.
"""
import ast
import contextlib
import copy
import logging
import re
import time
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
TREE = ast.parse(WORKER.read_text())
WORKER_CLASS = next(
    node for node in TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
)


def _load_weight_passes():
    klass = next(
        node for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "_WeightPasses"
    )
    namespace = {
        "contextmanager": contextlib.contextmanager,
        "logging": logging,
        "time": time,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(klass)], type_ignores=[])), str(WORKER), "exec"),
        namespace)
    return namespace["_WeightPasses"]


WeightPasses = _load_weight_passes()


class _CleanupError(RuntimeError):
    pass


class _ForwardError(RuntimeError):
    pass


class _StallError(RuntimeError):
    pass


class FakeEngine:
    """Records the pass calls and enforces the engine's pass contract."""

    def __init__(self, installed=True, cancel_error=None, end_error=None,
                 stop_error=None, abort_error=None):
        self.installed = installed
        self.cancel_error = cancel_error
        self.end_error = end_error
        self.stop_error = stop_error
        self.abort_error = abort_error
        self.producer_running = True
        self.calls = []
        self.open = []
        self._next = 1

    def stop_h2d_worker(self):
        # The engine joins the producer, then raises a recorded failure.
        self.producer_running = False
        self.calls.append("stop")
        if self.stop_error is not None:
            raise self.stop_error

    def abort_weight_passes(self):
        assert not self.producer_running, "abort requires a stopped producer"
        aborted = list(self.open)
        self.open = []
        self.calls.append(("abort", aborted))
        if self.abort_error is not None:
            raise self.abort_error
        return {"passes": len(aborted), "pending_skips": 0, "freed": {},
                "daemon_leases": 0, "schedule_lost": False}

    def begin_weight_pass(self):
        if not self.installed:
            self.calls.append(("begin", 0))
            return 0
        assert len(self.open) < 2, "at most two passes may be open"
        pass_id = self._next
        self._next += 1
        self.open.append(pass_id)
        self.calls.append(("begin", pass_id))
        return pass_id

    def end_weight_pass(self, pass_id):
        assert pass_id != 0
        assert self.open and self.open[0] == pass_id, "only the oldest pass ends"
        if self.end_error is not None:
            # The drain timed out: the pass stays open.
            self.calls.append(("end failed", pass_id))
            raise self.end_error
        self.open.pop(0)
        self.calls.append(("end", pass_id))
        return {
            "pass": pass_id,
            "copied": {"routed_expert": 3},
            "skipped": {"routed_expert": 1},
            "released_on_landing": {},
            "carried": 2,
        }

    def cancel_weight_pass(self, pass_id):
        assert pass_id != 0
        assert pass_id in self.open
        self.calls.append(("cancel", pass_id))
        if self.cancel_error is not None:
            raise self.cancel_error
        return {"pass": pass_id, "freed_present": 0,
                "release_on_landing": 0, "pending": 0}


def test_plain_pass_begins_before_and_ends_after_the_forward(caplog):
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=3)

    with caplog.at_level(logging.INFO):
        with passes.scope("prefill"):
            passes.begin()
            engine.calls.append("forward")
            passes.end()

    assert engine.calls == [("begin", 1), "forward", ("end", 1)]
    assert engine.open == []
    assert passes.totals == {
        "passes": 1,
        "copied": {"routed_expert": 3},
        "skipped": {"routed_expert": 1},
        "released_on_landing": {},
        "carried": 2,
    }
    assert (
        "[WEIGHT_PASS] rank 3 pass 1 copied=routed_expert:3 "
        "skipped=routed_expert:1 released_on_landing=- carried=2"
    ) in caplog.messages


def test_carry_arms_the_next_pass_at_the_start_and_the_next_forward_reuses_it():
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=0)

    with passes.scope("prefill"):
        for index in range(3):
            passes.begin(carry=index + 1 < 3)
            engine.calls.append(f"forward {index}")
            passes.end()

    assert engine.calls == [
        ("begin", 1), ("begin", 2), "forward 0", ("end", 1),
        ("begin", 3), "forward 1", ("end", 2),
        "forward 2", ("end", 3),
    ]
    assert engine.open == []
    assert passes.totals["passes"] == 3


def test_leaving_the_scope_cancels_and_ends_a_carried_pass_that_will_not_run():
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=0)

    with passes.scope("decode"):
        passes.begin(carry=True)
        passes.end()

    assert engine.calls == [
        ("begin", 1), ("begin", 2), ("end", 1), ("cancel", 2), ("end", 2),
    ]
    assert engine.open == []
    # A phase install closes again; nothing is left to close.
    passes.close()
    assert engine.calls[-1] == ("end", 2)


def test_an_install_time_close_ends_a_pass_left_open():
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=0)
    # A carried pass whose loop never reached its scope exit.
    passes._carried = engine.begin_weight_pass()

    passes.close()

    assert engine.calls == [("begin", 1), ("cancel", 1), ("end", 1)]
    assert engine.open == []


def test_a_forward_that_raises_stops_the_producer_then_aborts_its_passes():
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=0)
    failure = _ForwardError("forward failed")

    with pytest.raises(_ForwardError) as raised:
        with passes.scope("prefill"):
            passes.begin(carry=True)
            raise failure

    assert raised.value is failure
    assert raised.value.__context__ is None
    # No cancel and no end: ending would drain the producer.
    assert engine.calls == [("begin", 1), ("begin", 2), "stop", ("abort", [1, 2])]
    assert engine.open == []
    assert passes._current is None and passes._carried is None
    # The next install closes again; nothing is left to close.
    passes.close()
    assert engine.calls[-1] == ("abort", [1, 2])


def test_an_end_that_cannot_drain_aborts_and_reraises_the_drain_error():
    stall = _StallError("weight stream stalled")
    engine = FakeEngine(end_error=stall)
    passes = WeightPasses(engine, rank=0)

    with pytest.raises(_StallError) as raised:
        with passes.scope("decode"):
            passes.begin(carry=True)
            passes.end()

    assert raised.value is stall
    assert engine.calls == [
        ("begin", 1), ("begin", 2), ("end failed", 1), "stop", ("abort", [1, 2]),
    ]
    assert passes._current is None and passes._carried is None


def test_a_raise_with_no_open_pass_makes_no_engine_call():
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=0)

    with pytest.raises(_ForwardError):
        with passes.scope("prefill"):
            passes.begin()
            passes.end()
            raise _ForwardError("after the last forward")

    assert engine.calls == [("begin", 1), ("end", 1)]
    assert engine.producer_running

    nothing_installed = FakeEngine(installed=False)
    passes = WeightPasses(nothing_installed, rank=0)
    with pytest.raises(_ForwardError):
        with passes.scope("decode"):
            passes.begin(carry=True)
            raise _ForwardError("forward failed")
    assert nothing_installed.calls == [("begin", 0)]


def test_a_producer_failure_from_the_stop_is_chained_and_the_abort_still_runs():
    producer = _CleanupError("HtoD producer failed: copy failed")
    engine = FakeEngine(stop_error=producer)
    passes = WeightPasses(engine, rank=0)
    failure = _ForwardError("forward failed")

    with pytest.raises(_ForwardError) as raised:
        with passes.scope("prefill"):
            passes.begin()
            raise failure

    assert raised.value is failure
    assert raised.value.__context__ is producer
    assert engine.calls == [("begin", 1), "stop", ("abort", [1])]
    assert engine.open == []


def test_a_failed_abort_is_chained_and_forgets_the_passes():
    cleanup = _CleanupError("abort refused")
    engine = FakeEngine(abort_error=cleanup)
    passes = WeightPasses(engine, rank=0)
    failure = _ForwardError("forward failed")

    with pytest.raises(_ForwardError) as raised:
        with passes.scope("prefill"):
            passes.begin(carry=True)
            raise failure

    assert raised.value is failure
    assert raised.value.__context__ is cleanup
    # Nothing is cancelled or ended later for a pass the engine dropped.
    assert passes._current is None and passes._carried is None
    passes.close()
    assert engine.calls[-1] == ("abort", [1, 2])


def test_stop_and_abort_failures_are_both_chained_behind_the_forward_error():
    producer = _CleanupError("HtoD producer failed")
    cleanup = _CleanupError("abort refused")
    engine = FakeEngine(stop_error=producer, abort_error=cleanup)
    passes = WeightPasses(engine, rank=0)
    failure = _ForwardError("forward failed")

    with pytest.raises(_ForwardError) as raised:
        with passes.scope("prefill"):
            passes.begin()
            raise failure

    assert raised.value is failure
    assert raised.value.__context__ is cleanup
    assert cleanup.__context__ is producer


def test_a_cancel_failure_on_a_normal_exit_is_raised():
    cleanup = _CleanupError("cancel refused")
    engine = FakeEngine(cancel_error=cleanup)
    passes = WeightPasses(engine, rank=0)

    with pytest.raises(_CleanupError) as raised:
        with passes.scope("decode"):
            passes.begin(carry=True)
            passes.end()

    assert raised.value is cleanup
    assert "stop" not in engine.calls


def test_pass_zero_is_a_no_op(caplog):
    engine = FakeEngine(installed=False)
    passes = WeightPasses(engine, rank=0)

    with caplog.at_level(logging.INFO):
        with passes.scope("decode"):
            for _ in range(2):
                passes.begin(carry=True)
                passes.end()

    # Nothing installed: one begin per forward, no carry, end or cancel.
    assert engine.calls == [("begin", 0), ("begin", 0)]
    assert passes.totals["passes"] == 0
    assert not any("[WEIGHT_PASS]" in message for message in caplog.messages)


def test_decode_logs_one_summary_per_loop_exit(caplog):
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=5)

    with caplog.at_level(logging.INFO):
        with passes.scope("decode"):
            for _ in range(4):
                passes.begin(carry=True)
                passes.end()

    lines = [m for m in caplog.messages if "[WEIGHT_PASS]" in m]
    # Four steps plus the carried pass of the step that never ran.
    assert lines == [
        "[WEIGHT_PASS] rank 5 decode passes=5 copied=routed_expert:15 "
        "skipped=routed_expert:5 released_on_landing=- carried=10"
    ]


def test_misuse_fails_loud():
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=0)

    with pytest.raises(RuntimeError, match="outside a scope"):
        passes.begin()
    with pytest.raises(RuntimeError, match="no weight pass to end"):
        passes.end()
    with passes.scope("prefill"):
        with pytest.raises(RuntimeError, match="inside the prefill one"):
            with passes.scope("decode"):
                pass
        passes.begin()
        with pytest.raises(RuntimeError, match="was not ended"):
            passes.begin()
        passes.end()
    assert engine.open == []


# --------------------------------------------------------------------------- #
#  Static wiring of the worker
# --------------------------------------------------------------------------- #


def _method(name):
    return next(
        node for node in WORKER_CLASS.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _parents(function):
    parents = {}
    for node in ast.walk(function):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _ancestors(node, parents):
    while node in parents:
        node = parents[node]
        yield node


def _is_pass_call(node, name):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and ast.unparse(node.func.value) == "self._weight_passes"
    )


def _scope_phases(with_node):
    return [
        item.context_expr.args[0].value
        for item in with_node.items
        if _is_pass_call(item.context_expr, "scope")
    ]


def _forward_calls(function, matches):
    return [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call) and matches(ast.unparse(node.func))
    ]


def _assert_in_a_pass(function, call, phase=None):
    """``call`` runs between a begin and an end of the forward loop's
    iteration, and that loop runs inside a weight-pass scope (of this or the
    calling method). The forward loop is the innermost enclosing loop that
    begins passes: a decoder-layer call also sits inside the layer loop."""
    parents = _parents(function)
    loop = next(
        node for node in _ancestors(call, parents)
        if isinstance(node, (ast.For, ast.While))
        and any(_is_pass_call(n, "begin") for n in ast.walk(node))
    )
    begins = [n.lineno for n in ast.walk(loop) if _is_pass_call(n, "begin")]
    ends = [n.lineno for n in ast.walk(loop) if _is_pass_call(n, "end")]
    assert any(line < call.lineno for line in begins), ast.unparse(call)
    assert any(line > call.lineno for line in ends), ast.unparse(call)
    scopes = [
        phases
        for node in _ancestors(loop, parents)
        if isinstance(node, ast.With)
        for phases in _scope_phases(node)
    ]
    if phase is not None:
        assert scopes == [phase], ast.unparse(call)
    return scopes


def _assert_called_in_scope(caller, callee, phase):
    parents = _parents(_method(caller))
    calls = _forward_calls(_method(caller), lambda name: name == f"self.{callee}")
    assert calls, (caller, callee)
    for call in calls:
        phases = [
            p for node in _ancestors(call, parents)
            if isinstance(node, ast.With) for p in _scope_phases(node)
        ]
        assert phases == [phase], (caller, callee)


def test_every_prefill_forward_runs_inside_a_weight_pass():
    for name in ("prefill", "prefill_prepacked"):
        calls = _forward_calls(_method(name), _is_model_forward)
        assert calls, name
        for call in calls:
            _assert_in_a_pass(_method(name), call, phase="prefill")

    pool = _method("_prefill_prepacked_pool")
    calls = _forward_calls(pool, lambda func: func == "decoder_layer")
    assert len(calls) == 2
    for call in calls:
        # The pooled wave's scope is opened by its only caller.
        assert _assert_in_a_pass(pool, call) == []
    _assert_called_in_scope("prefill_prepacked", "_prefill_prepacked_pool", "prefill")


def test_every_transport_only_prefill_pass_is_a_weight_pass():
    generate = _method("generate")
    calls = _forward_calls(
        generate,
        lambda func: func in {
            "self.parallel_manager.run_streamed_sp8_transport_only_prefill",
            "self.parallel_manager.run_resident_ep_collective_only_prefill",
        },
    )
    assert len(calls) == 2
    for call in calls:
        _assert_in_a_pass(generate, call, phase="prefill")


def test_every_decode_step_runs_inside_a_weight_pass():
    continuous = _method("decoding_continuous")
    calls = _forward_calls(
        continuous, lambda func: func == "self._glm5_decode_model_forward"
    )
    assert len(calls) == 1
    # The decode loop's scope is opened by its caller.
    assert _assert_in_a_pass(continuous, calls[0]) == []

    legacy = _method("_decoding_legacy_modes")
    calls = _forward_calls(legacy, lambda func: func == "self.model")
    assert len(calls) == 2
    for call in calls:
        assert _assert_in_a_pass(legacy, call) == []

    # One scope around the whole call, so every return of the decode loop
    # closes the pass the last step carried. The legacy modes run inside the
    # same call.
    _assert_called_in_scope("generate", "decoding_continuous", "decode")
    legacy_callers = {
        function.name
        for function in WORKER_CLASS.body
        if isinstance(function, ast.FunctionDef)
        and _forward_calls(
            function, lambda func: func == "self._decoding_legacy_modes"
        )
    }
    assert legacy_callers == {"decoding_continuous"}


# Every way the worker drives the model: the model or its inner model (called
# or ``.forward``), a decoder layer, an indexed ``layers[...]`` entry, or a
# probing re-run of the model.
MODEL_FORWARD = re.compile(
    r"self\.model(\.model)?(\.forward)?"
    r"|decoder_layer"
    r"|.*\.layers\[.+\](\.forward)?"
    r"|.*\brun_model_with_probes"
)

# Model forwards that run outside every weight pass, each with its reason.
_GRAPH_COMPARE_RERUN = (
    "GLM-5 whole-model graph compare: the eager re-run of a step that a "
    "CUDA graph replayed. A graph cannot stream, so every weight it reads is "
    "resident and the re-run leases nothing; a streamed weight would fail "
    "loud in get_weights, which refuses a lease outside a pass."
)
UNSTREAMED_FORWARDS = {
    ("decoding_continuous", "self.model.model"): _GRAPH_COMPARE_RERUN,
    ("decoding_continuous", "self._whole_model_segment.run_model_with_probes"):
        _GRAPH_COMPARE_RERUN,
}


def _is_model_forward(func):
    return MODEL_FORWARD.fullmatch(func) is not None


def test_no_other_model_forward_escapes_the_weight_passes():
    """Every eager model forward or decoder-layer drive is one of the wrapped
    sites or a listed unstreamed one; ``_glm5_decode_model_forward`` is the
    forward itself."""
    wrapped = {
        "prefill", "prefill_prepacked", "_prefill_prepacked_pool",
        "_decoding_legacy_modes", "_glm5_decode_model_forward",
    }
    found = set()
    for function in WORKER_CLASS.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for call in _forward_calls(function, _is_model_forward):
            site = (function.name, ast.unparse(call.func))
            if site in UNSTREAMED_FORWARDS:
                found.add(site)
                continue
            assert function.name in wrapped, site
    # The list holds no stale entry.
    assert found == set(UNSTREAMED_FORWARDS)
    callers = {
        function.name
        for function in WORKER_CLASS.body
        if isinstance(function, ast.FunctionDef)
        and _forward_calls(
            function, lambda func: func == "self._glm5_decode_model_forward"
        )
    }
    assert callers == {"decoding_continuous"}


def test_the_unstreamed_forwards_only_rerun_a_graph_replay():
    """Each listed site sits in the graph-replay branch of a decode step,
    which arms no pass, inside the compare's forced-eager block."""
    for (name, callee), reason in UNSTREAMED_FORWARDS.items():
        function = _method(name)
        parents = _parents(function)
        calls = _forward_calls(function, lambda func: func == callee)
        assert len(calls) == 1, (name, callee)
        lineage = [calls[0], *_ancestors(calls[0], parents)]
        assert any(
            isinstance(node, ast.With)
            and any(
                ast.unparse(item.context_expr)
                == "self._glm5_force_segmented_graph_eager()"
                for item in node.items
            )
            for node in lineage
        ), reason
        graph_branch = next(
            node for node in lineage
            if isinstance(node, ast.If) and ast.unparse(node.test) == "_use_graph"
        )
        assert any(statement in lineage for statement in graph_branch.body), reason
        branch_calls = [
            node for statement in graph_branch.body
            for node in ast.walk(statement) if isinstance(node, ast.Call)
        ]
        assert any(
            ast.unparse(node.func) == "self._cuda_graph_manager.replay"
            for node in branch_calls
        ), reason
        assert not any(
            _is_pass_call(node, "begin") for node in branch_calls
        ), reason


def test_worker_calls_no_removed_weight_buffer_binding():
    removed = {"clear_weight_copy_queue", "get_weights_pinned", "clear_expert_buffer"}
    called = {
        node.func.attr
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not removed & called


def test_profiled_prefill_records_the_pass_counts_without_stopping_the_producer():
    prefill = _method("prefill_prepacked")
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "stop_h2d_worker"
        for node in ast.walk(prefill)
    )
    record = next(
        node for node in ast.walk(prefill)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "weight_stream"
            for key in node.keys
        )
    )
    values = {key.value: ast.unparse(value)
              for key, value in zip(record.keys, record.values)}
    assert values["weight_stream"] == "self.core_engine.get_weight_stream_profile()"
    assert values["weight_passes"] == "self._weight_passes.totals"


INSTALL_PATHS = (
    "_install_prefill_weight_copy_pipeline",
    "_load_decode_model",
    "_activate_persistent_decode_instance",
)


def test_every_install_path_closes_the_weight_passes_before_stopping():
    for name in INSTALL_PATHS:
        function = _method(name)
        closes = [n.lineno for n in ast.walk(function) if _is_pass_call(n, "close")]
        stops = [
            n.lineno for n in ast.walk(function)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "stop_h2d_worker"
        ]
        assert closes and stops, name
        assert min(closes) < min(stops), name

    # No other worker method stops the producer, so no install path is
    # missed. _WeightPasses._abort stops it only after a forward raised.
    stopping = {
        function.name
        for function in WORKER_CLASS.body
        if isinstance(function, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "stop_h2d_worker"
            for n in ast.walk(function)
        )
    }
    assert stopping == set(INSTALL_PATHS)


def test_every_phase_installs_its_schedule_through_one_installer():
    installing = {
        function.name
        for function in WORKER_CLASS.body
        if isinstance(function, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_weight_copy_queue"
            for n in ast.walk(function)
        )
    }
    assert installing == {
        "_install_prefill_weight_copy_pipeline",
        "_install_decode_weight_copy_queue",
    }
    # The persistent decode activation installs the decode schedule too, so
    # its decode forwards never see the prefill lists.
    activation = _method("_activate_persistent_decode_instance")
    assert any(
        isinstance(n, ast.Call)
        and ast.unparse(n.func) == "self._install_decode_weight_copy_queue"
        for n in ast.walk(activation)
    )


def test_the_weight_passes_are_bound_to_the_core_engine():
    source = ast.unparse(_method("_initialize_core_components"))
    assert "self._weight_passes = _WeightPasses(self.core_engine, self.rank)" in source
    assert "self._weight_passes = None" in ast.unparse(_method("__init__"))


def test_end_delay_sleeps_before_closing_a_prefill_pass_only(monkeypatch):
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=0)
    passes.end_delay_s = 2.5
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append((s, list(engine.open))))

    with passes.scope("prefill"):
        passes.begin()
        passes.end()
    # The delay runs while the pass is still open, i.e. before end_weight_pass.
    assert sleeps == [(2.5, [1])]

    with passes.scope("decode"):
        passes.begin()
        passes.end()
    assert len(sleeps) == 1


def test_weight_pass_end_delay_parses_only_this_rank():
    method = next(
        node for node in WORKER_CLASS.body
        if isinstance(node, ast.FunctionDef) and node.name == "_weight_pass_end_delay"
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(method)], type_ignores=[])), str(WORKER), "exec"),
        namespace)
    parse = namespace["_weight_pass_end_delay"]

    class _Worker:
        rank = 5

    assert parse(_Worker(), None) == 0.0
    assert parse(_Worker(), {"rank": 5, "seconds": 3}) == 3.0
    assert parse(_Worker(), {"rank": 4, "seconds": 3}) == 0.0
    for bad in ({"rank": 5}, {"rank": "5", "seconds": 1}, {"rank": 5, "seconds": 0},
                {"rank": 5, "seconds": 61}, 7):
        with pytest.raises(ValueError):
            parse(_Worker(), bad)


def test_prefill_config_installs_the_end_delay_before_the_forward():
    method = next(
        node for node in WORKER_CLASS.body
        if isinstance(node, ast.FunctionDef) and node.name == "_config_prefill_for_batch"
    )
    source = ast.unparse(method)
    assert "self._weight_passes.end_delay_s = self._weight_pass_end_delay(" in source
