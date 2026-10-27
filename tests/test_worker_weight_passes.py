"""Every streamed model forward of the worker runs inside one weight pass.

The core engine leases a weight only inside a pass (begin_weight_pass /
end_weight_pass), so the worker arms one per forward, carries the next one
while another forward follows, and closes every pass it armed before a loop
exits or a phase installs its weight-copy lists.

batchgen_worker imports the whole engine, so ``_WeightPasses`` is compiled in
isolation against a fake core engine and the wiring is checked statically.
"""
import ast
import contextlib
import copy
import logging
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
    namespace = {"contextmanager": contextlib.contextmanager, "logging": logging}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(klass)], type_ignores=[])), str(WORKER), "exec"),
        namespace)
    return namespace["_WeightPasses"]


WeightPasses = _load_weight_passes()


class _CleanupError(RuntimeError):
    pass


class _ForwardError(RuntimeError):
    pass


class FakeEngine:
    """Records the pass calls and enforces the engine's pass contract."""

    def __init__(self, installed=True, cancel_error=None):
        self.installed = installed
        self.cancel_error = cancel_error
        self.calls = []
        self.open = []
        self._next = 1

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


def test_a_forward_that_raises_cancels_and_ends_its_passes_and_reraises():
    engine = FakeEngine()
    passes = WeightPasses(engine, rank=0)
    failure = _ForwardError("forward failed")

    with pytest.raises(_ForwardError) as raised:
        with passes.scope("prefill"):
            passes.begin(carry=True)
            raise failure

    assert raised.value is failure
    assert engine.calls == [
        ("begin", 1), ("begin", 2),
        ("cancel", 1), ("end", 1), ("cancel", 2), ("end", 2),
    ]
    assert engine.open == []
    # The scope is reusable once it has unwound.
    with passes.scope("prefill"):
        passes.begin()
        passes.end()
    assert engine.open == []


def test_a_cleanup_failure_is_chained_and_never_masks_the_forward_error():
    cleanup = _CleanupError("cancel refused")
    engine = FakeEngine(cancel_error=cleanup)
    passes = WeightPasses(engine, rank=0)
    failure = _ForwardError("forward failed")

    with pytest.raises(_ForwardError) as raised:
        with passes.scope("prefill"):
            passes.begin()
            raise failure

    assert raised.value is failure
    assert raised.value.__context__ is cleanup


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
        calls = _forward_calls(
            _method(name),
            lambda func: func in {"self.model", "decoder_layer"},
        )
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


def test_no_other_model_forward_escapes_the_weight_passes():
    """Every eager ``self.model(...)`` / decoder-layer drive is one of the
    wrapped sites; ``_glm5_decode_model_forward`` is the forward itself."""
    wrapped = {
        "prefill", "prefill_prepacked", "_prefill_prepacked_pool",
        "_decoding_legacy_modes", "_glm5_decode_model_forward",
    }
    for function in WORKER_CLASS.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        calls = _forward_calls(
            function, lambda func: func in {"self.model", "decoder_layer"}
        )
        if calls:
            assert function.name in wrapped, function.name
    callers = {
        function.name
        for function in WORKER_CLASS.body
        if isinstance(function, ast.FunctionDef)
        and _forward_calls(
            function, lambda func: func == "self._glm5_decode_model_forward"
        )
    }
    assert callers == {"decoding_continuous"}


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

    # No other method stops the producer, so no install path is missed.
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
