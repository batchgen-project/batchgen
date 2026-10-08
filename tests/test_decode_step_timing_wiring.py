"""Wiring of the decode step timing into ``BatchGenWorker.decoding_continuous``.

batchgen_worker imports the whole engine, so these are static checks on the
shipping method body. The ring and split logic themselves are unit-tested in
``tests/worker/test_step_timing.py``.
"""

import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _decoding_continuous():
    source = WORKER.read_text()
    tree = ast.parse(source)
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    fn = next(
        node for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "decoding_continuous"
    )
    return source, fn


def _decode_loop(fn):
    return next(
        node for node in fn.body
        if isinstance(node, ast.While) and ast.unparse(node.test) == "decode_uuids"
    )


def _index(body, predicate, what):
    hits = [i for i, stmt in enumerate(body) if predicate(stmt)]
    assert len(hits) == 1, f"expected exactly one {what} in the decode loop body, got {hits}"
    return hits[0]


def _assigns(name):
    def predicate(stmt):
        return (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
        )
    return predicate


def _calls(func_src):
    def predicate(stmt):
        value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign)) else None
        return isinstance(value, ast.Call) and ast.unparse(value.func) == func_src
    return predicate


def _loop_exits(stmts):
    """Break/Continue nodes in ``stmts`` that would leave the ENCLOSING loop."""
    found = []

    def visit(node):
        if isinstance(node, (ast.Break, ast.Continue)):
            found.append(node)
            return
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # break/continue inside an inner loop body target that inner loop.
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in stmts:
        visit(stmt)
    return found


def test_forward_event_pair_brackets_forward_and_sample():
    _, fn = _decoding_continuous()
    body = _decode_loop(fn).body

    t0 = _index(body, _assigns("_split_t0"), "_split_t0")
    begin = _index(body, _calls("_fwd_ring.begin"), "_fwd_ring.begin")
    forward = _index(
        body,
        lambda s: isinstance(s, ast.With)
        and ast.unparse(s.items[0].context_expr) == "torch.inference_mode()",
        "inference_mode forward block",
    )
    t1 = _index(body, _assigns("_split_t1"), "_split_t1")
    end = _index(body, _calls("_fwd_ring.end"), "_fwd_ring.end")
    token_copy = _index(body, _calls("_new_tokens_pinned[:bs].copy_"), "token copy")

    assert begin == t0 + 1 and begin < forward < t1 < end < token_copy
    # The end marker is recorded on the stream that produced the tokens.
    assert "current_stream(self.torch_device)" in ast.unparse(body[end])
    # Every begun slot reaches end(): nothing between them can leave the
    # decode-loop iteration early.
    assert _loop_exits(body[begin + 1:end]) == []


def test_forward_events_are_timing_events_harvested_after_the_split():
    _, fn = _decoding_continuous()
    ring_ctor = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "ForwardEventRing"
    ]
    assert len(ring_ctor) == 1
    assert ast.unparse(ring_ctor[0].args[0]) == "FORWARD_EVENT_RING_SLOTS"
    assert "torch.cuda.Event(enable_timing=True)" in ast.unparse(ring_ctor[0].args[1])

    body = _decode_loop(fn).body
    t5 = _index(body, _assigns("_split_t5"), "_split_t5")
    accumulate = _index(body, _calls("accumulate_decode_step_split"), "split accumulate")
    harvest = _index(body, _calls("_fwd_ring.harvest"), "_fwd_ring.harvest")
    assert t5 < accumulate < harvest
    marks = ast.unparse(body[accumulate].value.args[1])
    assert marks == (
        "(forward_start, _split_t0, _split_t1, _split_t2, _split_t3, _split_t4, _split_t5)"
    )


def test_decode_loop_never_captures_a_cuda_graph():
    """The ring's record points live in this method; it must only replay."""
    _, fn = _decoding_continuous()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = ast.unparse(node.func)
            for forbidden in (
                "torch.cuda.graph",
                "CUDAGraph",
                "capture_begin",
                "capture_end",
                "warmup_and_capture",
                "_capture_one",
                "capture_graph",
            ):
                assert forbidden not in func, f"decode loop calls {func}"


def test_heartbeat_lines_use_the_new_split_labels():
    source, fn = _decoding_continuous()
    # The old labels changed meaning; no log line in the worker may keep
    # emitting them (string literals, including f-string parts).
    literals = [
        node.value for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    for old in ("forward+sample", "kv_flush", "token_readback"):
        assert not [s for s in literals if old in s], old
    # The split accumulator is round-local, never a worker attribute that a
    # hot reload could carry across a slot-layout change.
    assert not [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Attribute) and node.attr == "_decode_step_split"
    ]
    calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    # The 30 s heartbeat and the interval-end line.
    assert calls.count("format_decode_step_split") == 2
    assert calls.count("_fwd_ring.take_summary") == 2
