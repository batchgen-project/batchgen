"""An intentional shutdown interrupting a cohort swap is not a failure.

When a peer dies the worker manager begins a distributed-weight cohort swap
and finishes it after reaping the dead cohort. Stopping the server in that
window makes the daemon refuse the finish, which is expected: no restart
follows, so there is nothing to retry. The handler must then log one reason
line instead of a traceback that claims a retry (observed 2026-11-15 on a
four-node peer-death smoke).

This reads the handler's structure rather than running it: the enclosing
method drives process teardown and a live-behaviour test would have to stub
the whole manager, which would assert the stubs rather than the code.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "batchgen" / "server" / "worker_manager.py"


def _finish_swap_handlers() -> list[ast.Try]:
    """Every ``try`` whose own body statement calls the swap finisher.

    Only direct statements count: the enclosing ``try`` blocks of the failure
    path contain this one and would otherwise match too.
    """
    tree = ast.parse(SOURCE.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            call = stmt.value if isinstance(stmt, ast.Expr) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_finish_distributed_weight_swap"
            ):
                found.append(node)
                break
    return found


def test_the_handler_separates_a_shutdown_from_a_failure():
    handlers = _finish_swap_handlers()
    assert len(handlers) == 1, f"expected one finish-swap handler, found {len(handlers)}"
    (handler,) = handlers
    assert handler.handlers, "the finish must be guarded"
    body = handler.handlers[0].body
    branches = [stmt for stmt in body if isinstance(stmt, ast.If)]
    assert len(branches) == 1, "the handler must tell a shutdown from a failure"
    (branch,) = branches
    # The test is on the manager's own stopping state, not on the text of the
    # error the daemon raised.
    test_src = ast.dump(branch.test)
    assert "_stopping" in test_src, f"branch on self._stopping, got {test_src}"

    def log_calls(stmts):
        return {
            child.func.attr
            for stmt in stmts
            for child in ast.walk(stmt)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }

    stopping = log_calls(branch.body)
    failure = log_calls(branch.orelse)
    assert "info" in stopping, "a shutdown is reported as information"
    assert "exception" not in stopping, "a shutdown logs no traceback"
    assert "exception" in failure, "a real failure keeps its traceback"


def test_only_the_failure_branch_claims_a_retry():
    (handler,) = _finish_swap_handlers()
    (branch,) = [stmt for stmt in handler.handlers[0].body if isinstance(stmt, ast.If)]

    def text(stmts):
        return " ".join(
            node.value
            for stmt in stmts
            for node in ast.walk(stmt)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )

    assert "retries it" not in text(branch.body)
    assert "retries it" in text(branch.orelse)
    assert "shutdown" in text(branch.body)
