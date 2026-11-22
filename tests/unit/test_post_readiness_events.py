"""Post-readiness events: one marked log line and one count per compile or capture."""

import ast
import logging
from pathlib import Path

import pytest

from batchgen_kernels import post_readiness as pr

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh():
    pr._reset_for_tests()
    yield
    pr._reset_for_tests()


def test_nothing_is_recorded_before_the_rank_is_ready(caplog):
    with caplog.at_level(logging.INFO, logger=pr.__name__):
        assert pr.record(pr.GRAPH_CAPTURE, "startup capture") is False
    assert pr.counts() == {kind: 0 for kind in pr.KINDS}
    assert pr.MARKER not in caplog.text


def test_an_event_after_readiness_is_one_marked_line_and_a_count(caplog):
    with caplog.at_level(logging.INFO, logger=pr.__name__):
        pr.arm(3)
        assert pr.record(pr.GRAPH_CAPTURE, "decode graph\n capture #2") is True
        assert pr.record(pr.DEEPGEMM_COMPILE, "decode configure", count=4) is True
        assert pr.record(pr.DEEPGEMM_COMPILE, "decode configure", count=1) is True
    lines = [r.getMessage() for r in caplog.records if "kind=" in r.getMessage()]
    assert lines == [
        "[POST_READINESS] rank=3 kind=graph_capture count=1 total=1 where=decode graph capture #2",
        "[POST_READINESS] rank=3 kind=deepgemm_compile count=4 total=4 where=decode configure",
        "[POST_READINESS] rank=3 kind=deepgemm_compile count=1 total=5 where=decode configure",
    ]
    assert all(r.levelno == logging.WARNING for r in caplog.records if "kind=" in r.getMessage())
    assert pr.counts() == {pr.TRITON_COMPILE: 0, pr.DEEPGEMM_COMPILE: 5, pr.GRAPH_CAPTURE: 1}


def test_arming_twice_logs_once_and_an_unknown_kind_is_refused(caplog):
    with caplog.at_level(logging.INFO, logger=pr.__name__):
        pr.arm(0)
        pr.arm(0)
    assert sum("armed" in r.getMessage() for r in caplog.records) == 1
    with pytest.raises(ValueError, match="unknown post-readiness event kind"):
        pr.record("autotune", "somewhere")
    assert pr.record(pr.TRITON_COMPILE, "x", count=0) is False


def _calls(path: str, func: str):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            return {
                f"{sub.func.attr}:{ast.unparse(sub.args[0]) if sub.args else ''}"
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "post_readiness"
            }
    raise AssertionError(f"{func} not found in {path}")


@pytest.mark.parametrize("path, func, call", [
    ("batchgen/batchgen_worker.py", "arm_late_compile_guard_at_readiness", "arm:self.rank"),
    ("batchgen/batchgen_worker.py", "_record_decode_graph_capture",
     "record:post_readiness.GRAPH_CAPTURE"),
    ("batchgen/models/glm/glm5/deepgemm_prewarm.py", "check_late_compiles",
     "record:post_readiness.DEEPGEMM_COMPILE"),
    ("batchgen_kernels/triton_prewarm.py", "cache_hook", "record:post_readiness.TRITON_COMPILE"),
    ("batchgen_kernels/triton_prewarm.py", "post_compile_hook",
     "record:post_readiness.TRITON_COMPILE"),
])
def test_every_event_source_reports_to_the_record(path, func, call):
    assert call in _calls(path, func)
