"""One ``BatchGenWorker`` per process owns the model phase state.

The guard is exercised directly with stand-in owners; the worker side is
pinned at source level, because importing ``batchgen.batchgen_worker`` needs
the GPU toolchain.
"""

from __future__ import annotations

import ast
import gc
from pathlib import Path

import pytest

from batchgen.models.wrappers import phase_state
from batchgen.models.wrappers.phase_state import (
    claim_process_phase_state,
    release_process_phase_state,
)

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
MAIN_LOOP = ROOT / "batchgen" / "server_worker_main_loop.py"


class _Owner:
    pass


@pytest.fixture(autouse=True)
def _no_claim_leaks_between_tests():
    phase_state._owner_ref = None
    yield
    phase_state._owner_ref = None


def test_a_second_live_owner_is_refused():
    first, second = _Owner(), _Owner()
    claim_process_phase_state(first)
    with pytest.raises(
        RuntimeError,
        match="^a BatchGenWorker already owns the model phase state of this process$",
    ):
        claim_process_phase_state(second)
    # The refused claim did not displace the holder.
    with pytest.raises(RuntimeError):
        claim_process_phase_state(_Owner())
    claim_process_phase_state(first)


def test_the_same_owner_may_claim_again():
    owner = _Owner()
    claim_process_phase_state(owner)
    claim_process_phase_state(owner)


def test_release_lets_a_new_owner_claim():
    first, second = _Owner(), _Owner()
    claim_process_phase_state(first)
    release_process_phase_state(first)
    claim_process_phase_state(second)


def test_release_by_a_non_holder_is_a_no_op():
    holder, other = _Owner(), _Owner()
    claim_process_phase_state(holder)
    release_process_phase_state(other)
    with pytest.raises(RuntimeError):
        claim_process_phase_state(other)
    # Releasing with nothing claimed is also harmless.
    release_process_phase_state(holder)
    release_process_phase_state(holder)


def test_a_collected_owner_no_longer_blocks_a_new_one():
    first = _Owner()
    claim_process_phase_state(first)
    del first
    gc.collect()
    claim_process_phase_state(_Owner())


def _worker_init() -> ast.FunctionDef:
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    return next(
        node for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )


def _is_call(stmt: ast.stmt, name: str) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == name
    )


def test_worker_init_claims_before_any_other_work():
    body = _worker_init().body
    claims = [i for i, stmt in enumerate(body) if _is_call(stmt, "claim_process_phase_state")]
    assert len(claims) == 1
    claim = body[claims[0]].value
    assert [ast.unparse(arg) for arg in claim.args] == ["self"]
    assert not claim.keywords
    # Only the start-up log line precedes it: no configuration, allocation or
    # attribute assignment happens before the claim can refuse the worker.
    before = body[: claims[0]]
    assert len(before) == 1
    assert isinstance(before[0], ast.Expr)
    assert ast.unparse(before[0].value.func) == "logging.info"


def test_worker_process_releases_the_claim_after_the_ordered_teardown():
    tree = ast.parse(MAIN_LOOP.read_text())
    impl = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_server_worker_main_impl"
    )
    order = [
        stmt.value.func.id
        for stmt in impl.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id in {"_shutdown_worker_runtime", "release_process_phase_state"}
    ]
    assert order == ["_shutdown_worker_runtime", "release_process_phase_state"]
