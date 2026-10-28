"""``batchgen_debug.glm5_lmhead_fp32``: a batch flag, never an env guard.

The FP32 cast of the unpooled prefill lm_head matmul used to be opted into
with ``BATCHGEN_GLM5_LMHEAD_FP32=1``. An env var is fixed for the server's
lifetime, so toggling the cast cost a cold restart; the behavior now rides the
batch-level ``batchgen_debug`` seam, like ``k3_prefill_profile`` beside it.

The branch sits inside ``prefill_prepacked``'s micro-batch loop and cannot be
compiled in isolation, so it is pinned structurally against the module's own
AST -- the technique ``test_pool_probe_debug_seam.py`` uses for the call sites
inside the two prefill loops.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"
SOURCE = WORKER.read_text()
TREE = ast.parse(SOURCE)

FLAG = "glm5_lmhead_fp32"
LOCAL = "_lmhead_fp32"


def _method(name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_the_flag_is_read_once_per_prefill_from_the_batch_debug_dict():
    """One read, in the function body -- not once per micro-batch."""

    body_assigns = [
        node for node in _method("prefill_prepacked").body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == LOCAL
    ]
    assert len(body_assigns) == 1, body_assigns
    assert ast.unparse(body_assigns[0].value) == (
        f"self._debug_flag_enabled(prefill_debug.get('{FLAG}'))"
    )

    all_assigns = [
        node for node in ast.walk(_method("prefill_prepacked"))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == LOCAL
            for target in node.targets
        )
    ]
    assert len(all_assigns) == 1, all_assigns


def test_the_fp32_cast_branches_on_that_local():
    branches = [
        node for node in ast.walk(_method("prefill_prepacked"))
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == LOCAL
    ]
    assert len(branches) == 1, branches
    # The branch it guards is the FP32-cast matmul, not something else.
    assert ".float()" in ast.unparse(branches[0].body[0])


def test_no_lmhead_env_guard_survives_in_the_worker():
    """§1.3: debug behavior expressible as a batch flag carries no env var."""

    assert "BATCHGEN_GLM5_LMHEAD_FP32" not in SOURCE


def test_the_flag_key_is_spelled_once_and_only_in_that_read():
    reads = [
        node for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and node.value == FLAG
    ]
    assert len(reads) == 1, [node.lineno for node in reads]
