"""``debug_state.glm5_lmhead_fp32``: a batch flag, never an env guard.

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
# S5b: prefill_prepacked's body moved verbatim to PrefillEngine; the flag key
# literal stays in the worker behind _glm5_lmhead_fp32_flag_name (spec F3), so
# no model-name literal enters batchgen/worker/.
ENGINE = (
    Path(__file__).resolve().parents[2] / "batchgen" / "worker" / "prefill_engine.py"
)
ENGINE_SOURCE = ENGINE.read_text()
ENGINE_TREE = ast.parse(ENGINE_SOURCE)

FLAG = "glm5_lmhead_fp32"
LOCAL = "_lmhead_fp32"


_ENGINE_METHODS = {
    "prefill_prepacked", "_prefill_prepacked_pool", "_prefill_model_pass",
}


def _method(name: str) -> ast.FunctionDef:
    tree = ENGINE_TREE if name in _ENGINE_METHODS else TREE
    return next(
        node for node in ast.walk(tree)
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
        "self._debug_flag_enabled(prefill_debug.get("
        "self._glm5_lmhead_fp32_flag_name()))"
    )
    helper = _method("_glm5_lmhead_fp32_flag_name")
    assert ast.unparse(helper.body[-1]) == f"return '{FLAG}'"

    all_assigns = [
        node for node in ast.walk(_method("prefill_prepacked"))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == LOCAL
            for target in node.targets
        )
    ]
    assert len(all_assigns) == 1, all_assigns


def test_the_fp32_cast_branches_on_the_passed_flag_in_the_one_model_pass():
    """Step 4: the lm_head epilogue lives once, in ``_prefill_model_pass``;
    the cast branches on its ``lmhead_fp32`` parameter."""

    model_pass = _method("_prefill_model_pass")
    params = [arg.arg for arg in (
        model_pass.args.args + model_pass.args.kwonlyargs
    )]
    assert "lmhead_fp32" in params
    branches = [
        node for node in ast.walk(model_pass)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "lmhead_fp32"
    ]
    assert len(branches) == 1, branches
    # The branch it guards is the FP32-cast matmul, not something else.
    assert ".float()" in ast.unparse(branches[0].body[0])
    # No second lm_head epilogue survives in either driver.
    for driver in ("prefill_prepacked", "_prefill_prepacked_pool"):
        assert not [
            node for node in ast.walk(_method(driver))
            if isinstance(node, ast.Attribute) and node.attr == "lm_head"
        ], driver


def test_the_unpooled_driver_hands_its_read_to_the_pass_and_the_pool_keeps_false():
    """Behaviour-preserving step 4: the unpooled driver passes its one read
    of the flag into the pass; the pooled wave has never honored the key and
    keeps passing False (kept, reported -- not resolved here)."""

    expected = {"prefill_prepacked": LOCAL, "_prefill_prepacked_pool": "False"}
    for driver, value in expected.items():
        # D7.5 converted the pass, so the pooled driver (converted too) reaches
        # it as a keyword-only collaborator -- a bare ``Name`` callee -- while
        # the still-worker-shaped unpooled driver keeps the direct engine call.
        # The flag it hands over is the claim here, not the route.
        calls = [
            node for node in ast.walk(_method(driver))
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute)
                 and node.func.attr == "_prefill_model_pass")
                or (isinstance(node.func, ast.Name)
                    and node.func.id == "_prefill_model_pass")
            )
        ]
        assert len(calls) == 1, driver
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
        assert kwargs.get("lmhead_fp32") == value, (driver, kwargs)


def test_no_lmhead_env_guard_survives_in_the_worker():
    """§1.3: debug behavior expressible as a batch flag carries no env var."""

    assert "BATCHGEN_GLM5_LMHEAD_FP32" not in SOURCE
    assert "BATCHGEN_GLM5_LMHEAD_FP32" not in ENGINE_SOURCE


def test_the_flag_key_is_spelled_once_and_only_in_that_read():
    reads = [
        node for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and node.value == FLAG
    ]
    assert len(reads) == 1, [node.lineno for node in reads]
    assert not [
        node for node in ast.walk(ENGINE_TREE)
        if isinstance(node, ast.Constant) and node.value == FLAG
    ]
