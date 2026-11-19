"""The K3 block-residual buffer is released at the end of every serving pass.

``BlockResidualBuffer`` keeps one pass's ``(num_tokens, num_boundaries,
hidden)`` scratch as class state. A serving loop that seeds it through
``_new_block_residual`` and only drops its local view leaves the whole
allocation pinned until the next seed: it is then charged to everything the
next wave does first, and to that wave's memory headroom. These tests hold
the release on both serving loops and show why dropping the local is not
enough.
"""

import ast
import gc
import importlib.util
import weakref
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "batchgen" / "worker" / "prefill_engine.py"
MODEL = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "model.py"
BLOCK_RESIDUAL = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "block_residual.py"
)


def _load_block_residual():
    # Import-light by design (torch only): loaded by path, without batchgen.
    spec = importlib.util.spec_from_file_location("_k3_block_residual", BLOCK_RESIDUAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attr_calls(function, name):
    return [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_dropping_the_local_view_does_not_free_the_buffer():
    buffer_cls = _load_block_residual().BlockResidualBuffer
    buffer_cls.reset()
    view = buffer_cls.seed(64, 16, 8, dtype=torch.float32, device="cpu")
    storage = weakref.ref(buffer_cls._buf)
    view = None  # noqa: F841 - what a serving loop's `block_residual = None` does
    gc.collect()
    assert storage() is not None, "class state alone keeps the pass's buffer alive"

    buffer_cls.reset()
    gc.collect()
    assert storage() is None
    assert buffer_cls._buf is None and buffer_cls._view is None


def test_model_seam_releases_through_the_buffer_reset():
    tree = ast.parse(MODEL.read_text())
    release = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_release_block_residual"
    )
    resets = [
        call for call in _attr_calls(release, "reset")
        if isinstance(call.func.value, ast.Name)
        and call.func.value.id == "BlockResidualBuffer"
    ]
    assert len(resets) == 1


def test_every_serving_loop_that_seeds_also_releases():
    tree = ast.parse(ENGINE.read_text())
    seeding = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and _attr_calls(node, "_new_block_residual")
        and not any(
            isinstance(inner, ast.FunctionDef)
            and inner is not node
            and _attr_calls(inner, "_new_block_residual")
            for inner in ast.walk(node)
        )
    ]
    assert len(seeding) == 2, [node.name for node in seeding]
    for function in seeding:
        seeds = _attr_calls(function, "_new_block_residual")
        releases = _attr_calls(function, "_release_block_residual")
        assert len(releases) == len(seeds) == 1, function.name
        # the pass is seeded before its layers and released after them
        assert releases[0].lineno > seeds[0].lineno, function.name
