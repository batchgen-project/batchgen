"""The retired prompt-arena hook cannot reintroduce a second token store."""

import ast
import types
from pathlib import Path
from typing import Dict, Sequence, Set

import pytest


WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _load_publish_method():
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    method = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "_publish_prompt_arena")
    namespace = {
        "Dict": Dict,
        "Sequence": Sequence,
        "Set": Set,
        "dist": types.SimpleNamespace(),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(WORKER), "exec"), namespace)
    return namespace["_publish_prompt_arena"], namespace["dist"]


PUBLISH, DIST = _load_publish_method()


def test_prompt_arena_hook_is_retired():
    worker = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="PromptTokenArena is retired"):
        PUBLISH(worker, [], {}, set())
