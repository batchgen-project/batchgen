"""Admission refuses retired batchgen_debug keys before a worker sees them.

The prefill MoE mode is chosen once per server (--prefill-moe-mode); the
per-batch ``batchgen_debug.k3_prefill_moe_mode`` that used to switch one
wave's mode is retired. POST /v1/batches answers 400 for it; the turn and
stored-batch paths are covered in tests/test_turn_api.py.
"""

import ast
import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException, Request  # noqa: E402

SERVER = Path(__file__).resolve().parents[2] / "batchgen" / "server"


def _load_io_struct():
    path = SERVER / "io_struct.py"
    spec = importlib.util.spec_from_file_location("batchgen.server.io_struct", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["batchgen.server.io_struct"] = module
    spec.loader.exec_module(module)
    return module


io_struct = _load_io_struct()


def test_the_retired_mode_key_is_refused_naming_the_server_flag():
    message = io_struct.retired_batchgen_debug_error(
        {"k3_prefill_moe_mode": "resident_ep", "glm5_moe_mode": "graph"}
    )
    assert message is not None
    assert "batchgen_debug.k3_prefill_moe_mode is no longer accepted" in message
    assert "--prefill-moe-mode" in message
    assert "glm5_moe_mode" not in message


@pytest.mark.parametrize(
    "debug", [None, {}, {"k3_prefill_profile": True, "glm5_moe_mode": "eager"}]
)
def test_current_debug_keys_pass(debug):
    assert io_struct.retired_batchgen_debug_error(debug) is None


def _create_batch_handler():
    """The real POST /v1/batches handler, lifted out of create_app."""
    tree = ast.parse((SERVER / "http_server.py").read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "create_batch"
    )
    node = copy.deepcopy(node)
    node.decorator_list = []
    namespace = {
        "HTTPException": HTTPException,
        "Request": Request,
        "CreateBatchRequest": io_struct.CreateBatchRequest,
        "retired_batchgen_debug_error": io_struct.retired_batchgen_debug_error,
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "http_server.py", "exec"), namespace)
    return namespace["create_batch"]


class _Untouchable:
    def __getattr__(self, name):
        raise AssertionError(f"admission reached {name} for a refused batch")


def test_create_batch_answers_400_before_touching_pool_or_storage():
    handler = _create_batch_handler()
    state = SimpleNamespace(
        storage=_Untouchable(), scheduler=_Untouchable(), worker=_Untouchable()
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    body = io_struct.CreateBatchRequest(
        input_file_id="file-1",
        batchgen_debug={"k3_prefill_moe_mode": "resident_ep"},
    )
    with pytest.raises(HTTPException) as refused:
        asyncio.run(handler(request, body))
    assert refused.value.status_code == 400
    assert "--prefill-moe-mode" in refused.value.detail
