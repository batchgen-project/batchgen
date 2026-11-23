"""Admission refuses debug switches carried by a request.

Debug switches belong to the server (``batchgen/server_debug_state.py``) and are set
through the admin endpoint. A request that carries ``batchgen_debug`` is
answered 400 naming that endpoint, before a worker sees it; the turn and
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


def test_any_debug_switch_on_a_request_is_refused_naming_the_admin_endpoint():
    message = io_struct.batchgen_debug_refusal(
        {"k3_prefill_profile": True, "glm5_graph_path_log": True}
    )
    assert message is not None
    assert "POST /v1/admin/debug-state" in message
    assert "Keys sent: glm5_graph_path_log, k3_prefill_profile." in message


@pytest.mark.parametrize("debug", [None, {}])
def test_a_request_without_debug_switches_passes(debug):
    assert io_struct.batchgen_debug_refusal(debug) is None


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
        "batchgen_debug_refusal": io_struct.batchgen_debug_refusal,
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
        batchgen_debug={"k3_prefill_profile": True},
    )
    with pytest.raises(HTTPException) as refused:
        asyncio.run(handler(request, body))
    assert refused.value.status_code == 400
    assert "/v1/admin/debug-state" in refused.value.detail
    assert "k3_prefill_profile" in refused.value.detail
