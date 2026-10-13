"""The legacy non-pool inference path is gone; only the batch API remains.

Everything here is source-level or route-level: none of it needs a GPU, a
worker process, or the full server import stack.
"""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
HTTP_SERVER = ROOT / "batchgen" / "server" / "http_server.py"
MAIN_LOOP = ROOT / "batchgen" / "server_worker_main_loop.py"
SCHEDULER = ROOT / "batchgen" / "server" / "batch_scheduler.py"
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"


def _load_deprecation():
    """Import batchgen/deprecation.py alone (batchgen/__init__ pulls kernels)."""
    path = ROOT / "batchgen" / "deprecation.py"
    spec = importlib.util.spec_from_file_location("_batchgen_deprecation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deprecation = _load_deprecation()


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _class_method_names(path, class_name):
    cls = next(
        node for node in ast.walk(_tree(path))
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _function(path, name):
    return next(
        node for node in ast.walk(_tree(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


# ---------------------------------------------------------------- HTTP route


def test_v1_inference_returns_410_with_code_and_use_instead():
    """The route body itself, mounted on a bare app — no server stack import."""
    route = copy.deepcopy(_function(HTTP_SERVER, "run_inference"))
    decorators = [ast.unparse(dec) for dec in route.decorator_list]
    assert decorators == ["app.post('/v1/inference')"]
    route.decorator_list = []

    module = ast.Module(body=[route], type_ignores=[])
    namespace = {
        "HTTPException": HTTPException,
        "LEGACY_INFERENCE_ERROR_CODE": deprecation.LEGACY_INFERENCE_ERROR_CODE,
        "LEGACY_INFERENCE_MESSAGE": deprecation.LEGACY_INFERENCE_MESSAGE,
    }
    exec(compile(ast.fix_missing_locations(module), str(HTTP_SERVER), "exec"), namespace)

    app = FastAPI()
    app.post("/v1/inference")(namespace["run_inference"])
    # No lifespan: the route must refuse before touching any app state.
    response = TestClient(app).post(
        "/v1/inference", json={"prompts": ["hi"], "max_output_len": 8}
    )

    assert response.status_code == 410
    detail = response.json()["detail"]
    assert detail["code"] == deprecation.LEGACY_INFERENCE_ERROR_CODE
    assert detail["use_instead"] == "/v1/batches"
    assert "batch API" in detail["message"]


# ------------------------------------------------------------- deleted code


@pytest.mark.parametrize("name", ["process_new_batch", "_tokenize_global_batch"])
def test_worker_no_longer_defines_legacy_batch_entry_points(name):
    assert name not in _class_method_names(WORKER, "BatchGenWorker")


def test_worker_manager_no_longer_defines_infer():
    assert "infer" not in _class_method_names(WORKER_MANAGER, "WorkerManager")


def test_main_loop_never_calls_process_new_batch():
    impl = ast.unparse(_function(MAIN_LOOP, "_server_worker_main_impl"))
    assert "process_new_batch" not in impl
    assert "set_ignore_eos" not in impl
    assert "set_per_sequence_sampling_params" not in impl


def test_main_loop_fails_loudly_on_an_unhandled_message():
    """A message that is neither shutdown, reload nor init must raise."""
    impl = _function(MAIN_LOOP, "_server_worker_main_impl")
    loop = next(node for node in ast.walk(impl) if isinstance(node, ast.While))
    # Last statements of the loop body: the legacy guard, then a hard failure.
    legacy_guard, fallthrough = loop.body[-2], loop.body[-1]

    assert isinstance(legacy_guard, ast.If)
    assert "'prompts' in task_data" in ast.unparse(legacy_guard.test)
    assert isinstance(legacy_guard.body[0], ast.Raise)
    assert "LegacyInferenceDeprecated" in ast.unparse(legacy_guard.body[0])

    assert isinstance(fallthrough, ast.Raise)
    assert "RuntimeError" in ast.unparse(fallthrough)
    assert "unexpected worker message" in ast.unparse(fallthrough)


def test_scheduler_always_routes_to_the_pool_path():
    process_batch = ast.unparse(_function(SCHEDULER, "_process_batch"))
    assert "_process_batch_pool_mode" in process_batch
    assert "self._pool_mode" not in process_batch
    assert "self.worker.infer" not in process_batch

    scheduler_methods = _class_method_names(SCHEDULER, "BatchScheduler")
    for gone in (
        "_build_output_items",
        "_build_response_body",
        "_build_response_body_from_text",
        "_decode_tokens",
        "_trim_tokens",
        "_normalize_worker_results",
    ):
        assert gone not in scheduler_methods
