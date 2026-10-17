"""Batch admission rejects a request model from another tokenizer family.

The scheduler renders the chat template with the tokenizer resolved from the
request's ``model`` while the worker encodes with the served checkpoint's
tokenizer. A Kimi-K3 request on a Kimi-Linear-48B server was rendered with the
K3 template and its control markers encoded as plain text by the 48B tokenizer.

The server package imports the JIT engine, so batch_scheduler is loaded by
path with the real io_struct / IntakePool / SchedulingPool and stubs for the
engine-backed modules.
"""
import ast
import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.config.tokenizer_registry import resolve_tokenizer_type

SERVER = Path(__file__).resolve().parents[1] / "batchgen" / "server"

K3 = "moonshotai/Kimi-K3"
KIMI_LINEAR = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
KIMI_LINEAR_CACHE_DIR = "/models/kimi-linear-48b-a3b-instruct"


def _load_scheduler_module():
    names = ["batchgen.server"] + [
        f"batchgen.server.{m}" for m in (
            "io_struct", "intake_pool", "scheduling_pool", "server_args",
            "storage", "worker_manager", "batch_scheduler")]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        package = types.ModuleType("batchgen.server")
        package.__path__ = [str(SERVER)]
        sys.modules["batchgen.server"] = package
        for module, attr in (("server_args", "ServerArgs"), ("storage", "StorageManager"),
                             ("worker_manager", "WorkerManager")):
            stub = types.ModuleType(f"batchgen.server.{module}")
            setattr(stub, attr, object)
            sys.modules[stub.__name__] = stub
        loaded = None
        for module in ("io_struct", "intake_pool", "scheduling_pool", "batch_scheduler"):
            spec = importlib.util.spec_from_file_location(
                f"batchgen.server.{module}", SERVER / f"{module}.py")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
        return loaded
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


bs = _load_scheduler_module()


def _chat_line(model, cid="r0"):
    return json.dumps({
        "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
        "body": {"model": model, "max_tokens": 8,
                 "messages": [{"role": "user", "content": "hi"}]},
    })


def _completion_line(model, cid="r0"):
    return json.dumps({
        "custom_id": cid, "method": "POST", "url": "/v1/completions",
        "body": {"model": model, "prompt": "hi", "max_tokens": 8},
    })


def _file(*lines):
    return "\n".join(lines).encode()


def test_registry_resolves_family_from_its_own_pattern_table():
    assert resolve_tokenizer_type(K3) == "kimi_k3"
    assert resolve_tokenizer_type(KIMI_LINEAR) == "kimi_linear"
    assert resolve_tokenizer_type(KIMI_LINEAR_CACHE_DIR) == "kimi_linear"
    assert resolve_tokenizer_type("zai-org/GLM-5.1-FP8") == "glm_moe_dsa_5_1"
    assert resolve_tokenizer_type("not-a-registered-model") is None


def test_k3_request_on_kimi_linear_server_is_rejected_with_both_families():
    ok, error, requests = bs.parse_batch_file(
        _file(_chat_line(K3)), served_model=KIMI_LINEAR)
    assert not ok
    assert requests == []
    assert error.startswith("Line 1: ")
    for needle in (repr(K3), repr(KIMI_LINEAR), "'kimi_k3'", "'kimi_linear'"):
        assert needle in error


def test_completion_request_from_another_family_is_rejected():
    ok, error, _ = bs.parse_batch_file(
        _file(_completion_line(K3)), served_model=KIMI_LINEAR)
    assert not ok
    assert "'kimi_k3'" in error and "'kimi_linear'" in error


def test_served_model_name_is_accepted():
    ok, error, requests = bs.parse_batch_file(
        _file(_chat_line(KIMI_LINEAR), _chat_line(KIMI_LINEAR, "r1")),
        served_model=KIMI_LINEAR)
    assert ok, error
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("served", "requested"),
    [
        (KIMI_LINEAR_CACHE_DIR, KIMI_LINEAR),
        (KIMI_LINEAR, "Kimi-Linear-48B-A3B-Instruct"),
        ("/models/Kimi-K3", K3),
        ("zai-org/GLM-5.1", "zai-org/GLM-5.1-FP8"),
    ],
)
def test_same_family_alias_is_accepted(served, requested):
    ok, error, _ = bs.parse_batch_file(
        _file(_chat_line(requested)), served_model=served)
    assert ok, error


def test_glm_versions_with_different_templates_are_different_families():
    ok, error, _ = bs.parse_batch_file(
        _file(_chat_line("zai-org/GLM-5")), served_model="zai-org/GLM-5.1-FP8")
    assert not ok
    assert "'glm_moe_dsa'" in error and "'glm_moe_dsa_5_1'" in error


def test_unresolved_served_model_fails_loudly():
    ok, error, _ = bs.parse_batch_file(
        _file(_chat_line(K3)), served_model="not-a-registered-model")
    assert not ok
    assert "resolves to no registered tokenizer family" in error


def test_without_served_model_parse_is_unchanged():
    ok, error, _ = bs.parse_batch_file(_file(_chat_line(K3)))
    assert ok, error


def test_scheduler_fails_the_batch_before_rendering(tmp_path):
    input_id = "file-k3"
    (tmp_path / input_id).write_bytes(_file(_chat_line(K3)))
    updates = []
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)
    scheduler.server_args = SimpleNamespace(model=KIMI_LINEAR)
    scheduler.storage = SimpleNamespace(
        files_dir=tmp_path,
        load_batch=lambda batch_id: SimpleNamespace(
            status=bs.BatchStatus.VALIDATING, input_file_id=input_id),
        load_metadata=lambda file_id: {"id": file_id},
        update_batch_status=lambda batch_id, status, **kw: updates.append(
            (status, kw.get("error"))),
    )

    def must_not_render(*args, **kwargs):
        raise AssertionError("a mismatched request reached prompt rendering")

    scheduler._convert_requests_to_worker_inputs = must_not_render
    asyncio.run(scheduler._process_batch("batch_k3"))

    status, error = updates[-1]
    assert status == bs.BatchStatus.FAILED
    assert "'kimi_k3'" in error and "'kimi_linear'" in error


def test_upload_route_validates_against_the_served_model():
    tree = ast.parse((SERVER / "http_server.py").read_text())
    upload = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "upload_file"
    )
    calls = [
        node for node in ast.walk(upload)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "parse_batch_file"
    ]
    assert len(calls) == 1
    served = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert served.get("served_model") == "server_args.model"
