"""Tests for preserving batch-level create parameters.

``max_context_length`` on ``POST /v1/batches`` is deprecated: the deployment
``max_model_len`` bounds every request, so the field is accepted (existing
callers keep working), logged once per batch, and discarded before the batch
object exists.
"""

import ast
import asyncio
import importlib.util
import inspect
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
IO_STRUCT_LOGGER = "batchgen.server.io_struct"


def _load_io_struct():
    path = ROOT / "batchgen" / "server" / "io_struct.py"
    spec = importlib.util.spec_from_file_location("batchgen.server.io_struct", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["batchgen.server.io_struct"] = module
    spec.loader.exec_module(module)
    return module


def _deprecation_records(caplog):
    return [
        record for record in caplog.records
        if record.name == IO_STRUCT_LOGGER and "max_context_length" in record.getMessage()
    ]


def test_create_batch_request_preserves_debug_and_sampling_fields():
    io_struct = _load_io_struct()
    request = io_struct.CreateBatchRequest(
        input_file_id="file-test",
        max_decoding_length=128,
        temperature=0.0,
        top_p=1.0,
        top_k=5,
        batchgen_debug={
            "glm5_dsa_mode": "eager",
            "glm5_moe_mode": "graph",
        },
    )

    batch = io_struct.build_batch_object_from_create_request(
        batch_id="batch-test",
        body=request,
        created_at=100,
        expires_at=200,
    )

    assert batch.max_decoding_length == 128
    assert batch.temperature == 0.0
    assert batch.top_p == 1.0
    assert batch.top_k == 5
    assert batch.batchgen_debug == {
        "glm5_dsa_mode": "eager",
        "glm5_moe_mode": "graph",
    }


def test_deprecated_max_context_length_warns_once_and_is_discarded(caplog):
    io_struct = _load_io_struct()
    caplog.set_level(logging.WARNING, logger=IO_STRUCT_LOGGER)
    request = io_struct.CreateBatchRequest(
        input_file_id="file-test",
        max_decoding_length=64,
        max_context_length=4096,
    )

    records = _deprecation_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    message = records[0].getMessage()
    assert "max_context_length=4096 is deprecated and ignored" in message
    assert "max_model_len" in message
    assert "GET /v1/models" in message
    assert request.max_context_length is None

    batch = io_struct.build_batch_object_from_create_request(
        batch_id="batch-test", body=request, created_at=100, expires_at=200,
    )
    assert "max_context_length" not in io_struct.BatchObject.model_fields
    assert "max_context_length" not in batch.model_dump()
    assert batch.max_decoding_length == 64
    # Building the batch object does not re-parse the request: still one warning.
    assert len(_deprecation_records(caplog)) == 1


@pytest.mark.parametrize("payload", [{}, {"max_context_length": None}])
def test_absent_or_null_max_context_length_logs_nothing(caplog, payload):
    io_struct = _load_io_struct()
    caplog.set_level(logging.WARNING, logger=IO_STRUCT_LOGGER)
    request = io_struct.CreateBatchRequest(input_file_id="file-test", **payload)
    assert request.max_context_length is None
    assert _deprecation_records(caplog) == []


def test_invalid_max_context_length_is_still_rejected():
    io_struct = _load_io_struct()
    with pytest.raises(Exception, match="max_context_length"):
        io_struct.CreateBatchRequest(input_file_id="file-test", max_context_length=0)


def test_http_json_body_with_max_context_length_parses_and_discards(caplog):
    """The wire form FastAPI validates: raw JSON into the request schema."""
    io_struct = _load_io_struct()
    caplog.set_level(logging.WARNING, logger=IO_STRUCT_LOGGER)
    raw = json.dumps({
        "input_file_id": "file-test",
        "endpoint": "/v1/completions",
        "max_context_length": 262144,
        "max_decoding_length": 32,
    })
    request = io_struct.CreateBatchRequest.model_validate_json(raw)
    assert request.max_context_length is None
    assert request.max_decoding_length == 32
    assert len(_deprecation_records(caplog)) == 1
    schema = io_struct.CreateBatchRequest.model_json_schema()
    assert schema["properties"]["max_context_length"]["deprecated"] is True


def test_batch_persisted_with_the_old_field_still_loads(tmp_path):
    """The batch store survives restarts; records written before the field was
    removed carry ``max_context_length`` and must still load."""
    io_struct = _load_io_struct()
    spec = importlib.util.spec_from_file_location(
        "batchgen.server.storage", ROOT / "batchgen" / "server" / "storage.py")
    storage_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage_module)
    storage = storage_module.StorageManager(tmp_path)
    record = {
        "id": "batch-old", "object": "batch", "endpoint": "/v1/chat/completions",
        "input_file_id": "file-old", "completion_window": "24h",
        "status": "completed", "created_at": 1, "expires_at": 2,
        "max_decoding_length": 16, "max_context_length": 131072,
    }
    (storage.batches_dir / "batch-old.json").write_text(json.dumps(record))

    loaded = storage.load_batch("batch-old")
    assert isinstance(loaded, io_struct.BatchObject)
    assert loaded.max_decoding_length == 16
    assert "max_context_length" not in loaded.model_dump()


def test_batch_carrying_the_deprecated_field_still_runs_in_pool_mode():
    """A batch created with the deprecated field reaches the intake pool, and
    the pool init message no longer carries a context bound."""
    from tests.test_pool_request_identity import _load_scheduler_module

    bs = _load_scheduler_module()
    io_struct = _load_io_struct()
    batch = io_struct.build_batch_object_from_create_request(
        batch_id="batch-1",
        body=io_struct.CreateBatchRequest(input_file_id="file-1", max_context_length=4096),
        created_at=100,
        expires_at=200,
    )
    submitted = []
    scheduler = object.__new__(bs.BatchScheduler)
    scheduler._scheduling_pool = SimpleNamespace(register_batch=lambda **kw: None)
    scheduler._intake_pool = SimpleNamespace(
        submit_batch=lambda batch_id, entries, priority: submitted.extend(entries) or True,
        size=lambda: len(submitted),
        max_capacity=10,
    )
    scheduler.storage = SimpleNamespace()
    scheduler.server_args = SimpleNamespace(incremental_output_dir=None)
    scheduler._pool_request_meta = {}
    finalized = []

    async def record_finalize(batch_id, requests):
        finalized.append(batch_id)

    scheduler._wait_and_finalize_batch = record_finalize
    url = SimpleNamespace(value="/v1/completions")
    requests = [
        SimpleNamespace(
            custom_id=f"c{i}", url=url,
            body=bs.CompletionRequest(model="m", prompt=f"p{i}"),
        )
        for i in range(2)
    ]

    async def run():
        await scheduler._process_batch_pool_mode(
            "batch-1", batch, requests, ["p0", "p1"], [8, 8], [{}, {}],
        )
        await asyncio.sleep(0)

    asyncio.run(run())

    assert [entry.request_id for entry in submitted] == ["c0@batch-1", "c1@batch-1"]
    assert finalized == ["batch-1"]
    assert scheduler.pool_init_params() == {"max_output_len": 8}


def _create_batch_app(storage, scheduler, worker):
    """Compile the real POST /v1/batches handler, avoiding the CUDA server imports."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI, HTTPException, Request

    io_struct = _load_io_struct()
    app = FastAPI()
    app.state.storage = storage
    app.state.scheduler = scheduler
    app.state.worker = worker
    tree = ast.parse((ROOT / "batchgen/server/http_server.py").read_text())
    (node,) = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "create_batch"
    ]
    scope = dict(
        app=app, Request=Request, HTTPException=HTTPException,
        CreateBatchRequest=io_struct.CreateBatchRequest,
        BatchObject=io_struct.BatchObject,
        FilePurpose=io_struct.FilePurpose,
        build_batch_object_from_create_request=io_struct.build_batch_object_from_create_request,
        uuid=uuid, time=time,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "http_server.py", "exec"), scope)
    return app


def test_post_batches_accepts_and_discards_max_context_length(caplog):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    saved, enqueued = [], []

    async def enqueue(batch_id):
        enqueued.append(batch_id)

    storage = SimpleNamespace(
        load_metadata=lambda file_id: {"id": file_id, "purpose": "batch"},
        get_active_batch_for_file=lambda file_id: None,
        save_batch=saved.append,
    )
    scheduler = SimpleNamespace(intake_pool_usage_pct=lambda: 0.0, enqueue=enqueue)
    worker = SimpleNamespace(state="serving")
    caplog.set_level(logging.WARNING, logger=IO_STRUCT_LOGGER)

    with TestClient(_create_batch_app(storage, scheduler, worker)) as client:
        response = client.post("/v1/batches", json={
            "input_file_id": "file-1", "max_context_length": 131072,
            "max_decoding_length": 16,
        })

    assert response.status_code == 200, response.text
    body = response.json()
    assert "max_context_length" not in body
    assert body["max_decoding_length"] == 16
    assert [batch.id for batch in saved] == enqueued == [body["id"]]
    assert len(_deprecation_records(caplog)) == 1


def test_client_create_batch_has_no_max_context_length_parameter():
    spec = importlib.util.spec_from_file_location(
        "batchgen_client_under_test", ROOT / "batchgen" / "batchgen_client.py")
    client_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client_module)
    client_cls = client_module.BatchGenHttpClient
    for method in (client_cls.create_batch, client_cls.submit_batch):
        assert "max_context_length" not in inspect.signature(method).parameters

    sent = []
    client = object.__new__(client_cls)
    client.post_json = lambda path, payload: sent.append((path, payload)) or {"id": "b"}
    assert client.create_batch("file-1", max_decoding_length=8, temperature=0.5) == {"id": "b"}
    assert sent == [("/v1/batches", {
        "input_file_id": "file-1",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "max_decoding_length": 8,
        "temperature": 0.5,
    })]
    with pytest.raises(TypeError):
        client.create_batch("file-1", max_context_length=4096)
