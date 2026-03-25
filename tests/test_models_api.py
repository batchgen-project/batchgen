"""Tests for GET /v1/models and GET /v1/models/{model_id} endpoints.

Since batchgen.server.http_server has deep import dependencies (CUDA, torch, etc.),
we test by:
1. Importing io_struct models directly (lightweight, no GPU deps)
2. Building a minimal FastAPI app that replicates the endpoint logic
3. Verifying response schemas and behavior
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Direct import approach: load io_struct.py without going through
# batchgen.server.__init__ (which pulls in torch, CUDA, etc.)

# Import io_struct directly via importlib to bypass batchgen.server.__init__
# which pulls in the full server stack (torch, CUDA, etc.)
import importlib.util

_io_struct_path = str(Path(__file__).parent.parent / "batchgen" / "server" / "io_struct.py")
_spec = importlib.util.spec_from_file_location("batchgen.server.io_struct", _io_struct_path)
_io_struct = importlib.util.module_from_spec(_spec)
sys.modules["batchgen.server.io_struct"] = _io_struct
_spec.loader.exec_module(_io_struct)

ModelObject = _io_struct.ModelObject
ListModelsResponse = _io_struct.ListModelsResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient


def _build_test_app(model_id: str = "Kimi-K2.5", max_context_length: int = 262144) -> FastAPI:
    """Build a minimal FastAPI app with /v1/models endpoints matching http_server.py logic."""
    app = FastAPI()

    model_obj = ModelObject(
        id=model_id,
        created=int(time.time()),
        owned_by="batchgen",
        max_context_length=max_context_length,
    )
    app.state.model_object = model_obj

    @app.get("/v1/models", response_model=ListModelsResponse)
    async def list_models(request: Request):
        return ListModelsResponse(data=[request.app.state.model_object])

    @app.get("/v1/models/{model_id:path}", response_model=ModelObject)
    async def retrieve_model(request: Request, model_id: str):
        obj: ModelObject = request.app.state.model_object
        if model_id != obj.id:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return obj

    return app


@pytest.fixture()
def client():
    app = _build_test_app(model_id="Kimi-K2.5", max_context_length=262144)
    with TestClient(app) as tc:
        yield tc


class TestListModels:
    def test_returns_list_with_single_model(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1

    def test_model_object_fields(self, client):
        resp = client.get("/v1/models")
        model = resp.json()["data"][0]
        assert model["id"] == "Kimi-K2.5"
        assert model["object"] == "model"
        assert model["owned_by"] == "batchgen"
        assert model["max_context_length"] == 262144
        assert isinstance(model["created"], int)

    def test_created_timestamp_is_recent(self, client):
        resp = client.get("/v1/models")
        created = resp.json()["data"][0]["created"]
        now = int(time.time())
        assert now - created < 60


class TestRetrieveModel:
    def test_valid_model_id(self, client):
        resp = client.get("/v1/models/Kimi-K2.5")
        assert resp.status_code == 200
        model = resp.json()
        assert model["id"] == "Kimi-K2.5"
        assert model["max_context_length"] == 262144

    def test_invalid_model_id_returns_404(self, client):
        resp = client.get("/v1/models/nonexistent-model")
        assert resp.status_code == 404

    def test_404_detail_message(self, client):
        resp = client.get("/v1/models/wrong")
        assert "not found" in resp.json()["detail"].lower()


class TestMaxContextLength:
    def test_matches_configured_value(self, client):
        resp = client.get("/v1/models")
        model = resp.json()["data"][0]
        assert model["max_context_length"] == 262144

    def test_different_context_length(self):
        """Verify different max_context_length values propagate correctly."""
        app = _build_test_app(model_id="gpt-oss-120b", max_context_length=131072)
        with TestClient(app) as tc:
            resp = tc.get("/v1/models")
            model = resp.json()["data"][0]
            assert model["id"] == "gpt-oss-120b"
            assert model["max_context_length"] == 131072


class TestModelIdExtraction:
    def test_path_extraction_logic(self):
        """Verify Path.name extracts model ID correctly from various paths."""
        cases = [
            ("/data/models/moonshotai/Kimi-K2.5", "Kimi-K2.5"),
            ("/data2/tairan/workspace/models/openai/gpt-oss-120b", "gpt-oss-120b"),
            ("DeepSeek-R1", "DeepSeek-R1"),
            ("/mnt/data/GLM-5-FP8", "GLM-5-FP8"),
        ]
        for model_path, expected_id in cases:
            assert Path(model_path).name == expected_id


class TestPydanticModels:
    def test_model_object_defaults(self):
        obj = ModelObject(id="test", created=1000, max_context_length=4096)
        assert obj.object == "model"
        assert obj.owned_by == "batchgen"

    def test_list_models_response_defaults(self):
        obj = ModelObject(id="test", created=1000, max_context_length=4096)
        resp = ListModelsResponse(data=[obj])
        assert resp.object == "list"
        assert len(resp.data) == 1

    def test_model_object_serialization(self):
        obj = ModelObject(id="test-model", created=12345, max_context_length=8192)
        d = obj.dict()
        assert d == {
            "id": "test-model",
            "object": "model",
            "created": 12345,
            "owned_by": "batchgen",
            "max_context_length": 8192,
        }
