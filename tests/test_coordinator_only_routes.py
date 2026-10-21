"""Batch and file routes are served by node 0 only.

Only global rank 0, on node 0, reads the request queue, so a batch accepted
by any other node's server was stored and then never run. Those nodes now
refuse the routes with a 409 naming node 0, and they never touch the batch
store at startup: on a shared storage path their recovery sweep would fail
node 0's live batches.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Optional

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

HTTP_SERVER_PATH = (
    Path(__file__).resolve().parents[1] / "batchgen" / "server" / "http_server.py"
)


def _tree() -> ast.Module:
    return ast.parse(HTTP_SERVER_PATH.read_text(), filename=str(HTTP_SERVER_PATH))


def _function(tree: ast.Module, name: str):
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _load_rejection():
    """The real module-level helper and its prefix table, without the server
    stack that importing http_server.py would pull in."""
    tree = _tree()
    prefixes = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "_COORDINATOR_ONLY_PREFIXES"
    )
    helper = copy.deepcopy(_function(tree, "coordinator_only_rejection"))
    namespace = {"JSONResponse": JSONResponse, "Optional": Optional}
    module = ast.Module(body=[prefixes, helper], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(HTTP_SERVER_PATH), "exec"),
        namespace,
    )
    return namespace["coordinator_only_rejection"]


coordinator_only_rejection = _load_rejection()


def _app(node_rank: int) -> TestClient:
    """A bare app with the same middleware create_app installs off node 0."""
    app = FastAPI()

    @app.middleware("http")
    async def refuse(request: Request, call_next):
        rejection = coordinator_only_rejection(node_rank, request.url.path)
        if rejection is not None:
            return rejection
        return await call_next(request)

    for path in ("/v1/files", "/v1/files/{file_id}", "/v1/files/{file_id}/content",
                 "/v1/batches", "/v1/batches/{batch_id}",
                 "/v1/batches/{batch_id}/cancel", "/health",
                 "/v1/workers/status", "/v1/filesystem"):
        app.add_api_route(path, lambda: {"served": True}, methods=["GET", "POST", "DELETE"])
    return TestClient(app)


BATCH_STATE_REQUESTS = [
    ("POST", "/v1/files"),
    ("GET", "/v1/files"),
    ("GET", "/v1/files/file-1"),
    ("DELETE", "/v1/files/file-1"),
    ("GET", "/v1/files/file-1/content"),
    ("POST", "/v1/batches"),
    ("GET", "/v1/batches"),
    ("GET", "/v1/batches/batch-1"),
    ("POST", "/v1/batches/batch-1/cancel"),
]


@pytest.mark.parametrize("method,path", BATCH_STATE_REQUESTS)
def test_a_node_other_than_node_zero_refuses_every_batch_state_route(method, path):
    response = _app(node_rank=2).request(method, path)

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "message": "batches and files are served by node 0; send them there",
        "coordinator_node": 0,
    }


@pytest.mark.parametrize("method,path", BATCH_STATE_REQUESTS)
def test_node_zero_serves_every_batch_state_route(method, path):
    response = _app(node_rank=0).request(method, path)

    assert response.status_code == 200
    assert response.json() == {"served": True}


@pytest.mark.parametrize("path", ["/health", "/v1/workers/status", "/v1/filesystem"])
def test_other_routes_stay_served_on_every_node(path):
    """Health and pool status are per-node; a path that merely shares the
    prefix text is not a batch-state route."""
    response = _app(node_rank=1).get(path)

    assert response.status_code == 200


def _create_app() -> ast.FunctionDef:
    return _function(_tree(), "create_app")


def _guarding_if(func: ast.AST, call_name: str) -> Optional[ast.If]:
    """The innermost `if` whose body (not orelse) contains a call to call_name."""
    found = None
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        for child in node.body:
            if any(
                isinstance(call, ast.Call)
                and ast.unparse(call.func).endswith("." + call_name)
                for call in ast.walk(child)
            ):
                found = node
    return found


def test_create_app_installs_the_refusal_only_off_node_zero():
    guard = _guarding_if(_create_app(), "middleware")
    assert guard is not None
    assert ast.unparse(guard.test) == "server_args.node_rank > 0"
    assert "coordinator_only_rejection" in ast.unparse(guard)


@pytest.mark.parametrize(
    "call", ["acquire_storage_ownership", "fail_orphaned_batches"]
)
def test_only_node_zero_owns_and_recovers_the_batch_store(call):
    """Every call to the store's ownership and recovery sits under the
    node-0 guard, so another node never fails node 0's live batches."""
    lifespan = _function(_create_app(), "lifespan")
    calls = [
        node for node in ast.walk(lifespan)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("." + call)
    ]
    assert calls, call
    guard = _guarding_if(lifespan, call)
    assert guard is not None
    assert ast.unparse(guard.test) == "server_args.node_rank == 0"
    guarded = {id(node) for node in ast.walk(guard)}
    assert all(id(node) in guarded for node in calls)
