"""Per-turn agent API: POST /v1/turns and the session lifecycle endpoints.

An offline agent harness submits one model turn at a time and gets the answer
back in the same HTTP response: no file upload and no batch polling. Each turn
carries its trajectory and session identity in-band (the ``session`` object of
the body); the session lifecycle is signalled out of band:

    POST /v1/sessions                       open (traj_id, session_id, parent)
    POST /v1/sessions/{session_id}/close    reason end_of_session | end_of_trajectory
    POST /v1/sessions/{session_id}/abort    cancels the in-flight turn, if any
                                            (session_id may contain "/")
    POST /v1/turns                          one chat-completions turn

A turn is rendered exactly as a /v1/batches chat request, submitted to the
shared IntakePool as one entry, and answered when the completion listener
routes its worker completion; it coexists with running batches.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from batchgen.server.agent_sessions import SessionRegistry, TurnError
from batchgen.server.io_struct import (
    AbortSessionRequest,
    CloseSessionRequest,
    SessionIdentity,
    TurnRequest,
)


def _http_error(exc: TurnError) -> HTTPException:
    # A 503 (pool not serving, or failed and not yet restarted) is retryable,
    # as the batch path's 503 is.
    headers = {"Retry-After": "30"} if exc.status_code == 503 else None
    return HTTPException(
        status_code=exc.status_code, detail=exc.detail(), headers=headers
    )


def register_turn_routes(app: FastAPI) -> None:
    """Install the turn and session routes and the app's session registry.

    The handlers read ``app.state.scheduler`` (a BatchScheduler) and
    ``app.state.worker`` (its WorkerManager) at request time.
    """
    registry = SessionRegistry()
    app.state.session_registry = registry

    @app.post("/v1/sessions")
    async def open_session(body: SessionIdentity):
        try:
            record = registry.open(
                body.traj_id, body.session_id, body.parent_session_id
            )
        except TurnError as exc:
            raise _http_error(exc) from exc
        return record.to_dict()

    # ``:path`` so a session id may contain "/" (the documented
    # "<traj>/<name>" form); a plain parameter stops at the first slash and
    # the request 404s before reaching the handler.
    @app.post("/v1/sessions/{session_id:path}/close")
    async def close_session(session_id: str, body: CloseSessionRequest):
        try:
            closed = registry.close(session_id, body.reason)
        except TurnError as exc:
            raise _http_error(exc) from exc
        return {
            "object": "session.close",
            "reason": body.reason,
            "closed": [record.to_dict() for record in closed],
        }

    @app.post("/v1/sessions/{session_id:path}/abort")
    async def abort_session(
        request: Request,
        session_id: str,
        body: Optional[AbortSessionRequest] = None,
    ):
        reason = body.reason if body is not None else None
        try:
            record, turn_id = registry.abort(session_id, reason)
        except TurnError as exc:
            raise _http_error(exc) from exc
        # registry.abort refused (409 already_completed) a turn whose
        # completion is routed, so a turn cancelled here has not completed.
        cancelled = None
        if turn_id is not None:
            cancelled = request.app.state.scheduler.cancel_turn(
                turn_id, f"session {session_id} aborted ({record.end_reason})"
            )
        return {
            **record.to_dict(),
            "cancelled_turn_id": turn_id,
            "cancelled_turn": cancelled,
        }

    @app.post("/v1/turns")
    async def create_turn(request: Request):
        # Stamped before the body is read: the receipt -> submit leg includes
        # body parsing, validation and prompt rendering.
        received_at = time.time()
        scheduler = request.app.state.scheduler
        worker_manager = request.app.state.worker

        # Same admission gates as POST /v1/batches.
        pool_state = worker_manager.state
        if pool_state != "serving":
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Worker pool is not serving (state={pool_state}). "
                    "Turn submission is unavailable until the pool is serving "
                    "again. Retry later."
                ),
                headers={"Retry-After": "30"},
            )
        if scheduler.intake_pool_usage_pct() > 0.9:
            pool_size = scheduler.intake_pool_size()
            pool_cap = scheduler.intake_pool_capacity()
            raise HTTPException(
                status_code=429,
                detail=f"Server at capacity ({pool_size}/{pool_cap} requests in queue). Retry later.",
                headers={"Retry-After": "30"},
            )

        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"request body is not JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="request body must be a JSON object"
            )
        try:
            body = TurnRequest(**payload)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail=json.loads(exc.json())
            ) from exc

        session = body.session
        turn_id = f"turn_{uuid.uuid4().hex[:24]}"
        try:
            registry.begin_turn(
                session.traj_id,
                session.session_id,
                session.parent_session_id,
                turn_id,
            )
        except TurnError as exc:
            raise _http_error(exc) from exc
        try:
            waiter = scheduler.submit_turn(
                turn_id, body, received_at,
                on_routed=lambda: registry.complete_turn(
                    session.session_id, turn_id
                ),
            )
            try:
                result = await waiter.future
            except asyncio.CancelledError:
                # The request itself went away (client gone, server shutting
                # down): the turn must not stay pending.
                scheduler.cancel_turn(turn_id, "HTTP request cancelled")
                raise
            return scheduler.build_turn_response(waiter, result)
        except TurnError as exc:
            raise _http_error(exc) from exc
        finally:
            registry.end_turn(session.session_id, turn_id)
