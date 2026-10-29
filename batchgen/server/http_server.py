"""FastAPI server exposing OpenAI-compatible batch APIs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from batchgen.server.batch_scheduler import BatchScheduler, parse_batch_file
from batchgen.config.model_registry import (
    CONFIG_REGISTRY,
    _detect_model_type_from_identifier,
)
from batchgen.server.io_struct import (
    BatchObject,
    BatchStatus,
    CreateBatchRequest,
    DeleteFileResponse,
    FileObject,
    FilePurpose,
    FileStatus,
    ListBatchesRequest,
    ListBatchesResponse,
    ListFilesRequest,
    ListFilesResponse,
    ListModelsResponse,
    ModelObject,
    build_batch_object_from_create_request,
)
from batchgen.deprecation import (
    LEGACY_INFERENCE_ERROR_CODE,
    LEGACY_INFERENCE_MESSAGE,
)
from batchgen.server.health import ServerHealthState
from batchgen.eventlog.process import exit_process
from batchgen.server.restart_coordinator import PeerRestartError
from batchgen.server.server_args import (
    ServerArgs,
    validate_shared_runtime_capability,
)
from batchgen.server.storage import StorageManager
from batchgen.server.turn_api import register_turn_routes
from batchgen.server.worker_manager import WorkerExitState, WorkerManager

logger = logging.getLogger(__name__)

# How long a shutdown waits for open requests before cancelling them. Every
# request but a worker restart is short; a restart is cancelled here and
# aborted by WorkerManager.stop() at its next phase boundary.
_GRACEFUL_SHUTDOWN_TIMEOUT_S = 5

# Routes that read or change batch state. Only global rank 0, which runs on node
# 0, reads the request queue, so a batch accepted by any other node's server is
# stored and then never run. Turns and their sessions go the same way.
_COORDINATOR_ONLY_PREFIXES = ("/v1/files", "/v1/batches", "/v1/turns", "/v1/sessions")


def coordinator_only_rejection(
    node_rank: int, path: str
) -> Optional[JSONResponse]:
    """The 409 a node other than node 0 answers for a batch-state route."""
    if node_rank == 0:
        return None
    if not any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in _COORDINATOR_ONLY_PREFIXES
    ):
        return None
    return JSONResponse(
        status_code=409,
        content={"detail": {
            "message": "batches and files are served by node 0; send them there",
            "coordinator_node": 0,
        }},
    )


# Pattern for endpoints that should have quiet logging (only log errors)
# GET requests to batch status endpoints are very frequent during polling
QUIET_ENDPOINTS = re.compile(r"^GET /v1/batches/[^/]+$|^GET /v1/models|^GET /health$")


class QuietAccessLogMiddleware(BaseHTTPMiddleware):
    """Middleware to reduce verbose access logging for polling endpoints.

    Only logs requests that:
    - Return non-2xx status codes (errors)
    - Are POST/PUT/DELETE requests (state changes)
    - Are not in the QUIET_ENDPOINTS pattern
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)

        # Build request signature
        request_line = f"{request.method} {request.url.path}"

        # Decide whether to log
        should_log = True
        if QUIET_ENDPOINTS.match(request_line):
            # Only log if error status
            should_log = response.status_code >= 400

        if should_log:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"{request.client.host}:{request.client.port} - "
                f"\"{request_line}\" {response.status_code} ({process_time:.0f}ms)"
            )

        return response


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def make_main_thread_dispatcher(loop):
    """Bridge from a restart thread back onto the loop's main thread.

    Workers arm PR_SET_PDEATHSIG, which the kernel delivers when the thread
    that spawned them exits, so every respawn has to run on the main thread.
    A restart can start without a request behind it -- a peer node follows
    node 0 from the coordinator's thread -- so the bridge is installed for the
    server's whole life, not only per request.
    """

    def dispatch_on_main_thread(fn, dispatch_timeout):
        done = threading.Event()
        result = []

        def invoke():
            try:
                result.append((True, fn()))
            except BaseException as exc:
                result.append((False, exc))
            finally:
                done.set()

        loop.call_soon_threadsafe(invoke)
        if not done.wait(dispatch_timeout):
            raise TimeoutError("main-thread worker spawn dispatch timed out")
        ok, value = result[0]
        if not ok:
            raise value
        return value

    return dispatch_on_main_thread


def create_app(
    server_args: ServerArgs,
    worker_exit_state: Optional[WorkerExitState] = None,
    health_state: Optional[ServerHealthState] = None,
) -> FastAPI:
    worker_exit_state = worker_exit_state or WorkerExitState()
    health_state = health_state or ServerHealthState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        e2e_start = time.monotonic()
        validate_shared_runtime_capability(server_args)
        # A previous server process may have died with batches still active in
        # the shared storage; fail them now so their input files are reusable.
        storage = StorageManager(server_args.storage_path)
        # Only node 0 owns batch state: every other node refuses the batch
        # routes, so it never writes a batch. Its recovery sweep would instead
        # fail node 0's live batches whenever the nodes share a storage path,
        # and the ownership flock does not exclude across nodes on a network
        # filesystem.
        if server_args.node_rank == 0:
            # Exclusive cross-process ownership first: recovery must never
            # fail batches that belong to a live server sharing this path.
            storage.acquire_storage_ownership()
            orphaned = storage.fail_orphaned_batches()
            if orphaned:
                logger.warning(
                    "Failed %d batch(es) left active by a previous server "
                    "process: %s",
                    len(orphaned), orphaned,
                )
        worker = WorkerManager(server_args, worker_exit_state=worker_exit_state)
        app.state.server_args = server_args
        app.state.worker = worker
        scheduler = None
        scheduler_started = False
        try:
            worker._acquire_runtime_admission()
            storage = StorageManager(server_args.storage_path)
            scheduler = BatchScheduler(storage, worker, server_args)
            worker.set_lifecycle_hooks(
                quiesce=scheduler.quiesce,
                resume=scheduler.resume,
                fail_in_flight=scheduler.fail_in_flight,
                generation_ready=scheduler.worker_generation_ready,
                pool_init_params=scheduler.pool_init_params,
                retire_channels=scheduler.stop_completion_pump,
            )
            app.state.storage = storage
            app.state.scheduler = scheduler

            # Build model metadata once for the server lifetime. Only use
            # BatchGen-maintained configs; never fall back to HF config.json.
            detected_type = _detect_model_type_from_identifier(
                server_args.model
            )
            if detected_type is None or detected_type not in CONFIG_REGISTRY:
                raise ValueError(
                    f"Model '{server_args.model}' is not registered in "
                    "BatchGen CONFIG_REGISTRY. Cannot serve /v1/models "
                    f"metadata. Registered types: {list(CONFIG_REGISTRY.keys())}"
                )
            model_config = CONFIG_REGISTRY[detected_type]()
            model_id = Path(server_args.model).name
            app.state.model_object = ModelObject(
                id=model_id,
                created=int(time.time()),
                owned_by="batchgen",
                max_context_length=model_config.max_position_embeddings,
            )

            # Installed before the pool exists: worker.start() arms the
            # coordinator's follower, which may begin a coordinated restart as
            # soon as node 0 asks for one, with no request to carry a bridge.
            worker.set_main_thread_dispatcher(
                make_main_thread_dispatcher(asyncio.get_running_loop())
            )
            worker.start()
            await scheduler.start()
            scheduler_started = True
            health_state.mark_startup_complete()
            e2e_elapsed = time.monotonic() - e2e_start
            logger.info(
                "[startup] End-to-end server ready in %.2fs", e2e_elapsed,
            )
            yield
        finally:
            shutdown_start = time.monotonic()
            logging.getLogger("uvicorn.error").info("[shutdown] scheduler stop begin")
            try:
                if scheduler_started:
                    await scheduler.stop()
            finally:
                scheduler_stop_end = time.monotonic()
                logging.getLogger("uvicorn.error").info(
                    "[shutdown] scheduler stop elapsed=%.3fs",
                    scheduler_stop_end - shutdown_start,
                )
                try:
                    worker.stop()
                finally:
                    logging.getLogger("uvicorn.error").info(
                        "[shutdown] worker stop elapsed=%.3fs total=%.3fs",
                        time.monotonic() - scheduler_stop_end,
                        time.monotonic() - shutdown_start,
                    )

    app = FastAPI(title="BatchGen OpenAI-Compatible API", lifespan=lifespan)
    app.state.worker_exit_state = worker_exit_state
    app.state.health = health_state

    # Add custom access logging middleware (replaces uvicorn's default)
    app.add_middleware(QuietAccessLogMiddleware)

    if server_args.node_rank > 0:
        @app.middleware("http")
        async def refuse_batch_routes_off_node_zero(request: Request, call_next):
            rejection = coordinator_only_rejection(
                server_args.node_rank, request.url.path
            )
            if rejection is not None:
                return rejection
            return await call_next(request)

    # ==================== Model Endpoints ====================

    @app.get("/v1/models", response_model=ListModelsResponse)
    async def list_models(request: Request):
        return ListModelsResponse(data=[request.app.state.model_object])

    @app.get("/v1/models/{model_id:path}", response_model=ModelObject)
    async def retrieve_model(request: Request, model_id: str):
        model_obj: ModelObject = request.app.state.model_object
        if model_id != model_obj.id:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return model_obj

    # ==================== Pool Status Endpoint ====================

    @app.get("/v1/pool/status")
    async def pool_status(request: Request):
        scheduler: BatchScheduler = request.app.state.scheduler
        spool = scheduler._scheduling_pool
        ipool = scheduler._intake_pool
        active_batches = 0
        with spool._lock:
            for t in spool._batch_trackers.values():
                if not t.is_complete and not t.is_failed:
                    active_batches += 1
        return {
            "intake_pool_size": ipool.size(),
            "intake_pool_capacity": ipool.max_capacity,
            "scheduling_pool_active": spool.num_active_slots(),
            "scheduling_pool_free": spool.num_free_slots(),
            "scheduling_pool_capacity": spool.capacity,
            "trajectory_pool": scheduler._trajectory_pool_info,
            "active_batches": active_batches,
            "pool_mode": scheduler._pool_mode,
        }

    # ==================== File Endpoints ====================

    @app.post("/v1/files", response_model=FileObject)
    async def upload_file(
        request: Request, file: UploadFile = File(...), purpose: str = Form(...)
    ):
        storage: StorageManager = request.app.state.storage
        try:
            file_purpose = FilePurpose(purpose)
        except ValueError:
            valid = [p.value for p in FilePurpose]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid purpose '{purpose}'. Must be one of: {', '.join(valid)}",
            )

        content = await file.read()
        if file_purpose == FilePurpose.BATCH:
            ok, error_message, _ = parse_batch_file(
                content, served_model=server_args.model
            )
            if not ok:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid batch file: {error_message}",
                )

        checksum = hashlib.sha256(content).hexdigest()
        existing = storage.find_file_by_checksum(checksum)
        if existing:
            return FileObject(**existing)

        file_id = f"file-{uuid.uuid4().hex}"
        file_path = storage.files_dir / file_id
        with file_path.open("wb") as handle:
            handle.write(content)

        created_at = int(time.time())
        metadata = FileObject(
            id=file_id,
            bytes=len(content),
            created_at=created_at,
            filename=file.filename or "uploaded_file",
            purpose=file_purpose.value,
            status=FileStatus.PROCESSED.value,
            status_details=None,
            checksum=checksum,
        )
        storage.save_metadata(file_id, metadata.dict())
        return metadata

    @app.get("/v1/files", response_model=ListFilesResponse)
    async def list_files(
        request: Request,
        purpose: Optional[str] = None,
        limit: int = 10000,
        order: str = "desc",
        after: Optional[str] = None,
    ):
        storage: StorageManager = request.app.state.storage
        try:
            query_params = ListFilesRequest(
                purpose=purpose, limit=limit, order=order, after=after
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        all_metadata = storage.list_all_metadata()
        if query_params.purpose:
            all_metadata = [
                m
                for m in all_metadata
                if m.get("purpose") == query_params.purpose
            ]

        reverse_order = query_params.order == "desc"
        all_metadata.sort(key=lambda x: x["created_at"], reverse=reverse_order)

        if query_params.after:
            for idx, meta in enumerate(all_metadata):
                if meta.get("id") == query_params.after:
                    all_metadata = all_metadata[idx + 1 :]
                    break

        has_more = len(all_metadata) > query_params.limit
        all_metadata = all_metadata[: query_params.limit]
        return ListFilesResponse(
            data=[FileObject(**m) for m in all_metadata], has_more=has_more
        )

    @app.get("/v1/files/{file_id}", response_model=FileObject)
    async def retrieve_file(request: Request, file_id: str):
        storage: StorageManager = request.app.state.storage
        metadata = storage.load_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        return FileObject(**metadata)

    @app.delete("/v1/files/{file_id}", response_model=DeleteFileResponse)
    async def delete_file(request: Request, file_id: str):
        storage: StorageManager = request.app.state.storage
        metadata = storage.load_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")

        file_path = storage.files_dir / file_id
        if file_path.exists():
            file_path.unlink()

        storage.delete_file_metadata(file_id)
        return DeleteFileResponse(id=file_id, deleted=True)

    @app.get("/v1/files/{file_id}/content")
    async def retrieve_file_content(request: Request, file_id: str):
        storage: StorageManager = request.app.state.storage
        metadata = storage.load_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        file_path = storage.files_dir / file_id
        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail="File content not found"
            )
        return FileResponse(
            path=file_path,
            filename=metadata.get("filename"),
            media_type="application/octet-stream",
        )

    @app.post("/v1/batches", response_model=BatchObject, status_code=200)
    async def create_batch(request: Request, body: CreateBatchRequest):
        storage: StorageManager = request.app.state.storage
        scheduler: BatchScheduler = request.app.state.scheduler
        worker_manager: WorkerManager = request.app.state.worker

        # Admission control: a batch may only be created while the worker pool
        # is serving. Accepting one during a restart or while the pool is down
        # hands the client a success receipt for work the server knows it
        # cannot start, and the batch can never run.
        pool_state = worker_manager.state
        if pool_state != "serving":
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Worker pool is not serving (state={pool_state}). "
                    "Batch creation is unavailable until the pool is serving "
                    "again. Retry later."
                ),
                headers={"Retry-After": "30"},
            )

        # Admission control: reject early if intake pool is near capacity (>90%)
        if scheduler.intake_pool_usage_pct() > 0.9:
            pool_size = scheduler.intake_pool_size()
            pool_cap = scheduler.intake_pool_capacity()
            raise HTTPException(
                status_code=429,
                detail=f"Server at capacity ({pool_size}/{pool_cap} requests in queue). Retry later.",
                headers={"Retry-After": "30"},
            )

        input_meta = storage.load_metadata(body.input_file_id)
        if not input_meta:
            raise HTTPException(
                status_code=400,
                detail=f"No such file object: {body.input_file_id}",
            )
        if input_meta.get("purpose") != FilePurpose.BATCH.value:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file purpose. Expected 'batch', got '{input_meta.get('purpose')}'",
            )

        existing_batch = storage.get_active_batch_for_file(body.input_file_id)
        if existing_batch:
            raise HTTPException(
                status_code=400,
                detail=f"File '{body.input_file_id}' already has active batch "
                f"'{existing_batch.id}' with status '{existing_batch.status}'",
            )

        batch_id = f"batch_{uuid.uuid4().hex[:24]}"
        now = int(time.time())
        expires_at = now + 86400
        batch = build_batch_object_from_create_request(
            batch_id=batch_id,
            body=body,
            created_at=now,
            expires_at=expires_at,
        )
        storage.save_batch(batch)
        await scheduler.enqueue(batch_id)
        return batch

    @app.get("/v1/batches", response_model=ListBatchesResponse)
    async def list_batches(
        request: Request, after: Optional[str] = None, limit: int = 20
    ):
        storage: StorageManager = request.app.state.storage
        try:
            query_params = ListBatchesRequest(after=after, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        all_batches = storage.list_all_batches()
        if query_params.after:
            for idx, batch in enumerate(all_batches):
                if batch.id == query_params.after:
                    all_batches = all_batches[idx + 1 :]
                    break

        has_more = len(all_batches) > query_params.limit
        limited = all_batches[: query_params.limit]
        first_id = limited[0].id if limited else None
        last_id = limited[-1].id if limited else None
        return ListBatchesResponse(
            data=limited, first_id=first_id, last_id=last_id, has_more=has_more
        )

    @app.get("/v1/batches/{batch_id}", response_model=BatchObject)
    async def retrieve_batch(request: Request, batch_id: str):
        storage: StorageManager = request.app.state.storage
        batch = storage.load_batch(batch_id)
        if not batch:
            raise HTTPException(
                status_code=404, detail=f"Batch '{batch_id}' not found"
            )
        return batch

    @app.post("/v1/batches/{batch_id}/cancel", response_model=BatchObject)
    async def cancel_batch(request: Request, batch_id: str):
        storage: StorageManager = request.app.state.storage
        batch = storage.load_batch(batch_id)
        if not batch:
            raise HTTPException(
                status_code=404, detail=f"Batch '{batch_id}' not found"
            )

        cancellable = {BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS}
        if batch.status not in cancellable:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel batch with status '{batch.status}'."
                f" Valid statuses: validating, in_progress",
            )

        now = int(time.time())
        storage.update_batch_status(
            batch_id, BatchStatus.CANCELLED, cancelling_at=now, cancelled_at=now
        )
        updated = storage.load_batch(batch_id)
        return updated

    # ==================== Turn and Session Endpoints ====================

    register_turn_routes(app)

    @app.post("/v1/inference")
    async def run_inference():
        """DEPRECATED and disabled. All inference goes through the batch API.

        Rejected here, at the door, before any queue interaction. The old body's
        first act was to put a payload on the shared worker request queue, and
        in pool mode nothing downstream could undo that: the worker admission
        loop recognises no legacy message, so the payload was dropped there
        while the caller blocked on the shared response queue and took the next
        batch's completion.

        The route is kept rather than deleted so callers get this explanation
        instead of a 404 they would read as a typo. It takes no request body:
        an unparseable legacy payload must still get the deprecation notice,
        not a 422 about its schema.
        """
        raise HTTPException(
            status_code=410,
            detail={
                "code": LEGACY_INFERENCE_ERROR_CODE,
                "message": LEGACY_INFERENCE_MESSAGE,
                "use_instead": "/v1/batches",
            },
        )

    @app.post("/v1/workers/restart")
    async def restart_workers(request: Request):
        worker_manager: WorkerManager = request.app.state.worker
        server: ServerArgs = request.app.state.server_args
        if server.node_rank > 0:
            # Only node 0 may claim a restart epoch: a cohort spans every node,
            # so two entry points would tear the same world down twice.
            raise HTTPException(
                status_code=409,
                detail={
                    "message": (
                        "worker restart is coordinated by node 0; send it there"
                    ),
                    "coordinator_node": 0,
                },
            )
        try:
            body = await request.json()
        except Exception:
            body = {}
        timeout = float(body.get("timeout", 600.0))
        worker_manager.set_main_thread_dispatcher(
            make_main_thread_dispatcher(asyncio.get_running_loop())
        )
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(worker_manager.restart_workers, timeout_s=timeout),
                timeout=timeout,
            )
        except PeerRestartError as exc:
            # A peer's failure is the restart's outcome, not an internal error
            # of this node, so the node and phase that failed are named.
            return JSONResponse(
                status_code=500,
                content={
                    "state": worker_manager.state,
                    "generation": worker_manager.generation,
                    "epoch": worker_manager.status().get("restart_epoch"),
                    "error": str(exc),
                    "failed_node": exc.node,
                    "failed_phase": exc.phase,
                },
            )
        except RuntimeError as exc:
            if "restart conflict" in str(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            return JSONResponse(
                status_code=504,
                content={
                    "state": worker_manager.state,
                    "generation": worker_manager.generation,
                    "error": str(exc),
                    "partial": True,
                },
            )

    @app.get("/v1/workers/status")
    async def workers_status(request: Request):
        return request.app.state.worker.status()

    @app.get("/health")
    async def health_check(request: Request):
        health: ServerHealthState = request.app.state.health
        worker_exit: WorkerExitState = request.app.state.worker_exit_state
        worker_manager: WorkerManager = request.app.state.worker

        if worker_manager.state in {"workers_down", "restarting"}:
            if worker_manager.state == "restarting":
                reason = worker_exit.reason or "Worker restart in progress."
            else:
                reason = worker_exit.reason or "Worker pool is down."
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": reason},
            )

        if worker_exit.is_failed():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "reason": worker_exit.reason or "Worker process exited.",
                },
            )

        healthy, reason = health.is_healthy()
        if not healthy:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": reason},
            )

        return {"status": "healthy"}

    return app


def launch_server(server_args: ServerArgs) -> None:
    """Launch the HTTP server."""
    import os
    import signal
    import threading

    worker_exit_state = WorkerExitState()
    health_state = ServerHealthState()
    app = create_app(server_args, worker_exit_state, health_state)
    shutdown_requested = False

    # Startup timeout: if the server doesn't become ready in time, log error
    # and force-exit the process with exit code 1. The caller (shell, systemd,
    # K8s) sees the non-zero exit code and error logs.
    if server_args.startup_timeout is not None:
        def _startup_watchdog():
            logger.info(
                "Startup watchdog started (timeout=%ss, pid=%s).",
                server_args.startup_timeout, os.getpid(),
            )
            deadline = time.monotonic() + server_args.startup_timeout
            while time.monotonic() < deadline:
                if health_state.is_startup_complete():
                    logger.info("Startup completed successfully.")
                    return
                time.sleep(1.0)
            if not health_state.is_startup_complete():
                logger.error(
                    "Server failed to start within %ss. Exiting with code 1.",
                    server_args.startup_timeout,
                )
                exit_process(1, "startup timeout", logger=logger)

        t = threading.Thread(target=_startup_watchdog, daemon=True)
        t.start()

    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        logging.getLogger("uvicorn.error").info(
            "[shutdown] server received %s", sig_name
        )
        shutdown_requested = True
        # Re-raise to let uvicorn handle shutdown
        raise KeyboardInterrupt()

    # Install signal handlers for graceful shutdown
    # SIGQUIT is sent by watchdog when worker timeout occurs
    original_sigint = signal.signal(signal.SIGINT, shutdown_handler)
    original_sigterm = signal.signal(signal.SIGTERM, shutdown_handler)
    original_sigquit = signal.signal(signal.SIGQUIT, shutdown_handler)

    try:
        logger.info("Starting BatchGen HTTP server.")
        uvicorn.run(
            app,
            host=server_args.listen_ip,
            port=server_args.listen_port,
            # No logging config of uvicorn's own: its loggers propagate to the
            # root event-log handler and inherit the root level.
            log_config=None,
            log_level=None,
            access_log=False,  # Use custom QuietAccessLogMiddleware instead
            timeout_keep_alive=1000,
            # Without a bound uvicorn waits for every open request before the
            # lifespan shutdown runs, and POST /v1/workers/restart stays open
            # for the whole restart: a SIGTERM would wait out the restart
            # timeout before WorkerManager.stop() could tell the restart peers
            # and tear the cohort down.
            timeout_graceful_shutdown=_GRACEFUL_SHUTDOWN_TIMEOUT_S,
            loop="uvloop",
        )
    except KeyboardInterrupt:
        if shutdown_requested:
            logger.info("Graceful shutdown initiated by signal.")
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGQUIT, original_sigquit)

        logger.info("HTTP server stopped.")

        if worker_exit_state.is_failed():
            reason = worker_exit_state.reason or "Worker process exited."
            raise RuntimeError(reason)
