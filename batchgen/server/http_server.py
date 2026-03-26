"""FastAPI server exposing OpenAI-compatible batch APIs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
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
    RawInferenceRequest,
    normalize_inference_results,
)
from batchgen.server.health import ServerHealthState
from batchgen.server.server_args import ServerArgs
from batchgen.server.storage import StorageManager
from batchgen.server.worker_manager import WorkerExitState, WorkerManager

logger = logging.getLogger(__name__)

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
        storage = StorageManager(server_args.storage_path)
        worker = WorkerManager(server_args, worker_exit_state=worker_exit_state)
        scheduler = BatchScheduler(storage, worker, server_args)

        app.state.server_args = server_args
        app.state.storage = storage
        app.state.worker = worker
        app.state.scheduler = scheduler

        # Build model metadata object (cached for lifetime of server)
        # Only use BatchGen-maintained registered configs — never fall back to HF config.json
        detected_type = _detect_model_type_from_identifier(server_args.model)
        if detected_type is None or detected_type not in CONFIG_REGISTRY:
            raise ValueError(
                f"Model '{server_args.model}' is not registered in BatchGen CONFIG_REGISTRY. "
                f"Cannot serve /v1/models metadata. Registered types: {list(CONFIG_REGISTRY.keys())}"
            )
        model_config = CONFIG_REGISTRY[detected_type]()
        model_id = Path(server_args.model).name
        app.state.model_object = ModelObject(
            id=model_id,
            created=int(time.time()),
            owned_by="batchgen",
            max_context_length=model_config.max_position_embeddings,
        )

        worker.start()
        await scheduler.start()
        health_state.mark_startup_complete()
        e2e_elapsed = time.monotonic() - e2e_start
        logger.info(
            "[startup] End-to-end server ready in %.2fs", e2e_elapsed,
        )
        print(f"[startup] End-to-end server ready in {e2e_elapsed:.2f}s", flush=True)

        try:
            yield
        finally:
            await scheduler.stop()
            worker.stop()

    app = FastAPI(title="BatchGen OpenAI-Compatible API", lifespan=lifespan)
    app.state.worker_exit_state = worker_exit_state
    app.state.health = health_state

    # Add custom access logging middleware (replaces uvicorn's default)
    app.add_middleware(QuietAccessLogMiddleware)

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
            ok, error_message, _ = parse_batch_file(content)
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
        batch = BatchObject(
            id=batch_id,
            endpoint=body.endpoint,
            input_file_id=body.input_file_id,
            completion_window=body.completion_window,
            status=BatchStatus.VALIDATING,
            created_at=now,
            expires_at=expires_at,
            metadata=body.metadata,
            # Inference parameters (override per-request values)
            max_decoding_length=body.max_decoding_length,
            max_context_length=body.max_context_length,
            temperature=body.temperature,
            top_p=body.top_p,
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

    @app.post("/v1/inference")
    async def run_inference(request: Request, body: RawInferenceRequest):
        worker: WorkerManager = request.app.state.worker
        server_args: ServerArgs = request.app.state.server_args
        storage: StorageManager = request.app.state.storage

        max_input_len = body.max_input_len  # None = dynamically determined
        max_output_len = body.max_output_len or 128  # Default max output tokens

        start = time.perf_counter()
        try:
            results = await asyncio.to_thread(
                worker.infer,
                body.prompts,
                max_input_len,
                max_output_len,
                body.ignore_eos,
                body.temperature,  # None = greedy decoding
                body.top_p,  # None = disabled
            )
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=str(exc))

        latency_ms = int((time.perf_counter() - start) * 1000)
        # Worker returns dict {global_idx: str} — convert to ordered list
        if isinstance(results, dict):
            results = [results[k] for k in sorted(results.keys())]
        normalized_results = normalize_inference_results(results)

        response_data = {
            "status": "success",
            "results": normalized_results,
            "latency_ms": latency_ms,
        }

        # Save results to file if save_result is enabled
        if server_args.save_result:
            output_file_id = f"file-{uuid.uuid4().hex}"
            output_path = storage.output_dir / f"{output_file_id}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as f:
                for idx, result in enumerate(normalized_results):
                    record = {
                        "custom_id": f"inference-{idx}",
                        "prompt": body.prompts[idx] if idx < len(body.prompts) else "",
                        "response": result,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Save file metadata
            file_meta = FileObject(
                id=output_file_id,
                bytes=output_path.stat().st_size,
                created_at=int(time.time()),
                filename=f"{output_file_id}.jsonl",
                purpose=FilePurpose.BATCH_OUTPUT.value,
                status=FileStatus.PROCESSED.value,
                status_details=None,
                checksum=None,
            )
            storage.save_metadata(output_file_id, file_meta.dict())

            # Also copy to files_dir for download via /v1/files/{id}/content
            import shutil
            shutil.copy(output_path, storage.files_dir / output_file_id)

            response_data["output_file_id"] = output_file_id
            logger.info(f"Saved inference results to {output_path}")

        return response_data

    @app.get("/health")
    async def health_check(request: Request):
        health: ServerHealthState = request.app.state.health
        worker_exit: WorkerExitState = request.app.state.worker_exit_state

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

    from batchgen.server.process_utils import cleanup_resources

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
                os._exit(1)

        t = threading.Thread(target=_startup_watchdog, daemon=True)
        t.start()

    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
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
            log_level="info",
            access_log=False,  # Use custom QuietAccessLogMiddleware instead
            timeout_keep_alive=1000,
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

        # Final cleanup in case lifespan cleanup was incomplete
        # (e.g., if server was killed before lifespan could run, or watchdog triggered)
        # Always clean hugepages on shutdown for complete resource release
        cleanup_resources(
            shm_prefix=None,  # Clean all shared memory files
            clean_hugepages=True,  # Always clean hugepages on shutdown
            kill_workers=True,  # Force kill any remaining workers
        )

        if worker_exit_state.is_failed():
            reason = worker_exit_state.reason or "Worker process exited."
            raise RuntimeError(reason)
