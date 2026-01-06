"""FastAPI server exposing OpenAI-compatible batch APIs."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from batchgen.server.batch_scheduler import BatchScheduler, parse_batch_file
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
    RawInferenceRequest,
    normalize_inference_results,
)
from batchgen.server.server_args import ServerArgs
from batchgen.server.storage import StorageManager
from batchgen.server.worker_manager import WorkerExitState, WorkerManager

logger = logging.getLogger(__name__)


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def create_app(
    server_args: ServerArgs,
    worker_exit_state: Optional[WorkerExitState] = None,
) -> FastAPI:
    worker_exit_state = worker_exit_state or WorkerExitState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        storage = StorageManager(server_args.storage_path)
        worker = WorkerManager(server_args, worker_exit_state=worker_exit_state)
        scheduler = BatchScheduler(storage, worker, server_args)

        app.state.server_args = server_args
        app.state.storage = storage
        app.state.worker = worker
        app.state.scheduler = scheduler

        worker.start()
        await scheduler.start()

        try:
            yield
        finally:
            await scheduler.stop()
            worker.stop()

    app = FastAPI(title="BatchGen OpenAI-Compatible API", lifespan=lifespan)
    app.state.worker_exit_state = worker_exit_state

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

        max_input_len = body.max_input_len or server_args.max_input_len
        max_output_len = body.max_output_len or server_args.max_output_len

        start = time.perf_counter()
        try:
            results = await asyncio.to_thread(
                worker.infer,
                body.prompts,
                max_input_len,
                max_output_len,
                body.ignore_eos,
            )
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=str(exc))

        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "status": "success",
            "results": normalize_inference_results(results),
            "latency_ms": latency_ms,
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


def launch_server(server_args: ServerArgs) -> None:
    """Launch the HTTP server."""
    worker_exit_state = WorkerExitState()
    app = create_app(server_args, worker_exit_state)
    try:
        logger.info("Starting BatchGen HTTP server.")
        uvicorn.run(
            app,
            host=server_args.listen_ip,
            port=server_args.listen_port,
            log_level="info",
            timeout_keep_alive=1000,
            loop="uvloop",
        )
    finally:
        logger.info("HTTP server stopped.")
        if worker_exit_state.is_failed():
            reason = worker_exit_state.reason or "Worker process exited."
            raise RuntimeError(reason)
