import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from batchgen.managers.io_struct import (
    DeleteFileResponse,
    FileObject,
    ListFilesResponse,
    FilePurpose,
    FileStatus,
    ListFilesRequest,
    BatchObject,
    BatchStatus,
    CreateBatchRequest,
    ListBatchesRequest,
    ListBatchesResponse,
)
from batchgen.server_args import ServerArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
app = FastAPI(title="OpenAI Compatible Files API")

# Storage configuration
STORAGE_PATH = None
METADATA_PATH = None
BATCHES_PATH = None

@contextmanager
async def lifespan(app: FastAPI):
    # Startup code
    # load parameter, return a shm_name
    # allocate kv cache

    # init global ServerScheduler
    yield
    # Shutdown code
    # free kv cache
    # free parameter
    # free global ServerScheduler
    pass


def setup_storage(path: str):
    global STORAGE_PATH, METADATA_PATH, BATCHES_PATH
    STORAGE_PATH = Path(path)
    METADATA_PATH = STORAGE_PATH / "metadata"
    BATCHES_PATH = STORAGE_PATH / "batches"
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.mkdir(parents=True, exist_ok=True)
    BATCHES_PATH.mkdir(parents=True, exist_ok=True)


def save_metadata(file_id: str, metadata: dict):
    """Save file metadata to disk."""
    metadata_file = METADATA_PATH / f"{file_id}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def load_metadata(file_id: str) -> Optional[dict]:
    """Load file metadata from disk."""
    metadata_file = METADATA_PATH / f"{file_id}.json"
    if not metadata_file.exists():
        return None
    with open(metadata_file, "r") as f:
        return json.load(f)


def list_all_metadata() -> List[dict]:
    """List all file metadata."""
    metadata_list = []
    for metadata_file in (METADATA_PATH).glob("*.json"):
        with open(metadata_file, "r") as f:
            metadata_list.append(json.load(f))
    return metadata_list

def save_batch(batch: BatchObject) -> None:
    """Save batch metadata to disk"""
    batch_file = BATCHES_PATH / f"{batch.id}.json"
    with open(batch_file, "w") as f:
        json.dump(batch.dict(), f)

def load_batch(batch_id: str) -> Optional[BatchObject]:
    """Load batch metadata from disk"""
    batch_file = BATCHES_PATH / f"{batch_id}.json"
    if not batch_file.exists():
        return None
    with open(batch_file, "r") as f:
        data = json.load(f)
    return BatchObject(**data)

def list_all_batches() -> List[BatchObject]:
    """List all batches sorted by creation time (newest first)"""
    batches = []
    for batch_file in BATCHES_PATH.glob("*.json"):
        batch = load_batch(batch_file.stem)
        if batch:
            batches.append(batch)
    return sorted(batches, key=lambda x: x.created_at, reverse=True)


@app.post("/v1/files", response_model=FileObject)
async def upload_file(file: UploadFile = File(...), purpose: str = Form(...)):
    """
    Upload a file that can be used across various endpoints.
    
    The file upload API allows you to upload documents that you or your applications can use.
    
    Reference: https://platform.openai.com/docs/api-reference/files/create
    """
    # Validate purpose using Pydantic enum
    try:
        file_purpose = FilePurpose(purpose)
    except ValueError:
        valid_purposes = [p.value for p in FilePurpose]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid purpose '{purpose}'. Must be one of: {', '.join(valid_purposes)}",
        )

    # Generate unique file ID
    file_id = f"file-{uuid.uuid4().hex}"

    # Save file to disk
    file_path = app.server_args.file_path / file_id
    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    # Create metadata
    created_at = int(time.time())
    metadata = {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": created_at,
        "filename": file.filename or "uploaded_file",
        "purpose": file_purpose.value,
        "status": FileStatus.PROCESSED.value,
        "status_details": None,
    }

    save_metadata(file_id, metadata)

    return FileObject(**metadata)


@app.get("/v1/files", response_model=ListFilesResponse)
async def list_files(
    purpose: Optional[str] = None,
    limit: int = 10000,
    order: str = "desc",
    after: Optional[str] = None
):
    """
    Returns a list of files that belong to the user's organization.
    
    Reference: https://platform.openai.com/docs/api-reference/files/list
    """
    # Validate query parameters using Pydantic model
    try:
        query_params = ListFilesRequest(
            purpose=FilePurpose(purpose) if purpose else None,
            limit=limit,
            order=order,
            after=after
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    all_metadata = list_all_metadata()

    # Filter by purpose if specified
    if query_params.purpose:
        purpose_value = query_params.purpose if isinstance(query_params.purpose, str) else query_params.purpose.value
        all_metadata = [m for m in all_metadata if m["purpose"] == purpose_value]

    # Sort by created_at
    reverse_order = (query_params.order == "desc")
    all_metadata.sort(key=lambda x: x["created_at"], reverse=reverse_order)
    
    # Apply pagination with 'after' cursor
    if query_params.after:
        # Find the index of the 'after' file
        after_index = None
        for i, m in enumerate(all_metadata):
            if m["id"] == query_params.after:
                after_index = i
                break
        
        if after_index is not None:
            # Get items after this index
            all_metadata = all_metadata[after_index + 1:]
    
    # Apply limit
    has_more = len(all_metadata) > query_params.limit
    all_metadata = all_metadata[:query_params.limit]

    files = [FileObject(**m) for m in all_metadata]

    return ListFilesResponse(data=files, has_more=has_more)


@app.get("/v1/files/{file_id}", response_model=FileObject)
async def retrieve_file(file_id: str):
    """
    Returns information about a specific file.
    """
    metadata = load_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    return FileObject(**metadata)


@app.delete("/v1/files/{file_id}", response_model=DeleteFileResponse)
async def delete_file(file_id: str):
    """
    Delete a file.
    """
    metadata = load_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    # Delete actual file
    file_path = STORAGE_PATH / file_id
    if file_path.exists():
        file_path.unlink()

    # Delete metadata
    metadata_file = METADATA_PATH / f"{file_id}.json"
    if metadata_file.exists():
        metadata_file.unlink()

    return DeleteFileResponse(id=file_id, deleted=True)


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    """
    Returns the contents of the specified file.
    """
    metadata = load_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = STORAGE_PATH / file_id

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File content not found")

    return FileResponse(
        path=file_path,
        filename=metadata["filename"],
        media_type="application/octet-stream",
    )


@app.post("/v1/batches", response_model=BatchObject, status_code=200)
async def create_batch(request: CreateBatchRequest):
    """
    Creates and executes a batch from an uploaded file of requests.
    
    The batch will be processed asynchronously. You can poll the batch status
    using the retrieve batch endpoint.
    
    Reference: https://platform.openai.com/docs/api-reference/batch/create
    """
    # Validate that the input file exists
    input_file_metadata = load_metadata(request.input_file_id)
    if not input_file_metadata:
        raise HTTPException(
            status_code=400,
            detail=f"No such file object: {request.input_file_id}"
        )
    
    # Validate that the input file has the correct purpose
    if input_file_metadata.get("purpose") != "batch":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file purpose. Expected 'batch', got '{input_file_metadata.get('purpose')}'"
        )
    
    # Generate unique batch ID
    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    
    # Calculate timestamps
    now = int(datetime.now().timestamp())
    expires_at = now + 86400  # 24 hours from now
    
    # Create batch object
    batch = BatchObject(
        id=batch_id,
        endpoint=request.endpoint.value,
        input_file_id=request.input_file_id,
        completion_window=request.completion_window.value,
        status=BatchStatus.VALIDATING,
        created_at=now,
        expires_at=expires_at,
        metadata=request.metadata
    )
    
    # Save batch
    save_batch(batch)
    
    return batch


@app.get("/v1/batches", response_model=ListBatchesResponse)
async def list_batches(
    after: Optional[str] = None,
    limit: int = 20
):
    """
    List all batches with pagination support.
    
    Reference: https://platform.openai.com/docs/api-reference/batch/list
    """
    # Validate query parameters using Pydantic model
    try:
        query_params = ListBatchesRequest(
            after=after,
            limit=limit
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Get all batches (already sorted by creation time, newest first)
    all_batches = list_all_batches()
    
    # Apply pagination with 'after' cursor
    if query_params.after:
        # Find the index of the 'after' batch
        after_index = None
        for i, batch in enumerate(all_batches):
            if batch.id == query_params.after:
                after_index = i
                break
        
        if after_index is not None:
            # Get batches after this index
            all_batches = all_batches[after_index + 1:]
    
    # Apply limit
    has_more = len(all_batches) > query_params.limit
    limited_batches = all_batches[:query_params.limit]
    
    # Get first and last IDs
    first_id = limited_batches[0].id if limited_batches else None
    last_id = limited_batches[-1].id if limited_batches else None
    
    return ListBatchesResponse(
        data=limited_batches,
        first_id=first_id,
        last_id=last_id,
        has_more=has_more
    )


@app.get("/v1/batches/{batch_id}", response_model=BatchObject)
async def retrieve_batch(batch_id: str):
    """
    Retrieve information about a specific batch.
    
    Reference: https://platform.openai.com/docs/api-reference/batch/retrieve
    """
    batch = load_batch(batch_id)
    
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    
    return batch


@app.post("/v1/batches/{batch_id}/cancel", response_model=BatchObject)
async def cancel_batch(batch_id: str):
    """
    Cancel a batch that is in progress.
    
    You can only cancel batches with status 'validating' or 'in_progress'.
    
    Reference: https://platform.openai.com/docs/api-reference/batch/cancel
    """
    batch = load_batch(batch_id)
    
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    
    # Check if batch can be cancelled
    cancellable_statuses = [BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS]
    if batch.status not in [s.value for s in cancellable_statuses]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel batch with status '{batch.status}'. "
                   f"Only batches with status 'validating' or 'in_progress' can be cancelled."
        )
    
    # Update batch status to cancelling
    now = int(datetime.now().timestamp())
    batch.status = BatchStatus.CANCELLING
    batch.cancelling_at = now
    
    # In a real implementation, you would trigger the cancellation process here
    # For now, we'll immediately mark it as cancelled
    batch.status = BatchStatus.CANCELLED
    batch.cancelled_at = now
    
    # Save updated batch
    save_batch(batch)
    
    return batch


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def launch_server(server_args: ServerArgs):
    app.server_args = server_args
    setup_storage(server_args.file_path)
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level="info",
        timeout_keep_alive=300,
        loop="uvloop",
    )
