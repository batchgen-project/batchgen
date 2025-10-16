import asyncio
import json
import os
import time
import uuid
import hashlib
import logging
from pathlib import Path
import multiprocessing as mp
from typing import List, Optional
from datetime import datetime
from functools import partial
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from batchgen.managers.file_schema import (
    DeleteFileResponse,
    FileObject,
    ListFilesResponse,
    FilePurpose,
    FileStatus,
    ListFilesRequest,
)
from batchgen.managers.batch_schema import (
    BatchObject,
    BatchStatus,
    CreateBatchRequest,
    ListBatchesRequest,
    ListBatchesResponse,
)
from batchgen.server_args import ServerArgs
from batchgen.managers.scheduler import ServerScheduler
from batchgen.managers.storage import StorageManager

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Global rank tracking
CURRENT_RANK = 0

# Global scheduler instance
# SCHEDULER = None

# Global storage instance
STORAGE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    # load parameter, return a shm_name
    # allocate kv cache
    
    # create a seperate process to handle batch tasks
    stop_event = mp.Event()
    batch_process = mp.Process(target=partial(ServerScheduler(app.server_args), stop_event=stop_event), daemon=True)
    batch_process.start()

    # init global ServerScheduler
    global STORAGE
    
    # logging.info(f"Initializing ServerScheduler (rank={CURRENT_RANK}, nnodes={app.server_args.nnodes})")
    # SCHEDULER = ServerScheduler(app.server_args)
    # logging.info("ServerScheduler initialized successfully")

    # init global StorageManager
    STORAGE = StorageManager(app.server_args)
    logging.info("StorageManager initialized successfully")

    yield
    
    # Shutdown code
    # free kv cache
    # free parameter
    # free global ServerScheduler
    stop_event.set()
    batch_process.join()

app = FastAPI(title="OpenAI Compatible API", lifespan=lifespan)


def validate_batch_file_content(content: bytes, purpose: str) -> tuple[bool, Optional[str]]:
    """
    Validate batch file content for required fields and consistency.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if purpose != "batch":
        # Only validate batch purpose files
        return True, None
    
    try:
        lines = content.decode('utf-8').strip().split('\n')
    except UnicodeDecodeError:
        return False, "File must be UTF-8 encoded"
    
    if not lines or (len(lines) == 1 and not lines[0].strip()):
        return False, "Batch file cannot be empty"
    
    max_tokens_value = None
    model_value = None
    required_fields = ["custom_id", "method", "url", "body"]
    
    for line_num, line in enumerate(lines, start=1):
        if not line.strip():
            continue
            
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            return False, f"Line {line_num}: Invalid JSON - {str(e)}"
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in request]
        if missing_fields:
            return False, f"Line {line_num}: Missing required fields: {', '.join(missing_fields)}"
        
        # Validate body exists and is a dict
        if not isinstance(request.get("body"), dict):
            return False, f"Line {line_num}: 'body' must be an object"
        
        body = request["body"]
        
        # Check if model exists in body
        if "model" not in body:
            return False, f"Line {line_num}: Missing 'model' in request body"
        
        current_model = body["model"]
        
        # Validate model is a string
        if not isinstance(current_model, str) or not current_model.strip():
            return False, f"Line {line_num}: 'model' must be a non-empty string, got {current_model}"
        
        # Check consistency of model across all requests
        if model_value is None:
            model_value = current_model
        elif model_value != current_model:
            return False, f"Line {line_num}: Inconsistent model value. Expected '{model_value}', got '{current_model}'. All requests must have the same model value."
        
        # Check if max_tokens exists in body
        if "max_tokens" not in body:
            return False, f"Line {line_num}: Missing 'max_tokens' in request body"
        
        current_max_tokens = body["max_tokens"]
        
        # Validate max_tokens is a positive integer
        if not isinstance(current_max_tokens, int) or current_max_tokens <= 0:
            return False, f"Line {line_num}: 'max_tokens' must be a positive integer, got {current_max_tokens}"
        
        # Check consistency of max_tokens across all requests
        if max_tokens_value is None:
            max_tokens_value = current_max_tokens
        elif max_tokens_value != current_max_tokens:
            return False, f"Line {line_num}: Inconsistent max_tokens value. Expected {max_tokens_value}, got {current_max_tokens}. All requests must have the same max_tokens value."
        
        # Validate method
        if request["method"] not in ["POST", "GET", "PUT", "DELETE", "PATCH"]:
            return False, f"Line {line_num}: Invalid method '{request['method']}'"
        
        # Validate url format
        if not isinstance(request["url"], str) or not request["url"].startswith("/"):
            return False, f"Line {line_num}: 'url' must be a string starting with '/'"
    
    return True, None

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

    # Read file content
    content = await file.read()
    
    # Validate batch file content
    is_valid, error_message = validate_batch_file_content(content, file_purpose.value)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid batch file: {error_message}"
        )
    
    # Calculate SHA-256 checksum
    checksum = hashlib.sha256(content).hexdigest()
    
    # Check for duplicate file
    existing_file = STORAGE.find_file_by_checksum(checksum)
    if existing_file:
        # Return existing file metadata if duplicate is found
        return FileObject(**existing_file)

    # Generate unique file ID
    file_id = f"file-{uuid.uuid4().hex}"

    # Save file to disk
    file_path = app.server_args.file_path / file_id

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
        "checksum": checksum,
    }

    STORAGE.save_metadata(file_id, metadata)

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
    
    all_metadata = STORAGE.list_all_metadata()

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
    metadata = STORAGE.load_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    return FileObject(**metadata)


@app.delete("/v1/files/{file_id}", response_model=DeleteFileResponse)
async def delete_file(file_id: str):
    """
    Delete a file.
    """
    metadata = STORAGE.load_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    # Delete actual file
    file_path = STORAGE.storage_path / file_id
    if file_path.exists():
        file_path.unlink()

    # Delete metadata
    STORAGE.delete_file_metadata(file_id)

    return DeleteFileResponse(id=file_id, deleted=True)


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    """
    Returns the contents of the specified file.
    """
    metadata = STORAGE.load_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = STORAGE.storage_path / file_id

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
    input_file_metadata = STORAGE.load_metadata(request.input_file_id)
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
    
    # Check if the file already has an active batch
    existing_batch = STORAGE.get_active_batch_for_file(request.input_file_id)
    if existing_batch:
        raise HTTPException(
            status_code=400,
            detail=f"File '{request.input_file_id}' is already associated with an active batch '{existing_batch.id}' "
                   f"with status '{existing_batch.status}'. Please wait for it to complete or cancel it before creating a new batch."
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
    STORAGE.save_batch(batch)
    
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
    all_batches = STORAGE.list_all_batches()
    
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
    batch = STORAGE.load_batch(batch_id)
    
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
    batch = STORAGE.load_batch(batch_id)
    
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
    STORAGE.save_batch(batch)
    
    return batch


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rank": CURRENT_RANK,
        "api_enabled": CURRENT_RANK == 0
    }


@app.get("/rank")
async def get_rank():
    """Get current rank information."""
    return {
        "rank": CURRENT_RANK,
    }

def launch_server(server_args: ServerArgs, rank: int = None):
    """
    Launch the HTTP server.
    
    Args:
        server_args: Server configuration arguments
        rank: Process rank (0 to world_size-1). If None, uses server_args.node_rank.
              Only rank 0 can handle API requests.
    """
    
    
    global CURRENT_RANK
    # Use provided rank or fall back to node_rank from server_args
    if rank is None:
        rank = server_args.node_rank
    CURRENT_RANK = rank
    
    # Set up server args before lifespan (needed for scheduler initialization)
    # Storage setup is now handled by the scheduler during lifespan startup
    app.server_args = server_args
    
    # # Log configuration
    # logging.info(f"Starting server with rank={rank}, nnodes={server_args.nnodes}")
    # if server_args.nnodes > 1:
    #     logging.info(f"Broadcaster endpoint: {server_args.get_broadcaster_endpoint()}")
    
    # # Only rank 0 should actually start the HTTP server
    # if rank != 0:
    #     logging.warning(f"Rank {rank}: HTTP server is disabled. Only rank 0 can serve API requests.")
        
    #     # For non-zero ranks in multi-node setup, still need to participate in broadcaster
    #     if server_args.nnodes > 1:
    #         logging.info(f"Rank {rank}: Initializing as broadcaster subscriber")
    #         # The broadcaster will be initialized in the lifespan context
    #         # Keep process alive to participate in broadcasts
    #         import signal
    #         signal.pause()  # Wait indefinitely
    #     return
    
    # Rank 0 starts the HTTP server
    logging.info(f"Rank {rank}: Starting HTTP server on {server_args.listen_ip}:{server_args.listen_port}")
    uvicorn.run(
        app,
        host=server_args.listen_ip,
        port=server_args.listen_port,
        log_level="info",
        timeout_keep_alive=300,
        loop="uvloop",
    )
