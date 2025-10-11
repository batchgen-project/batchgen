import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from batchgen.managers.io_struct import (
    DeleteFileResponse,
    FileObject,
    ListFilesResponse,
)
from batchgen.server_args import ServerArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
app = FastAPI(title="OpenAI Compatible Files API")

# Storage configuration
STORAGE_PATH = None
METADATA_PATH = None


def setup_storage(path: str):
    global STORAGE_PATH, METADATA_PATH
    STORAGE_PATH = Path(path)
    METADATA_PATH = STORAGE_PATH / "metadata"
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.mkdir(parents=True, exist_ok=True)


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


@app.post("/v1/files", response_model=FileObject)
async def upload_file(file: UploadFile = File(...), purpose: str = Form(...)):
    """
    Upload a file that can be used across various endpoints.
    """
    # Validate purpose
    valid_purposes = ["batch"]
    if purpose not in valid_purposes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid purpose. Must be one of: {', '.join(valid_purposes)}",
        )

    # Generate unique file ID
    file_id = f"file-{uuid.uuid4().hex}"

    # Save file to disk
    file_path = app.server_args.file_path / file_id
    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    # Create metadata
    metadata = {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": file.filename,
        "purpose": purpose,
        "expires_at": None,
    }

    save_metadata(file_id, metadata)

    return FileObject(**metadata)


@app.get("/v1/files", response_model=ListFilesResponse)
async def list_files(purpose: Optional[str] = None):
    """
    Returns a list of files that belong to the user's organization.
    """
    all_metadata = list_all_metadata()

    # Filter by purpose if specified
    if purpose:
        all_metadata = [m for m in all_metadata if m["purpose"] == purpose]

    # Sort by created_at descending
    all_metadata.sort(key=lambda x: x["created_at"], reverse=True)

    files = [FileObject(**m) for m in all_metadata]

    return ListFilesResponse(data=files)


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
