from pydantic import BaseModel, Field
from typing import List, Optional

# ============= Pydantic Models for File API =============

class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str = "processed"
    status_details: Optional[str] = None


class ListFilesResponse(BaseModel):
    object: str = "list"
    data: List[FileObject]


class DeleteFileResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool


# HTTP Server for receiving requests and send back responses
class BatchGenerateRequest(BaseModel):
    input_texts: list[str] = Field(..., description="List of input texts to generate from")
    max_new_tokens: int = Field(..., description="Maximum number of new tokens to generate")

class BatchGenerateResponse(BaseModel):
    generated_texts: list[str] = Field(..., description="List of generated texts corresponding to input_texts")
    status: str = Field(..., description="Status of the generation request")
    message: str | None = Field(None, description="Optional message providing additional information")