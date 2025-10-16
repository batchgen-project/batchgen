"""
Pydantic models for OpenAI Files API

Reference: https://platform.openai.com/docs/api-reference/files
"""
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal
from enum import Enum


class FilePurpose(str, Enum):
    """Valid file purposes according to OpenAI API"""
    ASSISTANTS = "assistants"
    ASSISTANTS_OUTPUT = "assistants_output"
    BATCH = "batch"
    BATCH_OUTPUT = "batch_output"
    FINE_TUNE = "fine-tune"
    FINE_TUNE_RESULTS = "fine-tune-results"
    VISION = "vision"


class FileStatus(str, Enum):
    """File processing status"""
    UPLOADED = "uploaded"
    PROCESSED = "processed"
    ERROR = "error"


class FileObject(BaseModel):
    """
    The File object represents a document that has been uploaded to OpenAI.
    
    Reference: https://platform.openai.com/docs/api-reference/files/object
    """
    id: str = Field(..., description="The file identifier, which can be referenced in the API endpoints.")
    object: Literal["file"] = Field(default="file", description="The object type, which is always 'file'.")
    bytes: int = Field(..., description="The size of the file, in bytes.", ge=0)
    created_at: int = Field(..., description="The Unix timestamp (in seconds) for when the file was created.")
    filename: str = Field(..., description="The name of the file.")
    purpose: FilePurpose = Field(
        ..., 
        description="The intended purpose of the file. Supported values are 'assistants', "
                    "'assistants_output', 'batch', 'batch_output', 'fine-tune', 'fine-tune-results', and 'vision'."
    )
    status: FileStatus = Field(
        default=FileStatus.PROCESSED,
        description="Deprecated. The current status of the file, which can be either 'uploaded', 'processed', or 'error'."
    )
    status_details: Optional[str] = Field(
        default=None,
        description="Deprecated. For details on why a fine-tuning training file failed validation, see the error field on fine_tuning.job."
    )
    checksum: Optional[str] = Field(
        default=None,
        description="SHA-256 checksum of the file content for duplicate detection."
    )
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "id": "file-abc123",
                "object": "file",
                "bytes": 120000,
                "created_at": 1698107661,
                "filename": "training_data.jsonl",
                "purpose": "batch",
                "status": "processed",
                "status_details": None
            }
        }


class CreateFileRequest(BaseModel):
    """
    Request model for uploading a file to OpenAI.
    
    POST /v1/files
    Reference: https://platform.openai.com/docs/api-reference/files/create
    
    Note: In FastAPI, the actual file upload is handled via UploadFile,
    but this model represents the form data parameters.
    """
    purpose: FilePurpose = Field(
        ...,
        description="The intended purpose of the uploaded file. "
                    "Use 'assistants' for Assistants and Message files, "
                    "'vision' for Assistants image file inputs, "
                    "'batch' for Batch API, and "
                    "'fine-tune' for Fine-tuning."
    )
    
    @model_validator(mode='after')
    def cls_convert(self):
        """Ensure purpose is a valid FilePurpose enum value"""
        if isinstance(self.purpose, str):
            self.purpose = FilePurpose(self.purpose)
        return self
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "file": "@training_data.jsonl",
                "purpose": "batch"
            }
        }


class ListFilesRequest(BaseModel):
    """
    Query parameters for listing files.
    
    GET /v1/files
    Reference: https://platform.openai.com/docs/api-reference/files/list
    """
    purpose: Optional[FilePurpose] = Field(
        default=None,
        description="Only return files with the given purpose."
    )
    limit: Optional[int] = Field(
        default=10000,
        description="A limit on the number of objects to be returned. "
                    "Limit can range between 1 and 10,000, and the default is 10,000.",
        ge=1,
        le=10000
    )
    order: Optional[Literal["asc", "desc"]] = Field(
        default="desc",
        description="Sort order by the created_at timestamp of the objects. "
                    "asc for ascending order and desc for descending order."
    )
    after: Optional[str] = Field(
        default=None,
        description="A cursor for use in pagination. after is an object ID that defines your place in the list. "
                    "For instance, if you make a list request and receive 100 objects, ending with obj_foo, "
                    "your subsequent call can include after=obj_foo in order to fetch the next page of the list."
    )
    
    @model_validator(mode='after')
    def cls_convert(self):
        """Ensure purpose is a valid FilePurpose enum value"""
        if isinstance(self.purpose, str):
            self.purpose = FilePurpose(self.purpose)
        return self
    
    class Config:
        use_enum_values = True


class ListFilesResponse(BaseModel):
    """
    Response model for listing files.
    
    Reference: https://platform.openai.com/docs/api-reference/files/list
    """
    object: Literal["list"] = Field(default="list", description="The object type, which is always 'list'.")
    data: List[FileObject] = Field(..., description="List of file objects.")
    has_more: bool = Field(default=False, description="Whether there are more files available.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "id": "file-abc123",
                        "object": "file",
                        "bytes": 120000,
                        "created_at": 1698107661,
                        "filename": "training_data.jsonl",
                        "purpose": "batch",
                        "status": "processed",
                        "status_details": None
                    }
                ],
                "has_more": False
            }
        }


class DeleteFileResponse(BaseModel):
    """
    Response model for file deletion.
    
    DELETE /v1/files/{file_id}
    Reference: https://platform.openai.com/docs/api-reference/files/delete
    """
    id: str = Field(..., description="The ID of the deleted file.")
    object: Literal["file"] = Field(default="file", description="The object type, which is always 'file'.")
    deleted: bool = Field(..., description="Whether the file was successfully deleted.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "file-abc123",
                "object": "file",
                "deleted": True
            }
        }
