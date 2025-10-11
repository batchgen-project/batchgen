from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Optional, Dict, Literal
from enum import Enum

class ChatCompletionRequest(BaseModel):
    """Body structure for /v1/chat/completions batch requests"""
    
    model: str = Field(..., description="ID of the model to use")
    messages: List[Dict[str, str]] = Field(..., description="List of messages comprising the conversation")
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = Field(default=False, description="Must be false for batch requests")
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    
    @validator('stream')
    def validate_stream(cls, v):
        """Stream must be false for batch requests"""
        if v is True:
            raise ValueError("Streaming is not supported in batch requests")
        return v


class EmbeddingRequest(BaseModel):
    """Body structure for /v1/embeddings batch requests"""
    
    model: str = Field(..., description="ID of the model to use")
    input: str | List[str] = Field(
        ..., 
        description="Input text to embed, encoded as a string or array of strings"
    )
    encoding_format: Optional[Literal["float", "base64"]] = Field(default="float")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions for the embedding")
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    """Body structure for /v1/completions batch requests"""
    
    model: str = Field(..., description="ID of the model to use")
    prompt: str | List[str] = Field(..., description="The prompt(s) to generate completions for")
    max_tokens: Optional[int] = Field(default=16, ge=1)
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = Field(default=False)
    logprobs: Optional[int] = Field(default=None, ge=0, le=5)
    echo: Optional[bool] = Field(default=False)
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    user: Optional[str] = None
    
    @validator('stream')
    def validate_stream(cls, v):
        """Stream must be false for batch requests"""
        if v is True:
            raise ValueError("Streaming is not supported in batch requests")
        return v

# ============= Pydantic Models for OpenAI File API =============

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
    
    class Config:
        use_enum_values = True
        schema_extra = {
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
        schema_extra = {
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
        schema_extra = {
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
        schema_extra = {
            "example": {
                "id": "file-abc123",
                "object": "file",
                "deleted": True
            }
        }

# ============= Pydantic Models for OpenAI Batch API =============

class BatchStatus(str, Enum):
    """Batch processing status"""
    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    
class BatchEndpoint(str, Enum):
    """Supported batch endpoints"""
    CHAT_COMPLETIONS = "/v1/chat/completions"
    COMPLETIONS = "/v1/completions"
    EMBEDDINGS = "/v1/embeddings"


class CompletionWindow(str, Enum):
    """Time window for batch completion"""
    HOURS_24 = "24h"


class RequestCounts(BaseModel):
    """Request count statistics for a batch"""
    total: int = Field(default=0, description="Total number of requests in the batch.")
    completed: int = Field(default=0, description="Number of requests that have been completed successfully.")
    failed: int = Field(default=0, description="Number of requests that have failed.")


class BatchErrors(BaseModel):
    """Error information for failed batches"""
    object: Optional[str] = Field(default=None, description="The object type, always 'list'.")
    data: Optional[List[Dict]] = Field(default=None, description="List of error objects.")


class BatchObject(BaseModel):
    """
    The Batch object represents a batch of API requests.
    
    Reference: https://platform.openai.com/docs/api-reference/batch/object
    """
    id: str = Field(..., description="The batch identifier.")
    object: Literal["batch"] = Field(default="batch", description="The object type, which is always 'batch'.")
    endpoint: str = Field(..., description="The OpenAI API endpoint used by the batch.")
    errors: Optional[BatchErrors] = Field(default=None, description="Error details if the batch has failed.")
    input_file_id: str = Field(..., description="The ID of the input file for the batch.")
    completion_window: str = Field(..., description="The time frame within which the batch should be processed.")
    status: BatchStatus = Field(..., description="The current status of the batch.")
    output_file_id: Optional[str] = Field(default=None, description="The ID of the file containing the outputs of successfully executed requests.")
    error_file_id: Optional[str] = Field(default=None, description="The ID of the file containing the outputs of requests with errors.")
    created_at: int = Field(..., description="The Unix timestamp (in seconds) for when the batch was created.")
    in_progress_at: Optional[int] = Field(default=None, description="The Unix timestamp (in seconds) for when the batch started processing.")
    expires_at: Optional[int] = Field(default=None, description="The Unix timestamp (in seconds) for when the batch will expire.")
    finalizing_at: Optional[int] = Field(default=None, description="The Unix timestamp (in seconds) for when the batch started finalizing.")
    completed_at: Optional[int] = Field(default=None, description="The Unix timestamp (in seconds) for when the batch was completed.")
    failed_at: Optional[int] = Field(default=None, description="The Unix timestamp (in seconds) for when the batch failed.")
    expired_at: Optional[int] = Field(default=None, description="The Unix timestamp (in seconds) for when the batch expired.")
    cancelling_at: Optional[int] = Field(default=None, description="The Unix timestamp (in seconds) for when the batch started cancelling.")
    cancelled_at: Optional[int] = Field(default=None, description="The Unix timestamp (in seconds) for when the batch was cancelled.")
    request_counts: RequestCounts = Field(default_factory=RequestCounts, description="The request counts for different statuses within the batch.")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Set of 16 key-value pairs that can be attached to an object.")
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "id": "batch_abc123",
                "object": "batch",
                "endpoint": "/v1/chat/completions",
                "errors": None,
                "input_file_id": "file-abc123",
                "completion_window": "24h",
                "status": "completed",
                "output_file_id": "file-xyz789",
                "error_file_id": None,
                "created_at": 1711471533,
                "in_progress_at": 1711471538,
                "expires_at": 1711557933,
                "finalizing_at": 1711493133,
                "completed_at": 1711493163,
                "failed_at": None,
                "expired_at": None,
                "cancelling_at": None,
                "cancelled_at": None,
                "request_counts": {
                    "total": 100,
                    "completed": 95,
                    "failed": 5
                },
                "metadata": {
                    "customer_id": "user_123456789",
                    "batch_description": "Nightly eval job"
                }
            }
        }
    
class CreateBatchRequest(BaseModel):
    """
    Request model for creating a batch.
    
    POST /v1/batches
    https://platform.openai.com/docs/api-reference/batch/create
    """
    
    input_file_id: str = Field(
        ...,
        description="The ID of an uploaded file that contains requests for the new batch. "
                    "Your input file must be formatted as a JSONL file, and must be uploaded "
                    "with the purpose 'batch'. The file can contain up to 50,000 requests, "
                    "and can be up to 100 MB in size."
    )
    
    endpoint: BatchEndpoint = Field(
        ...,
        description="The endpoint to be used for all requests in the batch. "
                    "Currently /v1/chat/completions, /v1/embeddings, and /v1/completions are supported. "
                    "Note that /v1/embeddings batches are also restricted to a maximum of 50,000 "
                    "embedding inputs across all requests in the batch."
    )
    
    completion_window: CompletionWindow = Field(
        ...,
        description="The time frame within which the batch should be processed. "
                    "Currently only '24h' is supported."
    )
    
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional custom metadata for the batch. "
                    "You can add up to 16 key-value pairs, where keys are strings with a maximum "
                    "length of 64 characters, and values are strings with a maximum length of 512 characters."
    )
    
    # post-init to convert str to Enum
    @model_validator(mode='after')
    def cls_convert(self):
        if isinstance(self.endpoint, str):
            self.endpoint = BatchEndpoint(self.endpoint)
        if isinstance(self.completion_window, str):
            self.completion_window = CompletionWindow(self.completion_window)
        return self
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata constraints"""
        if v is None:
            return v
        
        # Maximum 16 key-value pairs
        if len(v) > 16:
            raise ValueError(
                f"Metadata can contain at most 16 key-value pairs, got {len(v)}"
            )
        
        # Validate key and value lengths
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key).__name__}")
            
            if not isinstance(value, str):
                raise ValueError(f"Metadata value must be a string, got {type(value).__name__}")
            
            if len(key) > 64:
                raise ValueError(
                    f"Metadata key '{key}' exceeds maximum length of 64 characters (length: {len(key)})"
                )
            
            if len(value) > 512:
                raise ValueError(
                    f"Metadata value for key '{key}' exceeds maximum length of 512 characters (length: {len(value)})"
                )
        
        return v
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "input_file_id": "file-abc123",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {
                    "customer_id": "user_123456789",
                    "batch_description": "Nightly eval job"
                }
            }
        }


class ListBatchesRequest(BaseModel):
    """
    Query parameters for listing batches.
    
    GET /v1/batches
    Reference: https://platform.openai.com/docs/api-reference/batch/list
    """
    after: Optional[str] = Field(
        default=None,
        description="A cursor for use in pagination. after is an object ID that defines your place in the list. "
                    "For instance, if you make a list request and receive 100 objects, ending with obj_foo, "
                    "your subsequent call can include after=obj_foo in order to fetch the next page of the list."
    )
    limit: Optional[int] = Field(
        default=20,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.",
        ge=1,
        le=100
    )
    
    class Config:
        schema_extra = {
            "example": {
                "limit": 20,
                "after": "batch_abc123"
            }
        }


class ListBatchesResponse(BaseModel):
    """
    Response model for listing batches.
    
    Reference: https://platform.openai.com/docs/api-reference/batch/list
    """
    object: Literal["list"] = Field(default="list", description="The object type, which is always 'list'.")
    data: List[BatchObject] = Field(..., description="List of batch objects.")
    first_id: Optional[str] = Field(default=None, description="The ID of the first batch in the list.")
    last_id: Optional[str] = Field(default=None, description="The ID of the last batch in the list.")
    has_more: bool = Field(default=False, description="Whether there are more batches available.")
    
    class Config:
        schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "id": "batch_abc123",
                        "object": "batch",
                        "endpoint": "/v1/chat/completions",
                        "errors": None,
                        "input_file_id": "file-abc123",
                        "completion_window": "24h",
                        "status": "completed",
                        "output_file_id": "file-xyz789",
                        "error_file_id": None,
                        "created_at": 1711471533,
                        "in_progress_at": 1711471538,
                        "expires_at": 1711557933,
                        "finalizing_at": 1711493133,
                        "completed_at": 1711493163,
                        "failed_at": None,
                        "expired_at": None,
                        "cancelling_at": None,
                        "cancelled_at": None,
                        "request_counts": {
                            "total": 100,
                            "completed": 95,
                            "failed": 5
                        },
                        "metadata": None
                    }
                ],
                "first_id": "batch_abc123",
                "last_id": "batch_abc456",
                "has_more": False
            }
        }


class RetrieveBatchResponse(BaseModel):
    """
    Response model for retrieving a specific batch.
    
    GET /v1/batches/{batch_id}
    Reference: https://platform.openai.com/docs/api-reference/batch/retrieve
    
    Path Parameters:
        batch_id (str): The ID of the batch to retrieve
    
    Returns the Batch object matching the specified ID.
    """
    # This returns a BatchObject directly
    # No request body needed - batch_id comes from URL path
    pass  # Marker class - actual response uses BatchObject directly


class CancelBatchResponse(BaseModel):
    """
    Response model for cancelling a batch.
    
    POST /v1/batches/{batch_id}/cancel
    Reference: https://platform.openai.com/docs/api-reference/batch/cancel
    
    Path Parameters:
        batch_id (str): The ID of the batch to cancel
    
    Returns the Batch object with status updated to 'cancelling' or 'cancelled'.
    Note: You can only cancel batches with status 'validating' or 'in_progress'.
    """
    # This is just a BatchObject with status updated to 'cancelling' or 'cancelled'
    # We can reuse BatchObject for the response
    pass  # Marker class - actual response uses BatchObject directly
