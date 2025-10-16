"""
Pydantic models for OpenAI Batch API

Reference: https://platform.openai.com/docs/api-reference/batch
"""
from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Optional, Dict, Literal
from enum import Enum


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
        json_schema_extra = {
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
                    "Currently /v1/chat/completions and /v1/completions are supported."
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
    
    @model_validator(mode='after')
    def cls_convert(self):
        """Convert string values to Enum types"""
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
        json_schema_extra = {
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
        json_schema_extra = {
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
        json_schema_extra = {
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
