"""
Pydantic models for batch request body structures and OpenAI-compatible schemas.

This module centralizes all IO models used by the FastAPI server, including
request/response shapes for batch execution, file management, and worker input.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """Body structure for /v1/chat/completions batch requests."""

    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(
        ..., description="Conversation messages"
    )
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0, description="Top-k filtering. None or 0 = disabled.")
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = Field(
        default=False, description="Must be false for batch requests"
    )
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # GPT-OSS reasoning effort control (low/medium/high)
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Reasoning effort level for GPT-OSS models (low, medium, high)",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of tools the model may call (OpenAI function-calling format)",
    )
    thinking: Optional[bool] = Field(
        default=None,
        description="Enable/disable thinking mode (None = model default)",
    )
    enable_thinking: Optional[bool] = Field(
        default=None,
        description=(
            "Alternate name for `thinking` (GLM/SGLang convention). "
            "If both set, `enable_thinking` takes precedence."
        ),
    )

    @validator("stream")
    def validate_stream(cls, value: Optional[bool]) -> Optional[bool]:
        if value is True:
            raise ValueError("Streaming is not supported in batch requests")
        return value


class CompletionRequest(BaseModel):
    """Body structure for /v1/completions batch requests."""

    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, List[str]] = Field(
        ..., description="Prompt(s) for completion"
    )
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0, description="Top-k filtering. None or 0 = disabled.")
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = Field(default=False)
    logprobs: Optional[int] = Field(default=None, ge=0, le=5)
    echo: Optional[bool] = Field(default=False)
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    user: Optional[str] = None

    @validator("stream")
    def validate_stream(cls, value: Optional[bool]) -> Optional[bool]:
        if value is True:
            raise ValueError("Streaming is not supported in batch requests")
        return value


class FilePurpose(str, Enum):
    BATCH = "batch"
    BATCH_OUTPUT = "batch_output"


class FileStatus(str, Enum):
    PROCESSED = "processed"
    FAILED = "failed"


class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str
    status_details: Optional[str] = None
    checksum: Optional[str] = None


class DeleteFileResponse(BaseModel):
    id: str
    deleted: bool
    object: Literal["file"] = "file"


class ListFilesRequest(BaseModel):
    purpose: Optional[str] = None
    limit: int = Field(default=10000, ge=1, le=10000)
    order: Literal["asc", "desc"] = "desc"
    after: Optional[str] = None


class ListFilesResponse(BaseModel):
    data: List[FileObject]
    has_more: bool


class BatchEndpoint(str, Enum):
    CHAT_COMPLETIONS = "/v1/chat/completions"
    COMPLETIONS = "/v1/completions"


class CompletionWindow(str, Enum):
    ONE_DAY = "24h"


class BatchStatus(str, Enum):
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class CreateBatchRequest(BaseModel):
    input_file_id: str
    endpoint: BatchEndpoint = BatchEndpoint.CHAT_COMPLETIONS
    completion_window: CompletionWindow = CompletionWindow.ONE_DAY
    metadata: Optional[Dict[str, Any]] = None
    # Inference parameters (serve as defaults when per-request values are None)
    max_decoding_length: Optional[int] = Field(default=None, ge=1)
    max_context_length: Optional[int] = Field(default=None, ge=1)  # Max total context (prompt + decode). None = use model max.
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)


class BatchObject(BaseModel):
    id: str
    object: Literal["batch"] = "batch"
    endpoint: BatchEndpoint
    input_file_id: str
    output_file_id: Optional[str] = None
    completion_window: CompletionWindow
    status: BatchStatus
    created_at: int
    expires_at: int
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Inference parameters (serve as defaults when per-request values are None)
    max_decoding_length: Optional[int] = None
    max_context_length: Optional[int] = None  # None = use model max
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    @root_validator(pre=True)
    def default_timestamps(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        created_at = values.get("created_at")
        if created_at is None:
            values["created_at"] = int(datetime.now().timestamp())
        return values


class ListBatchesRequest(BaseModel):
    after: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=1000)


class ListBatchesResponse(BaseModel):
    data: List[BatchObject]
    first_id: Optional[str]
    last_id: Optional[str]
    has_more: bool


class BatchError(BaseModel):
    code: str
    message: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatCompletionChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


BatchResponseBody = Union[ChatCompletionResponse, CompletionResponse]


class BatchResponse(BaseModel):
    status_code: int
    request_id: str
    body: BatchResponseBody


class BatchResultItem(BaseModel):
    """Result record persisted to the output file."""

    id: str
    custom_id: str
    response: Optional[BatchResponse] = None
    error: Optional[BatchError] = None


class BatchRequestItem(BaseModel):
    """Single line from an OpenAI batch file."""

    custom_id: str
    method: Literal["POST"]
    url: BatchEndpoint
    body: Union[ChatCompletionRequest, CompletionRequest]

    @validator("body", pre=True)
    def parse_body(
        cls, value: Dict[str, Any], values: Dict[str, Any]
    ) -> Union[ChatCompletionRequest, CompletionRequest]:
        url = values.get("url")
        if url == BatchEndpoint.CHAT_COMPLETIONS:
            return ChatCompletionRequest(**value)
        if url == BatchEndpoint.COMPLETIONS:
            return CompletionRequest(**value)
        raise ValueError(f"Unsupported url {url}")


class RawInferenceRequest(BaseModel):
    """Direct inference request without going through the batch file flow."""

    prompts: List[str] = Field(
        ..., min_items=1, description="List of prompt strings"
    )
    max_input_len: Optional[int] = Field(default=None, ge=1)
    max_output_len: Optional[int] = Field(default=None, ge=1)
    ignore_eos: bool = False
    # Sampling parameters (None = greedy decoding)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)

    @validator("prompts")
    def validate_prompts(cls, value: List[str]) -> List[str]:
        if not all(isinstance(p, str) and p.strip() for p in value):
            raise ValueError("All prompts must be non-empty strings")
        return value


def normalize_inference_results(results: List[Any]) -> List[Any]:
    return [_normalize_result_value(item) for item in results]


def _normalize_result_value(value: Any) -> Any:
    if (
        hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "tolist")
    ):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_normalize_result_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_result_value(val) for key, val in value.items()}
    return value


# ======================== Model Metadata ========================


class ModelObject(BaseModel):
    """OpenAI-compatible model object with BatchGen extensions."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "batchgen"
    max_context_length: int = Field(
        description="Maximum context length (prompt + completion tokens)"
    )


class ListModelsResponse(BaseModel):
    """OpenAI-compatible response for GET /v1/models."""

    object: Literal["list"] = "list"
    data: List[ModelObject]
