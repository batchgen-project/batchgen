"""
Pydantic models for batch request body structures

This module contains models for the request bodies used in batch processing.
For Files API and Batch API models, see file_schema.py and batch_schema.py
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Literal

# Request body models for batch processing
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
