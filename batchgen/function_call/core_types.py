"""Core dataclasses and protocol types used by the function-call parser.

This module is intentionally self-contained so the parser can be used without
pulling in the rest of the batchgen server stack. The OpenAI-style `Tool`,
`ToolChoice`, and `Function` schemas mirror the shape used by sglang/vLLM and
align with the dicts that batchgen's `ChatCompletionRequest.tools` field
already accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OpenAI-compatible tool schema (subset)
# ---------------------------------------------------------------------------


class Function(BaseModel):
    """Function descriptor inside a `Tool` definition."""

    description: Optional[str] = None
    name: str
    parameters: Optional[Any] = None
    strict: bool = False


class Tool(BaseModel):
    """Top-level OpenAI tool definition."""

    type: Literal["function"] = Field(default="function")
    function: Function

    @classmethod
    def from_dict(cls, data: Any) -> "Tool":
        """Build a `Tool` from a plain dict or pass through if already a Tool."""
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls.model_validate(data)
        raise TypeError(f"Cannot coerce {type(data)!r} into Tool")


class ToolChoiceFuncName(BaseModel):
    name: Optional[str] = None


class ToolChoice(BaseModel):
    """Specific tool selection (e.g. `{type: function, function: {name: ...}}`)."""

    function: ToolChoiceFuncName
    type: Literal["function"] = "function"


# ---------------------------------------------------------------------------
# Parser result types
# ---------------------------------------------------------------------------


class ToolCallItem(BaseModel):
    """A single parsed tool/function call.

    `parameters` always carries a JSON string (possibly partial during streaming)
    to mirror OpenAI's streaming protocol where argument chunks are emitted as
    text deltas.
    """

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string (may be a streaming fragment)


class StreamingParseResult(BaseModel):
    """Result of a (non-)streaming parse step."""

    normal_text: str = ""
    calls: List[ToolCallItem] = []


@dataclass
class StructureInfo:
    """Begin/end/trigger markers used for grammar-constrained decoding hints."""

    begin: str
    end: str
    trigger: str


# Helper alias: `name -> StructureInfo`
_GetInfoFunc = Callable[[str], StructureInfo]


def coerce_tools(tools: Any) -> List[Tool]:
    """Best-effort coercion of a list of dicts/Tools into `List[Tool]`."""
    if not tools:
        return []
    out: List[Tool] = []
    for t in tools:
        out.append(Tool.from_dict(t))
    return out
