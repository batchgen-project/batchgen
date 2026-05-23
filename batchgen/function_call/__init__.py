"""Tool/function-call parsers for batchgen, ported from sglang.

Public entry point is :class:`FunctionCallParser`.
"""

from batchgen.function_call.core_types import (
    Function,
    StreamingParseResult,
    StructureInfo,
    Tool,
    ToolCallItem,
    ToolChoice,
)
from batchgen.function_call.function_call_parser import FunctionCallParser

__all__ = [
    "FunctionCallParser",
    "Tool",
    "ToolChoice",
    "Function",
    "ToolCallItem",
    "StreamingParseResult",
    "StructureInfo",
]
