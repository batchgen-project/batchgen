"""Format-aware tool/function-call parser registry.

Ported from sglang's `function_call.function_call_parser`. The xgrammar
structural-tag / strict-mode plumbing has been omitted; batchgen only needs the
extract-from-text functionality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Type, Union

from batchgen.function_call.base_format_detector import BaseFormatDetector
from batchgen.function_call.core_types import Tool, ToolCallItem, coerce_tools
from batchgen.function_call.deepseekv31_detector import DeepSeekV31Detector
from batchgen.function_call.deepseekv32_detector import DeepSeekV32Detector
from batchgen.function_call.deepseekv3_detector import DeepSeekV3Detector
from batchgen.function_call.deepseekv4_detector import DeepSeekV4Detector
from batchgen.function_call.glm47_moe_detector import Glm47MoeDetector
from batchgen.function_call.glm4_moe_detector import Glm4MoeDetector
from batchgen.function_call.gpt_oss_detector import GptOssDetector
from batchgen.function_call.hermes_detector import HermesDetector
from batchgen.function_call.kimik2_detector import KimiK2Detector
from batchgen.function_call.llama32_detector import Llama32Detector
from batchgen.function_call.minimax_m2 import MinimaxM2Detector
from batchgen.function_call.mistral_detector import MistralDetector
from batchgen.function_call.pythonic_detector import PythonicDetector
from batchgen.function_call.qwen25_detector import Qwen25Detector
from batchgen.function_call.qwen3_coder_detector import Qwen3CoderDetector

logger = logging.getLogger(__name__)


ToolsArg = Union[List[Tool], List[Dict[str, Any]], None]


class FunctionCallParser:
    """Wraps a format-specific detector and exposes streaming + one-shot APIs.

    Example
    -------
    >>> parser = FunctionCallParser(tools=req.tools, tool_call_parser="qwen25")
    >>> normal_text, calls = parser.parse_non_stream(model_output)
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "deepseekv32": DeepSeekV32Detector,
        "deepseekv4": DeepSeekV4Detector,
        "glm": Glm4MoeDetector,
        "glm45": Glm4MoeDetector,
        "glm47": Glm47MoeDetector,
        "gpt-oss": GptOssDetector,
        "hermes": HermesDetector,
        "kimi_k2": KimiK2Detector,
        "llama3": Llama32Detector,
        "minimax-m2": MinimaxM2Detector,
        "mistral": MistralDetector,
        "pythonic": PythonicDetector,
        "qwen": Qwen25Detector,
        "qwen25": Qwen25Detector,
        "qwen3_coder": Qwen3CoderDetector,
    }

    def __init__(self, tools: ToolsArg, tool_call_parser: str) -> None:
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class is None:
            raise ValueError(
                f"Unsupported tool_call_parser: {tool_call_parser!r}. "
                f"Available: {sorted(self.ToolCallParserEnum)}"
            )
        self.detector: BaseFormatDetector = detector_class()
        self.tools: List[Tool] = coerce_tools(tools)

    def has_tool_call(self, text: str) -> bool:
        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> Tuple[str, List[ToolCallItem]]:
        if not self.tools:
            return full_text, []
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        if parsed_result.calls:
            return parsed_result.normal_text, parsed_result.calls
        return full_text, []

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, List[ToolCallItem]]:
        if not self.tools:
            return chunk_text, []
        sp = self.detector.parse_streaming_increment(chunk_text, self.tools)
        return sp.normal_text or "", list(sp.calls)
