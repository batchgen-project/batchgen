# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4 tokenizer for BatchGen."""

import logging
import re
import uuid
from typing import Optional

from batchgen.config.fast_tokenizer import FastTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer
from batchgen.models.deepseek.deepseekv3.tokenizer import TOKENIZER_DIR as DEEPSEEK_V3_TOKENIZER_DIR

logger = logging.getLogger(__name__)

TOKENIZER_DIR = DEEPSEEK_V3_TOKENIZER_DIR

DEEPSEEK_V4_BOS_TOKEN_ID = 0
DEEPSEEK_V4_EOS_TOKEN_ID = 1
DEEPSEEK_V4_VOCAB_SIZE = 129280


@register_tokenizer("deepseek_v4")
class DeepSeekV4Tokenizer(FastTokenizer):
    """DeepSeek-V4 tokenizer loaded from the vendored V4 Flash assets."""

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _TOOL_CALL_RE = re.compile(
        r"<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>", re.DOTALL
    )

    def __init__(self):
        super().__init__(str(TOKENIZER_DIR))
        self.bos_token_id = DEEPSEEK_V4_BOS_TOKEN_ID
        self.eos_token_id = DEEPSEEK_V4_EOS_TOKEN_ID
        self.pad_token_id = DEEPSEEK_V4_EOS_TOKEN_ID
        self.vocab_size = DEEPSEEK_V4_VOCAB_SIZE

        vocab = self.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        self.bos_token = id_to_token.get(self.bos_token_id)
        self.eos_token = id_to_token.get(self.eos_token_id)
        self.pad_token = self.eos_token
        if self.pad_token is not None:
            self.tokenizer.enable_padding(
                direction="right",
                pad_id=self.pad_token_id,
                pad_token=self.pad_token,
            )

        logger.info(
            "DeepSeek-V4 tokenizer initialized: vocab_size=%s, bos=%s, eos=%s",
            self.vocab_size,
            self.bos_token_id,
            self.eos_token_id,
        )

    def parse_thinking(self, text: str) -> tuple[Optional[str], str]:
        match = self._THINK_RE.search(text)
        if not match:
            return None, text
        reasoning = match.group(1).strip()
        visible = self._THINK_RE.sub("", text, count=1).strip()
        return reasoning, visible

    def parse_tool_calls(self, text: str) -> tuple[Optional[list], str]:
        matches = self._TOOL_CALL_RE.findall(text)
        if not matches:
            return None, text
        tool_calls = []
        for raw in matches:
            lines = raw.strip().split("\n", 1)
            name = lines[0].strip()
            arguments = lines[1].strip() if len(lines) > 1 else "{}"
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
        visible = self._TOOL_CALL_RE.sub("", text).strip()
        return tool_calls, visible
