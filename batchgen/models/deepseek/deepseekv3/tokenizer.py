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

"""DeepSeek-V3/R1 tokenizer for BatchGen.

This module provides the tokenizer implementation for DeepSeek-V3 and DeepSeek-R1
models, which use the same tokenizer architecture.

DeepSeek tokenizer specifications:
- Vocabulary size: 129,280 tokens
- BOS token: <｜begin▁of▁sentence｜> (ID: 0)
- EOS token: <｜end▁of▁sentence｜> (ID: 1)
- Uses HuggingFace tokenizer.json format

The tokenizer.json file is loaded from the converted checkpoint directory
provided by the BatchGen runtime.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from batchgen.config.fast_tokenizer import FastTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer

logger = logging.getLogger(__name__)


# DeepSeek-V3/R1 special token IDs
DEEPSEEK_V3_BOS_TOKEN_ID = 0  # <｜begin▁of▁sentence｜>
DEEPSEEK_V3_EOS_TOKEN_ID = 1  # <｜end▁of▁sentence｜>
DEEPSEEK_V3_VOCAB_SIZE = 129280


@register_tokenizer("deepseek_v3")
class DeepSeekV3Tokenizer(FastTokenizer):
    """DeepSeek-V3/R1 tokenizer.

    Loads tokenizer.json from the converted checkpoint directory.

    Attributes:
        bos_token_id: 0 (<｜begin▁of▁sentence｜>)
        eos_token_id: 1 (<｜end▁of▁sentence｜>)
        pad_token_id: 1 (uses EOS as pad token)
        vocab_size: 129,280
    """

    def __init__(self, tokenizer_path: str | Path):
        """Initialize the DeepSeek-V3 tokenizer.

        Loads tokenizer.json from the converted checkpoint directory.
        """
        super().__init__(str(tokenizer_path))

        # Override with DeepSeek-specific special token IDs
        # These are the correct values for DeepSeek-V3/R1 models
        self.bos_token_id = DEEPSEEK_V3_BOS_TOKEN_ID
        self.eos_token_id = DEEPSEEK_V3_EOS_TOKEN_ID
        self.pad_token_id = DEEPSEEK_V3_EOS_TOKEN_ID  # Use EOS as pad token
        self.vocab_size = DEEPSEEK_V3_VOCAB_SIZE

        # Get the actual token strings from vocabulary for padding setup
        vocab = self.tokenizer.get_vocab()
        self.bos_token = None
        self.eos_token = None
        self.pad_token = None

        # Find tokens by ID
        id_to_token = {v: k for k, v in vocab.items()}
        if self.bos_token_id in id_to_token:
            self.bos_token = id_to_token[self.bos_token_id]
        if self.eos_token_id in id_to_token:
            self.eos_token = id_to_token[self.eos_token_id]
            self.pad_token = self.eos_token  # Use EOS as pad

        # Re-enable padding with correct pad token
        if self.pad_token is not None:
            self.tokenizer.enable_padding(
                direction="right",
                pad_id=self.pad_token_id,
                pad_token=self.pad_token,
            )

        logger.info(
            f"DeepSeek-V3 tokenizer initialized: vocab_size={self.vocab_size}, "
            f"bos={self.bos_token_id}, eos={self.eos_token_id}"
        )

    # ---- Output parsing ----

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _TOOL_CALL_RE = re.compile(
        r"<｜tool▁call▁begin｜>(.*?)<｜tool▁call▁end｜>", re.DOTALL
    )

    def parse_thinking(self, text: str) -> tuple[Optional[str], str]:
        m = self._THINK_RE.search(text)
        if not m:
            return None, text
        reasoning = m.group(1).strip()
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
