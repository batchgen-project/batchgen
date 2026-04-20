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

"""MiniMax-M2.5 tokenizer for BatchGen.

MiniMax-M2.5 tokenizer specifications:
- Vocabulary size: 200,064 tokens
- Uses HuggingFace tokenizer.json format
- tokenizer.json is loaded from the converted checkpoint directory

Tokenizer assets are copied into the converted checkpoint directory during
BatchGen checkpoint conversion.
"""

import json
import logging
import re
import uuid
from importlib.resources import as_file, files
from pathlib import Path
from typing import Optional

from batchgen.config.fast_tokenizer import FastTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer

logger = logging.getLogger(__name__)

# MiniMax-M2.5 special token IDs (from tokenizer_config.json added_tokens_decoder)
MINIMAX_M25_BOS_TOKEN_ID = 200034
MINIMAX_M25_EOS_TOKEN_ID = 200020
MINIMAX_M25_VOCAB_SIZE = 200064


@register_tokenizer("minimax_m25")
class MiniMaxM25Tokenizer(FastTokenizer):
    """MiniMax-M2.5 tokenizer.

    Loads tokenizer.json from the converted checkpoint directory.

    Attributes:
        bos_token_id: BOS token ID
        eos_token_id: EOS token ID
        pad_token_id: PAD token ID (uses EOS)
        vocab_size: 200,064
    """

    def __init__(self, tokenizer_path: str | Path):
        """Initialize the MiniMax-M2.5 tokenizer."""
        super().__init__(str(tokenizer_path))

        self.bos_token_id = MINIMAX_M25_BOS_TOKEN_ID
        self.eos_token_id = MINIMAX_M25_EOS_TOKEN_ID
        self.pad_token_id = MINIMAX_M25_EOS_TOKEN_ID  # Use EOS as pad
        self.vocab_size = MINIMAX_M25_VOCAB_SIZE

        # Get the actual token strings from vocabulary for padding setup
        vocab = self.tokenizer.get_vocab()
        self.bos_token = None
        self.eos_token = None
        self.pad_token = None

        id_to_token = {v: k for k, v in vocab.items()}
        if self.bos_token_id in id_to_token:
            self.bos_token = id_to_token[self.bos_token_id]
        if self.eos_token_id in id_to_token:
            self.eos_token = id_to_token[self.eos_token_id]
            self.pad_token = self.eos_token

        if self.pad_token is not None:
            self.tokenizer.enable_padding(
                direction="right",
                pad_id=self.pad_token_id,
                pad_token=self.pad_token,
            )

        # chat_template.jinja remains packaged with BatchGen; only tokenizer JSONs move.
        chat_template_resource = files(__package__) / "chat_template.jinja"
        with as_file(chat_template_resource) as chat_template_path:
            if chat_template_path.exists():
                self.chat_template = chat_template_path.read_text()
                logger.info("Loaded chat template from packaged chat_template.jinja")

        logger.info(
            f"MiniMax-M2.5 tokenizer initialized: vocab_size={self.vocab_size}, "
            f"bos={self.bos_token_id}, eos={self.eos_token_id}"
        )

    # ---- Output parsing ----

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _TOOL_CALL_RE = re.compile(
        r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
    )
    _INVOKE_RE = re.compile(
        r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL
    )
    _PARAM_RE = re.compile(
        r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL
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
            for inv in self._INVOKE_RE.finditer(raw):
                name = inv.group(1)
                body = inv.group(2)
                arguments = {}
                for pm in self._PARAM_RE.finditer(body):
                    key = pm.group(1).strip()
                    val = pm.group(2).strip()
                    try:
                        arguments[key] = json.loads(val)
                    except (ValueError, json.JSONDecodeError):
                        arguments[key] = val
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                })
        visible = self._TOOL_CALL_RE.sub("", text).strip()
        return tool_calls or None, visible
