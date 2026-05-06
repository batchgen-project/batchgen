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

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from batchgen.config.fast_tokenizer import FastTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer
from batchgen.models.deepseek.deepseekv4_flash.assets.encoding.encoding_dsv4 import (
    encode_messages,
    parse_message_from_completion_text,
)

logger = logging.getLogger(__name__)

TOKENIZER_DIR = Path(__file__).parent / "assets"

DEEPSEEK_V4_BOS_TOKEN_ID = 0
DEEPSEEK_V4_EOS_TOKEN_ID = 1
DEEPSEEK_V4_VOCAB_SIZE = 129280


@register_tokenizer("deepseek_v4")
class DeepSeekV4Tokenizer(FastTokenizer):
    """DeepSeek-V4 tokenizer loaded from the vendored V4 Flash assets."""

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _TOOL_CALL_RE = re.compile(
        r"<｜DSML｜invoke name=\"(?P<name>[^\"]+)\">\n(?P<args>.*?)\n</｜DSML｜invoke>",
        re.DOTALL,
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

    def apply_chat_template(
        self,
        messages: List[Dict[str, object]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[str, List[int]]:
        if not add_generation_prompt:
            logger.debug(
                "DeepSeek-V4 encoding always follows the V4 message transition "
                "rules from assets/encoding; add_generation_prompt=False was ignored."
            )

        rendered_messages = [dict(message) for message in messages]
        tools = kwargs.get("tools")
        if tools is not None and not any("tools" in m for m in rendered_messages):
            if rendered_messages and rendered_messages[0].get("role") == "system":
                rendered_messages[0]["tools"] = tools
            else:
                rendered_messages.insert(0, {"role": "system", "content": "", "tools": tools})

        response_format = kwargs.get("response_format")
        if response_format is not None and not any(
            "response_format" in m for m in rendered_messages
        ):
            if rendered_messages and rendered_messages[0].get("role") == "system":
                rendered_messages[0]["response_format"] = response_format
            else:
                rendered_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": "",
                        "response_format": response_format,
                    },
                )

        enable_thinking = kwargs.get("enable_thinking", kwargs.get("thinking", False))
        thinking_mode = "thinking" if enable_thinking else "chat"
        preserve_thinking = kwargs.get("preserve_thinking", False)
        reasoning_effort = kwargs.get("reasoning_effort")
        rendered = encode_messages(
            rendered_messages,
            thinking_mode=thinking_mode,
            drop_thinking=not preserve_thinking,
            reasoning_effort=reasoning_effort,
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        truncation: bool = False,
        return_attention_mask: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        input_ids = [
            self.encode(text, add_special_tokens=False)
            for text in texts
        ]
        if truncation and max_length is not None:
            input_ids = [ids[:max_length] for ids in input_ids]

        attention_masks = [[1] * len(ids) for ids in input_ids]
        if padding and len(input_ids) > 1:
            max_len = max(len(ids) for ids in input_ids)
            for idx, ids in enumerate(input_ids):
                pad_len = max_len - len(ids)
                if pad_len <= 0:
                    continue
                input_ids[idx] = ids + [self.pad_token_id] * pad_len
                attention_masks[idx] = attention_masks[idx] + [0] * pad_len

        result = {}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
            if return_attention_mask:
                result["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
        else:
            result["input_ids"] = input_ids
            if return_attention_mask:
                result["attention_mask"] = attention_masks
        return result

    def parse_thinking(self, text: str) -> tuple[Optional[str], str]:
        match = self._THINK_RE.search(text)
        if not match:
            return None, text
        reasoning = match.group(1).strip()
        visible = self._THINK_RE.sub("", text, count=1).strip()
        return reasoning, visible

    def parse_tool_calls(self, text: str) -> tuple[Optional[list], str]:
        try:
            parsed = parse_message_from_completion_text(
                text if text.endswith(self.eos_token or "") else text + (self.eos_token or ""),
                thinking_mode="chat",
            )
            tool_calls = parsed.get("tool_calls") or []
            if tool_calls:
                return tool_calls, parsed.get("content", "").strip()
        except ValueError as exc:
            logger.debug("DeepSeek-V4 DSML parser did not accept completion text: %s", exc)

        matches = list(self._TOOL_CALL_RE.finditer(text))
        if not matches:
            return None, text
        tool_calls = []
        for match in matches:
            name = match.group("name").strip()
            arguments = match.group("args").strip() or "{}"
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                arguments = json.dumps({"arguments": arguments}, ensure_ascii=False)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
        visible = self._TOOL_CALL_RE.sub("", text).strip()
        return tool_calls, visible
