# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
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
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from batchgen.config.fast_tokenizer import FastTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer
from batchgen.models.deepseek.deepseekv4_flash.assets.encoding.encoding_dsv4 import (
    encode_messages,
    parse_message_from_completion_text,
    thinking_end_token,
)

logger = logging.getLogger(__name__)

TOKENIZER_DIR = Path(__file__).parent / "assets"

DEEPSEEK_V4_BOS_TOKEN_ID = 0
DEEPSEEK_V4_EOS_TOKEN_ID = 1
DEEPSEEK_V4_VOCAB_SIZE = 129280


def _openai_tool_call(name: str, arguments: str) -> Dict[str, object]:
    """One parsed call in the ``parse_tool_calls`` contract shape.

    DSML carries no call id and the vendored encoder's
    ``tool_calls_to_openai_format`` emits none, so the id is assigned here,
    once per parsed call, in the form the other families use.
    """
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@register_tokenizer("deepseek_v4")
class DeepSeekV4Tokenizer(FastTokenizer):
    """DeepSeek-V4 tokenizer loaded from the vendored V4 Flash assets."""

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
        """Split a completion at the ``</think>`` that ends its reasoning.

        A thinking-mode prompt ends with ``<think>`` and a chat-mode one with
        ``</think>`` (``encoding_dsv4.render_message``), so a thinking-mode
        completion is ``reasoning</think>`` followed by exactly what a
        chat-mode completion holds, and a chat-mode completion has no
        ``</think>``: the vendored parser rejects one in reasoning or content
        in either mode. Hence a completion with exactly one ``</think>`` is
        split there. The text after it is returned unchanged, not stripped:
        it is a chat-mode completion that ``parse_tool_calls`` parses next,
        and the DSML tool-call block opens with ``"\\n\\n"``.

        Without ``</think>`` there is no reasoning block in the text (a chat
        completion, or reasoning cut off before the tag), and with more than
        one the text is no V4 message; both stay content.
        """
        reasoning, closed, visible = text.partition(thinking_end_token)
        if not closed:
            return None, text
        if thinking_end_token in visible:
            logger.debug(
                "DeepSeek-V4 completion has more than one %s; kept as content",
                thinking_end_token,
            )
            return None, text
        return reasoning.strip() or None, visible

    def parse_tool_calls(self, text: str) -> tuple[Optional[list], str]:
        """Parse DSML tool calls with the vendored V4 parser.

        The vendored ``parse_message_from_completion_text`` is the only
        grammar, run in the one mode that can accept the text: ``thinking``
        when it holds the ``</think>`` that ends a thinking-mode reasoning
        block (the unparsed completion), else ``chat`` (a chat completion, or
        the content ``parse_thinking`` returns). Text it rejects is not a
        well-formed tool-call message, so it is returned unchanged as content
        with no calls, the same result as text without calls
        (``BaseTokenizer.parse_tool_calls``). It rejects with ``ValueError``
        (malformed DSML inside the tool-calls block) and with ``assert``
        (``AssertionError``: reasoning not ended by ``</think>``, content
        after the block, content after an end-of-sentence token, or a special
        token in the reasoning or content); nothing else is reachable from a
        completion string.

        The vendored parser does not check a ``string="false"`` parameter,
        whose value must be JSON (the DSML tool prompt), and pastes it into
        the arguments verbatim. A call whose arguments are not JSON is
        therefore malformed DSML and rejected the same way.

        With calls, the reasoning of an unparsed thinking-mode completion
        stays in the returned content with its ``</think>``: reasoning is
        ``parse_thinking``'s to split.
        """
        thinking_mode = "thinking" if thinking_end_token in text else "chat"
        try:
            parsed = parse_message_from_completion_text(
                text if text.endswith(self.eos_token or "") else text + (self.eos_token or ""),
                thinking_mode=thinking_mode,
            )
            for call in parsed["tool_calls"]:
                json.loads(call["function"]["arguments"])
        except (ValueError, AssertionError) as exc:
            logger.debug("DeepSeek-V4 DSML parser did not accept completion text: %s", exc)
            return None, text

        tool_calls = parsed["tool_calls"]
        if not tool_calls:
            return None, text
        content = parsed["content"]
        if thinking_mode == "thinking":
            content = parsed["reasoning_content"] + thinking_end_token + content
        return [
            _openai_tool_call(
                call["function"]["name"], call["function"]["arguments"],
            )
            for call in tool_calls
        ], content.strip()
