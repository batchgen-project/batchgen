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

"""GPT-OSS-120B tokenizer for BatchGen.

This module provides the tokenizer implementation for GPT-OSS-120B model
using tiktoken with o200k_base encoding and custom special tokens.

This follows the OpenAI reference implementation from gpt-oss/gpt_oss/tokenizer.py.

GPT-OSS tokenizer specifications:
- Based on o200k encoding with custom special tokens
- Vocabulary size: 201,088 tokens
- BOS token: <|startoftext|> (ID: 199998)
- EOS token: <|return|> (ID: 200002)
- PAD token: <|endoftext|> (ID: 199999)
- Stop tokens: <|return|>, <|endoftext|>, <|call|> (``generation_config.json``)

Output parsing (``--parse-thinking`` / ``--parse-tool-call``) reads the
harmony message grammar. Its markers (``<|start|>``, ``<|channel|>``,
``<|message|>``, ``<|end|>``, ``<|call|>``, ...) are special tokens, so the
server must also run with ``--detokenization-include-special-tokens``;
without it the completion text is ``analysis...assistantfinal...`` with every
message boundary gone and nothing can be parsed.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import tiktoken
import torch

from batchgen.config.base_tokenizer import BaseTokenizer
from batchgen.config.tokenizer_assets import O200K_BASE, load_tiktoken_ranks
from batchgen.config.tokenizer_registry import register_tokenizer

logger = logging.getLogger(__name__)

# Tokenizer config files are in the same directory as this module
TOKENIZER_DIR = Path(__file__).parent

# GPT-OSS-120B special token IDs (from OpenAI reference)
GPT_OSS_BOS_TOKEN_ID = 199998   # <|startoftext|>
GPT_OSS_EOS_TOKEN_ID = 200002   # <|return|>
GPT_OSS_PAD_TOKEN_ID = 199999   # <|endoftext|>
GPT_OSS_VOCAB_SIZE = 201088

# Special tokens for GPT-OSS Harmony format
GPT_OSS_SPECIAL_TOKENS = {
    "<|startoftext|>": 199998,
    "<|endoftext|>": 199999,
    "<|reserved_200000|>": 200000,
    "<|reserved_200001|>": 200001,
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|reserved_200004|>": 200004,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|reserved_200009|>": 200009,
    "<|reserved_200010|>": 200010,
    "<|reserved_200011|>": 200011,
    "<|call|>": 200012,
}

#: Decode stops at any of these ids. Mirrors ``generation_config.json``
#: ``eos_token_id``: ``<|return|>`` ends a final answer, ``<|call|>`` ends a
#: tool call (the caller runs the tool and sends its result in the next turn),
#: ``<|endoftext|>`` ends the document.
GPT_OSS_STOP_TOKEN_IDS = frozenset({
    GPT_OSS_SPECIAL_TOKENS["<|return|>"],
    GPT_OSS_SPECIAL_TOKENS["<|endoftext|>"],
    GPT_OSS_SPECIAL_TOKENS["<|call|>"],
})


# ---- Harmony output grammar ----
#
# A completion follows a prompt that ends with ``<|start|>assistant``, so its
# first message has an implicit role header; every later message opens with
# ``<|start|>``:
#
#   <|channel|>analysis<|message|>REASONING<|end|>
#   <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>
#   <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>ARGS<|call|>
#
# The recipient (``to=...``) may also sit in the role header
# (``<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>``).
# ``commentary`` without a recipient is a user-visible preamble.

_H_START = "<|start|>"
_H_CHANNEL = "<|channel|>"
_H_CONSTRAIN = "<|constrain|>"
_H_MESSAGE = "<|message|>"
_H_END = "<|end|>"
_H_CALL = "<|call|>"
_H_RETURN = "<|return|>"
_H_ENDOFTEXT = "<|endoftext|>"
_H_TERMINATORS = (_H_END, _H_CALL, _H_RETURN, _H_ENDOFTEXT)
_H_MARKER_RE = re.compile(
    "|".join(
        re.escape(m)
        for m in (_H_START, _H_CHANNEL, _H_CONSTRAIN, _H_MESSAGE)
        + _H_TERMINATORS
    )
)
_H_RECIPIENT_RE = re.compile(r"(?:^|\s)to=(\S+)")
_H_FUNCTIONS_PREFIX = "functions."
_H_ASSISTANT = "assistant"

# Message kinds.
_REASONING = "reasoning"
_CONTENT = "content"
_TOOL_CALL = "tool_call"
_MALFORMED = "malformed"


@dataclass(frozen=True)
class HarmonyMessage:
    """One parsed harmony message of a completion.

    ``raw`` is the exact source span (role header through terminator).
    ``explicit_start`` is False for a message whose ``<|start|>`` lies in the
    prompt (the first one) or is missing. ``header_complete`` is False when the
    header never reached ``<|message|>``. ``terminator`` is None when the text
    ends, or the next message starts, before one appears.
    """

    raw: str
    explicit_start: bool
    role_header: str
    channel_header: Optional[str]
    header_complete: bool
    body: str
    terminator: Optional[str]

    @property
    def role(self) -> str:
        words = [w for w in self.role_header.split() if not w.startswith("to=")]
        return words[0] if words else _H_ASSISTANT

    @property
    def channel(self) -> Optional[str]:
        if self.channel_header is None:
            return None
        head = self.channel_header.split(_H_CONSTRAIN, 1)[0]
        words = [w for w in head.split() if not w.startswith("to=")]
        return words[0] if words else None

    @property
    def recipient(self) -> Optional[str]:
        for header in (self.role_header, self.channel_header or ""):
            m = _H_RECIPIENT_RE.search(header.split(_H_CONSTRAIN, 1)[0])
            if m:
                return m.group(1)
        return None

    @property
    def kind(self) -> str:
        if not self.header_complete or self.role != _H_ASSISTANT:
            return _MALFORMED
        if self.recipient is not None:
            return _TOOL_CALL
        if self.channel == "analysis":
            return _REASONING
        if self.channel in ("final", "commentary"):
            return _CONTENT
        return _MALFORMED

    def serialize(self) -> str:
        """The message as standalone harmony text, ``<|start|>`` included."""
        if self.explicit_start:
            return self.raw
        return _H_START + _H_ASSISTANT + self.raw


def parse_harmony(text: str) -> List[Union[str, HarmonyMessage]]:
    """Split completion text into plain-text pieces and harmony messages.

    Never raises: text outside any message header is returned as a plain
    string piece, in order.
    """
    markers = list(_H_MARKER_RE.finditer(text))
    items: List[Union[str, HarmonyMessage]] = []
    i = 0
    pos = 0  # start of not-yet-consumed text
    while i < len(markers):
        m = markers[i]
        tok = m.group(0)
        if tok == _H_START:
            msg_start = m.start()
            explicit = True
            role_end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            role_header = text[m.end():role_end]
            i += 1
        elif tok == _H_CHANNEL:
            explicit = False
            if not items and pos == 0:
                # First message: its role header (possibly `` to=...``) follows
                # the prompt's ``<|start|>assistant`` directly.
                msg_start = 0
                role_header = text[:m.start()]
            else:
                msg_start = m.start()
                role_header = ""
        else:
            i += 1
            continue  # stray marker: stays in the surrounding plain text

        if msg_start > pos:
            items.append(text[pos:msg_start])

        channel_header: Optional[str] = None
        if i < len(markers) and markers[i].group(0) == _H_CHANNEL:
            ch_start = markers[i].end()
            j = i + 1
            while j < len(markers) and markers[j].group(0) == _H_CONSTRAIN:
                j += 1
            ch_end = markers[j].start() if j < len(markers) else len(text)
            channel_header = text[ch_start:ch_end]
            i = j
        if i >= len(markers) or markers[i].group(0) != _H_MESSAGE:
            # Header never reached <|message|>: the rest up to the next
            # message start is one malformed message with an empty body.
            end = len(text)
            k = i
            while k < len(markers):
                if markers[k].group(0) == _H_START:
                    end = markers[k].start()
                    break
                k += 1
            items.append(HarmonyMessage(
                raw=text[msg_start:end], explicit_start=explicit,
                role_header=role_header, channel_header=channel_header,
                header_complete=False, body="", terminator=None,
            ))
            pos = end
            i = k
            continue

        body_start = markers[i].end()
        i += 1
        terminator: Optional[str] = None
        while i < len(markers):
            t = markers[i].group(0)
            if t in _H_TERMINATORS:
                terminator = t
                break
            if t in (_H_START, _H_CHANNEL):
                break
            i += 1
        if terminator is not None:
            body_end = markers[i].start()
            end = markers[i].end()
            i += 1
        elif i < len(markers):
            body_end = end = markers[i].start()
        else:
            body_end = end = len(text)
        items.append(HarmonyMessage(
            raw=text[msg_start:end], explicit_start=explicit,
            role_header=role_header, channel_header=channel_header,
            header_complete=True, body=text[body_start:body_end],
            terminator=terminator,
        ))
        pos = end

    if pos < len(text):
        items.append(text[pos:])
    return items


def _join(parts: List[str]) -> str:
    return "\n\n".join(p for p in (s.strip() for s in parts) if p)


def _get_tiktoken_tokenizer() -> tiktoken.Encoding:
    """Create GPT-OSS tiktoken tokenizer.

    This follows the OpenAI reference implementation from
    gpt-oss/gpt_oss/tokenizer.py, except that the o200k_base ranks come from
    the installer-staged, hash-verified file (``batchgen.config.tokenizer_assets``)
    instead of ``tiktoken.get_encoding``, which downloads them on first use.
    A missing file raises ``TokenizerAssetError`` naming the staging command.

    Returns:
        tiktoken.Encoding configured for GPT-OSS
    """
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=O200K_BASE.pat_str,
        mergeable_ranks=load_tiktoken_ranks(O200K_BASE),
        special_tokens={
            **O200K_BASE.special_tokens,
            **GPT_OSS_SPECIAL_TOKENS,
            # Reserved tokens from 200013 to 201087
            **{f"<|reserved_{i}|>": i for i in range(200013, 201088)},
        },
    )
    return tokenizer


@register_tokenizer("gpt_oss")
class GPTOssTokenizer(BaseTokenizer):
    """GPT-OSS-120B tokenizer using tiktoken.

    This tokenizer uses tiktoken with o200k_base encoding plus custom special
    tokens for the Harmony response format, following the OpenAI reference
    implementation.

    Attributes:
        bos_token_id: 199998 (<|startoftext|>)
        eos_token_id: 200002 (<|return|>)
        eos_token_ids: {200002, 199999, 200012} -- decode also stops at
            <|endoftext|> and at <|call|>, the end of a tool call
        pad_token_id: 199999 (<|endoftext|>)
        vocab_size: 201,088
    """

    def __init__(self):
        """Initialize the GPT-OSS tokenizer using tiktoken."""
        self.tokenizer = _get_tiktoken_tokenizer()

        # Set special token IDs
        self.bos_token_id = GPT_OSS_BOS_TOKEN_ID
        self.eos_token_id = GPT_OSS_EOS_TOKEN_ID
        # Every id decode stops at; the worker reads this set.
        self.eos_token_ids = set(GPT_OSS_STOP_TOKEN_IDS)
        self.pad_token_id = GPT_OSS_PAD_TOKEN_ID
        self.vocab_size = GPT_OSS_VOCAB_SIZE

        # Token strings
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|return|>"
        self.pad_token = "<|endoftext|>"

        # Load chat template from tokenizer_config.json if available
        self.chat_template = self._load_chat_template()

        logger.info(
            f"GPT-OSS tokenizer initialized (tiktoken o200k_harmony): "
            f"vocab_size={self.vocab_size}, "
            f"bos={self.bos_token_id}, eos={self.eos_token_id}, "
            f"stop_ids={sorted(self.eos_token_ids)}, pad={self.pad_token_id}"
        )

    def _load_chat_template(self) -> Optional[str]:
        """Load chat template from tokenizer_config.json if available."""
        config_file = TOKENIZER_DIR / "tokenizer_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                return config.get("chat_template")
            except Exception as e:
                logger.warning(f"Failed to load chat template: {e}")
        return None

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add special tokens (ignored for tiktoken)

        Returns:
            List of token IDs
        """
        # tiktoken handles special tokens based on allowed_special
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        if skip_special_tokens:
            # Filter out special tokens
            special_ids = set(GPT_OSS_SPECIAL_TOKENS.values())
            special_ids.update(range(200013, 201088))  # Reserved tokens
            token_ids = [t for t in token_ids if t not in special_ids]
        return self.tokenizer.decode(token_ids)

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        truncation: bool = False,
        return_attention_mask: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Batch tokenize texts.

        Args:
            texts: Single text or list of texts to tokenize
            return_tensors: Output format ("pt" for PyTorch tensors, None for lists)
            padding: Whether to pad sequences to same length
            truncation: Whether to truncate sequences (currently unused)
            return_attention_mask: Whether to return attention mask
            max_length: Maximum sequence length (currently unused)

        Returns:
            Dict with "input_ids" and optionally "attention_mask"
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Encode all texts
        input_ids = [self.encode(text) for text in texts]

        # Create attention masks
        attention_masks = [[1] * len(ids) for ids in input_ids]

        # Pad if requested
        if padding and len(texts) > 1:
            max_len = max(len(ids) for ids in input_ids)

            # Pad each sequence to max length
            for i in range(len(input_ids)):
                pad_len = max_len - len(input_ids[i])
                if pad_len > 0:
                    input_ids[i] = input_ids[i] + [self.pad_token_id] * pad_len
                    attention_masks[i] = attention_masks[i] + [0] * pad_len

        # Build result dict
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

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Apply chat template to format messages.

        Uses the Jinja2 chat_template from tokenizer_config.json to format
        a list of chat messages into a prompt string.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tokenize: If True, return token IDs; if False, return string
            add_generation_prompt: Whether to add generation prompt at the end
            **kwargs: Additional arguments passed to template

        Returns:
            Formatted prompt string (if tokenize=False) or token IDs (if tokenize=True)

        Raises:
            ValueError: If no chat template is available
        """
        if not self.chat_template:
            raise ValueError(
                "This tokenizer does not have a chat_template. "
                "Cannot apply chat template formatting."
            )

        try:
            from jinja2 import Template, StrictUndefined
        except ImportError:
            raise ImportError(
                "jinja2 is required for apply_chat_template. "
                "Install it with: pip install jinja2"
            )

        # Create Jinja2 template
        template = Template(self.chat_template, undefined=StrictUndefined)

        # Render the template with messages and special tokens
        rendered = template.render(
            messages=messages,
            bos_token=self.bos_token or "",
            eos_token=self.eos_token or "",
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    # ---- Output parsing ----
    #
    # The scheduler runs parse_thinking, then feeds its visible text to
    # parse_tool_calls; either may also run alone. Each method resolves only
    # the messages it owns and hands every other message on as standalone
    # harmony text (HarmonyMessage.serialize), so the two compose:
    #   parse_thinking   -- analysis -> reasoning; final / commentary preamble
    #                       -> content; non-assistant or channel-less messages
    #                       -> raw content with a warning.
    #   parse_tool_calls -- assistant messages with a recipient -> tool_calls;
    #                       one not ended by <|call|> or with non-JSON
    #                       arguments -> raw content with a warning.

    def parse_thinking(self, text: str) -> Tuple[Optional[str], str]:
        items = parse_harmony(text)
        if not any(isinstance(item, HarmonyMessage) for item in items):
            if text.strip():
                self._warn_no_harmony_markers()
            return None, text

        reasoning: List[str] = []
        visible: List[str] = []
        has_reasoning = False
        for item in items:
            if isinstance(item, str):
                visible.append(item)
                continue
            kind = item.kind
            if kind == _REASONING:
                has_reasoning = True
                reasoning.append(item.body)
            elif kind == _CONTENT:
                visible.append(item.body)
            elif kind == _TOOL_CALL:
                visible.append(item.serialize())
            else:
                logger.warning(
                    "GPT-OSS parse_thinking: malformed harmony message "
                    "(role=%r channel=%r); returning it raw in content: %r",
                    item.role, item.channel, item.raw[:200],
                )
                visible.append(item.serialize())
        return (_join(reasoning) if has_reasoning else None), _join(visible)

    def parse_tool_calls(self, text: str) -> Tuple[Optional[list], str]:
        items = parse_harmony(text)
        if not any(
            isinstance(item, HarmonyMessage) and item.kind == _TOOL_CALL
            for item in items
        ):
            return None, text

        tool_calls: List[dict] = []
        visible: List[str] = []
        for item in items:
            if isinstance(item, str):
                visible.append(item)
            elif item.kind != _TOOL_CALL:
                visible.append(item.serialize())
            else:
                call = self._tool_call(item)
                if call is None:
                    visible.append(item.serialize())
                else:
                    tool_calls.append(call)
        return tool_calls or None, _join(visible)

    @staticmethod
    def _tool_call(message: HarmonyMessage) -> Optional[dict]:
        """The OpenAI tool_calls entry for one recipient message, or None."""
        recipient = message.recipient or ""
        name = (
            recipient[len(_H_FUNCTIONS_PREFIX):]
            if recipient.startswith(_H_FUNCTIONS_PREFIX)
            else recipient
        )
        arguments = message.body.strip()
        problem = None
        if message.terminator != _H_CALL:
            problem = (
                f"ended by {message.terminator!r}, not <|call|> "
                f"(truncated by max_tokens?)"
                if message.terminator is None
                else f"ended by {message.terminator!r}, not <|call|>"
            )
        elif not name:
            problem = "empty function name"
        else:
            try:
                json.loads(arguments)
            except ValueError as exc:
                problem = f"arguments are not JSON ({exc})"
        if problem is not None:
            logger.warning(
                "GPT-OSS parse_tool_calls: tool call to %r is %s; returning "
                "it raw in content: %r",
                recipient, problem, message.raw[:200],
            )
            return None
        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def _warn_no_harmony_markers(self) -> None:
        """Log once per tokenizer: parsing got text with no harmony grammar."""
        if getattr(self, "_warned_no_harmony_markers", False):
            logger.debug(
                "GPT-OSS parse_thinking: no harmony markers; content is raw"
            )
            return
        self._warned_no_harmony_markers = True
        logger.warning(
            "GPT-OSS parse_thinking: the completion text has no harmony "
            "message markers (<|channel|>...<|message|>), so it is returned "
            "raw in content. The harmony markers are special tokens: run the "
            "server with --detokenization-include-special-tokens so the "
            "decoded text keeps them. Logged once; later occurrences are "
            "DEBUG."
        )
