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

Chat prompts are rendered in the harmony format by
:func:`render_harmony_prompt` (``apply_chat_template``), including request
``tools``, assistant ``tool_calls`` history and ``tool`` result messages.

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
from typing import Dict, List, Optional, Tuple, Union

import tiktoken
import torch

from batchgen.config.base_tokenizer import BaseTokenizer
from batchgen.config.tokenizer_assets import O200K_BASE, load_tiktoken_ranks
from batchgen.config.tokenizer_registry import register_tokenizer
from batchgen.models.openai.gpt_oss_120b.harmony_tools import (
    FUNCTIONS_NAMESPACE,
    render_functions_namespace,
    validate_function_name,
)

logger = logging.getLogger(__name__)

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


# ---- Harmony prompt grammar ----
#
# One harmony message per chat message, in order; the prompt ends with
# ``<|start|>assistant`` when a generation prompt is requested:
#
#   <|start|>system<|message|>CONTENT\n\n# Valid channels: ...<|end|>
#   <|start|>developer<|message|># Tools ... namespace functions {...}<|end|>
#   <|start|>user<|message|>CONTENT<|end|>
#   <|start|>assistant<|channel|>final<|message|>CONTENT<|end|>
#   <|start|>assistant<|channel|>analysis<|message|>REASONING<|end|>
#   <|start|>assistant<|channel|>commentary<|message|>PREAMBLE<|end|>
#   <|start|>assistant to=functions.NAME<|channel|>commentary <|constrain|>json<|message|>ARGS<|call|>
#   <|start|>functions.NAME to=assistant<|channel|>commentary<|message|>RESULT<|end|>
#
# The developer message exists only when the request has tools; it follows
# the leading system message, which then also names the tool channel. A past
# turn ended by a final answer keeps no reasoning (harmony drops it); the
# reasoning of tool-calling assistant messages after the last final answer
# (the turn still in progress) is kept, so the model sees why it called.

_H_SYSTEM_SUFFIX = (
    "\n\n# Valid channels: analysis, commentary, final. "
    "Channel must be included for every message."
)
_H_TOOL_CHANNEL_NOTE = (
    "\nCalls to these tools must go to the commentary channel: 'functions'."
)
_H_JSON_CONTENT_TYPE = _H_CONSTRAIN + "json"


def _text_field(message: dict, field: str, index: int, *, optional: bool = False) -> str:
    value = message.get(field)
    if value is None and optional:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            f"messages[{index}] ({message.get('role')!r}): {field!r} must be a "
            f"string, got {type(value).__name__}"
        )
    return value


def _tool_call_parts(call: object, where: str) -> Tuple[str, str, Optional[str]]:
    """(name, arguments, id) of one OpenAI ``tool_calls`` entry."""
    if not isinstance(call, dict):
        raise ValueError(f"{where}: a tool call must be an object, got {type(call).__name__}")
    if call.get("type", "function") != "function":
        raise ValueError(f"{where}: only function tool calls are supported, got type={call.get('type')!r}")
    function = call.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"{where}: 'function' must be an object")
    name = validate_function_name(function.get("name"), where)
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError(
            f"{where}: function.arguments must be the JSON-encoded string the "
            f"model produced, got {type(arguments).__name__}"
        )
    call_id = call.get("id")
    if call_id is not None and not isinstance(call_id, str):
        raise ValueError(f"{where}: id must be a string, got {type(call_id).__name__}")
    return name, arguments, call_id


def _message(role_header: str, channel: Optional[str], body: str, terminator: str = _H_END) -> str:
    channel_part = "" if channel is None else _H_CHANNEL + channel
    return f"{_H_START}{role_header}{channel_part}{_H_MESSAGE}{body}{terminator}"


def render_harmony_prompt(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render OpenAI chat messages (and tools) as a GPT-OSS harmony prompt.

    Raises ``ValueError`` for a conversation the harmony format cannot carry
    faithfully: an unknown role, missing text, a malformed tool call, or a
    ``tool`` message that names no earlier call.
    """
    if not isinstance(messages, list):
        raise ValueError(f"messages must be a list, got {type(messages).__name__}")
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"tools must be a list, got {type(tools).__name__}")
    developer = None
    if tools:
        developer = _message(
            "developer", None, render_functions_namespace(tools)
        )

    def has_calls(message: dict) -> bool:
        return message.get("role") == "assistant" and bool(message.get("tool_calls"))

    # Reasoning is kept only for tool-calling messages after the last final
    # assistant answer: the turn the model is still working on.
    last_final = max(
        (
            i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "assistant" and not has_calls(m)
        ),
        default=-1,
    )

    parts: List[str] = []
    leading_system = bool(messages) and isinstance(messages[0], dict) and messages[0].get("role") == "system"
    if developer is not None and not leading_system:
        parts.append(developer)
    call_names: Dict[str, str] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object, got {type(message).__name__}")
        role = message.get("role")
        if role == "system":
            body = _text_field(message, "content", index) + _H_SYSTEM_SUFFIX
            if developer is not None and index == 0:
                body += _H_TOOL_CHANNEL_NOTE
            parts.append(_message("system", None, body))
            if developer is not None and index == 0:
                parts.append(developer)
        elif role == "user":
            parts.append(_message("user", None, _text_field(message, "content", index)))
        elif role == "assistant" and not has_calls(message):
            parts.append(_message(
                "assistant", "final", _text_field(message, "content", index)
            ))
        elif role == "assistant":
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                raise ValueError(f"messages[{index}]: tool_calls must be a list")
            reasoning = _text_field(message, "reasoning_content", index, optional=True)
            if reasoning and index > last_final:
                parts.append(_message("assistant", "analysis", reasoning))
            preamble = _text_field(message, "content", index, optional=True)
            if preamble:
                parts.append(_message("assistant", "commentary", preamble))
            for call_index, call in enumerate(calls):
                name, arguments, call_id = _tool_call_parts(
                    call, f"messages[{index}].tool_calls[{call_index}]"
                )
                if call_id is not None:
                    call_names[call_id] = name
                parts.append(_message(
                    f"assistant to={FUNCTIONS_NAMESPACE}.{name}",
                    f"commentary {_H_JSON_CONTENT_TYPE}",
                    arguments,
                    _H_CALL,
                ))
        elif role == "tool":
            name = _tool_result_name(message, index, call_names)
            parts.append(_message(
                f"{FUNCTIONS_NAMESPACE}.{name} to={_H_ASSISTANT}",
                "commentary",
                _text_field(message, "content", index),
            ))
        else:
            raise ValueError(
                f"messages[{index}]: role {role!r} is not supported by the "
                f"GPT-OSS chat format (system, user, assistant, tool)"
            )
    if add_generation_prompt:
        parts.append(_H_START + _H_ASSISTANT)
    return "".join(parts)


def _tool_result_name(message: dict, index: int, call_names: Dict[str, str]) -> str:
    """The function a ``tool`` message answers: its call id's, else its name."""
    call_id = message.get("tool_call_id")
    name = message.get("name")
    where = f"messages[{index}] (tool)"
    if call_id is not None:
        if call_id not in call_names:
            raise ValueError(
                f"{where}: tool_call_id {call_id!r} matches no earlier "
                f"assistant tool call"
            )
        if name is not None and name != call_names[call_id]:
            raise ValueError(
                f"{where}: name {name!r} disagrees with tool_call_id "
                f"{call_id!r}, which called {call_names[call_id]!r}"
            )
        return call_names[call_id]
    if name is None:
        raise ValueError(
            f"{where}: needs tool_call_id (or name) to say which function "
            f"returned this result"
        )
    return validate_function_name(name, where)


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

        logger.info(
            f"GPT-OSS tokenizer initialized (tiktoken o200k_harmony): "
            f"vocab_size={self.vocab_size}, "
            f"bos={self.bos_token_id}, eos={self.eos_token_id}, "
            f"stop_ids={sorted(self.eos_token_ids)}, pad={self.pad_token_id}"
        )

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
        messages: List[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Render chat messages as a harmony prompt (see render_harmony_prompt).

        Args:
            messages: OpenAI chat messages (system/user/assistant/tool).
            tokenize: If True, return token IDs; if False, return the string.
            add_generation_prompt: End with ``<|start|>assistant``.
            tools: OpenAI function tools; rendered as the developer message's
                ``functions`` namespace.
            **kwargs: Other chat-template switches the server forwards for
                every model (``enable_thinking``, ``clear_thinking``, ...).
                GPT-OSS has no such switch (its reasoning level is the
                system message's ``Reasoning:`` line), so they do not change
                the prompt.

        Raises:
            ValueError: the conversation or a tool definition cannot be
                rendered faithfully (the message names which one).
        """
        rendered = render_harmony_prompt(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
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
