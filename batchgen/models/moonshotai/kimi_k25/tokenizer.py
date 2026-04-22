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

"""Kimi K2.5 Tokenizer for BatchGen.

This module wraps the TikToken tokenizer from the assets directory
and registers it with BatchGen's tokenizer registry.

The Kimi K2.5 tokenizer uses TikToken format with 163,840 tokens.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from tokenizers import AddedToken

from batchgen.config.base_tokenizer import BaseTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer

# Kimi K2.5 tokenizer assets directory
TOKENIZER_DIR = Path(__file__).parent / "assets"

# Kimi K2.5 special token IDs (from tokenizer_config.json)
KIMI_K25_BOS_TOKEN_ID = 163584  # "[BOS]"
KIMI_K25_EOS_TOKEN_ID = 163586  # "<|im_end|>" (chat end-of-turn)
KIMI_K25_EOS_TOKEN_IDS = {163585, 163586}  # Both "[EOS]" and "<|im_end|>"
KIMI_K25_PAD_TOKEN_ID = 163839  # "[PAD]"
KIMI_K25_VOCAB_SIZE = 163840


@register_tokenizer("kimi_k25")
class KimiK25Tokenizer(BaseTokenizer):
    """Kimi K2.5 tokenizer using TikToken.

    This tokenizer wraps the TikTokenTokenizer from the assets directory
    and provides the BatchGen BaseTokenizer interface.

    Attributes:
        bos_token_id: 163584 ("[BOS]")
        eos_token_id: 163586 ("<|im_end|>")
        pad_token_id: 163839 ("[PAD]")
        vocab_size: 163840
    """

    def __init__(self):
        """Initialize Kimi K2.5 tokenizer from assets."""
        # Import TikTokenTokenizer from assets
        from batchgen.models.moonshotai.kimi_k25.assets.tokenization_kimi import (
            TikTokenTokenizer,
        )

        # Load tokenizer config
        config_file = TOKENIZER_DIR / "tokenizer_config.json"
        with open(config_file) as f:
            config = json.load(f)

        # Get added_tokens_decoder and convert to AddedToken format
        added_tokens_decoder_raw = config.get("added_tokens_decoder", {})
        added_tokens_decoder = {
            int(k): AddedToken(v["content"], special=v.get("special", False))
            for k, v in added_tokens_decoder_raw.items()
        }

        # Load tokenizer from tiktoken.model with added_tokens_decoder and special tokens
        vocab_file = str(TOKENIZER_DIR / "tiktoken.model")
        self._tokenizer = TikTokenTokenizer(
            vocab_file,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]",
            added_tokens_decoder=added_tokens_decoder
        )

        # Load chat template from jinja file
        jinja_file = TOKENIZER_DIR / "chat_template.jinja"
        if jinja_file.exists():
            self._tokenizer.chat_template = jinja_file.read_text()

        # Set special token IDs
        self.bos_token_id = KIMI_K25_BOS_TOKEN_ID
        self.eos_token_id = KIMI_K25_EOS_TOKEN_ID
        self.eos_token_ids = KIMI_K25_EOS_TOKEN_IDS
        self.pad_token_id = KIMI_K25_PAD_TOKEN_ID
        self.vocab_size = KIMI_K25_VOCAB_SIZE
        self.padding_side = "right"

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add BOS/EOS tokens (mapped to allow_special_tokens)

        Returns:
            List of token IDs
        """
        # TikTokenTokenizer uses allow_special_tokens, not add_special_tokens
        # When add_special_tokens=True, allow all special tokens
        return self._tokenizer.encode(text, allow_special_tokens=add_special_tokens)

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
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        conversation,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Apply chat template to format conversation.

        Args:
            conversation: List of message dicts with "role" and "content"
            tokenize: Whether to tokenize the output
            add_generation_prompt: Whether to add generation prompt
            **kwargs: Additional arguments passed to underlying tokenizer

        Returns:
            Formatted string or token IDs if tokenize=True
        """
        return self._tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

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
            truncation: Whether to truncate sequences
            return_attention_mask: Whether to return attention mask
            max_length: Maximum sequence length

        Returns:
            Dict with "input_ids" and optionally "attention_mask"
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        # Encode all texts
        encoded = [self.encode(text) for text in texts]

        # Pad if requested
        if padding:
            max_len = max(len(seq) for seq in encoded)
            padded = []
            attention_mask = []

            for seq in encoded:
                pad_len = max_len - len(seq)
                if self.padding_side == "right":
                    padded.append(seq + [self.pad_token_id] * pad_len)
                    attention_mask.append([1] * len(seq) + [0] * pad_len)
                else:
                    padded.append([self.pad_token_id] * pad_len + seq)
                    attention_mask.append([0] * pad_len + [1] * len(seq))

            encoded = padded
        else:
            attention_mask = [[1] * len(seq) for seq in encoded]

        # Convert to tensors if requested
        result = {}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(encoded, dtype=torch.long)
            if return_attention_mask:
                result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        else:
            result["input_ids"] = encoded
            if return_attention_mask:
                result["attention_mask"] = attention_mask

        return result

    # ---- Output parsing ----

    # Match both <think>...</think> and bare ...{reasoning}</think> (when <think> was in the prompt)
    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _THINK_CLOSE_RE = re.compile(r"^(.*?)</think>", re.DOTALL)
    _TOOL_CALLS_RE = re.compile(
        r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
        re.DOTALL,
    )
    # K2.5 tool call format: <|tool_call_begin|>functions.func_name:idx<|tool_call_argument_begin|>{json}<|tool_call_end|>
    _TOOL_CALL_RE = re.compile(
        r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*"
        r"<\|tool_call_argument_begin\|>\s*(?P<arguments>.*?)\s*<\|tool_call_end\|>",
        re.DOTALL,
    )

    def parse_thinking(self, text: str) -> tuple[Optional[str], str]:
        # Case 1: Full <think>...</think> tags present
        m = self._THINK_RE.search(text)
        if m:
            reasoning = m.group(1).strip()
            visible = self._THINK_RE.sub("", text, count=1).strip()
            return reasoning, visible
        # Case 2: Only closing </think> — opening <think> was in the prompt
        m = self._THINK_CLOSE_RE.search(text)
        if m:
            reasoning = m.group(1).strip()
            visible = text[m.end():].strip()
            return reasoning if reasoning else None, visible
        return None, text

    def parse_tool_calls(self, text: str) -> tuple[Optional[list], str]:
        section_match = self._TOOL_CALLS_RE.search(text)
        if not section_match:
            return None, text
        section = section_match.group(1)
        calls = []
        for m in self._TOOL_CALL_RE.finditer(section):
            tool_call_id = m.group("tool_call_id")
            args_str = m.group("arguments").strip()
            # Extract function name from "functions.func_name:idx" format
            func_name = tool_call_id.split(".")[1].split(":")[0] if "." in tool_call_id else tool_call_id
            calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": args_str,
                },
            })
        visible = self._TOOL_CALLS_RE.sub("", text, count=1).strip()
        return calls if calls else None, visible
