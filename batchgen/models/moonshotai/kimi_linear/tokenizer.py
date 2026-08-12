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

"""Kimi-Linear-48B Tokenizer for BatchGen.

The Kimi-Linear-48B testbed uses the same tiktoken 'kimi' merge file as Kimi
K2.5 (vocab 163840, bos=163584, eos=163586, pad=163839) — `tiktoken.model` is
byte-identical — so this module reuses the verified `TikTokenTokenizer`
implementation from `batchgen.models.moonshotai.kimi_k25.assets` and registers a
thin `BaseTokenizer` wrapper for the "kimi_linear" tokenizer type.

NOT Kimi-K3. A matching merge-file md5 does not imply matching special tokens.
K3 names the same reserved ids differently — 163586 `<|end_of_msg|>` vs
`<|im_end|>`, 163587 `<|open|>` vs `<|im_user|>`, 163588 `<|close|>` vs
`<|im_assistant|>`, plus `<|sep|>` at 163589 which the 48B does not have at all
— and K3 has no Jinja chat template (its XTML format is implemented in Python).
Cross-loading renders a 12-token K3 prompt fragment as 32 marker-free BPE
tokens, silently. K3 is served by `KimiK3Tokenizer`
(`batchgen/models/moonshotai/kimi_k3/tokenizer.py`, tokenizer type "kimi_k3").

By default the wrapper loads its vocab/config from this package's own assets
directory. A concrete model directory may be passed via `model_path`; note that
`load_tokenizer()` never does (`config/tokenizer_registry.py:144` constructs
with no arguments), so in production that parameter is always None — the branch
that silently substituted kimi_k25's chat template in bug_log.md 2026-07-31.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from tokenizers import AddedToken

logger = logging.getLogger(__name__)

from batchgen.config.base_tokenizer import BaseTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer

# The tiktoken.model / tokenizer_config.json bundled with kimi_k25 are the
# canonical fallback: the tiktoken merge file is byte-identical to the
# Kimi-Linear / Kimi-K3 checkpoints, so we reuse them rather than re-vendor.
_K25_ASSETS_DIR = (
    Path(__file__).resolve().parent.parent / "kimi_k25" / "assets"
)

# Special token IDs (from tokenizer_config.json) — shared with Kimi K2.5.
KIMI_LINEAR_BOS_TOKEN_ID = 163584  # "[BOS]"
KIMI_LINEAR_EOS_TOKEN_ID = 163586  # "<|im_end|>" (chat end-of-turn)
KIMI_LINEAR_EOS_TOKEN_IDS = {163585, 163586}  # Both "[EOS]" and "<|im_end|>"
KIMI_LINEAR_PAD_TOKEN_ID = 163839  # "[PAD]"
KIMI_LINEAR_VOCAB_SIZE = 163840


@register_tokenizer("kimi_linear")
class KimiLinearTokenizer(BaseTokenizer):
    """Kimi-Linear-48B tokenizer using TikToken.

    Wraps the verified ``TikTokenTokenizer`` shipped with kimi_k25 (the
    tiktoken merge file is byte-identical across the two families) and exposes
    the BatchGen ``BaseTokenizer`` interface.

    Does NOT serve Kimi-K3 — see the module docstring. K3's special tokens
    differ at ids 163586-163591; use ``KimiK3Tokenizer``.

    Attributes:
        bos_token_id: 163584 ("[BOS]")
        eos_token_id: 163586 ("<|im_end|>" in the 48B's config; the SAME id is
            "<|end_of_msg|>" in Kimi-K3's, which is why the two must not share
            a tokenizer class)
        pad_token_id: 163839 ("[PAD]")
        vocab_size: 163840
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize the tokenizer.

        Args:
            model_path: Optional path to a model directory containing
                ``tiktoken.model`` / ``tokenizer_config.json`` /
                ``chat_template.jinja`` (e.g. the testbed checkpoint dir).
                When omitted, the bundled kimi_k25 assets are used (identical
                vocab).
        """
        # Reuse the verified TikToken implementation from kimi_k25 assets.
        from batchgen.models.moonshotai.kimi_k25.assets.tokenization_kimi import (
            TikTokenTokenizer,
        )

        # Default to kimi_linear's OWN vendored assets (checkpoint's
        # chat_template.jinja + tokenizer_config.json). The k25 assets are
        # only a fallback for tiktoken.model (byte-identical vocab) — the k25
        # CHAT TEMPLATE is materially different (media tokens, tool sections,
        # think handling) and must never be used for kimi_linear.
        _own_assets = Path(__file__).parent / "assets"
        assets_dir = Path(model_path) if model_path is not None else _own_assets

        # tiktoken.model is byte-identical across families; fall back to the
        # bundled copy if a passed model dir happens not to ship it.
        # (No-silent-fallback policy: equivalent substitute, but still warn.)
        vocab_path = assets_dir / "tiktoken.model"
        if not vocab_path.exists():
            logger.warning(
                "kimi_linear tokenizer: %s has no tiktoken.model; using the "
                "kimi_k25 bundled copy (byte-identical vocab)", assets_dir)
            vocab_path = _K25_ASSETS_DIR / "tiktoken.model"

        # tokenizer_config.json drives added_tokens_decoder (special tokens).
        # Fail fast if missing — special-token drift is silent corruption.
        config_path = assets_dir / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"kimi_linear tokenizer_config.json not found in {assets_dir}; "
                "refusing to substitute another model's special-token config")
        with open(config_path) as f:
            config = json.load(f)

        added_tokens_decoder_raw = config.get("added_tokens_decoder", {})
        added_tokens_decoder = {
            int(k): AddedToken(v["content"], special=v.get("special", False))
            for k, v in added_tokens_decoder_raw.items()
        }

        self._tokenizer = TikTokenTokenizer(
            str(vocab_path),
            bos_token=config.get("bos_token", "[BOS]"),
            eos_token=config.get("eos_token", "[EOS]"),
            pad_token=config.get("pad_token", "[PAD]"),
            unk_token=config.get("unk_token", "[UNK]"),
            added_tokens_decoder=added_tokens_decoder,
        )

        # Chat template: NEVER substitute another model's template — that was
        # the silent-wrong-template bug (bug_log.md 2026-07-31). Fail fast.
        jinja_file = assets_dir / "chat_template.jinja"
        if not jinja_file.exists():
            raise FileNotFoundError(
                f"kimi_linear chat_template.jinja not found in {assets_dir}; "
                "refusing to fall back to another model's chat template")
        self._tokenizer.chat_template = jinja_file.read_text()

        # Special token IDs.
        self.bos_token_id = KIMI_LINEAR_BOS_TOKEN_ID
        self.eos_token_id = KIMI_LINEAR_EOS_TOKEN_ID
        self.eos_token_ids = KIMI_LINEAR_EOS_TOKEN_IDS
        self.pad_token_id = KIMI_LINEAR_PAD_TOKEN_ID
        self.vocab_size = KIMI_LINEAR_VOCAB_SIZE
        self.padding_side = "right"

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to allow special tokens (mapped to the
                underlying tokenizer's ``allow_special_tokens``)

        Returns:
            List of token IDs
        """
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
        """Apply chat template to format a conversation.

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
            truncation: Whether to truncate sequences (currently unused)
            return_attention_mask: Whether to return attention mask
            max_length: Maximum sequence length (currently unused)

        Returns:
            Dict with "input_ids" and optionally "attention_mask"
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self.encode(text) for text in texts]

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

    # ---- Output parsing (Kimi K2.5 think/tool format, kept as default) ----

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
