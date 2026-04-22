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

"""Fast tokenizer implementation using the tokenizers library.

This module provides a tokenizer implementation that loads tokenizer.json
files directly using the Rust-based tokenizers library, removing the
dependency on transformers.AutoTokenizer.

The tokenizers library is the same one used by HuggingFace's PreTrainedTokenizerFast,
so this provides identical functionality without the transformers dependency.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from .base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class FastTokenizer(BaseTokenizer):
    """Tokenizer using HuggingFace tokenizers library (Rust-based).

    This class loads tokenizer.json files directly, providing the same
    functionality as PreTrainedTokenizerFast without requiring the
    transformers library.

    Attributes:
        tokenizer: The underlying tokenizers.Tokenizer instance
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
        bos_token_id: Beginning-of-sequence token ID
        vocab_size: Size of the vocabulary
    """

    def __init__(
        self,
        tokenizer_path: str,
        tokenizer_config: Optional[Dict] = None,
    ):
        """Initialize the fast tokenizer.

        Args:
            tokenizer_path: Path to directory containing tokenizer.json
            tokenizer_config: Optional pre-loaded tokenizer config dict.
                             If None, will load from tokenizer_config.json if exists.
        """
        self.tokenizer_path = Path(tokenizer_path)

        # Load the tokenizer from tokenizer.json
        tokenizer_file = self.tokenizer_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found at {tokenizer_file}. "
                f"Please ensure the tokenizer file exists in the model directory."
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        logger.info(f"Loaded tokenizer from {tokenizer_file}")

        # Load config for special tokens and chat template
        self._config = tokenizer_config or self._load_config()
        self._setup_special_tokens(self._config)

        # Store chat template if available
        self.chat_template = self._config.get("chat_template")

        # Get vocab size from tokenizer
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Configure padding
        self._setup_padding()

    def _load_config(self) -> Dict:
        """Load tokenizer_config.json if it exists.

        Returns:
            Dict with tokenizer config, empty dict if file doesn't exist
        """
        config_file = self.tokenizer_path / "tokenizer_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                return json.load(f)
        return {}

    def _extract_token_string(self, token) -> Optional[str]:
        """Extract token string from config value.

        Handles both plain string tokens and HuggingFace AddedToken dicts:
        - String: "token" -> "token"
        - Dict: {"__type": "AddedToken", "content": "token", ...} -> "token"

        Args:
            token: Token value from config (str, dict, or None)

        Returns:
            Token string or None
        """
        if token is None:
            return None
        if isinstance(token, str):
            return token
        if isinstance(token, dict) and "content" in token:
            return token["content"]
        return None

    def _setup_special_tokens(self, config: Dict) -> None:
        """Set up special token IDs from config.

        Args:
            config: Tokenizer config dict
        """
        # Get special tokens from config (handles AddedToken dicts)
        bos_token = self._extract_token_string(config.get("bos_token"))
        eos_token = self._extract_token_string(config.get("eos_token"))
        pad_token = self._extract_token_string(config.get("pad_token"))

        # Try to get token IDs from the tokenizer vocabulary
        vocab = self.tokenizer.get_vocab()

        # BOS token
        if bos_token and bos_token in vocab:
            self.bos_token_id = vocab[bos_token]
            self.bos_token = bos_token
        else:
            self.bos_token_id = None
            self.bos_token = None

        # EOS token
        if eos_token and eos_token in vocab:
            self.eos_token_id = vocab[eos_token]
            self.eos_token = eos_token
        else:
            # Fallback: look for common EOS tokens
            for token in ["</s>", "<|endoftext|>", "<eos>"]:
                if token in vocab:
                    self.eos_token_id = vocab[token]
                    self.eos_token = token
                    break
            else:
                self.eos_token_id = 0  # Ultimate fallback
                self.eos_token = None

        # PAD token
        if pad_token and pad_token in vocab:
            self.pad_token_id = vocab[pad_token]
            self.pad_token = pad_token
        else:
            # Use EOS as pad token if no pad token specified
            self.pad_token_id = self.eos_token_id
            self.pad_token = self.eos_token

        logger.debug(
            f"Special tokens: bos={self.bos_token_id}, "
            f"eos={self.eos_token_id}, pad={self.pad_token_id}"
        )

    def _setup_padding(self) -> None:
        """Configure padding in the tokenizer."""
        self.padding_side = "right"

        # Enable padding with the pad token
        if self.pad_token is not None:
            self.tokenizer.enable_padding(
                direction="right",
                pad_id=self.pad_token_id,
                pad_token=self.pad_token,
            )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

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
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

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
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=True)

        # Extract token IDs
        input_ids = [e.ids for e in encodings]
        attention_masks = [e.attention_mask for e in encodings]

        # CRITICAL: the underlying Rust tokenizer was configured with
        # `enable_padding(...)` in `_setup_padding` (called once at __init__),
        # so `encode_batch` ALWAYS right-pads each result to the max length
        # in the current call's batch — regardless of the Python-level
        # `padding=` arg. When the caller asked for `padding=False`, we must
        # strip that pad ourselves, or every per-sequence length recorded
        # downstream is silently inflated by however much the longest
        # sibling in the same call needed. (This is the root cause of the
        # GLM-5 multi-seq prefill slot-0 corruption: when the worker
        # tokenizes 2 admitted prompts at once with `padding=False`, the
        # shorter one's input_ids comes back inflated with pad tokens, the
        # inflated count flows into seq.prompt_length, then into
        # cu_seqlens[1]-cu_seqlens[0], and finally
        # last_token_indices = batch_cu_seqlens[1:] - 1 picks a pad-token
        # hidden state for lm_head → garbage output for that sequence.)
        if not padding:
            for i in range(len(input_ids)):
                # attention_mask is 1 for valid tokens, 0 for Rust-added pad
                valid = sum(attention_masks[i])
                if valid != len(input_ids[i]):
                    input_ids[i] = input_ids[i][:valid]
                    attention_masks[i] = attention_masks[i][:valid]

        # Pad if requested (tokenizers library handles this automatically if enabled)
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

        # Lazily build + cache the Jinja Template. Template() is ~10 ms per
        # construction; render() is ~30 µs. Rebuilding every call is the
        # reason scheduler admission of 3361 L4 prompts took ~90 s.
        template = getattr(self, "_jinja_template", None)
        if template is None:
            try:
                from jinja2 import Template, Undefined
            except ImportError:
                raise ImportError(
                    "jinja2 is required for apply_chat_template. "
                    "Install it with: pip install jinja2"
                )
            # Match HuggingFace transformers' rendering: permissive Undefined
            # + trim_blocks=True + lstrip_blocks=True (HF uses
            # ImmutableSandboxedEnvironment which sets both to True). Without
            # these, raw newlines/whitespace from block tags leak into the
            # rendered prompt.
            template = Template(
                self.chat_template,
                undefined=Undefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self._jinja_template = template

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
