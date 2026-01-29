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
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import tiktoken
import torch

from batchgen.config.base_tokenizer import BaseTokenizer
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


def _get_tiktoken_tokenizer() -> tiktoken.Encoding:
    """Create GPT-OSS tiktoken tokenizer.

    This follows the OpenAI reference implementation from
    gpt-oss/gpt_oss/tokenizer.py.

    Returns:
        tiktoken.Encoding configured for GPT-OSS
    """
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
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
        pad_token_id: 199999 (<|endoftext|>)
        vocab_size: 201,088
    """

    def __init__(self):
        """Initialize the GPT-OSS tokenizer using tiktoken."""
        self.tokenizer = _get_tiktoken_tokenizer()

        # Set special token IDs
        self.bos_token_id = GPT_OSS_BOS_TOKEN_ID
        self.eos_token_id = GPT_OSS_EOS_TOKEN_ID
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
            f"bos={self.bos_token_id}, eos={self.eos_token_id}, pad={self.pad_token_id}"
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
