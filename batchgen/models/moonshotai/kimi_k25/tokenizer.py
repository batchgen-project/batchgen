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
KIMI_K25_EOS_TOKEN_ID = 163585  # "[EOS]"
KIMI_K25_PAD_TOKEN_ID = 163839  # "[PAD]"
KIMI_K25_VOCAB_SIZE = 163840


@register_tokenizer("kimi_k25")
class KimiK25Tokenizer(BaseTokenizer):
    """Kimi K2.5 tokenizer using TikToken.

    This tokenizer wraps the TikTokenTokenizer from the assets directory
    and provides the BatchGen BaseTokenizer interface.

    Attributes:
        bos_token_id: 163584 ("[BOS]")
        eos_token_id: 163585 ("[EOS]")
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

        # Set special token IDs
        self.bos_token_id = KIMI_K25_BOS_TOKEN_ID
        self.eos_token_id = KIMI_K25_EOS_TOKEN_ID
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
