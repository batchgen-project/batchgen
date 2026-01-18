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

"""HuggingFace tokenizer wrapper for fallback compatibility.

This module provides a wrapper around transformers.AutoTokenizer for models
that don't have a custom BatchGen tokenizer implementation yet. This allows
gradual migration to the new tokenizer abstraction.

Note: This wrapper still depends on the transformers library and should only
be used as a fallback for unsupported models.
"""

import logging
from typing import Dict, List, Optional, Union

import torch

from .base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class HuggingFaceTokenizerWrapper(BaseTokenizer):
    """Fallback wrapper for models without custom tokenizer implementation.

    This wrapper uses transformers.AutoTokenizer under the hood for
    migration compatibility. It should only be used for models that
    don't have a native BatchGen tokenizer implementation.

    Attributes:
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
        bos_token_id: Beginning-of-sequence token ID
        vocab_size: Size of the vocabulary
    """

    def __init__(self, model_path: str):
        """Initialize the HuggingFace tokenizer wrapper.

        Args:
            model_path: Path to model directory containing tokenizer files
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers library is required for HuggingFaceTokenizerWrapper. "
                "Please install it with: pip install transformers"
            ) from e

        logger.info(f"Loading HuggingFace tokenizer from {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._tokenizer.padding_side = "right"

        # Extract special token IDs
        self.eos_token_id = self._tokenizer.eos_token_id
        self.pad_token_id = self._tokenizer.pad_token_id or self.eos_token_id
        self.bos_token_id = self._tokenizer.bos_token_id
        self.vocab_size = self._tokenizer.vocab_size
        self.padding_side = "right"

        logger.debug(
            f"HuggingFace tokenizer loaded: vocab_size={self.vocab_size}, "
            f"eos={self.eos_token_id}, pad={self.pad_token_id}, bos={self.bos_token_id}"
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
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

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

        This method delegates to the underlying HuggingFace tokenizer.

        Args:
            texts: Single text or list of texts to tokenize
            return_tensors: Output format ("pt" for PyTorch tensors)
            padding: Whether to pad sequences to same length
            truncation: Whether to truncate sequences
            return_attention_mask: Whether to return attention mask
            max_length: Maximum sequence length

        Returns:
            Dict with "input_ids" and optionally "attention_mask"
        """
        return self._tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            max_length=max_length,
        )

    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        """Decode a batch of token sequences.

        Args:
            sequences: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded strings
        """
        return self._tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens
        )
