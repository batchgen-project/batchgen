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

"""Base tokenizer abstraction for BatchGen.

This module defines the abstract base class for all BatchGen tokenizers,
providing a unified API that removes dependency on transformers.AutoTokenizer.

Usage:
    from batchgen.config.tokenizer_registry import load_tokenizer

    tokenizer = load_tokenizer("/path/to/model")
    tokens = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(tokens)

    # Batch tokenization (HuggingFace-compatible API)
    batch = tokenizer(["Hello", "World"], return_tensors="pt", padding=True)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional

import torch


class BaseTokenizer(ABC):
    """Abstract base class for all BatchGen tokenizers.

    This class defines the unified API that all tokenizer implementations
    must follow. It is designed to be a drop-in replacement for HuggingFace's
    PreTrainedTokenizer in BatchGen's inference pipeline.

    Attributes:
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
        bos_token_id: Beginning-of-sequence token ID
        vocab_size: Size of the vocabulary
        padding_side: Side to pad on ("left" or "right")
    """

    eos_token_id: int
    pad_token_id: int
    bos_token_id: Optional[int]
    vocab_size: int
    padding_side: str = "right"

    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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

        This method provides HuggingFace-compatible API for batch tokenization,
        enabling drop-in replacement in existing BatchGen code.

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
        pass

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
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in sequences
        ]
