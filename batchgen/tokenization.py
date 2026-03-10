"""Tokenization utilities for batch inference.

This module provides utilities for batch tokenization and detokenization,
extracted from BatchGenWorker for better modularity.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from batchgen.config.base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SequenceTokens:
    """Tokenization result for a single sequence."""

    input_ids: torch.Tensor  # [1, seq_extended_size]
    attention_mask: torch.Tensor  # [1, seq_extended_size]
    decoded_tokens: torch.Tensor  # [1, max_decoding_length]
    prompt_length: int
    kv_token_budget: int  # Total token budget for this sequence


class BatchTokenizer:
    """Handles batch tokenization and detokenization for inference.

    This class wraps a BatchGen tokenizer and provides optimized batch
    tokenization with proper handling of variable-length sequences.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        eos_token_id: int,
        model_context_length: int,
        rank: int = 0,
    ):
        """Initialize the batch tokenizer.

        Args:
            tokenizer: BatchGen tokenizer instance (BaseTokenizer or subclass)
            eos_token_id: Token ID for end-of-sequence
            model_context_length: Maximum context length the model supports
            rank: Process rank for logging (default: 0)
        """
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.model_context_length = model_context_length
        self.rank = rank

    def tokenize_batch(
        self,
        texts: List[str],
        max_decoding_length: int,
    ) -> Tuple[List[SequenceTokens], int]:
        """Tokenize a batch of prompts without truncation.

        The max_prompt_length is determined dynamically as the longest prompt.
        All sequences are tokenized at once for efficiency using HuggingFace's
        batch tokenization with Rust multi-threading.

        Args:
            texts: List of prompt strings to tokenize
            max_decoding_length: Maximum number of tokens to decode per sequence

        Returns:
            Tuple of (list of SequenceTokens, max_prompt_length)
        """
        if not texts:
            return [], 0

        if self.rank == 0:
            logger.info(f"Batch tokenizing {len(texts)} sequences...")

        tokenize_start = time.perf_counter()

        # Phase 1: Batch tokenize all sequences at once
        # HuggingFace tokenizers use Rust with multi-threading for batch tokenization
        batch_tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=False,  # No truncation - keep full input
            padding=True,  # Pad to longest in batch for uniform tensor shape
            return_attention_mask=True,
        )
        tokenize_time = time.perf_counter() - tokenize_start

        if self.rank == 0:
            logger.info(f"Batch tokenization complete in {tokenize_time:.2f}s")

        # Phase 2: Extract individual sequences and compute prompt lengths
        # batch_tokenized["input_ids"] has shape [batch_size, max_seq_len]
        # BatchGen uses right-padding: valid tokens at [0:actual_len], padding at [actual_len:]
        prompt_lengths = []
        tokenized_inputs = []

        for i in range(len(texts)):
            # Count non-pad tokens (attention_mask == 1)
            actual_len = int(batch_tokenized["attention_mask"][i].sum().item())
            prompt_lengths.append(actual_len)
            tokenized_inputs.append(
                {
                    "input_ids": batch_tokenized["input_ids"][i : i + 1, :actual_len],
                    "attention_mask": batch_tokenized["attention_mask"][
                        i : i + 1, :actual_len
                    ],
                }
            )

        max_prompt_length = max(prompt_lengths)

        # Warn if any prompt exceeds model context length
        if max_prompt_length >= self.model_context_length:
            logger.warning(
                f"Rank {self.rank}: Longest prompt ({max_prompt_length} tokens) exceeds or equals "
                f"model context length ({self.model_context_length}). Some sequences may not decode."
            )

        logger.info(
            f"Rank {self.rank}: Dynamic max_prompt_length set to {max_prompt_length} "
            f"(prompt lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)})"
        )

        # Phase 3: Create per-sequence tensors sized to their actual prompt length
        # OPTIMIZATION: Don't pad all sequences to max_prompt_length - each sequence
        # only needs space for its own prompt + decoding. This is critical for long-tailed
        # distributions where a few sequences are very long but most are short.
        results = []
        for i, tokenized in enumerate(tokenized_inputs):
            actual_prompt_len = tokenized["input_ids"].size(1)

            # Each sequence gets its own sized tensor: actual_prompt_len + max_decoding_length
            # Capped by model context length to avoid wasting memory on impossible decoding
            seq_extended_size = min(
                actual_prompt_len + max_decoding_length, self.model_context_length
            )

            input_ids_extended = torch.zeros(
                (1, seq_extended_size), dtype=tokenized["input_ids"].dtype
            )
            attention_mask_extended = torch.zeros((1, seq_extended_size), dtype=torch.int64)

            # Copy the actual tokens (left-aligned, no truncation)
            input_ids_extended[0, :actual_prompt_len] = tokenized["input_ids"][0, :]
            # CRITICAL: Set attention mask to exactly match input_ids length
            # Don't copy from tokenizer's attention_mask - just set 1s for valid tokens
            # This ensures attention_mask.sum() == prompt_length == current_context_length
            attention_mask_extended[0, :actual_prompt_len] = 1

            decoded_tokens = torch.zeros(1, max_decoding_length, dtype=torch.int64)

            results.append(
                SequenceTokens(
                    input_ids=input_ids_extended,
                    attention_mask=attention_mask_extended,
                    decoded_tokens=decoded_tokens,
                    prompt_length=actual_prompt_len,
                    kv_token_budget=seq_extended_size,
                )
            )

        logger.info(f"Rank {self.rank}: Tokenized {len(texts)} sequences")

        return results, max_prompt_length

    def decode_tokens_to_string(
        self,
        tokens: torch.Tensor,
        min_tokens: int = 1,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs to string, stopping at first EOS token.

        Args:
            tokens: Tensor of token IDs, shape [1, seq_len] or [seq_len]
            min_tokens: Minimum tokens before considering EOS (to avoid empty outputs)
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded string, truncated at first valid EOS position
        """
        # Flatten to 1D if needed
        if tokens.dim() > 1:
            tokens = tokens.squeeze(0)

        tokens_list = tokens.tolist()

        # Find first EOS token position (after min_tokens)
        eos_positions = [
            i
            for i, t in enumerate(tokens_list)
            if t == self.eos_token_id and i >= min_tokens
        ]

        if eos_positions:
            end_pos = eos_positions[0]
        else:
            # No EOS found, use all non-padding tokens
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
            non_pad = [i for i, t in enumerate(tokens_list) if t != pad_id]
            end_pos = non_pad[-1] + 1 if non_pad else len(tokens_list)

        # Decode tokens up to end position
        return self.tokenizer.decode(
            tokens_list[:end_pos], skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        token_sequences: List[torch.Tensor],
        min_tokens: int = 1,
        skip_special_tokens: bool = False,
    ) -> List[str]:
        """Decode multiple token sequences to strings.

        Args:
            token_sequences: List of token tensors
            min_tokens: Minimum tokens before considering EOS
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded strings
        """
        decode_start = time.perf_counter()
        decoded_strings = []

        for tokens in token_sequences:
            decoded_str = self.decode_tokens_to_string(
                tokens, min_tokens=min_tokens, skip_special_tokens=skip_special_tokens
            )
            decoded_strings.append(decoded_str)

        decode_time = time.perf_counter() - decode_start
        logger.info(
            f"Detokenization complete: {len(decoded_strings)} sequences in {decode_time:.2f}s"
        )

        return decoded_strings
