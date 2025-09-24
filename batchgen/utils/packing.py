from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

__all__ = ["PackedBatch", "PrefillPacker", "SequenceSegment", "SequenceMapping"]


@dataclass
class SequenceMapping:
    """Maps original sequence IDs to their positions in packed batches."""

    original_id: int
    batch_idx: int
    start_pos: int
    length: int


@dataclass
class PackedBatch:
    """Represents a packed batch for prefill processing.

    Attributes:
        input_ids: List of token sequences, padded to max_length
        position_ids: List of position indices for each token in sequences
        attention_mask: Tensor indicating valid tokens (shape: batch_size, max_length)
        sequence_mappings: Maps original sequence IDs to their positions in packed batches
    """

    input_ids: List[List[int]]
    position_ids: List[List[int]]
    attention_mask: torch.Tensor  # Shape: (batch_size, max_length)
    sequence_mappings: Optional[List[SequenceMapping]] = None


class SequenceSegment(NamedTuple):
    """Represents a segment of a sequence to be packed."""

    sequence_id: int
    tokens: List[int]
    start_pos: int  # Position within original sequence
    length: int

    @classmethod
    def from_tokens(
        cls, sequence_id: int, tokens: List[int], start_pos: int = 0
    ) -> "SequenceSegment":
        """Create a segment from tokens."""
        return cls(sequence_id, tokens, start_pos, len(tokens))


class SequenceItem:
    """Helper class to track sequence information during packing."""

    def __init__(self, sequence_id: int, tokens: List[int]):
        self.sequence_id = sequence_id
        self.tokens = tokens
        self.length = len(tokens)

    def __lt__(self, other: "SequenceItem") -> bool:
        # For sorting by length (descending order for better packing)
        return self.length > other.length


class PrefillPacker:
    """
    A high-performance packer for prefill phase sequences.

    Uses a greedy bin-packing algorithm to minimize padding tokens and reduce
    redundant computation. The packer maintains sequence order information
    to enable restoration of original sequences.

    Features:
    - Efficient sequence packing with minimal padding
    - 2D attention mask indicating valid tokens (batch_size, max_length)
    - Support for variable-length sequences
    - Memory-efficient tensor operations
    - Sequence mapping for order restoration

    Example:
        >>> input_ids_list = [
        ...     [101, 102, 103],
        ...     [201, 202],
        ...     [301, 302, 303, 304],
        ...     [401]
        ... ]
        >>> packed_batch = PrefillPacker.pack(
        ...     input_ids_list, max_length=6, include_sequence_mappings=True
        ... )
        >>> print(packed_batch.input_ids)
        [[301, 302, 303, 304, 201, 202], [101, 102, 103, 401, 0, 0]]
        >>> print(packed_batch.attention_mask.shape)
        torch.Size([2, 6])
        >>> print(packed_batch.position_ids)
        [[0, 1, 2, 0, 1, 0], [0, 1, 2, 3, 0, 0]]
    """

    @classmethod
    def pack(
        cls,
        input_ids_list: List[List[int]],
        max_length: int,
        pad_token_id: int = 0,
        include_sequence_mappings: bool = False,
    ) -> PackedBatch:
        """
        Pack sequences into batches with minimal padding.

        Args:
            input_ids_list: List of token sequences to pack
            max_length: Maximum length per packed batch
            pad_token_id: Token ID used for padding
            include_sequence_mappings: Whether to include sequence mapping info

        Returns:
            PackedBatch containing packed sequences with proper masks

        Raises:
            ValueError: If any sequence exceeds max_length
        """
        if not input_ids_list:
            return PackedBatch([], [], torch.empty(0, 0, dtype=torch.int8))

        # Validate input sequences
        for i, seq in enumerate(input_ids_list):
            if len(seq) > max_length:
                raise ValueError(
                    f"Sequence {i} length {len(seq)} exceeds max_length {max_length}"
                )

        # Create sequence items and sort by length (descending) for better packing
        sequences = [
            SequenceItem(i, seq) for i, seq in enumerate(input_ids_list)
        ]
        sequences.sort(reverse=True, key=lambda x: x.length)

        # Pack sequences using greedy bin packing
        packed_batches = cls._pack_sequences(sequences, max_length)

        # Generate final outputs
        return cls._create_packed_batch(
            packed_batches, max_length, pad_token_id, include_sequence_mappings
        )

    @classmethod
    def _pack_sequences(
        cls, sequences: List[SequenceItem], max_length: int
    ) -> List[List[SequenceSegment]]:
        """
        Greedily pack sequences into batches.
        Each batch can hold sequences whose total length does not exceed max_length.
        """
        batches = []
        batch_remaining_space = []

        for seq in sequences:
            # Try to find an existing batch with enough space
            placed = False
            for batch_idx, remaining_space in enumerate(batch_remaining_space):
                if remaining_space >= seq.length:
                    # Place sequence in this batch
                    segment = SequenceSegment.from_tokens(
                        seq.sequence_id, seq.tokens
                    )
                    batches[batch_idx].append(segment)
                    batch_remaining_space[batch_idx] -= seq.length
                    placed = True
                    break

            if not placed:
                # Create new batch
                segment = SequenceSegment.from_tokens(
                    seq.sequence_id, seq.tokens
                )
                batches.append([segment])
                batch_remaining_space.append(max_length - seq.length)

        return batches

    @classmethod
    def _create_packed_batch(
        cls,
        packed_batches: List[List[SequenceSegment]],
        max_length: int,
        pad_token_id: int,
        include_sequence_mappings: bool,
    ) -> PackedBatch:
        batch_input_ids = []
        batch_position_ids = []
        sequence_mappings = [] if include_sequence_mappings else None

        for batch_idx, batch_segments in enumerate(packed_batches):
            # Initialize batch tensors
            input_ids_row = []
            position_ids_row = []

            current_pos = 0

            for segment in batch_segments:
                # Add tokens to batch
                input_ids_row.extend(segment.tokens)

                # Generate position IDs (restart from 0 for each sequence)
                position_ids_row.extend(list(range(segment.length)))

                # Track sequence mapping for restoration
                if include_sequence_mappings:
                    mapping = SequenceMapping(
                        original_id=segment.sequence_id,
                        batch_idx=batch_idx,
                        start_pos=current_pos,
                        length=segment.length,
                    )
                    sequence_mappings.append(mapping)

                current_pos += segment.length

            # Pad to max_length
            padding_length = max_length - len(input_ids_row)
            input_ids_row.extend([pad_token_id] * padding_length)
            position_ids_row.extend([0] * padding_length)

            batch_input_ids.append(input_ids_row)
            batch_position_ids.append(position_ids_row)

        # Generate attention mask (2D: batch_size x max_length)
        attention_mask = cls._create_attention_mask(packed_batches, max_length)

        return PackedBatch(
            input_ids=batch_input_ids,
            position_ids=batch_position_ids,
            attention_mask=attention_mask,
            sequence_mappings=sequence_mappings,
        )

    @classmethod
    def _create_attention_mask(
        cls, packed_batches: List[List[SequenceSegment]], max_length: int
    ) -> torch.Tensor:  # (batch_size, max_length)
        batch_size = len(packed_batches)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.int8)

        for batch_idx, segments in enumerate(packed_batches):
            current_pos = 0
            for segment in segments:
                seg_len = segment.length
                # Set attention for this segment
                attention_mask[
                    batch_idx, current_pos : current_pos + seg_len
                ] = 1
                current_pos += seg_len

        return attention_mask

    @classmethod
    def compute_packing_efficiency(cls, packed_batch: PackedBatch) -> float:
        """
        Compute the packing efficiency (ratio of non-padding tokens).

        Args:
            packed_batch: The packed batch to analyze

        Returns:
            Efficiency ratio between 0 and 1
        """
        # Handle empty batch case
        if not packed_batch.input_ids:
            return 1.0
        
        total_positions = len(packed_batch.input_ids) * len(
            packed_batch.input_ids[0]
        )
        if total_positions == 0:
            return 1.0

        # Count non-padding positions using attention mask
        non_padding_positions = packed_batch.attention_mask.sum().item()

        return non_padding_positions / total_positions
