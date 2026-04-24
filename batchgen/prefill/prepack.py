"""
Prepack utilities for efficient prefill batching.

Prepack combines multiple shorter sequences into rows to minimize padding waste.
This is especially important for MLP/MoE layers where padding tokens waste computation.

Example:
    Without prepack (3 sequences, max_len=7):
        [A A _ _ _ _ _]  <- 5 padding tokens
        [B B B _ _ _ _]  <- 4 padding tokens
        [C C C C C C C]  <- 0 padding tokens
        Total: 21 positions, 9 wasted

    With prepack:
        [A A B B B _ _]  <- 2 padding tokens (A and B packed together)
        [C C C C C C C]  <- 0 padding tokens
        Total: 14 positions, 2 wasted
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


@dataclass
class PrepackMetadata:
    """Metadata for prepacked sequences."""

    # Packed tensors - shape: [num_packed_rows, row_length]
    packed_input_ids: torch.Tensor
    packed_attention_mask: torch.Tensor
    packed_position_ids: torch.Tensor

    # Mapping info for unpacking
    # sequence_ids[i, j] = original sequence index for position (i, j), -1 for padding
    sequence_ids: torch.Tensor

    # For flash attention varlen
    # cu_seqlens_per_row[row_idx] = cumulative sequence lengths for sequences in that row
    # This is a list of tensors, one per row
    cu_seqlens_per_row: List[torch.Tensor]
    max_seqlen_per_row: List[int]

    # Original sequence info
    original_seq_lengths: List[int]  # Length of each original sequence
    num_original_sequences: int
    num_packed_rows: int
    row_length: int

    # Packing assignment: pack_assignment[seq_idx] = (row_idx, start_pos)
    pack_assignment: List[Tuple[int, int]]


def bin_pack_first_fit_decreasing(
    seq_lengths: List[int],
    row_capacity: int,
) -> List[List[Tuple[int, int]]]:
    """
    Bin-pack sequences using First-Fit Decreasing algorithm.

    Args:
        seq_lengths: List of sequence lengths
        row_capacity: Maximum tokens per row

    Returns:
        List of rows, where each row is a list of (seq_idx, seq_length) tuples
    """
    # Create (original_index, length) pairs and sort by length descending
    indexed_lengths = [(i, length) for i, length in enumerate(seq_lengths)]
    indexed_lengths.sort(key=lambda x: -x[1])  # Sort descending by length

    rows: List[List[Tuple[int, int]]] = []
    row_remaining: List[int] = []

    for seq_idx, seq_len in indexed_lengths:
        if seq_len > row_capacity:
            # Sequence too long for any row - give it its own row
            rows.append([(seq_idx, seq_len)])
            row_remaining.append(row_capacity - seq_len)  # Will be negative
            continue

        # Find first row with enough space
        placed = False
        for row_idx, remaining in enumerate(row_remaining):
            if remaining >= seq_len:
                rows[row_idx].append((seq_idx, seq_len))
                row_remaining[row_idx] -= seq_len
                placed = True
                break

        if not placed:
            # Create new row
            rows.append([(seq_idx, seq_len)])
            row_remaining.append(row_capacity - seq_len)

    return rows


def prepack_sequences(
    input_ids_list: List[torch.Tensor],
    attention_mask_list: List[torch.Tensor],
    row_capacity: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> PrepackMetadata:
    """
    Prepack multiple sequences into rows to minimize padding.

    Args:
        input_ids_list: List of input_ids tensors, each shape [1, seq_len] or [seq_len]
        attention_mask_list: List of attention_mask tensors, matching input_ids shapes
        row_capacity: Maximum tokens per row. If None, uses max sequence length.
        device: Target device for output tensors

    Returns:
        PrepackMetadata containing packed tensors and unpacking information
    """
    if device is None:
        device = input_ids_list[0].device

    # Normalize to 1D tensors and get lengths
    input_ids_1d = []
    attention_mask_1d = []
    seq_lengths = []

    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        # Flatten to 1D
        ids = input_ids.view(-1)
        mask = attention_mask.view(-1)

        # Get actual sequence length (sum of attention mask)
        actual_len = int(mask.sum().item())

        # Only keep the valid tokens
        input_ids_1d.append(ids[:actual_len])
        attention_mask_1d.append(mask[:actual_len])
        seq_lengths.append(actual_len)

    num_sequences = len(seq_lengths)
    max_seq_len = max(seq_lengths) if seq_lengths else 0

    # Determine row capacity
    if row_capacity is None:
        row_capacity = max_seq_len

    # Bin-pack sequences
    packed_rows = bin_pack_first_fit_decreasing(seq_lengths, row_capacity)
    num_packed_rows = len(packed_rows)

    # Determine actual row length (max of row_capacity and longest sequence)
    row_length = max(row_capacity, max_seq_len)

    # Initialize output tensors
    packed_input_ids = torch.zeros(
        (num_packed_rows, row_length), dtype=torch.long, device=device
    )
    packed_attention_mask = torch.zeros(
        (num_packed_rows, row_length), dtype=torch.long, device=device
    )
    packed_position_ids = torch.zeros(
        (num_packed_rows, row_length), dtype=torch.long, device=device
    )
    sequence_ids = torch.full(
        (num_packed_rows, row_length), -1, dtype=torch.long, device=device
    )

    # Track packing assignment
    pack_assignment: List[Tuple[int, int]] = [(-1, -1)] * num_sequences

    # Fill packed tensors
    cu_seqlens_per_row: List[torch.Tensor] = []
    max_seqlen_per_row: List[int] = []

    for row_idx, row_contents in enumerate(packed_rows):
        current_pos = 0
        cu_seqlens = [0]
        max_seqlen = 0

        for seq_idx, seq_len in row_contents:
            # Copy tokens
            packed_input_ids[row_idx, current_pos:current_pos + seq_len] = input_ids_1d[seq_idx]
            packed_attention_mask[row_idx, current_pos:current_pos + seq_len] = 1

            # Position IDs within each sequence (0, 1, 2, ...)
            packed_position_ids[row_idx, current_pos:current_pos + seq_len] = torch.arange(
                seq_len, device=device
            )

            # Track which sequence each position belongs to
            sequence_ids[row_idx, current_pos:current_pos + seq_len] = seq_idx

            # Record packing assignment
            pack_assignment[seq_idx] = (row_idx, current_pos)

            current_pos += seq_len
            cu_seqlens.append(current_pos)
            max_seqlen = max(max_seqlen, seq_len)

        cu_seqlens_per_row.append(
            torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        )
        max_seqlen_per_row.append(max_seqlen)

    return PrepackMetadata(
        packed_input_ids=packed_input_ids,
        packed_attention_mask=packed_attention_mask,
        packed_position_ids=packed_position_ids,
        sequence_ids=sequence_ids,
        cu_seqlens_per_row=cu_seqlens_per_row,
        max_seqlen_per_row=max_seqlen_per_row,
        original_seq_lengths=seq_lengths,
        num_original_sequences=num_sequences,
        num_packed_rows=num_packed_rows,
        row_length=row_length,
        pack_assignment=pack_assignment,
    )


def build_prefill_micro_batches(
    seq_lengths: List[int],
    token_cap: int,
    *,
    l2_balance: bool = True,
    l2_slack: float = 1.2,
    single_sequence_only: bool = False,
) -> Tuple[List[Tuple[int, int]], int]:
    """Plan contiguous sequence-index micro-batches for prefill.

    Returns:
        A pair ``(micro_batches, l2_cap)`` where each micro-batch is a
        ``(seq_start, seq_end)`` half-open range over the original sequence
        order. When ``single_sequence_only`` is set, each sequence gets its own
        micro-batch and ``l2_cap`` is reported as 0 because no balancing is
        applied.
    """
    if token_cap <= 0:
        raise ValueError(f"token_cap must be positive, got {token_cap}")

    num_sequences = len(seq_lengths)
    if num_sequences == 0:
        return [], 0

    if single_sequence_only:
        return [(seq_idx, seq_idx + 1) for seq_idx in range(num_sequences)], 0

    total_tokens_all = sum(seq_lengths)
    total_l2_all = sum(seq_len * seq_len for seq_len in seq_lengths)
    est_num_mb = max(1, (total_tokens_all + token_cap - 1) // token_cap)
    l2_cap = (
        int(l2_slack * total_l2_all / est_num_mb)
        if (l2_balance and est_num_mb > 0)
        else 0
    )

    micro_batches: List[Tuple[int, int]] = []
    current_batch_start = 0
    current_batch_tokens = 0
    current_batch_l2 = 0

    for seq_idx, seq_len in enumerate(seq_lengths):
        seq_l2 = seq_len * seq_len
        over_tokens = current_batch_tokens + seq_len > token_cap
        over_l2 = (l2_cap > 0) and (current_batch_l2 + seq_l2 > l2_cap)
        if (over_tokens or over_l2) and current_batch_tokens > 0:
            micro_batches.append((current_batch_start, seq_idx))
            current_batch_start = seq_idx
            current_batch_tokens = 0
            current_batch_l2 = 0

        current_batch_tokens += seq_len
        current_batch_l2 += seq_l2

    if current_batch_start < num_sequences:
        micro_batches.append((current_batch_start, num_sequences))

    return micro_batches, l2_cap


def unpack_outputs(
    packed_outputs: torch.Tensor,
    metadata: PrepackMetadata,
    pad_to_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Unpack outputs from prepacked format back to original batch format.

    Args:
        packed_outputs: Tensor of shape [num_packed_rows, row_length, hidden_dim]
        metadata: PrepackMetadata from prepack_sequences
        pad_to_length: If specified, pad output sequences to this length.
                      If None, uses max original sequence length.

    Returns:
        Tensor of shape [num_original_sequences, output_length, hidden_dim]
    """
    num_sequences = metadata.num_original_sequences
    hidden_dim = packed_outputs.shape[-1]
    device = packed_outputs.device

    if pad_to_length is None:
        pad_to_length = max(metadata.original_seq_lengths)

    # Initialize output tensor
    outputs = torch.zeros(
        (num_sequences, pad_to_length, hidden_dim),
        dtype=packed_outputs.dtype,
        device=device,
    )

    # Extract each sequence's output
    for seq_idx, (row_idx, start_pos) in enumerate(metadata.pack_assignment):
        seq_len = metadata.original_seq_lengths[seq_idx]
        outputs[seq_idx, :seq_len] = packed_outputs[row_idx, start_pos:start_pos + seq_len]

    return outputs


def unpack_last_token_logits(
    packed_logits: torch.Tensor,
    metadata: PrepackMetadata,
) -> torch.Tensor:
    """
    Extract the last token logits for each original sequence from prepacked output.

    This is useful for getting next-token predictions from prepacked prefill.

    Args:
        packed_logits: Tensor of shape [num_packed_rows, row_length, vocab_size]
        metadata: PrepackMetadata from prepack_sequences

    Returns:
        Tensor of shape [num_original_sequences, vocab_size] - last token logits per sequence
    """
    num_sequences = metadata.num_original_sequences
    vocab_size = packed_logits.shape[-1]
    device = packed_logits.device

    # Initialize output
    last_token_logits = torch.zeros(
        (num_sequences, vocab_size),
        dtype=packed_logits.dtype,
        device=device,
    )

    # Extract last token for each sequence
    for seq_idx, (row_idx, start_pos) in enumerate(metadata.pack_assignment):
        seq_len = metadata.original_seq_lengths[seq_idx]
        last_pos = start_pos + seq_len - 1
        last_token_logits[seq_idx] = packed_logits[row_idx, last_pos]

    return last_token_logits


def create_block_diagonal_attention_mask(
    metadata: PrepackMetadata,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create a block-diagonal attention mask for prepacked sequences.

    Each sequence in a row can only attend to tokens within the same sequence,
    creating a block-diagonal structure within each row.

    Args:
        metadata: PrepackMetadata from prepack_sequences
        dtype: Output dtype (bool for additive mask, float for multiplicative)

    Returns:
        Tensor of shape [num_packed_rows, row_length, row_length]
        True (or 1.0) where attention is allowed, False (or 0.0) where blocked
    """
    num_rows = metadata.num_packed_rows
    row_length = metadata.row_length
    device = metadata.sequence_ids.device

    # Create mask using sequence_ids
    # Position (i, j) can attend to position (i, k) if:
    # 1. Both positions belong to the same sequence (sequence_ids match)
    # 2. Position k <= position j within the sequence (causal)

    mask = torch.zeros(
        (num_rows, row_length, row_length),
        dtype=dtype,
        device=device,
    )

    for row_idx in range(num_rows):
        row_seq_ids = metadata.sequence_ids[row_idx]  # [row_length]
        row_positions = metadata.packed_position_ids[row_idx]  # [row_length]

        # For each query position
        for q_pos in range(row_length):
            q_seq_id = row_seq_ids[q_pos].item()
            if q_seq_id == -1:  # Padding position
                continue

            q_within_seq_pos = row_positions[q_pos].item()

            # This query can attend to keys where:
            # - Same sequence ID
            # - Key position within sequence <= query position within sequence
            same_seq = (row_seq_ids == q_seq_id)
            causal = (row_positions <= q_within_seq_pos)
            valid_keys = same_seq & causal

            if dtype == torch.bool:
                mask[row_idx, q_pos] = valid_keys
            else:
                mask[row_idx, q_pos] = valid_keys.to(dtype)

    return mask


def get_prepack_stats(metadata: PrepackMetadata) -> dict:
    """
    Get statistics about the prepacking efficiency.

    Returns:
        Dictionary with packing statistics
    """
    total_tokens = sum(metadata.original_seq_lengths)
    total_positions = metadata.num_packed_rows * metadata.row_length
    padding_positions = total_positions - total_tokens

    # What it would be without packing
    unpacked_positions = metadata.num_original_sequences * metadata.row_length
    unpacked_padding = unpacked_positions - total_tokens

    return {
        "num_sequences": metadata.num_original_sequences,
        "num_packed_rows": metadata.num_packed_rows,
        "row_length": metadata.row_length,
        "total_real_tokens": total_tokens,
        "total_positions_packed": total_positions,
        "padding_tokens_packed": padding_positions,
        "total_positions_unpacked": unpacked_positions,
        "padding_tokens_unpacked": unpacked_padding,
        "padding_saved": unpacked_padding - padding_positions,
        "packing_efficiency": total_tokens / total_positions if total_positions > 0 else 0,
        "rows_saved": metadata.num_original_sequences - metadata.num_packed_rows,
    }
