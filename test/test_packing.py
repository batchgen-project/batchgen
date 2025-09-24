from typing import List

import pytest
import torch

from batchgen.utils.packing import (
    PackedBatch,
    PrefillPacker,
    SequenceMapping,
    SequenceSegment,
)


class TestPrefillPacker:
    """Test suite for PrefillPacker class."""

    def test_empty_input(self):
        """Test packing with empty input list."""
        result = PrefillPacker.pack([], max_length=10)

        assert result.input_ids == []
        assert result.position_ids == []
        assert result.attention_mask.shape == (0, 0)
        assert result.sequence_mappings is None

    def test_single_sequence(self):
        """Test packing with a single sequence."""
        input_ids_list = [[101, 102, 103]]
        max_length = 5

        result = PrefillPacker.pack(
            input_ids_list, max_length, include_sequence_mappings=True
        )

        # Check basic structure
        assert len(result.input_ids) == 1
        assert len(result.position_ids) == 1
        assert result.attention_mask.shape == (1, 5)

        # Check content
        assert result.input_ids[0] == [101, 102, 103, 0, 0]
        assert result.position_ids[0] == [0, 1, 2, 0, 0]
        assert result.attention_mask[0].tolist() == [1, 1, 1, 0, 0]

        # Check sequence mapping
        assert len(result.sequence_mappings) == 1
        mapping = result.sequence_mappings[0]
        assert mapping.original_id == 0
        assert mapping.batch_idx == 0
        assert mapping.start_pos == 0
        assert mapping.length == 3

    def test_basic_packing_example(self):
        """Test the basic example from docstring."""
        input_ids_list = [
            [101, 102, 103],
            [201, 202],
            [301, 302, 303, 304],
            [401],
        ]
        max_length = 6

        result = PrefillPacker.pack(
            input_ids_list, max_length, include_sequence_mappings=True
        )

        # Should create 2 batches
        assert len(result.input_ids) == 2
        assert len(result.position_ids) == 2
        assert result.attention_mask.shape == (2, 6)

        # Check first batch (longest sequences first due to sorting)
        batch_0 = result.input_ids[0]
        pos_0 = result.position_ids[0]
        mask_0 = result.attention_mask[0]

        # Check sequence mappings
        assert len(result.sequence_mappings) == 4
        import pdb; pdb.set_trace()

        # Verify all original sequences are mapped
        mapped_ids = {m.original_id for m in result.sequence_mappings}
        assert mapped_ids == {0, 1, 2, 3}

    def test_perfect_fit_packing(self):
        """Test packing where sequences fit perfectly without padding."""
        input_ids_list = [
            [1, 2, 3],  # length 3
            [4, 5, 6],  # length 3
        ]
        max_length = 6

        result = PrefillPacker.pack(input_ids_list, max_length)

        # Should fit perfectly in one batch
        assert len(result.input_ids) == 1
        assert result.input_ids[0] == [1, 2, 3, 4, 5, 6]
        assert result.position_ids[0] == [0, 1, 2, 0, 1, 2]
        assert result.attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1]

    def test_sequence_too_long_error(self):
        """Test error handling for sequences longer than max_length."""
        input_ids_list = [[1, 2, 3, 4, 5, 6, 7]]  # length 7
        max_length = 5

        with pytest.raises(ValueError, match="exceeds max_length"):
            PrefillPacker.pack(input_ids_list, max_length)

    def test_custom_pad_token(self):
        """Test using custom padding token ID."""
        input_ids_list = [[1, 2], [3, 4, 5]]
        max_length = 6
        pad_token_id = 999

        result = PrefillPacker.pack(input_ids_list, max_length, pad_token_id)

        # Check that custom pad token is used
        batch = result.input_ids[0]
        assert batch[-1] == pad_token_id or batch[-2] == pad_token_id

    def test_position_ids_restart_per_sequence(self):
        """Test that position IDs restart from 0 for each sequence."""
        input_ids_list = [
            [100, 101, 102, 103],  # length 4
            [200, 201],  # length 2
        ]
        max_length = 6

        result = PrefillPacker.pack(input_ids_list, max_length)

        position_ids = result.position_ids[0]

        # Find where second sequence starts (where position resets to 0)
        reset_positions = [i for i, pos in enumerate(position_ids) if pos == 0]

        # Should have at least 2 reset positions (start of seq1 and seq2)
        # Note: padding positions also have 0, so might be more
        assert len([p for p in reset_positions if p < 6]) >= 2

    def test_attention_mask_shape_and_dtype(self):
        """Test attention mask has correct shape and dtype."""
        input_ids_list = [[1, 2], [3, 4, 5]]
        max_length = 8

        result = PrefillPacker.pack(input_ids_list, max_length)

        # Check shape is 2D (batch_size, max_length)
        assert result.attention_mask.ndim == 2
        assert result.attention_mask.shape[1] == max_length

        # Check dtype is int8 for memory efficiency
        assert result.attention_mask.dtype == torch.int8

        # Check values are only 0 or 1
        unique_values = result.attention_mask.unique().tolist()
        assert set(unique_values).issubset({0, 1})

    def test_packing_efficiency_calculation(self):
        """Test packing efficiency calculation."""
        input_ids_list = [
            [1, 2, 3],  # length 3
            [4, 5],  # length 2
        ]
        max_length = 8

        packed = PrefillPacker.pack(input_ids_list, max_length)
        efficiency = PrefillPacker.compute_packing_efficiency(packed)

        # Should be 5 valid tokens out of 8 total positions = 0.625
        expected_efficiency = 5.0 / 8.0
        assert abs(efficiency - expected_efficiency) < 1e-6

    def test_packing_efficiency_empty_batch(self):
        """Test packing efficiency for empty batch."""
        packed = PrefillPacker.pack([], max_length=10)
        efficiency = PrefillPacker.compute_packing_efficiency(packed)

        # Empty batch should have 100% efficiency
        assert efficiency == 1.0

    def test_greedy_packing_order(self):
        """Test that sequences are packed in length-descending order for efficiency."""
        input_ids_list = [
            [1],  # length 1
            [2, 3, 4, 5, 6],  # length 5
            [7, 8],  # length 2
            [9, 10, 11, 12],  # length 4
        ]
        max_length = 6

        result = PrefillPacker.pack(
            input_ids_list, max_length, include_sequence_mappings=True
        )

        # Longest sequence (length 5) should be in first batch
        # Next longest that fits (length 1) should join it
        # This tests the greedy first-fit decreasing algorithm

        batch_0_tokens = [t for t in result.input_ids[0] if t != 0]
        assert len(batch_0_tokens) == 6  # 5 + 1 = full capacity

    def test_multiple_small_sequences(self):
        """Test packing many small sequences."""
        input_ids_list = [[i] for i in range(20)]  # 20 sequences of length 1
        max_length = 5

        result = PrefillPacker.pack(input_ids_list, max_length)

        # Should pack 5 sequences per batch, so 4 batches total
        assert len(result.input_ids) == 4

        # Each batch should be fully utilized (except possibly the last)
        for i in range(3):  # First 3 batches
            valid_tokens = result.attention_mask[i].sum().item()
            assert valid_tokens == 5

        # Last batch should have the remaining sequences
        last_batch_tokens = result.attention_mask[3].sum().item()
        assert last_batch_tokens == 5  # 20 sequences, 5 per batch

    def test_single_token_sequences(self):
        """Test edge case with single-token sequences."""
        input_ids_list = [[42], [43], [44]]
        max_length = 4

        result = PrefillPacker.pack(
            input_ids_list, max_length, include_sequence_mappings=True
        )

        # Should pack all 3 single-token sequences in one batch
        assert len(result.input_ids) == 1
        assert result.input_ids[0] == [42, 43, 44, 0]
        assert result.position_ids[0] == [
            0,
            0,
            0,
            0,
        ]  # Each sequence starts at pos 0
        assert result.attention_mask[0].tolist() == [1, 1, 1, 0]

    def test_sequence_mapping_correctness(self):
        """Test that sequence mappings contain correct information."""
        input_ids_list = [
            [100, 101],  # seq 0, length 2
            [200, 201, 202],  # seq 1, length 3
            [300],  # seq 2, length 1
        ]
        max_length = 4

        result = PrefillPacker.pack(
            input_ids_list, max_length, include_sequence_mappings=True
        )

        # Should create 2 batches due to sorting by length
        assert len(result.input_ids) == 2
        assert len(result.sequence_mappings) == 3

        # Check that mappings are correct
        for mapping in result.sequence_mappings:
            original_seq = input_ids_list[mapping.original_id]

            # Extract the mapped sequence from packed batch
            batch_tokens = result.input_ids[mapping.batch_idx]
            extracted_tokens = batch_tokens[
                mapping.start_pos : mapping.start_pos + mapping.length
            ]

            assert extracted_tokens == original_seq
            assert mapping.length == len(original_seq)

    def test_large_batch_stress_test(self):
        """Stress test with many sequences of varying lengths."""
        import random

        random.seed(42)  # For reproducible tests

        # Generate 100 sequences with random lengths 1-10
        input_ids_list = []
        for i in range(100):
            length = random.randint(1, 10)
            sequence = list(range(i * 100, i * 100 + length))
            input_ids_list.append(sequence)

        max_length = 20

        result = PrefillPacker.pack(
            input_ids_list, max_length, include_sequence_mappings=True
        )

        # Basic sanity checks
        assert len(result.sequence_mappings) == 100
        assert all(len(batch) == max_length for batch in result.input_ids)
        assert all(len(batch) == max_length for batch in result.position_ids)

        # Check efficiency is reasonable (should be quite high with good packing)
        efficiency = PrefillPacker.compute_packing_efficiency(result)
        assert efficiency > 0.5  # Should be at least 50% efficient


class TestSequenceSegment:
    """Test the SequenceSegment helper class."""

    def test_from_tokens_basic(self):
        """Test basic SequenceSegment creation."""
        tokens = [1, 2, 3, 4]
        segment = SequenceSegment.from_tokens(0, tokens)

        assert segment.sequence_id == 0
        assert segment.tokens == tokens
        assert segment.start_pos == 0
        assert segment.length == 4

    def test_from_tokens_with_start_pos(self):
        """Test SequenceSegment creation with custom start position."""
        tokens = [5, 6, 7]
        segment = SequenceSegment.from_tokens(1, tokens, start_pos=10)

        assert segment.sequence_id == 1
        assert segment.tokens == tokens
        assert segment.start_pos == 10
        assert segment.length == 3


class TestSequenceMapping:
    """Test the SequenceMapping dataclass."""

    def test_sequence_mapping_creation(self):
        """Test SequenceMapping creation and attributes."""
        mapping = SequenceMapping(
            original_id=5, batch_idx=2, start_pos=10, length=7
        )

        assert mapping.original_id == 5
        assert mapping.batch_idx == 2
        assert mapping.start_pos == 10
        assert mapping.length == 7


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
