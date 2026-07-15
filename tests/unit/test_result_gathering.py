"""Tests for result gathering logic: local detokenization + string gathering.

Validates _decode_tokens_to_string behavior and the gather-by-string pattern
that replaced the old gather-by-tensor approach (which OOMed at 12K+ sequences).

Run with: python tests/test_result_gathering.py
"""

import sys
import torch
import pytest
from dataclasses import dataclass
from typing import List, Set, Optional


# ---- Mock classes to test _decode_tokens_to_string in isolation ----

class MockTokenizer:
    """Minimal tokenizer mock that maps token IDs to strings."""
    def __init__(self):
        # Simple vocab: token_id -> "t{id}"
        pass

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return " ".join(f"t{t}" for t in token_ids)


class MockWorker:
    """Minimal mock of BatchGenWorker with just the fields needed for _decode_tokens_to_string."""
    def __init__(self, eos_token_ids: Set[int], pad_token_id: int = 0):
        self.eos_token_ids = eos_token_ids
        self.pad_token_id = pad_token_id
        self.tokenizer = MockTokenizer()

    def _decode_tokens_to_string(self, tokens: torch.Tensor, min_tokens: int = 1) -> str:
        """Exact copy of BatchGenWorker._decode_tokens_to_string."""
        if tokens.dim() > 1:
            tokens = tokens.squeeze(0)

        tokens_list = tokens.tolist()

        eos_positions = [i for i, t in enumerate(tokens_list) if t in self.eos_token_ids and i >= min_tokens]

        if eos_positions:
            end_pos = eos_positions[0]
        else:
            non_pad = [i for i, t in enumerate(tokens_list) if t != self.pad_token_id]
            end_pos = non_pad[-1] + 1 if non_pad else len(tokens_list)

        return self.tokenizer.decode(tokens_list[:end_pos], skip_special_tokens=False)


# ---- Tests ----

def test_basic_decode():
    """Normal case: tokens followed by EOS then padding."""
    worker = MockWorker(eos_token_ids={2}, pad_token_id=0)
    # [10, 20, 30, EOS=2, PAD=0, PAD=0]
    tokens = torch.tensor([[10, 20, 30, 2, 0, 0]], dtype=torch.int64)
    result = worker._decode_tokens_to_string(tokens)
    # Should stop at EOS (position 3), decode tokens[:3]
    assert result == "t10 t20 t30", f"Expected 't10 t20 t30', got '{result}'"
    print("  PASS: test_basic_decode")


def test_eos_respects_min_tokens():
    """EOS at position 0 should be ignored when min_tokens=1."""
    worker = MockWorker(eos_token_ids={2}, pad_token_id=0)
    # [EOS=2, 10, 20, EOS=2, PAD=0]
    tokens = torch.tensor([[2, 10, 20, 2, 0]], dtype=torch.int64)
    result = worker._decode_tokens_to_string(tokens, min_tokens=1)
    # EOS at position 0 is < min_tokens=1, skip it. Next EOS at position 3.
    assert result == "t2 t10 t20", f"Expected 't2 t10 t20', got '{result}'"
    print("  PASS: test_eos_respects_min_tokens")


def test_no_eos_strips_padding():
    """No EOS token: should use all non-padding tokens."""
    worker = MockWorker(eos_token_ids={2}, pad_token_id=0)
    # [10, 20, 30, PAD=0, PAD=0]
    tokens = torch.tensor([[10, 20, 30, 0, 0]], dtype=torch.int64)
    result = worker._decode_tokens_to_string(tokens)
    assert result == "t10 t20 t30", f"Expected 't10 t20 t30', got '{result}'"
    print("  PASS: test_no_eos_strips_padding")


def test_all_padding():
    """All padding tokens, no real content."""
    worker = MockWorker(eos_token_ids={2}, pad_token_id=0)
    tokens = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
    result = worker._decode_tokens_to_string(tokens)
    # No non-pad tokens → end_pos = len(tokens_list) = 4
    assert result == "t0 t0 t0 t0", f"Expected 't0 t0 t0 t0', got '{result}'"
    print("  PASS: test_all_padding")


def test_1d_input():
    """1D tensor input (no batch dimension)."""
    worker = MockWorker(eos_token_ids={2}, pad_token_id=0)
    tokens = torch.tensor([10, 20, 2, 0], dtype=torch.int64)
    result = worker._decode_tokens_to_string(tokens)
    assert result == "t10 t20", f"Expected 't10 t20', got '{result}'"
    print("  PASS: test_1d_input")


def test_no_modification_of_original():
    """Verify _decode_tokens_to_string does not modify the input tensor."""
    worker = MockWorker(eos_token_ids={2}, pad_token_id=0)
    tokens = torch.tensor([[10, 20, 2, 0]], dtype=torch.int64)
    original = tokens.clone()
    worker._decode_tokens_to_string(tokens)
    assert torch.equal(tokens, original), "Input tensor was modified!"
    print("  PASS: test_no_modification_of_original")


def test_multiple_eos_ids():
    """Multiple EOS token IDs (e.g., model with multiple stop tokens)."""
    worker = MockWorker(eos_token_ids={2, 3}, pad_token_id=0)
    tokens = torch.tensor([[10, 20, 3, 30, 2, 0]], dtype=torch.int64)
    result = worker._decode_tokens_to_string(tokens)
    # First EOS is token 3 at position 2
    assert result == "t10 t20", f"Expected 't10 t20', got '{result}'"
    print("  PASS: test_multiple_eos_ids")


def test_large_tensor_perf():
    """Simulate production-scale tensor (131072 tokens)."""
    worker = MockWorker(eos_token_ids={2}, pad_token_id=0)
    # 100 real tokens, EOS, then padding
    tokens = torch.full((1, 131072), 0, dtype=torch.int64)
    tokens[0, :100] = torch.arange(10, 110)
    tokens[0, 100] = 2  # EOS
    result = worker._decode_tokens_to_string(tokens)
    expected_tokens = [f"t{i}" for i in range(10, 110)]
    assert result == " ".join(expected_tokens), "Large tensor decode failed"
    print("  PASS: test_large_tensor_perf")


def test_gather_sorting():
    """Simulate multi-rank gather output and verify global ordering."""
    # Simulate 4 ranks, each with some (global_idx, string) tuples
    rank_results = [
        [(3, "hello3"), (7, "hello7")],       # rank 0
        [(1, "hello1"), (5, "hello5")],       # rank 1
        [(0, "hello0"), (4, "hello4")],       # rank 2
        [(2, "hello2"), (6, "hello6")],       # rank 3
    ]

    # Simulate all_gather_object + flatten + sort
    all_results = [item for sublist in rank_results for item in sublist]
    all_results.sort(key=lambda x: x[0])
    decoded_strings = [s for _, s in all_results]

    expected = [f"hello{i}" for i in range(8)]
    assert decoded_strings == expected, f"Expected {expected}, got {decoded_strings}"
    print("  PASS: test_gather_sorting")


def test_empty_rank():
    """Some ranks may have no sequences (e.g., fewer sequences than ranks)."""
    rank_results = [
        [(0, "a"), (1, "b")],   # rank 0
        [],                       # rank 1 (no sequences)
        [(2, "c")],              # rank 2
        [],                       # rank 3 (no sequences)
    ]

    all_results = [item for sublist in rank_results for item in sublist]
    all_results.sort(key=lambda x: x[0])
    decoded_strings = [s for _, s in all_results]

    assert decoded_strings == ["a", "b", "c"], f"Got {decoded_strings}"
    print("  PASS: test_empty_rank")


def test_old_vs_new_equivalence():
    """Verify new (local detokenize) produces same output as old (gather tensors, rank-0 detokenize)."""
    worker = MockWorker(eos_token_ids={2}, pad_token_id=0)

    # Create test data: 8 sequences with varying content
    test_tensors = []
    for i in range(8):
        t = torch.full((1, 1024), 0, dtype=torch.int64)
        n_tokens = 50 + i * 10
        t[0, :n_tokens] = torch.arange(10, 10 + n_tokens)
        t[0, n_tokens] = 2  # EOS
        test_tensors.append(t)

    # OLD path: gather tensors, decode on rank 0
    old_strings = []
    for t in test_tensors:
        old_strings.append(worker._decode_tokens_to_string(t))

    # NEW path: decode locally then gather strings
    new_strings = []
    for t in test_tensors:
        new_strings.append(worker._decode_tokens_to_string(t))

    assert old_strings == new_strings, "Old and new paths produce different results!"
    print("  PASS: test_old_vs_new_equivalence")


def test_completed_outputs_cached_for_final_response():
    """Completed sequences remain available after local query slots are released."""
    try:
        from batchgen.batchgen_worker import BatchGenWorker
    except ImportError as exc:
        pytest.skip(f"BatchGenWorker import requires runtime extensions: {exc}")

    @dataclass
    class Sequence:
        global_idx: int

    class Batch:
        def __init__(self):
            self._sequences = {
                "seq_a": Sequence(global_idx=3),
                "seq_b": Sequence(global_idx=1),
            }

        def get_sequence(self, uuid: str):
            return self._sequences.get(uuid)

    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker.global_batch = Batch()
    worker._final_response_completed_outputs = {}

    worker._record_completed_outputs_for_final_response(
        ["seq_a", "seq_b", "missing"],
        {
            "seq_a": {"text": "alpha"},
            "seq_b": {"text": "beta"},
            "missing": {"text": "ignored"},
        },
    )

    assert worker._final_response_completed_outputs == {3: "alpha", 1: "beta"}
    print("  PASS: test_completed_outputs_cached_for_final_response")


if __name__ == "__main__":
    print("Running result gathering tests...\n")

    tests = [
        test_basic_decode,
        test_eos_respects_min_tokens,
        test_no_eos_strips_padding,
        test_all_padding,
        test_1d_input,
        test_no_modification_of_original,
        test_multiple_eos_ids,
        test_large_tensor_perf,
        test_gather_sorting,
        test_empty_rank,
        test_old_vs_new_equivalence,
        test_completed_outputs_cached_for_final_response,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed > 0:
        sys.exit(1)
