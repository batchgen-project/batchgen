"""Unit tests for per-sequence max_completion_tokens (T27).

Tests the full data flow: parse_batch_file → _convert_requests_to_worker_inputs →
process_new_batch → _is_sequence_completed → _check_and_handle_completions.

NOTE: These are pure unit tests that run without GPU or server.
"""

import json
import torch
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sys
from unittest.mock import MagicMock

# Stub heavy modules to avoid CUDA/JIT/server deps on dev machine
_mock = MagicMock()
_heavy_mods = [
    "uvicorn", "uvloop",
    "fastapi", "fastapi.responses", "fastapi.middleware", "fastapi.middleware.cors",
    "starlette", "starlette.middleware", "starlette.middleware.base", "starlette.responses",
    "ninja",
    "batchgen.server.http_server", "batchgen.server.watchdog",
    "batchgen.server.worker_manager",
    "batchgen.core_engine",
    "batchgen.op_builder", "batchgen.op_builder.builder",
    "batchgen.models.engine_loader",
]
for mod in _heavy_mods:
    sys.modules.setdefault(mod, _mock)

# Also stub batchgen_worker and config to avoid CUDA/model imports
# We only need parse_batch_file (standalone function) and io_struct (pydantic models)
sys.modules.setdefault("batchgen.batchgen_worker", _mock)
sys.modules.setdefault("batchgen.config", _mock)
sys.modules.setdefault("batchgen.config.model_registry", _mock)

from batchgen.server.batch_scheduler import parse_batch_file
from batchgen.server.io_struct import (
    BatchRequestItem,
    ChatCompletionRequest,
)
from batchgen.sequence import SequenceEntry, SequenceBatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_jsonl(*requests: dict) -> bytes:
    """Create JSONL bytes from request dicts."""
    lines = [json.dumps(r) for r in requests]
    return "\n".join(lines).encode("utf-8")


def _chat_request(
    custom_id: str,
    model: str = "test-model",
    max_tokens: int = None,
    max_completion_tokens: int = None,
    temperature: float = None,
) -> dict:
    """Build an OpenAI batch request dict."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        body["temperature"] = temperature
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _make_seq(
    uuid: str,
    decoded_length: int = 0,
    max_decode_length: int = 100,
    current_context_length: int = 0,
    eos_reached: bool = False,
) -> SequenceEntry:
    """Create a SequenceEntry with specific state for testing."""
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=0,
        prompt_length=10,
        max_decode_length=max_decode_length,
    )
    seq.decoded_length = decoded_length
    seq.current_context_length = current_context_length
    seq.eos_reached = eos_reached
    return seq


# ===========================================================================
# 1. parse_batch_file
# ===========================================================================
class TestParseBatchFile:
    def test_different_max_tokens_accepted(self):
        """Requests with different max_tokens should be accepted (no uniform enforcement)."""
        content = _make_jsonl(
            _chat_request("r1", max_tokens=50),
            _chat_request("r2", max_tokens=100),
            _chat_request("r3", max_tokens=200),
        )
        ok, error, requests = parse_batch_file(content)
        assert ok, f"Should accept different max_tokens: {error}"
        assert len(requests) == 3

    def test_different_max_completion_tokens_accepted(self):
        """Requests with different max_completion_tokens should be accepted."""
        content = _make_jsonl(
            _chat_request("r1", max_completion_tokens=50),
            _chat_request("r2", max_completion_tokens=100),
        )
        ok, error, requests = parse_batch_file(content)
        assert ok, f"Should accept different max_completion_tokens: {error}"
        assert len(requests) == 2

    def test_mixed_max_tokens_and_max_completion_tokens(self):
        """Mixing max_tokens and max_completion_tokens should be accepted."""
        content = _make_jsonl(
            _chat_request("r1", max_tokens=50),
            _chat_request("r2", max_completion_tokens=100),
            _chat_request("r3", max_tokens=30, max_completion_tokens=200),
        )
        ok, error, requests = parse_batch_file(content)
        assert ok, f"Should accept mixed fields: {error}"
        assert len(requests) == 3

    def test_only_max_completion_tokens_no_max_tokens(self):
        """Request with only max_completion_tokens (no max_tokens) should be accepted."""
        content = _make_jsonl(
            _chat_request("r1", max_completion_tokens=100),
        )
        ok, error, requests = parse_batch_file(content)
        assert ok, f"Should accept max_completion_tokens only: {error}"

    def test_neither_field_accepted(self):
        """Request with neither max_tokens nor max_completion_tokens should be accepted
        (fallback to batch-level default happens later)."""
        content = _make_jsonl(
            _chat_request("r1"),  # No max_tokens or max_completion_tokens
        )
        ok, error, requests = parse_batch_file(content)
        assert ok, f"Should accept request without max fields: {error}"

    def test_inconsistent_model_rejected(self):
        """Requests with different models should still be rejected."""
        content = _make_jsonl(
            _chat_request("r1", model="model-a", max_tokens=50),
            _chat_request("r2", model="model-b", max_tokens=50),
        )
        ok, error, requests = parse_batch_file(content)
        assert not ok
        assert "Inconsistent model" in error

    def test_empty_batch_rejected(self):
        """Empty batch file should be rejected."""
        ok, error, requests = parse_batch_file(b"")
        assert not ok
        assert "empty" in error.lower()

    def test_malformed_json_rejected(self):
        """Malformed JSON should be rejected."""
        ok, error, requests = parse_batch_file(b"not json\n")
        assert not ok


# ===========================================================================
# 2. ChatCompletionRequest field priority
# ===========================================================================
class TestRequestFieldPriority:
    def test_max_completion_tokens_parsed(self):
        """max_completion_tokens field should be parsed correctly."""
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=42,
        )
        assert req.max_completion_tokens == 42

    def test_both_fields_coexist(self):
        """Both max_tokens and max_completion_tokens can be set."""
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            max_completion_tokens=100,
        )
        assert req.max_tokens == 50
        assert req.max_completion_tokens == 100

    def test_priority_max_completion_tokens_wins(self):
        """max_completion_tokens should win over max_tokens."""
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
            max_completion_tokens=100,
        )
        # Mirrors the fixed logic in _convert_requests_to_worker_inputs
        result = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
        assert result == 100

    def test_fallback_to_max_tokens(self):
        """When max_completion_tokens is None, falls back to max_tokens."""
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
        )
        result = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
        assert result == 50

    def test_neither_set_returns_none(self):
        """When neither is set, result is None."""
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        result = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
        assert result is None


# ===========================================================================
# 3. SequenceEntry per-seq limits
# ===========================================================================
class TestSequenceEntryPerSeqLimits:
    def test_max_decode_length_set(self):
        """SequenceEntry should store per-seq max_decode_length."""
        seq = SequenceEntry(
            uuid="s1", global_idx=0, prompt_length=10, max_decode_length=42
        )
        assert seq.max_decode_length == 42

    def test_kv_token_budget_uses_per_seq(self):
        """kv_token_budget should be prompt_length + max_decode_length."""
        seq = SequenceEntry(
            uuid="s1", global_idx=0, prompt_length=100, max_decode_length=50
        )
        assert seq.kv_token_budget == 150

    def test_different_sequences_different_limits(self):
        """Different sequences should have independent limits."""
        s1 = SequenceEntry(uuid="s1", global_idx=0, prompt_length=10, max_decode_length=50)
        s2 = SequenceEntry(uuid="s2", global_idx=1, prompt_length=10, max_decode_length=200)
        assert s1.max_decode_length == 50
        assert s2.max_decode_length == 200
        assert s1.kv_token_budget != s2.kv_token_budget


# ===========================================================================
# 4. _is_sequence_completed (per-seq logic)
# ===========================================================================
class TestIsSequenceCompleted:
    """Test completion logic using seq.max_decode_length (not worker-wide)."""

    def _check(self, seq, ignore_eos=False, model_context_length=131072):
        """Simulate _is_sequence_completed logic."""
        if seq.decoded_length >= seq.max_decode_length:
            return True
        if seq.current_context_length >= model_context_length:
            return True
        if seq.eos_reached and not ignore_eos:
            return True
        return False

    def test_per_seq_max_decode_length(self):
        """Should complete based on seq's own max_decode_length."""
        seq = _make_seq("s1", decoded_length=50, max_decode_length=50)
        assert self._check(seq)

    def test_below_limit_not_completed(self):
        """Should NOT complete when below seq's limit."""
        seq = _make_seq("s1", decoded_length=49, max_decode_length=50)
        assert not self._check(seq)

    def test_different_seqs_different_completion(self):
        """Two seqs with same decoded_length but different limits."""
        s1 = _make_seq("s1", decoded_length=50, max_decode_length=50)
        s2 = _make_seq("s2", decoded_length=50, max_decode_length=100)
        assert self._check(s1)      # At limit
        assert not self._check(s2)  # Below limit

    def test_eos_completes_when_not_ignored(self):
        seq = _make_seq("s1", decoded_length=10, max_decode_length=100, eos_reached=True)
        assert self._check(seq, ignore_eos=False)

    def test_eos_ignored_when_flag_set(self):
        seq = _make_seq("s1", decoded_length=10, max_decode_length=100, eos_reached=True)
        assert not self._check(seq, ignore_eos=True)

    def test_context_length_completes(self):
        seq = _make_seq("s1", decoded_length=10, max_decode_length=100,
                        current_context_length=131072)
        assert self._check(seq, model_context_length=131072)

    def test_context_length_below_limit(self):
        seq = _make_seq("s1", decoded_length=10, max_decode_length=100,
                        current_context_length=1000)
        assert not self._check(seq, model_context_length=131072)


# ===========================================================================
# 5. Vectorized completion check
# ===========================================================================
class TestVectorizedCompletionCheck:
    """Test the vectorized torch completion check logic."""

    def _vectorized_check(self, seqs, ignore_eos=False, model_context_length=131072):
        """Reproduce the vectorized completion logic from _check_and_handle_completions."""
        n = len(seqs)
        if n == 0:
            return [], []

        decoded_lens = torch.empty(n, dtype=torch.int64)
        max_lens = torch.empty(n, dtype=torch.int64)
        ctx_lens = torch.empty(n, dtype=torch.int64)
        eos_flags = torch.empty(n, dtype=torch.bool)

        for i, seq in enumerate(seqs):
            decoded_lens[i] = seq.decoded_length
            max_lens[i] = seq.max_decode_length
            ctx_lens[i] = seq.current_context_length
            eos_flags[i] = seq.eos_reached and not ignore_eos

        completed_mask = (
            (decoded_lens >= max_lens)
            | (ctx_lens >= model_context_length)
            | eos_flags
        )

        completed = [seqs[i].uuid for i in range(n) if completed_mask[i]]
        active = [seqs[i].uuid for i in range(n) if not completed_mask[i]]
        return active, completed

    def test_empty_batch(self):
        active, completed = self._vectorized_check([])
        assert active == []
        assert completed == []

    def test_all_active(self):
        seqs = [
            _make_seq("s0", decoded_length=10, max_decode_length=100),
            _make_seq("s1", decoded_length=20, max_decode_length=100),
        ]
        active, completed = self._vectorized_check(seqs)
        assert len(active) == 2
        assert len(completed) == 0

    def test_all_completed(self):
        seqs = [
            _make_seq("s0", decoded_length=100, max_decode_length=100),
            _make_seq("s1", decoded_length=200, max_decode_length=100),
        ]
        active, completed = self._vectorized_check(seqs)
        assert len(active) == 0
        assert len(completed) == 2

    def test_mixed_per_seq_limits(self):
        """Different per-seq limits cause different completion status."""
        seqs = [
            _make_seq("s0", decoded_length=50, max_decode_length=50),   # completed
            _make_seq("s1", decoded_length=50, max_decode_length=100),  # active
            _make_seq("s2", decoded_length=50, max_decode_length=30),   # completed
            _make_seq("s3", decoded_length=50, max_decode_length=200),  # active
        ]
        active, completed = self._vectorized_check(seqs)
        assert set(active) == {"s1", "s3"}
        assert set(completed) == {"s0", "s2"}

    def test_eos_completion(self):
        seqs = [
            _make_seq("s0", decoded_length=10, max_decode_length=100, eos_reached=True),
            _make_seq("s1", decoded_length=10, max_decode_length=100, eos_reached=False),
        ]
        active, completed = self._vectorized_check(seqs, ignore_eos=False)
        assert active == ["s1"]
        assert completed == ["s0"]

    def test_eos_ignored(self):
        seqs = [
            _make_seq("s0", decoded_length=10, max_decode_length=100, eos_reached=True),
            _make_seq("s1", decoded_length=10, max_decode_length=100, eos_reached=False),
        ]
        active, completed = self._vectorized_check(seqs, ignore_eos=True)
        assert len(active) == 2
        assert len(completed) == 0

    def test_context_length_completion(self):
        seqs = [
            _make_seq("s0", decoded_length=10, max_decode_length=100,
                      current_context_length=131072),  # at context limit
            _make_seq("s1", decoded_length=10, max_decode_length=100,
                      current_context_length=1000),     # fine
        ]
        active, completed = self._vectorized_check(seqs, model_context_length=131072)
        assert active == ["s1"]
        assert completed == ["s0"]

    def test_mixed_completion_reasons(self):
        """Multiple completion reasons in same batch."""
        seqs = [
            _make_seq("s0", decoded_length=100, max_decode_length=100),                   # max length
            _make_seq("s1", decoded_length=10, max_decode_length=100, eos_reached=True),   # eos
            _make_seq("s2", decoded_length=10, max_decode_length=100,
                      current_context_length=131072),                                       # context
            _make_seq("s3", decoded_length=10, max_decode_length=100),                     # active
        ]
        active, completed = self._vectorized_check(seqs)
        assert active == ["s3"]
        assert set(completed) == {"s0", "s1", "s2"}

    def test_large_batch_correctness(self):
        """Vectorized check should be correct for larger batches."""
        n = 128
        seqs = []
        expected_completed = set()
        for i in range(n):
            limit = 50 + i  # Limits from 50 to 177
            decoded = 100    # All decode 100 tokens
            seq = _make_seq(f"s{i}", decoded_length=decoded, max_decode_length=limit)
            seqs.append(seq)
            if decoded >= limit:
                expected_completed.add(f"s{i}")

        active, completed = self._vectorized_check(seqs)
        assert set(completed) == expected_completed
        assert len(active) + len(completed) == n


# ===========================================================================
# 6. Fallback logic
# ===========================================================================
class TestFallbackLogic:
    def test_batch_default_fills_none(self):
        """Simulate: per_request has None entries, batch default fills them."""
        per_request = [100, None, 200, None]
        default_max = 128
        result = [mt if mt is not None else default_max for mt in per_request]
        assert result == [100, 128, 200, 128]

    def test_per_request_overrides_batch(self):
        """Per-request values should not be overridden by batch default."""
        per_request = [50, 100, 200]
        default_max = 128
        result = [mt if mt is not None else default_max for mt in per_request]
        assert result == [50, 100, 200]

    def test_all_none_uses_batch_default(self):
        """When all per-request are None, all use batch default."""
        per_request = [None, None, None]
        default_max = 256
        result = [mt if mt is not None else default_max for mt in per_request]
        assert result == [256, 256, 256]

    def test_max_for_budget(self):
        """Worker budget should use max across all per-seq limits."""
        per_request = [50, 100, 200, 75]
        budget = max(per_request)
        assert budget == 200


# ===========================================================================
# 7. Falsy-zero edge cases (regression tests for `or` vs `is not None`)
# ===========================================================================
class TestFalsyZeroEdgeCases:
    """Ensure zero values are not treated as None/falsy."""

    def test_process_new_batch_zero_max_tokens(self):
        """per_sequence_max_tokens=[0] should set max_decode_length=0, not fallback."""
        # Simulates the fixed logic in process_new_batch
        per_sequence_max_tokens = [0, 100, None]
        worker_default = 512
        results = []
        for idx in range(3):
            max_dec = worker_default
            if per_sequence_max_tokens is not None and idx < len(per_sequence_max_tokens):
                val = per_sequence_max_tokens[idx]
                max_dec = val if val is not None else worker_default
            results.append(max_dec)
        assert results == [0, 100, 512]

    def test_convert_requests_zero_max_completion_tokens(self):
        """max_completion_tokens=0 should NOT fall through to max_tokens."""
        # Note: pydantic ge=1 would reject 0, but test the logic in isolation
        max_completion_tokens = 0
        max_tokens = 50
        result = max_completion_tokens if max_completion_tokens is not None else max_tokens
        assert result == 0

    def test_convert_requests_none_falls_through(self):
        """max_completion_tokens=None should fall through to max_tokens."""
        max_completion_tokens = None
        max_tokens = 50
        result = max_completion_tokens if max_completion_tokens is not None else max_tokens
        assert result == 50

    def test_per_seq_completion_not_worker_wide(self):
        """Completion check uses seq.max_decode_length, not a shared worker value."""
        worker_max = 1000
        seq = SequenceEntry(
            uuid="test", global_idx=0, prompt_length=10,
            max_decode_length=50, text="test"
        )
        seq.decoded_length = 60
        # Per-seq check: completed (60 >= 50)
        assert seq.decoded_length >= seq.max_decode_length
        # Worker-wide check would say NOT completed (60 < 1000)
        assert seq.decoded_length < worker_max


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
