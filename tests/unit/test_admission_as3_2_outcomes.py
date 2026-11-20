"""CPU checks for AS3.2 seam outcome copy-back and failure state."""

from __future__ import annotations

from types import SimpleNamespace

from batchgen.worker.admission.context import MoeCountsOutcome, WaitOutcome
from tests.worker_method_source import load_worker_methods


_METHODS = load_worker_methods(
    [
        "_wait_pending_kv_append_tasks",
        "_sync_decode_moe_rank_counts",
    ],
    {},
)


class _Worker(SimpleNamespace):
    def _decode_phase_context(self):
        return "context"


def test_wait_outcome_creates_deferred_error_attribute_only_on_copy_back():
    outcome = WaitOutcome(3, ("ValueError: failed",))

    class Engine:
        def _wait_pending_kv_append_tasks(self, ctx, **kwargs):
            assert ctx == "context"
            assert kwargs["pending_tasks"] is None
            assert kwargs["pending_tensors"] is None
            assert kwargs["deferred_errors"] is None
            return outcome

    worker = _Worker(_kv_capacity_engine=Engine())
    method = _METHODS["_wait_pending_kv_append_tasks"]

    assert method(worker, defer_errors=True) == 3
    assert worker._deferred_kv_append_wait_errors == ["ValueError: failed"]


def test_moe_counts_outcome_copies_all_five_rebound_values():
    local_tensor = object()
    all_counts = object()
    outcome = MoeCountsOutcome(local_tensor, all_counts, 4, 7, all_counts)

    class Engine:
        def _sync_decode_moe_rank_counts(self, ctx, batch, **kwargs):
            assert ctx == "context"
            assert batch == [1, 2, 3, 4]
            assert kwargs["decode_local_count_tensor"] is None
            assert kwargs["decode_all_rank_counts"] is None
            assert kwargs["previous_max"] == 0
            return outcome

    worker = _Worker(
        _rank_sync_engine=Engine(),
        _decode_local_count_tensor=None,
        _decode_all_rank_counts=None,
    )
    assert _METHODS["_sync_decode_moe_rank_counts"](worker, [1, 2, 3, 4], reason="test") == 7
    assert worker._decode_local_count_tensor is local_tensor
    assert worker._decode_all_rank_counts is all_counts
    assert worker._current_decode_local_batch_size == 4
    assert worker._current_decode_max_rank_batch_size == 7
    assert worker._current_decode_rank_token_counts is all_counts
