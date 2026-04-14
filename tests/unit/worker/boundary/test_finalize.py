"""Unit tests for ``boundary/finalize.py`` — Phase 2.8.1g port.

Validates each step of the legacy ``_boundary_finalize`` port:

  * Page table rebuild fires on entry.
  * Per-rank batch-size gather uses exactly one
    ``all_gather_into_tensor`` call.
  * MoE setters fire when the gathered max > 0; skipped when max = 0.
  * Barrier fires exactly once.
  * Batch-mismatch repair rebuilds the page table a second time.
  * Page-table shape mismatch raises AssertionError (hard-fail per
    Phase 2.5 invariant).
  * Watermark trigger return value comes from the adapter.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from batchgen.worker.boundary.finalize import finalize
from batchgen.worker.state import WorkerState
from tests.unit.worker.fakes import FakeCollectiveBackend, FakeLegacyBackend


def _make_state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank, local_rank=rank, world_size=world_size, device=rank,
        torch_device=torch.device("cpu"),
    )


def _fake_gpu(
    *, is_initialized: bool = True, gpu_table_rows: int | None = None
) -> Any:
    """Build a fake gpu_manager with a page-table manager attached
    whose ``gpu_table.shape[0]`` can be asserted against."""
    if gpu_table_rows is not None:
        gpu_table = types.SimpleNamespace(shape=(gpu_table_rows,))
        page_table_mgr = types.SimpleNamespace(gpu_table=gpu_table)
    else:
        page_table_mgr = None
    return types.SimpleNamespace(
        is_initialized=is_initialized,
        _gpu_page_table_manager=page_table_mgr,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestFinalizeHappyPath:
    def test_single_rank_populates_moe_setters_and_barrier(self) -> None:
        state = _make_state(rank=0, world_size=1)
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}
        col = FakeCollectiveBackend(rank=0, world_size=1)
        gpu = _fake_gpu(gpu_table_rows=1)

        batch_out, watermark = finalize(
            state, legacy, col,
            decode_uuids=["u"], batch=[0], gpu_manager=gpu,
        )

        call_names = [c[0] for c in legacy.calls]
        # Step 1 rebuild, Step 3 MoE setters, Step 7 watermark.
        assert call_names.count("rebuild_page_table_for_batch") == 1
        assert "set_num_tokens_per_rank" in call_names
        assert "set_rank_token_counts" in call_names
        assert "check_host_kv_watermark_trigger" in call_names
        # One gather + one barrier.
        assert col.call_names().count("all_gather_into_tensor") == 1
        assert col.call_names().count("barrier") == 1
        # Fake default watermark trigger is False.
        assert watermark is False
        assert batch_out == [0]

    def test_empty_batch_skips_moe_setters(self) -> None:
        state = _make_state(rank=0, world_size=1)
        legacy = FakeLegacyBackend()
        col = FakeCollectiveBackend(rank=0, world_size=1)

        finalize(
            state, legacy, col,
            decode_uuids=[], batch=[], gpu_manager=_fake_gpu(gpu_table_rows=0),
        )
        call_names = [c[0] for c in legacy.calls]
        assert "set_num_tokens_per_rank" not in call_names
        assert "set_rank_token_counts" not in call_names

    def test_watermark_true_propagates(self) -> None:
        class _Adapter(FakeLegacyBackend):
            def check_host_kv_watermark_trigger(self) -> bool:
                self._record("check_host_kv_watermark_trigger")
                return True

        state = _make_state()
        legacy = _Adapter()
        legacy._uuid_to_local = {"u": 0}
        col = FakeCollectiveBackend(rank=0, world_size=1)
        _, watermark = finalize(
            state, legacy, col,
            decode_uuids=["u"], batch=[0], gpu_manager=_fake_gpu(gpu_table_rows=1),
        )
        assert watermark is True


# ---------------------------------------------------------------------------
# Batch mismatch repair
# ---------------------------------------------------------------------------


class TestBatchMismatchRepair:
    def test_mismatch_rewrites_batch_and_rebuilds(self) -> None:
        class _Adapter(FakeLegacyBackend):
            def get_local_indices_for_uuids(self, uuids: list[str]) -> list[int]:
                self._record("get_local_indices_for_uuids", uuids)
                # Adapter says decode_uuids map to [7], but the caller
                # passed batch=[3]. Finalize must repair to [7].
                return [7]

        state = _make_state()
        legacy = _Adapter()
        col = FakeCollectiveBackend(rank=0, world_size=1)
        gpu = _fake_gpu(gpu_table_rows=1)

        batch_out, _ = finalize(
            state, legacy, col,
            decode_uuids=["u"], batch=[3], gpu_manager=gpu,
        )
        assert batch_out == [7]
        # rebuild called twice: initial + repair.
        rebuilds = [
            c for c in legacy.calls if c[0] == "rebuild_page_table_for_batch"
        ]
        assert len(rebuilds) == 2
        assert rebuilds[0][1][0] == [3]
        assert rebuilds[1][1][0] == [7]


# ---------------------------------------------------------------------------
# Page-table shape mismatch hard-fail
# ---------------------------------------------------------------------------


class TestPageTableShapeAssertion:
    def test_shape_mismatch_raises(self) -> None:
        state = _make_state()
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}
        col = FakeCollectiveBackend(rank=0, world_size=1)
        # Page table reports 5 rows but batch is a single seq.
        gpu = _fake_gpu(gpu_table_rows=5)

        with pytest.raises(AssertionError, match="page table row count"):
            finalize(
                state, legacy, col,
                decode_uuids=["u"], batch=[0], gpu_manager=gpu,
            )

    def test_uninitialized_gpu_manager_skips_shape_check(self) -> None:
        state = _make_state()
        legacy = FakeLegacyBackend()
        legacy._uuid_to_local = {"u": 0}
        col = FakeCollectiveBackend(rank=0, world_size=1)
        gpu = _fake_gpu(is_initialized=False, gpu_table_rows=99)
        # Should NOT raise because manager is not live yet.
        finalize(
            state, legacy, col,
            decode_uuids=["u"], batch=[0], gpu_manager=gpu,
        )
