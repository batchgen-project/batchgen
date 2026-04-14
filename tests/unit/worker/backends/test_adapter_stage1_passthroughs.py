"""Unit tests for the Phase 2.8.1c boundary adapter passthroughs.

Covers the four thin adapter methods added in commit 1c so the later
boundary modules (finalize, executor) can route infrastructure calls
through ``LegacyInfraBackend`` without reaching into
``BatchGenWorker`` directly.

  * ``set_num_tokens_per_rank(n)`` + ``set_rank_token_counts(tensor)``
    → forward to ``parallel_manager``.
  * ``host_paged_kv_worker_view()`` → forward to
    ``core_engine.host_paged_kv_worker_view``; returns ``None`` when
    absent.
  * ``report_chunk_sizer_completion(decoded_length)`` → forward to
    ``adaptive_chunk_sizer.report_completion``; silent no-op otherwise.

Each test exercises either the ``LegacyWorkerBackend`` production wrap
against a plain mock worker, or the ``FakeLegacyBackend`` surface used
by the boundary/decode tests. The production wrap test confirms the
passthrough body hits the exact underscore-prefixed attribute we
planned for in the addendum (no drift between Protocol docstring and
implementation).
"""

from __future__ import annotations

import types

from batchgen.worker.backends.legacy_adapter import LegacyWorkerBackend
from tests.unit.worker.fakes import FakeLegacyBackend


# ---------------------------------------------------------------------------
# FakeLegacyBackend records calls
# ---------------------------------------------------------------------------


class TestFakeLegacyBackendStage1:
    def test_set_num_tokens_per_rank_is_recorded(self) -> None:
        fake = FakeLegacyBackend()
        fake.set_num_tokens_per_rank(64)
        assert ("set_num_tokens_per_rank", (64,), {}) in fake.calls

    def test_set_rank_token_counts_is_recorded(self) -> None:
        fake = FakeLegacyBackend()
        counts = object()  # opaque; the fake does not introspect
        fake.set_rank_token_counts(counts)
        assert fake.calls[-1] == ("set_rank_token_counts", (counts,), {})

    def test_host_paged_kv_worker_view_returns_preset(self) -> None:
        fake = FakeLegacyBackend()
        view = types.SimpleNamespace(name="fake_worker_view")
        fake._host_paged_kv_worker_view = view  # type: ignore[attr-defined]
        assert fake.host_paged_kv_worker_view() is view

    def test_host_paged_kv_worker_view_default_none(self) -> None:
        fake = FakeLegacyBackend()
        assert fake.host_paged_kv_worker_view() is None

    def test_report_chunk_sizer_completion_records_length(self) -> None:
        fake = FakeLegacyBackend()
        fake.report_chunk_sizer_completion(57)
        assert ("report_chunk_sizer_completion", (57,), {}) in fake.calls


# ---------------------------------------------------------------------------
# LegacyWorkerBackend production wrap
# ---------------------------------------------------------------------------


class _StubParallelManager:
    """Records the two new parallel_manager entry points."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def set_num_tokens_per_rank(self, n: int) -> None:
        self.calls.append(("set_num_tokens_per_rank", (n,)))

    def set_rank_token_counts(self, counts: object) -> None:
        self.calls.append(("set_rank_token_counts", (counts,)))


class _StubAdaptiveSizer:
    def __init__(self) -> None:
        self.completions: list[int] = []

    def report_completion(self, decoded_length: int) -> None:
        self.completions.append(decoded_length)


def _worker(
    *,
    view: object | None = None,
    sizer: object | None = None,
) -> object:
    """Build a minimal worker stub exposing just the attributes the
    Stage 1 passthroughs read. ``LegacyWorkerBackend`` does not hold
    a reference to any protected attribute beyond what it needs; this
    stub carries exactly those."""

    return types.SimpleNamespace(
        parallel_manager=_StubParallelManager(),
        core_engine=types.SimpleNamespace(host_paged_kv_worker_view=view),
        adaptive_chunk_sizer=sizer,
    )


class TestProductionAdapterStage1:
    def test_set_num_tokens_per_rank_forwards(self) -> None:
        worker = _worker()
        adapter = LegacyWorkerBackend(worker)  # type: ignore[arg-type]
        adapter.set_num_tokens_per_rank(12)
        assert worker.parallel_manager.calls == [
            ("set_num_tokens_per_rank", (12,))
        ]

    def test_set_rank_token_counts_forwards(self) -> None:
        worker = _worker()
        adapter = LegacyWorkerBackend(worker)  # type: ignore[arg-type]
        tensor = object()
        adapter.set_rank_token_counts(tensor)  # type: ignore[arg-type]
        assert worker.parallel_manager.calls == [
            ("set_rank_token_counts", (tensor,))
        ]

    def test_host_paged_kv_worker_view_returns_engine_attribute(self) -> None:
        view = types.SimpleNamespace(kind="worker_view")
        worker = _worker(view=view)
        adapter = LegacyWorkerBackend(worker)  # type: ignore[arg-type]
        assert adapter.host_paged_kv_worker_view() is view

    def test_host_paged_kv_worker_view_missing_returns_none(self) -> None:
        """When ``core_engine.host_paged_kv_worker_view`` was never set
        we return ``None`` so the caller can skip host-view operations
        without raising."""
        worker = types.SimpleNamespace(
            parallel_manager=_StubParallelManager(),
            core_engine=types.SimpleNamespace(),  # no attribute
            adaptive_chunk_sizer=None,
        )
        adapter = LegacyWorkerBackend(worker)  # type: ignore[arg-type]
        assert adapter.host_paged_kv_worker_view() is None

    def test_report_chunk_sizer_completion_forwards(self) -> None:
        sizer = _StubAdaptiveSizer()
        adapter = LegacyWorkerBackend(_worker(sizer=sizer))  # type: ignore[arg-type]
        adapter.report_chunk_sizer_completion(42)
        assert sizer.completions == [42]

    def test_report_chunk_sizer_completion_noop_when_sizer_absent(self) -> None:
        adapter = LegacyWorkerBackend(_worker(sizer=None))  # type: ignore[arg-type]
        # Should not raise.
        adapter.report_chunk_sizer_completion(42)
