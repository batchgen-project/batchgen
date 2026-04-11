"""Self-consistency replay tests — orchestrator determinism via trace.

Every test seeds two independent orchestrators with identical sequences
and asserts the resulting traces are byte-identical. This is the
determinism floor the plan centers on: before comparing main → new, we
have to prove the new pipeline is deterministic under replay of its own
inputs. Any non-determinism breaks cross-rank convergence in production.
"""

from __future__ import annotations

from batchgen.sequence import SequenceEntry
from batchgen.worker.config import WorkerConfig
from batchgen.worker.trace import SeqSpec
from tests.integration.worker.trace_replay.replayer import (
    ReplayResult,
    record_run,
    replay_roundtrip,
)


PAGE = SequenceEntry.PAGE_SIZE  # 64


def _config(max_pool_size: int = 0) -> WorkerConfig:
    return WorkerConfig(
        decision_frequency_pages=1,
        initial_gpu_page_buffer=32,
        extension_gpu_page_buffer=4,
        prefill_watermark_pct=70,
        eviction_watermark_pct=10,
        host_kv_total_pages=10000,
        rep_detection_enabled=False,
        preemption_enabled=True,
        ignore_eos=False,
        model_context_length=4096,
        max_pool_size=max_pool_size,
    )


def _spec(uuid: str, *, global_idx: int, text: str = "hi") -> SeqSpec:
    return SeqSpec(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=10,
        max_decode_length=PAGE,
        text=text,
    )


# ---------------------------------------------------------------------------
# Single-sequence determinism
# ---------------------------------------------------------------------------


class TestSingleSequenceDeterminism:
    def test_one_sequence_two_runs_match(self) -> None:
        result = replay_roundtrip(
            "single_seq",
            [_spec("u1", global_idx=0)],
            config=_config(),
        )
        assert result.matches, (
            f"divergences: {[(d.path, d.expected, d.actual) for d in result.divergences]}"
        )

    def test_checkpoint_count_matches(self) -> None:
        result = replay_roundtrip(
            "single_seq",
            [_spec("u1", global_idx=0)],
            config=_config(),
        )
        # initial, post_run_batch, final
        assert len(result.expected.checkpoints) == 3
        assert len(result.actual.checkpoints) == 3

    def test_final_checkpoint_shows_completed(self) -> None:
        result = replay_roundtrip(
            "single_seq",
            [_spec("u1", global_idx=0)],
            config=_config(),
        )
        # reported_uuids cumulative at final checkpoint
        assert result.expected.checkpoints[-1].reported_uuids == ("u1",)
        assert result.actual.checkpoints[-1].reported_uuids == ("u1",)


# ---------------------------------------------------------------------------
# Multi-sequence determinism
# ---------------------------------------------------------------------------


class TestMultiSequenceDeterminism:
    def test_three_sequences(self) -> None:
        result = replay_roundtrip(
            "triple",
            [
                _spec("uA", global_idx=0),
                _spec("uB", global_idx=1),
                _spec("uC", global_idx=2),
            ],
            config=_config(),
        )
        assert result.matches

    def test_five_sequences_shuffled_insertion(self) -> None:
        """Insertion order differs from global_idx order; the orchestrator
        should still see the same trace because every handler sorts by
        (global_idx, uuid)."""
        specs = [
            _spec("u2", global_idx=2),
            _spec("u0", global_idx=0),
            _spec("u4", global_idx=4),
            _spec("u1", global_idx=1),
            _spec("u3", global_idx=3),
        ]
        result = replay_roundtrip("shuffled", specs, config=_config())
        assert result.matches

    def test_final_snapshot_all_completed(self) -> None:
        specs = [_spec(f"u{i}", global_idx=i) for i in range(4)]
        result = replay_roundtrip("four", specs, config=_config())
        final = result.expected.checkpoints[-1]
        assert set(final.reported_uuids) == {"u0", "u1", "u2", "u3"}
        # Every sequence snapshot should show COMPLETED status (5)
        from batchgen.sequence import SequenceStatus

        for snap in final.state.sequences:
            assert snap.status == int(SequenceStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Empty run — trivial but catches regressions in the replayer scaffold
# ---------------------------------------------------------------------------


class TestEmptyRunDeterminism:
    def test_empty_seed_produces_matching_traces(self) -> None:
        result = replay_roundtrip("empty", [], config=_config())
        assert result.matches
        # Three checkpoints even when there's no work
        assert len(result.expected.checkpoints) == 3


# ---------------------------------------------------------------------------
# generate_persistent roundtrip
# ---------------------------------------------------------------------------


class TestGeneratePersistentDeterminism:
    def test_pool_mode_determinism(self) -> None:
        messages = [
            {
                "sequences": [
                    {"uuid": "p1", "text": "a", "max_decode_length": PAGE},
                    {"uuid": "p2", "text": "b", "max_decode_length": PAGE},
                ]
            }
        ]
        result = replay_roundtrip(
            "pool",
            specs=[],
            config=_config(max_pool_size=16),
            drive_generate_persistent=True,
            admission_messages=messages,
        )
        assert result.matches
        # Both pool sequences appear in the final reported_uuids
        assert set(result.expected.checkpoints[-1].reported_uuids) == {"p1", "p2"}


# ---------------------------------------------------------------------------
# record_run direct call
# ---------------------------------------------------------------------------


class TestRecordRunDirect:
    def test_returns_trace_with_expected_labels(self) -> None:
        trace = record_run(
            "labelled",
            [_spec("u1", global_idx=0)],
            config=_config(),
        )
        labels = [c.label for c in trace.checkpoints]
        assert labels == ["initial", "post_run_batch", "final"]

    def test_trace_name_roundtrip(self) -> None:
        trace = record_run(
            "my-trace",
            [_spec("u1", global_idx=0)],
            config=_config(),
        )
        assert trace.name == "my-trace"

    def test_initial_sequences_preserved(self) -> None:
        specs = [
            _spec("u1", global_idx=0),
            _spec("u2", global_idx=1),
        ]
        trace = record_run("seed", specs, config=_config())
        assert trace.initial_sequences == tuple(specs)
