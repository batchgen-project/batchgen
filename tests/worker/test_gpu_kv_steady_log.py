"""Unit tests for the steady-state GPU-KV occupancy heartbeat line.

Pure CPU: `format_gpu_kv_steady_state` renders from plain host-side
counters. The line shape is a parsing contract with the benchmark
recorder (batchgen-benchmark decode-run verdict, schema v2.1), so the
recorder's regexes are copied here LITERALLY to pin the contract: if
either side changes shape, this test fails before a GPU run does.
"""

from __future__ import annotations

import re

import pytest

from batchgen.worker.step_timing import format_gpu_kv_steady_state

# --- Recorder regexes, copied verbatim from the PR #42 parser -----------
# (batchgen-benchmark scripts/decode_run.py: GPU_KV_IN_USE_RE and
# ADMISSION_QUEUE_RE). Do not "fix" or simplify these: they must stay
# byte-identical to the recorder's compiled patterns.
GPU_KV_IN_USE_RE = re.compile(
    r"\[GPU-KV\][^\n]*?pages_in_use=(?P<in_use>\d+)(?:[^\n]*?pages_total=(?P<total>\d+))?"
)
ADMISSION_QUEUE_RE = re.compile(
    r"queued_sequences=(?P<queued>\d+)"
    r"|Breaking for prefill \(watermark\) - (?P<queued2>\d+) queued"
)
# The recorder's pool-size regex must NOT match the steady-state line
# (it is reserved for the three pool-lifecycle forms).
GPU_KV_POOL_PAGES_RE = re.compile(
    r"\[GPU-KV\] (?:Initialized: [\d.]+ GB, (?P<init>\d+) pages"
    r"|rank=\d+ released the reserved KV pool's physical pages in [\d.]+s \((?P<released>\d+) pages"
    r"|rank=\d+ remapped the reserved KV pool in [\d.]+s \((?P<remapped>\d+) pages\))"
)


def test_line_shape_exact() -> None:
    line = format_gpu_kv_steady_state(1234, 5678, 42)
    assert line == "[GPU-KV] pages_in_use=1234 pages_total=5678 queued_sequences=42"


def test_recorder_parses_occupancy_and_total() -> None:
    line = format_gpu_kv_steady_state(1234, 5678, 42)
    match = GPU_KV_IN_USE_RE.search(line)
    assert match is not None
    assert int(match.group("in_use")) == 1234
    assert int(match.group("total")) == 5678


def test_recorder_parses_admission_queue_depth() -> None:
    line = format_gpu_kv_steady_state(1234, 5678, 42)
    match = ADMISSION_QUEUE_RE.search(line)
    assert match is not None
    assert int(match.group("queued")) == 42


def test_queue_depth_zero_is_logged_not_dropped() -> None:
    # Saturation needs a POSITIVE depth; an idle queue must still parse
    # as an explicit 0 observation, not vanish from the log.
    line = format_gpu_kv_steady_state(10, 20, 0)
    match = ADMISSION_QUEUE_RE.search(line)
    assert match is not None
    assert int(match.group("queued")) == 0


def test_queue_depth_omitted_when_not_visible() -> None:
    line = format_gpu_kv_steady_state(10, 20, None)
    assert line == "[GPU-KV] pages_in_use=10 pages_total=20"
    assert ADMISSION_QUEUE_RE.search(line) is None
    match = GPU_KV_IN_USE_RE.search(line)
    assert match is not None
    assert int(match.group("in_use")) == 10
    assert int(match.group("total")) == 20


def test_parses_with_logging_prefix() -> None:
    # The recorder searches full log lines; a logging-formatter prefix
    # must not break the match.
    line = (
        "2026-11-10 22:00:00,123 - INFO - "
        + format_gpu_kv_steady_state(999, 1000, 7)
    )
    match = GPU_KV_IN_USE_RE.search(line)
    assert match is not None
    assert int(match.group("in_use")) == 999
    assert int(match.group("total")) == 1000
    queued = ADMISSION_QUEUE_RE.search(line)
    assert queued is not None
    assert int(queued.group("queued")) == 7


def test_does_not_collide_with_pool_size_regex() -> None:
    # The steady-state line must not be mistaken for a pool-size report
    # (one lifetime slice reporting two "pool sizes" voids the verdict).
    line = format_gpu_kv_steady_state(1234, 5678, 42)
    assert GPU_KV_POOL_PAGES_RE.search(line) is None


def test_counters_are_rendered_as_exact_integers() -> None:
    # Exact accounting: bool is an int subtype and numpy scalars coerce,
    # but a float that would silently truncate is a caller bug.
    assert "pages_in_use=1 pages_total=2 queued_sequences=0" in (
        format_gpu_kv_steady_state(True, 2, False)
    )
    with pytest.raises((TypeError, ValueError)):
        format_gpu_kv_steady_state(1.5, 2, 0)  # type: ignore[arg-type]
