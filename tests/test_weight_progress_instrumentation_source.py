"""CPU-only regressions for the weight daemon's progress-thread accounting.

Each rail's progress thread spins on ucp_worker_progress, so the only way it
stops making progress is losing the CPU. The loop therefore measures the gap
between its own consecutive iterations: a starved thread shows rare large
gaps while its loop count stays high, which separates "the thread was off the
CPU" from "the rail was idle". The counters are monotonic, per rail, written
only by that rail's thread, and surfaced through the summary file the daemon
already writes.

An operator can also pin the eight progress threads to named cores with
`progress_thread_cores`. The default (key absent) pins nothing and keeps the
old loop behaviour; a configured pin that cannot be applied records a daemon
failure rather than silently running unpinned.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAEMON = ROOT / "core/Weights_Storage/distributed_weight_daemon.cpp"


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _progress_spawn():
    return _section(
        DAEMON.read_text(),
        "progress_threads[rail] = std::thread([this, rail]() {",
        "// From here on only the prefetch thread may reset ring slots",
    )


def _config_block():
    return _section(
        DAEMON.read_text(),
        'if (config.contains("progress_thread_cores")) {',
        "\n        // Eight TP8 workers per node",
    )


def test_every_iteration_is_counted_and_bucketed_inside_the_loop():
    spawn = _progress_spawn()
    loop = _section(spawn, "while (!stop.load()) {", "});")
    # The gap is measured across the whole iteration, including the
    # ucp_worker_progress call and the yield, from one monotonic clock.
    progressed = loop.index("ucp_worker_progress(rails[rail].worker)")
    measured = loop.index("const Clock::time_point now = Clock::now();")
    assert progressed < measured
    assert "previous = now;" in loop
    assert "gaps.loops.fetch_add(1, std::memory_order_relaxed);" in loop
    # Buckets nest: a 50ms gap also counts as 1ms and 10ms.
    first = loop.index("if (gap_us >= 1000) {")
    second = loop.index("if (gap_us >= 10000) {", first)
    third = loop.index("if (gap_us >= 50000) {", second)
    assert first < second < third
    # The running maximum is kept lock-free, only advances, and the loop
    # never blocks on it: steady state is one relaxed load.
    assert "compare_exchange_weak" in loop
    # One condition guards both entry and every CAS retry, so a smaller gap
    # never writes and the steady state is a single relaxed load.
    assert re.search(r"while \(gap_us > seen &&", loop)
    assert "mutex" not in loop


def test_the_counters_are_per_rail_atomics_with_a_zero_default():
    daemon = DAEMON.read_text()
    struct = _section(daemon, "struct alignas(64) ProgressGaps {",
                      "std::array<ProgressGaps, kRails> progress_gaps;")
    for field in ("loops", "gap_ge_1ms", "gap_ge_10ms", "gap_ge_50ms",
                  "max_gap_us"):
        assert f"std::atomic<std::uint64_t> {field}{{0}};" in struct
    assert "std::array<ProgressGaps, kRails> progress_gaps;" in daemon
    # Each rail's counters own their cache line: every iteration writes
    # `loops`, and pinned neighbours must not ping-pong a shared line.
    assert "struct alignas(64) ProgressGaps {" in daemon
    # Written by the progress threads, read by the summary writer, nowhere
    # else: no reset site exists, so two summaries subtract cleanly.
    assert not re.search(r"progress_gaps\[[^\]]*\]\s*=", daemon)
    assert "gaps.loops.fetch_add" in daemon


def test_the_summary_carries_one_entry_per_rail_and_the_pin_list():
    daemon = DAEMON.read_text()
    summary = _section(daemon, "void WriteSummary() {", "const std::string temporary")
    assert 'summary["progress_gaps_by_rail"] = progress_by_rail;' in summary
    assert 'summary["progress_thread_cores"] = progress_thread_cores;' in summary
    block = _section(summary, "nlohmann::json progress_by_rail", 'summary["progress_gaps_by_rail"]')
    assert "for (int rail = 0; rail < kRails; ++rail) {" in block
    for key in ("loops", "gap_ge_1ms", "gap_ge_10ms", "gap_ge_50ms",
                "max_gap_us"):
        assert f'{{"{key}",' in block or f'{{"{key}"' in block


def test_pinning_is_config_gated_and_absent_config_pins_nothing():
    daemon = DAEMON.read_text()
    spawn = _progress_spawn()
    # The default is an empty list, and the spawn path checks it before any
    # affinity call, so an absent key leaves the thread untouched.
    assert "std::vector<int> progress_thread_cores;" in daemon
    guard = spawn.index("if (!progress_thread_cores.empty()) {")
    pinned = spawn.index("pthread_setaffinity_np(")
    loop = spawn.index("while (!stop.load()) {")
    assert guard < pinned < loop
    # The affinity call is the only one in the daemon and targets the thread
    # itself with exactly one core.
    assert daemon.count("pthread_setaffinity_np(") == 1
    assert "CPU_SET(progress_thread_cores[rail], &cores);" in spawn


def test_a_failed_pin_records_a_failure_and_stops_the_thread():
    spawn = _progress_spawn()
    failed = spawn.index("RecordFailure(")
    returned = spawn.index("return;", failed)
    loop = spawn.index("while (!stop.load()) {")
    # Fail-loud, then never enter the loop unpinned.
    assert failed < returned < loop
    diagnostic = spawn[failed:returned]
    assert "std::to_string(rail)" in diagnostic
    assert "std::to_string(progress_thread_cores[rail])" in diagnostic
    assert "std::strerror(rc)" in diagnostic


def test_core_list_validation_fails_by_name():
    block = _config_block()
    # Exactly kRails distinct integer core ids, each a valid core.
    assert "cores.size() != static_cast<size_t>(kRails)" in block
    assert '" distinct core ids");' in block
    assert "if (!entry.is_number_integer()) {" in block
    assert "core < 0 || core >= CPU_SETSIZE" in block
    assert '" is not a valid core id");' in block
    assert '" is repeated");' in block
    # The type check precedes the value read.
    assert block.index("is_number_integer") < block.index(
        "entry.get<std::int64_t>();")
