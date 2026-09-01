from batchgen.timing import TimingStats


def test_host_timed_preserves_layer_identity():
    timer = TimingStats("test", "prefill", [])
    timer.enable()

    with timer.host_timed("weight_wait", layer_idx=17):
        pass

    assert len(timer._records) == 1
    record = timer._records[0]
    assert record.op_name == "host:weight_wait"
    assert record.layer_idx == 17
    assert record.elapsed_ms >= 0.0


def test_prefill_timer_uses_phase_specific_output_controls(monkeypatch):
    monkeypatch.setenv("BATCHGEN_DECODE_TIMING_CSV", "/tmp/decode.csv")
    monkeypatch.setenv("BATCHGEN_DECODE_TIMING_RANKS", "7")
    monkeypatch.setenv("BATCHGEN_DECODE_TIMING_INTERVAL", "70")
    monkeypatch.setenv("BATCHGEN_PREFILL_TIMING_CSV", "/tmp/prefill_{rank}.csv")
    monkeypatch.setenv("BATCHGEN_PREFILL_TIMING_RANKS", "1,3")
    monkeypatch.setenv("BATCHGEN_PREFILL_TIMING_INTERVAL", "9")

    timer = TimingStats("test", "prefill", [])

    assert timer._csv_path == "/tmp/prefill_{rank}.csv"
    assert timer._emit_ranks == {1, 3}
    assert timer._interval == 9


def test_csv_path_expands_rank_and_creates_parent(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "BATCHGEN_PREFILL_TIMING_CSV",
        str(tmp_path / "nested" / "prefill_rank_{rank}.csv"),
    )
    monkeypatch.setenv("BATCHGEN_PREFILL_TIMING_RANKS", "4")
    monkeypatch.setattr("batchgen.timing._current_rank", lambda: 4)
    timer = TimingStats("test", "prefill", [])
    timer.enable()
    timer.record("host:probe", 2, 1.25)

    timer.log_summary()

    csv_path = tmp_path / "nested" / "prefill_rank_4.csv"
    assert csv_path.read_text().splitlines() == [
        "step,rank,layer,op,call,elapsed_ms",
        "0,4,2,host:probe,0,1.250000",
    ]
