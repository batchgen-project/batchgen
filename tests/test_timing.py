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
