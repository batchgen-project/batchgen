from batchgen.worker.prefill import host_kv_eviction_enabled


def test_chunked_host_kv_enables_eviction_independent_of_deprecated_switch():
    assert host_kv_eviction_enabled(8192) is True
    assert host_kv_eviction_enabled(64) is True
    assert host_kv_eviction_enabled(0) is False
