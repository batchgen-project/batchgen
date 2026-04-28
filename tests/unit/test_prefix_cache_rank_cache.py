from batchgen.batchgen_worker import BatchGenWorker


class _Stats:
    eviction_epoch = 3


class _WorkerView:
    def get_prefix_cache_stats(self):
        return _Stats()


class _CoreEngine:
    host_paged_kv_worker_view = _WorkerView()


def test_prefix_reuse_rank_cache_clears_on_eviction_epoch_change():
    worker = object.__new__(BatchGenWorker)
    worker.enable_prefix_reuse = True
    worker.core_engine = _CoreEngine()
    worker.rank = 0
    worker._prefix_reuse_prompt_rank_cache = {11: 1, 22: 2}
    worker._prefix_reuse_rank_cache_epoch = 2

    worker._maybe_clear_prefix_reuse_rank_cache_after_eviction()

    assert worker._prefix_reuse_prompt_rank_cache == {}
    assert worker._prefix_reuse_rank_cache_epoch == 3


def test_prefix_reuse_rank_cache_kept_when_epoch_unchanged():
    worker = object.__new__(BatchGenWorker)
    worker.enable_prefix_reuse = True
    worker.core_engine = _CoreEngine()
    worker.rank = 0
    worker._prefix_reuse_prompt_rank_cache = {11: 1}
    worker._prefix_reuse_rank_cache_epoch = 3

    worker._maybe_clear_prefix_reuse_rank_cache_after_eviction()

    assert worker._prefix_reuse_prompt_rank_cache == {11: 1}
