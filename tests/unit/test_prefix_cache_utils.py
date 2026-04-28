from batchgen.prefix_cache_utils import clear_rank_cache_if_prefix_evicted


class _Stats:
    def __init__(self, eviction_epoch: int):
        self.eviction_epoch = eviction_epoch


class _WorkerView:
    def __init__(self, eviction_epoch: int):
        self._eviction_epoch = eviction_epoch

    def get_prefix_cache_stats(self):
        return _Stats(self._eviction_epoch)


def test_rank_cache_not_cleared_without_epoch_change():
    cache = {11: 0, 22: 1}
    epoch = clear_rank_cache_if_prefix_evicted(
        enable_prefix_reuse=True,
        worker_view=_WorkerView(3),
        prompt_rank_cache=cache,
        current_epoch=3,
        rank=0,
    )
    assert epoch == 3
    assert cache == {11: 0, 22: 1}


def test_rank_cache_cleared_after_prefix_eviction_epoch_change():
    cache = {11: 0, 22: 1}
    epoch = clear_rank_cache_if_prefix_evicted(
        enable_prefix_reuse=True,
        worker_view=_WorkerView(4),
        prompt_rank_cache=cache,
        current_epoch=3,
        rank=0,
    )
    assert epoch == 4
    assert cache == {}


def test_rank_cache_unchanged_when_prefix_reuse_disabled():
    cache = {11: 0}
    epoch = clear_rank_cache_if_prefix_evicted(
        enable_prefix_reuse=False,
        worker_view=_WorkerView(4),
        prompt_rank_cache=cache,
        current_epoch=3,
        rank=0,
    )
    assert epoch == 3
    assert cache == {11: 0}
