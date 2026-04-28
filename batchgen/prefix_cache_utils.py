import logging
from typing import MutableMapping, Optional


def clear_rank_cache_if_prefix_evicted(
    *,
    enable_prefix_reuse: bool,
    worker_view: object,
    prompt_rank_cache: MutableMapping[int, int],
    current_epoch: int,
    rank: int,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Clear stale prompt->rank affinity after prefix cache eviction.

    The rank cache is an optimization only. If eviction removes an indexed
    prefix page, stale entries must be dropped so later requests can discover
    whichever rank still has a useful prefix entry.
    """
    if not enable_prefix_reuse or worker_view is None:
        return current_epoch
    try:
        stats = worker_view.get_prefix_cache_stats()
        eviction_epoch = int(getattr(stats, "eviction_epoch", 0))
    except Exception:
        return current_epoch
    if eviction_epoch == current_epoch:
        return current_epoch

    cached_entries = len(prompt_rank_cache)
    prompt_rank_cache.clear()
    if cached_entries:
        (logger or logging.getLogger(__name__)).info(
            "Rank %s prefix reuse rank cache cleared after prefix eviction "
            "(eviction_epoch=%d, entries=%d)",
            rank,
            eviction_epoch,
            cached_entries,
        )
    return eviction_epoch
