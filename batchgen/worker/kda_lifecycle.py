"""The default per-sequence KDA host lifecycle (方案甲), worker-side steps.

Every sequence of a model with recurrent state goes through the host on ONE
path, with no gate: its record pages are reserved at prefill admission, the
state is written into its per-sequence record at the REAL prefill end and the
GPU slot is freed, decode admission restores the record into a slot of the
decode layout, and growth eviction or completion releases the record pages
together with the sequence's host KV.

These helpers hold the lifecycle decisions so they are testable without a
worker: the worker and the KV capacity engine resolve their state (the record
store, the trajectory store, the sequence ids) and hand it in.  A model that
keeps no recurrent state has no record store (``store is None``) and every
step is structurally empty -- that is the absence of the state itself, not a
configuration gate.

One step runs before any store exists: ``release_restart_orphans``, the pool
startup sweep over the durable record map that releases the records a dead
worker cohort left behind (its ``shared_map is None`` case is the same
structural absence).
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Tuple


def offload_prefill_states(
    store,
    free_kda_sequences,
    sequences: Sequence[Tuple[int, int]],
    *,
    rank: int,
    logger=logging,
) -> Tuple[int, ...]:
    """The wave's prefill end: write every sequence's record, free its slot.

    ``sequences`` is ``[(global_sequence_id, true_token_count), ...]`` for the
    rows THIS rank computed; every rank of an owning group reaches here with
    the same rows and writes its own unit span.  The write is the mandatory
    lifecycle step, so a failure propagates -- there is no skip.  The GPU
    slots are freed only after every row's write returned, through
    ``free_kda_sequences`` (the same release the completion path uses).
    """

    if store is None or not sequences:
        return ()
    written: list[int] = []
    for sequence_id, token_count in sequences:
        store.write_rank_units(int(sequence_id), token_count=int(token_count))
        written.append(int(sequence_id))
    free_kda_sequences(written)
    logger.info(
        "[KDA_RECORD] rank=%s prefill end: %s sequence state(s) offloaded "
        "to host records; GPU slots freed",
        rank,
        len(written),
    )
    return tuple(written)


def restore_decode_states(
    store,
    sequence_ids: Iterable[int],
    *,
    group_size: int,
    group_rank: int,
    rank: int,
    logger=logging,
) -> Tuple[int, ...]:
    """Decode admission: restore every entering sequence from its record.

    A sequence that still holds a live slot (an ON_HOLD re-entry, or a row
    already decoding) carries newer state than its record and is left alone;
    a first entry from PREFILLED holds no slot -- its prefill end freed it --
    and restores from the record into a slot of the DECODE layout.  A row
    with neither a live slot nor a record is state loss and raises by name
    inside the store.
    """

    if store is None:
        return ()
    restored: list[int] = []
    for sequence_id in sequence_ids:
        sid = int(sequence_id)
        if store.state_writer_has_slot(sid):
            continue
        store.restore_rank_units(
            sid, group_size=int(group_size), group_rank=int(group_rank)
        )
        restored.append(sid)
    if restored:
        logger.info(
            "[KDA_RECORD] rank=%s decode admission: restored %s sequence "
            "state(s) from host records",
            rank,
            len(restored),
        )
    return tuple(restored)


def release_restart_orphans(
    shared_map,
    arena_allocator,
    live_sequence_ids: Iterable[int] = (),
    *,
    rank: int,
    logger=logging,
) -> int:
    """Pool startup: release the records a dead worker cohort left behind.

    ``shared_map`` is the node's durable sequence-id -> pages map read back
    from the shared region (``None`` when nothing was ever persisted -- a
    first cohort, or a model without recurrent state).  Runs on ONE rank
    per node, after the host KV views attach and before any admission; the
    server's readiness barrier orders it before every rank's store build.

    The sweep first opens a new cohort generation, then walks every entry
    of the previous ones:

    * a sequence in ``live_sequence_ids`` keeps its entry -- the new
      cohort's stores rehydrate it, so its record stays readable.  The
      restart route today fails every in-flight batch (#490 semantics), so
      the live set is empty; the parameter is the seam a future
      keep-sequences restart passes its survivors through.
    * a dead sequence with a RETAINED record is the orphan this sweep
      exists for: its pages are prefix-resident and named by the map
      alone, so they are released back to the Host pool here, exactly
      ``pages_per_group`` per record per group.
    * a dead sequence with an unretained record only drops its entry: its
      pages were still chain-owned by the arena's pseudo row, which the
      server's cohort reconcile (``release_all_sequences``) already
      recovered before this cohort spawned -- releasing them again here
      would double-free.

    Returns the resident pages released (per group, the same unit the
    store's ``release`` reports).
    """

    if shared_map is None:
        return 0
    live = {int(sequence_id) for sequence_id in live_sequence_ids}
    generation = shared_map.bump_generation()
    released_pages = 0
    released: list[int] = []
    dropped: list[int] = []
    kept: list[int] = []
    for sid, entry in sorted(shared_map.entries().items()):
        if int(entry.generation) == int(generation):
            # Written after the bump: not this sweep's to judge.
            continue
        if int(sid) in live:
            kept.append(int(sid))
            continue
        if entry.retained:
            arena_allocator.release(entry.to_arena())
            released_pages += int(entry.pages_per_group)
            released.append(int(sid))
        else:
            dropped.append(int(sid))
        shared_map.remove(int(sid))
    if released or dropped or kept:
        logger.info(
            "[KDA_RECORD] rank=%s pool restart recovery: released %s "
            "orphaned record page(s) of %s dead sequence(s) %s; dropped %s "
            "unretained entry/entries %s (pages already recovered by the "
            "cohort reconcile); kept %s surviving record(s) %s",
            rank,
            released_pages,
            len(released),
            released,
            len(dropped),
            dropped,
            len(kept),
            kept,
        )
    return released_pages


def release_records(
    store,
    owned_sequence_ids: Iterable[int],
    mirrored_sequence_ids: Iterable[int] = (),
    *,
    rank: int,
    where: str,
    logger=logging,
) -> int:
    """Eviction/completion: release owned records, drop mirrored ones.

    ``owned_sequence_ids`` are rows whose host KV this rank releases (the
    host-KV owner); their record pages go back to the Host pool here, in the
    same step as the KV chain, so eviction's benefit arithmetic
    (``kda_record_pages``) is truthful.  ``mirrored_sequence_ids`` are the
    G-1 followers' rows: rank-local bookkeeping only.  Returns the pages
    released.  A missing owned record raises by name inside the store -- a
    release seam that cannot find the record it accounted for is a bug.
    """

    if store is None:
        return 0
    released_pages = 0
    released_ids: list[int] = []
    for sequence_id in owned_sequence_ids:
        released_pages += store.release(int(sequence_id))
        released_ids.append(int(sequence_id))
    dropped = 0
    for sequence_id in mirrored_sequence_ids:
        if store.drop_mirror(int(sequence_id)):
            dropped += 1
    if released_ids or dropped:
        logger.info(
            "[KDA_RECORD] rank=%s %s: released %s record page(s) of %s "
            "sequence(s); dropped %s mirror(s)",
            rank,
            where,
            released_pages,
            len(released_ids),
            dropped,
        )
    return released_pages
