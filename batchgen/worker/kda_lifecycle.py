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
