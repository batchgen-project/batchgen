"""Byte-equality check for the pooled prefill wave's Host offload.

Every rank of a serve group offloads the SAME pooled rows into the SAME
mirrored Host pages: a segment row goes to its segment's pseudo sequence at
token 0, a tail row to its prompt's own pages at the tail's start. That the
bytes a rank hands to the D2H actually land at the destination it asked for
has never been asserted. The failure is silent by construction -- the wave
plan stays identical across ranks and runs, and only the DATA moves -- so a
wrong destination start or a wrong page stride reads as a model bug.

This module is that assertion, driven by the batch-level debug flag
``prefix_pool_offload_check``. At offload-issue time the recorder digests the
SOURCE bytes of every row (the whole row, plus one digest per destination Host
page). Once the wave's D2H has retired, the worker re-reads the DESTINATION
Host pages and compares. A mismatch names the rank, row kind, host sid,
destination span, layer, both page-table sizes, both digests, and the first
differing Host page -- everything needed to localize an offset bug without a
second run.

The destination is resolved through the LOGICAL page table, because that is
the table the offload itself walks. ``get_sequence_layer_page_pointers`` is
the wrong table for a pooled member: it lists the pages a sequence OWNS, while
a member's attached prefix chain lives only in the logical table, so a member
with 12 attached and 2 private pages reads back as 2. Mistaking that for a
truncated table is a probe artifact, reported as ``api-split`` and never as a
fault; only a LOGICAL table too short for the destination is a real
truncation, and that raises.

Nothing here runs unless the flag installed a recorder:
``AttnWrapperBase.prefill_pool_offload_recorder is None`` is the entire cost
of the check being off.
"""

from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# Batch-level ``batchgen_debug`` key. Batch-level only, by project debug
# policy: a per-sequence offload check would mix checked and unchecked rows
# inside one pooled chunk, and the chunk is the unit that gets offloaded.
DEBUG_FLAG = "prefix_pool_offload_check"

LOG_TAG = "[PREFIX_POOL_OFFLOAD_CHECK]"

# 16 bytes of blake2b per row and per page: enough to make a collision not the
# explanation, small enough to keep a 64-page row's digests at 1 KiB.
DIGEST_SIZE = 16


def digest_bytes(data) -> str:
    """blake2b over raw bytes, as a hex string."""

    return hashlib.blake2b(memoryview(data).cast("B"), digest_size=DIGEST_SIZE).hexdigest()


def page_digests(
    data, *, page_bytes: int, start_offset: int
) -> tuple[tuple[int, str], ...]:
    """Digest ``data`` split on the DESTINATION's Host page boundaries.

    ``start_offset`` is the byte offset of ``data[0]`` inside the destination
    sequence, so the returned indices are the destination's own page numbers
    and stay comparable even for a row that starts part way into a page.
    """

    if page_bytes <= 0:
        raise ValueError(f"{LOG_TAG} page_bytes must be positive, got {page_bytes}")
    if start_offset < 0:
        raise ValueError(f"{LOG_TAG} start_offset must not be negative, got {start_offset}")
    buf = memoryview(data).cast("B")
    out: list[tuple[int, str]] = []
    total = len(buf)
    pos = 0
    while pos < total:
        absolute = start_offset + pos
        chunk = min(page_bytes - (absolute % page_bytes), total - pos)
        out.append((absolute // page_bytes, digest_bytes(buf[pos : pos + chunk])))
        pos += chunk
    return tuple(out)


@dataclass(frozen=True)
class PoolOffloadRow:
    """What one pooled row asked the D2H to write, digested at issue time."""

    rank: int
    kind: str  # "segment" or "tail", the PlanItem kind of the row
    row: int
    layer_idx: int
    host_sid: int
    dest_start_tokens: int
    token_count: int
    token_bytes: int
    page_tokens: int
    row_digest: str
    page_digests: tuple[tuple[int, str], ...]
    # The exact worker view the offload went through, so the check reads back
    # the same Host region (a DSA model offloads latent and indexer keys to
    # two different views under the same sid).
    view: object = field(default=None, compare=False, repr=False)

    @property
    def num_bytes(self) -> int:
        return self.token_count * self.token_bytes

    @property
    def dest_end_tokens(self) -> int:
        return self.dest_start_tokens + self.token_count

    @property
    def page_bytes(self) -> int:
        return self.page_tokens * self.token_bytes


@dataclass(frozen=True)
class RowComparison:
    matched: bool
    source_digest: str
    dest_digest: str
    first_differing_page: Optional[int]
    detail: str = ""


def compare_row_bytes(record: PoolOffloadRow, dest) -> RowComparison:
    """Compare a row's recorded source digest against destination bytes."""

    dest_buf = memoryview(dest).cast("B")
    dest_digest = digest_bytes(dest_buf)
    if len(dest_buf) != record.num_bytes:
        return RowComparison(
            matched=False,
            source_digest=record.row_digest,
            dest_digest=dest_digest,
            first_differing_page=(
                record.page_digests[0][0] if record.page_digests else None
            ),
            detail=f"dest_bytes={len(dest_buf)} src_bytes={record.num_bytes}",
        )
    if dest_digest == record.row_digest:
        return RowComparison(
            matched=True,
            source_digest=record.row_digest,
            dest_digest=dest_digest,
            first_differing_page=None,
        )
    dest_pages = page_digests(
        dest_buf,
        page_bytes=record.page_bytes,
        start_offset=record.dest_start_tokens * record.token_bytes,
    )
    first: Optional[int] = None
    for (src_page, src_digest), (dest_page, page_digest) in zip(
        record.page_digests, dest_pages
    ):
        if src_page != dest_page or src_digest != page_digest:
            first = dest_page
            break
    return RowComparison(
        matched=False,
        source_digest=record.row_digest,
        dest_digest=dest_digest,
        first_differing_page=first,
    )


@dataclass(frozen=True)
class PageTableProbe:
    """How the two Host page tables of a destination sequence compare.

    ``logical_pages`` is what the offload walks; ``backend_pages`` is what the
    sequence OWNS. They differ for any member holding an attached prefix chain,
    and reading the owned list for such a destination is a probe artifact, not
    a runtime fault -- hence ``api-split`` is an observation, never an error.
    """

    logical_pages: int
    backend_pages: int
    dest_pages_needed: int
    classification: str  # "aligned" or "api-split"

    @property
    def api_split(self) -> bool:
        return self.classification == "api-split"


def destination_page_ids(view, *, host_sid: int) -> list[int]:
    """The LOGICAL Host page ids of a sequence: the list the offload walks.

    ``build_page_table`` and the range offload both read ``page_table_.Pages``
    (core/KV_Storage/host_paged_kv_worker_view.h:716 and :1556), so this is the
    same ordered page list the D2H addressed. ``get_sequence_layer_page_pointers``
    is NOT: it reads ``backend_.SequencePages`` (:733), the sequence-OWNED pages
    only, while ``AttachSharedPrefixPages`` prepends a member's chain to the
    logical table (:329) -- so a 12+2 member reads back as 2 pages there.
    """

    builder = getattr(view, "build_page_table", None)
    if builder is None:
        raise RuntimeError(
            f"{LOG_TAG} worker view {type(view).__name__} exposes no "
            f"build_page_table, so the logical Host page table of sid "
            f"{host_sid} cannot be resolved"
        )
    table = builder([int(host_sid)])
    if len(table) != 1:
        raise RuntimeError(
            f"{LOG_TAG} build_page_table returned {len(table)} rows for one "
            f"sequence id {host_sid}"
        )
    return [int(page) for page in table[0]]


def owned_page_count(view, *, host_sid: int, layer_idx: int) -> int:
    """How many pages the sequence OWNS, for the api-split discriminator."""

    pointers = getattr(view, "get_sequence_layer_page_pointers", None)
    if pointers is None:
        return -1
    k_pointers, _ = pointers(int(host_sid), int(layer_idx), None)
    return len(k_pointers)


def probe_destination_pages(record: PoolOffloadRow):
    """Resolve a row's destination pages and classify the two page tables.

    Hard-fails only when the LOGICAL table is itself too short for the
    destination the offload asked for -- the one case that is a real
    truncation rather than this probe reading the wrong table.
    """

    page_ids = destination_page_ids(record.view, host_sid=record.host_sid)
    owned = owned_page_count(
        record.view, host_sid=record.host_sid, layer_idx=record.layer_idx
    )
    needed = -(-record.dest_end_tokens // record.page_tokens)
    if needed > len(page_ids):
        raise RuntimeError(
            f"{LOG_TAG} TRUNCATED sid={record.host_sid} layer={record.layer_idx} "
            f"kind={record.kind} logical_pages={len(page_ids)} "
            f"backend_pages={owned} dest_ordinal_needed={needed}: the LOGICAL "
            f"Host page table is too short for dest_tokens="
            f"[{record.dest_start_tokens},{record.dest_end_tokens})"
        )
    return (
        PageTableProbe(
            logical_pages=len(page_ids),
            backend_pages=owned,
            dest_pages_needed=needed,
            classification="api-split" if needed > owned else "aligned",
        ),
        page_ids,
    )


def read_host_row_bytes(record: PoolOffloadRow, page_ids=None) -> bytes:
    """Re-read a row's destination Host bytes from the LOGICAL page table.

    The walk is the offload's own: ``ForEachPageChunk`` splits the raw token
    cursor by ``page_slot = cursor / page_tokens`` and
    ``page_offset = cursor % page_tokens``, indexes the logical page list, and
    addresses ``KPageAddress(base, physical_layer, page_id)``
    (host_paged_kv_worker_view.h:2780-2802). ``k_page_ptr(layer_idx, page_id)``
    is that same address by page id, resolving the logical layer id itself
    (KPagePtr, bound at core/batchgen_Binding.cpp:185).
    """

    if page_ids is None:
        page_ids = destination_page_ids(record.view, host_sid=record.host_sid)
    page_pointer = getattr(record.view, "k_page_ptr", None)
    if page_pointer is None:
        raise RuntimeError(
            f"{LOG_TAG} worker view {type(record.view).__name__} exposes no "
            f"k_page_ptr, so Host page bytes cannot be read by page id"
        )
    out = bytearray()
    token = record.dest_start_tokens
    remaining = record.token_count
    while remaining > 0:
        page_slot, offset_tokens = divmod(token, record.page_tokens)
        if page_slot >= len(page_ids):
            raise RuntimeError(
                f"{LOG_TAG} TRUNCATED sid={record.host_sid} "
                f"layer={record.layer_idx} needs logical page slot {page_slot} "
                f"for destination token {token}, but the logical table holds "
                f"{len(page_ids)} pages"
            )
        take = min(record.page_tokens - offset_tokens, remaining)
        address = int(page_pointer(int(record.layer_idx), page_ids[page_slot]))
        out += ctypes.string_at(
            address + offset_tokens * record.token_bytes,
            take * record.token_bytes,
        )
        token += take
        remaining -= take
    return bytes(out)


class PoolOffloadCheckRecorder:
    """Digest the source bytes of every pooled row the rank offloads."""

    def __init__(self, *, rank: int, page_tokens: int) -> None:
        if page_tokens <= 0:
            raise ValueError(
                f"{LOG_TAG} page_tokens must be positive, got {page_tokens}"
            )
        self.rank = int(rank)
        self.page_tokens = int(page_tokens)
        self.rows: list[PoolOffloadRow] = []

    def record_row(
        self,
        *,
        view,
        layer_idx: int,
        row: int,
        kind: str,
        host_sid: int,
        dest_start_tokens: int,
        token_count: int,
        source,
    ) -> PoolOffloadRow:
        """Digest one row's source tensor against its declared destination."""

        if token_count <= 0:
            raise ValueError(
                f"{LOG_TAG} row {row} of layer {layer_idx} has token_count "
                f"{token_count}"
            )
        host = source.detach().contiguous().reshape(-1).cpu()
        total_bytes = host.numel() * host.element_size()
        if total_bytes % token_count:
            raise RuntimeError(
                f"{LOG_TAG} row {row} of layer {layer_idx} is {total_bytes} B, "
                f"which does not divide into {token_count} tokens"
            )
        token_bytes = total_bytes // token_count
        data = ctypes.string_at(host.data_ptr(), total_bytes)
        record = PoolOffloadRow(
            rank=self.rank,
            kind=str(kind),
            row=int(row),
            layer_idx=int(layer_idx),
            host_sid=int(host_sid),
            dest_start_tokens=int(dest_start_tokens),
            token_count=int(token_count),
            token_bytes=int(token_bytes),
            page_tokens=self.page_tokens,
            row_digest=digest_bytes(data),
            page_digests=page_digests(
                data,
                page_bytes=self.page_tokens * token_bytes,
                start_offset=int(dest_start_tokens) * token_bytes,
            ),
            view=view,
        )
        self.rows.append(record)
        return record


@dataclass(frozen=True)
class PoolOffloadCheckResult:
    rows_checked: int
    bytes_hashed: int
    segment_rows: int
    tail_rows: int
    layers: tuple[int, ...]
    mismatches: tuple[tuple[PoolOffloadRow, RowComparison], ...]
    api_split_rows: int = 0

    @property
    def matched(self) -> bool:
        return not self.mismatches


def verify_pool_offload_rows(
    records: Sequence[PoolOffloadRow],
    *,
    logger,
    reader: Callable[..., bytes] = read_host_row_bytes,
    prober: Optional[Callable[[PoolOffloadRow], tuple]] = probe_destination_pages,
    wave_tag: str = "-",
) -> PoolOffloadCheckResult:
    """Re-read every recorded row's destination and compare it to the source.

    One ``MISMATCH`` line per bad row carries everything needed to localize
    the offset bug; one summary line per wave always lands, so a gate can
    assert the check actually ran rather than inferring it from silence.
    """

    rows_checked = 0
    bytes_hashed = 0
    segment_rows = 0
    tail_rows = 0
    api_split_rows = 0
    layers: set[int] = set()
    mismatches: list[tuple[PoolOffloadRow, RowComparison]] = []
    rank = records[0].rank if records else -1
    for record in records:
        probe, page_ids = prober(record) if prober is not None else (None, None)
        comparison = compare_row_bytes(record, reader(record, page_ids))
        rows_checked += 1
        bytes_hashed += record.num_bytes
        layers.add(record.layer_idx)
        if record.kind == "segment":
            segment_rows += 1
        else:
            tail_rows += 1
        if probe is not None and probe.api_split:
            api_split_rows += 1
            if api_split_rows == 1:
                # Once per wave: the owned-pages API cannot see this
                # destination, so the bytes below came from the logical table.
                logger.info(
                    "%s api-split rank=%s wave=%s kind=%s host_sid=%s layer=%s "
                    "logical_pages=%s backend_pages=%s dest_ordinal_needed=%s",
                    LOG_TAG,
                    record.rank,
                    wave_tag,
                    record.kind,
                    record.host_sid,
                    record.layer_idx,
                    probe.logical_pages,
                    probe.backend_pages,
                    probe.dest_pages_needed,
                )
        if not comparison.matched:
            mismatches.append((record, comparison))
            logger.error(
                "%s MISMATCH rank=%s kind=%s row=%s host_sid=%s layer=%s "
                "dest_start_tokens=%s dest_tokens=[%s,%s) token_bytes=%s "
                "page_tokens=%s logical_pages=%s backend_pages=%s "
                "dest_ordinal_needed=%s page_table=%s src_digest=%s "
                "dst_digest=%s first_bad_page=%s wave=%s detail=%s",
                LOG_TAG,
                record.rank,
                record.kind,
                record.row,
                record.host_sid,
                record.layer_idx,
                record.dest_start_tokens,
                record.dest_start_tokens,
                record.dest_end_tokens,
                record.token_bytes,
                record.page_tokens,
                probe.logical_pages if probe is not None else "-",
                probe.backend_pages if probe is not None else "-",
                probe.dest_pages_needed if probe is not None else "-",
                probe.classification if probe is not None else "-",
                comparison.source_digest,
                comparison.dest_digest,
                comparison.first_differing_page,
                wave_tag,
                comparison.detail or "-",
            )
    ordered_layers = tuple(sorted(layers))
    logger.info(
        "%s rank=%s wave=%s rows_checked=%s bytes_hashed=%s segment_rows=%s "
        "tail_rows=%s layers=%s layer_span=%s-%s api_split_rows=%s "
        "mismatches=%s",
        LOG_TAG,
        rank,
        wave_tag,
        rows_checked,
        bytes_hashed,
        segment_rows,
        tail_rows,
        len(ordered_layers),
        ordered_layers[0] if ordered_layers else "-",
        ordered_layers[-1] if ordered_layers else "-",
        api_split_rows,
        len(mismatches),
    )
    return PoolOffloadCheckResult(
        rows_checked=rows_checked,
        bytes_hashed=bytes_hashed,
        segment_rows=segment_rows,
        tail_rows=tail_rows,
        layers=ordered_layers,
        mismatches=tuple(mismatches),
        api_split_rows=api_split_rows,
    )
