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
Host pages through the worker view's own page pointers and compares. A
mismatch names the rank, row kind, host sid, destination span, layer, both
digests, and the first differing Host page -- everything needed to localize an
offset bug without a second run.

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


def read_host_row_bytes(record: PoolOffloadRow) -> bytes:
    """Re-read a row's destination Host bytes through the worker view.

    ``get_sequence_layer_page_pointers`` returns one host K pointer per page of
    the sequence's logical page table, in page-table order, resolving a logical
    layer id to its physical layer itself. A raw destination token index
    therefore splits into (page ordinal, offset inside the page) exactly as
    ``async_offload_layer_kv_range_to_host`` wrote it.
    """

    k_pointers, _ = record.view.get_sequence_layer_page_pointers(
        int(record.host_sid), int(record.layer_idx), None
    )
    out = bytearray()
    token = record.dest_start_tokens
    remaining = record.token_count
    while remaining > 0:
        page_ordinal, offset_tokens = divmod(token, record.page_tokens)
        if page_ordinal >= len(k_pointers):
            raise RuntimeError(
                f"{LOG_TAG} sid={record.host_sid} layer={record.layer_idx} needs "
                f"Host page ordinal {page_ordinal} for destination token {token}, "
                f"but the page table holds {len(k_pointers)} pages"
            )
        take = min(record.page_tokens - offset_tokens, remaining)
        address = int(k_pointers[page_ordinal]) + offset_tokens * record.token_bytes
        out += ctypes.string_at(address, take * record.token_bytes)
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

    @property
    def matched(self) -> bool:
        return not self.mismatches


def verify_pool_offload_rows(
    records: Sequence[PoolOffloadRow],
    *,
    logger,
    reader: Callable[[PoolOffloadRow], bytes] = read_host_row_bytes,
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
    layers: set[int] = set()
    mismatches: list[tuple[PoolOffloadRow, RowComparison]] = []
    rank = records[0].rank if records else -1
    for record in records:
        comparison = compare_row_bytes(record, reader(record))
        rows_checked += 1
        bytes_hashed += record.num_bytes
        layers.add(record.layer_idx)
        if record.kind == "segment":
            segment_rows += 1
        else:
            tail_rows += 1
        if not comparison.matched:
            mismatches.append((record, comparison))
            logger.error(
                "%s MISMATCH rank=%s kind=%s row=%s host_sid=%s layer=%s "
                "dest_start_tokens=%s dest_tokens=[%s,%s) token_bytes=%s "
                "page_tokens=%s src_digest=%s dst_digest=%s first_bad_page=%s "
                "wave=%s detail=%s",
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
                comparison.source_digest,
                comparison.dest_digest,
                comparison.first_differing_page,
                wave_tag,
                comparison.detail or "-",
            )
    ordered_layers = tuple(sorted(layers))
    logger.info(
        "%s rank=%s wave=%s rows_checked=%s bytes_hashed=%s segment_rows=%s "
        "tail_rows=%s layers=%s layer_span=%s-%s mismatches=%s",
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
        len(mismatches),
    )
    return PoolOffloadCheckResult(
        rows_checked=rows_checked,
        bytes_hashed=bytes_hashed,
        segment_rows=segment_rows,
        tail_rows=tail_rows,
        layers=ordered_layers,
        mismatches=tuple(mismatches),
    )
