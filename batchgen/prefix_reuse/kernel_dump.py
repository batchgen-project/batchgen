"""Flag-gated dump of what the pooled prefill kernel is actually handed.

The planner, the executor layout, the wrapper's kernel ARGUMENTS, the pooled
attention reference semantics, the KDA levels, the Host offload bytes and the
page-table padding have each been cleared for the gapped two-group chain wave.
What is left is the GPU-side truth at kernel read time: whether the pool slots
hold what the layout says they hold when ``gqa_extend_fa`` reads them, and
whether a row's query tokens are paired with the table row the layout assigned.

This records that at the one seam where both happen -
``pool_prefill_attention``: the chunk's K/V is written into the pool, then
every row attends through its page table. Per chunk and per dumped pool layer
it stores the kernel arguments, a digest of every pool page the table
addresses READ BACK from the pool tensor right before the kernel call, and a
digest of every row's SOURCE K at the write site. Digests and small integer
tensors only; no raw KV ever leaves the device.

Why the row digest alone was not enough (schema 2)
--------------------------------------------------
A row's write digest covers its whole source slice while the read digest
covers one page, so the two are only comparable where a row happens to own
exactly one full page. In the live gapped two-group wave that was true for a
single row out of 126 (the one tail whose own tokens are exactly
``page_tokens``), which left write->read byte identity unchecked for every
other row and for every ancestor page a row attends over.

Schema 2 therefore also splits each row's source K at the pool's PAGE
boundaries, the same split the write itself makes, and digests each piece
against exactly the slots it lands in. The split is derived from
``batch.slots`` - the very index tensor ``index_copy_`` is handed - as maximal
runs of consecutive slots inside one page, so it describes where the KV went
even if a row's slots were NOT one page-aligned ascending block. For each run
the dump carries the source digest and the digest of ``k_cache[page][offset:
offset + length]`` read back from the pool, which are the SAME bytes in the
same layout: unequal means that page does not hold what the layout says it
holds, per page, for every row.

A page written by an earlier chunk of the wave has no write run in this
chunk's dump (it was checkable in the dump of the chunk that wrote it); its
whole-page read digest is still recorded.

Two modes, because the sync is itself a hypothesis:

* ``sync``   - ``torch.cuda.synchronize()`` before reading the pool back, so
  the digests describe a settled pool. If the corruption disappears only in
  this mode, the defect is a write->attend ordering race, and the sync is the
  finding.
* ``nosync`` - dump with no explicit synchronize. The read-back may itself race
  with the chunk's writes, so a digest mismatch here is not by itself proof of
  a wrong pool; it is the control that tells ``sync`` apart from a probe
  artefact.

Off by default and nothing in this module runs.

Arming (batch level, no env var, no cold restart)::

    {"batchgen_debug": {"prefix_pool_dump_kernel_args": "sync",
                        "prefix_pool_dump_dir": "/dev/shm"}}
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

# Batch-level ``batchgen_debug`` keys.
DEBUG_FLAG = "prefix_pool_dump_kernel_args"
DUMP_DIR_FLAG = "prefix_pool_dump_dir"

MODE_SYNC = "sync"
MODE_NOSYNC = "nosync"
MODES = (MODE_SYNC, MODE_NOSYNC)

SCHEMA_VERSION = 2
DEFAULT_DUMP_PARENT = "/dev/shm"
# blake2b digest size; 16 bytes is plenty to separate page contents and keeps
# the npz small enough to scp off a node.
DIGEST_BYTES = 16
_HEX_WIDTH = DIGEST_BYTES * 2

_KIND_CODE = {"segment": 0, "tail": 1}


def parse_dump_mode(value) -> Optional[str]:
    """``None`` when the batch did not ask, else ``"sync"`` or ``"nosync"``.

    An unsupported value is refused by name rather than silently treated as
    off: a mistyped mode must not look like a clean run.
    """

    if value is None or value is False:
        return None
    if value is True:
        return MODE_SYNC
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return MODE_SYNC if value else None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "no", "off"}:
            return None
        if text in {"1", "true", "yes", "on"}:
            return MODE_SYNC
        if text in MODES:
            return text
    raise ValueError(
        f"{DEBUG_FLAG} must be one of {MODES} (or a boolean), got {value!r}"
    )


def dump_directory(parent, rank: int) -> str:
    """``<parent>/pool_dump_rank<rank>``; ``parent`` defaults to /dev/shm."""

    base = DEFAULT_DUMP_PARENT if parent in (None, "") else str(parent)
    return os.path.join(base, f"pool_dump_rank{int(rank)}")


def dumped_pool_layers(num_pool_layers: int) -> tuple[int, ...]:
    """Which dense pool rows to dump: the first and one in the middle.

    Bounds the volume to two layers per chunk; the first layer is where a
    wrong pool read shows up earliest and the middle one proves it is not a
    first-layer-only artefact.
    """

    if num_pool_layers <= 0:
        raise ValueError(f"pool has {num_pool_layers} layers")
    if num_pool_layers == 1:
        return (0,)
    return (0, num_pool_layers // 2)


def _digest(buffer: bytes) -> str:
    return hashlib.blake2b(buffer, digest_size=DIGEST_BYTES).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    """blake2b of a tensor's raw bytes, host side and contiguous."""

    host = tensor.detach().contiguous()
    if host.is_cuda:
        host = host.cpu()
    # numpy cannot represent bfloat16; hash the raw bytes bit-exactly instead.
    return _digest(host.view(torch.uint8).numpy().tobytes())


@dataclass(frozen=True)
class PageWriteRun:
    """One maximal run of a row's source K that lands inside ONE pool page.

    ``source_start``/``length`` index the row's own query tokens; the bytes go
    to ``page`` at token offset ``offset``. ``token_start`` is the absolute
    position of the run's first token, so a run names a (member, position)
    range without the reader re-deriving it.
    """

    row: int
    page: int
    offset: int
    length: int
    source_start: int
    token_start: int
    digest: str


def slot_write_runs(slots: Sequence[int], page_tokens: int) -> list[tuple]:
    """Cut one row's slot list into ``(source_start, length, page, offset)``.

    A run is maximal in both senses that matter to a per-page digest: the
    slots are consecutive AND they stay inside one page, so the run's bytes
    are exactly ``k_cache[page][offset:offset + length]`` in pool layout. The
    slots are the index tensor the pool write is handed, so this describes
    where the KV actually went, not where the layout meant to put it: a row
    whose slots are not one page-aligned ascending block simply yields more
    runs instead of being mis-described by a single digest.
    """

    if page_tokens <= 0:
        raise ValueError(f"page_tokens must be positive, got {page_tokens}")
    runs: list[tuple] = []
    for index, slot in enumerate(slots):
        slot = int(slot)
        page, offset = divmod(slot, page_tokens)
        if runs:
            start, length, last_page, last_offset = runs[-1]
            if page == last_page and offset == last_offset + length:
                runs[-1] = (start, length + 1, last_page, last_offset)
                continue
        runs.append((index, 1, page, offset))
    return runs


@dataclass
class PoolKernelDumpRecorder:
    """Collects one npz per (chunk, dumped pool layer) for one rank."""

    rank: int
    mode: str
    directory: str
    page_tokens: int
    # Wave index -> the sequence's global_idx, so a row maps to a live member.
    global_ids: Sequence[int] = ()
    layers: Optional[tuple[int, ...]] = None
    chunk_index: int = -1
    files: list = field(default_factory=list)
    bytes_hashed: int = 0
    # Set at the write site, consumed by the read site of the same call.
    _write_digests: dict = field(default_factory=dict)
    # Per-page source digests of the same write, in row then page order.
    _write_runs: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"dump mode must be one of {MODES}, got {self.mode!r}")
        os.makedirs(self.directory, exist_ok=True)

    # -- chunk lifecycle ---------------------------------------------------- #
    def begin_chunk(self, batch) -> None:
        """Called by the executor as it packs a chunk, before any layer runs."""

        self.chunk_index += 1
        self._write_digests.clear()
        self._write_runs.clear()

    def wants_layer(self, pool_layer: int, num_pool_layers: int) -> bool:
        if self.layers is None:
            self.layers = dumped_pool_layers(num_pool_layers)
        return int(pool_layer) in self.layers

    def maybe_synchronize(self) -> bool:
        """True when it synchronized; ``nosync`` never does."""

        if self.mode != MODE_SYNC:
            return False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    # -- the two sites ------------------------------------------------------ #
    def record_write_sources(self, batch, key: torch.Tensor) -> None:
        """Digest each row's SOURCE K, as handed to the pool write.

        Twice over: once for the whole row, and once per pool page the row's
        slots cover, so the read site can compare the SAME bytes it hashes
        out of the pool. Both digests come from ``key`` before the
        ``index_copy_``, so nothing here depends on the write having worked.
        """

        bounds = batch.row_bounds
        slots = batch.slots.detach().cpu().tolist()
        self._write_digests.clear()
        self._write_runs.clear()
        for row in range(len(batch.items)):
            lo, hi = bounds[row], bounds[row + 1]
            self._write_digests[row] = _tensor_digest(key[lo:hi])
            token_start = int(batch.items[row].token_start)
            for start, length, page, offset in slot_write_runs(
                slots[lo:hi], self.page_tokens
            ):
                piece = key[lo + start:lo + start + length]
                self._write_runs.append(PageWriteRun(
                    row=row, page=page, offset=offset, length=length,
                    source_start=start, token_start=token_start + start,
                    digest=_tensor_digest(piece),
                ))
                self.bytes_hashed += piece.numel() * piece.element_size()

    def record_kernel_inputs(
        self, batch, k_cache: torch.Tensor, pool_layer: int, num_pool_layers: int
    ) -> Optional[str]:
        """Digest the pool pages the table addresses, then write the npz.

        ``k_cache`` is this layer's pool K, shaped
        ``(num_pages, page_tokens, heads, head_dim)``.

        Two read-side digests per dump: the whole of every page the table
        addresses, and - for every write run the write site recorded - exactly
        the slots that run claims to have filled. The second is directly
        comparable with the run's source digest, so write->read byte identity
        is a per-page equality instead of an inference.
        """

        import numpy as np

        synced = self.maybe_synchronize()
        table = batch.page_table.detach().cpu().numpy()
        cache_seqlens = batch.cache_seqlens.detach().cpu().numpy()
        cu_seqlens = batch.cu_seqlens_q.detach().cpu().numpy()

        pages = sorted({int(page) for row in table for page in row})
        digests = []
        for page in pages:
            block = k_cache[page]
            digests.append(_tensor_digest(block))
            self.bytes_hashed += block.numel() * block.element_size()

        rows = len(batch.items)
        kinds = np.empty(rows, dtype=np.int8)
        refs = np.empty(rows, dtype=np.int32)
        global_idx = np.full(rows, -1, dtype=np.int64)
        token_start = np.empty(rows, dtype=np.int32)
        token_end = np.empty(rows, dtype=np.int32)
        write_digest = np.empty(rows, dtype=f"S{_HEX_WIDTH}")
        for row, item in enumerate(batch.items):
            kinds[row] = _KIND_CODE[item.kind]
            refs[row] = item.ref
            if item.kind == "tail" and item.ref < len(self.global_ids):
                global_idx[row] = int(self.global_ids[item.ref])
            token_start[row] = item.token_start
            token_end[row] = item.token_end
            write_digest[row] = self._write_digests.get(row, "").encode()

        # Per-page write runs, each read back over exactly its own slots.
        runs = self._write_runs
        run_count = len(runs)
        run_row = np.empty(run_count, dtype=np.int32)
        run_page = np.empty(run_count, dtype=np.int32)
        run_offset = np.empty(run_count, dtype=np.int32)
        run_length = np.empty(run_count, dtype=np.int32)
        run_source_start = np.empty(run_count, dtype=np.int32)
        run_token_start = np.empty(run_count, dtype=np.int32)
        run_write_digest = np.empty(run_count, dtype=f"S{_HEX_WIDTH}")
        run_read_digest = np.empty(run_count, dtype=f"S{_HEX_WIDTH}")
        for index, run in enumerate(runs):
            run_row[index] = run.row
            run_page[index] = run.page
            run_offset[index] = run.offset
            run_length[index] = run.length
            run_source_start[index] = run.source_start
            run_token_start[index] = run.token_start
            run_write_digest[index] = run.digest.encode()
            slice_ = k_cache[run.page][run.offset:run.offset + run.length]
            run_read_digest[index] = _tensor_digest(slice_).encode()
            self.bytes_hashed += slice_.numel() * slice_.element_size()

        path = os.path.join(
            self.directory,
            f"rank{self.rank}_chunk{self.chunk_index:04d}"
            f"_layer{int(pool_layer):02d}.npz",
        )
        np.savez(
            path,
            schema=np.int32(SCHEMA_VERSION),
            rank=np.int32(self.rank),
            mode=np.array(self.mode.encode()),
            synced=np.bool_(synced),
            chunk=np.int32(self.chunk_index),
            pool_layer=np.int32(pool_layer),
            num_pool_layers=np.int32(num_pool_layers),
            page_tokens=np.int32(self.page_tokens),
            num_pool_pages=np.int32(k_cache.shape[0]),
            bytes_hashed=np.int64(self.bytes_hashed),
            page_table=table.astype(np.int32),
            cache_seqlens=cache_seqlens.astype(np.int32),
            cu_seqlens_q=cu_seqlens.astype(np.int32),
            max_seqlen_q=np.int64(int(batch.max_seqlen_q)),
            row_table_row=np.arange(rows, dtype=np.int32),
            row_kind=kinds,
            row_ref=refs,
            row_global_idx=global_idx,
            row_token_start=token_start,
            row_token_end=token_end,
            read_pages=np.asarray(pages, dtype=np.int32),
            read_digest=np.asarray(
                [d.encode() for d in digests], dtype=f"S{_HEX_WIDTH}"
            ),
            write_digest=write_digest,
            write_run_row=run_row,
            write_run_page=run_page,
            write_run_offset=run_offset,
            write_run_length=run_length,
            write_run_source_start=run_source_start,
            write_run_token_start=run_token_start,
            write_run_digest=run_write_digest,
            write_run_read_digest=run_read_digest,
        )
        self.files.append(path)
        return path


def read_dump(path: str) -> dict:
    """Load one npz back as plain Python, for the collector and the tests."""

    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        schema = int(data["schema"])
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"{path} is dump schema {schema}, this build reads "
                f"{SCHEMA_VERSION}"
            )
        out = {
            "schema": schema,
            "rank": int(data["rank"]),
            "mode": bytes(data["mode"]).decode(),
            "synced": bool(data["synced"]),
            "chunk": int(data["chunk"]),
            "pool_layer": int(data["pool_layer"]),
            "num_pool_layers": int(data["num_pool_layers"]),
            "page_tokens": int(data["page_tokens"]),
            "num_pool_pages": int(data["num_pool_pages"]),
            "bytes_hashed": int(data["bytes_hashed"]),
            "max_seqlen_q": int(data["max_seqlen_q"]),
            "page_table": data["page_table"].tolist(),
            "cache_seqlens": data["cache_seqlens"].tolist(),
            "cu_seqlens_q": data["cu_seqlens_q"].tolist(),
            "row_table_row": data["row_table_row"].tolist(),
            "row_kind": data["row_kind"].tolist(),
            "row_ref": data["row_ref"].tolist(),
            "row_global_idx": data["row_global_idx"].tolist(),
            "row_token_start": data["row_token_start"].tolist(),
            "row_token_end": data["row_token_end"].tolist(),
            "read_pages": data["read_pages"].tolist(),
            "read_digest": [v.decode() for v in data["read_digest"]],
            "write_digest": [v.decode() for v in data["write_digest"]],
            "write_run_row": data["write_run_row"].tolist(),
            "write_run_page": data["write_run_page"].tolist(),
            "write_run_offset": data["write_run_offset"].tolist(),
            "write_run_length": data["write_run_length"].tolist(),
            "write_run_source_start": data["write_run_source_start"].tolist(),
            "write_run_token_start": data["write_run_token_start"].tolist(),
            "write_run_digest": [v.decode() for v in data["write_run_digest"]],
            "write_run_read_digest": [
                v.decode() for v in data["write_run_read_digest"]
            ],
        }
    out["kind"] = [
        "segment" if code == _KIND_CODE["segment"] else "tail"
        for code in out["row_kind"]
    ]
    # The verdict the per-page digests exist for: True means that page holds,
    # over exactly those slots, the bytes the row handed the pool.
    out["write_run_match"] = [
        write == read for write, read in
        zip(out["write_run_digest"], out["write_run_read_digest"])
    ]
    return out
