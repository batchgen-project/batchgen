"""Diagnostic prefix-cache KV integrity ledger (``--prefix-cache-integrity-check``).

The committer of a host page records exact content hashes of its K/V bytes and
the identity of the tokens it holds; every later reuse re-hashes the bytes it
actually reads and compares them against that node-shared record.
"""

from __future__ import annotations

import functools
import hashlib
import math
import mmap
import time
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_WEIGHT_SEED = 0x5EED_CAFE
_WEIGHT_LIMIT = 1 << 29
_MAX_PAGE_ELEMENTS = 1 << 16
_HASH_BLOCK_PAGES = 256  # <= 64 MiB per int64 page_hashes temporary
_ATTACH_TIMEOUT_S = 30.0
_INT63_MASK = (1 << 63) - 1


class PrefixIntegrityError(RuntimeError):
    """Prefix-cache KV bytes or token identity differ from the committed record."""


@functools.lru_cache(maxsize=None)
def _weights(num_elements: int, device: torch.device) -> torch.Tensor:
    # Drawn on CPU so every device hashes with identical weights.
    generator = torch.Generator().manual_seed(_WEIGHT_SEED)
    return torch.randint(
        1, _WEIGHT_LIMIT, (2, num_elements), generator=generator, dtype=torch.int64
    ).to(device)


def page_hashes(pages: torch.Tensor) -> torch.Tensor:
    """Exact int64 ``[N, 2]`` content hashes of ``N`` pages of a 2-byte dtype.

    Each page's raw 16-bit patterns are sign-extended and dotted with two fixed
    weight vectors. |x| <= 2**15, w < 2**29 and E <= 2**16 keep every sum
    below 2**60, so the result is exact and identical on CPU and CUDA.
    """
    if pages.element_size() != 2:
        raise ValueError(f"page_hashes needs a 2-byte dtype, got {pages.dtype}")
    num_elements = math.prod(pages.shape[1:])
    if num_elements > _MAX_PAGE_ELEMENTS:
        raise ValueError(f"exact hashing allows <= 2**16 elements, got {num_elements}")
    flat = pages.view(torch.int16).reshape(pages.shape[0], num_elements).long()
    weights = _weights(num_elements, flat.device)
    return torch.stack(
        [(flat * weights[0]).sum(dim=1), (flat * weights[1]).sum(dim=1)], dim=1
    )


def token_chain_hash(token_ids: Sequence[int]) -> int:
    """63-bit blake2b hash of the little-endian int32 token bytes."""
    data = np.asarray(token_ids, dtype="<i4").tobytes()
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "little") & _INT63_MASK


def page_identity(
    token_ids: Sequence[int], num_pages: int, page_tokens: int = 64
) -> tuple[list[int], list[int]]:
    """Chain hash and raw end token of each of the first ``num_pages`` pages.

    Entry ``i`` equals ``token_chain_hash(token_ids[:(i + 1) * page_tokens])``.
    """
    data = np.asarray(token_ids, dtype="<i4")
    if data.size < num_pages * page_tokens:
        raise PrefixIntegrityError(
            f"{data.size} tokens cannot fill {num_pages} pages of {page_tokens}"
        )
    hasher = hashlib.blake2b(digest_size=8)
    chains, raw_ends = [], []
    for page in range(num_pages):
        end = (page + 1) * page_tokens
        hasher.update(data[end - page_tokens : end].tobytes())
        chains.append(int.from_bytes(hasher.copy().digest(), "little") & _INT63_MASK)
        raw_ends.append(int(data[end - 1]))
    return chains, raw_ends


def sequence_page_hashes(
    k: torch.Tensor, v: torch.Tensor, positions: Sequence[int], device
) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V hashes ``[len(positions), L, 2]`` (CPU) of one sequence's pages.

    ``k``/``v`` use the ``read_sequence_kv_to_cpu`` layout
    ``[L, P, page_tokens, heads, head_dim]``; the page axis follows the
    sequence's Host page table, so ``positions`` are logical page indices.
    Each layer is hashed separately on ``device`` to bound temporary memory.
    """
    index = torch.as_tensor(list(positions), dtype=torch.long)

    def per_layer(kv: torch.Tensor) -> torch.Tensor:
        if kv.dim() != 5:
            raise ValueError(f"expected [L, P, tokens, heads, dim] KV, got {tuple(kv.shape)}")
        return torch.stack(
            [page_hashes(kv[layer].index_select(0, index).to(device))
             for layer in range(kv.shape[0])],
            dim=1,
        ).cpu()

    return per_layer(k), per_layer(v)


def gpu_chunk_hashes(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    chunk_ids: Sequence[int],
    page_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V hashes ``[len(chunk_ids), L, 2]`` (CPU) of Host-page-sized GPU chunks.

    ``k_cache``/``v_cache`` use the ``GPUPagedKVCacheManager`` layout
    ``[L, gpu_pages, gpu_page_tokens, heads, head_dim]`` (token-major). With
    ``r = gpu_page_tokens // page_tokens``, chunk ``c`` is tokens
    ``[(c % r) * page_tokens, (c % r + 1) * page_tokens)`` of GPU page
    ``c // r``: the bytes one Host page is loaded into, in the Host page's
    ``[tokens, heads, dim]`` order. Layer ``l`` is the raw cache row, the row
    Host loads write for Host layer ``l``. Hashed in blocks to bound memory.
    """
    index = torch.as_tensor(list(chunk_ids), dtype=torch.long, device=k_cache.device)

    def per_layer(cache: torch.Tensor) -> torch.Tensor:
        if cache.dim() != 5 or cache.shape[2] % page_tokens:
            raise ValueError(
                f"expected [L, pages, tokens, heads, dim] cache with tokens a "
                f"multiple of {page_tokens}, got {tuple(cache.shape)}"
            )
        # A view, never a copy: the paged caches are contiguous.
        chunks = cache.view(cache.shape[0], -1, page_tokens, *cache.shape[3:])
        return torch.stack(
            [torch.cat([page_hashes(chunks[layer].index_select(0, block))
                        for block in index.split(_HASH_BLOCK_PAGES)])
             for layer in range(cache.shape[0])],
            dim=1,
        ).cpu()

    return per_layer(k_cache), per_layer(v_cache)


def keyed_gpu_pages(
    gpu_page_by_key: Mapping[int, int], page_ids: Sequence[int], *, context: str
) -> list[int]:
    """GPU pages that decode shares by Host page id, in ``page_ids`` order."""
    missing = [int(page) for page in page_ids if int(page) not in gpu_page_by_key]
    if missing:
        raise PrefixIntegrityError(
            f"{context}: Host pages {missing[:8]} have no decode GPU page"
        )
    return [int(gpu_page_by_key[int(page)]) for page in page_ids]


def page_positions(
    page_table: Sequence[int], page_ids: Sequence[int], *, context: str
) -> list[int]:
    """Logical positions of ``page_ids`` in one sequence's Host page table."""
    index = {int(page): pos for pos, page in enumerate(page_table)}
    missing = [int(page) for page in page_ids if int(page) not in index]
    if missing:
        raise PrefixIntegrityError(
            f"{context}: pages {missing[:8]} are not in the sequence page table"
        )
    return [index[int(page)] for page in page_ids]


def split_commit_pages(
    page_ids: Sequence[int], committed_ids: Sequence[int], *, context: str
) -> tuple[list[int], list[int]]:
    """Positions in ``page_ids`` to verify (already committed) and to record."""
    verify = page_positions(page_ids, committed_ids, context=context)
    taken = set(verify)
    return verify, [pos for pos in range(len(page_ids)) if pos not in taken]


def check_compute_cached(
    attached: int, compute_cached: int, prompt_length: int, *, context: str
) -> None:
    """Compute resumes at the hit, except a raw full hit recomputes its last token."""
    expected = attached - 1 if attached == prompt_length else attached
    if compute_cached != expected:
        raise PrefixIntegrityError(
            f"{context}: compute_cached_tokens={compute_cached}, expected "
            f"{expected} (attached={attached}, prompt_length={prompt_length})"
        )


def check_host_growth_accounting(
    *, node: int, planned_free: int, actual_free: int
) -> None:
    """Fail when the growth plan credited more free Host pages than exist."""
    if actual_free < planned_free:
        raise PrefixIntegrityError(
            f"H8-growth-accounting: node {node} planned {planned_free} free "
            f"Host pages before growth but has {actual_free} "
            f"(over-credited by {planned_free - actual_free})"
        )


def assert_drained(live: Mapping[str, int], *, context: str) -> None:
    """Fail if any prefix bookkeeping is still live after a batch drained."""
    leaked = {name: int(count) for name, count in live.items() if count}
    if leaked:
        raise PrefixIntegrityError(
            f"{context}: prefix state still live after the batch drained: {leaked}"
        )


def _as_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


class IntegrityLedger:
    """Node-shared int64 ``[num_pages, 4 * num_layers + 4]`` table, one row per page.

    Row layout: layer ``l`` owns columns ``4l .. 4l+3`` = K hash0, K hash1,
    V hash0, V hash1; the last four columns are token_chain_hash,
    raw_end_token, committer_rank and valid (1 = row written).
    Hash arguments are int64 ``[N, num_layers, 2]`` (``page_hashes`` per
    layer, stacked on dim 1), aligned with ``page_ids`` of length ``N``.
    """

    # No lock: a host page id is written by exactly one committer at a time.

    def __init__(self, base_name: str, num_pages: int, num_layers: int):
        self.name = f"{base_name}_integrity"
        self.num_pages = int(num_pages)
        self.num_layers = int(num_layers)
        num_cols = 4 * self.num_layers + 4
        size = self.num_pages * num_cols * 8
        try:
            # A new POSIX segment is zero-filled, so every row starts invalid.
            self._shm = shared_memory.SharedMemory(
                name=self.name, create=True, size=size
            )
        except FileExistsError:
            self._shm = _attach(self.name, size)
        self._table = np.ndarray(
            (self.num_pages, num_cols), dtype=np.int64, buffer=self._shm.buf
        )

    def record(self, page_ids, k_hashes, v_hashes, token_chain_hashes,
               raw_end_tokens, rank: int) -> None:
        ids = self._page_ids(page_ids)
        kv = self._kv_columns(k_hashes, v_hashes, ids.size)
        chains, raw_ends = _as_numpy(token_chain_hashes), _as_numpy(raw_end_tokens)
        if chains.shape != ids.shape or raw_ends.shape != ids.shape:
            raise ValueError(
                f"need {ids.size} chain hashes and raw end tokens, got "
                f"{chains.shape} and {raw_ends.shape}"
            )
        base = 4 * self.num_layers
        self._table[ids, :base] = kv.reshape(ids.size, base)
        self._table[ids, base] = chains
        self._table[ids, base + 1] = raw_ends
        self._table[ids, base + 2] = int(rank)
        self._table[ids, base + 3] = 1

    def verify(self, page_ids, k_hashes, v_hashes, *, context: str) -> int:
        ids = self._page_ids(page_ids)
        actual = self._kv_columns(k_hashes, v_hashes, ids.size)
        rows = self._rows(ids, context)
        expected = rows[:, : 4 * self.num_layers].reshape(actual.shape)
        mismatch = np.argwhere(expected != actual)
        if mismatch.size:
            i, layer, col = (int(x) for x in mismatch[0])
            pair = slice(0, 2) if col < 2 else slice(2, 4)
            raise PrefixIntegrityError(
                f"{context}: page {int(ids[i])} layer {layer} "
                f"{'K' if col < 2 else 'V'} hash mismatch: expected "
                f"{tuple(expected[i, layer, pair].tolist())}, actual "
                f"{tuple(actual[i, layer, pair].tolist())} "
                f"(committed by rank {int(rows[i, -2])})"
            )
        return int(ids.size)

    def verify_identity(self, page_ids, token_ids_of_request,
                        page_tokens: int = 64, *, context: str) -> int:
        """Check that ``page_ids`` hold, in order, the request's leading pages."""
        ids = self._page_ids(page_ids)
        chains, raw_ends = page_identity(token_ids_of_request, ids.size, page_tokens)
        rows = self._rows(ids, context)
        base = 4 * self.num_layers
        want = np.array([chains, raw_ends], dtype=np.int64).T
        mismatch = np.argwhere(rows[:, base : base + 2] != want)
        if mismatch.size:
            i, col = (int(x) for x in mismatch[0])
            raise PrefixIntegrityError(
                f"{context}: page {int(ids[i])} (request page {i}) "
                f"{('token_chain_hash', 'raw_end_token')[col]} mismatch: "
                f"expected {int(rows[i, base + col])}, actual {int(want[i, col])} "
                f"(committed by rank {int(rows[i, -2])})"
            )
        return int(ids.size)

    def close(self) -> None:
        self._table = None  # drop the buffer export so the mapping can close
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

    def _page_ids(self, page_ids) -> np.ndarray:
        ids = _as_numpy(page_ids).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_pages):
            raise ValueError(
                f"page ids {int(ids.min())}..{int(ids.max())} outside "
                f"[0, {self.num_pages})"
            )
        return ids

    def _kv_columns(self, k_hashes, v_hashes, n: int) -> np.ndarray:
        k, v = _as_numpy(k_hashes), _as_numpy(v_hashes)
        shape = (n, self.num_layers, 2)
        if k.shape != shape or v.shape != shape:
            raise ValueError(
                f"K/V hashes must be {shape}, got {k.shape} and {v.shape}"
            )
        return np.concatenate([k, v], axis=2)

    def _rows(self, ids: np.ndarray, context: str) -> np.ndarray:
        rows = self._table[ids]
        missing = np.flatnonzero(rows[:, -1] != 1)
        if missing.size:
            raise PrefixIntegrityError(
                f"{context}: page {int(ids[missing[0]])} has no ledger row"
            )
        return rows


def _attach(name: str, size: int) -> shared_memory.SharedMemory:
    """Attach to a peer-created segment once the creator has sized it."""
    deadline = time.monotonic() + _ATTACH_TIMEOUT_S
    actual = 0
    while True:
        try:
            shm = shared_memory.SharedMemory(name=name)
        except ValueError:  # created but not yet sized: empty mmap
            shm = None
        if shm is not None:
            # Linux reports the exact size; macOS rounds it up to a page.
            if size <= shm.size < size + mmap.PAGESIZE:
                return shm
            actual = shm.size
            shm.close()
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"integrity ledger '{name}' is {actual} bytes after "
                f"{_ATTACH_TIMEOUT_S}s, expected {size}"
            )
        time.sleep(0.05)


def unlink_integrity_ledger(base_name: str) -> None:
    """Remove the ledger segment, if any; the prefix region owner calls this."""
    try:
        shm = shared_memory.SharedMemory(name=f"{base_name}_integrity")
    except FileNotFoundError:
        return
    shm.close()
    shm.unlink()


@dataclass
class IntegrityCounters:
    """Per-process tallies, emitted via ``emit_prefix_cache_metric``."""

    pages_ledgered: int = 0
    identity_checks: int = 0
    pages_verified: Counter = field(default_factory=Counter)  # by hook name

    def metric_record(self, phase: str) -> dict[str, Any]:
        """Keyword fields for ``emit_prefix_cache_metric(rank=..., **record)``."""
        return {
            "component": "prefix_integrity",
            "phase": phase,
            "pages_ledgered": self.pages_ledgered,
            "identity_checks": self.identity_checks,
            "pages_verified": dict(self.pages_verified),
        }
