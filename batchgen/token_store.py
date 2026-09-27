# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
# ---------------------------------------------------------------------------- #

"""Paged token storage: a node-shared prompt arena plus a rank-private decoded store.

Today a batch's token ids live in dense int64 matrices (rows = ``--max-pool-size``,
width = the widest request in the batch) that are regrown whenever a wider request
arrives. The width is set by one outlier, every row pays for it, and the superseded
matrices are leaked. This module stores tokens by the page instead:

* ``PromptTokenArena`` — ONE int32 POSIX shared-memory arena per node, created
  sparsely at a fixed capacity, handed out as chained 4096-token pages. The
  tokenized batch is identical on every rank, so one copy per node replaces
  ``world_size`` duplicates of the same bytes. A prompt costs
  ``ceil(len / page_size)`` pages, not the batch's widest request.
* ``DecodedTokenStore`` — process-local int32 chunks for generated tokens, which
  only the owning rank writes, and which are freed when the sequence finishes.

Nothing here is wired into the worker yet; a later change replaces
``QueryBookBufferPool``'s two dense buffers with these two objects.
"""

from __future__ import annotations

import mmap
import os
from multiprocessing import shared_memory
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
	"DEFAULT_DECODED_CHUNK_TOKENS",
	"DEFAULT_PAGE_SIZE_TOKENS",
	"DEFAULT_RELEASE_FREE_FRACTION",
	"END_OF_CHAIN",
	"PROMPT_ARENA_SHM_SUFFIX",
	"DecodedTokenStore",
	"PromptTokenArena",
	"TokenStoreCapacityError",
	"prompt_arena_shm_name",
]


# 4096 tokens x 4 B = one 16 KiB page. Large enough that the chain array and the
# per-chain Python work stay negligible, small enough that a 900-token prompt
# wastes well under a page.
DEFAULT_PAGE_SIZE_TOKENS = 4096
DEFAULT_DECODED_CHUNK_TOKENS = 1024
DEFAULT_RELEASE_FREE_FRACTION = 0.25

END_OF_CHAIN = -1
PROMPT_ARENA_SHM_SUFFIX = "prompt_tokens"

# Marks a page sitting on the writer's free list. Distinct from END_OF_CHAIN so
# that a stale handle raises while its pages are still free. Once a later write
# reuses a page the mark is gone, so this is a debugging aid, not a safety net:
# the caller must never use a handle after freeing it.
_FREE_PAGE = -2
_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)

# int64 header slots: magic, version, page_size_tokens, num_pages,
# next_page_offset, data_offset, and two spare slots for a later revision.
_HEADER_SLOTS = 8
_HEADER_MAGIC = 0x42475F544B4E5301  # b"BG_TKNS\x01"
_HEADER_VERSION = 1

_TOKEN_DTYPE = np.int32
_TOKEN_BYTES = 4
_PAGESIZE = mmap.PAGESIZE

#: A handle is the pair ``(first_page, length)`` — two plain integers, so it fits
#: a fixed-size record (one row of a shared table) without a side list.
TokenHandle = Tuple[int, int]


def _align_up(value: int, alignment: int) -> int:
	return -(-value // alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
	return (value // alignment) * alignment


def _as_token_ids(tokens) -> np.ndarray:
	"""Validate integer token ids in int32 range; return a contiguous int32 copy."""
	arr = np.asarray(tokens)
	if arr.size == 0:
		return np.empty(arr.shape, dtype=_TOKEN_DTYPE)
	if arr.dtype.kind not in "iu":
		raise TypeError(f"token ids must be integers, got dtype {arr.dtype}")
	low, high = int(arr.min()), int(arr.max())
	if low < _INT32_MIN or high > _INT32_MAX:
		raise ValueError(f"token ids [{low}, {high}] do not fit int32")
	return np.ascontiguousarray(arr, dtype=_TOKEN_DTYPE)


def _check_shm_space(nbytes: int) -> None:
	"""Refuse an arena larger than the free space of /dev/shm.

	Creation is sparse, so an over-committed tmpfs would only fail at the first
	write to an unbacked page — as SIGBUS, with no Python traceback. This check
	covers creation; the host memory budget must also leave room for the tokens
	actually written while other segments (host KV, weights) grow.
	"""
	try:
		stat = os.statvfs("/dev/shm")
	except OSError:
		return  # no /dev/shm (e.g. macOS); POSIX shm is not tmpfs-backed there
	available = stat.f_bavail * stat.f_frsize
	if nbytes > available:
		raise TokenStoreCapacityError(
			f"prompt arena needs {nbytes} bytes of /dev/shm, only {available} "
			"are free"
		)


def prompt_arena_shm_name(shm_prefix: str) -> str:
	"""Name the node's prompt arena under a run's shm prefix.

	The prefix comes from ``RuntimeIdentity.shm_prefix``, so the existing
	prefix-based shm cleanup and the startup refusal on a stale prefix cover this
	segment exactly as they cover ``host_kv`` and ``input_ids``.
	"""
	if not isinstance(shm_prefix, str) or not shm_prefix:
		raise ValueError("shm_prefix must be a non-empty string")
	return f"{shm_prefix}{PROMPT_ARENA_SHM_SUFFIX}"


class TokenStoreCapacityError(RuntimeError):
	"""A prompt-arena allocation exceeded the arena's free pages."""


class PromptTokenArena:
	"""Node-shared paged store for prompt token ids, with one writer per node.

	The single mapping is laid out as::

		[0, _HEADER_SLOTS*8)                  int64 header
		[next_page_offset, +num_pages*4)      int32 page chain: -1 ends a chain,
		                                      -2 marks a page on the free list
		[data_offset, +num_pages*page_bytes)  the token pages

	The chain array lives in a header region of the SAME mapping rather than in a
	second segment: a reader has one name to open and can therefore never observe
	tokens without the chain that indexes them, and the prefix-based cleanup has
	one fewer name to chase. It costs ``num_pages * 4`` bytes — 0.024% of the
	arena at the default page size — and stays sparse like the data region, since
	nothing writes a chain entry until a chain is built.

	Creation is sparse on purpose: the segment is ``ftruncate``'d to its full size
	and never zero-filled or otherwise touched, so a multi-GiB arena costs nothing
	at startup and resident memory tracks written pages only (POSIX guarantees the
	untouched pages read as zero).

	Exactly one process per node creates the arena and owns allocation; its free
	list is process-local, so there is no cross-process lock. Everybody else
	calls :meth:`attach` and reads by handle.

	Contracts for callers:

	* Publish a handle only after :meth:`write` returns (tokens and chain are
	  written by then), and free it only after every reader has dropped it:
	  reads take no lock, so a concurrent free + write can tear or alias a read.
	* Handles come only from :meth:`write`; they are not validated against the
	  start of a chain.
	* Attach after the creator has returned (a barrier), and only from
	  multiprocessing descendants of the server that also spawned the creator:
	  before Python 3.13 an attach registers the name with the resource tracker,
	  and a process with its own tracker would unlink the name when it exits.
	"""

	def __init__(
		self,
		name: str,
		*,
		create: bool,
		capacity_bytes: Optional[int] = None,
		page_size_tokens: int = DEFAULT_PAGE_SIZE_TOKENS,
		release_free_fraction: float = DEFAULT_RELEASE_FREE_FRACTION,
	):
		"""Low-level constructor; prefer :meth:`create` / :meth:`attach`."""
		self._name = name
		self._is_creator = bool(create)
		self._unlinked = False
		self._release_free_fraction = float(release_free_fraction)
		self._can_release = True
		# Pages freed since the last release pass; each pass punches only these.
		self._release_pending: List[int] = []
		self._shm: Optional[shared_memory.SharedMemory] = None
		# Kept after close() so the creator can still unlink the name.
		self._shm_ref: Optional[shared_memory.SharedMemory] = None
		self._next_page: Optional[np.ndarray] = None
		self._data: Optional[np.ndarray] = None

		layout: Optional[np.ndarray] = None
		if self._is_creator:
			if capacity_bytes is None:
				raise ValueError("capacity_bytes is required when creating an arena")
			if page_size_tokens <= 0:
				raise ValueError(
					f"page_size_tokens must be > 0, got {page_size_tokens}"
				)
			if not 0.0 < self._release_free_fraction <= 1.0:
				raise ValueError(
					"release_free_fraction must be in (0, 1], got "
					f"{release_free_fraction}"
				)
			page_bytes = page_size_tokens * _TOKEN_BYTES
			num_pages = int(capacity_bytes) // page_bytes
			if num_pages < 1:
				raise ValueError(
					f"capacity_bytes {capacity_bytes} does not hold one "
					f"{page_size_tokens}-token page ({page_bytes} B)"
				)
			next_page_offset = _align_up(_HEADER_SLOTS * 8, _PAGESIZE)
			data_offset = _align_up(next_page_offset + num_pages * 4, _PAGESIZE)
			layout = np.array(
				[
					_HEADER_MAGIC,
					_HEADER_VERSION,
					page_size_tokens,
					num_pages,
					next_page_offset,
					data_offset,
					0,
					0,
				],
				dtype=np.int64,
			)
			size = data_offset + num_pages * page_bytes
			_check_shm_space(size)
			# Exclusive creation (O_EXCL): a name collision must fail loudly
			# rather than silently adopt another run's or a stale segment.
			self._shm = shared_memory.SharedMemory(name=name, create=True, size=size)
		else:
			self._shm = shared_memory.SharedMemory(name=name)
		self._shm_ref = self._shm
		try:
			self._map_views(layout)
		except BaseException:
			# Never leave a mapping (or, for the creator, a name) behind on a
			# rejected segment: the next attempt would inherit the debris.
			self._unmap(destroy=self._is_creator)
			raise

		# Writer-private bookkeeping. ``_next_new_page`` is a bump pointer over
		# pages that have never been handed out, so creation stays O(1) instead
		# of seeding a free list with millions of entries; ``_free_pages`` is a
		# LIFO of returned pages.
		self._free_pages: List[int] = []
		self._next_new_page = 0

	@classmethod
	def create(
		cls,
		shm_prefix: str,
		*,
		capacity_bytes: int,
		page_size_tokens: int = DEFAULT_PAGE_SIZE_TOKENS,
		release_free_fraction: float = DEFAULT_RELEASE_FREE_FRACTION,
	) -> "PromptTokenArena":
		"""Create the node's arena at a fixed token capacity in bytes.

		The capacity sizes the TOKEN DATA; the header and chain array add about
		0.024% on top. Raises ``FileExistsError`` if the name is already taken.
		"""
		return cls(
			prompt_arena_shm_name(shm_prefix),
			create=True,
			capacity_bytes=capacity_bytes,
			page_size_tokens=page_size_tokens,
			release_free_fraction=release_free_fraction,
		)

	@classmethod
	def attach(cls, shm_prefix: str) -> "PromptTokenArena":
		"""Attach to the node's existing arena for reading only."""
		return cls(prompt_arena_shm_name(shm_prefix), create=False)

	# ---------------------------------------------------------------- geometry

	@property
	def name(self) -> str:
		return self._name

	@property
	def is_creator(self) -> bool:
		return self._is_creator

	@property
	def page_size_tokens(self) -> int:
		return self._page_size_tokens

	@property
	def num_pages(self) -> int:
		return self._num_pages

	@property
	def capacity_tokens(self) -> int:
		return self._num_pages * self._page_size_tokens

	@property
	def allocated_pages(self) -> int:
		"""Pages currently held by live chains (writer only)."""
		self._require_writer("report allocation")
		return self._next_new_page - len(self._free_pages)

	@property
	def free_pages(self) -> int:
		"""Pages available to :meth:`write` (writer only)."""
		return self._num_pages - self.allocated_pages

	def pages_for(self, n_tokens: int) -> int:
		"""Whole pages needed to hold ``n_tokens`` tokens."""
		n_tokens = int(n_tokens)
		if n_tokens < 0:
			raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
		return -(-n_tokens // self._page_size_tokens)

	# ------------------------------------------------------------ write / read

	def write(self, tokens: np.ndarray | Sequence[int]) -> TokenHandle:
		"""Copy ``tokens`` into a fresh page chain and return ``(first_page, length)``.

		``tokens`` may be any integer dtype or sequence; ids are stored as int32.
		"""
		self._require_writer("write tokens")
		src = _as_token_ids(tokens)
		if src.ndim != 1:
			raise ValueError(f"tokens must be 1-D, got shape {src.shape}")
		length = int(src.size)
		if length == 0:
			raise ValueError(
				"cannot write an empty token sequence: a handle must name a page"
			)
		pages = self._alloc_pages(self.pages_for(length))
		page_size = self._page_size_tokens
		last = len(pages) - 1
		for i, page in enumerate(pages):
			chunk = src[i * page_size : (i + 1) * page_size]
			base = page * page_size
			self._data[base : base + chunk.size] = chunk
			self._next_page[page] = END_OF_CHAIN if i == last else pages[i + 1]
		return (pages[0], length)

	def read(self, handle: TokenHandle) -> np.ndarray:
		"""Gather a handle's tokens into a fresh contiguous int32 array."""
		length = self._handle_length(handle)
		return self.read_into(handle, np.empty(length, dtype=_TOKEN_DTYPE))

	def read_into(self, handle: TokenHandle, out: np.ndarray) -> np.ndarray:
		"""Gather a handle's tokens into ``out`` without an intermediate array.

		``out`` may be int32 or int64 — int64 because the GPU input tensors are
		int64, and the per-page copy casts on the way in. ``out`` may be longer
		than the handle; the filled ``out[:length]`` view is returned.
		"""
		length = self._handle_length(handle)
		if not isinstance(out, np.ndarray):
			raise TypeError(f"out must be a numpy array, got {type(out).__name__}")
		if out.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
			raise ValueError(f"out must be int32 or int64, got {out.dtype}")
		if out.ndim != 1:
			raise ValueError(f"out must be 1-D, got shape {out.shape}")
		if out.size < length:
			raise ValueError(
				f"out holds {out.size} tokens, handle {tuple(handle)} has {length}"
			)
		page_size = self._page_size_tokens
		pos = 0
		for page in self.page_chain(handle):
			count = min(page_size, length - pos)
			base = page * page_size
			out[pos : pos + count] = self._data[base : base + count]
			pos += count
		return out[:length]

	def page_chain(self, handle: TokenHandle) -> List[int]:
		"""The pages a handle spans, first to last, validating chain integrity."""
		first_page = int(handle[0])
		length = self._handle_length(handle)
		count = self.pages_for(length)
		pages: List[int] = []
		page = first_page
		for i in range(count):
			if not 0 <= page < self._num_pages:
				raise ValueError(
					f"handle {tuple(handle)}: page {page} is outside "
					f"[0, {self._num_pages}) at chain position {i}"
				)
			pages.append(page)
			nxt = int(self._next_page[page])
			if nxt == _FREE_PAGE:
				raise ValueError(
					f"handle {tuple(handle)}: page {page} is on the free list; "
					"the chain was already freed"
				)
			if i + 1 < count and nxt == END_OF_CHAIN:
				raise ValueError(
					f"handle {tuple(handle)}: chain ended after {i + 1} of "
					f"{count} pages"
				)
			if i + 1 == count and nxt != END_OF_CHAIN:
				raise ValueError(
					f"handle {tuple(handle)}: chain continues past its "
					f"{count}th page"
				)
			page = nxt
		return pages

	def free(self, handle: TokenHandle) -> int:
		"""Return a chain's pages to the free list; returns the pages freed.

		Freeing or reading the handle again raises only while its pages are
		still free (marked ``_FREE_PAGE``); a later :meth:`write` may reuse them.
		"""
		self._require_writer("free a chain")
		pages = self.page_chain(handle)
		for page in pages:
			self._next_page[page] = _FREE_PAGE
		self._free_pages.extend(pages)
		self._release_pending.extend(pages)
		threshold = int(self._num_pages * self._release_free_fraction)
		if threshold > 0 and len(self._release_pending) >= threshold:
			self.release_free_memory()
		return len(pages)

	# -------------------------------------------------------------- lifecycle

	def release_free_memory(self) -> int:
		"""Drop the physical memory behind freed pages; returns bytes released.

		``posix_fadvise`` does not apply to an anonymous shm mapping, so this
		punches holes with ``MADV_REMOVE``, which zero-fills the range and frees
		its tmpfs pages. It is advisory and Linux-only: platforms without
		``mmap.MADV_REMOVE`` (macOS) and mappings the kernel refuses to punch
		skip silently, leaving freed pages resident but still reusable.

		Only byte ranges that are system-page aligned INSIDE a run of free pages
		are advised, so live tokens sharing a system page are never touched.
		"""
		self._require_writer("release free memory")
		pending, self._release_pending = self._release_pending, []
		# SharedMemory exposes no mmap or fd, and madvise is an mmap method, so
		# the private mapping is the only route to it — hence the getattr guard.
		mapping = getattr(self._shm, "_mmap", None)
		if (
			not self._can_release
			or mapping is None
			or not hasattr(mapping, "madvise")
			or not hasattr(mmap, "MADV_REMOVE")
		):
			return 0
		released = 0
		# Pages freed earlier and already reused by a write are no longer free.
		still_free = [p for p in pending if int(self._next_page[p]) == _FREE_PAGE]
		for start_page, end_page in self._runs(still_free):
			low = _align_up(self._data_offset + start_page * self._page_bytes, _PAGESIZE)
			high = _align_down(self._data_offset + end_page * self._page_bytes, _PAGESIZE)
			if high <= low:
				continue
			try:
				mapping.madvise(mmap.MADV_REMOVE, low, high - low)
			except (OSError, ValueError):
				# Not hole-punchable here; stop asking for the rest of the run.
				self._can_release = False
				return released
			released += high - low
		return released

	def close(self) -> None:
		"""Unmap this process's view. Idempotent; readers call only this."""
		self._unmap(destroy=False)

	def unlink(self) -> None:
		"""Unmap and remove the segment's name. Only the creator may do this."""
		if not self._is_creator:
			raise RuntimeError(
				f"prompt arena '{self._name}' was attached, not created; only "
				"the creator may unlink it"
			)
		self._unmap(destroy=True)

	# ----------------------------------------------------------------- private

	def _map_views(self, layout: Optional[np.ndarray]) -> None:
		"""Write or validate the header, then bind the chain and data views."""
		header_bytes = _HEADER_SLOTS * 8
		if layout is not None:
			self._shm.buf[:header_bytes] = layout.tobytes()
		# Read the header as a COPY. A live numpy view of the mapping exports a
		# buffer, and an export outstanding on any failure path below would make
		# mmap.close() raise BufferError instead of unmapping.
		header = np.frombuffer(bytes(self._shm.buf[:header_bytes]), dtype=np.int64)
		if layout is None:
			if int(header[0]) != _HEADER_MAGIC:
				raise ValueError(
					f"shared segment '{self._name}' is not a prompt arena "
					f"(magic {int(header[0]):#x})"
				)
			if int(header[1]) != _HEADER_VERSION:
				raise ValueError(
					f"prompt arena '{self._name}' has layout version "
					f"{int(header[1])}, this build reads version {_HEADER_VERSION}"
				)
		if layout is None and not (
			int(header[2]) > 0
			and int(header[3]) > 0
			and int(header[4]) >= header_bytes
			and int(header[5]) >= int(header[4]) + int(header[3]) * 4
		):
			raise ValueError(
				f"prompt arena '{self._name}' has an invalid header "
				f"{[int(v) for v in header[:6]]}"
			)
		self._page_size_tokens = int(header[2])
		self._num_pages = int(header[3])
		self._page_bytes = self._page_size_tokens * _TOKEN_BYTES
		self._next_page_offset = int(header[4])
		self._data_offset = int(header[5])
		required = self._data_offset + self._num_pages * self._page_bytes
		if self._shm.size < required:
			raise TokenStoreCapacityError(
				f"prompt arena '{self._name}' is {self._shm.size} bytes, its "
				f"header describes {required} ({self._num_pages} pages x "
				f"{self._page_size_tokens} tokens x {_TOKEN_BYTES} B)"
			)
		self._next_page = np.frombuffer(
			self._shm.buf,
			dtype=_TOKEN_DTYPE,
			count=self._num_pages,
			offset=self._next_page_offset,
		)
		self._data = np.frombuffer(
			self._shm.buf,
			dtype=_TOKEN_DTYPE,
			count=self._num_pages * self._page_size_tokens,
			offset=self._data_offset,
		)
		if not self._is_creator:
			# ``shared_memory`` has no read-only attach mode (it always opens
			# O_RDWR), so the read-only contract is enforced on the numpy views
			# and by _require_writer(), not by the page protection.
			self._next_page.flags.writeable = False
			self._data.flags.writeable = False

	def _unmap(self, *, destroy: bool) -> None:
		"""Drop the views and the mapping; optionally unlink the name.

		Unlinking does not depend on the mapping: a creator may close() first and
		unlink() later. The name goes before the mapping so that a failing
		close() cannot leave the segment behind.
		"""
		# The views must go first: they export buffers from the mapping, and
		# nothing this class hands out is a view into it, so dropping our own
		# references is enough to let mmap.close() through.
		self._next_page = None
		self._data = None
		if destroy and not self._unlinked and self._shm_ref is not None:
			self._shm_ref.unlink()
			self._unlinked = True
		shm = self._shm
		self._shm = None
		if shm is not None:
			shm.close()

	def _require_writer(self, action: str) -> None:
		if not self._is_creator:
			raise RuntimeError(
				f"prompt arena '{self._name}' is attached read-only and may not "
				f"{action}; the node's single writer owns allocation"
			)

	def _handle_length(self, handle: TokenHandle) -> int:
		length = int(handle[1])
		if length <= 0:
			raise ValueError(
				f"handle {tuple(handle)} has length {length}; every handle spans "
				"at least one token"
			)
		return length

	def _alloc_pages(self, count: int) -> List[int]:
		if count > self.free_pages:
			raise TokenStoreCapacityError(
				f"prompt arena '{self._name}': this allocation needs {count} "
				f"pages, only {self.free_pages} of {self._num_pages} capacity "
				f"pages are free"
			)
		pages: List[int] = []
		while len(pages) < count and self._free_pages:
			pages.append(self._free_pages.pop())
		while len(pages) < count:
			pages.append(self._next_new_page)
			self._next_new_page += 1
		return pages

	@staticmethod
	def _runs(pages: List[int]) -> Iterator[Tuple[int, int]]:
		"""Maximal runs of consecutive pages, as ``[start, end)`` indices."""
		run_start: Optional[int] = None
		prev: Optional[int] = None
		for page in sorted(set(pages)):
			if run_start is None:
				run_start = page
			elif page != prev + 1:
				yield run_start, prev + 1
				run_start = page
			prev = page
		if run_start is not None:
			yield run_start, prev + 1


class _SeqTokens:
	"""One sequence's decoded chunks and its logical token count."""

	__slots__ = ("chunks", "length")

	def __init__(self):
		self.chunks: List[np.ndarray] = []
		self.length = 0


class DecodedTokenStore:
	"""Rank-private chunked store for generated token ids (int32).

	Only the owning rank writes a sequence's decoded tokens, so nothing here is
	shared, named, or locked. A sequence holds a list of ``chunk_size_tokens``
	int32 chunks: it costs one chunk until it decodes past it, and :meth:`free`
	drops its chunks immediately. There is no rectangular
	``[max_pool_size, max_decoding_length]`` preallocation to grow or leak.

	Every sequence id starts empty — ``append*`` creates its record lazily, and
	reading or freeing an id that was never written is not an error, which keeps
	the decode hot path free of existence checks.
	"""

	def __init__(self, chunk_size_tokens: int = DEFAULT_DECODED_CHUNK_TOKENS):
		if chunk_size_tokens <= 0:
			raise ValueError(
				f"chunk_size_tokens must be > 0, got {chunk_size_tokens}"
			)
		self._chunk_size = int(chunk_size_tokens)
		self._seqs: Dict[int, _SeqTokens] = {}

	@property
	def chunk_size_tokens(self) -> int:
		return self._chunk_size

	@property
	def active_sequences(self) -> int:
		return len(self._seqs)

	def append(self, seq_id: int, tokens: np.ndarray | Sequence[int] | int) -> None:
		"""Append one sequence's tokens.

		O(chunks touched) numpy copies, independent of the token count.
		"""
		src = _as_token_ids(tokens)
		if src.ndim == 0:
			src = src.reshape(1)
		elif src.ndim != 1:
			raise ValueError(f"tokens must be 1-D, got shape {src.shape}")
		if src.size == 0:
			return
		record = self._record(int(seq_id))
		chunk_size = self._chunk_size
		pos = 0
		while pos < src.size:
			offset = record.length % chunk_size
			if offset == 0:
				record.chunks.append(np.empty(chunk_size, dtype=_TOKEN_DTYPE))
			count = min(chunk_size - offset, int(src.size) - pos)
			record.chunks[-1][offset : offset + count] = src[pos : pos + count]
			record.length += count
			pos += count

	def append_batch(self, seq_ids: np.ndarray, tokens: np.ndarray) -> None:
		"""Append ONE token per sequence — the decode-step hot path.

		Complexity is O(batch): one dict lookup and one int store per sequence,
		with the numpy-to-Python conversion done once per array (``tolist``)
		instead of once per element. A single vectorized scatter is not possible
		because the destination chunks are per-sequence, but nothing in the loop
		allocates unless a sequence crosses a chunk boundary.
		"""
		ids = np.ascontiguousarray(seq_ids).reshape(-1).tolist()
		# Validate before the loop so a bad id cannot leave a half-appended chunk.
		values = _as_token_ids(tokens).reshape(-1).tolist()
		if len(ids) != len(values):
			raise ValueError(
				f"seq_ids has {len(ids)} entries, tokens has {len(values)}"
			)
		chunk_size = self._chunk_size
		seqs = self._seqs
		for seq_id, token in zip(ids, values):
			record = seqs.get(seq_id)
			if record is None:
				record = seqs[seq_id] = _SeqTokens()
			offset = record.length % chunk_size
			if offset == 0:
				record.chunks.append(np.empty(chunk_size, dtype=_TOKEN_DTYPE))
			record.chunks[-1][offset] = token
			record.length += 1

	def extend_rows(
		self,
		seq_ids: np.ndarray,
		tokens: np.ndarray,
		counts: np.ndarray | int | None = None,
	) -> None:
		"""Flush a ``[steps, batch]`` step log column-wise.

		``tokens[:, j]`` holds the tokens generated for ``seq_ids[j]`` in step
		order. ``counts`` is the valid step count — one int for every column, one
		per column, or ``None`` for all ``steps`` rows. One numpy copy per column
		(a column is strided, so the copy is unavoidable) and O(batch) Python
		work.
		"""
		log = _as_token_ids(tokens)
		if log.ndim != 2:
			raise ValueError(f"tokens must be [steps, batch], got shape {log.shape}")
		steps, batch = log.shape
		ids = np.ascontiguousarray(seq_ids).reshape(-1).tolist()
		if len(ids) != batch:
			raise ValueError(
				f"seq_ids has {len(ids)} entries, tokens has {batch} columns"
			)
		if counts is None:
			per_column = [steps] * batch
		elif np.ndim(counts) == 0:
			per_column = [int(counts)] * batch
		else:
			per_column = np.ascontiguousarray(counts).reshape(-1).tolist()
			if len(per_column) != batch:
				raise ValueError(
					f"counts has {len(per_column)} entries, tokens has {batch} "
					"columns"
				)
		for column, seq_id in enumerate(ids):
			count = int(per_column[column])
			if not 0 <= count <= steps:
				raise ValueError(
					f"count {count} for column {column} is outside [0, {steps}]"
				)
			if count:
				self.append(seq_id, log[:count, column])

	def read(
		self, seq_id: int, start: int = 0, end: Optional[int] = None
	) -> np.ndarray:
		"""Copy ``[start, end)`` of a sequence's decoded tokens into a new array."""
		length = self.length(seq_id)
		start = int(start)
		end = length if end is None else int(end)
		if not 0 <= start <= end <= length:
			raise ValueError(
				f"sequence {seq_id}: [{start}, {end}) is outside its {length} "
				"decoded tokens"
			)
		out = np.empty(end - start, dtype=_TOKEN_DTYPE)
		if out.size == 0:
			return out
		chunk_size = self._chunk_size
		chunks = self._seqs[int(seq_id)].chunks
		index, offset = divmod(start, chunk_size)
		pos = 0
		while pos < out.size:
			count = min(chunk_size - offset, int(out.size) - pos)
			out[pos : pos + count] = chunks[index][offset : offset + count]
			pos += count
			index += 1
			offset = 0
		return out

	def length(self, seq_id: int) -> int:
		"""Decoded tokens held for ``seq_id``; 0 if it was never written."""
		record = self._seqs.get(int(seq_id))
		return 0 if record is None else record.length

	def free(self, seq_id: int) -> int:
		"""Drop a sequence's chunks; returns the tokens dropped (0 if unknown)."""
		record = self._seqs.pop(int(seq_id), None)
		return 0 if record is None else record.length

	def _record(self, seq_id: int) -> _SeqTokens:
		record = self._seqs.get(seq_id)
		if record is None:
			record = self._seqs[seq_id] = _SeqTokens()
		return record
