"""One-step lookahead state for the continuous-decode loop.

Synchronously, a decode step ends by waiting for its own sampled-token
readback and then doing the host bookkeeping (trajectory append, EOS/stop,
max-length, repetition).  That wait sits between two GPU forwards: the host
cannot launch step N+1 until step N's tokens have landed in pinned memory,
even though step N+1's *compute* needs nothing the host read back --- the
sampled token stays on device across steps and ``cache_seqlens`` advances
device-side.

With one-step lookahead the readback wait and the bookkeeping for step N run
during step N+1, after that step's forward and KV-append copies are already
launched.  This object owns the state that makes the reordering safe:

* which rows produced the in-flight tokens (``rows``), so a composition
  change between the two steps is a named error instead of a silent
  mis-attribution of tokens to rows;
* which of those rows may *accept* a token (``active``), evaluated exactly
  once per step at arm time --- a row retired by an earlier step's token
  still computes a token, and that token must be discarded, never appended;
* the per-row ``+1`` context offset that step N+1's forward needs, because
  the host has not yet advanced ``decoded_length`` for the unapplied token.

The offset is what keeps the GPU's view identical to the synchronous build.
``current_context_length`` and ``decoded_length`` keep meaning "tokens
actually accepted", so nothing is ever rolled back: the lookahead adds the
pending token to the *derived* per-step vector only.

Pure state plus pure arithmetic, no torch: the invariants above are the ones
worth testing on CPU, and the loop keeps the device buffers.
"""

from dataclasses import dataclass
from typing import Any, Sequence, Tuple


class DecodeLookaheadError(RuntimeError):
	"""Base class for lookahead contract violations."""


class LookaheadArmError(DecodeLookaheadError):
	"""The pipeline was armed twice, or read while empty."""


class LookaheadCompositionError(DecodeLookaheadError):
	"""The decode row set moved while a step's tokens were still in flight.

	Admission, retirement and page-boundary rebinds change the ordered row
	list.  The loop must apply the pending step before any of them, so this
	is a lost flush, not a recoverable condition: continuing would append
	one row's token to another row's trajectory.
	"""


@dataclass(frozen=True)
class PendingDecodeStep:
	"""One step's in-flight sampled tokens.

	``slot`` indexes the loop's pinned-buffer ring; ``sequences`` are the
	``SequenceEntry`` objects in row order.  ``active[i]`` is False for a row
	that was already retired when this step was launched: it still computed a
	token (identically to the synchronous path, which also forwards retired
	rows until the next boundary) and that token is discarded.
	"""

	rows: Tuple[int, ...]
	sequences: Tuple[Any, ...]
	active: Tuple[bool, ...]
	num_rows: int
	slot: int


class DecodeLookaheadState:
	"""At most one decode step's tokens in flight at a time."""

	__slots__ = ("_pending",)

	def __init__(self) -> None:
		self._pending: PendingDecodeStep | None = None

	@property
	def is_armed(self) -> bool:
		return self._pending is not None

	def arm(
		self,
		rows: Sequence[int],
		sequences: Sequence[Any],
		active: Sequence[bool],
		slot: int,
	) -> None:
		"""Record that ``rows`` have one sampled token in flight."""
		if self._pending is not None:
			raise LookaheadArmError(
				"decode lookahead armed twice: step "
				f"{self._pending.rows[:8]} is still pending; the loop must "
				"apply a pending step before launching the next one"
			)
		rows_t = tuple(int(r) for r in rows)
		sequences_t = tuple(sequences)
		active_t = tuple(bool(a) for a in active)
		if not (len(rows_t) == len(sequences_t) == len(active_t)):
			raise LookaheadArmError(
				"decode lookahead arm shape mismatch: "
				f"rows={len(rows_t)}, sequences={len(sequences_t)}, "
				f"active={len(active_t)}"
			)
		self._pending = PendingDecodeStep(
			rows=rows_t,
			sequences=sequences_t,
			active=active_t,
			num_rows=len(rows_t),
			slot=int(slot),
		)

	def take(self) -> PendingDecodeStep:
		"""Hand out the pending step and disarm."""
		pending = self._pending
		if pending is None:
			raise LookaheadArmError(
				"decode lookahead take() with nothing in flight"
			)
		self._pending = None
		return pending

	def context_offsets(self, rows: Sequence[int]) -> Tuple[int, ...]:
		"""Per-row ``+1`` for rows holding an unapplied produced token.

		Zero for every row when nothing is in flight, so the first step of a
		round and every step after a flush build exactly the synchronous
		vector.  A row that was inactive in the pending step contributes 0:
		its context is frozen, which is what the synchronous path does for a
		retired row too.
		"""
		rows_t = tuple(int(r) for r in rows)
		pending = self._pending
		if pending is None:
			return (0,) * len(rows_t)
		if rows_t != pending.rows:
			raise LookaheadCompositionError(
				"decode row set changed while a sampled step was in flight: "
				f"pending={pending.rows[:8]} (n={pending.num_rows}), "
				f"current={rows_t[:8]} (n={len(rows_t)}); the pending step "
				"must be applied before any composition change"
			)
		return tuple(1 if a else 0 for a in pending.active)


def effective_cache_seqlens(
	host_context_lengths: Sequence[int], offsets: Sequence[int]
) -> list:
	"""Per-row ``cache_seqlens`` for the next forward.

	``host_context_lengths[i]`` counts the tokens this row has *accepted*;
	``offsets[i]`` adds the one token it produced but whose host bookkeeping
	has not run yet.  The sum is exactly the value the synchronous path would
	have passed, which is what keeps the sampled tokens bit-identical.
	"""
	if len(host_context_lengths) != len(offsets):
		raise DecodeLookaheadError(
			"cache_seqlens offset shape mismatch: "
			f"contexts={len(host_context_lengths)}, offsets={len(offsets)}"
		)
	out = []
	for ctx, offset in zip(host_context_lengths, offsets):
		if offset not in (0, 1):
			raise DecodeLookaheadError(
				f"lookahead context offset must be 0 or 1, got {offset}"
			)
		out.append(int(ctx) + int(offset))
	return out
