"""Prefill scheduling — host-KV-capacity-bounded sequence selection.

Slice 6 of the worker decouple initiative (issue #175). Extracts the
pure *selection decision* from ``_prepare_prefill_batch``:

  - ``PrefillScheduler.select_prefill_batch`` — greedily admit candidate
    sequences (evicted first, by recompute priority; then queued, by
    arrival) into a prefill batch, bounded by each node's free host-KV
    pages.

Only the *decision* is ported. The cross-rank ``dist.all_gather`` that
collects per-node host-KV free pages, the candidate enumeration over
``global_batch``, and the rank-0 logging stay on the worker, which
builds the request and uses the returned uuid list.

Host KV is per-node: a sequence assigned to a rank on node N draws from
node N's host-KV capacity, so selection is a per-node bin-packing, not a
global one.

Design follows the per-slice frozen-snapshot pattern (no shared mutable
``WorkerState``): the worker passes exactly the fields this decision
consumes; the handler is pure and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple


def host_kv_eviction_enabled(host_kv_chunk_size: int) -> bool:
	"""Return whether bounded chunked host-KV mode requires re-entry."""
	return int(host_kv_chunk_size) > 0


def compute_prefill_host_reservation(
    *,
    prompt_length: int,
    kv_token_budget: int,
    page_size: int,
    chunk_size: int,
    initial_gpu_page_buffer: int,
) -> Tuple[int, int]:
	"""Return ``(pages, rounded_tokens)`` for one prefill reservation.

	The selector and the worker allocator must use the same reservation
	formula.  Keeping the arithmetic here makes the capacity snapshot and the
	actual allocation one contract instead of two copies that can drift.
	"""
	if prompt_length < 0:
		raise ValueError("prompt_length must be >= 0")
	if kv_token_budget <= 0:
		raise ValueError("kv_token_budget must be > 0")
	if page_size <= 0:
		raise ValueError("page_size must be > 0")
	if chunk_size < 0:
		raise ValueError("chunk_size must be >= 0")
	if initial_gpu_page_buffer < 0:
		raise ValueError("initial_gpu_page_buffer must be >= 0")

	post_prefill_length = prompt_length + 1
	gpu_initial_pages = (
		math.ceil(post_prefill_length / page_size) + initial_gpu_page_buffer
	)
	gpu_initial_tokens = gpu_initial_pages * page_size
	capacity = min(
		max(prompt_length + chunk_size, gpu_initial_tokens),
		kv_token_budget,
	)
	pages = math.ceil(capacity / page_size)
	return pages, pages * page_size


@dataclass(frozen=True)
class PrefillCandidate:
    """A sequence eligible for prefill admission (EVICTED or QUEUEING).

    ``is_evicted`` selects the priority group: evicted sequences are
    admitted first (ordered by most-decoded-first to minimise wasted
    recompute), then queueing sequences (ordered by ``prefill_order``,
    defaulting to arrival ``global_idx``).
    """

    uuid: str
    assigned_rank: int
    node_id: int
    is_evicted: bool
    global_idx: int
    total_decoded_before_eviction: int
    prompt_length: int
    kv_token_budget: int
    page_size: int
    # TP attention can replicate one sequence's host KV allocation on every
    # rank in its serve group.  The node-level allocator therefore consumes
    # this many copies of ``req_pages`` for one admitted sequence.
    host_kv_replication_factor: int = 1
    # Read-only prefix estimate used only when physical free pages cannot
    # admit any candidate. Attached pages already occupy the Host pool.
    cached_prefix_pages: int = 0
    # Queueing admission order. ``None`` means "use ``global_idx``" (arrival).
    # The worker stamps it so prompts sharing a prefix stay adjacent and land
    # in one wave, instead of being split by a capacity cut and recomputing
    # their shared prefix.
    prefill_order: Optional[int] = None


@dataclass(frozen=True)
class PrefillSelectionRequest:
    """Frozen snapshot for ``select_prefill_batch``.

    ``per_node_host_free`` is indexed by node id (gathered across ranks).
    ``initial_gpu_page_buffer`` is ``INITIAL_GPU_PAGE_BUFFER`` — pages
    reserved on a sequence's first GPU load.
    """

    candidates: Tuple[PrefillCandidate, ...]
    per_node_host_free: Tuple[int, ...]
    chunk_size: int
    num_nodes: int
    gpus_per_node: int
    initial_gpu_page_buffer: int
    # Persistent model state is a separate constraint from the token cap.
    # KDA keeps one state slot for a sequence until completion/eviction.
    # Free capacity can differ after prior work, so carry the gathered
    # rank/node vectors rather than one caller-local scalar.
    per_rank_sequence_free: Optional[Tuple[int, ...]] = None
    per_node_sequence_free: Optional[Tuple[int, ...]] = None
    # Retained prefix pages a node can reclaim right now (resident, no
    # attachment, no pending load), indexed by node id. They join the host
    # budget because a retained prefix either gets attached -- dropping demand
    # and supply by the same page count -- or stays evictable for the
    # reserve-time reclaim. ``None`` means "do not count them" and keeps every
    # non-prefix caller on physical free pages alone.
    per_node_host_reclaimable: Optional[Tuple[int, ...]] = None


class PrefillScheduler:
    """Prefill admission decision — pure, deterministic across ranks."""

    @staticmethod
    def required_host_pages(
        candidate: PrefillCandidate, req: PrefillSelectionRequest
    ) -> int:
        """Return the same initial Host-KV reservation used by admission.

        Delegates to ``compute_prefill_host_reservation`` so the selector and
        the worker allocator stay one formula, then discounts pages already
        resident in the prefix cache and multiplies the node replication
        factor.
        """
        pages_per_replica, _ = compute_prefill_host_reservation(
            prompt_length=candidate.prompt_length,
            kv_token_budget=candidate.kv_token_budget,
            page_size=candidate.page_size,
            chunk_size=req.chunk_size,
            initial_gpu_page_buffer=req.initial_gpu_page_buffer,
        )
        if not 0 <= candidate.cached_prefix_pages <= pages_per_replica:
            raise ValueError(
                f"candidate {candidate.uuid} has invalid cached_prefix_pages="
                f"{candidate.cached_prefix_pages} for reservation {pages_per_replica}"
            )
        return (
            pages_per_replica - candidate.cached_prefix_pages
        ) * candidate.host_kv_replication_factor

    @staticmethod
    def select_prefill_batch(req: PrefillSelectionRequest) -> List[str]:
        """Select which candidate sequences to prefill, bounded by host KV.

        Priority order: EVICTED sequences first (most decoded → least
        wasted recompute, ``global_idx`` tie-break), then QUEUEING (by
        ``prefill_order`` where set, else arrival ``global_idx``, with
        ``global_idx`` as tie-break). Each candidate is admitted iff its
        node still has room for its initial page reservation.

        Initial reservation (matches the worker's dynamic host-KV sizing):
        ``max(prompt_length + chunk_size, gpu_initial_tokens)`` capped at
        ``kv_token_budget``, rounded up to whole pages — where
        ``gpu_initial_tokens`` covers ``prompt_length + 1`` plus the GPU
        page buffer. No safety margin: selection and allocation use the
        same formula by design.

        Pure: reads only the candidate snapshots + per-node free pages.
        The NCCL gather and the ``global_batch`` enumeration stay on the
        worker. Deterministic across ranks (stable sort keys), so every
        rank produces the identical batch without communication.
        """
        return [
            candidate.uuid
            for candidate in PrefillScheduler._admit(
                req, bound_by_host_pages=True
            )
        ]

    @staticmethod
    def slot_admissible_candidates(
        req: PrefillSelectionRequest,
    ) -> List[PrefillCandidate]:
        """Candidates the persistent sequence-slot limits alone would admit.

        The same priority walk as ``select_prefill_batch`` with the Host-page
        bound lifted, so the result is exactly the set a fully reclaimed Host
        pool could admit right now. Without slot limits it is every candidate.
        Admission reclaim targets this set: Host pages reclaimed for a
        candidate that has no free KDA slot would buy nothing, and a model
        with slot limits is not exempt from reclaim (a full pool of retained
        prefixes otherwise starves admission forever).
        """
        return PrefillScheduler._admit(req, bound_by_host_pages=False)

    @staticmethod
    def _admit(
        req: PrefillSelectionRequest, *, bound_by_host_pages: bool
    ) -> List[PrefillCandidate]:
        evicted = [c for c in req.candidates if c.is_evicted]
        queueing = [c for c in req.candidates if not c.is_evicted]
        evicted.sort(key=lambda c: (-c.total_decoded_before_eviction, c.global_idx))
        queueing.sort(
            key=lambda c: (
                c.global_idx if c.prefill_order is None else c.prefill_order,
                c.global_idx,
            )
        )
        all_candidates = evicted + queueing
        if not all_candidates:
            return []

        if (
            req.per_rank_sequence_free is not None
            and req.per_node_sequence_free is not None
        ):
            raise ValueError(
                "prefill sequence capacity must be scoped to rank or node, "
                "not both"
            )
        if req.per_rank_sequence_free is not None:
            if len(req.per_rank_sequence_free) == 0:
                raise ValueError("per_rank_sequence_free must not be empty")
            if any(value < 0 for value in req.per_rank_sequence_free):
                raise ValueError("per_rank_sequence_free values must be >= 0")
        if req.per_node_sequence_free is not None:
            if len(req.per_node_sequence_free) != req.num_nodes:
                raise ValueError(
                    "per_node_sequence_free must have one value per node"
                )
            if any(value < 0 for value in req.per_node_sequence_free):
                raise ValueError("per_node_sequence_free values must be >= 0")
        if req.per_node_host_reclaimable is not None:
            if len(req.per_node_host_reclaimable) != req.num_nodes:
                raise ValueError(
                    "per_node_host_reclaimable must have one value per node"
                )
            if any(value < 0 for value in req.per_node_host_reclaimable):
                raise ValueError(
                    "per_node_host_reclaimable values must be >= 0"
                )

        per_node_effective_free = list(req.per_node_host_free)
        if bound_by_host_pages and req.per_node_host_reclaimable is not None:
            per_node_effective_free = [
                free + reclaimable
                for free, reclaimable in zip(
                    per_node_effective_free, req.per_node_host_reclaimable
                )
            ]
        node_pages_used = [0] * req.num_nodes
        rank_sequences_used: dict[int, int] = {}
        node_sequences_used = [0] * req.num_nodes
        admitted: List[PrefillCandidate] = []

        for c in all_candidates:
            seq_node = c.node_id
            if seq_node < 0 or seq_node >= req.num_nodes:
                raise ValueError(
                    f"candidate {c.uuid} has invalid node_id={seq_node} "
                    f"for num_nodes={req.num_nodes}"
                )
            if (
                req.per_rank_sequence_free is not None
                and c.assigned_rank >= len(req.per_rank_sequence_free)
            ):
                raise ValueError(
                    f"candidate {c.uuid} has assigned_rank={c.assigned_rank} "
                    "outside per_rank_sequence_free"
                )
            if (
                req.per_rank_sequence_free is not None
                and rank_sequences_used.get(c.assigned_rank, 0)
                >= req.per_rank_sequence_free[c.assigned_rank]
            ):
                continue
            if (
                req.per_node_sequence_free is not None
                and node_sequences_used[seq_node]
                >= req.per_node_sequence_free[seq_node]
            ):
                continue
            if c.host_kv_replication_factor <= 0:
                raise ValueError(
                    f"candidate {c.uuid} has invalid host_kv_replication_factor="
                    f"{c.host_kv_replication_factor}"
                )
            req_pages = PrefillScheduler.required_host_pages(c, req)

            if (
                not bound_by_host_pages
                or node_pages_used[seq_node] + req_pages
                <= per_node_effective_free[seq_node]
            ):
                admitted.append(c)
                node_pages_used[seq_node] += req_pages
                rank_sequences_used[c.assigned_rank] = (
                    rank_sequences_used.get(c.assigned_rank, 0) + 1
                )
                node_sequences_used[seq_node] += 1

        return admitted


# Consecutive scheduler iterations that may admit nothing, with nothing live
# and no change in the gathered admission state, before the worker fails by
# name. Such an iteration cannot make progress (no decode runs, no pages or
# slots come back), so any small bound is correct; this one sits well above
# transient retries, which the state components reset anyway -- a rejected
# reservation snapshot, an in-flight prefix load and a changed candidate set
# are each their own component (see ``PrefillProgressGuard``).
PREFILL_NO_PROGRESS_LIMIT = 64

# Candidates one admission reclaim buys room for. Minimal-progress reclaim:
# one is enough to unblock the wave, and the next wave reclaims again if it
# still cannot admit. See ``prefill_reclaim_target_pages``.
PREFILL_RECLAIM_TARGET_CANDIDATES = 1


def prefill_reclaim_target_pages(
    *,
    fitting_requirements: Sequence[int],
    free_pages: int,
    max_candidates: int = PREFILL_RECLAIM_TARGET_CANDIDATES,
) -> Tuple[int, int]:
	"""Return ``(required_pages, page_target)`` for a minimal-progress reclaim.

	``fitting_requirements`` are the Host-page reservations of this node's
	slot-admissible candidates that the pool can hold at all, in admission
	priority order. The target covers the first ``max_candidates`` of them --
	the minimum that lets the HIGHEST-PRIORITY admissible candidate proceed --
	rather than the sum over the whole admissible set.

	Policy (minimal-progress reclaim): a wave only has to admit something.
	Reclaiming for the sum evicts retained prefixes the wave will not use, and
	every evicted prefix is a hot prefix some later prompt must recompute; if
	one wave's admission is still not enough, the next wave reclaims again from
	the same LRU order. Taking the first requirement rather than the smallest
	also keeps the priority walk honest: targeting the cheapest candidate can
	admit later arrivals forever while the head of the queue never fits.

	Returns ``(0, 0)`` when no candidate fits the pool at all.
	"""
	if int(max_candidates) <= 0:
		raise ValueError("max_candidates must be > 0")
	head = [int(pages) for pages in fitting_requirements[: int(max_candidates)]]
	if not head:
		return 0, 0
	required = sum(head)
	return required, max(0, required - int(free_pages))


def prefill_host_reserve_page_deficit(
    node_rows: Mapping[int, Tuple[int, int]],
    node_id: int,
) -> int:
	"""Return the Host pages one node is short of the wave's gathered demand.

	``node_rows`` maps node id to the ``(need, free)`` pair the reservation
	capacity collective gathered. The deficit is what the reserve-time reclaim
	has to release on this node for the SAME wave to pass the collective a
	second time; it is exactly the part of the admission budget that was
	reclaimable rather than free.

	Zero means this node is not the short one -- it then evicts nothing and the
	second collective decides on whichever node is. A missing or negative row
	is a broken snapshot, not a zero deficit, so it is named here instead of
	silently reclaiming nothing.
	"""
	row = node_rows.get(int(node_id))
	if row is None:
		raise ValueError(f"no gathered host-KV capacity row for node {node_id}")
	need, free = int(row[0]), int(row[1])
	if need < 0 or free < 0:
		raise ValueError(
			f"invalid host-KV capacity row for node {node_id}: "
			f"need={need}, free={free}"
		)
	return max(0, need - free)


class PrefillAdmissionStalled(RuntimeError):
	"""Queued sequences remain, none is admissible, and nothing can free room.

	Raised instead of returning an empty wave forever: with no live sequence
	nothing decodes, completes, or releases Host pages or sequence slots, so an
	empty selection would repeat unchanged on every scheduler iteration.

	NOTHING handles it. It leaves ``_prepare_prefill_batch`` through
	``generate()`` / ``generate_persistent()`` into the worker fatal boundary,
	which exits the rank and fails every active batch. That is the right
	mapping for an engine invariant violation (pages pinned with nothing live;
	an EVICTED sequence that no longer fits a pool it did fit) and it is
	exactly why a single impossible REQUEST must not come here: an oversized
	queued prompt is failed on the per-request error channel instead, by
	``_reject_unsatisfiable_prefill_requests``.
	"""


class PrefillAdmissionNoProgress(PrefillAdmissionStalled):
	"""The scheduler loop's hard guard tripped: repeated identical empty waves."""


class PrefillAdmissionStateDivergence(RuntimeError):
	"""Ranks disagree about the prefill admission candidate set.

	Admission reclaim and the no-progress guard both branch on the candidate
	snapshot, which every rank derives independently from a ``global_batch``
	that is supposed to be replicated. A divergence sends ranks down different
	branches -- one reclaiming and gathering while another walks on to the next
	collective -- so it is named here instead of deadlocking or raising on a
	single rank later.
	"""


def describe_prefill_blockers(
    req: PrefillSelectionRequest,
    *,
    total_pages_by_node: Optional[Sequence[int]] = None,
    limit: int = 4,
) -> str:
	"""Name why the first unadmitted candidates do not fit, in priority order.

	Pure and deterministic: the same request yields the same text on every
	rank. ``total_pages_by_node`` (when known) separates a request that can
	never fit the pool from one waiting for pages.
	"""
	selected = set(PrefillScheduler.select_prefill_batch(req))
	slot_admissible = {
		candidate.uuid
		for candidate in PrefillScheduler.slot_admissible_candidates(req)
	}
	ordered = sorted(
		req.candidates,
		key=lambda c: (
			0 if c.is_evicted else 1,
			-c.total_decoded_before_eviction if c.is_evicted else 0,
			c.global_idx if c.is_evicted or c.prefill_order is None
			else c.prefill_order,
			c.global_idx,
		),
	)
	parts: List[str] = []
	blocked = [c for c in ordered if c.uuid not in selected]
	for candidate in blocked[: max(0, int(limit))]:
		if candidate.uuid not in slot_admissible:
			if req.per_node_sequence_free is not None:
				free = req.per_node_sequence_free[candidate.node_id]
				scope = f"node {candidate.node_id}"
			elif req.per_rank_sequence_free is not None:
				free = req.per_rank_sequence_free[candidate.assigned_rank]
				scope = f"rank {candidate.assigned_rank}"
			else:  # pragma: no cover - slot_admissible is every candidate
				free, scope = -1, "?"
			parts.append(
				f"seq {candidate.uuid[:8]} (idx {candidate.global_idx}): "
				f"no free sequence slot on {scope} (free={free})"
			)
			continue
		needed = PrefillScheduler.required_host_pages(candidate, req)
		text = (
			f"seq {candidate.uuid[:8]} (idx {candidate.global_idx}): needs "
			f"{needed} Host pages on node {candidate.node_id}, free="
			f"{req.per_node_host_free[candidate.node_id]}"
		)
		if total_pages_by_node is not None:
			total = int(total_pages_by_node[candidate.node_id])
			text += f", total={total}"
			if needed > total:
				text += " (can never fit)"
		parts.append(text)
	more = len(blocked) - len(parts)
	if more > 0:
		parts.append(f"... {more} more")
	return (
		f"{len(blocked)} of {len(req.candidates)} candidates unadmitted: "
		+ "; ".join(parts)
	)


class PrefillProgressGuard:
	"""Fail by name instead of spinning on identical empty prefill waves.

	``observe`` runs once per scheduler iteration, after the wave's Host
	reservation. An iteration is DEAD when it admitted nothing and no sequence
	is live (so the decode phase will not run either). A dead iteration whose
	admission ``state`` equals the previous one extends the streak; a change in
	ANY component resets it. ``limit`` consecutive identical dead iterations
	raise ``PrefillAdmissionNoProgress``.

	``state`` must distinguish "waiting on something in flight" from "truly
	stuck", or a recoverable wait repeats an identical snapshot until the limit
	and the guard fails a healthy worker. The worker therefore builds it from
	(see ``_prefill_admission_state``):

	  - gathered per-node Host free AND total pages,
	  - the gathered per-rank / per-node sequence-slot vectors,
	  - a digest of the candidate IDENTITIES and their required pages, not just
	    the candidate count,
	  - the prefix-cache pressure counters that move while a load or an
	    attachment release is in flight (resident nodes, active attachments,
	    pending loads),
	  - the admission reclaim's own outcome (whether it ran, pages released),
	  - the last Host reservation outcome: not attempted / committed /
	    rejected by the fresh capacity snapshot.

	Every component is rank-uniform -- gathered, reduced, or derived from a
	cross-rank-verified candidate digest -- and ``live`` is the OR across
	ranks, so every rank raises on the same iteration instead of diverging on
	the failure path.
	"""

	def __init__(self, limit: int = PREFILL_NO_PROGRESS_LIMIT) -> None:
		if int(limit) <= 0:
			raise ValueError("PrefillProgressGuard limit must be > 0")
		self._limit = int(limit)
		self._state: object = None
		self._streak = 0

	@property
	def streak(self) -> int:
		return self._streak

	def observe(
	    self,
	    *,
	    admitted: int,
	    live: bool,
	    state: object,
	    describe=None,
	) -> None:
		if int(admitted) > 0 or live or state is None:
			self._state = None
			self._streak = 0
			return
		if self._streak and state == self._state:
			self._streak += 1
		else:
			self._state = state
			self._streak = 1
		if self._streak >= self._limit:
			detail = describe() if describe is not None else ""
			raise PrefillAdmissionNoProgress(
				f"prefill admission made no progress for {self._streak} "
				"consecutive scheduler iterations: no sequence admitted, none "
				"live, and the Host KV / sequence-slot state unchanged"
				+ (f"; {detail}" if detail else "")
			)
