"""Agent-session KV hold: a finished turn keeps its Host KV for the next turn.

A /v1/turns request belongs to an agent session. With session hold enabled
(``--enable-session-kv-hold``), a turn that finishes while its session is open
does not free its KV:

* hold    -- at completion the owner rank moves the Host pages that cover the
             turn's KV out of the sequence's own chain into resident pages
             (``retain_sequence_pages``, the primitive the prefix cache uses),
             the rest of the sequence is released as usual (GPU pages eagerly,
             unused Host headroom), and every rank keeps the turn's trajectory
             slot, so the held tokens stay readable node-wide;
* resume  -- the session's next turn arrives as a full prompt. Its longest
             common token prefix (LCP) with the held tokens decides how much
             KV is reused: the held pages up to the LCP become the new turn's
             attached prefix pages, the pages past it are freed, and only the
             suffix is prefilled (the GPT-OSS prefix-hit path);
* release -- close_session / abort frees the held pages and the slot;
* expire  -- a hold idle longer than ``--session-hold-idle-ttl-s`` (no turn
             of its session queued or running) is released by rank 0's
             decision, broadcast like a close (abandoned sessions);
* LRU drop -- ``lru_hold_victims`` picks the least recently held sessions to
             free under pressure; the worker exposes it as a hook for the
             admission / tool-gap policies and calls it on no policy itself.

Held KV is never a sequence in ``global_batch``, so no victim selector (ON_HOLD,
EVICTED, Host growth eviction) can see it, and the prefix cache does not index
resident pages it did not insert.

This module is the replicated bookkeeping and the pure decisions. Every rank
holds the same ``SessionHoldBook`` (it is updated only from broadcast messages
and from completion and prefill-wave decisions that are identical on every
rank); only the owner rank also knows the Host page ids. No torch device work
and no worker imports here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

# Why held KV was released or not reused. Every drop is counted by reason.
DROP_LCP_BELOW_PAGE = "lcp_below_page"
DROP_SUPERSEDED = "superseded"
DROP_EVICTED_AFTER_RESUME = "evicted_after_resume"
DROP_CANCELLED = "cancelled"
# Freed by the LRU drop hook under memory pressure (caller-named reason).
DROP_PRESSURE_LRU = "pressure_lru"

# Release reason of a hold idle longer than the TTL.
RELEASE_IDLE_TTL = "idle_ttl"

# Default --session-hold-idle-ttl-s. The 2026-10-26 eval-workload census
# measured agent tool gaps (a turn's end to the session's next turn): over
# all tools p50 0.9-4.2 s, p90 20-30 s, p99 106-423 s; the slowest tool,
# WebFetch, has p90 308-337 s and p99 ~600 s, which is its timeout cap -- the
# longest wait a live session showed. 900 s is 1.5x that cap, so a live
# session is not expired in practice, while an abandoned one (a client that
# never closes) frees its KV within 15 minutes.
DEFAULT_IDLE_TTL_S = 900.0

# Upper bounds (tokens) of the LCP histogram buckets; the last bucket is open.
LCP_BUCKET_BOUNDS = (64, 1024, 4096, 16384, 32768, 65536, 131072)


class SessionHoldUnsupported(RuntimeError):
    """Session hold was requested on a configuration that cannot serve it."""


class SessionHoldInvariantError(RuntimeError):
    """Held KV bookkeeping disagrees with the Host page table."""


def ceil_div(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor))


@dataclass
class HeldSession:
    """One session's KV held between two turns (replicated on every rank)."""

    session_id: str
    traj_id: Optional[str]
    request_id: str
    # Rank that owns the Host KV (and is pinned for the session's next turn).
    rank: int
    # Node-shared trajectory slot of the finished turn: prompt + decoded tokens.
    trajectory_slot: int
    # Tokens whose KV is on Host: prompt + decoded - 1 (the last sampled token
    # was never fed back, so it has no KV).
    kv_tokens: int
    page_tokens: int
    # Owner rank only: the resident Host pages, in token order.
    page_ids: Optional[Tuple[int, ...]] = None
    # Order of the hold among all holds (SessionHoldBook.add); rank-uniform,
    # so it orders LRU victims identically on every rank.
    hold_seq: int = 0
    # time.monotonic() when the hold was recorded. Rank-local: only rank 0
    # reads it, to decide an idle-TTL release it then broadcasts.
    held_at: float = 0.0

    @property
    def pages(self) -> int:
        return ceil_div(self.kv_tokens, self.page_tokens)


@dataclass(frozen=True)
class ResumePlan:
    """How a session's next turn reuses its held KV."""

    session_id: str
    lcp_tokens: int
    # KV positions the turn takes from the hold (its cached_tokens); 0 = none.
    cached_tokens: int
    # Held pages the turn attaches (the last one may be partial) / frees.
    keep_pages: int
    release_pages: int
    drop_reason: Optional[str]

    @property
    def resumes(self) -> bool:
        return self.drop_reason is None


def longest_common_prefix(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of leading positions where two 1-D token tensors agree."""
    n = min(int(a.numel()), int(b.numel()))
    if n == 0:
        return 0
    diff = torch.ne(a[:n].to(torch.int64), b[:n].to(torch.int64)).nonzero()
    return n if diff.numel() == 0 else int(diff[0].item())


def plan_resume(
    *,
    session_id: str,
    lcp_tokens: int,
    kv_tokens: int,
    prompt_tokens: int,
    page_tokens: int,
) -> ResumePlan:
    """Decide how much held KV a new turn reuses.

    The reused span is the LCP, capped by the KV actually held and by
    ``prompt_tokens - 1``: prefill must compute at least the last prompt token
    to produce the first output token. Below one page the hold is dropped and
    the turn is prefilled in full. The page holding position ``cached - 1``
    is kept even when partial: its tokens past ``cached`` are overwritten by
    the suffix prefill before anything reads them, since the hold is consumed
    by this turn and nothing else references those pages.
    """
    if page_tokens <= 0:
        raise ValueError(f"page_tokens must be positive, got {page_tokens}")
    if prompt_tokens <= 0:
        raise ValueError(f"prompt_tokens must be positive, got {prompt_tokens}")
    held_pages = ceil_div(kv_tokens, page_tokens)
    usable = min(int(lcp_tokens), int(kv_tokens), int(prompt_tokens) - 1)
    if usable < page_tokens:
        return ResumePlan(
            session_id=session_id,
            lcp_tokens=int(lcp_tokens),
            cached_tokens=0,
            keep_pages=0,
            release_pages=held_pages,
            drop_reason=DROP_LCP_BELOW_PAGE,
        )
    keep = ceil_div(usable, page_tokens)
    return ResumePlan(
        session_id=session_id,
        lcp_tokens=int(lcp_tokens),
        cached_tokens=usable,
        keep_pages=keep,
        release_pages=held_pages - keep,
        drop_reason=None,
    )


def split_turn_pages_for_hold(
    *,
    page_table: Sequence[int],
    attached_pages: Sequence[int],
    kv_tokens: int,
    page_tokens: int,
) -> Tuple[List[int], List[int]]:
    """(pages to hold in token order, own pages to retain) for a finished turn.

    ``page_table`` is the sequence's Host page table: its attached pages (held
    pages a resumed turn borrowed, already resident) followed by its own
    chain. The hold covers ``kv_tokens``; own pages past that are headroom the
    normal release frees.
    """
    needed = ceil_div(kv_tokens, page_tokens)
    table = [int(page) for page in page_table]
    attached = [int(page) for page in attached_pages]
    if table[: len(attached)] != attached:
        raise SessionHoldInvariantError(
            "the Host page table does not start with the turn's attached "
            f"session pages (attached={len(attached)}, table={len(table)})"
        )
    if needed < len(attached):
        raise SessionHoldInvariantError(
            f"held KV of {kv_tokens} tokens needs {needed} pages, fewer than "
            f"the {len(attached)} pages the turn attached"
        )
    if len(table) < needed:
        raise SessionHoldInvariantError(
            f"Host page table has {len(table)} pages but the turn's KV covers "
            f"{kv_tokens} tokens ({needed} pages): the Host copy is incomplete"
        )
    held = table[:needed]
    return held, held[len(attached):]


@dataclass
class SessionHoldBook:
    """Held sessions and hold metrics; identical on every rank."""

    page_tokens: int
    sessions: Dict[str, HeldSession] = field(default_factory=dict)
    holds: int = 0
    resumes: int = 0
    reload_tokens: int = 0
    reload_seconds: float = 0.0
    reload_samples: int = 0
    # Holds that ended without reuse, by reason (DROP_*).
    drops: Dict[str, int] = field(default_factory=dict)
    # Holds freed by close_session / abort, by reason.
    releases: Dict[str, int] = field(default_factory=dict)
    lcp_histogram: List[int] = field(
        default_factory=lambda: [0] * (len(LCP_BUCKET_BOUNDS) + 1)
    )
    admission_waits: int = 0
    evictions_while_held: int = 0
    # --session-hold-idle-ttl-s; 0 = holds never expire.
    idle_ttl_s: float = DEFAULT_IDLE_TTL_S

    def get(self, session_id: Optional[str]) -> Optional[HeldSession]:
        if session_id is None:
            return None
        return self.sessions.get(session_id)

    def rank_of(self, session_id: Optional[str]) -> Optional[int]:
        record = self.get(session_id)
        return None if record is None else record.rank

    def add(self, record: HeldSession) -> Optional[HeldSession]:
        """Hold ``record``; return the record it supersedes, if any."""
        previous = self.sessions.get(record.session_id)
        self.sessions[record.session_id] = record
        self.holds += 1
        record.hold_seq = self.holds
        return previous

    def pop(self, session_id: str) -> Optional[HeldSession]:
        return self.sessions.pop(session_id, None)

    def note_drop(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    def note_release(self, reason: str) -> None:
        self.releases[reason] = self.releases.get(reason, 0) + 1

    def note_resume(self, plan: ResumePlan) -> None:
        self.lcp_histogram[_lcp_bucket(plan.lcp_tokens)] += 1
        if plan.resumes:
            self.resumes += 1
            self.reload_tokens += plan.cached_tokens
        else:
            self.note_drop(plan.drop_reason)

    def note_reload_seconds(self, seconds: float) -> None:
        self.reload_seconds += float(seconds)
        self.reload_samples += 1

    @property
    def held_kv_tokens(self) -> int:
        return sum(record.kv_tokens for record in self.sessions.values())

    @property
    def held_pages(self) -> int:
        return sum(record.pages for record in self.sessions.values())

    def snapshot(self) -> Dict[str, object]:
        return {
            "held_sessions": len(self.sessions),
            "held_kv_tokens": self.held_kv_tokens,
            "held_kv_pages": self.held_pages,
            "page_tokens": self.page_tokens,
            "holds": self.holds,
            "resumes": self.resumes,
            "reload_tokens": self.reload_tokens,
            "reload_ms": round(self.reload_seconds * 1000.0, 3),
            "reload_samples": self.reload_samples,
            "lcp_histogram": {
                label: count
                for label, count in zip(_LCP_BUCKET_LABELS, self.lcp_histogram)
            },
            "drops": dict(sorted(self.drops.items())),
            "releases": dict(sorted(self.releases.items())),
            "admission_waits_on_held_kv": self.admission_waits,
            "evictions_while_held": self.evictions_while_held,
            "idle_ttl_s": self.idle_ttl_s,
        }


def expired_holds(
    sessions: Mapping[str, HeldSession],
    *,
    now: float,
    idle_ttl_s: float,
    busy_sessions,
) -> List[Tuple[str, float]]:
    """(session, idle seconds) of holds idle longer than the TTL, oldest first.

    A session with a turn queued or running in the worker is not idle (that
    turn resumes the hold). ``idle_ttl_s <= 0`` disables expiry.
    """
    if idle_ttl_s <= 0:
        return []
    expired = [
        (session_id, now - record.held_at)
        for session_id, record in sessions.items()
        if session_id not in busy_sessions
        and now - record.held_at >= idle_ttl_s
    ]
    expired.sort(key=lambda item: (-item[1], item[0]))
    return expired


def lru_hold_victims(
    sessions: Mapping[str, HeldSession],
    *,
    pages_needed: int,
    exclude=(),
) -> List[str]:
    """Least recently held sessions whose pages together cover ``pages_needed``.

    Ordered by ``hold_seq`` (rank-uniform), so every rank picks the same
    victims. Returns every candidate when they cannot cover the need, and
    nothing for a need <= 0.
    """
    if pages_needed <= 0:
        return []
    candidates = sorted(
        (record for record in sessions.values()
         if record.session_id not in exclude),
        key=lambda record: record.hold_seq,
    )
    victims: List[str] = []
    freed = 0
    for record in candidates:
        if freed >= pages_needed:
            break
        victims.append(record.session_id)
        freed += record.pages
    return victims


def admission_blocked_by_held_kv(selection, held_pages: int) -> bool:
    """Whether held KV is what keeps a prefill wave from admitting anything.

    True when the wave selects nothing and some candidate that has a sequence
    slot needs more Host pages than are free but would fit if the held pages
    were free too. Anything else -- a slot limit, a candidate too large even
    with the held pages, a wave that did select -- is not a wait on held KV,
    and the no-progress guard must see it. ``selection`` is the wave's
    ``PrefillSelectionRequest`` (None when there were no candidates).
    """
    from batchgen.worker.prefill import PrefillScheduler

    if selection is None or int(held_pages) <= 0:
        return False
    if PrefillScheduler.select_prefill_batch(selection):
        return False
    slot_admissible = PrefillScheduler.slot_admissible_candidates(selection)
    for candidate in slot_admissible:
        needed = PrefillScheduler.required_host_pages(candidate, selection)
        free = int(selection.per_node_host_free[candidate.node_id])
        if free < needed <= free + int(held_pages):
            return True
    return False


def _lcp_bucket(lcp_tokens: int) -> int:
    for index, bound in enumerate(LCP_BUCKET_BOUNDS):
        if lcp_tokens < bound:
            return index
    return len(LCP_BUCKET_BOUNDS)


_LCP_BUCKET_LABELS = tuple(
    [f"<{bound}" for bound in LCP_BUCKET_BOUNDS]
    + [f">={LCP_BUCKET_BOUNDS[-1]}"]
)


def attribute_reload_seconds(
    seconds: float, cached_by_sequence: Mapping[int, int]
) -> Dict[int, float]:
    """Split one materialization's H2D wall time over its session hits by tokens."""
    total = sum(int(tokens) for tokens in cached_by_sequence.values())
    if total <= 0:
        return {}
    return {
        int(sequence_id): float(seconds) * int(tokens) / total
        for sequence_id, tokens in cached_by_sequence.items()
    }


def log_metrics(book: SessionHoldBook, rank: int, event: str) -> None:
    snap = book.snapshot()
    logging.info(
        "[SESSION_HOLD] metrics rank=%s event=%s held_sessions=%s "
        "held_kv_tokens=%s held_kv_pages=%s holds=%s resumes=%s "
        "reload_tokens=%s reload_ms=%s drops=%s releases=%s "
        "lcp_histogram=%s admission_waits_on_held_kv=%s "
        "evictions_while_held=%s",
        rank, event, snap["held_sessions"], snap["held_kv_tokens"],
        snap["held_kv_pages"], snap["holds"], snap["resumes"],
        snap["reload_tokens"], snap["reload_ms"], snap["drops"],
        snap["releases"], snap["lcp_histogram"],
        snap["admission_waits_on_held_kv"], snap["evictions_while_held"],
    )
