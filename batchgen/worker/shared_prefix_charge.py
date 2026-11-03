"""Shared-prefix-aware capacity accounting — the charge ledger.

Admission charges a page to a resource (a node's Host-KV pool, a rank's GPU
pool) so that the number of sequences it admits is bounded by what the
allocator will actually consume.  When several sequences read ONE physical
copy of a shared prefix, charging every row the whole prefix bounds admission
by the logical page sum instead of the physical one, and prefix sharing stops
converting into concurrency: the pool looks N times fuller than it is.

This module holds the arithmetic for charging a shared prefix ONCE per
resource, and nothing else.  It is pure: no torch, no distributed call, no
worker state.  Every decision is a function of the arguments, so each rank
computing the same inputs reaches the same verdict without communicating --
which is what makes it usable inside a collective admission walk.

Model
-----
A sequence's reservation is a chain of *shared segments* plus a private
*tail*.  A segment is a run of pages that >= 2 sequences read from one
physical copy; it carries a ``key`` that is equal exactly when the physical
pages are the same.  The first reader of a segment on a resource pays for it,
later readers pay nothing, and the pages come back only when the LAST reader
leaves -- so the ledger is a refcount per segment key, not a boolean.

Safety
------
The ledger never releases a segment that still has a reader, and an
unbalanced release is a named failure rather than a silently negative
refcount: a resource that under-counts its own occupancy hands the same page
to two owners.  ``savepoint``/``rollback`` exist because a charge is the first
phase of a two-phase acquire: the second phase (the real allocation, behind a
fresh collective snapshot) can still reject the wave, and without a rollback
every rejection would leak the charge and shrink the pool forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class SharedPrefixRefcountError(RuntimeError):
    """A segment was released more times than it was charged, or over-released.

    Nothing recovers from this: the ledger's refcount is the only record of
    how many sequences still read a physical page, so a corrupted count means
    the next release can hand a live page back to the free pool.  It is raised
    at the point the arithmetic breaks, while the caller is still identifiable,
    instead of surfacing later as reused KV.
    """


class SharedPrefixChargeError(ValueError):
    """A charge request is not self-consistent (bad pages, bad reservation)."""


@dataclass(frozen=True)
class SharedSegment:
    """One run of pages that several sequences read from one physical copy.

    ``key`` identifies the PHYSICAL copy: two sequences carry the same key for
    a segment exactly when charging it twice would double-count one set of
    pages.  Keys only have to be unique within the resource the ledger counts,
    and equal across ranks for the same segment (admission is collective).
    """

    key: int
    pages: int

    def __post_init__(self) -> None:
        if int(self.pages) <= 0:
            raise SharedPrefixChargeError(
                f"shared segment {self.key} must cover >= 1 page, "
                f"got {self.pages}"
            )


@dataclass(frozen=True)
class SharedChargeResult:
    """What one row costs the resource right now.

    ``marginal_pages`` is what admission must subtract from free pages;
    ``credited_pages`` is what sharing saved, kept separate so a caller can
    log or assert on the saving without recomputing it.
    """

    marginal_pages: int
    credited_pages: int
    first_reader_keys: Tuple[int, ...]


class SharedPrefixLedger:
    """Refcounted "who already paid for this prefix" ledger for ONE resource.

    One instance counts one pool: a node's Host-KV pages or one rank's GPU
    pages.  Resources never share a ledger, because a prefix resident on node
    0 saves nothing for a sequence admitted on node 1.

    ``resident`` names segments the resource ALREADY holds when the walk
    starts (a retained prefix-cache entry, a prefix already mapped on the GPU).
    They are seeded with one reference standing for the holder itself, so a
    reader's release can never take the count to zero and free pages the
    holder still owns.
    """

    __slots__ = ("_refcounts", "_pages", "_resident")

    def __init__(self, resident: Optional[Mapping[int, int]] = None) -> None:
        self._refcounts: Dict[int, int] = {}
        self._pages: Dict[int, int] = {}
        self._resident: set[int] = set()
        for key, pages in (resident or {}).items():
            segment = SharedSegment(key=int(key), pages=int(pages))
            self._pages[segment.key] = segment.pages
            # The holder's own reference. A reader that charges and releases
            # this segment returns to 1, never to 0, so the retained copy is
            # not freed by a reader that merely stopped reading it.
            self._refcounts[segment.key] = 1
            self._resident.add(segment.key)

    # ---------------------------------------------------------------- queries

    def is_charged(self, key: int) -> bool:
        """Whether this resource already holds pages for ``key``."""
        return int(key) in self._refcounts

    def refcount(self, key: int) -> int:
        return self._refcounts.get(int(key), 0)

    def charged_pages(self) -> int:
        """Pages this ledger has handed out, counting each segment once."""
        return sum(self._pages[key] for key in self._refcounts)

    def preview(
        self,
        *,
        total_pages: int,
        segments: Sequence[SharedSegment] = (),
        min_pages: int = 1,
    ) -> SharedChargeResult:
        """What ``charge`` would cost, without mutating the ledger.

        Used by the greedy admission test ("does this row still fit?") before
        the row is known to be admitted; a row that does not fit must leave no
        trace, or the next row is charged against a prefix nobody is reading.
        """
        return self._compute(
            total_pages=total_pages, segments=segments, min_pages=min_pages
        )

    # ---------------------------------------------------------------- charging

    def charge(
        self,
        *,
        total_pages: int,
        segments: Sequence[SharedSegment] = (),
        min_pages: int = 1,
    ) -> SharedChargeResult:
        """Record one reader and return the pages the resource actually loses.

        The first reader of a segment pays its pages; every later reader pays
        only its private tail.  ``min_pages`` floors the charge so a row can
        never be free: even a prompt that is entirely shared needs its own last
        page to hold the token whose logits it produces.
        """
        result = self._compute(
            total_pages=total_pages, segments=segments, min_pages=min_pages
        )
        for segment in self._normalize(segments):
            self._pages.setdefault(segment.key, segment.pages)
            self._refcounts[segment.key] = (
                self._refcounts.get(segment.key, 0) + 1
            )
        return result

    def release(self, segments: Sequence[SharedSegment] = ()) -> int:
        """Drop one reader and return the pages that became free.

        Pages come back only on the LAST reader: a segment whose refcount is
        still positive afterwards returns 0 pages, because another sequence is
        still reading that exact physical copy.  A release with no matching
        charge is a named failure -- see ``SharedPrefixRefcountError``.
        """
        freed = 0
        for segment in self._normalize(segments):
            count = self._refcounts.get(segment.key)
            if count is None:
                raise SharedPrefixRefcountError(
                    f"release of shared segment {segment.key} that this "
                    "resource never charged"
                )
            if count <= 0:  # pragma: no cover - guarded by the delete below
                raise SharedPrefixRefcountError(
                    f"shared segment {segment.key} has non-positive "
                    f"refcount {count} before release"
                )
            if count > 1:
                self._refcounts[segment.key] = count - 1
                continue
            if segment.key in self._resident:
                raise SharedPrefixRefcountError(
                    f"release of shared segment {segment.key} would drop the "
                    "holder's own reference on a resident prefix"
                )
            del self._refcounts[segment.key]
            freed += self._pages.pop(segment.key, segment.pages)
        return freed

    # --------------------------------------------------------- two-phase use

    def savepoint(self) -> Tuple[Tuple[int, int], ...]:
        """Snapshot the refcounts so a failed second phase can be undone."""
        return tuple(sorted(self._refcounts.items()))

    def rollback(self, savepoint: Tuple[Tuple[int, int], ...]) -> None:
        """Restore a ``savepoint``, undoing every charge taken since it.

        A rejected reservation must leave the ledger exactly as the wave found
        it.  Restoring the counts wholesale (rather than releasing row by row)
        keeps the undo correct even when the failure happened midway through a
        row's charge, which is precisely when a row-by-row undo would be
        unbalanced.
        """
        restored = {int(key): int(count) for key, count in savepoint}
        for key, count in restored.items():
            if count <= 0:
                raise SharedPrefixRefcountError(
                    f"savepoint holds non-positive refcount {count} for "
                    f"shared segment {key}"
                )
            if key not in self._pages:
                raise SharedPrefixRefcountError(
                    f"savepoint references unknown shared segment {key}"
                )
        for key in list(self._pages):
            if key not in restored:
                self._pages.pop(key, None)
        self._refcounts = restored

    # ----------------------------------------------------------------- detail

    def _compute(
        self,
        *,
        total_pages: int,
        segments: Sequence[SharedSegment],
        min_pages: int,
    ) -> SharedChargeResult:
        total = int(total_pages)
        floor = int(min_pages)
        if total <= 0:
            raise SharedPrefixChargeError(
                f"a reservation must cover >= 1 page, got {total}"
            )
        if floor < 0:
            raise SharedPrefixChargeError(
                f"min_pages must be >= 0, got {floor}"
            )
        normalized = self._normalize(segments)
        credited = 0
        first_reader: List[int] = []
        for segment in normalized:
            if segment.key in self._refcounts:
                credited += segment.pages
            else:
                first_reader.append(segment.key)
        # A credit larger than the reservation would make the row free (or
        # negative) and over-admit the wave. It means the segments and the
        # reservation were measured against different lengths, so cap the
        # credit at the reservation minus its floor rather than trusting it.
        credited = min(credited, max(0, total - floor))
        return SharedChargeResult(
            marginal_pages=total - credited,
            credited_pages=credited,
            first_reader_keys=tuple(first_reader),
        )

    @staticmethod
    def _normalize(
        segments: Sequence[SharedSegment],
    ) -> Tuple[SharedSegment, ...]:
        """Validate the chain and drop repeats of one key within one row.

        A key repeated inside ONE row is the same physical copy read twice by
        the same sequence: crediting it twice would claim a saving that does
        not exist, and refcounting it twice would keep it alive after its only
        reader left.
        """
        seen: Dict[int, int] = {}
        ordered: List[SharedSegment] = []
        for segment in segments:
            if not isinstance(segment, SharedSegment):
                segment = SharedSegment(
                    key=int(segment[0]), pages=int(segment[1])
                )
            previous = seen.get(segment.key)
            if previous is None:
                seen[segment.key] = segment.pages
                ordered.append(segment)
                continue
            if previous != segment.pages:
                raise SharedPrefixChargeError(
                    f"shared segment {segment.key} appears with {previous} "
                    f"and {segment.pages} pages in one reservation"
                )
        return tuple(ordered)


def plan_shared_runs(
    page_tables: Mapping[int, Sequence[int]],
) -> Dict[int, Tuple[int, int]]:
    """Per sequence, the leading page run it shares with another sequence.

    ``page_tables`` maps a sequence id to the PHYSICAL page ids backing its
    reservation, in logical order.  Two sequences share a run exactly when
    their tables agree page by page from the start, so the run is their
    page-granular common prefix -- the same identity the allocator dedups on
    when it maps one physical page per distinct page id.

    Returns ``seq_id -> (key, pages)`` for every sequence that shares at least
    one page, where ``key`` is the page id ending the run: it is equal for
    exactly the sequences that share that run, because a physical page id
    belongs to one copy.  Sequences sharing nothing are absent.

    Grouping by first page then comparing sorted neighbours makes this
    O(n log n) comparisons rather than all pairs: in sorted order a sequence's
    longest common prefix with the whole group is its longest with one of its
    two neighbours.
    """
    by_first: Dict[int, List[int]] = {}
    tables: Dict[int, Tuple[int, ...]] = {}
    for seq_id, pages in page_tables.items():
        row = tuple(int(page) for page in pages)
        if not row:
            continue
        tables[int(seq_id)] = row
        by_first.setdefault(row[0], []).append(int(seq_id))

    runs: Dict[int, Tuple[int, int]] = {}
    for group in by_first.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda seq_id: tables[seq_id])

        def common(left: int, right: int) -> int:
            if left < 0 or right >= len(ordered):
                return 0
            first, second = tables[ordered[left]], tables[ordered[right]]
            limit = min(len(first), len(second))
            index = 0
            while index < limit and first[index] == second[index]:
                index += 1
            return index

        for position, seq_id in enumerate(ordered):
            run = max(common(position - 1, position), common(position, position + 1))
            if run <= 0:
                continue
            runs[seq_id] = (tables[seq_id][run - 1], run)
    return runs


def admit_by_marginal_pages(
    rows: Sequence[Tuple[object, int, int, Optional[Tuple[int, int]], int]],
    free_pages: int,
    *,
    resident: Optional[Mapping[int, int]] = None,
) -> Tuple[List[object], int]:
    """Greedily admit rows against one resource, charging a prefix once.

    Each row is ``(row_id, resource_index, total_pages, run, resident_pages)``
    where ``run`` is its ``(key, pages)`` shared run (or ``None``) and
    ``resident_pages`` is how many of its leading pages the resource already
    holds.  ``resource_index`` is ignored here -- callers pass one resource per
    call -- and kept so a row's tuple is the same shape on every site.

    ``resident_pages`` and the in-round credit both describe a PREFIX of the
    same page list, so the larger subsumes the smaller and the credit is their
    maximum rather than their sum.  Returns the admitted row ids and the pages
    charged, so a caller can assert the charge against what it then allocates.
    """
    ledger = SharedPrefixLedger(resident=resident)
    used = 0
    admitted: List[object] = []
    for row_id, _resource, total_pages, run, resident_pages in rows:
        total = int(total_pages)
        if total <= 0:
            raise SharedPrefixChargeError(
                f"row {row_id} must reserve >= 1 page, got {total}"
            )
        credit = min(int(resident_pages), max(0, total - 1))
        segments: Tuple[SharedSegment, ...] = ()
        if run is not None:
            segments = (SharedSegment(key=int(run[0]), pages=int(run[1])),)
            preview = ledger.preview(total_pages=total, segments=segments)
            credit = max(credit, preview.credited_pages)
        charge = total - credit
        if used + charge > int(free_pages):
            continue
        used += charge
        admitted.append(row_id)
        if segments:
            ledger.charge(total_pages=total, segments=segments)
    return admitted, used


def segments_from_pairs(
    pairs: Iterable[Sequence[int]],
) -> Tuple[SharedSegment, ...]:
    """Build a segment chain from ``(key, pages)`` pairs.

    The wire form of a chain is pairs of ints, because the chain crosses an
    ``all_gather_object`` to make admission rank-identical and a dataclass
    there would pickle the class as well as the numbers.
    """
    return tuple(
        SharedSegment(key=int(key), pages=int(pages)) for key, pages in pairs
    )
