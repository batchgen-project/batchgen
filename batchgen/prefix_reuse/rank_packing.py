"""Pack one prefill wave onto DP ranks by predicted executor time.

``wave_plan.assign_ranks_for_sharing`` keeps shared prefixes on one rank, but
it only ever accepts a split that lowers the busiest rank, so a wave made of a
few large prefix families stops with ranks idle. This packer runs a short fixed
schedule of refinement rounds over the same split rule and keeps the round
whose busiest rank is cheapest under an offline executor cost model. Per-rank
prompt counts are therefore free to be uneven.

Round 0 is the unsplit wave, so splitting is only ever chosen when it beats
keeping every shared prefix together. Rounds 1..9 split every unit one further
level; the last four rounds instead split only the units above a falling score
target. Uniform levels are what find a balanced partition of units that all
score alike -- three equal prefix families have no target that keeps "every
family halved" -- while the targets refine one heavy unit inside an otherwise
light wave. Every distinct partition is placed and the cheapest span wins.

The wave's prefix tree is built once; every later step works on segment ids and
per-prompt segment paths, so no prompt token is hashed again.

A rank's features are those of the union of the members it holds, not the sum
of its units' features: two units on one rank do share every segment both pass
through. Its predicted tokens, pairs and rows are therefore exact -- they are
what ``prefill_plan_work`` reports for that rank group with an unlimited pool
-- so a rank's score depends only on which prompts it holds and never on how
they were partitioned into units. Only its chunk count stays approximate: a
lower bound from its tokens (see ``_score``) against an upper bound from its
segment depth (see ``_RankState``).

Complexity: O(total_blocks) to build the tree, then per round
O(n*D*W + U log U) for n prompts of segment depth D, U units and W ranks, over
at most ``_MAX_LEVELS + 4`` rounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from batchgen.prefix_reuse.executor_cost import ExecutorCostProfile
from batchgen.prefix_reuse.wave_plan import _Tree, _build_tree


@dataclass(frozen=True)
class UnitFeatures:
    """What one unit costs when it alone owns a rank's prefix pool.

    Mirrors ``prefill_plan_work`` of ``plan_wave_prefix_sharing`` over the
    unit's prompts with an unlimited pool: a segment at least two members pass
    through is computed once, every member computes the rest of its prompt.
    ``depth`` is the longest pooled chain, which floors the chunk count because
    a chunk may only read segments written by an earlier chunk.

    ``_RankState`` reports a whole rank's members in the same shape.
    """

    computed_tokens: int
    attention_pairs: int
    item_rows: int
    depth: int


@dataclass(frozen=True)
class RankPacking:
    ranks: tuple[int, ...]  # per prompt
    predicted: tuple[float, ...]  # per rank score, pinned load included
    round_used: int


_MAX_LEVELS = 10  # enough to halve a wave of a thousand identical prompts


def _tri(n: int) -> int:
    return n * (n + 1) // 2


def unit_features(
    tree: _Tree,
    members: Sequence[int],
    prompt_lengths: Sequence[int],
    block_tokens: int,
) -> UnitFeatures:
    """Exact plan features of ``members``, read off the whole wave's tree.

    A wave segment carries the same member set through all of its blocks, so a
    unit's segments are runs of wave segments of equal member count. Merging a
    run changes neither computed tokens nor attention pairs, but it does merge
    the run into one planned item, so rows count runs and not wave segments.
    """

    count: dict[int, int] = {}
    for index in members:
        for seg in tree.paths[index]:
            count[seg] = count.get(seg, 0) + 1
    computed = 0
    pairs = 0
    rows = 0
    for seg, seen in count.items():
        if seen < 2:
            continue
        start = tree.start_block[seg] * block_tokens
        end = tree.end_block[seg] * block_tokens
        computed += end - start
        pairs += _tri(end) - _tri(start)
        parent = tree.parent[seg]
        if parent is None or count.get(parent) != seen:
            rows += 1  # head of a run: its own planned item
    depth = 0
    for index in members:
        path = tree.paths[index]
        pooled = 0
        runs = 0
        previous = 0
        for seg in path:
            if count[seg] < 2:
                break  # counts fall monotonically, so the rest is private too
            if count[seg] != previous:
                runs += 1
            previous = count[seg]
            pooled += 1
        start = tree.end_block[path[pooled - 1]] * block_tokens if pooled else 0
        computed += prompt_lengths[index] - start
        pairs += _tri(prompt_lengths[index]) - _tri(start)
        depth = max(depth, runs)
    return UnitFeatures(computed, pairs, rows + len(members), depth)


def _score(
    profile: Optional[ExecutorCostProfile],
    computed_tokens: int,
    attention_pairs: int,
    item_rows: int,
    depth: int,
    chunk_tokens: int,
) -> float:
    """Predicted seconds, or computed tokens without a profile; 0 when empty.

    ``chunks`` is the floor the plan cannot beat: a chunk holds at most
    ``chunk_tokens`` tokens, and a pooled chain of ``depth`` segments needs
    ``depth`` chunks before the tails below it. The list scheduler can need
    more, so chunks is the one feature this model may under-count.
    """

    if computed_tokens <= 0:
        return 0.0
    if profile is None:
        return float(computed_tokens)
    chunks = max(-(-computed_tokens // chunk_tokens), depth + 1)
    return max(
        profile.seconds(computed_tokens, attention_pairs, item_rows, chunks), 0.0
    )


def _feature_score(
    profile: Optional[ExecutorCostProfile],
    features: UnitFeatures,
    chunk_tokens: int,
) -> float:
    return _score(
        profile,
        features.computed_tokens,
        features.attention_pairs,
        features.item_rows,
        features.depth,
        chunk_tokens,
    )


def prompt_score(
    tokens: Sequence[int],
    *,
    block_tokens: int,
    chunk_tokens: int,
    profile: Optional[ExecutorCostProfile],
) -> float:
    """Score one prompt as a unit of its own, exactly as the packer would."""

    return prompt_score_from_length(
        len(tokens),
        block_tokens=block_tokens,
        chunk_tokens=chunk_tokens,
        profile=profile,
    )


def prompt_score_from_length(
    length: int,
    *,
    block_tokens: int,
    chunk_tokens: int,
    profile: Optional[ExecutorCostProfile],
) -> float:
    """Score one prompt by its length alone, exactly as the packer would.

    This is what ``pack_wave_ranks``'s ``fixed`` expects for load already
    committed to a rank. A lone prompt shares nothing, so the score depends
    only on its length; ranks therefore agree without reading its token
    storage — which may not even exist locally for a sequence whose Host KV
    migrated to another node.
    """

    if block_tokens <= 0 or chunk_tokens <= 0:
        raise ValueError("block_tokens and chunk_tokens must be positive")
    if length <= 0:
        raise ValueError("every prompt must be non-empty")
    # unit_features of a one-prompt unit: nothing pooled, one item row.
    return _score(profile, length, _tri(length), 1, 0, chunk_tokens)


@dataclass(frozen=True)
class _SegInfo:
    """Static per-segment costs of the wave tree, shared by every rank state."""

    parent: tuple[Optional[int], ...]
    tokens: tuple[int, ...]
    pairs: tuple[int, ...]
    depth: tuple[int, ...]  # tree segments from the root down to it, inclusive


def _seg_info(tree: _Tree, block_tokens: int) -> _SegInfo:
    tokens: list[int] = []
    pairs: list[int] = []
    depth: list[int] = []
    for seg, up in enumerate(tree.parent):
        start = tree.start_block[seg] * block_tokens
        end = tree.end_block[seg] * block_tokens
        tokens.append(end - start)
        pairs.append(_tri(end) - _tri(start))
        # A segment is numbered after its parent, so one ascending pass does.
        depth.append(1 if up is None else depth[up] + 1)
    return _SegInfo(tree.parent, tuple(tokens), tuple(pairs), tuple(depth))


@dataclass(frozen=True)
class _UnitSegs:
    """A unit's segments, plus the work no other unit on its rank can share.

    ``segs`` is the union of its members' segment paths, ancestors first;
    ``counts`` and ``ends`` say how many members pass through resp. end at
    each. ``tail_tokens`` and ``tail_pairs`` are every member's work past its
    last tree segment, which is private to that member and so simply adds up.
    """

    segs: tuple[int, ...]
    counts: tuple[int, ...]
    ends: tuple[int, ...]
    members: int
    tail_tokens: int
    tail_pairs: int


def _unit_segs(
    tree: _Tree,
    members: Sequence[int],
    prompt_lengths: Sequence[int],
    block_tokens: int,
) -> _UnitSegs:
    counts: dict[int, int] = {}
    ends: dict[int, int] = {}
    tail_tokens = 0
    tail_pairs = 0
    for index in members:
        path = tree.paths[index]
        for seg in path:
            counts[seg] = counts.get(seg, 0) + 1
        if path:
            ends[path[-1]] = ends.get(path[-1], 0) + 1
        start = tree.end_block[path[-1]] * block_tokens if path else 0
        tail_tokens += prompt_lengths[index] - start
        tail_pairs += _tri(prompt_lengths[index]) - _tri(start)
    segs = tuple(counts)  # a path runs root to leaf, so ancestors come first
    return _UnitSegs(
        segs,
        tuple(counts[seg] for seg in segs),
        tuple(ends.get(seg, 0) for seg in segs),
        len(members),
        tail_tokens,
        tail_pairs,
    )


class _RankState:
    """One rank's plan features, updated as whole units are placed on it.

    A rank group pools every tree segment at least two of its members pass
    through, and each member computes its prompt from the end of its deepest
    pooled segment on. Splitting that per segment rather than per member gives
    the identity maintained here, for a member set ``S`` whose segment union
    ``U`` is every segment on a path of ``S``::

        computed = sum(tokens(s) for s in U)
                   + sum(len(m) - end(last segment of m) for m in S)
        pairs    = sum(tri(end(s)) - tri(start(s)) for s in U)
                   + sum(tri(len(m)) - tri(end(last segment of m)) for m in S)

    A segment one member passes through is private and one two or more pass
    through is pooled, but ``U`` counts it once either way: a member's private
    segments are exactly the contiguous run between its deepest pooled segment
    and its tail, so both sums telescope back to ``unit_features(S)``, i.e. to
    ``prefill_plan_work`` of that rank group with an unlimited pool.

    ``item_rows`` is exact too. Consecutive segments carrying the same members
    are planned as one item, and a segment keeps all of its members in its only
    child exactly when no member of it branches away or ends there, so::

        item_rows = pooled segments - merged segments + members

    ``depth`` is the one bound left: the deepest pooled segment's tree depth,
    which over-counts by every pooled run this rank happens to merge. It only
    raises the chunk floor in ``_score``.
    """

    def __init__(self, info: _SegInfo) -> None:
        self._info = info
        self._count: dict[int, int] = {}  # this rank's members through a seg
        self._kids: dict[int, int] = {}  # its children this rank holds
        self._ends: dict[int, int] = {}  # members whose path ends at it
        self._computed = 0
        self._pairs = 0
        self._members = 0
        self._pooled = 0  # segments with at least two members
        self._merged = 0  # segments whose only child keeps all their members
        self._depth = 0

    def _after(self, unit: _UnitSegs) -> tuple[int, int, int, int, int]:
        """This rank's totals once ``unit`` joins it; mutates nothing."""

        info = self._info
        computed = self._computed + unit.tail_tokens
        pairs = self._pairs + unit.tail_pairs
        depth = self._depth
        fresh: dict[int, int] = {}  # per segment, children new to this rank
        for seg in unit.segs:
            if seg not in self._count:
                computed += info.tokens[seg]
                pairs += info.pairs[seg]
                up = info.parent[seg]
                if up is not None:  # a path keeps ancestors, so up is in segs
                    fresh[up] = fresh.get(up, 0) + 1
        pooled = self._pooled
        merged = self._merged
        for position, seg in enumerate(unit.segs):
            was = self._count.get(seg, 0)
            now = was + unit.counts[position]
            if was < 2 <= now:
                pooled += 1
                depth = max(depth, info.depth[seg])
            kids = self._kids.get(seg, 0)
            ends = self._ends.get(seg, 0)
            # Only these three counts can change, and only for a segment of
            # this unit, so the running total telescopes to the final one.
            merged += int(
                now >= 2
                and kids + fresh.get(seg, 0) == 1
                and ends + unit.ends[position] == 0
            ) - int(was >= 2 and kids == 1 and ends == 0)
        return computed, pairs, pooled, merged, depth

    def trial(self, unit: _UnitSegs) -> UnitFeatures:
        """The features this rank would have with ``unit`` placed on it."""

        computed, pairs, pooled, merged, depth = self._after(unit)
        rows = pooled - merged + self._members + unit.members
        return UnitFeatures(computed, pairs, rows, depth)

    def add(self, unit: _UnitSegs) -> None:
        (
            self._computed,
            self._pairs,
            self._pooled,
            self._merged,
            self._depth,
        ) = self._after(unit)
        self._members += unit.members
        for position, seg in enumerate(unit.segs):
            if seg not in self._count:
                up = self._info.parent[seg]
                if up is not None:
                    self._kids[up] = self._kids.get(up, 0) + 1
            self._count[seg] = self._count.get(seg, 0) + unit.counts[position]
            if unit.ends[position]:
                self._ends[seg] = self._ends.get(seg, 0) + unit.ends[position]

    def features(self) -> UnitFeatures:
        rows = self._pooled - self._merged + self._members
        return UnitFeatures(self._computed, self._pairs, rows, self._depth)


_Unit = tuple[list[int], _UnitSegs, float]


def _refine(
    root: list[int],
    level: Optional[int],
    target: float,
    unit: Callable[[list[int]], _Unit],
    split: Callable[[list[int], int], Optional[list[tuple[list[int], int]]]],
) -> list[_Unit]:
    """One round's units: split to a uniform ``level``, or down to ``target``.

    Either way a unit stops splitting once it is a single prompt.
    """

    units: list[_Unit] = []
    stack = [(root, 0, 0)]
    while stack:
        members, depth, reached = stack.pop()
        made = unit(members)
        fits = reached >= level if level is not None else made[2] <= target
        if not fits:
            parts = split(members, depth)
            if parts is not None:
                stack.extend(
                    (part, at, reached + 1) for part, at in reversed(parts)
                )
                continue
        units.append(made)
    return units


def _place(
    units: list[_Unit],
    world_size: int,
    base: list[float],
    profile: Optional[ExecutorCostProfile],
    chunk_tokens: int,
    info: _SegInfo,
) -> tuple[float, list[int], tuple[float, ...]]:
    """Place units heaviest first on the rank minimising the resulting score.

    Each rank scores the union of the members it holds, so placing a unit only
    charges it for the segments that rank does not already have.
    """

    order = sorted(range(len(units)), key=lambda u: (-units[u][2], units[u][0][0]))
    state = [_RankState(info) for _ in range(world_size)]
    where = [0] * len(units)
    for u in order:
        segs = units[u][1]
        best: Optional[tuple[float, int]] = None
        for rank in range(world_size):
            trial = base[rank] + _feature_score(
                profile, state[rank].trial(segs), chunk_tokens
            )
            if best is None or (trial, rank) < best:
                best = (trial, rank)
        assert best is not None
        state[best[1]].add(segs)
        where[u] = best[1]
    predicted = tuple(
        base[rank] + _feature_score(profile, state[rank].features(), chunk_tokens)
        for rank in range(world_size)
    )
    return max(predicted), where, predicted


def pack_wave_ranks(
    prompts: Sequence[Sequence[int]],
    *,
    world_size: int,
    block_tokens: int,
    chunk_tokens: int,
    profile: Optional[ExecutorCostProfile],
    fixed: Sequence[tuple[int, float]] = (),
) -> RankPacking:
    """Balance one wave's predicted per-rank prefill executor time.

    ``fixed`` pins load already committed to a rank, such as an evicted
    sequence re-entering on the rank that still holds its pages. Each entry is
    added to that rank's starting score, so a busy rank simply receives fewer
    prompts.

    ``lower`` bounds the busiest rank from below: a rank cannot be cheaper than
    an even share of the unsplit wave, nor than the single costliest prompt.
    It seeds the score targets and stops the schedule once a round reaches it.

    Features assume every shared segment is pooled, as an unlimited prefix pool
    would allow. A rank whose pool is too small for that raises its own
    threshold and recomputes more, which this model does not predict.
    """

    if world_size <= 0 or block_tokens <= 0 or chunk_tokens <= 0:
        raise ValueError("world_size, block_tokens and chunk_tokens must be positive")
    base = [0.0] * world_size
    for rank, load in fixed:
        base[rank] += float(load)
    if len(prompts) == 0:
        return RankPacking((), tuple(base), 0)
    if any(len(tokens) == 0 for tokens in prompts):
        raise ValueError("every prompt must be non-empty")

    lengths = [len(tokens) for tokens in prompts]
    shareable = [(n - 1) // block_tokens for n in lengths]
    tree = _build_tree(prompts, block_tokens)
    info = _seg_info(tree, block_tokens)

    units_seen: dict[tuple[int, ...], _Unit] = {}

    def unit(members: list[int]) -> _Unit:
        # The score is the unit's own, as if it alone owned a rank: it orders
        # the placement and sets the round targets. What a rank pays for the
        # unit depends on the other units on it, so ``_place`` scores that.
        # Rounds re-derive the same split tree, so each unit is built once.
        key = tuple(members)
        made = units_seen.get(key)
        if made is None:
            feats = unit_features(tree, members, lengths, block_tokens)
            made = (
                members,
                _unit_segs(tree, members, lengths, block_tokens),
                _feature_score(profile, feats, chunk_tokens),
            )
            units_seen[key] = made
        return made

    def seg_at(index: int, depth: int) -> int:
        """Segment on ``index``'s path covering block ``depth``."""
        path = tree.paths[index]
        low, high = 0, len(path) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if tree.start_block[path[middle]] <= depth:
                low = middle
            else:
                high = middle - 1
        return path[low]

    def split(
        members: list[int], depth: int
    ) -> Optional[list[tuple[list[int], int]]]:
        """``assign_ranks_for_sharing``'s split rule, expressed on segment ids."""
        if len(members) == 1:
            return None
        while True:
            groups: dict[int, list[int]] = {}
            ended: list[int] = []
            for index in members:
                if shareable[index] > depth:
                    groups.setdefault(seg_at(index, depth), []).append(index)
                else:
                    ended.append(index)
            if not groups:  # identical shareable prefixes: halve the unit
                half = len(members) // 2
                return [(members[:half], depth), (members[half:], depth)]
            if len(groups) == 1 and not ended:
                # One segment holds every member: no member can branch or end
                # inside it, so skip straight past its last block.
                depth = tree.end_block[next(iter(groups))]
                continue
            parts = [list(group) for group in groups.values()]
            # Prompts ending here share [0, depth) with every child; keep them
            # with the largest child instead of recomputing that prefix alone.
            max(parts, key=len).extend(ended)
            if len(parts) == 1:  # only the ended prompts branched off
                rest = set(ended)
                parts = [[i for i in members if i not in rest], ended]
            return [(sorted(part), depth + 1) for part in parts]

    root = list(range(len(prompts)))
    whole = unit(root)
    lower = max(whole[2] / world_size, max(unit([i])[2] for i in root))

    rounds: list[tuple[Optional[int], float]] = [
        (level, math.inf) for level in range(_MAX_LEVELS)
    ]
    rounds += [(None, lower / 2 ** step) for step in range(4)]

    best: Optional[tuple[float, int, tuple[int, ...], tuple[float, ...]]] = None
    seen: set[tuple[tuple[int, ...], ...]] = set()
    levels_exhausted = False
    for index, (level, target) in enumerate(rounds):
        if level is not None and levels_exhausted:
            continue
        units = _refine(root, level, target, unit, split)
        shape = tuple(tuple(members) for members, _, _ in units)
        if shape in seen:
            # A uniform level that splits nothing leaves every deeper level
            # the same partition too.
            levels_exhausted = level is not None
            continue  # this round produced no new split
        seen.add(shape)
        span, where, predicted = _place(
            units, world_size, base, profile, chunk_tokens, info
        )
        if best is None or span < best[0]:
            ranks = [0] * len(prompts)
            for position, (members, _, _) in enumerate(units):
                for i in members:
                    ranks[i] = where[position]
            best = (span, index, tuple(ranks), predicted)
        if span <= lower * (1.0 + 1e-9):
            break  # a rank at the lower bound cannot be improved on

    assert best is not None
    return RankPacking(best[2], best[3], best[1])
