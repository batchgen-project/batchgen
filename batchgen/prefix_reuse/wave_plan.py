"""Plan in-wave prefix sharing for one prefill phase.

BatchGen receives a whole wave before prefill starts, so which prompts share
which prefix is known up front. The planner builds the prefix tree of the wave
from token ids, computes every shared segment exactly once into the prefill
prefix pool, and orders the work so that an item only runs after all of its
ancestor segments are resident in the pool.

Pure: no GPU, no Host KV, no coordinator. The executor consumes the plan.

Definitions
-----------
* A prompt is cut into ``block_tokens`` blocks. Only the first
  ``(len - 1) // block_tokens`` blocks are shareable, so the block holding the
  last prompt token is always private and produces the first output token.
* A segment is a maximal run of tree blocks that the same set of prompts
  passes through (no branch, no prompt ending inside it). ``count`` is the
  size of that set; counts never increase from a segment to its children.
* A segment with ``count >= threshold`` (and ``count >= 2``) is pooled: it is
  computed once, its KV stays in the pool until its whole subtree is done.
  The pooled set is closed under ancestors because counts are monotone.
* Every prompt has one tail item: the tokens after its deepest pooled segment.
  Pooling a segment saves ``(count - 1) * length`` computed tokens, i.e.
  ``count - 1`` per pool page, independent of the segment length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class PlanSegment:
    segment_id: int
    parent_id: Optional[int]  # nearest pooled ancestor segment, None at root
    token_start: int
    token_end: int
    members: tuple[int, ...]  # wave indices of the prompts sharing it

    @property
    def tokens(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class PlanItem:
    """One unit of prefill compute inside a prepacked chunk.

    ``kind == "segment"``: ``ref`` is a pooled segment id; its KV is written
    into the pool (and offloaded to Host once).
    ``kind == "tail"``: ``ref`` is a wave index; its KV is offloaded to Host
    and its last token produces the first output token.
    Either way attention reads ``prefix_segments`` from the pool as the
    already-computed prefix ``[0, token_start)``.
    """

    kind: str
    ref: int
    prefix_segments: tuple[int, ...]
    token_start: int
    token_end: int

    @property
    def tokens(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class PlanChunk:
    items: tuple[PlanItem, ...]
    pool_pages_in_use: int  # after this chunk's segment writes, before release
    release_after: tuple[int, ...]  # pooled segments freed once it completes


@dataclass(frozen=True)
class WavePrefixPlan:
    block_tokens: int
    pool_pages: int
    chunk_tokens: int
    threshold: Optional[int]  # None: nothing pooled
    segments: tuple[PlanSegment, ...]
    chunks: tuple[PlanChunk, ...]
    prompt_tokens: int
    computed_tokens: int
    peak_pool_pages: int

    @property
    def saved_tokens(self) -> int:
        return self.prompt_tokens - self.computed_tokens

    def segment_pages(self, segment_id: int) -> int:
        return self.segments[segment_id].tokens // self.block_tokens


@dataclass(frozen=True)
class _Tree:
    start_block: tuple[int, ...]
    end_block: tuple[int, ...]
    parent: tuple[Optional[int], ...]  # parent segment in the full tree
    count: tuple[int, ...]
    paths: tuple[tuple[int, ...], ...]  # per prompt: segments root to leaf


def _build_tree(prompts: Sequence[Sequence[int]], block_tokens: int) -> _Tree:
    children: list[dict[tuple[int, ...], int]] = [{}]
    parent_node = [-1]
    depth = [0]
    count = [0]
    node_paths: list[list[int]] = []
    for tokens in prompts:
        shareable = (len(tokens) - 1) // block_tokens
        node = 0
        path = []
        for block in range(shareable):
            key = tuple(tokens[block * block_tokens:(block + 1) * block_tokens])
            nxt = children[node].get(key)
            if nxt is None:
                nxt = len(count)
                children[node][key] = nxt
                children.append({})
                parent_node.append(node)
                depth.append(depth[node] + 1)
                count.append(0)
            count[nxt] += 1
            path.append(nxt)
            node = nxt
        node_paths.append(path)

    # Nodes are numbered after their parent, so one ascending pass compresses
    # every chain whose parent neither branches nor loses a prompt.
    seg_of = [-1] * len(count)
    start_block: list[int] = []
    end_block: list[int] = []
    seg_parent: list[Optional[int]] = []
    seg_count: list[int] = []
    for node in range(1, len(count)):
        up = parent_node[node]
        if up != 0 and len(children[up]) == 1 and count[up] == count[node]:
            seg = seg_of[up]
        else:
            seg = len(start_block)
            start_block.append(depth[node] - 1)
            end_block.append(depth[node])
            seg_parent.append(None if up == 0 else seg_of[up])
            seg_count.append(count[node])
        seg_of[node] = seg
        end_block[seg] = depth[node]

    paths = []
    for node_path in node_paths:
        segs: list[int] = []
        for node in node_path:
            if not segs or segs[-1] != seg_of[node]:
                segs.append(seg_of[node])
        paths.append(tuple(segs))
    return _Tree(
        tuple(start_block), tuple(end_block), tuple(seg_parent),
        tuple(seg_count), tuple(paths),
    )


def _schedule(
    tree: _Tree,
    prompt_lengths: Sequence[int],
    block_tokens: int,
    pool_pages: int,
    chunk_tokens: int,
    threshold: Optional[int],
) -> Optional[WavePrefixPlan]:
    """List-schedule one threshold; None if the pool deadlocks."""

    pooled_old = [
        threshold is not None and c >= max(threshold, 2) for c in tree.count
    ]
    # Renumber pooled segments densely in DFS order of the pooled tree.
    kids: dict[Optional[int], list[int]] = {}
    for seg, is_pooled in enumerate(pooled_old):
        if is_pooled:
            kids.setdefault(tree.parent[seg], []).append(seg)
    chains: list[tuple[int, ...]] = []  # per prompt, old ids, root to leaf
    tails_at: dict[Optional[int], list[int]] = {}
    for index, path in enumerate(tree.paths):
        chain = tuple(seg for seg in path if pooled_old[seg])
        if chain != path[:len(chain)]:
            raise AssertionError("pooled set must be closed under ancestors")
        chains.append(chain)
        tails_at.setdefault(chain[-1] if chain else None, []).append(index)

    order_old: list[int] = []
    stack = list(reversed(kids.get(None, [])))
    while stack:
        seg = stack.pop()
        order_old.append(seg)
        stack.extend(reversed(kids.get(seg, [])))
    new_id = {old: new for new, old in enumerate(order_old)}
    members: dict[int, list[int]] = {old: [] for old in order_old}
    for index, chain in enumerate(chains):
        for old in chain:
            members[old].append(index)
    segments = tuple(
        PlanSegment(
            segment_id=new_id[old],
            parent_id=None if tree.parent[old] is None else new_id[tree.parent[old]],
            token_start=tree.start_block[old] * block_tokens,
            token_end=tree.end_block[old] * block_tokens,
            members=tuple(members[old]),
        )
        for old in order_old
    )

    def ancestry(seg: Optional[int]) -> tuple[int, ...]:
        out = []
        while seg is not None:
            out.append(seg)
            seg = segments[seg].parent_id
        return tuple(reversed(out))

    # Work items in DFS order: a segment, then the tails ending at it, then
    # its child subtrees; prompts that share nothing come last.
    items: list[PlanItem] = []

    def emit_tails(old: Optional[int]) -> None:
        for index in tails_at.get(old, []):
            chain = tuple(new_id[seg] for seg in chains[index])
            start = segments[chain[-1]].token_end if chain else 0
            items.append(PlanItem("tail", index, chain, start, prompt_lengths[index]))

    for old in order_old:
        seg = segments[new_id[old]]
        items.append(PlanItem(
            "segment", seg.segment_id, ancestry(seg.parent_id),
            seg.token_start, seg.token_end,
        ))
        emit_tails(old)
    emit_tails(None)

    pages = [seg.tokens // block_tokens for seg in segments]
    # Deepest pooled page path below each segment: what must still fit to
    # finish one leaf of its subtree after admitting it.
    need = list(pages)
    for seg in reversed(range(len(segments))):
        parent = segments[seg].parent_id
        if parent is not None:
            need[parent] = max(need[parent], pages[parent] + need[seg])
    pending_below = [0] * len(segments)
    for item in items:
        for seg in item.prefix_segments:
            pending_below[seg] += 1

    done_segments: set[int] = set()
    remaining = list(items)
    live = 0
    peak = 0
    chunks: list[PlanChunk] = []
    while remaining:
        chosen: list[PlanItem] = []
        used = 0
        written = 0
        rest: list[PlanItem] = []
        for item in remaining:
            ready = all(seg in done_segments for seg in item.prefix_segments)
            fits = not chosen or used + item.tokens <= chunk_tokens
            if item.kind == "segment":
                fits = fits and live + written + need[item.ref] <= pool_pages
            if ready and fits:
                chosen.append(item)
                used += item.tokens
                if item.kind == "segment":
                    written += pages[item.ref]
            else:
                rest.append(item)
        if not chosen:
            return None
        live += written
        peak = max(peak, live)
        released = []
        for item in chosen:
            if item.kind == "segment":
                done_segments.add(item.ref)
            for seg in item.prefix_segments:
                pending_below[seg] -= 1
        for item in chosen:
            if item.kind == "segment" and pending_below[item.ref] == 0:
                released.append(item.ref)
            for seg in item.prefix_segments:
                if pending_below[seg] == 0 and seg not in released:
                    released.append(seg)
        # A segment is freed exactly once: after the chunk that completes its
        # subtree; mark it so later chunks never report it again.
        for seg in released:
            pending_below[seg] = -1
            live -= pages[seg]
        chunks.append(PlanChunk(tuple(chosen), live + sum(pages[s] for s in released), tuple(sorted(released))))
        remaining = rest

    prompt_tokens = sum(prompt_lengths)
    computed = sum(item.tokens for chunk in chunks for item in chunk.items)
    return WavePrefixPlan(
        block_tokens=block_tokens,
        pool_pages=pool_pages,
        chunk_tokens=chunk_tokens,
        threshold=threshold if segments else None,
        segments=segments,
        chunks=tuple(chunks),
        prompt_tokens=prompt_tokens,
        computed_tokens=computed,
        peak_pool_pages=peak,
    )


def plan_wave_prefix_sharing(
    prompts: Sequence[Sequence[int]],
    *,
    block_tokens: int,
    pool_pages: int,
    chunk_tokens: int,
) -> WavePrefixPlan:
    """Plan one wave; pools the most segments whose schedule fits the pool.

    The threshold starts at 2 (every shared segment pooled) and rises until
    the schedule fits ``pool_pages``; with nothing pooled every prompt is a
    single tail, which always fits.
    """

    if block_tokens <= 0 or chunk_tokens <= 0 or pool_pages < 0:
        raise ValueError("block_tokens and chunk_tokens must be positive, pool_pages >= 0")
    if any(len(tokens) == 0 for tokens in prompts):
        raise ValueError("every prompt must be non-empty")
    lengths = [len(tokens) for tokens in prompts]
    tree = _build_tree(prompts, block_tokens)
    # 2 means every shared segment; higher values only where counts exist.
    for threshold in sorted({2} | {c for c in tree.count if c > 2}):
        plan = _schedule(tree, lengths, block_tokens, pool_pages, chunk_tokens, threshold)
        if plan is not None:
            return plan
    plan = _schedule(tree, lengths, block_tokens, pool_pages, chunk_tokens, None)
    assert plan is not None
    return plan


def assign_ranks_for_sharing(
    prompts: Sequence[Sequence[int]],
    *,
    world_size: int,
    block_tokens: int,
) -> tuple[int, ...]:
    """Assign prompts to DP ranks so shared prefixes are computed on one rank.

    Each rank has its own prefix pool, so prompts sharing a prefix only save
    compute when they land on the same rank. Units start as the whole wave;
    the heaviest unit is split (at its next branch, or in halves when its
    prompts are identical) only while that lowers the busiest rank's load
    under first-fit-decreasing placement. Load is the unit's computed tokens
    with every shared segment pooled once.
    """

    if world_size <= 0 or block_tokens <= 0:
        raise ValueError("world_size and block_tokens must be positive")
    count = len(prompts)
    if count == 0:
        return ()

    def cost(members: list[int]) -> int:
        tree = _build_tree([prompts[i] for i in members], block_tokens)
        saved = sum(
            (c - 1) * (end - start)
            for c, start, end in zip(tree.count, tree.start_block, tree.end_block)
            if c >= 2
        )
        return sum(len(prompts[i]) for i in members) - saved * block_tokens

    def split(members: list[int], depth: int) -> Optional[list[tuple[list[int], int]]]:
        if len(members) == 1:
            return None
        while True:
            groups: dict[tuple[int, ...], list[int]] = {}
            ended: list[int] = []
            for i in members:
                if (len(prompts[i]) - 1) // block_tokens > depth:
                    block = tuple(prompts[i][depth * block_tokens:(depth + 1) * block_tokens])
                    groups.setdefault(block, []).append(i)
                else:
                    ended.append(i)
            if not groups:  # identical shareable prefixes: halve the unit
                half = len(members) // 2
                return [(members[:half], depth), (members[half:], depth)]
            if len(groups) == 1 and not ended:
                depth += 1  # no branch at this block, look one block deeper
                continue
            parts = [list(g) for g in groups.values()]
            # Prompts ending here share [0, depth) with every child; keep them
            # with the largest child instead of recomputing that prefix alone.
            max(parts, key=len).extend(ended)
            if len(parts) == 1:  # only the ended prompts branched off
                parts = [[i for i in members if i not in ended], ended]
            return [(sorted(p), depth + 1) for p in parts]

    def place(loads: list[int]) -> tuple[int, list[int]]:
        rank_load = [0] * world_size
        where = [0] * len(loads)
        for unit in sorted(range(len(loads)), key=lambda k: (-loads[k], k)):
            rank = min(range(world_size), key=lambda r: (rank_load[r], r))
            where[unit] = rank
            rank_load[rank] += loads[unit]
        return max(rank_load), where

    units: list[tuple[list[int], int]] = [(list(range(count)), 0)]
    loads = [cost(units[0][0])]
    span, where = place(loads)
    while True:
        heavy = max(range(len(units)), key=lambda k: (loads[k], -k))
        parts = split(*units[heavy])
        if parts is None:
            break
        trial_units = units[:heavy] + units[heavy + 1:] + parts
        trial_loads = loads[:heavy] + loads[heavy + 1:] + [cost(m) for m, _ in parts]
        trial_span, trial_where = place(trial_loads)
        if trial_span >= span:
            break
        units, loads, span, where = trial_units, trial_loads, trial_span, trial_where
    ranks = [0] * count
    for unit, (members, _) in enumerate(units):
        for i in members:
            ranks[i] = where[unit]
    return tuple(ranks)


def validate_wave_prefix_plan(
    plan: WavePrefixPlan, prompts: Sequence[Sequence[int]]
) -> None:
    """Assert every invariant the executor relies on; raises AssertionError."""

    block = plan.block_tokens
    segments = plan.segments
    for seg in segments:
        assert seg.token_start % block == 0 and seg.token_end % block == 0
        assert seg.token_end > seg.token_start
        assert len(seg.members) >= 2
        if seg.parent_id is not None:
            parent = segments[seg.parent_id]
            assert parent.token_end == seg.token_start
            assert set(seg.members) <= set(parent.members)
        else:
            assert seg.token_start == 0
        first = prompts[seg.members[0]][seg.token_start:seg.token_end]
        for index in seg.members:
            tokens = prompts[index]
            assert seg.token_end <= len(tokens) - 1, "last prompt token must stay private"
            assert list(tokens[seg.token_start:seg.token_end]) == list(first)

    computed_at: dict[tuple[str, int], int] = {}
    released_at: dict[int, int] = {}
    live = 0
    peak = 0
    for position, chunk in enumerate(plan.chunks):
        assert chunk.items, "empty chunk"
        size = sum(item.tokens for item in chunk.items)
        assert size <= plan.chunk_tokens or len(chunk.items) == 1
        for item in chunk.items:
            key = (item.kind, item.ref)
            assert key not in computed_at, f"{key} computed twice"
            computed_at[key] = position
            for seg in item.prefix_segments:
                assert computed_at.get(("segment", seg), position) < position, (
                    f"{key} reads segment {seg} before it is computed"
                )
                assert released_at.get(seg, position) >= position, (
                    f"{key} reads segment {seg} after it was released"
                )
            expected = 0 if not item.prefix_segments else segments[item.prefix_segments[-1]].token_end
            assert item.token_start == expected
            chain = item.prefix_segments
            for up, down in zip(chain, chain[1:]):
                assert segments[down].parent_id == up
            if item.kind == "segment":
                seg = segments[item.ref]
                assert (item.token_start, item.token_end) == (seg.token_start, seg.token_end)
                live += plan.segment_pages(item.ref)
            else:
                assert item.token_end == len(prompts[item.ref])
                assert item.tokens >= 1
                # The chain is exactly the pooled segments this prompt is in.
                assert set(chain) == {
                    s.segment_id for s in segments if item.ref in s.members
                }
        peak = max(peak, live)
        assert chunk.pool_pages_in_use == live
        for seg in chunk.release_after:
            assert seg not in released_at, f"segment {seg} released twice"
            released_at[seg] = position
            live -= plan.segment_pages(seg)
    assert live == 0, "pool must be empty after the wave"
    assert peak == plan.peak_pool_pages <= plan.pool_pages
    assert set(released_at) == set(range(len(segments)))
    expected_items = {("segment", s.segment_id) for s in segments}
    expected_items |= {("tail", i) for i in range(len(prompts))}
    assert set(computed_at) == expected_items

    # Released exactly after the last chunk that reads it.
    last_read = {s.segment_id: computed_at[("segment", s.segment_id)] for s in segments}
    for position, chunk in enumerate(plan.chunks):
        for item in chunk.items:
            for seg in item.prefix_segments:
                last_read[seg] = max(last_read[seg], position)
    assert released_at == last_read

    total = sum(len(tokens) for tokens in prompts)
    computed = sum(item.tokens for chunk in plan.chunks for item in chunk.items)
    saved = sum((len(seg.members) - 1) * seg.tokens for seg in segments)
    assert plan.prompt_tokens == total
    assert plan.computed_tokens == computed
    assert total == computed + saved, "computed + saved must equal prompt tokens"
