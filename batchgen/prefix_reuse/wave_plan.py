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

Materialized segments (M3 Slice 3)
----------------------------------
A prompt that continues a committed trajectory frontier carries a prefix whose
KV is already on the Host. Its plan contribution is a MATERIALIZED segment:
the same geometry as a pooled segment, so the prompt's tail reads it through
the same page table, but no row computes it -- the executor LOADS it. Such a
prompt takes no part in the in-wave prefix tree (the arbitration of design
note section 7.2: pooled waves keep recomputing, materialization serves only
trajectory re-entry), so its chain is exactly its materialized segment and its
tail starts at the frontier boundary.

Accounting therefore splits three ways instead of two::

    prompt_tokens == computed_tokens
                   + sum (members - 1) * tokens  over computed segments
                   + sum  members      * tokens  over materialized segments

A computed segment saves every member but the one that computes it; a
materialized segment computes for nobody, so it saves all of its members.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


class PoolMaterializationDoesNotFit(RuntimeError):
    """No threshold schedules this wave while its hits are materialized.

    Named so the caller falls back deliberately -- drop the materialized
    segments and plan the wave as a recompute -- instead of a silent reuse of
    whichever plan happened to fit.
    """


@dataclass(frozen=True)
class MaterializedPrefix:
    """One prompt's committed prefix, to be LOADED into the pool.

    ``index`` is the wave index of the prompt and ``tokens`` the frontier
    boundary: page-aligned, at least one block, and leaving the prompt's last
    token private exactly as a computed segment must.
    """

    index: int
    tokens: int


@dataclass(frozen=True)
class PlanSegment:
    segment_id: int
    parent_id: Optional[int]  # nearest pooled ancestor segment, None at root
    token_start: int
    token_end: int
    members: tuple[int, ...]  # wave indices of the prompts sharing it
    # LOADED from Host KV instead of computed by a row of this wave. The
    # geometry is a pooled segment's, so a reader's page table is unchanged;
    # only who writes the pages differs.
    materialized: bool = False

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

    ``kind == "materialize"``: ``ref`` is a MATERIALIZED segment id. It is NOT
    a forward row -- it carries no query token, contributes nothing to the
    chunk's token budget and nothing to ``computed_tokens``. It is an item so
    that the pool pages it occupies, the order it is resident in and the chunk
    it is released after are planned by the one scheduler that plans the
    computed segments.
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
    # An item may read a segment written earlier in its own chunk. Valid when
    # every layer writes the whole chunk's KV before attending over it.
    same_chunk_ancestors: bool = False
    # Cap on the keys a chunk's rows attend over (sum of item token_end), for
    # executors that gather every row's prefix keys; None: uncapped.
    chunk_keys: Optional[int] = None
    # Tokens served by materialized segments: loaded once each, never computed.
    materialized_tokens: int = 0

    @property
    def saved_tokens(self) -> int:
        return self.prompt_tokens - self.computed_tokens

    @property
    def materialized_segments(self) -> tuple[int, ...]:
        return tuple(
            seg.segment_id for seg in self.segments if seg.materialized
        )

    def segment_pages(self, segment_id: int) -> int:
        return self.segments[segment_id].tokens // self.block_tokens


@dataclass(frozen=True)
class PrefillPlanWork:
    computed_tokens: int
    full_attention_pairs: int
    item_rows: int
    chunks: int


def prefill_plan_work(plan: WavePrefixPlan) -> PrefillPlanWork:
    """Count executed tokens and causal attention pairs in one rank's plan.

    A token at absolute position ``p`` attends to ``p + 1`` keys in a full
    attention layer, including keys supplied by a pooled prefix. Each planned
    item is computed once, so shared prefixes are counted only once here.
    """
    items = [
        item
        for chunk in plan.chunks
        for item in chunk.items
        if item.kind != "materialize"  # no row, no query token, no pair
    ]
    pairs = sum(
        (
            item.token_end * (item.token_end + 1)
            - item.token_start * (item.token_start + 1)
        ) // 2
        for item in items
    )
    return PrefillPlanWork(
        computed_tokens=plan.computed_tokens,
        full_attention_pairs=pairs,
        item_rows=len(items),
        chunks=len(plan.chunks),
    )


@dataclass(frozen=True)
class _Tree:
    start_block: tuple[int, ...]
    end_block: tuple[int, ...]
    parent: tuple[Optional[int], ...]  # parent segment in the full tree
    count: tuple[int, ...]
    paths: tuple[tuple[int, ...], ...]  # per prompt: segments root to leaf


def _build_tree(
    prompts: Sequence[Sequence[int]],
    block_tokens: int,
    skip: frozenset[int] = frozenset(),
) -> _Tree:
    """The wave's prefix tree; ``skip``ped prompts contribute no block.

    A skipped prompt is one whose prefix is materialized: it shares nothing
    in-wave by construction, so it must not create or deepen a tree node --
    otherwise the same tokens would be both loaded and computed, and only one
    of the two would be read.
    """

    children: list[dict[tuple[int, ...], int]] = [{}]
    parent_node = [-1]
    depth = [0]
    count = [0]
    node_paths: list[list[int]] = []
    for index, tokens in enumerate(prompts):
        shareable = 0 if index in skip else (len(tokens) - 1) // block_tokens
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
    same_chunk_ancestors: bool = False,
    chunk_keys: Optional[int] = None,
    materialized: Sequence[tuple[int, tuple[int, ...]]] = (),
) -> Optional[WavePrefixPlan]:
    """List-schedule one threshold; None if the pool deadlocks.

    ``materialized`` is ``(tokens, members)`` per materialized segment, in the
    order the segments are numbered. Its members take no part in ``tree`` (the
    caller skipped them when building it), so a member's chain is exactly its
    materialized segment.
    """

    pooled_old = [
        threshold is not None and c >= max(threshold, 2) for c in tree.count
    ]
    num_materialized = len(materialized)
    # Which materialized segment each member reads; a non-member is absent.
    materialized_of: dict[int, int] = {}
    for segment_id, (_, members) in enumerate(materialized):
        for index in members:
            if index in materialized_of:
                raise ValueError(
                    f"prompt {index} is a member of two materialized segments"
                )
            materialized_of[index] = segment_id
    # Renumber pooled segments densely in DFS order of the pooled tree.
    kids: dict[Optional[int], list[int]] = {}
    for seg, is_pooled in enumerate(pooled_old):
        if is_pooled:
            kids.setdefault(tree.parent[seg], []).append(seg)
    chains: list[tuple[int, ...]] = []  # per prompt, old ids, root to leaf
    tails_at: dict[Optional[int], list[int]] = {}
    # Tails whose chain is a materialized segment, keyed by its NEW id.
    tails_at_materialized: dict[int, list[int]] = {
        segment_id: [] for segment_id in range(num_materialized)
    }
    for index, path in enumerate(tree.paths):
        chain = tuple(seg for seg in path if pooled_old[seg])
        if chain != path[:len(chain)]:
            raise AssertionError("pooled set must be closed under ancestors")
        chains.append(chain)
        segment_id = materialized_of.get(index)
        if segment_id is not None:
            if chain:
                raise AssertionError(
                    f"prompt {index} materializes its prefix but also shares "
                    "a computed segment"
                )
            tails_at_materialized[segment_id].append(index)
            continue
        tails_at.setdefault(chain[-1] if chain else None, []).append(index)

    order_old: list[int] = []
    stack = list(reversed(kids.get(None, [])))
    while stack:
        seg = stack.pop()
        order_old.append(seg)
        stack.extend(reversed(kids.get(seg, [])))
    # Materialized segments are numbered FIRST, so a plan's ids are stable
    # whatever the tree does behind them.
    new_id = {old: num_materialized + new for new, old in enumerate(order_old)}
    members: dict[int, list[int]] = {old: [] for old in order_old}
    for index, chain in enumerate(chains):
        for old in chain:
            members[old].append(index)
    segments = tuple(
        PlanSegment(
            segment_id=segment_id,
            parent_id=None,
            token_start=0,
            token_end=tokens,
            members=tuple(member_indices),
            materialized=True,
        )
        for segment_id, (tokens, member_indices) in enumerate(materialized)
    ) + tuple(
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

    # Materialized segments first, each followed by the tails that read it:
    # they are resident before any computed row needs the pool, and a reader
    # is never separated from its load by the whole computed tree.
    for segment_id in range(num_materialized):
        seg = segments[segment_id]
        items.append(PlanItem("materialize", segment_id, (), 0, seg.token_end))
        for index in tails_at_materialized[segment_id]:
            items.append(PlanItem(
                "tail", index, (segment_id,), seg.token_end,
                prompt_lengths[index],
            ))

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
        keys = 0
        written = 0
        rest: list[PlanItem] = []
        chosen_segments: set[int] = set()
        for item in remaining:
            # Items are in DFS order, so a segment precedes every reader.
            ready = all(
                seg in done_segments
                or (same_chunk_ancestors and seg in chosen_segments)
                for seg in item.prefix_segments
            )
            # A materialized segment is loaded, not forwarded: it spends pool
            # pages but no chunk tokens and no attention keys.
            loaded = item.kind == "materialize"
            fits = not chosen or loaded or (
                used + item.tokens <= chunk_tokens
                and (chunk_keys is None or keys + item.token_end <= chunk_keys)
            )
            if item.kind in ("segment", "materialize"):
                fits = fits and live + written + need[item.ref] <= pool_pages
            if ready and fits:
                chosen.append(item)
                if not loaded:
                    used += item.tokens
                    keys += item.token_end
                if item.kind in ("segment", "materialize"):
                    written += pages[item.ref]
                    chosen_segments.add(item.ref)
            else:
                rest.append(item)
        if not chosen:
            return None
        if all(item.kind == "materialize" for item in chosen):
            # Every chunk is one forward, and the load rides that forward's
            # layer loop. A chunk of loads alone would have no forward to ride,
            # so this schedule is refused here and the caller falls back to a
            # plan that recomputes -- rather than inventing a second pass.
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
    computed = sum(
        item.tokens
        for chunk in chunks
        for item in chunk.items
        if item.kind != "materialize"
    )
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
        same_chunk_ancestors=same_chunk_ancestors,
        chunk_keys=chunk_keys,
        materialized_tokens=sum(
            seg.tokens for seg in segments if seg.materialized
        ),
    )


def group_materialized_prefixes(
    prompts: Sequence[Sequence[int]],
    materialized: Sequence[MaterializedPrefix],
    *,
    block_tokens: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Validate the hits and fold identical prefixes into one segment each.

    Two prompts that re-enter the SAME frontier describe the same bytes, so
    they must read one segment: loading it twice would spend the pages and the
    H2D bandwidth twice for an identical result. Grouping is by (boundary,
    prefix tokens), and segments are ordered by their first member, so every
    rank that sees the same hits numbers them identically.
    """

    seen: set[int] = set()
    groups: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    order: list[tuple[int, tuple[int, ...]]] = []
    for hit in materialized:
        index, tokens = int(hit.index), int(hit.tokens)
        if not 0 <= index < len(prompts):
            raise ValueError(
                f"materialized prefix names prompt {index}, which is not one "
                f"of this wave's {len(prompts)} prompts"
            )
        if index in seen:
            raise ValueError(f"prompt {index} carries two materialized prefixes")
        seen.add(index)
        prompt = prompts[index]
        if tokens <= 0 or tokens % block_tokens:
            raise ValueError(
                f"materialized prefix of prompt {index} is {tokens} tokens, "
                f"which is not a positive multiple of the {block_tokens}-token "
                "block"
            )
        if tokens > len(prompt) - 1:
            raise ValueError(
                f"materialized prefix of prompt {index} covers {tokens} of "
                f"{len(prompt)} tokens; the last prompt token must stay "
                "private so the row produces a logit"
            )
        key = (tokens, tuple(prompt[:tokens]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(index)
    return tuple(
        (tokens, tuple(groups[(tokens, prefix)]))
        for tokens, prefix in order
    )


def plan_wave_prefix_sharing(
    prompts: Sequence[Sequence[int]],
    *,
    block_tokens: int,
    pool_pages: int,
    chunk_tokens: int,
    same_chunk_ancestors: bool = False,
    chunk_keys: Optional[int] = None,
    materialized: Sequence[MaterializedPrefix] = (),
    share_computed: bool = True,
) -> WavePrefixPlan:
    """Plan one wave; pools the most segments whose schedule fits the pool.

    The threshold starts at 2 (every shared segment pooled) and rises until
    the schedule fits ``pool_pages``; with nothing pooled every prompt is a
    single tail, which always fits.

    ``materialized`` names the prompts whose prefix is LOADED. They are kept
    out of the prefix tree and get one materialized segment per distinct
    frontier. When no threshold schedules the wave with them,
    ``PoolMaterializationDoesNotFit`` is raised: the caller then plans the
    same wave without them, which is the recompute fallback.

    ``share_computed=False`` pools no computed segment: only the
    materialized ones are segments, every other prompt is one tail. A
    session-hold wave plans this way -- it has no prefix cache to publish a
    computed segment into, so in-wave sharing would leave pages nobody owns.
    """

    if block_tokens <= 0 or chunk_tokens <= 0 or pool_pages < 0:
        raise ValueError("block_tokens and chunk_tokens must be positive, pool_pages >= 0")
    if any(len(tokens) == 0 for tokens in prompts):
        raise ValueError("every prompt must be non-empty")
    groups = group_materialized_prefixes(
        prompts, materialized, block_tokens=block_tokens
    )
    if groups and not same_chunk_ancestors:
        # The load rides the forward of the chunk it is planned in, so a
        # reader that may not read a segment resident in its own chunk would
        # have to wait a chunk for pages it already paid for -- and a
        # load-only chunk has no forward to ride. Refused by name instead of
        # planned into a shape this executor cannot run.
        raise ValueError(
            "materialized prefixes require a pool that lets a row read a "
            "segment resident in its own chunk (same_chunk_ancestors)"
        )
    lengths = [len(tokens) for tokens in prompts]
    skip = frozenset(
        index for _, members in groups for index in members
    )
    tree = _build_tree(prompts, block_tokens, skip)
    # 2 means every shared segment; higher values only where counts exist.
    thresholds = (
        sorted({2} | {c for c in tree.count if c > 2}) if share_computed else ()
    )
    for threshold in thresholds:
        plan = _schedule(tree, lengths, block_tokens, pool_pages, chunk_tokens,
                         threshold, same_chunk_ancestors, chunk_keys, groups)
        if plan is not None:
            return plan
    plan = _schedule(tree, lengths, block_tokens, pool_pages, chunk_tokens, None,
                     same_chunk_ancestors, chunk_keys, groups)
    if plan is None:
        if not groups:
            raise AssertionError(
                "a wave with nothing pooled is one tail per prompt and always "
                "schedules"
            )
        raise PoolMaterializationDoesNotFit(
            f"{len(groups)} materialized segment(s) covering "
            f"{sum(tokens for tokens, _ in groups)} token(s) do not schedule "
            f"in {pool_pages} pool page(s)"
        )
    return plan


def assign_ranks_for_sharing(
    prompts: Sequence[Sequence[int]],
    *,
    world_size: int,
    block_tokens: int,
    initial_loads: Optional[Sequence[float]] = None,
) -> tuple[int, ...]:
    """Assign prompts to DP ranks so shared prefixes are computed on one rank.

    Each rank has its own prefix pool, so prompts sharing a prefix only save
    compute when they land on the same rank. Units start as the whole wave;
    the heaviest unit is split (at its next branch, or in halves when its
    prompts are identical) only while that lowers the busiest rank's load
    under first-fit-decreasing placement. Load is the unit's computed tokens
    with every shared segment pooled once. ``initial_loads`` is each rank's
    load from prompts assigned earlier.
    """

    if world_size <= 0 or block_tokens <= 0:
        raise ValueError("world_size and block_tokens must be positive")
    count = len(prompts)
    if count == 0:
        return ()

    def cost(members: list[int]) -> float:
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

    def place(loads: list[float]) -> tuple[float, list[int]]:
        rank_load = list(initial_loads) if initial_loads is not None else [0] * world_size
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
        # A computed segment exists to be shared; a materialized one exists
        # because its member's prefix is already KV, which one member is
        # reason enough for.
        assert len(seg.members) >= (1 if seg.materialized else 2)
        if seg.materialized:
            assert seg.parent_id is None and seg.token_start == 0
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
    order_at: dict[tuple[str, int], int] = {}
    released_at: dict[int, int] = {}
    live = 0
    peak = 0
    for position, chunk in enumerate(plan.chunks):
        assert chunk.items, "empty chunk"
        forward = [item for item in chunk.items if item.kind != "materialize"]
        assert forward, "a chunk must carry the forward its loads ride"
        size = sum(item.tokens for item in forward)
        assert size <= plan.chunk_tokens or len(forward) == 1
        if plan.chunk_keys is not None:
            keys = sum(item.token_end for item in forward)
            assert keys <= plan.chunk_keys or len(forward) == 1
        for order, item in enumerate(chunk.items):
            # A materialized segment is resident for its readers exactly as a
            # computed one is, so both answer to the same key.
            key = ("segment" if item.kind == "materialize" else item.kind, item.ref)
            assert key not in computed_at, f"{key} computed twice"
            computed_at[key] = position
            order_at[key] = order
            for seg in item.prefix_segments:
                written = computed_at.get(("segment", seg), position + 1)
                same_chunk = (
                    plan.same_chunk_ancestors and written == position
                    and order_at[("segment", seg)] < order
                )
                assert written < position or same_chunk, (
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
            if item.kind in ("segment", "materialize"):
                seg = segments[item.ref]
                assert seg.materialized == (item.kind == "materialize")
                assert (item.token_start, item.token_end) == (seg.token_start, seg.token_end)
                assert item.prefix_segments == () or not seg.materialized
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
    computed = sum(
        item.tokens
        for chunk in plan.chunks
        for item in chunk.items
        if item.kind != "materialize"
    )
    saved = sum(
        (len(seg.members) - 1) * seg.tokens
        for seg in segments
        if not seg.materialized
    )
    # A materialized segment computes for nobody, so it saves every member.
    loaded = sum(
        len(seg.members) * seg.tokens for seg in segments if seg.materialized
    )
    assert plan.prompt_tokens == total
    assert plan.computed_tokens == computed
    assert plan.materialized_tokens == sum(
        seg.tokens for seg in segments if seg.materialized
    )
    assert total == computed + saved + loaded, (
        "computed + saved + loaded must equal prompt tokens"
    )
