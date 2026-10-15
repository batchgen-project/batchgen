# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The KDA level/fork path at the shape the live run ACTUALLY had.

``tests/unit/test_k3_kda_gapped_levels.py`` runs the same production code over
a wave of PAGE-ALIGNED prompt lengths nested in one chain, which has two
properties the live wave does not: every row's own length is a multiple of
``_KDA_CHUNK_SIZE``, and every prompt is a prefix of one global chain. The
schema-2 kernel dumps say the live wave is
``L(k) = 525 + 313k`` over 64 members with a 40-token per-member TEMPLATE
SUFFIX, one chunk per group, level-wise order - so every member tail is
RAGGED (40..103 tokens, never a chunk multiple except ch-011) and a member's
prompt is NOT a prefix of the next member's.

That is the difference that matters to a recurrent layer. A level is one
packed call over several rows, so a ragged row makes the NEXT row in the same
packed call start at a non-multiple of the chunk size; and the leaked tokens
the live run shows are template tokens, which only exist in this shape.

The wave shape (and its verification against the dumps) lives in
``tests/unit/test_k3_pool_live_wave_oracle.py``; the KDA harness - the CPU
stand-ins for fla's ``chunk_kda`` and the conv kernel, the state pools, the
slot allocator, the unsegmented reference - in
``tests/unit/test_k3_kda_pool_levels.py``. Both are reused so a verdict here
cannot differ by way of a different reference implementation.

What is asserted
----------------
* the LEVEL of every row and that ``chunk_levels`` cuts the chunk into
  exactly those contiguous runs, in increasing level order;
* the FORK of every row that continues a segment: the parent is its NEAREST
  pooled ancestor, the child is its own slot, a child never writes its
  parent's slot, and ``has_initial_state`` is set exactly for rows with a
  non-empty chain;
* the GATE, ``torch.equal``: for ALL 64 members, that member's pooled segment
  rows concatenated with its ragged tail row equal its own prompt run whole
  from zero state. Bit-exact, because K3's top-16 router turns a 1-ulp seam
  into an O(1) logit difference. A failure names the member, the piece, the
  level, the absolute position, that position's distance to the enclosing
  chunk boundary and whether it is inside the member's template suffix;
* every resume boundary is a multiple of ``_KDA_CHUNK_SIZE`` (the exactness
  premise) while every tail length is not (the live raggedness);
* copy-on-branch: after the wave each of the 31 segment slots still holds the
  state its OWN last token produced.

Run: PYTHONPATH=. python -m pytest tests/unit/test_k3_kda_live_wave_levels.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

torch.set_num_threads(1)

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, *relative):
    """Load by file path, reusing the module if pytest already imported it."""

    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = _ROOT.joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_KDA = _load("test_k3_kda_pool_levels", "tests", "unit",
             "test_k3_kda_pool_levels.py")
_WAVE = _load("test_k3_pool_live_wave_oracle", "tests", "unit",
              "test_k3_pool_live_wave_oracle.py")

SM = _KDA.SM
_LEVELS, _PLAN, _POOL, _EXEC = _KDA._LEVELS, _KDA._PLAN, _KDA._POOL, _KDA._EXEC

PAGE = _KDA.PAGE
assert PAGE == SM._KDA_CHUNK_SIZE == _WAVE.PAGE == 64, (
    "the pool page must be fla's chunk size and the live page, or no level "
    "could resume exactly"
)

FAMILY = _WAVE.FAMILY
LIVE_TOKENS = _WAVE.LIVE_TOKENS
MEMBERS = _WAVE.MEMBERS
GROUP_OF = _WAVE.GROUP_OF
NUM_GROUPS = _WAVE.NUM_GROUPS
LIVE_FLAGGED = _WAVE.LIVE_FLAGGED
DEEP_TAIL_MEMBER = _WAVE.DEEP_TAIL_MEMBER
SUFFIX = _WAVE.SUFFIX
GLOBAL_BASE = _WAVE.GLOBAL_BASE

CHUNK_TOKENS = 1 << 20  # the live cap is far above the wave: one chunk
SPLIT_CHUNK_TOKENS = 4096  # the control: several chunks, cross-chunk resume
POOL_PAGES = 4096


def member_prompt(member: int) -> list[int]:
    return _WAVE.member_prompt(member)


def chain_rows(end: int):
    """The shared chain's rows up to ``end``: what a segment computes."""

    return [(position, _WAVE.chain_token(position)) for position in range(end)]


# --------------------------------------------------------------------------- #
#  The oracle: levels and forks from the chain structure alone
# --------------------------------------------------------------------------- #
def oracle_wave(group: int):
    members = MEMBERS[group]
    prompts = [member_prompt(member) for member in members]
    pooled = _WAVE.oracle_pooled_runs(members)
    return members, prompts, pooled


def oracle_chain_length(member: int, pooled) -> int:
    """How many pooled segments this member's prompt passes through."""

    through = _WAVE.shared_blocks(member)
    return sum(1 for _, end, _ in pooled if end <= through)


def oracle_rows(group: int):
    """Per row: (level, parent segment or None, token_start, token_end)."""

    members, _, pooled = oracle_wave(group)
    levels: dict[tuple[str, int], tuple[int, int | None, int, int]] = {}
    for index, (start, end, _) in enumerate(pooled):
        levels[("segment", index)] = (
            index, None if index == 0 else index - 1, start * PAGE, end * PAGE,
        )
    for wave_index, member in enumerate(members):
        depth = oracle_chain_length(member, pooled)
        start = pooled[depth - 1][1] * PAGE if depth else 0
        levels[("tail", wave_index)] = (
            depth, depth - 1 if depth else None, start, LIVE_TOKENS[member],
        )
    return levels


# --------------------------------------------------------------------------- #
#  The wave through the real level/fork path
# --------------------------------------------------------------------------- #
def _state_pools(slots: int):
    return SimpleNamespace(
        recurrent_pool=torch.zeros(slots, _KDA.H, _KDA.D, _KDA.D),
        conv_q=torch.zeros(slots, _KDA.DIM, _KDA.W - 1),
        conv_k=torch.zeros(slots, _KDA.DIM, _KDA.W - 1),
        conv_v=torch.zeros(slots, _KDA.DIM, _KDA.W - 1),
    )


@pytest.fixture(autouse=True)
def _cpu_kernels(monkeypatch):
    """fla and the conv extension are CUDA-only; serve both from CPU code."""

    monkeypatch.setattr(SM, "_kda_prefill_conv", _KDA._cpu_conv)


def _plan(prompts, same_chunk: bool):
    plan = _PLAN.plan_wave_prefix_sharing(
        prompts, block_tokens=PAGE, pool_pages=POOL_PAGES,
        chunk_tokens=CHUNK_TOKENS if same_chunk else SPLIT_CHUNK_TOKENS,
        same_chunk_ancestors=same_chunk,
    )
    plan = _LEVELS.order_chunk_items_by_level(plan)
    _PLAN.validate_wave_prefix_plan(plan, prompts)
    assert plan.threshold == 2, "the pool budget forced the threshold up"
    return plan


def _batches(plan, prompts):
    scratch = _POOL.scratch_pages_bound(
        plan.chunk_tokens, max(len(p) for p in prompts), len(prompts), PAGE
    )
    pool = _POOL.PrefillPrefixPool(
        num_layers=1, num_pages=plan.pool_pages + scratch, page_tokens=PAGE,
        num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
    )
    executor = _EXEC.PoolChunkExecutor(plan, prompts, pool)
    batches = []
    for chunk in plan.chunks:
        batches.append(executor.build(chunk))
        executor.finish(chunk)
    executor.close()
    return batches


def run_live_wave(group: int, same_chunk: bool) -> dict:
    """Prefill one group's live-shaped wave level by level."""

    members, prompts, pooled = oracle_wave(group)
    plan = _plan(prompts, same_chunk)

    # The planner's segments must BE the oracle runs before indexing into them.
    assert len(plan.segments) == len(pooled)
    for index, (segment, (start, end, _)) in enumerate(
        zip(plan.segments, pooled)
    ):
        assert segment.segment_id == index
        assert segment.parent_id == (None if index == 0 else index - 1)
        assert (segment.token_start, segment.token_end) == (
            start * PAGE, end * PAGE,
        ), (
            f"group {group} segment {index}: planner "
            f"[{segment.token_start}, {segment.token_end}) != oracle "
            f"[{start * PAGE}, {end * PAGE})"
        )

    weights, kwargs = _KDA._conv_weights(), _KDA._kernel_kwargs()
    state = _state_pools(len(plan.segments) + len(members) + 4)
    slots = _KDA.Slots(state)
    sids = {seg.segment_id: -(1 + seg.segment_id) for seg in plan.segments}
    outputs: dict[tuple[str, int], torch.Tensor] = {}
    row_level: dict[tuple[str, int], int] = {}
    observed: list[dict] = []

    for position, batch in enumerate(_batches(plan, prompts)):
        row_ids = [
            sids[item.ref] if item.kind == "segment" else item.ref
            for item in batch.items
        ]
        slot_ids, has_initial_state = slots.prepare_prefill(row_ids)
        levels = _LEVELS.state_levels(
            batch, segment_sids=sids, row_ids=row_ids, slots=slot_ids,
            has_initial_state=has_initial_state, slot_of=slots.lookup,
        )
        cuts = _LEVELS.chunk_levels(batch)
        observed.append({
            "chunk": position, "batch": batch, "row_ids": row_ids,
            "slot_ids": slot_ids, "levels": levels,
            "resumes": has_initial_state,
            "forks": dict(_LEVELS.fork_pairs(batch, sids)),
            "cuts": cuts,
        })
        for cut in cuts:
            for row in range(cut.row_start, cut.row_end):
                item = batch.items[row]
                row_level[(item.kind, item.ref)] = cut.depth

        q, k, v, f, beta = _KDA._split([
            (int(pos), int(token))
            for pos, token in zip(batch.position_ids, batch.input_ids)
        ])
        out = SM._kda_prefill_levels(
            _KDA.FAKE_LAYER, _KDA._cpu_chunk_kda, state, levels,
            q, k, v, f, beta, weights, kwargs, None,
        )
        for row, item in enumerate(batch.items):
            lo, hi = batch.row_bounds[row], batch.row_bounds[row + 1]
            outputs[(item.kind, item.ref)] = out[0, lo:hi]

    return {
        "members": members, "prompts": prompts, "pooled": pooled, "plan": plan,
        "outputs": outputs, "row_level": row_level, "state": state,
        "slots": slots, "sids": sids, "observed": observed,
        "weights": weights, "kwargs": kwargs,
    }


_WAVES: dict[tuple[int, bool], dict] = {}


def live_wave(group: int, same_chunk: bool) -> dict:
    key = (group, same_chunk)
    if key not in _WAVES:
        _WAVES[key] = run_live_wave(group, same_chunk)
    return _WAVES[key]


# Per-member reference: at the live shape a prompt is NOT a prefix of the next
# member's (the template suffix ends every one of them), so the page-aligned
# file's one-chain shortcut does not apply and each member is run whole.
_REFERENCE: dict[int, torch.Tensor] = {}


def reference_output(member: int) -> torch.Tensor:
    out = _REFERENCE.get(member)
    if out is None:
        weights, kwargs = _KDA._conv_weights(), _KDA._kernel_kwargs()
        out, _ = _KDA._unsegmented(
            _KDA._rows_of(member_prompt(member)), weights, kwargs,
        )
        _REFERENCE[member] = out
    return out


# --------------------------------------------------------------------------- #
#  (0) The premise: ragged rows over chunk-aligned resume points
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_every_resume_is_chunk_aligned_and_every_tail_is_ragged(group):
    """The exactness premise, and the raggedness the old shape lacked.

    A row resumes a segment only at a multiple of ``_KDA_CHUNK_SIZE``, which
    is what makes a bit-exact comparison possible at all; but the row's own
    LENGTH is ragged, so inside a packed level call the next row starts off
    the chunk grid. Exactly one member (ch-011) is the exception.
    """

    members, prompts, pooled = oracle_wave(group)
    oracle = oracle_rows(group)
    ragged = 0
    for (kind, ref), (_, parent, start, end) in oracle.items():
        assert start % SM._KDA_CHUNK_SIZE == 0, (kind, ref, start)
        if (end - start) % SM._KDA_CHUNK_SIZE:
            ragged += 1
        elif kind == "tail":
            assert members[ref] in (11, DEEP_TAIL_MEMBER[group]), members[ref]
    # Every tail but at most one is ragged; segments are whole pages.
    tails = sum(1 for kind, _ in oracle if kind == "tail")
    assert tails == 32
    assert ragged >= tails - 1, (group, ragged)
    assert all(
        (end - start) % PAGE == 0
        for (kind, _), (_, _, start, end) in oracle.items() if kind == "segment"
    )


def test_the_live_flagged_members_are_the_short_end_of_their_group():
    """Guard the target set: ch-001/004 in group 0, ch-003/011/013 in group 1."""

    assert [GROUP_OF[ch] for ch in LIVE_FLAGGED] == [0, 1, 0, 1, 1]
    for ch in LIVE_FLAGGED:
        rank = sorted(MEMBERS[GROUP_OF[ch]]).index(ch)
        assert rank <= 6, (ch, rank)


# --------------------------------------------------------------------------- #
#  (i) Level and fork structure against the oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("same_chunk", [True, False])
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_row_levels_and_forks_match_the_chain_oracle(group, same_chunk):
    """Every row's level, fork parent, fork child and resume flag."""

    wave = live_wave(group, same_chunk)
    oracle = oracle_rows(group)
    slots, sids = wave["slots"], wave["sids"]
    seen: set[tuple[str, int]] = set()

    for record in wave["observed"]:
        batch, cuts = record["batch"], record["cuts"]
        context = f"group {group} same_chunk={same_chunk} chunk {record['chunk']}"

        assert cuts[0].row_start == 0
        assert cuts[-1].row_end == len(batch.items)
        for earlier, later in zip(cuts, cuts[1:]):
            assert earlier.row_end == later.row_start, context
            assert earlier.depth < later.depth, context

        for cut in cuts:
            for row in range(cut.row_start, cut.row_end):
                item = batch.items[row]
                key = (item.kind, item.ref)
                depth, parent, start, end = oracle[key]
                where = f"{context} row {row} {item.kind} {item.ref}"
                seen.add(key)

                assert cut.depth == depth, (
                    f"{where}: level {cut.depth} != oracle {depth}"
                )
                assert (item.token_start, item.token_end) == (start, end), (
                    f"{where}: item [{item.token_start}, {item.token_end}) != "
                    f"oracle [{start}, {end})"
                )
                resumed = bool(record["resumes"][row]) or row in record["forks"]
                assert resumed == (parent is not None), (
                    f"{where}: resume={resumed}, oracle parent={parent}"
                )
                if parent is None:
                    assert row not in record["forks"], (
                        f"{where}: forks although it has no pooled ancestor"
                    )
                    continue
                assert record["forks"][row] == sids[parent], (
                    f"{where}: forks from sequence {record['forks'][row]}, "
                    f"oracle nearest ancestor is segment {parent}"
                )

        for cut, level in zip(cuts, record["levels"]):
            children = [
                row for row in range(cut.row_start, cut.row_end)
                if row in record["forks"]
            ]
            # A level's packed call starts where the cut starts, whatever the
            # previous level's ragged rows left behind.
            assert level.token_start == batch.row_bounds[cut.row_start]
            assert level.token_end == batch.row_bounds[cut.row_end]
            if not children:
                assert level.fork_parents is None and level.fork_children is None
                continue
            assert level.fork_parents.tolist() == [
                slots.lookup(record["forks"][row]) for row in children
            ], context
            assert level.fork_children.tolist() == [
                slots.lookup(record["row_ids"][row]) for row in children
            ], context
            assert not bool((level.fork_parents == level.fork_children).any()), (
                f"{context}: a fork would overwrite its own parent's slot"
            )

    assert seen == set(oracle), "not every planned row went through a level"


# --------------------------------------------------------------------------- #
#  (ii) The gate: pooled rows are bit-exact per member, and WHERE if not
# --------------------------------------------------------------------------- #
def _describe_divergence(wave, member: int, wave_index: int, pieces, got,
                         expected) -> str:
    """Name the member, the piece, the level and the boundary that moved."""

    plan = wave["plan"]
    moved = (got != expected).any(dim=-1).any(dim=-1).nonzero()
    first = int(moved[0]) if moved.numel() else -1
    prompt_length = LIVE_TOKENS[member]
    template_at = prompt_length - SUFFIX
    where = "nowhere"
    if first >= 0:
        at = 0
        for key, rows in pieces:
            if at <= first < at + rows.shape[0]:
                level = wave["row_level"].get(key, -1)
                offset = first - at
                where = (
                    f"{key[0]} {key[1]} (level {level}), token {offset} of "
                    f"{rows.shape[0]} in that piece, absolute position {first}"
                )
                break
            at += rows.shape[0]
    return (
        f"ch-{member:03d} (global {GLOBAL_BASE + member}, group "
        f"{GROUP_OF[member]}, wave index {wave_index}, L={prompt_length}, "
        f"chain of {len(pieces) - 1} segments ending at "
        f"{plan.segments[[s.segment_id for s in plan.segments if wave_index in s.members][-1]].token_end if len(pieces) > 1 else 0})"
        f" diverges at {where}; first differing absolute position {first}, "
        f"which is {first % SM._KDA_CHUNK_SIZE} tokens past a "
        f"{SM._KDA_CHUNK_SIZE}-token chunk boundary and "
        f"{'INSIDE' if first >= template_at else 'before'} the 40-token "
        f"template suffix (starts at {template_at}); "
        f"max_abs={(got - expected).abs().max().item():.3e}, "
        f"{int(moved.numel())} of {got.shape[0]} tokens moved"
    )


def _gate(wave, group: int, wave_index: int, member: int) -> None:
    plan, outputs = wave["plan"], wave["outputs"]
    chain = [seg for seg in plan.segments if wave_index in seg.members]
    pieces = [
        (("segment", seg.segment_id), outputs[("segment", seg.segment_id)])
        for seg in chain
    ]
    pieces.append((("tail", wave_index), outputs[("tail", wave_index)]))
    got = torch.cat([rows for _, rows in pieces])
    expected = reference_output(member)

    assert got.shape == expected.shape, (
        f"group {group} ch-{member:03d}: pooled rows cover {got.shape[0]} "
        f"tokens, prompt has {expected.shape[0]}"
    )
    if not torch.equal(got, expected):
        raise AssertionError(
            _describe_divergence(wave, member, wave_index, pieces, got, expected)
        )


def _case_id(member: int) -> str:
    tag = f"g{GROUP_OF[member]}-ch{member:03d}"
    if member in LIVE_FLAGGED:
        tag += "-LIVE"
    if member == DEEP_TAIL_MEMBER[GROUP_OF[member]]:
        tag += "-deeptail"
    return tag


TARGETS = sorted(set(LIVE_FLAGGED) | set(DEEP_TAIL_MEMBER.values()))


@pytest.mark.parametrize("same_chunk", [True, False])
@pytest.mark.parametrize("member", TARGETS, ids=[_case_id(m) for m in TARGETS])
def test_the_live_corrupted_members_resume_their_chain_exactly(
    member, same_chunk
):
    """ch-001/004 (group 0), ch-003/011/013 (group 1), plus the deep tails."""

    group = GROUP_OF[member]
    wave = live_wave(group, same_chunk)
    _gate(wave, group, wave["members"].index(member), member)


@pytest.mark.parametrize("same_chunk", [True, False])
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_every_member_resumes_its_chain_exactly(group, same_chunk):
    """All 32 members of the group; the report names the shallowest failure."""

    wave = live_wave(group, same_chunk)
    failures = []
    for wave_index, member in enumerate(wave["members"]):
        try:
            _gate(wave, group, wave_index, member)
        except AssertionError as error:
            failures.append(str(error))
    if failures:
        pytest.fail(
            f"group {group} same_chunk={same_chunk}: {len(failures)} of "
            f"{len(wave['members'])} members moved; first:\n"
            + "\n".join(failures[:3])
        )


# --------------------------------------------------------------------------- #
#  (iii) Copy-on-branch: a segment's slot still holds its own end state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_every_segment_slot_still_holds_its_own_end_state(group):
    """Each of the 31 segments owns the state its own last token produced.

    A child writing its parent's slot would corrupt every deeper member,
    which is the shape of the live symptom.
    """

    wave = live_wave(group, same_chunk=False)
    weights, kwargs = wave["weights"], wave["kwargs"]
    state, slots, sids = wave["state"], wave["slots"], wave["sids"]

    for segment in wave["plan"].segments:
        _, expected = _KDA._unsegmented(
            chain_rows(segment.token_end), weights, kwargs,
        )
        got = _KDA._final_states(state, slots.lookup(sids[segment.segment_id]))
        for pool, reference in zip(got, _KDA._final_states(expected, 0)):
            assert torch.equal(pool, reference), (
                f"group {group} segment {segment.segment_id} "
                f"[{segment.token_start}, {segment.token_end}) "
                f"({segment.tokens // PAGE} pages) was overwritten by a "
                f"child: max_abs={(pool - reference).abs().max().item():.3e}"
            )
