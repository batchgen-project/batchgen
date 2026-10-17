# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The KDA level/fork path on a PAGE-ALIGNED two-group gapped chain wave.

CORRECTION (2026-10-15, from the schema-2 kernel dumps of the live two-node
run): this file's wave is not the live one. Every row's own length here is a
multiple of ``_KDA_CHUNK_SIZE`` and every prompt is a prefix of one chain; the
live wave is ``L(k) = 525 + 313k`` with a 40-token per-member template suffix,
so every tail is RAGGED and no prompt is a prefix of the next. That difference
is load-bearing for a recurrent layer: a packed level call whose earlier row is
ragged starts its next row off the chunk grid. With a chunk grid deliberately
anchored at the packed buffer instead of per sequence, this file's wave passes
- every row here begins on the grid by construction - while
``tests/unit/test_k3_kda_live_wave_levels.py`` (the live shape) fails for every
member that any ragged row precedes, which is all but the first of each group.
Keep both; a clean run here does not exonerate the live shape.


``tests/unit/test_k3_kda_pool_levels.py`` pins the recurrent half of the prefix
pool on small hand-built trees (one two-block segment, and a depth-3 binary
tree). This file runs the same production code over the wave shape the live
corruption appeared in: 64 nested members split across two serve groups by the
real admission balancer, so each group pools a 31-segment GAPPED chain whose
segments span 2, 4, 6 and 33 (group 0) or 31 (group 1) pages, and whose deepest
member keeps a 7-page (group 0) or 3-page (group 1) private tail.

The wave shape comes from ``test_k3_pool_attention_gapped_oracle`` (one source
of truth for the grouping), and the harness - CPU stand-ins for fla's chunk_kda
and the conv kernel, the state pools, the slot allocator - from
``test_k3_kda_pool_levels``. Only the page size differs from the attention
oracle: a pooled segment can resume exactly only on a multiple of fla's chunk
size, so here the page IS ``_KDA_CHUNK_SIZE`` (64), which is also the live page.

What is asserted against an oracle derived from the chain structure alone:

* the LEVEL of every row - a segment's is its own depth in the chain, a
  member tail's is the number of pooled segments its prompt passes through -
  and that ``chunk_levels`` cuts the chunk into exactly those runs;
* the FORK of every row that continues a segment: parent slot is the slot of
  its NEAREST pooled ancestor, child slot is its own, never the same slot, and
  ``has_initial_state`` is set exactly for the rows with a non-empty chain;
* the GATE, ``torch.equal``: a member's pooled segment rows concatenated with
  its tail row equal the same prompt run unsegmented from zero state (the
  default ``pool_levels=None`` shape). Bit-exact, not close: K3's top-16 router
  turns a 1-ulp seam into an O(1) logit difference;
* copy-on-branch: after the wave every segment's slot still holds the state its
  OWN last token produced, for all 31 multi-page segments of both groups;
* the deep multi-page tails explicitly.

Run: PYTHONPATH=. python -m pytest tests/unit/test_k3_kda_gapped_levels.py -q
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

torch.set_num_threads(1)

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, *relative):
    """Load a module by file path, or reuse it if pytest already imported it."""

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


# The KDA harness: SM (serving_modules by file path), the CPU chunk_kda and
# conv stand-ins, the state pools, the slot allocator, the unsegmented run.
_KDA = _load("test_k3_kda_pool_levels", "tests", "unit",
             "test_k3_kda_pool_levels.py")
# The live wave shape: the real balancer's split and the chain-run oracle.
_WAVE = _load("test_k3_pool_attention_gapped_oracle", "tests", "unit",
              "test_k3_pool_attention_gapped_oracle.py")

SM = _KDA.SM
_LEVELS = _KDA._LEVELS
_PLAN = _KDA._PLAN
_POOL = _KDA._POOL
_EXEC = _KDA._EXEC

PAGE = _KDA.PAGE  # == SM._KDA_CHUNK_SIZE == the live page
assert PAGE == SM._KDA_CHUNK_SIZE == 64

PAGES = _WAVE.PAGES  # 32..158 pages, ascending, page aligned
FAMILY = _WAVE.FAMILY
MEMBERS = _WAVE.MEMBERS
GROUP_OF = _WAVE.GROUP_OF
NUM_GROUPS = _WAVE.NUM_GROUPS
LIVE_FLAGGED = _WAVE.LIVE_FLAGGED
DEEP_TAIL_MEMBER = {0: 63, 1: 62}

CHUNK_TOKENS = 1 << 20  # generous: let same_chunk_ancestors pack one chunk
POOL_PAGES = 1024


def block_of(index: int) -> list[int]:
    """One shareable block; its token ids identify that chain block uniquely."""

    return [index * 1000 + offset for offset in range(PAGE)]


# Absolute position -> token id; member k is the first PAGES[k] blocks of it.
CHAIN_TOKENS = [
    token for index in range(max(PAGES)) for token in block_of(index)
]


def member_prompt(member: int) -> list[int]:
    return CHAIN_TOKENS[: PAGES[member] * PAGE]


# --------------------------------------------------------------------------- #
#  The oracle: levels and forks from the chain structure alone
# --------------------------------------------------------------------------- #
def oracle_wave(group: int):
    """(members, prompts, pooled runs in BLOCKS) of one group's wave.

    ``oracle_pooled_runs`` returns block indexes, which are page-size
    independent (a member's shareable block count is ``PAGES[k] - 1`` for any
    page size), so the attention oracle's runs describe this wave too.
    """

    members = MEMBERS[group]
    prompts = [member_prompt(member) for member in members]
    pooled = _WAVE.oracle_pooled_runs(members)
    return members, prompts, pooled


def oracle_chain_length(member: int, pooled) -> int:
    """How many pooled segments this member's prompt passes through."""

    through = (len(member_prompt(member)) - 1) // PAGE
    return sum(1 for _, end, _ in pooled if end <= through)


def oracle_rows(group: int):
    """Per-row oracle: kind -> ref -> (level, parent segment or None, range)."""

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
            depth, depth - 1 if depth else None, start, PAGES[member] * PAGE,
        )
    return levels


# --------------------------------------------------------------------------- #
#  The wave through the real level/fork path
# --------------------------------------------------------------------------- #
def _state_pools(slots: int):
    """The KDA harness's pools, sized for a 63-row wave instead of 32."""

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
        chunk_tokens=CHUNK_TOKENS, same_chunk_ancestors=same_chunk,
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


def run_gapped_wave(group: int, same_chunk: bool) -> dict:
    """Prefill one group's wave level by level; collect rows, states, levels."""

    members, prompts, pooled = oracle_wave(group)
    plan = _plan(prompts, same_chunk)

    # The planner's segments must BE the oracle runs before they index into it.
    assert len(plan.segments) == len(pooled)
    for index, (segment, (start, end, here)) in enumerate(
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
        observed.append({
            "chunk": position, "batch": batch, "row_ids": row_ids,
            "slot_ids": slot_ids, "levels": levels,
            "resumes": has_initial_state,
            "forks": dict(_LEVELS.fork_pairs(batch, sids)),
            "cuts": _LEVELS.chunk_levels(batch),
        })

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
        "outputs": outputs, "state": state, "slots": slots, "sids": sids,
        "observed": observed, "weights": weights, "kwargs": kwargs,
    }


_WAVES: dict[tuple[int, bool], dict] = {}


def gapped_wave(group: int, same_chunk: bool) -> dict:
    key = (group, same_chunk)
    if key not in _WAVES:
        _WAVES[key] = run_gapped_wave(group, same_chunk)
    return _WAVES[key]


# The whole 158-page chain from zero state, once: every member's prompt is a
# prefix of it and ends on a chunk-size multiple, so the reference output of a
# shorter member is the matching slice (test_the_sliced_reference_is_a_real_run
# proves that against independent full runs).
_REFERENCE: dict[str, torch.Tensor] = {}


def reference_output() -> torch.Tensor:
    if "o" not in _REFERENCE:
        weights, kwargs = _KDA._conv_weights(), _KDA._kernel_kwargs()
        out, _ = _KDA._unsegmented(
            _KDA._rows_of(CHAIN_TOKENS), weights, kwargs,
        )
        _REFERENCE["o"] = out
    return _REFERENCE["o"]


# --------------------------------------------------------------------------- #
#  (0) The premise and the reference shortcut
# --------------------------------------------------------------------------- #
def test_the_gapped_waves_have_the_live_multi_page_spans():
    """Guard the premise: spans and deep tails no small tree exercises."""

    for group in range(NUM_GROUPS):
        members, _, pooled = oracle_wave(group)
        spans = sorted({end - start for start, end, _ in pooled})
        assert len(pooled) == 31
        assert spans == [2, 4, 6, 33 if group == 0 else 31], (group, spans)
        tails = {
            member: PAGES[member] - (
                pooled[oracle_chain_length(member, pooled) - 1][1]
                if oracle_chain_length(member, pooled) else 0
            )
            for member in members
        }
        deep = sorted(m for m, pages in tails.items() if pages >= 3)
        assert deep == [DEEP_TAIL_MEMBER[group]], (group, deep)
        assert tails[DEEP_TAIL_MEMBER[group]] == (7 if group == 0 else 3)


@pytest.mark.parametrize(
    "member", sorted(set(LIVE_FLAGGED) | set(DEEP_TAIL_MEMBER.values())),
)
def test_the_sliced_reference_is_a_real_run(member):
    """The sliced whole-chain reference equals an independent full run.

    A member's prompt is a prefix of the chain, it ends on a multiple of the
    chunk size, and the sweep starts from zero state, so the prefix's chunk
    grid is the restriction of the chain's. Checked, not assumed.
    """

    weights, kwargs = _KDA._conv_weights(), _KDA._kernel_kwargs()
    prompt = member_prompt(member)
    own, _ = _KDA._unsegmented(_KDA._rows_of(prompt), weights, kwargs)

    assert torch.equal(own, reference_output()[: len(prompt)]), (
        f"member {member} ({PAGES[member]} pages): the sliced reference is not "
        f"the same run; max_abs="
        f"{(own - reference_output()[:len(prompt)]).abs().max().item():.3e}"
    )


# --------------------------------------------------------------------------- #
#  (i) Level and fork structure against the oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("same_chunk", [True, False])
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_row_levels_and_forks_match_the_chain_oracle(group, same_chunk):
    """Every row's level, fork parent, fork child and resume flag."""

    wave = gapped_wave(group, same_chunk)
    oracle = oracle_rows(group)
    slots, sids = wave["slots"], wave["sids"]
    seen: set[tuple[str, int]] = set()

    for record in wave["observed"]:
        batch, cuts = record["batch"], record["cuts"]
        context = f"group {group} same_chunk={same_chunk} chunk {record['chunk']}"

        # The cut runs must tile the rows, in strictly increasing level order.
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
                # Resume exactly when the member/segment continues a chain.
                resumed = bool(record["resumes"][row]) or row in record["forks"]
                assert resumed == (parent is not None), (
                    f"{where}: resume={resumed}, oracle parent={parent}"
                )
                if parent is None:
                    assert row not in record["forks"], (
                        f"{where}: forks although it has no pooled ancestor"
                    )
                    continue
                # The fork reads the NEAREST pooled ancestor's pseudo sequence.
                assert record["forks"][row] == sids[parent], (
                    f"{where}: forks from sequence {record['forks'][row]}, "
                    f"oracle nearest ancestor is segment {parent} "
                    f"(sequence {sids[parent]})"
                )

        # Per level, the fork tensors address exactly those rows' slots.
        for cut, level in zip(cuts, record["levels"]):
            children = [
                row for row in range(cut.row_start, cut.row_end)
                if row in record["forks"]
            ]
            if not children:
                assert level.fork_parents is None and level.fork_children is None
                continue
            assert level.fork_parents.tolist() == [
                slots.lookup(record["forks"][row]) for row in children
            ], context
            assert level.fork_children.tolist() == [
                slots.lookup(record["row_ids"][row]) for row in children
            ], context
            # A child never writes its parent's slot (copy-on-branch).
            assert not bool((level.fork_parents == level.fork_children).any()), (
                f"{context}: a fork would overwrite its own parent's slot"
            )

    assert seen == set(oracle), "not every planned row went through a level"


# --------------------------------------------------------------------------- #
#  (ii) The gate: pooled rows are bit-exact per member
# --------------------------------------------------------------------------- #
def _gate(wave, group: int, wave_index: int, member: int) -> None:
    plan, outputs = wave["plan"], wave["outputs"]
    chain = [seg for seg in plan.segments if wave_index in seg.members]
    pieces = [outputs[("segment", seg.segment_id)] for seg in chain]
    pieces.append(outputs[("tail", wave_index)])
    got = torch.cat(pieces)
    expected = reference_output()[: PAGES[member] * PAGE]

    assert got.shape == expected.shape, (
        f"group {group} member {member}: pooled rows cover {got.shape[0]} "
        f"tokens, prompt has {expected.shape[0]}"
    )
    assert torch.equal(got, expected), (
        f"group {group} member {member} ({PAGES[member]} pages, chain of "
        f"{len(chain)} segments ending at "
        f"{plan.segments[chain[-1]].token_end if chain else 0}) moved: "
        f"max_abs={(got - expected).abs().max().item():.3e}, first differing "
        f"token {int((got != expected).any(dim=-1).any(dim=-1).nonzero()[0])}"
    )


def _case_id(group: int, member: int) -> str:
    tag = f"g{group}-m{member:02d}"
    if member in LIVE_FLAGGED:
        tag += f"-LIVE-ch{member:03d}"
    if member == DEEP_TAIL_MEMBER[group]:
        tag += "-deeptail"
    return tag


FLAGGED = [
    (GROUP_OF[member], member)
    for member in sorted(set(LIVE_FLAGGED) | set(DEEP_TAIL_MEMBER.values()))
]


@pytest.mark.parametrize("same_chunk", [True, False])
@pytest.mark.parametrize(
    "group,member", FLAGGED,
    ids=[_case_id(group, member) for group, member in FLAGGED],
)
def test_live_flagged_members_resume_their_chain_exactly(
    group, member, same_chunk
):
    """The members the live runs corrupted, plus the deep multi-page tails."""

    wave = gapped_wave(group, same_chunk)
    _gate(wave, group, wave["members"].index(member), member)


@pytest.mark.parametrize("same_chunk", [True, False])
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_every_member_resumes_its_chain_exactly(group, same_chunk):
    """All 32 members; the first failure names the shallowest one."""

    wave = gapped_wave(group, same_chunk)
    for wave_index, member in enumerate(wave["members"]):
        try:
            _gate(wave, group, wave_index, member)
        except AssertionError as error:
            pytest.fail(f"same_chunk={same_chunk}: {error}")


# --------------------------------------------------------------------------- #
#  (iii) Copy-on-branch: a segment's slot still holds its own end state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_every_segment_slot_still_holds_its_own_end_state(group):
    """After the wave, each of the 31 multi-page segments owns its end state.

    That state is what a later chunk's children and the Host commit resume
    from, so a child writing its parent's slot would corrupt every deeper
    member - the shape of the live symptom.
    """

    wave = gapped_wave(group, same_chunk=False)
    weights, kwargs = wave["weights"], wave["kwargs"]
    state, slots, sids = wave["state"], wave["slots"], wave["sids"]

    for segment in wave["plan"].segments:
        prefix = _KDA._rows_of(CHAIN_TOKENS, 0, segment.token_end)
        _, expected = _KDA._unsegmented(prefix, weights, kwargs)
        got = _KDA._final_states(state, slots.lookup(sids[segment.segment_id]))
        for pool, reference in zip(got, _KDA._final_states(expected, 0)):
            assert torch.equal(pool, reference), (
                f"group {group} segment {segment.segment_id} "
                f"[{segment.token_start}, {segment.token_end}) "
                f"({segment.tokens // PAGE} pages) was overwritten by a "
                f"child: max_abs={(pool - reference).abs().max().item():.3e}"
            )
