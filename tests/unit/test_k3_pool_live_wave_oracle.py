# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The pooled wave at the shape the live run ACTUALLY had, not a variant of it.

``tests/unit/test_k3_pool_attention_gapped_oracle.py`` and
``tests/unit/test_k3_kda_gapped_levels.py`` both ran a wave of PAGE-ALIGNED
prompt lengths (an ascending progression of whole pages), nested with no
per-member suffix, which the planner then spreads over many chunks. The
schema-2 kernel dumps of the live two-node run say the wave was none of those
things. Reconstructed from the dumps of all 16 ranks and verified against
them, the live wave is:

* 64 members, ``global_idx`` 64..127, lengths ``L(k) = 525 + 313k`` - exact
  for all 64, NOT a multiple of the 64-token page for any of them;
* a per-member TEMPLATE SUFFIX of exactly 40 tokens, so a member leaves the
  shared chain INSIDE a page: the deepest block a member shares is
  ``min((L - 40) // 64, (L - 1) // 64)``, and 40 is the only suffix length
  consistent with all 62 constrainable members' shared-block counts;
* split over 2 serve groups by the real admission balancer, which reproduces
  the observed per-group membership exactly (pinned below);
* ONE chunk of 63 rows per group (31 pooled segments + 32 member tails),
  because K3 declares group-safe pool prefill, so the worker plans with
  ``same_chunk_ancestors=True`` and orders the chunk by level;
* every tail therefore ends on a PARTIALLY filled pool page, and the shortest
  members' tails are a few tokens past a page boundary.

That shape matters because the live corruption leaks TEMPLATE tokens into the
SHORTEST members of each group (ch-001 and ch-004 in group 0, ch-003, ch-011
and ch-013 in group 1 - members ``64 + N``), and the page-aligned variant has
no template to leak and no row that ends mid-page.

What is asserted here
---------------------
* the premise: the balancer's split IS the live split, and the planner plus
  the executor reproduce the layout the chain structure implies - the row
  table, the page tables, the table width, the distinct pool pages, the query
  token count, the widest row and each member's own token range, every one of
  them derived from ``L(k)``, the suffix and the page size by ``oracle_layout``
  rather than quoted from a run;
* every key every row attends over, walked through the kernel's own
  ``page_table`` for ``[0, cache_seqlens)``, is the token THAT ROW's member
  has at that absolute position. A row that reads a neighbour's template
  suffix fails here and the failure names the row, the member, the offset,
  the pool page and both tokens;
* the semantic gate: a member's pooled segment rows concatenated with its tail
  row equal the same prompt run whole through the same production function.

CPU-only, and the model bits are the smallest that still exercise the MLA
192/128-style split; the wave is ~60x the old one, so the attention reference
is run for a named member set (cost is O(L^2) per member) rather than for all
64. Every live-flagged member is in that set.

Run: PYTHONPATH=. python -m pytest tests/unit/test_k3_pool_live_wave_oracle.py -q
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from batchgen.decode_dp_group import assign_decode_dp_groups

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


# The page-aligned oracle owns the production-module loader, the plain-torch
# FA3 stand-ins and the per-token feature table. Reused verbatim so a verdict
# here cannot differ from it by way of a different reference implementation.
_OLD = _load("test_k3_pool_attention_gapped_oracle", "tests", "unit",
             "test_k3_pool_attention_gapped_oracle.py")
_PLAN, _POOL, _EXEC = _OLD._PLAN, _OLD._POOL, _OLD._EXEC
_LEVELS = _load("test_live_wave_levels_mod", "batchgen", "prefix_reuse",
                "levels.py")
SM = _OLD.SM
_attend = _OLD._attend
_varlen_reference = _OLD._varlen_reference
_hidden = _OLD._hidden
HIDDEN = _OLD.HIDDEN
FakeAttnWrapperBase = _OLD.FakeAttnWrapperBase


# --------------------------------------------------------------------------- #
#  The live wave shape, as the dumps gave it
# --------------------------------------------------------------------------- #
PAGE = 64  # the live serving page and the plan's shareable block
FAMILY = 64
GLOBAL_BASE = 64  # the chain corpus starts at global_idx 64, so ch-N is 64 + N
FIRST = 525
STRIDE = 313
SUFFIX = 40  # the per-member template suffix, derived from the dumps
LIVE_TOKENS = [FIRST + STRIDE * k for k in range(FAMILY)]
NUM_GROUPS = 2

GROUP_OF = assign_decode_dp_groups(LIVE_TOKENS, NUM_GROUPS)
MEMBERS = {
    group: [k for k in range(FAMILY) if GROUP_OF[k] == group]
    for group in range(NUM_GROUPS)
}

# The membership the live dumps show, as global ids: rank 0's and rank 8's
# tail rows in wave-index order. Data, so the balancer is checked not trusted.
LIVE_GROUP_GLOBAL_IDS = {
    0: (65, 66, 68, 71, 73, 74, 76, 79, 81, 82, 84, 87, 89, 90, 92, 95, 97,
        98, 100, 103, 105, 106, 108, 111, 113, 114, 116, 119, 121, 122, 124,
        127),
    1: (64, 67, 69, 70, 72, 75, 77, 78, 80, 83, 85, 86, 88, 91, 93, 94, 96,
        99, 101, 102, 104, 107, 109, 110, 112, 115, 117, 118, 120, 123, 125,
        126),
}

# Members whose output the live pooled runs corrupted, as ch-N.
LIVE_FLAGGED = (1, 3, 4, 11, 13)
# The deepest member of each group: its tail is everything past the point the
# next-shallower member left the chain, so it is the one multi-page tail.
DEEP_TAIL_MEMBER = {0: 63, 1: 62}


CHUNK_TOKENS = 1 << 20  # the live cap is far above the wave; one chunk
POOL_PAGES = 4096


def chain_token(position: int) -> int:
    """The shared chain's token at an absolute position; distinct per position."""

    return position


def template_token(offset: int) -> int:
    """A template token, from an id range the chain never uses."""

    return 10 ** 6 + offset


def member_prompt(member: int) -> list[int]:
    """Member ``member``'s prompt: shared chain, then its template suffix."""

    body = LIVE_TOKENS[member] - SUFFIX
    assert body > 0
    return [chain_token(p) for p in range(body)] + [
        template_token(offset) for offset in range(SUFFIX)
    ]


def shared_blocks(member: int) -> int:
    """Deepest shareable block count: the suffix cuts the chain inside a page.

    Only the first ``(len - 1) // PAGE`` blocks of a prompt are shareable at
    all, and a member stops agreeing with the chain at ``L - SUFFIX``.
    """

    length = LIVE_TOKENS[member]
    return min((length - SUFFIX) // PAGE, (length - 1) // PAGE)


def oracle_pooled_runs(members):
    """Pooled segments of one group's wave, from the lengths alone.

    Block ``j`` is passed through by the members that still agree with the
    chain there, i.e. ``shared_blocks(k) > j``. Maximal runs with the same
    member set are segments, and a run of two or more members is pooled.
    """

    through = {k: shared_blocks(k) for k in members}
    runs: list[list] = []
    block = 0
    while True:
        here = tuple(sorted(k for k in members if through[k] > block))
        if not here:
            break
        if runs and runs[-1][2] == here:
            runs[-1][1] = block + 1
        else:
            runs.append([block, block + 1, here])
        block += 1
    pooled = [run for run in runs if len(run[2]) >= 2]
    assert pooled == runs[:len(pooled)], "pooled set is not ancestor-closed"
    return [(start, end, here) for start, end, here in pooled]


def oracle_wave(group: int):
    """(members, prompts, pooled runs, pooled block count) of one group."""

    members = MEMBERS[group]
    prompts = [member_prompt(member) for member in members]
    pooled = oracle_pooled_runs(members)
    return members, prompts, pooled, (pooled[-1][1] if pooled else 0)


def oracle_chain_tokens(member: int, pooled_blocks: int) -> int:
    """How many tokens of ``member`` come out of the pool."""

    return PAGE * min(shared_blocks(member), pooled_blocks)


def oracle_tail_range(member: int, pooled_blocks: int) -> tuple[int, int]:
    """``(token_start, token_end)`` of a member's own tail row.

    Its start is where the pool stopped carrying the member and its end is the
    member's own length, so the range is a consequence of ``L(k)``, the suffix
    and the page size - never a number read off a run.
    """

    return oracle_chain_tokens(member, pooled_blocks), LIVE_TOKENS[member]


def oracle_layout(group: int) -> dict:
    """The chunk layout one group's wave MUST have, from the lengths alone.

    Every field here is implied by the chain structure, so the executor is
    checked against the shape the inputs force rather than against a
    fingerprint transcribed from one run - which would re-pass unchanged if
    the inputs drifted and would have to be re-measured on a real node to be
    updated at all.

    * a segment row spans its run of blocks and is addressed by every block up
      to its end, so the widest segment row is the pooled block count;
    * a tail row is addressed by the pooled blocks its member passes through
      plus its own pages, so its width follows from its tail length;
    * every pooled block is one pool page and every tail page is another, and
      one chunk allocates each exactly once, so the distinct pages in the
      table are their sum.
    """

    members, _, pooled, pooled_blocks = oracle_wave(group)
    tails = {
        member: LIVE_TOKENS[member] - oracle_chain_tokens(member, pooled_blocks)
        for member in members
    }

    def pages(tokens: int) -> int:
        return -(-tokens // PAGE)

    own_tokens = [PAGE * (end - start) for start, end, _ in pooled]
    own_tokens += [tails[member] for member in members]
    return {
        "rows": len(pooled) + len(members),
        "segments": len(pooled),
        "tails": len(members),
        "table_width": max(
            [end for _, end, _ in pooled]
            + [
                min(shared_blocks(member), pooled_blocks) + pages(tails[member])
                for member in members
            ]
        ),
        "pages": pooled_blocks + sum(pages(tails[member]) for member in members),
        "query_tokens": sum(own_tokens),
        "max_seqlen_q": max(own_tokens),
    }


# --------------------------------------------------------------------------- #
#  One tiny MLA layer, at this wave's scale
# --------------------------------------------------------------------------- #
# The wave is ~60x the page-aligned one (642k keys read per group), so the
# layer is the smallest that still has the MLA split: a K wider than V.
HEADS = 1
NOPE = 2
ROPE = 2  # K is 4 per head ...
V_DIM = 2  # ... V is 2
LORA = 5


def _mla_layer():
    generator = torch.Generator().manual_seed(20261015)

    def linear(fan_in, fan_out):
        layer = torch.nn.Linear(fan_in, fan_out, bias=False)
        with torch.no_grad():
            layer.weight.copy_(
                torch.randn(fan_out, fan_in, generator=generator) * 0.2
            )
        return layer

    attn = SimpleNamespace(
        q_lora_rank=None,
        use_output_gate=False,
        num_heads=HEADS,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        q_head_dim=NOPE + ROPE,
        v_head_dim=V_DIM,
        kv_lora_rank=LORA,
        scaling=(NOPE + ROPE) ** -0.5,
        attn_tp_size=1,
        q_proj=linear(HIDDEN, HEADS * (NOPE + ROPE)),
        kv_a_proj_with_mqa=linear(HIDDEN, LORA + ROPE),
        kv_a_layernorm=torch.nn.LayerNorm(LORA),
        kv_b_proj=linear(LORA, HEADS * (NOPE + V_DIM)),
        o_proj=linear(HEADS * V_DIM, HIDDEN),
    )
    attn.mla_prefill_nope_prepacked = types.MethodType(
        SM.mla_prefill_nope_prepacked, attn
    )
    return attn


# --------------------------------------------------------------------------- #
#  The kernel-argument oracle, per ROW instead of per chain
# --------------------------------------------------------------------------- #
class LiveKeyOracle:
    """Checks every key a row attends over against that row's OWN member.

    The page-aligned oracle could compare every key against one global chain,
    because every prompt was a prefix of it. Here a member leaves the chain
    inside a page and ends with its own template, so the expectation is per
    row: the key at absolute offset ``o`` of row ``r`` must be the token
    ``expected[r][o]``, which the caller derives from the member's prompt.
    This is the check a leaked template token fails.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.shadow: dict[int, tuple[int, int]] = {}
        self.expected: list[tuple] = []
        self.context = ""
        self.rows_checked = 0
        self.keys_checked = 0
        self.failures: list[str] = []

    def arm(self, context: str, expected) -> None:
        self.context = context
        self.expected = list(expected)

    def fail(self, message: str) -> None:
        self.failures.append(f"{self.label} {self.context}: {message}")

    def __call__(self, *, q, k_cache, v_cache, cache_seqlens, page_table,
                 cu_seqlens_q, max_seqlen_q, sinks=None, softmax_scale=None,
                 sliding_window=None):
        assert sinks is None and sliding_window is None
        bounds = cu_seqlens_q.tolist()

        if len(self.expected) != len(bounds) - 1:
            self.fail(
                f"cu_seqlens_q describes {len(bounds) - 1} rows, oracle has "
                f"{len(self.expected)}"
            )
        if int(max_seqlen_q) != max(row[2] for row in self.expected):
            self.fail(
                f"max_seqlen_q {int(max_seqlen_q)} != oracle "
                f"{max(row[2] for row in self.expected)}"
            )
        if q.shape[0] != bounds[-1]:
            self.fail(f"q has {q.shape[0]} rows, cu_seqlens_q ends at {bounds[-1]}")

        for row, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            if row >= len(self.expected):
                break
            start, end, tokens, who, keys = self.expected[row]
            where = f"row {row} ({who}, oracle item [{start}, {end}))"

            if hi - lo != tokens:
                self.fail(f"{where}: query slice is {hi - lo} tokens, oracle {tokens}")
            if int(cache_seqlens[row]) != end:
                self.fail(
                    f"{where}: cache_seqlens {int(cache_seqlens[row])} != oracle "
                    f"context length {end}"
                )

            pages = page_table[row].tolist()
            length = int(cache_seqlens[row])
            if length > len(pages) * PAGE:
                self.fail(
                    f"{where}: cache_seqlens {length} exceeds the {len(pages)} "
                    f"pages of the row's page table"
                )
                length = len(pages) * PAGE
            for offset in range(length):
                slot = pages[offset // PAGE] * PAGE + offset % PAGE
                seen = self.shadow.get(slot)
                if seen is None:
                    self.fail(
                        f"{where}: key {offset} (page index {offset // PAGE}, "
                        f"pool page {pages[offset // PAGE]}, slot {slot}) reads "
                        f"a slot this wave never wrote"
                    )
                    break
                token, position = seen
                if token != keys[offset] or position != offset:
                    self.fail(
                        f"{where}: key {offset} (page index {offset // PAGE}, "
                        f"pool page {pages[offset // PAGE]}, slot {slot}) holds "
                        f"token {token} at position {position}, oracle token "
                        f"{keys[offset]} at position {offset}"
                        + (" - a TEMPLATE token" if token >= 10 ** 6 else "")
                    )
                    break
                self.keys_checked += 1
            self.rows_checked += 1

        out = q.new_empty(q.shape[0], q.shape[1], v_cache.shape[-1])
        for row, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            pages = page_table[row].tolist()
            total = int(cache_seqlens[row])
            keys = torch.cat([k_cache[page] for page in pages])[:total]
            values = torch.cat([v_cache[page] for page in pages])[:total]
            out[lo:hi] = _attend(q[lo:hi], keys, values, softmax_scale)
        return out, None


@pytest.fixture
def harness(monkeypatch):
    """Serve FA3 from plain torch and the lazy executor import from the file."""

    monkeypatch.setitem(sys.modules, "batchgen.prefix_reuse.executor", _EXEC)
    oracle = LiveKeyOracle("unset")
    gqa = types.ModuleType("batchgen.attention.gqa")
    gqa.gqa_extend_fa = oracle
    monkeypatch.setitem(sys.modules, "batchgen.attention.gqa", gqa)
    fa3 = types.ModuleType("flash_attn_interface")
    fa3.flash_attn_varlen_func = _varlen_reference
    monkeypatch.setitem(sys.modules, "flash_attn_interface", fa3)
    FakeAttnWrapperBase.pinned = []
    FakeAttnWrapperBase.tracked = []
    yield oracle
    for name in ("prefill_pool", "prefill_pool_batch", "prefill_pool_row_starts",
                 "prefill_pool_offload_recorder", "cur_batch"):
        setattr(FakeAttnWrapperBase, name, None)


# --------------------------------------------------------------------------- #
#  Plan, lay out and run one group's wave
# --------------------------------------------------------------------------- #
def plan_group(group: int):
    """The live plan of one group: one chunk, level-ordered, 31 segments."""

    members, prompts, pooled, pooled_blocks = oracle_wave(group)
    plan = _PLAN.plan_wave_prefix_sharing(
        prompts, block_tokens=PAGE, pool_pages=POOL_PAGES,
        chunk_tokens=CHUNK_TOKENS, same_chunk_ancestors=True, chunk_keys=None,
    )
    plan = _LEVELS.order_chunk_items_by_level(plan)
    _PLAN.validate_wave_prefix_plan(plan, prompts)
    assert plan.threshold == 2, (
        f"group {group}: the pool budget forced the threshold up, so the "
        f"oracle below does not describe this plan"
    )
    return members, prompts, pooled, pooled_blocks, plan


def _expected_rows(batch, members, prompts, pooled, pooled_blocks):
    """Per row: (token_start, token_end, tokens, label, expected key tokens).

    Derived from the member lengths and the oracle runs, never from the item.
    A segment's keys are those of its first member, which is the whole point
    of pooling it: every member of the segment agrees that far.
    """

    expected = []
    for item in batch.items:
        if item.kind == "tail":
            member = members[item.ref]
            start = oracle_chain_tokens(member, pooled_blocks)
            end = LIVE_TOKENS[member]
            label = f"ch-{member:03d} (global {GLOBAL_BASE + member})"
            keys = prompts[item.ref]
        else:
            run_start, run_end, here = pooled[item.ref]
            start, end = run_start * PAGE, run_end * PAGE
            label = f"segment {item.ref} of {len(here)} members"
            keys = prompts[members.index(here[0])]
        expected.append((start, end, end - start, label, keys))
    return expected


def run_group_wave(group: int, oracle: LiveKeyOracle) -> dict:
    """Run one group's wave through the real wrapper path, checking every key."""

    members, prompts, pooled, pooled_blocks, plan = plan_group(group)
    oracle.label = f"group {group}"

    scratch = _POOL.scratch_pages_bound(
        CHUNK_TOKENS, max(len(p) for p in prompts), len(prompts), PAGE
    )
    pool = _POOL.PrefillPrefixPool(
        num_layers=1, num_pages=plan.pool_pages + scratch, page_tokens=PAGE,
        num_kv_heads=HEADS, head_dim=NOPE + ROPE, dtype=torch.float32,
        device="cpu", v_head_dim=V_DIM,
    )
    # Uninitialised pages would let a stale slot read as a plausible answer.
    pool.k.zero_()
    pool.v.zero_()

    attn = _mla_layer()
    executor = _EXEC.PoolChunkExecutor(plan, prompts, pool)
    rows: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    tails_seen: list[int] = []
    batches = []

    for position, chunk in enumerate(plan.chunks):
        batch = executor.build(chunk)
        batches.append(batch)
        expected = _expected_rows(batch, members, prompts, pooled, pooled_blocks)

        # What the executor packed, against the oracle, before the kernel.
        host_bounds = batch.row_bounds
        for row, item in enumerate(batch.items):
            start, end, tokens, label, keys = expected[row]
            lo, hi = host_bounds[row], host_bounds[row + 1]
            context = (f"group {group} chunk {position} row {row} {item.kind} "
                       f"{item.ref} [{label}]")
            assert (item.token_start, item.token_end) == (start, end), (
                f"{context}: item [{item.token_start}, {item.token_end}) != "
                f"oracle [{start}, {end})"
            )
            assert hi - lo == tokens, context
            assert batch.input_ids[lo:hi].tolist() == keys[start:end], (
                f"{context}: packed tokens are not the member's"
            )
            assert batch.position_ids[lo:hi].tolist() == list(range(start, end)), (
                f"{context}: position ids do not start at {start}"
            )

        # Mirror the chunk's pool write, then let the kernel read it back.
        for slot, token, pos in zip(
            batch.slots.tolist(), batch.input_ids.tolist(),
            batch.position_ids.tolist(),
        ):
            oracle.shadow[slot] = (int(token), int(pos))
        oracle.arm(f"chunk {position}", expected)

        attn_output, offload_kv = attn.mla_prefill_nope_prepacked(
            _hidden(batch.position_ids, batch.input_ids),
            batch.position_ids,
            batch.cu_seqlens_q,
            batch.max_seqlen_q,
            len(batch.items),
            pool=pool,
            pool_batch=batch,
            pool_layer=0,
        )
        for row, item in enumerate(batch.items):
            lo, hi = batch.row_bounds[row], batch.row_bounds[row + 1]
            rows[(item.kind, item.ref)] = (attn_output[lo:hi], offload_kv[lo:hi])
        tails_seen.extend(batch.tail_wave_indices)

        released_slots = []
        for segment_id in chunk.release_after:
            for page in executor.segment_pages[segment_id]:
                released_slots.extend(
                    page * PAGE + offset for offset in range(PAGE)
                )
        for row, item in enumerate(batch.items):
            if item.kind == "tail":
                lo, hi = batch.row_bounds[row], batch.row_bounds[row + 1]
                released_slots.extend(batch.slots[lo:hi].tolist())
        executor.finish(chunk)
        for slot in released_slots:
            oracle.shadow.pop(int(slot), None)

    executor.close()
    assert sorted(tails_seen) == list(range(len(members)))

    return {
        "members": members, "prompts": prompts, "plan": plan, "pooled": pooled,
        "pooled_blocks": pooled_blocks, "rows": rows, "attn": attn,
        "batches": batches, "rows_checked": oracle.rows_checked,
        "keys_checked": oracle.keys_checked, "failures": list(oracle.failures),
    }


_WAVES: dict[int, dict] = {}


def group_wave(group: int, oracle: LiveKeyOracle) -> dict:
    """One group's wave, run once per session and cached by group."""

    if group not in _WAVES:
        _WAVES[group] = run_group_wave(group, oracle)
    wave = _WAVES[group]
    assert not wave["failures"], (
        f"{len(wave['failures'])} kernel-argument/key mismatches in group "
        f"{group}; first:\n" + "\n".join(wave["failures"][:5])
    )
    return wave


def whole_prompt(attn, prompt):
    """The prompt as one packed sequence: the default (no pool) path."""

    return attn.mla_prefill_nope_prepacked(
        _hidden(range(len(prompt)), prompt),
        None,
        torch.tensor([0, len(prompt)], dtype=torch.int32),
        len(prompt),
        1,
    )


# --------------------------------------------------------------------------- #
#  (i) The premise: this really is the live wave
# --------------------------------------------------------------------------- #
def test_the_real_balancer_reproduces_the_live_membership():
    """The split is the balancer's, and it is the one the dumps show."""

    assert sorted(len(MEMBERS[g]) for g in range(NUM_GROUPS)) == [32, 32]
    for group in range(NUM_GROUPS):
        assert tuple(
            GLOBAL_BASE + member for member in MEMBERS[group]
        ) == LIVE_GROUP_GLOBAL_IDS[group], group
    # ... and the live-flagged members land in the groups the dumps show.
    assert [GROUP_OF[ch] for ch in LIVE_FLAGGED] == [0, 1, 0, 1, 1]


def test_the_lengths_are_ragged_and_ch011_is_the_only_aligned_one():
    """The property the page-aligned oracle could not have.

    63 of the 64 lengths are off a page boundary, and the one exception is
    ch-011 (3968 == 62 pages) - which is both a live-corrupted member and the
    only row in the whole live dump set whose own tokens fill a page exactly.
    """

    aligned = [member for member in range(FAMILY) if LIVE_TOKENS[member] % PAGE == 0]
    assert aligned == [11] and LIVE_TOKENS[11] == 3968
    assert 11 in LIVE_FLAGGED


def test_every_member_leaves_the_chain_inside_a_block():
    """The suffix is the divergence, so a tail is never just the last block."""

    for member in range(FAMILY):
        length = LIVE_TOKENS[member]
        assert shared_blocks(member) == min(
            (length - SUFFIX) // PAGE, (length - 1) // PAGE
        )
        # The tail carries the whole template plus the rest of the block the
        # template started in, so it is at least the suffix long.
        tail = length - PAGE * shared_blocks(member)
        assert SUFFIX <= tail <= SUFFIX + PAGE - 1, (member, tail)


# The segment page spans the live ranks dumped, per group: the gapped chain's
# own fingerprint, which no page-aligned variant produces.
LIVE_SEGMENT_SPANS = {
    0: (12, 5, 10, 14, 10, 5, 10, 14, 10, 5, 10, 15, 9, 5, 10, 15, 9, 5, 10,
        15, 10, 4, 10, 15, 10, 5, 9, 15, 10, 5, 10),
    1: (7, 15, 10, 4, 10, 15, 10, 5, 9, 15, 10, 5, 9, 15, 10, 5, 10, 14, 10,
        5, 10, 14, 10, 5, 10, 15, 9, 5, 10, 15, 9),
}


@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_each_group_is_the_31_segment_gapped_chain_the_ranks_dumped(group):
    """31 pooled segments per group, span by span as dumped."""

    members, _, pooled, pooled_blocks = oracle_wave(group)
    assert len(pooled) == 31, (group, len(pooled))
    assert tuple(end - start for start, end, _ in pooled) == \
        LIVE_SEGMENT_SPANS[group]
    tails = {
        member: LIVE_TOKENS[member] - oracle_chain_tokens(member, pooled_blocks)
        for member in members
    }
    deep = sorted(m for m, tokens in tails.items() if tokens > 3 * PAGE)
    assert deep == [DEEP_TAIL_MEMBER[group]], (group, deep)
    assert tails[DEEP_TAIL_MEMBER[group]] == (980 if group == 0 else 411)


@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_the_layout_is_the_one_the_chain_structure_forces(group):
    """The executor's chunk is the chunk the lengths imply, field by field."""

    members, prompts, pooled, pooled_blocks, plan = plan_group(group)
    assert len(plan.chunks) == 1, (
        f"group {group}: {len(plan.chunks)} chunks; the live wave is one"
    )
    scratch = _POOL.scratch_pages_bound(
        CHUNK_TOKENS, max(len(p) for p in prompts), len(prompts), PAGE
    )
    pool = _POOL.PrefillPrefixPool(
        num_layers=1, num_pages=plan.pool_pages + scratch, page_tokens=PAGE,
        num_kv_heads=1, head_dim=1, dtype=torch.uint8, device="cpu",
    )
    batch = _EXEC.PoolChunkExecutor(plan, prompts, pool).build(plan.chunks[0])
    expected = oracle_layout(group)
    table = batch.page_table.tolist()

    assert len(batch.items) == expected["rows"]
    assert sum(1 for i in batch.items if i.kind == "segment") == \
        expected["segments"]
    assert sum(1 for i in batch.items if i.kind == "tail") == \
        expected["tails"]
    assert len(table[0]) == expected["table_width"]
    assert len({page for row in table for page in row}) == expected["pages"]
    assert int(batch.cu_seqlens_q[-1]) == expected["query_tokens"]
    assert batch.max_seqlen_q == expected["max_seqlen_q"]
    # Rows are grouped by level, as the live level-wise order requires.
    _LEVELS.chunk_levels(batch)
    # Every member's own token range, the five live-corrupted ones included,
    # is where the pool stopped carrying it through to its own length.
    checked = 0
    for row, item in enumerate(batch.items):
        if item.kind != "tail":
            continue
        member = members[item.ref]
        assert (item.token_start, item.token_end) == \
            oracle_tail_range(member, pooled_blocks), (member, row)
        checked += 1
    assert checked == expected["tails"]
    # The flagged members really are in this wave, so the ranges above cover
    # the rows the live corruption appeared on.
    assert [m for m in LIVE_FLAGGED if GROUP_OF[m] == group]


# --------------------------------------------------------------------------- #
#  (ii) Every key of every row is that row's member's token
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_every_key_every_row_reads_is_its_own_members_token(group, harness):
    """A leaked template token fails here, named."""

    wave = group_wave(group, harness)

    assert wave["rows_checked"] == (
        len(wave["plan"].segments) + len(wave["members"])
    ), wave["rows_checked"]
    assert set(wave["rows"]) == (
        {("segment", seg.segment_id) for seg in wave["plan"].segments}
        | {("tail", index) for index in range(len(wave["members"]))}
    )
    # Every key of every row was compared, not just the row count.
    assert wave["keys_checked"] == sum(
        int(length) for batch in wave["batches"]
        for length in batch.cache_seqlens.tolist()
    )


def test_the_key_oracle_names_a_leaked_template_token():
    """The diagnostic itself: what a template leak would read as.

    Driven directly, because a leak has to be injected to be seen: one row,
    one page, and the pool slot holding a TEMPLATE token where the member's
    own chain token belongs. Without this the clean verdict above would rest
    on an unexercised failure path.
    """

    oracle = LiveKeyOracle("control")
    page = 7
    # Slot 0 of page 7 holds a template token instead of chain token 0.
    oracle.shadow[page * PAGE] = (template_token(0), 0)
    oracle.arm("injected", [(0, 1, 1, "ch-001 (global 65)", [chain_token(0)])])
    oracle(
        q=torch.zeros(1, HEADS, NOPE + ROPE),
        k_cache=torch.zeros(page + 1, PAGE, HEADS, NOPE + ROPE),
        v_cache=torch.zeros(page + 1, PAGE, HEADS, V_DIM),
        cache_seqlens=torch.tensor([1], dtype=torch.int32),
        page_table=torch.tensor([[page]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        max_seqlen_q=1, softmax_scale=0.5,
    )

    assert len(oracle.failures) == 1, oracle.failures
    message = oracle.failures[0]
    for fragment in ("row 0", "ch-001 (global 65)", f"pool page {page}",
                     f"slot {page * PAGE}", "TEMPLATE token"):
        assert fragment in message, (fragment, message)


# --------------------------------------------------------------------------- #
#  (iii) The semantic gate: pooled rows == the whole prompt
# --------------------------------------------------------------------------- #
def _gate(wave, index: int):
    """Pooled segment rows plus the tail row, against the whole-prompt run."""

    plan, prompts = wave["plan"], wave["prompts"]
    chain = [seg for seg in plan.segments if index in seg.members]
    pieces = [wave["rows"][("segment", seg.segment_id)] for seg in chain]
    pieces.append(wave["rows"][("tail", index)])
    expected_output, expected_latent = whole_prompt(wave["attn"], prompts[index])
    got_output = torch.cat([output for output, _ in pieces])
    got_latent = torch.cat([latent for _, latent in pieces])
    assert got_output.shape == expected_output.shape
    torch.testing.assert_close(got_output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_latent, expected_latent, rtol=1e-5, atol=1e-5)


# The gate costs O(L^2) per member and the deepest member is 20244 tokens, so
# the whole-prompt reference would need a 1.6 GB score matrix. It therefore
# runs for every member up to ch-013 - which is all five live-flagged members
# and nine controls, the whole short end the corruption prefers - plus one
# mid-chain member per group. Key-level coverage is NOT sampled: test (ii)
# walks every key of all 63 rows of both groups.
GATE_MEMBERS = list(range(14)) + [20, 21]
assert set(LIVE_FLAGGED) <= set(GATE_MEMBERS)


def _case_id(member: int) -> str:
    tag = f"g{GROUP_OF[member]}-ch{member:03d}"
    if member in LIVE_FLAGGED:
        tag += "-LIVE"
    return tag


@pytest.mark.parametrize(
    "member", GATE_MEMBERS, ids=[_case_id(m) for m in GATE_MEMBERS],
)
def test_the_live_corrupted_members_read_the_right_prefix(member, harness):
    """The members the live run corrupted, at the live shape, plus controls."""

    wave = group_wave(GROUP_OF[member], harness)
    _gate(wave, wave["members"].index(member))
