# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""A page-aligned two-group gapped chain wave, down to the pool attention args.

CORRECTION (2026-10-15, from the schema-2 kernel dumps of the live two-node
run): the wave shape below is NOT the one the live corruption appeared in. It
uses page-aligned prompt lengths (32..158 whole pages) nested in one chain,
which the planner then spreads over 32 chunks. The live wave is
``L(k) = 525 + 313k`` with a 40-token per-member TEMPLATE SUFFIX, so a member
leaves the chain INSIDE a page and every tail ends on a partially filled one,
and the whole wave is ONE level-ordered chunk per group.
``tests/unit/test_k3_pool_live_wave_oracle.py`` is that shape, verified
against the dumps; this file remains a valid - and cheaper - exercise of the
same code on a page-aligned wave, but a clean run here does NOT exonerate the
live shape. Measured: a chunk-grid defect that the true shape catches on 62 of
64 members is invisible to this one on all 64.


Round 1 (``tests/unit/test_prefix_gapped_chain_plan.py``) cleared the planner
and the executor layout; the Host-side offload bytes were then verified on the
node. What is left is the READ side: the arguments the model wrapper hands the
pooled prefix attention. This file drives the REAL
``mla_prefill_nope_prepacked`` pool branch (through the harness of
``tests/unit/test_k3_mla_pool_attention.py``) over the live wave shape and
checks every kernel argument against an oracle.

Wave shape
----------
The chain corpus is 64 nested members, lengths an ascending page-aligned
arithmetic progression (32 to 158 pages, i.e. 2048 to 10112 tokens at the live
64-token page). ``assign_decode_dp_groups`` - the real admission balancer,
imported, not reimplemented - splits them over 2 serve groups by L^2 load, and
each group's 32 members become one pooled wave with a GAPPED chain.

The split is scale invariant (FFD on the least-loaded group compares sums of
squares, so a common factor on every length cannot change a decision), which
``test_live_group_split_is_the_real_balancer_s`` pins. That lets the local
prompts use the harness's 8-token page while keeping the PAGE COUNT structure -
segment strides, chain depth, tail page counts - bit-identical to the live
64-token-page wave. The defect class under test is page-table and offset
arithmetic, which is indexed in pages, so the page count is what has to match,
not the page size.

What is asserted, per group, per chunk, per row, against the oracle
-------------------------------------------------------------------
``gqa_extend_fa`` is intercepted at the point the kernel consumes its
arguments, and a shadow of the pool records which (token, absolute position)
the chunk wrote into each slot:

* every key the row attends over, walked through ``page_table`` for
  ``[0, cache_seqlens)``, is the oracle token at the oracle position - so a
  misaligned or wrong prefix is caught key by key;
* ``cache_seqlens`` equals the row's true context length;
* ``cu_seqlens_q`` row slices, ``max_seqlen_q`` and the ``position_ids`` /
  ``input_ids`` the wrapper is fed equal the oracle item ranges, so a query row
  starting at the wrong ``token_start`` is caught;
* no row reads a slot the wave never wrote.

Then the semantic gate: a member's pooled segment rows concatenated with its
tail row must equal running the same production function over its whole prompt
as one packed sequence - the check that actually fails if the model attends the
wrong prefix. Run for every member of both groups, and separately for the
members flagged in the live runs plus the deep multi-page tails, so a failing
test id names them.

CPU-only, no ``import batchgen`` for the model modules: the package
``__init__`` JIT-builds a CUDA extension, so they are loaded by file path.

Run: PYTHONPATH=. python -m pytest tests/unit/test_k3_pool_attention_gapped_oracle.py -q
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# The balancer is pure (no torch, no worker), so the real module imports fine.
from batchgen.decode_dp_group import assign_decode_dp_groups

torch.set_num_threads(1)

_ROOT = Path(__file__).resolve().parents[2]
_KL_DIR = _ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"


def _load(name, *relative):
    path = _ROOT.joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Distinct alias names: test_k3_mla_pool_attention.py loads the same files under
# its own names, and the two must not clobber each other in one session.
_PLAN = _load("test_gapped_oracle_plan_mod", "batchgen", "prefix_reuse", "wave_plan.py")
_POOL = _load("test_gapped_oracle_pool_mod", "batchgen", "prefix_reuse", "pool.py")
_EXEC = _load("test_gapped_oracle_exec_mod", "batchgen", "prefix_reuse", "executor.py")


# --------------------------------------------------------------------------- #
#  The live wave shape
# --------------------------------------------------------------------------- #
LIVE_PAGE = 64  # the serving page / shareable block
FAMILY = 64  # nested members
# Ascending page-aligned arithmetic progression, deepest member 158 pages.
PAGES = [32 + 2 * k for k in range(FAMILY)]
LIVE_TOKENS = [pages * LIVE_PAGE for pages in PAGES]
NUM_GROUPS = 2

# The real balancer decides the split; nothing here second-guesses it.
GROUP_OF = assign_decode_dp_groups(LIVE_TOKENS, NUM_GROUPS)
MEMBERS = {
    group: [k for k in range(FAMILY) if GROUP_OF[k] == group]
    for group in range(NUM_GROUPS)
}

# Members whose output was corrupted in the live pooled runs, read as corpus
# member indexes (ch-001 -> member 1). The corpus naming is not recorded in the
# case file beyond these labels, so every member of both groups is checked
# anyway and the verdict does not depend on this reading.
LIVE_FLAGGED = (1, 3, 4, 11, 13)


# --------------------------------------------------------------------------- #
#  Model scale: the harness's tiny MLA layer and its 8-token page
# --------------------------------------------------------------------------- #
HIDDEN = 12
HEADS = 2
NOPE = 4
ROPE = 2  # K is 6 per head
V_DIM = 3  # ... V is 3: the 192/128 split, at test scale
LORA = 5
PAGE = 8  # pool page and the plan's shareable block

POOL_PAGES = 1024  # generous: every shared segment must be poolable
CHUNK_TOKENS = 4096


def block_of(index: int) -> list[int]:
    """One shareable block; its token ids identify that chain block uniquely."""

    return [index * 1000 + offset for offset in range(PAGE)]


# Absolute position -> token id. Member k is ORACLE_TOKEN[:PAGES[k] * PAGE],
# because every member is a prefix of the full chain.
ORACLE_TOKEN = [
    token for index in range(max(PAGES)) for token in block_of(index)
]


def member_prompt(member: int) -> list[int]:
    return ORACLE_TOKEN[: PAGES[member] * PAGE]


def oracle_pooled_runs(members):
    """Pooled segments of a wave, from the lengths alone.

    The last block of a prompt stays private, so member ``k`` passes through
    blocks ``[0, PAGES[k] - 1)``. Blocks carrying the same member set form one
    segment; a run shared by two or more members is pooled, and because counts
    never increase down the chain the pooled set is a prefix of it.
    """

    through = {k: (len(member_prompt(k)) - 1) // PAGE for k in members}
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
    assert pooled == runs[: len(pooled)], "pooled set is not ancestor-closed"
    return [(start, end, here) for start, end, here in pooled]


def oracle_wave(group: int):
    """(members, prompts, pooled runs, pooled block count) of one group."""

    members = MEMBERS[group]
    prompts = [member_prompt(member) for member in members]
    pooled = oracle_pooled_runs(members)
    return members, prompts, pooled, (pooled[-1][1] if pooled else 0)


def oracle_chain_tokens(member: int, pooled_blocks: int) -> int:
    through = (len(member_prompt(member)) - 1) // PAGE
    return PAGE * min(through, pooled_blocks)


# --------------------------------------------------------------------------- #
#  The model modules, loaded without the batchgen package
# --------------------------------------------------------------------------- #
class FakeAttnWrapperBase:
    """The class-variable seam ``wrappers.py`` reads for a prefill offload."""

    prefill_pool = None
    prefill_pool_batch = None
    prefill_pool_row_starts = None
    prefill_pool_offload_recorder = None
    prefill_pool_layer_map = None
    cur_batch = None
    prepack_num_sequences = None
    prepack_seq_lengths = None
    pinned: list = []
    tracked: list = []

    @classmethod
    def pin_prefill_offload_tensor(cls, tensor, layer_idx):
        cls.pinned.append((layer_idx, tensor))

    @classmethod
    def track_prefill_offload_task(cls, task, layer_idx):
        cls.tracked.append((layer_idx, task))


def _load_kimi_linear():
    """``serving_modules`` and ``wrappers`` as one alias package."""

    package = types.ModuleType("_kl_gapped_oracle")
    package.__path__ = [str(_KL_DIR)]
    sys.modules["_kl_gapped_oracle"] = package
    base = types.ModuleType("batchgen.models.wrappers")
    base.AttnWrapperBase = FakeAttnWrapperBase
    expert = types.ModuleType("batchgen.models.wrappers.expert")
    expert.ExpertWrapperBase = type("ExpertWrapperBase", (), {})
    previous = {
        name: sys.modules.get(name) for name in (base.__name__, expert.__name__)
    }
    sys.modules[base.__name__] = base
    sys.modules[expert.__name__] = expert
    try:
        return (
            importlib.import_module("_kl_gapped_oracle.serving_modules"),
            importlib.import_module("_kl_gapped_oracle.wrappers"),
        )
    finally:
        for name, module in previous.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


SM, WR = _load_kimi_linear()


# --------------------------------------------------------------------------- #
#  Plain-torch stand-ins for the two FA3 entry points
# --------------------------------------------------------------------------- #
def _attend(q, keys, values, scale):
    """Causal attention of ``q`` to ``keys``/``values``, in fp32.

    ``q`` holds the LAST ``q.shape[0]`` positions of the cached sequence, which
    is how FA3 aligns a varlen query block against a longer K cache.
    """

    total, queries = keys.shape[0], q.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) * scale
    future = (
        torch.arange(total)[None, :]
        > torch.arange(queries)[:, None] + (total - queries)
    )
    return torch.einsum(
        "hqk,khd->qhd",
        scores.masked_fill(future[None], float("-inf")).softmax(-1),
        values.float(),
    ).to(q.dtype)


def _varlen_reference(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                      max_seqlen_k, softmax_scale, causal):
    assert causal and torch.equal(cu_seqlens_q, cu_seqlens_k)
    bounds = cu_seqlens_q.tolist()
    out = q.new_empty(q.shape[0], q.shape[1], v.shape[2])
    for lo, hi in zip(bounds, bounds[1:]):
        out[lo:hi] = _attend(q[lo:hi], k[lo:hi], v[lo:hi], softmax_scale)
    return out


class KernelArgOracle:
    """Checks the pooled-attention arguments where the kernel consumes them.

    ``shadow`` mirrors the pool: the (token, absolute position) each chunk's
    write placed in every slot. Reading it back through the kernel's own
    ``page_table`` reconstructs exactly the key sequence the kernel will see.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.shadow: dict[int, tuple[int, int]] = {}
        self.expected: list[tuple[int, int, int]] = []  # per row
        self.context = ""
        self.rows_checked = 0
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
        if int(max_seqlen_q) != max(tokens for _, _, tokens in self.expected):
            self.fail(
                f"max_seqlen_q {int(max_seqlen_q)} != oracle "
                f"{max(tokens for _, _, tokens in self.expected)}"
            )
        if q.shape[0] != bounds[-1]:
            self.fail(f"q has {q.shape[0]} rows, cu_seqlens_q ends at {bounds[-1]}")

        for row, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            if row >= len(self.expected):
                break
            start, end, tokens = self.expected[row]
            where = f"row {row} (oracle item [{start}, {end}))"

            if hi - lo != tokens:
                self.fail(f"{where}: query slice is {hi - lo} tokens, oracle {tokens}")
            if int(cache_seqlens[row]) != end:
                self.fail(
                    f"{where}: cache_seqlens {int(cache_seqlens[row])} != oracle "
                    f"context length {end}"
                )

            # Walk the row's page table over its whole context and compare every
            # key against the oracle token at that absolute position.
            pages = page_table[row].tolist()
            length = int(cache_seqlens[row])
            if length > len(pages) * PAGE:
                self.fail(
                    f"{where}: cache_seqlens {length} exceeds the "
                    f"{len(pages)} pages of the row's page table"
                )
                length = len(pages) * PAGE
            for offset in range(length):
                slot = pages[offset // PAGE] * PAGE + offset % PAGE
                seen = self.shadow.get(slot)
                if seen is None:
                    self.fail(
                        f"{where}: key {offset} (page index {offset // PAGE}, "
                        f"pool page {pages[offset // PAGE]}) reads slot {slot}, "
                        f"which this wave never wrote"
                    )
                    break
                token, position = seen
                if token != ORACLE_TOKEN[offset] or position != offset:
                    self.fail(
                        f"{where}: key {offset} (page index {offset // PAGE}, "
                        f"pool page {pages[offset // PAGE]}) holds token "
                        f"{token} at position {position}, oracle token "
                        f"{ORACLE_TOKEN[offset]} at position {offset}"
                    )
                    break
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
    oracle = KernelArgOracle("unset")
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
#  One tiny MLA layer
# --------------------------------------------------------------------------- #
_FEATURES: dict[tuple[int, int], torch.Tensor] = {}


def _feature(position: int, token: int) -> torch.Tensor:
    """Activations of one token, shared by every member passing through it."""

    key = (position, token)
    row = _FEATURES.get(key)
    if row is None:
        generator = torch.Generator().manual_seed(
            (position * 1000003 + token) % (2 ** 31)
        )
        row = torch.randn(HIDDEN, generator=generator)
        _FEATURES[key] = row
    return row


def _hidden(positions, tokens) -> torch.Tensor:
    return torch.stack([
        _feature(int(position), int(token))
        for position, token in zip(positions, tokens)
    ])


def _mla_layer():
    generator = torch.Generator().manual_seed(20261001)

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
#  Driving one group's wave through the real wrapper path
# --------------------------------------------------------------------------- #
_WAVES: dict[int, dict] = {}


def run_group_wave(group: int, oracle: KernelArgOracle) -> dict:
    """Plan, lay out and run one group's wave; check every kernel argument."""

    members, prompts, pooled, pooled_blocks = oracle_wave(group)
    oracle.label = f"group {group}"

    plan = _PLAN.plan_wave_prefix_sharing(
        prompts, block_tokens=PAGE, pool_pages=POOL_PAGES,
        chunk_tokens=CHUNK_TOKENS,
    )
    _PLAN.validate_wave_prefix_plan(plan, prompts)
    assert plan.threshold == 2, (
        f"group {group}: the pool budget forced the threshold up, so the "
        f"oracle below does not describe this plan"
    )

    # The planner's segments must BE the oracle runs before they are used as an
    # index into them.
    assert len(plan.segments) == len(pooled), (
        f"group {group}: {len(plan.segments)} segments, oracle {len(pooled)}"
    )
    wave_index = {member: index for index, member in enumerate(members)}
    for position, (segment, (start, end, here)) in enumerate(
        zip(plan.segments, pooled)
    ):
        assert segment.segment_id == position
        assert segment.parent_id == (None if position == 0 else position - 1)
        assert (segment.token_start, segment.token_end) == (
            start * PAGE, end * PAGE,
        ), (
            f"group {group} segment {position}: planner "
            f"[{segment.token_start}, {segment.token_end}) != oracle "
            f"[{start * PAGE}, {end * PAGE})"
        )
        assert segment.members == tuple(wave_index[m] for m in here)

    scratch = _POOL.scratch_pages_bound(
        CHUNK_TOKENS, max(len(prompt) for prompt in prompts), len(prompts), PAGE
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

    for position, chunk in enumerate(plan.chunks):
        batch = executor.build(chunk)

        # Oracle item ranges for this chunk's rows, derived from the member
        # lengths and the oracle runs - never from the item itself.
        expected = []
        for item in batch.items:
            if item.kind == "tail":
                member = members[item.ref]
                start = oracle_chain_tokens(member, pooled_blocks)
                end = PAGES[member] * PAGE
            else:
                run_start, run_end, _ = pooled[item.ref]
                start, end = run_start * PAGE, run_end * PAGE
            expected.append((start, end, end - start))

        # The wrapper is fed exactly what the executor packed; check that
        # against the oracle before the kernel ever sees it.
        host_bounds = batch.row_bounds
        for row, item in enumerate(batch.items):
            start, end, tokens = expected[row]
            lo, hi = host_bounds[row], host_bounds[row + 1]
            context = f"group {group} chunk {position} row {row} {item.kind} {item.ref}"
            assert (item.token_start, item.token_end) == (start, end), (
                f"{context}: item [{item.token_start}, {item.token_end}) != "
                f"oracle [{start}, {end})"
            )
            assert hi - lo == tokens, context
            assert batch.input_ids[lo:hi].tolist() == ORACLE_TOKEN[start:end], (
                f"{context}: packed tokens are not the oracle's"
            )
            assert batch.position_ids[lo:hi].tolist() == list(range(start, end)), (
                f"{context}: position ids do not start at the oracle token_start "
                f"{start}"
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

        # A released segment's pages go back, so the shadow of those slots must
        # go with them: a later read of one is a stale read, not a lucky hit.
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
        "members": members, "prompts": prompts, "plan": plan,
        "pooled": pooled, "pooled_blocks": pooled_blocks, "rows": rows,
        "attn": attn, "chunks": len(plan.chunks),
        # Kept on the wave, so the argument verdict survives the result cache.
        "rows_checked": oracle.rows_checked, "failures": list(oracle.failures),
    }


def group_wave(group: int, oracle: KernelArgOracle) -> dict:
    """One group's wave, run once per session and cached by group."""

    if group not in _WAVES:
        _WAVES[group] = run_group_wave(group, oracle)
    wave = _WAVES[group]
    assert not wave["failures"], (
        f"{len(wave['failures'])} kernel-argument mismatches in group "
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
#  (i) The wave shape really is the live one
# --------------------------------------------------------------------------- #
def test_live_group_split_is_the_real_balancer_s():
    """The split comes from ``assign_decode_dp_groups`` and is scale invariant."""

    assert sorted(len(MEMBERS[group]) for group in range(NUM_GROUPS)) == [32, 32]
    assert sorted(MEMBERS[0] + MEMBERS[1]) == list(range(FAMILY))
    # L^2 FFD compares sums of squares, so a common factor on every length
    # cannot change a placement: the local 8-token page sees the same split.
    assert assign_decode_dp_groups(PAGES, NUM_GROUPS) == GROUP_OF
    assert assign_decode_dp_groups(
        [pages * PAGE for pages in PAGES], NUM_GROUPS
    ) == GROUP_OF
    # Not the alternating split a hand-made "every other member" repro assumes.
    assert MEMBERS[0] != list(range(1, FAMILY, 2))


def test_each_group_is_a_gapped_chain_with_multi_page_strides():
    """Guard the premise: both waves carry strides no single-page repro has."""

    for group in range(NUM_GROUPS):
        members, _, pooled, pooled_blocks = oracle_wave(group)
        strides = sorted({end - start for start, end, _ in pooled})
        assert len(pooled) == 31
        assert strides == [2, 4, 6, 33 if group == 0 else 31], (group, strides)
        # Every member's nearest ancestors are partly in the other group, so
        # the chain is gapped rather than one block per member.
        assert len(pooled) < len(members) * 2
        tails = {
            member: PAGES[member] * PAGE
            - oracle_chain_tokens(member, pooled_blocks)
            for member in members
        }
        deep = sorted(m for m, tokens in tails.items() if tokens >= 3 * PAGE)
        assert deep == [63 if group == 0 else 62], (group, deep)


# --------------------------------------------------------------------------- #
#  (ii) Every kernel argument of every row matches the oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_pool_attention_kernel_args_match_the_oracle(group, harness):
    """Each row's page table, cache_seqlens, query slice and position ids."""

    wave = group_wave(group, harness)

    # One checked kernel row per pooled segment and per member tail.
    assert wave["rows_checked"] == (
        len(wave["plan"].segments) + len(wave["members"])
    ), wave["rows_checked"]
    # Every pooled segment and every member tail went through the kernel.
    assert set(wave["rows"]) == (
        {("segment", seg.segment_id) for seg in wave["plan"].segments}
        | {("tail", index) for index in range(len(wave["members"]))}
    )


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
    torch.testing.assert_close(
        got_output, expected_output, rtol=1e-5, atol=1e-5,
    )
    torch.testing.assert_close(
        got_latent, expected_latent, rtol=1e-5, atol=1e-5,
    )


def _case_id(group: int, member: int) -> str:
    tag = f"g{group}-m{member:02d}"
    if member in LIVE_FLAGGED:
        tag += f"-LIVE-ch{member:03d}"
    if member in (62, 63):
        tag += "-deeptail"
    return tag


FLAGGED_CASES = [
    (GROUP_OF[member], member)
    for member in sorted(set(LIVE_FLAGGED) | {62, 63})
]


@pytest.mark.parametrize(
    "group,member", FLAGGED_CASES,
    ids=[_case_id(group, member) for group, member in FLAGGED_CASES],
)
def test_live_corrupted_members_read_the_right_prefix(group, member, harness):
    """The members the live pooled runs corrupted, plus the deep multi-page tails."""

    wave = group_wave(group, harness)
    _gate(wave, wave["members"].index(member))


@pytest.mark.parametrize("group", range(NUM_GROUPS))
def test_every_member_of_the_group_reads_the_right_prefix(group, harness):
    """All 32 members; the first failure names the shallowest one."""

    wave = group_wave(group, harness)
    for index, member in enumerate(wave["members"]):
        try:
            _gate(wave, index)
        except AssertionError as error:
            pytest.fail(
                f"group {group} member {member} (wave index {index}, "
                f"{PAGES[member]} pages, chain "
                f"{oracle_chain_tokens(member, wave['pooled_blocks'])} tokens) "
                f"reads a wrong prefix:\n{error}"
            )
