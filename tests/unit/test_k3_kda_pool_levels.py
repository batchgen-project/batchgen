# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""KDA prefill over a prefix-pool chunk runs level by level, and is EXACT.

Under test: ``_kda_prefill_levels`` in
``batchgen/models/moonshotai/kimi_linear/serving_modules.py`` driven by
``state_levels`` from ``batchgen/prefix_reuse/levels.py``, i.e. the recurrent
half of the in-wave prefix pool. A pooled segment is computed once as a prefill
row of its own, so a row that continues it cannot start from zero state: the
chunk's rows are grouped by level, each level runs as one packed call, and
between two levels the parent's state is FORKED into each child's slot.

THE GATE is ``torch.equal`` — against the same sequence run unsegmented, whole,
from zero state, which is the shape of the default (``pool_levels=None``) path.
K3's top-16 router turns a 1-ulp seam into an O(1) logit difference
(tests/gpu/test_kimi_k3_kda_fla_parity.py::test_E_kernel_seam_amplification),
so sharing a prefix has to be bit-exact, not close.

WHY IT CAN BE EXACT. Same argument as the segmented sweep
(tests/test_kimi_k3_kda_segmented.py): a pooled segment ends on a page multiple
and the page is ``_KDA_CHUNK_SIZE``, so a child's chunk grid is exactly the
restriction of the grid the whole sequence would have used. Only the fp32
recurrent state and the three raw conv rows cross the cut, now through the
state pools instead of a register.

WHAT IS PINNED: the level cut and its rebased bounds, the per-level slot slice,
the fork (recurrent AND the three conv states) for a parent in this chunk and
for one that ran in an earlier chunk, that a parent's own slot is never written
by its children, that a row with no parent starts from zero state, and the
worker's level-wise plan gate.

CPU-only. ``serving_modules.py`` is loaded by file path (the real package
``__init__`` needs the compiled kernels) through the segmented test's loader,
whose ``_cpu_chunk_kda`` reference is reused here; the conv kernel is CUDA-only
so a transcription of its contract stands in for it.

Run: python -m pytest tests/unit/test_k3_kda_pool_levels.py -q
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

torch.set_num_threads(1)


def _load(name, *relative):
    path = Path(__file__).resolve().parents[2].joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLAN = _load("test_kda_levels_plan_mod", "batchgen", "prefix_reuse", "wave_plan.py")
_POOL = _load("test_kda_levels_pool_mod", "batchgen", "prefix_reuse", "pool.py")
_EXEC = _load("test_kda_levels_exec_mod", "batchgen", "prefix_reuse", "executor.py")
_LEVELS = _load("test_kda_levels_mod", "batchgen", "prefix_reuse", "levels.py")
# The segmented test owns the CPU stand-in for fla's chunk_kda and the
# file-path loader for serving_modules; both are reused verbatim.
_SEGMENTED = _load("test_kda_levels_segmented_ref", "tests",
                   "test_kimi_k3_kda_segmented.py")
SM = _SEGMENTED.SM
_cpu_chunk_kda = _SEGMENTED._cpu_chunk_kda

H, D = _SEGMENTED.H, _SEGMENTED.D
DIM = H * D
W = 4            # conv width; the state is the last W - 1 raw input rows
SLOTS = 32
# The pool page, which is also every pooled segment's token boundary. It must
# be a multiple of fla's chunk size or no level could resume exactly.
PAGE = SM._KDA_CHUNK_SIZE
# q | k | v | f gate | beta, laid out as _kda_project returns them.
WIDTH = 3 * DIM + DIM + H


def blk(n):
    """One shareable block; its token ids identify the tree node uniquely."""
    return [n * 1000 + k for k in range(PAGE)]


# One pooled segment of two blocks, then a private block per prompt.
DEPTH1 = [blk(1) + blk(2) + blk(20 + i) + [910 + i] for i in range(3)]
# One root block, then three two-way branches: 7 pooled segments, 8 tails, and
# four levels of rows in one chunk.
DEPTH3 = [
    blk(1) + blk(2 + a) + blk(4 + 2 * a + b) + blk(8 + 4 * a + 2 * b + c) + [900 + c]
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
]


# --------------------------------------------------------------------------- #
#  Inputs: one feature row per distinct prefix-tree token
# --------------------------------------------------------------------------- #
_FEATURES = {}


def _feature(position, token):
    """The activations of one token, shared by every prompt passing through it.

    Keyed by (absolute position, token id), so two prompts that share a prefix
    feed the identical rows to the pooled segment and to their own reference
    run — which is what makes a bit-exact comparison meaningful at all.
    """
    key = (position, token)
    row = _FEATURES.get(key)
    if row is None:
        generator = torch.Generator().manual_seed(
            (position * 1000003 + token) % (2 ** 31)
        )
        row = torch.randn(WIDTH, generator=generator)
        _FEATURES[key] = row
    return row


def _rows_of(prompt, start=0, end=None):
    end = len(prompt) if end is None else end
    return [(position, prompt[position]) for position in range(start, end)]


def _split(rows):
    """``_kda_project``'s outputs for a packed range of tokens."""
    packed = torch.stack([_feature(position, token) for position, token in rows])
    q, k, v, f, beta = torch.split(packed, [DIM, DIM, DIM, DIM, H], dim=-1)
    return (q.contiguous(), k.contiguous(), v.contiguous(),
            f.contiguous().view(-1, H, D), beta.contiguous())


def _conv_weights():
    generator = torch.Generator().manual_seed(20261001)

    def pair():
        return (torch.randn(DIM, W, generator=generator).mul_(0.3),
                torch.randn(DIM, generator=generator).mul_(0.1))

    return (pair(), pair(), pair())


def _kernel_kwargs():
    generator = torch.Generator().manual_seed(4242)
    return dict(
        A_log=torch.randn(H, generator=generator),
        dt_bias=torch.randn(H * D, generator=generator),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        output_final_state=True,
    )


# --------------------------------------------------------------------------- #
#  CPU stand-in for _kda_prefill_conv
# --------------------------------------------------------------------------- #
def _cpu_conv(x, weight, bias, pool, cu_seqlens, slot_ids, has_initial_state):
    """``_kda_prefill_conv``'s contract, token-major, on CPU.

    Per sequence: silu(causal depthwise conv), reading ``pool[slot]`` as the
    W-1 rows before its first token when ``has_initial_state``, and leaving the
    last W-1 RAW input rows there (batchgen_kernels/conv1d.py, kda_conv_triton).
    """
    bounds = cu_seqlens.tolist()
    out = torch.empty_like(x)
    for row in range(len(bounds) - 1):
        lo, hi = bounds[row], bounds[row + 1]
        slot = int(slot_ids[row])
        raw = x[lo:hi].t()                                   # (dim, T)
        init = (pool[slot] if bool(has_initial_state[row])
                else torch.zeros_like(pool[slot]))
        padded = torch.cat([init, raw], dim=1)
        acc = bias[:, None].clone()
        for tap in range(W):
            acc = acc + padded[:, tap:tap + hi - lo] * weight[:, tap:tap + 1]
        out[lo:hi] = (acc * torch.sigmoid(acc)).t()
        pool[slot] = padded[:, -(W - 1):]
    return out


@pytest.fixture(autouse=True)
def _cpu_kernels(monkeypatch):
    """fla and the conv extension are CUDA-only; serve both from CPU code."""
    monkeypatch.setattr(SM, "_kda_prefill_conv", _cpu_conv)


# --------------------------------------------------------------------------- #
#  The state pools and the slot accounting a KDA layer sees
# --------------------------------------------------------------------------- #
def _state_pools():
    return SimpleNamespace(
        recurrent_pool=torch.zeros(SLOTS, H, D, D),
        conv_q=torch.zeros(SLOTS, DIM, W - 1),
        conv_k=torch.zeros(SLOTS, DIM, W - 1),
        conv_v=torch.zeros(SLOTS, DIM, W - 1),
    )


class Slots:
    """``KDALayerState.prepare_prefill`` and its manager, for one layer.

    A fresh allocation zeroes the slot in every pool (the F4 zero-on-alloc) and
    re-allocating a live sequence id returns its slot untouched, exactly as
    KDAStateGPUManager._ensure_state_item does.
    """

    def __init__(self, state):
        self.pools = (state.recurrent_pool, state.conv_q, state.conv_k,
                      state.conv_v)
        self.by_id = {}

    def contains(self, sequence_id):
        return int(sequence_id) in self.by_id

    def lookup(self, sequence_id):
        return self.by_id[int(sequence_id)]

    def alloc(self, sequence_id):
        slot = self.by_id.get(int(sequence_id))
        if slot is None:
            slot = len(self.by_id)
            self.by_id[int(sequence_id)] = slot
            for pool in self.pools:
                pool[slot].zero_()
        return slot

    def prepare_prefill(self, sequence_ids):
        has_initial_state = torch.tensor(
            [self.contains(s) for s in sequence_ids], dtype=torch.bool
        )
        slots = torch.tensor(
            [self.alloc(s) for s in sequence_ids], dtype=torch.int32
        )
        return slots, has_initial_state


FAKE_LAYER = SimpleNamespace(num_heads=H, head_dim=D)


# --------------------------------------------------------------------------- #
#  The two runs being compared
# --------------------------------------------------------------------------- #
def _unsegmented(rows, weights, kwargs, segment_tokens=None):
    """One whole sequence from zero state: the default path's shape."""
    state = _state_pools()
    slots = torch.tensor([0], dtype=torch.int32)
    fresh = torch.zeros(1, dtype=torch.bool)
    cu_seqlens = torch.tensor([0, len(rows)], dtype=torch.int32)
    q, k, v, f, beta = _split(rows)
    (qw, qb), (kw, kb), (vw, vb) = weights
    q = _cpu_conv(q, qw, qb, state.conv_q, cu_seqlens, slots, fresh)
    k = _cpu_conv(k, kw, kb, state.conv_k, cu_seqlens, slots, fresh)
    v = _cpu_conv(v, vw, vb, state.conv_v, cu_seqlens, slots, fresh)
    o = SM._kda_chunk_segments(
        _cpu_chunk_kda,
        q.view(1, -1, H, D), k.view(1, -1, H, D), v.view(1, -1, H, D),
        f.unsqueeze(0), beta.unsqueeze(0), cu_seqlens, slots,
        state.recurrent_pool, kwargs, segment_tokens,
    )
    return o[0], state


def _plan(prompts, same_chunk=True):
    plan = _PLAN.plan_wave_prefix_sharing(
        prompts, block_tokens=PAGE, pool_pages=1000, chunk_tokens=2048,
        same_chunk_ancestors=same_chunk,
    )
    plan = _LEVELS.order_chunk_items_by_level(plan)
    _PLAN.validate_wave_prefix_plan(plan, prompts)
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
    return batches


def _run_wave(plan, prompts, weights, kwargs, segment_tokens=None):
    """Prefill every chunk through the level/fork path, as the wrapper does.

    One layer, so one set of pools; a tail's row id is its wave index and a
    segment's is the negative pseudo id the worker hands out.
    """
    state = _state_pools()
    slots = Slots(state)
    sids = {seg.segment_id: -(1 + seg.segment_id) for seg in plan.segments}
    outputs = {}
    for batch in _batches(plan, prompts):
        row_ids = [
            sids[item.ref] if item.kind == "segment" else item.ref
            for item in batch.items
        ]
        slot_ids, has_initial_state = slots.prepare_prefill(row_ids)
        levels = _LEVELS.state_levels(
            batch,
            segment_sids=sids,
            row_ids=row_ids,
            slots=slot_ids,
            has_initial_state=has_initial_state,
            slot_of=slots.lookup,
        )
        q, k, v, f, beta = _split(
            [(int(position), int(token))
             for position, token in zip(batch.position_ids, batch.input_ids)]
        )
        o = SM._kda_prefill_levels(
            FAKE_LAYER, _cpu_chunk_kda, state, levels, q, k, v, f, beta,
            weights, kwargs, segment_tokens,
        )
        for row, item in enumerate(batch.items):
            lo, hi = batch.row_bounds[row], batch.row_bounds[row + 1]
            outputs[(item.kind, item.ref)] = o[0, lo:hi]
    return outputs, state, slots, sids


def _final_states(state, slot):
    return (state.recurrent_pool[slot], state.conv_q[slot], state.conv_k[slot],
            state.conv_v[slot])


# --------------------------------------------------------------------------- #
#  (i) A whole prefix tree, computed once, is bit-exact per prompt
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prompts,same_chunk,segment_tokens", [
    (DEPTH1, True, None),          # parent and children in ONE chunk
    (DEPTH1, True, PAGE),          # ... and each level segmented inside itself
    (DEPTH1, False, None),         # parent ran in an EARLIER chunk
    (DEPTH3, True, None),          # four levels, 7 segments, 8 tails
])
def test_level_and_fork_reproduce_the_unsegmented_sweep(
    prompts, same_chunk, segment_tokens
):
    weights, kwargs = _conv_weights(), _kernel_kwargs()
    plan = _plan(prompts, same_chunk=same_chunk)
    assert plan.segments, "nothing is pooled, so no fork is exercised"

    outputs, state, slots, sids = _run_wave(
        plan, prompts, weights, kwargs, segment_tokens
    )

    for index, prompt in enumerate(prompts):
        chain = [seg for seg in plan.segments if index in seg.members]
        assert chain, "every prompt of these waves shares its root block"
        # The prompt as the pool computed it: its chain of shared segments
        # (computed once, for every member) followed by its own tail.
        pieces = [outputs[("segment", seg.segment_id)] for seg in chain]
        pieces.append(outputs[("tail", index)])
        expected_o, expected = _unsegmented(
            _rows_of(prompt), weights, kwargs, segment_tokens
        )

        assert torch.equal(torch.cat(pieces), expected_o), (
            "prompt {} moved: max_abs={:.3e}".format(
                index, (torch.cat(pieces) - expected_o).abs().max().item())
        )
        got = _final_states(state, slots.lookup(index))
        for pool, reference in zip(got, _final_states(expected, 0)):
            assert torch.equal(pool, reference), (
                "prompt {} left a different final state: max_abs={:.3e}".format(
                    index, (pool - reference).abs().max().item())
            )


# --------------------------------------------------------------------------- #
#  (ii) A parent's own slot is never written by its children
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prompts,same_chunk", [
    (DEPTH1, True), (DEPTH1, False), (DEPTH3, True),
])
def test_a_segment_slot_still_holds_its_own_end_state(prompts, same_chunk):
    """Copy-on-branch: the fork writes the CHILD's slot only, so after the wave
    a segment's slot is still the state its own last token produced — which is
    what the next chunk's children, and the Host commit, depend on."""
    weights, kwargs = _conv_weights(), _kernel_kwargs()
    plan = _plan(prompts, same_chunk=same_chunk)

    _, state, slots, sids = _run_wave(plan, prompts, weights, kwargs)

    for seg in plan.segments:
        prefix = _rows_of(prompts[seg.members[0]], 0, seg.token_end)
        _, expected = _unsegmented(prefix, weights, kwargs)
        got = _final_states(state, slots.lookup(sids[seg.segment_id]))
        for pool, reference in zip(got, _final_states(expected, 0)):
            assert torch.equal(pool, reference), (
                "segment {} ending at {} was overwritten by a child".format(
                    seg.segment_id, seg.token_end)
            )


# --------------------------------------------------------------------------- #
#  (iii) The level path only forks where the plan says to
# --------------------------------------------------------------------------- #
def test_a_row_without_a_parent_starts_from_zero_state():
    """A level-0 row reads no state: the pool path must agree with a reference
    whose slot was never written, and must not be handed a resume flag."""
    weights, kwargs = _conv_weights(), _kernel_kwargs()
    plan = _plan(DEPTH3)
    batch, = _batches(plan, DEPTH3)
    sids = {seg.segment_id: -(1 + seg.segment_id) for seg in plan.segments}
    row_ids = [
        sids[item.ref] if item.kind == "segment" else item.ref
        for item in batch.items
    ]
    state = _state_pools()
    slots = Slots(state)
    slot_ids, has_initial_state = slots.prepare_prefill(row_ids)

    levels = _LEVELS.state_levels(
        batch, segment_sids=sids, row_ids=row_ids, slots=slot_ids,
        has_initial_state=has_initial_state, slot_of=slots.lookup,
    )

    assert not bool(levels[0].has_initial_state.any())
    assert levels[0].fork_parents is None and levels[0].fork_children is None
    assert all(bool(level.has_initial_state.all()) for level in levels[1:])
    # Every deeper row forks, from a slot that is not its own.
    for level in levels[1:]:
        assert level.fork_children.numel() == level.slots.numel()
        assert torch.equal(level.fork_children, level.slots.long())
        assert not bool((level.fork_parents == level.fork_children).any())


def test_state_levels_refuses_rows_it_cannot_slot():
    plan = _plan(DEPTH1)
    batch, = _batches(plan, DEPTH1)
    sids = {seg.segment_id: -(1 + seg.segment_id) for seg in plan.segments}

    with pytest.raises(ValueError, match="state slots"):
        _LEVELS.state_levels(
            batch, segment_sids=sids, row_ids=[0], slots=torch.zeros(1),
            has_initial_state=torch.zeros(1, dtype=torch.bool),
            slot_of=lambda sid: 0,
        )


def test_kda_prefill_serving_defaults_to_the_plain_packed_path():
    """``pool_levels`` is opt-in, so a prefill outside a pool wave is the same
    call it was: whole-range convs and one segmented chunk sweep."""
    parameters = inspect.signature(SM.kda_prefill_serving).parameters
    assert parameters["pool_levels"].default is None
    source = inspect.getsource(SM.kda_prefill_serving)
    assert "if pool_levels is not None:" in source


# --------------------------------------------------------------------------- #
#  (iv) The worker plans level-wise only for a model that declares it
# --------------------------------------------------------------------------- #
def _plan_harness():
    """The S3 group harness compiles ``_plan_prefill_pool_wave`` onto fakes;
    reuse it instead of rebuilding it. It imports the ``batchgen`` package,
    which requires the compiled kernels, so a box without them skips."""
    try:
        return importlib.import_module("test_k3_prefix_pool_group")
    except (ImportError, RuntimeError) as exc:  # no batchgen_kernels here
        pytest.skip(f"the worker plan harness is unavailable: {exc}")


def _planned(monkeypatch, *, declares):
    group = _plan_harness()
    group._stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker = group._plan_worker(
        free_pages=20, fake_dist=group.FakeDist(peer_budget=[16, 32]),
    )
    if not declares:
        del worker.parallel_manager.supports_group_pool_prefill
    wave = worker._plan_prefill_pool_wave(group.UUIDS, {})
    assert wave is not None
    return wave.plan


def test_a_level_wise_model_finishes_the_tree_in_one_forward(monkeypatch):
    plan = _planned(monkeypatch, declares=True)

    assert plan.same_chunk_ancestors is True
    assert len(plan.chunks) == 1
    # Rows grouped by level: the segment, then the two tails that read it.
    assert [(item.kind, len(item.prefix_segments))
            for item in plan.chunks[0].items] == [
        ("segment", 0), ("tail", 1), ("tail", 1)
    ]


def test_a_model_that_declares_nothing_keeps_the_plan_it_had(monkeypatch):
    """A model that pools K/V and declares nothing (GPT-OSS) keeps the flag it
    had, ``not with_v``, so a reader still waits for the next chunk."""
    plan = _planned(monkeypatch, declares=False)

    assert plan.same_chunk_ancestors is False
    assert len(plan.chunks) == 2
    assert [item.kind for item in plan.chunks[0].items] == ["segment"]
