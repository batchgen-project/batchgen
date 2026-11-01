# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""MLA prefill over a prefix-pool chunk, and the per-row latent offload.

Under test, the full-attention half of the in-wave prefix pool:
  - ``pool_prefill_attention`` (``batchgen/prefix_reuse/executor.py``), which
    writes one layer of a chunk's K/V into the pool and then attends through the
    page table. K and V are indexed with their OWN head dim, because an MLA
    model attending on the expanded heads keeps a 192-wide K beside a 128-wide
    V while a GQA model keeps one dim for both.
  - ``mla_prefill_nope_prepacked``'s pool branch
    (``kimi_linear/serving_modules.py``): same q/k/v math, pooled extend
    attention instead of varlen self-attention, same 576-wide latent returned.
  - ``KimiLinearAttnWrapper._offload_prefill_pool_rows`` and
    ``_pool_prefill_kwargs`` (``kimi_linear/wrappers.py``): each row's latent
    goes to ITS Host sequence at ITS start, and the layer is mapped onto its
    dense pool row through one seam.

THE GATE is the whole prefix tree against the default path: for every prompt,
the pooled segments' rows (computed ONCE, for every member) concatenated with
that prompt's tail row must equal running the same function over the whole
prompt as one packed sequence. Both runs go through the production function; the
only difference is the attention call, so a mismatch is this slice's bug and not
the model's. The gate is a tolerance, not ``torch.equal``: a chunk runs the
projections at a different M than the whole prompt does, which this model
amplifies (serving_modules.py's GEMM rule) — exactness is pinned per kernel on
the GPU (S6), not here.

Covered: a depth-1 tree whose segment and three tails are in the SAME chunk
(``same_chunk_ancestors``, so the write must precede the attention of that
layer), a parent computed in an EARLIER chunk, V head dim != K head dim, the
GPT-OSS case where they are equal, and the per-row offload targets.

CPU-only, no ``import batchgen``: the package ``__init__`` JIT-builds a CUDA
extension, so every module under test is loaded by file path, and FA3's two
entry points are served by the plain-torch references below. The paged FA3 call
itself is GPU work (S6).

Run: python -m pytest tests/unit/test_k3_mla_pool_attention.py -q
"""

import ctypes
import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


_PLAN = _load("test_mla_pool_plan_mod", "batchgen", "prefix_reuse", "wave_plan.py")
_POOL = _load("test_mla_pool_pool_mod", "batchgen", "prefix_reuse", "pool.py")
_EXEC = _load("test_mla_pool_exec_mod", "batchgen", "prefix_reuse", "executor.py")
_CHECK = _load(
    "test_mla_pool_check_mod", "batchgen", "prefix_reuse", "offload_check.py"
)


# --------------------------------------------------------------------------- #
#  Plain-torch stand-ins for the two FA3 entry points
# --------------------------------------------------------------------------- #
def _attend(q, keys, values, scale):
    """Causal attention of ``q`` to ``keys``/``values``, in fp32.

    ``q`` holds the LAST ``q.shape[0]`` positions of the cached sequence, which
    is how FA3 aligns a varlen query block against a longer K cache: query ``i``
    sees keys ``[0, len(keys) - len(q) + i]``.
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


class _Fa3Calls:
    """How often each FA3 entry point ran, so a branch can be pinned."""

    varlen = 0
    paged = 0


def _varlen_reference(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                      max_seqlen_k, softmax_scale, causal):
    """``flash_attn_varlen_func``: per-sequence causal self-attention."""
    assert causal and torch.equal(cu_seqlens_q, cu_seqlens_k)
    _Fa3Calls.varlen += 1
    bounds = cu_seqlens_q.tolist()
    out = q.new_empty(q.shape[0], q.shape[1], v.shape[2])
    for lo, hi in zip(bounds, bounds[1:]):
        out[lo:hi] = _attend(q[lo:hi], k[lo:hi], v[lo:hi], softmax_scale)
    return out


def _paged_extend_reference(*, q, k_cache, v_cache, cache_seqlens, page_table,
                            cu_seqlens_q, max_seqlen_q, sinks=None,
                            softmax_scale=None, sliding_window=None):
    """``gqa_extend_fa``: each row attends the first ``cache_seqlens`` tokens of
    its pages, which are its pooled ancestors followed by its own."""
    assert sinks is None and sliding_window is None
    _Fa3Calls.paged += 1
    bounds = cu_seqlens_q.tolist()
    out = q.new_empty(q.shape[0], q.shape[1], v_cache.shape[-1])
    for row, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        pages = page_table[row].tolist()
        total = int(cache_seqlens[row])
        keys = torch.cat([k_cache[page] for page in pages])[:total]
        values = torch.cat([v_cache[page] for page in pages])[:total]
        out[lo:hi] = _attend(q[lo:hi], keys, values, softmax_scale)
    return out, None


# --------------------------------------------------------------------------- #
#  The modules under test, loaded without the batchgen package
# --------------------------------------------------------------------------- #
class FakeAttnWrapperBase:
    """The class-variable seam ``wrappers.py`` reads for a prefill offload.

    The real base class lives in the batchgen package (compiled kernels); only
    these members are on the MLA prefill path.
    """

    prefill_pool = None
    prefill_pool_batch = None
    prefill_pool_row_starts = None
    # Batch-level batchgen_debug.prefix_pool_offload_check installs a recorder
    # here; None is the production default and the only state the offload path
    # needs to stay free of the check.
    prefill_pool_offload_recorder = None
    cur_batch = None
    prepack_num_sequences = None
    prepack_seq_lengths = None
    pinned = []
    tracked = []

    @classmethod
    def pin_prefill_offload_tensor(cls, tensor, layer_idx):
        cls.pinned.append((layer_idx, tensor))

    @classmethod
    def track_prefill_offload_task(cls, task, layer_idx):
        cls.tracked.append((layer_idx, task))


def _load_kimi_linear():
    """``serving_modules`` and ``wrappers`` as one alias package.

    Both are real files; only their ``batchgen.models.wrappers`` imports are
    stubbed, and the stubs are withdrawn from ``sys.modules`` as soon as the two
    modules hold their references, so no other test sees them.
    """
    package = types.ModuleType("_kl_mla_pool")
    package.__path__ = [str(_KL_DIR)]
    sys.modules["_kl_mla_pool"] = package
    base = types.ModuleType("batchgen.models.wrappers")
    base.AttnWrapperBase = FakeAttnWrapperBase
    expert = types.ModuleType("batchgen.models.wrappers.expert")
    expert.ExpertWrapperBase = type("ExpertWrapperBase", (), {})
    # serving_modules imports it at module level; the attention paths under
    # test never route experts, so nothing is idle.
    expert.skip_idle_streamed_experts = (
        lambda experts, expert_ids, token_counts: []
    )
    previous = {name: sys.modules.get(name) for name in (base.__name__,
                                                         expert.__name__)}
    sys.modules[base.__name__] = base
    sys.modules[expert.__name__] = expert
    try:
        return (importlib.import_module("_kl_mla_pool.serving_modules"),
                importlib.import_module("_kl_mla_pool.wrappers"))
    finally:
        for name, module in previous.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


SM, WR = _load_kimi_linear()


@pytest.fixture(autouse=True)
def _fa3_and_executor(monkeypatch):
    """Serve FA3 from plain torch, and the lazy import from the loaded file.

    ``mla_prefill_nope_prepacked`` imports ``batchgen.prefix_reuse.executor``
    and that imports ``batchgen.attention.gqa``; registering both names keeps
    the real package (and its compiled kernels) out of this test.
    """
    monkeypatch.setitem(
        sys.modules, "batchgen.prefix_reuse.executor", _EXEC,
    )
    gqa = types.ModuleType("batchgen.attention.gqa")
    gqa.gqa_extend_fa = _paged_extend_reference
    monkeypatch.setitem(sys.modules, "batchgen.attention.gqa", gqa)
    fa3 = types.ModuleType("flash_attn_interface")
    fa3.flash_attn_varlen_func = _varlen_reference
    monkeypatch.setitem(sys.modules, "flash_attn_interface", fa3)
    _Fa3Calls.varlen = _Fa3Calls.paged = 0
    FakeAttnWrapperBase.pinned = []
    FakeAttnWrapperBase.tracked = []
    yield
    for name in ("prefill_pool", "prefill_pool_batch",
                 "prefill_pool_row_starts", "prefill_pool_offload_recorder",
                 "cur_batch"):
        setattr(FakeAttnWrapperBase, name, None)


# --------------------------------------------------------------------------- #
#  One tiny MLA layer and its prompts
# --------------------------------------------------------------------------- #
HIDDEN = 12
HEADS = 2
NOPE = 4
ROPE = 2          # so K is 6 per head
V_DIM = 3         # ... and V is 3: the 192/128 split, at test scale
LORA = 5          # latent is LORA + ROPE = 7 wide
PAGE = 8          # pool page, and the plan's shareable block


def blk(n):
    """One shareable block; its token ids identify the tree node uniquely."""
    return [n * 1000 + k for k in range(PAGE)]


# Two shared blocks, then a private block and one odd token per prompt, so a
# tail ends mid-page and the pool's last page is partially filled.
DEPTH1 = [blk(1) + blk(2) + blk(20 + i) + [910 + i] for i in range(3)]

_FEATURES = {}


def _feature(position, token):
    """The activations of one token, shared by every prompt passing through it.

    Keyed by (absolute position, token id): two prompts sharing a prefix feed
    the identical rows to the pooled segment and to their own reference run,
    which is what makes the comparison meaningful at all.
    """
    key = (position, token)
    row = _FEATURES.get(key)
    if row is None:
        generator = torch.Generator().manual_seed(
            (position * 1000003 + token) % (2 ** 31)
        )
        row = torch.randn(HIDDEN, generator=generator)
        _FEATURES[key] = row
    return row


def _hidden(positions, tokens):
    return torch.stack([
        _feature(int(position), int(token))
        for position, token in zip(positions, tokens)
    ])


def _mla_layer():
    """The attributes ``mla_prefill_nope_prepacked`` reads, at tiny dims."""
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
#  Wave plumbing: a plan, a pool, and one forward per chunk
# --------------------------------------------------------------------------- #
def _plan(prompts, same_chunk):
    plan = _PLAN.plan_wave_prefix_sharing(
        prompts, block_tokens=PAGE, pool_pages=1000, chunk_tokens=2048,
        same_chunk_ancestors=same_chunk,
    )
    _PLAN.validate_wave_prefix_plan(plan, prompts)
    assert plan.segments, "nothing is pooled, so no pooled prefix is exercised"
    return plan


def _pool(prompts, plan, *, num_layers=1, head_dim=NOPE + ROPE,
          v_head_dim=V_DIM):
    scratch = _POOL.scratch_pages_bound(
        plan.chunk_tokens, max(len(p) for p in prompts), len(prompts), PAGE
    )
    pool = _POOL.PrefillPrefixPool(
        num_layers=num_layers, num_pages=plan.pool_pages + scratch,
        page_tokens=PAGE, num_kv_heads=HEADS, head_dim=head_dim,
        dtype=torch.float32, device="cpu", v_head_dim=v_head_dim,
    )
    # Pages are handed out uninitialised; zeroing makes an unwritten slot
    # visible as a wrong answer instead of as whatever was there.
    pool.k.zero_()
    pool.v.zero_()
    return pool


def _chunk_batches(prompts, plan, pool):
    """Build, run, finish — chunk by chunk, as the worker's pool driver does.

    Building every chunk up front would hand a later chunk the pages an earlier
    chunk's tails just gave back, and the pool CONTENT has to survive from a
    segment's chunk to the chunk of the tails that read it.
    """
    executor = _EXEC.PoolChunkExecutor(plan, prompts, pool)
    for chunk in plan.chunks:
        batch = executor.build(chunk)
        yield batch
        executor.finish(chunk)
    executor.close()


def _run_wave(attn, prompts, plan, pool, pool_layer=0):
    """Every row's attention output and latent, keyed by plan item."""
    rows = {}
    for batch in _chunk_batches(prompts, plan, pool):
        attn_output, offload_kv = attn.mla_prefill_nope_prepacked(
            _hidden(batch.position_ids, batch.input_ids),
            batch.position_ids,
            batch.cu_seqlens_q,
            batch.max_seqlen_q,
            len(batch.items),
            pool=pool,
            pool_batch=batch,
            pool_layer=pool_layer,
        )
        for row, item in enumerate(batch.items):
            lo, hi = batch.row_bounds[row], batch.row_bounds[row + 1]
            rows[(item.kind, item.ref)] = (attn_output[lo:hi], offload_kv[lo:hi])
    return rows


def _run_whole(attn, prompt):
    """The prompt as one packed sequence: the default (no pool) path."""
    return attn.mla_prefill_nope_prepacked(
        _hidden(range(len(prompt)), prompt),
        None,
        torch.tensor([0, len(prompt)], dtype=torch.int32),
        len(prompt),
        1,
    )


# --------------------------------------------------------------------------- #
#  (i) The pooled tree reproduces per-prompt causal self-attention
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("same_chunk", [True, False])
def test_pooled_rows_reproduce_the_whole_prompt(same_chunk):
    """A segment is computed once and read by every member — in the SAME chunk
    (the write must precede this layer's attention) or in an earlier one."""
    attn, prompts = _mla_layer(), DEPTH1
    plan = _plan(prompts, same_chunk)
    assert (len(plan.chunks) == 1) is same_chunk

    rows = _run_wave(attn, prompts, plan, _pool(prompts, plan))

    assert _Fa3Calls.varlen == 0, "the pool branch must not run self-attention"
    for index, prompt in enumerate(prompts):
        chain = [seg for seg in plan.segments if index in seg.members]
        assert chain, "every prompt of this wave shares its root block"
        pieces = [rows[("segment", seg.segment_id)] for seg in chain]
        pieces.append(rows[("tail", index)])
        expected_output, expected_latent = _run_whole(attn, prompt)

        torch.testing.assert_close(
            torch.cat([output for output, _ in pieces]), expected_output,
            rtol=1e-5, atol=1e-5,
            msg=lambda text, i=index: f"prompt {i} attention moved: {text}",
        )
        torch.testing.assert_close(
            torch.cat([latent for _, latent in pieces]), expected_latent,
            rtol=1e-5, atol=1e-5,
            msg=lambda text, i=index: f"prompt {i} latent moved: {text}",
        )


def test_the_pool_holds_every_token_of_the_chunk():
    """What a row reads is what the chunk wrote: the pool slot of each token
    carries that token's K and V, for the tails as well as for the segments."""
    attn, prompts = _mla_layer(), DEPTH1
    plan = _plan(prompts, same_chunk=True)
    pool = _pool(prompts, plan)

    batch, = _chunk_batches(prompts, plan, pool)
    attn.mla_prefill_nope_prepacked(
        _hidden(batch.position_ids, batch.input_ids), batch.position_ids,
        batch.cu_seqlens_q, batch.max_seqlen_q, len(batch.items),
        pool=pool, pool_batch=batch, pool_layer=0,
    )

    keys = pool.k[0].view(-1, HEADS, NOPE + ROPE).index_select(0, batch.slots)
    values = pool.v[0].view(-1, HEADS, V_DIM).index_select(0, batch.slots)
    assert keys.shape == (batch.input_ids.numel(), HEADS, NOPE + ROPE)
    assert values.shape == (batch.input_ids.numel(), HEADS, V_DIM)
    # The rope half is one broadcast row per token, so it is a cheap witness
    # that the right token landed in the right slot.
    assert not bool((keys[:, :, NOPE:] == 0).all(dim=-1).any())
    assert not bool((values == 0).all(dim=-1).any())


def test_the_default_path_is_the_varlen_self_attention_it_was():
    """``pool_batch`` is opt-in: without it the call is the one it was."""
    parameters = inspect.signature(SM.mla_prefill_nope_prepacked).parameters
    assert [parameters[name].default for name in
            ("pool", "pool_batch", "pool_layer")] == [None, None, None]

    _run_whole(_mla_layer(), DEPTH1[0])

    assert (_Fa3Calls.varlen, _Fa3Calls.paged) == (1, 0)


# --------------------------------------------------------------------------- #
#  (ii) K and V are indexed with their own head dim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("v_head_dim", [V_DIM, NOPE + ROPE])
def test_pool_prefill_attention_writes_each_cache_at_its_own_width(v_head_dim):
    """An MLA pool keeps a wider K than V; a GQA (GPT-OSS) pool keeps one dim
    for both, and takes the same path it always did."""
    plan = _plan(DEPTH1, same_chunk=True)
    pool = _pool(DEPTH1, plan, num_layers=2, v_head_dim=v_head_dim)
    batch, = _chunk_batches(DEPTH1, plan, pool)
    tokens = batch.input_ids.numel()
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(tokens, HEADS, NOPE + ROPE, generator=generator)
    key = torch.randn(tokens, HEADS, NOPE + ROPE, generator=generator)
    value = torch.randn(tokens, HEADS, v_head_dim, generator=generator)

    output, _ = _EXEC.pool_prefill_attention(
        pool=pool, layer_idx=1, batch=batch, query=query, key=key, value=value,
        sinks=None, softmax_scale=0.5, sliding_window=None,
    )

    assert output.shape == (tokens, HEADS, v_head_dim)
    assert torch.equal(
        pool.k[1].view(-1, HEADS, NOPE + ROPE).index_select(0, batch.slots), key
    )
    assert torch.equal(
        pool.v[1].view(-1, HEADS, v_head_dim).index_select(0, batch.slots), value
    )
    # Only the layer it was asked for.
    assert not bool(pool.k[0].any()) and not bool(pool.v[0].any())


# --------------------------------------------------------------------------- #
#  (iii) Each row's latent is offloaded to its own Host target
# --------------------------------------------------------------------------- #
class _Task:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _RecordingView:
    """The Host KV worker view, recording the range offloads it is handed."""

    def __init__(self, physical_of=None):
        self.calls = []
        self._physical_of = physical_of or {}

    def resolve_physical_layer(self, layer_idx):
        return self._physical_of[layer_idx]

    def async_offload_layer_kv_to_host(self, **kwargs):
        raise AssertionError("a pool row must offload at its own start")

    def async_offload_layer_kv_range_to_host(
        self, *, layer_idx, sequence_ids, k_tensor, v_tensor,
        raw_start_positions, token_counts,
    ):
        task = _Task()
        self.calls.append(SimpleNamespace(
            layer_idx=layer_idx, sequence_id=sequence_ids[0], kv=k_tensor,
            v_tensor=v_tensor, start=raw_start_positions[0],
            tokens=token_counts[0], task=task,
        ))
        return task


def _host_targets(batch, segment_sids, global_ids):
    """What the worker publishes per row: (Host sequence id, start)."""
    ids, starts = [], []
    for item in batch.items:
        if item.kind == "segment":
            ids.append(segment_sids[item.ref])
            starts.append(0)
        else:
            ids.append(global_ids[item.ref])
            starts.append(item.token_start)
    return ids, starts


def _offload_layer(layer_idx, view):
    """A wrapper stand-in carrying both offload methods it dispatches between."""
    layer = SimpleNamespace(
        layer_idx=layer_idx,
        core_engine=SimpleNamespace(host_paged_kv_worker_view=view),
    )
    layer._offload_prefill_pool_rows = types.MethodType(
        WR.KimiLinearAttnWrapper._offload_prefill_pool_rows, layer
    )
    return layer


def test_each_pool_row_offloads_its_own_latent_at_its_own_start():
    prompts = DEPTH1
    plan = _plan(prompts, same_chunk=True)
    pool = _pool(prompts, plan)
    batch, = _chunk_batches(prompts, plan, pool)
    segment_sids = {seg.segment_id: -(1 + seg.segment_id) for seg in plan.segments}
    global_ids = {index: 500 + index for index in range(len(prompts))}
    ids, starts = _host_targets(batch, segment_sids, global_ids)
    FakeAttnWrapperBase.prefill_pool_batch = batch
    FakeAttnWrapperBase.prefill_pool_row_starts = starts
    FakeAttnWrapperBase.cur_batch = ids
    view = _RecordingView()
    layer = _offload_layer(11, view)
    latent = torch.randn(batch.input_ids.numel(), LORA + ROPE)

    WR.KimiLinearAttnWrapper._offload_prepacked_kv(layer, latent, None)

    assert len(view.calls) == len(batch.items)
    for row, call in enumerate(view.calls):
        lo, hi = batch.row_bounds[row], batch.row_bounds[row + 1]
        assert (call.sequence_id, call.start, call.tokens) == (
            ids[row], starts[row], hi - lo
        )
        assert call.layer_idx == 11 and call.v_tensor is None
        # [1, tokens, 1, 576] on the real model: one latent head per token.
        assert call.kv.shape == (1, hi - lo, 1, LORA + ROPE)
        assert torch.equal(call.kv[0, :, 0], latent[lo:hi])
        assert call.task.waited, "the d2h copy must land before the next row"
    # The d2h source is pinned and the task registered, per row, as the packed
    # path does: the allocator must not hand that storage to the next layer.
    assert len(FakeAttnWrapperBase.pinned) == len(view.calls)
    assert len(FakeAttnWrapperBase.tracked) == len(view.calls)
    for (pin_layer, pinned), (task_layer, tracked), call in zip(
        FakeAttnWrapperBase.pinned, FakeAttnWrapperBase.tracked, view.calls
    ):
        assert pinned is call.kv and tracked is call.task
        assert pin_layer == task_layer == 11


class _Lines:
    """Collect the check's log lines as they would land in a server log."""

    def __init__(self):
        self.errors = []
        self.infos = []

    def error(self, message, *args):
        self.errors.append(message % args)

    def info(self, message, *args):
        self.infos.append(message % args)


def test_the_offload_check_digests_each_row_against_its_own_destination():
    """``prefix_pool_offload_check``: one record per row, keyed to its target.

    The digest covers the row's own latent, so a destination rebuilt from that
    latent matches, and flipping the last token of one tail names that tail's
    LAST Host page -- the multi-page stride the gapped-chain corruption lives
    in. Every other test in this file runs with the recorder left at None, so
    they are the flag-off path: nothing here is reachable without it.
    """
    prompts = DEPTH1
    plan = _plan(prompts, same_chunk=True)
    batch, = _chunk_batches(prompts, plan, _pool(prompts, plan))
    segment_sids = {seg.segment_id: -(1 + seg.segment_id) for seg in plan.segments}
    global_ids = {index: 500 + index for index in range(len(prompts))}
    ids, starts = _host_targets(batch, segment_sids, global_ids)
    FakeAttnWrapperBase.prefill_pool_batch = batch
    FakeAttnWrapperBase.prefill_pool_row_starts = starts
    FakeAttnWrapperBase.cur_batch = ids
    recorder = _CHECK.PoolOffloadCheckRecorder(rank=4, page_tokens=PAGE)
    FakeAttnWrapperBase.prefill_pool_offload_recorder = recorder
    layer = _offload_layer(11, _RecordingView())
    latent = torch.randn(batch.input_ids.numel(), LORA + ROPE)

    WR.KimiLinearAttnWrapper._offload_prepacked_kv(layer, latent, None)

    assert len(recorder.rows) == len(batch.items)
    for row, record in enumerate(recorder.rows):
        lo, hi = batch.row_bounds[row], batch.row_bounds[row + 1]
        assert (record.kind, record.host_sid, record.dest_start_tokens) == (
            batch.items[row].kind, ids[row], starts[row],
        )
        assert (record.row, record.layer_idx, record.rank) == (row, 11, 4)
        assert record.token_count == hi - lo
        assert record.token_bytes == (LORA + ROPE) * latent.element_size()
        assert record.page_tokens == PAGE

    def _destination(record, page_ids=None, corrupt_row=None):
        """The row's Host bytes, rebuilt from the activations it offloaded."""
        lo, hi = batch.row_bounds[record.row], batch.row_bounds[record.row + 1]
        tokens = latent[lo:hi].clone()
        if record.row == corrupt_row:
            tokens[-1, -1] = -tokens[-1, -1] - 1.0
        flat = tokens.contiguous().reshape(-1)
        return ctypes.string_at(
            flat.data_ptr(), flat.numel() * flat.element_size()
        )

    clean_log = _Lines()
    clean = _CHECK.verify_pool_offload_rows(
        recorder.rows, logger=clean_log, reader=_destination, prober=None,
        wave_tag="m3s1",
    )
    assert clean.matched and clean.rows_checked == len(batch.items)
    assert clean.segment_rows and clean.tail_rows
    assert clean_log.errors == [] and len(clean_log.infos) == 1
    assert "rows_checked=%d" % len(batch.items) in clean_log.infos[0]

    tail_row = max(
        row for row, item in enumerate(batch.items) if item.kind == "tail"
    )
    bad_log = _Lines()
    bad = _CHECK.verify_pool_offload_rows(
        recorder.rows,
        logger=bad_log,
        reader=lambda record, page_ids: _destination(
            record, page_ids, corrupt_row=tail_row
        ),
        prober=None,
        wave_tag="m3s1",
    )
    assert len(bad.mismatches) == 1
    record, comparison = bad.mismatches[0]
    assert record.row == tail_row and record.kind == "tail"
    # The last token of the row, so the walk reached the row's final page.
    assert comparison.first_differing_page == (record.dest_end_tokens - 1) // PAGE
    assert len(bad_log.errors) == 1
    assert f"host_sid={ids[tail_row]}" in bad_log.errors[0]
    assert "mismatches=1" in bad_log.infos[0]


def test_a_row_count_that_disagrees_with_the_published_targets_is_refused():
    prompts = DEPTH1
    plan = _plan(prompts, same_chunk=True)
    batch, = _chunk_batches(prompts, plan, _pool(prompts, plan))
    FakeAttnWrapperBase.prefill_pool_batch = batch
    FakeAttnWrapperBase.prefill_pool_row_starts = [0]
    FakeAttnWrapperBase.cur_batch = [0]
    layer = _offload_layer(3, _RecordingView())

    with pytest.raises(RuntimeError, match="row metadata size mismatch"):
        WR.KimiLinearAttnWrapper._offload_prepacked_kv(
            layer, torch.zeros(batch.input_ids.numel(), LORA + ROPE), None
        )


# --------------------------------------------------------------------------- #
#  (iv) The layer is mapped onto its pool row through one seam
# --------------------------------------------------------------------------- #
def test_the_pool_row_of_a_layer_comes_from_the_pool_layout():
    """K3's 24 MLA layers sit densely in the pool in engine order, and the row
    is read from the pool layout's engine-layer map."""
    mla_layers = list(range(3, 92, 4)) + [92]
    layer_map = tuple(
        mla_layers.index(idx) if idx in mla_layers else -1
        for idx in range(93)
    )
    pool = object()
    FakeAttnWrapperBase.prefill_pool = pool
    FakeAttnWrapperBase.prefill_pool_batch = "a chunk"
    FakeAttnWrapperBase.prefill_pool_layer_map = layer_map
    try:
        for engine_layer, row in ((3, 0), (92, 23)):
            layer = SimpleNamespace(layer_idx=engine_layer)
            assert WR.KimiLinearAttnWrapper._pool_prefill_kwargs(layer) == {
                "pool": pool, "pool_batch": "a chunk", "pool_layer": row,
            }
    finally:
        FakeAttnWrapperBase.prefill_pool_layer_map = None


def test_outside_a_wave_the_prefill_call_is_unchanged():
    FakeAttnWrapperBase.prefill_pool_batch = None
    layer = SimpleNamespace(layer_idx=3, core_engine=None)

    assert WR.KimiLinearAttnWrapper._pool_prefill_kwargs(layer) == {}


def test_pool_row_comes_from_the_layout_not_a_cache_layer_map():
    """The 48B Host-KV profile keeps all 27 layers and resolves a layer to
    itself, but its pool holds only the 7 MLA layers: taking the row from the
    cache indexed past the pool (IndexError at pool.k[7] on a 7-row pool)."""
    mla_layers = [4, 8, 12, 16, 20, 24, 26]
    layer_map = tuple(
        mla_layers.index(idx) if idx in mla_layers else -1 for idx in range(27)
    )
    FakeAttnWrapperBase.prefill_pool_batch = object()
    FakeAttnWrapperBase.prefill_pool = object()
    FakeAttnWrapperBase.prefill_pool_layer_map = layer_map
    try:
        for expected, engine_layer in enumerate(mla_layers):
            layer = SimpleNamespace(layer_idx=engine_layer)
            kwargs = WR.KimiLinearAttnWrapper._pool_prefill_kwargs(layer)
            assert kwargs["pool_layer"] == expected, (engine_layer, kwargs)
        # a KDA layer has no row and must be refused, not silently mapped
        with pytest.raises(RuntimeError, match="no prefix-pool row"):
            WR.KimiLinearAttnWrapper._pool_prefill_kwargs(SimpleNamespace(layer_idx=5))
    finally:
        FakeAttnWrapperBase.prefill_pool_batch = None
        FakeAttnWrapperBase.prefill_pool = None
        FakeAttnWrapperBase.prefill_pool_layer_map = None
