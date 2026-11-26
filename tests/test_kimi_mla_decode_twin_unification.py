# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3 / Kimi-Linear                                            #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""The dense NoPE-MLA decode exists ONCE (D13 step 5: operator twins merged).

``serving_modules.mla_decode_nope_core`` is the one spelling of the dense
decode math; ``mla_decoding_nope_with_pagekv`` (eager) and
``cuda_graph_segments._mla_decode_graph_safe`` (captured) are bindings of the
same function that differ ONLY in the injected KV write/table seam:

  * eager: the manager assigns the slot, writes the token, and hands back its
    live block table;
  * captured: the fused static-table kernel writes over the bucket's fixed
    table (padding rows dropped by ``num_valid_tokens``).

CPU oracle: both bindings run on identical weights and inputs with reference
seams — the manager write and the fused static write land the same bytes in
the same cache slots — and a deterministic stand-in for the capability-
selected dense decode. The outputs must match BITWISE (torch.equal): every op
on this path is deterministic on CPU, so any divergence is a real seam leak,
not tolerance noise.

Structure pins hold the merge itself: neither binding respells a projection,
and the K3.1 DSA steps share the same ``_project`` / ``_absorb`` / ``_finish``
objects instead of carrying a third copy of the dense statements.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from batchgen.attention.mla import paged_decode
from batchgen.models.moonshotai.kimi_linear import cuda_graph_segments as cgs
from batchgen.models.moonshotai.kimi_linear import serving_modules as sm

ROOT = Path(__file__).resolve().parents[1]
KIMI = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"

HEADS, NOPE, ROPE, LORA, VDIM, HIDDEN, PAGE = 2, 4, 8, 16, 4, 16, 4
KV_DIM = LORA + ROPE


class _Attn(nn.Module):
    def __init__(self, *, q_lora_rank=None, use_output_gate=False, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.q_lora_rank = q_lora_rank
        self.num_heads = HEADS
        self.qk_nope_head_dim = NOPE
        self.qk_rope_head_dim = ROPE
        self.q_head_dim = NOPE + ROPE
        self.kv_lora_rank = LORA
        self.v_head_dim = VDIM
        self.scaling = 0.07
        self.use_output_gate = use_output_gate
        if q_lora_rank is not None:
            self.q_a_proj = nn.Linear(HIDDEN, q_lora_rank, bias=False)
            self.q_a_layernorm = nn.Identity()
            self.q_b_proj = nn.Linear(q_lora_rank, HEADS * (NOPE + ROPE), bias=False)
        else:
            self.q_proj = nn.Linear(HIDDEN, HEADS * (NOPE + ROPE), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN, KV_DIM, bias=False)
        self.kv_a_layernorm = nn.Identity()
        self.kv_b_proj = nn.Linear(LORA, HEADS * (NOPE + VDIM), bias=False)
        self.o_proj = nn.Linear(HEADS * VDIM, HIDDEN, bias=False)
        if use_output_gate:
            self.g_proj = nn.Linear(HIDDEN, HEADS * VDIM, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.bfloat16()


def _reference_decode(query, blocked_k, block_table, cache_seqlens, head_dim_v,
                      softmax_scale, schedule=None):
    """Deterministic CPU stand-in for the capability-selected dense decode.

    Reads the cache THROUGH the block table it is handed, so a binding that
    feeds a wrong table or a wrong write placement changes the result.
    """
    bsz = query.shape[0]
    rows = []
    for b in range(bsz):
        n = int(cache_seqlens[b])
        idx = torch.arange(n)
        pages = block_table[b, idx // PAGE].long()
        keys = blocked_k[pages, idx % PAGE, 0, :].float()        # [n, kv_dim]
        q = query[b, 0].float()                                   # [H, kv_dim]
        weights = torch.softmax(q @ keys.t() * softmax_scale, dim=-1)
        rows.append((weights @ keys[:, :head_dim_v]).to(query.dtype))
    return torch.stack(rows).unsqueeze(1)                         # [B,1,H,c]


def _write_reference(blocked_k, k_tokens, page_table, positions, valid):
    for b in range(valid):
        pos = int(positions[b])
        page = int(page_table[b, pos // PAGE])
        blocked_k[page, pos % PAGE, 0, :] = k_tokens[b]


class _FakeManager:
    """The eager seam: slot assignment + write + live table, batch order."""

    def __init__(self, blocked_k, block_table):
        self.device = torch.device("cpu")
        self._blocked_k = blocked_k
        self._block_table = block_table
        self.writes = 0

    def update_layer_decode_new_token(self, *, k_tensor, v_tensor,
                                      sequence_lengths, layer_idx, batch_slice):
        assert v_tensor is None
        assert batch_slice is None
        self.writes += 1
        _write_reference(
            self._blocked_k, k_tensor.view(k_tensor.shape[0], -1),
            self._block_table, sequence_lengths, k_tensor.shape[0],
        )

    def get_layer_kv_with_page_table(self, *, layer_idx):
        return self._blocked_k, None, self._block_table


@pytest.fixture()
def _fused_static_writer(monkeypatch):
    """The captured seam's fused kernel, as a reference write that honors the
    static table and ``num_valid_tokens``."""
    calls = []

    def fused(*, k_cache, k_tokens, page_table, slot_indices, token_indices,
              page_size_tokens, num_valid_tokens):
        assert page_size_tokens == PAGE
        calls.append({"page_table": page_table,
                      "num_valid_tokens": num_valid_tokens})
        rows = page_table[slot_indices.long()]
        _write_reference(k_cache, k_tokens, rows, token_indices,
                         int(num_valid_tokens[0]))

    module = types.ModuleType("batchgen_kernels.triton.kv_cache")
    module.run_paged_kv_token_update_fused = fused
    monkeypatch.setitem(sys.modules, "batchgen_kernels.triton.kv_cache", module)
    return calls


def _statics(bucket, positions, block_table, bsz):
    width = block_table.shape[1]
    page_table = torch.zeros(bucket, width, dtype=torch.int32)
    page_table[:bsz] = block_table
    cache_seqlens = torch.ones(bucket, dtype=torch.int32)
    cache_seqlens[:bsz] = positions + 1
    token_indices = torch.zeros(bucket, dtype=torch.int32)
    token_indices[:bsz] = positions
    return SimpleNamespace(
        page_table=page_table,
        slot_indices=torch.arange(bucket, dtype=torch.int32),
        token_indices=token_indices,
        cache_seqlens=cache_seqlens,
        num_valid_tokens=torch.tensor([bsz], dtype=torch.int32),
    )


@pytest.mark.parametrize("q_lora_rank", [None, 12])
@pytest.mark.parametrize("use_output_gate", [False, True])
@pytest.mark.parametrize("bucket", [2, 4])
def test_eager_and_graph_bindings_are_bitwise_identical(
    monkeypatch, _fused_static_writer, q_lora_rank, use_output_gate, bucket
):
    bsz, pages = 2, 6
    monkeypatch.setattr(paged_decode, "mla_paged_decode", _reference_decode)

    torch.manual_seed(7)
    attn = _Attn(q_lora_rank=q_lora_rank, use_output_gate=use_output_gate)
    hidden = torch.randn(bsz, 1, HIDDEN).bfloat16()
    positions = torch.tensor([9, 5], dtype=torch.int32)
    block_table = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int32)
    history = torch.randn(pages, PAGE, 1, KV_DIM).bfloat16()

    # --- eager binding: manager-owned write + live table -------------------
    eager_cache = history.clone()
    manager = _FakeManager(eager_cache, block_table)
    eager_out, eager_k = sm.mla_decoding_nope_with_pagekv(
        attn, hidden, positions.view(bsz, 1).long(), positions + 1, 10,
        None, manager, 3, None,
    )
    assert manager.writes == 1

    # --- captured binding: fused static-table write, padded bucket ---------
    graph_cache = history.clone()
    monkeypatch.setattr(
        cgs.AttnWrapperBase, "gpu_paged_kv_manager",
        SimpleNamespace(get_layer_kv_cache=lambda layer_idx: (graph_cache, None)),
        raising=False,
    )
    statics = _statics(bucket, positions, block_table, bsz)
    padded = torch.zeros(bucket, 1, HIDDEN, dtype=hidden.dtype)
    padded[:bsz] = hidden
    graph_out, graph_k = cgs._mla_decode_graph_safe(
        attn,
        padded,
        layer_idx=3,
        page_table=statics.page_table,
        slot_indices=statics.slot_indices,
        token_indices=statics.token_indices,
        cache_seqlens=statics.cache_seqlens,
        num_valid_tokens=statics.num_valid_tokens,
        page_size_tokens=PAGE,
        mla_schedule=None,
    )
    assert len(_fused_static_writer) == 1
    assert _fused_static_writer[0]["page_table"] is statics.page_table
    assert _fused_static_writer[0]["num_valid_tokens"] is statics.num_valid_tokens

    # Same seam contents => bitwise-identical caches and outputs; the padding
    # rows of the bucket never reach the live rows.
    assert torch.equal(eager_cache, graph_cache)
    assert torch.equal(eager_out, graph_out[:bsz])
    assert torch.equal(eager_k, graph_k[:bsz])
    assert graph_out.shape == (bucket, 1, HIDDEN)
    assert graph_k.shape == (bucket, 1, 1, KV_DIM)


def test_eager_binding_keeps_the_batch_slice_table_window(monkeypatch):
    """The eager-only asymmetry (manager batch_slice) is preserved: the slice
    window is applied to the live table before the dense decode reads it."""
    seen = {}

    def spy_decode(query, blocked_k, block_table, cache_seqlens, head_dim_v,
                   softmax_scale, schedule=None):
        seen["table"] = block_table
        return _reference_decode(query, blocked_k, block_table, cache_seqlens,
                                 head_dim_v, softmax_scale)

    monkeypatch.setattr(paged_decode, "mla_paged_decode", spy_decode)
    attn = _Attn()
    full_table = torch.tensor(
        [[1, 2, 3], [4, 5, 0], [0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    cache = torch.randn(6, PAGE, 1, KV_DIM).bfloat16()

    class _SliceManager(_FakeManager):
        def update_layer_decode_new_token(self, *, k_tensor, v_tensor,
                                          sequence_lengths, layer_idx,
                                          batch_slice):
            assert batch_slice == (1, 3)
            _write_reference(
                self._blocked_k, k_tensor.view(k_tensor.shape[0], -1),
                self._block_table[1:3], sequence_lengths, k_tensor.shape[0],
            )

    manager = _SliceManager(cache, full_table)
    positions = torch.tensor([7, 3], dtype=torch.int32)
    out, _ = sm.mla_decoding_nope_with_pagekv(
        attn, torch.randn(2, 1, HIDDEN).bfloat16(),
        positions.view(2, 1).long(), positions + 1, 8,
        None, manager, 0, (1, 3),
    )
    assert out.shape == (2, 1, HIDDEN)
    assert torch.equal(seen["table"], full_table[1:3])


def test_zero_rows_short_circuit_never_touches_the_manager():
    attn = _Attn()
    hidden = torch.zeros(0, 1, HIDDEN, dtype=torch.bfloat16)
    sentinel = SimpleNamespace(device=torch.device("cpu"))
    out, k_tensor = sm.mla_decoding_nope_with_pagekv(
        attn, hidden, torch.zeros(0, 1, dtype=torch.long),
        torch.zeros(0, dtype=torch.int32), 4, None, sentinel, 0, None,
    )
    assert out is hidden
    assert k_tensor.shape == (0, 1, 1, KV_DIM)


# --------------------------------------------------------------------------- #
#  Structure: the math is written once                                        #
# --------------------------------------------------------------------------- #


def _function(path, name):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing `{name}` in {path.name}")


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (KIMI / "serving_modules.py", "mla_decoding_nope_with_pagekv"),
        (KIMI / "cuda_graph_segments.py", "_mla_decode_graph_safe"),
    ],
)
def test_bindings_do_not_respell_the_dense_math(path, name):
    binding = ast.unparse(_function(path, name))
    assert "mla_decode_nope_core" in binding
    # The projections, the absorb and the output stage live only in the core
    # and its helpers; a binding carries none of their statements.
    for respelled in ("kv_a_proj_with_mqa", "kv_b_proj", "einsum", "o_proj",
                      "mla_paged_decode"):
        assert respelled not in binding, (name, respelled)


def test_core_owns_the_dense_statements_once():
    core = ast.unparse(_function(KIMI / "serving_modules.py",
                                 "mla_decode_nope_core"))
    assert "_project(" in core
    assert "_absorb(" in core
    assert "_finish(" in core
    assert "bind_kv_write(" in core


def test_dsa_steps_share_the_same_helper_objects():
    from batchgen.models.moonshotai.kimi_linear import dsa_decode

    assert dsa_decode._project is sm._project
    assert dsa_decode._absorb is sm._absorb
    assert dsa_decode._finish is sm._finish
