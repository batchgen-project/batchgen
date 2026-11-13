# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Shared CPU references for the Kimi-K3.1 DSA serving tests.

The serving modules reach every GPU kernel through a module-level seam. The
tests replace each seam with the reference below, built from the kernel
package's own CPU statements wherever one exists, so the orchestration under
test is the production one and only the arithmetic inside each launch is a
reference. Not a test module (no ``test_`` prefix).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DSA_DIR = ROOT / "batchgen_kernels" / "attention" / "dsa"


def load_by_path(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_fla_shim():
    shim = load_by_path("_k31_fla_cpu_shim_refs",
                        ROOT / "tests" / "kimi_k3_oracle_assets" / "fla_cpu_shim.py")
    shim.install(force=True)


#: The shipped page-split helpers. Registered under their real module name so
#: ``paged_mqa_logits.py``'s absolute import resolves without importing the
#: ``batchgen_kernels.attention.dsa`` package (whose ``__init__`` needs triton).
QUANT = load_by_path("batchgen_kernels.attention.dsa.indexer_k_quant",
                     DSA_DIR / "indexer_k_quant.py")
PML = load_by_path("_k31_paged_mqa_logits", DSA_DIR / "paged_mqa_logits.py")


def head_gates_ref(hidden, weight, n_heads, head_dim):
    """``fused_indexer_score.compute_head_gates``, verbatim."""
    gates = F.linear(hidden.float(), weight.float())
    return (gates * ((n_heads ** -0.5) * (head_dim ** -0.5))).to(torch.float32)


def fa3_ref(*, q, k_cache, v_cache, qv, page_table, cache_seqlens, softmax_scale,
            causal, num_splits, return_softmax_lse):
    """Page-size-1 FA3 with ``qv``: score = q.k + qv.v, output = P v, per row
    over the first ``cache_seqlens[row]`` entries of its page table.

    The kernel reads ``cache_seqlens[row]`` page-table entries with no bounds
    check, so a length past the table width or a ``-1`` entry inside the
    length is an out-of-bounds read on GPU: asserted here, never truncated."""
    batch, _, heads, _ = q.shape
    out = torch.zeros(batch, 1, heads, v_cache.shape[-1])
    for b in range(batch):
        length = int(cache_seqlens[b])
        assert 0 <= length <= page_table.shape[1], (b, length, page_table.shape)
        tokens = page_table[b, :length].to(torch.long)
        assert bool((tokens >= 0).all()) and bool((tokens < k_cache.shape[0]).all()), b
        k = k_cache[tokens, 0, 0].float()
        v = v_cache[tokens, 0, 0].float()
        scores = (q[b, 0].float() @ k.T + qv[b, 0].float() @ v.T) * softmax_scale
        out[b, 0] = torch.softmax(scores, -1) @ v
    return out


def install_sparse_attention_ref(monkeypatch):
    """Route the sparse-attention seam (``sparse_dispatch``) to its Hopper
    backend with ``fa3_ref`` as the kernel: the orchestration under test is
    the production one, the FA3 launch the CPU reference."""
    from batchgen.attention.mla import sparse_dispatch

    monkeypatch.setattr(sparse_dispatch, "_BACKEND", "fa3")
    monkeypatch.setattr(sparse_dispatch, "_fa3_with_kvcache", lambda: fa3_ref)
    monkeypatch.setattr(sparse_dispatch, "_fa3_num_splits", lambda: 1)


def install_flash_attn_varlen(monkeypatch):
    """A CPU ``flash_attn_interface`` for the UNCHANGED dense K3 prefill:
    causal attention inside each packed row, expanded heads."""
    def flash_attn_varlen_func(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                               max_seqlen_k, softmax_scale, causal):
        out = torch.zeros(q.shape[0], q.shape[1], v.shape[-1])
        cu = cu_seqlens_q.tolist()
        for start, end in zip(cu, cu[1:]):
            qs, ks, vs = q[start:end].float(), k[start:end].float(), v[start:end].float()
            scores = torch.einsum("qhd,khd->hqk", qs, ks) * softmax_scale
            n = end - start
            mask = torch.ones(n, n, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            out[start:end] = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), vs)
        return out.to(q.dtype)

    module = types.ModuleType("flash_attn_interface")
    module.flash_attn_varlen_func = flash_attn_varlen_func
    module.flash_attn_with_kvcache = fa3_ref
    monkeypatch.setitem(sys.modules, "flash_attn_interface", module)


def prefill_topk_ref(*, q_idx, index_keys, head_gates, causal_starts, causal_ends,
                     max_seqlen, topk):
    """The causal prefill scorer + top-k, in fp32 with the production masking
    contract (``topk_from_logits_reference``: a row that sees at most ``topk``
    keys takes its dense prefix), returned as packed-global indices.

    ``sum_h gate_h relu(q_h . k)`` over each query's own ``[start, end)``.
    """
    total = q_idx.shape[0]
    keys = index_keys.float()
    lengths = (causal_ends - causal_starts).to(torch.int32)
    width = int(lengths.max()) if total else 0
    logits = torch.zeros(total, max(width, 1))
    for i in range(total):
        s, e = int(causal_starts[i]), int(causal_ends[i])
        per_head = torch.relu(q_idx[i].float() @ keys[s:e].T)
        logits[i, : e - s] = (head_gates[i][:, None] * per_head).sum(0)
    relative = PML.topk_from_logits_reference(logits, lengths, topk)
    starts = causal_starts.to(relative.dtype).unsqueeze(-1)
    return torch.where(relative >= 0, relative + starts, relative)
