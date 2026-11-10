"""Dispatch contract of batchgen.attention.mla.sparse_dispatch (CPU-only).

The kernels themselves are validated on GPU against an fp32 reference (node
probe); this covers capability selection, the unchanged Hopper FA3 call, and
the argument adaptation of each Blackwell kernel, with mocked capability and
mocked kernel modules.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from batchgen.attention.mla import sparse_dispatch as sd
from batchgen.attention.mla import trtllm_gen_mla

H, LORA, ROPE, PAGE = 12, 512, 64, 64


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(sd, "_BACKEND", None)
    monkeypatch.setattr(trtllm_gen_mla, "_FI_WORKSPACES", {})
    monkeypatch.setattr(sd, "_ANNOUNCED", set())


def _capability(monkeypatch, capability):
    monkeypatch.setattr(sd.torch.cuda, "get_device_capability", lambda *a: capability)


@pytest.mark.parametrize(
    "capability, expected",
    [((9, 0), "fa3"), ((10, 0), "blackwell"), ((10, 3), "blackwell")],
)
def test_backend_by_capability(monkeypatch, capability, expected):
    _capability(monkeypatch, capability)
    assert sd._backend() == expected
    assert sd.site_symbols() == sd.SITE_SYMBOLS[expected]
    assert sd.graph_capturable()


@pytest.mark.parametrize("capability", [(8, 0), (12, 0)])
def test_unknown_capability_fails_loud(monkeypatch, capability):
    _capability(monkeypatch, capability)
    with pytest.raises(RuntimeError, match="no sparse MLA attention backend"):
        sd._backend()


def _inputs(n=3, topk=256, pages=8):
    torch.manual_seed(0)
    q_pe = torch.randn(n, 1, H, ROPE).bfloat16()
    q_latent = torch.randn(n, 1, H, LORA).bfloat16()
    blocked_k = torch.randn(pages, PAGE, 1, LORA + ROPE).bfloat16()
    rows = torch.full((n, topk), -1, dtype=torch.int32)
    rows[:, :5] = torch.arange(5, dtype=torch.int32)
    lengths = torch.tensor([5, 0, 3][:n], dtype=torch.int32)
    return q_pe, q_latent, blocked_k, rows, lengths


def _fa3_recorder(calls):
    def fa3(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        return torch.zeros(q.shape[0], 1, q.shape[2], kwargs["v_cache"].shape[-1])
    return fa3


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_hopper_calls_fa3_exactly_as_before_the_seam(monkeypatch, phase):
    """The pre-seam call: FA3 qv over a page-size-1 view, ckv as V, k_pe as K,
    causal, the sparse split policy, no LSE."""
    _capability(monkeypatch, (9, 0))
    calls = []
    monkeypatch.setattr(sd, "_fa3_with_kvcache", lambda: _fa3_recorder(calls))
    monkeypatch.setattr(sd, "_fa3_num_splits", lambda: 1)
    q_pe, q_latent, blocked_k, rows, lengths = _inputs()
    if phase == "decode":
        sd.sparse_mla_decode(q_pe, q_latent, blocked_k, rows, lengths, 0.07)
    else:
        sd.sparse_mla_prefill(q_pe, q_latent, blocked_k.view(-1, LORA + ROPE), rows,
                              lengths, 0.07)
    (kw,) = calls
    flat = blocked_k.view(-1, 1, 1, LORA + ROPE)
    assert kw["q"] is q_pe and kw["qv"] is q_latent
    assert kw["page_table"] is rows and kw["cache_seqlens"] is lengths
    assert torch.equal(kw["k_cache"], flat[..., LORA:])
    assert torch.equal(kw["v_cache"], flat[..., :LORA])
    assert kw["k_cache"].data_ptr() == blocked_k.data_ptr() + LORA * 2  # a view, no copy
    assert kw["softmax_scale"] == 0.07 and kw["causal"] is True
    assert kw["num_splits"] == 1 and kw["return_softmax_lse"] is False


def _install_flashinfer(monkeypatch, calls):
    mla = types.ModuleType("flashinfer.mla")

    def trtllm(**kwargs):
        calls.append(kwargs)
        q = kwargs["query"]
        return torch.ones(q.shape[0], q.shape[1], q.shape[2], kwargs["kv_lora_rank"])

    mla.trtllm_batch_decode_with_kv_cache_mla = trtllm
    pkg = types.ModuleType("flashinfer")
    pkg.mla = mla
    monkeypatch.setitem(sys.modules, "flashinfer", pkg)
    monkeypatch.setitem(sys.modules, "flashinfer.mla", mla)


def test_blackwell_decode_adapts_to_trtllm_gen_sparse(monkeypatch):
    _capability(monkeypatch, (10, 3))
    calls = []
    _install_flashinfer(monkeypatch, calls)
    monkeypatch.setattr(trtllm_gen_mla, "_FI_WORKSPACE_BYTES", 64)
    q_pe, q_latent, blocked_k, rows, lengths = _inputs()
    out = sd.sparse_mla_decode(q_pe, q_latent, blocked_k, rows, lengths, 0.07)
    assert out.shape == (3, 1, H, LORA)
    (kw,) = calls
    # query = [latent | pe], the KV row order [ckv | k_pe]
    assert torch.equal(kw["query"][..., :LORA], q_latent)
    assert torch.equal(kw["query"][..., LORA:], q_pe)
    assert kw["kv_cache"].shape == (8, PAGE, LORA + ROPE)
    assert kw["kv_cache"].data_ptr() == blocked_k.data_ptr()
    assert kw["block_tables"].shape == (3, 1, 256)
    assert torch.equal(kw["block_tables"][:, 0], rows)
    # length-0 padding rows run as length 1 over their -1 slot (no NaN)
    assert kw["seq_lens"].tolist() == [5, 1, 3]
    assert kw["max_seq_len"] == 256 and kw["sparse_mla_top_k"] == 256
    assert kw["bmm1_scale"] == 0.07 and kw["bmm2_scale"] == 1.0
    assert kw["kv_lora_rank"] == LORA and kw["qk_rope_head_dim"] == ROPE
    assert kw["backend"] == "trtllm-gen"
    # the shared process-lifetime workspace of this device
    assert kw["workspace_buffer"] is trtllm_gen_mla._FI_WORKSPACES[("cpu", None)]


def test_blackwell_decode_rejects_a_foreign_cache_row(monkeypatch):
    _capability(monkeypatch, (10, 3))
    _install_flashinfer(monkeypatch, [])
    q_pe, q_latent, _, rows, lengths = _inputs()
    with pytest.raises(sd.SparseMlaShapeError, match="blocked_k must be"):
        sd.sparse_mla_decode(q_pe, q_latent, torch.zeros(8, PAGE, 1, 656).bfloat16(),
                             rows, lengths, 0.07)


def _install_flash_mla(monkeypatch, calls):
    module = types.ModuleType("flash_mla")

    def sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None,
                   use_legacy=False, return_log2_lse=None):
        calls.append(dict(q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v,
                          topk_length=topk_length, use_legacy=use_legacy,
                          return_log2_lse=return_log2_lse))
        # head h of row t carries t * 128 + h (exact in bf16 for t < 2)
        t = torch.arange(q.shape[0]).view(-1, 1, 1).float()
        h = torch.arange(q.shape[1]).view(1, -1, 1).float()
        return ((t * 128 + h).expand(q.shape[0], q.shape[1], d_v).bfloat16(), None, None)

    module.flash_mla_sparse_fwd = sparse_fwd
    monkeypatch.setitem(sys.modules, "flash_mla", module)


@pytest.mark.parametrize("heads, tile", [(12, 64), (64, 64), (96, 128)])
def test_blackwell_prefill_pads_heads_to_the_kernel_tile(monkeypatch, heads, tile):
    _capability(monkeypatch, (10, 3))
    calls = []
    _install_flash_mla(monkeypatch, calls)
    monkeypatch.setattr(sd, "_FLASHMLA_PREFILL_ROW_CHUNK", 2)
    n, topk = 5, 256
    q_pe = torch.randn(n, 1, heads, ROPE).bfloat16()
    q_latent = torch.randn(n, 1, heads, LORA).bfloat16()
    kv_rows = torch.randn(40, LORA + ROPE).bfloat16()
    rows = torch.full((n, topk), -1, dtype=torch.int64)
    lengths = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    out = sd.sparse_mla_prefill(q_pe, q_latent, kv_rows, rows, lengths, 0.07)
    assert out.shape == (n, 1, heads, LORA)
    assert [c["q"].shape[0] for c in calls] == [2, 2, 1]  # row chunks
    starts = [0, 2, 4]
    for start, call in zip(starts, calls):
        rows_in = call["q"].shape[0]
        assert call["q"].shape == (rows_in, tile, LORA + ROPE)
        assert torch.equal(call["q"][:, :heads, :LORA], q_latent[start:start + rows_in, 0])
        assert torch.equal(call["q"][:, :heads, LORA:], q_pe[start:start + rows_in, 0])
        assert not call["q"][:, heads:].any()  # zero padding heads
        assert call["kv"].shape == (40, 1, LORA + ROPE)
        assert call["kv"].data_ptr() == kv_rows.data_ptr()
        assert call["indices"].shape == (rows_in, 1, topk)
        assert call["indices"].dtype == torch.int32
        assert call["topk_length"].tolist() == lengths[start:start + rows_in].tolist()
        assert call["use_legacy"] is True and call["return_log2_lse"] is False
        assert call["sm_scale"] == 0.07 and call["d_v"] == LORA
    # output row t keeps the kernel's first `heads` heads of chunk row t - start
    for t in range(n):
        local = t - max(s for s in starts if s <= t)
        assert out[t, 0, :, 0].float().tolist() == [local * 128 + h for h in range(heads)]


def test_blackwell_prefill_refuses_unsupported_geometry(monkeypatch):
    _capability(monkeypatch, (10, 3))
    _install_flash_mla(monkeypatch, [])
    kv_rows = torch.randn(40, LORA + ROPE).bfloat16()
    lengths = torch.ones(2, dtype=torch.int32)
    with pytest.raises(sd.SparseMlaShapeError, match="at most 128 query heads"):
        sd.sparse_mla_prefill(torch.zeros(2, 1, 129, ROPE).bfloat16(),
                              torch.zeros(2, 1, 129, LORA).bfloat16(), kv_rows,
                              torch.zeros(2, 256, dtype=torch.int32), lengths, 0.1)
    with pytest.raises(sd.SparseMlaShapeError, match="multiple of 128"):
        sd.sparse_mla_prefill(torch.zeros(2, 1, H, ROPE).bfloat16(),
                              torch.zeros(2, 1, H, LORA).bfloat16(), kv_rows,
                              torch.zeros(2, 100, dtype=torch.int32), lengths, 0.1)
    with pytest.raises(sd.SparseMlaShapeError, match="kv_rows must be"):
        sd.sparse_mla_prefill(torch.zeros(2, 1, H, ROPE).bfloat16(),
                              torch.zeros(2, 1, H, LORA).bfloat16(),
                              kv_rows.view(40, 1, -1),
                              torch.zeros(2, 256, dtype=torch.int32), lengths, 0.1)


def test_the_dsa_modules_have_no_capability_branch():
    """Golden rule 7: the model code calls the seam; the seam alone picks the
    kernel by capability."""
    from pathlib import Path

    kimi = Path(sd.__file__).resolve().parents[2] / "models" / "moonshotai" / "kimi_linear"
    for name in ("dsa_decode.py", "dsa_sparse_prefill.py"):
        source = (kimi / name).read_text()
        assert "get_device_capability" not in source, name
        assert "from batchgen.attention.mla.sparse_dispatch import" in source, name


def test_each_phase_is_announced_once(monkeypatch, caplog):
    _capability(monkeypatch, (9, 0))
    monkeypatch.setattr(sd, "_fa3_with_kvcache", lambda: _fa3_recorder([]))
    monkeypatch.setattr(sd, "_fa3_num_splits", lambda: 1)
    q_pe, q_latent, blocked_k, rows, lengths = _inputs()
    with caplog.at_level("INFO"):
        for _ in range(3):
            sd.sparse_mla_decode(q_pe, q_latent, blocked_k, rows, lengths, 0.07)
            sd.sparse_mla_prefill(q_pe, q_latent, blocked_k.view(-1, LORA + ROPE), rows,
                                  lengths, 0.07)
    lines = [r.getMessage() for r in caplog.records if "[sparse-mla]" in r.getMessage()]
    assert lines == [
        "[sparse-mla] first sparse decode on backend fa3: rows=3 heads=12 topk=256",
        "[sparse-mla] first sparse prefill on backend fa3: rows=3 heads=12 topk=256",
    ]
