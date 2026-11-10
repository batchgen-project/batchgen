"""Dispatch contract of batchgen.attention.mla.paged_decode (CPU-only).

The kernels themselves are validated on GPU against an fp32 reference; this
covers capability selection and the flash_mla legacy-kwarg adaptation.
"""

from __future__ import annotations

import sys
import types

import pytest

from batchgen.attention.mla import paged_decode


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(paged_decode, "_BACKEND", None)
    monkeypatch.setattr(paged_decode, "_FLASHMLA_DENSE_KWARGS", None)


@pytest.mark.parametrize(
    "capability, expected",
    [((9, 0), "flashmla"), ((10, 0), "flashinfer-trtllm-gen"), ((10, 3), "flashinfer-trtllm-gen")],
)
def test_backend_by_capability(monkeypatch, capability, expected):
    monkeypatch.setattr(paged_decode.torch.cuda, "get_device_capability", lambda *a: capability)
    assert paged_decode._backend() == expected


def test_unknown_capability_fails_loud(monkeypatch):
    monkeypatch.setattr(paged_decode.torch.cuda, "get_device_capability", lambda *a: (8, 0))
    with pytest.raises(RuntimeError, match="no dense paged MLA decode backend"):
        paged_decode._backend()


def test_flashmla_legacy_kwarg_only_when_supported(monkeypatch):
    module = types.ModuleType("flash_mla")

    def current(q, k_cache, block_table, cache_seqlens, head_dim_v, meta, splits,
                softmax_scale=None, causal=False, use_legacy=False):
        pass

    module.flash_mla_with_kvcache = current
    monkeypatch.setitem(sys.modules, "flash_mla", module)
    assert paged_decode.flashmla_dense_kwargs() == {"use_legacy": True}

    paged_decode._FLASHMLA_DENSE_KWARGS = None

    def original(q, k_cache, block_table, cache_seqlens, head_dim_v, meta, splits,
                 softmax_scale=None, causal=False):
        pass

    module.flash_mla_with_kvcache = original
    assert paged_decode.flashmla_dense_kwargs() == {}


# --------------------------------------------------------------------------- #
#  Capturability, Hopper call shape, Blackwell trtllm-gen adaptation           #
# --------------------------------------------------------------------------- #

import torch  # noqa: E402

from batchgen.attention.mla import sparse_dispatch, trtllm_gen_mla  # noqa: E402

H, LORA, ROPE, PAGE = 12, 512, 64, 64
SCALE = 0.07


@pytest.fixture
def fresh_workspace(monkeypatch):
    monkeypatch.setattr(trtllm_gen_mla, "_FI_WORKSPACES", {})
    monkeypatch.setattr(trtllm_gen_mla, "_FI_WORKSPACE_BYTES", 64)


def _capability(monkeypatch, capability):
    monkeypatch.setattr(paged_decode.torch.cuda, "get_device_capability", lambda *a: capability)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3)])
def test_both_backends_are_graph_capturable(monkeypatch, capability):
    _capability(monkeypatch, capability)
    assert paged_decode.graph_capturable()


def _dense_inputs(batch=3, pages=4, table_pages=3):
    torch.manual_seed(0)
    query = torch.randn(batch, 1, H, LORA + ROPE).bfloat16()
    blocked_k = torch.randn(pages, PAGE, 1, LORA + ROPE).bfloat16()
    table = torch.arange(batch * table_pages, dtype=torch.int32).view(batch, table_pages) % pages
    seqlens = torch.tensor([1, 130, 0][:batch], dtype=torch.int32)
    return query, blocked_k, table, seqlens


def _install_flash_mla(monkeypatch, calls, *, legacy):
    module = types.ModuleType("flash_mla")

    def get_mla_metadata(cache_seqlens, num_q_tokens, num_kv_heads, **kwargs):
        calls.append(("metadata", cache_seqlens, num_q_tokens, num_kv_heads, kwargs))
        return torch.zeros(2, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)

    if legacy:
        def flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
                                   meta, splits, softmax_scale=None, causal=False,
                                   use_legacy=False):
            calls.append(("decode", (q, k_cache, block_table, cache_seqlens, head_dim_v,
                                     meta, splits, softmax_scale, causal),
                          {"use_legacy": use_legacy}))
            return torch.zeros(q.shape[0], 1, q.shape[2], head_dim_v), None
    else:
        def flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
                                   meta, splits, softmax_scale=None, causal=False):
            calls.append(("decode", (q, k_cache, block_table, cache_seqlens, head_dim_v,
                                     meta, splits, softmax_scale, causal), {}))
            return torch.zeros(q.shape[0], 1, q.shape[2], head_dim_v), None

    module.get_mla_metadata = get_mla_metadata
    module.flash_mla_with_kvcache = flash_mla_with_kvcache
    monkeypatch.setitem(sys.modules, "flash_mla", module)


@pytest.mark.parametrize("legacy", [False, True])
def test_hopper_call_is_unchanged(monkeypatch, legacy):
    """The pre-change call: per-call metadata from cache_seqlens, then
    flash_mla_with_kvcache with the caller's tensors, causal, legacy kwarg
    only when the wheel has it."""
    _capability(monkeypatch, (9, 0))
    calls = []
    _install_flash_mla(monkeypatch, calls, legacy=legacy)
    query, blocked_k, table, seqlens = _dense_inputs()
    paged_decode.mla_paged_decode(query, blocked_k, table, seqlens, LORA, SCALE)
    extra = {"use_legacy": True} if legacy else {}
    (metadata, decode) = calls
    assert metadata[0] == "metadata" and metadata[1] is seqlens
    assert metadata[2:] == (H, 1, extra)
    args = decode[1]
    assert args[0] is query and args[1] is blocked_k and args[2] is table
    assert args[3] is seqlens and args[4] == LORA
    assert args[7] == SCALE and args[8] is True
    assert decode[2] == extra
    # Hopper reads any table width in place.
    assert paged_decode.block_table_width(3, PAGE) == 3


def test_hopper_consumes_a_shared_schedule_without_replanning(monkeypatch):
    _capability(monkeypatch, (9, 0))
    calls = []
    _install_flash_mla(monkeypatch, calls, legacy=False)
    query, blocked_k, table, seqlens = _dense_inputs()
    schedule = paged_decode.decode_schedule(seqlens, H)
    assert [c[0] for c in calls] == ["metadata"]
    for _ in range(3):  # three MLA layers of one step
        paged_decode.mla_paged_decode(query, blocked_k, table, seqlens, LORA, SCALE,
                                      schedule=schedule)
    assert [c[0] for c in calls] == ["metadata", "decode", "decode", "decode"]
    for call in calls[1:]:
        assert call[1][5] is schedule.tile_scheduler_metadata
        assert call[1][6] is schedule.num_splits


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


def test_blackwell_dense_adapts_to_trtllm_gen(monkeypatch, fresh_workspace):
    _capability(monkeypatch, (10, 3))
    calls = []
    _install_flashinfer(monkeypatch, calls)
    query, blocked_k, table, seqlens = _dense_inputs(table_pages=3)
    out = paged_decode.mla_paged_decode(query, blocked_k, table, seqlens, LORA, SCALE)
    assert out.shape == (3, 1, H, LORA)
    (kw,) = calls
    assert kw["query"] is query  # packed [latent | pe] query, no copy
    assert kw["kv_cache"].shape == (4, PAGE, LORA + ROPE)
    assert kw["kv_cache"].data_ptr() == blocked_k.data_ptr()
    # page size 64: the kernel reads page ids in pairs (128 tokens), so a
    # 3-column table gains one -1 column that no length reaches
    assert kw["block_tables"].shape == (3, 4)
    assert torch.equal(kw["block_tables"][:, :3], table)
    assert kw["block_tables"][:, 3].tolist() == [-1, -1, -1]
    # zero-length padding rows run as length 1 (shared with the sparse seam)
    assert kw["seq_lens"].tolist() == [1, 130, 1]
    # static bound: the table capacity, never a device read of the lengths
    assert kw["max_seq_len"] == 4 * PAGE
    assert kw["sparse_mla_top_k"] == 0
    assert kw["bmm1_scale"] == SCALE and kw["bmm2_scale"] == 1.0
    assert kw["kv_lora_rank"] == LORA and kw["qk_rope_head_dim"] == ROPE
    assert kw["backend"] == "trtllm-gen"


def test_blackwell_reads_an_aligned_table_in_place(monkeypatch, fresh_workspace):
    _capability(monkeypatch, (10, 3))
    calls = []
    _install_flashinfer(monkeypatch, calls)
    query, blocked_k, table, seqlens = _dense_inputs(table_pages=4)
    paged_decode.mla_paged_decode(query, blocked_k, table, seqlens, LORA, SCALE)
    assert calls[0]["block_tables"] is table
    assert calls[0]["max_seq_len"] == 4 * PAGE


def test_blackwell_reuses_one_static_workspace_across_calls_and_seams(
    monkeypatch, fresh_workspace,
):
    _capability(monkeypatch, (10, 3))
    monkeypatch.setattr(sparse_dispatch, "_BACKEND", None)
    monkeypatch.setattr(sparse_dispatch.torch.cuda, "get_device_capability",
                        lambda *a: (10, 3))
    calls = []
    _install_flashinfer(monkeypatch, calls)
    query, blocked_k, table, seqlens = _dense_inputs(table_pages=4)
    paged_decode.mla_paged_decode(query, blocked_k, table, seqlens, LORA, SCALE)
    small = _dense_inputs(batch=2, table_pages=2)
    paged_decode.mla_paged_decode(*small, LORA, SCALE)
    rows = torch.full((3, 256), -1, dtype=torch.int32)
    sparse_dispatch.sparse_mla_decode(
        query[..., LORA:].contiguous(), query[..., :LORA].contiguous(), blocked_k, rows,
        torch.tensor([0, 1, 2], dtype=torch.int32), SCALE)
    buffers = [c["workspace_buffer"] for c in calls]
    assert len(calls) == 3
    assert all(b is buffers[0] for b in buffers)
    assert len(trtllm_gen_mla._FI_WORKSPACES) == 1
    assert not buffers[0].any()  # zero-initialized once (semaphore state)


def test_blackwell_launch_never_reads_device_values_on_the_host(
    monkeypatch, fresh_workspace,
):
    """No .item()/.tolist()/.cpu()/int(tensor)/bool(tensor) inside the call:
    a host read would synchronize every layer and break graph capture."""
    _capability(monkeypatch, (10, 3))
    calls = []
    _install_flashinfer(monkeypatch, calls)
    query, blocked_k, table, seqlens = _dense_inputs(table_pages=3)
    paged_decode.block_table_width(3, PAGE)  # resolve the backend first

    def forbidden(*args, **kwargs):
        raise AssertionError("host read of a device tensor in the decode launch")

    for name in ("item", "tolist", "cpu", "__int__", "__bool__", "__index__"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    paged_decode.mla_paged_decode(query, blocked_k, table, seqlens, LORA, SCALE)
    schedule = paged_decode.decode_schedule(seqlens, H)
    monkeypatch.undo()
    assert len(calls) == 1
    assert schedule == paged_decode.DecodeSchedule()


def test_blackwell_schedule_is_empty_and_touches_no_kernel(monkeypatch):
    _capability(monkeypatch, (10, 3))
    monkeypatch.setitem(sys.modules, "flash_mla", None)  # any import would fail
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setitem(sys.modules, "flashinfer.mla", None)
    _, _, _, seqlens = _dense_inputs()
    schedule = paged_decode.decode_schedule(seqlens, H)
    assert schedule.tile_scheduler_metadata is None and schedule.num_splits is None


@pytest.mark.parametrize(
    "max_pages, page_size, width",
    [(1, 64, 2), (3, 64, 4), (4, 64, 4), (5, 32, 8), (7, 16, 8), (3, 128, 3), (3, 256, 3)],
)
def test_blackwell_table_width(monkeypatch, max_pages, page_size, width):
    _capability(monkeypatch, (10, 3))
    assert paged_decode.block_table_width(max_pages, page_size) == width


@pytest.mark.parametrize("page_size", [0, 48, 96])
def test_blackwell_rejects_a_page_size_the_kernel_cannot_group(monkeypatch, page_size):
    _capability(monkeypatch, (10, 3))
    with pytest.raises(trtllm_gen_mla.MlaShapeError, match="page size"):
        paged_decode.block_table_width(3, page_size)


def test_blackwell_rejects_a_foreign_cache_row(monkeypatch, fresh_workspace):
    _capability(monkeypatch, (10, 3))
    _install_flashinfer(monkeypatch, [])
    query, _, table, seqlens = _dense_inputs(table_pages=4)
    with pytest.raises(trtllm_gen_mla.MlaShapeError, match="blocked_k must be"):
        paged_decode.mla_paged_decode(
            query, torch.zeros(4, PAGE, 1, 656).bfloat16(), table, seqlens, LORA, SCALE)
