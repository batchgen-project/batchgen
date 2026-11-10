"""The Kimi decode-graph MLA span on the capability-selected dense seam (CPU).

The captured span used to call flash_mla directly and refused Blackwell with
NotImplementedError. It now runs ``paged_decode.mla_paged_decode`` against the
bucket's static page table, whose width is the seam's ``block_table_width``,
so the Blackwell kernel reads that table in place. Kernels are mocked; the GPU
gate is the dense decode graph probe on the Blackwell node.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from batchgen.attention.mla import paged_decode, trtllm_gen_mla
from batchgen.models.moonshotai.kimi_linear import cuda_graph_segments as cgs

HEADS, NOPE, ROPE, LORA, VDIM, HIDDEN, PAGE = 2, 4, 64, 512, 4, 16, 64


@pytest.fixture(autouse=True)
def _blackwell(monkeypatch):
    monkeypatch.setattr(paged_decode, "_BACKEND", None)
    monkeypatch.setattr(paged_decode.torch.cuda, "get_device_capability",
                        lambda *a: (10, 3))
    monkeypatch.setattr(trtllm_gen_mla, "_FI_WORKSPACES", {})
    monkeypatch.setattr(trtllm_gen_mla, "_FI_WORKSPACE_BYTES", 64)


def test_statics_hold_the_seam_width_and_refresh_the_manager_columns():
    width = paged_decode.block_table_width(3, PAGE)
    assert width == 4
    statics = cgs._BucketStatics(4, width, torch.device("cpu"), kda_slots=None)
    statics.arm_for_capture()
    rows = torch.tensor([[5, 6, 7], [8, 9, -1]], dtype=torch.int32)
    statics.refresh(
        2,
        torch.tensor([130, 70], dtype=torch.int32),
        torch.tensor([129, 69], dtype=torch.int32),
        rows,
    )
    assert statics.page_table.shape == (4, 4)
    assert torch.equal(statics.page_table[:2, :3], rows)
    # extra columns and padding rows stay on page 0 (valid memory, never reached)
    assert statics.page_table[:, 3].tolist() == [0, 0, 0, 0]
    assert statics.page_table[2:].tolist() == [[0] * 4] * 2
    assert statics.cache_seqlens.tolist() == [130, 70, 1, 1]
    # an empty DP rank refreshes from a zero-row view of the static table
    statics.refresh(0, torch.zeros(0, dtype=torch.int32),
                    torch.zeros(0, dtype=torch.int32), statics.page_table[:0])
    assert statics.num_valid_tokens.tolist() == [0]
    assert statics.cache_seqlens.tolist() == [1, 1, 1, 1]


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.q_lora_rank = None
        self.num_heads = HEADS
        self.qk_nope_head_dim = NOPE
        self.qk_rope_head_dim = ROPE
        self.q_head_dim = NOPE + ROPE
        self.kv_lora_rank = LORA
        self.v_head_dim = VDIM
        self.scaling = 0.07
        self.use_output_gate = False
        self.q_proj = nn.Linear(HIDDEN, HEADS * (NOPE + ROPE), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN, LORA + ROPE, bias=False)
        self.kv_a_layernorm = nn.Identity()
        self.kv_b_proj = nn.Linear(LORA, HEADS * (NOPE + VDIM), bias=False)
        self.o_proj = nn.Linear(HEADS * VDIM, HIDDEN, bias=False)
        for module in (self.q_proj, self.kv_a_proj_with_mqa, self.kv_b_proj, self.o_proj):
            module.weight.data = module.weight.data.bfloat16()


def test_graph_span_decodes_through_the_seam_on_blackwell(monkeypatch):
    bucket, pages = 4, 6
    blocked_k = torch.zeros(pages, PAGE, 1, LORA + ROPE, dtype=torch.bfloat16)
    manager = SimpleNamespace(
        get_layer_kv_cache=lambda layer_idx: (blocked_k, None)
    )
    monkeypatch.setattr(cgs.AttnWrapperBase, "gpu_paged_kv_manager", manager,
                        raising=False)
    writes = []
    kv_cache = types.ModuleType("batchgen_kernels.triton.kv_cache")
    kv_cache.run_paged_kv_token_update_fused = lambda **kw: writes.append(kw)
    monkeypatch.setitem(sys.modules, "batchgen_kernels.triton.kv_cache", kv_cache)
    decodes = []
    mla = types.ModuleType("flashinfer.mla")

    def trtllm(**kw):
        decodes.append(kw)
        q = kw["query"]
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], kw["kv_lora_rank"],
                           dtype=q.dtype)

    mla.trtllm_batch_decode_with_kv_cache_mla = trtllm
    pkg = types.ModuleType("flashinfer")
    pkg.mla = mla
    monkeypatch.setitem(sys.modules, "flashinfer", pkg)
    monkeypatch.setitem(sys.modules, "flashinfer.mla", mla)
    monkeypatch.setitem(sys.modules, "flash_mla", None)  # never reached

    statics = cgs._BucketStatics(
        bucket, paged_decode.block_table_width(3, PAGE), torch.device("cpu"),
        kda_slots=None,
    )
    statics.arm_for_capture()
    hidden = torch.randn(bucket, 1, HIDDEN).bfloat16()
    for schedule in (None, paged_decode.decode_schedule(statics.cache_seqlens, HEADS)):
        out, k_tensor = cgs._mla_decode_graph_safe(
            _Attn(),
            hidden,
            layer_idx=3,
            page_table=statics.page_table,
            slot_indices=statics.slot_indices,
            token_indices=statics.token_indices,
            cache_seqlens=statics.cache_seqlens,
            num_valid_tokens=statics.num_valid_tokens,
            page_size_tokens=PAGE,
            mla_schedule=schedule,
        )
        assert out.shape == (bucket, 1, HIDDEN)
        assert k_tensor.shape == (bucket, 1, 1, LORA + ROPE)
    assert len(decodes) == 2 and len(writes) == 2
    for kw in decodes:
        # the static table is read in place: no per-call padding copy
        assert kw["block_tables"] is statics.page_table
        assert kw["max_seq_len"] == statics.page_table.shape[1] * PAGE
        assert kw["kv_cache"].data_ptr() == blocked_k.data_ptr()
        assert kw["query"].shape == (bucket, 1, HEADS, LORA + ROPE)
    assert decodes[0]["workspace_buffer"] is decodes[1]["workspace_buffer"]
    for kw in writes:
        assert kw["page_table"] is statics.page_table
        assert kw["num_valid_tokens"] is statics.num_valid_tokens
