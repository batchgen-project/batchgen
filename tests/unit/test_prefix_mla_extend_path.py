import sys
import types
from types import SimpleNamespace

import pytest
import torch

_FLASHINFER_STUB = types.ModuleType("flashinfer")
_FLASHINFER_STUB.BatchMLAPagedAttentionWrapper = object
sys.modules.setdefault("flashinfer", _FLASHINFER_STUB)

from batchgen.attention.forward_metadata import (  # noqa: E402
    ForwardBatchMetadata,
    PrefillAttentionMetadata,
)
from batchgen.attention.mla import flashinfer_extend  # noqa: E402
from batchgen.models.wrappers.prefix_mla_extend import (  # noqa: E402
    MlaExtendSpec,
    run_projected_mla_prefix_attention_from_gpu_pages,
)


class _FakeMlaGpuManager:
    def __init__(self, *, has_v_cache: bool = False):
        self.config = SimpleNamespace(has_v_cache=has_v_cache)
        self.append_calls = []
        self.blocked_k = torch.arange(
            4 * 8 * 1 * 6,
            dtype=torch.float32,
        ).reshape(4, 8, 1, 6)
        self.block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    def append_layer_prefill_suffix_tokens(
        self,
        *,
        k_tensor,
        v_tensor,
        append_plan,
        layer_idx,
    ):
        self.append_calls.append(
            {
                "k_tensor": k_tensor,
                "v_tensor": v_tensor,
                "append_plan": append_plan,
                "layer_idx": int(layer_idx),
            }
        )

    def get_layer_kv_with_page_table(self, layer_idx):
        return self.blocked_k, None, self.block_table


class _FakeMaterialization:
    def __init__(self, *, has_v_cache: bool = False):
        self.manager = _FakeMlaGpuManager(has_v_cache=has_v_cache)
        self.backend_state = {}
        self.append_plan = SimpleNamespace(
            cache_seqlens=torch.tensor([9, 10], dtype=torch.int32),
            slot_indices=torch.tensor([1, 0], dtype=torch.int32),
        )
        self.waited_layers = []

    def wait_for_layer(self, layer_idx):
        self.waited_layers.append(int(layer_idx))


def _metadata() -> ForwardBatchMetadata:
    return ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[101, 102],
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 9, 19], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=10,
            q_seq_lens=[1, 2],
            kv_seq_lens=[9, 10],
            position_ids=torch.tensor([8, 8, 9], dtype=torch.int64),
            append_seq_lens=[1, 2],
        ),
    )


def test_projected_mla_prefix_attention_appends_suffix_and_runs_flashinfer(
    monkeypatch,
):
    materialization = _FakeMaterialization()
    query_states = torch.zeros((1, 3, 2, 6), dtype=torch.float32)
    offload_kv = torch.ones((3, 1, 6), dtype=torch.float32)
    expected_output = torch.full((1, 3, 2, 4), 7.0, dtype=torch.float32)
    call = {}

    def fake_flashinfer_extend(**kwargs):
        call.update(kwargs)
        return expected_output

    monkeypatch.setattr(
        flashinfer_extend,
        "run_flashinfer_mla_extend_prefill",
        fake_flashinfer_extend,
    )

    output = run_projected_mla_prefix_attention_from_gpu_pages(
        layer_idx=5,
        query_states=query_states,
        offload_kv=offload_kv,
        metadata=_metadata(),
        spec=MlaExtendSpec(
            num_heads=2,
            kv_lora_rank=4,
            softmax_scale=0.25,
        ),
        materialization=materialization,
    )

    assert output is expected_output
    assert materialization.waited_layers == [5]
    assert len(materialization.manager.append_calls) == 1
    append_call = materialization.manager.append_calls[0]
    assert append_call["k_tensor"] is offload_kv
    assert append_call["v_tensor"] is None
    assert append_call["append_plan"] is materialization.append_plan
    assert append_call["layer_idx"] == 5
    assert call["query_states"].shape == query_states.shape
    assert call["compressed_kv_cache"] is materialization.manager.blocked_k
    assert call["page_table"] is materialization.manager.block_table
    assert call["slot_indices"] is materialization.append_plan.slot_indices
    assert call["cache_seqlens"] is materialization.append_plan.cache_seqlens
    assert call["cu_seqlens_q"].tolist() == [0, 1, 3]
    assert call["kv_lora_rank"] == 4
    assert call["num_heads"] == 2
    assert call["softmax_scale"] == 0.25
    assert call["plan_cache"] is materialization.backend_state


def test_flashinfer_mla_extend_prefill_reuses_materialization_plan(
    monkeypatch,
):
    flashinfer_extend._reset_flashinfer_mla_extend_prefill_cache_for_tests()
    created_wrappers = []

    class FakeWrapper:
        def __init__(self, workspace, backend="auto"):
            self.workspace = workspace
            self.backend = backend
            self.plan_calls = 0
            self.run_calls = 0
            created_wrappers.append(self)

        def plan(self, *args):
            self.plan_calls += 1

        def run(self, q_nope, q_pe, ckv_cache, kpe_cache):
            self.run_calls += 1
            return torch.zeros_like(q_nope)

    monkeypatch.setattr(
        flashinfer_extend,
        "BatchMLAPagedAttentionWrapper",
        FakeWrapper,
    )

    plan_cache = {}
    kwargs = dict(
        query_states=torch.zeros((1, 3, 2, 6), dtype=torch.float32),
        compressed_kv_cache=torch.zeros((4, 8, 1, 6), dtype=torch.float32),
        page_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_indices=torch.tensor([1, 0], dtype=torch.int32),
        cache_seqlens=torch.tensor([9, 10], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
        kv_lora_rank=4,
        num_heads=2,
        softmax_scale=0.25,
        plan_cache=plan_cache,
    )

    flashinfer_extend.run_flashinfer_mla_extend_prefill(**kwargs)
    flashinfer_extend.run_flashinfer_mla_extend_prefill(**kwargs)

    assert len(created_wrappers) == 1
    assert created_wrappers[0].plan_calls == 1
    assert created_wrappers[0].run_calls == 2


def test_projected_mla_prefix_attention_rejects_v_cache_before_append():
    materialization = _FakeMaterialization(has_v_cache=True)

    with pytest.raises(RuntimeError, match="K-only compressed KV"):
        run_projected_mla_prefix_attention_from_gpu_pages(
            layer_idx=5,
            query_states=torch.zeros((1, 1, 2, 6), dtype=torch.float32),
            offload_kv=torch.ones((1, 1, 6), dtype=torch.float32),
            metadata=_metadata(),
            spec=MlaExtendSpec(
                num_heads=2,
                kv_lora_rank=4,
                softmax_scale=0.25,
            ),
            materialization=materialization,
        )

    assert materialization.waited_layers == []
    assert materialization.manager.append_calls == []
