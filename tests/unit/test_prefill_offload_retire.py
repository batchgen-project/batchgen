from types import SimpleNamespace

import torch

from batchgen.attention.forward_metadata import (
    ForwardBatchMetadata,
    PrefillAttentionMetadata,
)
from batchgen.models.wrappers.attention import AttnWrapperBase


class _FakeTask:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


class _FakeHostWorkerView:
    def __init__(self):
        self.task = _FakeTask()
        self.range_calls = []

    def async_offload_layer_kv_range_to_host(self, **kwargs):
        self.range_calls.append(kwargs)
        return self.task


class _FakePrefixMaterialization:
    def __init__(self):
        self.finished_layers = []

    def finish_layer(self, layer_idx):
        self.finished_layers.append(int(layer_idx))


def _metadata(*, append_len: int) -> ForwardBatchMetadata:
    return ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[101],
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 6], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=6,
            q_seq_lens=[2],
            kv_seq_lens=[6],
            position_ids=torch.tensor([4, 5], dtype=torch.int64),
            append_seq_lens=[append_len],
        ),
    )


def _reset_pending_state() -> None:
    AttnWrapperBase.pending_prefill_offload_tasks = []
    AttnWrapperBase.pending_prefill_offload_tensors = []
    AttnWrapperBase.pending_prefill_offload_layer_idx = None
    AttnWrapperBase.prefill_prefix_materialization = None


def test_prefix_reuse_finish_layer_waits_for_tracked_prefill_offload():
    _reset_pending_state()
    host_view = _FakeHostWorkerView()
    materialization = _FakePrefixMaterialization()
    wrapper = object.__new__(AttnWrapperBase)
    wrapper.layer_idx = 7
    wrapper.core_engine = SimpleNamespace(host_paged_kv_worker_view=host_view)
    AttnWrapperBase.prefill_prefix_materialization = materialization

    key = torch.ones(2, 1, 4)
    value = torch.ones(2, 1, 4)
    wrapper.offload_prepacked_gqa_kv(
        key,
        value,
        metadata=_metadata(append_len=2),
        track_tasks=False,
    )

    assert materialization.finished_layers == []
    assert AttnWrapperBase.pending_prefill_offload_layer_idx == 7
    assert len(AttnWrapperBase.pending_prefill_offload_tasks) == 1
    assert len(AttnWrapperBase.pending_prefill_offload_tensors) >= 2
    assert host_view.range_calls[0]["raw_start_positions"] == [4]
    assert host_view.range_calls[0]["token_counts"] == [2]

    AttnWrapperBase.retire_pending_prefill_offloads(device=None)

    assert host_view.task.wait_calls == 1
    assert materialization.finished_layers == [7]
    assert AttnWrapperBase.pending_prefill_offload_layer_idx is None
    assert AttnWrapperBase.pending_prefill_offload_tasks == []
    assert AttnWrapperBase.pending_prefill_offload_tensors == []

    _reset_pending_state()


def test_prefix_reuse_zero_append_finishes_layer_on_retire():
    _reset_pending_state()
    host_view = _FakeHostWorkerView()
    materialization = _FakePrefixMaterialization()
    wrapper = object.__new__(AttnWrapperBase)
    wrapper.layer_idx = 3
    wrapper.core_engine = SimpleNamespace(host_paged_kv_worker_view=host_view)
    AttnWrapperBase.prefill_prefix_materialization = materialization

    key = torch.ones(2, 1, 4)
    value = torch.ones(2, 1, 4)
    wrapper.offload_prepacked_gqa_kv(
        key,
        value,
        metadata=_metadata(append_len=0),
        track_tasks=False,
    )

    assert host_view.range_calls == []
    assert materialization.finished_layers == []
    assert AttnWrapperBase.pending_prefill_offload_layer_idx == 3
    assert AttnWrapperBase.pending_prefill_offload_tasks == []
    assert len(AttnWrapperBase.pending_prefill_offload_tensors) == 2

    AttnWrapperBase.retire_pending_prefill_offloads(device=None)

    assert materialization.finished_layers == [3]
    assert AttnWrapperBase.pending_prefill_offload_layer_idx is None

    _reset_pending_state()
