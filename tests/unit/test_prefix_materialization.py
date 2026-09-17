from types import SimpleNamespace

import pytest
import torch

from batchgen.prefix_reuse.materialization import (
    materialize_gpt_oss_prefixes,
)


class _Task:
    def __init__(self):
        self.layers = []
        self.waited = False

    def wait_for_layer(self, layer_idx):
        self.layers.append(int(layer_idx))

    def wait(self):
        self.waited = True


class _Manager:
    def __init__(self, page_size=4):
        self.config = SimpleNamespace(
            page_size_tokens=page_size,
            num_k_heads=1,
            k_head_dim=2,
            num_v_heads=1,
            v_head_dim=2,
            kv_dtype=torch.bfloat16,
        )
        self.allocated = None
        self.rebuilt = None
        self.destroyed = False
        self.plan = SimpleNamespace(cache_seqlens=torch.tensor([7, 5]))

    def allocate_pages_for_sequences(self, sequence_ids, lengths):
        self.allocated = (list(sequence_ids), list(lengths))

    def rebuild_page_table(self, sequence_ids):
        self.rebuilt = list(sequence_ids)

    def prepare_prefill_suffix_append(self, **kwargs):
        self.plan_kwargs = kwargs
        return self.plan

    def get_padded_3d_page_pointers(self):
        return torch.tensor([[[100, 200], [300, 400]]]), torch.tensor(
            [[[500, 600], [700, 800]]]
        )

    def destroy(self, *, empty_cuda_cache=False):
        self.destroyed = True
        self.empty_cuda_cache = empty_cuda_cache


class _Host:
    def __init__(self):
        self.task = _Task()

    def async_load_prefix_pages_to_device(self, **kwargs):
        self.kwargs = kwargs
        return self.task


class _Coordinator:
    def __init__(self):
        self.begun = []
        self.ended = []

    def begin_attachment_load(self, handle):
        self.begun.append(int(handle))

    def end_attachment_load(self, handle):
        self.ended.append(int(handle))


def _lookup(handle, pages):
    return SimpleNamespace(
        attachment_handle=handle,
        materialization_spans=[
            SimpleNamespace(
                group_id=0,
                pages=[SimpleNamespace(page_id=page) for page in pages],
            )
        ]
        if pages
        else [],
    )


def test_materializes_mixed_hit_and_miss_and_releases_load_protection():
    manager = _Manager()
    host = _Host()
    coordinator = _Coordinator()

    materialization = materialize_gpt_oss_prefixes(
        gpu_manager=manager,
        host_worker_view=host,
        coordinator=coordinator,
        lookup_results=[_lookup(9, [101, 102]), _lookup(0, [])],
        sequence_ids=[11, 12],
        prompt_lengths=[7, 5],
        compute_cached_tokens=[6, 0],
        raw_page_tokens=4,
    )

    assert manager.allocated == ([11, 12], [7, 5])
    assert manager.plan_kwargs["prefix_lens"] == [6, 0]
    assert manager.plan_kwargs["suffix_lens"] == [1, 5]
    assert coordinator.begun == [9]
    assert host.kwargs["active_page_counts"].tolist() == [2, 0]
    assert host.kwargs["host_page_ids"].tolist() == [[101, 102], [0, 0]]

    materialization.wait_for_layer(3)
    materialization.close(empty_cuda_cache=True)
    assert host.task.layers == [3]
    assert host.task.waited
    assert coordinator.ended == [9]
    assert manager.destroyed
    assert manager.empty_cuda_cache


def test_materialization_maps_host_pages_into_larger_gpu_pages():
    manager = _Manager(page_size=8)
    host = _Host()

    materialization = materialize_gpt_oss_prefixes(
        gpu_manager=manager,
        host_worker_view=host,
        coordinator=_Coordinator(),
        lookup_results=[_lookup(1, [7, 8, 9])],
        sequence_ids=[11],
        prompt_lengths=[13],
        compute_cached_tokens=[12],
        raw_page_tokens=4,
    )

    # One Host page is 4 tokens * 1 head * dim 2 * bf16 = 16 bytes.
    assert host.kwargs["k_device_ptrs"].tolist() == [[[100, 116, 200]]]
    assert host.kwargs["v_device_ptrs"].tolist() == [[[500, 516, 600]]]
    materialization.close()


def test_materialization_rejects_non_multiple_page_sizes():
    with pytest.raises(ValueError, match="must be a multiple"):
        materialize_gpt_oss_prefixes(
            gpu_manager=_Manager(page_size=6),
            host_worker_view=_Host(),
            coordinator=_Coordinator(),
            lookup_results=[_lookup(1, [7])],
            sequence_ids=[11],
            prompt_lengths=[5],
            compute_cached_tokens=[4],
            raw_page_tokens=4,
        )
