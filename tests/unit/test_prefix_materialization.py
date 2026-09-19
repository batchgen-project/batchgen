from types import SimpleNamespace

import pytest
import torch

from batchgen.prefix_reuse.materialization import (
    materialize_gpt_oss_prefixes,
)


class _Task:
    def __init__(self):
        self.wait_count = 0
        self.done_count = 0
        self.is_done = False
        self.fail_on_wait = False

    def done(self):
        self.done_count += 1
        return self.is_done

    def wait(self):
        self.wait_count += 1
        if self.fail_on_wait:
            raise RuntimeError("load failed")
        self.is_done = True


class _Manager:
    def __init__(self, page_size=4):
        self.config = SimpleNamespace(
            page_size_tokens=page_size,
            num_layers=2,
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
        collect_metrics=True,
        host_page_bytes_all_layers=64,
    )

    assert manager.allocated == ([11, 12], [7, 5])
    assert manager.plan_kwargs["prefix_lens"] == [6, 0]
    assert manager.plan_kwargs["suffix_lens"] == [1, 5]
    assert coordinator.begun == [9]
    assert host.kwargs["active_page_counts"].tolist() == [2, 0]
    assert host.kwargs["host_page_ids"].tolist() == [[101, 102], [0, 0]]

    materialization.wait_for_layer(3)
    materialization.wait_for_layer(4)
    materialization.close(empty_cuda_cache=True)
    assert host.task.wait_count == 1
    assert coordinator.ended == [9]
    assert manager.destroyed
    assert manager.empty_cuda_cache
    metrics = materialization.take_metrics_fields()
    assert metrics is not None
    assert metrics["load_status"] == "complete"
    assert metrics["done_before_wait"] is False
    assert metrics["sequences"] == 2
    assert metrics["cached_tokens"] == 6
    assert metrics["host_pages"] == 2
    # 2 pages * 2 layers * (16-byte K page + 16-byte V page).
    assert metrics["host_bytes"] == 128
    assert metrics["wait_s"] >= 0.0
    assert metrics["launch_to_wait_return_s"] >= metrics["wait_s"]
    assert materialization.take_metrics_fields() is None


def test_metrics_disabled_does_not_poll_or_measure_task():
    host = _Host()
    materialization = materialize_gpt_oss_prefixes(
        gpu_manager=_Manager(),
        host_worker_view=host,
        coordinator=_Coordinator(),
        lookup_results=[_lookup(9, [101])],
        sequence_ids=[11],
        prompt_lengths=[5],
        compute_cached_tokens=[4],
        raw_page_tokens=4,
    )

    materialization.close()

    assert host.task.wait_count == 1
    assert host.task.done_count == 0
    assert materialization.take_metrics_fields() is None


def test_metrics_record_precompleted_and_failed_loads_once():
    completed_host = _Host()
    completed = materialize_gpt_oss_prefixes(
        gpu_manager=_Manager(),
        host_worker_view=completed_host,
        coordinator=_Coordinator(),
        lookup_results=[_lookup(9, [101])],
        sequence_ids=[11],
        prompt_lengths=[5],
        compute_cached_tokens=[4],
        raw_page_tokens=4,
        collect_metrics=True,
        host_page_bytes_all_layers=64,
    )
    completed_host.task.is_done = True
    completed.close()
    completed_metrics = completed.take_metrics_fields()
    assert completed_metrics is not None
    assert completed_metrics["done_before_wait"] is True
    assert completed_metrics["load_status"] == "complete"

    failed_host = _Host()
    failed_host.task.fail_on_wait = True
    failed = materialize_gpt_oss_prefixes(
        gpu_manager=_Manager(),
        host_worker_view=failed_host,
        coordinator=_Coordinator(),
        lookup_results=[_lookup(9, [101])],
        sequence_ids=[11],
        prompt_lengths=[5],
        compute_cached_tokens=[4],
        raw_page_tokens=4,
        collect_metrics=True,
        host_page_bytes_all_layers=64,
    )
    with pytest.raises(RuntimeError, match="load failed"):
        failed.close()
    failed_metrics = failed.take_metrics_fields()
    assert failed_metrics is not None
    assert failed_metrics["load_status"] == "failed"
    assert failed_metrics["wait_s"] >= 0.0
    assert failed.take_metrics_fields() is None


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
    assert host.kwargs["k_device_ptrs"].tolist() == [
        [[100, 116, 200], [300, 316, 400]]
    ]
    assert host.kwargs["v_device_ptrs"].tolist() == [
        [[500, 516, 600], [700, 716, 800]]
    ]
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
