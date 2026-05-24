from types import SimpleNamespace

import pytest
import torch

from batchgen.prefix_reuse.materialization import (
    PrefixMaterializationSequence,
    materialize_single_group_lookup_results,
    materialize_single_group_prefix_pages,
)


class _FakeTask:
    def __init__(self):
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


class _FakeHostWorkerView:
    def __init__(self):
        self.task = _FakeTask()
        self.calls = []

    def async_load_prefix_pages_to_device(self, **kwargs):
        self.calls.append(kwargs)
        return self.task


class _FailingHostWorkerView(_FakeHostWorkerView):
    def async_load_prefix_pages_to_device(self, **kwargs):
        super().async_load_prefix_pages_to_device(**kwargs)
        raise RuntimeError("load failed")


class _FakePrefixCoordinator:
    def __init__(self):
        self.begin_calls = []
        self.end_calls = []

    def begin_attachment_load(self, attachment_handle):
        self.begin_calls.append(int(attachment_handle))

    def end_attachment_load(self, attachment_handle):
        self.end_calls.append(int(attachment_handle))


class _FakeGpuManager:
    def __init__(self):
        self.config = SimpleNamespace(page_size_tokens=4)
        self.allocations = []
        self.rebuilt = []
        self.prepared = []
        self.k_ptrs = torch.ones((2, 2, 3), dtype=torch.int64)
        self.v_ptrs = torch.ones((2, 2, 3), dtype=torch.int64) * 2
        self.append_plan = SimpleNamespace(
            cache_seqlens=torch.tensor([7, 3], dtype=torch.int32),
            slot_indices=torch.tensor([0, 1], dtype=torch.int32),
            slot_values=(0, 1),
        )

    def allocate_pages_for_sequences(self, sequence_ids, num_tokens):
        self.allocations.append((list(sequence_ids), list(num_tokens)))

    def rebuild_page_table(self, sequence_ids):
        self.rebuilt.append(list(sequence_ids))

    def get_padded_3d_page_pointers(self):
        return self.k_ptrs, self.v_ptrs

    def prepare_prefill_suffix_append(
        self,
        *,
        sequence_ids,
        prefix_lens,
        suffix_lens,
        rebuild_page_table,
    ):
        self.prepared.append(
            (
                list(sequence_ids),
                list(prefix_lens),
                list(suffix_lens),
                rebuild_page_table,
            )
        )
        return self.append_plan


class _FailingAppendPlanGpuManager(_FakeGpuManager):
    def prepare_prefill_suffix_append(self, **kwargs):
        super().prepare_prefill_suffix_append(**kwargs)
        raise RuntimeError("append plan failed")


def test_materialize_single_group_prefix_pages_starts_page_id_load():
    gpu_manager = _FakeGpuManager()
    host_view = _FakeHostWorkerView()

    materialization = materialize_single_group_prefix_pages(
        gpu_manager=gpu_manager,
        host_worker_view=host_view,
        sequences=[
            PrefixMaterializationSequence(
                sequence_id=101,
                prefix_tokens=5,
                suffix_tokens=2,
                host_pages=[11, 12],
            ),
            PrefixMaterializationSequence(
                sequence_id=102,
                prefix_tokens=0,
                suffix_tokens=3,
                host_pages=[],
            ),
        ],
    )

    assert materialization.manager is gpu_manager
    assert materialization.append_plan is gpu_manager.append_plan
    assert gpu_manager.allocations == [([101, 102], [7, 3])]
    assert gpu_manager.rebuilt == [[101, 102]]
    assert gpu_manager.prepared == [([101, 102], [5, 0], [2, 3], False)]
    assert len(host_view.calls) == 1
    call = host_view.calls[0]
    assert call["host_page_ids"].tolist() == [[11, 12], [0, 0]]
    assert call["active_page_counts"].tolist() == [2, 0]
    assert call["k_device_ptrs"] is gpu_manager.k_ptrs
    assert call["v_device_ptrs"] is gpu_manager.v_ptrs

    materialization.wait_for_layer(0)
    materialization.wait_for_layer(1)
    assert host_view.task.wait_count == 1


def test_materialize_single_group_prefix_pages_skips_load_for_all_miss():
    gpu_manager = _FakeGpuManager()
    host_view = _FakeHostWorkerView()

    materialization = materialize_single_group_prefix_pages(
        gpu_manager=gpu_manager,
        host_worker_view=host_view,
        sequences=[
            PrefixMaterializationSequence(
                sequence_id=101,
                prefix_tokens=0,
                suffix_tokens=3,
                host_pages=[],
            ),
        ],
    )

    assert host_view.calls == []
    materialization.wait_for_layer(0)
    assert gpu_manager.allocations == [([101], [3])]


def test_materialize_single_group_prefix_pages_rejects_wrong_host_region():
    handle = SimpleNamespace(host_region_id=3, page_id=11)
    gpu_manager = _FakeGpuManager()
    with pytest.raises(ValueError, match="expected region"):
        materialize_single_group_prefix_pages(
            gpu_manager=gpu_manager,
            host_worker_view=_FakeHostWorkerView(),
            sequences=[
                PrefixMaterializationSequence(
                    sequence_id=101,
                    prefix_tokens=4,
                    suffix_tokens=1,
                    host_pages=[handle],
                ),
            ],
        )
    assert gpu_manager.allocations == []


def test_materialize_single_group_prefix_pages_guards_attachment_load():
    gpu_manager = _FakeGpuManager()
    host_view = _FakeHostWorkerView()
    coordinator = _FakePrefixCoordinator()

    materialization = materialize_single_group_prefix_pages(
        gpu_manager=gpu_manager,
        host_worker_view=host_view,
        prefix_cache_coordinator=coordinator,
        sequences=[
            PrefixMaterializationSequence(
                sequence_id=101,
                prefix_tokens=4,
                suffix_tokens=1,
                host_pages=[11],
                attachment_handle=91,
            ),
            PrefixMaterializationSequence(
                sequence_id=102,
                prefix_tokens=4,
                suffix_tokens=1,
                host_pages=[12],
                attachment_handle=91,
            ),
        ],
    )

    assert coordinator.begin_calls == [91]
    assert coordinator.end_calls == []
    materialization.wait_for_layer(0)
    materialization.wait_for_layer(1)
    assert host_view.task.wait_count == 1
    assert coordinator.end_calls == [91]


def test_materialize_single_group_prefix_pages_unwinds_attachment_on_load_error():
    gpu_manager = _FakeGpuManager()
    coordinator = _FakePrefixCoordinator()

    with pytest.raises(RuntimeError, match="load failed"):
        materialize_single_group_prefix_pages(
            gpu_manager=gpu_manager,
            host_worker_view=_FailingHostWorkerView(),
            prefix_cache_coordinator=coordinator,
            sequences=[
                PrefixMaterializationSequence(
                    sequence_id=101,
                    prefix_tokens=4,
                    suffix_tokens=1,
                    host_pages=[11],
                    attachment_handle=91,
                ),
            ],
        )

    assert coordinator.begin_calls == [91]
    assert coordinator.end_calls == [91]


def test_materialize_single_group_prefix_pages_does_not_load_before_append_plan():
    host_view = _FakeHostWorkerView()
    coordinator = _FakePrefixCoordinator()

    with pytest.raises(RuntimeError, match="append plan failed"):
        materialize_single_group_prefix_pages(
            gpu_manager=_FailingAppendPlanGpuManager(),
            host_worker_view=host_view,
            prefix_cache_coordinator=coordinator,
            sequences=[
                PrefixMaterializationSequence(
                    sequence_id=101,
                    prefix_tokens=4,
                    suffix_tokens=1,
                    host_pages=[11],
                    attachment_handle=91,
                ),
            ],
        )

    assert host_view.calls == []
    assert coordinator.begin_calls == []
    assert coordinator.end_calls == []


def test_materialize_single_group_lookup_results_builds_sequences():
    gpu_manager = _FakeGpuManager()
    host_view = _FakeHostWorkerView()
    coordinator = _FakePrefixCoordinator()
    lookup_result = SimpleNamespace(
        attachment_handle=91,
        common_cached_tokens=5,
        materialization_spans=[
            SimpleNamespace(
                group_id=7,
                raw_end_token=5,
                pages=[
                    SimpleNamespace(host_region_id=0, page_id=11),
                    SimpleNamespace(host_region_id=0, page_id=12),
                ],
            )
        ],
    )

    materialization = materialize_single_group_lookup_results(
        gpu_manager=gpu_manager,
        host_worker_view=host_view,
        prefix_cache_coordinator=coordinator,
        lookup_results=[lookup_result],
        sequence_ids=[101],
        prompt_lengths=[7],
        group_id=7,
    )

    assert materialization.append_plan is gpu_manager.append_plan
    assert gpu_manager.allocations == [([101], [7])]
    assert gpu_manager.prepared == [([101], [5], [2], False)]
    assert host_view.calls[0]["host_page_ids"].tolist() == [[11, 12]]
    assert host_view.calls[0]["active_page_counts"].tolist() == [2]
    assert coordinator.begin_calls == [91]
    materialization.wait()
    assert coordinator.end_calls == [91]


def test_materialize_single_group_lookup_results_rejects_mismatched_span():
    lookup_result = SimpleNamespace(
        attachment_handle=91,
        common_cached_tokens=5,
        materialization_spans=[
            SimpleNamespace(group_id=7, raw_end_token=4, pages=[11])
        ],
    )

    with pytest.raises(ValueError, match="cached token boundary"):
        materialize_single_group_lookup_results(
            gpu_manager=_FakeGpuManager(),
            host_worker_view=_FakeHostWorkerView(),
            lookup_results=[lookup_result],
            sequence_ids=[101],
            prompt_lengths=[7],
            group_id=7,
        )


def test_materialize_single_group_lookup_results_requires_attachment_for_hit():
    lookup_result = SimpleNamespace(
        attachment_handle=0,
        common_cached_tokens=5,
        materialization_spans=[
            SimpleNamespace(group_id=7, raw_end_token=5, pages=[11])
        ],
    )

    with pytest.raises(ValueError, match="attachment_handle"):
        materialize_single_group_lookup_results(
            gpu_manager=_FakeGpuManager(),
            host_worker_view=_FakeHostWorkerView(),
            lookup_results=[lookup_result],
            sequence_ids=[101],
            prompt_lengths=[7],
            group_id=7,
        )
