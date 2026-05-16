from __future__ import annotations

import pytest
import torch

from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator
from batchgen.kv_cache.gpu_paged_kv_manager import (
	GPUPagedKVCacheManager,
	GPUPagedKVConfig,
)
from batchgen.kv_cache.host_kv_coordinator import HostKVCoordinator
from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator


class _FakeTask:
	def __init__(self) -> None:
		self.waited = False

	def wait(self) -> None:
		self.waited = True


class _FakeHostView:
	def __init__(self, name: str, layer_mapping=None) -> None:
		self.name = name
		self.layer_mapping = layer_mapping or {}
		self.calls = []

	def initialize(self, *args, **kwargs) -> None:
		self.calls.append(("initialize", args, kwargs))

	def shutdown(self) -> None:
		self.calls.append(("shutdown",))

	def register_sequences(self, sequence_ids) -> None:
		self.calls.append(("register_sequences", list(sequence_ids)))

	def unregister_sequences(self, sequence_ids) -> None:
		self.calls.append(("unregister_sequences", list(sequence_ids)))

	def release_sequence_pages(self, sequence_ids) -> None:
		self.calls.append(("release_sequence_pages", list(sequence_ids)))

	def allocate_pages_for_sequences(self, seq_token_pairs):
		pairs = list(seq_token_pairs)
		self.calls.append(("allocate_pages_for_sequences", pairs))
		return {seq_id: tokens for seq_id, tokens in pairs}

	def grow_pages_for_sequences(self, seq_page_pairs):
		pairs = list(seq_page_pairs)
		self.calls.append(("grow_pages_for_sequences", pairs))
		return {seq_id: pages for seq_id, pages in pairs}

	def build_page_table(self, sequence_ids):
		self.calls.append(("build_page_table", list(sequence_ids)))
		return [list(sequence_ids), self.name]

	def get_stats(self):
		return f"stats:{self.name}"

	def resolve_physical_layer(self, layer_idx):
		return self.layer_mapping.get(layer_idx, layer_idx)

	def get_sequence_layer_page_pointers(self, sequence_id, layer_idx, max_tokens=None):
		self.calls.append(
			("get_sequence_layer_page_pointers", sequence_id, layer_idx, max_tokens)
		)
		return [layer_idx], None

	def async_offload_layer_kv_to_host(
		self, layer_idx, sequence_ids, k_tensor, v_tensor, sequence_lengths
	):
		self.calls.append(
			(
				"async_offload_layer_kv_to_host",
				layer_idx,
				list(sequence_ids),
				k_tensor,
				v_tensor,
				sequence_lengths,
			)
		)
		return _FakeTask()

	def async_append_decode_kv_to_host(
		self, layer_idx, sequence_ids, k_tensor, v_tensor, sequence_lengths
	):
		self.calls.append(
			(
				"async_append_decode_kv_to_host",
				layer_idx,
				list(sequence_ids),
				k_tensor,
				v_tensor,
				sequence_lengths,
			)
		)
		return _FakeTask()

	def async_load_layer_paged_kv_to_device(self, **kwargs):
		self.calls.append(("async_load_layer_paged_kv_to_device", kwargs))
		return _FakeTask()


def _make_gpu_manager(
	*, num_layers: int, num_pages: int, page_size_tokens: int
) -> GPUPagedKVCacheManager:
	config = GPUPagedKVConfig(
		num_layers=num_layers,
		num_pages=num_pages,
		page_size_tokens=page_size_tokens,
		num_k_heads=1,
		k_head_dim=2,
		num_v_heads=0,
		v_head_dim=0,
		kv_dtype=torch.float32,
	)
	return GPUPagedKVCacheManager(config=config, device="cpu")


def test_host_kv_coordinator_routes_component_layer_and_scaled_capacity():
	primary = _FakeHostView("primary")
	c4 = _FakeHostView("c4")
	coordinator = HostKVCoordinator(primary_component_name="primary")
	coordinator.register_component("primary", primary)
	coordinator.register_component(
		"compressor_c4",
		c4,
		logical_to_physical_layer=[-1, -1, 0, -1, 1],
		token_capacity_scale=0.25,
	)

	coordinator.initialize(0, create_region=False)
	coordinator.register_sequences([101, 102])
	result = coordinator.allocate_pages_for_sequences([(101, 9), (102, 16)])

	assert result == {101: 9, 102: 16}
	assert ("allocate_pages_for_sequences", [(101, 9), (102, 16)]) in primary.calls
	assert ("allocate_pages_for_sequences", [(101, 3), (102, 4)]) in c4.calls
	assert coordinator.resolve_physical_layer("compressor_c4", 4) == 1
	with pytest.raises(KeyError):
		coordinator.resolve_physical_layer("compressor_c4", 3)

	task = coordinator.async_offload_component_layer_to_host(
		"compressor_c4",
		layer_idx=4,
		sequence_ids=[101],
		k_tensor="k",
		v_tensor=None,
		sequence_lengths="lens",
	)
	task.wait()
	assert task.waited
	assert (
		"async_offload_layer_kv_to_host",
		1,
		[101],
		"k",
		None,
		"lens",
	) in c4.calls


def test_host_kv_coordinator_does_not_double_map_mapped_worker_views():
	mapped_view = _FakeHostView("mapped", layer_mapping={4: 1})
	coordinator = HostKVCoordinator(primary_component_name="mapped")
	coordinator.register_component("mapped", mapped_view)

	assert coordinator.resolve_physical_layer("mapped", 4) == 1
	coordinator.async_append_decode_component_to_host(
		"mapped",
		layer_idx=4,
		sequence_ids=[7],
		k_tensor="k",
		v_tensor=None,
		sequence_lengths="lens",
	)

	assert (
		"async_append_decode_kv_to_host",
		4,
		[7],
		"k",
		None,
		"lens",
	) in mapped_view.calls


def test_gpu_kv_coordinator_allows_independent_component_page_tables():
	primary = _make_gpu_manager(num_layers=3, num_pages=16, page_size_tokens=4)
	c4 = _make_gpu_manager(num_layers=2, num_pages=16, page_size_tokens=2)
	coordinator = GPUKVCoordinator(primary_component_name="primary")
	coordinator.register_component("primary", primary)
	coordinator.register_component(
		"compressor_c4",
		c4,
		logical_to_physical_layer={2: 0, 5: 1},
		token_capacity_scale=0.25,
	)

	coordinator.initialize()
	allocations = coordinator.allocate_pages_for_sequences([10, 20], [5, 9])
	primary_table = coordinator.rebuild_page_table([10, 20])
	c4_table = c4._gpu_page_table_manager.gpu_table

	assert allocations[10] and allocations[20]
	assert tuple(primary_table.shape) == (2, 4)
	assert tuple(c4_table.shape) == (2, 3)
	assert primary._sequences[10].pages.numel() == 2
	assert primary._sequences[20].pages.numel() == 3
	assert c4._sequences[10].pages.numel() == 1
	assert c4._sequences[20].pages.numel() == 2
	assert coordinator.resolve_physical_layer("compressor_c4", 5) == 1
	with pytest.raises(KeyError):
		coordinator.resolve_physical_layer("compressor_c4", 4)

	k_cache, v_cache, table = coordinator.get_layer_kv_with_page_table(
		"compressor_c4", 5
	)
	assert v_cache is None
	assert k_cache.data_ptr() == c4._k_cache[1].data_ptr()
	assert table.data_ptr() == c4_table.data_ptr()
	assert coordinator.get_stats().num_used_pages == 5
	assert coordinator.get_component_stats()["compressor_c4"].num_used_pages == 3


def test_dual_gpu_coordinator_remains_a_mirrored_special_case():
	primary = _make_gpu_manager(num_layers=1, num_pages=8, page_size_tokens=4)
	auxiliary = _make_gpu_manager(num_layers=1, num_pages=8, page_size_tokens=4)
	dual = DualKVCacheCoordinator(primary, auxiliary)

	assert dual.component_names == ["primary", "auxiliary"]
	dual.initialize()
	dual.allocate_pages_for_sequences([1, 2], [5, 8])
	primary_table = dual.rebuild_page_table([1, 2])
	aux_table = auxiliary._gpu_page_table_manager.gpu_table

	assert torch.equal(primary_table, aux_table)
	assert dual.get_component("primary").manager is primary
	assert dual.get_component("auxiliary").manager is auxiliary
