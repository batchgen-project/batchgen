from __future__ import annotations

import pytest
import torch

from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator
from batchgen.kv_cache.gpu_paged_kv_manager import (
	GPUPagedKVCacheManager,
	GPUPagedKVConfig,
)
from batchgen.kv_cache.host_kv_coordinator import HostKVCoordinator


class _FakeHostView:
	def __init__(self, name: str, layer_mapping=None) -> None:
		self.name = name
		self.layer_mapping = layer_mapping or {}
		self.calls = []

	def resolve_physical_layer(self, layer_idx):
		return self.layer_mapping.get(layer_idx, layer_idx)

	def register_sequences(self, sequence_ids) -> None:
		self.calls.append(("register_sequences", list(sequence_ids)))


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


def test_host_kv_coordinator_is_a_lightweight_registry():
	primary = _FakeHostView("primary")
	c4 = _FakeHostView("c4")
	coordinator = HostKVCoordinator()
	coordinator.register_component("primary", primary)
	coordinator.register_component(
		"compressor_c4",
		c4,
		logical_to_physical_layer=[-1, -1, 0, -1, 1],
		token_capacity_scale=0.25,
	)

	assert coordinator.component_names == ["primary", "compressor_c4"]
	assert coordinator.primary is primary
	assert coordinator.get_view("compressor_c4") is c4
	assert coordinator.resolve_physical_layer("compressor_c4", 4) == 1
	assert coordinator.view_layer_id("compressor_c4", 4) == 1
	assert coordinator.map_sequence_tokens(
		"compressor_c4", [(101, 9), (102, 16)]
	) == [(101, 3), (102, 4)]

	with pytest.raises(KeyError):
		coordinator.resolve_physical_layer("compressor_c4", 3)

	coordinator.call_all("register_sequences", [101, 102])
	assert primary.calls == [("register_sequences", [101, 102])]
	assert c4.calls == [("register_sequences", [101, 102])]


def test_host_kv_coordinator_does_not_double_map_mapped_worker_views():
	mapped_view = _FakeHostView("mapped", layer_mapping={4: 1})
	coordinator = HostKVCoordinator()
	coordinator.register_component("mapped", mapped_view)

	assert coordinator.resolve_physical_layer("mapped", 4) == 1
	assert coordinator.view_layer_id("mapped", 4) == 4


def test_gpu_kv_coordinator_keeps_managers_independent():
	primary = _make_gpu_manager(num_layers=3, num_pages=16, page_size_tokens=4)
	c4 = _make_gpu_manager(num_layers=2, num_pages=16, page_size_tokens=2)
	coordinator = GPUKVCoordinator()
	coordinator.register_component("primary", primary)
	coordinator.register_component(
		"compressor_c4",
		c4,
		logical_to_physical_layer={2: 0, 5: 1},
		token_capacity_scale=0.25,
	)

	assert coordinator.primary is primary
	assert coordinator.get_manager("compressor_c4") is c4
	assert coordinator.resolve_physical_layer("compressor_c4", 5) == 1
	assert coordinator.map_token_counts("compressor_c4", [5, 9]) == [2, 3]
	with pytest.raises(KeyError):
		coordinator.resolve_physical_layer("compressor_c4", 4)

	coordinator.call_all("initialize")
	primary.allocate_pages_for_sequences([10, 20], [5, 9])
	c4.allocate_pages_for_sequences(
		[10, 20], coordinator.map_token_counts("compressor_c4", [5, 9])
	)
	primary_table = primary.rebuild_page_table([10, 20])
	c4_table = c4.rebuild_page_table([10, 20])

	assert tuple(primary_table.shape) == (2, 4)
	assert tuple(c4_table.shape) == (2, 3)
	assert primary._sequences[20].pages.numel() == 3
	assert c4._sequences[20].pages.numel() == 2
