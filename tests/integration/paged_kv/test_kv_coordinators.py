from __future__ import annotations

import pytest
import torch

from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator
from batchgen.kv_cache.gpu_paged_kv_manager import (
	GPUPagedKVCacheManager,
	GPUPagedKVConfig,
)
from batchgen.kv_cache.host_kv_coordinator import HostKVCoordinator
from batchgen.kv_cache.deepseek_v4_kv_coordinator import (
	COMPRESSOR_C4,
	COMPRESSOR_C128,
	INDEXER_C4,
	PRIMARY_MLA,
	SWA,
	DeepSeekV4GPUKVCoordinator,
	DeepSeekV4HostKVCoordinator,
	DeepSeekV4KVLayout,
)


class _FakeHostView:
	def __init__(self, name: str, layer_mapping=None, *, mapped: bool = False) -> None:
		self.name = name
		self.layer_mapping = layer_mapping or {}
		self.calls = []
		self.uses_logical_layer_mapping = mapped

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


def test_deepseek_v4_layout_builds_compact_component_maps():
	layout = DeepSeekV4KVLayout.from_compression_ratios(
		[0, 128, 4, 128, 4],
		sliding_window=128,
	)

	assert layout.num_layers == 5
	assert layout.c4_layer_map == (-1, -1, 0, -1, 1)
	assert layout.c128_layer_map == (-1, 0, -1, 1, -1)
	assert layout.num_c4_layers == 2
	assert layout.num_c128_layers == 2
	assert layout.physical_layer_count(PRIMARY_MLA) == 5
	assert layout.physical_layer_count(SWA) == 5
	assert layout.token_capacity(COMPRESSOR_C4, 65) == 17
	assert layout.token_capacity(COMPRESSOR_C128, 257) == 3
	assert layout.token_capacity(INDEXER_C4, 9) == 3
	assert layout.token_capacity(SWA, 4096) == 128

	with pytest.raises(ValueError):
		DeepSeekV4KVLayout.from_compression_ratios([0, 8, 4])


def test_deepseek_v4_host_coordinator_registers_dsv4_components():
	primary = _FakeHostView("primary")
	swa = _FakeHostView("swa")
	c4 = _FakeHostView("c4")
	c128 = _FakeHostView("c128")
	indexer = _FakeHostView("indexer")

	coordinator = DeepSeekV4HostKVCoordinator(
		compression_ratios=[0, 128, 4, 128, 4],
		primary_mla=primary,
		swa=swa,
		compressor_c4=c4,
		compressor_c128=c128,
		indexer_c4=indexer,
		sliding_window=128,
	)

	assert coordinator.component_names == [
		PRIMARY_MLA,
		SWA,
		COMPRESSOR_C4,
		COMPRESSOR_C128,
		INDEXER_C4,
	]
	assert coordinator.primary_mla is primary
	assert coordinator.swa is swa
	assert coordinator.compressor_c4 is c4
	assert coordinator.compressor_c128 is c128
	assert coordinator.indexer_c4 is indexer
	assert coordinator.resolve_physical_layer(COMPRESSOR_C4, 4) == 1
	assert coordinator.resolve_physical_layer(COMPRESSOR_C128, 3) == 1
	assert coordinator.resolve_physical_layer(INDEXER_C4, 2) == 0
	assert coordinator.map_sequence_tokens(
		COMPRESSOR_C4, [(101, 65), (102, 128)]
	) == [(101, 17), (102, 32)]
	assert coordinator.map_sequence_tokens(COMPRESSOR_C128, [(101, 257)]) == [
		(101, 3)
	]
	assert coordinator.map_sequence_tokens(SWA, [(101, 4096)]) == [(101, 128)]

	with pytest.raises(KeyError):
		coordinator.resolve_physical_layer(COMPRESSOR_C4, 3)


def test_deepseek_v4_host_coordinator_keeps_mapped_view_layer_ids_logical():
	c4 = _FakeHostView("mapped_c4", {2: 0, 4: 1}, mapped=True)
	coordinator = DeepSeekV4HostKVCoordinator(
		compression_ratios=[0, 128, 4, 128, 4],
		primary_mla=_FakeHostView("primary"),
		compressor_c4=c4,
	)

	assert coordinator.resolve_physical_layer(COMPRESSOR_C4, 4) == 1
	assert coordinator.view_layer_id(COMPRESSOR_C4, 4) == 4


def test_deepseek_v4_gpu_coordinator_registers_dsv4_components():
	primary = _make_gpu_manager(num_layers=5, num_pages=32, page_size_tokens=4)
	swa = _make_gpu_manager(num_layers=5, num_pages=32, page_size_tokens=4)
	c4 = _make_gpu_manager(num_layers=2, num_pages=32, page_size_tokens=4)
	c128 = _make_gpu_manager(num_layers=2, num_pages=32, page_size_tokens=4)
	indexer = _make_gpu_manager(num_layers=2, num_pages=32, page_size_tokens=4)

	coordinator = DeepSeekV4GPUKVCoordinator(
		compression_ratios=[0, 128, 4, 128, 4],
		primary_mla=primary,
		swa=swa,
		compressor_c4=c4,
		compressor_c128=c128,
		indexer_c4=indexer,
		sliding_window=128,
	)

	assert coordinator.get_manager(PRIMARY_MLA) is primary
	assert coordinator.get_manager(COMPRESSOR_C4) is c4
	assert coordinator.resolve_physical_layer(COMPRESSOR_C4, 4) == 1
	assert coordinator.resolve_physical_layer(COMPRESSOR_C128, 1) == 0
	assert coordinator.resolve_physical_layer(INDEXER_C4, 2) == 0
	assert coordinator.map_token_counts(COMPRESSOR_C4, [65, 128]) == [17, 32]
	assert coordinator.map_token_counts(COMPRESSOR_C128, [257]) == [3]
	assert coordinator.map_token_counts(SWA, [64, 4096]) == [64, 128]

	coordinator.call_all("initialize")
	primary.allocate_pages_for_sequences([10], [65])
	c4.allocate_pages_for_sequences(
		[10], coordinator.map_token_counts(COMPRESSOR_C4, [65])
	)
	c128.allocate_pages_for_sequences(
		[10], coordinator.map_token_counts(COMPRESSOR_C128, [65])
	)

	assert primary._sequences[10].pages.numel() == 17
	assert c4._sequences[10].pages.numel() == 5
	assert c128._sequences[10].pages.numel() == 1
