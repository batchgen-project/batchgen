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
	SWA,
	DeepSeekV4GPUKVCoordinator,
	DeepSeekV4HostKVCoordinator,
	DeepSeekV4KVLayout,
)


class _FakeTask:
	def __init__(self, name: str) -> None:
		self.name = name
		self.waited = False

	def wait(self) -> None:
		self.waited = True


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

	def async_offload_layer_kv_to_host(
		self, layer_idx, sequence_ids, k_tensor, v_tensor=None, sequence_lengths=None
	):
		self.calls.append(
			(
				"async_offload_layer_kv_to_host",
				int(layer_idx),
				list(sequence_ids),
				k_tensor,
				v_tensor,
				list(sequence_lengths),
			)
		)
		return _FakeTask(f"{self.name}:offload")

	def async_append_decode_kv_to_host(
		self, layer_idx, sequence_ids, k_tensor, v_tensor=None, sequence_lengths=None
	):
		self.calls.append(
			(
				"async_append_decode_kv_to_host",
				int(layer_idx),
				list(sequence_ids),
				k_tensor,
				v_tensor,
				list(sequence_lengths),
			)
		)
		return _FakeTask(f"{self.name}:append")

	def async_append_decode_kv_to_host_batched_kernel(
		self, entries, sequence_ids, sequence_lengths
	):
		self.calls.append(
			(
				"async_append_decode_kv_to_host_batched_kernel",
				list(entries),
				list(sequence_ids),
				list(sequence_lengths),
			)
		)
		return _FakeTask(f"{self.name}:append_batched")

	def async_load_layer_kv_to_device(
		self, sequence_ids, k_device_ptrs, v_device_ptrs=None
	):
		self.calls.append(
			(
				"async_load_layer_kv_to_device",
				list(sequence_ids),
				k_device_ptrs,
				v_device_ptrs,
			)
		)
		return _FakeTask(f"{self.name}:load")

	def async_load_layer_paged_kv_to_device(
		self, sequence_ids, active_page_counts, k_device_ptrs, v_device_ptrs=None
	):
		self.calls.append(
			(
				"async_load_layer_paged_kv_to_device",
				list(sequence_ids),
				active_page_counts,
				k_device_ptrs,
				v_device_ptrs,
			)
		)
		return _FakeTask(f"{self.name}:load_paged")


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


def test_host_kv_coordinator_routes_async_data_movement_by_component():
	primary = _FakeHostView("primary")
	c4 = _FakeHostView("c4")
	coordinator = HostKVCoordinator()
	coordinator.register_component("primary", primary)
	coordinator.register_component(
		"compressor_c4",
		c4,
		logical_to_physical_layer=[-1, -1, 0, -1, 1],
	)

	task = coordinator.async_offload_layer_kv_to_host(
		4,
		[101],
		"k",
		"v",
		[65],
		component_name="compressor_c4",
	)
	assert task.name == "c4:offload"
	assert c4.calls[-1] == (
		"async_offload_layer_kv_to_host",
		1,
		[101],
		"k",
		"v",
		[65],
	)

	coordinator.async_append_decode_kv_to_host_batched_kernel(
		[(4, "k4", "v4"), (2, "k2", "v2")],
		[101, 102],
		[65, 66],
		component_name="compressor_c4",
	)
	assert c4.calls[-1] == (
		"async_append_decode_kv_to_host_batched_kernel",
		[(1, "k4", "v4"), (0, "k2", "v2")],
		[101, 102],
		[65, 66],
	)

	composite = coordinator.async_offload_components_kv_to_host(
		{
			"primary": {
				"layer_idx": 0,
				"sequence_ids": [101],
				"k_tensor": "pk",
				"v_tensor": "pv",
				"sequence_lengths": [65],
			},
			"compressor_c4": {
				"layer_idx": 2,
				"sequence_ids": [101],
				"k_tensor": "ck",
				"v_tensor": "cv",
				"sequence_lengths": [17],
			},
		},
		tensors="held",
	)
	assert composite.tensors == "held"
	composite.wait()
	assert composite.tasks["primary"].waited
	assert composite.tasks["compressor_c4"].waited
	assert primary.calls[-1] == (
		"async_offload_layer_kv_to_host",
		0,
		[101],
		"pk",
		"pv",
		[65],
	)
	assert c4.calls[-1] == (
		"async_offload_layer_kv_to_host",
		0,
		[101],
		"ck",
		"cv",
		[17],
	)

	with pytest.raises(KeyError):
		coordinator.async_offload_layer_kv_to_host(
			3,
			[101],
			"k",
			"v",
			[65],
			component_name="compressor_c4",
		)


def test_gpu_kv_coordinator_keeps_managers_independent():
	primary = _make_gpu_manager(num_layers=3, num_pages=16, page_size_tokens=4)
	c4 = _make_gpu_manager(num_layers=2, num_pages=16, page_size_tokens=2)
	coordinator = GPUKVCoordinator()
	coordinator.register_component("primary", primary)
	coordinator.register_component(
		"compressor_c4",
		c4,
		logical_to_physical_layer=[-1, -1, 0, -1, -1, 1],
		token_capacity_scale=0.25,
	)

	assert coordinator.primary is primary
	assert coordinator.get_manager("compressor_c4") is c4
	assert coordinator.resolve_physical_layer("compressor_c4", 5) == 1
	assert coordinator.map_token_counts("compressor_c4", [5, 9]) == [2, 3]
	with pytest.raises(KeyError):
		coordinator.resolve_physical_layer("compressor_c4", 4)

	coordinator.initialize()
	coordinator.allocate_pages_for_sequences([10, 20], [5, 9])
	primary_table = coordinator.rebuild_page_table([10, 20])
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
	assert layout.physical_layer_count(SWA) == 5
	assert layout.token_capacity(COMPRESSOR_C4, 65) == 17
	assert layout.token_capacity(COMPRESSOR_C128, 257) == 3
	assert layout.token_capacity(INDEXER_C4, 9) == 3
	assert layout.token_capacity(SWA, 4096) == 128

	with pytest.raises(ValueError):
		DeepSeekV4KVLayout.from_compression_ratios([0, 8, 4])


def test_deepseek_v4_host_coordinator_registers_dsv4_components():
	swa = _FakeHostView("swa")
	c4 = _FakeHostView("c4")
	c128 = _FakeHostView("c128")
	indexer = _FakeHostView("indexer")

	coordinator = DeepSeekV4HostKVCoordinator(
		compression_ratios=[0, 128, 4, 128, 4],
		swa=swa,
		compressor_c4=c4,
		compressor_c128=c128,
		indexer_c4=indexer,
		sliding_window=128,
	)

	assert coordinator.component_names == [
		SWA,
		COMPRESSOR_C4,
		COMPRESSOR_C128,
		INDEXER_C4,
	]
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
		swa=_FakeHostView("swa"),
		compressor_c4=c4,
	)

	assert coordinator.resolve_physical_layer(COMPRESSOR_C4, 4) == 1
	assert coordinator.view_layer_id(COMPRESSOR_C4, 4) == 4


def test_deepseek_v4_gpu_coordinator_registers_dsv4_components():
	swa = _make_gpu_manager(num_layers=5, num_pages=32, page_size_tokens=4)
	c4 = _make_gpu_manager(num_layers=2, num_pages=32, page_size_tokens=4)
	c128 = _make_gpu_manager(num_layers=2, num_pages=32, page_size_tokens=4)
	indexer = _make_gpu_manager(num_layers=2, num_pages=32, page_size_tokens=4)

	coordinator = DeepSeekV4GPUKVCoordinator(
		compression_ratios=[0, 128, 4, 128, 4],
		swa=swa,
		compressor_c4=c4,
		compressor_c128=c128,
		indexer_c4=indexer,
		sliding_window=128,
	)

	assert coordinator.get_manager(SWA) is swa
	assert coordinator.get_manager(COMPRESSOR_C4) is c4
	assert coordinator.resolve_physical_layer(COMPRESSOR_C4, 4) == 1
	assert coordinator.resolve_physical_layer(COMPRESSOR_C128, 1) == 0
	assert coordinator.resolve_physical_layer(INDEXER_C4, 2) == 0
	assert coordinator.map_token_counts(COMPRESSOR_C4, [65, 128]) == [17, 32]
	assert coordinator.map_token_counts(COMPRESSOR_C128, [257]) == [3]
	assert coordinator.map_token_counts(SWA, [64, 4096]) == [64, 128]

	coordinator.initialize()
	coordinator.allocate_pages_for_sequences([10], [65])

	assert swa._sequences[10].pages.numel() == 17
	assert c4._sequences[10].pages.numel() == 5
	assert c128._sequences[10].pages.numel() == 1
