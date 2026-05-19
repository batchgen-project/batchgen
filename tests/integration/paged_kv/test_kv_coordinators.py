from __future__ import annotations

import pytest
import torch

from batchgen.kv_cache.component_coordinator import (
    GPUKVCoordinator,
    HostKVCoordinator,
)
from batchgen.kv_cache.compressed_ratio_gpu_paged_kv_manager import (
    CompressedRatioGPUPagedKVCacheManager,
)
from batchgen.kv_cache.deepseek_v4_kv_coordinator import (
    COMPRESSOR_C4,
    COMPRESSOR_C4_STATE,
    COMPRESSOR_C128,
    COMPRESSOR_C128_STATE,
    INDEXER_C4,
    INDEXER_C4_STATE,
    SWA,
    DeepSeekV4GPUKVCoordinator,
    DeepSeekV4HostKVCoordinator,
)
from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)
from batchgen.kv_cache.swa_gpu_paged_kv_manager import SWAGPUPagedKVCacheManager


class _FakeTask:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class _FakeHostView:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []

    def initialize(self, *args, **kwargs) -> None:
        self.calls.append(("initialize", args, kwargs))

    def shutdown(self) -> None:
        self.calls.append(("shutdown",))

    def register_sequences(self, sequence_ids) -> None:
        self.calls.append(("register_sequences", list(sequence_ids)))

    def async_offload_layer_kv_to_host(
        self,
        layer_idx,
        sequence_ids,
        k_tensor,
        v_tensor=None,
        sequence_lengths=None,
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
        self,
        layer_idx,
        sequence_ids,
        k_tensor,
        v_tensor=None,
        sequence_lengths=None,
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
        self,
        sequence_ids,
        active_page_counts,
        k_device_ptrs,
        v_device_ptrs=None,
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


class _FakeStateManager:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []
        self.is_initialized = False

    def initialize(self, *args, **kwargs) -> None:
        self.is_initialized = True
        self.calls.append(("initialize", args, kwargs))

    def shutdown(self) -> None:
        self.is_initialized = False
        self.calls.append(("shutdown",))

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        self.is_initialized = False
        self.calls.append(("destroy", empty_cuda_cache))

    def allocate_state_items_for_sequences(self, sequence_ids):
        self.calls.append(
            ("allocate_state_items_for_sequences", list(sequence_ids))
        )
        return {
            int(seq_id): int(offset)
            for offset, seq_id in enumerate(sequence_ids)
        }

    def release_sequence_states(self, sequence_ids) -> None:
        self.calls.append(("release_sequence_states", list(sequence_ids)))

    def async_offload_decode_state_to_host(
        self, layer_idx, sequence_ids, state_tensor, raw_positions
    ):
        self.calls.append(
            (
                "async_offload_decode_state_to_host",
                int(layer_idx),
                list(sequence_ids),
                state_tensor,
                raw_positions,
            )
        )
        return _FakeTask(f"{self.name}:offload_state")

    def async_load_decode_state_to_device(
        self, layer_idx, sequence_ids, state_tensor, raw_positions
    ):
        self.calls.append(
            (
                "async_load_decode_state_to_device",
                int(layer_idx),
                list(sequence_ids),
                state_tensor,
                raw_positions,
            )
        )
        return _FakeTask(f"{self.name}:load_state")

    def async_append_decode_state_to_host_batched_kernel(
        self, entries, sequence_ids, raw_positions
    ):
        self.calls.append(
            (
                "async_append_decode_state_to_host_batched_kernel",
                list(entries),
                list(sequence_ids),
                raw_positions,
            )
        )
        return _FakeTask(f"{self.name}:append_state")

    def async_offload_state_items_to_host(
        self, sequence_ids, state_device_ptrs
    ):
        self.calls.append(
            (
                "async_offload_state_items_to_host",
                list(sequence_ids),
                state_device_ptrs,
            )
        )
        return _FakeTask(f"{self.name}:offload_state_items")

    def async_load_state_items_to_device(self, sequence_ids, state_device_ptrs):
        self.calls.append(
            (
                "async_load_state_items_to_device",
                list(sequence_ids),
                state_device_ptrs,
            )
        )
        return _FakeTask(f"{self.name}:load_state_items")

    def prepare_decode_step(self, sequence_ids, raw_positions) -> None:
        self.calls.append(
            ("prepare_decode_step", list(sequence_ids), raw_positions)
        )

    def update_layer_decode_state(
        self,
        state_tensor,
        raw_positions,
        layer_idx,
        *,
        sequence_ids=None,
        batch_slice=None,
        assume_prepared=False,
    ) -> None:
        self.calls.append(
            (
                "update_layer_decode_state",
                state_tensor,
                raw_positions,
                int(layer_idx),
                sequence_ids,
                batch_slice,
                assume_prepared,
            )
        )

    def export_state_item_pointers(self, sequence_ids):
        self.calls.append(("export_state_item_pointers", list(sequence_ids)))
        return "state_ptrs"

    def get_sequence_layer_state_item_pointer(self, sequence_id, layer_idx):
        self.calls.append(
            (
                "get_sequence_layer_state_item_pointer",
                int(sequence_id),
                int(layer_idx),
            )
        )
        return 12345


def _make_gpu_manager(
    *,
    num_layers: int,
    num_pages: int,
    page_size_tokens: int,
    logical_to_physical_layer=None,
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
        logical_to_physical_layer=logical_to_physical_layer,
    )
    return GPUPagedKVCacheManager(config=config, device="cpu")


def test_host_kv_coordinator_is_a_generic_component_registry():
    primary = _FakeHostView("primary")
    c4 = _FakeHostView("c4")
    c4_state = _FakeStateManager("c4_state")
    coordinator = HostKVCoordinator()
    coordinator.register_component("primary", primary)
    coordinator.register_component("compressor_c4", c4)
    coordinator.register_component("compressor_c4_state", c4_state)

    assert coordinator.component_names == [
        "primary",
        "compressor_c4",
        "compressor_c4_state",
    ]
    assert coordinator.primary is primary
    assert coordinator.get_component("compressor_c4") is c4
    assert coordinator.compressor_c4_state is c4_state

    coordinator.call_all("initialize", 0)
    assert primary.calls == [("initialize", (0,), {})]
    assert c4.calls == [("initialize", (0,), {})]
    assert c4_state.calls == [("initialize", (0,), {})]


def test_host_kv_coordinator_can_call_one_component_explicitly():
    coordinator = HostKVCoordinator()
    mapped_view = _FakeHostView("mapped")
    coordinator.register_component("mapped", mapped_view)

    task = coordinator.call_component(
        "mapped",
        "async_offload_layer_kv_to_host",
        4,
        [101],
        "k",
        "v",
        [65],
    )
    assert task.name == "mapped:offload"
    assert mapped_view.calls[-1] == (
        "async_offload_layer_kv_to_host",
        4,
        [101],
        "k",
        "v",
        [65],
    )


def test_host_kv_coordinator_rejects_component_metadata():
    coordinator = HostKVCoordinator()

    with pytest.raises(ValueError, match="does not accept component metadata"):
        coordinator.register_component(
            "empty_component",
            _FakeHostView("empty"),
            logical_to_physical_layer=[-1, -1, 0],
        )


def test_gpu_kv_coordinator_is_a_generic_component_registry():
    primary = _make_gpu_manager(num_layers=3, num_pages=16, page_size_tokens=4)
    c4 = _make_gpu_manager(num_layers=2, num_pages=16, page_size_tokens=2)
    c4_state = _FakeStateManager("c4_state")
    coordinator = GPUKVCoordinator()
    coordinator.register_component("primary", primary)
    coordinator.register_component("compressor_c4", c4)
    coordinator.register_component("compressor_c4_state", c4_state)

    assert coordinator.primary is primary
    assert coordinator.get_component("compressor_c4") is c4
    assert coordinator.compressor_c4_state is c4_state

    coordinator.call_component("primary", "initialize")
    coordinator.call_component("compressor_c4", "initialize")
    primary.allocate_pages_for_sequences([10, 20], [5, 9])
    c4.allocate_pages_for_sequences([10, 20], [5, 9])
    primary_table = primary.rebuild_page_table([10, 20])
    c4_table = c4.rebuild_page_table([10, 20])

    assert tuple(primary_table.shape) == (2, 4)
    assert tuple(c4_table.shape) == (2, 6)
    assert primary._sequences[20].pages.numel() == 3
    assert c4._sequences[20].pages.numel() == 5


def test_gpu_kv_coordinator_rejects_component_metadata():
    coordinator = GPUKVCoordinator()

    with pytest.raises(ValueError, match="does not accept component metadata"):
        coordinator.register_component(
            "compressor_c4",
            _make_gpu_manager(num_layers=2, num_pages=8, page_size_tokens=4),
            logical_to_physical_layer=[-1, -1, 0, -1, 1],
        )


def test_mapped_gpu_paged_kv_manager_resolves_logical_layers():
    manager = _make_gpu_manager(
        num_layers=2,
        num_pages=8,
        page_size_tokens=4,
        logical_to_physical_layer=[-1, -1, 0, -1, 1],
    )
    assert manager.uses_logical_layer_mapping
    assert manager.resolve_physical_layer(2) == 0
    assert manager.resolve_physical_layer(4) == 1

    manager.initialize()
    manager.allocate_pages_for_sequences([10], [8])
    manager.rebuild_page_table([10])
    page_id = int(manager._sequences[10].pages[0].item())

    k_ptrs, _ = manager.get_sequence_layer_page_pointers(10, 4)
    assert k_ptrs[0] == manager._k_cache[1, page_id].data_ptr()

    k_layer, _, _ = manager.get_layer_kv_with_page_table(2)
    assert k_layer.data_ptr() == manager._k_cache[0].data_ptr()

    with pytest.raises(KeyError):
        manager.get_sequence_layer_page_pointers(10, 3)


def test_gpu_kv_coordinator_can_call_one_component_explicitly():
    manager = _make_gpu_manager(
        num_layers=2,
        num_pages=8,
        page_size_tokens=4,
        logical_to_physical_layer=[-1, -1, 0, -1, 1],
    )
    manager.initialize()
    manager.allocate_pages_for_sequences([10], [8])
    manager.rebuild_page_table([10])
    page_id = int(manager._sequences[10].pages[0].item())

    coordinator = GPUKVCoordinator()
    coordinator.register_component("compressor_c4", manager)

    k_ptrs, _ = coordinator.call_component(
        "compressor_c4",
        "get_sequence_layer_page_pointers",
        10,
        4,
    )

    assert k_ptrs[0] == manager._k_cache[1, page_id].data_ptr()


def test_compressed_ratio_gpu_manager_uses_floor_positions_without_padding():
    manager = CompressedRatioGPUPagedKVCacheManager(
        config=GPUPagedKVConfig(
            num_layers=1,
            num_pages=16,
            page_size_tokens=2,
            num_k_heads=1,
            k_head_dim=2,
            num_v_heads=0,
            v_head_dim=0,
            kv_dtype=torch.float32,
        ),
        device="cpu",
        compression_ratio=4,
    )
    manager.initialize()
    manager.allocate_pages_for_sequences([10, 20, 30, 40], [3, 4, 7, 8])

    assert manager._sequences[10].pages.numel() == 1
    assert manager._sequences[40].pages.numel() == 1
    assert manager.map_raw_lengths_to_compressed_lengths(
        [3, 4, 7, 8]
    ).tolist() == [
        0,
        1,
        1,
        2,
    ]

    manager.prepare_decode_step(
        [10, 20, 30, 40],
        torch.tensor([2, 3, 6, 7], dtype=torch.int32),
        refresh_page_table=True,
    )
    assert manager._prepared_decode_positions[:4].tolist() == [-1, 0, -1, 1]


def test_swa_gpu_manager_releases_prefix_pages_at_page_boundary():
    manager = SWAGPUPagedKVCacheManager(
        config=GPUPagedKVConfig(
            num_layers=1,
            num_pages=8,
            page_size_tokens=4,
            num_k_heads=1,
            k_head_dim=2,
            num_v_heads=0,
            v_head_dim=0,
            kv_dtype=torch.float32,
        ),
        device="cpu",
        window_size_tokens=8,
    )
    manager.initialize()
    manager.allocate_pages_for_sequences([10], [9])
    before_pages = manager._sequences[10].pages.tolist()
    assert len(before_pages) == 3

    manager.prepare_decode_step(
        [10],
        torch.tensor([12], dtype=torch.int32),
        refresh_page_table=True,
    )
    after_pages = manager._sequences[10].pages.tolist()

    assert manager._states[10].window_start_page == 1
    assert len(after_pages) == 3
    assert after_pages[0] != before_pages[0]
    assert manager._prepared_decode_positions[:1].tolist() == [8]


def test_deepseek_v4_host_coordinator_registers_dsv4_components():
    swa = _FakeHostView("swa")
    c4 = _FakeHostView("c4")
    c128 = _FakeHostView("c128")
    indexer = _FakeHostView("indexer")

    coordinator = DeepSeekV4HostKVCoordinator(
        swa=swa,
        compressor_c4=c4,
        compressor_c128=c128,
        indexer_c4=indexer,
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

    coordinator.compressor_c4.async_offload_layer_kv_to_host(
        4,
        [101],
        "k",
        None,
        [17],
    )
    assert c4.calls[-1] == (
        "async_offload_layer_kv_to_host",
        4,
        [101],
        "k",
        None,
        [17],
    )


def test_deepseek_v4_host_coordinator_routes_state_components():
    swa = _FakeHostView("swa")
    c4_state = _FakeStateManager("c4_state")
    indexer_state = _FakeStateManager("indexer_state")

    coordinator = DeepSeekV4HostKVCoordinator(
        swa=swa,
        compressor_c4_state=c4_state,
        indexer_c4_state=indexer_state,
    )

    assert coordinator.compressor_c4_state is c4_state
    assert coordinator.compressor_c128_state is None
    assert coordinator.indexer_c4_state is indexer_state
    assert coordinator.component_names == [
        SWA,
        COMPRESSOR_C4_STATE,
        INDEXER_C4_STATE,
    ]
    assert coordinator.get_component(COMPRESSOR_C4_STATE) is c4_state
    assert coordinator.get_component(INDEXER_C4_STATE) is indexer_state

    coordinator.initialize(0)
    assert c4_state.calls[-1] == ("initialize", (0,), {})

    allocation = (
        coordinator.compressor_c4_state.allocate_state_items_for_sequences(
            [101, 102]
        )
    )
    assert allocation == {101: 0, 102: 1}

    task = coordinator.compressor_c4_state.async_append_decode_state_to_host_batched_kernel(
        [(2, "state_l2")],
        [101],
        [7],
    )
    assert task.name == "c4_state:append_state"
    assert c4_state.calls[-1] == (
        "async_append_decode_state_to_host_batched_kernel",
        [(2, "state_l2")],
        [101],
        [7],
    )

    coordinator.compressor_c4_state.release_sequence_states([101])
    coordinator.indexer_c4_state.release_sequence_states([101])
    assert c4_state.calls[-1] == ("release_sequence_states", [101])
    assert indexer_state.calls[-1] == ("release_sequence_states", [101])


def test_deepseek_v4_gpu_coordinator_registers_dsv4_components():
    swa = _make_gpu_manager(num_layers=5, num_pages=32, page_size_tokens=4)
    c4 = _make_gpu_manager(
        num_layers=2,
        num_pages=32,
        page_size_tokens=4,
        logical_to_physical_layer=[-1, -1, 0, -1, 1],
    )
    c128 = _make_gpu_manager(
        num_layers=2,
        num_pages=32,
        page_size_tokens=4,
        logical_to_physical_layer=[-1, 0, -1, 1, -1],
    )
    indexer = _make_gpu_manager(
        num_layers=2,
        num_pages=32,
        page_size_tokens=4,
        logical_to_physical_layer=[-1, -1, 0, -1, 1],
    )

    coordinator = DeepSeekV4GPUKVCoordinator(
        swa=swa,
        compressor_c4=c4,
        compressor_c128=c128,
        indexer_c4=indexer,
    )

    assert coordinator.get_component(SWA) is swa
    assert coordinator.get_component(COMPRESSOR_C4) is c4

    coordinator.initialize()
    coordinator.compressor_c4.allocate_pages_for_sequences([10], [8])
    coordinator.compressor_c4.rebuild_page_table([10])
    page_id = int(c4._sequences[10].pages[0].item())
    k_ptrs, _ = coordinator.compressor_c4.get_sequence_layer_page_pointers(
        10,
        4,
    )

    assert k_ptrs[0] == c4._k_cache[1, page_id].data_ptr()


def test_deepseek_v4_gpu_coordinator_routes_state_components():
    swa = _make_gpu_manager(num_layers=5, num_pages=32, page_size_tokens=4)
    c4_state = _FakeStateManager("c4_state")
    c128_state = _FakeStateManager("c128_state")

    coordinator = DeepSeekV4GPUKVCoordinator(
        swa=swa,
        compressor_c4_state=c4_state,
        compressor_c128_state=c128_state,
    )

    assert coordinator.compressor_c4_state is c4_state
    assert coordinator.compressor_c128_state is c128_state
    assert coordinator.indexer_c4_state is None
    assert coordinator.component_names == [
        SWA,
        COMPRESSOR_C4_STATE,
        COMPRESSOR_C128_STATE,
    ]
    assert coordinator.get_component(COMPRESSOR_C4_STATE) is c4_state
    assert coordinator.get_component(COMPRESSOR_C128_STATE) is c128_state

    coordinator.initialize()
    assert c4_state.calls[-1] == ("initialize", (), {})

    coordinator.compressor_c4_state.prepare_decode_step(
        [10],
        torch.tensor([3], dtype=torch.int32),
    )
    assert c4_state.calls[-1][0] == "prepare_decode_step"

    coordinator.compressor_c4_state.update_layer_decode_state(
        "state",
        None,
        2,
        assume_prepared=True,
    )
    assert c4_state.calls[-1] == (
        "update_layer_decode_state",
        "state",
        None,
        2,
        None,
        None,
        True,
    )
