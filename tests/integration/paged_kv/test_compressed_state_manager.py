import pytest
import torch

from batchgen.kv_cache.compressed_state_gpu_manager import (
    CompressedStateGPUConfig,
    CompressedStateGPUManager,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="compressed-state manager tests require CUDA",
)


STATE_DIM = 8
NUM_LAYERS = 2
NUM_STATE_ITEMS = 16
RING_SIZE = 8


@pytest.fixture(scope="module")
def bg():
    from batchgen.models.engine_loader import core_engine as bg_module

    return bg_module


def _gpu_config(*, mapped: bool) -> CompressedStateGPUConfig:
    return CompressedStateGPUConfig(
        num_layers=NUM_LAYERS,
        num_state_items=NUM_STATE_ITEMS,
        ring_size=RING_SIZE,
        state_dim=STATE_DIM,
        state_dtype=torch.float32,
        cuda_graph_max_slots=8,
        logical_to_physical_layer=[1, -1, 0] if mapped else None,
    )


def _host_config(bg, *, mapped: bool):
    cfg = bg.CompressedStateHostConfig()
    cfg.num_layers = NUM_LAYERS
    cfg.num_state_items = NUM_STATE_ITEMS
    cfg.ring_size = RING_SIZE
    cfg.state_token_bytes = (
        STATE_DIM * torch.empty((), dtype=torch.float32).element_size()
    )
    cfg.sequence_table_capacity = 32
    cfg.alignment_bytes = 64
    cfg.logical_to_physical_layer = [1, -1, 0] if mapped else []
    cfg.logger_name = "CompressedStateHostTest"
    return cfg


def test_gpu_compressed_state_maps_raw_positions_and_prepared_slots():
    device = torch.device("cuda:0")
    manager = CompressedStateGPUManager(
        config=_gpu_config(mapped=True),
        device=device,
        ratio=4,
        overlap=True,
    )
    manager.initialize()

    sequence_ids = [101, 202]
    allocations = manager.allocate_state_items_for_sequences(sequence_ids)
    assert set(allocations) == set(sequence_ids)

    values = torch.arange(
        len(sequence_ids) * STATE_DIM,
        dtype=torch.float32,
        device=device,
    ).view(len(sequence_ids), STATE_DIM)
    raw_positions = torch.tensor([18, 3], dtype=torch.int32, device=device)
    manager.update_layer_decode_state(
        values,
        raw_positions,
        layer_idx=2,
        sequence_ids=sequence_ids,
    )
    torch.cuda.synchronize()

    layer_buffer = manager.get_layer_state_buffer(2)
    for row, raw_position in enumerate(raw_positions.tolist()):
        state_item_id = allocations[sequence_ids[row]]
        ring_offset = raw_position % RING_SIZE
        assert torch.equal(
            layer_buffer[state_item_id, ring_offset], values[row]
        )

    prepared_values = values + 100
    prepared_positions = torch.tensor(
        [19, 11], dtype=torch.int32, device=device
    )
    manager.prepare_decode_step(
        sequence_ids,
        prepared_positions,
    )
    manager.update_layer_decode_state(
        prepared_values,
        None,
        layer_idx=2,
        assume_prepared=True,
    )
    torch.cuda.synchronize()

    for row, raw_position in enumerate(prepared_positions.tolist()):
        state_item_id = allocations[sequence_ids[row]]
        ring_offset = raw_position % RING_SIZE
        assert torch.equal(
            layer_buffer[state_item_id, ring_offset], prepared_values[row]
        )

    explicit_values = values + 200
    explicit_slots = torch.tensor(
        [
            allocations[101] * RING_SIZE + 0,
            allocations[101] * RING_SIZE + 3,
        ],
        dtype=torch.int32,
        device=device,
    )
    manager.update_layer_state_slots(
        explicit_values,
        explicit_slots,
        layer_idx=2,
    )
    torch.cuda.synchronize()

    assert torch.equal(layer_buffer[allocations[101], 0], explicit_values[0])
    assert torch.equal(layer_buffer[allocations[101], 3], explicit_values[1])


def test_host_compressed_state_decode_round_trip(bg):
    device = torch.device("cuda:0")
    manager = bg.OverlapCompressedState4HostManager(
        _host_config(bg, mapped=True)
    )
    manager.initialize(0)
    sequence_ids = [301, 302]
    raw_positions = [18, 3]

    try:
        manager.allocate_state_items_for_sequences(sequence_ids)
        values = torch.arange(
            len(sequence_ids) * STATE_DIM,
            dtype=torch.float32,
            device=device,
        ).view(len(sequence_ids), STATE_DIM)
        manager.async_offload_decode_state_to_host(
            2,
            sequence_ids,
            values,
            raw_positions,
        ).wait()

        restored = torch.empty_like(values)
        manager.async_load_decode_state_to_device(
            2,
            sequence_ids,
            restored,
            raw_positions,
        ).wait()
        torch.cuda.synchronize()
        assert torch.equal(restored, values)

        overwritten = values + 50
        manager.async_offload_decode_state_to_host(
            2,
            sequence_ids,
            overwritten,
            [raw_positions[0] + RING_SIZE, raw_positions[1] + RING_SIZE],
        ).wait()
        manager.async_load_decode_state_to_device(
            2,
            sequence_ids,
            restored,
            raw_positions,
        ).wait()
        torch.cuda.synchronize()
        assert torch.equal(restored, overwritten)
    finally:
        manager.release_sequence_states(sequence_ids)
        manager.shutdown()


def test_host_compressed_state_batched_append_round_trip(bg):
    device = torch.device("cuda:0")
    manager = bg.NonOverlapCompressedState4HostManager(
        _host_config(bg, mapped=False)
    )
    manager.initialize(0)
    sequence_ids = [351, 352]
    raw_positions = [18, 3]

    try:
        manager.allocate_state_items_for_sequences(sequence_ids)
        layer0 = torch.arange(
            len(sequence_ids) * STATE_DIM,
            dtype=torch.float32,
            device=device,
        ).view(len(sequence_ids), STATE_DIM)
        layer1 = layer0 + 100
        manager.async_append_decode_state_to_host_batched_kernel(
            [(0, layer0), (1, layer1)],
            sequence_ids,
            raw_positions,
        ).wait()

        restored0 = torch.empty_like(layer0)
        restored1 = torch.empty_like(layer1)
        manager.async_load_decode_state_to_device(
            0,
            sequence_ids,
            restored0,
            raw_positions,
        ).wait()
        manager.async_load_decode_state_to_device(
            1,
            sequence_ids,
            restored1,
            raw_positions,
        ).wait()
        torch.cuda.synchronize()
        assert torch.equal(restored0, layer0)
        assert torch.equal(restored1, layer1)
    finally:
        manager.release_sequence_states(sequence_ids)
        manager.shutdown()


def test_host_compressed_state_item_copy_round_trip(bg):
    device = torch.device("cuda:0")
    host = bg.NonOverlapCompressedState4HostManager(
        _host_config(bg, mapped=False)
    )
    gpu = CompressedStateGPUManager(
        config=_gpu_config(mapped=False),
        device=device,
        ratio=4,
        overlap=False,
    )
    host.initialize(0)
    gpu.initialize()
    sequence_ids = [401]

    try:
        host.allocate_state_items_for_sequences(sequence_ids)
        allocations = gpu.allocate_state_items_for_sequences(sequence_ids)
        state_item_id = allocations[401]

        state_cache = gpu.state_cache
        state_cache.zero_()
        for layer_idx in range(NUM_LAYERS):
            state_cache[layer_idx, state_item_id].fill_(10 * layer_idx + 1)

        ptrs = gpu.export_state_item_pointers(sequence_ids)

        expected = state_cache.clone()
        host.async_offload_state_items_to_host(
            sequence_ids,
            ptrs,
        ).wait()
        state_cache.zero_()
        host.async_load_state_items_to_device(
            sequence_ids,
            ptrs,
        ).wait()
        torch.cuda.synchronize()

        for layer_idx in range(NUM_LAYERS):
            assert torch.equal(
                state_cache[layer_idx, state_item_id],
                expected[layer_idx, state_item_id],
            )
    finally:
        host.release_sequence_states(sequence_ids)
        host.shutdown()
        gpu.destroy()
