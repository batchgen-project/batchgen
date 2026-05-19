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
NUM_PAGES = 16
STATE_PAGE_TOKENS = 16
RING_SIZE = 8


@pytest.fixture(scope="module")
def bg():
    from batchgen.models.engine_loader import core_engine as bg_module

    return bg_module


def _gpu_config(*, mapped: bool) -> CompressedStateGPUConfig:
    return CompressedStateGPUConfig(
        num_layers=NUM_LAYERS,
        num_pages=NUM_PAGES,
        state_page_size_tokens=STATE_PAGE_TOKENS,
        ring_size=RING_SIZE,
        state_dim=STATE_DIM,
        state_dtype=torch.float32,
        cuda_graph_max_pages_per_sequence=4,
        cuda_graph_max_slots=8,
        logical_to_physical_layer=[1, -1, 0] if mapped else None,
    )


def _host_config(bg, *, mapped: bool):
    cfg = bg.CompressedStateHostConfig()
    cfg.num_layers = NUM_LAYERS
    cfg.num_pages = NUM_PAGES
    cfg.state_page_size_tokens = STATE_PAGE_TOKENS
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
    allocations = manager.allocate_pages_for_sequences(sequence_ids, [20, 4])
    assert all(len(pages) == 1 for pages in allocations.values())
    page_table = manager.rebuild_page_table(sequence_ids)
    assert page_table.shape[0] == len(sequence_ids)
    assert page_table.shape[1] >= 1
    assert torch.all(page_table[:, 1:] == -1)

    values = torch.arange(
        len(sequence_ids) * STATE_DIM,
        dtype=torch.float32,
        device=device,
    ).view(len(sequence_ids), STATE_DIM)
    raw_positions = torch.tensor([18, 3], dtype=torch.int32, device=device)
    manager.update_layer_decode_state(values, raw_positions, layer_idx=2)
    torch.cuda.synchronize()

    layer_buffer = manager.get_layer_state_buffer(2)
    for row, raw_position in enumerate(raw_positions.tolist()):
        page_id = int(page_table[row, 0].item())
        ring_offset = raw_position % RING_SIZE
        assert torch.equal(layer_buffer[page_id, ring_offset], values[row])

    prepared_values = values + 100
    prepared_positions = torch.tensor(
        [19, 11], dtype=torch.int32, device=device
    )
    manager.prepare_decode_step(
        sequence_ids,
        prepared_positions,
        use_cuda_graph_page_table=True,
    )
    graph_state = manager.get_cuda_graph_page_table_state()
    assert graph_state.table.shape == (8, 4)
    assert graph_state.num_valid_slots == len(sequence_ids)
    manager.update_layer_decode_state(
        prepared_values,
        None,
        layer_idx=2,
        assume_prepared=True,
    )
    torch.cuda.synchronize()

    for row, raw_position in enumerate(prepared_positions.tolist()):
        page_id = int(page_table[row, 0].item())
        ring_offset = raw_position % RING_SIZE
        assert torch.equal(
            layer_buffer[page_id, ring_offset], prepared_values[row]
        )


def test_host_compressed_state_decode_round_trip(bg):
    device = torch.device("cuda:0")
    manager = bg.OverlapCompressedState4HostManager(
        _host_config(bg, mapped=True)
    )
    manager.initialize(0)
    sequence_ids = [301, 302]
    raw_positions = [18, 3]

    try:
        manager.allocate_pages_for_sequences([(301, 20), (302, 4)])
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
        manager.release_sequence_pages(sequence_ids)
        manager.shutdown()


def test_host_compressed_state_page_copy_round_trip(bg):
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
        host.allocate_pages_for_sequences([(401, 20)])
        gpu.allocate_pages_for_sequences(sequence_ids, [20])
        pages = gpu._sequences[401].pages.tolist()
        assert len(pages) == 1
        active_page_counts = [len(pages)]

        state_cache = gpu.state_cache
        state_cache.zero_()
        for layer_idx in range(NUM_LAYERS):
            for page_ordinal, page_id in enumerate(pages):
                state_cache[layer_idx, page_id].fill_(
                    10 * layer_idx + page_ordinal + 1
                )

        ptrs = torch.empty(
            (NUM_LAYERS, len(sequence_ids), len(pages)),
            dtype=torch.int64,
        )
        for layer_idx in range(NUM_LAYERS):
            ptrs[layer_idx, 0] = torch.tensor(
                gpu.get_sequence_layer_state_page_pointers(401, layer_idx),
                dtype=torch.int64,
            )

        expected = state_cache.clone()
        host.async_offload_state_pages_to_host(
            sequence_ids,
            active_page_counts,
            ptrs,
        ).wait()
        state_cache.zero_()
        host.async_load_state_pages_to_device(
            sequence_ids,
            active_page_counts,
            ptrs,
        ).wait()
        torch.cuda.synchronize()

        for layer_idx in range(NUM_LAYERS):
            for page_id in pages:
                assert torch.equal(
                    state_cache[layer_idx, page_id],
                    expected[layer_idx, page_id],
                )
    finally:
        host.release_sequence_pages(sequence_ids)
        host.shutdown()
        gpu.destroy()
