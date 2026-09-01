import ctypes
import errno
import os
import random
import string

import pytest
import torch


_libc = ctypes.CDLL("libc.so.6", use_errno=True)

PAGE_TOKENS = 64
NUM_PHYSICAL_LAYERS = 4
NUM_PAGES = 128
NUM_K_HEADS = 1
K_HEAD_DIM = 8
K_ELEMENT_SIZE_BYTES = 2
ALIGNMENT_BYTES = 64

# logical 0, 3, and 6 are intentionally absent.
LOGICAL_TO_PHYSICAL = [-1, 3, 0, -1, 2, 1, -1]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mapped HostKV tests require CUDA"
)


@pytest.fixture(scope="module")
def bg():
    if not torch.cuda.is_available():
        pytest.skip("mapped HostKV tests require CUDA")
    from batchgen.models.engine_loader import core_engine as bg_module

    return bg_module


def _random_shm_name() -> str:
    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )
    return f"/batchgen_mapped_kv_{suffix}"


def _shm_unlink(name: str) -> None:
    if not name:
        return
    res = _libc.shm_unlink(name.encode("utf-8"))
    if res != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, f"shm_unlink({name}) failed")


def _make_mapped_mla_config(bg, shm_name: str):
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = NUM_PHYSICAL_LAYERS
    cfg.num_pages = NUM_PAGES
    cfg.page_size_tokens = PAGE_TOKENS
    cfg.num_k_heads = NUM_K_HEADS
    cfg.k_head_dim = K_HEAD_DIM
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = K_ELEMENT_SIZE_BYTES
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = 256
    cfg.alignment_bytes = ALIGNMENT_BYTES
    cfg.logical_to_physical_layer = list(LOGICAL_TO_PHYSICAL)
    cfg.logger_name = "MappedHostKVTest"
    return cfg


def _k_page_bytes() -> int:
    return PAGE_TOKENS * NUM_K_HEADS * K_HEAD_DIM * K_ELEMENT_SIZE_BYTES


def _layer_stride_bytes() -> int:
    raw_layer_bytes = NUM_PAGES * _k_page_bytes()
    return (
        (raw_layer_bytes + ALIGNMENT_BYTES - 1)
        // ALIGNMENT_BYTES
        * ALIGNMENT_BYTES
    )


def _read_bf16_token(page_ptr: int, token_offset: int) -> torch.Tensor:
    token_elems = NUM_K_HEADS * K_HEAD_DIM
    byte_offset = token_offset * token_elems * K_ELEMENT_SIZE_BYTES
    array_type = ctypes.c_uint16 * token_elems
    raw = array_type.from_address(page_ptr + byte_offset)
    return torch.frombuffer(raw, dtype=torch.uint16).view(torch.bfloat16).clone()


def _expected_token(value: float) -> torch.Tensor:
    return torch.full(
        (NUM_K_HEADS * K_HEAD_DIM,), value, dtype=torch.bfloat16
    )


def _token_value_for_cpu_write(
    sequence_id: int, physical_layer: int, page_ordinal: int
) -> float:
    return float(physical_layer * 16 + page_ordinal + sequence_id % 7)


def _close_view(view, sequence_ids) -> None:
    if view is None:
        return
    try:
        view.release_sequence_pages(list(sequence_ids))
    except Exception:
        pass
    try:
        view.shutdown()
    except Exception:
        pass


def test_mapped_mla_packed_prefill_offload_variable_lengths(bg):
    shm_name = _random_shm_name()
    cfg = _make_mapped_mla_config(bg, shm_name)
    view = None
    sequence_ids = [11, 22, 33, 44]
    sequence_lengths = [1, PAGE_TOKENS, PAGE_TOKENS + 3, PAGE_TOKENS * 2 + 5]

    try:
        torch.cuda.set_device(0)
        view = bg.MappedMLAHostPagedKVWorkerView(cfg)
        view.initialize(0, True)
        view.register_sequences(sequence_ids)
        view.allocate_pages_for_sequences(
            list(zip(sequence_ids, sequence_lengths))
        )

        packed = torch.empty(
            (sum(sequence_lengths), NUM_K_HEADS, K_HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        offset = 0
        for batch_idx, length in enumerate(sequence_lengths):
            packed[offset : offset + length].fill_(float(30 + batch_idx))
            offset += length

        task = view.async_offload_packed_layer_kv_to_host(
            4, sequence_ids, packed, None, sequence_lengths
        )
        task.wait()

        for batch_idx, (sequence_id, length) in enumerate(
            zip(sequence_ids, sequence_lengths)
        ):
            k_cpu, v_cpu = view.read_sequence_kv_to_cpu(sequence_id)
            assert v_cpu.numel() == 0
            flattened = k_cpu[2].reshape(-1, NUM_K_HEADS, K_HEAD_DIM)
            expected = torch.full_like(
                flattened[:length], float(30 + batch_idx)
            )
            assert torch.equal(flattened[:length], expected)

        with pytest.raises(ValueError, match="sum\(sequence_lengths\)"):
            view.async_offload_packed_layer_kv_to_host(
                4, sequence_ids, packed, None, [1, 1, 1, 1]
            )
    finally:
        _close_view(view, sequence_ids)
        _shm_unlink(shm_name)


def test_mapped_mla_view_routes_logical_layers_after_cpu_write(bg):
    shm_name = _random_shm_name()
    cfg = _make_mapped_mla_config(bg, shm_name)
    mapping_repr = "logical_to_physical_layer=[-1,3,0,-1,2,1,-1]"
    assert mapping_repr in repr(cfg)
    view = None
    sequence_lengths = {
        101: 1,
        202: PAGE_TOKENS,
        303: PAGE_TOKENS + 1,
        404: PAGE_TOKENS * 3 + 1,
    }

    try:
        view = bg.MappedMLAHostPagedKVWorkerView(cfg)
        assert mapping_repr in repr(view)
        assert view.uses_logical_layer_mapping is True
        assert view.has_v_cache is False
        for logical_layer, physical_layer in enumerate(LOGICAL_TO_PHYSICAL):
            if physical_layer >= 0:
                assert view.resolve_physical_layer(logical_layer) == physical_layer

        sequence_ids = list(sequence_lengths)
        view.initialize(0, True)
        view.register_sequences(sequence_ids)
        requests = list(sequence_lengths.items())
        allocations = view.allocate_pages_for_sequences(requests)
        assert view.build_page_table(sequence_ids) == allocations

        stats = view.get_stats()
        assert stats.num_active_sequences == len(sequence_ids)
        assert stats.num_used_pages == sum(len(pages) for pages in allocations)

        for sequence_id, pages in zip(sequence_ids, allocations):
            num_pages = len(pages)
            k_tensor = torch.empty(
                (
                    NUM_PHYSICAL_LAYERS,
                    num_pages,
                    PAGE_TOKENS,
                    NUM_K_HEADS,
                    K_HEAD_DIM,
                ),
                dtype=torch.bfloat16,
            )
            for physical_layer in range(NUM_PHYSICAL_LAYERS):
                for page_ordinal in range(num_pages):
                    k_tensor[physical_layer, page_ordinal].fill_(
                        _token_value_for_cpu_write(
                            sequence_id, physical_layer, page_ordinal
                        )
                    )
            view.write_sequence_kv_from_cpu(sequence_id, k_tensor, None)

        base_addr = view.data_base_address()
        k_page_bytes = _k_page_bytes()
        layer_stride = _layer_stride_bytes()

        for sequence_id, pages in zip(sequence_ids, allocations):
            for logical_layer, physical_layer in enumerate(
                LOGICAL_TO_PHYSICAL
            ):
                if physical_layer < 0:
                    with pytest.raises(IndexError):
                        view.get_sequence_layer_page_pointers(
                            sequence_id, logical_layer, None
                        )
                    continue

                k_ptrs, v_ptrs = view.get_sequence_layer_page_pointers(
                    sequence_id, logical_layer, None
                )
                assert v_ptrs is None
                assert len(k_ptrs) == len(pages)
                for page_ordinal, physical_page_id in enumerate(pages):
                    page_ptr = int(k_ptrs[page_ordinal])
                    assert page_ptr == view.k_page_ptr(
                        logical_layer, physical_page_id
                    )
                    expected_offset = (
                        physical_layer * layer_stride
                        + physical_page_id * k_page_bytes
                    )
                    assert page_ptr - base_addr == expected_offset

                    expected_value = _token_value_for_cpu_write(
                        sequence_id, physical_layer, page_ordinal
                    )
                    assert torch.equal(
                        _read_bf16_token(page_ptr, 0),
                        _expected_token(expected_value),
                    )

        with pytest.raises(IndexError):
            view.resolve_physical_layer(len(LOGICAL_TO_PHYSICAL))

        view.release_sequence_pages(sequence_ids)
        stats_after_free = view.get_stats()
        assert stats_after_free.num_used_pages == 0
        assert stats_after_free.num_active_sequences == 0
        view.shutdown()
        view = None
    finally:
        _close_view(view, sequence_lengths.keys())
        _shm_unlink(shm_name)


def test_mapped_mla_view_routes_prefill_and_batched_decode_writes(bg):
    shm_name = _random_shm_name()
    cfg = _make_mapped_mla_config(bg, shm_name)
    view = None
    sequence_ids = [11, 22, 33, 44]
    capacity_tokens = PAGE_TOKENS * 4
    prefill_lengths = [1, PAGE_TOKENS, PAGE_TOKENS + 3, PAGE_TOKENS * 2 + 5]
    decode_positions = [0, PAGE_TOKENS - 1, PAGE_TOKENS, PAGE_TOKENS * 2 + 7]

    try:
        torch.cuda.set_device(0)
        view = bg.MappedMLAHostPagedKVWorkerView(cfg)
        view.initialize(0, True)
        view.register_sequences(sequence_ids)
        allocations = view.allocate_pages_for_sequences(
            [(sequence_id, capacity_tokens) for sequence_id in sequence_ids]
        )
        assert all(len(pages) == 4 for pages in allocations)

        device = torch.device("cuda:0")
        max_prefill_len = max(prefill_lengths)
        prefill = torch.zeros(
            (
                len(sequence_ids),
                max_prefill_len,
                NUM_K_HEADS,
                K_HEAD_DIM,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        for batch_idx, length in enumerate(prefill_lengths):
            prefill[batch_idx, :length].fill_(float(20 + batch_idx))

        # logical layer 4 routes to physical layer 2.
        prefill_task = view.async_offload_layer_kv_to_host(
            4, sequence_ids, prefill, None, prefill_lengths
        )
        prefill_task.wait()

        for batch_idx, sequence_id in enumerate(sequence_ids):
            k_cpu, v_cpu = view.read_sequence_kv_to_cpu(sequence_id)
            assert v_cpu.numel() == 0
            for token_idx in {0, prefill_lengths[batch_idx] - 1}:
                page_ordinal = token_idx // PAGE_TOKENS
                page_offset = token_idx % PAGE_TOKENS
                actual = k_cpu[2, page_ordinal, page_offset].flatten()
                assert torch.equal(
                    actual, _expected_token(float(20 + batch_idx))
                )

        batched_entries = []
        decode_specs = [
            (1, 3, 50.0),  # logical 1 -> physical 3
            (2, 0, 60.0),  # logical 2 -> physical 0
            (5, 1, 70.0),  # logical 5 -> physical 1
        ]
        for logical_layer, _physical_layer, base_value in decode_specs:
            decode = torch.empty(
                (len(sequence_ids), 1, NUM_K_HEADS, K_HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            for batch_idx in range(len(sequence_ids)):
                decode[batch_idx].fill_(base_value + batch_idx)
            batched_entries.append((logical_layer, decode, None))

        decode_task = view.async_append_decode_kv_to_host_batched_kernel(
            batched_entries, sequence_ids, decode_positions
        )
        decode_task.wait()

        for batch_idx, sequence_id in enumerate(sequence_ids):
            k_cpu, _ = view.read_sequence_kv_to_cpu(sequence_id)
            token_idx = decode_positions[batch_idx]
            page_ordinal = token_idx // PAGE_TOKENS
            page_offset = token_idx % PAGE_TOKENS
            for _logical_layer, physical_layer, base_value in decode_specs:
                actual = k_cpu[physical_layer, page_ordinal, page_offset]
                assert torch.equal(
                    actual.flatten(), _expected_token(base_value + batch_idx)
                )

        view.release_sequence_pages(sequence_ids)
        stats_after_free = view.get_stats()
        assert stats_after_free.num_used_pages == 0
        assert stats_after_free.num_active_sequences == 0
        view.shutdown()
        view = None
    finally:
        _close_view(view, sequence_ids)
        _shm_unlink(shm_name)


@pytest.mark.parametrize("use_custom_stream", [False, True])
def test_mapped_mla_view_copies_exact_dirty_gpu_page_ranges(
    bg, use_custom_stream
):
    shm_name = _random_shm_name()
    cfg = _make_mapped_mla_config(bg, shm_name)
    view = None
    sequence_ids = [110, 220, 330, 440]
    active_pages = 3
    allocated_pages = 4
    dirty_ranges = [
        (0, 0),
        (1, PAGE_TOKENS + 3),
        (PAGE_TOKENS - 1, PAGE_TOKENS * 2 + 1),
        (PAGE_TOKENS * 2 - 2, PAGE_TOKENS * 3),
    ]
    host_sentinel = -37.0

    try:
        torch.cuda.set_device(0)
        view = bg.MappedMLAHostPagedKVWorkerView(cfg)
        view.initialize(0, True)
        view.register_sequences(sequence_ids)
        allocations = view.allocate_pages_for_sequences(
            [
                (sequence_id, PAGE_TOKENS * allocated_pages)
                for sequence_id in sequence_ids
            ]
        )
        assert all(len(pages) == allocated_pages for pages in allocations)

        host_initial = torch.full(
            (
                NUM_PHYSICAL_LAYERS,
                allocated_pages,
                PAGE_TOKENS,
                NUM_K_HEADS,
                K_HEAD_DIM,
            ),
            host_sentinel,
            dtype=torch.bfloat16,
        )
        for sequence_id in sequence_ids:
            view.write_sequence_kv_from_cpu(sequence_id, host_initial, None)

        source_shape = (
            NUM_PHYSICAL_LAYERS,
            len(sequence_ids),
            active_pages,
            PAGE_TOKENS,
            NUM_K_HEADS,
            K_HEAD_DIM,
        )
        source_cpu = torch.empty(
            source_shape,
            dtype=torch.bfloat16,
            pin_memory=True,
        )
        layer_values = torch.arange(NUM_PHYSICAL_LAYERS).view(-1, 1, 1, 1)
        sequence_values = torch.arange(len(sequence_ids)).view(1, -1, 1, 1)
        token_values = torch.arange(active_pages * PAGE_TOKENS).view(
            1, 1, active_pages, PAGE_TOKENS
        )
        source_bits = (
            1
            + layer_values
            * len(sequence_ids)
            * active_pages
            * PAGE_TOKENS
            + sequence_values * active_pages * PAGE_TOKENS
            + token_values
        ).to(torch.uint16)
        source_values = source_bits.view(torch.bfloat16)
        source_cpu.copy_(
            source_values[..., None, None].expand(source_shape)
        )

        device_pages = torch.empty(
            source_shape,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        device_ptrs = torch.zeros(
            (
                NUM_PHYSICAL_LAYERS,
                len(sequence_ids),
                allocated_pages,
            ),
            dtype=torch.int64,
        )
        for physical_layer in range(NUM_PHYSICAL_LAYERS):
            for sequence_idx in range(len(sequence_ids)):
                for page_idx in range(active_pages):
                    device_ptrs[
                        physical_layer, sequence_idx, page_idx
                    ] = device_pages[
                        physical_layer, sequence_idx, page_idx
                    ].data_ptr()

        sequence_tensor = torch.tensor(sequence_ids, dtype=torch.int64)
        page_count_tensor = torch.full(
            (len(sequence_ids),), active_pages, dtype=torch.int32
        )
        range_tensor = torch.tensor(dirty_ranges, dtype=torch.int64)

        if use_custom_stream:
            producer_stream = torch.cuda.Stream(device=0)
            with torch.cuda.stream(producer_stream):
                device_pages.copy_(source_cpu, non_blocking=True)
                task = view.async_copy_dirty_kv_token_ranges_to_host(
                    sequence_tensor,
                    page_count_tensor,
                    range_tensor,
                    device_ptrs,
                    None,
                )
        else:
            device_pages.copy_(source_cpu, non_blocking=True)
            task = view.async_copy_dirty_kv_token_ranges_to_host(
                sequence_tensor,
                page_count_tensor,
                range_tensor,
                device_ptrs,
                None,
            )
        task.wait()

        for sequence_idx, sequence_id in enumerate(sequence_ids):
            actual, v_cpu = view.read_sequence_kv_to_cpu(sequence_id)
            assert v_cpu.numel() == 0
            expected = host_initial.clone()
            start_token, end_token = dirty_ranges[sequence_idx]
            expected_tokens = expected.view(
                NUM_PHYSICAL_LAYERS,
                allocated_pages * PAGE_TOKENS,
                NUM_K_HEADS,
                K_HEAD_DIM,
            )
            source_active = source_cpu[:, sequence_idx].reshape(
                NUM_PHYSICAL_LAYERS,
                active_pages * PAGE_TOKENS,
                NUM_K_HEADS,
                K_HEAD_DIM,
            )
            expected_tokens[:, start_token:end_token].copy_(
                source_active[:, start_token:end_token]
            )
            assert torch.equal(actual, expected)

        null_active_pointer = device_ptrs.clone()
        null_active_pointer[0, 1, 0] = 0
        with pytest.raises(RuntimeError, match="contains null pointer"):
            view.async_copy_dirty_kv_token_ranges_to_host(
                sequence_tensor,
                page_count_tensor,
                range_tensor,
                null_active_pointer,
                None,
            )

        beyond_capacity = range_tensor.clone()
        beyond_capacity[1] = torch.tensor(
            [0, active_pages * PAGE_TOKENS + 1], dtype=torch.int64
        )
        with pytest.raises(IndexError, match="exceeds active token capacity"):
            view.async_copy_dirty_kv_token_ranges_to_host(
                sequence_tensor,
                page_count_tensor,
                beyond_capacity,
                device_ptrs,
                None,
            )

        reversed_range = range_tensor.clone()
        reversed_range[1] = torch.tensor([2, 1], dtype=torch.int64)
        with pytest.raises(IndexError, match="invalid token range"):
            view.async_copy_dirty_kv_token_ranges_to_host(
                sequence_tensor,
                page_count_tensor,
                reversed_range,
                device_ptrs,
                None,
            )

        with pytest.raises(ValueError, match="V cache is disabled"):
            view.async_copy_dirty_kv_token_ranges_to_host(
                sequence_tensor,
                page_count_tensor,
                range_tensor,
                device_ptrs,
                device_ptrs,
            )

        view.release_sequence_pages(sequence_ids)
        view.shutdown()
        view = None
    finally:
        _close_view(view, sequence_ids)
        _shm_unlink(shm_name)


def test_mapped_mla_view_rejects_all_absent_mapping(bg):
    cfg = _make_mapped_mla_config(bg, _random_shm_name())
    cfg.logical_to_physical_layer = [-1, -1, -1]

    with pytest.raises(ValueError, match="non-negative physical layer id"):
        bg.MappedMLAHostPagedKVWorkerView(cfg)
