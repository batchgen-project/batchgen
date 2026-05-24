import ctypes
import errno
import random
import string

import pytest
import torch

_libc = ctypes.CDLL("libc.so.6", use_errno=True)

PAGE_TOKENS = 4
NUM_PHYSICAL_LAYERS = 2
NUM_PAGES = 64
NUM_K_HEADS = 1
K_HEAD_DIM = 4
K_ELEMENT_SIZE_BYTES = 2
ALIGNMENT_BYTES = 64
LOGICAL_TO_PHYSICAL = [-1, 0, 1]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="transformed HostKV tests require CUDA",
)


@pytest.fixture(scope="module")
def bg():
    from batchgen.models.engine_loader import core_engine as bg_module

    return bg_module


def _random_shm_name(prefix: str) -> str:
    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )
    return f"/batchgen_{prefix}_{suffix}"


def _shm_unlink(name: str) -> None:
    if not name:
        return
    res = _libc.shm_unlink(name.encode("utf-8"))
    if res != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, f"shm_unlink({name}) failed")


def _make_config(bg, shm_name: str):
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
    cfg.sequence_table_capacity = 128
    cfg.alignment_bytes = ALIGNMENT_BYTES
    cfg.logical_to_physical_layer = list(LOGICAL_TO_PHYSICAL)
    cfg.logger_name = "TransformedHostKVTest"
    return cfg


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


def _expected_token(value: float) -> torch.Tensor:
    return torch.full(
        (NUM_K_HEADS, K_HEAD_DIM),
        value,
        dtype=torch.bfloat16,
    )


def test_compressed_ratio_host_batched_append_skips_pending_rows(bg):
    shm_name = _random_shm_name("compressed_ratio_host")
    view = None
    sequence_ids = [101, 102, 103, 104]
    raw_positions = [2, 3, 6, 7]

    try:
        torch.cuda.set_device(0)
        view = bg.CompressedRatio4MappedMLAHostPagedKVWorkerView(
            _make_config(bg, shm_name)
        )
        view.initialize(0, True)
        view.register_sequences(sequence_ids)
        allocations = view.allocate_pages_for_sequences(
            [(sequence_id, 8) for sequence_id in sequence_ids]
        )
        assert all(len(pages) == 1 for pages in allocations)

        tokens = torch.empty(
            (len(sequence_ids), 1, NUM_K_HEADS, K_HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        for batch_idx in range(len(sequence_ids)):
            tokens[batch_idx].fill_(float(10 + batch_idx))

        view.async_append_decode_kv_to_host_batched_kernel(
            [(2, tokens, None)],
            sequence_ids,
            raw_positions,
        ).wait()

        # Only raw positions 3 and 7 complete ratio-4 blocks, so only seq 102
        # and 104 should be materialized at compressed positions 0 and 1.
        k_102, _ = view.read_sequence_kv_to_cpu(102)
        assert torch.equal(k_102[1, 0, 0], _expected_token(11.0))

        k_104, _ = view.read_sequence_kv_to_cpu(104)
        assert torch.equal(k_104[1, 0, 1], _expected_token(13.0))

        k_101, _ = view.read_sequence_kv_to_cpu(101)
        k_103, _ = view.read_sequence_kv_to_cpu(103)
        assert torch.equal(k_101[1, 0, 0], torch.zeros_like(k_101[1, 0, 0]))
        assert torch.equal(k_103[1, 0, 0], torch.zeros_like(k_103[1, 0, 0]))
    finally:
        _close_view(view, sequence_ids)
        _shm_unlink(shm_name)


def test_swa_host_view_keeps_full_history_and_exposes_window_range(bg):
    shm_name = _random_shm_name("swa_host")
    view = None
    sequence_ids = [201]

    try:
        torch.cuda.set_device(0)
        view = bg.SWAMappedMLAHostPagedKVWorkerView(
            _make_config(bg, shm_name),
            8,
        )
        view.initialize(0, True)
        view.register_sequences(sequence_ids)
        allocations = view.allocate_pages_for_sequences([(201, 13)])
        assert len(allocations[0]) == 4
        first_page = allocations[0][0]

        token = torch.full(
            (1, 1, NUM_K_HEADS, K_HEAD_DIM),
            42.0,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        view.async_append_decode_kv_to_host(
            1,
            sequence_ids,
            token,
            None,
            [12],
        ).wait()

        page_table = view.build_page_table(sequence_ids)
        assert len(page_table[0]) == 4
        assert page_table[0][0] == first_page

        window_range = view.compute_swa_host_page_range(201, 13)
        assert window_range.sequence_id == 201
        assert window_range.raw_context_len == 13
        assert window_range.window_start_token == 5
        assert window_range.first_page == 1
        assert window_range.page_count == 3
        assert window_range.local_kv_len == 9
        assert window_range.mask_start == 1

        ranges = view.compute_swa_host_page_ranges([201], [13])
        assert len(ranges) == 1
        assert ranges[0].first_page == window_range.first_page
        assert ranges[0].page_count == window_range.page_count

        all_k_ptrs, all_v_ptrs = view.get_sequence_layer_page_pointers(
            201,
            1,
            None,
        )
        window_k_ptrs, window_v_ptrs = (
            view.get_sequence_layer_swa_window_page_pointers(
                201,
                1,
                13,
            )
        )
        assert all_v_ptrs is None
        assert window_v_ptrs is None
        assert window_k_ptrs == all_k_ptrs[1:4]

        k_cpu, _ = view.read_sequence_kv_to_cpu(201)
        assert torch.equal(k_cpu[0, 3, 0], _expected_token(42.0))
    finally:
        _close_view(view, sequence_ids)
        _shm_unlink(shm_name)
