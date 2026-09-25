import ctypes
import os
import uuid

import pytest
import torch

from batchgen.models.engine_loader import core_engine as bg


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _shm_unlink(name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.shm_unlink(name.encode()) != 0 and ctypes.get_errno() != 2:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))


def test_prefix_range_offload_preserves_shared_prefix_tokens():
    shm_name = f"/prefix_range_{uuid.uuid4().hex}"
    config = bg.HostPagedKVConfig()
    config.shm_name = shm_name
    config.num_layers = 1
    config.num_pages = 8
    config.page_size_tokens = 4
    config.num_k_heads = 1
    config.k_head_dim = 2
    config.num_v_heads = 1
    config.v_head_dim = 2
    config.k_element_size_bytes = 2
    config.v_element_size_bytes = 2
    config.sequence_table_capacity = 8
    view = bg.DefaultHostPagedKVWorkerView(config)
    sequence_id = 91
    try:
        view.initialize(0, True)
        view.register_sequences([sequence_id])
        view.allocate_pages_for_sequences([(sequence_id, 8)])
        key = torch.full(
            (1, 4, 1, 2), 7.0, dtype=torch.bfloat16, device="cuda:0"
        )
        value = torch.full_like(key, 11.0)

        task = view.async_offload_layer_kv_range_to_host(
            layer_idx=0,
            sequence_ids=[sequence_id],
            k_tensor=key,
            v_tensor=value,
            raw_start_positions=[2],
            token_counts=[4],
        )
        task.wait()

        k_cpu, v_cpu = view.read_sequence_kv_to_cpu(sequence_id)
        torch.testing.assert_close(k_cpu[0, 0, :2], torch.zeros_like(k_cpu[0, 0, :2]))
        torch.testing.assert_close(k_cpu[0, 0, 2:], torch.full_like(k_cpu[0, 0, 2:], 7.0))
        torch.testing.assert_close(k_cpu[0, 1, :2], torch.full_like(k_cpu[0, 1, :2], 7.0))
        torch.testing.assert_close(v_cpu[0, 0, :2], torch.zeros_like(v_cpu[0, 0, :2]))
        torch.testing.assert_close(v_cpu[0, 0, 2:], torch.full_like(v_cpu[0, 0, 2:], 11.0))
        torch.testing.assert_close(v_cpu[0, 1, :2], torch.full_like(v_cpu[0, 1, :2], 11.0))
    finally:
        try:
            view.release_sequence_pages([sequence_id])
        except Exception:
            pass
        try:
            view.shutdown()
        except Exception:
            pass
        del view
        _shm_unlink(shm_name)
