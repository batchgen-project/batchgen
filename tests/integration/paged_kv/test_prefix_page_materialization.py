import ctypes
import errno
import math
import random
import string

import pytest
import torch

from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)
from batchgen.models.engine_loader import core_engine as bg


_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def _random_shm_name() -> str:
    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )
    return f"/batchgen_prefix_pages_{suffix}"


def _shm_unlink(name: str) -> None:
    result = _LIBC.shm_unlink(name.encode("utf-8"))
    if result != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, f"shm_unlink({name}) failed")


def _host_config(shm_name: str) -> bg.HostPagedKVConfig:
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = 2
    cfg.num_pages = 16
    cfg.page_size_tokens = 4
    cfg.num_k_heads = 1
    cfg.k_head_dim = 2
    cfg.num_v_heads = 1
    cfg.v_head_dim = 2
    cfg.k_element_size_bytes = 2
    cfg.v_element_size_bytes = 2
    cfg.sequence_table_capacity = 16
    cfg.alignment_bytes = 64
    return cfg


def _gpu_config() -> GPUPagedKVConfig:
    return GPUPagedKVConfig(
        num_layers=2,
        num_pages=16,
        page_size_tokens=4,
        num_k_heads=1,
        k_head_dim=2,
        num_v_heads=1,
        v_head_dim=2,
        kv_dtype=torch.bfloat16,
    )


def _read_sequence_tokens(
    manager: GPUPagedKVCacheManager,
    *,
    sequence_id: int,
    layer_idx: int,
    length: int,
    value_cache: bool,
) -> torch.Tensor:
    cache = manager._v_cache if value_cache else manager._k_cache
    pages = manager._sequences[sequence_id].pages.tolist()
    chunks = []
    remaining = int(length)
    for page in pages:
        if remaining <= 0:
            break
        take = min(remaining, manager.config.page_size_tokens)
        chunks.append(cache[layer_idx, page, :take].detach().cpu())
        remaining -= take
    return torch.cat(chunks, dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_load_prefix_pages_to_device_uses_host_page_ids():
    shm_name = _random_shm_name()
    source_seq = 101
    target_seq = 202
    prefix_tokens = 5
    full_tokens = 7
    page_size = 4
    prefix_pages = math.ceil(prefix_tokens / page_size)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    host_manager = bg.DefaultHostPagedKVManager(_host_config(shm_name))
    host_manager.initialize(True)
    worker = bg.DefaultHostPagedKVWorkerView(_host_config(shm_name))
    worker.initialize(0, False)

    try:
        worker.register_sequences([source_seq])
        host_pages = worker.allocate_pages_for_sequences(
            [(source_seq, prefix_pages * page_size)]
        )[0]

        expected_k = {}
        expected_v = {}
        for layer_idx in range(2):
            base = float(10 * (layer_idx + 1))
            k_tensor = (
                torch.arange(
                    prefix_tokens * 2, dtype=torch.float32, device=device
                )
                .reshape(1, prefix_tokens, 1, 2)
                .add(base)
                .to(torch.bfloat16)
            )
            v_tensor = (k_tensor + 100).contiguous()
            expected_k[layer_idx] = k_tensor.detach().cpu().squeeze(0)
            expected_v[layer_idx] = v_tensor.detach().cpu().squeeze(0)
            task = worker.async_offload_layer_kv_to_host(
                layer_idx=layer_idx,
                sequence_ids=[source_seq],
                k_tensor=k_tensor.contiguous(),
                v_tensor=v_tensor,
                sequence_lengths=[prefix_tokens],
            )
            task.wait()

        gpu_manager = GPUPagedKVCacheManager(
            config=_gpu_config(),
            device=device,
        )
        gpu_manager.initialize()
        gpu_manager.allocate_pages_for_sequences([target_seq], [full_tokens])
        gpu_manager.rebuild_page_table([target_seq])
        k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
        active_page_counts = torch.tensor([prefix_pages], dtype=torch.int64)
        host_page_ids = torch.tensor(
            [host_pages[:prefix_pages]],
            dtype=torch.int64,
        )

        load_task = worker.async_load_prefix_pages_to_device(
            host_page_ids=host_page_ids,
            active_page_counts=active_page_counts,
            k_device_ptrs=k_ptrs,
            v_device_ptrs=v_ptrs,
        )
        load_task.wait()
        torch.cuda.synchronize(device)

        for layer_idx in range(2):
            actual_k = _read_sequence_tokens(
                gpu_manager,
                sequence_id=target_seq,
                layer_idx=layer_idx,
                length=prefix_tokens,
                value_cache=False,
            )
            actual_v = _read_sequence_tokens(
                gpu_manager,
                sequence_id=target_seq,
                layer_idx=layer_idx,
                length=prefix_tokens,
                value_cache=True,
            )
            torch.testing.assert_close(actual_k, expected_k[layer_idx])
            torch.testing.assert_close(actual_v, expected_v[layer_idx])
    finally:
        try:
            host_manager.free_sequence(source_seq)
        except Exception:
            pass
        _shm_unlink(shm_name)
