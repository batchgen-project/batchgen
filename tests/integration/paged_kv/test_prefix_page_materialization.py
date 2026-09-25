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
from batchgen.prefix_reuse.materialization import (
    _expand_device_ptrs_for_host_pages,
)


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


def _gpu_config(page_size_tokens: int) -> GPUPagedKVConfig:
    return GPUPagedKVConfig(
        num_layers=2,
        num_pages=16,
        page_size_tokens=page_size_tokens,
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
@pytest.mark.parametrize("gpu_page_size", [4, 8])
def test_async_load_prefix_pages_to_device_uses_host_page_ids(gpu_page_size):
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
                torch.arange(prefix_tokens * 2, dtype=torch.float32, device=device)
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
            config=_gpu_config(gpu_page_size),
            device=device,
        )
        gpu_manager.initialize()
        gpu_manager.allocate_pages_for_sequences([target_seq], [full_tokens])
        gpu_manager.rebuild_page_table([target_seq])
        k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
        active_page_counts = torch.tensor([prefix_pages], dtype=torch.int64)
        k_ptrs, v_ptrs = _expand_device_ptrs_for_host_pages(
            gpu_manager=gpu_manager,
            k_device_ptrs=k_ptrs,
            v_device_ptrs=v_ptrs,
            active_page_counts=active_page_counts,
            host_page_tokens=page_size,
        )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_gpu_prefix_page_survives_one_sequence_release():
    shm_name = _random_shm_name()
    source_seq = 301
    target_seqs = [401, 402]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    host_manager = bg.DefaultHostPagedKVManager(_host_config(shm_name))
    host_manager.initialize(True)
    worker = bg.DefaultHostPagedKVWorkerView(_host_config(shm_name))
    worker.initialize(0, False)
    gpu_manager = None
    try:
        worker.register_sequences([source_seq])
        host_page = worker.allocate_pages_for_sequences([(source_seq, 4)])[0][0]
        expected = {}
        for layer_idx in range(2):
            key = torch.arange(
                8, dtype=torch.float32, device=device
            ).reshape(1, 4, 1, 2).add(10 * layer_idx).to(torch.bfloat16)
            value = (key + 100).contiguous()
            expected[layer_idx] = (key.cpu().squeeze(0), value.cpu().squeeze(0))
            worker.async_offload_layer_kv_to_host(
                layer_idx=layer_idx,
                sequence_ids=[source_seq],
                k_tensor=key.contiguous(),
                v_tensor=value,
                sequence_lengths=[4],
            ).wait()

        gpu_manager = GPUPagedKVCacheManager(
            config=_gpu_config(4), device=device
        )
        gpu_manager.initialize()
        gpu_manager.allocate_pages_for_sequences_with_page_keys(
            target_seqs,
            [4, 4],
            [[host_page], [host_page]],
        )
        first_page = int(gpu_manager._sequences[target_seqs[0]].pages[0])
        second_page = int(gpu_manager._sequences[target_seqs[1]].pages[0])
        assert first_page == second_page

        gpu_manager.rebuild_page_table(target_seqs)
        k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
        worker.async_load_prefix_pages_to_device(
            host_page_ids=torch.tensor(
                [[host_page], [host_page]], dtype=torch.int64
            ),
            active_page_counts=torch.tensor([1, 1], dtype=torch.int64),
            k_device_ptrs=k_ptrs,
            v_device_ptrs=v_ptrs,
        ).wait()
        torch.cuda.synchronize(device)

        gpu_manager.free_pages_for_sequences([target_seqs[0]])
        assert gpu_manager.get_stats().num_used_pages == 1
        for layer_idx in range(2):
            actual_k = _read_sequence_tokens(
                gpu_manager,
                sequence_id=target_seqs[1],
                layer_idx=layer_idx,
                length=4,
                value_cache=False,
            )
            actual_v = _read_sequence_tokens(
                gpu_manager,
                sequence_id=target_seqs[1],
                layer_idx=layer_idx,
                length=4,
                value_cache=True,
            )
            torch.testing.assert_close(actual_k, expected[layer_idx][0])
            torch.testing.assert_close(actual_v, expected[layer_idx][1])

        gpu_manager.free_pages_for_sequences([target_seqs[1]])
        assert gpu_manager.get_stats().num_used_pages == 0
    finally:
        if gpu_manager is not None:
            gpu_manager.destroy()
        try:
            host_manager.free_sequence(source_seq)
        except Exception:
            pass
        _shm_unlink(shm_name)
