import ctypes
import errno
import logging
import math
import multiprocessing as mp
import os
import random
import string
import time
from concurrent.futures import ProcessPoolExecutor

import torch
from tqdm import tqdm

from batchgen.config.config import EngineConfig
from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)
from batchgen.models.engine_loader import core_engine as bg

logging.basicConfig(
    level=logging.INFO,  # Set to the lowest level to capture all messages
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Customize timestamp format
)

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _random_shm_name() -> str:
    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )
    return f"/batchgen_kv_{suffix}"


def _shm_unlink(name: str) -> None:
    if not name:
        return
    res = _libc.shm_unlink(name.encode("utf-8"))
    if res != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, f"shm_unlink({name}) failed")


def _make_deepseek_r1_config(
    shm_name: str, device_index: int = 0
) -> EngineConfig:  # type: ignore
    cfg = GPUPagedKVConfig(
        num_layers=61,
        num_pages=10000,
        page_size_tokens=64,
        num_k_heads=1,
        k_head_dim=512 + 64,
        num_v_heads=0,
        v_head_dim=0,
        kv_dtype=torch.bfloat16,
    )

    engine_config = EngineConfig()
    engine_config.gpu_kv_config = cfg
    engine_config.Basic_Config.device = device_index
    return engine_config


def _make_deepseek_r1_host_config(shm_name: str) -> bg.HostPagedKVConfig:  # type: ignore
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = 61
    cfg.num_pages = 10000
    cfg.page_size_tokens = 64
    cfg.num_k_heads = 1
    cfg.k_head_dim = 512 + 64
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = 2
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = 10240
    cfg.alignment_bytes = 64
    return cfg


# ---------------------------------------------------------------------------
# Test: allocate pages
# ---------------------------------------------------------------------------
def test_alloc_page(shm_name):
    cfg = _make_deepseek_r1_config(shm_name)
    kv_manager = GPUPagedKVCacheManager(cfg)

    kv_manager.initialize(None)

    seq_ids = [i for i in range(10)]
    num_tokens = [1000 + i * 100 for i in range(10)]

    allocated_pages = kv_manager.allocate_pages_for_sequences(
        seq_ids, num_tokens
    )
    print("Allocated pages:", allocated_pages)


# ---------------------------------------------------------------------------
# Test: update_new_token
# ---------------------------------------------------------------------------
def test_update_new_token(shm_name: str) -> None:
    engine_cfg = _make_deepseek_r1_config(shm_name)
    kv_manager = GPUPagedKVCacheManager(engine_cfg)
    kv_manager.initialize(core_engine=None)

    kv_cfg = engine_cfg.gpu_kv_config
    layer_idx = 3

    # ---- choose token positions to cover不同场景 ----
    page_size = kv_cfg.page_size_tokens
    seq_ids = [i for i in range(32)]

    # 依次覆盖：
    #  - 第一页开头      (page 0, offset 0)
    #  - 第二页开头      (page 1, offset 0)
    #  - 更深的某一页中间 (page 16, offset 39)
    seq_lens = [((page_size * i) + i) for i in range(32)]
    print(seq_lens)
    seq_lens_tensor = torch.tensor(
        seq_lens, dtype=torch.int32, device=kv_manager.device
    )
    max_len = max(seq_lens) + 1  # 至少能容纳这些 token index

    # ---- allocate enough pages for each sequence ----
    kv_manager.allocate_pages_for_sequences(
        seq_ids,
        [max_len] * len(seq_ids),
    )

    kv_manager.rebuild_page_table(sequence_ids=seq_ids)

    # ---- build single-token K tensor for each sequence ----
    batch = len(seq_ids)
    heads = kv_cfg.num_k_heads
    dim = kv_cfg.k_head_dim

    torch.manual_seed(1234)
    k_tensor = torch.randn(
        (batch, 1, heads, dim),
        dtype=kv_cfg.kv_dtype,
        device=kv_manager.device,
    )

    # ---- call update_new_token ----
    torch.cuda.set_device(kv_manager.device)
    for i in range(100):
        kv_manager.update_new_token(
            k_tensor=k_tensor,
            v_tensor=None,
            sequence_lengths=seq_lens_tensor,
            layer_idx=layer_idx,
        )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(10):
        kv_manager.update_new_token(
            k_tensor=k_tensor,
            v_tensor=None,
            sequence_lengths=seq_lens_tensor,
            layer_idx=layer_idx,
        )

    end_event.record()
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event)  # milliseconds
    print(f"update_new_token took {elapsed:.6f} ms")

    block_k, _, page_table = kv_manager.get_layer_kv_with_page_table(layer_idx)

    print(page_table)

    # ---- assertions ----
    # 使用内部的 _resolve_token_location 复用逻辑，
    # 避免测试里再手写一遍 page/offset 计算。
    for b, seq_id in enumerate(seq_ids):
        token_index = seq_lens[b]
        state = kv_manager._get_sequence_state(seq_id)

        gpu_page, offset = kv_manager._resolve_token_location(
            state,
            sequence_id=seq_id,
            token_index=token_index,
            context="test_update_new_token",
        )

        cached = kv_manager._k_cache[layer_idx, gpu_page, offset]
        src = k_tensor[b, 0]

        # copy_ 是逐元素拷贝，这里用 equal 就行
        assert torch.equal(cached, src), (
            f"update_new_token wrote wrong data for "
            f"seq={seq_id}, token_index={token_index}, "
            f"page={gpu_page}, offset={offset}"
        )

    print("update_new_token test passed ✓")


def _host_transfer_worker(
    shm_name: str,  
    device_index: int,
    sequence_ids: list[int],
):
    cfg = _make_deepseek_r1_host_config(shm_name)
    worker = bg.MLAHostPagedKVWorkerView(cfg)
    worker.initialize(device_index, False)

    worker.register_sequences(sequence_ids)
    capacity_tokens = cfg.page_size_tokens * 10
    requests = [(seq_id, capacity_tokens) for seq_id in sequence_ids]
    worker.allocate_pages_for_sequences(requests)


    # offload kv from device to host
    for i in range(cfg.num_layers):
        device_tensor = torch.full(
            (
                SEQ_NUM_PER_WORKER,
                capacity_tokens,
                cfg.num_k_heads,
                cfg.k_head_dim,
            ),
            i + 1,
            device="cuda:6",
            dtype=torch.bfloat16,
        )

        requests = [(seq_id, capacity_tokens) for seq_id in sequence_ids]
        task = worker.async_offload_layer_kv_to_host(
            layer_idx=i,
            sequence_ids=sequence_ids,
            k_tensor=device_tensor,
            v_tensor=None,
            sequence_lengths=[length for (_, length) in requests],
        )

        task.wait()




def test_host_transfer_layer(shm_name):
    PAGE_NUM_PER_SEQ = 10
    NUM_WORKERS = 8
    SEQ_NUM_PER_WORKER = 10
    shm_name = _random_shm_name()
    cfg = _make_deepseek_r1_host_config(shm_name)
    host_manager = bg.MLAHostPagedKVManager(cfg)

    sequence_ids = [i for i in range(0, SEQ_NUM_PER_WORKER)]
    capacity_tokens = cfg.page_size_tokens * PAGE_NUM_PER_SEQ
    capacities = [capacity_tokens] * (SEQ_NUM_PER_WORKER)

    host_manager.initialize(True)

    worker = bg.MLAHostPagedKVWorkerView(cfg)
    worker.initialize(6, False)

    worker.register_sequences(sequence_ids)
    requests = [(seq_id, capacity_tokens) for seq_id in sequence_ids]
    worker.allocate_pages_for_sequences(requests)

    # offload kv from device to host
    for i in range(cfg.num_layers):
        device_tensor = torch.full(
            (
                SEQ_NUM_PER_WORKER,
                capacity_tokens,
                cfg.num_k_heads,
                cfg.k_head_dim,
            ),
            i + 1,
            device="cuda:6",
            dtype=torch.bfloat16,
        )

        requests = [(seq_id, capacity_tokens) for seq_id in sequence_ids]
        task = worker.async_offload_layer_kv_to_host(
            layer_idx=i,
            sequence_ids=sequence_ids,
            k_tensor=device_tensor,
            v_tensor=None,
            sequence_lengths=[length for (_, length) in requests],
        )

        task.wait()

    # offload complete
    torch.cuda.set_device(6)
    torch.cuda.synchronize()

    # device manager init
    device_config = _make_deepseek_r1_config(shm_name)
    device_manager = GPUPagedKVCacheManager(device_config)
    device_manager.initialize(core_engine=None)
    device_manager.allocate_pages_for_sequences(
        sequence_ids, [capacity_tokens] * (SEQ_NUM_PER_WORKER)
    )
    device_manager.rebuild_page_table(sequence_ids=sequence_ids)

    # transfer kv from host to device
    tasks = []
    
    for seq in sequence_ids:
        pages = device_manager._get_sequence_state(seq).pages.tolist()
        layer_page_pointer_batch = (
            device_manager.build_layer_page_pointer_batch(
                page_indices=pages,
            )
        )
        tasks.append(
            worker.async_load_layer_kv_to_device(
                layer_indices=layer_page_pointer_batch.layer_ids,
                page_indices=layer_page_pointer_batch.page_indices,
                k_device_ptrs=layer_page_pointer_batch.k_ptrs,
                v_device_ptrs=layer_page_pointer_batch.v_ptrs,
            )
        )

    for task in tasks:
        task.wait()

    # check gpu data correctness
    torch.cuda.set_device(device_manager.device)
    k_cache = device_manager._k_cache
    if k_cache is None:
        raise AssertionError("K cache must be initialized for verification")

    for layer_idx in range(cfg.num_layers):
        layer_tensor = k_cache[layer_idx]
        expected_value = float(layer_idx + 1)
        for seq_id in sequence_ids:
            state = device_manager._get_sequence_state(seq_id)
            page_indices = state.pages.to(
                device_manager.device, dtype=torch.long
            )
            gathered = layer_tensor.index_select(0, page_indices)
            print(f"Layer {layer_idx}, Seq {seq_id}, Gathered: {gathered}")
            expected = torch.full_like(gathered, expected_value)
            if not torch.equal(gathered, expected):
                raise AssertionError(
                    "Host→device copy mismatch for layer "
                    f"{layer_idx}, seq {seq_id}"
                )

    print("Host to device transfer complete")

    # device_manager.free_pages_for_sequences(sequence_ids)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    shm_name = _random_shm_name()
    try:
        # test_alloc_page(shm_name)
        # test_update_new_token(shm_name)
        test_host_transfer_layer(shm_name)
    finally:
        _shm_unlink(shm_name)
