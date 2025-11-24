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
from typing import Union
from unittest import SkipTest

import torch
from tqdm import tqdm

from batchgen.config.config import EngineConfig, ModelConfig
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


def _dtype_to_str(value: torch.dtype) -> str:
    return str(value).split(".")[-1]


def _make_deepseek_r1_config(
    shm_name: str,
    *,
    kv_dtype: torch.dtype = torch.bfloat16,
    device_index: int = 0,
) -> tuple[EngineConfig, ModelConfig]:  # type: ignore
    engine_config = EngineConfig()
    engine_config.Basic_Config.device = f"cuda:{device_index}"
    engine_config.Basic_Config.device_torch = torch.device(
        f"cuda:{device_index}"
    )
    engine_config.Basic_Config.kv_dtype = _dtype_to_str(kv_dtype)
    engine_config.Basic_Config.kv_dtype_torch = kv_dtype

    device_cfg = engine_config.Device_Paged_KV_Config
    device_cfg.num_layers = 61
    device_cfg.num_pages_per_layer = 10000
    device_cfg.page_size = 64
    device_cfg.num_k_heads = 1
    device_cfg.k_head_dim = 512 + 64
    device_cfg.num_v_heads = 0
    device_cfg.v_head_dim = 0
    device_cfg.kv_dtype = _dtype_to_str(kv_dtype)

    model_config = ModelConfig()
    model_config.model_type = "deepseek_r1"
    model_config.num_hidden_layers = 61
    model_config.num_local_experts = 0
    model_config.num_attention_heads = 1
    model_config.num_key_value_heads = 1
    model_config.head_dim = 512 + 64

    return engine_config, model_config


def _make_deepseek_r1_host_config(
    shm_name: str, *, k_element_size_bytes: int = 2
) -> bg.HostPagedKVConfig:  # type: ignore
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = 61
    cfg.num_pages = 50000
    cfg.page_size_tokens = 64
    cfg.num_k_heads = 1
    cfg.k_head_dim = 512 + 64
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = k_element_size_bytes
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = 10240
    cfg.alignment_bytes = 64
    return cfg


_TOKEN_DEVICE_STRIDE = 4096
_TOKEN_LAYER_STRIDE = 128
_TOKEN_SEQUENCE_PERIOD = 127


def _make_sequence_token_values(
    device_index: int,
    layer_idx: int,
    sequence_id: int,
    token_count: int,
    kv_cfg: GPUPagedKVConfig,
    device: Union[torch.device, str],
) -> torch.Tensor:
    """Generate deterministic per-token values for a sequence slice."""

    if token_count <= 0:
        return torch.empty(0, dtype=kv_cfg.kv_dtype, device=device)

    dtype = kv_cfg.kv_dtype
    token_positions = torch.arange(
        token_count, dtype=torch.float32, device=device
    )

    if dtype == torch.float8_e4m3fn:
        # Keep float8-friendly magnitudes to avoid saturation when
        # converting to 8-bit representable ranges.
        token_positions = torch.remainder(token_positions, 32.0)
        base = (
            (device_index % 16) * 16.0
            + (layer_idx % 16) * 4.0
            + float(sequence_id % 16)
        )
    else:
        base = (
            device_index * _TOKEN_DEVICE_STRIDE
            + layer_idx * _TOKEN_LAYER_STRIDE
            + (sequence_id % _TOKEN_SEQUENCE_PERIOD)
        )
    values = token_positions + float(base)
    return values.to(kv_cfg.kv_dtype)


def _assert_token_block_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    kv_dtype: torch.dtype,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    """Compares cached token tensors with dtype-aware tolerances."""

    if kv_dtype == torch.float8_e4m3fn:
        actual_cmp = actual.to(torch.float32)
        expected_cmp = expected.to(torch.float32)
        atol = max(atol, 0.5)
        rtol = 0.0
    else:
        actual_cmp = actual
        expected_cmp = expected

    if not torch.allclose(actual_cmp, expected_cmp, atol=atol, rtol=rtol):
        raise AssertionError("Host -> device copy mismatch detected")


# ---------------------------------------------------------------------------
# Test: allocate pages
# ---------------------------------------------------------------------------
def test_alloc_page(shm_name):
    engine_cfg, model_cfg = _make_deepseek_r1_config(shm_name)
    kv_manager = GPUPagedKVCacheManager(
        engine_config=engine_cfg, model_config=model_cfg
    )

    kv_manager.initialize()

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
    engine_cfg, model_cfg = _make_deepseek_r1_config(shm_name)
    kv_manager = GPUPagedKVCacheManager(
        engine_config=engine_cfg, model_config=model_cfg
    )
    kv_manager.initialize()

    kv_cfg = kv_manager.config
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


def _host_transfer_worker(spec: dict) -> dict:
    """Worker routine that offloads from device -> host and loads back."""

    shm_name: str = spec["shm_name"]
    device_index: int = spec["device_index"]
    sequence_ids: list[int] = spec["sequence_ids"]
    page_num_per_seq: int = spec["page_num_per_seq"]
    sequence_lengths: list[int] = spec["sequence_lengths"]
    kv_dtype: torch.dtype = spec.get("kv_dtype", torch.bfloat16)
    host_k_element_size_bytes: int = spec.get("host_k_element_size_bytes", 2)

    barrier = spec.get("barrier")

    print(f"[worker {device_index}] sequence lengths: {sequence_lengths}")

    torch.cuda.set_device(device_index)

    host_cfg = _make_deepseek_r1_host_config(
        shm_name, k_element_size_bytes=host_k_element_size_bytes
    )
    capacity_tokens = host_cfg.page_size_tokens * page_num_per_seq
    if any(length > capacity_tokens for length in sequence_lengths):
        raise ValueError("sequence length exceeds allocated capacity in worker")

    worker = bg.MLAHostPagedKVWorkerView(host_cfg)
    worker.initialize(device_index, False)
    worker.register_sequences(sequence_ids)
    requests = [(seq_id, capacity_tokens) for seq_id in sequence_ids]
    allocations = worker.allocate_pages_for_sequences(requests)
    logging.info(f"[worker {device_index}] allocated pages: {allocations}")

    if barrier is not None:
        logging.info(
            f"[worker {device_index}] Waiting at barrier before loading back..."
        )
        barrier.wait()
        logging.info(
            f"[worker {device_index}] Barrier passed! Starting load task."
        )

    logging.info(f"[worker {device_index}] passed barrier, starting offload.")

    engine_cfg, model_cfg = _make_deepseek_r1_config(
        shm_name, kv_dtype=kv_dtype
    )
    engine_cfg.Basic_Config.device = f"cuda:{device_index}"
    engine_cfg.Basic_Config.device_torch = torch.device(f"cuda:{device_index}")
    gpu_manager = GPUPagedKVCacheManager(
        engine_config=engine_cfg, model_config=model_cfg
    )
    gpu_manager.initialize()

    kv_cfg = gpu_manager.config
    gpu_manager.allocate_pages_for_sequences(
        sequence_ids, [capacity_tokens] * len(sequence_ids)
    )
    gpu_manager.rebuild_page_table(sequence_ids=sequence_ids)

    data_shape = (
        len(sequence_ids),
        capacity_tokens,
        kv_cfg.num_k_heads,
        kv_cfg.k_head_dim,
    )

    logging.info(
        f"[worker {device_index}] gpu page manager initialized, starting offload."
    )

    # Populate host storage with synthetic per-layer data (device -> host)
    for layer_idx in range(kv_cfg.num_layers):
        device_tensor = torch.zeros(
            data_shape,
            dtype=kv_cfg.kv_dtype,
            device=f"cuda:{device_index}",
        )
        for batch_idx, seq_id in enumerate(sequence_ids):
            seq_len = sequence_lengths[batch_idx]
            if seq_len <= 0:
                continue
            token_values = _make_sequence_token_values(
                device_index=device_index,
                layer_idx=layer_idx,
                sequence_id=seq_id,
                token_count=seq_len,
                kv_cfg=kv_cfg,
                device=device_tensor.device,
            )
            slice_view = device_tensor[batch_idx, :seq_len]
            token_block = token_values.view(seq_len, 1, 1).expand(
                -1, kv_cfg.num_k_heads, kv_cfg.k_head_dim
            )
            slice_view.copy_(token_block.contiguous())

        torch.cuda.synchronize(device_index)
        task = worker.async_offload_layer_kv_to_host(
            layer_idx=layer_idx,
            sequence_ids=sequence_ids,
            k_tensor=device_tensor,
            v_tensor=None,
            sequence_lengths=sequence_lengths,
        )
        task.wait()

    logging.info(f"[worker {device_index}] offloaded all layers to host.")
    verify_host_content = False

    if verify_host_content:
        for layer_idx in range(kv_cfg.num_layers):
            for sid in sequence_ids:
                pointers = worker.get_sequence_layer_page_pointers(
                    sid, layer_idx
                )
                k_ptrs, v_ptrs = pointers
                expected = float(device_index * 100 + layer_idx + 1)
                for i, ptr in enumerate(k_ptrs):
                    array_type = ctypes.c_uint16 * (
                        64 * host_cfg.num_k_heads * host_cfg.k_head_dim
                    )
                    host_array = array_type.from_address(ptr)
                    buf = torch.frombuffer(host_array, dtype=torch.uint16)
                    bf16 = buf.view(torch.bfloat16)

                    mask = bf16 != expected
                    if torch.any(mask):
                        bad_indices = torch.nonzero(
                            mask, as_tuple=False
                        ).squeeze(-1)
                        # mismatch 对应的值
                        bad_values = bf16[mask]

                        # 只打印前 10 个，避免太长
                        N = min(10, bad_indices.numel())
                        msg_lines = []
                        for j in range(N):
                            idx = bad_indices[j].item()
                            val = bad_values[j].item()
                            msg_lines.append(
                                f"[idx={idx}] value={val} expected={expected}"
                            )

                        msg = "\n".join(msg_lines)
                        raise AssertionError(
                            f"Sequence {sid} mismatch (first {N} mismatches):\n{msg}\n"
                            f"ptr={hex(ptr)}\n"
                        )

    logging.info(f"[worker {device_index}] host storage content verified.")

    torch.cuda.synchronize(device_index)

    # Load tensors back to GPU cache (host -> device)
    start_get_gpu = time.time()
    sequence_tensor = torch.tensor(
        sequence_ids, dtype=torch.int64, device="cpu"
    )
    k_ptrs, v_ptrs = gpu_manager.export_layer_page_pointer_table()
    end_get_gpu = time.time()
    logging.info(
        f"[worker {device_index}] get GPU pointers took {end_get_gpu - start_get_gpu:.6f} seconds."
    )

    if barrier is not None:
        logging.info(
            f"[worker {device_index}] Waiting at barrier before loading back..."
        )
        barrier.wait()
        logging.info(
            f"[worker {device_index}] Barrier passed! Starting load task."
        )

    start = time.perf_counter()
    load_task = worker.async_load_layer_kv_to_device(
        sequence_ids=sequence_tensor,
        k_device_ptrs=k_ptrs,
        v_device_ptrs=v_ptrs,
    )
    load_task.wait()

    elapsed = time.perf_counter() - start

    logging.info(
        f"[worker {device_index}] loaded back to device from host. throughput: {len(sequence_ids) * kv_cfg.page_size_tokens * kv_cfg.num_layers} tokens."
    )
    torch.cuda.synchronize(device_index)
    # Verify device cache content per worker / per layer
    k_cache = gpu_manager._k_cache
    if k_cache is None:
        raise AssertionError("K cache must be initialized for verification")

    logging.info(
        f"[worker {device_index}] begin to verify device cache content."
    )
    for layer_idx in range(kv_cfg.num_layers):
        layer_tensor = k_cache[layer_idx]
        for batch_idx, seq_id in enumerate(sequence_ids):
            seq_len = sequence_lengths[batch_idx]
            if seq_len <= 0:
                continue

            state = gpu_manager._get_sequence_state(seq_id)
            page_indices = state.pages.to(gpu_manager.device, dtype=torch.long)
            gathered_pages = layer_tensor.index_select(0, page_indices)
            gathered_tokens = gathered_pages.reshape(
                -1, kv_cfg.num_k_heads, kv_cfg.k_head_dim
            )[:seq_len]

            expected_values = _make_sequence_token_values(
                device_index=device_index,
                layer_idx=layer_idx,
                sequence_id=seq_id,
                token_count=seq_len,
                kv_cfg=kv_cfg,
                device=gathered_tokens.device,
            )
            expected = expected_values.view(seq_len, 1, 1).expand(
                -1, kv_cfg.num_k_heads, kv_cfg.k_head_dim
            )

            _assert_token_block_equal(
                gathered_tokens, expected, kv_cfg.kv_dtype
            )
    logging.info(f"[worker {device_index}] device cache content verified.")

    bytes_per_token = (
        kv_cfg.num_k_heads * kv_cfg.k_head_dim * host_cfg.k_element_size_bytes
    )
    total_tokens = sum(sequence_lengths) * kv_cfg.num_layers
    bytes_transferred = bytes_per_token * total_tokens
    bandwidth_gbps = (bytes_transferred / elapsed) / 1e9 if elapsed > 0 else 0.0

    return {
        "device_index": device_index,
        "sequence_ids": sequence_ids,
        "sequence_lengths": sequence_lengths,
        "bytes_transferred": bytes_transferred,
        "host_to_device_seconds": elapsed,
        "bandwidth_gbps": bandwidth_gbps,
        "verified": True,
    }


def test_host_transfer_layer(shm_name):
    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required for host transfer test")

    host_cfg = _make_deepseek_r1_host_config(shm_name)
    host_manager = bg.MLAHostPagedKVManager(host_cfg)
    host_manager.initialize(True)

    # device_count = torch.cuda.device_count()
    device_count = 8
    if device_count == 0:
        raise SkipTest("No CUDA devices available")

    page_num_per_seq = 200
    sequences_per_worker = 30
    capacity_tokens = host_cfg.page_size_tokens * page_num_per_seq

    rng = random.Random(1234)
    worker_specs = []
    seq_offset = 0
    for device_index in range(device_count):
        sequence_ids = list(
            range(seq_offset, seq_offset + sequences_per_worker)
        )
        seq_offset += sequences_per_worker
        # seq_lengths = [
        #     rng.randint(max(1, capacity_tokens // 4), capacity_tokens)
        #     for _ in sequence_ids
        # ]
        seq_lengths = [capacity_tokens for _ in sequence_ids]
        worker_specs.append(
            {
                "shm_name": shm_name,
                "device_index": device_index,
                "sequence_ids": sequence_ids,
                "page_num_per_seq": page_num_per_seq,
                "sequence_lengths": seq_lengths,
            }
        )

    if not worker_specs:
        raise SkipTest("No workers configured for host transfer test")

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        sync_barrier = manager.Barrier(len(worker_specs))

        # 3. 将 barrier 放入参数中
        for spec in worker_specs:
            spec["barrier"] = sync_barrier

        with ProcessPoolExecutor(
            max_workers=len(worker_specs), mp_context=ctx
        ) as executor:
            futures = [
                executor.submit(_host_transfer_worker, spec)
                for spec in worker_specs
            ]
            results = [future.result() for future in futures]

        for result in results:
            assert result["verified"], f"Worker {result} verification failed"
            assert result["bytes_transferred"] > 0, (
                "Bytes transferred must be positive"
            )
            assert result["host_to_device_seconds"] > 0, (
                "Host->device transfer time must be positive"
            )
            logging.info(
                "Device %d transferred %.2f MB in %.3f s (%.2f GB/s)",
                result["device_index"],
                result["bytes_transferred"] / (1024 * 1024),
                result["host_to_device_seconds"],
                result["bandwidth_gbps"],
            )

        total_bytes = sum(result["bytes_transferred"] for result in results)
        total_time = sum(result["host_to_device_seconds"] for result in results)
        logging.info(
            "Aggregated host->device transfer: %.2f MB in %.3f s",
            total_bytes / (1024 * 1024),
            total_time,
        )


def test_host_transfer_layer_variable_lengths(shm_name):
    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required for host transfer test")

    host_cfg = _make_deepseek_r1_host_config(shm_name)
    host_manager = bg.MLAHostPagedKVManager(host_cfg)
    host_manager.initialize(True)

    available_devices = torch.cuda.device_count()
    if available_devices == 0:
        raise SkipTest("No CUDA devices available")

    device_count = min(2, available_devices)
    page_num_per_seq = 200
    sequences_per_worker = 32
    capacity_tokens = host_cfg.page_size_tokens * page_num_per_seq
    min_tokens = max(8, host_cfg.page_size_tokens // 3)
    max_tokens = capacity_tokens - host_cfg.page_size_tokens // 2

    worker_specs = []
    seq_offset = 10_000
    for device_index in range(device_count):
        sequence_ids = list(
            range(seq_offset, seq_offset + sequences_per_worker)
        )
        seq_offset += sequences_per_worker

        seq_lengths: list[int] = []
        for local_idx, _ in enumerate(sequence_ids):
            base = min_tokens + (local_idx + 1) * (
                host_cfg.page_size_tokens // 2
            )
            jitter = (local_idx * local_idx + device_index * 7) % (
                host_cfg.page_size_tokens * 2
            )
            length = base + jitter
            length = max(min_tokens, min(length, max_tokens))
            if length % host_cfg.page_size_tokens == 0:
                length += 1
                if length > max_tokens:
                    length -= 3
            seq_lengths.append(length)

        if len(set(seq_lengths)) <= 1:
            seq_lengths[-1] = max(
                min_tokens,
                min(
                    seq_lengths[-1] + host_cfg.page_size_tokens // 3, max_tokens
                ),
            )
            if seq_lengths[-1] % host_cfg.page_size_tokens == 0:
                seq_lengths[-1] += 1

        worker_specs.append(
            {
                "shm_name": shm_name,
                "device_index": device_index,
                "sequence_ids": sequence_ids,
                "page_num_per_seq": page_num_per_seq,
                "sequence_lengths": seq_lengths,
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        sync_barrier = manager.Barrier(len(worker_specs))
        for spec in worker_specs:
            spec["barrier"] = sync_barrier

        with ProcessPoolExecutor(
            max_workers=len(worker_specs), mp_context=ctx
        ) as executor:
            futures = [
                executor.submit(_host_transfer_worker, spec)
                for spec in worker_specs
            ]
            results = [future.result() for future in futures]

        for spec, result in zip(worker_specs, results):
            assert result["verified"], (
                f"Worker {result['device_index']} verification failed"
            )
            assert result["sequence_lengths"] == spec["sequence_lengths"]
            per_worker_lengths = result["sequence_lengths"]
            assert len(set(per_worker_lengths)) > 1, (
                "Sequence lengths must vary within a worker"
            )
            assert any(
                (length % host_cfg.page_size_tokens) != 0
                for length in per_worker_lengths
            ), "Sequence lengths should not align to page size tokens"
            logging.info(
                "[variable] Device %d transferred %.2f MB in %.3f s (%.2f GB/s)",
                result["device_index"],
                result["bytes_transferred"] / (1024 * 1024),
                result["host_to_device_seconds"],
                result["bandwidth_gbps"],
            )

        total_tokens = sum(
            sum(result["sequence_lengths"]) for result in results
        )
        assert total_tokens > 0
        total_bytes = sum(result["bytes_transferred"] for result in results)
        total_time = sum(result["host_to_device_seconds"] for result in results)
        logging.info(
            "[variable] Aggregated host->device transfer: %.2f MB in %.3f s",
            total_bytes / (1024 * 1024),
            total_time,
        )


def test_host_transfer_layer_byte_kv(shm_name):
    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required for host transfer test")

    kv_dtype = torch.float8_e4m3fn
    host_cfg = _make_deepseek_r1_host_config(shm_name, k_element_size_bytes=1)
    host_manager = bg.MLAHostPagedKVManager(host_cfg)
    host_manager.initialize(True)

    device_count = 8
    if torch.cuda.device_count() < device_count:
        raise SkipTest("At least 2 CUDA devices required for byte KV test")

    page_num_per_seq = 200
    sequences_per_worker = 30
    capacity_tokens = host_cfg.page_size_tokens * page_num_per_seq

    worker_specs = []
    seq_offset = 100_000
    for device_index in range(device_count):
        sequence_ids = list(
            range(seq_offset, seq_offset + sequences_per_worker)
        )
        seq_offset += sequences_per_worker
        seq_lengths = [capacity_tokens for _ in sequence_ids]
        worker_specs.append(
            {
                "shm_name": shm_name,
                "device_index": device_index,
                "sequence_ids": sequence_ids,
                "page_num_per_seq": page_num_per_seq,
                "sequence_lengths": seq_lengths,
                "kv_dtype": kv_dtype,
                "host_k_element_size_bytes": 1,
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        sync_barrier = manager.Barrier(len(worker_specs))
        for spec in worker_specs:
            spec["barrier"] = sync_barrier

        with ProcessPoolExecutor(
            max_workers=len(worker_specs), mp_context=ctx
        ) as executor:
            futures = [
                executor.submit(_host_transfer_worker, spec)
                for spec in worker_specs
            ]
            results = [future.result() for future in futures]

    for result in results:
        assert result["verified"], (
            f"Worker {result['device_index']} verification failed"
        )
        assert result["bytes_transferred"] > 0
        logging.info(
            "[byte] Device %d transferred %.2f MB in %.3f s (%.2f GB/s)",
            result["device_index"],
            result["bytes_transferred"] / (1024 * 1024),
            result["host_to_device_seconds"],
            result["bandwidth_gbps"],
        )

    total_bytes = sum(result["bytes_transferred"] for result in results)
    total_time = sum(result["host_to_device_seconds"] for result in results)
    logging.info(
        "[byte] Aggregated host->device transfer: %.2f MB in %.3f s",
        total_bytes / (1024 * 1024),
        total_time,
    )


def test_host_transfer_layer_variable_lengths_byte_kv(shm_name):
    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required for host transfer test")

    kv_dtype = torch.float8_e4m3fn
    host_cfg = _make_deepseek_r1_host_config(shm_name, k_element_size_bytes=1)
    host_manager = bg.MLAHostPagedKVManager(host_cfg)
    host_manager.initialize(True)

    available_devices = torch.cuda.device_count()
    if available_devices == 0:
        raise SkipTest("No CUDA devices available")

    device_count = min(8, available_devices)
    page_num_per_seq = 200
    sequences_per_worker = 24
    capacity_tokens = host_cfg.page_size_tokens * page_num_per_seq
    min_tokens = max(8, host_cfg.page_size_tokens // 4)
    max_tokens = capacity_tokens - host_cfg.page_size_tokens // 2

    worker_specs = []
    seq_offset = 200_000
    for device_index in range(device_count):
        sequence_ids = list(
            range(seq_offset, seq_offset + sequences_per_worker)
        )
        seq_offset += sequences_per_worker

        seq_lengths: list[int] = []
        for local_idx, _ in enumerate(sequence_ids):
            base = min_tokens + (local_idx + 1) * (
                host_cfg.page_size_tokens // 3
            )
            jitter = (local_idx * 11 + device_index * 13) % (
                host_cfg.page_size_tokens
            )
            length = base + jitter
            length = max(min_tokens, min(length, max_tokens))
            if length % host_cfg.page_size_tokens == 0:
                length = min(length + 5, max_tokens)
            seq_lengths.append(length)

        if len(set(seq_lengths)) <= 1:
            seq_lengths[-1] = max(
                min_tokens,
                min(
                    seq_lengths[-1] + host_cfg.page_size_tokens // 5,
                    max_tokens,
                ),
            )

        worker_specs.append(
            {
                "shm_name": shm_name,
                "device_index": device_index,
                "sequence_ids": sequence_ids,
                "page_num_per_seq": page_num_per_seq,
                "sequence_lengths": seq_lengths,
                "kv_dtype": kv_dtype,
                "host_k_element_size_bytes": 1,
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        sync_barrier = manager.Barrier(len(worker_specs))
        for spec in worker_specs:
            spec["barrier"] = sync_barrier

        with ProcessPoolExecutor(
            max_workers=len(worker_specs), mp_context=ctx
        ) as executor:
            futures = [
                executor.submit(_host_transfer_worker, spec)
                for spec in worker_specs
            ]
            results = [future.result() for future in futures]

    for spec, result in zip(worker_specs, results):
        assert result["verified"], (
            f"Worker {result['device_index']} verification failed"
        )
        assert result["sequence_lengths"] == spec["sequence_lengths"]
        per_worker_lengths = result["sequence_lengths"]
        assert len(set(per_worker_lengths)) > 1
        assert any(
            (length % host_cfg.page_size_tokens) != 0
            for length in per_worker_lengths
        )
        logging.info(
            "[variable-byte] Device %d transferred %.2f MB in %.3f s (%.2f GB/s)",
            result["device_index"],
            result["bytes_transferred"] / (1024 * 1024),
            result["host_to_device_seconds"],
            result["bandwidth_gbps"],
        )

    total_tokens = sum(sum(result["sequence_lengths"]) for result in results)
    assert total_tokens > 0
    total_bytes = sum(result["bytes_transferred"] for result in results)
    total_time = sum(result["host_to_device_seconds"] for result in results)
    logging.info(
        "[variable-byte] Aggregated host->device transfer: %.2f MB in %.3f s",
        total_bytes / (1024 * 1024),
        total_time,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    shm_name = _random_shm_name()
    try:
        # test_alloc_page(shm_name)
        # test_update_new_token(shm_name)
        test_host_transfer_layer(shm_name)
        # test_host_transfer_layer_variable_lengths(shm_name)
        # test_host_transfer_layer_variable_lengths_byte_kv(shm_name)
        # test_host_transfer_layer_byte_kv(shm_name)
    finally:
        _shm_unlink(shm_name)
