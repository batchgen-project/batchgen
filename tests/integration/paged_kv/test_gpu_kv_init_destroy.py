import ctypes
import errno
import logging
import multiprocessing as mp
import random
import string
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


def test_gpu_kv_init_destroy() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available")

    device_index = 6
    shm_name = _random_shm_name()
    try:
        engine_config, model_config = _make_deepseek_r1_config(
            shm_name, device_index=device_index
        )
        kv_manager = GPUPagedKVCacheManager(
            engine_config=engine_config,
            model_config=model_config,
        )
        kv_manager.initialize()

        # print gpu memory utilization
        torch.cuda.synchronize(device_index)
        mem_allocated = torch.cuda.memory_allocated(device_index)
        mem_reserved = torch.cuda.memory_reserved(device_index)
        logging.info(
            f"GPU memory allocated: {mem_allocated / (1024**2):.2f} MB"
            f", reserved: {mem_reserved / (1024**2):.2f} MB"
        )

        kv_manager.destroy()
        logging.info("GPUPagedKVCacheManager destroyed successfully")

        mem_allocated_after = torch.cuda.memory_allocated(device_index)
        mem_reserved_after = torch.cuda.memory_reserved(device_index)
        logging.info(
            f"After destroy - GPU memory allocated: {mem_allocated_after / (1024**2):.2f} MB"
            f", reserved: {mem_reserved_after / (1024**2):.2f} MB"
        )

        kv_manager.initialize()
        logging.info("GPUPagedKVCacheManager re-initialized successfully")
        mem_allocated_reinit = torch.cuda.memory_allocated(device_index)
        mem_reserved_reinit = torch.cuda.memory_reserved(device_index)
        logging.info(
            f"After re-initialization - GPU memory allocated: {mem_allocated_reinit / (1024**2):.2f} MB"
            f", reserved: {mem_reserved_reinit / (1024**2):.2f} MB"
        )

    finally:
        # Clean up shared memory
        _shm_unlink(shm_name)


if __name__ == "__main__":
    test_gpu_kv_init_destroy()
