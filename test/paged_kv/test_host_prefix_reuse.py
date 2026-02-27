import ctypes
import errno
import random
import string
from unittest import SkipTest

import torch

from batchgen.models.engine_loader import core_engine as bg

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _random_shm_name() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"/batchgen_prefix_{suffix}"


def _shm_unlink(name: str) -> None:
    if not name:
        return
    rc = _libc.shm_unlink(name.encode("utf-8"))
    if rc != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, f"shm_unlink({name}) failed")


def _make_mla_config(
    shm_name: str, enable_prefix_reuse: bool
) -> bg.HostPagedKVConfig:  # type: ignore
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = 1
    cfg.num_pages = 512
    cfg.page_size_tokens = 64
    cfg.num_k_heads = 1
    cfg.k_head_dim = 576
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = 2
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = 1024
    cfg.alignment_bytes = 64

    cfg.enable_prefix_reuse = enable_prefix_reuse
    cfg.prefix_min_reuse_pages = 1
    cfg.prefix_min_store_pages = 1
    cfg.sequence_page_node_capacity = 2048
    cfg.radix_node_capacity = 1024
    cfg.radix_edge_capacity = 2048
    cfg.prefix_entry_capacity = 256
    cfg.prefix_page_ref_capacity = 1024
    cfg.prefix_page_budget = 128
    return cfg


def test_prefix_reuse_disabled_regression() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available")

    shm_name = _random_shm_name()
    cfg = _make_mla_config(shm_name, enable_prefix_reuse=False)
    worker = bg.MLAHostPagedKVWorkerView(cfg)

    try:
        worker.initialize(device_index=0, create_region=True)
        worker.register_sequences([101])

        prompt_tokens = list(range(64))
        pages, reused = worker.allocate_pages_for_sequences_with_prefix(
            [(101, 64)], prompt_tokens, [0, 64]
        )

        assert len(pages) == 1
        assert len(pages[0]) == 1
        assert reused == [0]

        worker.release_sequence_pages([101])
        stats = worker.get_stats()
        assert stats.num_used_pages == 0
    finally:
        worker.shutdown()
        del worker
        _shm_unlink(shm_name)


def test_prefix_reuse_hits_after_commit() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available")

    shm_name = _random_shm_name()
    cfg = _make_mla_config(shm_name, enable_prefix_reuse=True)
    worker = bg.MLAHostPagedKVWorkerView(cfg)

    try:
        worker.initialize(device_index=0, create_region=True)

        prompt_tokens = [i % 32000 for i in range(64)]

        # First sequence: allocate and offload layer 0 to trigger prefix commit.
        worker.register_sequences([201])
        pages_1, reused_1 = worker.allocate_pages_for_sequences_with_prefix(
            [(201, 64)], prompt_tokens, [0, 64]
        )
        assert reused_1 == [0]
        assert len(pages_1[0]) == 1

        k_tensor = torch.full(
            (1, 64, cfg.num_k_heads, cfg.k_head_dim),
            fill_value=1.0,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        task = worker.async_offload_layer_kv_to_host(
            layer_idx=0,
            sequence_ids=[201],
            k_tensor=k_tensor,
            v_tensor=None,
            sequence_lengths=[64],
        )
        task.wait()

        worker.release_sequence_pages([201])

        # Second sequence with same prompt should reuse full first page.
        worker.register_sequences([202])
        pages_2, reused_2 = worker.allocate_pages_for_sequences_with_prefix(
            [(202, 64)], prompt_tokens, [0, 64]
        )

        assert len(pages_2) == 1
        assert len(pages_2[0]) == 1
        assert reused_2 == [64]

        stats = worker.get_stats()
        assert stats.num_prefix_hits >= 1
        assert stats.num_shared_pages >= 1

        worker.release_sequence_pages([202])
    finally:
        worker.shutdown()
        del worker
        _shm_unlink(shm_name)
