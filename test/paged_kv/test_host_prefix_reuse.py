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
    shm_name: str, enable_prefix_reuse: bool, page_size_tokens: int = 64
) -> bg.HostPagedKVConfig:  # type: ignore
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = 1
    cfg.num_pages = 512
    cfg.page_size_tokens = page_size_tokens
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


def test_prefix_reuse_includes_decode_tokens() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available")

    shm_name = _random_shm_name()
    cfg = _make_mla_config(
        shm_name, enable_prefix_reuse=True, page_size_tokens=8
    )
    worker = bg.MLAHostPagedKVWorkerView(cfg)

    try:
        worker.initialize(device_index=0, create_region=True)

        prompt_tokens = [100 + i for i in range(cfg.page_size_tokens)]
        decode_tokens = [200 + i for i in range(cfg.page_size_tokens)]
        full_tokens = prompt_tokens + decode_tokens

        worker.register_sequences([301])
        pages_1, reused_1 = worker.allocate_pages_for_sequences_with_prefix(
            [(301, len(full_tokens))], prompt_tokens, [0, len(prompt_tokens)]
        )
        assert reused_1 == [0]
        assert len(pages_1[0]) == 2

        prefill_k = torch.full(
            (1, len(prompt_tokens), cfg.num_k_heads, cfg.k_head_dim),
            fill_value=1.0,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        worker.async_offload_layer_kv_to_host(
            layer_idx=0,
            sequence_ids=[301],
            k_tensor=prefill_k,
            v_tensor=None,
            sequence_lengths=[len(prompt_tokens)],
        ).wait()

        for step, token_id in enumerate(decode_tokens):
            decode_k = torch.full(
                (1, 1, cfg.num_k_heads, cfg.k_head_dim),
                fill_value=2.0 + step,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            worker.async_append_decode_kv_to_host(
                layer_idx=0,
                sequence_ids=[301],
                k_tensor=decode_k,
                v_tensor=None,
                sequence_lengths=[len(prompt_tokens) + step],
                decode_token_ids=[token_id],
            ).wait()

        worker.release_sequence_pages([301])

        worker.register_sequences([302])
        pages_2, reused_2 = worker.allocate_pages_for_sequences_with_prefix(
            [(302, len(full_tokens))], full_tokens, [0, len(full_tokens)]
        )
        assert len(pages_2[0]) == 2
        assert reused_2 == [len(full_tokens)]

        stats = worker.get_stats()
        assert stats.num_prefix_hits >= 1
        assert stats.num_shared_pages >= 1

        worker.release_sequence_pages([302])
    finally:
        worker.shutdown()
        del worker
        _shm_unlink(shm_name)


def test_prefix_reuse_waits_for_full_decode_page_before_commit() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available")

    shm_name = _random_shm_name()
    cfg = _make_mla_config(
        shm_name, enable_prefix_reuse=True, page_size_tokens=8
    )
    worker = bg.MLAHostPagedKVWorkerView(cfg)

    try:
        worker.initialize(device_index=0, create_region=True)

        prompt_tokens = [500 + i for i in range(cfg.page_size_tokens)]
        decode_tokens = [600 + i for i in range(cfg.page_size_tokens - 1)]
        full_tokens = prompt_tokens + decode_tokens

        worker.register_sequences([351])
        pages_1, reused_1 = worker.allocate_pages_for_sequences_with_prefix(
            [(351, len(full_tokens))], prompt_tokens, [0, len(prompt_tokens)]
        )
        assert reused_1 == [0]
        assert len(pages_1[0]) == 2

        prefill_k = torch.full(
            (1, len(prompt_tokens), cfg.num_k_heads, cfg.k_head_dim),
            fill_value=1.0,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        worker.async_offload_layer_kv_to_host(
            layer_idx=0,
            sequence_ids=[351],
            k_tensor=prefill_k,
            v_tensor=None,
            sequence_lengths=[len(prompt_tokens)],
        ).wait()

        for step, token_id in enumerate(decode_tokens):
            decode_k = torch.full(
                (1, 1, cfg.num_k_heads, cfg.k_head_dim),
                fill_value=2.0 + step,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            worker.async_append_decode_kv_to_host(
                layer_idx=0,
                sequence_ids=[351],
                k_tensor=decode_k,
                v_tensor=None,
                sequence_lengths=[len(prompt_tokens) + step],
                decode_token_ids=[token_id],
            ).wait()

        worker.release_sequence_pages([351])

        worker.register_sequences([352])
        pages_2, reused_2 = worker.allocate_pages_for_sequences_with_prefix(
            [(352, len(full_tokens))], full_tokens, [0, len(full_tokens)]
        )

        assert len(pages_2[0]) == 2
        assert reused_2 == [len(prompt_tokens)]

        stats = worker.get_stats()
        assert stats.num_prefix_hits >= 1
        assert stats.num_shared_pages >= 1

        worker.release_sequence_pages([352])
    finally:
        worker.shutdown()
        del worker
        _shm_unlink(shm_name)


def test_prefix_reuse_skips_decode_extension_when_token_ids_missing() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available")

    shm_name = _random_shm_name()
    cfg = _make_mla_config(
        shm_name, enable_prefix_reuse=True, page_size_tokens=8
    )
    worker = bg.MLAHostPagedKVWorkerView(cfg)

    try:
        worker.initialize(device_index=0, create_region=True)

        prompt_tokens = [700 + i for i in range(cfg.page_size_tokens)]
        decode_tokens = [800 + i for i in range(cfg.page_size_tokens)]
        full_tokens = prompt_tokens + decode_tokens

        worker.register_sequences([361])
        pages_1, reused_1 = worker.allocate_pages_for_sequences_with_prefix(
            [(361, len(full_tokens))], prompt_tokens, [0, len(prompt_tokens)]
        )
        assert reused_1 == [0]
        assert len(pages_1[0]) == 2

        prefill_k = torch.full(
            (1, len(prompt_tokens), cfg.num_k_heads, cfg.k_head_dim),
            fill_value=1.0,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        worker.async_offload_layer_kv_to_host(
            layer_idx=0,
            sequence_ids=[361],
            k_tensor=prefill_k,
            v_tensor=None,
            sequence_lengths=[len(prompt_tokens)],
        ).wait()

        for step in range(len(decode_tokens)):
            decode_k = torch.full(
                (1, 1, cfg.num_k_heads, cfg.k_head_dim),
                fill_value=3.0 + step,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            worker.async_append_decode_kv_to_host(
                layer_idx=0,
                sequence_ids=[361],
                k_tensor=decode_k,
                v_tensor=None,
                sequence_lengths=[len(prompt_tokens) + step],
            ).wait()

        worker.release_sequence_pages([361])

        worker.register_sequences([362])
        pages_2, reused_2 = worker.allocate_pages_for_sequences_with_prefix(
            [(362, len(full_tokens))], full_tokens, [0, len(full_tokens)]
        )

        assert len(pages_2[0]) == 2
        assert reused_2 == [len(prompt_tokens)]

        stats = worker.get_stats()
        assert stats.num_prefix_hits >= 1
        assert stats.num_shared_pages >= 1

        worker.release_sequence_pages([362])
    finally:
        worker.shutdown()
        del worker
        _shm_unlink(shm_name)


def test_prefix_batch_allocation_failure_rolls_back() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available")

    shm_name = _random_shm_name()
    cfg = _make_mla_config(shm_name, enable_prefix_reuse=True)
    cfg.num_pages = 2
    cfg.sequence_page_node_capacity = 16
    worker = bg.MLAHostPagedKVWorkerView(cfg)

    try:
        worker.initialize(device_index=0, create_region=True)
        worker.register_sequences([401, 402])

        flat_prompt_tokens = list(range(128))
        raised = False
        try:
            worker.allocate_pages_for_sequences_with_prefix(
                [(401, 64), (402, 128)],
                flat_prompt_tokens,
                [0, 64, 128],
            )
        except RuntimeError:
            raised = True
        assert raised

        stats = worker.get_stats()
        assert stats.num_used_pages == 0
        assert stats.num_active_sequences == 0
    finally:
        worker.shutdown()
        del worker
        _shm_unlink(shm_name)
