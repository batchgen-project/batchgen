import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_gpu_manager_module():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "batchgen" / "config" / "config.py"
    config_spec = importlib.util.spec_from_file_location(
        "_batchgen_config_config_for_gpu_suffix_append_test",
        config_path,
    )
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules[config_spec.name] = config_module
    config_spec.loader.exec_module(config_module)

    previous_config_pkg = sys.modules.get("batchgen.config")
    previous_config_module = sys.modules.get("batchgen.config.config")
    previous_gpu_kv_kernels = sys.modules.get("batchgen.kv_cache.gpu_kv_kernels")
    config_pkg = types.ModuleType("batchgen.config")
    config_pkg.__path__ = [str(repo_root / "batchgen" / "config")]
    config_pkg.config = config_module
    sys.modules["batchgen.config"] = config_pkg
    sys.modules["batchgen.config.config"] = config_module

    gpu_kv_kernels = types.ModuleType("batchgen.kv_cache.gpu_kv_kernels")

    def _unused_gpu_kernel(*args, **kwargs):
        raise RuntimeError("GPU KV kernels are not used by this suffix append test")

    gpu_kv_kernels.run_paged_kv_token_update = _unused_gpu_kernel
    gpu_kv_kernels.run_paged_kv_token_update_fused = _unused_gpu_kernel
    sys.modules["batchgen.kv_cache.gpu_kv_kernels"] = gpu_kv_kernels

    try:
        manager_path = repo_root / "batchgen" / "kv_cache" / "gpu_paged_kv_manager.py"
        manager_spec = importlib.util.spec_from_file_location(
            "_batchgen_gpu_paged_kv_manager_for_suffix_append_test",
            manager_path,
        )
        manager_module = importlib.util.module_from_spec(manager_spec)
        sys.modules[manager_spec.name] = manager_module
        manager_spec.loader.exec_module(manager_module)
        return manager_module
    finally:
        if previous_config_pkg is None:
            sys.modules.pop("batchgen.config", None)
        else:
            sys.modules["batchgen.config"] = previous_config_pkg
        if previous_config_module is None:
            sys.modules.pop("batchgen.config.config", None)
        else:
            sys.modules["batchgen.config.config"] = previous_config_module
        if previous_gpu_kv_kernels is None:
            sys.modules.pop("batchgen.kv_cache.gpu_kv_kernels", None)
        else:
            sys.modules["batchgen.kv_cache.gpu_kv_kernels"] = previous_gpu_kv_kernels


_gpu_manager_module = _load_gpu_manager_module()
GPUPagedKVCacheManager = _gpu_manager_module.GPUPagedKVCacheManager
GPUPagedKVConfig = _gpu_manager_module.GPUPagedKVConfig


def _make_config(
    *,
    has_v: bool = True,
) -> GPUPagedKVConfig:
    return GPUPagedKVConfig(
        num_layers=2,
        num_pages=16,
        page_size_tokens=4,
        num_k_heads=1,
        k_head_dim=2,
        num_v_heads=1 if has_v else 0,
        v_head_dim=2 if has_v else 0,
        kv_dtype=torch.float32,
    )


def _make_manager(*, has_v: bool = True) -> GPUPagedKVCacheManager:
    manager = GPUPagedKVCacheManager(config=_make_config(has_v=has_v), device="cpu")
    manager.initialize()
    return manager


def _read_sequence_k(manager: GPUPagedKVCacheManager, sequence_id: int, length: int):
    k_cache, _ = manager.get_kv_tensors()
    pages = manager._sequences[sequence_id].pages.tolist()
    chunks = []
    remaining = length
    for page in pages:
        if remaining <= 0:
            break
        take = min(remaining, manager.config.page_size_tokens)
        chunks.append(k_cache[0, page, :take].clone())
        remaining -= take
    return torch.cat(chunks, dim=0)


def _read_sequence_v(manager: GPUPagedKVCacheManager, sequence_id: int, length: int):
    _, v_cache = manager.get_kv_tensors()
    pages = manager._sequences[sequence_id].pages.tolist()
    chunks = []
    remaining = length
    for page in pages:
        if remaining <= 0:
            break
        take = min(remaining, manager.config.page_size_tokens)
        chunks.append(v_cache[0, page, :take].clone())
        remaining -= take
    return torch.cat(chunks, dim=0)


def test_prepare_prefill_suffix_append_auto_allocates_miss_sequence():
    manager = _make_manager()

    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[101],
        prefix_lens=[0],
        suffix_lens=[3],
    )

    assert plan.sequence_ids == [101]
    assert plan.prefix_lens.tolist() == [0]
    assert plan.suffix_lens.tolist() == [3]
    assert plan.cache_seqlens.tolist() == [3]
    assert plan.token_starts.tolist() == [0]
    assert plan.slot_indices.tolist() == [0]
    assert plan.page_table.shape[0] == 1
    assert 101 in manager._sequences


def test_prepare_prefill_suffix_append_requires_allocated_reused_prefix():
    manager = _make_manager()

    with pytest.raises(KeyError, match="prefix-reused sequence"):
        manager.prepare_prefill_suffix_append(
            sequence_ids=[101],
            prefix_lens=[2],
            suffix_lens=[3],
        )


def test_append_layer_prefill_suffix_tokens_writes_across_page_boundary():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([101], [7])
    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[101],
        prefix_lens=[3],
        suffix_lens=[4],
    )
    suffix_k = torch.tensor(
        [[[1.0, 1.5]], [[2.0, 2.5]], [[3.0, 3.5]], [[4.0, 4.5]]],
        dtype=torch.float32,
    )
    suffix_v = suffix_k + 10

    manager.append_layer_prefill_suffix_tokens(
        k_tensor=suffix_k,
        v_tensor=suffix_v,
        append_plan=plan,
        layer_idx=0,
    )

    full_k = _read_sequence_k(manager, 101, 7)
    full_v = _read_sequence_v(manager, 101, 7)
    torch.testing.assert_close(full_k[:3], torch.zeros_like(full_k[:3]))
    torch.testing.assert_close(full_v[:3], torch.zeros_like(full_v[:3]))
    torch.testing.assert_close(full_k[3:7], suffix_k)
    torch.testing.assert_close(full_v[3:7], suffix_v)


def test_append_layer_prefill_suffix_tokens_handles_mixed_batch():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([101], [5])
    manager.allocate_pages_for_sequences([103], [4])
    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[101, 102, 103],
        prefix_lens=[3, 0, 4],
        suffix_lens=[2, 3, 0],
    )
    suffix_k = torch.arange(10, dtype=torch.float32).view(5, 1, 2)
    suffix_v = suffix_k + 100

    manager.append_layer_prefill_suffix_tokens(
        k_tensor=suffix_k,
        v_tensor=suffix_v,
        append_plan=plan,
        layer_idx=0,
    )

    torch.testing.assert_close(_read_sequence_k(manager, 101, 5)[3:5], suffix_k[:2])
    torch.testing.assert_close(_read_sequence_v(manager, 101, 5)[3:5], suffix_v[:2])
    torch.testing.assert_close(_read_sequence_k(manager, 102, 3), suffix_k[2:5])
    torch.testing.assert_close(_read_sequence_v(manager, 102, 3), suffix_v[2:5])
    torch.testing.assert_close(_read_sequence_k(manager, 103, 4), torch.zeros(4, 1, 2))


def test_append_layer_prefill_suffix_tokens_accepts_mla_2d_k_tensor():
    manager = _make_manager(has_v=False)
    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[101],
        prefix_lens=[0],
        suffix_lens=[2],
    )
    suffix_k = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    manager.append_layer_prefill_suffix_tokens(
        k_tensor=suffix_k,
        v_tensor=None,
        append_plan=plan,
        layer_idx=0,
    )

    torch.testing.assert_close(
        _read_sequence_k(manager, 101, 2),
        suffix_k.unsqueeze(1),
    )


def test_append_layer_prefill_suffix_tokens_rejects_bad_token_count():
    manager = _make_manager()
    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[101],
        prefix_lens=[0],
        suffix_lens=[2],
    )

    with pytest.raises(ValueError, match="token count mismatch"):
        manager.append_layer_prefill_suffix_tokens(
            k_tensor=torch.zeros(3, 1, 2),
            v_tensor=None,
            append_plan=plan,
            layer_idx=0,
        )
