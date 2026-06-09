from __future__ import annotations

from types import SimpleNamespace

import pytest

from batchgen.batchgen_worker import BatchGenWorker


class _FakeV4Coordinator:
    constructed = None

    @classmethod
    def bytes_per_page_unit_for(
        cls, *, compress_ratios, base_page_size=256, swa_page_size=128
    ):
        return 1024

    def __init__(
        self,
        *,
        compress_ratios,
        num_pages,
        device,
        base_page_size=256,
        swa_page_size=128,
    ):
        self.compress_ratios = list(compress_ratios)
        self.num_pages = num_pages
        self.device = device
        self.is_initialized = False
        _FakeV4Coordinator.constructed = self

    def initialize(self):
        self.is_initialized = True


def _make_worker(ckpt_name, *, compress_ratios=None, num_hidden_layers=None):
    worker = object.__new__(BatchGenWorker)
    worker.huggingface_ckpt_name = ckpt_name
    worker.gpu_kv_cache_size_gb = 1.0
    worker.local_rank = 0
    worker.rank = 0
    worker.core_engine = SimpleNamespace(
        gpu_paged_kv_manager=None, gpu_paged_kv_manager_aux=None
    )
    worker.model_config = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        compress_ratios=compress_ratios,
    )
    worker.loaded_model_config = None
    return worker


def test_v4_branch_builds_and_binds_coordinator(monkeypatch):
    import batchgen.kv_cache.deepseek_v4_kv_coordinator as v4_module

    monkeypatch.setattr(
        v4_module, "DeepSeekV4KVCoordinator", _FakeV4Coordinator
    )
    _FakeV4Coordinator.constructed = None

    worker = _make_worker(
        "deepseek-v4-flash",
        compress_ratios=[0, 0, 4, 128] * 11,
        num_hidden_layers=43,
    )
    manager = worker._initialize_gpu_kv_manager_fixed_size()

    assert manager is _FakeV4Coordinator.constructed
    assert manager.is_initialized
    assert len(manager.compress_ratios) == 43
    assert manager.num_pages == (1 * 1024**3) // 1024
    assert worker.gpu_paged_kv_cache_manager is manager
    assert worker.core_engine.gpu_paged_kv_manager is manager


def test_v4_compress_ratios_normalized_to_num_layers():
    worker = _make_worker(
        "deepseek-v4-flash",
        compress_ratios=[0, 4, 128, 4, 0],
        num_hidden_layers=3,
    )
    assert worker._get_deepseek_v4_compress_ratios() == [0, 4, 128]


def test_v4_compress_ratios_rejects_invalid_value():
    worker = _make_worker(
        "deepseek-v4-flash",
        compress_ratios=[0, 7],
        num_hidden_layers=2,
    )
    with pytest.raises(ValueError):
        worker._get_deepseek_v4_compress_ratios()


def test_dsa_and_plain_models_do_not_take_v4_branch(monkeypatch):
    from batchgen.kv_cache.host_kv_mananger_config import (
        is_dsa_model,
        is_v4_model,
    )

    assert is_v4_model("deepseek-v4-flash") is True
    assert is_v4_model("zai-org/GLM-5-FP8") is False
    assert is_dsa_model("deepseek-v4-flash") is False
