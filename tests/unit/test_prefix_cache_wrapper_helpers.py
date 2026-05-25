import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeCuSeqlens:
    def __init__(self, values):
        self._values = list(values)

    def __len__(self):
        return len(self._values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class _FakeSeqTensor:
    def __init__(self, name, dim=3):
        self.name = name
        self._dim = dim

    def unsqueeze(self, dim):
        return _FakeSeqTensor(f"{self.name}.unsqueeze({dim})", self._dim + 1)

    def dim(self):
        return self._dim


class _FakeFlatTensor:
    def __init__(self, name, dim=3):
        self.name = name
        self._dim = dim

    def __getitem__(self, key):
        return _FakeSeqTensor(f"{self.name}[{key.start}:{key.stop}]", self._dim)


class _FakeWorkerView:
    def __init__(self):
        self.calls = []

    def async_offload_layer_kv_to_host(self, **kwargs):
        self.calls.append(("normal", kwargs))
        return SimpleNamespace(done=lambda: True, wait=lambda: None)

    def async_offload_layer_kv_to_host_with_offsets(self, **kwargs):
        self.calls.append(("offset", kwargs))
        return SimpleNamespace(done=lambda: True, wait=lambda: None)


class _NoOffsetWorkerView:
    def async_offload_layer_kv_to_host(self, **kwargs):
        del kwargs
        return None


def _install_torch_stub(monkeypatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float16 = "float16"
    torch_stub.int32 = "int32"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    batchgen_stub = types.ModuleType("batchgen")
    batchgen_stub.__path__ = [str(REPO_ROOT / "batchgen")]
    monkeypatch.setitem(sys.modules, "batchgen", batchgen_stub)
    models_stub = types.ModuleType("batchgen.models")
    models_stub.__path__ = [str(REPO_ROOT / "batchgen" / "models")]
    monkeypatch.setitem(sys.modules, "batchgen.models", models_stub)
    wrappers_stub = types.ModuleType("batchgen.models.wrappers")
    wrappers_stub.__path__ = [
        str(REPO_ROOT / "batchgen" / "models" / "wrappers")
    ]
    monkeypatch.setitem(sys.modules, "batchgen.models.wrappers", wrappers_stub)


def _prefix_cache_module(monkeypatch):
    _install_torch_stub(monkeypatch)
    return importlib.import_module("batchgen.models.wrappers.prefix_cache")


class _Wrapper:
    prepack_cu_seqlens = _FakeCuSeqlens([0, 2, 5])
    prepack_max_seqlen = 3
    prepack_num_sequences = 2
    prepack_seq_lengths = [2, 3]
    cur_batch = [10, 20]
    prepack_prefix_reuse_mode = True
    prepack_full_hit_mode = False
    prepack_prefix_shared_tokens = [7, 11]
    prepack_full_seq_lengths = [9, 14]


def test_prefix_cache_metadata_validates_prefix_lengths(monkeypatch):
    mod = _prefix_cache_module(monkeypatch)

    metadata = mod.PrefixCachePrepackMetadata.from_wrapper_cls(_Wrapper)

    assert metadata.global_sequence_ids == [10, 20]
    assert metadata.prefix_shared_tokens == [7, 11]


def test_prefix_offloader_uses_destination_offsets(monkeypatch):
    mod = _prefix_cache_module(monkeypatch)
    metadata = mod.PrefixCachePrepackMetadata.from_wrapper_cls(_Wrapper)
    worker_view = _FakeWorkerView()
    tracked = []
    offloader = mod.PrefixAwarePrefillOffloader(
        worker_view=worker_view,
        layer_idx=3,
        metadata=metadata,
        track_task=lambda task, layer_idx: tracked.append((task, layer_idx)),
    )

    offloader.offload_gqa(
        key=_FakeFlatTensor("k"),
        value=_FakeFlatTensor("v"),
    )

    assert [kind for kind, _ in worker_view.calls] == ["offset", "offset"]
    assert worker_view.calls[0][1]["destination_token_starts"] == [7]
    assert worker_view.calls[1][1]["destination_token_starts"] == [11]
    assert [layer_idx for _, layer_idx in tracked] == [3, 3]


def test_prefix_offloader_rejects_missing_offset_api(monkeypatch):
    mod = _prefix_cache_module(monkeypatch)
    metadata = mod.PrefixCachePrepackMetadata.from_wrapper_cls(_Wrapper)
    offloader = mod.PrefixAwarePrefillOffloader(
        worker_view=_NoOffsetWorkerView(),
        layer_idx=0,
        metadata=metadata,
    )

    with pytest.raises(RuntimeError, match="with_offsets"):
        offloader.offload_mla(key=_FakeFlatTensor("kv", dim=2))
