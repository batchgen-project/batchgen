import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class _FakeInputIds:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, key):
        row, token_slice = key
        assert row == 0
        return _FakeTensor(self._values[token_slice])


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


def _install_torch_stub(monkeypatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.device = lambda value: value
    torch_stub.cuda = _FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    batchgen_stub = types.ModuleType("batchgen")
    batchgen_stub.__path__ = [str(REPO_ROOT / "batchgen")]
    monkeypatch.setitem(sys.modules, "batchgen", batchgen_stub)
    prefill_stub = types.ModuleType("batchgen.prefill")
    prefill_stub.__path__ = [str(REPO_ROOT / "batchgen" / "prefill")]
    monkeypatch.setitem(sys.modules, "batchgen.prefill", prefill_stub)


def _runtime_module(monkeypatch):
    _install_torch_stub(monkeypatch)
    return importlib.import_module("batchgen.prefix_reuse.runtime_state")


def _seq(global_idx, tokens, assigned_rank=None):
    return SimpleNamespace(
        uuid=f"seq-{global_idx}",
        global_idx=global_idx,
        input_ids=_FakeInputIds(tokens),
        prompt_length=len(tokens),
        assigned_rank=assigned_rank,
        prefix_shared_tokens=0,
    )


def test_namespace_hash_is_stable_and_config_specific(monkeypatch):
    mod = _runtime_module(monkeypatch)

    first = mod.PrefixReuseRuntime.build_namespace_hash(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bf16",
        page_size=64,
    )
    second = mod.PrefixReuseRuntime.build_namespace_hash(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bf16",
        page_size=64,
    )
    different_page = mod.PrefixReuseRuntime.build_namespace_hash(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bf16",
        page_size=128,
    )

    assert first == second
    assert first != different_page


def test_prompt_rank_key_uses_full_pages_and_caches(monkeypatch):
    mod = _runtime_module(monkeypatch)
    runtime = mod.PrefixReuseRuntime(
        enabled=True,
        model_name="openai/gpt-oss-120b",
        kv_dtype="bf16",
        page_size=64,
        rank=0,
        world_size=2,
        torch_device="cuda:0",
    )
    seq = _seq(7, range(130))

    key = runtime.prompt_rank_key(seq)
    assert key == runtime.prompt_rank_key(seq)

    cached = runtime.state.prompt_rank_key_cache[7]
    assert cached[0] == 128
    assert cached[1] == 64


def test_cached_rank_can_be_discovered_from_existing_sequence(monkeypatch):
    mod = _runtime_module(monkeypatch)
    runtime = mod.PrefixReuseRuntime(
        enabled=True,
        model_name="openai/gpt-oss-120b",
        kv_dtype="bf16",
        page_size=64,
        rank=0,
        world_size=4,
        torch_device="cuda:0",
    )
    existing = _seq(1, range(128), assigned_rank=3)
    incoming = _seq(2, range(128))

    rank = runtime.cached_rank_for_sequence(
        incoming,
        existing_sequences=[existing, incoming],
        pending_uuids={incoming.uuid},
    )

    assert rank == 3
    key = runtime.prompt_rank_key(incoming)
    assert runtime.state.prompt_rank_cache[key] == 3


def test_shared_tokens_prefers_sequence_then_allocation(monkeypatch):
    mod = _runtime_module(monkeypatch)
    runtime = mod.PrefixReuseRuntime(
        enabled=True,
        model_name="openai/gpt-oss-120b",
        kv_dtype="bf16",
        page_size=64,
        rank=0,
        world_size=1,
        torch_device="cuda:0",
    )
    seq = _seq(11, range(128))

    runtime.state.allocations_by_global_id[11] = {"shared_prefix_tokens": 64}
    assert runtime.shared_tokens_for_sequence(seq, worker_view=None) == 64

    seq.prefix_shared_tokens = 128
    assert runtime.shared_tokens_for_sequence(seq, worker_view=None) == 128
