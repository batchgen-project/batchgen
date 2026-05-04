import importlib
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_torch_stub(monkeypatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    batchgen_stub = types.ModuleType("batchgen")
    batchgen_stub.__path__ = [str(REPO_ROOT / "batchgen")]
    monkeypatch.setitem(sys.modules, "batchgen", batchgen_stub)


def _full_hit_module(monkeypatch):
    _install_torch_stub(monkeypatch)
    return importlib.import_module("batchgen.prefix_reuse.full_hit_runtime")


class _Wrapper:
    prepack_mode = False
    prepack_cu_seqlens = None
    prepack_max_seqlen = None
    prepack_num_sequences = None
    prepack_seq_lengths = None
    position_ids = None
    cur_batch = None
    prepack_prefix_reuse_mode = False
    prepack_prefix_shared_tokens = None
    prepack_full_seq_lengths = None
    prepack_full_hit_mode = False


def test_full_hit_attention_state_restores_wrapper_state(monkeypatch):
    mod = _full_hit_module(monkeypatch)
    cu_seqlens = object()
    position_ids = object()

    with mod.full_hit_attention_state(
        wrapper_classes=(_Wrapper,),
        cu_seqlens=cu_seqlens,
        position_ids=position_ids,
        global_sequence_ids=[1, 2],
        prompt_lengths=[64, 128],
    ):
        assert _Wrapper.prepack_mode is True
        assert _Wrapper.prepack_cu_seqlens is cu_seqlens
        assert _Wrapper.prepack_max_seqlen == 1
        assert _Wrapper.prepack_num_sequences == 2
        assert _Wrapper.position_ids is position_ids
        assert _Wrapper.cur_batch == [1, 2]
        assert _Wrapper.prepack_full_hit_mode is True

    assert _Wrapper.prepack_mode is False
    assert _Wrapper.prepack_cu_seqlens is None
    assert _Wrapper.prepack_max_seqlen is None
    assert _Wrapper.prepack_num_sequences is None
    assert _Wrapper.prepack_seq_lengths is None
    assert _Wrapper.prepack_prefix_reuse_mode is False
    assert _Wrapper.prepack_prefix_shared_tokens is None
    assert _Wrapper.prepack_full_seq_lengths is None
    assert _Wrapper.prepack_full_hit_mode is False
