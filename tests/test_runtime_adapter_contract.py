"""M2 Phase A: contract test for the per-model runtime adapter.

GPU-free. Mirrors tests/cuda_graph_contract/test_adapter_contract.py.
"""
import pytest
import torch

from batchgen.contracts.runtime_adapter import (
    ModelRuntimeAdapter,
    RuntimePhase,
    RuntimeState,
)


def test_abstract_method_set():
    # Only past_kv_byte_size is mandatory; the rest have model-agnostic defaults.
    assert ModelRuntimeAdapter.__abstractmethods__ == frozenset({"past_kv_byte_size"})


def test_runtime_state_is_frozen():
    s = RuntimeState(
        phase=RuntimePhase.DECODE, attention_mask=None, max_input_length=8, token_idx=0
    )
    with pytest.raises(Exception):
        s.token_idx = 1  # frozen dataclass -> FrozenInstanceError


def test_phase_values():
    assert RuntimePhase.PREFILL == "prefill"
    assert RuntimePhase.DECODE == "decode"


class _DummyAdapter(ModelRuntimeAdapter):
    def past_kv_byte_size(self, state: RuntimeState) -> int:
        return (state.max_input_length + state.token_idx) * 4


def test_default_position_ids_shapes():
    mask = torch.ones(2, 5, dtype=torch.long)
    a = _DummyAdapter(model_config=None)
    pre = a.compute_position_ids(
        RuntimeState(RuntimePhase.PREFILL, mask, max_input_length=5, token_idx=0)
    )
    dec = a.compute_position_ids(
        RuntimeState(RuntimePhase.DECODE, mask, max_input_length=5, token_idx=0)
    )
    assert tuple(pre.shape) == (2, 5)   # prefill: full
    assert tuple(dec.shape) == (2, 1)   # decode: last token only


def test_default_attention_backend_is_noop():
    a = _DummyAdapter(model_config=None)
    a.configure_attention_backend(object(), phase=RuntimePhase.DECODE)  # must not raise


def test_past_kv_byte_size_uses_state():
    a = _DummyAdapter(model_config=None)
    n = a.past_kv_byte_size(
        RuntimeState(RuntimePhase.DECODE, None, max_input_length=10, token_idx=3)
    )
    assert n == (10 + 3) * 4
