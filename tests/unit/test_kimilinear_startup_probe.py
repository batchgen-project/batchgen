"""CUDA-free contracts for the Kimi-Linear startup decode probe."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
    KimiLinearParallelStrategyManager,
    StartupDecodeProbeError,
    StartupDecodeProbePlan,
    plan_startup_decode_probe,
    uniform_expert_assignment,
)
from batchgen.models.moonshotai.kimi_linear.wrappers import (
    KDASlotManager,
    KimiLinearAttnWrapper,
    KimiLinearExpertWrapper,
    KimiLinearKDAWrapper,
)
from batchgen.models.wrappers import AttnWrapperBase


# --------------------------------------------------------------------------- #
#  plan / assignment helpers
# --------------------------------------------------------------------------- #


def test_probe_shape_plan_uses_requested_width_and_one_past_token():
    plan = plan_startup_decode_probe(8, 16)
    assert plan == StartupDecodeProbePlan(batch_size=8, past_tokens=1)


def test_probe_shape_plan_refuses_width_above_configured_cap():
    with pytest.raises(StartupDecodeProbeError, match="exceeds configured decode cap"):
        plan_startup_decode_probe(17, 16)


def test_probe_shape_plan_refuses_non_integer_width_without_truncation():
    with pytest.raises(StartupDecodeProbeError, match="requires integer"):
        plan_startup_decode_probe(8.5, 16)


def test_uniform_expert_assignment_balances_topk_rows():
    assignment = uniform_expert_assignment(32, 8, 2)
    counts = torch.bincount(assignment.reshape(-1), minlength=8)
    assert assignment.shape == (32, 2)
    assert int(counts.max()) - int(counts.min()) <= 1
    assert torch.all((assignment >= 0) & (assignment < 8))


def test_uniform_expert_assignment_refuses_fractional_dimensions():
    with pytest.raises(StartupDecodeProbeError, match="requires integer dimensions"):
        uniform_expert_assignment(8.5, 8, 2)


# --------------------------------------------------------------------------- #
#  fakes
# --------------------------------------------------------------------------- #


class FakeKDAStateManager:
    """Free-list + seq->slot map with the probe-consumed manager surface."""

    def __init__(self, num_slots):
        self.num_slots = int(num_slots)
        self._free = list(range(self.num_slots))
        self._seq_map = {}

    # -- surface the probe and the KDASlotManager facade consume ---------- #
    def get_stats(self):
        return SimpleNamespace(
            num_total_state_items=self.num_slots,
            num_free_state_items=len(self._free),
            num_used_state_items=self.num_slots - len(self._free),
            num_active_sequences=len(self._seq_map),
        )

    def has_sequence_state_item(self, seq_id):
        return int(seq_id) in self._seq_map

    def allocate_state_item(self, seq_id):
        seq_id = int(seq_id)
        if seq_id in self._seq_map:
            return self._seq_map[seq_id]
        if not self._free:
            raise RuntimeError("Insufficient free KDA state items")
        slot = self._free.pop()
        self._seq_map[seq_id] = slot
        return slot

    def get_sequence_state_item(self, seq_id):
        return self._seq_map[int(seq_id)]

    def release_sequence_states(self, seq_ids):
        for seq_id in seq_ids:
            slot = self._seq_map.pop(int(seq_id), None)
            if slot is not None:
                self._free.append(slot)

    def snapshot(self):
        return (len(self._free), dict(self._seq_map))


class _FakeResidentBF16:
    num_tokens_per_rank = None

    @classmethod
    def set_num_tokens_per_rank(cls, num_tokens_per_rank):
        cls.num_tokens_per_rank = int(num_tokens_per_rank)


class _FakeResidentMXFP4:
    num_tokens_per_rank = None

    @classmethod
    def set_num_tokens_per_rank(cls, num_tokens_per_rank):
        cls.num_tokens_per_rank = int(num_tokens_per_rank)


class FakeDecodeModel:
    """Callable model double: checks the probe's step inputs, runs the patched
    gates, and returns HF-style outputs with .logits."""

    def __init__(self, layers, vocab=32, hidden=8):
        self.model = SimpleNamespace(layers=layers)
        self.vocab = vocab
        self.hidden = hidden
        self.calls = 0

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)

    def __call__(self, input_ids=None, position_ids=None, **kwargs):
        assert input_ids is not None and input_ids.shape[1] == 1
        assert position_ids is not None and position_ids.shape == input_ids.shape
        bsz = input_ids.shape[0]
        hidden = torch.zeros(bsz, 1, self.hidden)
        for layer in self.model.layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is None:
                continue
            topk_idx, topk_weight = moe.gate(hidden)
            assert topk_idx.shape == (bsz, moe.gate.top_k)
            assert topk_weight.shape == (bsz, moe.gate.top_k)
            assert int(topk_idx.max()) < moe.gate.num_experts
        self.calls += 1
        return SimpleNamespace(logits=torch.zeros(bsz, 1, self.vocab))


class FakeGate:
    """nn.Module-like gate: __call__ resolves ``self.forward`` dynamically, so
    the probe's instance-dict forward patch takes effect exactly as it does on
    the real KimiMoEGate."""

    def __init__(self, num_experts=8, top_k=2):
        self.num_experts = num_experts
        self.top_k = top_k

    def __call__(self, hidden_states):
        return self.forward(hidden_states)

    def forward(self, hidden_states):
        raise AssertionError(
            "the probe must route through its uniform gate patch"
        )


def _fake_mla_wrapper(indexer=None):
    wrapper = object.__new__(KimiLinearAttnWrapper)
    wrapper.module = SimpleNamespace(
        indexer=indexer, kv_lora_rank=16, qk_rope_head_dim=4
    )
    return wrapper


def _fake_kda_wrapper():
    wrapper = object.__new__(KimiLinearKDAWrapper)
    wrapper.module = SimpleNamespace()
    return wrapper


def _build_probe_manager(monkeypatch, *, num_slots=8, cap=4, indexer=None,
                         resident=True):
    """A manager + environment that lets the probe run end to end on CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(AttnWrapperBase, "phase", "decode", raising=False)
    monkeypatch.setattr(
        KimiLinearExpertWrapper, "phase", "decode", raising=False
    )

    bf16_mod = types.ModuleType("batchgen.moe.fused_moe_bf16_resident")
    bf16_mod.ResidentEPMoELayer = _FakeResidentBF16
    mxfp4_mod = types.ModuleType("batchgen.moe.fused_moe_mxfp4_resident")
    mxfp4_mod.ResidentEPMXFP4MoELayer = _FakeResidentMXFP4
    monkeypatch.setitem(
        sys.modules, "batchgen.moe.fused_moe_bf16_resident", bf16_mod
    )
    monkeypatch.setitem(
        sys.modules, "batchgen.moe.fused_moe_mxfp4_resident", mxfp4_mod
    )
    _FakeResidentBF16.num_tokens_per_rank = None
    _FakeResidentMXFP4.num_tokens_per_rank = None

    state_manager = FakeKDAStateManager(num_slots)
    monkeypatch.setattr(
        KimiLinearKDAWrapper, "state_manager", state_manager, raising=False
    )
    monkeypatch.setattr(
        KimiLinearKDAWrapper,
        "slot_manager",
        KDASlotManager(state_manager),
        raising=False,
    )
    layer_pool = SimpleNamespace(
        cur_batch_ids=[], has_initial_state=None, cur_decode_slots=None
    )
    monkeypatch.setattr(
        KimiLinearKDAWrapper, "layer_pools", {0: layer_pool}, raising=False
    )

    gate = FakeGate()
    moe_block = SimpleNamespace(
        gate=gate,
        _resident_ep_moe=object() if resident else None,
    )
    layers = [
        SimpleNamespace(self_attn=_fake_kda_wrapper()),
        SimpleNamespace(
            self_attn=_fake_mla_wrapper(indexer=indexer),
            block_sparse_moe=moe_block,
        ),
    ]
    model = FakeDecodeModel(layers)

    manager = object.__new__(KimiLinearParallelStrategyManager)
    manager.padding_bsz = cap
    manager.model = model
    manager.world_size = 1
    manager._comm = None
    manager._attn_tp_size = 1
    manager._decode_graph = None
    manager.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch=torch.device("cpu")),
        Device_Paged_KV_Config=SimpleNamespace(page_size=64),
    )
    return manager, state_manager, gate, layer_pool


# --------------------------------------------------------------------------- #
#  named refusals
# --------------------------------------------------------------------------- #


def test_probe_refuses_without_cuda(monkeypatch):
    manager = object.__new__(KimiLinearParallelStrategyManager)
    manager.padding_bsz = 8
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(StartupDecodeProbeError, match="requires CUDA"):
        manager.startup_decode_probe_step(8)


def test_probe_refuses_without_configured_cap(monkeypatch):
    manager = object.__new__(KimiLinearParallelStrategyManager)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(StartupDecodeProbeError, match="configure_decoding"):
        manager.startup_decode_probe_step(8)


def test_probe_refuses_width_above_cap(monkeypatch):
    manager, _, _, _ = _build_probe_manager(monkeypatch, cap=4)

    with pytest.raises(StartupDecodeProbeError, match="exceeds configured decode cap"):
        manager.startup_decode_probe_step(5)


def test_probe_refuses_wrong_phase(monkeypatch):
    manager, _, _, _ = _build_probe_manager(monkeypatch)
    monkeypatch.setattr(AttnWrapperBase, "phase", "prefill", raising=False)

    with pytest.raises(StartupDecodeProbeError, match="decode phase"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_dsa_variant_by_name(monkeypatch):
    manager, state_manager, _, _ = _build_probe_manager(
        monkeypatch, indexer=object()
    )
    before = state_manager.snapshot()

    with pytest.raises(
        StartupDecodeProbeError, match="DSA decode probe not yet supported"
    ):
        manager.startup_decode_probe_step(4)
    # Refused before any slot was touched.
    assert state_manager.snapshot() == before


def test_probe_refuses_streamed_decode_moe(monkeypatch):
    manager, _, _, _ = _build_probe_manager(monkeypatch, resident=False)

    with pytest.raises(StartupDecodeProbeError, match="resident-EP decode MoE"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_insufficient_free_slots(monkeypatch):
    manager, state_manager, _, _ = _build_probe_manager(
        monkeypatch, num_slots=2, cap=4
    )
    before = state_manager.snapshot()

    with pytest.raises(StartupDecodeProbeError, match="free KDA state slots"):
        manager.startup_decode_probe_step(4)
    assert state_manager.snapshot() == before


# --------------------------------------------------------------------------- #
#  slot accounting + full state restore
# --------------------------------------------------------------------------- #


def test_two_consecutive_probe_calls_leave_pool_identical(monkeypatch):
    manager, state_manager, gate, layer_pool = _build_probe_manager(
        monkeypatch, num_slots=8, cap=4
    )
    initial = state_manager.snapshot()
    saved_attention = {
        name: getattr(AttnWrapperBase, name)
        for name in (
            "gpu_paged_kv_manager",
            "gpu_paged_kv_manager_aux",
            "cache_seqlens",
            "max_seqlen",
            "cur_batch",
            "position_ids",
            "attention_mask",
            "kv_append_callback",
            "kv_append_callback_aux",
            "batchgen_debug",
        )
    }

    manager.startup_decode_probe_step(4)
    first = state_manager.snapshot()
    manager.startup_decode_probe_step(4)
    second = state_manager.snapshot()

    # Free count AND seq map end exactly as they began, both times (the
    # measure_eager contract runs the step warm + twice measured).
    assert first == initial
    assert second == initial
    assert manager.model.calls == 2

    # Gate patch removed, resident layout scalar restored.
    assert "forward" not in gate.__dict__
    assert _FakeResidentBF16.num_tokens_per_rank is None
    assert _FakeResidentMXFP4.num_tokens_per_rank is None

    # Per-step KDA pool context and every AttnWrapperBase attr restored.
    assert layer_pool.cur_batch_ids == []
    assert layer_pool.has_initial_state is None
    assert layer_pool.cur_decode_slots is None
    for name, value in saved_attention.items():
        assert getattr(AttnWrapperBase, name) is value


def test_probe_releases_slots_and_restores_state_on_step_failure(monkeypatch):
    manager, state_manager, gate, _ = _build_probe_manager(
        monkeypatch, num_slots=8, cap=4
    )
    initial = state_manager.snapshot()

    def _boom(**kwargs):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(FakeDecodeModel, "__call__", lambda self, **kw: _boom())

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        manager.startup_decode_probe_step(4)

    assert state_manager.snapshot() == initial
    assert "forward" not in gate.__dict__
    assert AttnWrapperBase.gpu_paged_kv_manager is None
    assert AttnWrapperBase.cur_batch is None
