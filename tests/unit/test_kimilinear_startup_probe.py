"""CUDA-free contracts for the Kimi-Linear startup decode probe."""

import gc
import sys
import types
import weakref
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
        # Optional hook a test installs to observe the class state the probe
        # publishes for the step (synthetic managers, batch metadata).
        self.probe_recorder = None

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)

    def __call__(self, input_ids=None, position_ids=None, **kwargs):
        assert input_ids is not None and input_ids.shape[1] == 1
        assert position_ids is not None and position_ids.shape == input_ids.shape
        if self.probe_recorder is not None:
            self.probe_recorder()
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


# Larger than the fake deployment page size (64), so the DSA arm's aliased
# page tables carry a genuine shared-history entry next to the per-row write
# entry, and smaller than any real index_topk so the CPU fakes stay tiny.
DSA_TOPK = 100
DSA_HEAD_DIM = 128
PAGE_SIZE = 64


def _dsa_indexer(topk=DSA_TOPK, head_dim=DSA_HEAD_DIM):
    return SimpleNamespace(index_topk=topk, index_head_dim=head_dim)


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


def test_probe_refuses_dsa_indexer_without_geometry(monkeypatch):
    manager, state_manager, _, _ = _build_probe_manager(
        monkeypatch, indexer=SimpleNamespace()
    )
    before = state_manager.snapshot()

    with pytest.raises(
        StartupDecodeProbeError, match="DSA indexer geometry"
    ):
        manager.startup_decode_probe_step(4)
    # Refused before any slot was touched.
    assert state_manager.snapshot() == before


def test_probe_refuses_disagreeing_dsa_index_head_dims(monkeypatch):
    manager, state_manager, _, _ = _build_probe_manager(
        monkeypatch, indexer=_dsa_indexer()
    )
    manager.model.model.layers.append(
        SimpleNamespace(
            self_attn=_fake_mla_wrapper(
                indexer=SimpleNamespace(
                    index_topk=DSA_TOPK, index_head_dim=DSA_HEAD_DIM // 2
                )
            )
        )
    )
    before = state_manager.snapshot()

    with pytest.raises(
        StartupDecodeProbeError, match="disagreeing DSA index_head_dim"
    ):
        manager.startup_decode_probe_step(4)
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


# --------------------------------------------------------------------------- #
#  Kimi-K3.1 DSA arm
# --------------------------------------------------------------------------- #


def _expected_aliased_table(bsz, past=DSA_TOPK, page_size=PAGE_SIZE):
    pages_per_row = (past + 1 + page_size - 1) // page_size
    table = torch.zeros(bsz, pages_per_row, dtype=torch.int32)
    table[:, past // page_size] = torch.arange(1, bsz + 1, dtype=torch.int32)
    return table


def test_probe_dsa_arm_installs_long_rows_and_synthetic_aux(monkeypatch):
    manager, _, _, _ = _build_probe_manager(monkeypatch, indexer=_dsa_indexer())
    seen = {}

    def record():
        primary = AttnWrapperBase.gpu_paged_kv_manager
        aux = AttnWrapperBase.gpu_paged_kv_manager_aux
        assert aux is not None
        aux_blocked, aux_v, aux_table = aux.get_layer_kv_with_page_table(1)
        _, _, primary_table = primary.get_layer_kv_with_page_table(1)
        seen.update(
            aux_shape=tuple(aux_blocked.shape),
            aux_dtype=aux_blocked.dtype,
            aux_v=aux_v,
            aux_table=aux_table.clone(),
            primary_table=primary_table.clone(),
            aux_page_size=aux.config.page_size_tokens,
            primary_page_size=primary.config.page_size_tokens,
            slot_map=dict(aux._gpu_page_table_manager.seq_id_to_slot),
            cur_batch=list(AttnWrapperBase.cur_batch),
            cache_seqlens=AttnWrapperBase.cache_seqlens.clone(),
            max_seqlen=AttnWrapperBase.max_seqlen,
            position_ids=AttnWrapperBase.position_ids.clone(),
            callbacks=(
                AttnWrapperBase.kv_append_callback,
                AttnWrapperBase.kv_append_callback_aux,
            ),
        )

    manager.model.probe_recorder = record
    manager.startup_decode_probe_step(4)

    # The page-split FP8 carrier: index_head_dim e4m3 bytes + 4 scale bytes
    # per token, one K head, plan_bsz + 1 physical pages (shared history
    # page 0 + one write page per row), at the deployment page size.
    assert seen["aux_shape"] == (5, PAGE_SIZE, 1, DSA_HEAD_DIM + 4)
    assert seen["aux_dtype"] is torch.uint8
    assert seen["aux_v"] is None
    assert seen["aux_page_size"] == PAGE_SIZE
    assert seen["primary_page_size"] == PAGE_SIZE

    # Every row is minimally LONG (index_topk + 1), so the sparse selection
    # path is the one measured, never the dense fallback.
    assert seen["cache_seqlens"].tolist() == [DSA_TOPK + 1] * 4
    assert seen["max_seqlen"] == DSA_TOPK + 1
    assert seen["position_ids"].flatten().tolist() == [DSA_TOPK] * 4

    expected = _expected_aliased_table(4)
    assert torch.equal(seen["aux_table"], expected)
    assert torch.equal(seen["primary_table"], expected)

    # The aux slot indirection resolves every synthetic sequence id, and no
    # Host-append callback is bound during the probe (the DSA wrapper then
    # appends nothing instead of raising).
    assert seen["slot_map"] == {
        sid: row for row, sid in enumerate(seen["cur_batch"])
    }
    assert seen["callbacks"] == (None, None)


def test_probe_dsa_synthetic_aux_write_read_round_trip(monkeypatch):
    """The aux surface addresses like the real manager: each row's write
    position resolves through seq_id_to_slot + the page table to its OWN
    physical page, and bytes written there read back intact."""
    manager, _, _, _ = _build_probe_manager(monkeypatch, indexer=_dsa_indexer())

    def record():
        aux = AttnWrapperBase.gpu_paged_kv_manager_aux
        page_size = aux.config.page_size_tokens
        blocked, _, table = aux.get_layer_kv_with_page_table(0)
        pos = DSA_TOPK
        slot_of = aux._gpu_page_table_manager.seq_id_to_slot
        pages = []
        for row, sid in enumerate(AttnWrapperBase.cur_batch):
            page = int(table[slot_of[sid], pos // page_size])
            pages.append(page)
            blocked[page, pos % page_size, 0, :DSA_HEAD_DIM] = row + 1
        assert len(set(pages)) == len(pages)  # write pages never alias
        for row, sid in enumerate(AttnWrapperBase.cur_batch):
            page = int(table[slot_of[sid], pos // page_size])
            got = blocked[page, pos % page_size, 0, :DSA_HEAD_DIM]
            assert torch.equal(
                got, torch.full((DSA_HEAD_DIM,), row + 1, dtype=torch.uint8)
            )
            # History entries alias the shared read-only zero page and were
            # not touched by any row's write.
            history_page = int(table[slot_of[sid], 0])
            assert history_page == 0
            assert int(blocked[history_page].to(torch.int64).sum()) == 0

    manager.model.probe_recorder = record
    manager.startup_decode_probe_step(4)


def test_two_dsa_probe_calls_free_aux_and_leave_pool_identical(monkeypatch):
    manager, state_manager, gate, layer_pool = _build_probe_manager(
        monkeypatch, indexer=_dsa_indexer()
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
            "kv_append_callback",
            "kv_append_callback_aux",
        )
    }
    aux_refs = []
    manager.model.probe_recorder = lambda: aux_refs.append(
        weakref.ref(AttnWrapperBase.gpu_paged_kv_manager_aux)
    )

    manager.startup_decode_probe_step(4)
    assert state_manager.snapshot() == initial
    manager.startup_decode_probe_step(4)
    assert state_manager.snapshot() == initial
    assert manager.model.calls == 2

    # Every synthetic aux surface is dropped with the step: nothing retains
    # it once the finally-restore has run.
    manager.model.probe_recorder = None
    gc.collect()
    assert len(aux_refs) == 2
    assert all(ref() is None for ref in aux_refs)

    assert "forward" not in gate.__dict__
    assert layer_pool.cur_batch_ids == []
    for name, value in saved_attention.items():
        assert getattr(AttnWrapperBase, name) is value
