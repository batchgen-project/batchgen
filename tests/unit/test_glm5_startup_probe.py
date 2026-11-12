"""CUDA-free contracts for the GLM-5 startup decode probe."""

import gc
import importlib.machinery
import importlib.util
import sys
import types
import weakref
from types import SimpleNamespace

import pytest
import torch

# ---------------------------------------------------------------------------- #
#  triton stub: glm5/model.py defines its Triton kernels at import time, so a
#  CPU environment without triton needs a decorator-only stand-in. The kernel
#  bodies never run here (no fake is registered under a real triton install).
# ---------------------------------------------------------------------------- #

if importlib.util.find_spec("triton") is None:  # pragma: no branch

    def _stub_module(name):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__version__ = "0.0.0-cpu-stub"
        return mod

    _triton = _stub_module("triton")

    def _jit(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    _triton.jit = _jit
    _triton.cdiv = lambda a, b: (a + b - 1) // b
    _triton.autotune = lambda *a, **k: (lambda f: f)
    _triton.heuristics = lambda *a, **k: (lambda f: f)
    _triton.Config = lambda *a, **k: None

    class _TlName:
        def __getitem__(self, item):
            return item

        def __call__(self, *args, **kwargs):
            return None

    _tl = _stub_module("triton.language")

    def _tl_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _TlName()

    _tl.__getattr__ = _tl_getattr
    _triton.language = _tl
    sys.modules["triton"] = _triton
    sys.modules["triton.language"] = _tl

from batchgen.models.glm.glm5.Parallel_Strategy_Manager import (  # noqa: E402
    GLM5ParallelStrategyManager,
    StartupDecodeProbeError,
    StartupDecodeProbePlan,
    plan_startup_decode_probe,
    uniform_expert_assignment,
)
from batchgen.models.glm.glm5.model import Glm5MoE  # noqa: E402
from batchgen.models.glm.glm5.wrappers import (  # noqa: E402
    GLM5AttnWrapper,
    GLM5ExpertWrapper,
)
from batchgen.models.wrappers import AttnWrapperBase  # noqa: E402


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


class FakeGate:
    """Gate double with the attributes the probe reads to size its patch."""

    def __init__(self, num_experts=8, num_experts_per_tok=2):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok


class FakeMoE:
    """Decode MoE double: EP-configured, fully resident, DP-width tracked."""

    def __init__(self, *, num_experts=8, top_k=2, world_size=1,
                 resident=True, offloading=False, comm=None,
                 ep_configured=True):
        self.gate = FakeGate(num_experts, top_k)
        self.num_experts_per_tok = top_k
        per_rank = num_experts // world_size
        if ep_configured:
            self.experts_per_rank = per_rank
            self.num_persistent_local_experts = (
                per_rank if resident else per_rank - 1
            )
        self.enable_ep_offloading = offloading
        self.comm = comm
        self.num_tokens_per_rank = None
        self.ntp_calls = []

    def set_num_tokens_per_rank(self, num_tokens_per_rank):
        self.num_tokens_per_rank = int(num_tokens_per_rank)
        self.ntp_calls.append(int(num_tokens_per_rank))

    def _gate_decode(self, x):
        raise AssertionError(
            "the probe must route through its uniform _gate_decode patch"
        )


class FakeDecodeModel:
    """Callable model double: checks the probe's step inputs, runs the patched
    decode gates, and returns HF-style outputs with .logits."""

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
        hidden = torch.zeros(bsz, self.hidden)
        for layer in self.model.layers:
            moe = getattr(layer, "mlp", None)
            if moe is None or getattr(moe, "gate", None) is None:
                continue
            topk_idx, topk_weight = moe._gate_decode(hidden)
            assert topk_idx.dtype is torch.int32
            assert topk_weight.dtype is torch.float32
            assert topk_idx.shape == (bsz, moe.gate.num_experts_per_tok)
            assert topk_weight.shape == (bsz, moe.gate.num_experts_per_tok)
            assert int(topk_idx.max()) < moe.gate.num_experts
        self.calls += 1
        return SimpleNamespace(logits=torch.zeros(bsz, 1, self.vocab))


def _fake_attn_wrapper(module):
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.module = module
    return wrapper


# Larger than the fake deployment page size (64), so the aliased page tables
# carry a genuine shared-history entry next to the per-row write entry, and
# smaller than the real index_topk (2048) so the CPU fakes stay tiny.
DSA_TOPK = 100
DSA_HEAD_DIM = 128
PAGE_SIZE = 64
LATENT_DIM = 20  # kv_lora_rank 16 + qk_rope_head_dim 4 in the fakes
SCALE_BYTES = 4


def _dsa_indexer(topk=DSA_TOPK, head_dim=DSA_HEAD_DIM):
    return SimpleNamespace(index_topk=topk, index_head_dim=head_dim)


def _mla_module(indexer, skip_topk=False):
    """A Glm5MLA double WITH the DSA ``indexer`` attribute (None on a
    skip_topk shared layer, as GLM-5.2/5.3 construct it)."""
    return SimpleNamespace(
        indexer=indexer,
        skip_topk=skip_topk,
        kv_lora_rank=16,
        qk_rope_head_dim=4,
    )


def _dense_mla_module():
    """A use_dense_mla double: the ``indexer`` attribute is never set."""
    module = SimpleNamespace(kv_lora_rank=16, qk_rope_head_dim=4)
    assert not hasattr(module, "indexer")
    return module


def _build_probe_manager(monkeypatch, *, cap=4, model_type="glm_moe_dsa_5_2",
                         world_size=1, resident=True, offloading=False,
                         ep_configured=True, comm=None, page_size=PAGE_SIZE,
                         layer1_indexer="skip"):
    """A manager + environment that lets the probe run end to end on CPU.

    The two-layer model mirrors the GLM-5.2/5.3 schedule: layer 0 computes
    DSA top-k (live indexer), layer 1 is a skip_topk shared layer whose
    ``indexer`` attribute is None (or a second full layer when
    ``layer1_indexer`` passes one).
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(AttnWrapperBase, "phase", "decode", raising=False)
    monkeypatch.setattr(GLM5ExpertWrapper, "phase", "decode", raising=False)

    moe0 = FakeMoE(world_size=world_size, resident=resident,
                   offloading=offloading, comm=comm,
                   ep_configured=ep_configured)
    moe1 = FakeMoE(world_size=world_size, resident=resident,
                   offloading=offloading, comm=comm,
                   ep_configured=ep_configured)
    if layer1_indexer == "skip":
        layer1_module = _mla_module(indexer=None, skip_topk=True)
    else:
        layer1_module = _mla_module(indexer=layer1_indexer)
    layers = [
        SimpleNamespace(
            self_attn=_fake_attn_wrapper(_mla_module(indexer=_dsa_indexer())),
            mlp=moe0,
        ),
        SimpleNamespace(
            self_attn=_fake_attn_wrapper(layer1_module),
            mlp=moe1,
        ),
    ]
    model = FakeDecodeModel(layers)

    manager = object.__new__(GLM5ParallelStrategyManager)
    manager.padding_bsz = cap
    manager.model = model
    manager.world_size = world_size
    manager.loaded_model_config = SimpleNamespace(model_type=model_type)
    manager.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch=torch.device("cpu")),
        Device_Paged_KV_Config=SimpleNamespace(page_size=page_size),
    )
    return manager, model, (moe0, moe1)


_SAVED_ATTENTION_NAMES = (
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

_SAVED_GLM5_WRAPPER_NAMES = (
    "_dsa_short_count",
    "_dsa_prev_topk_indices",
    "glm5_decode_primary_slot_indices",
    "glm5_decode_aux_slot_indices",
    "glm5_dsa_graph_forward_state",
    "glm5_dsa_flashmla_graph_metadata",
)


# --------------------------------------------------------------------------- #
#  named refusals
# --------------------------------------------------------------------------- #


def test_probe_refuses_without_cuda(monkeypatch):
    manager = object.__new__(GLM5ParallelStrategyManager)
    manager.padding_bsz = 8
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(StartupDecodeProbeError, match="requires CUDA"):
        manager.startup_decode_probe_step(8)


def test_probe_refuses_without_configured_cap(monkeypatch):
    manager = object.__new__(GLM5ParallelStrategyManager)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(StartupDecodeProbeError, match="configure_decoding"):
        manager.startup_decode_probe_step(8)


def test_probe_refuses_width_above_cap(monkeypatch):
    manager, _, _ = _build_probe_manager(monkeypatch, cap=4)

    with pytest.raises(StartupDecodeProbeError, match="exceeds configured decode cap"):
        manager.startup_decode_probe_step(5)


def test_probe_refuses_wrong_phase(monkeypatch):
    manager, _, _ = _build_probe_manager(monkeypatch)
    monkeypatch.setattr(AttnWrapperBase, "phase", "prefill", raising=False)

    with pytest.raises(StartupDecodeProbeError, match="decode phase"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_dense_mla_decode_instance(monkeypatch):
    manager, model, _ = _build_probe_manager(monkeypatch)
    for layer in model.model.layers:
        layer.self_attn = _fake_attn_wrapper(_dense_mla_module())

    with pytest.raises(StartupDecodeProbeError, match="use_dense_mla"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_mixed_dense_and_dsa_layers(monkeypatch):
    manager, model, _ = _build_probe_manager(monkeypatch)
    model.model.layers[1].self_attn = _fake_attn_wrapper(_dense_mla_module())

    with pytest.raises(StartupDecodeProbeError, match="mix of DSA and dense-MLA"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_all_skip_topk_layers(monkeypatch):
    manager, model, _ = _build_probe_manager(monkeypatch)
    model.model.layers[0].self_attn = _fake_attn_wrapper(
        _mla_module(indexer=None, skip_topk=True)
    )

    with pytest.raises(StartupDecodeProbeError, match="every layer is skip_topk"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_skip_topk_first_layer(monkeypatch):
    manager, model, _ = _build_probe_manager(monkeypatch)
    model.model.layers[0].self_attn = _fake_attn_wrapper(
        _mla_module(indexer=None, skip_topk=True)
    )
    model.model.layers[1].self_attn = _fake_attn_wrapper(
        _mla_module(indexer=_dsa_indexer())
    )

    with pytest.raises(StartupDecodeProbeError, match="first attention layer"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_dsa_indexer_without_geometry(monkeypatch):
    manager, model, _ = _build_probe_manager(monkeypatch)
    model.model.layers[0].self_attn = _fake_attn_wrapper(
        _mla_module(indexer=SimpleNamespace())
    )

    with pytest.raises(StartupDecodeProbeError, match="DSA indexer geometry"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_disagreeing_dsa_index_head_dims(monkeypatch):
    manager, _, _ = _build_probe_manager(
        monkeypatch,
        layer1_indexer=_dsa_indexer(head_dim=DSA_HEAD_DIM // 2),
    )

    with pytest.raises(
        StartupDecodeProbeError, match="disagreeing DSA index_head_dim"
    ):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_streamed_decode_moe(monkeypatch):
    manager, _, _ = _build_probe_manager(monkeypatch, resident=False)

    with pytest.raises(
        StartupDecodeProbeError, match="fully resident local-expert decode MoE"
    ):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_offloaded_decode_moe(monkeypatch):
    manager, _, _ = _build_probe_manager(monkeypatch, offloading=True)

    with pytest.raises(
        StartupDecodeProbeError, match="fully resident local-expert decode MoE"
    ):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_ep_unconfigured_decode_moe(monkeypatch):
    manager, _, _ = _build_probe_manager(monkeypatch, ep_configured=False)

    with pytest.raises(StartupDecodeProbeError, match="EP-configured"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_missing_communicator_in_multi_rank_world(monkeypatch):
    manager, _, _ = _build_probe_manager(monkeypatch, world_size=2, comm=None)

    with pytest.raises(StartupDecodeProbeError, match="EP communicator"):
        manager.startup_decode_probe_step(4)


def test_probe_refuses_without_deployment_page_size(monkeypatch):
    manager, _, _ = _build_probe_manager(monkeypatch, page_size=0)

    with pytest.raises(StartupDecodeProbeError, match="deployment page size"):
        manager.startup_decode_probe_step(4)


# --------------------------------------------------------------------------- #
#  full state restore across calls (what the capture flavor relies on: the
#  probe runs BEFORE the lifecycle's decode graph capture, so everything it
#  publishes for the step must be back to its pre-probe value afterwards)
# --------------------------------------------------------------------------- #


def test_two_consecutive_probe_calls_restore_all_published_state(monkeypatch):
    manager, model, moes = _build_probe_manager(monkeypatch, cap=4)

    # Sentinels in the graph-owned per-step slots: the subsequent capture must
    # find them exactly as the worker left them, not nulled by the probe.
    sentinel_state = {"path": "graph", "bucket": 4}
    sentinel_metadata = {"bucket_size": 4}
    sentinel_slots = torch.arange(4, dtype=torch.int32)
    monkeypatch.setattr(
        GLM5AttnWrapper, "glm5_dsa_graph_forward_state", sentinel_state,
        raising=False,
    )
    monkeypatch.setattr(
        GLM5AttnWrapper, "glm5_dsa_flashmla_graph_metadata", sentinel_metadata,
        raising=False,
    )
    monkeypatch.setattr(
        GLM5AttnWrapper, "glm5_decode_primary_slot_indices", sentinel_slots,
        raising=False,
    )
    monkeypatch.setattr(
        GLM5AttnWrapper, "glm5_decode_aux_slot_indices", sentinel_slots,
        raising=False,
    )
    monkeypatch.setattr(
        GLM5AttnWrapper, "_dsa_short_count", 3, raising=False
    )
    rank_counts = torch.tensor([4])
    monkeypatch.setattr(
        Glm5MoE, "_rank_token_counts", rank_counts, raising=False
    )
    for moe in moes:
        moe.num_tokens_per_rank = 7  # worker-synced pre-probe width

    saved_attention = {
        name: getattr(AttnWrapperBase, name) for name in _SAVED_ATTENTION_NAMES
    }
    saved_wrapper = {
        name: getattr(GLM5AttnWrapper, name)
        for name in _SAVED_GLM5_WRAPPER_NAMES
    }

    during = {}

    def record():
        during.update(
            short_count=GLM5AttnWrapper._dsa_short_count,
            prev_topk=GLM5AttnWrapper._dsa_prev_topk_indices,
            primary_slots=GLM5AttnWrapper.glm5_decode_primary_slot_indices,
            aux_slots=GLM5AttnWrapper.glm5_decode_aux_slot_indices,
            graph_state=GLM5AttnWrapper.glm5_dsa_graph_forward_state,
            graph_metadata=GLM5AttnWrapper.glm5_dsa_flashmla_graph_metadata,
            rank_counts=Glm5MoE._rank_token_counts,
            debug=dict(AttnWrapperBase.batchgen_debug),
            ntp=[moe.num_tokens_per_rank for moe in moes],
        )

    model.probe_recorder = record

    manager.startup_decode_probe_step(4)
    manager.startup_decode_probe_step(4)
    assert model.calls == 2

    # During the step: eager forced, per-step graph state cleared, all rows
    # long, pure-DP width at the probe's plan on every MoE layer.
    assert during["short_count"] == 0
    assert during["prev_topk"] is None
    assert during["primary_slots"] is None
    assert during["aux_slots"] is None
    assert during["graph_state"] is None
    assert during["graph_metadata"] is None
    assert during["rank_counts"] is None
    assert during["debug"] == {
        "glm5_dsa_mode": "eager", "glm5_moe_mode": "eager",
    }
    assert during["ntp"] == [4, 4]

    # After the step: gate patch removed, DP width restored, every published
    # class attribute back to its sentinel.
    for moe in moes:
        assert "_gate_decode" not in moe.__dict__
        assert moe.num_tokens_per_rank == 7
        assert moe.ntp_calls[-1] == 7
    assert Glm5MoE._rank_token_counts is rank_counts
    for name, value in saved_wrapper.items():
        assert getattr(GLM5AttnWrapper, name) is value
    for name, value in saved_attention.items():
        assert getattr(AttnWrapperBase, name) is value


def test_probe_restores_state_on_step_failure(monkeypatch):
    manager, model, moes = _build_probe_manager(monkeypatch, cap=4)
    for moe in moes:
        moe.num_tokens_per_rank = 7

    saved_attention = {
        name: getattr(AttnWrapperBase, name) for name in _SAVED_ATTENTION_NAMES
    }
    saved_wrapper = {
        name: getattr(GLM5AttnWrapper, name)
        for name in _SAVED_GLM5_WRAPPER_NAMES
    }

    def _boom(**kwargs):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(FakeDecodeModel, "__call__", lambda self, **kw: _boom())

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        manager.startup_decode_probe_step(4)

    for moe in moes:
        assert "_gate_decode" not in moe.__dict__
        assert moe.num_tokens_per_rank == 7
    for name, value in saved_wrapper.items():
        assert getattr(GLM5AttnWrapper, name) is value
    for name, value in saved_attention.items():
        assert getattr(AttnWrapperBase, name) is value


# --------------------------------------------------------------------------- #
#  synthetic surfaces
# --------------------------------------------------------------------------- #


def _expected_aliased_table(bsz, past=DSA_TOPK, page_size=PAGE_SIZE):
    pages_per_row = (past + 1 + page_size - 1) // page_size
    table = torch.zeros(bsz, pages_per_row, dtype=torch.int32)
    table[:, past // page_size] = torch.arange(1, bsz + 1, dtype=torch.int32)
    return table


def test_probe_installs_long_rows_and_page_split_aux_for_glm52(monkeypatch):
    manager, model, _ = _build_probe_manager(
        monkeypatch, model_type="glm_moe_dsa_5_2"
    )
    seen = {}

    def record():
        primary = AttnWrapperBase.gpu_paged_kv_manager
        aux = AttnWrapperBase.gpu_paged_kv_manager_aux
        assert aux is not None
        aux_blocked, aux_v, aux_table = aux.get_layer_kv_with_page_table(1)
        primary_blocked, primary_v, primary_table = (
            primary.get_layer_kv_with_page_table(1)
        )
        seen.update(
            aux_shape=tuple(aux_blocked.shape),
            aux_dtype=aux_blocked.dtype,
            aux_v=aux_v,
            aux_table=aux_table.clone(),
            aux_config=(aux.config.kv_dtype, aux.config.k_head_dim),
            primary_shape=tuple(primary_blocked.shape),
            primary_dtype=primary_blocked.dtype,
            primary_v=primary_v,
            primary_table=primary_table.clone(),
            aux_page_size=aux.config.page_size_tokens,
            primary_page_size=primary.config.page_size_tokens,
            primary_slot_map=dict(primary._gpu_page_table_manager.seq_id_to_slot),
            aux_slot_map=dict(aux._gpu_page_table_manager.seq_id_to_slot),
            cur_batch=list(AttnWrapperBase.cur_batch),
            cache_seqlens=AttnWrapperBase.cache_seqlens.clone(),
            max_seqlen=AttnWrapperBase.max_seqlen,
            position_ids=AttnWrapperBase.position_ids.clone(),
            callbacks=(
                AttnWrapperBase.kv_append_callback,
                AttnWrapperBase.kv_append_callback_aux,
            ),
        )

    model.probe_recorder = record
    manager.startup_decode_probe_step(4)

    # The page-split FP8 carrier: index_head_dim e4m3 bytes + 4 scale bytes
    # per token, one K head, plan_bsz + 1 physical pages (shared history
    # page 0 + one write page per row), at the deployment page size.
    assert seen["aux_shape"] == (5, PAGE_SIZE, 1, DSA_HEAD_DIM + SCALE_BYTES)
    assert seen["aux_dtype"] is torch.uint8
    assert seen["aux_config"] == (torch.uint8, DSA_HEAD_DIM + SCALE_BYTES)
    assert seen["aux_v"] is None
    # The primary latent surface: kv_lora_rank + qk_rope_head_dim bf16 per
    # token, one KV head, same page geometry.
    assert seen["primary_shape"] == (5, PAGE_SIZE, 1, LATENT_DIM)
    assert seen["primary_dtype"] is torch.bfloat16
    assert seen["primary_v"] is None
    assert seen["aux_page_size"] == PAGE_SIZE
    assert seen["primary_page_size"] == PAGE_SIZE

    # Every row is minimally LONG (index_topk + 1), so the sparse selection
    # path is the one measured, never the dense all-short fallback.
    assert seen["cache_seqlens"].tolist() == [DSA_TOPK + 1] * 4
    assert seen["max_seqlen"] == DSA_TOPK + 1
    assert seen["position_ids"].flatten().tolist() == [DSA_TOPK] * 4

    expected = _expected_aliased_table(4)
    assert torch.equal(seen["aux_table"], expected)
    assert torch.equal(seen["primary_table"], expected)

    # Both slot indirections resolve every synthetic sequence id in batch
    # order, and no Host-append callback is bound during the probe.
    expected_slots = {sid: row for row, sid in enumerate(seen["cur_batch"])}
    assert seen["primary_slot_map"] == expected_slots
    assert seen["aux_slot_map"] == expected_slots
    assert seen["callbacks"] == (None, None)


def test_probe_installs_bf16_aos_aux_for_glm5_base(monkeypatch):
    manager, model, _ = _build_probe_manager(
        monkeypatch, model_type="glm_moe_dsa", layer1_indexer=_dsa_indexer()
    )
    seen = {}

    def record():
        aux = AttnWrapperBase.gpu_paged_kv_manager_aux
        aux_blocked, _, _ = aux.get_layer_kv_with_page_table(0)
        seen.update(
            aux_shape=tuple(aux_blocked.shape),
            aux_dtype=aux_blocked.dtype,
            aux_config=(aux.config.kv_dtype, aux.config.k_head_dim),
        )

    model.probe_recorder = record
    manager.startup_decode_probe_step(4)

    # GLM-5 / GLM-5.1 store bf16 AoS indexer keys: carrier == index_head_dim.
    assert seen["aux_shape"] == (5, PAGE_SIZE, 1, DSA_HEAD_DIM)
    assert seen["aux_dtype"] is torch.bfloat16
    assert seen["aux_config"] == (torch.bfloat16, DSA_HEAD_DIM)


def test_probe_primary_write_read_round_trip_through_slot_indices(monkeypatch):
    """The primary surface addresses like the real manager: the selector's
    slot-indexed paged write lands in each row's OWN physical page and reads
    back intact, while history entries alias the untouched zero page."""
    manager, model, _ = _build_probe_manager(monkeypatch)

    def record():
        primary = AttnWrapperBase.gpu_paged_kv_manager
        page_size = primary.config.page_size_tokens
        slot_of = primary._gpu_page_table_manager.seq_id_to_slot
        bsz = len(AttnWrapperBase.cur_batch)
        slot_indices = torch.tensor(
            [slot_of[sid] for sid in AttnWrapperBase.cur_batch],
            dtype=torch.int32,
        )
        positions = torch.full((bsz,), DSA_TOPK, dtype=torch.int32)
        k_tensor = torch.arange(
            1, bsz + 1, dtype=torch.bfloat16
        ).view(bsz, 1, 1, 1).expand(bsz, 1, 1, LATENT_DIM)
        primary.update_layer_decode_new_token(
            k_tensor=k_tensor,
            v_tensor=None,
            sequence_lengths=positions,
            layer_idx=1,
            slot_indices=slot_indices,
        )
        blocked, _, table = primary.get_layer_kv_with_page_table(1)
        pages = []
        for row, sid in enumerate(AttnWrapperBase.cur_batch):
            page = int(table[slot_of[sid], DSA_TOPK // page_size])
            pages.append(page)
            got = blocked[page, DSA_TOPK % page_size, 0]
            assert torch.equal(
                got, torch.full((LATENT_DIM,), row + 1, dtype=torch.bfloat16)
            )
            history_page = int(table[slot_of[sid], 0])
            assert history_page == 0
            assert float(blocked[history_page].abs().sum()) == 0.0
        assert len(set(pages)) == len(pages)  # write pages never alias
        # The surface holds the MLA latent only; a V write is a defect.
        with pytest.raises(StartupDecodeProbeError, match="no V cache"):
            primary.update_layer_decode_new_token(
                k_tensor=k_tensor,
                v_tensor=k_tensor,
                sequence_lengths=positions,
                layer_idx=1,
                slot_indices=slot_indices,
            )

    model.probe_recorder = record
    manager.startup_decode_probe_step(4)


def test_two_probe_calls_drop_every_synthetic_surface(monkeypatch):
    manager, model, _ = _build_probe_manager(monkeypatch)
    surface_refs = []
    model.probe_recorder = lambda: surface_refs.extend(
        (
            weakref.ref(AttnWrapperBase.gpu_paged_kv_manager),
            weakref.ref(AttnWrapperBase.gpu_paged_kv_manager_aux),
        )
    )

    manager.startup_decode_probe_step(4)
    manager.startup_decode_probe_step(4)
    assert model.calls == 2

    # Every synthetic surface is dropped with the step: nothing retains it
    # once the finally-restore has run.
    model.probe_recorder = None
    gc.collect()
    assert len(surface_refs) == 4
    assert all(ref() is None for ref in surface_refs)
