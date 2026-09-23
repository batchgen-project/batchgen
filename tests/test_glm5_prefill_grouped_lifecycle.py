from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5.model import (
    Glm5DecoderLayer,
    Glm5MoE,
    _glm5_accumulate_shared_expert_chunked,
)


class _Event:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def synchronize(self):
        self.log.append(("sync", self.name))

    def record(self, stream=None):
        self.log.append(("record", self.name))


class _DeferredFuture:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        self.completed = False

    def result(self):
        if not self.completed:
            self.fn(*self.args)
            self.completed = True


class _DeferredExecutor:
    def submit(self, fn, *args):
        return _DeferredFuture(fn, args)

    def shutdown(self, wait=True):
        assert wait is True


class _Core:
    def __init__(self, log):
        self.log = log

    def free_weights_buffer(self, key):
        self.log.append(("free", key))


class _Expert:
    def __init__(self, key, core, log, *, fail=False):
        self.module_key = key
        self.core_engine = core
        self.log = log
        self.fail = fail
        self.persistent = False
        self.is_fp8 = True
        self.weight_dequant_scale = {
            "gate_proj.weight_scale_inv": torch.ones(1, 1),
            "up_proj.weight_scale_inv": torch.ones(1, 1),
            "down_proj.weight_scale_inv": torch.ones(1, 1),
        }
        self.weights = {
            "gate_proj.weight": torch.ones(1, 1),
            "up_proj.weight": torch.ones(1, 1),
            "down_proj.weight": torch.ones(1, 1),
        }

    def load_weights_pinned(self):
        self.log.append(("load", self.module_key))
        if self.fail:
            raise RuntimeError("load failed")
        return self.weights


class _FailingAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(phase="prefill")

    def forward(self, **kwargs):
        raise RuntimeError("attention failed")


def _make_moe(experts, shared):
    moe = object.__new__(Glm5MoE)
    torch.nn.Module.__init__(moe)
    moe.config = SimpleNamespace(phase="prefill")
    moe._prefill_grouped_enabled = True
    moe._prefill_prepared_keys = None
    moe._prefill_weight_prototypes = None
    moe._prefill_release_event = None
    moe._prefill_shared_key = None
    moe._prefill_shared_release_event = None
    moe.layer_idx = 3
    moe.device = torch.device("cpu")
    moe.experts = experts
    moe.shared_experts = shared
    return moe


def test_shared_expert_prefill_is_accumulated_in_bounded_chunks():
    class _Shared:
        def __init__(self):
            self.rows = []

        def _forward_impl(self, hidden_states):
            self.rows.append(hidden_states.shape[0])
            return hidden_states * 3

    hidden = torch.arange(44, dtype=torch.float32).view(11, 4)
    output = torch.ones_like(hidden)
    shared = _Shared()

    result = _glm5_accumulate_shared_expert_chunked(
        output,
        hidden,
        shared,
        chunk_rows=4,
    )

    assert result is output
    assert shared.rows == [4, 4, 3]
    torch.testing.assert_close(result, 1 + hidden * 3)


@pytest.fixture(autouse=True)
def _reset_prefill_class_state():
    Glm5MoE._prefill_buf = object()
    Glm5MoE._prefill_ptrs_pinned = torch.empty(78, 6, 2, dtype=torch.int64)
    Glm5MoE._prefill_ptrs_dev = torch.empty(6, 2, dtype=torch.int64)
    Glm5MoE._prefill_ring_pending = None
    Glm5MoE._prefill_shared_pending = None
    Glm5MoE._prefill_retire_executor = _DeferredExecutor()
    Glm5MoE._prefill_retire_future = None
    yield
    Glm5MoE.retire_prefill_grouped_weights()
    Glm5MoE._prefill_buf = None
    Glm5MoE._prefill_ptrs_pinned = None
    Glm5MoE._prefill_ptrs_dev = None
    Glm5MoE._prefill_ring_pending = None
    Glm5MoE._prefill_shared_pending = None
    Glm5MoE._prefill_retire_executor = None
    Glm5MoE._prefill_retire_future = None


def test_prepare_uses_distinct_pinned_h2d_source_per_layer():
    log = []
    core = _Core(log)
    layer3 = _make_moe(
        [_Expert("l3_e0", core, log), _Expert("l3_e1", core, log)],
        _Expert("l3_shared", core, log),
    )
    layer3._prefill_prepare_weights()
    layer3_stage = Glm5MoE._prefill_ptrs_pinned[3].clone()

    layer4 = _make_moe(
        [_Expert("l4_e0", core, log), _Expert("l4_e1", core, log)],
        _Expert("l4_shared", core, log),
    )
    layer4.layer_idx = 4
    layer4._prefill_prepare_weights()

    torch.testing.assert_close(Glm5MoE._prefill_ptrs_pinned[3], layer3_stage)
    assert not torch.equal(
        Glm5MoE._prefill_ptrs_pinned[3],
        Glm5MoE._prefill_ptrs_pinned[4],
    )


def test_prepare_retires_previous_layers_without_blocking_current_acquisition():
    log = []
    core = _Core(log)
    experts = [_Expert("routed_0", core, log), _Expert("routed_1", core, log)]
    shared = _Expert("shared", core, log)
    moe = _make_moe(experts, shared)

    Glm5MoE._prefill_ring_pending = (
        _Event(log, "routed_prev"),
        ["routed_prev_0", "routed_prev_1"],
        core,
    )
    Glm5MoE._prefill_shared_pending = (
        _Event(log, "shared_prev"),
        "shared_prev",
        core,
    )

    moe._prefill_prepare_weights()

    assert log == [
        ("load", "routed_0"),
        ("load", "routed_1"),
        ("load", "shared"),
    ]
    assert moe._prefill_prepared_keys == ["routed_0", "routed_1"]
    assert moe._prefill_shared_key == "shared"
    assert Glm5MoE._prefill_ring_pending is None
    assert Glm5MoE._prefill_shared_pending is None

    Glm5MoE.retire_prefill_grouped_weights()
    assert log == [
        ("load", "routed_0"),
        ("load", "routed_1"),
        ("load", "shared"),
        ("sync", "routed_prev"),
        ("free", "routed_prev_0"),
        ("free", "routed_prev_1"),
        ("sync", "shared_prev"),
        ("free", "shared_prev"),
    ]

    staged = Glm5MoE._prefill_ptrs_dev
    for expert_idx, expert in enumerate(experts):
        assert staged[0, expert_idx].item() == expert.weights[
            "gate_proj.weight"
        ].data_ptr()
        assert staged[1, expert_idx].item() == expert.weight_dequant_scale[
            "gate_proj.weight_scale_inv"
        ].data_ptr()
        assert staged[2, expert_idx].item() == expert.weights[
            "up_proj.weight"
        ].data_ptr()
        assert staged[4, expert_idx].item() == expert.weights[
            "down_proj.weight"
        ].data_ptr()


def test_prepare_failure_releases_every_successfully_acquired_routed_slot():
    log = []
    core = _Core(log)
    experts = [
        _Expert("routed_0", core, log),
        _Expert("routed_1", core, log, fail=True),
    ]
    moe = _make_moe(experts, _Expert("shared", core, log))

    with pytest.raises(RuntimeError, match="load failed"):
        moe._prefill_prepare_weights()

    assert log == [
        ("load", "routed_0"),
        ("load", "routed_1"),
        ("free", "routed_0"),
    ]
    assert moe._prefill_prepared_keys is None


def test_prepare_failure_after_shared_acquisition_releases_and_clears_caches():
    class _FailingCopy:
        def copy_(self, source, non_blocking=False):
            raise RuntimeError("pointer copy failed")

    log = []
    core = _Core(log)
    experts = [_Expert("routed_0", core, log), _Expert("routed_1", core, log)]
    shared = _Expert("shared", core, log)
    moe = _make_moe(experts, shared)
    Glm5MoE._prefill_ptrs_dev = _FailingCopy()

    with pytest.raises(RuntimeError, match="pointer copy failed"):
        moe._prefill_prepare_weights()

    assert log == [
        ("load", "routed_0"),
        ("load", "routed_1"),
        ("load", "shared"),
        ("free", "routed_0"),
        ("free", "routed_1"),
        ("free", "shared"),
    ]
    assert shared.cached_gate is None
    assert shared.cached_up is None
    assert shared.cached_down is None


def test_forward_failure_fences_and_releases_every_owned_slot(monkeypatch):
    log = []
    core = _Core(log)
    experts = [_Expert("routed_0", core, log), _Expert("routed_1", core, log)]
    shared = _Expert("shared", core, log)
    moe = _make_moe(experts, shared)
    moe._prefill_prepared_keys = ["routed_0", "routed_1"]
    moe._prefill_weight_prototypes = (object(),) * 6
    moe._prefill_shared_key = "shared"
    shared.cached_gate = shared.cached_up = shared.cached_down = object()
    moe._prefill_release_event = _Event(log, "routed")
    moe._prefill_shared_release_event = _Event(log, "shared")

    def _fail_forward(self, hidden_states):
        raise RuntimeError("grouped compute failed")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: object())
    monkeypatch.setattr(Glm5MoE, "_forward_prefill_grouped_impl", _fail_forward)

    with pytest.raises(RuntimeError, match="grouped compute failed"):
        moe._forward_prefill_grouped(torch.ones(1, 1))

    assert log == [
        ("record", "routed"),
        ("record", "shared"),
        ("sync", "routed"),
        ("free", "routed_0"),
        ("free", "routed_1"),
        ("sync", "shared"),
        ("free", "shared"),
    ]
    assert moe._prefill_prepared_keys is None
    assert moe._prefill_weight_prototypes is None
    assert moe._prefill_shared_key is None
    assert shared.cached_gate is None
    assert shared.cached_up is None
    assert shared.cached_down is None


def test_attention_failure_releases_weights_prefetched_by_decoder(monkeypatch):
    log = []
    core = _Core(log)
    experts = [_Expert("routed_0", core, log), _Expert("routed_1", core, log)]
    shared = _Expert("shared", core, log)
    moe = _make_moe(experts, shared)
    moe._prefill_release_event = _Event(log, "routed")
    moe._prefill_shared_release_event = _Event(log, "shared")

    layer = object.__new__(Glm5DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 3
    layer.mlp = moe
    layer.self_attn = _FailingAttention()
    layer.input_layernorm = torch.nn.Identity()

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: object())

    with pytest.raises(RuntimeError, match="attention failed"):
        layer(torch.ones(1, 1))

    assert log == [
        ("load", "routed_0"),
        ("load", "routed_1"),
        ("load", "shared"),
        ("record", "routed"),
        ("record", "shared"),
        ("sync", "routed"),
        ("free", "routed_0"),
        ("free", "routed_1"),
        ("sync", "shared"),
        ("free", "shared"),
    ]


def test_terminal_retire_waits_once_then_releases_both_weight_classes():
    log = []
    core = _Core(log)
    Glm5MoE._prefill_ring_pending = (
        _Event(log, "routed"),
        ["routed_0", "routed_1"],
        core,
    )
    Glm5MoE._prefill_shared_pending = (
        _Event(log, "shared"),
        "shared_0",
        core,
    )

    Glm5MoE.retire_prefill_grouped_weights()
    Glm5MoE.retire_prefill_grouped_weights()

    assert log == [
        ("sync", "routed"),
        ("free", "routed_0"),
        ("free", "routed_1"),
        ("sync", "shared"),
        ("free", "shared_0"),
    ]
