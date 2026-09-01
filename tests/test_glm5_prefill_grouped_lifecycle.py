from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5.model import Glm5MoE


class _Event:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def synchronize(self):
        self.log.append(("sync", self.name))


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


def _make_moe(experts, shared):
    moe = object.__new__(Glm5MoE)
    torch.nn.Module.__init__(moe)
    moe.config = SimpleNamespace(phase="prefill")
    moe._prefill_grouped_enabled = True
    moe._prefill_prepared_keys = None
    moe._prefill_weight_prototypes = None
    moe._prefill_shared_key = None
    moe.experts = experts
    moe.shared_experts = shared
    return moe


@pytest.fixture(autouse=True)
def _reset_prefill_class_state():
    Glm5MoE._prefill_buf = object()
    Glm5MoE._prefill_ptrs_pinned = torch.empty(6, 2, dtype=torch.int64)
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
