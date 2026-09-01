from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5 import model as glm5_model
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


class _AsyncCore(_Core):
    def free_weights_buffers_async(self, keys):
        self.log.append(("free_many_async", tuple(keys)))

    def free_weights_buffer_async(self, key):
        self.log.append(("free_async", key))


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


def test_grouped_forward_uses_core_async_release_without_python_pending_state(
    monkeypatch,
):
    log = []
    routed_core = _AsyncCore(log)
    shared_core = _AsyncCore(log)
    moe = object.__new__(Glm5MoE)
    torch.nn.Module.__init__(moe)
    moe.layer_idx = 3
    moe.num_experts_per_tok = 1
    moe.device = torch.device("cpu")
    moe.gate = lambda hidden: (
        torch.ones(hidden.shape[0], 1),
        torch.zeros(hidden.shape[0], 1, dtype=torch.int64),
    )
    moe.experts = [SimpleNamespace(core_engine=routed_core)]
    moe.shared_experts = SimpleNamespace(
        core_engine=shared_core,
        cached_gate=torch.ones(1),
        cached_up=torch.ones(1),
        cached_down=torch.ones(1),
        _forward_impl=lambda hidden: torch.ones_like(hidden),
    )
    moe._prefill_prepared_keys = ["routed_0", "routed_1"]
    moe._prefill_weight_prototypes = tuple(torch.ones(1) for _ in range(6))
    moe._prefill_shared_key = "shared"

    Glm5MoE._prefill_ptrs_dev = torch.zeros(6, 2, dtype=torch.int64)
    Glm5MoE._prefill_ring_pending = None
    Glm5MoE._prefill_shared_pending = None
    Glm5MoE._prefill_grouped_logged = True
    Glm5MoE._prefill_buf = SimpleNamespace(
        token_window=1,
        num_experts=2,
        dispatched_x=torch.empty(1, 1),
        expert_counts=torch.zeros(2, dtype=torch.int32),
        expert_counters=torch.zeros(2, dtype=torch.int32),
        cu_seqlens=torch.zeros(3, dtype=torch.int32),
        topk_pos=torch.zeros(1, dtype=torch.int32),
        x_fp8=torch.empty(1, 1, dtype=torch.float8_e4m3fn),
        x_scale=torch.empty(1, 1),
        intermediate=torch.empty(1, 1),
        inter_fp8=torch.empty(1, 1, dtype=torch.float8_e4m3fn),
        inter_scale=torch.empty(1, 1),
        expert_out=torch.empty(1, 1),
        s1_tma_desc=None,
        s3_tma_desc=None,
        tiles=None,
        cu_tiles=None,
    )

    monkeypatch.setattr(
        glm5_model,
        "_glm5_dispatch_scatter_ragged",
        lambda *args, **kwargs: (
            Glm5MoE._prefill_buf.expert_counts,
            Glm5MoE._prefill_buf.cu_seqlens,
            Glm5MoE._prefill_buf.topk_pos,
        ),
    )
    monkeypatch.setattr(
        glm5_model, "_glm5_act_quant_ragged", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        glm5_model,
        "grouped_fp8_blockwise_fused_s1_ptrs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        glm5_model,
        "grouped_fp8_blockwise_s3_ptrs",
        lambda *args, **kwargs: None,
    )

    def _reduce(*args, output, **kwargs):
        output.fill_(2)

    monkeypatch.setattr(glm5_model, "reduce_weighted_scatter_bf16_ordered", _reduce)

    output = moe._forward_prefill_grouped(torch.ones(1, 1))

    torch.testing.assert_close(output, torch.full((1, 1), 3.0))
    assert log == [
        ("free_many_async", ("routed_0", "routed_1")),
        ("free_async", "shared"),
    ]
    assert Glm5MoE._prefill_ring_pending is None
    assert Glm5MoE._prefill_shared_pending is None
    assert moe._prefill_prepared_keys is None
    assert moe._prefill_shared_key is None
