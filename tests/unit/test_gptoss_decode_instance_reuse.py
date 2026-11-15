"""GPT-OSS decode-instance reuse: the reuse gate, the fixed-address expert
arena, and the capture-once lifecycle through the worker's transitions.

The production ``GptOssParallelStrategyManager`` runs its real
``configure_decoding`` / ``activate_decoding`` / ``phase_exit`` against
boundary fakes: a CPU byte buffer stands in for the CUDA VMM arena (``map``
and ``unmap`` scramble its contents, as fresh physical memory is undefined),
a dict of host tensors stands in for Weights_Storage, and a small module tree
stands in for the GptOss model. The build steps that need a GPU
(skeleton/attention upload, wrapper configuration, decode-MoE setup) are
replaced by recorders; the expert loading, the arena binding and the phase
hooks are the production code.
"""

import ast
import copy
import gc
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager as psm_module
from batchgen.models.glm.glm5.cuda_graph_policy import (
    should_warmup_cuda_graphs_before_decode,
)
from batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager import (
    GptOssParallelStrategyManager,
)
from batchgen.phase_lifecycle import ExitMode, PhaseEnterContext, PhaseKey
from tests.worker.worker_fakes import attach_graph_state

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"

NUM_LAYERS = 2
NUM_EXPERTS = 8
WIDTH = 16
WEIGHT_KEYS = (
    "gate_proj.weight", "gate_proj.weight_scales",
    "up_proj.weight", "up_proj.weight_scales",
    "down_proj.weight", "down_proj.weight_scales",
)
BIAS_KEYS = ("gate_proj.bias", "up_proj.bias", "down_proj.bias")


# --------------------------------------------------------------------------
# boundary fakes
# --------------------------------------------------------------------------

class _CpuArena:
    """VmmArena's contract on a CPU buffer: fixed addresses, undefined
    contents after every map, carving only while mapped."""

    def __init__(self, nbytes):
        self.nbytes = nbytes
        self._buf = torch.zeros(nbytes + 512, dtype=torch.uint8)
        self._base = (-self._buf.data_ptr()) % 256
        self._mapped = False
        self._carved = 0
        self.maps = 0
        self.unmaps = 0

    @property
    def is_mapped(self):
        return self._mapped

    def map(self):
        if self._mapped:
            raise RuntimeError("already mapped")
        self._mapped = True
        self.maps += 1
        self._buf.fill_(0xA5)

    def unmap(self):
        if not self._mapped:
            raise RuntimeError("not mapped")
        self._mapped = False
        self.unmaps += 1
        self._buf.fill_(0x5A)

    def carve(self, shape, dtype):
        if not self._mapped:
            raise RuntimeError("map() the arena first")
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        start = (self._carved + 255) // 256 * 256
        if start + nbytes > self.nbytes:
            raise RuntimeError("exceeds the arena")
        self._carved = start + nbytes
        lo = self._base + start
        return self._buf[lo:lo + nbytes].view(dtype).view(tuple(shape))


def _host_store(seed=0, bias=True):
    gen = torch.Generator().manual_seed(seed)
    store = {}
    for layer in range(NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            tensors = {}
            for key in WEIGHT_KEYS:
                shape = (6, 3) if key.endswith("weight") else (6, 1)
                tensors[key] = torch.randint(
                    0, 255, shape, generator=gen, dtype=torch.uint8
                )
            if bias:
                for key in BIAS_KEYS:
                    tensors[key] = torch.randn(6, generator=gen).to(torch.bfloat16)
            store[f"routed_expert_{layer}_{expert}"] = tensors
    return store


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = nn.Module()
        self.mlp.experts = nn.ModuleList(nn.Module() for _ in range(NUM_EXPERTS))


class _FakeGptOss(nn.Module):
    built = 0

    def __init__(self, _config):
        super().__init__()
        type(self).built += 1
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_Layer() for _ in range(NUM_LAYERS))


def _manager(monkeypatch, *, world_size=2, global_rank=1, offloading=False,
             ratio=0.0, pre_dequant=False, store=None):
    monkeypatch.setattr(psm_module, "GptOss", _FakeGptOss)
    monkeypatch.setattr(psm_module, "log_gpu_memory", lambda *_a, **_k: None)
    monkeypatch.setattr(
        torch.cuda, "current_stream",
        lambda *_a, **_k: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_a, **_k: None)
    store = _host_store() if store is None else store
    engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch=torch.device("cpu")),
        EP_Config=SimpleNamespace(
            enable_offloading=offloading, offloading_ratio=ratio
        ),
    )
    model_config = SimpleNamespace(
        num_hidden_layers=NUM_LAYERS,
        num_local_experts=NUM_EXPERTS,
        pre_dequantize_weights=pre_dequant,
    )
    m = GptOssParallelStrategyManager(
        loaded_model_config=SimpleNamespace(phase=None),
        engine_config=engine_config,
        model_config=model_config,
        core_engine=SimpleNamespace(get_tensor=lambda key: store[key]),
        skeleton_state_dict={},
        local_rank=0,
        global_rank=global_rank,
        world_size=world_size,
    )
    m.NUM_TOTAL_EXPERTS = NUM_EXPERTS
    m.arenas = []

    def make_arena(nbytes):
        arena = _CpuArena(nbytes)
        m.arenas.append(arena)
        return arena

    m._make_expert_arena = make_arena
    m.legacy_loads = []
    m._load_expert_module_ep = lambda: m.legacy_loads.append("ep")
    m._load_expert_module = lambda: m.legacy_loads.append("all")
    for name in ("_load_model_skeleton", "_load_attn_module",
                 "_config_attn_module", "_config_expert_module",
                 "_config_lm_head_hook", "_swap_to_prefill_moe"):
        setattr(m, name, lambda: None)
    m.pointer_tables = []

    def setup_decode_moe(ep_enabled, weight_format):
        # What GptOssMoEDecode's pointer arrays capture: one address per
        # (layer, expert, tensor) of the resident experts.
        start, end = m._local_resident_expert_range()
        m.pointer_tables.append(_pointer_table(m, m.model))

    m._setup_decode_moe = setup_decode_moe
    m._dequant_experts_to_bf16 = lambda: None
    m.store = store
    return m


def _pointer_table(m, model):
    start, end = m._local_resident_expert_range()
    table = []
    for layer in range(NUM_LAYERS):
        for expert in range(start, end):
            module = model.model.layers[layer].mlp.experts[expert]
            for attr in ("mxfp4_gate_packed", "mxfp4_down_scales", "mxfp4_up_bias"):
                tensor = getattr(module, attr, None)
                table.append(None if tensor is None else tensor.data_ptr())
    return table


def _rows_match_host(m):
    start, end = m._local_resident_expert_range()
    for layer in range(NUM_LAYERS):
        for expert in range(start, end):
            module = m.model.model.layers[layer].mlp.experts[expert]
            host = m.store[f"routed_expert_{layer}_{expert}"]
            for key, attr in (("gate_proj.weight", "mxfp4_gate_packed"),
                              ("down_proj.weight_scales", "mxfp4_down_scales"),
                              ("up_proj.bias", "mxfp4_up_bias")):
                if not torch.equal(getattr(module, attr), host[key]):
                    return False
    return True


def _decode_ctx(comm):
    return PhaseEnterContext(comm=comm, max_num_seq=WIDTH)


DECODE = PhaseKey("decode", None)
PREFILL = PhaseKey("prefill", None)


# --------------------------------------------------------------------------
# the reuse gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "world_size,offloading,ratio,resident,owned",
    [
        (1, False, 0.0, NUM_EXPERTS, NUM_EXPERTS),
        (1, True, 0.5, NUM_EXPERTS, NUM_EXPERTS),
        (2, False, 0.0, 4, 4),
        (4, True, 0.5, 1, 2),
        (2, True, 0.0, 4, 4),
    ],
)
def test_planned_split_is_known_before_the_first_configure(
    monkeypatch, world_size, offloading, ratio, resident, owned
):
    m = _manager(monkeypatch, world_size=world_size, global_rank=0,
                 offloading=offloading, ratio=ratio)
    assert m._plan_local_expert_split()[1:] == (resident, owned)
    m.reuse_phase_instances = True
    assert m._instances_are_reused is (resident == owned)


def test_reuse_needs_the_worker_request(monkeypatch):
    m = _manager(monkeypatch)
    assert m._instances_are_reused is False


def test_bf16_pre_dequant_is_not_reusable(monkeypatch):
    m = _manager(monkeypatch, pre_dequant=True)
    m.reuse_phase_instances = True
    assert m._instances_are_reused is False


def test_ep_ranges_match_the_planner(monkeypatch):
    m = _manager(monkeypatch, world_size=4, global_rank=3, offloading=True,
                 ratio=0.5)
    m._compute_expert_ranges()
    assert (m.routed_expert_gpu_start_idx, m.routed_expert_gpu_end_idx) == (6, 7)
    assert m._local_resident_expert_range() == (6, 7)
    assert m.enable_ep_offloading is True


# --------------------------------------------------------------------------
# configure / suspend / activate / retire on the arena
# --------------------------------------------------------------------------

def test_unrequested_or_partial_rank_builds_always_with_the_legacy_loader(monkeypatch):
    for kwargs in ({}, {"offloading": True, "ratio": 0.5}):
        m = _manager(monkeypatch, **kwargs)
        if kwargs:
            m.reuse_phase_instances = True
        m.phase_enter(DECODE, _decode_ctx(object()))
        assert m.legacy_loads == ["ep"]
        assert m.arenas == [] and m.holds_instance("decode") is False


def test_world1_reuse_puts_every_expert_in_the_arena(monkeypatch):
    m = _manager(monkeypatch, world_size=1, global_rank=0)
    m.reuse_phase_instances = True
    m.phase_enter(DECODE, _decode_ctx(object()))
    assert m.legacy_loads == [] and len(m._expert_rows) == NUM_LAYERS * NUM_EXPERTS
    assert _rows_match_host(m)


def test_pointers_are_stable_across_suspend_and_activate(monkeypatch):
    m = _manager(monkeypatch)
    m.reuse_phase_instances = True
    comm = object()
    model, _ = m.phase_enter(DECODE, _decode_ctx(comm))
    assert m.holds_instance("decode") and m.decode_experts_resident
    assert m.legacy_loads == [] and _rows_match_host(m)
    table = m.pointer_tables[0]
    arena = m.arenas[0]

    for _ in range(2):
        m.phase_exit(DECODE, ExitMode.SUSPEND)
        assert not m.decode_experts_resident and m.holds_instance("decode")
        m.phase_enter(PREFILL, PhaseEnterContext())  # stubbed-out prefill build
        resumed, _ = m.phase_enter(DECODE, _decode_ctx(comm))
        assert resumed is model and m.model is model
        assert m.loaded_model_config.phase == "decode"
        # Same arena, same rows, same addresses; refilled bytes.
        assert m.arenas == [arena]
        assert _pointer_table(m, model) == table
        assert _rows_match_host(m)
    assert m.arenas == [arena] and arena.maps == 3 and arena.unmaps == 2
    # Only the first decode entry built anything.
    assert len(m.pointer_tables) == 1


def test_retire_unmaps_and_a_rebuild_reuses_the_same_addresses(monkeypatch):
    m = _manager(monkeypatch)
    m.reuse_phase_instances = True
    comm = object()
    m.phase_enter(DECODE, _decode_ctx(comm))
    table = m.pointer_tables[0]
    arena = m.arenas[0]

    m.phase_exit(DECODE, ExitMode.RETIRE)
    assert not m.holds_instance("decode") and not arena.is_mapped
    m.phase_enter(DECODE, _decode_ctx(comm))
    assert m.arenas == [arena] and arena.is_mapped
    assert m.pointer_tables[1] == table and _rows_match_host(m)
    assert m.holds_instance("decode")


def test_build_refuses_an_arena_left_mapped(monkeypatch):
    m = _manager(monkeypatch)
    m.reuse_phase_instances = True
    m.phase_enter(DECODE, _decode_ctx(object()))
    m.decode_instance = None  # lost without a RETIRE
    with pytest.raises(RuntimeError, match="still mapped"):
        m.phase_enter(DECODE, _decode_ctx(m.comm))


def test_host_shape_mismatch_fails_loud(monkeypatch):
    store = _host_store()
    store["routed_expert_1_5"]["up_proj.weight"] = torch.zeros(6, 4, dtype=torch.uint8)
    m = _manager(monkeypatch, store=store)
    m.reuse_phase_instances = True
    with pytest.raises(RuntimeError, match="does not match the arena row"):
        m.phase_enter(DECODE, _decode_ctx(object()))


def test_bias_set_mismatch_fails_loud(monkeypatch):
    store = _host_store()
    del store["routed_expert_1_6"]["down_proj.bias"]
    m = _manager(monkeypatch, store=store)
    m.reuse_phase_instances = True
    with pytest.raises(RuntimeError, match="bias set differs"):
        m.phase_enter(DECODE, _decode_ctx(object()))


def test_arena_size_covers_every_row_with_carve_alignment(monkeypatch):
    m = _manager(monkeypatch, store=_host_store(bias=False))
    m.reuse_phase_instances = True
    m.phase_enter(DECODE, _decode_ctx(object()))
    # 6 tensors per expert, each padded to 256 bytes, 4 experts, 2 layers.
    assert m.arenas[0].nbytes == 6 * 256 * 4 * NUM_LAYERS
    assert all(
        getattr(m.model.model.layers[0].mlp.experts[4], attr, None) is None
        for attr in ("mxfp4_gate_bias", "mxfp4_up_bias", "mxfp4_down_bias")
    )


# --------------------------------------------------------------------------
# the worker lifecycle: capture once across two decode intervals
# --------------------------------------------------------------------------

_WORKER_METHODS = (
    "_phase_exit_mode",
    "_prepare_phase_exit",
    "_transition_phase",
    "_finish_decode_interval",
    "_assert_decode_graph_capture_allowed",
    "_record_decode_graph_capture",
    "_release_decode_graph_state",
    "deep_free_model_memory",
    "_request_phase_instance_reuse",
)


def _isolated_worker_class():
    tree = ast.parse(WORKER.read_text())
    source = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    methods = [
        copy.deepcopy(node) for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in _WORKER_METHODS
    ]
    assert {node.name for node in methods} == set(_WORKER_METHODS)
    module = ast.Module(
        body=[ast.ClassDef(name="Worker", bases=[], keywords=[], body=methods,
                           decorator_list=[])],
        type_ignores=[],
    )
    namespace = {
        "ExitMode": ExitMode, "PhaseEnterContext": PhaseEnterContext,
        "PhaseKey": PhaseKey, "torch": torch, "sys": sys, "time": time,
        "gc": gc, "logging": logging,
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace["Worker"]


def _worker(manager, events):
    worker = _isolated_worker_class()()
    worker.rank = 0
    worker._live_phase_key = None
    worker.parallel_manager = manager
    worker.comm = object()
    worker.args = SimpleNamespace(persistent_phase_instances=False)
    worker._streamed_sp8_h2d_installed = False
    worker._weight_passes = SimpleNamespace(close=lambda: None)
    worker.core_engine = SimpleNamespace(
        stop_h2d_worker=lambda: None,
        clear_kv_copy_queue=lambda: None,
        reset_decoding_buffer=lambda: None,
    )
    worker._destroy_gpu_paged_kv_cache = lambda **kw: events.append("destroy_kv")
    worker._persistent_phase_enabled = lambda: False
    worker._release_suspended_decode_kv_pool = lambda: events.append("release_kv")
    worker._log_phase_memory_ledger = lambda label: None
    worker._unregister_fp8_weights = lambda: events.append("unregister")
    worker.init_nvshmem = lambda: None
    # S2: the decode-graph state is GraphRuntime's (adapter/manager None,
    # capture counter 0 by default).
    attach_graph_state(worker)
    worker.model = None
    worker.weight_copy_task = {}
    return worker


def _capture(worker, events):
    """The `_setup_cuda_graphs` contract: guard, capture, count."""
    worker._assert_decode_graph_capture_allowed()
    events.append("capture")
    worker._graph.cuda_graph_manager = object()
    worker._graph.whole_model_segment = object()
    worker._record_decode_graph_capture()


def _decode_interval(worker, events):
    worker._request_phase_instance_reuse()  # as _load_decode_model does
    changed, resumed = worker._transition_phase(
        DECODE, comm=worker.comm, max_num_seq=WIDTH
    )
    if should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=worker._graph.cuda_graph_manager is not None,
        global_batch_has_queueing=False,
        model_name="gpt-oss-120b",
    ):
        _capture(worker, events)
    worker._finish_decode_interval()
    worker._transition_phase(PREFILL)
    return changed, resumed


def test_two_decode_intervals_capture_exactly_once(monkeypatch):
    m = _manager(monkeypatch)
    events = []
    worker = _worker(m, events)
    worker._transition_phase(PREFILL)

    first = _decode_interval(worker, events)
    second = _decode_interval(worker, events)

    assert first == (True, False)
    # The prefill instance is never kept, so prefill exits RETIRE; the decode
    # entry still resumes the kept decode instance.
    assert second == (True, True)
    assert events.count("capture") == 1
    assert worker._graph.decode_graph_captures == 1
    assert worker._graph.cuda_graph_manager is not None
    assert m.arenas[0].maps == 2 and m.arenas[0].unmaps == 2
    assert len(m.pointer_tables) == 1
    assert "unregister" not in events  # decode interval keeps the instance


def test_capture_requested_again_for_a_kept_instance_fails_loud(monkeypatch):
    m = _manager(monkeypatch)
    events = []
    worker = _worker(m, events)
    _decode_interval(worker, events)
    worker._graph.cuda_graph_manager = None  # graph state lost under a kept instance
    with pytest.raises(RuntimeError, match="captured once per"):
        _decode_interval(worker, events)
    worker._criticality_probe_recapture = True
    worker._transition_phase(PREFILL)
    _decode_interval(worker, events)  # the armed probe recapture is admitted
    assert events.count("capture") == 2


def test_build_always_rank_recaptures_every_interval(monkeypatch):
    # A rank the gate declines (partial residency) keeps today's lifecycle.
    m = _manager(monkeypatch, offloading=True, ratio=0.5)
    events = []
    worker = _worker(m, events)
    _decode_interval(worker, events)
    _decode_interval(worker, events)
    assert events.count("capture") == 2
    assert worker._graph.decode_graph_captures == 0  # released with the instance
