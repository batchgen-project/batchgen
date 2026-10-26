"""Kimi-K3 implements the worker's default phase-instance reuse protocol.

K3 serves both phases from one module tree and switches the routed-expert
backend per phase. Without the protocol the worker's rebuild path deep-freed
that tree on every admission wave and ran ``_build_model`` again. With it, the
tree is built once per process; each switch only swaps the expert backend:

* decode -> prefill: the worker releases the decode graph and this rank's
  resident EP shard (``release_decode_routed_experts``), then
  ``activate_prefill`` re-enters the selected prefill mode;
* prefill -> prefill (back to back): ``activate_prefill`` again, idempotent,
  with an unchanged streamed-SP8 weight-copy schedule;
* prefill -> decode: ``activate_decoding`` rebuilds the shard and installs a
  fresh decode graph adapter.

The real ``KimiLinearParallelStrategyManager`` methods run here. The fakes sit
only at the GPU boundary: the module tree (``_build_model`` needs checkpoint
weights on eight GPUs), the resident-shard builder, the streamed-SP8 layer
buffer and the decode-graph adapter. The worker side of each switch is
reproduced from the worker's own branch conditions (``batchgen_worker.py``
``_config_prefill_for_batch`` / ``_release_reused_decode_instance`` /
``_load_decode_model``).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import batchgen.models.moonshotai.kimi_linear.cuda_graph_segments as graph_module
import batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert as mxfp4_expert
import batchgen.moe.fused_moe_mxfp4_resident as resident_module
import batchgen.moe.streamed_sp8_mxfp4 as streamed_module
from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
    KimiLinearParallelStrategyManager,
)
from batchgen.models.moonshotai.kimi_linear.wrappers import (
    KimiLinearExpertWrapper,
)
from batchgen.models.wrappers import AttnWrapperBase
from batchgen.moe.fused_moe_mxfp4_resident import ResidentEPMXFP4MoELayer


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"

WORLD = 16
G = 8
NUM_LAYERS = 4
MOE_LAYERS = (1, 2, 3)     # layer 0 is dense
NUM_EXPERTS = 32           # divisible by WORLD and by G


# --------------------------------------------------------------------------
# worker helpers, exec'd from the worker source
# --------------------------------------------------------------------------

def _worker_method(name, **globals_override):
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = dict(globals_override)
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"),
         namespace)
    return namespace[name]


_FINGERPRINT = _worker_method("_weight_copy_task_fingerprint")


def _fingerprint(task):
    return _FINGERPRINT(task)


# --------------------------------------------------------------------------
# GPU-boundary fakes
# --------------------------------------------------------------------------

class _Moe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            torch.nn.Identity() for _ in range(NUM_EXPERTS)
        )
        self.routed_expert_down_proj = torch.nn.Identity()
        self.routed_expert_up_proj = torch.nn.Identity()
        self.latent_moe_use_norm = False


class _Layer(torch.nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.self_attn = torch.nn.Identity()
        if moe:
            self.block_sparse_moe = _Moe()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            _Layer(idx in MOE_LAYERS) for idx in range(NUM_LAYERS)
        )


class _Shard:
    def nbytes(self):
        return 1 << 20


class _Recorder:
    """Counts the GPU-boundary events a phase switch produces."""

    def __init__(self):
        self.model_builds = 0
        self.shard_builds = 0
        self.buffers = []
        self.graphs = []


def _install_fakes(monkeypatch, recorder):
    def build_resident_ep_mxfp4_layers(layers, *args, **kwargs):
        recorder.shard_builds += 1
        for layer in layers:
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is not None:
                moe._resident_ep_moe = SimpleNamespace(shard=_Shard())

    class _Buffer:
        def __init__(self, **kwargs):
            self.closed = 0
            recorder.buffers.append(self)

        def order_tp_collective_after_cross_launch(self):
            return None

        def close(self):
            self.closed += 1

    class _StreamedLayer:
        def __init__(self, **kwargs):
            self.layer_idx = kwargs["layer_idx"]

    class _DecodeGraph:
        def __init__(self, model, *args, **kwargs):
            self.model = model
            self.installed = 0
            self.released = 0
            recorder.graphs.append(self)

        def install(self):
            self.installed += 1

        def release(self):
            self.released += 1

        def set_mode(self, mode):
            pass

    monkeypatch.setattr(mxfp4_expert, "is_mxfp4_quantized", lambda cfg: True)
    monkeypatch.setattr(
        resident_module,
        "build_resident_ep_mxfp4_layers",
        build_resident_ep_mxfp4_layers,
    )
    monkeypatch.setattr(streamed_module, "StreamedSP8LayerBuffer", _Buffer)
    monkeypatch.setattr(
        streamed_module, "StreamedSP8MXFP4MoELayer", _StreamedLayer
    )
    monkeypatch.setattr(graph_module, "KimiLinearDecodeGraph", _DecodeGraph)


# --------------------------------------------------------------------------
# the real PSM
# --------------------------------------------------------------------------

def _loaded_model_config():
    cfg = SimpleNamespace(
        model_type="kimi_k3",
        n_routed_experts=NUM_EXPERTS,
        kda_num_heads=32,
        kda_head_dim=64,
        num_hidden_layers=NUM_LAYERS,
        moe_intermediate_size=64,
        routed_expert_hidden_size=32,
    )
    cfg.is_kda_layer = lambda layer_idx: False
    cfg.is_mla_layer = lambda layer_idx: True
    return cfg


def _engine_config(*, distributed=True, decode_moe_mode="resident_ep",
                   resident_token_cap=0):
    return SimpleNamespace(
        Basic_Config=SimpleNamespace(
            distributed_weight_sharded=distributed,
            distributed_weight_transport="host_rdma",
            attention_group_size=G,
            stream_all_modules=False,
            device_torch="cpu",
            decode_graph_mode="graph",
            decode_graph_buckets=None,
            decode_graph_compare_every=None,
        ),
        GPU_Buffer_Config=SimpleNamespace(
            kda_state_slots=8,
            kda_prefix_segment_slots=0,
            num_prefill_module_buffer={"routed_expert": 2},
        ),
        EP_Config=SimpleNamespace(decode_moe_mode=decode_moe_mode),
        Module_Batching_Config=SimpleNamespace(
            k3_resident_prefill_token_cap=resident_token_cap,
            k3_deepep_ll=False,
        ),
    )


def _psm(monkeypatch, recorder, **config):
    psm = KimiLinearParallelStrategyManager(
        loaded_model_config=_loaded_model_config(),
        engine_config=_engine_config(**config),
        model_config=SimpleNamespace(),
        core_engine=SimpleNamespace(get_tensor=lambda name: None),
        skeleton_state_dict={},
        rank=0,
        global_rank=0,
        world_size=WORLD,
    )

    def build_model():
        recorder.model_builds += 1
        psm.model = _Model()
        psm._reset_model_attached_state()

    monkeypatch.setattr(psm, "_build_model", build_model)
    psm.set_comm(object())
    return psm


@pytest.fixture(autouse=True)
def _restore_class_state():
    phases = (AttnWrapperBase.phase, KimiLinearExpertWrapper.phase)
    ResidentEPMXFP4MoELayer._prefill_y = None
    yield
    AttnWrapperBase.phase, KimiLinearExpertWrapper.phase = phases
    ResidentEPMXFP4MoELayer._prefill_y = None


@pytest.fixture
def recorder(monkeypatch):
    recorder = _Recorder()
    _install_fakes(monkeypatch, recorder)
    return recorder


# --------------------------------------------------------------------------
# the worker's side of each switch, from its own branch conditions
# --------------------------------------------------------------------------

def _startup(psm):
    """prepare_kimi_k3_startup: decode first, then the default prefill mode."""
    psm.reuse_phase_instances = True          # _request_phase_instance_reuse
    decode = psm.configure_decoding(padding_bsz=4, comm=object())
    psm.set_prefill_moe_mode(psm.default_prefill_moe_mode())
    prefill = psm.configure_prefill()
    return decode, prefill


def _decode_to_prefill(psm, mode="streamed_sp8"):
    """_config_prefill_for_batch on the reuse branch."""
    psm.set_prefill_moe_mode(mode)
    assert psm.decode_instance is not None   # `reuse` is True
    if psm.decode_experts_resident:          # _release_reused_decode_instance
        psm.release_decode_routed_experts()
    assert not psm.decode_experts_resident   # _assert_phase_switch_clean
    assert psm.prefill_instance is not None
    return psm.activate_prefill()


def _prefill_to_decode(psm):
    """_load_decode_model on the reuse branch."""
    if psm.decode_experts_resident:          # _decode_instance_is_active
        return None
    model, task = psm.activate_decoding()
    assert psm.decode_experts_resident       # _assert_phase_switch_clean
    assert task["routed_expert"] == []       # _install_decode_weight_copy_queue
    return model, task


def _resident_layers(psm):
    return [
        layer.block_sparse_moe._resident_ep_moe
        for idx, layer in enumerate(psm.model.model.layers)
        if idx in MOE_LAYERS
    ]


# --------------------------------------------------------------------------
# 1. protocol surface
# --------------------------------------------------------------------------

def test_the_protocol_surface_exists_and_starts_unpublished(monkeypatch, recorder):
    psm = _psm(monkeypatch, recorder)

    assert psm.reuse_phase_instances is False
    assert psm.prefill_instance is None
    assert psm.decode_instance is None
    assert psm.decode_experts_resident is False
    for name in ("activate_prefill", "activate_decoding",
                 "release_decode_routed_experts"):
        assert callable(getattr(psm, name))


def test_configures_publish_only_when_reuse_is_honoured(monkeypatch, recorder):
    psm = _psm(monkeypatch, recorder)

    # Not requested: the worker's rebuild path, nothing published.
    psm.configure_decoding(padding_bsz=4)
    psm.set_prefill_moe_mode("streamed_sp8")
    psm.configure_prefill()
    assert psm.decode_instance is None
    assert psm.prefill_instance is None

    psm.reuse_phase_instances = True
    decode = psm.configure_decoding(padding_bsz=4)
    assert psm.decode_instance == decode
    prefill = psm.configure_prefill()
    assert psm.prefill_instance == prefill


def test_activation_never_builds_a_model(monkeypatch, recorder):
    psm = _psm(monkeypatch, recorder)
    _startup(psm)
    assert recorder.model_builds == 1

    _decode_to_prefill(psm)
    _prefill_to_decode(psm)
    _decode_to_prefill(psm)

    assert recorder.model_builds == 1


@pytest.mark.parametrize("activate", ["activate_prefill", "activate_decoding"])
def test_activation_without_a_model_fails_loud(monkeypatch, recorder, activate):
    psm = _psm(monkeypatch, recorder)
    _startup(psm)
    psm.model = None                         # e.g. a stray deep free

    with pytest.raises(RuntimeError, match="found no model"):
        getattr(psm, activate)()
    assert recorder.model_builds == 1


def test_a_rebuild_drops_the_published_instances(monkeypatch, recorder):
    psm = _psm(monkeypatch, recorder)
    _startup(psm)
    assert psm.decode_instance is not None and psm.prefill_instance is not None

    psm._reset_model_attached_state()

    assert psm.decode_instance is None
    assert psm.prefill_instance is None
    assert psm.decode_experts_resident is False


# --------------------------------------------------------------------------
# 2. a driven cycle
# --------------------------------------------------------------------------

def test_a_full_cycle_builds_once_and_swaps_only_the_expert_backend(
    monkeypatch, recorder
):
    psm = _psm(monkeypatch, recorder)

    _startup(psm)
    assert recorder.model_builds == 1
    # Startup decode built the shard and a graph; the startup streamed-SP8
    # prefill released both.
    assert recorder.shard_builds == 1
    assert len(recorder.graphs) == 1
    assert recorder.graphs[0].released == 1
    assert psm._decode_graph is None
    assert _resident_layers(psm) == [None] * len(MOE_LAYERS)
    assert len(recorder.buffers) == 1
    assert psm.decode_experts_resident is False

    # First admission and a back-to-back wave: no decode in between.
    for _ in range(2):
        _decode_to_prefill(psm)
        assert AttnWrapperBase.phase == "prefill"
        assert KimiLinearExpertWrapper.phase == "prefill"
        assert psm.prefill_uses_streamed_sp8()
        assert len(recorder.buffers) == 1
        assert recorder.buffers[0].closed == 0
        assert recorder.shard_builds == 1

    # prefill -> decode: shard rebuilt, fresh graph adapter, SP8 buffer closed.
    _prefill_to_decode(psm)
    assert AttnWrapperBase.phase == "decode"
    assert recorder.shard_builds == 2
    assert all(layer is not None for layer in _resident_layers(psm))
    assert len(recorder.graphs) == 2
    assert psm._decode_graph is recorder.graphs[1]
    assert recorder.graphs[1].installed == 1
    assert recorder.buffers[0].closed == 1
    assert psm._streamed_sp8_buffer is None
    assert psm.decode_experts_resident is True
    # decode -> decode interval: the worker returns before any switch work.
    assert _prefill_to_decode(psm) is None
    assert recorder.shard_builds == 2

    # decode -> prefill: graph first, then shard.
    _decode_to_prefill(psm)
    assert recorder.graphs[1].released == 1
    assert _resident_layers(psm) == [None] * len(MOE_LAYERS)
    assert psm._resident_ep_built is False
    assert len(recorder.buffers) == 2

    # And back: a third graph adapter for the third decode activation.
    _prefill_to_decode(psm)
    assert recorder.shard_builds == 3
    assert len(recorder.graphs) == 3
    assert psm._decode_graph is recorder.graphs[2]
    psm.release_decode_routed_experts()
    assert recorder.graphs[2].released == 1
    assert psm._decode_graph is None
    assert psm.decode_experts_resident is False

    assert recorder.model_builds == 1


def test_a_resident_ep_prefill_does_not_pass_for_an_active_decode(
    monkeypatch, recorder
):
    """A resident-EP prefill builds the shard too; the switch must still run.

    The worker skips prefill->decode when ``decode_experts_resident`` reports
    the decode instance active. With the shard alone as the signal the model
    would stay in prefill mode through decode.
    """
    psm = _psm(monkeypatch, recorder)
    _startup(psm)
    _decode_to_prefill(psm)
    _prefill_to_decode(psm)
    builds = recorder.shard_builds

    _decode_to_prefill(psm, mode="resident_ep")
    assert psm._resident_ep_built is True
    assert psm.decode_experts_resident is False
    assert recorder.shard_builds == builds + 1

    assert _prefill_to_decode(psm) is not None
    assert AttnWrapperBase.phase == "decode"
    # The shard the resident prefill built is kept for decode.
    assert recorder.shard_builds == builds + 1
    assert psm._decode_graph is recorder.graphs[-1]


def test_a_streamed_prefill_after_a_resident_prefill_releases_the_shard(
    monkeypatch, recorder
):
    """No decode between the two waves, so the worker releases nothing.

    ``decode_experts_resident`` is False after a resident-EP prefill, so the
    worker's reuse branch skips ``release_decode_routed_experts``; entering
    streamed-SP8 prefill must drop the shard itself or the whole resident
    shard would stay mapped through the streamed wave.
    """
    psm = _psm(monkeypatch, recorder)
    _startup(psm)
    _decode_to_prefill(psm)
    _prefill_to_decode(psm)       # resident-EP prefill needs a built shard
    _decode_to_prefill(psm, mode="resident_ep")
    assert psm._resident_ep_built is True
    assert psm.decode_experts_resident is False

    _decode_to_prefill(psm, mode="streamed_sp8")

    assert psm._resident_ep_built is False
    assert all(layer is None for layer in _resident_layers(psm))
    assert psm._decode_graph is None
    assert recorder.model_builds == 1


# --------------------------------------------------------------------------
# 3. fingerprint stability
# --------------------------------------------------------------------------

def test_the_streamed_sp8_schedule_fingerprint_survives_every_switch(
    monkeypatch, recorder
):
    psm = _psm(monkeypatch, recorder)
    _, (_, startup_task) = _startup(psm)
    installed = _fingerprint(startup_task)
    # This rank's 1/G expert slice of every MoE layer, nothing else.
    per_rank = NUM_EXPERTS // G
    assert len(startup_task["routed_expert"]) == len(MOE_LAYERS) * per_rank

    fingerprints = [_fingerprint(_decode_to_prefill(psm)[1])]
    fingerprints.append(_fingerprint(_decode_to_prefill(psm)[1]))
    _prefill_to_decode(psm)
    fingerprints.append(_fingerprint(_decode_to_prefill(psm)[1]))
    fingerprints.append(_fingerprint(_decode_to_prefill(psm)[1]))

    assert fingerprints == [installed] * 4


# --------------------------------------------------------------------------
# 4. deployments that decline
# --------------------------------------------------------------------------

def test_a_resident_prefill_handoff_deployment_declines(monkeypatch, recorder):
    psm = _psm(monkeypatch, recorder, resident_token_cap=4096)
    _startup(psm)

    assert psm._instances_are_reused is False
    assert psm.decode_instance is None
    assert psm.prefill_instance is None


def test_a_declined_handoff_prefill_keeps_the_resident_shard(
    monkeypatch, recorder
):
    """The worker's ``resident_handoff`` branch, unchanged by this protocol.

    It keeps the model and the shard, releases only the decode graph, then
    calls ``configure_prefill`` in resident-EP mode, which must reuse the shard
    instead of rebuilding it from the node store.
    """
    psm = _psm(monkeypatch, recorder, resident_token_cap=4096)
    _startup(psm)
    psm.configure_decoding(padding_bsz=4)
    builds = recorder.shard_builds
    shards = _resident_layers(psm)

    psm.release_decode_graph()
    psm.set_prefill_moe_mode("resident_ep")
    psm.configure_prefill()

    assert recorder.shard_builds == builds
    assert _resident_layers(psm) == shards
    assert recorder.model_builds == 1
    assert psm.decode_instance is None
    assert psm.prefill_instance is None


def test_the_worker_tries_reuse_before_the_startup_and_handoff_branches():
    """The protocol relies on this order in ``_config_prefill_for_batch``.

    A K3 decode instance is published at startup, so the first admission must
    take ``reuse`` (not the startup reuse) and the H2D pipeline installed at
    startup must still be skipped on that admission.
    """
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node for node in worker.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_config_prefill_for_batch"
    )

    chain = None
    for node in ast.walk(method):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "persistent"
        ):
            chain = node
            break
    assert chain is not None
    order = []
    while isinstance(chain, ast.If) and isinstance(chain.test, ast.Name):
        order.append(chain.test.id)
        chain = chain.orelse[0] if len(chain.orelse) == 1 else None
    assert order[:4] == [
        "persistent", "reuse", "reuse_startup_prefill", "resident_handoff",
    ]

    guarded_install = [
        node for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "reuse_startup_prefill"
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_install_prefill_weight_copy_pipeline"
            for stmt in node.body
            for call in ast.walk(stmt)
        )
    ]
    assert guarded_install


def test_streamed_decode_declines(monkeypatch, recorder):
    psm = _psm(
        monkeypatch, recorder, distributed=False, decode_moe_mode="streamed"
    )
    _startup(psm)

    assert psm._instances_are_reused is False
    assert psm.decode_instance is None
    assert psm.prefill_instance is None


# --------------------------------------------------------------------------
# 5. --persistent-phase-instances
# --------------------------------------------------------------------------

def test_persistent_phase_instances_is_refused_by_name(monkeypatch, recorder):
    psm = _psm(monkeypatch, recorder)

    assert getattr(psm, "persistent_phase_instances", False) is False
    psm.persistent_phase_instances = False
    with pytest.raises(
        RuntimeError, match="does not support --persistent-phase-instances"
    ):
        psm.persistent_phase_instances = True
    assert psm.persistent_phase_instances is False


def test_the_worker_flag_gate_fails_at_its_first_call(monkeypatch, recorder):
    """The gate runs in the startup ``_load_decode_model``, before readiness."""
    gate = _worker_method("_persistent_phase_enabled")
    worker = SimpleNamespace(
        args=SimpleNamespace(persistent_phase_instances=True),
        _trajectory_book=object(),
        parallel_manager=_psm(monkeypatch, recorder),
    )

    with pytest.raises(
        RuntimeError, match="does not support --persistent-phase-instances"
    ):
        gate(worker)
