"""A rebuild cycle driven through the REAL Kimi-Linear lifecycle objects.

Every prefill<->decode switch that does not keep its phase instances runs
``deep_free_model_memory()`` and then rebuilds: ``configure_decoding`` finds
``self.model is None`` and calls ``_build_model()`` again. Five consumers were
coupled to a model identity that no longer holds across that boundary, and
each one failed only on the *second* switch, inside a live serve run.

The sibling test files pin the decision functions, the structure and the
fakes. This one walks the cycle itself — build -> reset -> rebuild, twice —
through the real ``KimiLinearParallelStrategyManager``, the real
``KimiLinearKDAWrapper`` classmethods, the real ``KDAStateGPUManager`` pools
and the real ``ResidentEPMXFP4MoELayer`` class-level buffer, so an invariant
that holds per function but breaks when they are composed cannot pass here.

Boundary substitutions (everything else is the production object and the
production bound method):

* ``device="cpu"`` for the state pools. ``KDAStateGPUManager.initialize()``
  allocates plain ``torch.zeros`` and touches ``torch.cuda.set_device`` only
  when the device is CUDA, so the slot accounting, the per-layer views and
  their fixed base addresses are exercised unchanged — just in host memory.
* ``dist.new_group`` / ``dist.is_initialized`` are monkeypatched, because a
  NCCL sub-group needs a 16-rank world. The call COUNT is what the once-guard
  is about, and that is what is recorded.
* The decode graph is a recorder with ``release()``, and the resident MLA
  module used for the pool-layout cross-check is a value holder. Both are
  read-only leaves: the PSM only calls ``release()`` on one and reads head
  counts off the other.

No model is built (that needs 93 layers of checkpoint weights on 8 GPUs), so
the pool-geometry arguments are assembled exactly as ``_build_model`` step 4
assembles them, from the live PSM's own derived attributes.
"""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
    KimiLinearParallelStrategyManager,
)
from batchgen.models.moonshotai.kimi_linear.wrappers import (
    KimiLinearKDAWrapper,
)
from batchgen.moe.fused_moe_mxfp4_resident import ResidentEPMXFP4MoELayer


ROOT = Path(__file__).resolve().parents[2]


def _load_kda_state_manager():
    """Import the REAL KDA state manager without ``batchgen.kv_cache.__init__``.

    That package ``__init__`` eagerly pulls the Triton-backed token-update
    kernels, which a CPU-only machine does not have and which are not on the
    state-pool path at all (the fla kernel updates each sequence's state in
    place; this manager only allocates, hands out views and accounts slots).
    So the package is served as a path-only namespace and the kernel
    re-export module is stubbed; the manager, its config and the view helpers
    it imports are the production code. Every ``sys.modules`` entry added
    here is removed again, so no other test module sees a leaked stub.
    """
    saved = dict(sys.modules)
    try:
        package = types.ModuleType("batchgen.kv_cache")
        package.__path__ = [str(ROOT / "batchgen" / "kv_cache")]
        sys.modules["batchgen.kv_cache"] = package
        kernels = types.ModuleType("batchgen.kv_cache.gpu_kv_kernels")
        kernels.run_paged_kv_token_update = lambda *a, **k: None
        kernels.run_paged_kv_token_update_fused = lambda *a, **k: None
        sys.modules["batchgen.kv_cache.gpu_kv_kernels"] = kernels
        return importlib.import_module(
            "batchgen.kv_cache.kda_state_gpu_manager"
        )
    finally:
        for name in [name for name in sys.modules if name not in saved]:
            del sys.modules[name]
        sys.modules.update(saved)


KDA_STATE = _load_kda_state_manager()

WORLD = 16
G = 8                     # attention_group_size: head-parallel KDA (M2a)
NUM_LAYERS = 6
KDA_LAYERS = (0, 2)       # the rest are MLA layers
POOL_SLOTS = 8
KDA_HEADS = 32            # 4 local heads per rank at G=8
KDA_HEAD_DIM = 64
CONV_WIDTH = 4
# prefill_pool_layout's MLA geometry.
MLA_HEADS = 32
K_NOPE, K_ROPE, V_DIM = 128, 64, 128
LIVE_SEQUENCES = (41, 42, 43)


# --------------------------------------------------------------------------
# the real PSM, constructed from minimal config objects
# --------------------------------------------------------------------------

def _loaded_model_config():
    cfg = SimpleNamespace(
        model_type="kimi_k3",
        n_routed_experts=896,
        kda_num_heads=KDA_HEADS,
        kda_head_dim=KDA_HEAD_DIM,
        kda_conv_size=CONV_WIDTH,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=MLA_HEADS,
        qk_nope_head_dim=K_NOPE,
        qk_rope_head_dim=K_ROPE,
        v_head_dim=V_DIM,
    )
    cfg.is_kda_layer = lambda layer_idx: layer_idx in KDA_LAYERS
    cfg.is_mla_layer = lambda layer_idx: layer_idx not in KDA_LAYERS
    return cfg


def _engine_config(**basic):
    fields = dict(
        distributed_weight_sharded=False,
        distributed_weight_transport="host_rdma",
        attention_group_size=G,
        stream_all_modules=False,
        device_torch="cpu",
    )
    fields.update(basic)
    return SimpleNamespace(
        Basic_Config=SimpleNamespace(**fields),
        GPU_Buffer_Config=SimpleNamespace(
            kda_state_slots=POOL_SLOTS,
            kda_prefix_segment_slots=0,
        ),
    )


def _psm(*, rank=0, **basic):
    """A real KimiLinearParallelStrategyManager. No CUDA in ``__init__``."""
    return KimiLinearParallelStrategyManager(
        loaded_model_config=_loaded_model_config(),
        engine_config=_engine_config(**basic),
        model_config=SimpleNamespace(),
        core_engine=SimpleNamespace(),
        skeleton_state_dict={},
        rank=rank,
        global_rank=rank,
        world_size=WORLD,
    )


# --------------------------------------------------------------------------
# the rebuild steps, as _build_model performs them
# --------------------------------------------------------------------------

def _build_state_pools(psm, *, slots=None):
    """``_build_model`` step 4, with this PSM's own derived geometry."""
    cfg = psm.loaded_model_config
    kda_indices = [
        i for i in range(cfg.num_hidden_layers) if cfg.is_kda_layer(i)
    ]
    KimiLinearKDAWrapper.init_state_pools(
        kda_indices,
        num_slots=psm._kda_pool_slots if slots is None else slots,
        num_heads=psm._attn_tp_hl,
        head_dim=cfg.kda_head_dim,
        conv_width=cfg.kda_conv_size,
        proj_size=psm._attn_tp_hl * cfg.kda_head_dim,
        device="cpu",
        dtype=torch.bfloat16,
    )


def _pool_addresses():
    """Base addresses of the live pools; CUDA-graph capture bakes these."""
    manager = KimiLinearKDAWrapper.state_manager
    views = [manager.get_layer_recurrent_view(physical) for physical in (0, 1)]
    views += list(manager.get_layer_conv_views(0))
    return tuple(view.data_ptr() for view in views)


class _FakeDecodeGraph:
    """Read-only leaf: the PSM only ever calls ``release()`` on it."""

    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1


def _attach_decode_phase_state(psm):
    """What ``configure_decoding`` hangs off the model it just built."""
    graph = _FakeDecodeGraph()
    psm._resident_ep_built = True
    psm._streamed_sp8_buffer = torch.zeros(4)
    psm._decode_graph = graph
    # Class-level, so nothing on the instance drops it.
    ResidentEPMXFP4MoELayer._prefill_y = torch.zeros(4)
    return graph


def _resident_mla_model(num_heads, *, v_head_dim=V_DIM):
    attn = SimpleNamespace(
        num_heads=num_heads,
        qk_nope_head_dim=K_NOPE,
        qk_rope_head_dim=K_ROPE,
        v_head_dim=v_head_dim,
    )
    layers = [SimpleNamespace(self_attn=attn) for _ in range(NUM_LAYERS)]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


@pytest.fixture(autouse=True)
def _clean_class_state(monkeypatch):
    """The KDA pools and the FP32 combine buffer are class-level (per process)."""
    # ``init_state_pools`` imports the manager at call time; serve the module
    # loaded above so the import inside the production method resolves.
    monkeypatch.setitem(
        sys.modules, "batchgen.kv_cache.kda_state_gpu_manager", KDA_STATE
    )
    KimiLinearKDAWrapper.reset()
    ResidentEPMXFP4MoELayer._prefill_y = None
    yield
    KimiLinearKDAWrapper.reset()
    ResidentEPMXFP4MoELayer._prefill_y = None


@pytest.fixture
def groups(monkeypatch):
    """Count real ``dist.new_group`` calls; a 16-rank world needs no NCCL here."""
    created = []

    def new_group(ranks=None, **kwargs):
        created.append(tuple(ranks or ()))
        return f"group#{len(created)}"

    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    return created


# --------------------------------------------------------------------------
# the cycle
# --------------------------------------------------------------------------

def test_two_rebuild_cycles_keep_every_cross_phase_holder(groups):
    """build -> reset -> rebuild, twice, through the real objects.

    Each assertion below is one of the five regressions, checked in the
    composed order a serve run executes them.
    """
    psm = _psm()

    # ---- cycle 1 build -------------------------------------------------
    _build_state_pools(psm)
    psm._build_attn_tp_group()
    first_manager = KimiLinearKDAWrapper.state_manager
    first_addresses = _pool_addresses()
    groups_after_first_build = len(groups)
    first_attn_group = psm._attn_tp_group
    # attention + node-local weight families, one group per block of G ranks.
    assert groups_after_first_build == 2 * (WORLD // G)
    assert first_attn_group is not None

    # Prefill writes each sequence's recurrent + conv state: for a linear
    # attention model that state IS its KV.
    slots = {
        seq: KimiLinearKDAWrapper.slot_manager.alloc(seq)
        for seq in LIVE_SEQUENCES
    }
    assert sorted(slots) == sorted(LIVE_SEQUENCES)
    live_map = first_manager.sequence_state_items
    assert live_map == slots

    decode_graph = _attach_decode_phase_state(psm)
    layout_with_model = None

    for cycle in (1, 2):
        # ---- the switch: deep free + _reset_model_attached_state --------
        psm.model = None
        psm._reset_model_attached_state()
        assert psm._resident_ep_built is False, cycle
        assert psm._streamed_sp8_buffer is None, cycle
        assert psm._decode_graph is None, cycle
        assert decode_graph.releases == 1, cycle
        assert ResidentEPMXFP4MoELayer._prefill_y is None, cycle
        # The NCCL groups are world-level: the reset must NOT drop them.
        assert psm._attn_tp_groups_built is True, cycle
        assert psm._attn_tp_group_signature == (WORLD, G), cycle

        # ---- planning in the phase window, with no model ---------------
        layout = psm.prefill_pool_layout()
        assert psm.model is None, cycle
        assert layout["num_kv_heads"] == MLA_HEADS // G, cycle
        assert layout["k_head_dim"] == K_NOPE + K_ROPE, cycle
        assert layout["v_head_dim"] == V_DIM, cycle
        assert layout["num_layers"] == NUM_LAYERS - len(KDA_LAYERS), cycle
        assert layout["layer_map"] == (-1, 0, -1, 1, 2, 3), cycle

        # ---- rebuild: _build_model runs again --------------------------
        _build_state_pools(psm)
        psm._build_attn_tp_group()

        # The pools are kept, so the sequences keep their state AND their
        # addresses (what CUDA-graph capture baked).
        assert KimiLinearKDAWrapper.state_manager is first_manager, cycle
        assert first_manager.sequence_state_items == slots, cycle
        assert _pool_addresses() == first_addresses, cycle
        # The facade and the per-layer views are re-derived onto them.
        assert KimiLinearKDAWrapper.slot_manager.state_manager is first_manager
        assert sorted(KimiLinearKDAWrapper.layer_pools) == list(KDA_LAYERS)
        for layer_idx in KDA_LAYERS:
            state = KimiLinearKDAWrapper.layer_pools[layer_idx]
            assert state.slot_manager is KimiLinearKDAWrapper.slot_manager
            assert state.slot_manager.contains(LIVE_SEQUENCES[0])
        # No new communicators on any later build.
        assert len(groups) == groups_after_first_build, cycle
        assert psm._attn_tp_group is first_attn_group, cycle

        # The rebuilt model is resident again: now the layout cross-checks
        # against the module instead of trusting the derivation.
        psm.model = _resident_mla_model(MLA_HEADS // G)
        layout_with_model = psm.prefill_pool_layout()
        assert layout_with_model == layout, cycle

        decode_graph = _attach_decode_phase_state(psm)

    assert layout_with_model is not None


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------

def test_a_changed_pool_layout_with_live_state_is_refused_by_name():
    """Honouring it would silently destroy decoded context."""
    psm = _psm()
    _build_state_pools(psm)
    manager = KimiLinearKDAWrapper.state_manager
    slots = {
        seq: KimiLinearKDAWrapper.slot_manager.alloc(seq)
        for seq in LIVE_SEQUENCES
    }
    addresses = _pool_addresses()

    psm._reset_model_attached_state()
    with pytest.raises(RuntimeError) as excinfo:
        _build_state_pools(psm, slots=POOL_SLOTS * 2)

    message = str(excinfo.value)
    assert "KDA state pools must be rebuilt with a different layout" in message
    assert f"{len(LIVE_SEQUENCES)} sequences still hold state" in message
    # The refusal leaves the live pools exactly as they were.
    assert KimiLinearKDAWrapper.state_manager is manager
    assert manager.sequence_state_items == slots
    assert _pool_addresses() == addresses


def test_a_changed_pool_layout_rebuilds_once_nothing_is_live():
    psm = _psm()
    _build_state_pools(psm)
    first = KimiLinearKDAWrapper.state_manager
    for seq in LIVE_SEQUENCES:
        KimiLinearKDAWrapper.slot_manager.alloc(seq)
    # The worker's completion/eviction hook, for real.
    KimiLinearKDAWrapper.free_sequences(LIVE_SEQUENCES)
    assert first.sequence_state_items == {}

    psm._reset_model_attached_state()
    _build_state_pools(psm, slots=POOL_SLOTS * 2)

    rebuilt = KimiLinearKDAWrapper.state_manager
    assert rebuilt is not first
    assert rebuilt.config.num_state_items == POOL_SLOTS * 2
    assert KimiLinearKDAWrapper.slot_manager.state_manager is rebuilt


def test_the_pool_layout_disagreeing_with_the_resident_module_is_refused():
    """The module is ground truth whenever it is resident."""
    psm = _psm()
    psm.model = _resident_mla_model(MLA_HEADS)  # unsharded: G was not applied

    with pytest.raises(RuntimeError) as excinfo:
        psm.prefill_pool_layout()

    message = str(excinfo.value)
    assert "disagrees with the resident MLA module" in message
    assert f"num_kv_heads module={MLA_HEADS} config={MLA_HEADS // G}" in message


def test_a_changed_attention_topology_is_refused_after_the_groups_exist(groups):
    psm = _psm()
    psm._build_attn_tp_group()
    built = len(groups)
    psm._attn_tp_size = 4  # in-process topology is immutable today

    with pytest.raises(RuntimeError) as excinfo:
        psm._build_attn_tp_group()

    message = str(excinfo.value)
    assert str((WORLD, G)) in message
    assert str((WORLD, 4)) in message
    assert len(groups) == built


# --------------------------------------------------------------------------
# single-shard path: every seam above must still be a no-op at G == 1
# --------------------------------------------------------------------------

def test_the_single_shard_path_builds_no_groups_and_keeps_full_heads(groups):
    psm = _psm(attention_group_size=1)
    _build_state_pools(psm)
    psm._build_attn_tp_group()

    assert groups == []
    assert psm._attn_tp_groups_built is False
    assert psm._attn_tp_group_signature is None
    assert psm.prefill_pool_layout()["num_kv_heads"] == MLA_HEADS
    assert KimiLinearKDAWrapper.state_manager.config.num_heads == KDA_HEADS

    psm._reset_model_attached_state()
    _build_state_pools(psm)
    assert groups == []
