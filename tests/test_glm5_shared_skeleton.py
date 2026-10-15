"""GLM-5 PSM: the kept prefill and decode instances share ONE device skeleton.

With the default phase-switch lifecycle both phase instances are kept for the
whole process. Each used to ``model.to(device)`` its own copy of the skeleton
(norms, router gates, indexer, dense MLP, embeddings, LM head), so the first
decode configure — the single point where the GPU KV pool is sized — found
the kept prefill instance's private copy still resident and sized the pool
that much smaller. ``share_skeleton_on_device()`` (the persistent path's
mechanism) moves the skeleton to the device once, and both instances bind it.

Sharing is only sound if neither phase's own weight loads ever write into a
shared tensor. These tests run the REAL ``configure_prefill`` and
``configure_decoding`` (name-map construction, skeleton binding, the decode
attention / shared-expert loads) against a small GLM-5-shaped model and pin:

* the bound skeleton key set is disjoint from both phases' phase-managed keys,
  and both phases bind the identical key -> tensor alias set;
* after both configures, every skeleton-bound parameter of BOTH instances is
  the one shared tensor, and its values are unchanged by the decode loads;
* phase-managed parameters never alias a skeleton entry;
* the runtime checks fail loud on an overlap, a non-aliasing binding, a
  different key set, or a configure that rebinds a skeleton parameter;
* a rank that does not keep its instances never moves the skeleton.

The PSM module cannot be imported on a CPU-only machine (its model module needs
Triton), so methods are exec'd in isolation — the idiom of
``test_glm5_psm_switch_cost.py``. On CPU, ``tensor.to("cpu")`` returns the same
tensor, which is exactly the aliasing a device-resident skeleton gives.
"""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
PSM = ROOT / "batchgen" / "models" / "glm" / "glm5" / "Parallel_Strategy_Manager.py"
PSM_SRC = PSM.read_text()

_METHODS = [
    "_instances_are_reused",
    "_local_routed_experts_fully_resident",
    "_plan_local_expert_split",
    "configure_prefill",
    "configure_decoding",
    "share_skeleton_on_device",
    "_skeleton_bucket",
    "_check_shared_skeleton_binding",
    "_assert_skeleton_still_shared",
    "_extract_dequantize_scale",
    "_load_model_skeleton",
    "_load_attn_module",
    "_load_shared_expert_module",
]


def _psm_class():
    tree = ast.parse(PSM_SRC)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GLM5ParallelStrategyManager"
    )


class _TorchProxy:
    """Real torch, with the CUDA calls the configures make stubbed out."""

    cuda = SimpleNamespace(
        empty_cache=lambda: None,
        memory_allocated=lambda *a, **k: 0,
        current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
        is_available=lambda: False,
    )

    def __getattr__(self, name):
        return getattr(torch, name)


class _Logged:
    def __init__(self):
        self.lines = []

    def info(self, *args, **kwargs):
        self.lines.append(args[0] % args[1:] if len(args) > 1 else args[0])

    warning = info
    debug = info


class _Glm5MoEStub:
    @classmethod
    def reset_prefill_grouped_state(cls):
        pass


# ---------------------------------------------------------------------------
# A small GLM-5-shaped model: the parameter NAMES the real one has
# ---------------------------------------------------------------------------

HIDDEN = 8
LAYERS = 3
FIRST_K_DENSE = 1
NUM_EXPERTS = 4
WORLD = 2


def _lin():
    return nn.Linear(HIDDEN, HIDDEN, bias=False)


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN))


class _Indexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wk = _lin()
        self.wq_b = _lin()
        self.k_norm = _Norm()
        self.weights_proj = _lin()


class _Attn(nn.Module):
    def __init__(self, indexer):
        super().__init__()
        self.q_a_proj = _lin()
        self.q_b_proj = _lin()
        self.kv_a_proj_with_mqa = _lin()
        self.kv_b_proj = _lin()
        self.o_proj = _lin()
        self.q_a_layernorm = _Norm()
        self.kv_a_layernorm = _Norm()
        self.indexer = _Indexer() if indexer else None


class _Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _lin()
        self.up_proj = _lin()
        self.down_proj = _lin()


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(NUM_EXPERTS, HIDDEN))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(NUM_EXPERTS))


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()
        self.shared_experts = _Dense()
        self.experts = nn.ModuleList(_Dense() for _ in range(NUM_EXPERTS))


class _Layer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.self_attn = _Attn(indexer=idx % 2 == 0)
        self.mlp = _Dense() if idx < FIRST_K_DENSE else _MoE()
        self.input_layernorm = _Norm()
        self.post_attention_layernorm = _Norm()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, HIDDEN)
        self.layers = nn.ModuleList(_Layer(i) for i in range(LAYERS))
        self.norm = _Norm()


class _CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = _Inner()
        self.lm_head = _lin()


def _phase_managed(key):
    """Keys the phase loaders own (attention projections, experts)."""
    if ".self_attn." in key:
        name = key.split(".self_attn.", 1)[1]
        return not (
            name.startswith("indexer.")
            or name in ("q_a_layernorm.weight", "kv_a_layernorm.weight")
        )
    return ".mlp.shared_experts." in key or ".mlp.experts." in key


def _skeleton():
    """What the parameter server ships: every non-phase-managed tensor, the
    FP8 scales, and one entry no model parameter binds."""
    reference = _CausalLM(None)
    sd = {}
    for key, param in reference.named_parameters():
        if not _phase_managed(key):
            sd[key] = torch.randn_like(param)
    for layer in range(LAYERS):
        sd[f"model.layers.{layer}.self_attn.q_a_proj.weight_scale_inv"] = (
            torch.rand(2, 2)
        )
    sd["model.layers.3.eh_proj.weight"] = torch.randn(HIDDEN, HIDDEN)
    return sd


def _core_engine():
    def get_tensor(module_key):
        if module_key.startswith("attn_"):
            names = (
                "q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight",
                "kv_b_proj.weight", "o_proj.weight",
            )
        else:
            names = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
        return {name: torch.full((HIDDEN, HIDDEN), 7.0) for name in names}

    return SimpleNamespace(get_tensor=get_tensor)


def _psm(*, reuse=True, resident_split=None):
    log = _Logged()
    psm_ast = _psm_class()
    wanted = [
        copy.deepcopy(
            next(
                node
                for node in psm_ast.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
        )
        for name in _METHODS
    ]
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Psm", bases=[], keywords=[], body=wanted, decorator_list=[]
            )
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(PSM), "exec"), namespace)
    cls = namespace["Psm"]
    overrides = dict(
        torch=_TorchProxy(),
        logging=log,
        time=__import__("time"),
        os=os,
        Glm5MoE=_Glm5MoEStub,
        Glm5ForCausalLM=_CausalLM,
        _synchronize_prefill_preloads=lambda: None,
    )
    for name in _METHODS:
        member = getattr(cls, name)
        target = member.fget if isinstance(member, property) else member
        if isinstance(member, staticmethod):
            target = member.__func__
        target.__globals__.update(overrides)

    psm = cls()
    psm.NUM_TOTAL_EXPERTS = NUM_EXPERTS
    psm.FIRST_K_DENSE = FIRST_K_DENSE
    psm._SKELETON_KEY_REMAP = {}
    psm.rank = 0
    psm.global_rank = 0
    psm.world_size = WORLD
    psm.is_fp8_experts = False
    psm.persistent_phase_instances = False
    psm.reuse_phase_instances = reuse
    psm.prefill_instance = None
    psm.decode_instance = None
    psm._device_scales = None
    psm._skeleton_on_device = False
    psm._shared_skeleton_keys = None
    psm._skeleton_bound_params = []
    psm.skeleton_state_dict = _skeleton()
    psm.core_engine = _core_engine()
    psm.loaded_model_config = SimpleNamespace(model_type="glm5-test")
    psm.model_config = SimpleNamespace(
        num_hidden_layers=LAYERS, num_local_experts=NUM_EXPERTS
    )
    offloading = resident_split is not None
    psm.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch="cpu"),
        EP_Config=SimpleNamespace(
            enable_offloading=offloading,
            offloading_ratio=0.5 if offloading else 0.0,
            num_local_expert_per_layer=None,
        ),
    )
    # Collaborators outside the skeleton/weight-load path: GPU kernels,
    # wrappers, comm and buffer setup. They are not under test here.
    for name in (
        "_config_attn_module", "_config_expert_module", "_config_lm_head_hook",
        "_setup_fp8_scales", "_init_fused_kernels", "_configure_decode_moe",
        "_init_mode_decoding", "_init_decoding_padding_bsz",
        "_load_local_routed_experts_into_arena",
        "_load_local_routed_experts_prealloc", "_load_local_routed_experts",
    ):
        setattr(psm, name, lambda *a, **k: None)
    return psm, log


def _bound_names(model, name_map):
    return {
        key
        for key, _ in model.named_parameters()
        if key not in name_map
    }


def _configure_both():
    psm, log = _psm()
    psm.configure_prefill()
    prefill_map = dict(psm.state_dict_name_map)
    psm.configure_decoding(padding_bsz=8, comm=None)
    decode_map = dict(psm.state_dict_name_map)
    return psm, log, prefill_map, decode_map


# ---------------------------------------------------------------------------
# S1: disjointness and one alias set, through the real configures
# ---------------------------------------------------------------------------


def test_the_reused_instances_move_the_skeleton_to_the_device_once():
    psm, log, _, _ = _configure_both()

    assert psm._skeleton_on_device is True
    assert sum("[SKELETON] shared on device" in line for line in log.lines) == 1


def test_bound_skeleton_keys_are_disjoint_from_both_phases_managed_keys():
    psm, _, prefill_map, decode_map = _configure_both()
    prefill_model, _ = psm.prefill_instance
    decode_model, _ = psm.decode_instance

    bound_prefill = _bound_names(prefill_model, prefill_map)
    bound_decode = _bound_names(decode_model, decode_map)

    assert bound_prefill == bound_decode == set(psm._shared_skeleton_keys)
    assert not bound_prefill & set(prefill_map)
    assert not bound_prefill & set(decode_map)
    # Every phase-managed key of the reference model is in both maps, and
    # nothing the skeleton binds is phase-managed.
    assert {key for key in prefill_map} == {key for key in decode_map}
    assert not any(_phase_managed(key) for key in bound_prefill)


def test_both_instances_alias_the_one_shared_tensor_per_skeleton_key():
    psm, _, prefill_map, _ = _configure_both()
    prefill_params = dict(psm.prefill_instance[0].named_parameters())
    decode_params = dict(psm.decode_instance[0].named_parameters())

    for key in psm._shared_skeleton_keys:
        shared = psm.skeleton_state_dict[key]
        assert prefill_params[key].data_ptr() == shared.data_ptr(), key
        assert decode_params[key].data_ptr() == shared.data_ptr(), key


def test_decode_weight_loads_leave_the_shared_values_untouched():
    """The decode configure's own loads (attention projections, shared
    experts) run after the prefill instance is bound; none of them may write
    into a tensor the prefill instance reads."""
    psm, _ = _psm()
    psm.configure_prefill()
    snapshot = {
        key: psm.skeleton_state_dict[key].clone()
        for key in psm._shared_skeleton_keys
    }

    psm.configure_decoding(padding_bsz=8, comm=None)

    for key, before in snapshot.items():
        assert torch.equal(psm.skeleton_state_dict[key], before), key


def test_phase_managed_parameters_never_alias_a_skeleton_entry():
    psm, _, _, decode_map = _configure_both()
    skeleton_ptrs = {
        tensor.data_ptr() for tensor in psm.skeleton_state_dict.values()
    }
    for model in (psm.prefill_instance[0], psm.decode_instance[0]):
        for key, param in model.named_parameters():
            if key in decode_map and param.numel():
                assert param.data_ptr() not in skeleton_ptrs, key


def test_unbound_skeleton_entries_are_reported():
    _, log, _, _ = _configure_both()

    unbound = [line for line in log.lines if "unbound non-scale entries" in line]
    assert len(unbound) == 1
    assert "unbound non-scale entries on device: 1," in unbound[0]


def test_a_rank_that_keeps_no_instances_never_moves_the_skeleton():
    """A streaming rank rebuilds each phase; its skeleton stays on the host."""
    psm, log = _psm(resident_split=1)
    host = dict(psm.skeleton_state_dict)

    psm.configure_prefill()
    psm.configure_decoding(padding_bsz=8, comm=None)

    assert psm._skeleton_on_device is False
    assert all(psm.skeleton_state_dict[k] is v for k, v in host.items())
    assert not any("[SKELETON] shared" in line for line in log.lines)


# ---------------------------------------------------------------------------
# The runtime checks fail loud
# ---------------------------------------------------------------------------


def test_sharing_after_the_scales_were_uploaded_fails_loud():
    psm, _ = _psm()
    psm._device_scales = {}

    with pytest.raises(RuntimeError, match="after the FP8 scales were uploaded"):
        psm.share_skeleton_on_device()


def test_sharing_twice_is_a_no_op():
    psm, log = _psm()
    psm.share_skeleton_on_device()
    first = dict(psm.skeleton_state_dict)

    psm.share_skeleton_on_device()

    assert all(psm.skeleton_state_dict[k] is v for k, v in first.items())
    assert sum("shared on device" in line for line in log.lines) == 1


def test_upload_log_breaks_the_bytes_down_by_bucket():
    psm, log = _psm()
    psm.share_skeleton_on_device()

    line = next(line for line in log.lines if "shared on device" in line)
    expected = len(psm.skeleton_state_dict)
    assert f": {expected} tensors," in line
    for bucket in ("global=", "layer_norm=", "indexer=", "scale=", "other="):
        assert bucket in line


def test_a_phase_managed_key_in_the_binding_fails_loud():
    psm, _ = _psm()
    psm.share_skeleton_on_device()
    key = "model.layers.0.self_attn.q_a_layernorm.weight"
    param = nn.Parameter(psm.skeleton_state_dict[key])
    psm.state_dict_name_map = {key: {}}
    psm.loaded_model_config.phase = "decode"

    with pytest.raises(RuntimeError, match="phase-managed keys"):
        psm._check_shared_skeleton_binding({key: (key, param)})


def test_a_binding_that_does_not_alias_the_shared_tensor_fails_loud():
    psm, _ = _psm()
    psm.share_skeleton_on_device()
    key = "model.norm.weight"
    private = nn.Parameter(psm.skeleton_state_dict[key].clone())
    psm.state_dict_name_map = {}
    psm.loaded_model_config.phase = "decode"

    with pytest.raises(RuntimeError, match="does not alias"):
        psm._check_shared_skeleton_binding({key: (key, private)})


def test_a_phase_binding_a_different_key_set_fails_loud():
    psm, _ = _psm()
    psm.share_skeleton_on_device()
    psm.state_dict_name_map = {}
    psm.loaded_model_config.phase = "prefill"
    norm = "model.norm.weight"
    embed = "model.embed_tokens.weight"
    psm._check_shared_skeleton_binding(
        {norm: (norm, nn.Parameter(psm.skeleton_state_dict[norm]))}
    )
    psm.loaded_model_config.phase = "decode"

    with pytest.raises(RuntimeError, match="different shared skeleton key set"):
        psm._check_shared_skeleton_binding(
            {
                norm: (norm, nn.Parameter(psm.skeleton_state_dict[norm])),
                embed: (embed, nn.Parameter(psm.skeleton_state_dict[embed])),
            }
        )


def test_a_configure_that_rebinds_a_skeleton_parameter_fails_loud():
    """A cast or wrapper that gives one instance a private copy again."""
    psm, _ = _psm()
    psm.configure_prefill()
    param, ckpt_key = psm._skeleton_bound_params[0]
    param.data = param.data.clone()

    with pytest.raises(RuntimeError, match="off the shared device copy"):
        psm._assert_skeleton_still_shared("prefill")


def test_configures_share_before_the_first_scale_lookup_and_bind():
    """Order inside both configures: share -> scales -> skeleton bind."""
    psm_ast = _psm_class()
    for name in ("configure_prefill", "configure_decoding"):
        method = next(
            node
            for node in psm_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        calls = [
            node.func.attr
            for node in sorted(
                (
                    node
                    for node in ast.walk(method)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                ),
                key=lambda node: (node.lineno, node.col_offset),
            )
        ]
        order = [
            c
            for c in calls
            if c in (
                "share_skeleton_on_device",
                "_extract_dequantize_scale",
                "_load_model_skeleton",
                "_assert_skeleton_still_shared",
            )
        ]
        assert order == [
            "share_skeleton_on_device",
            "_extract_dequantize_scale",
            "_load_model_skeleton",
            "_assert_skeleton_still_shared",
        ], (name, order)
