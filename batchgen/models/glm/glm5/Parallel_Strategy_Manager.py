# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 Parallel Strategy Manager for BatchGen.

Standalone — no cross-model imports.

Key differences from DeepSeek V3 PSM:
- 78 layers (vs 61), hidden_size=6144 (vs 7168)
- Indexer params (self_attn.indexer.*) excluded from state_dict_name_map → skeleton
- kv_a_proj_with_mqa naming (same as DeepSeek)
- DSA dual KV cache handled automatically by batchgen_worker.py
- MoE gate has e_score_correction_bias
- Dense MLP layers 0-2, MoE layers 3-77
- Prefill: all modules offloaded (DP), Decode: EP
"""

import gc
import logging
import os
import time
import types

import torch
import torch.distributed as dist

from .model import Glm5ForCausalLM, Glm5MoE
from .wrappers import GLM5ExpertWrapper, GLM5AttnWrapper


def _attn_of(model, layer_idx):
    """Return the raw attention module, unwrapping if wrapped."""
    a = model.model.layers[layer_idx].self_attn
    return a.module if hasattr(a, "module") else a




class GLM5ParallelStrategyManager:
    NUM_TOTAL_EXPERTS = 256
    NUM_LAYERS = 78
    FIRST_K_DENSE = 3
    HIDDEN_SIZE = 6144

    def __init__(
        self,
        loaded_model_config,
        engine_config,
        model_config,
        core_engine,
        skeleton_state_dict,
        local_rank,
        global_rank,
        world_size,
    ):
        self.loaded_model_config = loaded_model_config
        self.engine_config = engine_config
        self.model_config = model_config
        self.core_engine = core_engine
        self.skeleton_state_dict = skeleton_state_dict
        self.weight_copy_task = {}
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.rank = global_rank
        # Detect FP8 variant by checking for expert scale tensors in skeleton
        self.is_fp8_experts = any(
            "experts.0.gate_proj.weight_scale_inv" in k for k in skeleton_state_dict
        )

    def _dump_weight_stats_once(self):
        """Rank-0 one-shot log of loaded norm/gate stats for sanity-check vs disk."""
        if self.rank != 0:
            return

        def _attn(layer_idx):
            a = self.model.model.layers[layer_idx].self_attn
            return a.module if hasattr(a, "module") else a

        targets = [
            ("L0.input_layernorm.weight", lambda: self.model.model.layers[0].input_layernorm.weight),
            ("L0.post_attn_layernorm.weight", lambda: self.model.model.layers[0].post_attention_layernorm.weight),
            ("L0.q_a_layernorm.weight", lambda: _attn(0).q_a_layernorm.weight),
            ("L0.kv_a_layernorm.weight", lambda: _attn(0).kv_a_layernorm.weight),
            ("L3.mlp.gate.weight", lambda: self.model.model.layers[3].mlp.gate.weight),
            ("L3.mlp.gate.e_score_correction_bias",
                lambda: self.model.model.layers[3].mlp.gate.e_score_correction_bias),
            ("L3.indexer.k_norm.weight",
                lambda: _attn(3).indexer.k_norm.weight),
            ("L3.indexer.k_norm.bias",
                lambda: _attn(3).indexer.k_norm.bias),
            ("model.norm.weight", lambda: self.model.model.norm.weight),
        ]
        logging.info("[GLM5 WEIGHT-VERIFY] dumping loaded stats (compare to disk)")
        dense_mla = not hasattr(_attn(3), "indexer")
        for name, getter in targets:
            try:
                t = getter()
            except AttributeError as e:
                if "indexer" in name and dense_mla:
                    logging.info(f"[GLM5 WEIGHT-VERIFY] {name}: skipped (dense-MLA mode)")
                else:
                    logging.warning(f"[GLM5 WEIGHT-VERIFY] {name}: skipped — {e}")
                continue
            f32 = t.float()
            first = [f"{x:.6g}" for x in t.flatten()[:5].tolist()]
            logging.info(
                f"[GLM5 WEIGHT-VERIFY] {name}: dtype={t.dtype} shape={tuple(t.shape)} "
                f"min={f32.min().item():.6g} max={f32.max().item():.6g} "
                f"mean={f32.mean().item():.6g} std={f32.std().item():.6g} "
                f"first5={first}"
            )

    def _verify_all_weights_loaded(self):
        """Comprehensive weight-load audit. Always runs on rank 0; assert only
        when BATCHGEN_GLM5_WEIGHT_ASSERT=1.

        Two checks in one pass:
          1) Enumerate every expected tensor key, confirm presence in
             `skeleton_state_dict`. Report MISSING keys grouped by type.
          2) For every expected module weight (post-load), read the live
             `param.data` and assert it's not at its `nn.Module` default
             (zeros / N(0, 0.02) / copy of a template row). Heuristic:
             norms/biases should NOT be uniform-zero and NOT uniform-one
             without the trained structure. FP8 weights should have a wide
             value range consistent with quantized trained weights.

        If BATCHGEN_GLM5_WEIGHT_ASSERT=1, raises on the first violation.
        Otherwise logs a report and continues.
        """
        import os as _os
        if self.rank != 0:
            return
        mode = _os.environ.get("BATCHGEN_GLM5_WEIGHT_ASSERT", "").strip()
        fail_fast = mode == "1"
        cfg = self.loaded_model_config
        num_layers = cfg.num_hidden_layers
        first_k_dense = cfg.first_k_dense_replace
        n_experts = cfg.n_routed_experts
        has_indexer = hasattr(_attn_of(self.model, 3), "indexer")

        # ==== 1. Skeleton coverage ====
        skel = self.skeleton_state_dict
        present = set(skel.keys())
        expected = set()

        # Global
        expected.update([
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ])
        # Per-layer
        for i in range(num_layers):
            # Layer norms
            expected.add(f"model.layers.{i}.input_layernorm.weight")
            expected.add(f"model.layers.{i}.post_attention_layernorm.weight")
            # Attn
            expected.add(f"model.layers.{i}.self_attn.q_a_proj.weight")
            expected.add(f"model.layers.{i}.self_attn.q_a_proj.weight_scale_inv")
            expected.add(f"model.layers.{i}.self_attn.q_a_layernorm.weight")
            expected.add(f"model.layers.{i}.self_attn.q_b_proj.weight")
            expected.add(f"model.layers.{i}.self_attn.q_b_proj.weight_scale_inv")
            expected.add(f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight")
            expected.add(f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight_scale_inv")
            expected.add(f"model.layers.{i}.self_attn.kv_a_layernorm.weight")
            expected.add(f"model.layers.{i}.self_attn.kv_b_proj.weight")
            expected.add(f"model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv")
            expected.add(f"model.layers.{i}.self_attn.o_proj.weight")
            expected.add(f"model.layers.{i}.self_attn.o_proj.weight_scale_inv")
            # Indexer
            if has_indexer:
                for t in [
                    "indexer.wq_b.weight", "indexer.wq_b.weight_scale_inv",
                    "indexer.wk.weight", "indexer.wk.weight_scale_inv",
                    "indexer.weights_proj.weight",
                    "indexer.k_norm.weight", "indexer.k_norm.bias",
                ]:
                    expected.add(f"model.layers.{i}.self_attn.{t}")
            # MLP
            if i < first_k_dense:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    expected.add(f"model.layers.{i}.mlp.{proj}.weight")
                    expected.add(f"model.layers.{i}.mlp.{proj}.weight_scale_inv")
            else:
                expected.add(f"model.layers.{i}.mlp.gate.weight")
                expected.add(f"model.layers.{i}.mlp.gate.e_score_correction_bias")
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    expected.add(f"model.layers.{i}.mlp.shared_experts.{proj}.weight")
                    expected.add(f"model.layers.{i}.mlp.shared_experts.{proj}.weight_scale_inv")
                for j in range(n_experts):
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        expected.add(f"model.layers.{i}.mlp.experts.{j}.{proj}.weight")
                        expected.add(f"model.layers.{i}.mlp.experts.{j}.{proj}.weight_scale_inv")

        missing = sorted(expected - present)
        extra = sorted(present - expected)

        def _bucket(key):
            if "input_layernorm" in key or "post_attention_layernorm" in key:
                return "layer_norm"
            if "q_a_layernorm" in key or "kv_a_layernorm" in key:
                return "attn_norm"
            if "indexer" in key:
                return "indexer"
            if ".experts." in key:
                return "routed_expert"
            if "shared_experts" in key:
                return "shared_expert"
            if ".gate." in key or ".gate_proj" in key or ".up_proj" in key or ".down_proj" in key:
                return "mlp_or_gate"
            if "embed_tokens" in key or "lm_head" in key or "model.norm" in key:
                return "global"
            if "self_attn" in key:
                return "attn"
            return "other"

        from collections import Counter
        miss_bucket = Counter(_bucket(k) for k in missing)
        logging.warning(
            f"[GLM5 WEIGHT-VERIFY] expected={len(expected)} present={len(present)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
        if missing:
            logging.warning(f"[GLM5 WEIGHT-VERIFY] missing by bucket: {dict(miss_bucket)}")
            # Print up to 30 example missing keys
            for k in missing[:30]:
                logging.warning(f"[GLM5 WEIGHT-VERIFY] MISSING: {k}")
            if len(missing) > 30:
                logging.warning(f"[GLM5 WEIGHT-VERIFY]  ... +{len(missing) - 30} more missing")

        # ==== 2. Module live-weight audit ====
        violations = []

        def _check(name, tensor, kind):
            if tensor is None:
                violations.append((name, f"{kind} is None"))
                return
            f32 = tensor.detach().float()
            abs_mean = f32.abs().mean().item()
            stdev = f32.std().item()
            # Any tensor should have nonzero absmean (unless deliberately zero bias)
            if abs_mean < 1e-9:
                violations.append((name, f"{kind} all-zero (abs_mean={abs_mean:.2e})"))
                return
            # Default-init detection for RMSNorm.weight: `Glm5RMSNorm.__init__`
            # uses `torch.ones(hidden_size)` → tensor is uniform 1.0 with
            # stdev ≈ 0. Only flag if uniform — GLM-5's actual trained norm
            # weights have small abs_mean (0.02-0.2) with stdev > 1e-4, which
            # must NOT be mistaken for un-loaded init.
            if kind == "norm_weight" and stdev < 1e-6 and abs(abs_mean - 1.0) < 1e-4:
                violations.append((name, f"{kind} is uniform-1.0 (un-loaded ones_init) "
                                          f"(abs_mean={abs_mean:.6f} std={stdev:.2e})"))
            # N(0, 0.02) init for Linear.weight — absmean ≈ 0.016, std ≈ 0.02.
            if kind in ("gate_weight", "embed") and abs_mean < 1e-6:
                violations.append((name, f"{kind} all-zero "
                                          f"(abs_mean={abs_mean:.2e})"))
            # Bias correctness is too model-specific for a heuristic — GLM-5
            # gate e_score_correction_bias has abs_mean ≈ 6-35 with stdev
            # ≈ 0.1-0.2 (verified on disk). Only flag all-zero.

        # Walk all layers and spot-check loaded-ness
        for i in range(num_layers):
            L = self.model.model.layers[i]
            _check(f"L{i}.input_layernorm.weight", L.input_layernorm.weight, "norm_weight")
            _check(f"L{i}.post_attn_layernorm.weight", L.post_attention_layernorm.weight, "norm_weight")
            attn = L.self_attn.module if hasattr(L.self_attn, "module") else L.self_attn
            wrapper = L.self_attn if hasattr(L.self_attn, "module") else None
            _check(f"L{i}.q_a_layernorm.weight", attn.q_a_layernorm.weight, "norm_weight")
            _check(f"L{i}.kv_a_layernorm.weight", attn.kv_a_layernorm.weight, "norm_weight")
            # Attn projection FP8 weights: read from the wrapper's captured
            # references (`self.fp8_q_a_proj` etc.) set by `_register_fp8_weights`
            # — the underlying `attn.q_a_proj.weight.data` is reset/cleared
            # after capture so reading it would misleadingly report all-zero.
            _wrapper_attrs = {
                "q_a_proj":            "fp8_q_a_proj",
                "q_b_proj":            "fp8_q_b_proj",
                "kv_a_proj_with_mqa":  "fp8_kv_a_proj",
                "kv_b_proj":           "fp8_kv_b_proj",
                "o_proj":              "fp8_o_proj",
            }
            for _proj, _wattr in _wrapper_attrs.items():
                _t = getattr(wrapper, _wattr, None) if wrapper is not None else None
                if _t is None:
                    # Wrapper not populated yet (e.g. prefill phase) — fall
                    # back to the live module weight.
                    _mod = getattr(attn, _proj, None)
                    if _mod is not None and getattr(_mod, "weight", None) is not None:
                        _t = _mod.weight.data
                if _t is not None:
                    _check(f"L{i}.self_attn.{_proj}.weight", _t, "fp8_weight")
            # Indexer
            if has_indexer and hasattr(attn, "indexer"):
                _check(f"L{i}.indexer.k_norm.weight", attn.indexer.k_norm.weight, "norm_weight")
                _check(f"L{i}.indexer.k_norm.bias", attn.indexer.k_norm.bias, "norm_bias")
            # Dense MLP (0..first_k_dense)
            if i < first_k_dense:
                mlp = L.mlp
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    mod = getattr(mlp, proj)
                    _check(f"L{i}.mlp.{proj}.weight", mod.weight.data, "fp8_weight")
                    scale = getattr(mod, "weight_scale_inv", None)
                    if scale is None:
                        # Scale may live as side-attr after _setup_fp8_scales
                        scale = getattr(mlp, f"{proj.split('_')[0]}_scale", None)
                    _check(f"L{i}.mlp.{proj}.weight_scale_inv", scale, "fp32_scale")
            else:
                # MoE layer: gate + bias
                gate = L.mlp.gate
                _check(f"L{i}.mlp.gate.weight", gate.weight, "gate_weight")
                _check(f"L{i}.mlp.gate.e_score_correction_bias",
                       gate.e_score_correction_bias, "norm_bias")
                # Explicit per-layer bias stats — POIS flagged L18 specifically.
                # Zero-everywhere here means the disk key was missed and routing
                # silently degrades to raw-sigmoid top-K (no bias correction).
                if i in (first_k_dense, 18, 39, num_layers - 1):
                    _b = gate.e_score_correction_bias.detach().float()
                    logging.warning(
                        f"[GLM5 WEIGHT-VERIFY] L{i}.mlp.gate.e_score_correction_bias "
                        f"abs_mean={_b.abs().mean().item():.6f} "
                        f"min={_b.min().item():.4f} max={_b.max().item():.4f} "
                        f"nonzero_frac={(_b.abs() > 1e-9).float().mean().item():.3f} "
                        f"first5={[round(x, 4) for x in _b[:5].tolist()]}"
                    )

        # Global
        _check("model.norm.weight", self.model.model.norm.weight, "norm_weight")
        _check("model.embed_tokens.weight", self.model.model.embed_tokens.weight, "embed")
        _check("lm_head.weight", self.model.lm_head.weight, "embed")

        if violations:
            logging.warning(f"[GLM5 WEIGHT-VERIFY] {len(violations)} live-weight violation(s):")
            for name, reason in violations[:50]:
                logging.warning(f"[GLM5 WEIGHT-VERIFY] LIVE-VIOLATION: {name}: {reason}")
            if len(violations) > 50:
                logging.warning(f"[GLM5 WEIGHT-VERIFY]  ... +{len(violations) - 50} more")

        if fail_fast and (missing or violations):
            raise RuntimeError(
                f"BATCHGEN_GLM5_WEIGHT_ASSERT=1 tripped: "
                f"{len(missing)} skeleton-missing + {len(violations)} live-violations. "
                f"See [GLM5 WEIGHT-VERIFY] log lines."
            )
        logging.warning("[GLM5 WEIGHT-VERIFY] audit complete")

    def configure_prefill(self):
        """Configure model for prefill (pure DP, all modules offloaded)."""
        start_time = time.perf_counter()
        timings = {}

        self.loaded_model_config.phase = "prefill"

        step_start = time.perf_counter()
        self.model = Glm5ForCausalLM(self.loaded_model_config)
        timings['model_init'] = time.perf_counter() - step_start

        self.state_dict_name_map = {}
        self.weight_copy_task = {
            "attn": [],
            "routed_expert": [],
            "shared_expert": [],
        }

        step_start = time.perf_counter()
        for layer_idx in range(self.model_config.num_hidden_layers):
            # Attention (EXCLUDING indexer → skeleton; EXCLUDING q_a_layernorm /
            # kv_a_layernorm → skeleton too. Both are tiny BF16 RMSNorm weights
            # that don't need the copy-task machinery, and routing them through
            # state_dict_name_map makes _load_model_skeleton skip them, leaving
            # the live module at its ones_() init → silently-wrong Q/K RMSNorm.)
            for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
                if name.startswith("indexer."):
                    continue
                if name in ("q_a_layernorm.weight", "kv_a_layernorm.weight"):
                    continue
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            if layer_idx >= self.FIRST_K_DENSE:
                # Shared experts — use static param names (no nn.Module traversal)
                _shared_expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for name in _shared_expert_param_names:
                    tensor_full_name = f"model.layers.{layer_idx}.mlp.shared_experts.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(f"shared_expert_{layer_idx}")

                # Routed experts — use static param names (experts are placeholders)
                _expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for expert_idx in range(self.NUM_TOTAL_EXPERTS):
                    module_key = f"routed_expert_{layer_idx}_{expert_idx}"
                    for name in _expert_param_names:
                        tensor_full_name = (
                            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": module_key,
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(module_key)
        timings['weight_mappings'] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._extract_dequantize_scale()
        timings['dequantize'] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._load_model_skeleton()
        timings['skeleton'] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._config_attn_module()
        timings['attn'] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._config_expert_module()
        timings['expert'] = time.perf_counter() - step_start

        self._config_lm_head_hook()
        self.model.eval()

        step_start = time.perf_counter()
        self.model.to(self.engine_config.Basic_Config.device_torch)
        self._setup_fp8_scales()
        self._init_fused_kernels()
        self._dump_weight_stats_once()
        self._verify_all_weights_loaded()
        timings['to_device'] = time.perf_counter() - step_start

        total_time = time.perf_counter() - start_time
        if self.rank == 0:
            logging.info(
                f"[PREFILL] Model configured in {total_time:.2f}s "
                f"(init={timings['model_init']:.1f}s, skeleton={timings['skeleton']:.1f}s, "
                f"expert={timings['expert']:.1f}s, to_device={timings['to_device']:.1f}s)"
            )
        return self.model, self.weight_copy_task

    def configure_decoding(self, padding_bsz=None, comm=None):
        """Configure model for decode: DP + EP."""
        self.loaded_model_config.phase = "decode"
        self.loaded_model_config._attn_implementation = "eager"
        self.model = None
        torch.cuda.empty_cache()

        self.model = Glm5ForCausalLM(self.loaded_model_config)

        self.weight_copy_task = {
            "attn": [],
            "routed_expert": [],
            "shared_expert": [],
        }
        self.state_dict_name_map = {}
        self.local_routed_experts = []
        self.host_routed_experts = []

        NUM_EXPERT_PER_RANK = self.NUM_TOTAL_EXPERTS // self.world_size

        if self.world_size > 8:
            self.enable_ep_offloading = False
            NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
        elif self.engine_config.EP_Config.enable_offloading:
            self.enable_ep_offloading = True
            offload_ratio = self.engine_config.EP_Config.offloading_ratio
            NUM_LOCAL_EXPERT_PER_LAYER = int(NUM_EXPERT_PER_RANK * (1 - offload_ratio))
        else:
            self.enable_ep_offloading = False
            NUM_LOCAL_EXPERT_PER_LAYER = self.engine_config.EP_Config.num_local_expert_per_layer
            if NUM_LOCAL_EXPERT_PER_LAYER is None or NUM_LOCAL_EXPERT_PER_LAYER == 0:
                NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK

        self.num_local_expert_per_layer = NUM_LOCAL_EXPERT_PER_LAYER

        routed_expert_gpu_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
        routed_expert_gpu_end_idx = routed_expert_gpu_start_idx + NUM_LOCAL_EXPERT_PER_LAYER
        routed_expert_host_end_idx = (self.global_rank + 1) * NUM_EXPERT_PER_RANK

        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            for expert_idx in range(routed_expert_gpu_start_idx, routed_expert_gpu_end_idx):
                self.local_routed_experts.append(f"routed_expert_{layer_idx}_{expert_idx}")
            for expert_idx in range(routed_expert_gpu_end_idx, routed_expert_host_end_idx):
                self.host_routed_experts.append(f"routed_expert_{layer_idx}_{expert_idx}")

        self.weight_copy_task["routed_expert"] = self.host_routed_experts

        # Build state_dict_name_map for all modules (skip indexer and
        # q_a/kv_a_layernorm — those route through skeleton, see
        # configure_prefill for rationale).
        for layer_idx in range(self.model_config.num_hidden_layers):
            for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
                if name.startswith("indexer."):
                    continue
                if name in ("q_a_layernorm.weight", "kv_a_layernorm.weight"):
                    continue
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }

            if layer_idx >= self.FIRST_K_DENSE:
                _shared_expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for name in _shared_expert_param_names:
                    tensor_full_name = f"model.layers.{layer_idx}.mlp.shared_experts.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }

                _expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for expert_idx in range(self.model_config.num_local_experts):
                    module_key = f"routed_expert_{layer_idx}_{expert_idx}"
                    for name in _expert_param_names:
                        tensor_full_name = (
                            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": module_key,
                            "tensor_key": name,
                        }

        torch.cuda.empty_cache()
        self._extract_dequantize_scale()
        self._load_model_skeleton()
        self._load_local_routed_experts()
        self._load_attn_module()
        self._load_shared_expert_module()
        self._config_attn_module()
        self._config_expert_module()
        self._configure_decode_moe(comm)
        self._config_lm_head_hook()

        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)
        self._setup_fp8_scales()
        self._init_fused_kernels()

        if self.rank == 0:
            used = torch.cuda.memory_allocated(self.engine_config.Basic_Config.device_torch)
            logging.info(f"[MODEL] GPU memory after init: {used / (1024**3):.2f} GB used")

        self._verify_all_weights_loaded()
        self._init_mode_decoding()
        effective_bsz = padding_bsz if padding_bsz is not None else 128
        self._init_decoding_padding_bsz(effective_bsz)

        if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "1":
            self._init_ata_comms(effective_bsz)

        return self.model, self.weight_copy_task

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "set_num_tokens_per_rank"):
                layer.set_num_tokens_per_rank(num_tokens_per_rank)

    def set_rank_token_counts(self, counts: torch.Tensor):
        """Store per-rank real-token counts [world_size] on GPU for 3D-MoE padding masking.

        Mirrors KimiK25ParallelStrategyManager.set_rank_token_counts
        (moonshotai/kimi_k25/Parallel_Strategy_Manager.py:590). Worker
        (batchgen_worker.py ~line 7805) calls this each decode iter when the
        active batch has mixed real/padded rows, so the 3D-MoE path can
        mask topk_idx for padded positions before dispatch_scatter_3d.
        """
        from .model import Glm5MoE
        Glm5MoE._rank_token_counts = counts

    def _init_decoding_padding_bsz(self, padding_bsz):
        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        max_rank_bsz = int(env_max_bsz) if env_max_bsz else padding_bsz
        if self.rank == 0:
            logging.info(f"[DECODE] Padding batch size: {max_rank_bsz}")

        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init_num_tokens"):
                layer.init_num_tokens(max_rank_bsz)

        # Initialize shared buffer manager (pre-allocated comm buffers for all MoE layers)
        device = self.engine_config.Basic_Config.device_torch
        Glm5MoE.init_buffer_manager(
            max_rank_bsz, self.world_size, self.HIDDEN_SIZE, device,
        )

    def _init_mode_decoding(self):
        has_persistent = self.num_local_expert_per_layer > 0
        if not has_persistent:
            if self.rank == 0:
                logging.info("EP offloading: no persistent experts, skipping grouped GEMM init")
            return
        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init"):
                layer.init(self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size)

    def _init_ata_comms(self, padding_bsz):
        """Initialize All-to-All communications for EP."""
        try:
            from pplx_kernels.all_to_all import AllToAll
        except ImportError:
            logging.warning("pplx_kernels not available, skipping ATA init")
            return

        in_type = torch.float8_e4m3fn
        out_type = torch.bfloat16
        dp_size = 1
        hidden_size = self.HIDDEN_SIZE
        block_size = 128
        device = self.engine_config.Basic_Config.device_torch

        experts_per_rank = self.NUM_TOTAL_EXPERTS // self.world_size
        num_experts_per_tok = 8
        num_dp = self.world_size // dp_size

        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        max_rank_bsz = int(env_max_bsz) if env_max_bsz else padding_bsz

        self.expert_num_tokens = torch.empty(experts_per_rank, dtype=torch.int32, device=device)
        self.expert_x = torch.empty(
            (experts_per_rank, max_rank_bsz * num_dp, hidden_size), dtype=in_type, device=device
        )
        self.expert_x_scale = torch.empty(
            (experts_per_rank, self.expert_x.size(1), (hidden_size + block_size - 1) // block_size),
            dtype=torch.float32, device=device
        )
        self.expert_y = torch.empty_like(self.expert_x, dtype=out_type)
        self.indices = torch.empty(
            (max_rank_bsz, num_experts_per_tok), dtype=torch.uint32, device=device
        )
        self.weights = torch.empty(
            (max_rank_bsz, num_experts_per_tok), dtype=torch.float32, device=device
        )
        self.y = torch.empty((max_rank_bsz, hidden_size), dtype=out_type, device=device)
        self.dp_x = torch.empty((max_rank_bsz, hidden_size), dtype=in_type, device=device)
        self.dp_x_scale = torch.empty(
            (max_rank_bsz, (hidden_size + block_size - 1) // block_size),
            dtype=torch.float32, device=device
        )

        if self.world_size <= 8:
            ata = AllToAll.intranode(
                max_num_tokens=max_rank_bsz, num_experts=self.NUM_TOTAL_EXPERTS,
                experts_per_token=num_experts_per_tok, rank=self.rank,
                world_size=self.world_size, dp_size=dp_size, hidden_dim=hidden_size,
                hidden_dim_bytes=hidden_size * in_type.itemsize,
                hidden_dim_scale_bytes=(hidden_size + block_size - 1) // block_size * 4,
            )
        else:
            ata = AllToAll.internode(
                max_num_tokens=max_rank_bsz, num_experts=self.NUM_TOTAL_EXPERTS,
                experts_per_token=num_experts_per_tok, rank=self.rank,
                world_size=self.world_size, dp_size=dp_size, hidden_dim=hidden_size,
                hidden_dim_bytes=hidden_size * in_type.itemsize,
                hidden_dim_scale_bytes=(hidden_size + block_size - 1) // block_size * 4,
            )

        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init_ata_comm"):
                layer.init_ata_comm(
                    padding_bsz, self.expert_num_tokens, self.expert_x,
                    self.expert_x_scale, self.expert_y, self.indices,
                    self.weights, self.y, self.dp_x, self.dp_x_scale, ata,
                )

    def _load_attn_module(self):
        """Load attention FP8 weights for decode (persistent on GPU).

        q_a_layernorm / kv_a_layernorm now route through the skeleton path at
        model init (see glm5_parameter_server.py), so this function only moves
        the 5 FP8 projections onto device.
        """
        device = self.engine_config.Basic_Config.device_torch
        for layer_idx in range(len(self.model.model.layers)):
            attn = self.model.model.layers[layer_idx].self_attn
            tensors = self.core_engine.get_tensor(f"attn_{layer_idx}")
            attn.q_a_proj.weight.data = tensors["q_a_proj.weight"].to(device)
            attn.q_b_proj.weight.data = tensors["q_b_proj.weight"].to(device)
            attn.kv_a_proj_with_mqa.weight.data = tensors["kv_a_proj_with_mqa.weight"].to(device)
            attn.kv_b_proj.weight.data = tensors["kv_b_proj.weight"].to(device)
            attn.o_proj.weight.data = tensors["o_proj.weight"].to(device)

    def _load_shared_expert_module(self):
        """Load shared expert FP8 weights for decode (persistent on GPU)."""
        device = self.engine_config.Basic_Config.device_torch
        for layer_idx in range(self.FIRST_K_DENSE, len(self.model.model.layers)):
            tensors = self.core_engine.get_tensor(f"shared_expert_{layer_idx}")
            shared = self.model.model.layers[layer_idx].mlp.shared_experts
            shared.gate_proj.weight.data = tensors["gate_proj.weight"].to(device)
            shared.up_proj.weight.data = tensors["up_proj.weight"].to(device)
            shared.down_proj.weight.data = tensors["down_proj.weight"].to(device)

    def _load_local_routed_experts(self):
        """Load persistent routed expert FP8 weights for decode.

        Stores weights as flat attributes on placeholder (following GPT-OSS pattern).
        GLM5ExpertWrapper._register_fp8_weights() reads these during _config_expert_module.
        """
        device = self.engine_config.Basic_Config.device_torch
        for routed_expert_idx in self.local_routed_experts:
            tensors = self.core_engine.get_tensor(routed_expert_idx)
            parts = routed_expert_idx.split("_")
            layer_idx = int(parts[2])
            expert_idx = int(parts[3])
            expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]
            # Store as flat attrs on placeholder (no nn.Module needed)
            expert.fp8_gate = tensors["gate_proj.weight"].to(device)
            expert.fp8_up = tensors["up_proj.weight"].to(device)
            expert.fp8_down = tensors["down_proj.weight"].to(device)
        logging.debug("Local routed experts loaded")

    def _config_attn_module(self):
        """Replace attention modules with GLM5AttnWrapper."""
        start_time = time.perf_counter()
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            if self.engine_config.Basic_Config.gpu_arch == "hopper":
                from batchgen.attention.mla.fa3_backend import (
                    mla_prefill_flashattention3,
                    mla_prefill_flashattention3_w8a16_deepgemm,
                    mla_prefill_flashattention3_prepacked,
                    mla_prefill_flashattention3_w8a16_deepgemm_prepacked,
                )
                from batchgen.attention.mla.flashmla_backend import (
                    mla_decoding_flashmla,
                    mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv,
                    mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn,
                    fused_get_query_states_triton,
                )
                setattr(attn_module, "prefill_attn", types.MethodType(
                    mla_prefill_flashattention3, attn_module))
                setattr(attn_module, "prefill_attn_w8a16", types.MethodType(
                    mla_prefill_flashattention3_w8a16_deepgemm, attn_module))
                setattr(attn_module, "prefill_attn_prepacked", types.MethodType(
                    mla_prefill_flashattention3_prepacked, attn_module))
                setattr(attn_module, "prefill_attn_w8a16_prepacked", types.MethodType(
                    mla_prefill_flashattention3_w8a16_deepgemm_prepacked, attn_module))
                setattr(attn_module, "decoding_attn", types.MethodType(
                    mla_decoding_flashmla, attn_module))
                setattr(attn_module, "decoding_attn_mode_3_bf16", types.MethodType(
                    mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv, attn_module))
                setattr(attn_module, "decoding_attn_mode_3_fp8", types.MethodType(
                    mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn, attn_module))
                setattr(attn_module, "fused_get_query_states_triton", types.MethodType(
                    fused_get_query_states_triton, attn_module))
            elif self.engine_config.Basic_Config.gpu_arch == "ampere":
                from batchgen.attention.mla.fa2_backend import mla_chunked_prefill_flashattention2
                from batchgen.attention.mla.torch_backend import mla_decoding_torch
                setattr(attn_module, "prefill_attn", types.MethodType(
                    mla_chunked_prefill_flashattention2, attn_module))
                setattr(attn_module, "decoding_attn", types.MethodType(
                    mla_decoding_torch, attn_module))
            else:
                raise ValueError(f"Unsupported GPU arch: {self.engine_config.Basic_Config.gpu_arch}")

            # Determine persistence
            persistent = f"attn_{layer_idx}" not in self.weight_copy_task.get("attn", [])

            # Extract FP8 dequant scales from skeleton
            weight_dequant_scales = {}
            prefix = f"model.layers.{layer_idx}.self_attn."
            postfix = ".weight_scale_inv"
            for name, param in self.skeleton_state_dict.items():
                if name.startswith(prefix) and name.endswith(postfix):
                    # Skip indexer scales
                    if ".indexer." in name:
                        continue
                    key = name[len(prefix):]
                    weight_dequant_scales[key] = param.to(
                        self.engine_config.Basic_Config.device_torch
                    )

            wrapper = GLM5AttnWrapper(
                attn_module, layer_idx, self.core_engine,
                self.engine_config, self.model_config,
                persistent, weight_dequant_scales,
            )
            self.model.model.layers[layer_idx].self_attn = wrapper
            if persistent:
                wrapper._register_fp8_weights()
                wrapper.initialize_decode_absorb()

        elapsed = time.perf_counter() - start_time
        logging.debug(f"Attn module config time: {elapsed:.2f}s")

    def _config_expert_module(self):
        """Replace expert modules with GLM5ExpertWrapper."""
        start_time = time.perf_counter()
        mlp_names = ["gate_proj", "up_proj", "down_proj"]
        postfix = ".weight_scale_inv"

        for layer_idx in range(self.FIRST_K_DENSE, len(self.model.model.layers)):
            layer = self.model.model.layers[layer_idx]

            # Shared expert
            shared_persistent = f"shared_expert_{layer_idx}" not in self.weight_copy_task.get("shared_expert", [])
            prefix = f"model.layers.{layer_idx}.mlp.shared_experts."
            shared_scales = {}
            for name in mlp_names:
                key = prefix + name + postfix
                if key in self.skeleton_state_dict:
                    shared_scales[name + postfix] = self.skeleton_state_dict[key].to(
                        self.engine_config.Basic_Config.device_torch
                    )
            layer.mlp.shared_experts = GLM5ExpertWrapper(
                layer.mlp.shared_experts, layer_idx, -1,
                self.core_engine, self.engine_config, self.model_config,
                shared_persistent, shared_scales, is_fp8=self.is_fp8_experts,
            )
            if shared_persistent:
                layer.mlp.shared_experts._register_fp8_weights()

            # Routed experts — wrap placeholders directly (no nn.Module needed)
            local_set = set(self.local_routed_experts) if hasattr(self, 'local_routed_experts') else set()
            for expert_idx in range(len(layer.mlp.experts)):
                routed_key = f"routed_expert_{layer_idx}_{expert_idx}"
                persistent = routed_key not in self.weight_copy_task.get("routed_expert", [])

                prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
                expert_scales = {}
                for name in mlp_names:
                    key = prefix + name + postfix
                    if key in self.skeleton_state_dict:
                        expert_scales[name + postfix] = self.skeleton_state_dict[key].to(
                            self.engine_config.Basic_Config.device_torch
                        )
                layer.mlp.experts[expert_idx] = GLM5ExpertWrapper(
                    layer.mlp.experts[expert_idx], layer_idx, expert_idx,
                    self.core_engine, self.engine_config, self.model_config,
                    persistent, expert_scales, is_fp8=self.is_fp8_experts,
                )
                # Only register weights for experts that had weights loaded
                if persistent and routed_key in local_set:
                    layer.mlp.experts[expert_idx]._register_fp8_weights()

        elapsed = time.perf_counter() - start_time
        logging.debug(f"Expert module config time: {elapsed:.2f}s")

    def _configure_decode_moe(self, comm):
        """Configure existing Glm5MoE instances for EP decode.

        No class swap — same Glm5MoE is used for both prefill and
        decode. Just inject comm and set EP attributes.
        """
        NUM_EXPERT_PER_RANK = self.NUM_TOTAL_EXPERTS // self.world_size

        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            moe = self.model.model.layers[layer_idx].mlp
            moe.comm = comm
            moe.device = self.engine_config.Basic_Config.device_torch
            moe.routed_expert_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
            moe.routed_expert_end_idx = (self.global_rank + 1) * NUM_EXPERT_PER_RANK
            moe.experts_per_rank = NUM_EXPERT_PER_RANK
            moe.num_persistent_local_experts = self.num_local_expert_per_layer
            moe.enable_ep_offloading = self.enable_ep_offloading

        if self.rank == 0:
            logging.info(
                f"[DECODE] Configured Glm5MoE for {self.model_config.num_hidden_layers - self.FIRST_K_DENSE} "
                f"MoE layers (experts_per_rank={NUM_EXPERT_PER_RANK}, "
                f"offloading={self.enable_ep_offloading})"
            )

    _SKELETON_KEY_REMAP = {}

    def _load_model_skeleton(self):
        """Load skeleton weights as-is (no CPU dequant). FP8 dequant happens on-the-fly."""
        from collections import Counter
        loaded, skipped, remapped = 0, 0, 0
        qa_trace = []
        # Per-bucket counters so a zero-count category (e.g. attn_norm,
        # e_score_correction_bias) pops immediately in the summary.
        loaded_bucket = Counter()
        missing_bucket = Counter()

        def _bucket_for(k: str) -> str:
            if "q_a_layernorm" in k or "kv_a_layernorm" in k:
                return "attn_norm"
            if "input_layernorm" in k or "post_attention_layernorm" in k:
                return "layer_norm"
            if "e_score_correction_bias" in k:
                return "gate_bias"
            if "indexer" in k:
                return "indexer"
            if ".mlp.gate.weight" in k:
                return "gate_weight"
            if "embed_tokens" in k or "lm_head" in k or "model.norm" in k:
                return "global"
            if ".self_attn." in k:
                return "attn_other"
            if ".mlp." in k:
                return "mlp_other"
            return "other"

        for key, param in self.model.named_parameters():
            if key in self.state_dict_name_map:
                skipped += 1
                if self.rank == 0 and ("q_a_layernorm" in key or "kv_a_layernorm" in key):
                    qa_trace.append(f"SKIPPED (in state_dict_name_map): {key}")
                continue
            # Try direct match first, then remapped key
            ckpt_key = key
            for src, dst in self._SKELETON_KEY_REMAP.items():
                if src in key:
                    ckpt_key = key.replace(src, dst)
                    break
            if ckpt_key in self.skeleton_state_dict:
                param.data = self.skeleton_state_dict[ckpt_key]
                loaded += 1
                loaded_bucket[_bucket_for(key)] += 1
                if ckpt_key != key:
                    remapped += 1
                if self.rank == 0 and ("q_a_layernorm" in key or "kv_a_layernorm" in key):
                    qa_trace.append(f"LOADED: {key} (ckpt_key={ckpt_key})")
            elif key in self.skeleton_state_dict:
                param.data = self.skeleton_state_dict[key]
                loaded += 1
                loaded_bucket[_bucket_for(key)] += 1
                if self.rank == 0 and ("q_a_layernorm" in key or "kv_a_layernorm" in key):
                    qa_trace.append(f"LOADED (fallback): {key}")
            else:
                missing_bucket[_bucket_for(key)] += 1
                if self.rank == 0 and ("q_a_layernorm" in key or "kv_a_layernorm" in key):
                    qa_trace.append(f"MISSING from skeleton: {key} (tried ckpt_key={ckpt_key})")
                if self.rank == 0 and ("gate" in key or "e_score_correction_bias" in key):
                    logging.warning(f"[SKELETON] Missing key: {key} (tried ckpt_key={ckpt_key})")

        if self.rank == 0 and qa_trace:
            # Log first few samples from each bucket
            seen_types = {}
            for entry in qa_trace:
                bucket = entry.split(":", 1)[0]
                seen_types.setdefault(bucket, []).append(entry)
            for bucket, entries in seen_types.items():
                logging.warning(
                    f"[SKELETON q_a/kv_a trace] {bucket}: {len(entries)} entries; "
                    f"examples: {entries[:3]}"
                )

        if self.rank == 0:
            logging.info(f"[SKELETON] loaded={loaded}, skipped={skipped}, remapped={remapped}")
            # Bucket summary — a zero count for attn_norm or gate_bias means
            # those keys never matched the checkpoint and silently remain at
            # init (ones for norms, zeros for bias).
            logging.warning(
                f"[SKELETON-BUCKETS loaded] {dict(loaded_bucket)}"
            )
            if missing_bucket:
                logging.warning(
                    f"[SKELETON-BUCKETS missing] {dict(missing_bucket)}"
                )

        skeleton_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024**3)
        if self.rank == 0:
            logging.info(f"Model skeleton size: {skeleton_size:.2f} GB")

    def _setup_fp8_scales(self):
        """Attach FP8 scale tensors to indexer and dense MLP for on-the-fly dequant."""
        device = self.engine_config.Basic_Config.device_torch
        for layer_idx in range(self.model_config.num_hidden_layers):
            attn = self.model.model.layers[layer_idx].self_attn
            # After wrapping, self_attn is GLM5AttnWrapper; original Glm5MLA is at .module
            inner = attn.module if hasattr(attn, 'module') else attn
            # When use_dense_mla is set, Glm5MLA skips indexer construction;
            # skip the scale attach too (no destination).
            if hasattr(inner, "indexer"):
                indexer = inner.indexer
                for proj, attr in [("wk", "wk_scale"), ("wq_b", "wq_b_scale")]:
                    key = f"model.layers.{layer_idx}.self_attn.indexer.{proj}.weight_scale_inv"
                    if key in self.dequant_scale:
                        setattr(indexer, attr, self.dequant_scale[key].to(device))
        for layer_idx in range(self.FIRST_K_DENSE):
            mlp = self.model.model.layers[layer_idx].mlp
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"model.layers.{layer_idx}.mlp.{proj}.weight_scale_inv"
                if key in self.dequant_scale:
                    setattr(mlp, f"{proj.split('_')[0]}_scale",
                            self.dequant_scale[key].to(device))

    def _init_fused_kernels(self):
        """Initialize TMA-based CUDA kernels after FP8 scales are attached.

        Counts WP2/WP4 init failures so a silent fallback to PyTorch
        doesn't regress perf unnoticed.

        Skipped entirely in dense-MLA mode (no indexer module → nothing to
        fuse). Structural check on the first attention layer — more robust
        than threading a config flag through two parallel config types.
        """
        first_attn = self.model.model.layers[0].self_attn
        first_inner = first_attn.module if hasattr(first_attn, "module") else first_attn
        if not hasattr(first_inner, "indexer"):
            if self.rank == 0:
                logging.info("[DSA kernels] skipped (no indexer — dense-MLA mode)")
            return
        total = len(self.model.model.layers)
        inited = 0
        wp2_ok = 0
        wp4_ok = 0
        for layer_idx in range(total):
            wrapper = self.model.model.layers[layer_idx].self_attn
            if hasattr(wrapper, 'initialize_fused_kernels'):
                wrapper.initialize_fused_kernels()
                inited += 1
                if getattr(wrapper, '_indexer_cuda_weights', None) is not None:
                    wp2_ok += 1
                if getattr(wrapper, '_fused_wqb_weights', None) is not None:
                    wp4_ok += 1
        if self.rank == 0:
            logging.info(
                f"[DSA kernels] init={inited}/{total} layers, "
                f"WP2={wp2_ok}/{inited}, WP4={wp4_ok}/{inited}"
            )

    def _lm_head_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self):
        self.model.lm_head.register_forward_pre_hook(self._lm_head_forward_pre_hook)

    def _extract_dequantize_scale(self):
        self.dequant_scale = {}
        for key, param in self.skeleton_state_dict.items():
            if "weight_scale_inv" in key:
                self.dequant_scale[key] = param
