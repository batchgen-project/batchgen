# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
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
import operator
import os
import time
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist

from batchgen.models.glm.glm5.releases import SPARSE_PREFILL_MODEL_TYPES
from batchgen.models.wrappers import AttnWrapperBase
from batchgen.phase_lifecycle import (
    AdmitContext,
    ExitMode,
    PhaseEnterContext,
    PhaseKey,
)

from .model import (
    Glm5ForCausalLM,
    Glm5MoE,
    _required_dsa_model_kernel_import_failures,
)
from .triton_prewarm import PHASE_DECODE, PHASE_GEOMETRY, PHASE_PREFILL
from .wrappers import (
    GLM5ExpertWrapper,
    GLM5AttnWrapper,
    _required_dsa_kernel_import_failures,
)


def _synchronize_prefill_preloads():
    """Keep one rank's forward from overlapping another rank's preload."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# Reserved sequence-id space for the startup decode probe. Real sequence ids
# are non-negative worker-assigned globals; GLM-5's prefix-pool segment pseudo
# sids are ``-(rank + 1 + world_size * counter)`` (batchgen_worker.py
# ``_next_pool_segment_sid``) — small negatives that grow by one per segment.
# Probing from -3_000_000_001 downward is outside both for any realistic
# campaign, and the probe's synthetic surfaces own their whole seq->slot map,
# so no shared pool can collide with them.
_GLM5_PROBE_SEQ_ID_BASE = -3_000_000_001

# Per-token fp32 scale bytes of DeepGEMM's page-split FP8 index-key carrier.
# The authoritative constant is
# ``batchgen_kernels.attention.dsa.indexer_k_quant.SCALE_BYTES_PER_TOKEN``;
# it is not imported here because that package's ``__init__`` needs triton
# (absent on CPU test environments), and the layout itself is a fixed kernel
# ABI the real writer validates (``cache_is_page_split``) before any launch.
_DSA_PAGE_SPLIT_SCALE_BYTES = 4


class StartupDecodeProbeError(RuntimeError):
    """The startup decode probe cannot honor its cap-width contract."""


@dataclass(frozen=True)
class StartupDecodeProbePlan:
    """The bounded synthetic decode shape used by the startup probe."""

    batch_size: int
    past_tokens: int = 1


def plan_startup_decode_probe(
    padding_bsz: int, configured_cap: int
) -> StartupDecodeProbePlan:
    """Validate and describe one cap-width synthetic decode step.

    Local copy of the gpt-oss/kimi-linear probe seam
    (batchgen/models/moonshotai/kimi_linear/Parallel_Strategy_Manager.py):
    model families do not import one another's modules, so the helper is
    duplicated here rather than cross-imported.
    """
    try:
        if isinstance(padding_bsz, bool) or isinstance(configured_cap, bool):
            raise TypeError("bool is not a batch-width integer")
        requested = operator.index(padding_bsz)
        cap = operator.index(configured_cap)
    except (TypeError, ValueError) as exc:
        raise StartupDecodeProbeError(
            "startup decode probe requires integer padding_bsz and configured cap"
        ) from exc
    if requested <= 0:
        raise StartupDecodeProbeError(
            f"startup decode probe requires padding_bsz > 0, got {requested}"
        )
    if cap <= 0:
        raise StartupDecodeProbeError(
            f"startup decode probe has no configured positive cap (got {cap})"
        )
    if requested > cap:
        raise StartupDecodeProbeError(
            "startup decode probe width exceeds configured decode cap: "
            f"padding_bsz={requested}, configured_cap={cap}"
        )
    return StartupDecodeProbePlan(batch_size=requested)


def uniform_expert_assignment(
    num_rows: int,
    num_experts: int,
    top_k: int,
    *,
    device=None,
) -> torch.Tensor:
    """Return a row-major, round-robin top-k assignment over all experts.

    Local copy of the gpt-oss/kimi-linear helper (same rationale as
    ``plan_startup_decode_probe``).
    """
    try:
        if any(isinstance(value, bool) for value in (num_rows, num_experts, top_k)):
            raise TypeError("bool is not an assignment dimension")
        rows = operator.index(num_rows)
        experts = operator.index(num_experts)
        k = operator.index(top_k)
    except (TypeError, ValueError) as exc:
        raise StartupDecodeProbeError(
            "uniform expert assignment requires integer dimensions"
        ) from exc
    if rows < 0 or experts <= 0 or k <= 0 or k > experts:
        raise StartupDecodeProbeError(
            "invalid uniform expert assignment shape: "
            f"rows={rows}, experts={experts}, top_k={k}"
        )
    row_starts = (torch.arange(rows, device=device, dtype=torch.long) * k) % experts
    offsets = torch.arange(k, device=device, dtype=torch.long)
    return (row_starts[:, None] + offsets[None, :]) % experts


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
        # --persistent-phase-instances: each phase's (model, weight_copy_task)
        # is built once and switched in by activate_prefill/activate_decoding.
        self.persistent_phase_instances = False
        # The same build-once/activate machinery, requested by the worker's
        # default phase-switch lifecycle instead of by the CLI flag. It is a
        # capability condition, not a mode: the worker turns it on only for a
        # rank that keeps every one of its local routed experts resident, which
        # is already the hard precondition of the 3D grouped-GEMM decode path.
        # A rank that streams part of its experts leaves it off, keeps the
        # legacy per-expert loader and reports no reusable instance, so the
        # worker rebuilds that phase exactly as before.
        self.reuse_phase_instances = False
        self.prefill_instance = None
        self.decode_instance = None
        self._prefill_moe_mode = None
        self._pending_prefill_moe_mode = None
        self._expert_arena = None
        self._expert_copy_pairs = []
        # Device copies of the skeleton's FP8 dequant scales. The scales never
        # change, so they are uploaded once per process in one bulk copy and
        # handed out as views; see _device_scale.
        self._device_scales = None
        # Shared device skeleton (see share_skeleton_on_device): whether it is
        # on the device, which skeleton keys were uploaded, the model keys the
        # first instance bound to it, and the (parameter, checkpoint key) pairs
        # of the instance configured last.
        self._skeleton_on_device = False
        self._skeleton_uploaded_keys = frozenset()
        self._shared_skeleton_keys = None
        self._skeleton_bound_params = []
        # The decode CUDA graph's capture bucket list, for the Triton prewarm's
        # bucket-keyed entries. The worker derives it from its CUDA-graph
        # arguments; None until something hands it over (see triton_prewarm).
        # Nothing does yet: with the graph on, the prewarm logs that once
        # (``report_unset_decode_buckets``).
        self.decode_graph_buckets = None

    @property
    def _instances_are_reused(self) -> bool:
        """True when this process keeps its phase instances and switches them in.

        Either the CLI flag or the worker's default lifecycle asks for it. The
        arena-backed routed-expert stacks come with it: they are what lets the
        experts' physical memory be released for prefill and remapped at the
        same addresses afterwards.
        """
        if self.persistent_phase_instances:
            return True
        # The default lifecycle only gets reusable instances when the arena
        # path is actually available for this rank. A streaming rank keeps
        # decode_instance None, which is how the worker learns to rebuild.
        return bool(
            self.reuse_phase_instances
            and self._local_routed_experts_fully_resident()
        )

    def configure_prefill(self):
        """Configure model for prefill (pure DP, all modules offloaded)."""
        self._check_late_deepgemm_compiles("configure_prefill")
        start_time = time.perf_counter()
        timings = {}

        pending_prefill_moe_mode = getattr(
            self, "_pending_prefill_moe_mode", None
        )
        if pending_prefill_moe_mode is not None:
            self._prefill_moe_mode = pending_prefill_moe_mode
        self.loaded_model_config.phase = "prefill"
        Glm5MoE.reset_prefill_grouped_state()

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
        if self._instances_are_reused:
            # Both kept instances bind one device skeleton; see
            # share_skeleton_on_device. Must precede the first scale lookup.
            self.share_skeleton_on_device()
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
        self._assert_skeleton_still_shared("prefill")
        # GLM-5.2 long prompts run packed sparse absorbed FlashMLA prefill, which
        # has no dense fallback. Prove the required runtime exists here, at
        # configure time, instead of failing minutes into a long prefill.
        if (
            getattr(self.loaded_model_config, "model_type", None)
            in SPARSE_PREFILL_MODEL_TYPES
        ):
            from .sparse_prefill import validate_glm52_sparse_prefill_runtime

            validate_glm52_sparse_prefill_runtime()
        timings['triton_prewarm'] = self._prewarm_triton_kernels(PHASE_PREFILL)
        if self.is_fp8_experts:
            from batchgen.attention.fused_kernels import (
                preload_fused_attention_kernels,
            )

            preload_fused_attention_kernels()
            grouped_layers = [
                layer.mlp
                for layer in self.model.model.layers[self.FIRST_K_DENSE :]
                if isinstance(layer.mlp, Glm5MoE)
                and layer.mlp._prefill_grouped_enabled
            ]
            expected_layers = (
                self.model_config.num_hidden_layers - self.FIRST_K_DENSE
            )
            if len(grouped_layers) != expected_layers:
                raise RuntimeError(
                    "GLM-5 FP8 prefill requires grouped full-offload MoE on "
                    f"all {expected_layers} routed layers; got {len(grouped_layers)}"
                )
            Glm5MoE.init_prefill_grouped_buffers(
                self.loaded_model_config,
                self.engine_config.Basic_Config.device_torch,
            )
            for moe in grouped_layers:
                moe._prefill_release_event = torch.cuda.Event()
                moe._prefill_shared_release_event = torch.cuda.Event()
            _synchronize_prefill_preloads()
        timings['to_device'] = (
            time.perf_counter() - step_start - timings['triton_prewarm']
        )

        total_time = time.perf_counter() - start_time
        if self.rank == 0:
            logging.info(
                f"[PREFILL] Model configured in {total_time:.2f}s "
                f"(init={timings['model_init']:.1f}s, skeleton={timings['skeleton']:.1f}s, "
                f"expert={timings['expert']:.1f}s, to_device={timings['to_device']:.1f}s, "
                f"triton_prewarm={timings['triton_prewarm']:.1f}s)"
            )
        if self._instances_are_reused:
            self.prefill_instance = (self.model, self.weight_copy_task)
        return self.model, self.weight_copy_task

    def configure_decoding(self, padding_bsz=None, comm=None):
        """Configure model for decode: DP + EP."""
        self._check_late_deepgemm_compiles("configure_decoding")
        Glm5MoE.reset_prefill_grouped_state()
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

        # Same planner the reuse capability is decided with before the first
        # configure, so the two can never disagree.
        self.enable_ep_offloading, NUM_LOCAL_EXPERT_PER_LAYER = (
            self._plan_local_expert_split()
        )

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
        if self._instances_are_reused:
            self.share_skeleton_on_device()
        self._extract_dequantize_scale()
        self._load_model_skeleton()
        if self.persistent_phase_instances:
            # Unchanged: the flag requires full residency and says so loudly.
            self._load_local_routed_experts_into_arena()
        elif self.reuse_phase_instances and self._local_routed_experts_fully_resident():
            # Default lifecycle: same fixed-address stacks, so the experts can
            # be released for prefill and remapped at the same addresses.
            self._load_local_routed_experts_into_arena()
        elif self._local_routed_experts_fully_resident():
            self._load_local_routed_experts_prealloc()
        else:
            # Partially resident rank: the rest of its experts stream through
            # the H2D copy engine, so there is no full stack row per expert
            # and the 3D grouped-GEMM path is disabled anyway.
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
        self._assert_skeleton_still_shared("decode")
        self._prewarm_triton_kernels(PHASE_DECODE)

        if self.rank == 0:
            used = torch.cuda.memory_allocated(self.engine_config.Basic_Config.device_torch)
            logging.info(f"[MODEL] GPU memory after init: {used / (1024**3):.2f} GB used")

        self._init_mode_decoding()
        effective_bsz = padding_bsz if padding_bsz is not None else 128
        self._init_decoding_padding_bsz(effective_bsz)

        if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "1":
            self._init_ata_comms(effective_bsz)

        if self._instances_are_reused:
            self.decode_instance = (self.model, self.weight_copy_task)
        return self.model, self.weight_copy_task

    def prewarm_decode_graphs(self, gpu_manager=None):
        """Compile the decode kernels keyed on the KV page-table geometry.

        The worker calls this with its GPU KV manager right before CUDA-graph
        warmup on every decode configure. It runs the geometry phase of the
        Triton prewarm the first time and again whenever the page-table
        widths change, then arms the late-compile guard; any other call
        returns at once.
        """
        if gpu_manager is None:
            return 0.0
        return self._prewarm_triton_kernels(PHASE_GEOMETRY,
                                            gpu_manager=gpu_manager)

    def _triton_prewarm_partition(self):
        """``(part_index, part_count, barrier)`` for one prewarm phase.

        Attention is pure DP and every rank of a node shares one Triton cache
        directory, so the node-local ranks split each phase's compiles and
        meet at a global barrier before every rank loads the full set.
        """
        count = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0)
        if count <= 0:
            count = torch.cuda.device_count()
        index = int(self.local_rank)
        if count < 1 or not 0 <= index < count:
            logging.warning(
                "[GLM5_TRITON_WARMUP] rank=%s local rank %s outside a local "
                "world of %s; this rank runs every prewarm entry itself",
                self.global_rank, index, count,
            )
            return 0, 1, None
        return index, count, _synchronize_prefill_preloads

    def _prewarm_triton_kernels(self, phase, gpu_manager=None):
        """Run one phase of the GLM-5 Triton prewarm; returns its wall time.

        Each phase runs once per process (the geometry phase once per
        geometry). The serving kernels' keys and the reasons for every
        witness are in ``triton_prewarm``.
        """
        if not torch.cuda.is_available():
            return 0.0
        from .triton_prewarm import (
            prewarm_glm5_decode,
            prewarm_glm5_geometry,
            prewarm_glm5_prefill,
            report_unset_decode_buckets,
        )

        if phase in (PHASE_DECODE, PHASE_GEOMETRY):
            report_unset_decode_buckets(
                self.decode_graph_buckets,
                whole_model_graph=bool(
                    self.engine_config.Basic_Config.enable_cuda_graphs),
                rank=self.global_rank,
            )
        part_index, part_count, barrier = self._triton_prewarm_partition()
        common = dict(part_index=part_index, part_count=part_count,
                      barrier=barrier, rank=self.global_rank)
        start = time.perf_counter()
        if phase == PHASE_PREFILL:
            prewarm_glm5_prefill(self.loaded_model_config,
                                 max_rows=self._deepgemm_max_rows(), **common)
        elif phase == PHASE_DECODE:
            prewarm_glm5_decode(
                self.model, self.loaded_model_config,
                decode_buckets=self.decode_graph_buckets,
                max_rows=self._deepgemm_max_rows(), **common,
            )
        elif phase == PHASE_GEOMETRY:
            prewarm_glm5_geometry(
                self.model, self.loaded_model_config, gpu_manager,
                decode_buckets=self.decode_graph_buckets,
                world_size=self.world_size, **common,
            )
        else:
            raise ValueError(f"unknown GLM-5 Triton prewarm phase {phase!r}")
        return time.perf_counter() - start

    def _deepgemm_max_rows(self) -> int:
        """Largest row count a DeepGEMM GEMM of the prewarm must cover.

        A prefill micro-batch holds at most the token cap, except a prompt
        longer than the cap, which is a micro-batch of its own. Eager decode
        rows are assumed to stay below it. See ``deepgemm_prewarm``.
        """
        cap = int(self.engine_config.Module_Batching_Config
                  .prefill_micro_batch_token_cap)
        prompt = self.engine_config.Basic_Config.get_max_prompt_length()
        return max(cap, int(prompt or 0))

    def _check_late_deepgemm_compiles(self, where):
        """Log the DeepGEMM kernels compiled since the last geometry
        prewarm; ``batchgen_debug.glm5_triton_strict`` makes it raise."""
        if not torch.cuda.is_available():
            return
        from .deepgemm_prewarm import check_late_compiles
        from .triton_prewarm import _batch_debug, strict_requested

        check_late_compiles(self.global_rank, where,
                            strict=strict_requested(_batch_debug()))

    def set_prefill_moe_mode(self, mode):
        """Record the worker-selected prefill MoE mode for reuse checks."""
        self._pending_prefill_moe_mode = mode

    def _prepare_prefill_activation(self, instance=None):
        Glm5MoE.reset_prefill_grouped_state()
        self.loaded_model_config.phase = "prefill"
        if instance is not None:
            self.model, self.weight_copy_task = instance
        Glm5MoE.init_prefill_grouped_buffers(
            self.loaded_model_config,
            self.engine_config.Basic_Config.device_torch,
        )
        _synchronize_prefill_preloads()

    def activate_prefill(self):
        """Switch the already-built prefill instance in (persistent instances)."""
        self._check_late_deepgemm_compiles("activate_prefill")
        self._prepare_prefill_activation(self.prefill_instance)
        return self.model, self.weight_copy_task

    def activate_decoding(self):
        """Switch the already-built decode instance in and refill its experts."""
        self._check_late_deepgemm_compiles("activate_decoding")
        prep_start = time.perf_counter()
        Glm5MoE.reset_prefill_grouped_state()
        torch.cuda.empty_cache()
        self.loaded_model_config.phase = "decode"
        self.model, self.weight_copy_task = self.decode_instance
        start = time.perf_counter()
        self._expert_arena.map()
        mapped = time.perf_counter()
        self.refill_local_routed_experts()
        done = time.perf_counter()
        nbytes = sum(dst.nbytes for dst, _ in self._expert_copy_pairs)
        logging.info(
            "[PERSISTENT_PHASE] rank=%d prep=%.3fs map=%.3fs refill=%.3fs %.1f GB/s",
            self.rank,
            start - prep_start,
            mapped - start,
            done - mapped,
            nbytes / (done - mapped) / 1e9,
        )
        return self.model, self.weight_copy_task

    @property
    def decode_experts_resident(self) -> bool:
        return self._expert_arena is not None and self._expert_arena.is_mapped

    def release_decode_routed_experts(self):
        """Return the decode routed experts' HBM; their addresses stay reserved."""
        torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
        start = time.perf_counter()
        self._expert_arena.unmap()
        logging.info(
            "[PERSISTENT_PHASE] rank=%d unmap=%.3fs",
            self.rank,
            time.perf_counter() - start,
        )

    def phase_enter(self, key: PhaseKey, ctx: PhaseEnterContext):
        """Enter ``key.phase``; return its ``(model, weight_copy_task)``.

        Resumes the instance this process keeps for the phase, otherwise
        builds it. A prefill build uses the mode recorded by
        set_prefill_moe_mode.
        """
        if key.phase == "prefill":
            if self.prefill_instance is not None:
                return self.activate_prefill()
            return self.configure_prefill()
        if key.phase == "decode":
            if ctx.comm is None:
                raise ValueError(
                    "decode phase_enter requires PhaseEnterContext.comm"
                )
            if ctx.max_num_seq is None:
                raise ValueError(
                    "decode phase_enter requires PhaseEnterContext.max_num_seq"
                )
            if self.decode_instance is not None:
                return self.activate_decoding()
            return self.configure_decoding(
                padding_bsz=ctx.max_num_seq, comm=ctx.comm
            )
        raise ValueError(f"phase_enter: unsupported phase {key.phase!r}")

    def admit(self, key: PhaseKey, ctx: AdmitContext) -> None:
        """Prepare the live phase instance for one admission wave.

        Prefill resets the grouped prefill state and re-initializes its
        buffers on the live model. Decode has no manager-side admission work:
        KV allocation and local-row resolution are worker-owned.
        """
        if key.phase == "prefill":
            self._check_late_deepgemm_compiles("admit")
            self._prepare_prefill_activation()
            return
        if key.phase == "decode":
            return
        raise ValueError(f"admit: unsupported phase {key.phase!r}")

    def phase_exit(self, key: PhaseKey, mode: ExitMode) -> None:
        """Leave ``key.phase`` in ``mode`` (see ExitMode)."""
        if key.phase not in ("prefill", "decode"):
            raise ValueError(f"phase_exit: unsupported phase {key.phase!r}")
        if mode is ExitMode.RETIRE:
            # The worker's lifecycle orchestration performs the retire
            # mechanics (deep_free_model_memory, GPU KV destroy); this manager
            # owns no further state to free.
            if key.phase == "prefill":
                self.prefill_instance = None
            else:
                self.decode_instance = None
            self.model = None
            self.weight_copy_task = {}
            return
        if mode is not ExitMode.SUSPEND:
            raise ValueError(f"phase_exit: unsupported mode {mode!r}")
        if key.phase == "prefill":
            # Prefill keeps no manager-owned state to suspend; the worker
            # stops the H2D worker and releases the ring.
            return
        if not self.decode_experts_resident:
            raise RuntimeError(
                f"rank {self.rank}: decode SUSPEND needs a resident reusable "
                "decode instance; a rank without one must RETIRE"
            )
        self.release_decode_routed_experts()

    def holds_instance(self, phase: str) -> bool:
        """Return whether a resumable instance is retained for ``phase``."""
        if phase == "prefill":
            return self.prefill_instance is not None
        if phase == "decode":
            return self.decode_instance is not None
        raise ValueError(f"holds_instance: unsupported phase {phase!r}")

    @staticmethod
    def _copy_expert_pairs(pairs):
        """H2D-copy every (stack row, pinned host source) pair, then await once.

        The destinations already exist, so these are plain async copies into
        existing memory rather than one blocking copy-plus-allocate per tensor.
        """
        for dst, src in pairs:
            dst.copy_(src, non_blocking=True)
        torch.cuda.current_stream().synchronize()

    def refill_local_routed_experts(self):
        """H2D-copy every local routed expert into its fixed-address stack row."""
        self._copy_expert_pairs(self._expert_copy_pairs)

    def share_skeleton_on_device(self):
        """Upload the shared skeleton once so both kept phase instances alias it.

        Both instances' ``_load_model_skeleton`` bind ``param.data`` to the
        entries of ``skeleton_state_dict``; once those entries live on the
        device, ``model.to(device)`` keeps the bound tensors as they are, so
        the prefill and decode instances hold ONE device copy of the skeleton
        (norms, gates, indexer, dense MLP, embeddings, LM head) instead of one
        each. The phase-specific weights never come from here: they are loaded
        under ``state_dict_name_map`` keys, which the skeleton binding skips,
        and ``_assert_skeleton_still_shared`` checks both properties after
        every configure.

        Only what is read on the device moves: the entries the model binds
        (``_skeleton_bind_keys``, the same rule ``_load_model_skeleton``
        applies) plus the FP8 ``weight_scale_inv`` tensors the scale lookups
        consume by name. Everything else — chiefly the MTP layer's weights,
        which no instance binds — stays in host memory, as it did before.

        Idempotent. It has to run before the first scale lookup: scales
        uploaded from the host copies would otherwise stay a second copy.
        The upload set comes from a built model, so a call made before any
        configure (the persistent path asks at startup) leaves it to the
        configure that builds one, which calls this again.
        """
        if self._skeleton_on_device:
            return
        if self._device_scales is not None:
            raise RuntimeError(
                "share_skeleton_on_device() after the FP8 scales were uploaded "
                "from the host skeleton: share the skeleton before the first "
                "phase configure"
            )
        if (
            getattr(self, "model", None) is None
            or getattr(self, "state_dict_name_map", None) is None
        ):
            return
        upload = self._skeleton_bind_keys() | {
            key for key in self.skeleton_state_dict if "weight_scale_inv" in key
        }
        device = self.engine_config.Basic_Config.device_torch
        start = time.perf_counter()
        buckets = {}
        kept_count, kept_bytes = 0, 0
        shared = {}
        for key, tensor in self.skeleton_state_dict.items():
            nbytes = tensor.numel() * tensor.element_size()
            if key not in upload:
                kept_count += 1
                kept_bytes += nbytes
                shared[key] = tensor
                continue
            bucket = self._skeleton_bucket(key)
            count, total = buckets.get(bucket, (0, 0))
            buckets[bucket] = (count + 1, total + nbytes)
            shared[key] = tensor.to(device)
        self.skeleton_state_dict = shared
        self._skeleton_uploaded_keys = frozenset(upload)
        self._skeleton_on_device = True
        if self.rank == 0:
            total = sum(nbytes for _, nbytes in buckets.values())
            logging.info(
                "[SKELETON] shared on device in %.2fs: %d tensors, %.3f GiB | %s"
                " | kept on host (bound by no instance): %d tensors, %.3f GiB",
                time.perf_counter() - start,
                len(upload),
                total / (1024 ** 3),
                ", ".join(
                    f"{name}={count}/{nbytes / (1024 ** 3):.3f}GiB"
                    for name, (count, nbytes) in sorted(
                        buckets.items(), key=lambda item: -item[1][1]
                    )
                ),
                kept_count,
                kept_bytes / (1024 ** 3),
            )

    def _skeleton_bind_keys(self):
        """Checkpoint keys ``_load_model_skeleton`` binds for ``self.model``.

        The same rule, applied without binding: every model parameter outside
        ``state_dict_name_map``, remapped through ``_SKELETON_KEY_REMAP``, that
        the skeleton holds.
        """
        keys = set()
        for key, _ in self.model.named_parameters():
            if key in self.state_dict_name_map:
                continue
            ckpt_key = key
            for src, dst in self._SKELETON_KEY_REMAP.items():
                if src in key:
                    ckpt_key = key.replace(src, dst)
                    break
            if ckpt_key in self.skeleton_state_dict:
                keys.add(ckpt_key)
            elif key in self.skeleton_state_dict:
                keys.add(key)
        return keys

    @staticmethod
    def _skeleton_bucket(key: str) -> str:
        """Coarse category of a skeleton key, for the binding and upload logs."""
        if "weight_scale_inv" in key:
            return "scale"
        if "q_a_layernorm" in key or "kv_a_layernorm" in key:
            return "attn_norm"
        if "input_layernorm" in key or "post_attention_layernorm" in key:
            return "layer_norm"
        if "e_score_correction_bias" in key:
            return "gate_bias"
        if "indexer" in key:
            return "indexer"
        if ".mlp.gate.weight" in key:
            return "gate_weight"
        if "embed_tokens" in key or "lm_head" in key or "model.norm" in key:
            return "global"
        if ".self_attn." in key:
            return "attn_other"
        if ".mlp." in key:
            return "mlp_other"
        return "other"

    def _check_shared_skeleton_binding(self, bound):
        """Pin what one instance bound to the shared device skeleton.

        ``bound`` maps model key -> (checkpoint key, parameter). Fails loud if
        a phase-managed key was bound (its loads would then write into memory
        the other phase reads), if a parameter does not alias the shared
        tensor, or if this instance binds a different key set than the first
        one did. Logs, once, the skeleton entries no instance binds.
        """
        overlap = sorted(set(bound) & set(self.state_dict_name_map))
        if overlap:
            raise RuntimeError(
                f"shared skeleton bound {len(overlap)} phase-managed keys, "
                f"e.g. {overlap[:3]}"
            )
        missing = sorted(
            ckpt_key
            for ckpt_key, _ in bound.values()
            if ckpt_key not in self._skeleton_uploaded_keys
        )
        if missing:
            raise RuntimeError(
                f"shared skeleton bound {len(missing)} entries that were not "
                f"uploaded to the device, e.g. {missing[:3]}"
            )
        for key, (ckpt_key, param) in bound.items():
            if param.data_ptr() != self.skeleton_state_dict[ckpt_key].data_ptr():
                raise RuntimeError(
                    f"{key} does not alias the shared device skeleton entry "
                    f"{ckpt_key}"
                )
        keys = frozenset(bound)
        phase = getattr(self.loaded_model_config, "phase", None)
        if self._shared_skeleton_keys is None:
            self._shared_skeleton_keys = keys
            if self.rank == 0:
                used = {ckpt_key for ckpt_key, _ in bound.values()}
                unbound = {}
                for key in self._skeleton_uploaded_keys:
                    if key in used or "weight_scale_inv" in key:
                        continue
                    tensor = self.skeleton_state_dict[key]
                    bucket = self._skeleton_bucket(key)
                    count, nbytes = unbound.get(bucket, (0, 0))
                    unbound[bucket] = (
                        count + 1,
                        nbytes + tensor.numel() * tensor.element_size(),
                    )
                logging.info(
                    "[SKELETON] unbound non-scale entries on device: %d, %.3f GiB | %s",
                    sum(count for count, _ in unbound.values()),
                    sum(nbytes for _, nbytes in unbound.values()) / (1024 ** 3),
                    ", ".join(
                        f"{name}={count}/{nbytes / (1024 ** 3):.3f}GiB"
                        for name, (count, nbytes) in sorted(unbound.items())
                    ) or "none",
                )
        elif keys != self._shared_skeleton_keys:
            raise RuntimeError(
                f"phase {phase!r} binds a different shared skeleton key set: "
                f"{len(keys - self._shared_skeleton_keys)} extra, "
                f"{len(self._shared_skeleton_keys - keys)} missing"
            )
        if self.rank == 0:
            nbytes = sum(
                param.numel() * param.element_size() for _, param in bound.values()
            )
            logging.info(
                "[SKELETON] phase=%s binds %d shared device tensors (%.3f GiB); "
                "same key set as the first binding; no overlap with %d "
                "phase-managed keys",
                phase,
                len(bound),
                nbytes / (1024 ** 3),
                len(self.state_dict_name_map),
            )

    def _assert_skeleton_still_shared(self, phase: str) -> None:
        """After a configure: every bound parameter still aliases the shared copy.

        A phase-specific load, wrapper or cast that rebinds one of them would
        silently give this instance a private copy again.
        """
        if not self._skeleton_on_device:
            return
        moved = [
            ckpt_key
            for param, ckpt_key in self._skeleton_bound_params
            if param.data_ptr() != self.skeleton_state_dict[ckpt_key].data_ptr()
        ]
        if moved:
            raise RuntimeError(
                f"{phase} configure left {len(moved)} skeleton parameters off the "
                f"shared device copy, e.g. {moved[:3]}"
            )

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

    def startup_decode_probe_step(self, padding_bsz: int) -> None:
        """Run one cap-width synthetic DSA decode step for startup accounting.

        The worker owns sequence admission, paged-KV pool pages and the
        per-step decode metadata for real rows; at this startup seam none of
        those rows exist yet. The probe therefore:

        - installs a tiny synthetic PRIMARY paged-KV surface holding the MLA
          latent geometry (``kv_lora_rank + qk_rope_head_dim`` per token, one
          KV head, bf16) at the deployment page size (never a guessed one).
          Physical pages are O(batch): the page-table entry that holds each
          row's write position maps to that row's own page, and every history
          entry maps to one shared read-only zero page (the probe writes only
          the new token, so sharing history pages is race-free at any
          synthetic context length);
        - installs a synthetic AUXILIARY index-key surface with the same page
          geometry in the layout the deployment allocates: DeepGEMM's
          page-split FP8 carrier (``index_head_dim`` e4m3 bytes + 4 scale
          bytes per token, uint8) for the releases whose geometry reuses
          top-k across layers (GLM-5.2/5.3, the same
          ``SPARSE_PREFILL_MODEL_TYPES`` predicate the host-KV profiles key
          on), and the bf16 AoS carrier for GLM-5/5.1. Both surfaces expose
          the ``_gpu_page_table_manager.seq_id_to_slot`` indirection the
          eager selector resolves (reserved synthetic sequence ids,
          ``_GLM5_PROBE_SEQ_ID_BASE`` downward);
        - makes every row's context ``index_topk + 1`` tokens, because a row
          at or under ``index_topk`` runs the dense all-short FA3 path and
          the probe would never execute the sparse selection kernels (the
          fused index-K writer, the page-split top-k scorer / AoS scoring
          pipeline and the selected-token sparse FA3 attention all run REAL
          at cap width). On GLM-5.2/5.3 the skip_topk shared layers reuse the
          carried top-k of the preceding full layer, exactly as in serving.
          Host-append callbacks stay None for both regions, so nothing is
          appended host-side. Honest exclusion: the scorer's context width is
          ``index_topk + 1``, the minimal long context, so transients that
          grow with a longer decode context are not measured here;
        - forces the eager decode path for the whole step
          (``batchgen_debug.glm5_dsa_mode/glm5_moe_mode = "eager"``): graph
          replays bind the worker's real page tables and shared graph
          buffers, which this synthetic step must not touch. At the startup
          seam no decode graph exists yet anyway — the probe runs BEFORE
          ``_warmup_cuda_graphs`` in the persistent lifecycle — and the
          full finally-restore below leaves the instance exactly as the
          capture found it;
        - drives every configured decode MoE layer at a fabricated
          round-robin uniform top-k routing through the real EP collectives
          (symmetric shapes on all ranks) by patching the layer's
          ``_gate_decode`` in its instance dict (no gate parameter changes,
          no retained probe-only attribute), and finishes the step through
          the LM head and ``sample_tokens``.

        Width semantics: GLM-5 decode attention is pure DP — every rank holds
        its own ``padding_bsz`` rows (the per-rank cap the worker configures
        via ``configure_decoding(padding_bsz=...)``), and the MoE path
        all-gathers ``num_tokens_per_rank == padding_bsz`` rows per rank, the
        exact ``set_num_tokens_per_rank`` layout the worker syncs per step.

        Named refusals (StartupDecodeProbeError): no CUDA; width above the
        configured cap; missing model or decode phase; a decode instance
        without the DSA indexer attribute (``use_dense_mla`` decode is not
        probed); missing/invalid DSA indexer geometry or disagreeing
        per-layer ``index_head_dim``; a skip_topk first layer (no carried
        top-k to reuse); no configured decode MoE layers; an EP-unconfigured,
        offloaded or partially resident (streamed) decode MoE — a streamed
        layer's experts flow through the worker's H2D copy engine, which has
        no admission-independent probe surface; a missing EP communicator in
        a multi-rank world; and no deployment page size.
        """
        if not torch.cuda.is_available():
            raise StartupDecodeProbeError(
                "startup decode probe requires CUDA; refusing to measure on CPU"
            )

        configured_cap = getattr(self, "padding_bsz", None)
        if configured_cap is None:
            raise StartupDecodeProbeError(
                "startup decode probe requires configure_decoding() to set a cap"
            )
        plan = plan_startup_decode_probe(padding_bsz, configured_cap)
        plan_bsz = plan.batch_size

        model = getattr(self, "model", None)
        if model is None:
            raise StartupDecodeProbeError(
                "startup decode probe requires a configured decode model"
            )
        if getattr(AttnWrapperBase, "phase", None) != "decode":
            raise StartupDecodeProbeError(
                "startup decode probe requires the decode phase to be live"
            )
        if getattr(GLM5ExpertWrapper, "phase", None) != "decode":
            raise StartupDecodeProbeError(
                "startup decode probe requires the expert wrappers in decode phase"
            )

        layers = list(model.model.layers)
        attn_wrappers = [
            layer.self_attn for layer in layers
            if isinstance(layer.self_attn, GLM5AttnWrapper)
        ]
        if len(attn_wrappers) != len(layers):
            raise StartupDecodeProbeError(
                "startup decode probe requires every attention module wrapped "
                "as GLM5AttnWrapper for decode"
            )

        # GLM-5 decode is the DSA path: layers carry an ``indexer`` attribute
        # (a live Glm5Indexer on full layers, None on GLM-5.2/5.3 skip_topk
        # shared layers). A ``use_dense_mla`` instance constructs no indexer
        # attribute at all and decodes on the dense FlashMLA fallback, which
        # this probe does not fabricate surfaces for.
        dsa_attr_wrappers = [
            wrapper for wrapper in attn_wrappers
            if hasattr(wrapper.module, "indexer")
        ]
        if not dsa_attr_wrappers:
            raise StartupDecodeProbeError(
                "startup decode probe covers the GLM-5 DSA decode path only; "
                "a use_dense_mla decode instance has no indexer to probe"
            )
        if len(dsa_attr_wrappers) != len(attn_wrappers):
            raise StartupDecodeProbeError(
                "startup decode probe found a mix of DSA and dense-MLA "
                "attention layers; one synthetic surface cannot serve both"
            )
        indexer_modules = [
            wrapper.module.indexer for wrapper in attn_wrappers
            if wrapper.module.indexer is not None
        ]
        if not indexer_modules:
            raise StartupDecodeProbeError(
                "startup decode probe found no live DSA indexer layer: every "
                "layer is skip_topk, so no top-k is ever computed to reuse"
            )
        if getattr(attn_wrappers[0].module, "skip_topk", False):
            raise StartupDecodeProbeError(
                "startup decode probe requires the first attention layer to "
                "compute DSA top-k; a skip_topk first layer has no carried "
                "top-k to reuse"
            )
        dsa_index_topk = 0
        dsa_index_head_dim = 0
        for indexer in indexer_modules:
            topk = int(getattr(indexer, "index_topk", 0) or 0)
            head_dim = int(getattr(indexer, "index_head_dim", 0) or 0)
            if topk <= 0 or head_dim <= 0:
                raise StartupDecodeProbeError(
                    "startup decode probe requires positive DSA indexer "
                    f"geometry, got index_topk={topk}, "
                    f"index_head_dim={head_dim}"
                )
            if dsa_index_head_dim not in (0, head_dim):
                raise StartupDecodeProbeError(
                    "startup decode probe found disagreeing DSA "
                    f"index_head_dim values ({dsa_index_head_dim} vs "
                    f"{head_dim}); one auxiliary carrier cannot serve both"
                )
            dsa_index_topk = max(dsa_index_topk, topk)
            dsa_index_head_dim = head_dim

        moe_blocks = [
            layer.mlp for layer in layers
            if getattr(getattr(layer, "mlp", None), "gate", None) is not None
            and hasattr(layer.mlp, "num_experts_per_tok")
        ]
        if not moe_blocks:
            raise StartupDecodeProbeError(
                "startup decode probe found no configured decode MoE layers"
            )
        for moe in moe_blocks:
            persistent = getattr(moe, "num_persistent_local_experts", None)
            per_rank = getattr(moe, "experts_per_rank", None)
            if persistent is None or per_rank is None:
                raise StartupDecodeProbeError(
                    "startup decode probe requires EP-configured decode MoE "
                    "layers (_configure_decode_moe has not run)"
                )
            if getattr(moe, "enable_ep_offloading", False) or persistent != per_rank:
                raise StartupDecodeProbeError(
                    "startup decode probe supports fully resident local-expert "
                    "decode MoE only; a streamed/offloaded decode MoE layer "
                    "has no resident shard to probe without the worker's H2D "
                    "streamer"
                )
            if self.world_size > 1 and getattr(moe, "comm", None) is None:
                raise StartupDecodeProbeError(
                    "startup decode probe requires the configured EP "
                    "communicator on every decode MoE layer"
                )

        device = self.engine_config.Basic_Config.device_torch

        # MLA latent geometry comes from the live attention module: the paged
        # cache stores one 1-head latent token of kv_lora_rank + qk_rope_head_dim
        # (what the selector's paged write stores and what the FA3 sparse
        # attention reads back through the selected physical token ids).
        sample_mla = attn_wrappers[0].module
        latent_dim = int(getattr(sample_mla, "kv_lora_rank", 0)) + int(
            getattr(sample_mla, "qk_rope_head_dim", 0)
        )
        if latent_dim <= 0:
            raise StartupDecodeProbeError(
                "startup decode probe requires positive MLA latent geometry"
            )

        num_layers = len(layers)
        synthetic_sids = [_GLM5_PROBE_SEQ_ID_BASE - i for i in range(plan_bsz)]

        # The decode kernels support only the deployment page size; at this
        # startup seam the worker's manager may not be installed on the
        # wrapper, so the page size comes from the ENGINE CONFIG, with the
        # live manager honored when present. The real DSA deployment asserts
        # equal primary/aux page sizes (DualKVCacheCoordinator), so one value
        # serves both synthetic regions.
        source_config = getattr(
            AttnWrapperBase.gpu_paged_kv_manager, "config", None
        )
        page_size = int(
            getattr(source_config, "page_size_tokens", 0)
            or getattr(
                getattr(self.engine_config, "Device_Paged_KV_Config", None),
                "page_size",
                0,
            )
            or 0
        )
        if page_size <= 0:
            raise StartupDecodeProbeError(
                "startup decode probe found no deployment page size"
            )

        # Aux carrier layout, from the same release geometry the deployment
        # allocates from (releases.py): a top-k-reusing checkpoint (GLM-5.2 /
        # GLM-5.3) stores page-split FP8 index keys, GLM-5/5.1 stores bf16
        # AoS. The eager writer and scorer both dispatch on the cache layout
        # (``cache_is_page_split``), so a mis-declared carrier fails loud in
        # the real kernels rather than silently probing the wrong format.
        page_split_aux = (
            getattr(self.loaded_model_config, "model_type", None)
            in SPARSE_PREFILL_MODEL_TYPES
        )
        if page_split_aux:
            aux_carrier_dim = dsa_index_head_dim + _DSA_PAGE_SPLIT_SCALE_BYTES
            aux_dtype = torch.uint8
        else:
            aux_carrier_dim = dsa_index_head_dim
            aux_dtype = torch.bfloat16

        # Synthetic decode context: the eager DSA step routes any row with
        # cache_seqlen <= index_topk to the dense all-short path, so the probe
        # makes every row MINIMALLY LONG (index_topk + 1 tokens): the fused
        # index-K writer, the top-k scorer and the selected-token sparse FA3
        # attention all execute with their real cap-width shapes.
        probe_past = dsa_index_topk
        pages_per_row = (probe_past + 1 + page_size - 1) // page_size
        write_entry = probe_past // page_size

        def _aliased_page_table() -> torch.Tensor:
            """[plan_bsz, pages_per_row] int32 over plan_bsz + 1 real pages.

            The entry holding the write position maps to that row's own page
            (1 + row); every history entry maps to the shared read-only zero
            page 0. The probe writes only the new token, so sharing history
            pages is race-free, and the synthetic caches stay O(batch) pages
            at any synthetic context length (the index_topk + 1 context would
            otherwise need O(batch * index_topk) real tokens).
            """
            table = torch.zeros(
                plan_bsz, pages_per_row, dtype=torch.int32, device=device
            )
            table[:, write_entry] = torch.arange(
                1, plan_bsz + 1, dtype=torch.int32, device=device
            )
            return table

        slot_map = {sid: row for row, sid in enumerate(synthetic_sids)}

        class _SyntheticPagedKV:
            """Minimal PRIMARY paged-KV surface for the GLM-5 DSA decode.

            The eager selector (``build_glm5_dsa_flashmla_inputs``) consumes
            exactly ``.device``, ``.config.page_size_tokens``,
            ``._gpu_page_table_manager.seq_id_to_slot`` (the batch-order slot
            build), ``update_layer_decode_new_token(..., slot_indices=...)``
            and ``get_layer_kv_with_page_table(layer_idx)``. MLA stores no V,
            so no V cache is allocated and None is returned in its place.
            """

            def __init__(self):
                self.device = device
                self.page_size = page_size
                self.config = types.SimpleNamespace(
                    page_size_tokens=page_size
                )
                self.page_table = _aliased_page_table()
                self._gpu_page_table_manager = types.SimpleNamespace(
                    seq_id_to_slot=dict(slot_map)
                )
                # One latent head per token: [L, pages, page_size, 1, latent].
                self.k_cache = torch.zeros(
                    (num_layers, plan_bsz + 1, page_size, 1, latent_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )

            def update_layer_decode_new_token(
                self,
                *,
                k_tensor,
                v_tensor,
                sequence_lengths,
                layer_idx,
                slot_indices=None,
                **_kwargs,
            ):
                if v_tensor is not None:
                    raise StartupDecodeProbeError(
                        "startup decode probe synthetic KV holds the MLA "
                        "latent only; no V cache exists to write"
                    )
                positions = sequence_lengths.to(dtype=torch.long)
                if slot_indices is None:
                    rows = torch.arange(
                        positions.shape[0], device=self.page_table.device,
                        dtype=torch.long,
                    )
                else:
                    rows = slot_indices.to(
                        dtype=torch.long, device=self.page_table.device,
                    )
                page_entries = positions // self.page_size
                offsets = positions % self.page_size
                pages = self.page_table[rows].gather(
                    1, page_entries[:, None]
                ).squeeze(1)
                self.k_cache[layer_idx, pages, offsets] = k_tensor[:, 0].to(
                    self.k_cache.dtype
                )

            def get_layer_kv_with_page_table(self, layer_idx):
                return self.k_cache[layer_idx], None, self.page_table

        class _SyntheticPagedKVAux:
            """Minimal AUXILIARY (index-key) surface for the GLM-5 DSA decode.

            The eager selector consumes exactly ``.device``,
            ``.config.page_size_tokens``, ``.config.kv_dtype`` /
            ``.config.k_head_dim`` (``aux_manager_applies_hadamard`` builds
            its meta probe from them), ``._gpu_page_table_manager
            .seq_id_to_slot`` and ``get_layer_kv_with_page_table(layer_idx)``;
            the write goes through ``write_aux_indexer_k``, which dispatches
            on the cache layout itself. Page-split FP8 zero bytes decode to
            all-zero keys and scales — finite, so the ReLU-gated scorer and
            the sparse attention stay NaN-free; bf16 AoS zeros are trivially
            finite.
            """

            def __init__(self):
                self.device = device
                self.config = types.SimpleNamespace(
                    page_size_tokens=page_size,
                    kv_dtype=aux_dtype,
                    k_head_dim=aux_carrier_dim,
                )
                self.page_table = _aliased_page_table()
                self._gpu_page_table_manager = types.SimpleNamespace(
                    seq_id_to_slot=dict(slot_map)
                )
                self.k_cache = torch.zeros(
                    (num_layers, plan_bsz + 1, page_size, 1, aux_carrier_dim),
                    dtype=aux_dtype,
                    device=device,
                )

            def get_layer_kv_with_page_table(self, layer_idx):
                return self.k_cache[layer_idx], None, self.page_table

        saved_attention_state = {
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
        # GLM-5-owned per-step class state the worker normally resets before
        # every decode step; the probe is its own step, so it sets and
        # restores all of it.
        saved_glm5_wrapper_state = {
            name: getattr(GLM5AttnWrapper, name)
            for name in (
                "_dsa_short_count",
                "_dsa_prev_topk_indices",
                "glm5_decode_primary_slot_indices",
                "glm5_decode_aux_slot_indices",
                "glm5_dsa_graph_forward_state",
                "glm5_dsa_flashmla_graph_metadata",
            )
        }
        saved_rank_token_counts = Glm5MoE._rank_token_counts
        saved_moe_ntp = [
            (moe, getattr(moe, "num_tokens_per_rank", None))
            for moe in moe_blocks
        ]
        patched_gates = []

        try:
            # Force the eager decode path for both the DSA attention and the
            # MoE: graph replays bind the worker's real page tables and
            # shared graph buffers, which this synthetic step must not touch.
            # The configured batch debug dict is restored below.
            AttnWrapperBase.batchgen_debug = {
                "glm5_dsa_mode": "eager",
                "glm5_moe_mode": "eager",
            }
            # Every synthetic row is long, so the worker's per-step short-row
            # hint is 0 and no selector falls back to a host-side reduction.
            GLM5AttnWrapper._dsa_short_count = 0
            GLM5AttnWrapper._dsa_prev_topk_indices = None
            GLM5AttnWrapper.glm5_decode_primary_slot_indices = None
            GLM5AttnWrapper.glm5_decode_aux_slot_indices = None
            GLM5AttnWrapper.glm5_dsa_graph_forward_state = None
            GLM5AttnWrapper.glm5_dsa_flashmla_graph_metadata = None
            # No rank padding in the probe: every rank's plan_bsz rows are
            # real, so dispatch masks nothing.
            Glm5MoE._rank_token_counts = None

            AttnWrapperBase.gpu_paged_kv_manager = _SyntheticPagedKV()
            AttnWrapperBase.gpu_paged_kv_manager_aux = _SyntheticPagedKVAux()
            AttnWrapperBase.cache_seqlens = torch.full(
                (plan_bsz,), probe_past + 1,
                dtype=torch.int32,
                device=device,
            )
            AttnWrapperBase.max_seqlen = probe_past + 1
            AttnWrapperBase.cur_batch = list(synthetic_sids)
            AttnWrapperBase.position_ids = torch.full(
                (plan_bsz, 1), probe_past,
                dtype=torch.long,
                device=device,
            )
            AttnWrapperBase.attention_mask = None
            # Both Host-append callbacks stay None for the probe; the eager
            # selector then appends nothing host-side.
            AttnWrapperBase.kv_append_callback = None
            AttnWrapperBase.kv_append_callback_aux = None

            # The worker syncs this scalar before every decode step; GLM-5
            # decode is pure DP, so every rank drives its full plan_bsz rows.
            for moe in moe_blocks:
                if hasattr(moe, "set_num_tokens_per_rank"):
                    moe.set_num_tokens_per_rank(plan_bsz)

            # Make the routed workload uniform without changing any gate
            # parameter or retaining a probe-only module attribute: the
            # decode gate seam is the layer's ``_gate_decode`` (the gate
            # module's ``forward`` serves prefill only).
            for moe in moe_blocks:
                gate = moe.gate
                num_experts = int(gate.num_experts)
                top_k = int(gate.num_experts_per_tok)
                original = moe.__dict__.get("_gate_decode", None)

                def _uniform_gate_decode(
                    x,
                    *,
                    _num_experts=num_experts,
                    _top_k=top_k,
                ):
                    rows = x.reshape(-1, x.shape[-1]).shape[0]
                    topk_idx = uniform_expert_assignment(
                        rows,
                        _num_experts,
                        _top_k,
                        device=x.device,
                    ).to(torch.int32)
                    topk_weight = torch.full(
                        (rows, _top_k),
                        1.0 / _top_k,
                        dtype=torch.float32,
                        device=x.device,
                    )
                    return topk_idx, topk_weight

                moe.__dict__["_gate_decode"] = _uniform_gate_decode
                patched_gates.append((moe, original))

            input_ids = torch.zeros(
                (plan_bsz, 1), dtype=torch.long, device=device
            )
            position_ids = torch.full(
                (plan_bsz, 1), probe_past,
                dtype=torch.long,
                device=device,
            )
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
            )
            logits = outputs.logits[:, -1, :]
            from batchgen.sampling import sample_tokens

            sampled_tokens = sample_tokens(
                logits, temperature=1.0, top_p=1.0, top_k=0
            )
            del sampled_tokens, logits, outputs
            del position_ids, input_ids
        finally:
            for moe, original in reversed(patched_gates):
                if original is None:
                    moe.__dict__.pop("_gate_decode", None)
                else:
                    moe.__dict__["_gate_decode"] = original
            for moe, ntp in saved_moe_ntp:
                if ntp is not None and hasattr(moe, "set_num_tokens_per_rank"):
                    moe.set_num_tokens_per_rank(ntp)
            Glm5MoE._rank_token_counts = saved_rank_token_counts
            for name, value in saved_glm5_wrapper_state.items():
                setattr(GLM5AttnWrapper, name, value)
            for name, value in saved_attention_state.items():
                setattr(AttnWrapperBase, name, value)

    def _init_decoding_padding_bsz(self, padding_bsz):
        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        max_rank_bsz = int(env_max_bsz) if env_max_bsz else padding_bsz
        # The configured per-rank decode cap; startup_decode_probe_step
        # validates its requested width against it.
        self.padding_bsz = max_rank_bsz
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
        """Load shared expert FP8 weights for decode (persistent on GPU).

        Gate and up land as the two row halves of ONE [2I, K] device
        allocation, ``shared.gate_up_weight``; ``gate_proj`` / ``up_proj`` bind
        views of it. The decode shared expert then runs gate and up as a single
        GEMM over the same bytes (GLM5ExpertWrapper._bind_fused_gate_up checks
        the aliasing), and no weight is stored twice: the footprint equals the
        two separate allocations it replaces.
        """
        device = self.engine_config.Basic_Config.device_torch
        for layer_idx in range(self.FIRST_K_DENSE, len(self.model.model.layers)):
            tensors = self.core_engine.get_tensor(f"shared_expert_{layer_idx}")
            shared = self.model.model.layers[layer_idx].mlp.shared_experts
            gate, up = tensors["gate_proj.weight"], tensors["up_proj.weight"]
            if gate.shape != up.shape or gate.dtype != up.dtype or gate.dim() != 2:
                raise RuntimeError(
                    f"Layer {layer_idx}: shared expert gate {tuple(gate.shape)}/"
                    f"{gate.dtype} and up {tuple(up.shape)}/{up.dtype} must be "
                    "same-shape 2D weights to share one gate/up allocation")
            inter = gate.shape[0]
            gate_up = torch.empty(
                (2 * inter, gate.shape[1]), dtype=gate.dtype, device=device)
            gate_up[:inter].copy_(gate)
            gate_up[inter:].copy_(up)
            shared.gate_up_weight = gate_up
            shared.gate_proj.weight.data = gate_up[:inter]
            shared.up_proj.weight.data = gate_up[inter:]
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

    def _local_routed_experts_fully_resident(self) -> bool:
        """True when this rank keeps every one of its routed experts on GPU.

        The 3D grouped-GEMM decode path requires it (see
        ``_glm5_moe_3d_blockwise_supported``), and so do the preallocated
        stacks: a partially resident rank streams the rest through the H2D
        copy engine and cannot own a full stack row per expert.
        """
        resident = getattr(self, "num_local_expert_per_layer", None)
        if resident is None:
            # configure_decoding has not run yet. The split is a deployment
            # constant (world size + EP config), so plan it instead of
            # reporting "not resident": the FIRST prefill configure asks this
            # question to decide whether this process keeps its phase
            # instances, and answering False there meant the first prefill
            # model was discarded and rebuilt at the next switch.
            _, resident = self._plan_local_expert_split()
        return resident == self.NUM_TOTAL_EXPERTS // self.world_size

    def _plan_local_expert_split(self) -> tuple[bool, int]:
        """(offloading enabled, routed experts this rank keeps resident).

        Derived from the world size and the EP configuration only, so it is
        the same answer before and after ``configure_decoding`` assigns it.
        """
        experts_per_rank = self.NUM_TOTAL_EXPERTS // self.world_size
        if self.world_size > 8:
            # Multi-node: every rank's shard is resident.
            return False, experts_per_rank
        if self.engine_config.EP_Config.enable_offloading:
            offload_ratio = self.engine_config.EP_Config.offloading_ratio
            return True, int(experts_per_rank * (1 - offload_ratio))
        configured = self.engine_config.EP_Config.num_local_expert_per_layer
        if configured is None or configured == 0:
            return False, experts_per_rank
        return False, configured

    def _bind_local_routed_expert_stacks(self, carve):
        """Point every local routed expert at a row of its layer's 3D stack.

        ``carve(shape, dtype) -> tensor`` supplies the per-layer stacks. Both
        decode paths use this: the stacks become ``moe._w3d_prealloc``, so
        ``Glm5MoE._init_fp8_blockwise_weights`` adopts them instead of
        allocating its own and copying every expert in again. That second copy
        moved this rank's whole routed-expert shard device-to-device on every
        prefill -> decode switch, for no change in the bytes.

        Returns the (destination row, pinned host source) pairs, which
        ``refill_local_routed_experts`` copies. The host sources are
        address-stable, so the pairs stay valid for the life of the stacks.
        """
        experts_per_rank = self.NUM_TOTAL_EXPERTS // self.world_size
        if not self._local_routed_experts_fully_resident():
            raise RuntimeError(
                "preallocated routed-expert stacks need every local routed "
                f"expert resident: {self.num_local_expert_per_layer}/"
                f"{experts_per_rank} per layer"
            )
        names = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
        start = self.global_rank * experts_per_rank
        first = self.core_engine.get_tensor(self.local_routed_experts[0])
        moe_layers = range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers)
        pairs = []
        for layer_idx in moe_layers:
            moe = self.model.model.layers[layer_idx].mlp
            # A1: on the DeepEP decode configuration gate and up share ONE
            # [E, 2I, H] carve so the masked m-grouped GEMM runs them in a
            # single launch. The carve is the same total bytes as the two
            # separate stacks. Glm5MoE._init_fp8_blockwise_weights decides the
            # layout from enable_deepep and adopts a 2-tuple here, a 3-tuple
            # otherwise; the two must agree, which the destination rows below
            # and that method's alias check together enforce.
            fused = bool(getattr(moe, "glm5_deepep_ll", False))
            gate_shape = first[names[0]].shape
            up_shape = first[names[1]].shape
            if fused and (tuple(gate_shape) != tuple(up_shape)):
                raise RuntimeError(
                    f"layer {layer_idx}: the fused gate/up carve needs matching "
                    f"gate {tuple(gate_shape)} and up {tuple(up_shape)} shapes"
                )
            if fused:
                inter = int(gate_shape[0])
                gate_up = carve(
                    (experts_per_rank, 2 * inter, int(gate_shape[1])),
                    first[names[0]].dtype,
                )
                down = carve(
                    (experts_per_rank, *first[names[2]].shape),
                    first[names[2]].dtype,
                )
                moe._w3d_prealloc = (gate_up, down)
                # Destination row per (expert, projection), in ``names`` order.
                def _rows(i, _gu=gate_up, _d=down, _n=inter):
                    return (_gu[i, :_n, :], _gu[i, _n:, :], _d[i])
            else:
                stacks = tuple(
                    carve((experts_per_rank, *first[name].shape), first[name].dtype)
                    for name in names
                )
                moe._w3d_prealloc = stacks

                def _rows(i, _s=stacks):
                    return tuple(s[i] for s in _s)
            for i in range(experts_per_rank):
                tensors = self.core_engine.get_tensor(
                    f"routed_expert_{layer_idx}_{start + i}"
                )
                rows = _rows(i)
                for row, name in zip(rows, names):
                    src = tensors[name]
                    if src.shape != row.shape or src.dtype != row.dtype:
                        raise RuntimeError(
                            f"routed_expert_{layer_idx}_{start + i}.{name}: host "
                            f"{tuple(src.shape)}/{src.dtype} does not match stack row "
                            f"{tuple(row.shape)}/{row.dtype}"
                        )
                    pairs.append((row, src))
                expert = moe.experts[start + i]
                expert.fp8_gate, expert.fp8_up, expert.fp8_down = rows
        return pairs

    def _expert_stack_layer_bytes(self, first, names, experts_per_rank):
        return sum(
            experts_per_rank * first[name].numel() * first[name].element_size() + 256
            for name in names
        )

    def _load_local_routed_experts_prealloc(self):
        """Load this rank's routed experts straight into per-layer 3D stacks.

        The default decode path used to upload every expert tensor into its own
        fresh allocation and then let ``_init_fp8_blockwise_weights`` copy the
        whole shard again, device-to-device, into the stacks the grouped GEMM
        actually reads. Allocating the stacks first makes the host-to-device
        copy land where the kernels want it and removes the second copy
        entirely. The copies are issued asynchronously into memory that already
        exists and are awaited once, instead of one blocking copy per tensor
        each of which also had to allocate its destination.
        """
        device = self.engine_config.Basic_Config.device_torch

        def carve(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        # Deliberately NOT stored on self: on this path the stacks belong to
        # the decode model and must die with it at the next switch. Keeping a
        # manager-level reference would pin this rank's whole routed-expert
        # shard across the prefill phase.
        pairs = self._bind_local_routed_expert_stacks(carve)
        self._copy_expert_pairs(pairs)
        logging.debug("Local routed experts loaded into preallocated 3D stacks")

    def _load_local_routed_experts_into_arena(self):
        """Load this rank's routed experts into fixed-address per-layer 3D stacks.

        Persistent decode keeps its CUDA graphs across prefill phases, so the
        stacks those graphs read must never move. They live in one VmmArena
        whose physical memory is released while prefill runs and refilled
        from the pinned host copy on the way back.
        """
        from batchgen.cuda_graph.vmm_arena import VmmArena

        experts_per_rank = self.NUM_TOTAL_EXPERTS // self.world_size
        if self.num_local_expert_per_layer != experts_per_rank:
            raise RuntimeError(
                "persistent phase instances need every local routed expert resident: "
                f"{self.num_local_expert_per_layer}/{experts_per_rank} per layer"
            )
        names = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
        first = self.core_engine.get_tensor(self.local_routed_experts[0])
        moe_layers = range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers)
        layer_bytes = self._expert_stack_layer_bytes(
            first, names, experts_per_rank
        )
        # 512 MiB chunks keep the round-up waste under 0.5 GiB of HBM.
        self._expert_arena = VmmArena(
            self.engine_config.Basic_Config.device_torch,
            layer_bytes * len(moe_layers),
            chunk_bytes=512 << 20,
        )
        self._expert_arena.map()
        self._expert_copy_pairs = self._bind_local_routed_expert_stacks(
            self._expert_arena.carve
        )
        self.refill_local_routed_experts()

    def _config_attn_module(self):
        """Replace attention modules with GLM5AttnWrapper."""
        start_time = time.perf_counter()
        # Only the decode configure registers resident attention weights
        # (configure_prefill lists every attn module in weight_copy_task, so
        # its wrappers are non-persistent and stream per forward). The phase is
        # read explicitly anyway, because the kv_b_proj release below is only
        # correct for the instance that owns the BF16 absorb pair.
        decode_phase = getattr(self.loaded_model_config, "phase", None) == "decode"
        kv_b_released = 0
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            if self.engine_config.Basic_Config.gpu_arch in ("hopper", "blackwell"):
                if self.engine_config.Basic_Config.gpu_arch == "blackwell":
                    raise NotImplementedError(
                        "GLM-5 attention on Blackwell is unsupported pending "
                        "sub-project work"
                    )
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
                attn_module.single_sequence_prefill_chunk_size = 16_384
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
            for name in self.skeleton_state_dict:
                if name.startswith(prefix) and name.endswith(postfix):
                    # Skip indexer scales
                    if ".indexer." in name:
                        continue
                    weight_dequant_scales[name[len(prefix):]] = (
                        self._device_scale(name)
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
                if decode_phase:
                    # The absorb pair now holds everything decode reads out of
                    # kv_b_proj; the FP8 weight is dead residency competing
                    # with the GPU KV pool. See release_decode_kv_b_proj.
                    kv_b_released += wrapper.release_decode_kv_b_proj()

        if kv_b_released and self.rank == 0:
            logging.info(
                "[ATTN] released decode FP8 kv_b_proj on %d layers: %.3f GiB "
                "(BF16 absorb pair is the decode authority)",
                len(self.model.model.layers),
                kv_b_released / (1024 ** 3),
            )

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
                scale = self._device_scale(prefix + name + postfix)
                if scale is not None:
                    shared_scales[name + postfix] = scale
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
                    scale = self._device_scale(prefix + name + postfix)
                    if scale is not None:
                        expert_scales[name + postfix] = scale
                layer.mlp.experts[expert_idx] = GLM5ExpertWrapper(
                    layer.mlp.experts[expert_idx], layer_idx, expert_idx,
                    self.core_engine, self.engine_config, self.model_config,
                    persistent, expert_scales, is_fp8=self.is_fp8_experts,
                )
                # Only register weights for experts that had weights loaded
                if persistent and routed_key in local_set:
                    layer.mlp.experts[expert_idx]._register_fp8_weights()

            layer.mlp._prefill_grouped_enabled = (
                getattr(self.loaded_model_config, "phase", "decode") == "prefill"
                and self.is_fp8_experts
                and not layer.mlp.shared_experts.persistent
                and all(not expert.persistent for expert in layer.mlp.experts)
            )

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

        bound = {}
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
                bound[key] = (ckpt_key, param)
                loaded += 1
                loaded_bucket[_bucket_for(key)] += 1
                if ckpt_key != key:
                    remapped += 1
                if self.rank == 0 and ("q_a_layernorm" in key or "kv_a_layernorm" in key):
                    qa_trace.append(f"LOADED: {key} (ckpt_key={ckpt_key})")
            elif key in self.skeleton_state_dict:
                param.data = self.skeleton_state_dict[key]
                bound[key] = (key, param)
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
        if self._skeleton_on_device:
            self._check_shared_skeleton_binding(bound)
            self._skeleton_bound_params = [
                (param, ckpt_key) for ckpt_key, param in bound.values()
            ]

    def _setup_fp8_scales(self):
        """Attach FP8 scale tensors to indexer and dense MLP for on-the-fly dequant."""
        for layer_idx in range(self.model_config.num_hidden_layers):
            attn = self.model.model.layers[layer_idx].self_attn
            # After wrapping, self_attn is GLM5AttnWrapper; original Glm5MLA is at .module
            inner = attn.module if hasattr(attn, 'module') else attn
            # When use_dense_mla is set, Glm5MLA skips indexer construction;
            # skip the scale attach too (no destination). GLM-5.2 "shared" layers
            # carry no indexer weights either (indexer is None) — skip them.
            if getattr(inner, "indexer", None) is not None:
                indexer = inner.indexer
                for proj, attr in [("wk", "wk_scale"), ("wq_b", "wq_b_scale")]:
                    key = f"model.layers.{layer_idx}.self_attn.indexer.{proj}.weight_scale_inv"
                    scale = self._device_scale(key)
                    if scale is not None:
                        setattr(indexer, attr, scale)
        for layer_idx in range(self.FIRST_K_DENSE):
            mlp = self.model.model.layers[layer_idx].mlp
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"model.layers.{layer_idx}.mlp.{proj}.weight_scale_inv"
                scale = self._device_scale(key)
                if scale is not None:
                    setattr(mlp, f"{proj.split('_')[0]}_scale", scale)

    def _init_fused_kernels(self):
        """Initialize and require the production DSA kernels before serving."""
        dsa_wrappers = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            wrapper = layer.self_attn
            inner = wrapper.module if hasattr(wrapper, "module") else wrapper
            if getattr(inner, "indexer", None) is not None:
                dsa_wrappers.append((layer_idx, wrapper))

        if not dsa_wrappers:
            if self.rank == 0:
                logging.info("[DSA kernels] skipped (no active indexer layers)")
            return

        import_failures = {
            **_required_dsa_model_kernel_import_failures(),
            **_required_dsa_kernel_import_failures(),
        }
        if import_failures:
            details = "; ".join(
                f"{name}: {error!r}" for name, error in import_failures.items()
            )
            raise RuntimeError(
                "GLM-5 DSA required kernel imports are unavailable before "
                f"serving: {details}"
            )

        for _, wrapper in dsa_wrappers:
            wrapper.initialize_fused_kernels()

        missing = {}
        for name, attr in (
            ("WP2 fused indexer KV projection", "_indexer_cuda_weights"),
            ("WP4 fused indexer scoring", "_fused_wqb_weights"),
        ):
            layers = [
                layer_idx
                for layer_idx, wrapper in dsa_wrappers
                if getattr(wrapper, attr, None) is None
            ]
            if layers:
                missing[name] = layers

        phase = getattr(self.loaded_model_config, "phase", None)
        if phase == "decode":
            layers = [
                layer_idx
                for layer_idx, wrapper in dsa_wrappers
                if getattr(wrapper, "_fp8_absorb_weights", None) is None
            ]
            if layers:
                missing["WP5 FP8 absorb"] = layers

        if missing:
            details = "; ".join(
                f"{name} missing on layers {layers}"
                for name, layers in missing.items()
            )
            raise RuntimeError(
                "GLM-5 DSA required kernels failed to initialize before "
                f"serving (phase={phase!r}): {details}"
            )

        if self.rank == 0:
            total = len(dsa_wrappers)
            logging.info(
                f"[DSA kernels] WP2={total}/{total}, WP4={total}/{total}, "
                f"WP5={'import-ready' if phase != 'decode' else f'{total}/{total}'}"
            )

    def _lm_head_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self):
        self.model.lm_head.register_forward_pre_hook(self._lm_head_forward_pre_hook)

    @staticmethod
    def _pack_scales(entries, dtype):
        """Flatten ``[(key, tensor), ...]`` of one dtype into one pinned buffer.

        Returns ``(staging, layout)`` where ``layout`` is
        ``[(key, offset, shape), ...]``: every tensor's elements sit
        contiguously at its offset, so one host-to-device copy of ``staging``
        followed by one reshaped slice per key reproduces each tensor exactly.
        """
        total = sum(tensor.numel() for _, tensor in entries)
        # Pinning is what makes the single copy async; it is also only
        # meaningful with a CUDA runtime present.
        staging = torch.empty(
            total, dtype=dtype, pin_memory=torch.cuda.is_available()
        )
        layout = []
        offset = 0
        for key, tensor in entries:
            count = tensor.numel()
            staging[offset:offset + count] = tensor.reshape(-1)
            layout.append((key, offset, tuple(tensor.shape)))
            offset += count
        return staging, layout

    def _upload_skeleton_scales(self):
        """One host-to-device copy for every host-resident FP8 scale.

        Each phase configure used to move these per tensor: the routed experts
        alone are num_layers x num_experts x 3 separate `.to(device)` calls for
        a very small total payload, i.e. almost pure launch overhead, paid again
        on every switch in both directions although the scales never change.
        They are packed per dtype into one pinned buffer, copied once, and
        handed out as views.

        Tensors already on the target device are returned as themselves, not
        repacked: that keeps `share_skeleton_on_device()` (the persistent phase
        path) handing out exactly the objects it handed out before.
        """
        device = self.engine_config.Basic_Config.device_torch
        target = torch.device(device)
        scales = {}
        by_dtype = {}
        for key, tensor in self.skeleton_state_dict.items():
            if "weight_scale_inv" not in key:
                continue
            if tensor.device == target:
                scales[key] = tensor
                continue
            by_dtype.setdefault(tensor.dtype, []).append((key, tensor))

        uploaded = 0
        for dtype, entries in by_dtype.items():
            staging, layout = self._pack_scales(entries, dtype)
            packed = staging.to(device, non_blocking=True)
            for key, offset, shape in layout:
                count = 1
                for dim in shape:
                    count *= dim
                scales[key] = packed[offset:offset + count].view(shape)
            uploaded += len(layout)
        if by_dtype:
            torch.cuda.current_stream().synchronize()
        if self.rank == 0:
            logging.info(
                "[SCALES] %d skeleton scale tensors on device (%d uploaded in "
                "%d bulk copies)",
                len(scales),
                uploaded,
                len(by_dtype),
            )
        return scales

    def _device_scale(self, key):
        """Device copy of skeleton scale ``key``, or None when absent."""
        if self._device_scales is None:
            self._device_scales = self._upload_skeleton_scales()
        return self._device_scales.get(key)

    def _extract_dequantize_scale(self):
        self.dequant_scale = {}
        for key, param in self.skeleton_state_dict.items():
            if "weight_scale_inv" in key:
                self.dequant_scale[key] = param
