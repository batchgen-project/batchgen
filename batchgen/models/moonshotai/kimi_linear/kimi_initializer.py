# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear Initializer for BatchGen.

All-BF16 48B-A3B hybrid model: 20 KDA (linear attention) + 7 NoPE-MLA layers,
256 routed experts (BF16) + 1 shared. Adds a new "kda_attn" module type for
KDA attention weights. KV storage holds MLA latent KV (576/token); KDA state
is managed Python-side (KimiLinearKDAWrapper pools), outside KV_Storage.
"""

import logging
import os
from typing import Tuple

import torch

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.config.model_registry import load_config

from .config import KimiLinearConfig, require_num_routed_experts
from .planner import KimiLinearPlanner

try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    from batchgen.models.engine_loader import core_engine as loader_module
    core_engine = loader_module.batchgen


class KimiLinearInitializer:
    """Initialize Kimi-Linear model for BatchGen inference."""

    def __init__(self, input_arguments):
        # Unified BatchGen config (registry resolves local config.json,
        # including linear_attn_config).
        self.batchgen_config = load_config(input_arguments.huggingface_ckpt_name)

        # Name-pattern lookup drops data-driven fields (linear_attn_config);
        # the checkpoint config.json under --cache-dir is authoritative.
        _ckpt_dir = getattr(input_arguments, "cache_dir", None)
        if (
            getattr(self.batchgen_config, "linear_attn_config", None) is None
            and _ckpt_dir
        ):
            cfg_json = os.path.join(_ckpt_dir, "config.json")
            if os.path.isfile(cfg_json):
                self.batchgen_config = KimiLinearConfig.from_json(cfg_json)
                self.batchgen_config._name_or_path = (
                    input_arguments.huggingface_ckpt_name
                )

        # Kimi-K3 must reach KimiLinearConfig.from_json — it flattens
        # `text_config` and stamps model_type="kimi_k3". If it did not, the
        # name-pattern registry shortcut (model_registry.py:232-238) has handed
        # us KimiLinearConfig() with its 48B DEFAULTS, which builds a model that
        # loads, runs, and is wrong.
        self.is_k3 = self.batchgen_config.model_type == "kimi_k3"
        if "kimi-k3" in input_arguments.huggingface_ckpt_name.lower() \
                and not self.is_k3:
            raise RuntimeError(
                "Kimi-K3 identifier "
                f"{input_arguments.huggingface_ckpt_name!r} resolved to a "
                f"config with model_type={self.batchgen_config.model_type!r}. "
                f"Pass --cache-dir pointing at the checkpoint directory so its "
                "config.json is read; there is no K3 default config."
            )

        # Model instantiation config (same class; structure from config.json).
        self.loaded_model_config = self.batchgen_config
        self.loaded_model_config._name_or_path = input_arguments.huggingface_ckpt_name

        self.host_kv_cache_size = input_arguments.host_kv_cache_size
        self.host_kv_cache_byte_size = input_arguments.host_kv_cache_size * (1024**3)
        self.global_kv_cache_size_gb = input_arguments.global_host_kv_cache_size_gb

        self.local_rank = input_arguments.local_rank
        self.global_rank = input_arguments.global_rank
        self.world_size = input_arguments.world_size
        # hugetlbfs opt-in is not wired for kimi-linear yet; enabling it is a
        # dedicated follow-up PR (policy: no new server-side env guards here).
        self.enable_hugetlbfs = False

        self.model_config = self._parse_model_config()

        self.engine_config = EngineConfig()
        self.engine_config = self._set_basic_config(self.engine_config, input_arguments)
        self._default_engine_config()
        self.planner = KimiLinearPlanner(is_k3=self.is_k3)
        self.engine_config = self.planner.generate_config(self.engine_config)
        if self.global_rank == 0:
            logging.info(f"Engine config after planning: {self.engine_config}")

        self.shm_name = input_arguments.shm_name
        self.tensor_meta_shm_name = input_arguments.tensor_meta_shm_name

    def _set_basic_config(self, engine_config: EngineConfig, args) -> EngineConfig:
        """Basic engine config: all-BF16 weights and KV, hybrid module types."""
        engine_config.Basic_Config.device = args.device
        engine_config.Basic_Config.device_torch = torch.device(f"cuda:{args.device}")

        engine_config.Basic_Config.weight_dtype = "bfloat16"
        engine_config.Basic_Config.weight_dtype_torch = torch.bfloat16

        kv_dtype = getattr(args, "kv_dtype", None)
        if kv_dtype and kv_dtype.lower() not in ("bf16", "bfloat16"):
            logging.warning(
                f"Kimi-Linear KV is BF16 — ignoring kv_dtype={kv_dtype}"
            )
        engine_config.Basic_Config.kv_dtype = "bfloat16"
        engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16

        engine_config.Basic_Config.activation_dtype = "bfloat16"
        engine_config.Basic_Config.activation_dtype_torch = torch.bfloat16

        # Module types: NoPE-MLA attention + KDA attention + experts
        engine_config.Basic_Config.module_types = [
            "attn", "kda_attn", "routed_expert", "shared_expert",
        ]

        engine_config.Basic_Config.padding_length = args.padding_length
        engine_config.Basic_Config.max_decoding_length = args.max_decoding_length
        engine_config.Basic_Config.world_size = args.world_size
        engine_config.Basic_Config.rank = args.rank
        engine_config.Basic_Config.num_queries = getattr(args, "num_queries", 1)
        engine_config.Basic_Config.num_threads = 0

        gpu_arch = getattr(args, "gpu_arch", "hopper")
        if gpu_arch and gpu_arch.lower() not in ["hopper", "ampere", "blackwell"]:
            raise ValueError("Currently gpu_arch must be 'hopper', 'ampere', or 'blackwell'")
        engine_config.Basic_Config.gpu_arch = gpu_arch.lower() if gpu_arch else "hopper"

        return engine_config

    def _default_engine_config(self):
        """KV storage layout + module weight shapes for the C++ engine."""
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        logging.info(
            f"Current device total memory: {props.total_memory / (1024**3):.2f} GB"
        )

        cfg = self.batchgen_config
        hidden_size = cfg.hidden_size                    # 48B 2304 / K3 7168
        moe_intermediate = cfg.moe_intermediate_size     # 48B 1024 / K3 3072
        kv_lora_rank = cfg.kv_lora_rank                  # 512 (both)
        qk_rope_head_dim = cfg.qk_rope_head_dim          # 64  (both)
        num_heads = cfg.num_attention_heads              # 48B 32   / K3 96
        qk_nope_head_dim = cfg.qk_nope_head_dim          # 128 (both)
        v_head_dim = cfg.v_head_dim                      # 128 (both)
        compressed_kv_dim = kv_lora_rank + qk_rope_head_dim  # 576 (both)

        lac = getattr(cfg, "linear_attn_config", None) or {}
        kda_num_heads = lac.get("num_heads", 32)         # 48B 32   / K3 96
        kda_head_dim = lac.get("head_dim", 128)          # 128 (both)
        kda_proj = kda_num_heads * kda_head_dim          # 48B 4096 / K3 12288
        conv_w = lac.get("short_conv_kernel_size", 4)

        # ---- KV cache (MLA latent KV per token; KDA layers never append) ----
        self.engine_config.KV_Storage_Config.reserved_length = (
            self.engine_config.Basic_Config.padding_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * compressed_kv_dim
            * torch.finfo(self.engine_config.Basic_Config.kv_dtype_torch).bits // 8
        )
        self.engine_config.KV_Storage_Config.num_host_slots = (
            self.host_kv_cache_byte_size
            // self.engine_config.KV_Storage_Config.slot_byte_size
            // self.model_config.num_hidden_layers
        )
        self.engine_config.KV_Storage_Config.storage_byte_size = (
            self.host_kv_cache_byte_size
        )
        logging.info(
            f"Number of host kv slots: "
            f"{self.engine_config.KV_Storage_Config.num_host_slots}"
        )

        self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
            * (
                self.engine_config.Basic_Config.max_decoding_length
                + self.engine_config.Basic_Config.padding_length
            )
        )

        # ---- Module shapes ----
        if self.is_k3:
            # K3 shapes live next to the K3 name map so the two cannot drift.
            # reconcile_k3_checkpoint proves sum(shape x dtype) equals the
            # checkpoint's per-tensor shapes and dtypes exactly (verified: 0
            # delta and 0 tensor mismatches against the released
            # 1,560,860,324,864 B checkpoint, all 497,220 tensors).
            # if/else, not an early return: an early return would silently skip
            # anything appended to this method later on the K3 path only.
            from .k3.tensor_map import k3_module_shapes, validate_k3_config

            validate_k3_config(cfg)
            (
                self.engine_config.GPU_Buffer_Config.module_shapes,
                self.engine_config.GPU_Buffer_Config.weight_dtypes,
                self.engine_config.GPU_Buffer_Config.tensor_dtypes,
            ) = k3_module_shapes(cfg)
        else:
            q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
            self.engine_config.GPU_Buffer_Config.module_shapes = {
                # NoPE-MLA (q_lora_rank=None → direct q_proj)
                "attn": {
                    "q_proj.weight": [num_heads * q_head_dim, hidden_size],
                    "kv_a_proj_with_mqa.weight": [compressed_kv_dim, hidden_size],
                    "kv_a_layernorm.weight": [kv_lora_rank],
                    "kv_b_proj.weight": [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank],
                    "o_proj.weight": [hidden_size, num_heads * v_head_dim],
                },
                # KDA attention
                "kda_attn": {
                    "q_proj.weight": [kda_proj, hidden_size],
                    "k_proj.weight": [kda_proj, hidden_size],
                    "v_proj.weight": [kda_proj, hidden_size],
                    "q_conv1d.weight": [kda_proj, 1, conv_w],
                    "k_conv1d.weight": [kda_proj, 1, conv_w],
                    "v_conv1d.weight": [kda_proj, 1, conv_w],
                    "A_log": [kda_num_heads],
                    "f_a_proj.weight": [kda_head_dim, hidden_size],
                    "f_b_proj.weight": [kda_proj, kda_head_dim],
                    "dt_bias": [kda_proj],
                    "b_proj.weight": [kda_num_heads, hidden_size],
                    "g_a_proj.weight": [kda_head_dim, hidden_size],
                    "g_b_proj.weight": [kda_proj, kda_head_dim],
                    "o_norm.weight": [kda_head_dim],
                    "o_proj.weight": [hidden_size, kda_proj],
                },
                # BF16 routed experts (w1/w3: gate/up, w2: down)
                "routed_expert": {
                    "w1.weight": [moe_intermediate, hidden_size],
                    "w2.weight": [hidden_size, moe_intermediate],
                    "w3.weight": [moe_intermediate, hidden_size],
                },
                # BF16 shared expert
                "shared_expert": {
                    "gate_proj.weight": [moe_intermediate, hidden_size],
                    "up_proj.weight": [moe_intermediate, hidden_size],
                    "down_proj.weight": [hidden_size, moe_intermediate],
                },
            }

            self.engine_config.GPU_Buffer_Config.weight_dtypes = {
                "attn": torch.bfloat16,
                "kda_attn": torch.bfloat16,
                "routed_expert": torch.bfloat16,
                "shared_expert": torch.bfloat16,
            }

            # Per-tensor dtype overrides.
            # A_log and dt_bias are F32 in the 48B checkpoint; o_norm.weight is
            # NOT — it ships BF16[128] (256 B), verified from the 48B shard
            # headers. Declaring it float32 sized the GPU slot at 512 B, which
            # is a live 256 B under-copy the moment `kda_attn` becomes a
            # streamed ring: blocking_copy_ writes the host byte size with no
            # bound check (HtoD_Engine.cu:232-238), leaving the slot's second
            # half at its zeros init and reinterpreting two BF16 values as one
            # F32. Inert today only because no `kda_attn` ring is ever allocated
            # (num_prefill_module_buffer has no such key, base_planner.py:90-94)
            # and the resident path reads the checkpoint dtype from the host
            # side instead.
            self.engine_config.GPU_Buffer_Config.tensor_dtypes = {
                "attn": {
                    "kv_a_layernorm.weight": torch.bfloat16,
                },
                "kda_attn": {
                    "A_log": torch.float32,
                    "dt_bias": torch.float32,
                },
            }

    def _parse_model_config(self) -> ModelConfig:
        cfg = self.batchgen_config
        model_config = ModelConfig()
        model_config.model_type = cfg.model_type
        model_config.num_hidden_layers = cfg.num_hidden_layers
        model_config.num_local_experts = require_num_routed_experts(cfg)
        model_config.num_attention_heads = cfg.num_attention_heads
        model_config.num_key_value_heads = cfg.num_key_value_heads
        model_config.head_dim = getattr(cfg, "head_dim", 128)
        model_config.compressed_kv_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        return model_config

    def Init(self, weights_storage) -> Tuple:
        """Initialize the core engine. Returns
        (core_engine, engine_config, model_config, loaded_model_config)."""
        try:
            torch.cuda.set_device(self.local_rank)
            if self.global_rank == 0:
                logging.info(f"Engine config: {self.engine_config}")

            self.core_engine = core_engine(
                self.engine_config, self.model_config, weights_storage
            )
            logging.info("Core engine created")
            self.core_engine.Init()
            logging.info("Core engine initialized")
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise e

        return (
            self.core_engine,
            self.engine_config,
            self.model_config,
            self.loaded_model_config,
        )
