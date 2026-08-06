# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear / Kimi-K3 Parameter Server for BatchGen.

TWO checkpoint families share this class — and therefore share the dispatch at
`server/worker_manager.py:806`, which already routes both and needs no change.

Kimi-Linear-48B — plain HF safetensors, BF16, no name prefix:
    model.layers.{L}.self_attn.{name}                          → MLA / KDA layers
    model.layers.{L}.block_sparse_moe.experts.{E}.w{1,2,3}.weight
    model.layers.{L}.block_sparse_moe.shared_experts.{...}

Kimi-K3 (2.8T) — `language_model.` prefix, MXFP4-packed routed experts:
    language_model.model.layers.{L}.self_attn.{name}
    language_model.model.layers.{L}.block_sparse_moe.experts.{E}
                                    .w{1,2,3}.{weight_packed,weight_scale}
    language_model.model.layers.{L}.block_sparse_moe.shared_experts.{...}
    vision_tower.* / mm_projector.*  → explicitly ignored (168 tensors)

The K3 name map, module shapes and the checkpoint-completeness proof live in
`k3/tensor_map.py`; this file only dispatches, sizes the reservation, and runs
the reconciliation gate.

BatchGen module mapping (both families):
    attn_{L}              → NoPE-MLA layers
    kda_attn_{L}          → KDA layers
    routed_expert_{L}_{E} → routed experts (BF16 in the 48B, MXFP4 in K3)
    shared_expert_{L}     → BF16 shared expert (MoE layers)

Skeleton (NOT in map; loaded by the PSM from skeleton_state_dict):
    norms, router gate (+ e_score_correction_bias), dense layer-0 MLP,
    embeddings, final norm, lm_head; K3 adds the AttnRes proj/norm pairs and the
    LatentMoE down/up/norm projections.
"""

import gc
import json
import logging
import os
import shutil
import uuid

import torch
from tqdm import trange

from batchgen.config.model_registry import load_config
from batchgen.ckpt_converter.ckpt_converter import ckpt_converter

from .config import require_num_routed_experts

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server


_MLA_ATTN_TENSOR_NAMES = [
    "q_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
]

_KDA_ATTN_TENSOR_NAMES = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "dt_bias",
    "b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
]

_SHARED_EXPERT_TENSOR_NAMES = [
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
]

_ROUTED_EXPERT_TENSOR_NAMES = [
    "w1.weight",
    "w2.weight",
    "w3.weight",
]

# Kimi-Linear-48B: ~96 GiB BF16 total; 110 GiB with buffer. K3 does NOT get a
# constant — it is computed from the checkpoint index (k3_shm_byte_size).
_KIMI_LINEAR_48B_BYTE_SIZE = 110 * 1024 * 1024 * 1024


def _detect_kimi_family(model_name, cache_dir):
    """Return "kimi_k3" or "kimi_linear", from the CHECKPOINT'S OWN config.json.

    NOT from `load_config(model_name).model_type`: for an HF-style identifier the
    registry takes the name-pattern shortcut (`model_registry.py:232-238`) and
    returns `CONFIG_REGISTRY["kimi_k3"]()` — which is `KimiLinearConfig()` with
    its **48B defaults**, whose `model_type` field is the literal string
    "kimi_linear" (`config.py:48`). Dispatching on that would silently select the
    48B name lists for a K3 checkpoint: zero routed-expert matches and 494,592
    tensors promoted to skeleton.

    The identifier is used for ONE thing: deciding whether a MISSING config.json
    is fatal. K3 cannot be served without it, the 48B can. (An identifier ↔
    checkpoint cross-check was considered and dropped: `worker_manager.py:806`
    already requires "kimi-linear" or "kimi-k3" in the name to construct this
    class at all, and the only reachable mismatch — a K3 checkpoint reached
    under a 48B-looking name — is caught three lines later by the `model_type`
    assertion in `__init__`. It would have rejected a legitimately-named staged
    directory for nothing.)
    """
    cfg_json = os.path.join(cache_dir, "config.json") if cache_dir else None
    if cfg_json and os.path.isfile(cfg_json):
        with open(cfg_json, "r") as handle:
            raw = json.load(handle)
        if raw.get("model_type") == "kimi_k3" or "text_config" in raw:
            return "kimi_k3"
        if raw.get("model_type") == "kimi_linear":
            return "kimi_linear"
        raise ValueError(
            f"{cfg_json}: model_type={raw.get('model_type')!r} is neither "
            "'kimi_k3' nor 'kimi_linear'. KimiLinear_Parameter_Server "
            "serves only those two families."
        )

    if "kimi-k3" in model_name.lower():
        raise FileNotFoundError(
            "Kimi-K3 requires an on-disk config.json under --cache-dir "
            f"(looked for {cfg_json}). Every K3 feature switch "
            "(routed_expert_hidden_size, attn_res_block_size, q_lora_rank, "
            "use_full_rank_gate, ...) is Optional/None in the built-in "
            "defaults, and a miss on any one of them loads a model that "
            "runs and is wrong."
        )
    return "kimi_linear"


class KimiLinear_Parameter_Server:
    """Parameter server for Kimi-Linear (BF16) and Kimi-K3 (MXFP4 experts)."""

    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir,
                 enable_hugetlbfs, enable_memfd=False):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        self.enable_memfd = enable_memfd

        self.family = _detect_kimi_family(huggingface_ckpt_name, cache_dir)

        self.model_config = load_config(huggingface_ckpt_name)
        # Name-pattern lookup drops linear_attn_config; config.json in
        # cache_dir is authoritative for the KDA/MLA layer split. Mandatory for
        # K3 (every K3 switch is absent from the built-in defaults).
        if (
            getattr(self.model_config, "linear_attn_config", None) is None
            or self.family == "kimi_k3"
        ) and cache_dir:
            cfg_json = os.path.join(cache_dir, "config.json")
            if os.path.isfile(cfg_json):
                from .config import KimiLinearConfig

                self.model_config = KimiLinearConfig.from_json(cfg_json)
        if self.family == "kimi_k3" and self.model_config.model_type != "kimi_k3":
            raise RuntimeError(
                "Kimi-K3 checkpoint resolved to a config whose model_type is "
                f"{self.model_config.model_type!r}. KimiLinearConfig.from_json "
                "must be reached for K3 — it flattens text_config and stamps "
                "model_type='kimi_k3'; the 48B defaults were loaded instead."
            )
        self.num_layers = self.model_config.num_hidden_layers  # 27
        self.num_experts = require_num_routed_experts(self.model_config)
        self.first_k_dense_replace = self.model_config.first_k_dense_replace  # 1

        free_memory, total_memory = torch.cuda.mem_get_info()
        logging.info(
            f"Python PM instantiation ({self.family}): GPU 0 free memory: "
            f"{free_memory / 1024**3:.2f} GB / {total_memory / 1024**3:.2f} GB"
        )

    def _is_kda_layer(self, layer_idx: int) -> bool:
        return self.model_config.is_kda_layer(layer_idx)

    def _shm_byte_size(self) -> int:
        """Shm reservation. Constant for the 48B; computed for K3.

        K3 needs 1,560,860,718,080 B = 1453.66 GiB — 13.2x the 48B constant.
        Reserving the constant would SIGBUS mid-load (the C++ side memsets and
        from_blobs into the mapping); over-reserving is the OOM-kill class
        recorded in bug_log.md:426. Neither may be guessed.
        """
        if self.family == "kimi_linear":
            return _KIMI_LINEAR_48B_BYTE_SIZE
        from .k3.tensor_map import k3_shm_byte_size

        return k3_shm_byte_size(self.cache_dir)

    def Init(self):
        self._parse_state_dict()

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs, self.enable_memfd)

        byte_size = self._shm_byte_size()

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free / 1024**3:.2f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size is not enough for {self.family}. "
                f"Required: {byte_size} ({byte_size / 1024**3:.2f} GiB), "
                f"Available: {free}. Remount: mount -o remount,"
                f"size={int(byte_size / 1024**3) + 64}G /dev/shm (and set "
                "/sys/kernel/mm/transparent_hugepage/shmem_enabled to 'always' "
                "first, or page tables cost ~26 GB across 8 workers)."
            )

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(f"Byte size: {byte_size}")

        # Plain HF safetensors — convert to BatchGen binary format on first run
        # (cached under <cache_dir>/converted_ckpt afterwards).
        if self.converted_ckpt_dir is None or not os.path.isdir(self.converted_ckpt_dir):
            converter = ckpt_converter()
            self.converted_ckpt_dir = converter.convert_model_directory(
                self.cache_dir, marlin=False
            )
        else:
            logging.info(f"Using pre-converted checkpoint: {self.converted_ckpt_dir}")

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            str(self.converted_ckpt_dir),
            self.state_dict_name_map,
        )
        return self.shm_name, self.tensor_meta_shm_name

    def _parse_state_dict(self):
        if self.family == "kimi_k3":
            self._parse_state_dict_k3()
        else:
            self._parse_state_dict_48b()

        logging.info(
            f"state_dict_name_map ({self.family}): "
            f"{len(self.state_dict_name_map):,} entries "
            f"(kda_attn={len(self.weight_copy_task['kda_attn'])} layers, "
            f"attn={len(self.weight_copy_task['attn'])} layers, "
            f"shared_expert={len(self.weight_copy_task['shared_expert'])}, "
            f"routed_expert={len(self.weight_copy_task['routed_expert']):,} modules)"
        )
        gc.collect()
        torch.cuda.empty_cache()

    def _parse_state_dict_k3(self):
        """K3: build the name map, then PROVE the partition is total.

        The reconciliation is a mandatory startup gate, not just a test. Without
        it a checkpoint tensor with no destination is silently promoted to
        skeleton_state_dict_ (Parameter_Server.cpp:368), and a wrong shape or
        dtype silently overruns its GPU slot (HtoD_Engine.cu:232-238 has no
        bound check).

        `use_shard_headers=True` is passed explicitly even though it is the
        default: index mode compares only names and the AGGREGATE byte total, so
        a byte-neutral shape error — e.g. o_proj declared [12288, 7168] instead
        of [7168, 12288] — passes it, allocates a correctly-sized slot, copies
        cleanly and hands the GEMM a transposed weight. Header mode compares
        shape and dtype per tensor. Measured on the released checkpoint: 6.1 s
        (vs 2.5 s), against a multi-hour conversion and a 1.5 TB load. The
        shards are necessarily mounted here — the converter is about to read
        them.
        """
        from .k3.tensor_map import (
            build_k3_state_dict_name_map,
            reconcile_k3_checkpoint,
        )

        self.state_dict_name_map, self.weight_copy_task = (
            build_k3_state_dict_name_map(self.model_config)
        )
        report = reconcile_k3_checkpoint(
            self.cache_dir, self.model_config, use_shard_headers=True,
        )
        logging.info("\n" + report.render())
        report.raise_for_status()

    def _parse_state_dict_48b(self):
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["kda_attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        for layer_idx in trange(self.num_layers, desc="Parsing state_dict"):
            # --- Attention: MLA vs KDA by layer type ---
            if self._is_kda_layer(layer_idx):
                for name in _KDA_ATTN_TENSOR_NAMES:
                    full = f"model.layers.{layer_idx}.self_attn.{name}"
                    self.state_dict_name_map[full] = {
                        "module_key": f"kda_attn_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["kda_attn"].append(f"kda_attn_{layer_idx}")
            else:
                for name in _MLA_ATTN_TENSOR_NAMES:
                    full = f"model.layers.{layer_idx}.self_attn.{name}"
                    self.state_dict_name_map[full] = {
                        "module_key": f"attn_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # --- MoE (layer 0 is dense) ---
            if layer_idx >= self.first_k_dense_replace:
                for name in _SHARED_EXPERT_TENSOR_NAMES:
                    full = (
                        f"model.layers.{layer_idx}.block_sparse_moe."
                        f"shared_experts.{name}"
                    )
                    self.state_dict_name_map[full] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    f"shared_expert_{layer_idx}"
                )

                for expert_idx in range(self.num_experts):
                    for name in _ROUTED_EXPERT_TENSOR_NAMES:
                        full = (
                            f"model.layers.{layer_idx}.block_sparse_moe."
                            f"experts.{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[full] = {
                            "module_key": f"routed_expert_{layer_idx}_{expert_idx}",
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(
                        f"routed_expert_{layer_idx}_{expert_idx}"
                    )
