# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 parameter server for BatchGen.

Handles checkpoint loading and tensor name mapping for zai-org/GLM-5-FP8.
Standalone — no cross-model imports.

Key differences from DeepSeek:
- 78 layers (vs 61), 256 experts, first_k_dense_replace=3
- kv_a_proj_with_mqa naming (same as DeepSeek)
- Indexer tensors (wk, wq_b, k_norm, weights_proj) — kept in skeleton (BF16/FP8 mixed)
- e_score_correction_bias in MoE gate
- MTP layer at index 78 (eh_proj, enorm, hnorm, shared_head.norm)
- byte_size: ~760 GB (FP8 experts) or ~1400 GB (BF16 experts)
"""

import sys as _diag_sys
import time as _diag_time
def _diag(msg):
    print(f"[DIAG {_diag_time.time():.3f}] glm5_ps_import: {msg}", flush=True)
    _diag_sys.stdout.flush()

_diag("start")
import gc
import json
import logging
import os
import shutil
from multiprocessing import Process
_diag("stdlib done")

import torch
_diag("torch done")
from safetensors.torch import load_file
_diag("safetensors done")
from tqdm import tqdm, trange
_diag("tqdm done")

from .model import Glm5ForCausalLM
_diag("model (Glm5ForCausalLM) done")
from batchgen.config.batchgen_model_config import BatchGenModelConfig
_diag("model_registry done")

try:
    from batchgen.core_engine import Parameter_Server
    _diag("core_engine (prebuilt) done")
except ImportError:
    _diag("core_engine ImportError -> engine_loader JIT")
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server
    _diag("engine_loader done")


class GLM5_Parameter_Server:
    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir, enable_hugetlbfs, enable_memfd=False):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        self.enable_memfd = enable_memfd
        # Single resolved internal config (checkpoint-backed GLM5Config/GLM52Config).
        # Used both for metadata reads and to build the model graph — GLM-5 no
        # longer uses an HF transformers.PretrainedConfig.
        self.model_config = BatchGenModelConfig.resolve(huggingface_ckpt_name, cache_dir)
        self.hf_config = self.model_config
        self.hf_config._name_or_path = huggingface_ckpt_name

        free_memory, total_memory = torch.cuda.mem_get_info()
        logging.info(
            f"Python PM instantiation: GPU 0 free memory: "
            f"{free_memory / 1024**3:.1f} GB / {total_memory / 1024**3:.1f} GB"
        )

    def Init(self):
        free_memory, total_memory = torch.cuda.mem_get_info()
        logging.info(
            f"GPU 0 free mem pm start Init: {free_memory / 1024**3:.1f} GB / "
            f"{total_memory / 1024**3:.1f} GB"
        )

        self._parse_state_dict()

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs, self.enable_memfd)

        # GLM-5-FP8: FP8 experts (~675 GB) + FP8 attn + rest ≈ 700 GB
        # GLM-5: BF16 experts (~1350 GB) + FP8 attn + rest ≈ 1380 GB
        if "fp8" in self.huggingface_ckpt_name.lower():
            byte_size = 760 * 1024 * 1024 * 1024
        else:
            byte_size = 1400 * 1024 * 1024 * 1024

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free / 1024**3:.1f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size not enough. Required: {byte_size / 1024**3:.0f} GB, "
                f"Available: {free / 1024**3:.0f} GB. "
                f"Please clear /dev/shm or increase size."
            )

        import uuid
        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(f"Tensor meta shared memory name: {self.tensor_meta_shm_name}")
        logging.info(f"Byte size: {byte_size}")

        # Convert checkpoint files
        from batchgen.ckpt_converter.ckpt_converter import ckpt_converter
        converter = ckpt_converter()
        self.converted_ckpt_dir = converter.convert_model_directory(self.cache_dir)

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            self.converted_ckpt_dir,
            self.state_dict_name_map,
        )
        inventory = self._validate_loaded_prefill_host_weight_inventory(byte_size)
        shm_objects = self._prefill_host_backing_inventory(byte_size)
        logging.info("[PREFILL_HOST_WEIGHT_INVENTORY] %s", json.dumps({
            "init_complete": True,
            "host_backed": True,
            "backing": shm_objects[0]["backend"],
            "shm_objects": shm_objects,
            "expected_module_counts": inventory["expected_module_counts"],
            "actual_module_counts": inventory["actual_module_counts"],
            "mapped_tensor_counts": inventory["mapped_tensor_counts"],
            "mapped_tensors_total": inventory["mapped_tensors_total"],
            "mapped_weight_bytes": inventory["mapped_weight_bytes"],
            "mapped_byte_range": inventory["mapped_byte_range"],
            "tensor_metadata_validated": True,
            "all_expected_bulk_tensors_present": True,
            "scale_metadata_offloaded": False,
            "tensors_per_shared_expert": 3,
            "tensors_per_routed_expert": 3,
        }, separators=(",", ":")))
        return self.shm_name, self.tensor_meta_shm_name

    def _prefill_host_backing_inventory(self, byte_size):
        objects = []
        hugepage_path = os.path.join(
            "/dev/hugepages", self.shm_name.lstrip("/")
        )
        if self.enable_hugetlbfs and os.path.exists(hugepage_path):
            size_bytes = os.path.getsize(hugepage_path)
            if size_bytes < byte_size:
                raise RuntimeError(
                    "GLM-5 prefill hugetlbfs weight backing is undersized: "
                    f"path={hugepage_path}, size={size_bytes}, expected>={byte_size}"
                )
            objects.append({
                "kind": "weights",
                "backend": "hugetlbfs",
                "name": self.shm_name,
                "path": hugepage_path,
                "size_bytes": size_bytes,
            })
        elif self.enable_memfd:
            fd = self.parameter_server.weights_memfd_fd()
            if fd < 0:
                raise RuntimeError(
                    "GLM-5 prefill memfd weight backing has no live file descriptor"
                )
            path = f"/proc/self/fd/{fd}"
            target = os.readlink(path)
            stat = os.fstat(fd)
            if "memfd:batchgen_weights" not in target or stat.st_size < byte_size:
                raise RuntimeError(
                    "GLM-5 prefill memfd weight backing is invalid: "
                    f"target={target!r}, size={stat.st_size}, expected>={byte_size}"
                )
            objects.append({
                "kind": "weights",
                "backend": "memfd",
                "fd": fd,
                "path": path,
                "target": target,
                "size_bytes": stat.st_size,
            })
        else:
            path = os.path.join("/dev/shm", self.shm_name.lstrip("/"))
            size_bytes = os.path.getsize(path)
            if size_bytes < byte_size:
                raise RuntimeError(
                    "GLM-5 prefill POSIX-SHM weight backing is undersized: "
                    f"path={path}, size={size_bytes}, expected>={byte_size}"
                )
            objects.append({
                "kind": "weights",
                "backend": "posix_shm",
                "name": self.shm_name,
                "path": path,
                "size_bytes": size_bytes,
            })

        metadata_path = os.path.join(
            "/dev/shm", self.tensor_meta_shm_name.lstrip("/")
        )
        metadata_size = os.path.getsize(metadata_path)
        if metadata_size <= 0:
            raise RuntimeError(
                "GLM-5 prefill tensor-metadata shared memory is empty: "
                f"{metadata_path}"
            )
        objects.append({
            "kind": "tensor_metadata",
            "backend": "posix_shm",
            "name": self.tensor_meta_shm_name,
            "path": metadata_path,
            "size_bytes": metadata_size,
        })
        return objects

    def _expected_prefill_host_modules(self):
        num_layers = self.model_config.num_hidden_layers
        first_k_dense = self.model_config.first_k_dense_replace
        num_experts = self.model_config.n_routed_experts
        return {
            "attn": [f"attn_{layer_idx}" for layer_idx in range(num_layers)],
            "routed_expert": [
                f"routed_expert_{layer_idx}_{expert_idx}"
                for layer_idx in range(first_k_dense, num_layers)
                for expert_idx in range(num_experts)
            ],
            "shared_expert": [
                f"shared_expert_{layer_idx}"
                for layer_idx in range(first_k_dense, num_layers)
            ],
        }

    def _validate_prefill_host_weight_inventory(self, expected_tensor_keys):
        expected_modules = self._expected_prefill_host_modules()
        if list(self.weight_copy_task) != list(expected_modules):
            raise RuntimeError(
                "GLM-5 prefill host module groups mismatch: "
                f"expected={list(expected_modules)}, actual={list(self.weight_copy_task)}"
            )
        for kind, expected_keys in expected_modules.items():
            actual_keys = self.weight_copy_task.get(kind, [])
            if actual_keys != expected_keys:
                raise RuntimeError(
                    f"GLM-5 prefill host {kind} inventory mismatch: "
                    f"expected={len(expected_keys)} ordered modules, "
                    f"actual={len(actual_keys)}"
                )

        actual_tensor_keys = {}
        for full_name, mapping in self.state_dict_name_map.items():
            if set(mapping) != {"module_key", "tensor_key"}:
                raise RuntimeError(
                    f"GLM-5 prefill malformed tensor mapping for {full_name}: {mapping}"
                )
            module_key = mapping["module_key"]
            tensor_key = mapping["tensor_key"]
            if tensor_key in actual_tensor_keys.setdefault(module_key, []):
                raise RuntimeError(
                    f"GLM-5 prefill duplicate mapped tensor {module_key}/{tensor_key}"
                )
            actual_tensor_keys[module_key].append(tensor_key)

        if list(actual_tensor_keys) != list(expected_tensor_keys):
            raise RuntimeError(
                "GLM-5 prefill ordered mapped-module inventory mismatch"
            )
        for module_key, expected_keys in expected_tensor_keys.items():
            actual_keys = actual_tensor_keys.get(module_key, [])
            if actual_keys != expected_keys:
                raise RuntimeError(
                    f"GLM-5 prefill mapped tensors mismatch for {module_key}: "
                    f"expected={expected_keys}, actual={actual_keys}"
                )

        expected_flat_modules = {
            module_key
            for keys in expected_modules.values()
            for module_key in keys
        }
        if set(expected_tensor_keys) != expected_flat_modules:
            missing = sorted(expected_flat_modules - set(expected_tensor_keys))
            extra = sorted(set(expected_tensor_keys) - expected_flat_modules)
            raise RuntimeError(
                "GLM-5 prefill expected tensor inventory does not match host modules: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

        expected_module_counts = {
            kind: len(keys) for kind, keys in expected_modules.items()
        }
        actual_module_counts = {
            kind: len(self.weight_copy_task[kind]) for kind in expected_modules
        }
        mapped_tensor_counts = {
            kind: sum(len(actual_tensor_keys[key]) for key in keys)
            for kind, keys in expected_modules.items()
        }
        for kind in ("shared_expert", "routed_expert"):
            bad = [
                key for key in expected_modules[kind]
                if len(actual_tensor_keys[key]) != 3
            ]
            if bad:
                raise RuntimeError(
                    f"GLM-5 prefill {kind} modules must map exactly three bulk "
                    f"weight tensors; bad={bad[:5]}"
                )
        return {
            "expected_module_counts": expected_module_counts,
            "actual_module_counts": actual_module_counts,
            "mapped_tensor_counts": mapped_tensor_counts,
            "mapped_tensors_total": sum(mapped_tensor_counts.values()),
            "expected_tensor_keys": expected_tensor_keys,
        }

    def _validate_loaded_prefill_host_weight_inventory(self, backing_size):
        planned = self._prefill_host_weight_inventory
        expected_tensor_keys = planned["expected_tensor_keys"]
        loaded = self.parameter_server.module_weights_shm()

        expected_modules = set(expected_tensor_keys)
        actual_modules = set(loaded)
        if actual_modules != expected_modules:
            missing = sorted(expected_modules - actual_modules)
            extra = sorted(actual_modules - expected_modules)
            raise RuntimeError(
                "GLM-5 prefill loaded host module inventory mismatch: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

        dtype_bytes = {
            "float8_e4m3fn": 1,
            "uint8": 1,
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "int32": 4,
            "int64": 8,
        }
        intervals = []
        for module_key, expected_keys in expected_tensor_keys.items():
            actual_keys = set(loaded[module_key])
            if actual_keys != set(expected_keys):
                raise RuntimeError(
                    f"GLM-5 prefill loaded host tensors mismatch for {module_key}: "
                    f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}"
                )
            for tensor_key in expected_keys:
                metadata = loaded[module_key][tensor_key]
                try:
                    offset = int(metadata.offset)
                    byte_size = int(metadata.byte_size)
                    shape = [int(dim) for dim in metadata.tensor_shape]
                    dtype = str(metadata.dtype)
                except (AttributeError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        "GLM-5 prefill loaded host tensor metadata is malformed for "
                        f"{module_key}/{tensor_key}: {error}"
                    ) from error
                if offset < 0 or byte_size <= 0 or not shape or any(dim <= 0 for dim in shape):
                    raise RuntimeError(
                        "GLM-5 prefill loaded host tensor metadata has an invalid "
                        f"range/shape for {module_key}/{tensor_key}: "
                        f"offset={offset}, byte_size={byte_size}, shape={shape}"
                    )
                element_size = dtype_bytes.get(dtype)
                numel = 1
                for dim in shape:
                    numel *= dim
                if element_size is None or numel * element_size != byte_size:
                    raise RuntimeError(
                        "GLM-5 prefill loaded host tensor metadata has an invalid "
                        f"dtype/size for {module_key}/{tensor_key}: dtype={dtype}, "
                        f"shape={shape}, byte_size={byte_size}"
                    )
                end = offset + byte_size
                if end > backing_size:
                    raise RuntimeError(
                        "GLM-5 prefill loaded host tensor range exceeds backing for "
                        f"{module_key}/{tensor_key}: end={end}, backing={backing_size}"
                    )
                intervals.append((offset, end, module_key, tensor_key))

        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if previous[1] > current[0]:
                raise RuntimeError(
                    "GLM-5 prefill loaded host tensor ranges overlap: "
                    f"{previous[2]}/{previous[3]}=[{previous[0]},{previous[1]}) "
                    f"and {current[2]}/{current[3]}=[{current[0]},{current[1]})"
                )

        expected_by_kind = self._expected_prefill_host_modules()
        actual_module_counts = {
            kind: sum(module_key in actual_modules for module_key in module_keys)
            for kind, module_keys in expected_by_kind.items()
        }
        mapped_tensor_counts = {
            kind: sum(len(loaded[module_key]) for module_key in module_keys)
            for kind, module_keys in expected_by_kind.items()
        }
        return {
            "expected_module_counts": planned["expected_module_counts"],
            "actual_module_counts": actual_module_counts,
            "mapped_tensor_counts": mapped_tensor_counts,
            "mapped_tensors_total": sum(mapped_tensor_counts.values()),
            "mapped_weight_bytes": sum(end - start for start, end, _, _ in intervals),
            "mapped_byte_range": [intervals[0][0], intervals[-1][1]],
        }

    def _parse_state_dict(self):
        """Build state_dict_name_map by walking model named_parameters.

        GLM-5 differences from DeepSeek:
        - Indexer tensors (self_attn.indexer.*) are EXCLUDED from the map
          (they go to skeleton). Indexer has wk, wq_b (FP8), k_norm, weights_proj.
        - Dense MLP layers (0-2) have mlp.gate_proj/up_proj/down_proj (no experts)
        - MoE layers (3-77) have mlp.gate, mlp.shared_experts, mlp.experts
        - MTP layer 78 has eh_proj, enorm, hnorm, shared_head.norm
        """
        self.hf_config._attn_implementation = "eager"
        model = Glm5ForCausalLM(self.hf_config).to('cpu')
        model.eval()

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []
        expected_tensor_keys = {}

        num_layers = self.model_config.num_hidden_layers  # 78
        first_k_dense = self.model_config.first_k_dense_replace  # 3

        for layer_idx in trange(num_layers, desc="Parsing state_dict"):
            # Attention parameters (EXCLUDING indexer — indexer goes to skeleton;
            # EXCLUDING q_a_layernorm/kv_a_layernorm — tiny BF16 RMSNorm weights,
            # route them through skeleton too. If they stay in state_dict_name_map
            # they get stripped from the skeleton loader AND never actually
            # written into the live module at prefill time, so the module keeps
            # its `ones_()` init and every attention layer's Q/K norm is wrong.
            # q_a_layernorm is [2048] bf16 = 4KB/layer, kv_a_layernorm is [512]
            # bf16 = 1KB/layer → ~400KB total, trivial to hold in skeleton.)
            attn_tensor_keys = []
            for name, _ in model.model.layers[layer_idx].self_attn.named_parameters():
                if name.startswith("indexer."):
                    continue
                if name in ("q_a_layernorm.weight", "kv_a_layernorm.weight"):
                    continue
                attn_tensor_keys.append(name)
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            attn_key = f"attn_{layer_idx}"
            expected_tensor_keys[attn_key] = attn_tensor_keys
            self.weight_copy_task["attn"].append(attn_key)

            # MoE layers (layer_idx >= first_k_dense)
            if layer_idx >= first_k_dense:
                # Shared experts — use static param names (experts are placeholders)
                _shared_expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                expected_tensor_keys[f"shared_expert_{layer_idx}"] = list(
                    _shared_expert_param_names
                )
                for name in _shared_expert_param_names:
                    tensor_full_name = f"model.layers.{layer_idx}.mlp.shared_experts.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    f"shared_expert_{layer_idx}"
                )

                # Routed experts — use static param names (experts are placeholders)
                _expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                num_experts = self.model_config.n_routed_experts  # 256
                for expert_idx in range(num_experts):
                    module_key = f"routed_expert_{layer_idx}_{expert_idx}"
                    expected_tensor_keys[module_key] = list(_expert_param_names)
                    for name in _expert_param_names:
                        tensor_full_name = (
                            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": module_key,
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(module_key)

        self._prefill_host_weight_inventory = (
            self._validate_prefill_host_weight_inventory(expected_tensor_keys)
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
