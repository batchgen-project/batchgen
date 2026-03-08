# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Kimi K2.5 Parallel Strategy Manager for BatchGen.

Handles model initialization, weight loading, and EP configuration for Kimi K2.5.

Key differences from DeepSeek-V3 PSM:
- 384 routed experts (vs 256)
- INT4 W4A16 weight loading (vs FP8 W8A8)
- BF16 attention (no FP8 dequant scales)
- No _extract_dequantize_scale() — K2.5 doesn't use FP8
- Loads INT4 packed/scale tensors for persistent experts
"""

from .model import KimiK25ForCausalLM, KimiK25MoE, KimiK25MoEBufferManager
from .wrappers import KimiK25ExpertWrapper, KimiK25AttnWrapper
import logging
import types
import torch.distributed as dist
import time
import torch
import gc
import os
from batchgen.utils import torch_gpu_mem_usage
if os.environ.get("BATCHGEN_ENABLE_ALL_TO_ALL") == "1":
    from pplx_kernels.all_to_all import AllToAll
else:
    AllToAll = None


NUM_TOTAL_EXPERTS = 384  # K2.5 has 384 routed experts per MoE layer


class KimiK25ParallelStrategyManager:
    def __init__(
        self,
        loaded_model_config,
        engine_config,
        model_config,
        core_engine,
        skeleton_state_dict,
        local_rank,
        global_rank,
        world_size
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

    def _cleanup_decode_gpu_state(self):
        """Free all phase-specific GPU allocations (model, buffers, class state).

        Called before both configure_prefill() and configure_decoding() to ensure
        no GPU memory leaks across phase transitions. Cleans up:
        - Model instance and its nn.Parameters
        - INT4 contiguous GPU weight buffers (PSM attributes)
        - KimiK25MoE._buf class-level activation buffer manager
        - KimiK25MoE._shared_expert_stream class-level CUDA stream
        - AllToAll communication buffers (if enabled)
        """
        device = self.engine_config.Basic_Config.device_torch
        alloc_before = torch.cuda.memory_allocated(device)

        # 1. Delete old model (frees nn.Parameters on GPU)
        if getattr(self, 'model', None) is not None:
            del self.model
            self.model = None

        # 2. INT4 contiguous GPU buffers (stored on PSM, not on model)
        if hasattr(self, '_int4_packed_gpu_buf'):
            del self._int4_packed_gpu_buf
        if hasattr(self, '_int4_scale_gpu_buf'):
            del self._int4_scale_gpu_buf

        # 3. MoE buffer manager (class variable — survives model deletion)
        if hasattr(KimiK25MoE, '_buf') and KimiK25MoE._buf is not None:
            KimiK25MoE._buf = None

        # 4. MoE shared CUDA stream (class variable)
        if hasattr(KimiK25MoE, '_shared_expert_stream'):
            KimiK25MoE._shared_expert_stream = None

        # 5. AllToAll communication buffers (if ATA was enabled)
        for attr in ('expert_x', 'expert_y', 'y', 'dp_x', 'ata',
                     'expert_num_tokens', 'indices', 'weights'):
            if hasattr(self, attr):
                delattr(self, attr)

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        alloc_after = torch.cuda.memory_allocated(device)
        logging.info(
            f"Rank {self.rank}: [HBM] phase cleanup freed "
            f"{(alloc_before - alloc_after) / (1024**3):.2f} GiB "
            f"(allocated: {alloc_before / (1024**3):.2f} → {alloc_after / (1024**3):.2f} GiB)"
        )

    def configure_prefill(self):
        """Configure model skeleton for prefill (pure DP) and weight copy task."""
        import time
        start_time = time.perf_counter()
        timings = {}

        # Step 1: Set phase (pure DP - no EP in prefill)
        self.loaded_model_config.phase = "prefill"
        self.loaded_model_config.ep_size = 1  # Pure DP: all 384 experts on each rank

        # Step 1.5: Free decode-phase GPU allocations before creating prefill model
        self._cleanup_decode_gpu_state()

        # Step 2: Initialize model
        # K2.5 reuses KimiK25ForCausalLM with K2.5 config overrides
        step_start = time.perf_counter()
        self.model = KimiK25ForCausalLM(self.loaded_model_config)
        timings['model_init'] = time.perf_counter() - step_start

        # Step 3: Initialize data structures
        self.state_dict_name_map = {}
        self.weight_copy_task = {}
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        # Step 4: Build weight copy task mappings
        step_start = time.perf_counter()
        for layer_idx in range(self.model_config.num_hidden_layers):
            # Attention parameters
            for name, _ in self.model.model.layers[
                layer_idx
            ].self_attn.named_parameters():
                tensor_full_name = (
                    "model.layers." + str(layer_idx) + ".self_attn." + name
                )
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": "attn_" + str(layer_idx),
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append("attn_" + str(layer_idx))

            if layer_idx >= self.loaded_model_config.first_k_dense_replace:
                # Shared experts
                for name, _ in self.model.model.layers[
                    layer_idx
                ].mlp.shared_experts.named_parameters():
                    tensor_full_name = (
                        "model.layers."
                        + str(layer_idx)
                        + ".mlp.shared_experts."
                        + name
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": "shared_expert_" + str(layer_idx),
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    "shared_expert_" + str(layer_idx)
                )

                # Routed experts — module keys only (INT4 tensors handled by wrappers)
                # Prefill uses pure DP: all 384 experts on each rank
                # In EP mode, non-local experts are None — skip them
                for expert_idx in range(NUM_TOTAL_EXPERTS):
                    expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]
                    if expert is None:
                        continue
                    for name, _ in expert.named_parameters():
                        tensor_full_name = (
                            "model.layers."
                            + str(layer_idx)
                            + ".mlp.experts."
                            + str(expert_idx)
                            + "."
                            + name
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": "routed_expert_"
                            + str(layer_idx)
                            + "_"
                            + str(expert_idx),
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(
                        "routed_expert_"
                        + str(layer_idx)
                        + "_"
                        + str(expert_idx)
                    )
        timings['weight_mappings'] = time.perf_counter() - step_start

        # Step 5: Load model skeleton (BF16 — no FP8 dequant needed)
        step_start = time.perf_counter()
        self._load_model_skeleton()
        timings['skeleton'] = time.perf_counter() - step_start

        # Step 6: Config attention module
        step_start = time.perf_counter()
        self._config_attn_module()
        timings['attn'] = time.perf_counter() - step_start

        # Step 7: Config expert module
        step_start = time.perf_counter()
        self._config_expert_module()
        timings['expert'] = time.perf_counter() - step_start

        # Step 8: Config lm_head hook
        self._config_lm_head_hook()

        # Step 9: Set model to eval mode
        self.model.eval()

        # Step 10: Move model to device
        step_start = time.perf_counter()
        self.model.to(self.engine_config.Basic_Config.device_torch)
        timings['to_device'] = time.perf_counter() - step_start

        total_time = time.perf_counter() - start_time

        if self.rank == 0:
            logging.info(
                f"[PREFILL] Model configured in {total_time:.2f}s "
                f"(init={timings['model_init']:.1f}s, skeleton={timings['skeleton']:.1f}s, "
                f"expert={timings['expert']:.1f}s, to_device={timings['to_device']:.1f}s)"
            )

        return self.model, self.weight_copy_task

    def _warmup(self):
        """Warmup compiled kernels for MoE gate."""
        torch._dynamo.config.inline_inbuilt_nn_modules = True
        if self.rank == 0:
            logging.info("Start torch compile warmup")
        for layer_idx in range(self.loaded_model_config.first_k_dense_replace, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp.gate
            if hasattr(layer, "warmup"):
                if self.global_rank == 0:
                    logging.debug(f"Warming up layer {layer_idx}")
                    dummy_hidden_states = torch.randn(
                        128, 1, 7168, dtype=torch.bfloat16,
                        device=self.engine_config.Basic_Config.device_torch
                    )
                    _ = layer.decoding_forward(dummy_hidden_states)
                    torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)

    def configure_decoding(self, padding_bsz=None, comm=None):
        """Configure model for decoding: DP + EP with optional offloading.

        K2.5: 384 experts per layer, INT4 W4A16 quantization, BF16 attention.

        Args:
            padding_bsz: Maximum batch size per rank for token buffer allocation.
            comm: NCCL communicator for all-gather/all-reduce operations.
        """
        self.loaded_model_config.phase = "decode"
        self.loaded_model_config._attn_implementation = "eager"
        self.loaded_model_config.ep_size = self.world_size

        # Log GPU memory before deep free (use GiB = /1024^3, matching PyTorch OOM messages)
        device = self.engine_config.Basic_Config.device_torch
        alloc_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        logging.info(
            f"Rank {self.rank}: HBM BEFORE deep free: "
            f"allocated={alloc_before / (1024**3):.2f} GiB, "
            f"reserved={reserved_before / (1024**3):.2f} GiB"
        )

        # Deep free prefill model and all phase-specific GPU allocations
        self._cleanup_decode_gpu_state()

        # Log GPU memory after deep free
        alloc_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        logging.info(
            f"Rank {self.rank}: HBM AFTER deep free: "
            f"allocated={alloc_after / (1024**3):.2f} GiB, "
            f"reserved={reserved_after / (1024**3):.2f} GiB, "
            f"freed={(alloc_before - alloc_after) / (1024**3):.2f} GiB"
        )

        # Create model on CPU first to avoid GPU memory allocation
        with torch.device('cpu'):
            self.model = KimiK25ForCausalLM(self.loaded_model_config, comm)

        # Inject comm and device into MoE layers for EP AllGather/AllReduce
        device = self.engine_config.Basic_Config.device_torch
        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            moe = self.model.model.layers[layer_idx].mlp
            moe.comm = comm
            moe.device = device

        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        self.local_routed_experts = []
        self.host_routed_experts = []

        NUM_EXPERT_PER_RANK = NUM_TOTAL_EXPERTS // self.world_size

        # Determine offloading behavior
        if self.world_size > 8:
            offload_ratio = 0.0
            self.enable_ep_offloading = False
            NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
            logging.info(
                f"Rank {self.rank}: Multi-node mode (world_size={self.world_size}). "
                f"All {NUM_EXPERT_PER_RANK} experts per rank are persistent."
            )
        elif self.engine_config.EP_Config.enable_offloading:
            offload_ratio = self.engine_config.EP_Config.offloading_ratio
            self.enable_ep_offloading = True
            NUM_LOCAL_EXPERT_PER_LAYER = int(NUM_EXPERT_PER_RANK * (1 - offload_ratio))
            logging.info(
                f"Rank {self.rank}: EP with offloading enabled. "
                f"Experts per rank: {NUM_EXPERT_PER_RANK}, "
                f"Persistent (GPU): {NUM_LOCAL_EXPERT_PER_LAYER}, "
                f"Offloaded (host): {NUM_EXPERT_PER_RANK - NUM_LOCAL_EXPERT_PER_LAYER}"
            )
        else:
            offload_ratio = 0.0
            self.enable_ep_offloading = False
            NUM_LOCAL_EXPERT_PER_LAYER = self.engine_config.EP_Config.num_local_expert_per_layer
            if NUM_LOCAL_EXPERT_PER_LAYER is None or NUM_LOCAL_EXPERT_PER_LAYER == 0:
                NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
            logging.info(
                f"Rank {self.rank}: Single-node mode without offloading. "
                f"{NUM_LOCAL_EXPERT_PER_LAYER} experts persistent per rank."
            )

        self.num_local_expert_per_layer = NUM_LOCAL_EXPERT_PER_LAYER

        routed_expert_gpu_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
        routed_expert_gpu_end_idx = routed_expert_gpu_start_idx + NUM_LOCAL_EXPERT_PER_LAYER
        routed_expert_host_start_idx = routed_expert_gpu_end_idx
        routed_expert_host_end_idx = (self.global_rank + 1) * NUM_EXPERT_PER_RANK
        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            for expert_idx in range(routed_expert_gpu_start_idx, routed_expert_gpu_end_idx):
                self.local_routed_experts.append(
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                )
            for expert_idx in range(routed_expert_host_start_idx, routed_expert_host_end_idx):
                self.host_routed_experts.append(
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                )

        self.weight_copy_task["routed_expert"] = self.host_routed_experts

        # Build state_dict_name_map (internal use only)
        for layer_idx in range(self.model_config.num_hidden_layers):
            for name, _ in self.model.model.layers[
                layer_idx
            ].self_attn.named_parameters():
                tensor_full_name = (
                    "model.layers." + str(layer_idx) + ".self_attn." + name
                )
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": "attn_" + str(layer_idx),
                    "tensor_key": name,
                }

            if layer_idx >= self.loaded_model_config.first_k_dense_replace:
                for name, _ in self.model.model.layers[
                    layer_idx
                ].mlp.shared_experts.named_parameters():
                    tensor_full_name = (
                        "model.layers."
                        + str(layer_idx)
                        + ".mlp.shared_experts."
                        + name
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": "shared_expert_" + str(layer_idx),
                        "tensor_key": name,
                    }

                # Loop over this rank's experts (use global indices with placeholder structure)
                expert_start_idx = self.global_rank * (NUM_TOTAL_EXPERTS // self.world_size)
                expert_end_idx = expert_start_idx + (NUM_TOTAL_EXPERTS // self.world_size)
                for global_expert_idx in range(expert_start_idx, expert_end_idx):
                    # With placeholder structure, use global index directly
                    expert = self.model.model.layers[layer_idx].mlp.experts[global_expert_idx]

                    # Skip None placeholders (shouldn't happen in this range, but defensive)
                    if expert is None:
                        continue

                    for name, _ in expert.named_parameters():
                        tensor_full_name = (
                            "model.layers."
                            + str(layer_idx)
                            + ".mlp.experts."
                            + str(global_expert_idx)
                            + "."
                            + name
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": "routed_expert_"
                            + str(layer_idx)
                            + "_"
                            + str(global_expert_idx),
                            "tensor_key": name,
                        }

        # Load weights and configure modules — log HBM after each step to find leaks
        device = self.engine_config.Basic_Config.device_torch
        def _log_hbm(step_name):
            a = torch.cuda.memory_allocated(device) / (1024**3)
            r = torch.cuda.memory_reserved(device) / (1024**3)
            logging.debug(f"Rank {self.rank}: HBM after {step_name}: allocated={a:.2f} GiB, reserved={r:.2f} GiB")

        torch.cuda.empty_cache()
        _log_hbm("empty_cache")

        self._load_model_skeleton()
        _log_hbm("_load_model_skeleton")

        self._load_local_routed_experts()
        _log_hbm("_load_local_routed_experts")

        self._load_attn_module()
        _log_hbm("_load_attn_module")

        self._load_shared_expert_module()
        _log_hbm("_load_shared_expert_module")

        self._config_attn_module()
        _log_hbm("_config_attn_module")

        self._config_expert_module()
        _log_hbm("_config_expert_module")

        self._config_lm_head_hook()
        _log_hbm("_config_lm_head_hook")

        # Set per-layer persistent/non-persistent expert ID lists
        routed_expert_gpu_start = self.global_rank * (NUM_TOTAL_EXPERTS // self.world_size)
        routed_expert_gpu_end = routed_expert_gpu_start + self.num_local_expert_per_layer
        routed_expert_host_end = (self.global_rank + 1) * (NUM_TOTAL_EXPERTS // self.world_size)
        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            layer = self.model.model.layers[layer_idx]
            layer.mlp.persistent_expert_ids = list(range(routed_expert_gpu_start, routed_expert_gpu_end))
            layer.mlp.nonpersistent_expert_ids = list(range(routed_expert_gpu_end, routed_expert_host_end))
            layer.mlp.num_persistent_local_experts = self.num_local_expert_per_layer

        logging.info(
            f"Rank {self.rank}: K2.5 MoE expert split — "
            f"persistent: {self.num_local_expert_per_layer}, "
            f"non-persistent: {routed_expert_host_end - routed_expert_gpu_end}, "
            f"for layers {self.loaded_model_config.first_k_dense_replace}-{self.model_config.num_hidden_layers - 1}"
        )

        self.model.eval()

        device = self.engine_config.Basic_Config.device_torch

        # model.to(device) moves only nn.Parameters (skeleton params still on CPU).
        # Attn + shared expert params are already on GPU from _load_*_module().
        # INT4 weights are plain attributes — NOT moved by model.to().
        self.model.to(device)
        _log_hbm("model.to (params only)")

        # Move INT4 weights to GPU using 2 contiguous allocations (not 17,280 individual ones).
        # This avoids ~20 GiB CUDA allocator fragmentation from small scale tensors.
        self._move_int4_to_gpu_contiguous()
        _log_hbm("_move_int4_to_gpu_contiguous")

        # Initialize grouped WGMMA for persistent experts (after INT4 weights on GPU)
        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            moe = self.model.model.layers[layer_idx].mlp
            if hasattr(moe, 'init_grouped_wgmma'):
                moe.init_grouped_wgmma()
        _log_hbm("init_grouped_wgmma")

        # Initialize MoE layers for decoding
        self._init_mode_decoding()
        effective_padding_bsz = padding_bsz if padding_bsz is not None else 128
        self._init_decoding_padding_bsz(effective_padding_bsz)

        # Allocate shared MoE buffer manager (one instance for all 60 MoE layers)
        max_global_bsz = self.world_size * effective_padding_bsz
        KimiK25MoE._buf = KimiK25MoEBufferManager(
            E_local=NUM_LOCAL_EXPERT_PER_LAYER,
            max_global_bsz=max_global_bsz,
            H=self.loaded_model_config.hidden_size,
            N_inter=self.loaded_model_config.moe_intermediate_size,
            topk=self.loaded_model_config.num_experts_per_tok,
            num_tokens_per_rank=effective_padding_bsz,
            device=device,
        )
        _log_hbm("MoEBufferManager")

        # Initialize All-to-All comms if enabled
        if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "1":
            self._init_ata_comms(effective_padding_bsz)

        self._warmup()

        return self.model, self.weight_copy_task

    def _init_decoding_padding_bsz(self, padding_bsz):
        """Initialize padding batch size for decoding."""
        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        if env_max_bsz is not None:
            max_rank_bsz = int(env_max_bsz)
            if self.rank == 0:
                logging.info(f"[DECODE] Padding batch size: {max_rank_bsz} (from BATCHGEN_MAX_RANK_BSZ)")
        else:
            max_rank_bsz = padding_bsz
            if self.rank == 0:
                logging.info(f"[DECODE] Padding batch size: {padding_bsz}")

        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init_num_tokens"):
                layer.init_num_tokens(max_rank_bsz)

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        """Dynamically update num_tokens_per_rank for all MoE layers."""
        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "set_num_tokens_per_rank"):
                layer.set_num_tokens_per_rank(num_tokens_per_rank)

    def _init_ata_comms(self, padding_bsz):
        """Initialize All-to-All communication for multi-GPU MoE.

        K2.5: uses BF16 dispatch (not FP8), 384 experts.
        """
        # K2.5 uses BF16 dispatch (no FP8 activation quantization)
        in_type = torch.bfloat16
        out_type = torch.bfloat16
        dp_size = 1
        world_size = self.world_size
        num_dp = world_size // dp_size
        hidden_size = 7168
        self.device = self.engine_config.Basic_Config.device_torch

        self.experts_per_rank = NUM_TOTAL_EXPERTS // world_size
        self.num_experts_per_tok = 8

        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        if env_max_bsz is not None:
            max_rank_bsz = int(env_max_bsz)
            logging.info(
                f"Rank {self.rank}: _init_ata_comms - Using BATCHGEN_MAX_RANK_BSZ={max_rank_bsz}"
            )
        else:
            max_rank_bsz = padding_bsz

        self.num_tokens_per_rank = max_rank_bsz

        self.expert_num_tokens = torch.empty(
            self.experts_per_rank, dtype=torch.int32, device=self.device
        )
        self.expert_x = torch.empty(
            (self.experts_per_rank, self.num_tokens_per_rank * num_dp, hidden_size),
            dtype=in_type,
            device=self.device
        )
        # BF16 dispatch — no per-block scaling needed (unlike FP8)
        self.expert_y = torch.empty_like(self.expert_x, dtype=out_type)
        self.indices = torch.empty(
            (self.num_tokens_per_rank, self.num_experts_per_tok),
            dtype=torch.uint32,
            device=self.device
        )
        self.weights = torch.empty(
            (self.num_tokens_per_rank, self.num_experts_per_tok),
            dtype=torch.float32,
            device=self.device
        )
        self.y = torch.empty(
            (self.num_tokens_per_rank, hidden_size),
            dtype=out_type,
            device=self.device
        )
        self.dp_x = torch.empty(
            (self.num_tokens_per_rank, hidden_size),
            dtype=in_type,
            device=self.device
        )

        if self.world_size <= 8:
            self.ata = AllToAll.intranode(
                max_num_tokens=self.num_tokens_per_rank,
                num_experts=NUM_TOTAL_EXPERTS,
                experts_per_token=self.num_experts_per_tok,
                rank=self.rank,
                world_size=self.world_size,
                dp_size=dp_size,
                hidden_dim=hidden_size,
                hidden_dim_bytes=hidden_size * in_type.itemsize,
                hidden_dim_scale_bytes=0  # BF16 dispatch — no scale
            )
        else:
            self.ata = AllToAll.internode(
                max_num_tokens=self.num_tokens_per_rank,
                num_experts=NUM_TOTAL_EXPERTS,
                experts_per_token=self.num_experts_per_tok,
                rank=self.rank,
                world_size=self.world_size,
                dp_size=dp_size,
                hidden_dim=hidden_size,
                hidden_dim_bytes=hidden_size * in_type.itemsize,
                hidden_dim_scale_bytes=0
            )

        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init_ata_comm"):
                layer.init_ata_comm(
                    padding_bsz,
                    self.expert_num_tokens,
                    self.expert_x,
                    None,  # No FP8 scale for BF16 dispatch
                    self.expert_y,
                    self.indices,
                    self.weights,
                    self.y,
                    self.dp_x,
                    None,  # No FP8 scale for BF16 dispatch
                    self.ata
                )

    def _init_mode_decoding(self):
        """Initialize MoE layers for decoding."""
        if self.enable_ep_offloading:
            if self.rank == 0:
                logging.info("EP offloading mode: skipping grouped GEMM init (using loop-based execution)")
            return

        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init"):
                layer.init(self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size)

    def _load_attn_module(self):
        """Load BF16 attention weights from core engine."""
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            attn_module_name = "attn_" + str(layer_idx)
            tensors = self.core_engine.get_tensor(attn_module_name)
            attn_module.q_a_proj.weight.data = tensors["q_a_proj.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )
            attn_module.q_b_proj.weight.data = tensors["q_b_proj.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )
            attn_module.kv_a_proj_with_mqa.weight.data = tensors["kv_a_proj_with_mqa.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )
            attn_module.kv_b_proj.weight.data = tensors["kv_b_proj.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )
            attn_module.o_proj.weight.data = tensors["o_proj.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )
            attn_module.q_a_layernorm.weight.data = tensors["q_a_layernorm.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )
            attn_module.kv_a_layernorm.weight.data = tensors["kv_a_layernorm.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )

    def _load_shared_expert_module(self):
        """Load BF16 shared expert weights from core engine."""
        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            len(self.model.model.layers),
        ):
            layer = self.model.model.layers[layer_idx]
            shared_expert_name = "shared_expert_" + str(layer_idx)
            tensors = self.core_engine.get_tensor(shared_expert_name)
            shared_expert = layer.mlp.shared_experts
            shared_expert.gate_proj.weight.data = tensors["gate_proj.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )
            shared_expert.up_proj.weight.data = tensors["up_proj.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )
            shared_expert.down_proj.weight.data = tensors["down_proj.weight"].to(
                self.engine_config.Basic_Config.device_torch
            )

    def _config_attn_module(self):
        """Configure attention wrappers with BF16 MLA methods.

        K2.5 uses BF16 attention (no FP8 dequant scales needed).
        Injects MLA prefill/decode methods based on GPU architecture.
        """
        start_time = time.perf_counter()
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            if self.engine_config.Basic_Config.gpu_arch == "hopper":
                from batchgen.attention.mla.fa3_backend import (
                    mla_prefill_flashattention3,
                    mla_prefill_flashattention3_prepacked,
                )
                from batchgen.attention.mla.flashmla_backend import (
                    mla_decoding_flashmla,
                    fused_get_query_states_triton,
                    mla_decoding_flashmla_attn_mode_3_bf16,
                    mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv,
                )
                # K2.5 pure BF16 attention (no FP8 quantization)
                setattr(
                    attn_module, "prefill_attn_bf16",
                    types.MethodType(mla_prefill_flashattention3, attn_module),
                )
                setattr(
                    attn_module, "prefill_attn_bf16_prepacked",
                    types.MethodType(mla_prefill_flashattention3_prepacked, attn_module),
                )
                setattr(
                    attn_module, "decoding_attn",
                    types.MethodType(mla_decoding_flashmla, attn_module),
                )
                setattr(
                    attn_module, "decoding_attn_mode_3_bf16",
                    types.MethodType(
                        mla_decoding_flashmla_attn_mode_3_pure_bf16_with_pagekv, attn_module
                    ),
                )
                setattr(
                    attn_module, "decoding_attn_bf16",
                    types.MethodType(mla_decoding_flashmla, attn_module),
                )
                setattr(
                    attn_module, "fused_get_query_states_triton",
                    types.MethodType(fused_get_query_states_triton, attn_module),
                )
            elif self.engine_config.Basic_Config.gpu_arch == "ampere":
                from batchgen.attention.mla.fa2_backend import (
                    mla_prefill_flashattention2,
                    mla_chunked_prefill_flashattention2,
                )
                from batchgen.attention.mla.torch_backend import (
                    mla_decoding_torch,
                )
                setattr(
                    attn_module, "prefill_attn_bf16",
                    types.MethodType(mla_chunked_prefill_flashattention2, attn_module),
                )
                setattr(
                    attn_module, "decoding_attn",
                    types.MethodType(mla_decoding_torch, attn_module),
                )
                setattr(
                    attn_module, "decoding_attn_bf16",
                    types.MethodType(mla_decoding_torch, attn_module),
                )
            else:
                raise ValueError(
                    "Unsupported GPU architecture: "
                    + self.engine_config.Basic_Config.gpu_arch
                )

            # Attention: persistent if NOT in weight_copy_task
            if "attn_" + str(layer_idx) in self.weight_copy_task["attn"]:
                persistent = False
            else:
                persistent = True

            # K2.5: No FP8 weight_dequant_scales — BF16 attention
            attn_wrapper_instance = KimiK25AttnWrapper(
                attn_module,
                layer_idx,
                self.core_engine,
                self.engine_config,
                self.model_config,
                persistent,
            )
            self.model.model.layers[layer_idx].self_attn = attn_wrapper_instance

        end_time = time.perf_counter()
        logging.debug(
            f"Attn module configuration time: {end_time - start_time:.2f} seconds"
        )

    def _load_local_routed_experts(self):
        """Load INT4 packed/scale tensors for persistent (GPU-resident) routed experts.

        For K2.5, routed expert weights are INT4 packed (int32) + scale (bf16).
        Weights are loaded to CPU first, then moved to GPU in bulk via
        _move_int4_to_gpu_contiguous() to avoid CUDA allocator fragmentation.
        """
        for routed_expert_idx in self.local_routed_experts:
            tensors = self.core_engine.get_tensor(routed_expert_idx)
            layer_idx = int(routed_expert_idx.split("_")[2])
            global_expert_idx = int(routed_expert_idx.split("_")[3])

            # With placeholder structure, use global index directly
            expert = self.model.model.layers[layer_idx].mlp.experts[global_expert_idx]

            if expert is None:
                raise RuntimeError(
                    f"Expert at global index {global_expert_idx} is None - "
                    f"placeholder structure mismatch for rank {self.global_rank}"
                )

            # Delete unused nn.Linear default weights.
            # These are created during DeepseekV3MLP.__init__ but never used
            # (we use INT4 weights instead).
            del expert.gate_proj.weight
            del expert.up_proj.weight
            del expert.down_proj.weight
            expert.gate_proj.weight = None
            expert.up_proj.weight = None
            expert.down_proj.weight = None

            # Store INT4 packed/scale tensors on CPU as plain attributes.
            # They will be moved to GPU in bulk by _move_int4_to_gpu_contiguous().
            expert.int4_gate_packed = tensors["gate_proj.weight_packed"]
            expert.int4_gate_scale = tensors["gate_proj.weight_scale"]
            expert.int4_up_packed = tensors["up_proj.weight_packed"]
            expert.int4_up_scale = tensors["up_proj.weight_scale"]
            expert.int4_down_packed = tensors["down_proj.weight_packed"]
            expert.int4_down_scale = tensors["down_proj.weight_scale"]

        logging.debug(f"Local routed experts loaded ({len(self.local_routed_experts)} experts)")

    def _move_int4_to_gpu_contiguous(self):
        """Move all INT4 expert weights to GPU using 2 contiguous allocations.

        Instead of 17,280 individual .to(device) calls (which fragments the CUDA
        allocator by ~20 GiB due to block rounding on small scale tensors), we:
        1. Pre-allocate one contiguous int32 buffer for all packed weights
        2. Pre-allocate one contiguous bf16 buffer for all scale weights
        3. Copy CPU tensors into slices, replace attributes with GPU views

        This reduces GPU allocations from 17,280 to 2, eliminating fragmentation.
        """
        device = self.engine_config.Basic_Config.device_torch

        # Collect all INT4 tensors and compute total sizes
        # Each entry: (expert_module, attr_name, cpu_tensor)
        packed_entries = []
        scale_entries = []

        for routed_expert_idx in self.local_routed_experts:
            layer_idx = int(routed_expert_idx.split("_")[2])
            global_expert_idx = int(routed_expert_idx.split("_")[3])
            expert = self.model.model.layers[layer_idx].mlp.experts[global_expert_idx]
            if expert is None:
                continue
            # After wrapping, expert is a KimiK25ExpertWrapper — get underlying module
            module = expert.module if hasattr(expert, 'module') else expert

            for proj in ('gate', 'up', 'down'):
                packed_attr = f'int4_{proj}_packed'
                scale_attr = f'int4_{proj}_scale'
                packed_entries.append((module, packed_attr, getattr(module, packed_attr)))
                scale_entries.append((module, scale_attr, getattr(module, scale_attr)))

        # Pre-allocate contiguous GPU buffers
        total_packed_numel = sum(t.numel() for _, _, t in packed_entries)
        total_scale_numel = sum(t.numel() for _, _, t in scale_entries)

        packed_gpu_buf = torch.empty(total_packed_numel, dtype=torch.int32, device=device)
        scale_gpu_buf = torch.empty(total_scale_numel, dtype=torch.bfloat16, device=device)

        if self.rank == 0:
            logging.info(
                f"[MODEL] INT4 contiguous GPU buffers: "
                f"packed={total_packed_numel * 4 / (1024**3):.2f} GiB, "
                f"scale={total_scale_numel * 2 / (1024**3):.2f} GiB"
            )

        # Copy CPU tensors into GPU buffer slices, replace attributes with views
        offset = 0
        for module, attr_name, cpu_tensor in packed_entries:
            n = cpu_tensor.numel()
            gpu_view = packed_gpu_buf[offset:offset + n].view(cpu_tensor.shape)
            gpu_view.copy_(cpu_tensor)
            setattr(module, attr_name, gpu_view)
            offset += n

        offset = 0
        for module, attr_name, cpu_tensor in scale_entries:
            n = cpu_tensor.numel()
            gpu_view = scale_gpu_buf[offset:offset + n].view(cpu_tensor.shape)
            gpu_view.copy_(cpu_tensor)
            setattr(module, attr_name, gpu_view)
            offset += n

        # Keep references to prevent GC of the backing buffers
        self._int4_packed_gpu_buf = packed_gpu_buf
        self._int4_scale_gpu_buf = scale_gpu_buf

        logging.debug(f"INT4 weights moved to GPU contiguously for {len(self.local_routed_experts)} experts")

    def _load_model_skeleton(self):
        """Load skeleton weights (norms, embeddings, router) to model.

        K2.5: No FP8 dequantization needed — all skeleton weights are BF16.

        Note: Checkpoint has 'language_model.' prefix (from KimiK25ForConditionalGeneration),
        but BatchGen uses only the language model (KimiK25ForCausalLM), so we strip the prefix.
        """
        for key, param in self.model.named_parameters():
            # Try with and without language_model. prefix
            checkpoint_key = f"language_model.{key}"
            if checkpoint_key in self.skeleton_state_dict:
                # K2.5: direct BF16 assignment (no FP8 dequant)
                param.data = self.skeleton_state_dict[checkpoint_key]
            elif key in self.skeleton_state_dict:
                # Fallback: try without prefix
                param.data = self.skeleton_state_dict[key]

        model_skeleton_byte_size = (
            sum(p.numel() * p.element_size() for p in self.model.parameters())
            / (1024**3)
        )
        if self.rank == 0:
            logging.info(f"Model skeleton size: {model_skeleton_byte_size:.2f} GB")

    def _config_expert_module(self):
        """Replace expert modules with K2.5 wrappers.

        persistent flag:
        - True: INT4 weights pre-loaded on GPU via registered buffers
        - False: INT4 weights loaded from core_engine buffer each forward

        K2.5 differences from DeepSeek-V3:
        - No FP8 weight_dequant_scales (K2.5 uses INT4 W4A16)
        - INT4 weights registered as module buffers, moved by model.to(device)
        - Wrapper accesses INT4 weights through self.module (no cached pointers)
        - Pre-dequant to BF16 handled by _pre_dequant_experts() after model.to()
        """
        start_time = time.perf_counter()

        for layer_idx in range(
            self.loaded_model_config.first_k_dense_replace,
            len(self.model.model.layers),
        ):
            layer = self.model.model.layers[layer_idx]

            # Shared expert: always persistent, BF16 (not quantized)
            if (
                "shared_expert_" + str(layer_idx)
                in self.weight_copy_task["shared_expert"]
            ):
                shared_persistent = False
            else:
                shared_persistent = True

            # K2.5 shared expert: standard BF16, no INT4 dequant
            layer.mlp.shared_experts = KimiK25ExpertWrapper(
                layer.mlp.shared_experts,
                layer_idx,
                -1,
                self.core_engine,
                self.engine_config,
                self.model_config,
                shared_persistent,
            )
            if shared_persistent:
                # Shared expert is BF16 — set use_bf16_weights directly
                layer.mlp.shared_experts.use_bf16_weights = True
                layer.mlp.shared_experts.gate_weight_bf16 = layer.mlp.shared_experts.module.gate_proj.weight.data
                layer.mlp.shared_experts.up_weight_bf16 = layer.mlp.shared_experts.module.up_proj.weight.data
                layer.mlp.shared_experts.down_weight_bf16 = layer.mlp.shared_experts.module.down_proj.weight.data

            # Offloading summary (rank 0 only, first MoE layer only)
            if layer_idx == self.loaded_model_config.first_k_dense_replace and self.rank == 0:
                n_offloaded = len(self.weight_copy_task['routed_expert'])
                n_total = len(layer.mlp.experts)
                attn_offloaded = len(self.weight_copy_task.get('attn', [])) > 0
                shared_offloaded = len(self.weight_copy_task.get('shared_expert', [])) > 0
                logging.info(
                    f"Offloading summary: attention={'offloaded' if attn_offloaded else 'persistent'}, "
                    f"shared_experts={'offloaded' if shared_offloaded else 'persistent'}, "
                    f"routed_experts={n_offloaded}/{n_total} offloaded ({100*n_offloaded/n_total:.0f}%)"
                )

            # Loop through all expert slots (384 total)
            # In prefill: all slots are instantiated
            # In decode: only local expert slots are instantiated (rest are None)
            for expert_idx in range(len(layer.mlp.experts)):
                expert = layer.mlp.experts[expert_idx]

                # Skip None placeholders (EP mode - non-local experts)
                if expert is None:
                    continue

                # Index in ModuleList IS the global expert index
                global_expert_idx = expert_idx

                # Routed expert: persistent if NOT in weight_copy_task
                module_key = "routed_expert_" + str(layer_idx) + "_" + str(global_expert_idx)
                if module_key in self.weight_copy_task["routed_expert"]:
                    persistent = False
                else:
                    persistent = True

                # K2.5: No FP8 weight_dequant_scales
                layer.mlp.experts[expert_idx] = KimiK25ExpertWrapper(
                    expert,
                    layer_idx,
                    global_expert_idx,  # Use global index for wrapper
                    self.core_engine,
                    self.engine_config,
                    self.model_config,
                    persistent,
                )
                # Note: pre-dequant (INT4→BF16) is deferred to _pre_dequant_experts()
                # which runs AFTER model.to(device), so dequant happens on GPU.

        end_time = time.perf_counter()
        logging.debug(
            f"Expert module configuration time: {end_time - start_time:.2f} seconds"
        )

    def _lm_head_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self):
        self.model.lm_head.register_forward_pre_hook(
            self._lm_head_forward_pre_hook
        )
