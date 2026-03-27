from .model import (
	DeepSeekR1ForCausalLM,
	DeepSeekR1MoE,
	DeepSeekR1MoEBufferManager,
)
from .wrappers import DeepSeekExpertWrapper, DeepSeekAttnWrapper as Attn_Wrapper
import logging
from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization
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
	AllToAll = None  # Optional dependency
	
class DeepseekV3ParallelStrategyManager:
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
		
	# def configure_prefill(self):
	# 	"""
	# 		Configure a model skeletion for prefill pure dp 
	# 		and the corresponding weight copy task.
	# 	"""
	# 	self.loaded_model_config.phase = "prefill"
	# 	# self.model = DeepseekV3ForCausalLM._from_config(
	# 	# 	self.loaded_model_config
	# 	# )
	# 	# logging.info(f"loaded_model_config: {self.loaded_model_config}")
	# 	self.model = DeepseekV3ForCausalLM(self.loaded_model_config)
	# 	self.state_dict_name_map = {}
	# 	self.weight_copy_task = {}
	# 	self.weight_copy_task["attn"] = []
	# 	self.weight_copy_task["routed_expert"] = []
	# 	self.weight_copy_task["shared_expert"] = []

	# 	for layer_idx in range(self.model_config.num_hidden_layers):
	# 		for name, _ in self.model.model.layers[
	# 			layer_idx
	# 		].self_attn.named_parameters():
	# 			tensor_full_name = (
	# 				"model.layers." + str(layer_idx) + ".self_attn." + name
	# 			)
	# 			self.state_dict_name_map[tensor_full_name] = {
	# 				"module_key": "attn_" + str(layer_idx),
	# 				"tensor_key": name,
	# 			}
	# 		self.weight_copy_task["attn"].append("attn_" + str(layer_idx))

	# 		if layer_idx >= self.loaded_model_config.first_k_dense_replace:
	# 			for name, _ in self.model.model.layers[
	# 				layer_idx
	# 			].mlp.shared_experts.named_parameters():
	# 				tensor_full_name = (
	# 					"model.layers."
	# 					+ str(layer_idx)
	# 					+ ".mlp.shared_experts."
	# 					+ name
	# 				)
	# 				self.state_dict_name_map[tensor_full_name] = {
	# 					"module_key": "shared_expert_" + str(layer_idx),
	# 					"tensor_key": name,
	# 				}
	# 			self.weight_copy_task["shared_expert"].append(
	# 				"shared_expert_" + str(layer_idx)
	# 			)

	# 			for expert_idx in range(self.model_config.num_local_experts):
	# 				for name, _ in (
	# 					self.model.model.layers[layer_idx]
	# 					.mlp.experts[expert_idx]
	# 					.named_parameters()
	# 				):
	# 					tensor_full_name = (
	# 						"model.layers."
	# 						+ str(layer_idx)
	# 						+ ".mlp.experts."
	# 						+ str(expert_idx)
	# 						+ "."
	# 						+ name
	# 					)
	# 					self.state_dict_name_map[tensor_full_name] = {
	# 						"module_key": "routed_expert_"
	# 						+ str(layer_idx)
	# 						+ "_"
	# 						+ str(expert_idx),
	# 						"tensor_key": name,
	# 					}
	# 				self.weight_copy_task["routed_expert"].append(
	# 					"routed_expert_"
	# 					+ str(layer_idx)
	# 					+ "_"
	# 					+ str(expert_idx)
	# 				)

	# 	# Load Model Skeleton
	# 	self._extract_dequantize_scale()
	# 	self._load_model_skeleton()
	# 	self._config_attn_module()
	# 	self._config_expert_module()
	# 	self._config_lm_head_hook()
	# 	self.model.eval()
	# 	self.model.to(self.engine_config.Basic_Config.device_torch)
	# 	# self._warmup()
	# 	return self.model, self.weight_copy_task

	def configure_prefill(self):
		"""
			Configure a model skeletion for prefill pure dp
			and the corresponding weight copy task.
		"""
		import time
		start_time = time.perf_counter()
		timings = {}

		# Step 1: Set phase
		self.loaded_model_config.phase = "prefill"

		# Step 2: Initialize model
		step_start = time.perf_counter()
		self.model = DeepSeekR1ForCausalLM(self.loaded_model_config)
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

				# Routed experts
				for expert_idx in range(self.model_config.num_local_experts):
					for name, _ in (
						self.model.model.layers[layer_idx]
						.mlp.experts[expert_idx]
						.named_parameters()
					):
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

		# Step 5: Extract dequantize scale
		step_start = time.perf_counter()
		self._extract_dequantize_scale()
		timings['dequantize'] = time.perf_counter() - step_start

		# Step 6: Load model skeleton
		step_start = time.perf_counter()
		self._load_model_skeleton()
		timings['skeleton'] = time.perf_counter() - step_start

		# Step 7: Config attention module
		step_start = time.perf_counter()
		self._config_attn_module()
		timings['attn'] = time.perf_counter() - step_start

		# Step 8: Config expert module
		step_start = time.perf_counter()
		self._config_expert_module()
		timings['expert'] = time.perf_counter() - step_start

		# Step 9: Config lm_head hook
		self._config_lm_head_hook()

		# Step 10: Set model to eval mode
		self.model.eval()

		# Step 11: Move model to device
		step_start = time.perf_counter()
		self.model.to(self.engine_config.Basic_Config.device_torch)
		timings['to_device'] = time.perf_counter() - step_start

		total_time = time.perf_counter() - start_time

		# Log summary (rank 0 only)
		if self.rank == 0:
			logging.info(
				f"[PREFILL] Model configured in {total_time:.2f}s "
				f"(init={timings['model_init']:.1f}s, skeleton={timings['skeleton']:.1f}s, "
				f"expert={timings['expert']:.1f}s, to_device={timings['to_device']:.1f}s)"
			)

		return self.model, self.weight_copy_task
	
	def _warmup(self):
		# Currently only need to warmup the MoEGate
		torch._dynamo.config.inline_inbuilt_nn_modules = True
		if self.rank == 0:
			logging.info("Start torch compile warmup")
		# warmup_compiled_moe_gate removed — using new model.py gate
		# device = self.engine_config.Basic_Config.device_torch
		# with torch.inference_mode():
		# 	warmup_compiled_moe_gate(device)
		for layer_idx in range(self.loaded_model_config.first_k_dense_replace, self.model_config.num_hidden_layers):
			layer = self.model.model.layers[layer_idx].mlp.gate
			if hasattr(layer, "warmup"):
				if self.global_rank == 0:
					logging.debug(f"Warming up layer {layer_idx}")
					dummy_hidden_states = torch.randn(128, 1, 7168, dtype=torch.bfloat16, device=self.engine_config.Basic_Config.device_torch)
					_ = layer.decoding_forward(dummy_hidden_states)
					torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
				# layer.warmup()
			# with torch.inference_mode():
			# 	for t in range(5):
			# 		dummy_hidden_states = torch.randn(128, 1, 7168, dtype=torch.bfloat16, device=self.engine_config.Basic_Config.device_torch)
			# 		_ = layer.decoding_forward(dummy_hidden_states)

		
		# for layer_idx in range(self.loaded_model_config.first_k_dense_replace, self.loaded_model_config.first_k_dense_replace + 1):
		# 	layer = self.model.model.layers[layer_idx].mlp.gate
		# 	if hasattr(layer, "warmup"):
		# 		layer.warmup()


	def configure_decoding(self, padding_bsz=None, comm=None):
		"""
		Configure model for decoding: DP + EP with optional offloading.

		Handles all deployment scenarios:
		- Multi-node (world_size > 8): all experts persistent (no offloading)
		- Single-node with EP offloading: partial persistence based on offloading_ratio
		- Single-node without offloading: all experts persistent

		Args:
			padding_bsz: Maximum batch size per rank for token buffer allocation.
				Required for EP offloading mode (moe_infer_loop_with_offloading).
				If None, uses BATCHGEN_MAX_RANK_BSZ env var or defaults to 128.
			comm: NCCL communicator for all-gather/all-reduce operations.
				Required for distributed MoE forward.

		When enable_offloading is True:
		- Uses offloading_ratio to determine which experts are persistent (GPU-resident)
		- persistent=True: weights pre-loaded on GPU
		- persistent=False: weights loaded from buffer each forward
		"""
		self.loaded_model_config.phase = "decode"
		self.loaded_model_config._attn_implementation = "eager"
		self.loaded_model_config.ep_size = self.world_size
		self.model = None
		torch.cuda.empty_cache()

		# Always use comm for NCCL collectives
		self.model = DeepSeekR1ForCausalLM(self.loaded_model_config, comm)

		self.weight_copy_task = {}
		self.state_dict_name_map = {}
		self.weight_copy_task["attn"] = []
		self.weight_copy_task["routed_expert"] = []
		self.weight_copy_task["shared_expert"] = []

		self.local_routed_experts = []
		self.host_routed_experts = []

		NUM_TOTAL_EXPERTS = 256          # Total experts per layer
		NUM_EXPERT_PER_RANK = NUM_TOTAL_EXPERTS // self.world_size

		# Determine offloading behavior based on deployment scenario
		if self.world_size > 8:
			# Multi-node: all experts persistent (no offloading across nodes)
			offload_ratio = 0.0
			self.enable_ep_offloading = False
			NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
			logging.info(
				f"Rank {self.rank}: Multi-node mode (world_size={self.world_size}). "
				f"All {NUM_EXPERT_PER_RANK} experts per rank are persistent."
			)
		elif self.engine_config.EP_Config.enable_offloading:
			# Single-node with EP offloading
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
			# Single-node without offloading: all experts persistent
			offload_ratio = 0.0
			self.enable_ep_offloading = False
			NUM_LOCAL_EXPERT_PER_LAYER = self.engine_config.EP_Config.num_local_expert_per_layer
			if NUM_LOCAL_EXPERT_PER_LAYER is None or NUM_LOCAL_EXPERT_PER_LAYER == 0:
				NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
			logging.info(
				f"Rank {self.rank}: Single-node mode without offloading. "
				f"{NUM_LOCAL_EXPERT_PER_LAYER} experts persistent per rank."
			)

		# Store for later use in _config_expert_module
		self.num_local_expert_per_layer = NUM_LOCAL_EXPERT_PER_LAYER


		routed_expert_gpu_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
		routed_expert_gpu_end_idx = routed_expert_gpu_start_idx + NUM_LOCAL_EXPERT_PER_LAYER
		routed_expert_host_start_idx = routed_expert_gpu_end_idx
		routed_expert_host_end_idx = (self.global_rank + 1) * NUM_EXPERT_PER_RANK
		for layer_idx in range(
			self.loaded_model_config.first_k_dense_replace,
			self.model_config.num_hidden_layers,
		):
			# The first NUM_LOCAL_EXPERT_PER_LAYER in each part associated with the corresponding rank.
			# The rest of the experts in the part are stored in the host memory.
			for expert_idx in range(routed_expert_gpu_start_idx, routed_expert_gpu_end_idx):
				self.local_routed_experts.append(
					"routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
				)
			for expert_idx in range(routed_expert_host_start_idx, routed_expert_host_end_idx):
				self.host_routed_experts.append(
					"routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
				)

		self.weight_copy_task["routed_expert"] = self.host_routed_experts
		# NOTE: For decoding mode, attention and shared experts are persistent (loaded via _load_model_skeleton).
		# Only offloaded routed experts need weight copying. DO NOT add attention/shared_expert to weight_copy_task.
		# weight_copy_task["attn"] and weight_copy_task["shared_expert"] stay EMPTY.

		# Build state_dict_name_map for all modules (needed for weight loading lookups)
		# but do NOT add to weight_copy_task (attention and shared experts are persistent)
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
			# DO NOT add attention to weight_copy_task - it's persistent for decoding

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
				# DO NOT add shared_expert to weight_copy_task - it's persistent for decoding

				for expert_idx in range(self.model_config.num_local_experts):
					for name, _ in (
						self.model.model.layers[layer_idx]
						.mlp.experts[expert_idx]
						.named_parameters()
					):
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
		# Load Model Skeleton and Local Routed Experts
		# Clear torch cache
		torch.cuda.empty_cache()
		self._extract_dequantize_scale()
		self._load_model_skeleton()
		self._load_local_routed_experts()
		# Load attention and shared expert FP8 weights (required for attn_mode=3 / EP offloading)
		# These are persistent on GPU, but need explicit loading
		self._load_attn_module()
		self._load_shared_expert_module()
		self._config_attn_module()
		self._config_expert_module()
		self._config_lm_head_hook()

		# --- NEW: Inject comm/device and init FP8 blockwise weights ---
		device = self.engine_config.Basic_Config.device_torch
		NUM_EXPERT_PER_RANK = NUM_TOTAL_EXPERTS // self.world_size
		for layer_idx in range(
			self.loaded_model_config.first_k_dense_replace,
			self.model_config.num_hidden_layers,
		):
			moe = self.model.model.layers[layer_idx].mlp
			moe.comm = comm
			moe.device = device
			# Stack per-expert FP8 weights into 3D tensors for grouped GEMM
			moe.init_fp8_blockwise_weights()

		# Allocate shared MoE buffer manager (singleton across all layers)
		effective_padding_bsz_buf = padding_bsz if padding_bsz is not None else 128
		env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
		if env_max_bsz is not None:
			effective_padding_bsz_buf = int(env_max_bsz)
		DeepSeekR1MoE._buf = DeepSeekR1MoEBufferManager(
			E_local=NUM_EXPERT_PER_RANK,
			max_global_bsz=self.world_size * effective_padding_bsz_buf,
			H=self.loaded_model_config.hidden_size,
			N_inter=self.loaded_model_config.moe_intermediate_size,
			topk=self.loaded_model_config.num_experts_per_tok,
			num_tokens_per_rank=effective_padding_bsz_buf,
			device=device,
		)
		if self.rank == 0:
			logging.info(
				f"[MoE] FP8 blockwise weights stacked, buffer manager allocated "
				f"(E_local={NUM_EXPERT_PER_RANK}, global_bsz={self.world_size * effective_padding_bsz_buf})"
			)
		# --- END NEW ---

		# Set enable_ep_offloading flag on MoE layers for loop-based execution
		if self.enable_ep_offloading:
			for layer_idx in range(
				self.loaded_model_config.first_k_dense_replace,
				self.model_config.num_hidden_layers,
			):
				layer = self.model.model.layers[layer_idx]
				# layer.mlp is DeepseekV3MoE_Decoding_FP8
				layer.mlp.enable_ep_offloading = True
			logging.info(
				f"Rank {self.rank}: Set enable_ep_offloading=True on MoE layers "
				f"(layers {self.loaded_model_config.first_k_dense_replace}-{self.model_config.num_hidden_layers - 1})"
			)

		self.model.eval()
		self.model.to(self.engine_config.Basic_Config.device_torch)
		# Log final GPU memory (rank 0 only)
		if self.rank == 0:
			used_memory = torch.cuda.memory_allocated(self.engine_config.Basic_Config.device_torch)
			logging.info(f"[MODEL] GPU memory after init: {used_memory / (1024**3):.2f} GB used")

		# Initialize MoE layers for decoding (required for EP offloading mode)
		# This sets up num_tokens_per_rank and other buffers needed for all-gather/all-reduce
		self._init_mode_decoding()
		# Use provided padding_bsz, or default to 128 if not provided
		# _init_decoding_padding_bsz will also check BATCHGEN_MAX_RANK_BSZ env var
		effective_padding_bsz = padding_bsz if padding_bsz is not None else 128
		self._init_decoding_padding_bsz(effective_padding_bsz)

		# Initialize All-to-All comms if enabled (used for multi-node or benchmark scenarios)
		if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "1":
			self._init_ata_comms(effective_padding_bsz)

		# Warmup compiled kernels
		self._warmup()

		return self.model, self.weight_copy_task

	def _init_decoding_padding_bsz(self, padding_bsz):
		"""
		Initialize the padding batch size for decoding.
		This is used to set the padding size for the input sequences.
		
		Uses BATCHGEN_MAX_RANK_BSZ environment variable if set, to pre-allocate
		large enough buffers for continuous batching scenarios.
		"""
		# Use BATCHGEN_MAX_RANK_BSZ environment variable if set, otherwise use padding_bsz
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
		"""
		Dynamically update num_tokens_per_rank for all MoE layers.
		Called at page boundaries to reduce all-gather/all-reduce communication.
		
		Args:
			num_tokens_per_rank: The max batch size across all ranks for this page
		"""
		for layer_idx in range(
			self.loaded_model_config.first_k_dense_replace,
			self.model_config.num_hidden_layers,
		):
			layer = self.model.model.layers[layer_idx].mlp
			if hasattr(layer, "set_num_tokens_per_rank"):
				layer.set_num_tokens_per_rank(num_tokens_per_rank)

	def _init_ata_comms(self, padding_bsz):
		# Current default ata impl is perplexity all-to-all dispatch and combine.
		# USe fp8e4m3 dispatch by default.
		in_type = torch.float8_e4m3fn
		out_type = torch.bfloat16
		dp_size = 1 # Each rank is a dp worker.
		world_size = self.world_size
		num_dp = world_size // dp_size	
		hidden_size = 7168
		self.device = self.engine_config.Basic_Config.device_torch
		block_size = 128

		self.experts_per_rank = 256 // world_size
		self.num_experts_per_tok = 8
		
		# Use BATCHGEN_MAX_RANK_BSZ environment variable if set, otherwise use padding_bsz
		# This allows pre-setting a large enough buffer size for continuous batching
		env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
		if env_max_bsz is not None:
			max_rank_bsz = int(env_max_bsz)
			logging.info(
				f"Rank {self.rank}: _init_ata_comms - Using BATCHGEN_MAX_RANK_BSZ={max_rank_bsz} "
				f"(padding_bsz from comms was {padding_bsz})"
			)
		else:
			max_rank_bsz = padding_bsz
			logging.info(
				f"Rank {self.rank}: _init_ata_comms - Using padding_bsz={padding_bsz} from comms "
				f"(BATCHGEN_MAX_RANK_BSZ not set)"
			)
		
		self.num_tokens_per_rank = max_rank_bsz

		self.expert_num_tokens = torch.empty(self.experts_per_rank, dtype=torch.int32, device=self.device)
		self.expert_x = torch.empty(
			(self.experts_per_rank, self.num_tokens_per_rank * num_dp, hidden_size),
			dtype=in_type,
			device=self.device
		)
		self.expert_x_scale = torch.empty(
			(self.experts_per_rank, self.expert_x.size(1), (self.expert_x.size(2) + block_size -1)//block_size),
			dtype=torch.float32,
			device=self.device
		)
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
		self.dp_x_scale = torch.empty(
			(self.dp_x.size(0), (self.dp_x.size(1) + block_size -1)//block_size),
			dtype=torch.float32,
			device=self.device
		)
		if self.world_size <= 8:
			# We does not support devices less than 8 but locates on different nodes.
			self.ata = AllToAll.intranode(
				max_num_tokens = self.num_tokens_per_rank,
				num_experts = 256,
				experts_per_token = self.num_experts_per_tok,
				rank = self.rank,
				world_size = self.world_size,
				dp_size = dp_size,
				hidden_dim = hidden_size,
				hidden_dim_bytes = hidden_size * in_type.itemsize,
				hidden_dim_scale_bytes = (hidden_size + block_size -1) // block_size * torch.float32.itemsize
			)
		else:
			self.ata = AllToAll.internode(
				max_num_tokens = self.num_tokens_per_rank,
				num_experts = 256,
				experts_per_token = self.num_experts_per_tok,
				rank = self.rank,
				world_size = self.world_size,
				dp_size = dp_size,
				hidden_dim = hidden_size,
				hidden_dim_bytes = hidden_size * in_type.itemsize,
				hidden_dim_scale_bytes = (hidden_size + block_size -1) // block_size * torch.float32.itemsize
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
					self.expert_x_scale,
					self.expert_y,
					self.indices,
					self.weights,
					self.y,
					self.dp_x,
					self.dp_x_scale,
					self.ata
				)
		




	def _init_mode_decoding(self):
		# Skip grouped GEMM initialization for EP offloading mode
		# In EP offloading, non-persistent experts don't have fp8_gate/fp8_up/fp8_down registered
		# and moe_infer_loop_with_offloading() doesn't use these pointer lists anyway
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

			# attn_module.initialize()

	

	def _load_shared_expert_module(self):
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

			# for name, param in layer.mlp.shared_experts.named_parameters():
			# 	if name in tensors:
			# 		if self.local_rank == 0:
			# 			logging.debug(f"Loading {name} for shared expert module {shared_expert_name}")
					
					# param.data = tensors[name].to(
					# 	self.engine_config.Basic_Config.device_torch
					# )

	def _config_attn_module(self):
		"""
		- Configure the wrapper.
		"""
		start_time = time.perf_counter()
		for layer_idx in range(len(self.model.model.layers)):
			attn_module = self.model.model.layers[layer_idx].self_attn
			if self.engine_config.Basic_Config.gpu_arch == "hopper":
				from ....attention.mla.fa3_backend import (
					mla_prefill_flashattention3,
					mla_prefill_flashattention3_w8a16_deepgemm,
					mla_prefill_flashattention3_fused_dequant,
					mla_prefill_flashattention3_prepacked,
					mla_prefill_flashattention3_w8a16_deepgemm_prepacked,
				)
				from ....attention.mla.flashmla_backend import (
					mla_decoding_flashmla,
					mla_decoding_flashmla_v2,
					fused_get_query_states_triton,
					# mla_decoding_flashmla_attn_mode_3,
					mla_decoding_flashmla_attn_mode_3_bf16,
					mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv,
					mla_decoding_flashmla_attn_mode_3_dequant_fusion,
					mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn
				)
				setattr(
					attn_module,
					"prefill_attn",
					types.MethodType(
						mla_prefill_flashattention3, attn_module
					),
				)
				setattr(
					attn_module,
					"prefill_attn_w8a16",
					types.MethodType(
						mla_prefill_flashattention3_w8a16_deepgemm, attn_module
					),
				)

				# Prepacked prefill methods for efficient batching
				setattr(
					attn_module,
					"prefill_attn_prepacked",
					types.MethodType(
						mla_prefill_flashattention3_prepacked, attn_module
					),
				)
				setattr(
					attn_module,
					"prefill_attn_w8a16_prepacked",
					types.MethodType(
						mla_prefill_flashattention3_w8a16_deepgemm_prepacked, attn_module
					),
				)

				setattr(
					attn_module,
					"decoding_attn",
					types.MethodType(
						mla_decoding_flashmla, attn_module
					),
				)

				setattr(
					attn_module,
					"decoding_attn_mode_3_fp8",
					types.MethodType(
						mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn, attn_module
					),
				)

				setattr(
					attn_module,
					"decoding_attn_mode_3_bf16",
					types.MethodType(
						# mla_decoding_flashmla_attn_mode_3_bf16, attn_module
						mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv, attn_module
					),
				)

				setattr(
					attn_module,
					"decoding_attn_mode_3_dequant_fusion",
					types.MethodType(
						mla_decoding_flashmla_attn_mode_3_dequant_fusion, attn_module
					),
				)


				setattr(
					attn_module,
					"fused_get_query_states_triton",
					types.MethodType(
						fused_get_query_states_triton, attn_module
					),
				)
			elif self.engine_config.Basic_Config.gpu_arch == "ampere":
				from ....attention.mla.fa2_backend import mla_prefill_flashattention2, mla_chunked_prefill_flashattention2
				from ....attention.mla.torch_backend import mla_decoding_torch, mla_chunked_prefill_torch
				setattr(
					attn_module,
					"prefill_attn",
					types.MethodType(
						mla_chunked_prefill_flashattention2, attn_module
					),
				)
				setattr(
					attn_module,
					"decoding_attn",
					types.MethodType(
						mla_decoding_torch, attn_module
					),
				)
			else:
				raise ValueError(
					"Unsupported GPU architecture: "
					+ self.engine_config.Basic_Config.gpu_arch
				)


			# Attention: persistent if NOT in weight_copy_task
			if "attn_" + str(layer_idx) in self.weight_copy_task["attn"]:
				persistent = False  # In offload list, needs loading
			else:
				persistent = True  # Not in offload list, pre-loaded on GPU
			weight_dequant_scales = {}
			prefix = "model.layers." + str(layer_idx) + ".self_attn."
			postfix = ".weight_scale_inv"
			for name, param in self.skeleton_state_dict.items():
				if name.startswith(prefix) and name.endswith(postfix):
					# Use simplified key: e.g: "q_a_proj.weight_scale_inv"
					key = name[len(prefix) :]
					weight_dequant_scales[key] = param.to(
						self.engine_config.Basic_Config.device_torch
					)
			attn_wrapper_instance = Attn_Wrapper(
				attn_module,
				layer_idx,
				self.core_engine,
				self.engine_config,
				self.model_config,
				persistent,
				weight_dequant_scales,
			)
			self.model.model.layers[layer_idx].self_attn = attn_wrapper_instance
			if persistent:
				# Persistent attention: register FP8 weights for direct GPU access
				attn_wrapper_instance._register_fp8_weights()
				for key, value in attn_wrapper_instance.weight_dequant_scale.items():
					value = value.to(
						self.engine_config.Basic_Config.device_torch
					)
				
		
		end_time = time.perf_counter()
		logging.debug(
			f"Attn module configuration time: {end_time - start_time:.2f} seconds"
		)
	
	def _unregister_fp8_weights(self):
		# set all fp8 weights to None
		for layer_idx in range(len(self.model.model.layers)):
			attn_module = self.model.model.layers[layer_idx].self_attn
			attn_module._unregister_fp8_weights()
			if layer_idx >= self.loaded_model_config.first_k_dense_replace:
				for routed_expert_idx in self.local_routed_experts:
					self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx]._unregister_fp8_weights()




	def _load_local_routed_experts(self):
		for routed_expert_idx in self.local_routed_experts:
			tensors = self.core_engine.get_tensor(routed_expert_idx)
			layer_idx = int(routed_expert_idx.split("_")[2])
			expert_idx = int(routed_expert_idx.split("_")[3])
			# logging.info(tensors.keys())
			self.model.model.layers[layer_idx].mlp.experts[expert_idx].gate_proj.weight.data = tensors["gate_proj.weight"].to(
				self.engine_config.Basic_Config.device_torch
			)
			self.model.model.layers[layer_idx].mlp.experts[expert_idx].up_proj.weight.data = tensors["up_proj.weight"].to(
				self.engine_config.Basic_Config.device_torch
			)
			self.model.model.layers[layer_idx].mlp.experts[expert_idx].down_proj.weight.data = tensors["down_proj.weight"].to(
				self.engine_config.Basic_Config.device_torch
			)
			# del tensors
		logging.debug(f"Local routed experts loaded")

	def _load_model_skeleton(self):
		for key, param in self.model.named_parameters():
			if key in self.skeleton_state_dict:
				dequant_key = key + "_scale_inv"
				if dequant_key in self.dequant_scale:
					param.data = deepseek_v3_dequantization(
						self.skeleton_state_dict[key],
						self.dequant_scale[dequant_key],
					)
				else:
					param.data = self.skeleton_state_dict[key]

		model_skeletion_byte_size = (
			sum(p.numel() * p.element_size() for p in self.model.parameters())
			/ (1024**3)
		)
		if self.rank == 0:
			logging.info(f"Model skeleton size: {model_skeletion_byte_size:.2f} GB")
		# Rank 0 print out all the tensors in the model with tensor size in MB
		# if dist.get_rank() == 0:
		# 	logging.info("Model skeleton tensors:")
		# 	for name, param in self.model.named_parameters():
		# 		tensor_size_mb = (
		# 			param.numel() * param.element_size() / (1024**2)
		# 		)
		# 		logging.info(
		# 			f"{name}: {tensor_size_mb:.2f} MB, dtype: {param.dtype}"
		# 		)
		# 	for name, buffer in self.model.named_buffers():
		# 		tensor_size_mb = (
		# 			buffer.numel() * buffer.element_size() / (1024**2)
		# 		)
		# 		logging.info(
		# 			f"{name}: {tensor_size_mb:.2f} MB, dtype: {buffer.dtype}"
		# 		)
		# dist.barrier()

	def _config_expert_module_(self):
		"""
		Replace expert module with the wrapper.

		persistent flag semantics:
		- True: weights are pre-loaded on GPU, no buffer fetch needed
		- False: weights need to be loaded from buffer each forward
		"""
		start_time = time.perf_counter()
		for layer_idx in range(
			self.loaded_model_config.first_k_dense_replace,
			len(self.model.model.layers),
		):
			layer = self.model.model.layers[layer_idx]
			# Shared expert: persistent if NOT in weight_copy_task
			if (
				"shared_expert_" + str(layer_idx)
				in self.weight_copy_task["shared_expert"]
			):
				persistent = False  # In offload list, needs loading
			else:
				persistent = True  # Not in offload list, pre-loaded on GPU

			prefix = "model.layers." + str(layer_idx) + ".mlp.shared_experts."
			postfix = ".weight_scale_inv"
			weight_dequant_scales = {}
			for name, param in self.skeleton_state_dict.items():
				if name.startswith(prefix) and name.endswith(postfix):
					key = name[len(prefix) :]
					weight_dequant_scales[key] = param.to(
						self.engine_config.Basic_Config.device_torch
					)

			layer.mlp.shared_experts = DeepSeekExpertWrapper(
				layer.mlp.shared_experts,
				layer_idx,
				-1,
				self.core_engine,
				self.engine_config,
				self.model_config,
				persistent,
				weight_dequant_scales,
			)
			for expert_idx in range(len(layer.mlp.experts)):
				# Routed expert: persistent if NOT in weight_copy_task
				if (
					"routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
					in self.weight_copy_task["routed_expert"]
				):
					persistent = False  # In offload list, needs loading
				else:
					persistent = True  # Not in offload list, pre-loaded on GPU

				prefix = (
					"model.layers."
					+ str(layer_idx)
					+ ".mlp.experts."
					+ str(expert_idx)
					+ "."
				)
				postfix = ".weight_scale_inv"
				weight_dequant_scales = {}
				for name, param in self.skeleton_state_dict.items():
					if name.startswith(prefix) and name.endswith(postfix):
						key = name[len(prefix) :]
						weight_dequant_scales[key] = param.to(
							self.engine_config.Basic_Config.device_torch
						)
				layer.mlp.experts[expert_idx] = DeepSeekExpertWrapper(
					layer.mlp.experts[expert_idx],
					layer_idx,
					expert_idx,
					self.core_engine,
					self.engine_config,
					self.model_config,
					persistent,
					weight_dequant_scales,
				)
				if persistent:
					# Persistent expert: register FP8 weights for direct GPU access
					layer.mlp.experts[expert_idx]._register_fp8_weights()
					for key, value in layer.mlp.experts[expert_idx].weight_dequant_scale.items():
						value = value.to(
							self.engine_config.Basic_Config.device_torch
						)
					# routed_expert_name = "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
					# self.fp8_weights_IPC_handle[routed_expert_name] = {}
		end_time = time.perf_counter()
		logging.debug(
			f"Expert module configuration time: {end_time - start_time:.2f} seconds"
		)

	def _config_expert_module(self):
		"""
		Replace expert module with the wrapper.

		persistent flag semantics:
		- True: weights are pre-loaded on GPU, no buffer fetch needed
		- False: weights need to be loaded from buffer each forward

		An expert is persistent if it is NOT in weight_copy_task (i.e., already on GPU).
		An expert is non-persistent if it IS in weight_copy_task (needs dynamic loading).
		"""
		start_time = time.perf_counter()
		mlp_names = ["gate_proj", "up_proj", "down_proj"]
		for layer_idx in range(
			self.loaded_model_config.first_k_dense_replace,
			len(self.model.model.layers),
		):
			layer = self.model.model.layers[layer_idx]
			# Shared expert: persistent if NOT in weight_copy_task
			if (
				"shared_expert_" + str(layer_idx)
				in self.weight_copy_task["shared_expert"]
			):
				persistent = False  # In offload list, needs loading
			else:
				persistent = True  # Not in offload list, pre-loaded on GPU

			prefix = "model.layers." + str(layer_idx) + ".mlp.shared_experts."
			postfix = ".weight_scale_inv"
			weight_dequant_scales = {}
			for name in mlp_names:
				key = prefix + name + postfix
				if key in self.skeleton_state_dict:
					weight_dequant_scales[name + postfix] = self.skeleton_state_dict[key].to(
						self.engine_config.Basic_Config.device_torch
					)



			layer.mlp.shared_experts = DeepSeekExpertWrapper(
				layer.mlp.shared_experts,
				layer_idx,
				-1,
				self.core_engine,
				self.engine_config,
				self.model_config,
				persistent,
				weight_dequant_scales,
			)
			if persistent:
					# Persistent expert: register FP8 weights for direct GPU access
					layer.mlp.shared_experts._register_fp8_weights()
					for key, value in layer.mlp.shared_experts.weight_dequant_scale.items():
						value = value.to(
							self.engine_config.Basic_Config.device_torch
						)

			for expert_idx in range(len(layer.mlp.experts)):
				# Routed expert: persistent if NOT in weight_copy_task
				if (
					"routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
					in self.weight_copy_task["routed_expert"]
				):
					persistent = False  # In offload list, needs loading
				else:
					persistent = True  # Not in offload list, pre-loaded on GPU

				prefix = (
					"model.layers."
					+ str(layer_idx)
					+ ".mlp.experts."
					+ str(expert_idx)
					+ "."
				)
				postfix = ".weight_scale_inv"
				weight_dequant_scales = {}
				# for name, param in self.skeleton_state_dict.items():
				# 	if name.startswith(prefix) and name.endswith(postfix):
				# 		key = name[len(prefix) :]
				# 		weight_dequant_scales[key] = param.to(
				# 			self.engine_config.Basic_Config.device_torch
				# 		)
				for name in mlp_names:
					key = prefix + name + postfix
					if key in self.skeleton_state_dict:
						weight_dequant_scales[name + postfix] = self.skeleton_state_dict[key].to(
							self.engine_config.Basic_Config.device_torch
						)
				layer.mlp.experts[expert_idx] = DeepSeekExpertWrapper(
					layer.mlp.experts[expert_idx],
					layer_idx,
					expert_idx,
					self.core_engine,
					self.engine_config,
					self.model_config,
					persistent,
					weight_dequant_scales,
				)
				if persistent:
					# Persistent expert: register FP8 weights for direct GPU access
					layer.mlp.experts[expert_idx]._register_fp8_weights()
					for key, value in layer.mlp.experts[expert_idx].weight_dequant_scale.items():
						value = value.to(
							self.engine_config.Basic_Config.device_torch
						)
					routed_expert_name = "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
					# self.fp8_weights_IPC_handle[routed_expert_name] = {}
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

	def _extract_dequantize_scale(self):
		self.dequant_scale = {}
		for key, param in self.skeleton_state_dict.items():
			if "weight_scale_inv" in key:
				self.dequant_scale[key] = param			

