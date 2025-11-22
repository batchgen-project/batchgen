from .modeling_deepseek_v3 import (
	DeepseekV3ForCausalLM
)
from ...Wrapper import (
	Attn_Wrapper,
	Expert_Wrapper
)
import logging
from ....quantization.fp8e4m3 import (
	deepseek_v3_dequantization
)
import types
import torch.distributed as dist	
import time
import torch 
import gc
import os
from batchgen.utils import torch_gpu_mem_usage
from pplx_kernels.all_to_all import AllToAll
	
class DeepseekV3ParallelStrategyManager:
	def __init__(
		self, 
		hf_model_config, 
		engine_config, 
		model_config,
		core_engine,
		skeleton_state_dict,
		local_rank,
		global_rank,
		world_size
	):
		self.hf_model_config = hf_model_config
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
	# 	self.hf_model_config.phase = "prefill"
	# 	# self.model = DeepseekV3ForCausalLM._from_config(
	# 	# 	self.hf_model_config
	# 	# )
	# 	# logging.info(f"hf_model_config: {self.hf_model_config}")
	# 	self.model = DeepseekV3ForCausalLM(self.hf_model_config)
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

	# 		if layer_idx >= self.hf_model_config.first_k_dense_replace:
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
		logging.info("Starting configure_prefill")
		
		# Step 1: Set phase
		step_start = time.perf_counter()
		self.hf_model_config.phase = "prefill"
		logging.info(f"Set phase to prefill took {time.perf_counter() - step_start:.4f}s")
		
		# Step 2: Initialize model
		step_start = time.perf_counter()
		self.model = DeepseekV3ForCausalLM(self.hf_model_config)
		logging.info(f"DeepseekV3ForCausalLM initialization took {time.perf_counter() - step_start:.4f}s")
		
		# Step 3: Initialize data structures
		step_start = time.perf_counter()
		self.state_dict_name_map = {}
		self.weight_copy_task = {}
		self.weight_copy_task["attn"] = []
		self.weight_copy_task["routed_expert"] = []
		self.weight_copy_task["shared_expert"] = []
		logging.info(f"Initialize data structures took {time.perf_counter() - step_start:.4f}s")
		
		# Step 4: Build weight copy task mappings
		step_start = time.perf_counter()
		loop_start = time.perf_counter()
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

			if layer_idx >= self.hf_model_config.first_k_dense_replace:
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
		
		logging.info(f"Build weight copy task mappings (loop over {self.model_config.num_hidden_layers} layers) took {time.perf_counter() - loop_start:.4f}s")
		
		# Step 5: Extract dequantize scale
		step_start = time.perf_counter()
		self._extract_dequantize_scale()
		logging.info(f"_extract_dequantize_scale took {time.perf_counter() - step_start:.4f}s")
		
		# Step 6: Load model skeleton
		step_start = time.perf_counter()
		self._load_model_skeleton()
		logging.info(f"_load_model_skeleton took {time.perf_counter() - step_start:.4f}s")
		
		# Step 7: Config attention module
		step_start = time.perf_counter()
		self._config_attn_module()
		logging.info(f"_config_attn_module took {time.perf_counter() - step_start:.4f}s")
		
		# Step 8: Config expert module
		step_start = time.perf_counter()
		self._config_expert_module()
		logging.info(f"_config_expert_module took {time.perf_counter() - step_start:.4f}s")
		
		# Step 9: Config lm_head hook
		step_start = time.perf_counter()
		self._config_lm_head_hook()
		logging.info(f"_config_lm_head_hook took {time.perf_counter() - step_start:.4f}s")
		
		# Step 10: Set model to eval mode
		step_start = time.perf_counter()
		self.model.eval()
		logging.info(f"model.eval() took {time.perf_counter() - step_start:.4f}s")
		
		# Step 11: Move model to device
		step_start = time.perf_counter()
		self.model.to(self.engine_config.Basic_Config.device_torch)
		logging.info(f"model.to(device) took {time.perf_counter() - step_start:.4f}s")
		
		total_time = time.perf_counter() - start_time
		logging.info(f"configure_prefill completed in {total_time:.4f}s")
		
		return self.model, self.weight_copy_task
	
	def _warmup(self):
		# Currently only need to warmup the MoEGate
		torch._dynamo.config.inline_inbuilt_nn_modules = True
		logging.info("Start torch compile warmup")
		# from .modeling_deepseek_v3 import warmup_compiled_moe_gate
		# device = self.engine_config.Basic_Config.device_torch
		# with torch.inference_mode():
		# 	warmup_compiled_moe_gate(device)
		for layer_idx in range(self.hf_model_config.first_k_dense_replace, self.model_config.num_hidden_layers):
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

		
		# for layer_idx in range(self.hf_model_config.first_k_dense_replace, self.hf_model_config.first_k_dense_replace + 1):
		# 	layer = self.model.model.layers[layer_idx].mlp.gate
		# 	if hasattr(layer, "warmup"):
		# 		layer.warmup()


	def configure_decoding(self):
		"""
			Configure a model skeletion for decoding, 
			DP + EP 
		"""
		self.hf_model_config.phase = "decode"
		self.hf_model_config._attn_implementation = "eager"
		self.model = None
		torch.cuda.empty_cache()
		self.model = DeepseekV3ForCausalLM._from_config(
			self.hf_model_config
		)
		self.weight_copy_task = {}
		self.state_dict_name_map = {}
		self.weight_copy_task["attn"] = []
		self.weight_copy_task["routed_expert"] = []
		self.weight_copy_task["shared_expert"] = []

		# We have 8 devices and 256 experts per layer.
		# In this case, we hold NUM_LOCAL_EXPERT_PER_LAYER in the GPU and the 32 - NUM_LOCAL_EXPERT_PER_LAYER in the host memory.
		# So the self.local_routed_experts in just the names of experts in each rank's GPU.

		self.local_routed_experts = []
		self.host_routed_experts = []
		# self.expert_location_map = {}

		NUM_LOCAL_EXPERT_PER_LAYER = self.engine_config.EP_Config.num_local_expert_per_layer  
		NUM_TOTAL_EXPERTS = 256          # Total experts per layer
		NUM_EXPERT_PER_RANK = NUM_TOTAL_EXPERTS // self.world_size


		routed_expert_gpu_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
		routed_expert_gpu_end_idx = routed_expert_gpu_start_idx + NUM_LOCAL_EXPERT_PER_LAYER
		routed_expert_host_start_idx = routed_expert_gpu_end_idx
		routed_expert_host_end_idx = (self.global_rank + 1) * NUM_EXPERT_PER_RANK
		for layer_idx in range(
			self.hf_model_config.first_k_dense_replace,
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
		# assert len(self.weight_copy_task["routed_expert"]) == 0


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
			self.weight_copy_task["attn"].append("attn_" + str(layer_idx))

			if layer_idx >= self.hf_model_config.first_k_dense_replace:
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
		self._config_attn_module()
		self._config_expert_module()
		self._config_lm_head_hook()
		# Log used GPU memory
		used_memory = torch.cuda.memory_allocated(self.engine_config.Basic_Config.device_torch)
		used_memory_gb = used_memory / (1024**3)
		logging.info(f"Used GPU memory: {used_memory_gb:.2f} GB")
		self.model.eval()
		self.model.to(self.engine_config.Basic_Config.device_torch)			
		return self.model, self.weight_copy_task


	def pure_gpu_decoding(self, padding_bsz, comm=None):
		"""
			Beta 1: Load full mode into GPU.
			Duplicate attention modules and shared experts in each dp worker.
			Split routed experts.
		"""
		self.hf_model_config.phase = "decode"
		self.hf_model_config._attn_implementation = "eager"

		self.model = None
		torch.cuda.empty_cache()
		# self.model = DeepseekV3ForCausalLM._from_config(
		# 	self.hf_model_config, comm
		# )
		self.model = DeepseekV3ForCausalLM(self.hf_model_config, comm)
		""" In this case, empty copy task. """
		self.weight_copy_task = {}
		self.state_dict_name_map = {}
		self.weight_copy_task["attn"] = []
		self.weight_copy_task["routed_expert"] = []
		self.weight_copy_task["shared_expert"] = []

		NUM_TOTAL_EXPERTS = 256          # Total experts per layer
		NUM_EXPERT_PER_RANK = NUM_TOTAL_EXPERTS // self.world_size

		routed_expert_gpu_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
		routed_expert_gpu_end_idx = routed_expert_gpu_start_idx + NUM_EXPERT_PER_RANK

		self.local_routed_experts = []
		for layer_idx in range(
			self.hf_model_config.first_k_dense_replace,
			self.model_config.num_hidden_layers,
		):
			# The first NUM_LOCAL_EXPERT_PER_LAYER in each part associated with the corresponding rank.
			# The rest of the experts in the part are stored in the host memory.
			for expert_idx in range(routed_expert_gpu_start_idx, routed_expert_gpu_end_idx):
				self.local_routed_experts.append(
					"routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
				)

		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage before configure wrapper: {torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory before configure wrapper: {free_memory} GB / {total_memory} GB"
		)

		self._extract_dequantize_scale()
		self._load_model_skeleton()
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage after load model skeleton {torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory after load model skeleton: {free_memory} GB / {total_memory} GB"
		)
		self._load_local_routed_experts()
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage after load local experts{torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory after load local experts: {free_memory} GB / {total_memory} GB"
		)
		self._load_attn_module()
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage after load attn: {torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory after load attn: {free_memory} GB / {total_memory} GB"
		)
		self._load_shared_expert_module()
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage after load shared experts{torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory after load shared experts: {free_memory} GB / {total_memory} GB"
		)
		self._config_attn_module()
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage after config attn{torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory after config attn: {free_memory} GB / {total_memory} GB"
		)
		self._config_expert_module()
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage after _config_expert_module: {torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory after _config_expert_module: {free_memory} GB / {total_memory} GB"
		)
		self._config_lm_head_hook()
		self._init_mode_decoding()
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage after _init_mode_decoding: {torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory after _init_mode_decoding: {free_memory} GB / {total_memory} GB"
		)		
		self._init_decoding_padding_bsz(padding_bsz)
		
		enable_ata = os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0")
		if enable_ata == "1":
			self._init_ata_comms(padding_bsz)
		used_memory = torch.cuda.memory_allocated(self.engine_config.Basic_Config.device_torch)
		used_memory_gb = used_memory / (1024**3)
		logging.info(f"Used GPU memory: {used_memory_gb:.2f} GB")
		self.model.eval()
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage before copy model instance to device: {torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory before copy model instance to device: {free_memory} GB / {total_memory} GB"
		)
		self.model.to(self.engine_config.Basic_Config.device_torch)
		free_memory, total_memory = torch.cuda.mem_get_info()
		free_memory = free_memory / 1024 / 1024 / 1024
		total_memory = total_memory / 1024 / 1024 / 1024
		logging.info(
			f"Rank: {self.rank} Device torch memory usage after copy model instance to device: {torch.cuda.memory_allocated(self.local_rank) / (1024**3)} GB / {total_memory} GB"
		)
		logging.info(
			f"Rank: {self.rank} Device torch free memory after copy model instance to device: {free_memory} GB / {total_memory} GB"
		)
		self._warmup()
		return self.model, self.weight_copy_task

	

	def _init_decoding_padding_bsz(self, padding_bsz):
		"""
		Initialize the padding batch size for decoding.
		This is used to set the padding size for the input sequences.
		"""
		for layer_idx in range(
			self.hf_model_config.first_k_dense_replace,
			self.model_config.num_hidden_layers,
		):
			layer = self.model.model.layers[layer_idx].mlp
			if hasattr(layer, "init_num_tokens"):
				layer.init_num_tokens(padding_bsz)

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
		self.num_tokens_per_rank = padding_bsz

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
			self.hf_model_config.first_k_dense_replace,
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
		for layer_idx in range(
			self.hf_model_config.first_k_dense_replace,
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
			self.hf_model_config.first_k_dense_replace,
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
			if self.engine_config.Basic_Config.gpu_arch == "hooper":
				from ....attention.mla.fa3_backend import (
					mla_prefill_flashattention3, 
					mla_prefill_flashattention3_w8a16_deepgemm,
					mla_prefill_flashattention3_fused_dequant
				)
				from ....attention.mla.flashmla_backend import (
					mla_decoding_flashmla,
					mla_decoding_flashmla_v2,
					fused_get_query_states_triton,
					# mla_decoding_flashmla_attn_mode_3,
					mla_decoding_flashmla_attn_mode_3_bf16,
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
						mla_decoding_flashmla_attn_mode_3_bf16, attn_module
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


			if "attn_" + str(layer_idx) in self.weight_copy_task["attn"]:
				get_weights = True
			else:
				get_weights = False
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
				get_weights,
				weight_dequant_scales,
			)
			self.model.model.layers[layer_idx].self_attn = attn_wrapper_instance
			if get_weights == False:
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
			if layer_idx >= self.hf_model_config.first_k_dense_replace:
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
					logging.info(f"Dequantizing and loading parameter: {key}")
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
		if dist.get_rank() == 0:
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
		"""
		start_time = time.perf_counter()
		for layer_idx in range(
			self.hf_model_config.first_k_dense_replace,
			len(self.model.model.layers),
		):
			layer = self.model.model.layers[layer_idx]
			if (
				"shared_expert_" + str(layer_idx)
				in self.weight_copy_task["shared_expert"]
			):
				get_weights = True
			else:
				get_weights = False

			prefix = "model.layers." + str(layer_idx) + ".mlp.shared_experts."
			postfix = ".weight_scale_inv"
			weight_dequant_scales = {}
			for name, param in self.skeleton_state_dict.items():
				if name.startswith(prefix) and name.endswith(postfix):
					key = name[len(prefix) :]
					weight_dequant_scales[key] = param.to(
						self.engine_config.Basic_Config.device_torch
					)

			layer.mlp.shared_experts = Expert_Wrapper(
				layer.mlp.shared_experts,
				layer_idx,
				-1,
				self.core_engine,
				self.engine_config,
				self.model_config,
				get_weights,
				weight_dequant_scales,
			)
			for expert_idx in range(len(layer.mlp.experts)):
				if (
					"routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
					in self.weight_copy_task["routed_expert"]
				):
					get_weights = True
				else:
					get_weights = False

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
				layer.mlp.experts[expert_idx] = Expert_Wrapper(
					layer.mlp.experts[expert_idx],
					layer_idx,
					expert_idx,
					self.core_engine,
					self.engine_config,
					self.model_config,
					get_weights,
					weight_dequant_scales,
				)
				if get_weights == False:
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
		"""
		start_time = time.perf_counter()
		mlp_names = ["gate_proj", "up_proj", "down_proj"]
		for layer_idx in range(
			self.hf_model_config.first_k_dense_replace,
			len(self.model.model.layers),
		):
			layer = self.model.model.layers[layer_idx]
			if (
				"shared_expert_" + str(layer_idx)
				in self.weight_copy_task["shared_expert"]
			):
				get_weights = True
			else:
				get_weights = False

			prefix = "model.layers." + str(layer_idx) + ".mlp.shared_experts."
			postfix = ".weight_scale_inv"
			weight_dequant_scales = {}
			for name in mlp_names:
				key = prefix + name + postfix
				if key in self.skeleton_state_dict:
					weight_dequant_scales[name + postfix] = self.skeleton_state_dict[key].to(
						self.engine_config.Basic_Config.device_torch
					)
				


			layer.mlp.shared_experts = Expert_Wrapper(
				layer.mlp.shared_experts,
				layer_idx,
				-1,
				self.core_engine,
				self.engine_config,
				self.model_config,
				get_weights,
				weight_dequant_scales,
			)
			if get_weights == False:
					layer.mlp.shared_experts._register_fp8_weights()
					for key, value in layer.mlp.shared_experts.weight_dequant_scale.items():
						value = value.to(
							self.engine_config.Basic_Config.device_torch
						)
			
			for expert_idx in range(len(layer.mlp.experts)):
				if (
					"routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
					in self.weight_copy_task["routed_expert"]
				):
					get_weights = True
				else:
					get_weights = False

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
				layer.mlp.experts[expert_idx] = Expert_Wrapper(
					layer.mlp.experts[expert_idx],
					layer_idx,
					expert_idx,
					self.core_engine,
					self.engine_config,
					self.model_config,
					get_weights,
					weight_dequant_scales,
				)
				if get_weights == False:
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

