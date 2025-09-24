from batchgen.config.config import EngineConfig
import logging
class Scheduler:
	def __version__(self):
		""" 
			Currently exclusively for H20
		"""
		return "0.1.6"
	
	def __init__(self):
		# self.config = EngineConfig()

		pass

	def generate_config(self, config) -> EngineConfig:
		self.config = config
		self.Max_Prompt_Length = config.Basic_Config.padding_length
		self.Max_Response_Length = config.Basic_Config.max_decoding_length
		self.Max_Context_Length = self.Max_Prompt_Length + self.Max_Response_Length
		self.world_size = config.Basic_Config.world_size	

		self._set_default_configs()
		"""
			Configure the rest.
		"""
		DEFAULT_MEM_FRAC = 0.75
		# MAGIC_NUM = self.compute_profiler.profile(attn_decoding_module)
		MAGIC_NUM = 224000
		EXPERT_PER_RANK = 256 // self.world_size
		assert EXPERT_PER_RANK > 0, "EXPERT_PER_RANK must be greater than 0"
		if self.world_size > 8:
			self.config.Basic_Config.attn_mode = 3 # DUAL-NODE.
		else:
			self.config.Basic_Config.attn_mode = 1 # SINGLE-NODE.

		
		attn_decoding_micro_batch_size = MAGIC_NUM // self.Max_Prompt_Length
		# Down round attn_decoding_micro_batch_size to nearest expon of 2
		attn_decoding_micro_batch_size = 2 ** (attn_decoding_micro_batch_size.bit_length() - 1)
		est_kv_cp_t_per_micro_batch = attn_decoding_micro_batch_size * self.Max_Context_Length * 576 / (1024 ** 3) / 52 * 1000 # in ms
		# num_k_buffer = self.compute_profiler.profile(MoE_module) // est_kv_cp_t_per_micro_batch + 2
		num_k_buffer = 6
		k_buffer_size = num_k_buffer * attn_decoding_micro_batch_size * self.Max_Context_Length * 576 / (1024 ** 3) # in GB


		available_gpu_mem = 96 * DEFAULT_MEM_FRAC  # Assuming 96GB GPU memory
		# non_static_memory_usage = 6 + max(self.mem_profiler.profile(attn_decoding_module), self.mem_profiler.profile(MoE_module)) + k_buffer_size 
		model_skeleton_size = 6
		cuda_page_table_default_size = 5
		NCCL_default_buffer_usage = 2.5 # in GB
		non_static_memory_usage = k_buffer_size + model_skeleton_size + cuda_page_table_default_size
		available_memory_for_expert_cache = available_gpu_mem - non_static_memory_usage
		num_local_expert_per_layer = min(EXPERT_PER_RANK, int(available_memory_for_expert_cache // 2.4)) # Each expert cache is around 2.4GB
		num_decoding_module_buffer_routed_expert = EXPERT_PER_RANK - num_local_expert_per_layer + 2
		if self.config.Basic_Config.attn_mode == 3:
			num_local_expert_per_layer = EXPERT_PER_RANK
			expert_size = num_local_expert_per_layer * 2.4
			# Fix 
			self.per_seq_size = self.Max_Context_Length * 61 * 576 / (1024 ** 3) # in GB
			self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = (
				int((available_gpu_mem - model_skeleton_size - cuda_page_table_default_size - expert_size - NCCL_default_buffer_usage) / self.per_seq_size)
			)
			logging.info(f"Max Available MoE decoding micro batch size: {self.config.Module_Batching_Config.MoE_decoding_micro_batch_size}")
		if num_local_expert_per_layer == EXPERT_PER_RANK:
			num_decoding_module_buffer_routed_expert = 0

		# Update the config with the computed values
		self.config.Module_Batching_Config.attn_decoding_micro_batch_size = attn_decoding_micro_batch_size
		self.config.GPU_Buffer_Config.num_k_buffer = num_k_buffer
		self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = num_decoding_module_buffer_routed_expert
		self.config.EP_Config.num_local_expert_per_layer = num_local_expert_per_layer

			
		if self.config.Basic_Config.attn_mode == 3:
			# decoding module buffer are zero.
			self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = 0
			self.config.GPU_Buffer_Config.num_decoding_module_buffer["attn"] = 0
			self.config.GPU_Buffer_Config.num_decoding_module_buffer["shared_expert"] = 0
			
			self.config.GPU_Buffer_Config.num_k_buffer = 0



		if self.Max_Prompt_Length > 14000:
			# For longer context, shink prefill attention micro batch size and MoE prefill micro batch size accordingly
			factor = self.Max_Prompt_Length / 14000
			self.config.Module_Batching_Config.attn_prefill_micro_batch_size = int(self.config.Module_Batching_Config.attn_prefill_micro_batch_size / factor)
			self.config.Module_Batching_Config.MoE_prefill_micro_batch_size = int(self.config.Module_Batching_Config.MoE_prefill_micro_batch_size / factor)
		
		else:
			# For shorter context, increse prefill attention micro batch size and MoE prefill micro batch size accordingly
			factor = self.Max_Prompt_Length / 14000
			self.config.Module_Batching_Config.attn_prefill_micro_batch_size = min(32, int(self.config.Module_Batching_Config.attn_prefill_micro_batch_size / factor))
			self.config.Module_Batching_Config.MoE_prefill_micro_batch_size = min(32, int(self.config.Module_Batching_Config.MoE_prefill_micro_batch_size / factor))

		

		return self.config


	def _set_default_configs(self):
		""" Default Basic Config """
		# self.config.Basic_Config.log_level = "info"
		# self.config.Basic_Config.weight_dtype = "float8_e4m3fn"
		# self.config.Basic_Config.kv_dtype = "float8_e4m3fn"
		# self.config.Basic_Config.activation_dtype = "bfloat16"
		# self.config.Basic_Config.module_types = ["attn", "routed_expert", "shared_expert"]
		# self.config.Basic_Config.gpu_arch = "hooper"
		
		
		""" Default Module Batching Config """
		self.config.Module_Batching_Config.attn_prefill_micro_batch_size = 8
		self.config.Module_Batching_Config.MoE_prefill_micro_batch_size = 8	
		self.config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 4096

		self.config.Module_Batching_Config.attn_decoding_micro_batch_size = None
		self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = None
		self.config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 2048
				

		""" Default GPU Buffer Config """
		self.config.GPU_Buffer_Config.num_prefill_module_buffer = {
			"attn": 1,
			"routed_expert": 8,
			"shared_expert": 1
		}
		self.config.GPU_Buffer_Config.num_decoding_module_buffer = {
			"attn": 1,
			"routed_expert": None,  # This will be set later based on available memory
			"shared_expert": 1
		}
		self.config.GPU_Buffer_Config.num_k_buffer = None  # This will be set later based on available memory
		self.config.GPU_Buffer_Config.num_v_buffer = 0  # No value buffer for now
		# self.config.GPU_Buffer_Config.kv_buffer_num_tokens = None  # This will be set later based on available memory
		

		""" Default EP Config """
		self.config.EP_Config.enable = True
		self.config.EP_Config.num_local_expert_per_layer = None
