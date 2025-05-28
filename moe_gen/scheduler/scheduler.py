from ..config.config import EngineConfig
class Scheduler:
	def __version__(self):
		""" 
			Exclusively for H20 Single Node and DeepSeek-R1
		"""
		return "0.1.0"
	
	def __init__(self, Max_Prompt_Length, Max_Response_Length, ):
		self.config = EngineConfig()
		self.Max_Prompt_Length = Max_Prompt_Length
		self.Max_Response_Length = Max_Response_Length
		self.Max_Context_Length = Max_Prompt_Length + Max_Response_Length

	def generate_config(self) -> EngineConfig:
		self._set_default_configs()
		"""
			Configure the rest.
		"""
		DEFAULT_MEM_FRAC = 0.9
		# MAGIC_NUM = self.compute_profiler.profile(attn_decoding_module)
		MAGIC_NUM =  224000
		
		attn_decoding_micro_batch_size = MAGIC_NUM // self.Max_Prompt_Length
		est_kv_cp_t_per_micro_batch = attn_decoding_micro_batch_size * self.Max_Context_Length * 576 / (1024 ** 3) / 52 * 1000 # in ms
		# num_k_buffer = self.compute_profiler.profile(MoE_module) // est_kv_cp_t_per_micro_batch + 2
		num_k_buffer = 8
		k_buffer_size = num_k_buffer * attn_decoding_micro_batch_size * self.Max_Context_Length * 576 / (1024 ** 3) # in GB


		available_gpu_mem = 96 * DEFAULT_MEM_FRAC  # Assuming 96GB GPU memory
		non_static_memory_usage = 6 + max(self.mem_profiler.profile(attn_decoding_module), self.mem_profiler.profile(MoE_module)) + k_buffer_size 
		available_memory_for_expert_cache = available_gpu_mem - non_static_memory_usage
		num_local_expert_per_layer = int(available_memory_for_expert_cache // 2.4) # Each expert cache is around 2.4GB
		num_decoding_module_buffer_routed_expert = 32 - num_local_expert_per_layer + 2

		# Update the config with the computed values
		self.config.Module_Batching_Config.attn_decoding_micro_batch_size = attn_decoding_micro_batch_size
		self.config.GPU_Buffer_Config.num_k_buffer = num_k_buffer
		self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = num_decoding_module_buffer_routed_expert
		self.config.EP_Config.num_local_expert_per_layer = num_local_expert_per_layer


	def _set_default_configs(self):
		""" Default Basic Config """
		self.config.Basic_Config = {
			"log_level": "info",
			"weight_dtype": "float8_e4m3fn",
			"kv_dtype": "float8_e4m3fn",
			"activation_dtype": "bfloat16",
			"module_types": ["attn", "routed_expert", "shared_expert"],
			"gpu_arch": "hooper"
		}

		""" Default Module Batching Config """
		self.config.Module_Batching_Config = {
			"attn_prefill_micro_batch_size": 8,
			"MoE_prefill_micro_batch_size": 8,
			"expert_prefill_batch_size_upper_bound": 2048,
			"attn_decoding_micro_batch_size": None,
			"MoE_decoding_micro_batch_size": None,
			"expert_decoding_batch_size_upper_bound": 2048
		}

		""" Default GPU Buffer Config """
		self.config.GPU_Buffer_Config = {
			"num_prefill_module_buffer": {
				"attn": 1,
				"routed_expert": 8,
				"shared_expert": 1
			},
			"num_decoding_module_buffer": {
				"attn": 1,
				"routed_expert": None,
				"shared_expert": 1
			},
			"num_k_buffer": None,
			"num_v_buffer": 0,
			"kv_buffer_num_tokens": None
		}

		""" Default EP Config """
		self.config.EP_Config = {
			"enable": True,
			"num_local_expert_per_layer": None
		}

