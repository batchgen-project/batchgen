from batchgen.config.config import EngineConfig
import torch
import logging

def set_basic_config(engine_config: EngineConfig, input_arguments):
	"""
		Basic Config
	"""
	engine_config.Basic_Config.log_level = "info"
	
	""" Weight Dtype """
	engine_config.Basic_Config.weight_dtype = "float8_e4m3fn"
	engine_config.Basic_Config.weight_dtype_torch = torch.float8_e4m3fn
	
	""" KV Dtype """
	# If kv_dtype is not provided, use bf16
	if not input_arguments.get('kv_dtype', None):
		logging.info("kv_dtype is not provided, using bfloat16 as default")
		engine_config.Basic_Config.kv_dtype = "bfloat16"
	else:
		logging.info(f"kv_dtype is set to {input_arguments.kv_dtype}")
		logging.info(f"attn_mode is set to {input_arguments.get('attn_mode')}")
		if input_arguments.kv_dtype.lower() in ['bfloat16', 'bf16']:
			engine_config.Basic_Config.kv_dtype = "bfloat16"
		elif input_arguments.kv_dtype.lower() in ['fp8', 'float8', 'float8_e4m3fn']:
			engine_config.Basic_Config.kv_dtype = "float8_e4m3fn"
		else:
			raise ValueError(f"Unsupported kv_dtype: {input_arguments.kv_dtype}, only support ['bfloat16','float8_e4m3fn']")
	# engine_config.Basic_Config.kv_dtype_torch = torch.dtype(engine_config.Basic_Config.kv_dtype)
	if engine_config.Basic_Config.kv_dtype == "bfloat16":
		engine_config.Basic_Config.kv_dtype_torch = torch.bfloat16
	elif engine_config.Basic_Config.kv_dtype == "float8_e4m3fn":
		engine_config.Basic_Config.kv_dtype_torch = torch.float8_e4m3fn


	""" Attention Dtype """
	# If attention_dtype is not provided, use bf16
	if not input_arguments.get('attention_dtype', None):
		logging.info("attention_dtype is not provided, using bfloat16 as default")
		engine_config.Basic_Config.attention_dtype = "bfloat16"
	else:
		if input_arguments.attention_dtype.lower() in ['bfloat16', 'bf16']:
			engine_config.Basic_Config.attention_dtype = "bfloat16"
		elif input_arguments.attention_dtype.lower() in ['fp8', 'float8', 'float8_e4m3fn']:
			engine_config.Basic_Config.attention_dtype = "float8_e4m3fn"
		else:
			raise ValueError(f"Unsupported attention_dtype: {input_arguments.attention_dtype}, only support ['bfloat16','float8_e4m3fn']")
	

	""" Activation Dtype """
	engine_config.Basic_Config.activation_dtype = "bfloat16"
	# engine_config.Basic_Config.activation_dtype_torch = torch.dtype(engine_config.Basic_Config.activation_dtype)
	engine_config.Basic_Config.activation_dtype_torch = torch.bfloat16

	""" Device """
	logging.info(f"device: {input_arguments.get('device')}")
	if not input_arguments.get('device', None):
		raise ValueError("Device must be specified")
	else:
		engine_config.Basic_Config.device = input_arguments.device
		engine_config.Basic_Config.device_torch = torch.device(f"cuda:{input_arguments.device}")

	# """ Attn Mode """
	# if not input_arguments.get('attn_mode', None):
	# 	# raise ValueError("Attn mode must be specified")
	# else:
	# 	if input_arguments.attn_mode not in [1, 2, 3]:
	# 		raise ValueError("Currently attn_mode must be 1, 2, or 3")
	# 	engine_config.Basic_Config.attn_mode = input_arguments.attn_mode

	""" Module Types """
	engine_config.Basic_Config.module_types = ["attn", "routed_expert", "shared_expert"]


	""" Num Threads """
	# Deprecated
	engine_config.Basic_Config.num_threads = 0
	
	""" Padding Length """
	if not input_arguments.get('padding_length', None):
		raise ValueError("Padding length must be specified")
	else:
		engine_config.Basic_Config.padding_length = input_arguments.padding_length

	""" Max Decoding Length """
	if not input_arguments.get('max_decoding_length', None):
		raise ValueError("Max decoding length must be specified")
	else:
		engine_config.Basic_Config.max_decoding_length = input_arguments.max_decoding_length


	""" Num Queries """
	if not input_arguments.get('num_queries', None):
		raise ValueError("Num queries must be specified")
	else:
		engine_config.Basic_Config.num_queries = input_arguments.num_queries

	""" Rank """
	if not input_arguments.get('rank', None):
		raise ValueError("Rank must be specified")
	else:
		engine_config.Basic_Config.rank = input_arguments.rank

	""" World Size """
	if not input_arguments.get('world_size', None):
		raise ValueError("World size must be specified")
	else:
		engine_config.Basic_Config.world_size = input_arguments.world_size

	""" GPU Arch """
	if not input_arguments.get('gpu_arch', None):
		raise ValueError("GPU architecture must be specified")
	else:
		if input_arguments.gpu_arch.lower() not in ['hooper', 'ampere']:
			raise ValueError("Currently gpu_arch must be 'hooper', or 'ampere'")
		engine_config.Basic_Config.gpu_arch = input_arguments.gpu_arch.lower()


	return engine_config



