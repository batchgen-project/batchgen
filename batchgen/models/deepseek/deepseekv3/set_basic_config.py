from batchgen.config.config import EngineConfig
import torch

def set_basic_config(engine_config: EngineConfig, **kwargs):
	"""
		Basic Config
	"""
	engine_config.Basic_Config.log_level = "info"
	
	""" Weight Dtype """
	engine_config.Basic_Config.weight_dtype = "float8_e4m3fn"
	engine_config.Basic_Config.weight_dtype_torch = torch.dtype(engine_config.Basic_Config.weight_dtype)
	
	""" KV Dtype """
	# If kv_dtype is not provided, use bf16
	if not kwargs.get('kv_dtype', None):
		engine_config.Basic_Config.kv_dtype = "bfloat16"
	else:
		if kwargs.kv_dtype.lower() in ['bfloat16', 'bf16']:
			engine_config.Basic_Config.kv_dtype = "bfloat16"
		elif kwargs.kv_dtype.lower() in ['fp8', 'float8', 'float8_e4m3fn']:
			engine_config.Basic_Config.kv_dtype = "float8_e4m3fn"
		else:
			raise ValueError(f"Unsupported kv_dtype: {kwargs.kv_dtype}, only support ['bfloat16','float8_e4m3fn']")
	engine_config.Basic_Config.kv_dtype_torch = torch.dtype(engine_config.Basic_Config.kv_dtype)

	""" Attention Dtype """
	# If attention_dtype is not provided, use bf16
	if not kwargs.get('attention_dtype', None):
		engine_config.Basic_Config.attention_dtype = "bfloat16"
	else:
		if kwargs.attention_dtype.lower() in ['bfloat16', 'bf16']:
			engine_config.Basic_Config.attention_dtype = "bfloat16"
		elif kwargs.attention_dtype.lower() in ['fp8', 'float8', 'float8_e4m3fn']:
			engine_config.Basic_Config.attention_dtype = "float8_e4m3fn"
		else:
			raise ValueError(f"Unsupported attention_dtype: {kwargs.attention_dtype}, only support ['bfloat16','float8_e4m3fn']")
	

	""" Activation Dtype """
	engine_config.Basic_Config.activation_dtype = "bfloat16"
	engine_config.Basic_Config.activation_dtype_torch = torch.dtype(engine_config.Basic_Config.activation_dtype)

	""" Device """
	if not kwargs.get('device', None):
		raise ValueError("Device must be specified")
	else:
		engine_config.Basic_Config.device = kwargs.device
		engine_config.Basic_Config.device_torch = torch.device(f"cuda:{kwargs.device}")

	""" Attn Mode """
	if not kwargs.get('attn_mode', None):
		engine_config.Basic_Config.attn_mode = 1
	else:
		if kwargs.attn_mode not in [1, 2, 3]:
			raise ValueError("Currently attn_mode must be 1, 2, or 3")
		engine_config.Basic_Config.attn_mode = kwargs.attn_mode

	""" Module Types """
	engine_config.Basic_Config.module_types = ["attn", "routed_expert", "shared_expert"]


	""" Num Threads """
	# Deprecated
	engine_config.Basic_Config.num_threads = 0
	
	""" Padding Length """
	if not kwargs.get('padding_length', None):
		raise ValueError("Padding length must be specified")
	else:
		engine_config.Basic_Config.padding_length = kwargs.padding_length

	""" Max Decoding Length """
	if not kwargs.get('max_decoding_length', None):
		raise ValueError("Max decoding length must be specified")
	else:
		engine_config.Basic_Config.max_decoding_length = kwargs.max_decoding_length


	""" Num Queries """
	if not kwargs.get('num_queries', None):
		raise ValueError("Num queries must be specified")
	else:
		engine_config.Basic_Config.num_queries = kwargs.num_queries

	""" Rank """
	if not kwargs.get('rank', None):
		raise ValueError("Rank must be specified")
	else:
		engine_config.Basic_Config.rank = kwargs.rank

	""" World Size """
	if not kwargs.get('world_size', None):
		raise ValueError("World size must be specified")
	else:
		engine_config.Basic_Config.world_size = kwargs.world_size

	""" GPU Arch """
	if not kwargs.get('gpu_arch', None):
		raise ValueError("GPU architecture must be specified")
	else:
		if kwargs.gpu_arch.lower() not in ['hooper', 'ampere']:
			raise ValueError("Currently gpu_arch must be 'hooper', or 'ampere'")
		engine_config.Basic_Config.gpu_arch = kwargs.gpu_arch.lower()


	return engine_config



