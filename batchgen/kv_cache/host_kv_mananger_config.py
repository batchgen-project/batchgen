from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig
from batchgen.models.engine_loader import core_engine as bg_lib

HOST_KV_SHM_NAME = os.environ.get(
	"BATCHGEN_HOST_KV_SHM_NAME", "batchgen_host_kv_cache"
)

__all__ = [
	"build_host_kv_config",
	"build_gpu_kv_config",
	"HOST_KV_SHM_NAME",
]


def _dtype_size_bytes(dtype: str) -> int:
	"""Returns the storage size in bytes for the provided dtype string."""

	normalized = dtype.lower()
	if normalized in {"bfloat16", "float16"}:
		return 2
	if normalized == "float32":
		return 4
	if normalized in {"float8_e4m3fn", "float8_e5m2"}:
		return 1
	raise ValueError(f"Unsupported kv dtype '{dtype}'")


def _torch_dtype_from_string(dtype: str) -> torch.dtype:
	mapping = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
		"float8_e4m3fn": torch.float8_e4m3fn,
		"float8_e5m2": torch.float8_e5m2,
	}
	key = dtype.strip().lower()
	if key not in mapping:
		raise ValueError(f"Unsupported kv dtype '{dtype}' for torch")
	return mapping[key]


@dataclass(frozen=True)
class _HostKVModelProfile:
	num_layers: int
	num_k_heads: int
	k_head_dim: int
	page_size: int = 64
	num_v_heads: int = 0
	v_head_dim: int = 0
	kv_dtype: str = "bfloat16"
	sequence_table_capacity: int | None = None
	alignment_bytes: int = 64

	def bytes_per_page(self) -> int:
		element_bytes = _dtype_size_bytes(self.kv_dtype)
		k_bytes = (
			self.page_size * self.num_k_heads * self.k_head_dim * element_bytes
		)
		v_bytes = (
			self.page_size * self.num_v_heads * self.v_head_dim * element_bytes
		)
		return k_bytes + v_bytes


_DEEPSEEK_MLA_PROFILE = _HostKVModelProfile(
	num_layers=61,
	num_k_heads=1,
	k_head_dim=576,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="bfloat16",
)

_DEEPSEEK_V4_FLASH_PROFILE = _HostKVModelProfile(
	num_layers=43,
	num_k_heads=1,
	k_head_dim=512,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="bfloat16",
)

_DEEPSEEK_V4_PRO_PROFILE = _HostKVModelProfile(
	num_layers=61,
	num_k_heads=1,
	k_head_dim=512,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="bfloat16",
)

# GPT-OSS-120B: GQA with 8 KV heads, head_dim=64, 36 layers
_GPT_OSS_GQA_PROFILE = _HostKVModelProfile(
	num_layers=36,
	num_k_heads=8,
	k_head_dim=64,
	num_v_heads=8,
	v_head_dim=64,
	kv_dtype="bfloat16",
)

# DeepSeek-V3.2 DSA: same MLA cache as V3, plus a separate indexer cache
_DEEPSEEK_V3_2_INDEXER_PROFILE = _HostKVModelProfile(
	num_layers=61,
	num_k_heads=1,
	k_head_dim=128,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="bfloat16",
)

# MiniMax-M2.5: GQA with 8 KV heads, head_dim=128, 62 layers
_MINIMAX_M25_GQA_PROFILE = _HostKVModelProfile(
	num_layers=62,
	num_k_heads=8,
	k_head_dim=128,
	num_v_heads=8,
	v_head_dim=128,
	kv_dtype="bfloat16",
)

# GLM-5: MLA cache (78 layers, compressed_kv_dim=576, same as DeepSeek)
_GLM5_MLA_PROFILE = _HostKVModelProfile(
	num_layers=78,
	num_k_heads=1,
	k_head_dim=576,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="bfloat16",
)

# GLM-5 DSA: indexer cache (78 layers, MQA single-head K, head_dim=128)
_GLM5_INDEXER_PROFILE = _HostKVModelProfile(
	num_layers=78,
	num_k_heads=1,
	k_head_dim=128,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="bfloat16",
)

_PROFILE_REGISTRY: Dict[str, _HostKVModelProfile] = {
	"deepseek_mla": _DEEPSEEK_MLA_PROFILE,
	"deepseek_v4_flash": _DEEPSEEK_V4_FLASH_PROFILE,
	"deepseek_v4_pro": _DEEPSEEK_V4_PRO_PROFILE,
	"deepseek_v3_2_indexer": _DEEPSEEK_V3_2_INDEXER_PROFILE,
	"gpt_oss_gqa": _GPT_OSS_GQA_PROFILE,
	"minimax_m25_gqa": _MINIMAX_M25_GQA_PROFILE,
	"glm5_mla": _GLM5_MLA_PROFILE,
	"glm5_indexer": _GLM5_INDEXER_PROFILE,
}

_PROFILE_ALIASES: Dict[str, str] = {}
for canonical, aliases in {
	"deepseek_mla": (
		"deepseek-ai/deepseek-r1",
		"deepseek-ai/deepseek-v3",
		"deepseek/deepseek-r1",
		"deepseek/deepseek-v3",
		"deepseek-r1",
		"deepseek-v3",
		"deepseek-ai/deepseek-v3.2",
		"deepseek/deepseek-v3.2",
		"deepseek-v3.2",
		"moonshotai/kimi-k2.5",
		"moonshotai/kimi-k2.6",
		"moonshotai/kimi-k25",
		"moonshotai/kimi-k26",
		"kimi-k2.5",
		"kimi-k2.6",
		"kimi-k25",
		"kimi-k26",
		"kimi",
	),
	"gpt_oss_gqa": (
		"openai/gpt-oss-120b",
		"gpt-oss-120b",
	),
	"deepseek_v4_flash": (
		"deepseek-ai/deepseek-v4-flash",
		"deepseek/deepseek-v4-flash",
		"deepseek-v4-flash",
	),
	"deepseek_v4_pro": (
		"deepseek-ai/deepseek-v4-pro",
		"deepseek/deepseek-v4-pro",
		"deepseek-v4-pro",
	),
	"minimax_m25_gqa": (
		"minimaxai/minimax-m2.5",
		"minimax-m2.5",
		"minimax",
	),
	"glm5_mla": (
		"zai-org/glm-5-fp8",
		"zai-org/glm-5",
		"glm-5-fp8",
		"glm-5",
		# GLM-5.1: architecturally identical to GLM-5 (same 78-layer MLA graph,
		# compressed_kv_dim=576), shares the MLA host-KV profile.
		"zai-org/glm-5.1-fp8",
		"zai-org/glm-5.1",
		"glm-5.1-fp8",
		"glm-5.1",
	),
}.items():
	for alias in aliases:
		_PROFILE_ALIASES[alias.lower()] = canonical

# DSA indexer profile aliases (used by build_*_aux functions)
_INDEXER_PROFILE_ALIASES: Dict[str, str] = {}
for canonical, aliases in {
	"deepseek_v3_2_indexer": (
		"deepseek-ai/deepseek-v3.2",
		"deepseek/deepseek-v3.2",
		"deepseek-v3.2",
	),
	"glm5_indexer": (
		"zai-org/glm-5-fp8",
		"zai-org/glm-5",
		"glm-5-fp8",
		"glm-5",
		# GLM-5.1: identical DSA indexer (32 heads, head_dim=128, 78 layers).
		"zai-org/glm-5.1-fp8",
		"zai-org/glm-5.1",
		"glm-5.1-fp8",
		"glm-5.1",
	),
}.items():
	for alias in aliases:
		_INDEXER_PROFILE_ALIASES[alias.lower()] = canonical


def _resolve_indexer_profile(model_name: str) -> _HostKVModelProfile | None:
	"""Maps a model name to its DSA indexer profile, or None if not a DSA model."""
	alias = model_name.strip().lower()
	canonical = _INDEXER_PROFILE_ALIASES.get(alias)
	if canonical is None:
		return None
	return _PROFILE_REGISTRY[canonical]


def _resolve_profile(model_name: str) -> _HostKVModelProfile:
	"""Maps a user supplied model name to a cached profile."""

	if not isinstance(model_name, str):
		raise ValueError("model_name must be a string")
	alias = model_name.strip().lower()
	if alias not in _PROFILE_ALIASES:
		raise ValueError(f"Unsupported model '{model_name}' for host KV cache")
	return _PROFILE_REGISTRY[_PROFILE_ALIASES[alias]]


def build_host_kv_config(model_name: str, host_kv_cache_size: int) -> Any:
	"""Builds a core HostPagedKVConfig for the given model and host budget."""

	if host_kv_cache_size is None:
		raise ValueError("host_kv_cache_size must be a positive integer")
	try:
		host_budget = int(host_kv_cache_size)
	except (TypeError, ValueError) as exc:
		raise ValueError(
			"host_kv_cache_size must be a positive integer"
		) from exc

	if host_budget <= 0:
		raise ValueError("host_kv_cache_size must be a positive integer")

	profile = _resolve_profile(model_name)
	bytes_per_page = profile.bytes_per_page()
	if bytes_per_page <= 0:
		raise ValueError(f"Invalid profile definition for '{model_name}'")

	denom = profile.num_layers * bytes_per_page
	if host_budget < denom:
		raise ValueError(
			"host_kv_cache_size is too small to allocate even one page per layer"
		)

	num_pages_per_layer = host_budget // denom
	config = bg_lib.HostPagedKVConfig()
	config.shm_name = HOST_KV_SHM_NAME
	config.num_layers = profile.num_layers
	config.num_pages = num_pages_per_layer
	config.page_size_tokens = profile.page_size
	config.num_k_heads = profile.num_k_heads
	config.k_head_dim = profile.k_head_dim
	config.num_v_heads = profile.num_v_heads
	config.v_head_dim = profile.v_head_dim
	config.k_element_size_bytes = _dtype_size_bytes(profile.kv_dtype)
	config.v_element_size_bytes = (
		0 if profile.num_v_heads == 0 else config.k_element_size_bytes
	)
	config.sequence_table_capacity = (
		profile.sequence_table_capacity or config.num_pages
	)
	config.alignment_bytes = profile.alignment_bytes
	return config


def _normalize_sequence_tokens(sequence_tokens: Sequence[int]) -> list[int]:
	if not sequence_tokens:
		raise ValueError("sequence_tokens must contain at least one element")
	normalized: list[int] = []
	for idx, value in enumerate(sequence_tokens):
		try:
			token_count = int(value)
		except (
			TypeError,
			ValueError,
		) as exc:  # pragma: no cover - defensive branch
			raise ValueError(
				f"sequence_tokens[{idx}] must be an integer, got {value!r}"
			) from exc
		if token_count <= 0:
			raise ValueError(
				f"sequence_tokens[{idx}] must be > 0, got {token_count}"
			)
		normalized.append(token_count)
	return normalized


def _compute_gpu_page_capacity(
	sequence_tokens: Sequence[int], page_size_tokens: int
) -> int:
	normalized = _normalize_sequence_tokens(sequence_tokens)
	total_pages = 0
	for token_count in normalized:
		total_pages += (token_count // page_size_tokens) + 1
	if total_pages <= 0:
		raise ValueError("Computed GPU page capacity must be positive")
	return total_pages


def build_gpu_kv_config(
	model_name: str, sequence_tokens: Sequence[int]
) -> GPUPagedKVConfig:
	"""Builds a GPUPagedKVConfig sized for the provided sequence lengths."""

	profile = _resolve_profile(model_name)
	num_pages = _compute_gpu_page_capacity(sequence_tokens, profile.page_size)
	return GPUPagedKVConfig(
		num_layers=profile.num_layers,
		num_pages=num_pages,
		page_size_tokens=profile.page_size,
		num_k_heads=profile.num_k_heads,
		k_head_dim=profile.k_head_dim,
		num_v_heads=profile.num_v_heads,
		v_head_dim=profile.v_head_dim,
		kv_dtype=_torch_dtype_from_string(profile.kv_dtype),
	)


HOST_KV_AUX_SHM_NAME = os.environ.get(
	"BATCHGEN_HOST_KV_AUX_SHM_NAME", "batchgen_host_kv_cache_aux"
)


def is_dsa_model(model_name: str) -> bool:
	"""Returns True if the model uses DeepSeek Sparse Attention (has indexer cache)."""
	return _resolve_indexer_profile(model_name) is not None


def build_gpu_kv_config_aux(
	model_name: str, sequence_tokens: Sequence[int]
) -> GPUPagedKVConfig | None:
	"""Builds a GPUPagedKVConfig for the DSA indexer cache, or None if not a DSA model."""

	profile = _resolve_indexer_profile(model_name)
	if profile is None:
		return None
	num_pages = _compute_gpu_page_capacity(sequence_tokens, profile.page_size)
	return GPUPagedKVConfig(
		num_layers=profile.num_layers,
		num_pages=num_pages,
		page_size_tokens=profile.page_size,
		num_k_heads=profile.num_k_heads,
		k_head_dim=profile.k_head_dim,
		num_v_heads=profile.num_v_heads,
		v_head_dim=profile.v_head_dim,
		kv_dtype=_torch_dtype_from_string(profile.kv_dtype),
	)


def build_host_kv_config_aux(model_name: str, host_kv_cache_size: int) -> Any | None:
	"""Builds a HostPagedKVConfig for the DSA indexer host cache, or None."""

	profile = _resolve_indexer_profile(model_name)
	if profile is None:
		return None

	host_budget = int(host_kv_cache_size)
	if host_budget <= 0:
		raise ValueError("host_kv_cache_size must be a positive integer")

	bytes_per_page = profile.bytes_per_page()
	denom = profile.num_layers * bytes_per_page
	num_pages_per_layer = host_budget // denom

	config = bg_lib.HostPagedKVConfig()
	config.shm_name = HOST_KV_AUX_SHM_NAME
	config.num_layers = profile.num_layers
	config.num_pages = num_pages_per_layer
	config.page_size_tokens = profile.page_size
	config.num_k_heads = profile.num_k_heads
	config.k_head_dim = profile.k_head_dim
	config.num_v_heads = profile.num_v_heads
	config.v_head_dim = profile.v_head_dim
	config.k_element_size_bytes = _dtype_size_bytes(profile.kv_dtype)
	config.v_element_size_bytes = 0
	config.sequence_table_capacity = (
		profile.sequence_table_capacity or config.num_pages
	)
	config.alignment_bytes = profile.alignment_bytes
	return config


# Legacy function below

def build_gpu_kv_config_fixed_size(
	model_name: str,
	gpu_kv_cache_size_gb: float,
	page_size_tokens: int = 64,
) -> 'GPUPagedKVConfig':
	"""
	Build GPU KV config with a fixed memory budget.
	
	Args:
		model_name: Model identifier for KV dimensions
		gpu_kv_cache_size_gb: GPU memory budget for KV cache in GB
		page_size_tokens: Tokens per page
	
	Returns:
		GPUPagedKVConfig with num_pages calculated from memory budget
	"""
	from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig
	
	profile = _resolve_profile(model_name)

	# Calculate total pages from memory budget using profile
	bytes_per_page_all_layers = profile.bytes_per_page() * profile.num_layers
	total_bytes = int(gpu_kv_cache_size_gb * (1024 ** 3))
	num_pages = total_bytes // bytes_per_page_all_layers
	return GPUPagedKVConfig(
		num_layers=profile.num_layers,
		num_pages=num_pages,
		page_size_tokens=profile.page_size,
		num_k_heads=profile.num_k_heads,
		k_head_dim=profile.k_head_dim,
		num_v_heads=profile.num_v_heads,
		v_head_dim=profile.v_head_dim,
		kv_dtype=_torch_dtype_from_string(profile.kv_dtype),
	)
