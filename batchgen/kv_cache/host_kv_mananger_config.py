from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig
from batchgen.models.engine_loader import core_engine as bg_lib

HOST_KV_SHM_NAME = "batchgen_host_kv_cache"

# Arrangement of one host/GPU KV page. "contiguous_aos" keeps token-major
# records. "page_split_fp8_scale" is the SoA page deep_gemm.fp8_paged_mqa_logits
# reads: the page's e4m3 K bytes, then one fp32 scale per token. In the
# page-split format `k_head_dim` carries TOTAL BYTES per token (data + scale),
# which keeps every page/layer/budget size a plain product.
PAGE_FORMAT_CONTIGUOUS_AOS = "contiguous_aos"
PAGE_FORMAT_FP8_PAGE_SPLIT = "page_split_fp8_scale"
PAGE_SPLIT_SCALE_BYTES_PER_TOKEN = 4

# FlashMLA's packed FP8 MLA token record, used by the GLM-5.2 PRIMARY cache.
# Contract: references/FlashMLA-1408756/csrc/pybind.cpp:249-256 and
# README.md:117-121 -- one token is 656 contiguous bytes,
#
#   [   0, 512)  512 x float8_e4m3  quantized NoPE (compressed KV latent)
#   [ 512, 528)    4 x float32      one scale per 128-element NoPE block
#   [ 528, 656)   64 x bfloat16     RoPE, NOT quantized
#
# and the page block size is hard-required to be 64 (pybind.cpp:236) with the
# whole block contiguous (`kcache.stride(1) == 656`). The record is token-major
# inside the page, so this is the ordinary contiguous-AoS page format with a
# 656-BYTE token; nothing about the host page geometry is special. `is_fp8` is
# an explicit kernel flag rather than inferred from the tensor dtype
# (pybind.cpp:180 accepts kUInt8), which is why a plain uint8 carrier is legal.
PACKED_MLA_NOPE_DIM = 512
PACKED_MLA_ROPE_DIM = 64
PACKED_MLA_QUANT_TILE = 128
PACKED_MLA_NUM_SCALES = PACKED_MLA_NOPE_DIM // PACKED_MLA_QUANT_TILE
PACKED_MLA_SCALE_BYTES = PACKED_MLA_NUM_SCALES * 4
PACKED_MLA_TOKEN_BYTES = (
	PACKED_MLA_NOPE_DIM + PACKED_MLA_SCALE_BYTES + PACKED_MLA_ROPE_DIM * 2
)
PACKED_MLA_PAGE_SIZE = 64
# The unpacked compressed-KV dim this record encodes (512 NoPE + 64 RoPE).
PACKED_MLA_SOURCE_DIM = PACKED_MLA_NOPE_DIM + PACKED_MLA_ROPE_DIM

__all__ = [
	"build_host_kv_config",
	"build_host_kv_worker_view",
	"build_gpu_kv_config",
	"HOST_KV_SHM_NAME",
	"PAGE_FORMAT_CONTIGUOUS_AOS",
	"PAGE_FORMAT_FP8_PAGE_SPLIT",
	"PAGE_SPLIT_SCALE_BYTES_PER_TOKEN",
	"PACKED_MLA_NOPE_DIM",
	"PACKED_MLA_ROPE_DIM",
	"PACKED_MLA_QUANT_TILE",
	"PACKED_MLA_NUM_SCALES",
	"PACKED_MLA_SCALE_BYTES",
	"PACKED_MLA_TOKEN_BYTES",
	"PACKED_MLA_PAGE_SIZE",
	"PACKED_MLA_SOURCE_DIM",
	"glm52_packed_kv_enabled",
	"set_glm52_packed_kv",
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
	# Opaque byte carrier; k_head_dim counts BYTES per token. Mirrors
	# ResolveElementSizeBytes in core/KV_Storage/host_paged_kv_config_utils.h.
	if normalized == "uint8":
		return 1
	raise ValueError(f"Unsupported kv dtype '{dtype}'")


def _torch_dtype_from_string(dtype: str) -> torch.dtype:
	mapping = {
		"float32": torch.float32,
		"float16": torch.float16,
		"bfloat16": torch.bfloat16,
		"float8_e4m3fn": torch.float8_e4m3fn,
		"float8_e5m2": torch.float8_e5m2,
		"uint8": torch.uint8,
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
	logical_to_physical_layer: tuple[int, ...] | None = None
	page_format: str = PAGE_FORMAT_CONTIGUOUS_AOS
	# True only for the FlashMLA packed FP8 MLA record. The page geometry IS
	# contiguous AoS (token-major 656-byte records), so no new page format is
	# needed; this flag exists so the exact byte layout the kernel hard-checks
	# is asserted where the profile is declared instead of first failing inside
	# a CUDA graph capture.
	packed_mla_fp8: bool = False

	def __post_init__(self) -> None:
		if self.packed_mla_fp8:
			self._validate_packed_mla_fp8()
		if self.page_format == PAGE_FORMAT_CONTIGUOUS_AOS:
			return
		if self.page_format != PAGE_FORMAT_FP8_PAGE_SPLIT:
			raise ValueError(f"Unsupported page_format '{self.page_format}'")
		# Mirror of the C++ SanitizeConfig gate, applied where the profile is
		# declared: the SoA page carries no head dimension and no V cache, its
		# data elements are e4m3 bytes, and k_head_dim is a byte count.
		if self.kv_dtype != "float8_e4m3fn":
			raise ValueError(
				f"{PAGE_FORMAT_FP8_PAGE_SPLIT} requires kv_dtype="
				f"'float8_e4m3fn', got '{self.kv_dtype}'"
			)
		if self.num_k_heads != 1 or self.num_v_heads != 0:
			raise ValueError(
				f"{PAGE_FORMAT_FP8_PAGE_SPLIT} requires num_k_heads=1 and "
				"num_v_heads=0"
			)
		data_bytes = self.k_head_dim - PAGE_SPLIT_SCALE_BYTES_PER_TOKEN
		if data_bytes <= 0 or data_bytes % 16 != 0:
			raise ValueError(
				f"{PAGE_FORMAT_FP8_PAGE_SPLIT} requires k_head_dim to be the "
				"per-token byte count (a positive multiple of 16, plus 4 "
				f"scale bytes), got {self.k_head_dim}"
			)

	def _validate_packed_mla_fp8(self) -> None:
		"""Every geometry FlashMLA's FP8 path hard-checks, checked up front.

		``csrc/pybind.cpp:249-256`` requires a ``[blocks, 64, 1, 656]`` uint8
		cache with the whole block contiguous, and ``:236`` requires
		``page_block_size == 64``. A profile that satisfies all of it cannot
		produce a cache the kernel rejects; one that does not is named here,
		at declaration time, rather than at the first decode step.
		"""

		if self.page_format != PAGE_FORMAT_CONTIGUOUS_AOS:
			raise ValueError(
				"packed_mla_fp8 is token-major and therefore requires "
				f"page_format='{PAGE_FORMAT_CONTIGUOUS_AOS}', got "
				f"'{self.page_format}'"
			)
		if self.kv_dtype != "uint8":
			raise ValueError(
				"packed_mla_fp8 mixes e4m3 data, fp32 scales and bf16 RoPE in "
				"one record, so its carrier must be kv_dtype='uint8', got "
				f"'{self.kv_dtype}'"
			)
		if self.k_head_dim != PACKED_MLA_TOKEN_BYTES:
			raise ValueError(
				"packed_mla_fp8 requires k_head_dim == "
				f"{PACKED_MLA_TOKEN_BYTES} bytes per token "
				f"({PACKED_MLA_NOPE_DIM} e4m3 + {PACKED_MLA_SCALE_BYTES} "
				f"scale + {PACKED_MLA_ROPE_DIM * 2} bf16 RoPE), got "
				f"{self.k_head_dim}"
			)
		if self.page_size != PACKED_MLA_PAGE_SIZE:
			raise ValueError(
				"packed_mla_fp8 requires page_size == "
				f"{PACKED_MLA_PAGE_SIZE} (FlashMLA hard requirement), got "
				f"{self.page_size}"
			)
		if self.num_k_heads != 1 or self.num_v_heads != 0:
			raise ValueError(
				"packed_mla_fp8 requires num_k_heads == 1 and "
				f"num_v_heads == 0, got {self.num_k_heads}/{self.num_v_heads}"
			)

	@property
	def is_page_split(self) -> bool:
		return self.page_format == PAGE_FORMAT_FP8_PAGE_SPLIT

	def k_data_dim(self) -> int:
		"""Elements of K data per token, i.e. the real indexer head dim.

		Equal to ``k_head_dim`` in the AoS format; ``k_head_dim`` minus the
		fp32 scale bytes in the page-split format, where ``k_head_dim`` counts
		the token's total bytes.
		"""

		if self.is_page_split:
			return self.k_head_dim - PAGE_SPLIT_SCALE_BYTES_PER_TOKEN
		return self.k_head_dim

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

# GLM-5.2 primary MLA cache in FlashMLA's packed FP8 record (656 B/token vs
# 1,152 B for bf16 576). Same 78 layers and same page size 64; only a token's
# bytes change, and they change identically in the GPU pool, the host region,
# every staging buffer and every prefix-reuse span -- the format is produced
# once, at the writer, and byte-copied everywhere after. GLM-5, GLM-5.1 and
# GLM-5.3 keep `glm5_mla` (bf16 576); only the glm-5.2 aliases move here, and
# only while `glm52_packed_kv_enabled()` is true.
_GLM5_2_MLA_PACKED_PROFILE = _HostKVModelProfile(
	num_layers=78,
	num_k_heads=1,
	k_head_dim=PACKED_MLA_TOKEN_BYTES,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="uint8",
	page_size=PACKED_MLA_PAGE_SIZE,
	packed_mla_fp8=True,
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

# GLM-5.2 runs the DSA indexer on 21 of 78 layers ([0, 1, 2] + range(6, 78, 4),
# i.e. index_topk_freq=4 / index_skip_topk_offset=3 — see
# batchgen/models/glm/glm5/dsa_schedule.py); the other 57 skip_topk layers
# reuse the previous full layer's top-k and never read the aux K cache. Store
# only the 21 indexer layers densely in both host and GPU aux pools; the
# logical map keeps engine-layer indexing at every caller (-1 = no slot).
_GLM5_2_INDEXER_LOGICAL_LAYERS = (0, 1, 2) + tuple(range(6, 78, 4))
_GLM5_2_INDEXER_PHYSICAL_BY_LOGICAL = {
	logical_layer: physical_layer
	for physical_layer, logical_layer in enumerate(_GLM5_2_INDEXER_LOGICAL_LAYERS)
}
_GLM5_2_INDEXER_LAYER_MAP = tuple(
	_GLM5_2_INDEXER_PHYSICAL_BY_LOGICAL.get(layer_idx, -1)
	for layer_idx in range(78)
)
# GLM-5.2 stores its indexer keys in DeepGEMM's page-split FP8 layout, the one
# deep_gemm.fp8_paged_mqa_logits reads directly: per page
# [64 x 128 B e4m3 K][64 x 4 B fp32 scale] = 8,448 B. k_head_dim is therefore
# the per-token BYTE count (128 data + 4 scale = 132), not an element count:
# 132 B/token against 256 B/token for bf16 AoS. GLM-5/5.1/5.3 keep
# `glm5_indexer` (bf16 AoS) untouched.
_GLM5_2_INDEXER_PROFILE = _HostKVModelProfile(
	num_layers=len(_GLM5_2_INDEXER_LOGICAL_LAYERS),
	num_k_heads=1,
	k_head_dim=128 + PAGE_SPLIT_SCALE_BYTES_PER_TOKEN,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="float8_e4m3fn",
	logical_to_physical_layer=_GLM5_2_INDEXER_LAYER_MAP,
	page_format=PAGE_FORMAT_FP8_PAGE_SPLIT,
)

# Kimi-K3 has 93 logical engine layers but only these 24 MLA layers own paged
# KV.  The logical map preserves engine-layer indexing at every caller while
# both host and GPU pools store the MLA layers densely.  Host->GPU bulk loads
# therefore also see the same 24 physical rows on both sides.
_KIMI_K3_MLA_LOGICAL_LAYERS = tuple(range(3, 92, 4)) + (92,)
_KIMI_K3_MLA_PHYSICAL_BY_LOGICAL = {
	logical_layer: physical_layer
	for physical_layer, logical_layer in enumerate(_KIMI_K3_MLA_LOGICAL_LAYERS)
}
_KIMI_K3_MLA_LAYER_MAP = tuple(
	_KIMI_K3_MLA_PHYSICAL_BY_LOGICAL.get(layer_idx, -1)
	for layer_idx in range(93)
)
_KIMI_K3_MLA_PROFILE = _HostKVModelProfile(
	num_layers=len(_KIMI_K3_MLA_LOGICAL_LAYERS),
	num_k_heads=1,
	k_head_dim=576,
	num_v_heads=0,
	v_head_dim=0,
	kv_dtype="bfloat16",
	logical_to_physical_layer=_KIMI_K3_MLA_LAYER_MAP,
)

# Kimi-Linear: MLA latent KV (compressed_kv_dim=576, 27 engine layers; only
# the 7 MLA layers ever append — KDA layers hold no KV).
_KIMI_LINEAR_MLA_PROFILE = _HostKVModelProfile(
	num_layers=27,
	num_k_heads=1,
	k_head_dim=576,
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
	"glm5_2_mla_packed": _GLM5_2_MLA_PACKED_PROFILE,
	"glm5_indexer": _GLM5_INDEXER_PROFILE,
	"glm5_2_indexer": _GLM5_2_INDEXER_PROFILE,
	"kimi_linear_mla": _KIMI_LINEAR_MLA_PROFILE,
	"kimi_k3_mla": _KIMI_K3_MLA_PROFILE,
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
	"kimi_linear_mla": (
		"moonshotai/kimi-linear-48b-a3b-instruct",
		"moonshotai/kimi-linear",
		"kimi-linear",
		"kimi_linear",
	),
	# K3 is NOT an alias of kimi-linear here: same KV geometry, but a distinct
	# 93-logical-layer -> 24-physical-MLA map. See _KIMI_K3_MLA_PROFILE.
	"kimi_k3_mla": (
		"moonshotai/kimi-k3",
		"kimi-k3",
		"kimi_k3",
	),
	"glm5_mla": (
		"zai-org/glm-5.2-fp8",
		"zai-org/glm-5.2",
		"glm-5.2-fp8",
		"glm-5.2",
		"zai-org/glm-5.3-fp8",
		"zai-org/glm-5.3",
		"glm-5.3-fp8",
		"glm-5.3",
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
	# GLM-5.2 stores only its 21 indexer layers (dense mapped profile).
	# GLM-5.3's skip-layer schedule is not established here; it stays on the
	# full 78-layer profile with GLM-5/5.1.
	"glm5_2_indexer": (
		"zai-org/glm-5.2-fp8",
		"zai-org/glm-5.2",
		"glm-5.2-fp8",
		"glm-5.2",
	),
	"glm5_indexer": (
		"zai-org/glm-5.3-fp8",
		"zai-org/glm-5.3",
		"glm-5.3-fp8",
		"glm-5.3",
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


def apply_page_format(config: Any, page_format: str) -> None:
	"""Stamp a profile's page format onto a core HostPagedKVConfig.

	The format is part of the physical layout and is hashed into the region's
	config fingerprint, so a core_engine build without the field cannot serve a
	page-split region: that combination raises instead of silently creating an
	AoS region a page-split reader would misread.
	"""

	if page_format not in (PAGE_FORMAT_CONTIGUOUS_AOS, PAGE_FORMAT_FP8_PAGE_SPLIT):
		raise ValueError(f"Unsupported page_format '{page_format}'")
	enum_type = getattr(bg_lib, "HostKVPageFormat", None)
	if enum_type is None or not hasattr(config, "page_format"):
		if page_format != PAGE_FORMAT_CONTIGUOUS_AOS:
			raise RuntimeError(
				f"page_format='{page_format}' requires a core_engine build "
				"with HostPagedKVConfig.page_format"
			)
		return
	config.page_format = (
		enum_type.PAGE_SPLIT_FP8_SCALE
		if page_format == PAGE_FORMAT_FP8_PAGE_SPLIT
		else enum_type.CONTIGUOUS_AOS
	)


# GLM-5.2's primary MLA cache moves to FlashMLA's packed FP8 record. The move
# is expressed as an OVERRIDE applied inside `_resolve_profile`, not as an edit
# to `_PROFILE_ALIASES`: with the kill-switch off, resolution is byte-for-byte
# the pre-migration mapping (bf16 576 via `glm5_mla`), which is what makes the
# rollback claim structural rather than a promise.
_PACKED_PRIMARY_PROFILE_ALIASES: Dict[str, str] = {
	alias: "glm5_2_mla_packed"
	for alias in (
		"zai-org/glm-5.2-fp8",
		"zai-org/glm-5.2",
		"glm-5.2-fp8",
		"glm-5.2",
	)
}

# Process-wide resolution switch for the packed GLM-5.2 primary cache.
#
# The primary profile is read by ~12 independent call sites across the server
# process, every worker process, the prefix-cache coordinator and three model
# initializers, and the GPU pool, the host shm region, the staging buffers and
# the prefix-reuse spans must ALL agree on the format or pages silently
# mis-decode. Threading a parameter through every signature makes a single
# missed site a silent corruption, so the switch is process state that every
# resolution reads, and it LATCHES: once a profile has been resolved, changing
# it raises instead of handing out two different geometries in one process.
_GLM5_2_PACKED_KV_DEFAULT = True
_glm52_packed_kv: bool = _GLM5_2_PACKED_KV_DEFAULT
_glm52_packed_kv_locked: bool = False


def glm52_packed_kv_enabled() -> bool:
	"""Whether GLM-5.2's primary MLA cache uses the packed FP8 record."""

	return _glm52_packed_kv


def set_glm52_packed_kv(enabled: bool) -> None:
	"""Select the GLM-5.2 primary KV format for this process.

	Must be called before the first primary-profile resolution. Re-selecting
	the value already in force is a no-op (initializers may set it more than
	once); changing it after any resolution raises, because the caches already
	sized from the old profile cannot be re-laid out.
	"""

	global _glm52_packed_kv
	enabled = bool(enabled)
	if enabled == _glm52_packed_kv:
		return
	if _glm52_packed_kv_locked:
		raise RuntimeError(
			"glm52_packed_kv cannot change after a host/GPU KV profile has "
			"been resolved in this process: the primary cache geometry is "
			f"already committed to packed={_glm52_packed_kv}. Set it before "
			"any KV cache is built."
		)
	_glm52_packed_kv = enabled


def _resolve_profile(model_name: str) -> _HostKVModelProfile:
	"""Maps a user supplied model name to a cached profile."""

	global _glm52_packed_kv_locked

	if not isinstance(model_name, str):
		raise ValueError("model_name must be a string")
	alias = model_name.strip().lower()
	if alias not in _PROFILE_ALIASES:
		raise ValueError(f"Unsupported model '{model_name}' for host KV cache")
	_glm52_packed_kv_locked = True
	if _glm52_packed_kv and alias in _PACKED_PRIMARY_PROFILE_ALIASES:
		return _PROFILE_REGISTRY[_PACKED_PRIMARY_PROFILE_ALIASES[alias]]
	return _PROFILE_REGISTRY[_PROFILE_ALIASES[alias]]


def build_host_kv_config(
	model_name: str,
	host_kv_cache_size: int,
	shm_name: str = HOST_KV_SHM_NAME,
) -> Any:
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
	if not isinstance(shm_name, str) or not shm_name:
		raise ValueError("shm_name must be a non-empty string")

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
	config.shm_name = shm_name
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
	apply_page_format(config, profile.page_format)
	config.logical_to_physical_layer = list(
		profile.logical_to_physical_layer or ()
	)
	return config


def build_host_kv_worker_view(core_engine_module: Any, config: Any) -> Any:
	"""Construct the host worker view matching the physical layout and map."""

	mapped_layers = bool(config.logical_to_physical_layer)
	if config.num_v_heads == 0:
		view_cls = (
			core_engine_module.MappedMLAHostPagedKVWorkerView
			if mapped_layers
			else core_engine_module.MLAHostPagedKVWorkerView
		)
	else:
		view_cls = (
			core_engine_module.MappedDefaultHostPagedKVWorkerView
			if mapped_layers
			else core_engine_module.DefaultHostPagedKVWorkerView
		)
	return view_cls(config)


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
		logical_to_physical_layer=profile.logical_to_physical_layer,
	)


HOST_KV_AUX_SHM_NAME = "batchgen_host_kv_cache_aux"


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
		logical_to_physical_layer=profile.logical_to_physical_layer,
	)


def build_host_kv_config_aux(
	model_name: str,
	host_kv_cache_size: int,
	shm_name: str = HOST_KV_AUX_SHM_NAME,
) -> Any | None:
	"""Builds a HostPagedKVConfig for the DSA indexer host cache, or None."""

	profile = _resolve_indexer_profile(model_name)
	if profile is None:
		return None

	host_budget = int(host_kv_cache_size)
	if host_budget <= 0:
		raise ValueError("host_kv_cache_size must be a positive integer")
	if not isinstance(shm_name, str) or not shm_name:
		raise ValueError("shm_name must be a non-empty string")

	bytes_per_page = profile.bytes_per_page()
	denom = profile.num_layers * bytes_per_page
	num_pages_per_layer = host_budget // denom

	config = bg_lib.HostPagedKVConfig()
	config.shm_name = shm_name
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
	apply_page_format(config, profile.page_format)
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
		logical_to_physical_layer=profile.logical_to_physical_layer,
	)
