from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig
from batchgen.models.engine_loader import core_engine as bg_lib

HOST_KV_SHM_NAME = "batchgen_host_kv_cache"

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

_PROFILE_REGISTRY: Dict[str, _HostKVModelProfile] = {
    "deepseek_mla": _DEEPSEEK_MLA_PROFILE,
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
    ),
}.items():
    for alias in aliases:
        _PROFILE_ALIASES[alias.lower()] = canonical


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
