"""Derived runtime configuration for Host-side prefix-cache reuse."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


_GPT_OSS_CANONICAL_MODEL = "openai/gpt-oss-120b"
_GPT_OSS_ALIASES = frozenset({_GPT_OSS_CANONICAL_MODEL, "gpt-oss-120b"})


class PrefixKVGroupSemantic(str, Enum):
    FULL_KV = "full_kv"


@dataclass(frozen=True)
class PrefixKVGroupSpec:
    group_id: int
    semantic: PrefixKVGroupSemantic
    required_for_reuse: bool
    raw_page_tokens: int
    compression_ratio: int = 1


@dataclass(frozen=True)
class PrefixCacheRuntimeConfig:
    shm_name: str
    namespace_digest: tuple[int, int, int, int]
    group_specs: tuple[PrefixKVGroupSpec, ...]
    hash_block_tokens: int
    publish_boundary_tokens: int
    max_nodes: int
    max_group_entries: int
    max_page_handles: int
    max_attachments: int
    debug_stats: bool = False

    def to_core_config(self, core_engine_module: Any):
        config = core_engine_module.HostPrefixCacheConfig()
        config.shm_name = self.shm_name
        config.hash_block_tokens = int(self.hash_block_tokens)
        config.max_nodes = int(self.max_nodes)
        config.max_group_entries = int(self.max_group_entries)
        config.max_page_handles = int(self.max_page_handles)
        config.max_attachments = int(self.max_attachments)
        config.group_specs = [
            _to_core_group_spec(core_engine_module, spec)
            for spec in self.group_specs
        ]
        return config


def require_prefix_cache_model_support(model_name: str) -> str:
    """Return the canonical model name or fail before allocating resources."""

    if not isinstance(model_name, str):
        raise ValueError("prefix cache model name must be a string")
    normalized = model_name.strip().lower()
    if normalized not in _GPT_OSS_ALIASES:
        raise ValueError(
            "prefix cache reuse currently supports only "
            f"{_GPT_OSS_CANONICAL_MODEL}; model {model_name!r} is unsupported"
        )
    return _GPT_OSS_CANONICAL_MODEL


def build_prefix_cache_runtime_config(
    *,
    model_name: str,
    kv_dtype: str,
    host_kv_config: Any,
    debug_stats: bool = False,
) -> PrefixCacheRuntimeConfig:
    """Derive the single-group GPT-OSS prefix-cache shared-memory layout."""

    canonical_model = require_prefix_cache_model_support(model_name)
    page_tokens = int(host_kv_config.page_size_tokens)
    num_pages = int(host_kv_config.num_pages)
    sequence_capacity = int(host_kv_config.sequence_table_capacity)
    if page_tokens <= 0 or num_pages <= 0 or sequence_capacity <= 0:
        raise ValueError("host KV configuration must have positive capacities")

    group_specs = (
        PrefixKVGroupSpec(
            group_id=0,
            semantic=PrefixKVGroupSemantic.FULL_KV,
            required_for_reuse=True,
            raw_page_tokens=page_tokens,
        ),
    )
    return PrefixCacheRuntimeConfig(
        shm_name=_derive_prefix_cache_shm_name(canonical_model),
        namespace_digest=_build_namespace_digest(
            model_name=canonical_model,
            kv_dtype=kv_dtype,
            page_tokens=page_tokens,
        ),
        group_specs=group_specs,
        hash_block_tokens=page_tokens,
        publish_boundary_tokens=page_tokens,
        max_nodes=num_pages + 1,
        max_group_entries=num_pages + 1,
        max_page_handles=num_pages,
        max_attachments=max(1024, sequence_capacity),
        debug_stats=bool(debug_stats),
    )


def create_host_prefix_cache_coordinator(
    *,
    core_engine_module: Any,
    runtime_config: PrefixCacheRuntimeConfig,
    create_region: bool,
):
    coordinator = core_engine_module.HostPrefixCacheCoordinator(
        runtime_config.to_core_config(core_engine_module)
    )
    coordinator.initialize(bool(create_region))
    return coordinator


def unlink_prefix_cache_shared_memory(runtime_config: PrefixCacheRuntimeConfig):
    """Remove the exact POSIX SHM name after every worker has exited."""

    path = f"/dev/shm/{runtime_config.shm_name.lstrip('/')}"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _derive_prefix_cache_shm_name(model_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    digest = hashlib.blake2b(model_name.encode("utf-8"), digest_size=4)
    return f"batchgen_prefix_cache_{normalized}_{digest.hexdigest()}"


def _build_namespace_digest(
    *, model_name: str, kv_dtype: str, page_tokens: int
) -> tuple[int, int, int, int]:
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(kv_dtype.strip().lower().encode("utf-8"))
    hasher.update(b"\0full_kv\0")
    hasher.update(int(page_tokens).to_bytes(4, "little"))
    digest = hasher.digest()
    return tuple(
        int.from_bytes(digest[offset : offset + 8], "little")
        for offset in range(0, 32, 8)
    )


def _to_core_group_spec(core_engine_module: Any, spec: PrefixKVGroupSpec):
    core_spec = core_engine_module.HostKVGroupSpec()
    core_spec.group_id = int(spec.group_id)
    core_spec.semantic = core_engine_module.HostKVGroupSemantic.FULL_KV
    core_spec.required_for_reuse = bool(spec.required_for_reuse)
    core_spec.raw_page_tokens = int(spec.raw_page_tokens)
    core_spec.compression_ratio = int(spec.compression_ratio)
    return core_spec
