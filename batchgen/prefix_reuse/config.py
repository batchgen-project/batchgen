"""Runtime configuration helpers for Host-side prefix reuse.

This module deliberately keeps the Python-side configuration lightweight:
user-facing CLI only enables/disables prefix reuse, while shared-memory names,
group semantics, hash granularity, and table capacities are derived from the
model and Host KV profile.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class PrefixKVGroupSemantic(str, Enum):
    FULL_KV = "full_kv"
    MLA_COMPRESSED_KV = "mla_compressed_kv"
    SWA_KV = "swa_kv"
    COMPRESSED_RATIO_KV = "compressed_ratio_kv"


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

    def to_core_config(self, core_engine_module):
        """Build a core_engine.HostPrefixCacheConfig instance."""

        core_config = core_engine_module.HostPrefixCacheConfig()
        core_config.shm_name = self.shm_name
        core_config.hash_block_tokens = int(self.hash_block_tokens)
        core_config.max_nodes = int(self.max_nodes)
        core_config.max_group_entries = int(self.max_group_entries)
        core_config.max_page_handles = int(self.max_page_handles)
        core_config.max_attachments = int(self.max_attachments)
        core_config.group_specs = [
            _to_core_group_spec(core_engine_module, spec)
            for spec in self.group_specs
        ]
        return core_config


def build_prefix_cache_runtime_config(
    *,
    model_name: str,
    kv_dtype: str,
    host_kv_cache_size_bytes: int,
    node_rank: int = 0,
    debug_stats: bool = False,
) -> PrefixCacheRuntimeConfig:
    """Derive a Host prefix-cache config from existing Host KV profiles."""

    group_specs, required_pages = _derive_group_specs_and_page_count(
        model_name=model_name,
        host_kv_cache_size_bytes=host_kv_cache_size_bytes,
    )
    return build_prefix_cache_runtime_config_from_specs(
        model_name=model_name,
        kv_dtype=kv_dtype,
        host_kv_pages_per_required_group=required_pages,
        node_rank=node_rank,
        group_specs=group_specs,
        debug_stats=debug_stats,
    )


def build_prefix_cache_runtime_config_from_specs(
    *,
    model_name: str,
    kv_dtype: str,
    host_kv_pages_per_required_group: int,
    node_rank: int = 0,
    group_specs: Sequence[PrefixKVGroupSpec],
    debug_stats: bool = False,
) -> PrefixCacheRuntimeConfig:
    """Build a runtime config from already-derived logical KV groups."""

    specs = tuple(group_specs)
    if not specs:
        raise ValueError("prefix cache requires at least one KV group")
    required_specs = tuple(spec for spec in specs if spec.required_for_reuse)
    if not required_specs:
        raise ValueError("prefix cache requires at least one required KV group")

    hash_block_tokens = _gcd(spec.raw_page_tokens for spec in required_specs)
    publish_boundary_tokens = _lcm(
        spec.raw_page_tokens for spec in required_specs
    )
    if hash_block_tokens <= 0 or publish_boundary_tokens <= 0:
        raise ValueError("prefix cache token boundaries must be positive")

    pages_per_group = int(host_kv_pages_per_required_group)
    if pages_per_group <= 0:
        raise ValueError("host_kv_pages_per_required_group must be positive")

    max_nodes = max(1024, pages_per_group + 1)
    max_group_entries = max_nodes * len(specs)
    max_page_handles = _derive_page_handle_capacity(
        max_nodes=max_nodes,
        pages_per_group=pages_per_group,
        group_count=len(specs),
    )
    max_attachments = max(1024, max_nodes // 4)

    return PrefixCacheRuntimeConfig(
        shm_name=derive_prefix_cache_shm_name(model_name, node_rank=node_rank),
        namespace_digest=build_prefix_cache_namespace_digest(
            model_name=model_name,
            kv_dtype=kv_dtype,
            group_specs=specs,
        ),
        group_specs=specs,
        hash_block_tokens=hash_block_tokens,
        publish_boundary_tokens=publish_boundary_tokens,
        max_nodes=max_nodes,
        max_group_entries=max_group_entries,
        max_page_handles=max_page_handles,
        max_attachments=max_attachments,
        debug_stats=debug_stats,
    )


def create_host_prefix_cache_coordinator(
    *,
    core_engine_module,
    runtime_config: PrefixCacheRuntimeConfig,
    create_region: bool,
):
    coordinator = core_engine_module.HostPrefixCacheCoordinator(
        runtime_config.to_core_config(core_engine_module)
    )
    coordinator.initialize(bool(create_region))
    return coordinator


def derive_prefix_cache_shm_name(model_name: str, *, node_rank: int) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    normalized = normalized[:64] or "model"
    digest = hashlib.blake2b(model_name.encode("utf-8"), digest_size=4)
    suffix = int.from_bytes(digest.digest(), "little")
    return f"batchgen_prefix_cache_{normalized}_{suffix:08x}_node{node_rank}"


def build_prefix_cache_namespace_digest(
    *,
    model_name: str,
    kv_dtype: str,
    group_specs: Sequence[PrefixKVGroupSpec],
) -> tuple[int, int, int, int]:
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(model_name.strip().lower().encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(kv_dtype.strip().lower().encode("utf-8"))
    for spec in sorted(group_specs, key=lambda item: int(item.group_id)):
        hasher.update(b"\0")
        hasher.update(int(spec.group_id).to_bytes(4, "little"))
        hasher.update(spec.semantic.value.encode("ascii"))
        hasher.update(b"\0")
        hasher.update(int(spec.required_for_reuse).to_bytes(1, "little"))
        hasher.update(int(spec.raw_page_tokens).to_bytes(4, "little"))
        hasher.update(int(spec.compression_ratio).to_bytes(4, "little"))
    digest = hasher.digest()
    return tuple(
        int.from_bytes(digest[offset : offset + 8], "little")
        for offset in range(0, 32, 8)
    )


def _derive_group_specs_and_page_count(
    *, model_name: str, host_kv_cache_size_bytes: int
) -> tuple[tuple[PrefixKVGroupSpec, ...], int]:
    from batchgen.kv_cache.host_kv_mananger_config import (
        resolve_host_kv_group_profiles,
    )

    group_profiles = resolve_host_kv_group_profiles(model_name)
    specs = tuple(
        PrefixKVGroupSpec(
            group_id=profile.group_id,
            semantic=_semantic_from_group_profile(profile),
            required_for_reuse=profile.required_for_reuse,
            raw_page_tokens=profile.raw_page_tokens,
            compression_ratio=profile.compression_ratio,
        )
        for profile in group_profiles
    )
    required_profiles = tuple(
        profile for profile in group_profiles if profile.required_for_reuse
    )
    if not required_profiles:
        raise ValueError("prefix cache requires at least one required KV group")

    publish_boundary_tokens = _lcm(
        profile.raw_page_tokens for profile in required_profiles
    )
    bytes_per_publish_boundary = sum(
        profile.bytes_per_page()
        * profile.num_layers
        * (publish_boundary_tokens // profile.raw_page_tokens)
        for profile in required_profiles
    )
    publish_units = int(host_kv_cache_size_bytes) // bytes_per_publish_boundary
    if publish_units <= 0:
        raise ValueError("host KV cache is too small for prefix cache")

    return specs, publish_units


def _semantic_from_group_profile(profile) -> PrefixKVGroupSemantic:
    try:
        return PrefixKVGroupSemantic(profile.semantic)
    except ValueError as exc:
        raise ValueError(
            f"unsupported prefix KV group semantic {profile.semantic!r}"
        ) from exc


def _to_core_group_spec(core_engine_module, spec: PrefixKVGroupSpec):
    core_spec = core_engine_module.HostKVGroupSpec()
    core_spec.group_id = int(spec.group_id)
    core_spec.semantic = _to_core_semantic(core_engine_module, spec.semantic)
    core_spec.required_for_reuse = bool(spec.required_for_reuse)
    core_spec.raw_page_tokens = int(spec.raw_page_tokens)
    core_spec.compression_ratio = int(spec.compression_ratio)
    return core_spec


def _to_core_semantic(core_engine_module, semantic: PrefixKVGroupSemantic):
    enum_cls = core_engine_module.HostKVGroupSemantic
    return getattr(enum_cls, semantic.name)


def _gcd(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = int(value) if result == 0 else math.gcd(result, int(value))
    return result


def _lcm(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result = math.lcm(result, int(value))
    return result


def _derive_page_handle_capacity(
    *, max_nodes: int, pages_per_group: int, group_count: int
) -> int:
    # The current C++ entry stores enough page handles to materialize a node,
    # so long prompts need more than one handle per node. Use a derived,
    # bounded estimate instead of a user-tunable knob.
    average_pages_per_node = max(
        16, min(512, int(math.sqrt(max(1, pages_per_group))))
    )
    return max_nodes * max(1, group_count) * average_pages_per_node
