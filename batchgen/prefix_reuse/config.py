"""Derived runtime configuration for Host-side prefix-cache reuse."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


# Support follows the prefill compute path (model type), never a model-name list.
# One Host-KV region holding full K/V:
_FULL_KV_MODEL_TYPES = frozenset({"gpt_oss"})
# GLM-5.2 sparse DSA prefill: MLA latent in the primary Host-KV region and the
# indexer keys in the auxiliary one. GLM-5/5.1 (glm_moe_dsa) prefill is dense.
_DSA_MODEL_TYPES = frozenset({"glm_moe_dsa_5_2", "glm_moe_dsa_5_3"})
# Kimi-K3 and Kimi-Linear-48B dense MLA prefill: one Host-KV region holding
# the latent -- the kimi_k3_mla profile (24 physical layers of 1 head x 576
# bf16, page 64) over 93 engine layers, or kimi_linear_mla over 27. No indexer,
# so one group. Both are NoPE MLA of the same per-head widths, so they share
# every pool seam.
_MLA_LATENT_MODEL_TYPES = frozenset({"kimi_k3", "kimi_linear"})
# Reuse for these types exists only on the in-wave prefill prefix pool: they
# have no cross-wave materialization path, so prefix caching without a planner
# pool workspace must be refused rather than silently served the other way.
_POOL_ONLY_MODEL_TYPES = _DSA_MODEL_TYPES | _MLA_LATENT_MODEL_TYPES


class PrefixKVGroupSemantic(str, Enum):
    FULL_KV = "full_kv"
    MLA_LATENT = "mla_latent"
    DSA_INDEXER_KEYS = "dsa_indexer_keys"


# The core coordinator only stores and cross-checks the semantic.
_CORE_SEMANTIC = {
    PrefixKVGroupSemantic.FULL_KV: "FULL_KV",
    PrefixKVGroupSemantic.MLA_LATENT: "MLA_COMPRESSED_KV",
    PrefixKVGroupSemantic.DSA_INDEXER_KEYS: "MLA_COMPRESSED_KV",
}


@dataclass(frozen=True)
class PrefixKVGroupSpec:
    group_id: int
    semantic: PrefixKVGroupSemantic
    required_for_reuse: bool
    raw_page_tokens: int
    compression_ratio: int = 1


@dataclass(frozen=True)
class PrefixCacheRuntimeConfig:
    # Diagnostic label; the region is an anonymous memfd reached through
    # /proc/<memfd_creator_pid>/fd/<memfd_fd> when attaching.
    shm_name: str
    namespace_digest: tuple[int, int, int, int]
    group_specs: tuple[PrefixKVGroupSpec, ...]
    hash_block_tokens: int
    publish_boundary_tokens: int
    max_nodes: int
    max_group_entries: int
    max_page_handles: int
    max_attachments: int
    host_page_bytes_all_layers: int
    debug_stats: bool = False
    memfd_creator_pid: int = -1
    memfd_fd: int = -1

    def to_core_config(self, core_engine_module: Any):
        config = core_engine_module.HostPrefixCacheConfig()
        config.shm_name = self.shm_name
        config.memfd_creator_pid = int(self.memfd_creator_pid)
        config.memfd_fd = int(self.memfd_fd)
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
    """Return the model type or fail before allocating resources."""

    if not isinstance(model_name, str):
        raise ValueError("prefix cache model name must be a string")
    from batchgen.config.model_registry import _detect_model_type_from_identifier

    model_type = _detect_model_type_from_identifier(model_name.strip())
    if (
        model_type
        not in _FULL_KV_MODEL_TYPES | _DSA_MODEL_TYPES | _MLA_LATENT_MODEL_TYPES
    ):
        raise ValueError(
            "prefix cache reuse currently supports only GPT-OSS, Kimi-K3 and "
            f"models on the GLM-5.2 sparse DSA prefill path; model "
            f"{model_name!r} (type {model_type!r}) is unsupported"
        )
    return model_type


def prefix_reuse_requires_prefill_pool(model_type: str) -> bool:
    """True when only the in-wave prefix pool can serve this model's reuse."""

    return model_type in _POOL_ONLY_MODEL_TYPES


def _host_page_bytes_all_layers(host_kv_config: Any) -> int:
    return int(host_kv_config.num_layers) * int(host_kv_config.page_size_tokens) * (
        int(host_kv_config.num_k_heads)
        * int(host_kv_config.k_head_dim)
        * int(host_kv_config.k_element_size_bytes)
        + int(host_kv_config.num_v_heads)
        * int(host_kv_config.v_head_dim)
        * int(host_kv_config.v_element_size_bytes)
    )


def build_prefix_cache_runtime_config(
    *,
    model_name: str,
    kv_dtype: str,
    host_kv_config: Any,
    shm_name: str | None = None,
    aux_host_kv_config: Any = None,
    debug_stats: bool = False,
    memfd_creator_pid: int = -1,
    memfd_fd: int = -1,
) -> PrefixCacheRuntimeConfig:
    """Derive the prefix-cache shared-memory layout, one group per Host-KV region.

    GPT-OSS has one full-K/V group and Kimi-K3 one MLA-latent group. A DSA
    model has two, sharing page size and page count: group 0 the MLA latent
    (primary region), group 1 the indexer keys (``aux_host_kv_config``). A
    prefix is reusable only when every group holds it.
    """

    model_type = require_prefix_cache_model_support(model_name)
    if not shm_name:
        # Diagnostic label only: the region is an anonymous memfd reached
        # through /proc/<pid>/fd/<fd>, so the name never hits /dev/shm.
        shm_name = _derive_prefix_cache_shm_name(model_name)
    if model_type in _DSA_MODEL_TYPES:
        if aux_host_kv_config is None:
            raise ValueError(
                "DSA prefix reuse needs the indexer-key (aux) Host-KV config"
            )
        layouts = (
            (PrefixKVGroupSemantic.MLA_LATENT, host_kv_config),
            (PrefixKVGroupSemantic.DSA_INDEXER_KEYS, aux_host_kv_config),
        )
    else:
        if aux_host_kv_config is not None:
            raise ValueError(f"model type {model_type!r} has no aux Host-KV region")
        layouts = ((
            PrefixKVGroupSemantic.MLA_LATENT
            if model_type in _MLA_LATENT_MODEL_TYPES
            else PrefixKVGroupSemantic.FULL_KV,
            host_kv_config,
        ),)

    page_tokens = int(host_kv_config.page_size_tokens)
    num_pages = int(host_kv_config.num_pages)
    sequence_capacity = int(host_kv_config.sequence_table_capacity)
    if page_tokens <= 0 or num_pages <= 0 or sequence_capacity <= 0:
        raise ValueError("host KV configuration must have positive capacities")
    for _, layout in layouts:
        if (
            int(layout.page_size_tokens) != page_tokens
            or int(layout.num_pages) != num_pages
        ):
            raise ValueError(
                "every prefix-cache group must share page size and page count"
            )
    host_page_bytes_all_layers = sum(
        _host_page_bytes_all_layers(layout) for _, layout in layouts
    )
    if host_page_bytes_all_layers <= 0:
        raise ValueError("host KV page byte size must be positive")

    group_specs = tuple(
        PrefixKVGroupSpec(
            group_id=group_id,
            semantic=semantic,
            required_for_reuse=True,
            raw_page_tokens=page_tokens,
        )
        for group_id, (semantic, _) in enumerate(layouts)
    )
    groups = len(group_specs)
    return PrefixCacheRuntimeConfig(
        shm_name=shm_name,
        namespace_digest=_build_namespace_digest(
            model_type=model_type,
            kv_dtype=kv_dtype,
            page_tokens=page_tokens,
            semantics=[spec.semantic for spec in group_specs],
        ),
        group_specs=group_specs,
        hash_block_tokens=page_tokens,
        publish_boundary_tokens=page_tokens,
        max_nodes=num_pages + 1,
        max_group_entries=groups * (num_pages + 1),
        max_page_handles=groups * num_pages,
        max_attachments=max(1024, sequence_capacity),
        host_page_bytes_all_layers=host_page_bytes_all_layers,
        debug_stats=bool(debug_stats),
        memfd_creator_pid=int(memfd_creator_pid),
        memfd_fd=int(memfd_fd),
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


def _derive_prefix_cache_shm_name(model_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    digest = hashlib.blake2b(model_name.encode("utf-8"), digest_size=4)
    return f"batchgen_prefix_cache_{normalized}_{digest.hexdigest()}"


def _build_namespace_digest(
    *, model_type: str, kv_dtype: str, page_tokens: int, semantics
) -> tuple[int, int, int, int]:
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(model_type.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(kv_dtype.strip().lower().encode("utf-8"))
    for semantic in semantics:
        hasher.update(b"\0" + semantic.value.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(int(page_tokens).to_bytes(4, "little"))
    digest = hasher.digest()
    return tuple(
        int.from_bytes(digest[offset : offset + 8], "little")
        for offset in range(0, 32, 8)
    )


def _to_core_group_spec(core_engine_module: Any, spec: PrefixKVGroupSpec):
    core_spec = core_engine_module.HostKVGroupSpec()
    core_spec.group_id = int(spec.group_id)
    core_spec.semantic = getattr(
        core_engine_module.HostKVGroupSemantic, _CORE_SEMANTIC[spec.semantic]
    )
    core_spec.required_for_reuse = bool(spec.required_for_reuse)
    core_spec.raw_page_tokens = int(spec.raw_page_tokens)
    core_spec.compression_ratio = int(spec.compression_ratio)
    return core_spec
