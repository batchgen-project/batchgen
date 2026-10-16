"""Host-side prefix-cache reuse primitives."""

from batchgen.prefix_reuse.config import (
    PrefixCacheRuntimeConfig,
    PrefixKVGroupSemantic,
    PrefixKVGroupSpec,
    build_prefix_cache_runtime_config,
    create_host_prefix_cache_coordinator,
    require_prefix_cache_model_support,
)
from batchgen.prefix_reuse.commit import (
    PrefixCommitRequest,
    aligned_prefix_tokens,
    build_prefix_commit_request,
    collect_group_pages_for_commit,
    release_evicted_prefix_pages,
    retain_inserted_prefix_pages,
)
from batchgen.prefix_reuse.prefill import (
    PrefixCachePrefillLookup,
    lookup_prefix_cache_for_prefill,
    release_prefix_lookup_attachments,
)
from batchgen.prefix_reuse.trajectory_frontier import (
    FrontierCharge,
    KDASnapshotDescriptor,
    KDASnapshotProfile,
    TrajectoryFrontierRecord,
    TrajectoryFrontierRegistry,
    build_kda_snapshot_profile,
    frontier_index_key,
    is_frontier_hit,
    longest_common_prefix_length,
    session_key_for_baseline,
    snapshot_bytes,
)

__all__ = [
    "PrefixCacheRuntimeConfig",
    "PrefixCommitRequest",
    "PrefixKVGroupSemantic",
    "PrefixKVGroupSpec",
    "build_prefix_cache_runtime_config",
    "aligned_prefix_tokens",
    "build_prefix_commit_request",
    "collect_group_pages_for_commit",
    "create_host_prefix_cache_coordinator",
    "require_prefix_cache_model_support",
    "PrefixCachePrefillLookup",
    "lookup_prefix_cache_for_prefill",
    "release_prefix_lookup_attachments",
    "release_evicted_prefix_pages",
    "retain_inserted_prefix_pages",
    "FrontierCharge",
    "KDASnapshotDescriptor",
    "KDASnapshotProfile",
    "TrajectoryFrontierRecord",
    "TrajectoryFrontierRegistry",
    "build_kda_snapshot_profile",
    "frontier_index_key",
    "is_frontier_hit",
    "longest_common_prefix_length",
    "session_key_for_baseline",
    "snapshot_bytes",
]
