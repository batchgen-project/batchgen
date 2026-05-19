"""Generic registry for GPU-side KV components."""

from __future__ import annotations

from batchgen.kv_cache.component_coordinator import ComponentCoordinator


class GPUKVCoordinator(ComponentCoordinator):
    """GPU-side named component registry.

    Components may be paged KV managers, compressed-state managers, or any
    future GPU-side KV object. Paged-KV operations are intentionally left on the
    component itself and should be called explicitly through the named member.
    """

    component_label = "GPU KV component"
