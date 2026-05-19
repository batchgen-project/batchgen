"""Generic registry for host-side KV components."""

from __future__ import annotations

from batchgen.kv_cache.component_coordinator import ComponentCoordinator


class HostKVCoordinator(ComponentCoordinator):
    """Host-side named component registry.

    Components may be paged KV views, compressed-state managers, or any future
    host-side KV object. Paged-KV operations are intentionally left on the
    component itself and should be called explicitly through the named member.
    """

    component_label = "host KV component"
