"""DeepSeek-V4 KV coordinators.

DeepSeek-V4 has multiple logical KV components whose layer sets and token
rates differ. Layer and allocation policy live in each KV view/manager; the
DSV4 coordinators below only register the model's component names.
"""

from __future__ import annotations

from typing import Any

from batchgen.kv_cache.gpu_kv_coordinator import GPUKVCoordinator
from batchgen.kv_cache.host_kv_coordinator import HostKVCoordinator


SWA = "swa"
COMPRESSOR_C4 = "compressor_c4"
COMPRESSOR_C128 = "compressor_c128"
INDEXER_C4 = "indexer_c4"


class DeepSeekV4HostKVCoordinator(HostKVCoordinator):
    """Runtime facade for DeepSeek-V4 host KV worker views.

    The base coordinator provides lifecycle, page, query, and data movement for
    registered components. Each worker view owns its layer mapping.
    """

    def __init__(
        self,
        *,
        swa: Any,
        compressor_c4: Any = None,
        compressor_c128: Any = None,
        indexer_c4: Any = None,
    ) -> None:
        super().__init__()
        for component_name, view in (
            (SWA, swa),
            (COMPRESSOR_C4, compressor_c4),
            (COMPRESSOR_C128, compressor_c128),
            (INDEXER_C4, indexer_c4),
        ):
            if view is None:
                continue
            self.register_component(component_name, view)


class DeepSeekV4GPUKVCoordinator(GPUKVCoordinator):
    """Runtime facade for DeepSeek-V4 GPU paged KV managers."""

    def __init__(
        self,
        *,
        swa: Any,
        compressor_c4: Any = None,
        compressor_c128: Any = None,
        indexer_c4: Any = None,
    ) -> None:
        super().__init__()
        for component_name, manager in (
            (SWA, swa),
            (COMPRESSOR_C4, compressor_c4),
            (COMPRESSOR_C128, compressor_c128),
            (INDEXER_C4, indexer_c4),
        ):
            if manager is None:
                continue
            self.register_component(component_name, manager)
