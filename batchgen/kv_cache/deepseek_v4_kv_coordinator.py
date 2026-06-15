from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from batchgen.kv_cache.deepseek_v4_single_kv_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
)
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVStats

_GPU_RESIDENT_ONLY_ERROR = "V4 KV is GPU-resident only for this milestone"


@dataclass(frozen=True)
class DeepSeekV4LayerRouting:
    swa_layer_idx: int
    c4_layer_idx: Optional[int]
    c128_layer_idx: Optional[int]
    indexer_layer_idx: Optional[int]


class DeepSeekV4KVCoordinator:
    """Standalone 4-pool KV coordinator for DeepSeek-V4 Flash.

    This is intentionally *not* a DualKVCacheCoordinator subclass and does not
    expose `.primary` / `.auxiliary`. V4's SWA/c4/c128/indexer pools are not a
    mirrored 2-pool layout, so the only safe contract for Phase 1 is a
    standalone coordinator that duck-types the worker-facing page-allocation API.
    """

    indexer_head_dim = 128

    def __init__(
        self,
        *,
        compress_ratios: Sequence[int],
        num_pages: int,
        device: str | int,
        base_page_size: int = 256,
        swa_page_size: int = 128,
    ) -> None:
        if not compress_ratios:
            raise ValueError("compress_ratios must be non-empty")
        if base_page_size <= 0:
            raise ValueError(
                f"base_page_size must be > 0, got {base_page_size}"
            )
        if base_page_size % 128 != 0:
            raise ValueError(
                f"base_page_size must be divisible by 128, got {base_page_size}"
            )
        if swa_page_size <= 0:
            raise ValueError(f"swa_page_size must be > 0, got {swa_page_size}")
        if num_pages <= 0:
            raise ValueError(f"num_pages must be > 0, got {num_pages}")

        self.compress_ratios = [int(ratio) for ratio in compress_ratios]
        self.device = device
        self.base_page_size = int(base_page_size)
        self.swa_page_size = int(swa_page_size)
        self.c4_page_size = self.base_page_size // 4
        self.c128_page_size = self.base_page_size // 128
        self.num_layers = len(self.compress_ratios)

        c4_layers = [
            idx for idx, ratio in enumerate(self.compress_ratios) if ratio == 4
        ]
        c128_layers = [
            idx
            for idx, ratio in enumerate(self.compress_ratios)
            if ratio == 128
        ]

        self.swa = DeepSeekV4SingleKVPool(
            num_layers=self.num_layers,
            num_pages=num_pages,
            page_size_tokens=self.swa_page_size,
            device=device,
        )
        self.c4 = DeepSeekV4SingleKVPool(
            num_layers=len(c4_layers),
            num_pages=num_pages,
            page_size_tokens=self.c4_page_size,
            device=device,
        )
        self.c128 = DeepSeekV4SingleKVPool(
            num_layers=len(c128_layers),
            num_pages=num_pages,
            page_size_tokens=self.c128_page_size,
            device=device,
        )
        self.indexer = DeepSeekV4IndexerPool(
            num_layers=len(c4_layers),
            num_pages=num_pages,
            page_size_tokens=self.c4_page_size,
            indexer_head_dim=self.indexer_head_dim,
            device=device,
        )

        c4_map = {
            layer_idx: local_idx
            for local_idx, layer_idx in enumerate(c4_layers)
        }
        c128_map = {
            layer_idx: local_idx
            for local_idx, layer_idx in enumerate(c128_layers)
        }
        self.layer_routing: Dict[int, DeepSeekV4LayerRouting] = {
            layer_idx: DeepSeekV4LayerRouting(
                swa_layer_idx=layer_idx,
                c4_layer_idx=c4_map.get(layer_idx),
                c128_layer_idx=c128_map.get(layer_idx),
                indexer_layer_idx=c4_map.get(layer_idx),
            )
            for layer_idx in range(self.num_layers)
        }
        self._active_sequence_ids: tuple[int, ...] = tuple()
        self.is_initialized = False
        self._gpu_page_table_manager = None

    @staticmethod
    def _ceil_div(value: int, divisor: int) -> int:
        return -(-value // divisor)

    @classmethod
    def _single_pool_bytes_per_page(cls, page_size_tokens: int) -> int:
        alignment = int(DeepSeekV4SingleKVPool.token_body_bytes)
        raw = int(page_size_tokens) * int(
            DeepSeekV4SingleKVPool.bytes_per_token
        )
        return cls._ceil_div(raw, alignment) * alignment

    @classmethod
    def bytes_per_page_unit_for(
        cls,
        *,
        compress_ratios: Sequence[int],
        base_page_size: int = 256,
        swa_page_size: int = 128,
    ) -> int:
        """Total bytes for one page-unit across all 4 differently-sized pools (worker uses it to size num_pages from a GB budget)."""
        ratios = [int(ratio) for ratio in compress_ratios]
        if not ratios:
            raise ValueError("compress_ratios must be non-empty")
        invalid = sorted(set(ratios) - {0, 4, 128})
        if invalid:
            raise ValueError(
                f"Unsupported DeepSeek-V4 compress_ratios: {invalid}"
            )

        c4_layers = sum(1 for ratio in ratios if ratio == 4)
        c128_layers = sum(1 for ratio in ratios if ratio == 128)

        c4_page_size = int(base_page_size) // 4
        c128_page_size = int(base_page_size) // 128

        swa_bytes = len(ratios) * cls._single_pool_bytes_per_page(swa_page_size)
        c4_bytes = c4_layers * cls._single_pool_bytes_per_page(c4_page_size)
        c128_bytes = c128_layers * cls._single_pool_bytes_per_page(
            c128_page_size
        )
        indexer_bytes = (
            c4_layers
            * c4_page_size
            * int(DeepSeekV4IndexerPool.bytes_per_token)
        )
        return swa_bytes + c4_bytes + c128_bytes + indexer_bytes

    def bytes_per_page_unit(self) -> int:
        return type(self).bytes_per_page_unit_for(
            compress_ratios=self.compress_ratios,
            base_page_size=self.base_page_size,
            swa_page_size=self.swa_page_size,
        )

    def initialize(self) -> None:
        self.swa.initialize()
        self.c4.initialize()
        self.c128.initialize()
        self.indexer.initialize()
        self.is_initialized = True

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        self.swa.destroy(empty_cuda_cache=empty_cuda_cache)
        self.c4.destroy(empty_cuda_cache=empty_cuda_cache)
        self.c128.destroy(empty_cuda_cache=empty_cuda_cache)
        self.indexer.destroy(empty_cuda_cache=empty_cuda_cache)
        self._active_sequence_ids = tuple()
        self.is_initialized = False

    def _ensure_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("DeepSeekV4KVCoordinator is not initialized")

    def get_layer_routing(self, layer_idx: int) -> DeepSeekV4LayerRouting:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        return self.layer_routing[layer_idx]

    def _pool_order(self) -> list[tuple[str, object]]:
        return [
            ("swa", self.swa),
            ("c4", self.c4),
            ("c128", self.c128),
            ("indexer", self.indexer),
        ]

    # Raw context tokens map to stored rows per pool at these ratios: swa keeps
    # raw tokens, c4/indexer keep raw/4 rows, c128 keeps raw/128 rows. The store
    # paths in v4_prefill_populate.py write arange(compressed.shape[0]) slots, so
    # page accounting must use these compressed counts, not the raw token count.
    _POOL_COMPRESS_RATIO: Dict[str, int] = {
        "swa": 1,
        "c4": 4,
        "c128": 128,
        "indexer": 4,
    }

    def _pool_logical_tokens(
        self, pool_name: str, num_tokens: Sequence[int]
    ) -> List[int]:
        # ceil (not floor) is a safe never-under bound on stored rows; the max(1)
        # floor reserves at least one page per pool for a fresh sequence.
        ratio = self._POOL_COMPRESS_RATIO[pool_name]
        if ratio == 1:
            return [int(t) for t in num_tokens]
        return [max(1, self._ceil_div(int(t), ratio)) for t in num_tokens]

    def allocate_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        num_tokens: Sequence[int],
    ) -> Dict[int, List[int]]:
        self._ensure_initialized()
        allocations_by_pool: dict[str, Dict[int, List[int]]] = {}
        try:
            for pool_name, pool in self._pool_order():
                pool_tokens = self._pool_logical_tokens(pool_name, num_tokens)
                allocations_by_pool[pool_name] = (
                    pool.allocate_pages_for_sequences(sequence_ids, pool_tokens)
                )
        except Exception:
            for pool_name, allocations in allocations_by_pool.items():
                getattr(self, pool_name)._rollback_allocations(allocations)
            raise
        return allocations_by_pool.get("swa", {})

    def can_allocate_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        num_tokens: Sequence[int],
    ) -> bool:
        # Per-pool preflight for the scheduler: a summed free-page count hides an
        # exhausted pool, so admission must verify EVERY pool independently and
        # fall back to ON_HOLD/skip rather than hit the pool's hard pop() raise.
        self._ensure_initialized()
        shortages = self.additional_pages_needed_by_pool(
            sequence_ids, num_tokens
        )
        for pool_name, pool in self._pool_order():
            if shortages.get(pool_name, 0) > pool.get_stats().num_free_pages:
                return False
        return True

    def additional_pages_needed_by_pool(
        self,
        sequence_ids: Sequence[int],
        num_tokens: Sequence[int],
    ) -> Dict[str, int]:
        self._ensure_initialized()
        needed: Dict[str, int] = {}
        for pool_name, pool in self._pool_order():
            pool_tokens = self._pool_logical_tokens(pool_name, num_tokens)
            total_missing = 0
            for seq_id, token_count in zip(sequence_ids, pool_tokens):
                required = pool._required_pages(int(token_count))
                state = pool._sequences.get(int(seq_id))
                current = 0 if state is None else int(state.pages.numel())
                total_missing += max(0, required - current)
            needed[pool_name] = total_missing
        return needed

    def free_worker_pages(self, page_size_tokens: int = 64) -> int:
        # Coarse scalar for legacy greedy paths only: the binding (min over pools)
        # free raw-token headroom, in 64-token worker pages. Real admission must
        # still go through can_allocate_pages_for_sequences.
        self._ensure_initialized()
        if page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be > 0")
        free_raw_tokens = None
        for pool_name, pool in self._pool_order():
            ratio = self._POOL_COMPRESS_RATIO[pool_name]
            pool_free_raw = (
                pool.get_stats().num_free_pages * pool.page_size_tokens * ratio
            )
            if free_raw_tokens is None or pool_free_raw < free_raw_tokens:
                free_raw_tokens = pool_free_raw
        if free_raw_tokens is None:
            return 0
        return free_raw_tokens // page_size_tokens

    def rebuild_page_table(
        self, sequence_ids: Sequence[int]
    ) -> Mapping[str, object]:
        self._ensure_initialized()
        self._active_sequence_ids = tuple(
            int(seq_id) for seq_id in sequence_ids
        )
        return {
            "swa": self.swa.rebuild_page_table(sequence_ids),
            "c4": self.c4.rebuild_page_table(sequence_ids),
            "c128": self.c128.rebuild_page_table(sequence_ids),
            "indexer": self.indexer.rebuild_page_table(sequence_ids),
        }

    def clear_page_table(self) -> None:
        self._ensure_initialized()
        for _name, pool in self._pool_order():
            pool._clear_page_table()

    def extend_pages_for_sequence(
        self, sequence_id: int, new_total_tokens: int
    ) -> int:
        self._ensure_initialized()
        allocations = self.allocate_pages_for_sequences(
            [sequence_id], [new_total_tokens]
        )
        return len(allocations.get(sequence_id, []))

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        self._ensure_initialized()
        self.swa.free_pages_for_sequences(sequence_ids)
        self.c4.free_pages_for_sequences(sequence_ids)
        self.c128.free_pages_for_sequences(sequence_ids)
        self.indexer.free_pages_for_sequences(sequence_ids)
        if self._active_sequence_ids:
            remaining = tuple(
                seq_id
                for seq_id in self._active_sequence_ids
                if seq_id not in set(sequence_ids)
            )
            self._active_sequence_ids = remaining

    def get_stats(self) -> GPUPagedKVStats:
        self._ensure_initialized()
        stats = [
            self.swa.get_stats(),
            self.c4.get_stats(),
            self.c128.get_stats(),
            self.indexer.get_stats(),
        ]
        return GPUPagedKVStats(
            num_total_pages=sum(item.num_total_pages for item in stats),
            num_free_pages=sum(item.num_free_pages for item in stats),
            num_used_pages=sum(item.num_used_pages for item in stats),
            num_total_pages_allocated=sum(
                item.num_total_pages_allocated for item in stats
            ),
        )

    def resident_only_error(self) -> RuntimeError:
        return RuntimeError(_GPU_RESIDENT_ONLY_ERROR)

    def copy_kv_to_tensor(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def copy_tensor_to_kv(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def get_context_kv_page_ptrs(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def get_sequence_layer_page_pointers(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def get_padded_3d_page_pointers(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def export_active_sequence_page_counts(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def async_offload_layer_kv_to_host(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def load_cpu_copy(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def migrate_to_host(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()

    def offload_to_host(self, *args, **kwargs):
        del args, kwargs
        raise self.resident_only_error()


__all__ = [
    "DeepSeekV4KVCoordinator",
    "DeepSeekV4LayerRouting",
]
