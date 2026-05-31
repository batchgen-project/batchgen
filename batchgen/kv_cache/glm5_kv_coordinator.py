"""GLM-5 KV coordinators.

GLM-5 uses two logical KV groups:

- group 0: primary MLA compressed KV
- group 1: DSA/indexer KV

Unlike the legacy dual coordinators, these classes do not require primary and
indexer managers to allocate identical physical page ids. The shared invariant
is the logical sequence set, token/page counts, and active slot order. Prefix
cache metadata keeps the per-group physical page handles separate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from batchgen.config.model_name_utils import is_glm5_backend_model
from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
    GPUPagedKVStats,
)
from batchgen.kv_cache.host_kv_mananger_config import (
    HOST_KV_AUX_SHM_NAME,
    HOST_KV_SHM_NAME,
    HostKVGroupProfile,
    _dtype_size_bytes,
    resolve_host_kv_group_profiles,
)
from batchgen.models.engine_loader import core_engine as bg_lib

logger = logging.getLogger(__name__)

GLM5_PRIMARY_GROUP_ID = 0
GLM5_INDEXER_GROUP_ID = 1


@dataclass
class GLM5AsyncKVTask:
    """Composite async task for primary + indexer Host->GPU KV loads."""

    primary_task: Any
    indexer_task: Any
    tensors: Any = None

    def wait(self) -> None:
        errors = []
        for name, task in (
            ("primary", self.primary_task),
            ("indexer", self.indexer_task),
        ):
            try:
                task.wait()
            except Exception as exc:
                errors.append((name, exc))
        if not errors:
            return
        if len(errors) == 1:
            raise errors[0][1]
        names = ", ".join(name for name, _ in errors)
        raise RuntimeError(
            f"GLM5AsyncKVTask wait failed for KV loads: {names}"
        ) from errors[0][1]


def is_glm5_dual_kv_model(model_name: str | None) -> bool:
    """Return whether this model should use the GLM-5 KV coordinator."""

    return is_glm5_backend_model(model_name)


def _try_set_logger_name(config: Any, name: str) -> None:
    try:
        config.logger_name = name
    except AttributeError:
        return


def _glm5_group_profiles(
    model_name: str,
) -> tuple[HostKVGroupProfile, HostKVGroupProfile]:
    if not is_glm5_dual_kv_model(model_name):
        raise ValueError(f"Model '{model_name}' is not a GLM-5 KV model")
    profiles = {
        int(profile.group_id): profile
        for profile in resolve_host_kv_group_profiles(model_name)
    }
    try:
        primary = profiles[GLM5_PRIMARY_GROUP_ID]
        indexer = profiles[GLM5_INDEXER_GROUP_ID]
    except KeyError as exc:
        raise ValueError(
            f"GLM-5 KV profiles must include groups "
            f"{GLM5_PRIMARY_GROUP_ID} and {GLM5_INDEXER_GROUP_ID}"
        ) from exc
    if primary.raw_page_tokens != indexer.raw_page_tokens:
        raise ValueError(
            "GLM-5 primary/indexer raw page mismatch: "
            f"primary={primary.raw_page_tokens}, indexer={indexer.raw_page_tokens}"
        )
    return primary, indexer


def _compute_glm5_page_count(
    model_name: str,
    host_kv_cache_size: int,
) -> tuple[HostKVGroupProfile, HostKVGroupProfile, int]:
    primary, indexer = _glm5_group_profiles(model_name)
    combined_bytes_per_page = (
        primary.bytes_per_page() * primary.num_layers
        + indexer.bytes_per_page() * indexer.num_layers
    )
    num_pages = int(host_kv_cache_size) // combined_bytes_per_page
    if num_pages <= 0:
        raise ValueError(
            f"host_kv_cache_size ({host_kv_cache_size}) too small for "
            f"GLM-5 KV cache (combined bytes per page = "
            f"{combined_bytes_per_page})"
        )
    return primary, indexer, num_pages


def _build_host_config_from_group(
    profile: HostKVGroupProfile,
    *,
    shm_name: str,
    num_pages: int,
) -> Any:
    config = bg_lib.HostPagedKVConfig()
    config.shm_name = shm_name
    config.num_layers = profile.num_layers
    config.num_pages = int(num_pages)
    config.page_size_tokens = profile.storage_page_tokens
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


class GLM5HostKVCoordinator:
    """Host-side GLM-5 KV facade with independent primary/indexer pages."""

    def __init__(self, primary: Any, indexer: Any) -> None:
        self.primary = primary
        self.indexer = indexer
        self.auxiliary = indexer

    def views_by_group(self) -> dict[int, Any]:
        return {
            GLM5_PRIMARY_GROUP_ID: self.primary,
            GLM5_INDEXER_GROUP_ID: self.indexer,
        }

    @classmethod
    def from_budget(
        cls,
        *,
        model_name: str,
        host_kv_cache_size: int,
        core_engine_module: Any,
        enable_memfd: bool = False,
        memfd_creator_pid: int = -1,
        memfd_fd: int = -1,
        aux_memfd_fd: int = -1,
    ) -> Optional["GLM5HostKVCoordinator"]:
        if not is_glm5_dual_kv_model(model_name):
            return None
        primary_profile, indexer_profile, num_pages = _compute_glm5_page_count(
            model_name, host_kv_cache_size
        )
        primary_config = _build_host_config_from_group(
            primary_profile,
            shm_name=HOST_KV_SHM_NAME,
            num_pages=num_pages,
        )
        indexer_config = _build_host_config_from_group(
            indexer_profile,
            shm_name=HOST_KV_AUX_SHM_NAME,
            num_pages=num_pages,
        )
        _try_set_logger_name(primary_config, "GLM5HostPagedKVWorkerView")
        _try_set_logger_name(indexer_config, "GLM5IndexerHostPagedKVWorkerView")

        if enable_memfd:
            primary_config.enable_memfd = True
            primary_config.memfd_creator_pid = memfd_creator_pid
            primary_config.memfd_fd = memfd_fd
            indexer_config.enable_memfd = True
            indexer_config.memfd_creator_pid = memfd_creator_pid
            indexer_config.memfd_fd = aux_memfd_fd

        primary_view = core_engine_module.MLAHostPagedKVWorkerView(
            primary_config
        )
        indexer_view = core_engine_module.MLAHostPagedKVWorkerView(
            indexer_config
        )
        logger.info(
            "GLM5HostKVCoordinator created: %d pages, primary dim=%d, "
            "indexer dim=%d",
            num_pages,
            primary_profile.k_head_dim,
            indexer_profile.k_head_dim,
        )
        return cls(primary_view, indexer_view)

    @classmethod
    def create_managers(
        cls,
        *,
        model_name: str,
        host_kv_cache_size: int,
        enable_memfd: bool = False,
    ) -> Optional[tuple[Any, Any]]:
        if not is_glm5_dual_kv_model(model_name):
            return None
        primary_profile, indexer_profile, num_pages = _compute_glm5_page_count(
            model_name, host_kv_cache_size
        )
        primary_config = _build_host_config_from_group(
            primary_profile,
            shm_name=HOST_KV_SHM_NAME,
            num_pages=num_pages,
        )
        indexer_config = _build_host_config_from_group(
            indexer_profile,
            shm_name=HOST_KV_AUX_SHM_NAME,
            num_pages=num_pages,
        )
        _try_set_logger_name(primary_config, "GLM5HostPagedKVManager")
        _try_set_logger_name(indexer_config, "GLM5IndexerHostPagedKVManager")

        if enable_memfd:
            primary_config.enable_memfd = True
            indexer_config.enable_memfd = True

        primary_manager = bg_lib.MLAHostPagedKVManager(primary_config)
        primary_manager.initialize(True)
        indexer_manager = bg_lib.MLAHostPagedKVManager(indexer_config)
        indexer_manager.initialize(True)
        logger.info(
            "GLM5HostKVCoordinator managers created: %d pages, primary dim=%d, "
            "indexer dim=%d",
            num_pages,
            primary_profile.k_head_dim,
            indexer_profile.k_head_dim,
        )
        return primary_manager, indexer_manager

    def initialize(self, **kwargs: Any) -> None:
        self.primary.initialize(**kwargs)
        self.indexer.initialize(**kwargs)

    def register_sequences(self, sequence_ids: Sequence[int]) -> None:
        self.primary.register_sequences(sequence_ids)
        try:
            self.indexer.register_sequences(sequence_ids)
        except Exception:
            self.primary.unregister_sequences(sequence_ids)
            raise

    def allocate_pages_for_sequences(self, seq_token_pairs: Sequence[Any]) -> None:
        pairs = list(seq_token_pairs)
        sequence_ids = [int(seq_id) for seq_id, _ in pairs]
        self.primary.allocate_pages_for_sequences(pairs)
        try:
            self.indexer.allocate_pages_for_sequences(pairs)
        except Exception:
            self.primary.release_sequence_pages(sequence_ids)
            raise

    def grow_pages_for_sequences(self, seq_page_pairs: Sequence[Any]) -> None:
        pairs = list(seq_page_pairs)
        needed = sum(int(pages) for _, pages in pairs)
        primary_free = int(self.primary.get_stats().num_free_pages)
        indexer_free = int(self.indexer.get_stats().num_free_pages)
        if needed > primary_free or needed > indexer_free:
            raise RuntimeError(
                "GLM-5 grow_pages_for_sequences: insufficient Host KV free "
                f"pages: need={needed}, primary_free={primary_free}, "
                f"indexer_free={indexer_free}"
            )
        self.primary.grow_pages_for_sequences(pairs)
        self.indexer.grow_pages_for_sequences(pairs)

    def release_sequence_pages(self, sequence_ids: Sequence[int]) -> None:
        self.primary.release_sequence_pages(sequence_ids)
        self.indexer.release_sequence_pages(sequence_ids)

    def unregister_sequences(self, sequence_ids: Sequence[int]) -> None:
        self.primary.unregister_sequences(sequence_ids)
        self.indexer.unregister_sequences(sequence_ids)

    def get_stats(self) -> Any:
        primary_stats = self.primary.get_stats()
        indexer_stats = self.indexer.get_stats()
        if indexer_stats.num_free_pages < primary_stats.num_free_pages:
            return indexer_stats
        return primary_stats

    def async_load_layer_paged_kv_to_device(self, **kwargs: Any) -> None:
        raise RuntimeError(
            "GLM-5 KV load must use "
            "async_load_layer_paged_kv_to_device_dual()"
        )

    def async_load_layer_paged_kv_to_device_dual(
        self,
        *,
        sequence_ids: torch.Tensor,
        primary_active_page_counts: torch.Tensor,
        primary_k_device_ptrs: torch.Tensor,
        primary_v_device_ptrs: Optional[torch.Tensor],
        aux_active_page_counts: torch.Tensor,
        aux_k_device_ptrs: torch.Tensor,
        aux_v_device_ptrs: Optional[torch.Tensor],
        tensors: Any = None,
    ) -> GLM5AsyncKVTask:
        if primary_active_page_counts.tolist() != aux_active_page_counts.tolist():
            raise RuntimeError(
                "GLM-5 primary/indexer load page-count mismatch: "
                f"primary={primary_active_page_counts.tolist()}, "
                f"indexer={aux_active_page_counts.tolist()}"
            )
        primary_task = self.primary.async_load_layer_paged_kv_to_device(
            sequence_ids=sequence_ids,
            active_page_counts=primary_active_page_counts,
            k_device_ptrs=primary_k_device_ptrs,
            v_device_ptrs=primary_v_device_ptrs,
        )
        try:
            indexer_task = self.indexer.async_load_layer_paged_kv_to_device(
                sequence_ids=sequence_ids,
                active_page_counts=aux_active_page_counts,
                k_device_ptrs=aux_k_device_ptrs,
                v_device_ptrs=aux_v_device_ptrs,
            )
        except Exception:
            primary_task.wait()
            raise
        return GLM5AsyncKVTask(
            primary_task=primary_task,
            indexer_task=indexer_task,
            tensors=tensors,
        )

    def async_offload_layer_kv_to_host(self, **kwargs: Any) -> None:
        raise RuntimeError(
            "GLM-5 KV offload must explicitly offload primary and indexer KV"
        )


class GLM5GPUKVCoordinator:
    """GPU-side GLM-5 KV facade with per-group physical page ownership."""

    def __init__(
        self,
        primary: GPUPagedKVCacheManager,
        indexer: GPUPagedKVCacheManager,
    ) -> None:
        self.primary = primary
        self.indexer = indexer
        self.auxiliary = indexer
        if primary.config.page_size_tokens != indexer.config.page_size_tokens:
            raise ValueError(
                "GLM-5 primary/indexer GPU KV page size mismatch: "
                f"primary={primary.config.page_size_tokens}, "
                f"indexer={indexer.config.page_size_tokens}"
            )

    def managers_by_group(self) -> dict[int, GPUPagedKVCacheManager]:
        return {
            GLM5_PRIMARY_GROUP_ID: self.primary,
            GLM5_INDEXER_GROUP_ID: self.indexer,
        }

    def initialize(self) -> None:
        self.primary.initialize()
        self.indexer.initialize()
        logger.info(
            "GLM5GPUKVCoordinator initialized: primary=%s, indexer=%s",
            self.primary.get_stats(),
            self.indexer.get_stats(),
        )

    def destroy(self, *, empty_cuda_cache: bool = False) -> None:
        self.primary.destroy(empty_cuda_cache=empty_cuda_cache)
        self.indexer.destroy(empty_cuda_cache=empty_cuda_cache)

    @property
    def is_initialized(self) -> bool:
        return self.primary.is_initialized and self.indexer.is_initialized

    def allocate_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        num_tokens: Sequence[int],
    ) -> Any:
        result = self.primary.allocate_pages_for_sequences(
            sequence_ids, num_tokens
        )
        try:
            self.indexer.allocate_pages_for_sequences(sequence_ids, num_tokens)
        except Exception:
            self._rollback_primary_allocations(result)
            raise
        self.assert_aligned_state("allocate_pages_for_sequences", sequence_ids)
        return result

    def grow_pages_for_sequences(
        self,
        sequence_ids: Sequence[int],
        additional_tokens: Sequence[int],
    ) -> Any:
        needed = sum(int(tokens) for tokens in additional_tokens)
        primary_free = int(self.primary.get_stats().num_free_pages)
        indexer_free = int(self.indexer.get_stats().num_free_pages)
        if needed > primary_free or needed > indexer_free:
            raise RuntimeError(
                "GLM-5 grow_pages_for_sequences: insufficient GPU KV free "
                f"pages: need={needed}, primary_free={primary_free}, "
                f"indexer_free={indexer_free}"
            )
        result = self.primary.grow_pages_for_sequences(
            sequence_ids, additional_tokens
        )
        self.indexer.grow_pages_for_sequences(sequence_ids, additional_tokens)
        self.assert_aligned_state("grow_pages_for_sequences", sequence_ids)
        return result

    def extend_pages_for_sequence(
        self,
        sequence_id: int,
        new_total_tokens: int,
    ) -> int:
        primary_state = self.primary._sequences.get(sequence_id)
        indexer_state = self.indexer._sequences.get(sequence_id)
        if primary_state is None or indexer_state is None:
            raise KeyError(
                f"extend_pages_for_sequence: GLM-5 sequence {sequence_id} "
                "is not allocated in both primary and indexer managers"
            )
        primary_required = int(
            self.primary._geometry.required_pages(new_total_tokens)
        )
        indexer_required = int(
            self.indexer._geometry.required_pages(new_total_tokens)
        )
        primary_missing = max(0, primary_required - int(primary_state.pages.numel()))
        indexer_missing = max(0, indexer_required - int(indexer_state.pages.numel()))
        if primary_missing != indexer_missing:
            raise RuntimeError(
                "GLM-5 primary/indexer page growth mismatch: "
                f"primary_missing={primary_missing}, "
                f"indexer_missing={indexer_missing}"
            )
        if primary_missing <= 0:
            return 0
        if primary_missing > self.primary.get_stats().num_free_pages:
            raise RuntimeError(
                "GLM-5 primary GPU KV has insufficient free pages: "
                f"need={primary_missing}"
            )
        if indexer_missing > self.indexer.get_stats().num_free_pages:
            raise RuntimeError(
                "GLM-5 indexer GPU KV has insufficient free pages: "
                f"need={indexer_missing}"
            )
        added = self.primary.extend_pages_for_sequence(
            sequence_id, new_total_tokens
        )
        self.indexer.extend_pages_for_sequence(sequence_id, new_total_tokens)
        self.assert_aligned_state("extend_pages_for_sequence", [sequence_id])
        return added

    def rebuild_page_table(self, sequence_ids: Sequence[int]) -> torch.Tensor:
        table = self.primary.rebuild_page_table(sequence_ids)
        self.indexer.rebuild_page_table(sequence_ids)
        self._assert_slot_order("rebuild_page_table")
        return table

    def clear_page_table(self) -> None:
        self.primary.clear_page_table()
        self.indexer.clear_page_table()

    def free_pages_for_sequences(self, sequence_ids: Sequence[int]) -> None:
        self.primary.free_pages_for_sequences(sequence_ids)
        self.indexer.free_pages_for_sequences(sequence_ids)

    def get_stats(self) -> GPUPagedKVStats:
        primary_stats = self.primary.get_stats()
        indexer_stats = self.indexer.get_stats()
        if indexer_stats.num_free_pages < primary_stats.num_free_pages:
            return indexer_stats
        return primary_stats

    def get_page_table_version(self) -> int:
        return self.primary.get_page_table_version()

    @property
    def config(self) -> GPUPagedKVConfig:
        return self.primary.config

    @property
    def device(self) -> torch.device:
        return self.primary.device

    @property
    def _gpu_page_table_manager(self) -> Any:
        return self.primary._gpu_page_table_manager

    @property
    def _sequences(self) -> Any:
        return self.primary._sequences

    def copy_kv_to_tensor(self, sequence_id: int) -> torch.Tensor:
        return self.primary.copy_kv_to_tensor(sequence_id)

    def copy_tensor_to_kv(self, sequence_id: int, k_tensor: torch.Tensor) -> None:
        self.primary.copy_tensor_to_kv(sequence_id, k_tensor)

    def get_context_kv_page_ptrs(self, *args: Any, **kwargs: Any) -> Any:
        return self.primary.get_context_kv_page_ptrs(*args, **kwargs)

    def get_sequence_layer_page_pointers(self, *args: Any, **kwargs: Any) -> Any:
        return self.primary.get_sequence_layer_page_pointers(*args, **kwargs)

    def export_layer_page_pointer_table(self, *args: Any, **kwargs: Any) -> Any:
        return self.primary.export_layer_page_pointer_table(*args, **kwargs)

    def get_kv_tensors(self) -> Any:
        raise RuntimeError(
            "GLM5GPUKVCoordinator.get_kv_tensors() is primary-only; "
            "use .primary or .indexer explicitly"
        )

    def get_layer_kv_with_page_table(self, layer_idx: int) -> Any:
        raise RuntimeError(
            "GLM5GPUKVCoordinator.get_layer_kv_with_page_table() is "
            "primary-only; use .primary or .indexer explicitly"
        )

    def export_active_sequence_page_counts(self) -> torch.Tensor:
        raise RuntimeError(
            "GLM5GPUKVCoordinator.export_active_sequence_page_counts() is "
            "primary-only; use .primary or .indexer explicitly"
        )

    def get_padded_3d_page_pointers(self) -> Any:
        raise RuntimeError(
            "GLM5GPUKVCoordinator.get_padded_3d_page_pointers() is "
            "primary-only; use .primary or .indexer explicitly"
        )

    def assert_aligned_state(
        self,
        op_name: str,
        sequence_ids: Optional[Sequence[int]] = None,
    ) -> None:
        primary_ids = set(self.primary._sequences.keys())
        indexer_ids = set(self.indexer._sequences.keys())
        if primary_ids != indexer_ids:
            raise RuntimeError(
                f"{op_name}: GLM-5 primary/indexer sequence set mismatch: "
                f"primary_only={sorted(primary_ids - indexer_ids)[:10]}, "
                f"indexer_only={sorted(indexer_ids - primary_ids)[:10]}"
            )
        check_ids = (
            list(sequence_ids) if sequence_ids is not None else sorted(primary_ids)
        )
        for seq_id in check_ids:
            primary_state = self.primary._sequences.get(seq_id)
            indexer_state = self.indexer._sequences.get(seq_id)
            if primary_state is None or indexer_state is None:
                continue
            primary_pages = int(primary_state.pages.numel())
            indexer_pages = int(indexer_state.pages.numel())
            if primary_pages != indexer_pages:
                raise RuntimeError(
                    f"{op_name}: GLM-5 primary/indexer page-count mismatch "
                    f"for seq {seq_id}: primary={primary_pages}, "
                    f"indexer={indexer_pages}"
                )
        self._assert_slot_order(op_name)

    def _assert_slot_order(self, op_name: str) -> None:
        primary_slots = list(self.primary._gpu_page_table_manager.slot_to_seq_id)
        indexer_slots = list(self.indexer._gpu_page_table_manager.slot_to_seq_id)
        if primary_slots != indexer_slots:
            raise RuntimeError(
                f"{op_name}: GLM-5 primary/indexer slot order mismatch: "
                f"primary={primary_slots[:10]}, indexer={indexer_slots[:10]}"
            )

    def _rollback_primary_allocations(self, allocations: Any) -> None:
        if not allocations:
            return
        reclaimed = []
        for seq_id, pages in allocations.items():
            if not pages:
                continue
            state = self.primary._sequences.get(seq_id)
            if state is None:
                continue
            count = len(pages)
            tail = state.pages[-count:].tolist()
            if tail != pages:
                raise RuntimeError(
                    f"Cannot rollback GLM-5 primary KV allocation for seq "
                    f"{seq_id}: tail={tail}, allocated={pages}"
                )
            reclaimed.append(state.pages[-count:].clone())
            if state.pages.numel() == count:
                del self.primary._sequences[seq_id]
            else:
                state.pages = state.pages[:-count].clone()
        if reclaimed:
            self.primary._free_pages.push(torch.cat(reclaimed, dim=0))
            self.primary._clear_active_page_pointer_tables()
