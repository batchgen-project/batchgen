"""Materialize attached Host prefix pages into temporary GPU paged KV."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


class _AsyncTask(Protocol):
    def wait(self) -> None: ...


@dataclass
class PrefixMaterialization:
    manager: object
    append_plan: object
    load_task: _AsyncTask | None
    coordinator: object
    attachment_handles: tuple[int, ...]
    _load_complete: bool = False
    _closed: bool = False

    def wait_for_layer(self, layer_idx: int) -> None:
        if self._closed:
            raise RuntimeError("prefix materialization is already closed")
        del layer_idx
        self._wait_for_load()

    def _wait_for_load(self) -> None:
        if self.load_task is not None and not self._load_complete:
            self.load_task.wait()
            self._load_complete = True

    def close(self, *, empty_cuda_cache: bool = False) -> None:
        if self._closed:
            return
        try:
            self._wait_for_load()
        finally:
            for handle in reversed(self.attachment_handles):
                self.coordinator.end_attachment_load(handle)
            self.manager.destroy(empty_cuda_cache=empty_cuda_cache)
            self._closed = True


def materialize_gpt_oss_prefixes(
    *,
    gpu_manager: object,
    host_worker_view: object,
    coordinator: object,
    lookup_results: Sequence[object],
    sequence_ids: Sequence[int],
    prompt_lengths: Sequence[int],
    compute_cached_tokens: Sequence[int],
    raw_page_tokens: int,
) -> PrefixMaterialization:
    """Build one mixed hit/miss GPT-OSS prefill materialization."""

    count = len(lookup_results)
    if not (
        len(sequence_ids)
        == len(prompt_lengths)
        == len(compute_cached_tokens)
        == count
    ):
        raise ValueError("prefix materialization inputs must have equal lengths")
    if count == 0:
        raise ValueError("prefix materialization requires at least one sequence")

    host_page_tokens = int(raw_page_tokens)
    if host_page_tokens <= 0:
        raise ValueError("raw_page_tokens must be positive")

    prefix_lens = [int(value) for value in compute_cached_tokens]
    full_lens = [int(value) for value in prompt_lengths]
    suffix_lens = [
        full - prefix for full, prefix in zip(full_lens, prefix_lens)
    ]
    page_counts = [
        math.ceil(prefix / host_page_tokens) if prefix else 0
        for prefix in prefix_lens
    ]
    if any(full <= 0 for full in full_lens):
        raise ValueError("prompt lengths must be positive")
    if any(prefix < 0 or suffix <= 0 for prefix, suffix in zip(prefix_lens, suffix_lens)):
        raise ValueError(
            "computed prefix must leave at least one prompt token for prefill"
        )

    gpu_manager.allocate_pages_for_sequences(sequence_ids, full_lens)
    gpu_manager.rebuild_page_table(sequence_ids)
    append_plan = gpu_manager.prepare_prefill_suffix_append(
        sequence_ids=sequence_ids,
        prefix_lens=prefix_lens,
        suffix_lens=suffix_lens,
        rebuild_page_table=False,
    )

    max_pages = max(page_counts)
    if max_pages == 0:
        return PrefixMaterialization(
            manager=gpu_manager,
            append_plan=append_plan,
            load_task=None,
            coordinator=coordinator,
            attachment_handles=(),
        )

    host_rows: list[list[int]] = []
    attachment_handles: list[int] = []
    for result, page_count in zip(lookup_results, page_counts):
        pages: list[int] = []
        if page_count:
            spans = list(result.materialization_spans)
            if len(spans) != 1 or int(spans[0].group_id) != 0:
                raise RuntimeError(
                    "GPT-OSS prefix materialization requires FULL_KV group 0"
                )
            pages = [int(page.page_id) for page in spans[0].pages]
            if len(pages) < page_count:
                raise RuntimeError(
                    "prefix lookup returned too few Host pages for materialization"
                )
            handle = int(result.attachment_handle)
            if handle == 0:
                raise RuntimeError("prefix hit is missing its attachment handle")
            attachment_handles.append(handle)
        row = pages[:page_count]
        row.extend([0] * (max_pages - len(row)))
        host_rows.append(row)

    host_page_ids = torch.tensor(host_rows, dtype=torch.int64)
    active_page_counts = torch.tensor(page_counts, dtype=torch.int64)
    k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
    k_ptrs, v_ptrs = _expand_device_ptrs_for_host_pages(
        gpu_manager=gpu_manager,
        k_device_ptrs=k_ptrs,
        v_device_ptrs=v_ptrs,
        active_page_counts=active_page_counts,
        host_page_tokens=host_page_tokens,
    )

    begun: list[int] = []
    try:
        for handle in attachment_handles:
            coordinator.begin_attachment_load(handle)
            begun.append(handle)
        load_task = host_worker_view.async_load_prefix_pages_to_device(
            host_page_ids=host_page_ids,
            active_page_counts=active_page_counts,
            k_device_ptrs=k_ptrs,
            v_device_ptrs=v_ptrs,
        )
    except Exception:
        for handle in reversed(begun):
            coordinator.end_attachment_load(handle)
        gpu_manager.destroy()
        raise

    return PrefixMaterialization(
        manager=gpu_manager,
        append_plan=append_plan,
        load_task=load_task,
        coordinator=coordinator,
        attachment_handles=tuple(begun),
    )


def _expand_device_ptrs_for_host_pages(
    *,
    gpu_manager: object,
    k_device_ptrs: torch.Tensor,
    v_device_ptrs: torch.Tensor | None,
    active_page_counts: torch.Tensor,
    host_page_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map smaller Host pages into byte ranges within GPU FA pages."""

    gpu_page_tokens = int(gpu_manager.config.page_size_tokens)
    host_page_tokens = int(host_page_tokens)
    if host_page_tokens <= 0 or gpu_page_tokens <= 0:
        raise ValueError("Host and GPU page sizes must be positive")
    if host_page_tokens == gpu_page_tokens:
        return k_device_ptrs, v_device_ptrs
    if host_page_tokens > gpu_page_tokens or gpu_page_tokens % host_page_tokens:
        raise ValueError(
            "GPU page size must be a multiple of Host page size for prefix "
            f"materialization, got gpu={gpu_page_tokens}, "
            f"host={host_page_tokens}"
        )

    pages_per_gpu_page = gpu_page_tokens // host_page_tokens
    k_page_bytes = _host_page_bytes(
        gpu_manager=gpu_manager,
        host_page_tokens=host_page_tokens,
        is_value=False,
    )
    expanded_k = _expand_pointer_tensor_for_host_pages(
        k_device_ptrs,
        active_page_counts=active_page_counts,
        host_page_bytes=k_page_bytes,
        host_pages_per_gpu_page=pages_per_gpu_page,
    )

    expanded_v = None
    if v_device_ptrs is not None:
        v_page_bytes = _host_page_bytes(
            gpu_manager=gpu_manager,
            host_page_tokens=host_page_tokens,
            is_value=True,
        )
        expanded_v = _expand_pointer_tensor_for_host_pages(
            v_device_ptrs,
            active_page_counts=active_page_counts,
            host_page_bytes=v_page_bytes,
            host_pages_per_gpu_page=pages_per_gpu_page,
        )
    return expanded_k, expanded_v


def _host_page_bytes(
    *,
    gpu_manager: object,
    host_page_tokens: int,
    is_value: bool,
) -> int:
    config = gpu_manager.config
    if is_value:
        heads = int(config.num_v_heads)
        head_dim = int(config.v_head_dim)
    else:
        heads = int(config.num_k_heads)
        head_dim = int(config.k_head_dim)
    element_size = torch.empty((), dtype=config.kv_dtype).element_size()
    return int(host_page_tokens) * heads * head_dim * element_size


def _expand_pointer_tensor_for_host_pages(
    pointer_tensor: torch.Tensor,
    *,
    active_page_counts: torch.Tensor,
    host_page_bytes: int,
    host_pages_per_gpu_page: int,
) -> torch.Tensor:
    max_host_pages = int(active_page_counts.max().item())
    if max_host_pages == 0:
        return pointer_tensor[:, :, :0].contiguous()

    host_slots = torch.arange(max_host_pages, dtype=torch.long)
    gpu_slots = torch.div(
        host_slots,
        int(host_pages_per_gpu_page),
        rounding_mode="floor",
    )
    if int(gpu_slots[-1].item()) >= int(pointer_tensor.shape[2]):
        raise ValueError(
            "GPU page pointer tensor is too small for Host prefix pages: "
            f"host_pages={max_host_pages}, "
            f"host_pages_per_gpu_page={host_pages_per_gpu_page}, "
            f"gpu_pointer_pages={pointer_tensor.shape[2]}"
        )
    offsets = (
        torch.remainder(host_slots, int(host_pages_per_gpu_page)).to(
            dtype=torch.int64
        )
        * int(host_page_bytes)
    )
    expanded = pointer_tensor.index_select(2, gpu_slots).contiguous()
    return expanded + offsets.view(1, 1, -1)
