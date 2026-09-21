"""Materialize attached Host prefix pages into temporary GPU paged KV."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


class _AsyncTask(Protocol):
    def done(self) -> bool: ...

    def wait(self) -> None: ...


@dataclass
class PrefixMaterialization:
    manager: object
    append_plan: object
    load_task: _AsyncTask | None
    coordinator: object
    attachment_handles: tuple[int, ...]
    collect_metrics: bool = False
    sequence_count: int = 0
    cached_tokens: int = 0
    host_pages: int = 0
    host_bytes: int = 0
    _load_launched_at: float | None = None
    load_done_before_wait: bool | None = None
    load_wait_s: float = 0.0
    load_launch_to_wait_return_s: float = 0.0
    _load_failed: bool = False
    _metrics_emitted: bool = False
    _load_complete: bool = False
    _closed: bool = False
    # Host pages loaded per sequence, in ``append_plan`` order.
    host_page_ids: tuple[tuple[int, ...], ...] = ()
    host_page_tokens: int = 0
    integrity_ledger: object | None = None
    integrity_counters: object | None = None

    def wait_for_layer(self, layer_idx: int) -> None:
        if self._closed:
            raise RuntimeError("prefix materialization is already closed")
        del layer_idx
        self._wait_for_load()

    def _wait_for_load(self) -> None:
        if self.load_task is not None and not self._load_complete:
            if not self.collect_metrics:
                self.load_task.wait()
                self._load_complete = True
                return
            self.load_done_before_wait = bool(self.load_task.done())
            wait_started_at = time.perf_counter()
            try:
                self.load_task.wait()
            except Exception:
                self._load_failed = True
                raise
            else:
                self._load_complete = True
            finally:
                wait_returned_at = time.perf_counter()
                self.load_wait_s = wait_returned_at - wait_started_at
                if self._load_launched_at is not None:
                    self.load_launch_to_wait_return_s = (
                        wait_returned_at - self._load_launched_at
                    )

    def take_metrics_fields(
        self,
    ) -> dict[str, int | float | bool | None | str] | None:
        """Return the one-shot evidence record after close is attempted.

        ``launch_to_wait_return_s`` is an observed upper bound on GPU-complete
        H2D latency.  If ``done_before_wait`` is true, completion occurred at
        an unknown earlier point within that interval.
        """

        if not self.collect_metrics or self._metrics_emitted:
            return None
        self._metrics_emitted = True
        if self.load_task is None:
            status = "no_load"
        elif self._load_failed:
            status = "failed"
        elif self._load_complete:
            status = "complete"
        else:
            status = "pending"
        return {
            "sequences": int(self.sequence_count),
            "cached_tokens": int(self.cached_tokens),
            "host_pages": int(self.host_pages),
            "host_bytes": int(self.host_bytes),
            "load_status": status,
            "done_before_wait": self.load_done_before_wait,
            "wait_s": float(self.load_wait_s),
            "launch_to_wait_return_s": float(
                self.load_launch_to_wait_return_s
            ),
        }

    def close(
        self, *, empty_cuda_cache: bool = False, verify_integrity: bool = False
    ) -> None:
        """Release the load protection and the temporary GPU pages.

        Only the successful prefill path passes ``verify_integrity``; cleanup
        after an exception closes without it.
        """
        if self._closed:
            return
        try:
            self._wait_for_load()
            if verify_integrity and self.integrity_ledger is not None:
                self._verify_gpu_prefix_pages()
        finally:
            for handle in reversed(self.attachment_handles):
                self.coordinator.end_attachment_load(handle)
            self.manager.destroy(empty_cuda_cache=empty_cuda_cache)
            self._closed = True

    def _verify_gpu_prefix_pages(self) -> None:
        """H2: the GPU prefix bytes attention read equal their ledgered Host pages.

        Runs after every layer appended its suffix, so it also proves no suffix
        write landed in the prefix. A raw full hit recomputes its last prompt
        token into the last loaded page, so only pages wholly below the
        computed-prefix boundary are compared.
        """
        from batchgen.prefix_reuse.integrity import gpu_chunk_hashes

        plan = self.append_plan
        page_table = plan.page_table.cpu()
        per_gpu_page = (
            int(self.manager.config.page_size_tokens) // self.host_page_tokens
        )
        page_ids: list[int] = []
        chunk_ids: list[int] = []
        for slot, prefix, pages in zip(
            plan.slot_values, plan.prefix_values, self.host_page_ids
        ):
            count = int(prefix) // self.host_page_tokens
            page_ids.extend(pages[:count])
            chunk_ids.extend(
                _host_page_chunks(page_table[slot].tolist(), count, per_gpu_page)
            )
        if not page_ids:
            return
        k_cache, v_cache = self.manager.get_kv_tensors()
        k_hashes, v_hashes = gpu_chunk_hashes(
            k_cache, v_cache, chunk_ids, self.host_page_tokens
        )
        self.integrity_counters.pages_verified["H2-materialization"] += (
            self.integrity_ledger.verify(
                page_ids, k_hashes, v_hashes, context="H2-materialization"
            )
        )


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
    collect_metrics: bool = False,
    host_page_bytes_all_layers: int = 0,
    integrity_ledger: object | None = None,
    integrity_counters: object | None = None,
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
    if collect_metrics and int(host_page_bytes_all_layers) <= 0:
        raise ValueError(
            "host_page_bytes_all_layers must be positive when metrics are enabled"
        )

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
            collect_metrics=bool(collect_metrics),
            sequence_count=count,
        )

    host_rows: list[list[int]] = []
    loaded_pages: list[tuple[int, ...]] = []
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
        loaded_pages.append(tuple(row))
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
        load_launched_at = time.perf_counter() if collect_metrics else None
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
        collect_metrics=bool(collect_metrics),
        sequence_count=count,
        cached_tokens=sum(prefix_lens) if collect_metrics else 0,
        host_pages=sum(page_counts) if collect_metrics else 0,
        host_bytes=(
            sum(page_counts) * int(host_page_bytes_all_layers)
            if collect_metrics
            else 0
        ),
        _load_launched_at=load_launched_at,
        host_page_ids=tuple(loaded_pages),
        host_page_tokens=host_page_tokens,
        integrity_ledger=integrity_ledger,
        integrity_counters=integrity_counters,
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


def _host_page_chunks(
    gpu_page_row: Sequence[int], num_host_pages: int, host_pages_per_gpu_page: int
) -> list[int]:
    """``gpu_chunk_hashes`` ids of one sequence's first Host slots.

    Host slot ``s`` fills sub-page ``s % r`` of GPU page ``row[s // r]``: the
    byte range ``_expand_pointer_tensor_for_host_pages`` targets.
    """
    r = int(host_pages_per_gpu_page)
    return [int(gpu_page_row[s // r]) * r + s % r for s in range(num_host_pages)]


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
