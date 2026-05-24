"""GPU paged materialization helpers for prefix-reuse prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import torch


class _AsyncTask(Protocol):
    def wait(self) -> None: ...


@dataclass(frozen=True)
class PrefixMaterializationSequence:
    """Host prefix pages needed by one target GPU sequence."""

    sequence_id: int
    prefix_tokens: int
    suffix_tokens: int
    host_pages: Sequence[int | object]

    @property
    def full_tokens(self) -> int:
        return int(self.prefix_tokens) + int(self.suffix_tokens)


@dataclass
class SingleGroupPrefixMaterialization:
    """Single KV-group materialization view consumed by current adapters."""

    manager: object
    append_plan: object
    load_task: Optional[_AsyncTask] = None
    _loaded: bool = False

    def wait_for_layer(self, layer_idx: int) -> None:
        del layer_idx
        self.wait()

    def wait(self) -> None:
        if self._loaded:
            return
        if self.load_task is not None:
            self.load_task.wait()
        self._loaded = True


def materialize_single_group_prefix_pages(
    *,
    gpu_manager: object,
    host_worker_view: object,
    sequences: Sequence[PrefixMaterializationSequence],
    expected_host_region_id: int = 0,
) -> SingleGroupPrefixMaterialization:
    """Materialize Host prefix pages into target GPU paged KV slots.

    This helper is intentionally below the Host prefix-cache coordinator. The
    caller provides already attached/pinned Host page handles and target
    sequence ids; this function only allocates GPU pages, starts the page-id
    based Host->GPU copy, and prepares suffix append metadata.
    """

    if not sequences:
        raise ValueError("prefix materialization requires at least one sequence")

    sequence_ids = [int(item.sequence_id) for item in sequences]
    prefix_lens = [int(item.prefix_tokens) for item in sequences]
    suffix_lens = [int(item.suffix_tokens) for item in sequences]
    full_lens = [prefix + suffix for prefix, suffix in zip(prefix_lens, suffix_lens)]
    for seq_id, prefix_len, suffix_len, full_len in zip(
        sequence_ids, prefix_lens, suffix_lens, full_lens
    ):
        if prefix_len < 0 or suffix_len < 0:
            raise ValueError(
                "prefix/suffix lengths must be non-negative for sequence "
                f"{seq_id}: prefix={prefix_len}, suffix={suffix_len}"
            )
        if full_len <= 0:
            raise ValueError(
                f"full sequence length must be positive for sequence {seq_id}"
            )

    page_size = int(gpu_manager.config.page_size_tokens)
    prefix_page_counts = [
        int(math.ceil(prefix_len / page_size)) if prefix_len > 0 else 0
        for prefix_len in prefix_lens
    ]
    has_prefix_pages = any(count > 0 for count in prefix_page_counts)
    host_page_ids = None
    active_page_counts = None
    if has_prefix_pages:
        host_page_ids = _build_host_page_id_tensor(
            sequences,
            prefix_page_counts=prefix_page_counts,
            expected_host_region_id=expected_host_region_id,
        )
        active_page_counts = torch.tensor(prefix_page_counts, dtype=torch.int64)

    gpu_manager.allocate_pages_for_sequences(sequence_ids, full_lens)
    gpu_manager.rebuild_page_table(sequence_ids)
    k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()

    load_task = None
    if has_prefix_pages:
        load_task = host_worker_view.async_load_prefix_pages_to_device(
            host_page_ids=host_page_ids,
            active_page_counts=active_page_counts,
            k_device_ptrs=k_ptrs,
            v_device_ptrs=v_ptrs,
        )

    append_plan = gpu_manager.prepare_prefill_suffix_append(
        sequence_ids=sequence_ids,
        prefix_lens=prefix_lens,
        suffix_lens=suffix_lens,
        rebuild_page_table=False,
    )
    return SingleGroupPrefixMaterialization(
        manager=gpu_manager,
        append_plan=append_plan,
        load_task=load_task,
    )


def _build_host_page_id_tensor(
    sequences: Sequence[PrefixMaterializationSequence],
    *,
    prefix_page_counts: Sequence[int],
    expected_host_region_id: int,
) -> torch.Tensor:
    max_pages = max(int(count) for count in prefix_page_counts)
    rows: list[list[int]] = []
    for item, page_count in zip(sequences, prefix_page_counts):
        pages = [
            _host_page_id(handle, expected_host_region_id=expected_host_region_id)
            for handle in item.host_pages
        ]
        if len(pages) < int(page_count):
            raise ValueError(
                "host prefix page list is shorter than required for sequence "
                f"{item.sequence_id}: need {page_count}, got {len(pages)}"
            )
        row = pages[: int(page_count)]
        row.extend([0] * (max_pages - len(row)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def _host_page_id(handle: int | object, *, expected_host_region_id: int) -> int:
    if isinstance(handle, int):
        return int(handle)
    region_id = getattr(handle, "host_region_id", expected_host_region_id)
    if int(region_id) != int(expected_host_region_id):
        raise ValueError(
            "prefix materialization cannot load host page from region "
            f"{region_id}; expected region {expected_host_region_id}"
        )
    page_id = getattr(handle, "page_id", None)
    if page_id is None:
        raise TypeError("host page handle must be an int or expose page_id")
    return int(page_id)
