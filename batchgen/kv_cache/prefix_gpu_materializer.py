"""Prefill-scoped prefix KV materialization into GPU paged KV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import (
	GPUPagedKVCacheManager,
	GPUPagedKVSuffixAppendPlan,
)


def _to_int_list(values: Sequence[int] | torch.Tensor, name: str) -> list[int]:
	tensor = torch.as_tensor(values, dtype=torch.long, device="cpu")
	if tensor.dim() != 1:
		raise ValueError(f"{name} must be 1-D, got shape={tuple(tensor.shape)}")
	return [int(value) for value in tensor.tolist()]


def _unique_in_order(values: Sequence[int]) -> list[int]:
	return list(dict.fromkeys(int(value) for value in values))


@dataclass
class PrefillPrefixGpuMaterialization:
	"""Temporary GPU materialization for one prefix-reuse prefill microbatch."""

	manager: GPUPagedKVCacheManager
	sequence_ids: list[int]
	append_plan: GPUPagedKVSuffixAppendPlan
	load_task: Optional[object]
	host_pages_loaded: list[int]
	gpu_pages_loaded: list[int]
	_destroy_manager_on_cleanup: bool = False
	_load_waited: bool = False
	_cleaned: bool = False

	def wait_for_load(self) -> None:
		if self._load_waited:
			return
		if self.load_task is not None:
			self.load_task.wait()
		self._load_waited = True

	def cleanup(self) -> None:
		if self._cleaned:
			return
		self.wait_for_load()
		if self.sequence_ids:
			self.manager.free_pages_for_sequences(self.sequence_ids)
		if self._destroy_manager_on_cleanup:
			self.manager.destroy(empty_cuda_cache=False)
		self._cleaned = True


def materialize_prefill_prefix_pages(
	*,
	manager: GPUPagedKVCacheManager,
	worker_view: object,
	sequence_ids: Sequence[int],
	full_lengths: Sequence[int] | torch.Tensor,
	prefix_lens: Sequence[int] | torch.Tensor,
	suffix_lens: Sequence[int] | torch.Tensor,
	shared_prefix_pages: Sequence[Sequence[int]],
	destroy_manager_on_cleanup: bool = False,
	require_page_aligned_prefix: bool = True,
) -> PrefillPrefixGpuMaterialization:
	"""Allocate temporary GPU pages and async-load cached host prefix pages.

	The host KV cache remains the source of truth. The allocated GPU pages are
	only a prefill-scoped materialized view used by paged extend attention.
	"""

	seq_ids = _to_int_list(sequence_ids, "sequence_ids")
	full = _to_int_list(full_lengths, "full_lengths")
	prefix = _to_int_list(prefix_lens, "prefix_lens")
	suffix = _to_int_list(suffix_lens, "suffix_lens")
	if not (
		len(seq_ids) == len(full)
		and len(seq_ids) == len(prefix)
		and len(seq_ids) == len(suffix)
		and len(seq_ids) == len(shared_prefix_pages)
	):
		raise ValueError(
			"materialize_prefill_prefix_pages: sequence_ids, lengths, and "
			"shared_prefix_pages must have the same length"
		)
	if not seq_ids:
		raise ValueError("materialize_prefill_prefix_pages requires sequences")
	if any(prefix_len <= 0 for prefix_len in prefix):
		raise ValueError(
			"materialize_prefill_prefix_pages only supports prefix-hit sequences"
		)
	for idx, (prefix_len, suffix_len, full_len) in enumerate(zip(prefix, suffix, full)):
		if prefix_len + suffix_len != full_len:
			raise ValueError(
				"materialize_prefill_prefix_pages length mismatch at "
				f"idx={idx}: prefix={prefix_len}, suffix={suffix_len}, "
				f"full={full_len}"
			)

	normalized_shared = [
		[int(page) for page in pages] for pages in shared_prefix_pages
	]
	page_size = int(manager.config.page_size_tokens)
	for idx, (prefix_len, pages) in enumerate(zip(prefix, normalized_shared)):
		if not pages:
			raise ValueError(
				"materialize_prefill_prefix_pages: prefix-hit sequence has "
				f"no shared host pages at idx={idx}"
			)
		if len(pages) * page_size < prefix_len:
			raise ValueError(
				"materialize_prefill_prefix_pages: shared host pages do not "
				f"cover prefix tokens at idx={idx}: pages={len(pages)}, "
				f"page_size={page_size}, prefix_len={prefix_len}"
			)
		if require_page_aligned_prefix and len(pages) * page_size != prefix_len:
			raise ValueError(
				"materialize_prefill_prefix_pages requires page-aligned "
				f"prefix reuse for the GPU paged path at idx={idx}: "
				f"pages={len(pages)}, page_size={page_size}, "
				f"prefix_len={prefix_len}"
			)
	manager.allocate_pages_for_sequences_with_prefix(
		seq_ids,
		full,
		normalized_shared,
	)
	append_plan = manager.prepare_prefill_suffix_append(
		sequence_ids=seq_ids,
		prefix_lens=prefix,
		suffix_lens=suffix,
		rebuild_page_table=True,
	)

	host_pages_to_load = _unique_in_order(
		page for pages in normalized_shared for page in pages
	)
	load_task = None
	gpu_pages_to_load: list[int] = []
	if host_pages_to_load:
		gpu_pages_to_load = [
			int(manager._shared_prefix_gpu_pages[host_page])
			for host_page in host_pages_to_load
		]
		k_ptrs, v_ptrs = manager.get_page_pointer_matrix(gpu_pages_to_load)
		host_page_tensor = torch.tensor(
			host_pages_to_load, dtype=torch.int32, device="cpu"
		)
		load_task = worker_view.async_load_prefix_pages_to_device(
			host_page_tensor,
			k_ptrs,
			v_ptrs,
		)

	return PrefillPrefixGpuMaterialization(
		manager=manager,
		sequence_ids=seq_ids,
		append_plan=append_plan,
		load_task=load_task,
		host_pages_loaded=host_pages_to_load,
		gpu_pages_loaded=gpu_pages_to_load,
		_destroy_manager_on_cleanup=destroy_manager_on_cleanup,
	)
