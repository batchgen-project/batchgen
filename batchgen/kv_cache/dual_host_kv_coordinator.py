"""Coordinator for dual host KV caches (MLA + Indexer) used by DSA models.

DSA models (DeepSeek-V3.2, GLM-5) require two host paged KV caches:
  - Primary: MLA compressed KV (dim=576)
  - Auxiliary: Indexer KV for token scoring (dim=128)

This coordinator wraps two MLAHostPagedKVWorkerView instances and delegates
all lifecycle operations to both, keeping them synchronized. It duck-types
the worker view API so callers can use it as a drop-in replacement.

For operations that require different data per pool (offload, load-to-device),
callers must use explicit dual APIs. Primary-only operations are invalid for
DSA because they can leave the indexer KV stale while the sequence remains
marked GPU/host-resident.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from batchgen.models.engine_loader import core_engine as bg_lib
from batchgen.prefix_reuse.dual_prefix_cache import (
	assert_matching_prefix_allocation_results,
	assert_matching_prefix_eviction_results,
	assert_matching_prefix_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class DualAsyncKVTask:
	"""Composite async task that owns both primary and auxiliary KV work."""

	primary_task: Any
	aux_task: Any
	tensors: Any = None

	def __post_init__(self) -> None:
		if self.primary_task is None or self.aux_task is None:
			raise RuntimeError(
				"DualAsyncKVTask requires both primary and auxiliary tasks"
			)

	def wait(self) -> None:
		errors = []
		for name, task in (
			("primary", self.primary_task),
			("auxiliary", self.aux_task),
		):
			try:
				task.wait()
			except Exception as e:
				errors.append((name, e))
		if errors:
			if len(errors) == 1:
				raise errors[0][1]
			names = ", ".join(name for name, _ in errors)
			raise RuntimeError(
				f"DualAsyncKVTask wait failed for both KV loads: {names}"
			) from errors[0][1]


def _try_set_logger_name(config, name: str) -> bool:
	"""Try to set a custom logger name on a C++ HostPagedKVConfig.

	Returns True if the field exists and was set, False otherwise.
	"""
	try:
		config.logger_name = name
		return True
	except AttributeError:
		return False


def _build_host_config_from_profile(
	profile,
	shm_name: str,
	num_pages: int,
) -> Any:
	"""Build a HostPagedKVConfig from a _HostKVModelProfile."""
	from batchgen.kv_cache.host_kv_mananger_config import _dtype_size_bytes

	config = bg_lib.HostPagedKVConfig()
	config.shm_name = shm_name
	config.num_layers = profile.num_layers
	config.num_pages = num_pages
	config.page_size_tokens = profile.page_size
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


def _compute_dual_page_count(
	model_name: str, host_kv_cache_size: int,
) -> Tuple[Any, Any, int]:
	"""Compute shared page count from combined budget.

	Returns:
		(primary_profile, aux_profile, num_pages)
	"""
	from batchgen.kv_cache.host_kv_mananger_config import (
		_resolve_profile, _resolve_indexer_profile,
	)
	primary_profile = _resolve_profile(model_name)
	aux_profile = _resolve_indexer_profile(model_name)
	if primary_profile.page_size != aux_profile.page_size:
		raise ValueError(
			f"DSA primary/aux host KV page size mismatch: "
			f"primary={primary_profile.page_size}, aux={aux_profile.page_size}"
		)

	primary_layer_bytes = primary_profile.bytes_per_page() * primary_profile.num_layers
	aux_layer_bytes = aux_profile.bytes_per_page() * aux_profile.num_layers
	combined = primary_layer_bytes + aux_layer_bytes

	num_pages = int(host_kv_cache_size) // combined
	if num_pages <= 0:
		raise ValueError(
			f"host_kv_cache_size ({host_kv_cache_size}) too small for dual KV cache "
			f"(combined bytes per page = {combined})"
		)
	return primary_profile, aux_profile, num_pages


class DualHostKVCoordinator:
	"""Synchronizes a primary and auxiliary host paged KV worker view.

	Both views track identical sequences with identical page allocations.
	All lifecycle operations are mirrored. Return values come from the
	primary view for backward compatibility.

	Attributes:
		primary: MLA host KV worker view (dim=576 for GLM-5)
		auxiliary: Indexer host KV worker view (dim=128 for GLM-5)
	"""

	def __init__(self, primary, auxiliary) -> None:
		if auxiliary is None:
			raise RuntimeError("DualHostKVCoordinator requires auxiliary host KV for DSA")
		self.primary = primary
		self.auxiliary = auxiliary

	@classmethod
	def from_budget(
		cls,
		model_name: str,
		host_kv_cache_size: int,
		core_engine_module,
		enable_memfd: bool = False,
		memfd_creator_pid: int = -1,
		memfd_fd: int = -1,
		aux_memfd_fd: int = -1,
	) -> Optional["DualHostKVCoordinator"]:
		"""Factory: split budget proportionally, create both worker views.

		Returns DualHostKVCoordinator or None if not a DSA model.
		"""
		from batchgen.kv_cache.host_kv_mananger_config import (
			is_dsa_model, HOST_KV_SHM_NAME, HOST_KV_AUX_SHM_NAME,
		)
		if not is_dsa_model(model_name):
			return None

		primary_profile, aux_profile, num_pages = _compute_dual_page_count(
			model_name, host_kv_cache_size,
		)

		primary_config = _build_host_config_from_profile(
			primary_profile,
			HOST_KV_SHM_NAME,
			num_pages,
		)
		aux_config = _build_host_config_from_profile(
			aux_profile,
			HOST_KV_AUX_SHM_NAME,
			num_pages,
		)

		# Set distinct logger names to avoid C++ logger name collision
		_try_set_logger_name(primary_config, "HostPagedKVWorkerView")
		_try_set_logger_name(aux_config, "HostPagedKVWorkerView_aux")

		if enable_memfd:
			primary_config.enable_memfd = True
			primary_config.memfd_creator_pid = memfd_creator_pid
			primary_config.memfd_fd = memfd_fd
			aux_config.enable_memfd = True
			aux_config.memfd_creator_pid = memfd_creator_pid
			aux_config.memfd_fd = aux_memfd_fd

		primary_view = core_engine_module.MLAHostPagedKVWorkerView(primary_config)
		try:
			aux_view = core_engine_module.MLAHostPagedKVWorkerView(aux_config)
		except RuntimeError as e:
			raise RuntimeError(
				"Cannot create auxiliary KV worker view for DSA model; "
				"primary-only DSA mode is invalid"
			) from e

		coordinator = cls(primary_view, aux_view)
		logger.info(
			"DualHostKVCoordinator created: %d pages, "
			"primary k_dim=%d, aux k_dim=%d",
			num_pages, primary_profile.k_head_dim, aux_profile.k_head_dim,
		)
		return coordinator

	@classmethod
	def create_managers(
		cls,
		model_name: str,
		host_kv_cache_size: int,
		enable_memfd: bool = False,
	) -> Optional[Tuple[Any, Any]]:
		"""Server-side factory: create and initialize both host KV managers.

		Returns (primary_manager, aux_manager) or None if not DSA.
		"""
		from batchgen.kv_cache.host_kv_mananger_config import (
			is_dsa_model, HOST_KV_SHM_NAME, HOST_KV_AUX_SHM_NAME,
		)
		if not is_dsa_model(model_name):
			return None

		primary_profile, aux_profile, num_pages = _compute_dual_page_count(
			model_name, host_kv_cache_size,
		)

		primary_config = _build_host_config_from_profile(
			primary_profile,
			HOST_KV_SHM_NAME,
			num_pages,
		)
		aux_config = _build_host_config_from_profile(
			aux_profile,
			HOST_KV_AUX_SHM_NAME,
			num_pages,
		)

		# Set distinct logger names to avoid C++ logger name collision
		_try_set_logger_name(primary_config, "HostPagedKVManager")
		_try_set_logger_name(aux_config, "HostPagedKVManager_aux")

		if enable_memfd:
			primary_config.enable_memfd = True
			aux_config.enable_memfd = True

		primary_mgr = bg_lib.MLAHostPagedKVManager(primary_config)
		primary_mgr.initialize(True)
		try:
			aux_mgr = bg_lib.MLAHostPagedKVManager(aux_config)
			aux_mgr.initialize(True)
		except RuntimeError as e:
			raise RuntimeError(
				"Cannot create auxiliary KV manager for DSA model; "
				"primary-only DSA mode is invalid"
			) from e

		logger.info(
			"DualHostKVCoordinator managers created: %d pages, "
			"primary k_dim=%d, aux k_dim=%d",
			num_pages, primary_profile.k_head_dim, aux_profile.k_head_dim,
		)
		return primary_mgr, aux_mgr

	# -- Lifecycle --

	def initialize(self, **kwargs) -> None:
		self.primary.initialize(**kwargs)
		self.require_auxiliary("initialize").initialize(**kwargs)

	def require_auxiliary(self, context: str):
		if self.auxiliary is None:
			raise RuntimeError(f"{context}: auxiliary host KV is required for DSA")
		return self.auxiliary

	# -- Sequence management (mirrored) --

	def register_sequences(self, sequence_ids) -> None:
		self.primary.register_sequences(sequence_ids)
		try:
			self.require_auxiliary("register_sequences").register_sequences(sequence_ids)
		except Exception:
			try:
				self.primary.unregister_sequences(sequence_ids)
			except Exception:
				logger.exception(
					"Failed to rollback primary host KV registration for %s",
					list(sequence_ids)[:10],
				)
			raise

	def allocate_pages_for_sequences(self, seq_token_pairs) -> None:
		seq_token_pairs = list(seq_token_pairs)
		sequence_ids = [seq_id for seq_id, _ in seq_token_pairs]
		self.primary.allocate_pages_for_sequences(seq_token_pairs)
		try:
			self.require_auxiliary("allocate_pages_for_sequences").allocate_pages_for_sequences(seq_token_pairs)
		except Exception:
			try:
				self.primary.release_sequence_pages(sequence_ids)
			except Exception:
				logger.exception(
					"Failed to rollback primary host KV allocation for %s",
					sequence_ids[:10],
				)
			raise

	def allocate_pages_for_sequences_with_prefix(self, prefix_requests):
		"""Allocate primary/aux host pages with identical prefix reuse plans."""
		prefix_requests = list(prefix_requests)
		sequence_ids = [int(request[0]) for request in prefix_requests]
		primary_results = self.primary.allocate_pages_for_sequences_with_prefix(
			prefix_requests
		)
		try:
			auxiliary_results = self.require_auxiliary(
				"allocate_pages_for_sequences_with_prefix"
			).allocate_pages_for_sequences_with_prefix(prefix_requests)
		except Exception:
			self._release_primary_prefix_allocation(sequence_ids)
			raise
		try:
			assert_matching_prefix_allocation_results(
				primary_results,
				auxiliary_results,
				"allocate_pages_for_sequences_with_prefix",
			)
		except Exception:
			self._release_dual_prefix_allocation(sequence_ids)
			raise
		return primary_results

	def estimate_pages_for_sequences_with_prefix(self, prefix_requests):
		"""Estimate dual prefix allocations without mutating either view."""
		prefix_requests = list(prefix_requests)
		primary_results = self.primary.estimate_pages_for_sequences_with_prefix(
			prefix_requests
		)
		auxiliary_results = self.require_auxiliary(
			"estimate_pages_for_sequences_with_prefix"
		).estimate_pages_for_sequences_with_prefix(prefix_requests)
		assert_matching_prefix_allocation_results(
			primary_results,
			auxiliary_results,
			"estimate_pages_for_sequences_with_prefix",
		)
		return primary_results

	def commit_sequence_prefix_pages(
		self,
		sequence_id: int,
		token_ids,
		namespace_hash: int = 0,
	):
		"""Commit one logical prefix page chain to both host prefix caches."""
		primary_inserted = self.primary.commit_sequence_prefix_pages(
			sequence_id,
			token_ids,
			namespace_hash,
		)
		auxiliary_inserted = self.require_auxiliary(
			"commit_sequence_prefix_pages"
		).commit_sequence_prefix_pages(
			sequence_id,
			token_ids,
			namespace_hash,
		)
		if int(primary_inserted) != int(auxiliary_inserted):
			raise RuntimeError(
				"commit_sequence_prefix_pages: primary/auxiliary inserted-page "
				f"mismatch for seq {sequence_id}: primary={primary_inserted}, "
				f"auxiliary={auxiliary_inserted}"
			)
		return primary_inserted

	def grow_pages_for_sequences(self, seq_page_pairs) -> None:
		seq_page_pairs = list(seq_page_pairs)
		needed = sum(int(pages) for _, pages in seq_page_pairs)
		primary_stats = self.primary.get_stats()
		aux = self.require_auxiliary("grow_pages_for_sequences")
		aux_stats = aux.get_stats()
		if needed > primary_stats.num_free_pages or needed > aux_stats.num_free_pages:
			raise RuntimeError(
				"grow_pages_for_sequences: insufficient mirrored host KV free pages: "
				f"need={needed}, primary_free={primary_stats.num_free_pages}, "
				f"aux_free={aux_stats.num_free_pages}"
			)
		self.primary.grow_pages_for_sequences(seq_page_pairs)
		aux.grow_pages_for_sequences(seq_page_pairs)

	def release_sequence_pages(self, sequence_ids) -> None:
		self.primary.release_sequence_pages(sequence_ids)
		self.require_auxiliary("release_sequence_pages").release_sequence_pages(sequence_ids)

	def unregister_sequences(self, sequence_ids) -> None:
		self.primary.unregister_sequences(sequence_ids)
		self.require_auxiliary("unregister_sequences").unregister_sequences(sequence_ids)

	# -- Query: report the capacity-tighter of {primary, auxiliary} --

	def get_stats(self):
		primary_stats = self.primary.get_stats()
		aux_stats = self.require_auxiliary("get_stats").get_stats()
		if primary_stats.num_total_pages != aux_stats.num_total_pages:
			raise RuntimeError(
				"primary/auxiliary host KV total-page mismatch: "
				f"primary={primary_stats.num_total_pages}, aux={aux_stats.num_total_pages}"
			)
		# Both views share num_pages (see _compute_dual_page_count), but they
		# can drift if mirroring ever fails partway (e.g. aux register raises
		# after primary succeeds). Report whichever side has fewer free pages
		# so the watermark fires on the tighter bound, not the optimistic one.
		if aux_stats.num_free_pages < primary_stats.num_free_pages:
			return aux_stats
		return primary_stats

	def shared_prefix_pages(self, sequence_id: int):
		primary_pages = list(self.primary.shared_prefix_pages(sequence_id))
		auxiliary_pages = list(
			self.require_auxiliary("shared_prefix_pages").shared_prefix_pages(
				sequence_id
			)
		)
		if len(primary_pages) != len(auxiliary_pages):
			raise RuntimeError(
				"shared_prefix_pages: primary/auxiliary shared-page count "
				f"mismatch for seq {sequence_id}: primary={len(primary_pages)}, "
				f"auxiliary={len(auxiliary_pages)}"
			)
		return primary_pages

	def shared_prefix_tokens(self, sequence_id: int) -> int:
		primary_tokens = int(self.primary.shared_prefix_tokens(sequence_id))
		auxiliary_tokens = int(
			self.require_auxiliary("shared_prefix_tokens").shared_prefix_tokens(
				sequence_id
			)
		)
		if primary_tokens != auxiliary_tokens:
			raise RuntimeError(
				"shared_prefix_tokens: primary/auxiliary token mismatch for "
				f"seq {sequence_id}: primary={primary_tokens}, "
				f"auxiliary={auxiliary_tokens}"
			)
		return primary_tokens

	def get_prefix_cache_stats(self):
		primary_stats = self.primary.get_prefix_cache_stats()
		auxiliary_stats = self.require_auxiliary(
			"get_prefix_cache_stats"
		).get_prefix_cache_stats()
		assert_matching_prefix_stats(
			primary_stats,
			auxiliary_stats,
			"get_prefix_cache_stats",
		)
		return primary_stats

	def prefix_cache_debug_entries(self, limit: int = 0, cold_first: bool = True):
		primary_entries = self.primary.prefix_cache_debug_entries(
			limit,
			cold_first,
		)
		auxiliary_entries = self.require_auxiliary(
			"prefix_cache_debug_entries"
		).prefix_cache_debug_entries(
			limit,
			cold_first,
		)
		if len(primary_entries) != len(auxiliary_entries):
			raise RuntimeError(
				"prefix_cache_debug_entries: primary/auxiliary entry-count "
				f"mismatch: primary={len(primary_entries)}, "
				f"auxiliary={len(auxiliary_entries)}"
			)
		return primary_entries

	def clear_prefix_cache(self) -> None:
		self.primary.clear_prefix_cache()
		self.require_auxiliary("clear_prefix_cache").clear_prefix_cache()

	def evict_prefix_cache_until_free(
		self,
		target_free_pages: int,
		protected_pages=None,
		max_entries_to_scan: int = 0,
	):
		if max_entries_to_scan:
			raise RuntimeError(
				"evict_prefix_cache_until_free: max_entries_to_scan is not "
				"supported by the underlying host prefix cache binding"
			)
		primary_result = self.primary.evict_prefix_cache_until_free(
			target_free_pages,
			protected_pages=protected_pages,
		)
		auxiliary_result = self.require_auxiliary(
			"evict_prefix_cache_until_free"
		).evict_prefix_cache_until_free(
			target_free_pages,
			protected_pages=protected_pages,
		)
		assert_matching_prefix_eviction_results(
			primary_result,
			auxiliary_result,
			"evict_prefix_cache_until_free",
		)
		return primary_result

	# -- Migration / load helpers --

	def async_load_layer_paged_kv_to_device(self, **kwargs):
		raise RuntimeError(
			"DSA dual host KV load must use async_load_layer_paged_kv_to_device_dual(); "
			"primary-only loads can leave auxiliary/indexer KV stale"
		)

	def async_load_layer_paged_kv_to_device_dual(
		self,
		*,
		sequence_ids,
		primary_active_page_counts,
		primary_k_device_ptrs,
		primary_v_device_ptrs,
		aux_active_page_counts,
		aux_k_device_ptrs,
		aux_v_device_ptrs,
		tensors=None,
	) -> DualAsyncKVTask:
		aux = self.require_auxiliary("async_load_layer_paged_kv_to_device_dual")
		if primary_active_page_counts.tolist() != aux_active_page_counts.tolist():
			raise RuntimeError(
				"primary/auxiliary host load page-count mismatch: "
				f"primary={primary_active_page_counts.tolist()}, "
				f"auxiliary={aux_active_page_counts.tolist()}"
			)
		primary_task = self.primary.async_load_layer_paged_kv_to_device(
			sequence_ids=sequence_ids,
			active_page_counts=primary_active_page_counts,
			k_device_ptrs=primary_k_device_ptrs,
			v_device_ptrs=primary_v_device_ptrs,
		)
		try:
			aux_task = aux.async_load_layer_paged_kv_to_device(
				sequence_ids=sequence_ids,
				active_page_counts=aux_active_page_counts,
				k_device_ptrs=aux_k_device_ptrs,
				v_device_ptrs=aux_v_device_ptrs,
			)
		except Exception as aux_launch_error:
			try:
				primary_task.wait()
			except Exception as primary_wait_error:
				raise RuntimeError(
					"Auxiliary KV load launch failed after primary launch, "
					"and primary load did not drain cleanly"
				) from primary_wait_error
			raise aux_launch_error
		return DualAsyncKVTask(primary_task, aux_task, tensors=tensors)

	def async_offload_layer_kv_to_host(self, **kwargs):
		raise RuntimeError(
			"DSA dual host KV offload must explicitly offload both primary and auxiliary KV; "
			"primary-only offload is unsafe"
		)

	def _release_primary_prefix_allocation(self, sequence_ids: Sequence[int]) -> None:
		if not sequence_ids:
			return
		try:
			self.primary.release_sequence_pages(sequence_ids)
		except Exception:
			logger.exception(
				"Failed to rollback primary host prefix allocation for %s",
				list(sequence_ids)[:10],
			)

	def _release_dual_prefix_allocation(self, sequence_ids: Sequence[int]) -> None:
		if not sequence_ids:
			return
		for name, view in (
			("primary", self.primary),
			("auxiliary", self.require_auxiliary("_release_dual_prefix_allocation")),
		):
			try:
				view.release_sequence_pages(sequence_ids)
			except Exception:
				logger.exception(
					"Failed to rollback %s host prefix allocation for %s",
					name,
					list(sequence_ids)[:10],
				)
