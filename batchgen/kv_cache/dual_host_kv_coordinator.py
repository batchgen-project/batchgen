"""Coordinator for dual host KV caches (MLA + Indexer) used by DSA models.

DSA models (DeepSeek-V3.2, GLM-5) require two host paged KV caches:
  - Primary: MLA compressed KV (dim=576)
  - Auxiliary: Indexer KV for token scoring (dim=128)

This coordinator wraps two MLAHostPagedKVWorkerView instances and delegates
all lifecycle operations to both, keeping them synchronized. It duck-types
the worker view API so callers can use it as a drop-in replacement.

For operations that require different data per pool (offload, load-to-device),
callers access .primary / .auxiliary directly.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

from batchgen.models.engine_loader import core_engine as bg_lib

logger = logging.getLogger(__name__)


def _try_set_logger_name(config, name: str) -> bool:
	"""Try to set a custom logger name on a C++ HostPagedKVConfig.

	Returns True if the field exists and was set, False otherwise.
	"""
	try:
		config.logger_name = name
		return True
	except AttributeError:
		return False


def _build_host_config_from_profile(profile, shm_name: str, num_pages: int) -> Any:
	"""Build a bg_lib.HostPagedKVConfig from a _HostKVModelProfile."""
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
			primary_profile, HOST_KV_SHM_NAME, num_pages,
		)
		aux_config = _build_host_config_from_profile(
			aux_profile, HOST_KV_AUX_SHM_NAME, num_pages,
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
			if "logger" in str(e) and "already exists" in str(e):
				logger.warning(
					"Cannot create auxiliary KV view (duplicate C++ logger). "
					"Using primary-only mode. DSA indexer KV will "
					"be unavailable. Error: %s", e
				)
				coordinator = cls(primary_view, None)
				logger.info(
					"DualHostKVCoordinator created (primary-only): %d pages, "
					"primary k_dim=%d",
					num_pages, primary_profile.k_head_dim,
				)
				return coordinator
			raise

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
			primary_profile, HOST_KV_SHM_NAME, num_pages,
		)
		aux_config = _build_host_config_from_profile(
			aux_profile, HOST_KV_AUX_SHM_NAME, num_pages,
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
			if "logger" in str(e) and "already exists" in str(e):
				logger.warning(
					"Cannot create auxiliary KV manager (duplicate C++ logger). "
					"Using primary-only mode. Error: %s", e
				)
				return primary_mgr, None
			raise

		logger.info(
			"DualHostKVCoordinator managers created: %d pages, "
			"primary k_dim=%d, aux k_dim=%d",
			num_pages, primary_profile.k_head_dim, aux_profile.k_head_dim,
		)
		return primary_mgr, aux_mgr

	# -- Lifecycle --

	def initialize(self, **kwargs) -> None:
		self.primary.initialize(**kwargs)
		if self.auxiliary is not None:
			self.auxiliary.initialize(**kwargs)

	# -- Sequence management (mirrored) --

	def register_sequences(self, sequence_ids) -> None:
		self.primary.register_sequences(sequence_ids)
		if self.auxiliary is not None:
			self.auxiliary.register_sequences(sequence_ids)

	def allocate_pages_for_sequences(self, seq_token_pairs) -> None:
		self.primary.allocate_pages_for_sequences(seq_token_pairs)
		if self.auxiliary is not None:
			self.auxiliary.allocate_pages_for_sequences(seq_token_pairs)

	def release_sequence_pages(self, sequence_ids) -> None:
		self.primary.release_sequence_pages(sequence_ids)
		if self.auxiliary is not None:
			self.auxiliary.release_sequence_pages(sequence_ids)

	# -- Query: report the capacity-tighter of {primary, auxiliary} --

	def get_stats(self):
		primary_stats = self.primary.get_stats()
		if self.auxiliary is None:
			return primary_stats
		aux_stats = self.auxiliary.get_stats()
		# Both views share num_pages (see _compute_dual_page_count), but they
		# can drift if mirroring ever fails partway (e.g. aux register raises
		# after primary succeeds). Report whichever side has fewer free pages
		# so the watermark fires on the tighter bound, not the optimistic one.
		if aux_stats.num_free_pages < primary_stats.num_free_pages:
			return aux_stats
		return primary_stats

	# -- Migration (primary only, aux rebuilt during prefill) --

	def async_load_layer_paged_kv_to_device(self, **kwargs):
		return self.primary.async_load_layer_paged_kv_to_device(**kwargs)

	def async_offload_layer_kv_to_host(self, **kwargs):
		return self.primary.async_offload_layer_kv_to_host(**kwargs)
