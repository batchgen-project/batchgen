"""Consistency helpers for dual prefix cache operations.

DSA models keep two host/GPU KV pools for the same logical sequence. Prefix
reuse must therefore be attached and committed to both pools as one logical
operation; a primary-only hit is unsafe because decode depends on mirrored
primary and auxiliary page tables.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


PREFIX_ALLOCATION_FIELDS = (
	"sequence_id",
	"shared_prefix_tokens",
	"private_start_token",
	"logical_page_count",
	"physical_pages_allocated",
	"full_hit",
	"miss_reason",
)

PREFIX_ALLOCATION_PAGE_LIST_FIELDS = (
	"shared_prefix_pages",
	"private_pages",
)

PREFIX_STATS_FIELDS = (
	"entries",
	"lookup_hits",
	"lookup_misses",
	"shared_pages_attached",
	"prefix_pin_increments",
	"prefix_pin_decrements",
	"host_pages_saved",
	"eviction_epoch",
	"eviction_runs",
	"evicted_entries",
	"evicted_prefix_pins",
	"evicted_pages_immediately_freed",
	"evicted_active_ref_entries",
	"eviction_protected_skips",
	"eviction_target_failures",
)

PREFIX_EVICTION_FIELDS = (
	"entries_removed",
	"pages_immediately_freed",
	"prefix_pins_released",
	"protected_entries_skipped",
	"active_ref_entries_removed",
	"reached_target",
)


def assert_matching_prefix_allocation_results(
	primary_results: Sequence[dict],
	auxiliary_results: Sequence[dict],
	context: str,
) -> None:
	"""Raise if primary/aux prefix allocation plans diverge."""
	if len(primary_results) != len(auxiliary_results):
		raise RuntimeError(
			f"{context}: primary/auxiliary prefix allocation result-count "
			f"mismatch: primary={len(primary_results)}, "
			f"auxiliary={len(auxiliary_results)}"
		)
	for idx, (primary, auxiliary) in enumerate(
		zip(primary_results, auxiliary_results)
	):
		for field in PREFIX_ALLOCATION_FIELDS:
			primary_value = _normalize_value(primary.get(field))
			auxiliary_value = _normalize_value(auxiliary.get(field))
			if primary_value != auxiliary_value:
				raise RuntimeError(
					f"{context}: primary/auxiliary prefix allocation mismatch "
					f"at result {idx} field {field}: "
					f"primary={primary_value}, auxiliary={auxiliary_value}"
				)
		for field in PREFIX_ALLOCATION_PAGE_LIST_FIELDS:
			primary_len = _page_list_length(primary.get(field))
			auxiliary_len = _page_list_length(auxiliary.get(field))
			if primary_len != auxiliary_len:
				raise RuntimeError(
					f"{context}: primary/auxiliary prefix allocation mismatch "
					f"at result {idx} field {field} length: "
					f"primary={primary_len}, auxiliary={auxiliary_len}"
				)


def assert_matching_prefix_stats(
	primary_stats: Any,
	auxiliary_stats: Any,
	context: str,
) -> None:
	"""Raise if primary/aux prefix-cache stats diverge."""
	_assert_matching_attributes(
		primary_stats,
		auxiliary_stats,
		PREFIX_STATS_FIELDS,
		context,
		"prefix stats",
	)


def assert_matching_prefix_eviction_results(
	primary_result: Any,
	auxiliary_result: Any,
	context: str,
) -> None:
	"""Raise if primary/aux prefix eviction results diverge."""
	_assert_matching_attributes(
		primary_result,
		auxiliary_result,
		PREFIX_EVICTION_FIELDS,
		context,
		"prefix eviction result",
	)


def _assert_matching_attributes(
	primary_obj: Any,
	auxiliary_obj: Any,
	fields: Iterable[str],
	context: str,
	label: str,
) -> None:
	for field in fields:
		if not hasattr(primary_obj, field) or not hasattr(auxiliary_obj, field):
			continue
		primary_value = _normalize_value(getattr(primary_obj, field))
		auxiliary_value = _normalize_value(getattr(auxiliary_obj, field))
		if primary_value != auxiliary_value:
			raise RuntimeError(
				f"{context}: primary/auxiliary {label} mismatch at field "
				f"{field}: primary={primary_value}, auxiliary={auxiliary_value}"
			)


def _normalize_value(value: Any) -> Any:
	if isinstance(value, tuple):
		return [_normalize_value(item) for item in value]
	if isinstance(value, list):
		return [_normalize_value(item) for item in value]
	return value


def _page_list_length(value: Any) -> int:
	if value is None:
		return 0
	if isinstance(value, (list, tuple)):
		return len(value)
	return len(list(value))
