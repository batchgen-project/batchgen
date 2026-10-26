"""Token sampling utilities for batch inference.

Supports per-sequence sampling parameters via [B] tensors for temperature,
top_p, and top_k. Scalar params are auto-broadcast for backward compatibility.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F

from batchgen.worker.sampling_residency import (
	SamplingHostDecisions,
	SamplingResidencyShapeError,
	SamplingRowIndexError,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# The one-sample draw on the host-decided (decode step) path
# --------------------------------------------------------------------------
#
# True: the sampled rows are drawn with `_draw_one_per_row` below, which
# launches the same kernels as `torch.multinomial(probs, 1)` without its two
# device->host validity readbacks. False: `torch.multinomial` exactly as
# before (byte-identical behaviour, including its RuntimeError on an invalid
# distribution). Read at call time, so it can be flipped on the module. Only
# the host-decided path consults it; the reference path (`host_decisions`
# None) syncs on `bad_rows.any()` anyway and keeps `torch.multinomial` as
# the oracle the switch is tested against.
SAMPLED_DRAW_SYNC_FREE = True


def _draw_one_per_row(probs: torch.Tensor) -> torch.Tensor:
	"""`torch.multinomial(probs, 1)` minus its host syncs; token-identical.

	For `n_sample == 1` ATen's multinomial is itself the exponential race
	(Gumbel-max with the exp applied): it checks
	`((max < inf) & (min >= 0)).item()` and `(sum(1) == 0).sum().item()` --
	two cudaStreamSynchronize per call -- and then runs exactly

	    q = empty_like(self).exponential_(1, gen)
	    div_out(q, self, q)
	    argmax_out(result, q, -1, keepdim=True)

	This is that tail, statement for statement: same allocation shape, dtype
	and layout, same default generator, same single `exponential_` over
	`probs.numel()` elements -- so the Philox (CUDA) / mt19937 (CPU) stream
	consumed, the values drawn, and the generator state left behind are the
	ones `torch.multinomial` would produce. Nothing else in `sample_tokens`
	touches the RNG, so the surrounding order is unchanged too.

	It does NOT validate. On an invalid row the race returns garbage silently
	(argmax picks the first NaN; an all-zero row yields token 0; an +inf entry
	always wins; negative entries never do), so the caller must route every
	row multinomial would have rejected to its fallback BEFORE trusting the
	draw -- see `_invalid_draw_rows`.
	"""
	q = torch.empty_like(probs).exponential_(1)
	torch.div(probs, q, out=q)
	return q.argmax(dim=-1, keepdim=True)


def _invalid_draw_rows(probs: torch.Tensor, bad_rows: torch.Tensor) -> torch.Tensor:
	"""Per-row form of multinomial's validity check, on device, no readback.

	`torch.multinomial` refuses a distribution with a NaN, an inf, a negative
	entry, or a zero row sum. Per row, that is exactly
	`~((min >= 0) & (max < inf) & (max > 0))`: `aminmax` propagates NaN, so
	both comparisons are False on a NaN row; and once every entry is finite
	and non-negative the row sum is zero iff the max is zero (a sum of
	non-negative floats cannot round to 0 unless every term is 0). One fused
	pass replaces multinomial's separate global max, min and row sum.

	`bad_rows` (NaN or sum < 1e-10) already covers NaN and zero sum, and a
	softmax output cannot hold inf or negatives, so for today's caller the
	result equals `bad_rows`. The full check is kept anyway so that the draw
	can never silently consume a distribution multinomial would have raised
	on: such a row now takes the deterministic greedy fallback and is counted
	in `bad_row_counter` (drained and reported by the caller), instead of
	raising -- raising is the one thing that needs a sync.
	"""
	row_min, row_max = torch.aminmax(probs, dim=-1)
	valid = (row_min >= 0) & (row_max < float("inf")) & (row_max > 0)
	return bad_rows | ~valid


# --------------------------------------------------------------------------
# The greedy/sampling row split
# --------------------------------------------------------------------------
#
# `sample_tokens` handles greedy and sampled rows separately. How the two
# halves are gathered and written back is the only thing that differs between
# the reference path and the host-decided decode step path, so it lives in
# these three small objects:
#
# * `_MaskSplit` -- reference path. The split is decided on device from the
#   cast temperatures. Every boolean-mask gather/scatter is an implicit
#   `nonzero`, i.e. a cudaStreamSynchronize; acceptable off the step path, and
#   kept as the oracle the other two are tested against.
# * `_WholeSplit` -- host-decided, uniform batch. Only one half exists and it
#   is the whole batch, so there is nothing to gather.
# * `_IndexSplit` -- host-decided, mixed batch. Gathers and scatters with int64
#   row-index tensors that were built on device once per row-set change
#   (`SamplingHostDecisions.greedy_index` / `.sampling_index`). index_select /
#   index_copy_ take their sizes from tensor metadata, so neither syncs.
#
# Boolean indexing gathers rows in ascending row order; the index tensors hold
# the same rows in the same ascending order, so all three produce
# element-identical half-batches.


class _MaskSplit:
	__slots__ = ("greedy_mask", "sampling_mask")

	def __init__(self, greedy_mask: torch.Tensor):
		self.greedy_mask = greedy_mask
		self.sampling_mask = ~greedy_mask

	def greedy(self, t: torch.Tensor) -> torch.Tensor:
		return t[self.greedy_mask]

	def sampling(self, t: torch.Tensor) -> torch.Tensor:
		return t[self.sampling_mask]

	def write_greedy(self, out: torch.Tensor, values: torch.Tensor) -> None:
		out[self.greedy_mask] = values

	def write_sampling(self, out: torch.Tensor, values: torch.Tensor) -> None:
		out[self.sampling_mask] = values


class _WholeSplit:
	__slots__ = ()

	def greedy(self, t: torch.Tensor) -> torch.Tensor:
		return t

	def sampling(self, t: torch.Tensor) -> torch.Tensor:
		return t

	def write_greedy(self, out: torch.Tensor, values: torch.Tensor) -> None:
		out.copy_(values)

	def write_sampling(self, out: torch.Tensor, values: torch.Tensor) -> None:
		out.copy_(values)


class _IndexSplit:
	__slots__ = ("greedy_index", "sampling_index")

	def __init__(self, greedy_index: torch.Tensor, sampling_index: torch.Tensor):
		self.greedy_index = greedy_index
		self.sampling_index = sampling_index

	def greedy(self, t: torch.Tensor) -> torch.Tensor:
		return t.index_select(0, self.greedy_index)

	def sampling(self, t: torch.Tensor) -> torch.Tensor:
		return t.index_select(0, self.sampling_index)

	def write_greedy(self, out: torch.Tensor, values: torch.Tensor) -> None:
		out.index_copy_(0, self.greedy_index, values)

	def write_sampling(self, out: torch.Tensor, values: torch.Tensor) -> None:
		out.index_copy_(0, self.sampling_index, values)


_WHOLE_SPLIT = _WholeSplit()


def _host_split(
	decisions: SamplingHostDecisions, rows: int, device: torch.device
) -> Union[_WholeSplit, _IndexSplit]:
	"""The row split for a host-decided batch; fail-loud, never a mask fallback.

	Every check reads tensor metadata only (shape, dtype, device), so none of
	them touches the device.
	"""
	if not decisions.mixed:
		return _WHOLE_SPLIT
	gi = decisions.greedy_index
	si = decisions.sampling_index
	if gi is None or si is None:
		raise SamplingRowIndexError(
			f"mixed greedy/sampling batch of {rows} rows "
			f"({len(decisions.greedy_rows)} greedy, {len(decisions.sampling_rows)} "
			"sampled) reached sample_tokens without its row-index tensors; build "
			"them once per row-set change (DecodeSamplingResidency, or "
			"attach_row_index off the step path)"
		)
	for name, index, host_rows in (
		("greedy_index", gi, decisions.greedy_rows),
		("sampling_index", si, decisions.sampling_rows),
	):
		if index.dim() != 1 or index.dtype != torch.int64 or index.device != device:
			raise SamplingRowIndexError(
				f"{name} must be a 1-D int64 tensor on {device}, got "
				f"shape={tuple(index.shape)} dtype={index.dtype} device={index.device}"
			)
		if index.numel() != len(host_rows):
			raise SamplingRowIndexError(
				f"{name} holds {index.numel()} rows, the host decision lists "
				f"{len(host_rows)}"
			)
	if gi.numel() + si.numel() != rows:
		raise SamplingRowIndexError(
			f"greedy ({gi.numel()}) + sampled ({si.numel()}) rows do not "
			f"partition the {rows}-row batch"
		)
	return _IndexSplit(gi, si)


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
	"""
	Greedily decode the next token from logits.

	Args:
		logits: Tensor of shape [batch_size, vocab_size] containing the logits

	Returns:
		Tensor of shape [batch_size, 1] containing the indices of the selected tokens
	"""
	return torch.argmax(logits.float(), dim=-1, keepdim=True)


@torch.inference_mode()
def sample_tokens(
	logits: torch.Tensor,
	temperature: Union[float, torch.Tensor, None] = None,
	top_p: Union[float, torch.Tensor, None] = None,
	top_k: Union[int, torch.Tensor] = 0,
	host_decisions: Optional[SamplingHostDecisions] = None,
	bad_row_counter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
	"""
	Efficient token sampling with per-sequence temperature, top-p, and top-k.

	Supports both scalar params (applied uniformly) and [B] tensor params
	(per-sequence). Mixed greedy/sampling batches are split and handled
	separately for mathematical exactness.

	Args:
		logits: [batch_size, vocab_size] raw logits from model
		temperature: Scaling factor. None or <=0 = greedy. Scalar or [B] tensor.
		top_p: Nucleus sampling threshold. None or >=1.0 = disabled. Scalar or [B] tensor.
		top_k: Top-k filtering. 0 = disabled. Scalar or [B] tensor.
		host_decisions: Branch decisions the caller already knows host-side,
			from `batchgen.worker.sampling_residency`. When given, the four
			device->host reductions that only re-derive what the host itself
			wrote (`greedy_mask.any()`, `sampling_mask.any()`,
			`s_top_ks.max().item()`, `(s_top_ps < 1.0).any()`) are skipped,
			the greedy/sampling split is taken from the host decision instead
			of from a device boolean mask (a uniform batch is not split at
			all; a mixed batch is split with its per-batch row-index tensors,
			and refused with `SamplingRowIndexError` without them), and the
			invalid-probability fallback is selected on device instead of
			being branched on host. Every decision must be derived from the
			*cast* parameter values (see that module) or a row can take the
			wrong branch. When None, every branch is decided on device exactly
			as before -- that is the reference path all non-decode callers use.
		bad_row_counter: Optional 0-d int64 device tensor. With
			`host_decisions`, the number of rows that fell back to greedy
			because filtering left them with invalid probabilities is added to
			it on device, because the host can no longer see that count without
			the readback this path exists to remove. Drained by the caller at
			heartbeat rate.

	Returns:
		[batch_size, 1] sampled token indices
	"""
	B, V = logits.shape

	if host_decisions is not None and host_decisions.n != B:
		raise SamplingResidencyShapeError(
			f"host sampling decisions describe {host_decisions.n} rows, "
			f"logits carry {B}"
		)

	# --- Determine greedy mask ---
	# Scalar fast path: all greedy or all same params
	if temperature is None or (isinstance(temperature, (int, float)) and temperature <= 0):
		return logits.float().argmax(dim=-1, keepdim=True)

	# Convert scalars to [B] tensors for uniform code path
	if isinstance(temperature, (int, float)):
		temps = torch.full((B,), temperature, device=logits.device, dtype=logits.dtype)
	else:
		temps = temperature.to(device=logits.device, dtype=logits.dtype)

	if isinstance(top_p, (int, float)):
		top_ps = torch.full((B,), top_p if top_p is not None else 1.0, device=logits.device, dtype=logits.dtype)
	elif top_p is None:
		top_ps = torch.ones(B, device=logits.device, dtype=logits.dtype)
	else:
		top_ps = top_p.to(device=logits.device, dtype=logits.dtype)

	if isinstance(top_k, int):
		top_ks = torch.full((B,), top_k, device=logits.device, dtype=torch.int64)
	else:
		top_ks = top_k.to(device=logits.device, dtype=torch.int64)

	# --- Split greedy vs sampling ---
	# The split is derived from values the host wrote, so deciding it on device
	# just reads the host's own decision back through a stall -- twice: once
	# in `.any()`, and once in every boolean-mask gather/scatter below (an
	# implicit `nonzero`). When the caller already knows the answer, use it.
	if host_decisions is None:
		split = _MaskSplit(temps <= 0)
		any_greedy = bool(split.greedy_mask.any())
		any_sampling = bool(split.sampling_mask.any())
	else:
		split = _host_split(host_decisions, B, logits.device)
		any_greedy = host_decisions.any_greedy
		any_sampling = host_decisions.any_sampling

	result = torch.empty(B, 1, dtype=torch.long, device=logits.device)

	# Handle greedy sequences
	if any_greedy:
		split.write_greedy(
			result, split.greedy(logits).float().argmax(dim=-1, keepdim=True)
		)

	# Handle sampling sequences
	if not any_sampling:
		return result

	s_raw_logits = split.sampling(logits)  # [B_s, V]
	s_temps = split.sampling(temps)    # [B_s]
	s_top_ps = split.sampling(top_ps)  # [B_s]
	s_top_ks = split.sampling(top_ks)  # [B_s]
	B_s = s_raw_logits.shape[0]

	# The invalid-probability fallback below is an argmax over the UNSCALED
	# masked logits. On the sync-free path it has to be available without
	# knowing yet whether any row needs it, so take it here, off the masked
	# copy that already exists, and keep only the [B_s, 1] result. The copy
	# itself is dropped right after the temperature divide, exactly as before,
	# so peak memory is unchanged.
	fallback_tokens_all = None
	if host_decisions is not None:
		fallback_tokens_all = s_raw_logits.argmax(dim=-1, keepdim=True)

	# --- Temperature scaling ---
	s_logits = s_raw_logits / s_temps.unsqueeze(-1)  # [B_s, V] / [B_s, 1]
	del s_raw_logits

	# --- Top-k filtering (vectorized per-sequence k) ---
	if host_decisions is None:
		max_k = s_top_ks.max().item()
	else:
		max_k = host_decisions.max_top_k
	if max_k > 0:
		max_k = min(max_k, V)
		# Sort descending to get ranks
		sorted_logits, sorted_indices = torch.sort(s_logits, dim=-1, descending=True)
		# Create rank tensor [1, V] and compare with per-seq top_k [B_s, 1]
		ranks = torch.arange(V, device=logits.device).unsqueeze(0)  # [1, V]
		# For sequences with top_k=0, set effective_k=V (no filtering)
		effective_k = s_top_ks.clone()
		effective_k[effective_k <= 0] = V
		effective_k = effective_k.clamp(max=V)
		topk_mask = ranks >= effective_k.unsqueeze(-1)  # [B_s, V]
		sorted_logits[topk_mask] = float('-inf')
		# Scatter back to original order
		s_logits = torch.zeros_like(s_logits).scatter(1, sorted_indices, sorted_logits)

	# --- Top-p (nucleus) filtering (vectorized per-sequence threshold) ---
	if host_decisions is None:
		needs_top_p = (s_top_ps < 1.0).any()
	else:
		needs_top_p = host_decisions.needs_top_p
	if needs_top_p:
		sorted_logits, sorted_indices = torch.sort(s_logits, dim=-1, descending=True)
		cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

		# Remove tokens with cumulative prob above threshold, but keep at least one.
		# Same logic as original scalar code: shift right so the token that
		# crosses the threshold is kept, but everything after is removed.
		sorted_mask = cumulative_probs > s_top_ps.unsqueeze(-1)  # [B_s, V]
		# Shift: position i gets mask from position i-1
		sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
		sorted_mask[:, 0] = False  # always keep top-1

		# For sequences with top_p >= 1.0, disable filtering entirely
		no_filter = (s_top_ps >= 1.0).unsqueeze(-1)
		sorted_mask[no_filter.expand_as(sorted_mask)] = False

		# Scatter mask back to original indices and apply
		mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
		s_logits = s_logits.masked_fill(mask, float('-inf'))

	# --- Sample ---
	probs = F.softmax(s_logits, dim=-1)

	# Per-sequence NaN fallback: if filtering removed all tokens, use argmax
	bad_rows = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) < 1e-10)

	if host_decisions is not None:
		# Sync-free form: `bad_rows.any()` is a genuine data dependency on
		# `probs`, so it cannot be answered host-side -- it has to stop being
		# asked. Draw over all B_s rows and select, which keeps the common
		# (no bad row) case bitwise identical to the branched form: the filled
		# tensor equals `probs` element-for-element, and the one-sample draw
		# (an `exponential_` over the whole [B_s, V] tensor, with either
		# switch setting) consumes the same random numbers at the same
		# positions regardless of the rows' values.
		#
		# When a bad row IS present the good rows' draws differ from the
		# branched form, because that form compacts the good rows into a
		# smaller tensor and so shifts which random numbers each row consumes. The
		# fallback rows themselves are bit-identical (same unscaled argmax) and
		# the good rows are still correct draws from their own distributions;
		# only the branched form's incidental row-to-uniform mapping is lost,
		# and that was never a contract.
		bad_col = bad_rows.unsqueeze(-1)
		# In place: `probs` is a fresh softmax output with no other reader, so
		# this adds no allocation over the branched form. Bad rows become
		# uniform-over-vocab purely so the draw has a valid distribution;
		# their draw is discarded by the select below.
		probs = probs.masked_fill_(bad_col, 1.0)
		if SAMPLED_DRAW_SYNC_FREE:
			# Same draw as multinomial, without its validity readbacks. The
			# check it skipped is re-done per row on the filled tensor (what
			# the draw actually consumes), and every row it would have
			# rejected is selected away from the unvalidated draw below.
			bad_rows = _invalid_draw_rows(probs, bad_rows)
			bad_col = bad_rows.unsqueeze(-1)
			sampled = _draw_one_per_row(probs)
		else:
			sampled = torch.multinomial(probs, num_samples=1)
		sampled = torch.where(bad_col, fallback_tokens_all, sampled)
		if bad_row_counter is not None:
			# The host can no longer see this count without the readback this
			# path exists to remove, so accumulate on device and let the
			# caller drain it where the device is already caught up.
			bad_row_counter.add_(bad_rows.sum())
		split.write_sampling(result, sampled)
		return result

	if bad_rows.any():
		logger.warning(
			f"Invalid probabilities after filtering for {bad_rows.sum().item()} sequences, "
			"falling back to greedy for those"
		)
		# Fix bad rows with uniform over original logits
		fallback_tokens = split.sampling(logits)[bad_rows].argmax(dim=-1, keepdim=True)
		# Sample good rows
		good_rows = ~bad_rows
		sampled = torch.empty(B_s, 1, dtype=torch.long, device=logits.device)
		if good_rows.any():
			sampled[good_rows] = torch.multinomial(probs[good_rows], num_samples=1)
		sampled[bad_rows] = fallback_tokens
	else:
		sampled = torch.multinomial(probs, num_samples=1)

	split.write_sampling(result, sampled)
	return result


# Backward compatibility alias
def sample_with_temperature_top_p(
	logits: torch.Tensor,
	temperature: float = 1.0,
	top_p: float = 0.9,
	top_k: int = None,
	eps: float = 1e-10,
) -> torch.Tensor:
	"""
	Legacy sampling function for backward compatibility.

	Args:
		logits: Tensor of shape [batch_size, vocab_size] containing the logits
		temperature: Float controlling randomness (0.0 = deterministic)
		top_p: Float for nucleus sampling (0.0 to 1.0)
		top_k: Optional int to limit sampling to top k tokens
		eps: Ignored (kept for API compatibility)

	Returns:
		Sampled token indices of shape [batch_size, 1]
	"""
	return sample_tokens(
		logits,
		temperature=temperature if temperature >= 1e-6 else None,
		top_p=top_p,
		top_k=top_k or 0,
	)
