"""
CompletionHandler: Sequence completion detection and distributed health checks.

Extracted from batchgen_worker.py (Step 11 of scheduler split).
Handles: EOS detection, completion checks, finish reasons, distributed reinit.
"""
import logging
from datetime import timedelta
from typing import List, Tuple

import torch
import torch.distributed as dist

from batchgen.worker.state import WorkerState


class CompletionHandler:
	"""Sequence completion detection, EOS handling, and distributed health checks."""

	def __init__(self, state: WorkerState, worker):
		self.state = state
		self._worker = worker

	def should_stop_at_eos(self, token_id: int) -> bool:
		"""
		Check if we should stop at this token.

		Returns True if token is EOS AND we're not ignoring EOS.
		"""
		if self._worker._ignore_eos:
			return False
		return token_id in self._worker.eos_token_ids

	def is_completed(self, seq) -> bool:
		"""
		Unified completion check that respects ignore_eos.

		A sequence is completed if:
		1. It reached max_decoding_length (always checked), OR
		2. It hit EOS AND ignore_eos is False, OR
		3. current_context_length >= model_context_length (context limit reached)
		"""
		# Always complete at per-sequence max decoding length
		if seq.decoded_length >= seq.max_decode_length:
			return True

		# Complete if context length limit reached (prompt + decoded >= model max)
		if seq.current_context_length >= self.state.model_context_length:
			return True

		# Only complete at EOS if not ignoring EOS
		if seq.eos_reached and not self._worker._ignore_eos:
			return True

		return False

	def get_finish_reason(self, seq) -> str:
		"""Return OpenAI-compatible finish_reason for a completed sequence."""
		# EOS takes priority (natural completion)
		if seq.eos_reached and not self._worker._ignore_eos:
			return "stop"
		# Otherwise it's a length limit (decode cap or context cap)
		return "length"

	def check_and_handle(
		self,
		decode_uuids: List[str],
		local_decode_indices: List[int],
		new_token_idx: int
	) -> Tuple[List[str], List[int], List[str]]:
		"""
		Check for completed sequences at page boundaries.
		FIXED: Respects ignore_eos flag.
		"""
		n = len(decode_uuids)
		if n == 0:
			return [], [], []

		# Vectorized completion check: build tensors once, compare in batch
		decoded_lens = torch.empty(n, dtype=torch.int64)
		max_lens = torch.empty(n, dtype=torch.int64)
		ctx_lens = torch.empty(n, dtype=torch.int64)
		eos_flags = torch.empty(n, dtype=torch.bool)
		ignore_eos = self._worker._ignore_eos

		seqs = []
		for i, uuid in enumerate(decode_uuids):
			seq = self.state.global_batch.get_sequence(uuid)
			seqs.append(seq)
			decoded_lens[i] = seq.decoded_length
			max_lens[i] = seq.max_decode_length
			ctx_lens[i] = seq.current_context_length
			eos_flags[i] = seq.eos_reached and not ignore_eos

		completed_mask = (
			(decoded_lens >= max_lens)
			| (ctx_lens >= self.state.model_context_length)
			| eos_flags
		)

		completed_uuids = []
		active_uuids = []
		active_local_indices = []
		for i in range(n):
			uuid = decode_uuids[i]
			if completed_mask[i]:
				completed_uuids.append(uuid)
				seq = seqs[i]
				logging.info(
					f"Rank {self.state.rank}: Sequence {uuid} completed at token {new_token_idx} "
					f"(decoded_length={seq.decoded_length}, eos_reached={seq.eos_reached}, "
					f"ignore_eos={self._worker._ignore_eos})"
				)
			else:
				active_uuids.append(uuid)
				if uuid in self.state.uuid_to_local_map:
					active_local_indices.append(self.state.uuid_to_local_map[uuid])

		return active_uuids, active_local_indices, completed_uuids

	def check_and_reinit_distributed(self) -> bool:
		"""
		Check if torch.distributed is healthy. If not, attempt to reinitialize.
		Returns True if distributed is healthy (or was successfully reinitialized).
		Returns False if reinitialization failed.
		"""
		if not dist.is_initialized():
			logging.warning(f"Rank {self.state.rank}: torch.distributed not initialized, attempting to initialize...")
			try:
				self._worker._init_torch_dist()
				return True
			except Exception as e:
				logging.error(f"Rank {self.state.rank}: Failed to initialize torch.distributed: {e}")
				return False

		# Perform a quick health check with a short timeout
		try:
			health_tensor = torch.ones(1, device=self.state.torch_device)
			work = dist.all_reduce(health_tensor, op=dist.ReduceOp.SUM, async_op=True)

			# Wait with a short timeout (30 seconds)
			success = work.wait(timeout=timedelta(seconds=30))

			if not success:
				raise RuntimeError("Health check timed out")

			expected = float(self.state.world_size)
			if abs(health_tensor.item() - expected) > 1e-6:
				raise RuntimeError(f"Health check result mismatch: got {health_tensor.item()}, expected {expected}")

			logging.debug(f"Rank {self.state.rank}: Distributed health check passed")
			return True

		except Exception as e:
			logging.warning(f"Rank {self.state.rank}: Distributed health check failed: {e}")
			logging.info(f"Rank {self.state.rank}: Attempting to reinitialize torch.distributed...")

			try:
				dist.destroy_process_group()
			except Exception as destroy_e:
				logging.warning(f"Rank {self.state.rank}: Error destroying process group: {destroy_e}")

			try:
				self._worker._init_torch_dist()
				logging.info(f"Rank {self.state.rank}: Successfully reinitialized torch.distributed")
				return True
			except Exception as reinit_e:
				logging.error(f"Rank {self.state.rank}: Failed to reinitialize torch.distributed: {reinit_e}")
				return False
