"""GLM-5 prefix-cache helpers.

The GLM-5 DSA path writes two logical caches during prefill:
primary MLA KV and auxiliary indexer KV. Both can reuse the common
PrefixAwarePrefillOffloader, but GLM-5 still needs model-local tensor lifetime
management because the host offload tasks read CUDA tensors asynchronously.
"""

from __future__ import annotations

import torch

from batchgen.models.wrappers import AttnWrapperBase
from batchgen.models.wrappers.prefix_cache import (
	PrefixAwarePrefillOffloader,
	PrefixCachePrepackMetadata,
)


def offload_glm5_prepacked_mla_kv(
	*,
	key: torch.Tensor,
	worker_view: object,
	layer_idx: int,
	metadata: PrefixCachePrepackMetadata,
) -> None:
	"""Offload prepacked GLM-5 k-only MLA/indexer KV with prefix offsets."""
	_pin_parent_tensor_until_prefill_offload_done(key)
	offloader = PrefixAwarePrefillOffloader(
		worker_view=worker_view,
		layer_idx=layer_idx,
		metadata=metadata,
		track_task=AttnWrapperBase.track_prefill_offload_task,
	)
	offloader.offload_mla(
		key=key,
		sequence_callback=_pin_sequence_tensor_until_prefill_offload_done,
	)


def _pin_parent_tensor_until_prefill_offload_done(tensor: torch.Tensor) -> None:
	AttnWrapperBase.pending_prefill_offload_tensors.append(tensor)
	if tensor.is_cuda:
		event = torch.cuda.Event()
		event.record(torch.cuda.current_stream())
		event.synchronize()


def _pin_sequence_tensor_until_prefill_offload_done(
	_seq_idx: int,
	_sequence_id: int,
	_seq_len: int,
	seq_tensor: torch.Tensor,
) -> None:
	AttnWrapperBase.pending_prefill_offload_tensors.append(seq_tensor)
