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


def run_glm5_prefix_aware_prefill(
	*,
	wrapper: object,
	hidden_states_2d: torch.Tensor,
	position_ids: torch.Tensor,
	metadata: PrefixCachePrepackMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Run GLM-5 suffix prefill against cached prefix MLA KV.

	The current worker isolates prefix-reuse prefill into one sequence per
	micro-batch. Keeping that invariant here avoids changing the surrounding
	batching architecture while preserving correct causal alignment: FlashMLA
	sees the full compressed KV sequence, and the suffix query is treated as the
	tail of that sequence.
	"""
	if not metadata.prefix_reuse_mode:
		raise RuntimeError("GLM-5 prefix-aware prefill requires prefix reuse mode")
	if metadata.num_sequences != 1:
		raise RuntimeError(
			"GLM-5 prefix-aware prefill currently requires single-sequence "
			"micro-batches"
		)
	if metadata.prefix_shared_tokens is None or metadata.full_seq_lengths is None:
		raise RuntimeError("GLM-5 prefix-aware prefill requires prefix metadata")

	attn = wrapper.module
	q_states, offload_kv = _project_suffix_query_and_kv(
		wrapper=wrapper,
		hidden_states_2d=hidden_states_2d,
		position_ids=position_ids,
		full_length=max(metadata.full_seq_lengths),
		weight_scale=wrapper.weight_dequant_scale,
	)
	compressed_kv, cu_k, _ = (
		wrapper.prefix_attention_kv_builder().build_mla_prefix_kv(
			key=offload_kv,
			metadata=metadata,
			kv_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
		)
	)
	blocked_k, block_table, cache_seqlens = _blocked_mla_kv_by_sequence(
		compressed_kv=compressed_kv,
		cu_k=cu_k,
		page_size=wrapper.host_prefix_reader().page_size(),
	)
	attn_output = _run_flash_mla_prefix_attention(
		wrapper=wrapper,
		query_states=q_states,
		blocked_k=blocked_k,
		block_table=block_table,
		cache_seqlens=cache_seqlens,
		query_len=int(metadata.seq_lengths[0]),
	)
	return attn_output, offload_kv


def run_glm5_full_hit_prefill(
	*,
	wrapper: object,
	hidden_states_2d: torch.Tensor,
	position_ids: torch.Tensor,
	metadata: PrefixCachePrepackMetadata,
) -> torch.Tensor:
	"""Run GLM-5 exact full-hit prefill against fully cached MLA KV."""
	if not metadata.full_hit_mode:
		raise RuntimeError("GLM-5 full-hit prefill requires full-hit mode")
	if metadata.full_seq_lengths is None:
		raise RuntimeError("GLM-5 full-hit prefill requires full sequence lengths")
	metadata.validate_full_hit_query_lengths()

	attn = wrapper.module
	q_states = _project_query_states(
		wrapper=wrapper,
		hidden_states_2d=hidden_states_2d,
		position_ids=position_ids,
		full_length=max(metadata.full_seq_lengths),
		weight_scale=wrapper.weight_dequant_scale,
	)
	compressed_kv, cu_k, _ = wrapper.prefix_attention_kv_builder().build_mla_full_hit_kv(
		metadata=metadata,
		kv_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
		dtype=q_states.dtype,
		device=hidden_states_2d.device,
	)
	blocked_k, block_table, cache_seqlens = _blocked_mla_kv_by_sequence(
		compressed_kv=compressed_kv,
		cu_k=cu_k,
		page_size=wrapper.host_prefix_reader().page_size(),
	)
	return _run_flash_mla_prefix_attention(
		wrapper=wrapper,
		query_states=q_states,
		blocked_k=blocked_k,
		block_table=block_table,
		cache_seqlens=cache_seqlens,
		query_len=1,
	)


def _run_flash_mla_prefix_attention(
	*,
	wrapper: object,
	query_states: torch.Tensor,
	blocked_k: torch.Tensor,
	block_table: torch.Tensor,
	cache_seqlens: torch.Tensor,
	query_len: int,
) -> torch.Tensor:
	attn = wrapper.module

	from batchgen.attention.mla.flashmla_backend import (
		flash_mla_with_kvcache,
		get_mla_metadata,
	)

	tile_scheduler_metadata, num_splits = get_mla_metadata(
		cache_seqlens,
		attn.num_heads,
		int(query_len),
	)
	attn_out, _ = flash_mla_with_kvcache(
		query_states,
		blocked_k,
		block_table,
		cache_seqlens,
		attn.kv_lora_rank,
		tile_scheduler_metadata,
		num_splits,
		attn.softmax_scale,
		True,
	)
	out_absorb = _out_absorb_weights(wrapper)
	attn_output = torch.einsum("bqhc,hdc->bqhd", attn_out, out_absorb)
	attn_output = attn_output.reshape(
		query_states.shape[0] * int(query_len),
		attn.num_heads * attn.v_head_dim,
	)
	attn_output = _w8a16_gemm(
		attn.o_proj.weight.data,
		wrapper.weight_dequant_scale["o_proj.weight_scale_inv"],
		attn_output,
	)
	return attn_output


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


def _project_suffix_query_and_kv(
	*,
	wrapper: object,
	hidden_states_2d: torch.Tensor,
	position_ids: torch.Tensor,
	full_length: int,
	weight_scale: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
	attn = wrapper.module
	q_states = _w8a16_gemm(
		attn.q_a_proj.weight.data,
		weight_scale["q_a_proj.weight_scale_inv"],
		hidden_states_2d,
	)
	q_states = attn.q_a_layernorm(q_states)
	q_states = _w8a16_gemm(
		attn.q_b_proj.weight.data,
		weight_scale["q_b_proj.weight_scale_inv"],
		q_states,
	)
	total_tokens = hidden_states_2d.shape[0]
	q_states = q_states.view(total_tokens, attn.num_heads, attn.q_head_dim)
	q_nope, q_pe = torch.split(
		q_states,
		[attn.qk_nope_head_dim, attn.qk_rope_head_dim],
		dim=-1,
	)

	compressed_kv = _w8a16_gemm(
		attn.kv_a_proj_with_mqa.weight.data,
		weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
		hidden_states_2d,
	)
	kv, k_pe = torch.split(
		compressed_kv,
		[attn.kv_lora_rank, attn.qk_rope_head_dim],
		dim=-1,
	)
	kv = attn.kv_a_layernorm(kv)
	k_pe = k_pe.view(total_tokens, 1, attn.qk_rope_head_dim)

	from batchgen.attention.mla.rotary_embedding import (
		rotary_pos_emb_interleaved_native,
	)

	rotary_seq_len = max(int(full_length), int(position_ids.max().item()) + 1)
	cos, sin = attn.rotary_emb(q_pe.unsqueeze(0), seq_len=rotary_seq_len)
	q_pe = rotary_pos_emb_interleaved_native(
		q_pe.unsqueeze(0),
		cos,
		sin,
		position_ids.unsqueeze(0),
		2,
	).squeeze(0)
	k_pe = rotary_pos_emb_interleaved_native(
		k_pe.unsqueeze(0),
		cos,
		sin,
		position_ids.unsqueeze(0),
		2,
	).squeeze(0)

	offload_kv = torch.cat(
		[kv, k_pe.view(total_tokens, attn.qk_rope_head_dim)],
		dim=-1,
	)
	q_absorb = _q_absorb_weights(wrapper)
	query_states = torch.empty(
		1,
		total_tokens,
		attn.num_heads,
		attn.kv_lora_rank + attn.qk_rope_head_dim,
		dtype=offload_kv.dtype,
		device=offload_kv.device,
	)
	query_states[0, :, :, : attn.kv_lora_rank] = torch.einsum(
		"thd,hdc->thc",
		q_nope,
		q_absorb,
	)
	query_states[0, :, :, attn.kv_lora_rank :] = q_pe
	return query_states.contiguous(), offload_kv


def _project_query_states(
	*,
	wrapper: object,
	hidden_states_2d: torch.Tensor,
	position_ids: torch.Tensor,
	full_length: int,
	weight_scale: dict,
) -> torch.Tensor:
	attn = wrapper.module
	q_states = _w8a16_gemm(
		attn.q_a_proj.weight.data,
		weight_scale["q_a_proj.weight_scale_inv"],
		hidden_states_2d,
	)
	q_states = attn.q_a_layernorm(q_states)
	q_states = _w8a16_gemm(
		attn.q_b_proj.weight.data,
		weight_scale["q_b_proj.weight_scale_inv"],
		q_states,
	)
	total_tokens = hidden_states_2d.shape[0]
	q_states = q_states.view(total_tokens, attn.num_heads, attn.q_head_dim)
	q_nope, q_pe = torch.split(
		q_states,
		[attn.qk_nope_head_dim, attn.qk_rope_head_dim],
		dim=-1,
	)

	from batchgen.attention.mla.rotary_embedding import (
		rotary_pos_emb_interleaved_native,
	)

	rotary_seq_len = max(int(full_length), int(position_ids.max().item()) + 1)
	cos, sin = attn.rotary_emb(q_pe.unsqueeze(0), seq_len=rotary_seq_len)
	q_pe = rotary_pos_emb_interleaved_native(
		q_pe.unsqueeze(0),
		cos,
		sin,
		position_ids.unsqueeze(0),
		2,
	).squeeze(0)

	q_absorb = _q_absorb_weights(wrapper)
	query_states = torch.empty(
		total_tokens,
		attn.num_heads,
		attn.kv_lora_rank + attn.qk_rope_head_dim,
		dtype=q_pe.dtype,
		device=q_pe.device,
	)
	query_states[:, :, : attn.kv_lora_rank] = torch.einsum(
		"thd,hdc->thc",
		q_nope,
		q_absorb,
	)
	query_states[:, :, attn.kv_lora_rank :] = q_pe
	return query_states.view(
		total_tokens,
		1,
		attn.num_heads,
		attn.kv_lora_rank + attn.qk_rope_head_dim,
	).contiguous()


def _blocked_mla_kv_by_sequence(
	*,
	compressed_kv: torch.Tensor,
	cu_k: torch.Tensor,
	page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	if compressed_kv.dim() != 3:
		raise RuntimeError(
			f"GLM-5 compressed KV must be [tokens, 1, dim], got "
			f"{tuple(compressed_kv.shape)}"
		)
	page_size = int(page_size)
	cu_values = [int(value) for value in cu_k.detach().cpu().tolist()]
	if len(cu_values) < 2:
		raise RuntimeError("GLM-5 blocked KV build requires at least one sequence")

	page_blocks = []
	block_rows = []
	cache_lengths = []
	next_page_idx = 0
	for seq_idx in range(len(cu_values) - 1):
		start = cu_values[seq_idx]
		end = cu_values[seq_idx + 1]
		seq_len = end - start
		if seq_len <= 0:
			raise RuntimeError(
				f"GLM-5 blocked KV build got empty sequence at index {seq_idx}"
			)
		segment = compressed_kv[start:end]
		num_pages = (seq_len + page_size - 1) // page_size
		padded_tokens = num_pages * page_size
		if padded_tokens != seq_len:
			padding = torch.zeros(
				padded_tokens - seq_len,
				compressed_kv.shape[1],
				compressed_kv.shape[2],
				dtype=compressed_kv.dtype,
				device=compressed_kv.device,
			)
			segment = torch.cat([segment, padding], dim=0)
		page_blocks.append(
			segment.contiguous().view(
				num_pages,
				page_size,
				compressed_kv.shape[1],
				compressed_kv.shape[2],
			)
		)
		block_rows.append(
			torch.arange(
				next_page_idx,
				next_page_idx + num_pages,
				dtype=torch.int32,
				device=compressed_kv.device,
			)
		)
		cache_lengths.append(seq_len)
		next_page_idx += num_pages

	blocked_k = torch.cat(page_blocks, dim=0)
	max_pages = max(int(row.numel()) for row in block_rows)
	block_table = torch.zeros(
		(len(block_rows), max_pages),
		dtype=torch.int32,
		device=compressed_kv.device,
	)
	for row_idx, row in enumerate(block_rows):
		block_table[row_idx, : row.numel()] = row
	cache_seqlens = torch.tensor(
		cache_lengths,
		dtype=torch.int32,
		device=compressed_kv.device,
	)
	return blocked_k, block_table, cache_seqlens


def _q_absorb_weights(wrapper: object) -> torch.Tensor:
	if getattr(wrapper, "_cached_q_absorb", None) is not None:
		return wrapper._cached_q_absorb
	attn = wrapper.module
	if getattr(attn, "q_absorb", None) is not None:
		return attn.q_absorb
	kv_b_proj = _dequantized_kv_b_proj(wrapper)
	return kv_b_proj[:, : attn.qk_nope_head_dim, :]


def _out_absorb_weights(wrapper: object) -> torch.Tensor:
	if getattr(wrapper, "_cached_out_absorb", None) is not None:
		return wrapper._cached_out_absorb
	attn = wrapper.module
	if getattr(attn, "out_absorb", None) is not None:
		return attn.out_absorb
	kv_b_proj = _dequantized_kv_b_proj(wrapper)
	return kv_b_proj[:, attn.qk_nope_head_dim :, :]


def _dequantized_kv_b_proj(wrapper: object) -> torch.Tensor:
	attn = wrapper.module
	weight_scale = getattr(wrapper, "weight_dequant_scale", None)
	if weight_scale is None or "kv_b_proj.weight_scale_inv" not in weight_scale:
		raise RuntimeError("GLM-5 prefix prefill requires kv_b_proj weight scale")

	from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization

	return deepseek_v3_dequantization(
		attn.kv_b_proj.weight.data,
		weight_scale["kv_b_proj.weight_scale_inv"],
	).view(
		attn.num_heads,
		-1,
		attn.kv_lora_rank,
	)


def _w8a16_gemm(
	weight_data_fp8: torch.Tensor,
	weight_scale_inv_fp32: torch.Tensor,
	activation_bf16: torch.Tensor,
) -> torch.Tensor:
	import os as _os_gemm

	from batchgen.attention.mla.fa3_backend import (
		w8a16_gemm,
		w8a16_gemm_dequant,
	)

	use_dequant_path = _os_gemm.environ.get("BATCHGEN_W8A16_DEQUANT", "0") == "1"
	gemm = w8a16_gemm_dequant if use_dequant_path else w8a16_gemm
	return gemm(weight_data_fp8, weight_scale_inv_fp32, activation_bf16)
