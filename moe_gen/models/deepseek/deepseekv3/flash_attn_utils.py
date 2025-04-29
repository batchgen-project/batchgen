import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

"""
	Test Performance and Correctness of BF16 MLA implementation using FA3 as backend.
"""
class IndexFirstAxis(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, indices):
		ctx.save_for_backward(indices)
		assert input.ndim >= 2
		ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
		second_dim = other_shape.numel()
		# TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
		# return input[indices]
		return torch.gather(
			rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
		).reshape(-1, *other_shape)

	@staticmethod
	def backward(ctx, grad_output):
		(indices,) = ctx.saved_tensors
		assert grad_output.ndim >= 2
		other_shape = grad_output.shape[1:]
		grad_output = rearrange(grad_output, "b ... -> b (...)")
		grad_input = torch.zeros(
			[ctx.first_axis_dim, grad_output.shape[1]],
			device=grad_output.device,
			dtype=grad_output.dtype,
		)
		# TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
		# grad_input[indices] = grad_output
		grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
		return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply

class IndexPutFirstAxis(torch.autograd.Function):
	@staticmethod
	def forward(ctx, values, indices, first_axis_dim):
		ctx.save_for_backward(indices)
		assert indices.ndim == 1
		assert values.ndim >= 2
		output = torch.zeros(
			first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
		)
		# TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
		output[indices] = values
		# output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		(indices,) = ctx.saved_tensors
		# TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
		grad_values = grad_output[indices]
		# grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
		return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply

def unpad_input(hidden_states, attention_mask, unused_mask=None):
	"""
	Arguments:
		hidden_states: (batch, seqlen, ...)
		attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
		unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
	Return:
		hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
		indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
		cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
		max_seqlen_in_batch: int
		seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
	"""
	all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
	seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
	used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
	indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
	max_seqlen_in_batch = seqlens_in_batch.max().item()
	cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
	# TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
	# bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
	# times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
	# index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
	# so we write custom forward and backward to make it a bit faster.
	return (
		index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
		indices,
		cu_seqlens,
		max_seqlen_in_batch,
		used_seqlens_in_batch, 
	)


def pad_input(hidden_states, indices, batch, seqlen):
	"""
	Arguments:
		hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
		indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
		batch: int, batch size for the padded sequence.
		seqlen: int, maximum sequence length for the padded sequence.
	Return:
		hidden_states: (batch, seqlen, ...)
	"""
	dim = hidden_states.shape[1:]
	output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
	output[indices] = hidden_states
	return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _get_unpad_data(attention_mask):
	"""
		Unpads the input data based on the attention mask.
		Args:
			attention_mask (torch.Tensor): The attention mask tensor.[bsz, seq_len]
		Returns:
			indices (torch.Tensor): The indices of the unpadded data.
			cu_seqlens (torch.Tensor): The cumulative sequence lengths.
			max_seqlen_in_batch (int): The maximum sequence length in the batch.	
	"""
	seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
	indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
	max_seqlen_in_batch = seqlens_in_batch.max().item()
	cu_seqlens = F.pad(
		torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
	)
	return (
		indices,
		cu_seqlens,
		max_seqlen_in_batch,
	)

def _upad_input(
	query_layer, key_layer, value_layer, attention_mask, query_length
):
	num_heads = query_layer.shape[2]
	head_dim_k = query_layer.shape[-1]
	head_dim_v = value_layer.shape[-1]
	indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
	batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

	key_layer = index_first_axis(
		key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim_k),
		indices_k,
	)
	value_layer = index_first_axis(
		value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim_v),
		indices_k,
	)
	if query_length == kv_seq_len:
		query_layer = index_first_axis(
			query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim_k),
			indices_k,
		)
		cu_seqlens_q = cu_seqlens_k
		max_seqlen_in_batch_q = max_seqlen_in_batch_k
		indices_q = indices_k
	else:
		raise ValueError(
			"Query length must be equal to key value length for MLA prefill."
		)
	# elif query_length == 1:
	# 	max_seqlen_in_batch_q = 1
	# 	cu_seqlens_q = torch.arange(
	# 		batch_size + 1, dtype=torch.int32, device=query_layer.device
	# 	)  # There is a memcpy here, that is very bad.
	# 	indices_q = cu_seqlens_q[:-1]
	# 	query_layer = query_layer.squeeze(1)
	# else:
	# 	# The -q_len: slice assumes left padding.
	# 	attention_mask = attention_mask[:, -query_length:]
	# 	query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
	# 		query_layer, attention_mask
	# 	)

	return (
		query_layer,
		key_layer,
		value_layer,
		indices_q,
		(cu_seqlens_q, cu_seqlens_k),
		(max_seqlen_in_batch_q, max_seqlen_in_batch_k),
	)
