"""
	For Hooper GPU.
	- prefill_fa3()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn_interface import flash_attn_varlen_func 
from .padding import _upad_input, pad_input
from .rotary_embedding import mla_rotary_pos_emb, rotary_pos_emb, apply_rotary_pos_emb
import deep_gemm
from deep_gemm import get_col_major_tma_aligned_tensor
import logging
from typing import Tuple
import torch.distributed as dist
from ...moe.fused_dequant_gemm import fused_fp8_bf16_gemm

@torch.inference_mode()
def mla_prefill_flashattention3(
	self,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
		MLA prefifill on hooper device.
		Materialize QKV and call flash_attn_varlen_func(). (flash_attn_3 backend)
	"""
	bsz, seq_len, _ = hidden_states.shape
	query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	query_states = query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim)
	q_nope, q_pe = torch.split(
		query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)	
	compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	compressed_kv, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)	
	normed_kv = self.kv_a_layernorm(compressed_kv)
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(q_pe, seq_len=seq_len)
	q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids, 2)
	k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids, 2)
	k_pe = k_pe.view(bsz, seq_len, self.qk_rope_head_dim)
	offload_kv = torch.cat(
		[normed_kv, k_pe], dim=-1
	)

	kv = self.kv_b_proj(normed_kv)
	kv = kv.view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)		

	
	query_states = k_pe.new_empty(bsz, seq_len, self.num_heads, self.q_head_dim)
	query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
	query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

	key_states = k_pe.new_empty(
		bsz, seq_len, self.num_heads, self.q_head_dim
	)
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
	key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
	key_states[:, :, :, self.qk_nope_head_dim :] = k_pe	

	query_states = query_states.contiguous()
	key_states = key_states.contiguous()
	value_states = value_states.contiguous()

	# Call flash_attn_varlen_func
	(
		query_states,
		key_states,
		value_states,
		indices_q,
		cu_seq_lens,
		max_seq_lens,
	) = _upad_input(
		query_states, key_states, value_states, attention_mask, seq_len
	)      	
	cu_seqlens_q, cu_seqlens_k = cu_seq_lens
	max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
	attn_output_unpad = flash_attn_varlen_func(
		query_states,
		key_states,
		value_states,
		cu_seqlens_q=cu_seqlens_q,
		cu_seqlens_k=cu_seqlens_k,
		max_seqlen_q=max_seqlen_in_batch_q,
		max_seqlen_k=max_seqlen_in_batch_k,
		softmax_scale=self.softmax_scale,
		causal=True
	)		

	attn_output = pad_input(attn_output_unpad, indices_q, bsz, seq_len).view(
		bsz, seq_len, self.num_heads * self.v_head_dim
	).contiguous()

	attn_output = self.o_proj(attn_output)

	return attn_output, offload_kv


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	assert x.dim() == 2 and x.size(1) % 128 == 0
	m, n = x.shape
	x_view = x.view(m, -1, 128)
	x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
	return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
		torch.float8_e4m3fn
	).view(m, n), (x_amax / 448.0).view(m, -1)

# def w8a16_gemm(
# 	weight_data_fp8: torch.Tensor,
# 	weight_scale_inv_fp32: torch.Tensor,
# 	activation_bf16: torch.Tensor,
# ) -> torch.Tensor:
# 	"""
# 	activation_bf16: [n_group, m, k]
# 	weight_data_fp8: [m, n]
# 	weight_scale_inv_fp32: [m, n]
# 	"""
# 	assert weight_data_fp8.dim() == 2
# 	assert activation_bf16.dim() == 3
# 	n_group, m, k = activation_bf16.size()
# 	out = torch.empty_like(activation_bf16, dtype=torch.bfloat16)
# 	y_fp8 = (weight_data_fp8, weight_scale_inv_fp32)
# 	for i in range(n_group):
# 		x_fp8 = per_token_cast_to_fp8(activation_bf16[i])
# 		x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
# 		deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out[i])
# 	return out

import triton
import triton.language as tl
from triton import Config


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

def w8a16_gemm(
	weight_data_fp8: torch.Tensor,
	weight_scale_inv_fp32: torch.Tensor,
	activation_bf16: torch.Tensor,
) -> torch.Tensor:
	"""
		activation_bf16: [n_group, m, k]
		weight_data_fp8: [m, n]
		weight_scale_inv_fp32: [m, n]
	"""
	assert weight_data_fp8.dim() == 2
	assert activation_bf16.dim() == 3 or activation_bf16.dim() == 2
	if activation_bf16.dim() == 3:
		n_group, l, _ = activation_bf16.size()

	x = activation_bf16.view(-1, activation_bf16.size(-1))
	m, k = x.size()
	n, _ = weight_data_fp8.size()
	out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
	y_fp8 = (weight_data_fp8, weight_scale_inv_fp32)
	
	# x_fp8 = per_token_cast_to_fp8(x)
	x_fp8 = act_quant(x)
	x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
	deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
	if activation_bf16.dim() == 3:
		out = out.view(n_group, l, n)
	else:
		out = out.view(m, n)
	return out

@torch.inference_mode()
def mla_prefill_flashattention3_w8a16_deepgemm(
	self,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
	weight_scale: dict[str, torch.Tensor],

) -> tuple[torch.Tensor, torch.Tensor]:
	"""
		MLA prefifill on hooper device.
		Materialize QKV and call flash_attn_varlen_func(). (flash_attn_3 backend)
	"""
	bsz, seq_len, _ = hidden_states.shape
	# query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	query_states = w8a16_gemm(
		self.q_a_proj.weight.data,
		weight_scale["q_a_proj.weight_scale_inv"],
		hidden_states
	)
	# query_states = fused_fp8_bf16_gemm(hidden_states, self.q_a_proj.weight.data, weight_scale["q_a_proj.weight_scale_inv"])
	# torch.cuda.current_stream().synchronize()
	query_states = self.q_a_layernorm(query_states)
	query_states = w8a16_gemm(
		self.q_b_proj.weight.data,
		weight_scale["q_b_proj.weight_scale_inv"],
		query_states
	)
	# query_states = fused_fp8_bf16_gemm(query_states, self.q_b_proj.weight.data, weight_scale["q_b_proj.weight_scale_inv"])
	# torch.cuda.current_stream().synchronize()

	query_states = query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim)
	q_nope, q_pe = torch.split(
		query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)	
	# compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	compressed_kv = w8a16_gemm(
		self.kv_a_proj_with_mqa.weight.data,
		weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
		hidden_states
	)
	# compressed_kv = fused_fp8_bf16_gemm(hidden_states, self.kv_a_proj_with_mqa.weight.data, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"])
	# torch.cuda.current_stream().synchronize()
	
	compressed_kv, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)	
	normed_kv = self.kv_a_layernorm(compressed_kv)
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
	cos, sin = self.rotary_emb(q_pe, seq_len=seq_len)
	q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids, 2)
	k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids, 2)
	k_pe = k_pe.view(bsz, seq_len, self.qk_rope_head_dim)
	offload_kv = torch.cat(
		[normed_kv, k_pe], dim=-1
	)

	# kv = self.kv_b_proj(normed_kv)
	kv = w8a16_gemm(
		self.kv_b_proj.weight.data,
		weight_scale["kv_b_proj.weight_scale_inv"],
		normed_kv
	)
	kv = kv.view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)		

	
	query_states = k_pe.new_empty(bsz, seq_len, self.num_heads, self.q_head_dim)
	query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
	query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

	key_states = k_pe.new_empty(
		bsz, seq_len, self.num_heads, self.q_head_dim
	)
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
	key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
	key_states[:, :, :, self.qk_nope_head_dim :] = k_pe	

	query_states = query_states.contiguous()
	key_states = key_states.contiguous()
	value_states = value_states.contiguous()


	# Call flash_attn_varlen_func
	(
		query_states,
		key_states,
		value_states,
		indices_q,
		cu_seq_lens,
		max_seq_lens,
	) = _upad_input(
		query_states, key_states, value_states, attention_mask, seq_len
	)      	
	cu_seqlens_q, cu_seqlens_k = cu_seq_lens
	max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
	logging.info(f"cu_seqlens_q: {cu_seqlens_q}")
	logging.info(f"cu_seqlens_k: {cu_seqlens_k}")
	logging.info(f"max_seqlen_in_batch_q: {max_seqlen_in_batch_q}")
	logging.info(f"max_seqlen_in_batch_k: {max_seqlen_in_batch_k}")
	logging.info(f"query_states shape: {query_states.shape}, dtype: {query_states.dtype}")
	logging.info(f"key_states shape: {key_states.shape}, dtype: {key_states.dtype}")	
	logging.info(f"value_states shape: {value_states.shape}, dtype: {value_states.dtype}")
	# exit()
	attn_output_unpad = flash_attn_varlen_func(
		query_states,
		key_states,
		value_states,
		cu_seqlens_q=cu_seqlens_q,
		cu_seqlens_k=cu_seqlens_k,
		max_seqlen_q=max_seqlen_in_batch_q,
		max_seqlen_k=max_seqlen_in_batch_k,
		softmax_scale=self.softmax_scale,
		causal=True
	)
	# if attn_output_unpad is a tuple, we use attn_output_unpad[0]
	if isinstance(attn_output_unpad, tuple):
		attn_output_unpad = attn_output_unpad[0]		

	attn_output = pad_input(attn_output_unpad, indices_q, bsz, seq_len).view(
		bsz, seq_len, self.num_heads * self.v_head_dim
	).contiguous()

	# attn_output = self.o_proj(attn_output)
	attn_output = w8a16_gemm(
		self.o_proj.weight.data,
		weight_scale["o_proj.weight_scale_inv"],
		attn_output
	)

	# logging.info(f"offload_kv: {offload_kv.shape}")
	return attn_output, offload_kv


@torch.inference_mode()
def mla_prefill_flashattention3_fused_dequant(
	self,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
	weight_scale: dict[str, torch.Tensor],

) -> tuple[torch.Tensor, torch.Tensor]:
	"""
		MLA prefifill on hooper device.
		Materialize QKV and call flash_attn_varlen_func(). (flash_attn_3 backend)
	"""
	# logging.info(f"Rank {dist.get_rank()} start")
	bsz, seq_len, _ = hidden_states.shape
	# query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	query_states = w8a16_gemm(
		self.q_a_proj.weight.data,
		weight_scale["q_a_proj.weight_scale_inv"],
		hidden_states
	)
	# query_states = fused_fp8_bf16_gemm(hidden_states, self.q_a_proj.weight.data, weight_scale["q_a_proj.weight_scale_inv"])
	# logging.info(f"Rank {dist.get_rank()} first GEMM passed")
	# logging.info(f"query_states: {query_states.shape}")
	# logging.info(f"query_states dtype: {query_states.dtype}")
	# logging.info(f"query_states device: {query_states.device}")
	# logging.info(f"query_states item 0: {query_states[1, 0, 0]}")
	query_states = self.q_a_layernorm(query_states)
	# query_states = w8a16_gemm(
	# 	self.q_b_proj.weight.data,
	# 	weight_scale["q_b_proj.weight_scale_inv"],
	# 	query_states
	# )
	query_states = fused_fp8_bf16_gemm(query_states, self.q_b_proj.weight.data, weight_scale["q_b_proj.weight_scale_inv"])

	query_states = query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim)
	q_nope, q_pe = torch.split(
		query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)	
	cos, sin = self.rotary_emb(q_pe, seq_len=seq_len)
	# compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	compressed_kv = w8a16_gemm(
		self.kv_a_proj_with_mqa.weight.data,
		weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
		hidden_states
	)
	# compressed_kv = fused_fp8_bf16_gemm(hidden_states, self.kv_a_proj_with_mqa.weight.data, weight_scale["kv_a_proj_with_mqa.weight_scale_inv"])
	# torch.cuda.current_stream(query_states.device).synchronize()
	# logging.info(f"Rank {dist.get_rank()} third GEMM passed")
	# logging.info(f"compressed_kv: {compressed_kv.shape}")
	compressed_kv, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)	
	normed_kv = self.kv_a_layernorm(compressed_kv)
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
	# cos, sin = self.rotary_emb(q_pe, seq_len=seq_len)

	# logging.info(f"Rank {dist.get_rank()} cos, sin: {cos.shape}, {sin.shape}")
	q_pe = rotary_pos_emb(q_pe, cos, sin, position_ids, 2)
	k_pe = rotary_pos_emb(k_pe, cos, sin, position_ids, 2)
	k_pe = k_pe.view(bsz, seq_len, self.qk_rope_head_dim)
	offload_kv = torch.cat(
		[normed_kv, k_pe], dim=-1
	)


	# logging.info(f"Rank {dist.get_rank()} offload_kv: {offload_kv.shape}")
	# kv = self.kv_b_proj(normed_kv)
	kv = w8a16_gemm(
		self.kv_b_proj.weight.data,
		weight_scale["kv_b_proj.weight_scale_inv"],
		normed_kv
	)
	# kv = fused_fp8_bf16_gemm(normed_kv, self.kv_b_proj.weight.data, weight_scale["kv_b_proj.weight_scale_inv"])
	# logging.info(f"Rank {dist.get_rank()} fourth GEMM passed")
	# logging.info(f"kv: {kv.shape}")
	kv = kv.view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)		

	
	query_states = k_pe.new_empty(bsz, seq_len, self.num_heads, self.q_head_dim)
	query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
	query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

	key_states = k_pe.new_empty(
		bsz, seq_len, self.num_heads, self.q_head_dim
	)
	k_pe = k_pe.view(bsz, seq_len, 1, self.qk_rope_head_dim)
	key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
	key_states[:, :, :, self.qk_nope_head_dim :] = k_pe	

	query_states = query_states.contiguous()
	key_states = key_states.contiguous()
	value_states = value_states.contiguous()


	# Call flash_attn_varlen_func
	(
		query_states,
		key_states,
		value_states,
		indices_q,
		cu_seq_lens,
		max_seq_lens,
	) = _upad_input(
		query_states, key_states, value_states, attention_mask, seq_len
	)      	
	cu_seqlens_q, cu_seqlens_k = cu_seq_lens
	max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
	attn_output_unpad = flash_attn_varlen_func(
		query_states,
		key_states,
		value_states,
		cu_seqlens_q=cu_seqlens_q,
		cu_seqlens_k=cu_seqlens_k,
		max_seqlen_q=max_seqlen_in_batch_q,
		max_seqlen_k=max_seqlen_in_batch_k,
		softmax_scale=self.softmax_scale,
		causal=True
	)		

	attn_output = pad_input(attn_output_unpad, indices_q, bsz, seq_len).view(
		bsz, seq_len, self.num_heads * self.v_head_dim
	).contiguous()

	# attn_output = self.o_proj(attn_output)
	attn_output = w8a16_gemm(
		self.o_proj.weight.data,
		weight_scale["o_proj.weight_scale_inv"],
		attn_output
	)
	# attn_output = fused_fp8_bf16_gemm(attn_output, self.o_proj.weight.data, weight_scale["o_proj.weight_scale_inv"])
	# logging.info(f"Rank {dist.get_rank()} fifth GEMM passed")
	# logging.info(f"attn_output: {attn_output.shape}")
	# logging.info(f"offload_kv: {offload_kv.shape}")
	return attn_output, offload_kv




