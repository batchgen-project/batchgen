# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
	AttentionMaskConverter,
	_prepare_4d_attention_mask,
	_prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
	BaseModelOutputWithPast,
	CausalLMOutputWithPast,
	SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
	ALL_LAYERNORM_LAYERS,
	is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	is_flash_attn_2_available,
	is_flash_attn_greater_or_equal_2_10,
	logging,
	replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_deepseek_v3 import DeepseekV3Config
import torch.distributed as dist
import numpy as np
import os
import triton
import gc

if is_flash_attn_2_available():
	from flash_attn import flash_attn_func, flash_attn_varlen_func
	from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
	if not is_torch_greater_or_equal_than_1_13:
		import torch.fx

	_prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV3Config"


def _get_unpad_data(attention_mask):
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


# class DeepseekV3RMSNorm(nn.Module):
# 	def __init__(self, hidden_size, eps=1e-6):
# 		"""
# 		DeepseekV3RMSNorm is equivalent to T5LayerNorm
# 		"""
# 		super().__init__()
# 		self.weight = nn.Parameter(torch.ones(hidden_size))
# 		self.variance_epsilon = eps

# 	def forward(self, hidden_states):
# 		input_dtype = hidden_states.dtype
# 		hidden_states = hidden_states.to(torch.float32)
# 		variance = hidden_states.pow(2).mean(-1, keepdim=True)
# 		hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
# 		return self.weight * hidden_states.to(input_dtype)

# class DeepseekV3RMSNorm(nn.Module):
# 	def __init__(self, hidden_size, eps=1e-6):
# 		"""
# 		DeepseekV3RMSNorm is equivalent to T5LayerNorm
# 		"""
# 		super().__init__()
# 		self.weight = nn.Parameter(torch.ones(hidden_size))
# 		self.variance_epsilon = eps
# 		self.dim = hidden_size

# 	def forward(self, hidden_states):
# 		# logger.info(f"Hidden states dtype: {hidden_states.dtype}")
# 		# logger.info(f"Weight dtype: {self.weight.dtype}")
# 		return F.rms_norm(hidden_states, (self.dim,), self.weight, self.variance_epsilon)

# from batchgen.other_kernels.fused_rmsnorm import fused_rmsnorm_func
from mgn_kernel import fused_rmsnorm

class DeepseekV3RMSNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-6):
		"""
		DeepseekV3RMSNorm is equivalent to T5LayerNorm
		"""
		super().__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.variance_epsilon = eps
		self.dim = hidden_size

	def forward(self, hidden_states):
		# return fused_rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)
		return fused_rmsnorm(hidden_states, self.weight, self.variance_epsilon)

ALL_LAYERNORM_LAYERS.append(DeepseekV3RMSNorm)


# class DeepseekV3RotaryEmbedding(nn.Module):
# 	def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
# 		super().__init__()

# 		self.dim = dim
# 		self.max_position_embeddings = max_position_embeddings
# 		self.base = base
# 		inv_freq = 1.0 / (
# 			self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
# 		)
# 		self.register_buffer("inv_freq", inv_freq, persistent=False)

# 		# Build here to make `torch.jit.trace` work.
# 		self._set_cos_sin_cache(
# 			seq_len=max_position_embeddings,
# 			device=self.inv_freq.device,
# 			dtype=torch.get_default_dtype(),
# 		)
# 		self.max_seq_len_cached = None

# 	def _set_cos_sin_cache(self, seq_len, device, dtype):
# 		self.max_seq_len_cached = seq_len
# 		t = torch.arange(
# 			self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
# 		)

# 		freqs = torch.outer(t, self.inv_freq.to(t.device))
# 		# Different from paper, but it uses a different permutation in order to obtain the same calculation
# 		emb = torch.cat((freqs, freqs), dim=-1)
# 		self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
# 		self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

# 	def forward(self, x, seq_len=None):
# 		# x: [bs, num_attention_heads, seq_len, head_size]
# 		if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
# 			self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

# 		return (
# 			self.cos_cached[:seq_len].to(dtype=x.dtype),
# 			self.sin_cached[:seq_len].to(dtype=x.dtype),
# 		)

class DeepseekV3RotaryEmbedding(nn.Module):
	def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
		super().__init__()

		self.dim = dim
		self.max_position_embeddings = max_position_embeddings
		self.base = base
		
		# Store the original device
		self.init_device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		# Calculate on specified device and don't move it around
		inv_freq = 1.0 / (
			self.base ** (torch.arange(0, self.dim, 2, device=self.init_device).float() / self.dim)
		)
		self.register_buffer("inv_freq", inv_freq, persistent=False)

		# Build here to make `torch.jit.trace` work.
		self._set_cos_sin_cache(
			seq_len=max_position_embeddings,
			device=self.init_device,
			dtype=torch.get_default_dtype(),
		)
		# self.max_seq_len_cached = None

	def _set_cos_sin_cache(self, seq_len, device, dtype):
		self.max_seq_len_cached = seq_len
		
		# Ensure we're creating on the requested device
		t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
		
		# Move inv_freq to match t's device
		inv_freq_on_device = self.inv_freq.to(device)
		
		# Compute freqs
		freqs = torch.outer(t, inv_freq_on_device)
		
		# Different from paper, but it uses a different permutation in order to obtain the same calculation
		emb = torch.cat((freqs, freqs), dim=-1)
		
		# Register buffers with correct dtype
		self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
		self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

	def forward(self, x, seq_len=None):
		# x: [bs, num_attention_heads, seq_len, head_size]
		
		# Default seq_len if not provided
		if seq_len is None:
			seq_len = x.size(-2)  # Assuming seq_len dimension is -2
			
		# Check if we need to recompute the cache
		if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
			# Make sure to set cache on the same device as x
			logger.warning(f"Recomputing cos/sin cache for rotary embedding. THIS SHOULD NOT HAPPEN. seq_len: {seq_len}, max_seq_len_cached: {self.max_seq_len_cached}")
			self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
		
		# Return cached values, ensuring they're on the right device and dtype
		return (
			self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
			self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
		)


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->DeepseekV3
class DeepseekV3LinearScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
	"""DeepseekV3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

	def __init__(
		self,
		dim,
		max_position_embeddings=2048,
		base=10000,
		device=None,
		scaling_factor=1.0,
	):
		self.scaling_factor = scaling_factor
		super().__init__(dim, max_position_embeddings, base, device)

	def _set_cos_sin_cache(self, seq_len, device, dtype):
		self.max_seq_len_cached = seq_len
		t = torch.arange(
			self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
		)
		t = t / self.scaling_factor

		freqs = torch.outer(t, self.inv_freq)
		# Different from paper, but it uses a different permutation in order to obtain the same calculation
		emb = torch.cat((freqs, freqs), dim=-1)
		self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
		self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->DeepseekV3
class DeepseekV3DynamicNTKScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
	"""DeepseekV3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

	def __init__(
		self,
		dim,
		max_position_embeddings=2048,
		base=10000,
		device=None,
		scaling_factor=1.0,
	):
		self.scaling_factor = scaling_factor
		super().__init__(dim, max_position_embeddings, base, device)

	def _set_cos_sin_cache(self, seq_len, device, dtype):
		self.max_seq_len_cached = seq_len

		if seq_len > self.max_position_embeddings:
			base = self.base * (
				(self.scaling_factor * seq_len / self.max_position_embeddings)
				- (self.scaling_factor - 1)
			) ** (self.dim / (self.dim - 2))
			inv_freq = 1.0 / (
				base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
			)
			self.register_buffer("inv_freq", inv_freq, persistent=False)

		t = torch.arange(
			self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
		)

		freqs = torch.outer(t, self.inv_freq)
		# Different from paper, but it uses a different permutation in order to obtain the same calculation
		emb = torch.cat((freqs, freqs), dim=-1)
		self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
		self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
	num_rotations, dim, base=10000, max_position_embeddings=2048
):
	return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
		2 * math.log(base)
	)


# Find dim range bounds based on rotations
def yarn_find_correction_range(
	low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
	low = math.floor(
		yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
	)
	high = math.ceil(
		yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
	)
	return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
	if scale <= 1:
		return 1.0
	return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
	if min == max:
		max += 0.001  # Prevent singularity

	linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
	ramp_func = torch.clamp(linear_func, 0, 1)
	return ramp_func


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):

	def __init__(
		self,
		dim,
		max_position_embeddings=2048,
		base=10000,
		device=None,
		scaling_factor=1.0,
		original_max_position_embeddings=4096,
		beta_fast=32,
		beta_slow=1,
		mscale=1,
		mscale_all_dim=0,
	):
		self.scaling_factor = scaling_factor
		self.original_max_position_embeddings = original_max_position_embeddings
		self.beta_fast = beta_fast
		self.beta_slow = beta_slow
		self.mscale = mscale
		self.mscale_all_dim = mscale_all_dim
		super().__init__(dim, max_position_embeddings, base, device)

	def _set_cos_sin_cache(self, seq_len, device, dtype):
		# logger.warning(f"Recomputing cos/sin cache for yarn rotary embedding. THIS SHOULD NOT HAPPEN. seq_len: {seq_len}")
		self.max_seq_len_cached = seq_len
		dim = self.dim

		freq_extra = 1.0 / (
			self.base
			** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
		)
		freq_inter = 1.0 / (
			self.scaling_factor
			* self.base
			** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
		)

		low, high = yarn_find_correction_range(
			self.beta_fast,
			self.beta_slow,
			dim,
			self.base,
			self.original_max_position_embeddings,
		)
		inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
			device=device, dtype=torch.float32
		)
		inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
		self.register_buffer("inv_freq", inv_freq, persistent=False)

		t = torch.arange(seq_len, device=device, dtype=torch.float32)

		freqs = torch.outer(t, inv_freq)

		_mscale = float(
			yarn_get_mscale(self.scaling_factor, self.mscale)
			/ yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
		)

		emb = torch.cat((freqs, freqs), dim=-1)
		self.register_buffer(
			"cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
		)
		self.register_buffer(
			"sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
		)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
	"""Applies Rotary Position Embedding to the query and key tensors.

	Args:
		q (`torch.Tensor`): The query tensor.
		k (`torch.Tensor`): The key tensor.
		cos (`torch.Tensor`): The cosine part of the rotary embedding.
		sin (`torch.Tensor`): The sine part of the rotary embedding.
		position_ids (`torch.Tensor`):
			The position indices of the tokens corresponding to the query and key tensors. For example, this can be
			used to pass offsetted position ids when working with a KV-cache.
		unsqueeze_dim (`int`, *optional*, defaults to 1):
			The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
			sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
			that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
			k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
			cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
			the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
	Returns:
		`tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
	"""
	cos = cos[position_ids].unsqueeze(unsqueeze_dim)
	sin = sin[position_ids].unsqueeze(unsqueeze_dim)

	b, h, s, d = q.shape
	q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

	b, h, s, d = k.shape
	k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

	q_embed = (q * cos) + (rotate_half(q) * sin)
	k_embed = (k * cos) + (rotate_half(k) * sin)
	return q_embed, k_embed

from ....moe.fused_dequant_gemm import fused_fp8_bf16_gemm
from ....attention.mla.fa3_backend import w8a16_gemm
class DeepseekV3MLP(nn.Module):
	def __init__(self, config, hidden_size=None, intermediate_size=None):
		super().__init__()
		self.config = config
		self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
		self.intermediate_size = (
			config.intermediate_size if intermediate_size is None else intermediate_size
		)

		self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
		self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
		self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
		self.act_fn = ACT2FN[config.hidden_act]

	@torch.inference_mode()
	def forward(self, x):
		down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		return down_proj

	@torch.inference_mode
	def fused_fp8_forward(self, x, scale, out=None, offset=None):
		up = fused_fp8_bf16_gemm(x, self.up_proj.weight.data, scale['up_proj.weight_scale_inv'], block_size = [64, 16, 64])
		gate = fused_fp8_bf16_gemm(x, self.gate_proj.weight.data, scale['gate_proj.weight_scale_inv'], block_size = [64, 16, 64])
		intermediate = self.act_fn(gate) * up
		if out is not None:
			fused_fp8_bf16_gemm(intermediate, self.down_proj.weight.data, scale['down_proj.weight_scale_inv'], out=out, offset=offset, block_size = [64, 16, 64])
		else:
			res = fused_fp8_bf16_gemm(intermediate, self.down_proj.weight.data, scale['down_proj.weight_scale_inv'], block_size = [64, 16, 32])
			return res
	
	@torch.inference_mode
	def deepgemm_forward(self, x, scale):
		up = w8a16_gemm(self.up_proj.weight.data, scale['up_proj.weight_scale_inv'], x)
		gate = w8a16_gemm(self.gate_proj.weight.data, scale['gate_proj.weight_scale_inv'], x)
		intermediate = self.act_fn(gate) * up
		return w8a16_gemm(self.down_proj.weight.data, scale['down_proj.weight_scale_inv'], intermediate)


# torch.set_float32_matmul_precision('highest')
# import torch._dynamo.config as dynamo_config
# import os

# # Set up persistent disk cache
# cache_dir = "./torch_compile_cache"
# os.makedirs(cache_dir, exist_ok=True)

# # Configure disk caching
# dynamo_config.cache_dir = cache_dir
# dynamo_config.accumulated_cache_size_limit = 1024  # MB
# dynamo_config.cache_size_limit = 1024

# Enable FX graph caching  
# torch._dynamo.config.fx_graph_cache = True


@torch.inference_mode()
@torch.compile(mode="max-autotune", fullgraph=True, disable=True, backend="inductor")
def compiled_moe_gate_forward(hidden_states, weight, e_score_correction_bias, 
							 n_group, topk_group, n_routed_experts, top_k, 
							 routed_scaling_factor):
	bsz, seq_len, h = hidden_states.shape
	### compute gating score
	hidden_states = hidden_states.view(-1, h)
	logits = F.linear(
		hidden_states.type(torch.float32), weight.type(torch.float32), None
	)
	scores = logits.sigmoid()

	### select top-k experts
	scores_for_choice = scores.view(bsz * seq_len, -1) + e_score_correction_bias.unsqueeze(0)
	group_scores = (
		scores_for_choice.view(bsz * seq_len, n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
	)  # [n, n_group]
	group_idx = torch.topk(
		group_scores, k=topk_group, dim=-1, sorted=False
	)[
		1
	]  # [n, top_k_group]
	group_mask = torch.zeros_like(group_scores)  # [n, n_group]
	group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
	score_mask = (
		group_mask.unsqueeze(-1)
		.expand(
			bsz * seq_len, n_group, n_routed_experts // n_group
		)
		.reshape(bsz * seq_len, -1)
	)  # [n, e]
	tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
	_, topk_idx = torch.topk(
		tmp_scores, k=top_k, dim=-1, sorted=False
	)
	topk_weight = scores.gather(1, topk_idx)

	denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
	topk_weight = topk_weight / denominator
	topk_weight = topk_weight * routed_scaling_factor # must multiply the scaling factor

	# return topk_idx, topk_weight.to(hidden_states.dtype)
	return topk_idx, topk_weight

from torch._inductor import config as ind_config
# ind_config.triton.force_cudagraphs_warmup = True
# ind_config.triton.cudagraphs = True
# ind_config.triton.cudagraphs
def warmup_compiled_moe_gate(device):
	
	dummy_weight = nn.Parameter(
		torch.zeros(256, 7168, dtype=torch.bfloat16, device=device)
	)
	dummy_e_score_correction_bias = torch.nn.Parameter(
		torch.zeros(256, dtype=torch.float32, device=device)
	)
	
	with torch.inference_mode(): 
		for t in range(5):
			dummy_hidden_states = torch.randn(128, 1, 7168, dtype=torch.bfloat16, device=device)
			_ = compiled_moe_gate_forward(
				dummy_hidden_states, 
				dummy_weight,
				dummy_e_score_correction_bias,
				8, 
				4, 
				256, 
				8, 
				2.5
			)
			torch.cuda.synchronize(device=device)

from torch.utils.cpp_extension import load
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, "..", "..", "..", "test", "fused_moe_gate.cu")
parallel_moe = load(
    name="parallel_moe_gate",
    sources=[source_dir],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True
)
class MoEGate(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.top_k = config.num_experts_per_tok
		self.n_routed_experts = config.n_routed_experts
		self.routed_scaling_factor = config.routed_scaling_factor
		self.scoring_func = config.scoring_func
		self.topk_method = config.topk_method
		self.n_group = config.n_group
		self.topk_group = config.topk_group

		# topk selection algorithm
		self.norm_topk_prob = config.norm_topk_prob
		self.gating_dim = config.hidden_size
		self.weight = nn.Parameter(
			torch.empty((self.n_routed_experts, self.gating_dim))
		)
		if self.topk_method == "noaux_tc":
			self.e_score_correction_bias = nn.Parameter(
				torch.empty((self.n_routed_experts))
			)
		self.reset_parameters()
		# self.device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
		# self.input_buf = torch.empty(128,1,7168, dtype=torch.bfloat16, device=self.device)

	@torch.inference_mode()
	def warmup(self):	
		# with torch.inference_mode():
		# 	for t in range(5):
		# 		_ = compiled_moe_gate_forward(
		# 			self.input_buf, 
		# 			self.weight,
		# 			self.e_score_correction_bias,
		# 			self.n_group, 
		# 			self.topk_group, 
		# 			self.n_routed_experts, 
		# 			self.top_k, 
		# 			self.routed_scaling_factor
		# 		)
		pass

	def reset_parameters(self) -> None:
		import torch.nn.init as init

		init.kaiming_uniform_(self.weight, a=math.sqrt(5))

	def forward(self, hidden_states):
		# log self.weight and self.e_score_correction_bias shape and dtype
		bsz, seq_len, h = hidden_states.shape
		### compute gating score
		hidden_states = hidden_states.view(-1, h)
		logits = F.linear(
			hidden_states.type(torch.float32), self.weight.type(torch.float32), None
		)
		if self.scoring_func == "sigmoid":
			scores = logits.sigmoid()
		else:
			raise NotImplementedError(
				f"insupportable scoring function for MoE gating: {self.scoring_func}"
			)

		### select top-k experts
		if self.topk_method == "noaux_tc":
			assert not self.training
			scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
			group_scores = (
				scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
			)  # [n, n_group]
			group_idx = torch.topk(
				group_scores, k=self.topk_group, dim=-1, sorted=False
			)[
				1
			]  # [n, top_k_group]
			group_mask = torch.zeros_like(group_scores)  # [n, n_group]
			group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
			score_mask = (
				group_mask.unsqueeze(-1)
				.expand(
					bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
				)
				.reshape(bsz * seq_len, -1)
			)  # [n, e]
			tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
			_, topk_idx = torch.topk(
				tmp_scores, k=self.top_k, dim=-1, sorted=False
			)
			topk_weight = scores.gather(1, topk_idx)
		else:
			raise NotImplementedError(
				f"insupportable TopK function for MoE gating: {self.topk_method}"
			)

		### norm gate to sum 1
		if self.top_k > 1 and self.norm_topk_prob:
			denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
			topk_weight = topk_weight / denominator
		topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor

		return topk_idx, topk_weight
	
	# @torch.compile(fullgraph=True, mode="reduce-overhead")
	# @torch.compile(mode="reduce-overhead", backend="inductor")
	@torch.inference_mode()
	@torch.compile(mode="max-autotune", backend="inductor")
	def _decoding_forward_compiled(self, hidden_states, weight, e_score_correction_bias, 
							  n_group, topk_group, n_routed_experts, top_k, 
							  routed_scaling_factor):
		bsz, seq_len, h = hidden_states.shape
		### compute gating score
		hidden_states = hidden_states.view(-1, h)
		logits = F.linear(
			hidden_states.type(torch.float32), weight.type(torch.float32), None
		)
		scores = logits.sigmoid()

		### select top-k experts
		scores_for_choice = scores.view(bsz * seq_len, -1) + e_score_correction_bias.unsqueeze(0)
		group_scores = (
			scores_for_choice.view(bsz * seq_len, n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
		)  # [n, n_group]
		group_idx = torch.topk(
			group_scores, k=topk_group, dim=-1, sorted=False
		)[
			1
		]  # [n, top_k_group]
		group_mask = torch.zeros_like(group_scores)  # [n, n_group]
		group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
		score_mask = (
			group_mask.unsqueeze(-1)
			.expand(
				bsz * seq_len, n_group, n_routed_experts // n_group
			)
			.reshape(bsz * seq_len, -1)
		)  # [n, e]
		tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
		_, topk_idx = torch.topk(
			tmp_scores, k=top_k, dim=-1, sorted=False
		)
		topk_weight = scores.gather(1, topk_idx)

		denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
		topk_weight = topk_weight / denominator
		topk_weight = topk_weight * routed_scaling_factor # must multiply the scaling factor

		return topk_idx, topk_weight.to(hidden_states.dtype)
	
	@torch.inference_mode()
	def moe_gate_forward_hybrid(self, hidden_states):
		"""Hybrid: PyTorch matmul + sigmoid, then custom kernel"""
		bsz, seq_len, h = hidden_states.shape
		
		# PyTorch handles heavy lifting
		hidden_states_flat = hidden_states.view(-1, h)
		logits = F.linear(hidden_states_flat.float(), self.weight.float(), None)
		scores = torch.sigmoid(logits)
		
		# Custom kernel handles MoE routing
		topk_idx, topk_weight = parallel_moe.forward(
			scores,
			self.e_score_correction_bias,
			self.n_group,
			self.topk_group,
			self.n_routed_experts,
			self.top_k,
			self.routed_scaling_factor
		)
		
		return topk_idx, topk_weight
	
	@torch.inference_mode()
	def _decoding_forward(self, hidden_states):
		bsz, seq_len, h = hidden_states.shape
		### compute gating score
		hidden_states = hidden_states.view(-1, h)
		logits = F.linear(
			hidden_states.type(torch.float32), self.weight.type(torch.float32), None
		)
		scores = logits.sigmoid()

		### select top-k experts
		scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
		group_scores = (
			scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
		)  # [n, n_group]
		group_idx = torch.topk(
			group_scores, k=self.topk_group, dim=-1, sorted=False
		)[
			1
		]  # [n, top_k_group]
		group_mask = torch.zeros_like(group_scores)  # [n, n_group]
		group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
		score_mask = (
			group_mask.unsqueeze(-1)
			.expand(
				bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
			)
			.reshape(bsz * seq_len, -1)
		)  # [n, e]
		tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
		_, topk_idx = torch.topk(
			tmp_scores, k=self.top_k, dim=-1, sorted=False
		)
		topk_weight = scores.gather(1, topk_idx)

		denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
		topk_weight = topk_weight / denominator
		topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor

		return topk_idx, topk_weight.to(hidden_states.dtype)
	
	@torch.inference_mode()
	def decoding_forward(self, hidden_states):
		# out = self._decoding_forward_compiled(
		# 	hidden_states, self.weight, self.e_score_correction_bias,
		# 	self.n_group, self.topk_group, self.n_routed_experts, 
		# 	self.top_k, self.routed_scaling_factor
		# )
		# return out

		# logger.warning_once(f"MoE Gate weight shape: {self.weight.shape}, dtype: {self.weight.dtype}")
		# if hasattr(self, 'e_score_correction_bias'):
		# 	logger.warning_once(f"MoE Gate e_score_correction_bias shape: {self.e_score_correction_bias.shape}, dtype: {self.e_score_correction_bias.dtype}")
	
		# self.input_buf.copy_(hidden_states)
		return compiled_moe_gate_forward(
			hidden_states, self.weight, self.e_score_correction_bias,
			self.n_group, self.topk_group, self.n_routed_experts, 
			self.top_k, self.routed_scaling_factor
		)

import triton
import triton.language as tl
@triton.jit
def moe_fp32_accum_kernel_v2(
    outs_ptr,
    inv_idxs_ptr,
    topk_weights_ptr,
    output_ptr,
    total_tokens: tl.constexpr,
    topk: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized version with better memory access patterns.
    Each block processes multiple tokens to improve memory coalescing.
    """
    # Program handles a block of tokens and a chunk of hidden dims
    token_block_id = tl.program_id(0)
    h_block_id = tl.program_id(1)
    
    # Token range for this block
    TOKENS_PER_BLOCK: tl.constexpr = 4
    token_start = token_block_id * TOKENS_PER_BLOCK
    
    # Hidden dim range
    h_start = h_block_id * BLOCK_SIZE
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE)
    h_mask = h_offsets < hidden_dim
    
    # Process each token in this block
    for t_idx in range(TOKENS_PER_BLOCK):
        token_id = token_start + t_idx
        
        # Check if this token is valid
        if token_id < total_tokens:
            # Accumulator for this token
            accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            # Accumulate over topk experts
            for k in range(topk):
                new_x_idx = token_id * topk + k
                outs_idx = tl.load(inv_idxs_ptr + new_x_idx)
                
                # topk_weights is [total_tokens, topk], need 2D indexing
                weight_offset = token_id * topk + k
                weight = tl.load(topk_weights_ptr + weight_offset).to(tl.float32)
                
                outs_offsets = outs_idx * hidden_dim + h_offsets
                expert_out = tl.load(outs_ptr + outs_offsets, mask=h_mask, other=0.0)
                accum += expert_out.to(tl.float32) * weight
            
            # Store result
            output_offsets = token_id * hidden_dim + h_offsets
            tl.store(output_ptr + output_offsets, accum.to(output_ptr.dtype.element_ty), mask=h_mask)


def moe_fp32_accum_triton_v2(
    outs: torch.Tensor,
    idxs: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Version 2 with better memory coalescing."""
    total_tokens, topk = topk_weights.shape
    hidden_dim = outs.shape[1]
    
    # Create inverse index
    inv_idxs = torch.empty_like(idxs)
    inv_idxs[idxs] = torch.arange(len(idxs), device=idxs.device, dtype=idxs.dtype)
    
    output = torch.empty((total_tokens, hidden_dim), device=outs.device, dtype=outs.dtype)
    
    BLOCK_SIZE = 128
    TOKENS_PER_BLOCK = 4
    
    grid = lambda META: (
        triton.cdiv(total_tokens, TOKENS_PER_BLOCK),
        triton.cdiv(hidden_dim, META['BLOCK_SIZE'])
    )
    
    moe_fp32_accum_kernel_v2[grid](
        outs, inv_idxs, topk_weights, output,
        total_tokens=total_tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

class DeepseekV3MoE_Prefill(nn.Module):
	"""
	A mixed expert module containing shared experts.
	"""

	def __init__(self, config, comm=None):
		super().__init__()
		# logger.info("Initializing DeepseekV3MoE_Prefill")
		self.config = config
		self.num_experts_per_tok = config.num_experts_per_tok

		if hasattr(config, "ep_size") and config.ep_size > 1:
			assert config.ep_size == dist.get_world_size()
			self.ep_size = config.ep_size
			self.experts_per_rank = config.n_routed_experts // config.ep_size
			self.ep_rank = dist.get_rank()
			self.experts = nn.ModuleList(
				[
					(
						DeepseekV3MLP(
							config, intermediate_size=config.moe_intermediate_size
						)
						if i >= self.ep_rank * self.experts_per_rank
						and i < (self.ep_rank + 1) * self.experts_per_rank
						else None
					)
					for i in range(config.n_routed_experts)
				]
			)
		else:
			self.ep_size = 1
			self.experts_per_rank = config.n_routed_experts
			self.ep_rank = 0
			self.experts = nn.ModuleList(
				[
					DeepseekV3MLP(
						config, intermediate_size=config.moe_intermediate_size
					)
					for i in range(config.n_routed_experts)
				]
			)
		self.gate = MoEGate(config)
		if config.n_shared_experts is not None:
			intermediate_size = config.moe_intermediate_size * config.n_shared_experts
			self.shared_experts = DeepseekV3MLP(
				config=config, intermediate_size=intermediate_size
			)

	def forward(self, hidden_states):
		identity = hidden_states
		orig_shape = hidden_states.shape
		topk_idx, topk_weight = self.gate(hidden_states)
		hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
		flat_topk_idx = topk_idx.view(-1)
		if not self.training:
			y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
		if self.config.n_shared_experts is not None:
			y = y + self.shared_experts(identity)
			# accumulate in fp32
			# y = y.to(torch.float32) + self.shared_experts(identity).to(torch.float32)
			# y = y.type(identity.dtype)
		return y



	@torch.no_grad()
	def moe_infer(self, x, topk_ids, topk_weight):
		cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
		cnts.scatter_(1, topk_ids, 1)
		tokens_per_expert = cnts.sum(dim=0)
		idxs = topk_ids.view(-1).argsort()
		sorted_tokens = x[idxs // topk_ids.shape[1]]
		sorted_tokens_shape = sorted_tokens.shape
		if self.ep_size > 1:
			tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
			tokens_per_expert_group = tokens_per_expert.new_empty(
				tokens_per_expert.shape[0]
			)
			dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
			output_splits = (
				tokens_per_expert_group.view(self.ep_size, -1)
				.sum(1)
				.cpu()
				.numpy()
				.tolist()
			)
			gathered_tokens = sorted_tokens.new_empty(
				tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
			)
			input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
			dist.all_to_all(
				list(gathered_tokens.split(output_splits)),
				list(sorted_tokens.split(input_split_sizes)),
			)
			tokens_per_expert_post_gather = tokens_per_expert_group.view(
				self.ep_size, self.experts_per_rank
			).sum(dim=0)
			gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
			s = 0
			for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
				gatherd_idxs[s : s + k] = i % self.experts_per_rank
				s += k
			gatherd_idxs = gatherd_idxs.argsort()
			sorted_tokens = gathered_tokens[gatherd_idxs]
			tokens_per_expert = tokens_per_expert_post_gather
		tokens_per_expert = tokens_per_expert.cpu().numpy()

		outputs = []
		start_idx = 0
		for i, num_tokens in enumerate(tokens_per_expert):
			end_idx = start_idx + num_tokens
			if num_tokens == 0:
				continue
			expert = self.experts[i + self.ep_rank * self.experts_per_rank]
			tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
			expert_out = expert(tokens_for_this_expert)
			outputs.append(expert_out)
			start_idx = end_idx

		outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
		if self.ep_size > 1:
			new_x = torch.empty_like(outs)
			new_x[gatherd_idxs] = outs
			gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
			dist.all_to_all(
				list(gathered_tokens.split(input_split_sizes)),
				list(new_x.split(output_splits)),
			)
			outs = gathered_tokens

		# new_x = torch.empty_like(outs)
		# new_x[idxs] = outs
		# topk_weight = topk_weight.to(torch.bfloat16)
		# final_out = (
		# 	new_x.view(*topk_ids.shape, -1)
		# 	.mul_(topk_weight.unsqueeze(dim=-1))
		# 	.sum(dim=1)
		# 	.type(new_x.dtype)
		# )

		# new_x = torch.empty_like(outs)
		# new_x[idxs] = outs
		# assert topk_weight.dtype == torch.float32
		# topk_weight = topk_weight.to(new_x.dtype)
		# final_out = (
		# 	new_x.view(*topk_ids.shape, -1)
		# 	.type(topk_weight.dtype)
		# 	.mul_(topk_weight.unsqueeze(dim=-1))
		# 	.sum(dim=1)
		# 	.type(new_x.dtype)
		# )

		final_out = moe_fp32_accum_triton_v2(
			outs, idxs, topk_weight
		)
		return final_out

from ....moe.fused_grouped_dequant_gemm import (
	fused_dequant_grouped_gemm_bf16_fp8_triton,
	fused_dequant_grouped_gemm_bf16_fp8_triton_v2,
	fused_dequant_grouped_gemm_fp8_fp8_triton
)
from ....moe.fused_dequant_moe import (
	fused_dequant_weighted_moe_stage_1, 
	fused_fp8_moe_stage_1,
	fused_fp8_moe_stage_1_optimized
)
from ....attention.mla.fa3_backend import act_quant
class DeepseekV3MoE_Decoding(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.num_experts_per_tok = config.num_experts_per_tok

		# --- distributed/world metadata -------------------------------------
		if not dist.is_initialized():
			self.rank, self.world_size = 0, 1
		else:
			self.rank        = dist.get_rank()
			self.world_size  = dist.get_world_size()

		self.experts_per_rank   = 256 // self.world_size
		self.total_experts      = self.world_size * self.experts_per_rank
		self.routed_expert_start_idx = self.rank * self.experts_per_rank
		self.routed_expert_end_idx   = (self.rank + 1) * self.experts_per_rank

		# --- experts, gate, shared MLP --------------------------------------
		self.experts = nn.ModuleList([
			DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
			for _ in range(self.total_experts)
		])
		self.gate = MoEGate(config)
		if config.n_shared_experts:
			self.shared_experts = DeepseekV3MLP(
				config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
			)		
		# --- 🔑  pre-allocate the tiny communication buffers ---------------
		self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
		# They are the same size every call, so we keep them as buffers
		self.register_buffer(
			"send_counts_buf",
			torch.zeros(self.world_size, dtype=torch.int64)
		)
		self.register_buffer(
			"recv_counts_buf",
			torch.zeros(self.world_size, dtype=torch.int64)
		)
		self.act_fn = ACT2FN[config.hidden_act]

	def forward(self, hidden_states):
		identity = hidden_states
		orig_shape = hidden_states.shape
		topk_idx, topk_weight = self.gate(hidden_states)
		hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
		out = self.moe_infer(hidden_states, topk_idx, topk_weight)
		out = out.view(*orig_shape)
		out = out + self.shared_experts(identity)
		return out

	@torch.no_grad()
	def moe_infer(self, x, topk_idx, topk_weight):
		num_tokens, hidden_size = x.shape
		K = self.num_experts_per_tok
		device = x.device

		# ---- 1) flatten, sort by expert ------------------------------------
		flat_eids   = topk_idx.flatten()
		flat_wts    = topk_weight.flatten()
		expanded_x  = x.repeat_interleave(K, dim=0)
		# token_idx   = torch.arange(num_tokens, device=device).repeat_interleave(K)

		sorted_eids, sort_idx = flat_eids.sort()
		sorted_x   = expanded_x[sort_idx]
		# sorted_tok = token_idx[sort_idx]
		sorted_wt  = flat_wts[sort_idx]

		# ---- 2) fill the pre-allocated send_counts tensor ------------------
		local_counts = torch.bincount(sorted_eids, minlength=self.total_experts)
		sc = self.send_counts_buf               # alias for readability
		rc = self.recv_counts_buf

		reshaped_counts = local_counts.view(self.world_size, -1)
		sc = reshaped_counts.sum(dim=1)

		gathered_counts = [torch.zeros_like(sc) for _ in range(self.world_size)]
		# ---- 3) first all-to-all (counts) on its own stream ---------------
		dist.all_gather(gathered_counts, sc)
		gathered_tensor = torch.stack(gathered_counts)  # Shape: [world_size, world_size]
		rc = gathered_tensor[:, self.rank]  # Extract column for this rank
		recv_total = rc.sum()  # Keep as tensor until final use		
		# sc_cpu = sc.to("cpu", non_blocking=True)
		# rc_cpu = rc.to("cpu", non_blocking=True)

		# ---- 4) allocate data buffers (variable size) ----------------------
		send_x   = sorted_x
		send_eid = sorted_eids
		recv_x   = torch.empty(recv_total, hidden_size, device=device, dtype=x.dtype)
		recv_eid = torch.empty(recv_total, device=device, dtype=sorted_eids.dtype)

		# convert counts to python lists once – NCCL needs that
		sc_list, rc_list = sc.tolist(), rc.tolist()
		# torch.cuda.current_stream(device).synchronize()
		# sc_list, rc_list = sc_cpu.tolist(), rc_cpu.tolist()

		# ---- 5) main all-to-alls --------------------
		dist.all_to_all_single(recv_x, send_x, rc_list, sc_list)
		dist.all_to_all_single(recv_eid, send_eid, rc_list, sc_list)

		if recv_total:
			recv_eid_sorted, local_sort_idx = recv_eid.sort()
			res = self.grouped_dequant_moe(recv_x[local_sort_idx], recv_eid_sorted)
			recv_x[local_sort_idx] = res

		# ---- 7) all-to-all (return) ---------------------------------------
		out_sorted = torch.empty_like(sorted_x)
		dist.all_to_all_single(out_sorted, recv_x, sc_list, rc_list)

		# ---- 8) unsort, accumulate, normalise -----------------------------
		unsort_idx = sort_idx.argsort()
		final_x   = out_sorted[unsort_idx]
		# final_tok = sorted_tok[unsort_idx]
		final_wt  = sorted_wt[unsort_idx]

		# out_acc = torch.zeros_like(x, dtype=torch.float32)
		# out_acc.index_add_(0, final_tok, final_x.float() * final_wt.unsqueeze(-1))
		final_x = final_x.float() * final_wt.unsqueeze(-1)
		final_x = final_x.view(num_tokens, K, -1)
		final_x = final_x.sum(dim=1)  # sum over experts
		return final_x.to(x.dtype)

		
	def grouped_dequant_moe(self, recv_x, recv_eid):
		# This function assumes that recv_x and recv_eid are already sorted by expert id
		gate_list = []
		up_list = []
		down_list = []
		gate_scale_list = []
		up_scale_list = []
		down_scale_list = []
		for e in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
			gate_list.append(self.experts[e].fp8_gate)
			up_list.append(self.experts[e].fp8_up)
			down_list.append(self.experts[e].fp8_down)
			gate_scale_list.append(self.experts[e].weight_dequant_scale['gate_proj.weight_scale_inv'])
			up_scale_list.append(self.experts[e].weight_dequant_scale['up_proj.weight_scale_inv'])
			down_scale_list.append(self.experts[e].weight_dequant_scale['down_proj.weight_scale_inv'])
		eids = recv_eid - self.routed_expert_start_idx
		counts = torch.bincount(eids, minlength=self.experts_per_rank)
		group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)	
		group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=self.device)
		group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
		group_start_indices[0] = 0  # The first group starts at index 0	

		intermediate = fused_dequant_weighted_moe_stage_1(
			recv_x, gate_list, up_list, gate_scale_list, up_scale_list, group_sizes, group_start_indices
		)	
		
		res = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
			intermediate, down_list, down_scale_list, group_sizes, group_start_indices, gemm_block_size = [64, 16, 64]
		)
		return res

from batchgen.moe.token_permutation.token_permutation_launcher import FusedMoETokenPermutation
# from batchgen.moe.expert_bincount.expert_bincount_launcher import FusedExpertBincount
from mgn_kernel import expert_bincount, fused_moe_token_dispatch, moe_fused_gate
from batchgen.moe.moe_weighted_sum import moe_weighted_sum_triton_v2, moe_weighted_sum_v3
@triton.jit
def scatter_weight_reduce_optimized_kernel(
    # Input pointers
    res_ptr,                    # [nnz, hidden_size]
    nnz_indices_ptr,            # [num_tokens, num_experts_per_tok] - mapping to nnz indices (-1 if empty)
    topk_weight_ptr,            # [num_tokens, num_experts_per_tok]
    # Output pointer
    output_ptr,                 # [num_tokens, hidden_size]
    # Dimensions
    num_tokens,
    num_experts_per_tok,
    hidden_size,
    nnz,                        # Total number of non-zero entries (for bounds checking)
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Optimized version that uses pre-computed inverse mapping.
    This avoids scanning all nnz entries for each token.
    """
    token_idx = tl.program_id(0)
    
    if token_idx >= num_tokens:
        return
    
    h_offset = tl.program_id(1) * BLOCK_SIZE_H
    h_indices = h_offset + tl.arange(0, BLOCK_SIZE_H)
    h_mask = h_indices < hidden_size
    
    accumulator = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    
    # Only loop over experts for this specific token
    for k in range(num_experts_per_tok):
        # Get the nnz index for this token's k-th expert
        mapping_offset = token_idx * num_experts_per_tok + k
        nnz_idx = tl.load(nnz_indices_ptr + mapping_offset)
        
        # Create mask for valid entries (use mask instead of if statement)
        is_valid = (nnz_idx >= 0) & (nnz_idx < nnz)
        
        # Load weight (masked)
        weight = tl.load(topk_weight_ptr + mapping_offset)
        
        # Load result values with proper masking
        # Use tl.where to handle invalid indices safely
        safe_nnz_idx = tl.where(is_valid, nnz_idx, 0)  # Use 0 as safe fallback
        res_offset = safe_nnz_idx * hidden_size + h_indices
        
        # Load with combined mask: valid entry AND within hidden_size bounds
        load_mask = h_mask & is_valid
        res_vals = tl.load(res_ptr + res_offset, mask=load_mask, other=0.0)
        
        # Convert to FP32 and accumulate
        res_vals_fp32 = res_vals.to(tl.float32)
        
        # Only accumulate if valid (weight is already 0 for invalid entries conceptually)
        weighted = tl.where(is_valid, res_vals_fp32 * weight, 0.0)
        accumulator += weighted
    
    # Write result
    output_offset = token_idx * hidden_size + h_indices
    tl.store(output_ptr + output_offset, accumulator, mask=h_mask)


def build_inverse_mapping(
    global_indices: torch.Tensor,     # [nnz]
    token_topk_pos: torch.Tensor,     # [nnz]
    num_tokens: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Build inverse mapping: [num_tokens, num_experts_per_tok] -> nnz_idx"""
    # Use int64 for better compatibility with Triton indexing
    mapping = torch.full((num_tokens, num_experts_per_tok), -1, 
                         dtype=torch.int64, device=global_indices.device)
    
    # Handle empty case
    if global_indices.numel() == 0:
        return mapping
    
    # Ensure indices are within bounds
    assert global_indices.max() < num_tokens, "global_indices out of bounds"
    assert token_topk_pos.max() < num_experts_per_tok, "token_topk_pos out of bounds"
    
    mapping[global_indices, token_topk_pos] = torch.arange(
        len(global_indices), dtype=torch.int64, device=global_indices.device
    )
    return mapping


def scatter_weight_reduce_optimized(
    res: torch.Tensor,
    global_indices: torch.Tensor,
    token_topk_pos: torch.Tensor,
    topk_weight: torch.Tensor,
    num_tokens: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Optimized version using inverse mapping."""
    assert topk_weight.dtype == torch.float32, "topk_weight must be float32"
    assert topk_weight.shape == (num_tokens, num_experts_per_tok), "topk_weight shape mismatch"
    
    nnz, hidden_size = res.shape
    
    # Handle empty res case
    if nnz == 0:
        return torch.zeros((num_tokens, hidden_size), device=res.device, dtype=torch.float32)
    
    # Build inverse mapping (can be cached if indices don't change)
    nnz_indices = build_inverse_mapping(
        global_indices, token_topk_pos, num_tokens, num_experts_per_tok
    )
    
    output = torch.zeros((num_tokens, hidden_size), device=res.device, dtype=torch.float32)
    
    # Skip kernel launch if no work to do
    if num_tokens == 0:
        return output
    
    # Adaptive block size
    BLOCK_SIZE_H = min(triton.next_power_of_2(hidden_size), 256)
    grid = (num_tokens, triton.cdiv(hidden_size, BLOCK_SIZE_H))
    
    scatter_weight_reduce_optimized_kernel[grid](
        res, nnz_indices, topk_weight,
        output,
        num_tokens, num_experts_per_tok, hidden_size, nnz,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return output

class DeepseekV3MoE_Decoding_FP8(nn.Module): 
	"""
		EP with two ALL-to-ALLs.
	"""
	def __init__(self, config, comm):
		super().__init__()
		self.config = config
		self.num_experts_per_tok = config.num_experts_per_tok
		self.comm = comm
		

		# --- distributed/world metadata -------------------------------------
		if not dist.is_initialized():
			self.rank, self.world_size = 0, 1
		else:
			self.rank = dist.get_rank()
			self.world_size = dist.get_world_size()

		self.experts_per_rank   = 256 // self.world_size
		self.total_experts      = self.world_size * self.experts_per_rank
		self.routed_expert_start_idx = self.rank * self.experts_per_rank
		self.routed_expert_end_idx   = (self.rank + 1) * self.experts_per_rank

		# --- experts, gate, shared MLP --------------------------------------
		self.experts = nn.ModuleList([
			DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
			for _ in range(self.total_experts)
		])
		self.gate = MoEGate(config)
		if config.n_shared_experts:
			self.shared_experts = DeepseekV3MLP(
				config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
			)
		self.act_fn = ACT2FN[config.hidden_act]
		self.register_buffer(
			"send_counts_buf",
			torch.zeros(self.world_size, dtype=torch.int32)
		)
		self.register_buffer(
			"recv_counts_buf",
			torch.zeros(self.world_size, dtype=torch.int32)
		)

		# --- communication setup -------------------------------------------
		self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
		self.comm_stream = torch.cuda.Stream(device=self.device)

		# --- Pre-allocate Buffers. --------------------------------
		self.num_tokens_per_rank = None		# This is a placeholder, adjust as needed

	def init_num_tokens(self, num_tokens_per_rank):
		self.num_tokens_per_rank = num_tokens_per_rank
		global_num_tokens = self.num_tokens_per_rank * self.world_size
		K = self.num_experts_per_tok
		self.token_idx = torch.arange(global_num_tokens, dtype=torch.int32, device=self.device).repeat_interleave(K)
		self.topk_pos = torch.arange(K, dtype=torch.int32, device=self.device).repeat(global_num_tokens)
		self.gate_bias = torch.zeros(self.config.n_routed_experts, device=self.device, dtype=torch.bfloat16)

	def init(self, num_tokens_per_rank):
		# self.num_tokens_per_rank = num_tokens_per_rank
		
		self.gate_list = []
		self.up_list = []
		self.down_list = []
		self.gate_scale_list = []
		self.up_scale_list = []
		self.down_scale_list = []
		for e in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
			self.gate_list.append(self.experts[e].fp8_gate)
			self.up_list.append(self.experts[e].fp8_up)
			self.down_list.append(self.experts[e].fp8_down)
			self.gate_scale_list.append(self.experts[e].weight_dequant_scale['gate_proj.weight_scale_inv'])
			self.up_scale_list.append(self.experts[e].weight_dequant_scale['up_proj.weight_scale_inv'])
			self.down_scale_list.append(self.experts[e].weight_dequant_scale['down_proj.weight_scale_inv'])
		
			self.gate_ptrs_ptr = torch.tensor([r.data_ptr() for r in self.gate_list], dtype=torch.int64, device=self.device)
			self.up_ptrs_ptr = torch.tensor([r.data_ptr() for r in self.up_list], dtype=torch.int64, device=self.device)
			self.down_ptrs_ptr = torch.tensor([r.data_ptr() for r in self.down_list], dtype=torch.int64, device=self.device)
			self.gate_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in self.gate_scale_list], dtype=torch.int64, device=self.device)
			self.up_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in self.up_scale_list], dtype=torch.int64, device=self.device)
			self.down_scale_ptrs_ptr = torch.tensor([s.data_ptr() for s in self.down_scale_list], dtype=torch.int64, device=self.device)
	
	def cleanup(self):
		self.gate_list = None
		self.up_list = None
		self.down_list = None
		self.gate_scale_list = None
		self.up_scale_list = None
		self.down_scale_list = None
		self.gate_ptrs_ptr = None
		self.up_ptrs_ptr = None
		self.down_ptrs_ptr = None
		self.gate_scale_ptrs_ptr = None
		self.up_scale_ptrs_ptr = None
		self.down_scale_ptrs_ptr = None
		# gc.collect()



	def forward(self, hidden_states):
		orig_shape = hidden_states.shape
		hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
		identity = hidden_states
		
		out = self.moe_infer_allgather_allreduce_opt(hidden_states)
		# out = self.moe_infer_alltoall(hidden_states)
		out = out + self.shared_experts(identity)
		# accumulate with fp32
		# out = out.float()
		# out = out + self.shared_experts(identity).float()
		# out = out.to(hidden_states.dtype)


		return out.view(*orig_shape)
	
	@torch.inference_mode()
	def moe_infer_alltoall(self, x):
		num_tokens, hidden_size = x.shape
		K = self.num_experts_per_tok
		device = x.device
		topk_idx, topk_weight = self.gate.decoding_forward(x.view(num_tokens, 1, hidden_size)) #API Comp
		# ---- 1) flatten, sort by expert ------------------------------------
		flat_eids   = topk_idx.flatten()
		flat_wts    = topk_weight.flatten()
		expanded_x  = x.repeat_interleave(K, dim=0)
		# token_idx   = torch.arange(num_tokens, device=device).repeat_interleave(K)

		sorted_eids, sort_idx = flat_eids.sort()
		sorted_x   = expanded_x[sort_idx]
		# sorted_tok = token_idx[sort_idx]
		sorted_wt  = flat_wts[sort_idx]

		# ---- 2) fill the pre-allocated send_counts tensor ------------------
		local_counts = torch.bincount(sorted_eids, minlength=self.total_experts)
		sc = self.send_counts_buf               # alias for readability
		rc = self.recv_counts_buf

		reshaped_counts = local_counts.view(self.world_size, -1)
		sc = reshaped_counts.sum(dim=1)

		gathered_counts = [torch.zeros_like(sc) for _ in range(self.world_size)]
		# ---- 3) first all-to-all (counts) on its own stream ---------------
		dist.all_gather(gathered_counts, sc)
		gathered_tensor = torch.stack(gathered_counts)  # Shape: [world_size, world_size]
		rc = gathered_tensor[:, self.rank]  # Extract column for this rank
		recv_total = rc.sum()  # Keep as tensor until final use		
		# sc_cpu = sc.to("cpu", non_blocking=True)
		# rc_cpu = rc.to("cpu", non_blocking=True)

		# ---- 4) allocate data buffers (variable size) ----------------------
		send_x   = sorted_x
		send_eid = sorted_eids
		recv_x   = torch.empty(recv_total, hidden_size, device=device, dtype=x.dtype)
		recv_eid = torch.empty(recv_total, device=device, dtype=sorted_eids.dtype)

		# convert counts to python lists once – NCCL needs that
		sc_list, rc_list = sc.tolist(), rc.tolist()
		# torch.cuda.current_stream(device).synchronize()
		# sc_list, rc_list = sc_cpu.tolist(), rc_cpu.tolist()

		# ---- 5) main all-to-alls --------------------
		dist.all_to_all_single(recv_x, send_x, rc_list, sc_list)
		dist.all_to_all_single(recv_eid, send_eid, rc_list, sc_list)

		if recv_total:
			recv_eid_sorted, local_sort_idx = recv_eid.sort()
			recv_eid_sorted = recv_eid_sorted.to(torch.int32)
			res = self.grouped_dequant_moe_fp8(recv_x[local_sort_idx], recv_eid_sorted)
			recv_x[local_sort_idx] = res

		# ---- 7) all-to-all (return) ---------------------------------------
		out_sorted = torch.empty_like(sorted_x)
		dist.all_to_all_single(out_sorted, recv_x, sc_list, rc_list)

		# ---- 8) unsort, accumulate, normalise -----------------------------
		unsort_idx = sort_idx.argsort()
		final_x   = out_sorted[unsort_idx]
		# final_tok = sorted_tok[unsort_idx]
		final_wt  = sorted_wt[unsort_idx]

		# out_acc = torch.zeros_like(x, dtype=torch.float32)
		# out_acc.index_add_(0, final_tok, final_x.float() * final_wt.unsqueeze(-1))
		final_x = final_x * final_wt.unsqueeze(-1)
		final_x = final_x.view(num_tokens, K, -1)
		final_x = final_x.sum(dim=1)  # sum over experts
		return final_x.to(x.dtype)

	@torch.inference_mode()
	def moe_infer_allgather_allreduce(self, x):
		num_tokens, hidden_size = x.shape
		# Fix
		self.num_tokens_per_rank = 8
		self.num_tokens_per_rank = min(self.num_tokens_per_rank, triton.next_power_of_2(num_tokens))
		# logger.warning_once(f"Actuall num tokens per rank is {self.num_tokens_per_rank}")
		global_num_tokens = self.num_tokens_per_rank * self.world_size
		K = self.num_experts_per_tok
		token_idx = torch.arange(global_num_tokens, device=self.device).repeat_interleave(K)
		topk_pos = torch.arange(K, device=self.device).repeat(global_num_tokens)
		

		device = x.device
		# ---- 1) First all-gather: collect all tokens on all workers -------
		# Prepare buffers for all-gather
		all_tokens = torch.zeros((self.world_size * self.num_tokens_per_rank, self.config.hidden_size),
		 							  device=self.device, dtype=torch.bfloat16)
		# all_tokens[self.rank * self.num_tokens_per_rank: self.rank * self.num_tokens_per_rank + num_tokens] = x
		# with self.comm.change_state(enable=True):
		# 	self.comm.all_reduce(all_tokens, op=dist.ReduceOp.SUM, stream=torch.cuda.default_stream(self.device))
		with self.comm.change_state(enable=True):
			self.comm.all_gather(all_tokens, x, stream=torch.cuda.default_stream(self.device))
		# self.comm.stream.synchronize()  # Ensure all-gather is complete
		# dist.all_gather_into_tensor(all_tokens, x, async_op=False)
		# ---- 2) Gate computation on global tokens --------------------------
		global_x = all_tokens
		global_x = global_x.view(global_x.shape[0], 1, global_x.shape[1])  # Add dummy dimension for compatibility
		topk_idx, topk_weight = self.gate.decoding_forward(global_x)
		global_x = global_x.squeeze(1)  # Remove the dummy dimension


		# ---- 3) Process tokens assigned to local experts ------------------
		# Find out which tokens are assigned to which local experts.
		# ---- 3.1) flatten, sort by expert ------------------------------------
		flat_eids   = topk_idx.flatten()
		expanded_x  = global_x.repeat_interleave(K, dim=0)
		sorted_eids, sort_idx = flat_eids.sort()
		sorted_x   = expanded_x[sort_idx]
		sorted_tok = token_idx[sort_idx]
		sorted_pos = topk_pos[sort_idx]

		# ---- 3.2) Build tensor for local expert input sorted by expert id ----
		"""
			We need a tensor x that contains tokens assigned to local expert sorted by expert id.
			We need a tensor eids that contains expert ids for each token in x.
		"""
		local_token_expanded_x_indices = (sorted_eids >= self.routed_expert_start_idx) & (sorted_eids < self.routed_expert_end_idx)
		input_x = sorted_x[local_token_expanded_x_indices]
		input_eids = sorted_eids[local_token_expanded_x_indices]
		global_indices = sorted_tok[local_token_expanded_x_indices]
		token_topk_pos = sorted_pos[local_token_expanded_x_indices]
		# ---- 3) Process tokens assigned to local experts ------------------
		res = self.grouped_dequant_moe_fp8(input_x, input_eids)
		# self.local_expert_results.zero_()  # Reset results buffer
		global_results = torch.zeros((self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok, self.config.hidden_size),
		 									 device=self.device, dtype=torch.bfloat16)
		global_results[global_indices, token_topk_pos, :] = res
		# weighted_output = global_results * topk_weight.unsqueeze(-1)
		# global_results = weighted_output.sum(dim=1)
		global_results = moe_weighted_sum_triton_v2(global_results, topk_weight)

		# ---- 3.3) All-reduce to combine results from all workers ------------
		with self.comm.change_state(enable=True):
			self.comm.all_reduce(global_results, op=dist.ReduceOp.SUM, stream=torch.cuda.default_stream(self.device))
		# ---- 3.4) Extract results for local tokens and aggregate ------------
		start_token_ids = self.rank * num_tokens
		end_token_ids = start_token_ids + num_tokens

		final_output = global_results[start_token_ids:end_token_ids]
		return final_output

	@torch.inference_mode()
	def moe_infer_allgather_allreduce_opt(self, x):
		num_tokens, hidden_size = x.shape
		device = x.device
		# ---- 1) First all-gather: collect all tokens on all workers -------
		# Prepare buffers for all-gather
		all_tokens = torch.zeros((self.world_size * self.num_tokens_per_rank, self.config.hidden_size),
		 							  device=self.device, dtype=torch.bfloat16)
		if x.shape[0] < self.num_tokens_per_rank:
			padded_hidden_states = torch.zeros((self.num_tokens_per_rank, hidden_size), device=self.device, dtype=x.dtype)
			padded_hidden_states[:x.shape[0]] = x
		else:
			padded_hidden_states = x
		with self.comm.change_state(enable=True):
			self.comm.all_gather(all_tokens, padded_hidden_states, stream=torch.cuda.default_stream(self.device))
		# ---- 2) Gate computation on global tokens --------------------------
		global_x = all_tokens
		global_x = global_x.view(global_x.shape[0], 1, global_x.shape[1])  # Add dummy dimension for compatibility
		# topk_idx, topk_weight = self.gate.decoding_forward(global_x)
		topk_idx, topk_weight = self.gate.moe_gate_forward_hybrid(global_x)
		
		# logits = F.linear(
		# 	global_x.type(torch.float32), self.gate.weight.type(torch.float32), None
		# ).to(torch.bfloat16)

		# logits = logits.squeeze(1)  
		# topk_weight, topk_idx = moe_fused_gate(
		# 	logits, 
		# 	self.gate_bias,
		# 	self.config.n_group,
		# 	self.config.topk_group,
		# 	self.config.num_experts_per_tok,
		# 	0,
		# 	self.config.routed_scaling_factor
		# )
		global_x = global_x.squeeze(1) 
		


		# ---- 3) Process tokens assigned to local experts ------------------
		topk_idx = topk_idx.to(torch.int32)
		# dispatcher = FusedMoETokenPermutation(use_cuda_if_available=True)
		# input_x, input_eids, global_indices, token_topk_pos = dispatcher(
		# 	global_x, topk_idx, self.token_idx, self.topk_pos,
		# 	self.routed_expert_start_idx, self.routed_expert_end_idx
		# )
		
		input_x, input_eids, global_indices, token_topk_pos, _ = fused_moe_token_dispatch(
			global_x, topk_idx, self.token_idx, self.topk_pos,
			self.routed_expert_start_idx, self.routed_expert_end_idx,
		)

		# ---- 3) Process tokens assigned to local experts ------------------
		res = self.grouped_dequant_moe_fp8(input_x, input_eids)
		# res = self.grouped_weight_dequant_moe_a16w8(input_x, input_eids)
		# global_results = torch.zeros((self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok, self.config.hidden_size),
		#  									 device=self.device, dtype=torch.bfloat16)
		# global_results[global_indices, token_topk_pos, :] = res

		# """ FP32 Weighting """
		# assert topk_weight.dtype == torch.float32
		# weighted_output = global_results.to(torch.float32) * topk_weight.unsqueeze(-1)
		# global_results = weighted_output.sum(dim=1)
		global_results = scatter_weight_reduce_optimized(
			res, global_indices, token_topk_pos, topk_weight,
			self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok
		)
		
		""" BF16 Weighting """
		# topk_weight = topk_weight.to(x.dtype)
		# global_results = moe_weighted_sum_triton_v2(global_results, topk_weight)
		
		# ---- 3.3) All-reduce to combine results from all workers ------------
		with self.comm.change_state(enable=True):
			self.comm.all_reduce(global_results, op=dist.ReduceOp.SUM, stream=torch.cuda.default_stream(self.device))
		# ---- 3.4) Extract results for local tokens and aggregate ------------
		start_token_ids = self.rank * num_tokens
		end_token_ids = start_token_ids + num_tokens

		final_output = global_results[start_token_ids:end_token_ids]
		return final_output.to(x.dtype)
	
	# def expert_bincount(self, eids, routed_expert_start_idx, experts_per_rank, device):
	# 	eids_adjusted = eids - routed_expert_start_idx  
	# 	counts = torch.bincount(eids_adjusted, minlength=experts_per_rank)
		
	# 	nonzero_mask = counts > 0
	# 	activated_group_idx = torch.nonzero(nonzero_mask, as_tuple=True)[0].to(torch.int32)
	# 	group_size = counts[nonzero_mask].to(torch.int32)
		
	# 	group_start_indices = torch.zeros_like(group_size)
	# 	if group_size.numel() > 1:
	# 		group_start_indices[1:] = torch.cumsum(group_size[:-1], dim=0)
		
	# 	return group_size, activated_group_idx, group_start_indices

	def grouped_dequant_moe_fp8(self, x, eids):
		# group_size, activated_group_idx, group_start_indices = self.expert_bincount(
		# 	eids, self.routed_expert_start_idx, self.experts_per_rank, self.device
		# )
		if(len(eids) == 0):
			# logger.warning_once("No tokens routed to this rank.")
			assert len(x) == 0, "If no tokens routed, x should be empty too."
			return x

		group_size, activated_group_idx, group_start_indices = expert_bincount(
			eids, self.routed_expert_start_idx, self.experts_per_rank, self.device
		)
		# eids = eids - self.routed_expert_start_idx
		# counts = torch.bincount(eids, minlength=self.experts_per_rank)
		# group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)	
		# group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=self.device)
		# group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
		# group_start_indices[0] = 0  # The first group starts at index 0	
		# activated_group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=self.device)

		# Quantize the recv_x tensor to fp8_e4m3
		x, x_scale = act_quant(x)
		intermediate = fused_fp8_moe_stage_1(
			x, x_scale, 
			self.gate_list, self.gate_ptrs_ptr,
			self.up_list, self.up_ptrs_ptr,
			self.gate_scale_list, self.gate_scale_ptrs_ptr,
			self.up_scale_list, self.up_scale_ptrs_ptr,
			group_size, activated_group_idx, group_start_indices
		)	
		
		intermediate, intermediate_scale = act_quant(intermediate)
		res = fused_dequant_grouped_gemm_fp8_fp8_triton(
			intermediate, intermediate_scale, 
			self.down_list, self.down_ptrs_ptr,
			self.down_scale_list, self.down_scale_ptrs_ptr,
			group_size, activated_group_idx, group_start_indices
		)
		return res

	def grouped_weight_dequant_moe_a16w8(self, x, eids):
		# group_size, activated_group_idx, group_start_indices = self.expert_bincount(
		# 	eids, self.routed_expert_start_idx, self.experts_per_rank, self.device
		# )
		if(len(eids) == 0):
			# logger.warning_once("No tokens routed to this rank.")
			assert len(x) == 0, "If no tokens routed, x should be empty too."
			return x

		group_size, activated_group_idx, group_start_indices = expert_bincount(
			eids, self.routed_expert_start_idx, self.experts_per_rank, self.device
		)

		# Quantize the recv_x tensor to fp8_e4m3
		# x, x_scale = act_quant(x)
		# intermediate = fused_fp8_moe_stage_1(
		# 	x, x_scale, 
		# 	self.gate_list, self.gate_ptrs_ptr,
		# 	self.up_list, self.up_ptrs_ptr,
		# 	self.gate_scale_list, self.gate_scale_ptrs_ptr,
		# 	self.up_scale_list, self.up_scale_ptrs_ptr,
		# 	group_size, activated_group_idx, group_start_indices
		# )	
		
		# intermediate, intermediate_scale = act_quant(intermediate)
		# res = fused_dequant_grouped_gemm_fp8_fp8_triton(
		# 	intermediate, intermediate_scale, 
		# 	self.down_list, self.down_ptrs_ptr,
		# 	self.down_scale_list, self.down_scale_ptrs_ptr,
		# 	group_size, activated_group_idx, group_start_indices
		# )
		# group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
		# Please get group_sizes from group_size tensor
		group_sizes = [(int(idx), int(size)) for idx, size in zip(activated_group_idx.tolist(), group_size.tolist())]
		intermediate = fused_dequant_weighted_moe_stage_1(
			x, self.gate_list, self.up_list, self.gate_scale_list, self.up_scale_list, group_sizes, group_start_indices
		)	
		
		res = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
			intermediate, self.down_list, self.down_scale_list, group_sizes, group_start_indices, gemm_block_size = [64, 16, 64]
		)
		return res


	

class DeepseekV3MoE_Decoding_v2_bak2(nn.Module): 
	"""
		EP with two ALL-to-ALLs.
	"""
	def __init__(self, config, comm):
		super().__init__()
		self.config = config
		self.num_experts_per_tok = config.num_experts_per_tok
		self.comm = comm
		

		# --- distributed/world metadata -------------------------------------
		if not dist.is_initialized():
			self.rank, self.world_size = 0, 1
		else:
			self.rank = dist.get_rank()
			self.world_size = dist.get_world_size()

		self.experts_per_rank   = 256 // self.world_size
		self.total_experts      = self.world_size * self.experts_per_rank
		self.routed_expert_start_idx = self.rank * self.experts_per_rank
		self.routed_expert_end_idx   = (self.rank + 1) * self.experts_per_rank

		# --- experts, gate, shared MLP --------------------------------------
		self.experts = nn.ModuleList([
			DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
			for _ in range(self.total_experts)
		])
		self.gate = MoEGate(config)
		if config.n_shared_experts:
			self.shared_experts = DeepseekV3MLP(
				config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
			)

		# --- communication setup -------------------------------------------
		self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
		self.comm_stream = torch.cuda.Stream(device=self.device)

		# --- Pre-allocate Buffers. --------------------------------
		self.num_tokens_per_rank = 48		# This is a placeholder, adjust as needed
		global_num_tokens = self.num_tokens_per_rank * self.world_size
		K = self.num_experts_per_tok
		# self.all_tokens = torch.zeros((self.world_size * num_tokens_per_rank, config.hidden_size),
		# 							  device=self.device, dtype=torch.bfloat16)
		# self.local_expert_results = torch.zeros((num_tokens_per_rank * self.world_size, self.num_experts_per_tok, config.hidden_size),
		# 									 device=self.device, dtype=torch.bfloat16)

		self.act_fn = ACT2FN[config.hidden_act]
		self.token_idx = torch.arange(global_num_tokens, device=self.device).repeat_interleave(K)
		self.topk_pos = torch.arange(K, device=self.device).repeat(global_num_tokens)


	
	def forward(self, hidden_states):
		orig_shape = hidden_states.shape
		hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
		identity = hidden_states
		
		out = self.moe_infer(hidden_states)
		out = out + self.shared_experts(identity)
		return out.view(*orig_shape)

	@torch.inference_mode()
	def moe_infer(self, x):
		# self.local_expert_results.zero_()  # Reset results buffer
		num_tokens, hidden_size = x.shape
		K = self.num_experts_per_tok
		device = x.device
		# ---- 1) First all-gather: collect all tokens on all workers -------
		# Prepare buffers for all-gather
		# logger.info(f"Rank {self.rank} starting all-gather for tokens.")
		# dist.all_gather(self.all_tokens, x, async_op=False)
		# torch.cuda.synchronize(self.device)  # Ensure all-gather is complete
		# self.all_tokens.zero_()  # Reset the all_tokens buffer
		all_tokens = torch.zeros((self.world_size * self.num_tokens_per_rank, self.config.hidden_size),
		 							  device=self.device, dtype=torch.bfloat16)
		# dist.all_gather_into_tensor(all_tokens, x, async_op=False)
		# self.comm.all_gather(all_tokens, x, self.comm_stream)
		# self.comm_stream.synchronize()  # Ensure all-gather is complete
		# torch.cuda.current_stream(self.device).wait_stream(self.comm_stream)  # Ensure
		# torch.cuda.synchronize(self.device)  # Ensure all-gather is complete
		with self.comm.change_state(enable=True):
			self.comm.all_gather(all_tokens, x, stream=torch.cuda.default_stream(self.device))
		# self.comm.stream.synchronize()  # Ensure all-gather is complete
		# dist.barrier()  # Ensure all-gather is complete before proceeding

		# logger.info(f"Rank {self.rank} gathered {len(self.all_tokens)} tokens from all workers.")

		# Concatenate all tokens from all workers
		# global_x = torch.cat(self.all_tokens, dim=0)  # Shape: [num_tokens * world_size, hidden_size]
		# global_num_tokens = global_x.shape[0]

		# ---- 2) Gate computation on global tokens --------------------------
		# logger.info(f"all_tokens.shape: {all_tokens.shape}, dtype: {all_tokens.dtype}, device: {all_tokens.device}")
		# if self.rank == 0:
		# 	logger.info(f"{all_tokens[0]}")  # Log the first token for debugging
		global_x = all_tokens
		global_x = global_x.view(global_x.shape[0], 1, global_x.shape[1])  # Add dummy dimension for compatibility
		topk_idx, topk_weight = self.gate.decoding_forward(global_x)
		global_x = global_x.squeeze(1)  # Remove the dummy dimension


		# ---- 3) Process tokens assigned to local experts ------------------
		"""
		if recv_total:
			recv_eid_sorted, local_sort_idx = recv_eid.sort()
			res = self.grouped_dequant_moe(recv_x[local_sort_idx], recv_eid_sorted)
			recv_x[local_sort_idx] = res

		"""	

		# Find out which tokens are assigned to which local experts.
		# ---- 1) flatten, sort by expert ------------------------------------
		flat_eids   = topk_idx.flatten()
		expanded_x  = global_x.repeat_interleave(K, dim=0)
		# token_idx   = torch.arange(global_num_tokens, device=device).repeat_interleave(K)
		# topk_pos    = torch.arange(K, device=device).repeat(global_num_tokens)

		sorted_eids, sort_idx = flat_eids.sort()
		sorted_x   = expanded_x[sort_idx]
		sorted_tok = self.token_idx[sort_idx]
		sorted_pos = self.topk_pos[sort_idx]

		# ---- 2) Build tensor for local expert input sorted by expert id ----
		"""
			We need a tensor x that contains tokens assigned to local expert sorted by expert id.
			We need a tensor eids that contains expert ids for each token in x.
		"""
		local_token_expanded_x_indices = (sorted_eids >= self.routed_expert_start_idx) & (sorted_eids < self.routed_expert_end_idx)
		input_x = sorted_x[local_token_expanded_x_indices]
		input_eids = sorted_eids[local_token_expanded_x_indices]
		global_indices = sorted_tok[local_token_expanded_x_indices]
		token_topk_pos = sorted_pos[local_token_expanded_x_indices]
		# torch.cuda.synchronize(self.device)  # Ensure all operations are complete
		# ---- 3) Process tokens assigned to local experts ------------------
		res = self.grouped_dequant_moe(input_x, input_eids)
		# self.local_expert_results.zero_()  # Reset results buffer
		global_results = torch.zeros((self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok, self.config.hidden_size),
		 									 device=self.device, dtype=torch.bfloat16)
		global_results[global_indices, token_topk_pos, :] = res
		# Compute the weighted sum
		# weighted_output = global_results.to(torch.float32) * topk_weight.unsqueeze(-1)
		# assert topk_weight.dtype == torch.bfloat16, f"Expected topk_weight to be bfloat16, got {topk_weight.dtype}"
		weighted_output = global_results * topk_weight.unsqueeze(-1)
		global_results = weighted_output.sum(dim=1)


				
		# ---- 4) All-reduce to combine results from all workers ------------
		with self.comm.change_state(enable=True):
			self.comm.all_reduce(global_results, op=dist.ReduceOp.SUM, stream=torch.cuda.default_stream(self.device))
		# self.comm.stream.synchronize()  # Ensure all-reduce is complete
		# ---- 5) Extract results for local tokens and aggregate ------------
		start_token_ids = self.rank * num_tokens
		end_token_ids = start_token_ids + num_tokens

		# local_results = local_expert_results[start_token_ids:end_token_ids]
		# local_token_weights = topk_weight[start_token_ids:end_token_ids]

		# weighted_output = local_results.to(torch.float32) * local_token_weights.unsqueeze(-1)
		# final_output = weighted_output.sum(dim=1).to(x.dtype)
		final_output = global_results[start_token_ids:end_token_ids]
		return final_output

	def grouped_dequant_moe(self, recv_x, recv_eid):
		# This function assumes that recv_x and recv_eid are already sorted by expert id
		gate_list = []
		up_list = []
		down_list = []
		gate_scale_list = []
		up_scale_list = []
		down_scale_list = []
		for e in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
			gate_list.append(self.experts[e].fp8_gate)
			up_list.append(self.experts[e].fp8_up)
			down_list.append(self.experts[e].fp8_down)
			gate_scale_list.append(self.experts[e].weight_dequant_scale['gate_proj.weight_scale_inv'])
			up_scale_list.append(self.experts[e].weight_dequant_scale['up_proj.weight_scale_inv'])
			down_scale_list.append(self.experts[e].weight_dequant_scale['down_proj.weight_scale_inv'])
		eids = recv_eid - self.routed_expert_start_idx
		counts = torch.bincount(eids, minlength=self.experts_per_rank)
		group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)	
		group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=self.device)
		group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
		group_start_indices[0] = 0  # The first group starts at index 0	

		intermediate = fused_dequant_weighted_moe_stage_1(
			recv_x, gate_list, up_list, gate_scale_list, up_scale_list, group_sizes, group_start_indices
		)	
		
		res = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
			intermediate, down_list, down_scale_list, group_sizes, group_start_indices, gemm_block_size = [64, 16, 64]
		)
		return res




class DeepseekV3MoE_Decoding_v2_bak(nn.Module): 
	"""
		EP with two ALL-to-ALLs.
	"""
	def __init__(self, config, comm):
		super().__init__()
		self.config = config
		self.num_experts_per_tok = config.num_experts_per_tok
		self.comm = comm
		

		# --- distributed/world metadata -------------------------------------
		if not dist.is_initialized():
			self.rank, self.world_size = 0, 1
		else:
			self.rank = dist.get_rank()
			self.world_size = dist.get_world_size()

		self.experts_per_rank   = 256 // self.world_size
		self.total_experts      = self.world_size * self.experts_per_rank
		self.routed_expert_start_idx = self.rank * self.experts_per_rank
		self.routed_expert_end_idx   = (self.rank + 1) * self.experts_per_rank

		# --- experts, gate, shared MLP --------------------------------------
		self.experts = nn.ModuleList([
			DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
			for _ in range(self.total_experts)
		])
		self.gate = MoEGate(config)
		if config.n_shared_experts:
			self.shared_experts = DeepseekV3MLP(
				config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
			)

		# --- communication setup -------------------------------------------
		self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
		self.comm_stream = torch.cuda.Stream(device=self.device)

		# --- Pre-allocate Buffers. --------------------------------
		self.num_tokens_per_rank = 48		# This is a placeholder, adjust as needed
		global_num_tokens = self.num_tokens_per_rank * self.world_size
		K = self.num_experts_per_tok
		# self.all_tokens = torch.zeros((self.world_size * num_tokens_per_rank, config.hidden_size),
		# 							  device=self.device, dtype=torch.bfloat16)
		# self.local_expert_results = torch.zeros((num_tokens_per_rank * self.world_size, self.num_experts_per_tok, config.hidden_size),
		# 									 device=self.device, dtype=torch.bfloat16)

		self.act_fn = ACT2FN[config.hidden_act]
		self.token_idx = torch.arange(global_num_tokens, device=self.device).repeat_interleave(K)
		self.topk_pos = torch.arange(K, device=self.device).repeat(global_num_tokens)


	
	def forward(self, hidden_states):
		orig_shape = hidden_states.shape
		hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
		identity = hidden_states
		
		out = self.moe_infer(hidden_states)
		out = out + self.shared_experts(identity)
		return out.view(*orig_shape)

	@torch.inference_mode()
	def moe_infer(self, x):
		# self.local_expert_results.zero_()  # Reset results buffer
		num_tokens, hidden_size = x.shape
		K = self.num_experts_per_tok
		device = x.device
		# ---- 1) First all-gather: collect all tokens on all workers -------
		# Prepare buffers for all-gather
		# logger.info(f"Rank {self.rank} starting all-gather for tokens.")
		# dist.all_gather(self.all_tokens, x, async_op=False)
		# torch.cuda.synchronize(self.device)  # Ensure all-gather is complete
		# self.all_tokens.zero_()  # Reset the all_tokens buffer
		all_tokens = torch.zeros((self.world_size * self.num_tokens_per_rank, self.config.hidden_size),
		 							  device=self.device, dtype=torch.bfloat16)
		# dist.all_gather_into_tensor(all_tokens, x, async_op=False)
		# self.comm.all_gather(all_tokens, x, self.comm_stream)
		# self.comm_stream.synchronize()  # Ensure all-gather is complete
		# torch.cuda.current_stream(self.device).wait_stream(self.comm_stream)  # Ensure
		# torch.cuda.synchronize(self.device)  # Ensure all-gather is complete
		with self.comm.change_state(enable=True):
			self.comm.all_gather(all_tokens, x, stream=torch.cuda.default_stream(self.device))
		self.comm.stream.synchronize()  # Ensure all-gather is complete
		# dist.barrier()  # Ensure all-gather is complete before proceeding

		# logger.info(f"Rank {self.rank} gathered {len(self.all_tokens)} tokens from all workers.")

		# Concatenate all tokens from all workers
		# global_x = torch.cat(self.all_tokens, dim=0)  # Shape: [num_tokens * world_size, hidden_size]
		# global_num_tokens = global_x.shape[0]

		# ---- 2) Gate computation on global tokens --------------------------
		# logger.info(f"all_tokens.shape: {all_tokens.shape}, dtype: {all_tokens.dtype}, device: {all_tokens.device}")
		# if self.rank == 0:
		# 	logger.info(f"{all_tokens[0]}")  # Log the first token for debugging
		global_x = all_tokens
		global_x = global_x.view(global_x.shape[0], 1, global_x.shape[1])  # Add dummy dimension for compatibility
		topk_idx, topk_weight = self.gate(global_x)
		global_x = global_x.squeeze(1)  # Remove the dummy dimension


		# ---- 3) Process tokens assigned to local experts ------------------
		"""
		if recv_total:
			recv_eid_sorted, local_sort_idx = recv_eid.sort()
			res = self.grouped_dequant_moe(recv_x[local_sort_idx], recv_eid_sorted)
			recv_x[local_sort_idx] = res

		"""	

		# Find out which tokens are assigned to which local experts.
		# ---- 1) flatten, sort by expert ------------------------------------
		flat_eids   = topk_idx.flatten()
		expanded_x  = global_x.repeat_interleave(K, dim=0)
		# token_idx   = torch.arange(global_num_tokens, device=device).repeat_interleave(K)
		# topk_pos    = torch.arange(K, device=device).repeat(global_num_tokens)

		sorted_eids, sort_idx = flat_eids.sort()
		sorted_x   = expanded_x[sort_idx]
		sorted_tok = self.token_idx[sort_idx]
		sorted_pos = self.topk_pos[sort_idx]

		# ---- 2) Build tensor for local expert input sorted by expert id ----
		"""
			We need a tensor x that contains tokens assigned to local expert sorted by expert id.
			We need a tensor eids that contains expert ids for each token in x.
		"""
		local_token_expanded_x_indices = (sorted_eids >= self.routed_expert_start_idx) & (sorted_eids < self.routed_expert_end_idx)
		input_x = sorted_x[local_token_expanded_x_indices]
		input_eids = sorted_eids[local_token_expanded_x_indices]
		global_indices = sorted_tok[local_token_expanded_x_indices]
		token_topk_pos = sorted_pos[local_token_expanded_x_indices]
		# torch.cuda.synchronize(self.device)  # Ensure all operations are complete
		# ---- 3) Process tokens assigned to local experts ------------------
		res = self.grouped_dequant_moe(input_x, input_eids)
		# self.local_expert_results.zero_()  # Reset results buffer
		local_expert_results = torch.zeros((self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok, self.config.hidden_size),
		 									 device=self.device, dtype=torch.bfloat16)
		local_expert_results[global_indices, token_topk_pos, :] = res
		# torch.cuda.default_stream(self.device).synchronize()  # Ensure all operations are complete
		# torch.cuda.synchronize(self.device)  # Ensure all operations are complete	
				
		# ---- 4) All-reduce to combine results from all workers ------------
		# logger.info(f"Rank {self.rank} starting all-reduce for expert results.")
		# dist.all_reduce(local_expert_results, op=dist.ReduceOp.SUM, async_op=False)
		# self.comm.all_reduce(local_expert_results, stream=self.comm_stream)
		# self.comm_stream.synchronize()  # Ensure all-reduce is complete
		# torch.cuda.current_stream(self.device).wait_stream(self.comm_stream)  # Ensure all
		# torch.cuda.synchronize(self.device)
		with self.comm.change_state(enable=True):
			self.comm.all_reduce(local_expert_results, op=dist.ReduceOp.SUM, stream=torch.cuda.default_stream(self.device))
		self.comm.stream.synchronize()  # Ensure all-reduce is complete
		# ---- 5) Extract results for local tokens and aggregate ------------
		start_token_ids = self.rank * num_tokens
		end_token_ids = start_token_ids + num_tokens

		local_results = local_expert_results[start_token_ids:end_token_ids]
		local_token_weights = topk_weight[start_token_ids:end_token_ids]

		weighted_output = local_results.to(torch.float32) * local_token_weights.unsqueeze(-1)
		final_output = weighted_output.sum(dim=1).to(x.dtype)

		return final_output

	def grouped_dequant_moe(self, recv_x, recv_eid):
		# This function assumes that recv_x and recv_eid are already sorted by expert id
		gate_list = []
		up_list = []
		down_list = []
		gate_scale_list = []
		up_scale_list = []
		down_scale_list = []
		for e in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
			gate_list.append(self.experts[e].fp8_gate)
			up_list.append(self.experts[e].fp8_up)
			down_list.append(self.experts[e].fp8_down)
			gate_scale_list.append(self.experts[e].weight_dequant_scale['gate_proj.weight_scale_inv'])
			up_scale_list.append(self.experts[e].weight_dequant_scale['up_proj.weight_scale_inv'])
			down_scale_list.append(self.experts[e].weight_dequant_scale['down_proj.weight_scale_inv'])
		eids = recv_eid - self.routed_expert_start_idx
		counts = torch.bincount(eids, minlength=self.experts_per_rank)
		group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)	
		group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=self.device)
		group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
		group_start_indices[0] = 0  # The first group starts at index 0	

		intermediate = fused_dequant_weighted_moe_stage_1(
			recv_x, gate_list, up_list, gate_scale_list, up_scale_list, group_sizes, group_start_indices
		)	
		
		res = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
			intermediate, down_list, down_scale_list, group_sizes, group_start_indices, gemm_block_size = [64, 16, 64]
		)
		return res

class DeepseekV3MoE_Decoding_v2_(nn.Module):
	"""
		EP with two ALL-to-ALLs.
	"""
	def __init__(self, config, comm):
		super().__init__()
		self.config = config
		self.num_experts_per_tok = config.num_experts_per_tok
		self.comm = comm
		

		# --- distributed/world metadata -------------------------------------
		if not dist.is_initialized():
			self.rank, self.world_size = 0, 1
		else:
			self.rank = dist.get_rank()
			self.world_size = dist.get_world_size()

		self.experts_per_rank   = 256 // self.world_size
		self.total_experts      = self.world_size * self.experts_per_rank
		self.routed_expert_start_idx = self.rank * self.experts_per_rank
		self.routed_expert_end_idx   = (self.rank + 1) * self.experts_per_rank

		# --- experts, gate, shared MLP --------------------------------------
		self.experts = nn.ModuleList([
			DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
			for _ in range(self.total_experts)
		])
		self.gate = MoEGate(config)
		if config.n_shared_experts:
			self.shared_experts = DeepseekV3MLP(
				config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
			)

		# --- communication setup -------------------------------------------
		self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
		# self.comm_stream = torch.cuda.Stream()

		# --- Pre-allocate Buffers. --------------------------------
		num_tokens_per_rank = 8		# This is a placeholder, adjust as needed
		global_num_tokens = num_tokens_per_rank * self.world_size
		K = self.num_experts_per_tok
		self.all_tokens = torch.zeros((self.world_size * num_tokens_per_rank, config.hidden_size),
									  device=self.device, dtype=torch.bfloat16)
		self.local_expert_results = torch.zeros((num_tokens_per_rank * self.world_size, self.num_experts_per_tok, config.hidden_size),
											 device=self.device, dtype=torch.bfloat16)

		self.act_fn = ACT2FN[config.hidden_act]
		self.token_idx = torch.arange(global_num_tokens, device=self.device).repeat_interleave(K)
		self.topk_pos = torch.arange(K, device=self.device).repeat(global_num_tokens)

	
	def forward(self, hidden_states):
		orig_shape = hidden_states.shape
		hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
		identity = hidden_states
		
		out = self.moe_infer(hidden_states)
		out = out + self.shared_experts(identity)
		return out.view(*orig_shape)

	@torch.inference_mode()
	def moe_infer(self, x):
		# self.local_expert_results.zero_()  # Reset results buffer
		num_tokens, hidden_size = x.shape
		K = self.num_experts_per_tok
		device = x.device
		# ---- 1) First all-gather: collect all tokens on all workers -------
		# Prepare buffers for all-gather
		# logger.info(f"Rank {self.rank} starting all-gather for tokens.")
		# dist.all_gather(self.all_tokens, x, async_op=False)
		# torch.cuda.synchronize(self.device)  # Ensure all-gather is complete
		self.all_tokens.zero_()  # Reset the all_tokens buffer
		dist.all_gather_into_tensor(self.all_tokens, x, async_op=False)
		# torch.cuda.synchronize(self.device)  # Ensure all-gather is complete
		# dist.barrier()  # Ensure all-gather is complete before proceeding

		# logger.info(f"Rank {self.rank} gathered {len(self.all_tokens)} tokens from all workers.")

		# Concatenate all tokens from all workers
		# global_x = torch.cat(self.all_tokens, dim=0)  # Shape: [num_tokens * world_size, hidden_size]
		# global_num_tokens = global_x.shape[0]

		# ---- 2) Gate computation on global tokens --------------------------
		global_x = self.all_tokens
		global_x = global_x.view(global_x.shape[0], 1, global_x.shape[1])  # Add dummy dimension for compatibility
		topk_idx, topk_weight = self.gate(global_x)
		global_x = global_x.squeeze(1)  # Remove the dummy dimension


		# ---- 3) Process tokens assigned to local experts ------------------
		"""
		if recv_total:
			recv_eid_sorted, local_sort_idx = recv_eid.sort()
			res = self.grouped_dequant_moe(recv_x[local_sort_idx], recv_eid_sorted)
			recv_x[local_sort_idx] = res

		"""	

		# Find out which tokens are assigned to which local experts.
		# ---- 1) flatten, sort by expert ------------------------------------
		flat_eids   = topk_idx.flatten()
		expanded_x  = global_x.repeat_interleave(K, dim=0)
		# token_idx   = torch.arange(global_num_tokens, device=device).repeat_interleave(K)
		# topk_pos    = torch.arange(K, device=device).repeat(global_num_tokens)

		sorted_eids, sort_idx = flat_eids.sort()
		sorted_x   = expanded_x[sort_idx]
		sorted_tok = self.token_idx[sort_idx]
		sorted_pos = self.topk_pos[sort_idx]

		# ---- 2) Build tensor for local expert input sorted by expert id ----
		"""
			We need a tensor x that contains tokens assigned to local expert sorted by expert id.
			We need a tensor eids that contains expert ids for each token in x.
		"""
		local_token_expanded_x_indices = (sorted_eids >= self.routed_expert_start_idx) & (sorted_eids < self.routed_expert_end_idx)
		input_x = sorted_x[local_token_expanded_x_indices]
		input_eids = sorted_eids[local_token_expanded_x_indices]
		global_indices = sorted_tok[local_token_expanded_x_indices]
		token_topk_pos = sorted_pos[local_token_expanded_x_indices]
		# torch.cuda.synchronize(self.device)  # Ensure all operations are complete
		# ---- 3) Process tokens assigned to local experts ------------------
		res = self.grouped_dequant_moe(input_x, input_eids)
		self.local_expert_results.zero_()  # Reset results buffer
		self.local_expert_results[global_indices, token_topk_pos, :] = res
		# torch.cuda.default_stream(self.device).synchronize()  # Ensure all operations are complete
		


		# for expert_idx in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
		# 	# Get the mask for tokens assigned to this expert
		# 	mask = (topk_idx == expert_idx)
		# 	if mask.any():
		# 		# Process only tokens assigned to this expert
		# 		token_indices, topk_pos = torch.where(mask)

		# 		if len(token_indices) > 0:
		# 			# Get the corresponding tokens and weights
		# 			input_tokens = global_x[token_indices]

		# 			# Apply the expert to these tokens
		# 			expert_output = self.experts[expert_idx](input_tokens)

		# 			self.local_expert_results[token_indices, topk_pos, :] = expert_output
		
		# ---- 4) All-reduce to combine results from all workers ------------
		# logger.info(f"Rank {self.rank} starting all-reduce for expert results.")
		dist.all_reduce(self.local_expert_results, op=dist.ReduceOp.SUM, async_op=False)
		# ---- 5) Extract results for local tokens and aggregate ------------
		start_token_ids = self.rank * num_tokens
		end_token_ids = start_token_ids + num_tokens

		local_results = self.local_expert_results[start_token_ids:end_token_ids]
		local_token_weights = topk_weight[start_token_ids:end_token_ids]

		weighted_output = local_results.to(torch.float32) * local_token_weights.unsqueeze(-1)
		final_output = weighted_output.sum(dim=1).to(x.dtype)

		return final_output


	# def grouped_dequant_moe(self, recv_x, recv_eid):
	# 	from ....moe.fused_grouped_dequant_gemm import (
	# 		fused_dequant_grouped_gemm_bf16_fp8_triton,
	# 		fused_dequant_grouped_gemm_bf16_fp8_triton_v2
	# 	)

	# 	# This function assumes that recv_x and recv_eid are already sorted by expert id
	# 	gate_list = []
	# 	up_list = []
	# 	down_list = []
	# 	gate_scale_list = []
	# 	up_scale_list = []
	# 	down_scale_list = []
	# 	for e in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
	# 		gate_list.append(self.experts[e].fp8_gate)
	# 		up_list.append(self.experts[e].fp8_up)
	# 		down_list.append(self.experts[e].fp8_down)
	# 		gate_scale_list.append(self.experts[e].weight_dequant_scale['gate_proj.weight_scale_inv'])
	# 		up_scale_list.append(self.experts[e].weight_dequant_scale['up_proj.weight_scale_inv'])
	# 		down_scale_list.append(self.experts[e].weight_dequant_scale['down_proj.weight_scale_inv'])
	# 	eids = recv_eid - self.routed_expert_start_idx
	# 	counts = torch.bincount(eids, minlength=self.experts_per_rank)
	# 	group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)		
	# 	up = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
	# 		recv_x, up_list, up_scale_list, group_sizes, gemm_block_size = [64, 16, 128]
	# 	)
	# 	gate = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
	# 		recv_x, gate_list, gate_scale_list, group_sizes, gemm_block_size = [64, 16, 128]
	# 	)
	# 	intermediate = self.act_fn(gate) * up
	# 	res = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
	# 		intermediate, down_list, down_scale_list, group_sizes, gemm_block_size = [64, 16, 64]
	# 	)
	# 	return res
	def grouped_dequant_moe(self, recv_x, recv_eid):
		# This function assumes that recv_x and recv_eid are already sorted by expert id
		gate_list = []
		up_list = []
		down_list = []
		gate_scale_list = []
		up_scale_list = []
		down_scale_list = []
		for e in range(self.routed_expert_start_idx, self.routed_expert_end_idx):
			gate_list.append(self.experts[e].fp8_gate)
			up_list.append(self.experts[e].fp8_up)
			down_list.append(self.experts[e].fp8_down)
			gate_scale_list.append(self.experts[e].weight_dequant_scale['gate_proj.weight_scale_inv'])
			up_scale_list.append(self.experts[e].weight_dequant_scale['up_proj.weight_scale_inv'])
			down_scale_list.append(self.experts[e].weight_dequant_scale['down_proj.weight_scale_inv'])
		eids = recv_eid - self.routed_expert_start_idx
		counts = torch.bincount(eids, minlength=self.experts_per_rank)
		group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)	
		group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=self.device)
		group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
		group_start_indices[0] = 0  # The first group starts at index 0	

		intermediate = fused_dequant_weighted_moe_stage_1(
			recv_x, gate_list, up_list, gate_scale_list, up_scale_list, group_sizes, group_start_indices
		)	
		
		res = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
			intermediate, down_list, down_scale_list, group_sizes, group_start_indices, gemm_block_size = [64, 16, 64]
		)
		return res

				
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
	"""
	This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
	num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
	"""
	batch, num_key_value_heads, slen, head_dim = hidden_states.shape
	if n_rep == 1:
		return hidden_states
	hidden_states = hidden_states[:, :, None, :, :].expand(
		batch, num_key_value_heads, n_rep, slen, head_dim
	)
	return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DeepseekV3Attention(nn.Module):
	"""Multi-headed attention from 'Attention Is All You Need' paper"""

	def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
		super().__init__()
		self.config = config
		self.layer_idx = layer_idx
		if layer_idx is None:
			logger.warning_once(
				f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
				"to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
				"when creating this class."
			)

		self.attention_dropout = config.attention_dropout
		self.hidden_size = config.hidden_size
		self.num_heads = config.num_attention_heads

		self.max_position_embeddings = config.max_position_embeddings
		self.rope_theta = config.rope_theta
		self.q_lora_rank = config.q_lora_rank
		self.qk_rope_head_dim = config.qk_rope_head_dim
		self.kv_lora_rank = config.kv_lora_rank
		self.v_head_dim = config.v_head_dim
		self.qk_nope_head_dim = config.qk_nope_head_dim
		self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

		self.is_causal = True

		if self.q_lora_rank is None:
			self.q_proj = nn.Linear(
				self.hidden_size, self.num_heads * self.q_head_dim, bias=False
			)
		else:
			self.q_a_proj = nn.Linear(
				self.hidden_size, config.q_lora_rank, bias=config.attention_bias
			)
			self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
			self.q_b_proj = nn.Linear(
				config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
			)

		self.kv_a_proj_with_mqa = nn.Linear(
			self.hidden_size,
			config.kv_lora_rank + config.qk_rope_head_dim,
			bias=config.attention_bias,
		)
		self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
		self.kv_b_proj = nn.Linear(
			config.kv_lora_rank,
			self.num_heads
			* (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
			bias=False,
		)

		self.o_proj = nn.Linear(
			self.num_heads * self.v_head_dim,
			self.hidden_size,
			bias=config.attention_bias,
		)
		self._init_rope()

		# self.softmax_scale = self.q_head_dim ** (-0.5)
		# self.softmax_scale = 576 ** (-0.5)  # use fixed scale to match DeepseekV3
		self.qkv_materialized_softmax_scale = (self.q_head_dim) ** -0.5
		self.qkv_unmaterialized_softmax_scale = (576) ** -0.5
		if self.config.rope_scaling is not None:
			mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
			scaling_factor = self.config.rope_scaling["factor"]
			if mscale_all_dim:
				mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
				# self.softmax_scale = self.softmax_scale * mscale * mscale
				self.qkv_materialized_softmax_scale = (
					self.qkv_materialized_softmax_scale * mscale * mscale
				)
				self.qkv_unmaterialized_softmax_scale = (
					self.qkv_unmaterialized_softmax_scale * mscale * mscale
				)
		self.softmax_scale = self.qkv_materialized_softmax_scale

	def initialize(self):
		if self.config.phase == "decoding":
			kv_b_proj = self.kv_b_proj.weight.view(
				self.num_heads, -1, self.kv_lora_rank
			)
			self.q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
			self.out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

	def _init_rope(self):
		if self.config.rope_scaling is None:
			self.rotary_emb = DeepseekV3RotaryEmbedding(
				self.qk_rope_head_dim,
				max_position_embeddings=self.max_position_embeddings,
				base=self.rope_theta,
			)
		else:
			scaling_type = self.config.rope_scaling["type"]
			scaling_factor = self.config.rope_scaling["factor"]
			if scaling_type == "linear":
				self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
					self.qk_rope_head_dim,
					max_position_embeddings=self.max_position_embeddings,
					scaling_factor=scaling_factor,
					base=self.rope_theta,
				)
			elif scaling_type == "dynamic":
				self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
					self.qk_rope_head_dim,
					max_position_embeddings=self.max_position_embeddings,
					scaling_factor=scaling_factor,
					base=self.rope_theta,
				)
			elif scaling_type == "yarn":
				kwargs = {
					key: self.config.rope_scaling[key]
					for key in [
						"original_max_position_embeddings",
						"beta_fast",
						"beta_slow",
						"mscale",
						"mscale_all_dim",
					]
					if key in self.config.rope_scaling
				}
				self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
					self.qk_rope_head_dim,
					max_position_embeddings=self.max_position_embeddings,
					scaling_factor=scaling_factor,
					base=self.rope_theta,
					**kwargs,
				)
			else:
				raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

	def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
		return (
			tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
			.transpose(1, 2)
			.contiguous()
		)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
		**kwargs,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		if "padding_mask" in kwargs:
			warnings.warn(
				"Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
			)
		bsz, q_len, _ = hidden_states.size()

		if self.q_lora_rank is None:
			q = self.q_proj(hidden_states)
		else:
			q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
		q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
		q_nope, q_pe = torch.split(
			q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
		)

		compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
		compressed_kv, k_pe = torch.split(
			compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
		)
		k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
		kv = (
			self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
			.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
			.transpose(1, 2)
		)

		k_nope, value_states = torch.split(
			kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
		)
		kv_seq_len = value_states.shape[-2]
		if past_key_value is not None:
			if self.layer_idx is None:
				raise ValueError(
					f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
					"for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
					"with a layer index."
				)
			kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
		cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

		q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

		query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
		query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
		query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

		key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
		key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
		key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
		if past_key_value is not None:
			cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
			key_states, value_states = past_key_value.update(
				key_states, value_states, self.layer_idx, cache_kwargs
			)

		attn_weights = (
			torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
		)

		if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
			raise ValueError(
				f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
				f" {attn_weights.size()}"
			)
		assert attention_mask is not None
		if attention_mask is not None:
			if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
				raise ValueError(
					f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
				)
			attn_weights = attn_weights + attention_mask

		# upcast attention to fp32
		attn_weights = nn.functional.softmax(
			attn_weights, dim=-1, dtype=torch.float32
		).to(query_states.dtype)
		attn_weights = nn.functional.dropout(
			attn_weights, p=self.attention_dropout, training=self.training
		)
		attn_output = torch.matmul(attn_weights, value_states)

		if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
			raise ValueError(
				f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
				f" {attn_output.size()}"
			)

		attn_output = attn_output.transpose(1, 2).contiguous()

		attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

		attn_output = self.o_proj(attn_output)

		if not output_attentions:
			attn_weights = None

		return attn_output, attn_weights, past_key_value


class DeepseekV3Attention_FlashMLA_Decoding_CUDAGraph(nn.Module):
	"""
		TODO: since input kv tensor is not the same in each run. 
	"""
	def __init__(self, base_model):
		super().__init__()
		self.base_model = base_model
		self.graph = None
		self.static_inputs = {}
		self.static_outputs = {}
		self.is_graph_captured = False
		self.input_shapes = {}	
	
	def _create_static_tensors(self, **kwargs):
		"""Create static versions of all input tensors"""
		static_inputs = {}		
		for key, value in kwargs.items():
			if isinstance(value, torch.Tensor):
				# Create static tensor with same properties
				static_inputs[key] = torch.empty_like(value)
				# Store shape info for validation
				self.input_shapes[key] = tuple(value.shape)
			elif isinstance(value, dict):
				# Handle dictionary of tensors (like weight_scale)
				static_dict = {}
				for dict_key, dict_value in value.items():
					if isinstance(dict_value, torch.Tensor):
						static_dict[dict_key] = torch.empty_like(dict_value)
					else:
						static_dict[dict_key] = dict_value
				static_inputs[key] = static_dict
			else:
				# Non-tensor inputs (keep as-is)
				static_inputs[key] = value
				
		return static_inputs	

	def _copy_inputs_to_static(self, static_inputs, **kwargs):
		"""Copy input data to static tensors"""
		for key, value in kwargs.items():
			if isinstance(value, torch.Tensor):
				static_inputs[key].copy_(value, non_blocking=True)
			elif isinstance(value, dict):
				# Handle dictionary of tensors
				for dict_key, dict_value in value.items():
					if isinstance(dict_value, torch.Tensor):
						static_inputs[key][dict_key].copy_(dict_value, non_blocking=True)

	def _validate_input_shapes(self, **kwargs):
		"""Validate that input shapes match captured shapes"""
		for key, value in kwargs.items():
			if isinstance(value, torch.Tensor):
				current_shape = tuple(value.shape)
				if key in self.input_shapes and self.input_shapes[key] != current_shape:
					raise ValueError(
						f"Input tensor '{key}' shape mismatch: "
						f"expected {self.input_shapes[key]}, got {current_shape}. "
						f"CUDA graphs require static shapes."
					)
							
	
	def capture_graph(self, **kwargs):
		"""Capture CUDA graph for the MLA decoding method"""
		self.base_model.eval()
		
		# Create static versions of all inputs
		self.static_inputs = self._create_static_tensors(**kwargs)
		
		# Copy input data to static tensors
		self._copy_inputs_to_static(self.static_inputs, **kwargs)
		
		# Warmup phase - critical for proper graph capture
		print("Starting warmup for CUDA graph capture...")
		torch.cuda.synchronize()
		
		# Use a specific stream for better control
		stream = torch.cuda.Stream()
		with torch.cuda.stream(stream):
			for i in range(10):
				with torch.inference_mode():
					try:
						outputs = self.base_model.mla_decoding_flashmla_attn_mode_3(**self.static_inputs)
						if i == 0:
							print(f"Warmup successful, output types: {[type(o) for o in outputs]}")
					except Exception as e:
						print(f"Warmup iteration {i} failed: {e}")
						raise
		
		stream.synchronize()
		print("Warmup completed, starting graph capture...")
		
		# Capture the graph on the same stream
		with torch.cuda.stream(stream):
			self.graph = torch.cuda.CUDAGraph()
			with torch.cuda.graph(self.graph, stream=stream):
				# Capture the method call with all arguments
				self.static_outputs = self.base_model.mla_decoding_flashmla_attn_mode_3(**self.static_inputs)
		
		stream.synchronize()
		self.is_graph_captured = True
		print("CUDA graph capture completed successfully!")
	
	def forward(self, 
				hidden_states: torch.Tensor,
				past_key_states: torch.Tensor,
				past_value_states: torch.Tensor,
				attention_mask: torch.Tensor,
				position_ids: torch.Tensor,
				scale: torch.Tensor
	):
		
		# Collect all arguments
		kwargs = {
			'hidden_states': hidden_states,
			'past_key_states': past_key_states,
			'past_value_states': past_value_states,
			'attention_mask': attention_mask,
			'position_ids': position_ids,
			'scale': scale
		}
		
		if not self.is_graph_captured:
			print("First call - capturing CUDA graph...")
			self.capture_graph(**kwargs)
		else:
			# Validate shapes match captured shapes
			self._validate_input_shapes(**kwargs)
		
		# Copy new input data to static tensors
		self._copy_inputs_to_static(self.static_inputs, **kwargs)
		
		# Replay the graph
		self.graph.replay()
		
		# Extract outputs (the method returns tuple of tensors)
		# Note: past_key_states and scale are modified in-place, so we return the static versions
		if isinstance(self.static_outputs, tuple):
			attn_output, updated_past_key_states, updated_scale = self.static_outputs
			return (
				attn_output.clone(),
				self.static_inputs['past_key_states'].clone(),  # This was modified in-place
				self.static_inputs['scale'].clone()            # This was modified in-place
			)
		else:
			return self.static_outputs.clone()


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->DeepseekV3
class DeepseekV3FlashAttention2(DeepseekV3Attention):
	"""
	DeepseekV3 flash attention module. This module inherits from `DeepseekV3Attention` as the weights of the module stays
	untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
	flash attention and deal with padding tokens in case the input contains any of them.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
		# flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
		# Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
		self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Cache] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
		**kwargs,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		# DeepseekV3FlashAttention2 attention does not support output_attentions
		if "padding_mask" in kwargs:
			warnings.warn(
				"Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
			)

			# overwrite attention_mask with padding_mask
			attention_mask = kwargs.pop("padding_mask")

		output_attentions = False

		bsz, q_len, _ = hidden_states.size()

		if self.q_lora_rank is None:
			q = self.q_proj(hidden_states)
		else:
			q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
		q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
		q_nope, q_pe = torch.split(
			q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
		)

		# Flash attention requires the input to have the shape
		# batch_size x seq_length x head_dim x hidden_dim
		# therefore we just need to keep the original shape
		compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
		compressed_kv, k_pe = torch.split(
			compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
		)
		k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
		kv = (
			self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
			.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
			.transpose(1, 2)
		)

		k_nope, value_states = torch.split(
			kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
		)
		kv_seq_len = value_states.shape[-2]

		kv_seq_len = value_states.shape[-2]
		if past_key_value is not None:
			kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

		cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
		q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

		query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
		query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
		query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

		key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
		key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
		key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

		if self.q_head_dim != self.v_head_dim:
			value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

		if past_key_value is not None:
			cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
			key_states, value_states = past_key_value.update(
				key_states, value_states, self.layer_idx, cache_kwargs
			)

		# TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
		# to be able to avoid many of these transpose/reshape/view.
		query_states = query_states.transpose(1, 2)
		key_states = key_states.transpose(1, 2)
		value_states = value_states.transpose(1, 2)

		dropout_rate = self.attention_dropout if self.training else 0.0

		# In PEFT, usually we cast the layer norms in float32 for training stability reasons
		# therefore the input hidden states gets silently casted in float32. Hence, we need
		# cast them back in the correct dtype just to be sure everything works as expected.
		# This might slowdown training & inference so it is recommended to not cast the LayerNorms
		# in fp32. (DeepseekV3RMSNorm handles it correctly)

		input_dtype = query_states.dtype
		if input_dtype == torch.float32:
			# Handle the case where the model is quantized
			if hasattr(self.config, "_pre_quantization_dtype"):
				target_dtype = self.config._pre_quantization_dtype
			elif torch.is_autocast_enabled():
				target_dtype = torch.get_autocast_gpu_dtype()
			else:
				target_dtype = (
					self.q_proj.weight.dtype
					if self.q_lora_rank is None
					else self.q_a_proj.weight.dtype
				)

			logger.warning_once(
				f"The input hidden states seems to be silently casted in float32, this might be related to"
				f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
				f" {target_dtype}."
			)

			query_states = query_states.to(target_dtype)
			key_states = key_states.to(target_dtype)
			value_states = value_states.to(target_dtype)

		attn_output = self._flash_attention_forward(
			query_states,
			key_states,
			value_states,
			attention_mask,
			q_len,
			dropout=dropout_rate,
			softmax_scale=self.softmax_scale,
		)
		if self.q_head_dim != self.v_head_dim:
			attn_output = attn_output[:, :, :, : self.v_head_dim]

		attn_output = attn_output.reshape(
			bsz, q_len, self.num_heads * self.v_head_dim
		).contiguous()
		attn_output = self.o_proj(attn_output)

		if not output_attentions:
			attn_weights = None

		return attn_output, attn_weights, past_key_value

	def _flash_attention_forward(
		self,
		query_states,
		key_states,
		value_states,
		attention_mask,
		query_length,
		dropout=0.0,
		softmax_scale=None,
	):
		"""
		Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
		first unpad the input, then computes the attention scores and pad the final attention scores.

		Args:
			query_states (`torch.Tensor`):
				Input query states to be passed to Flash Attention API
			key_states (`torch.Tensor`):
				Input key states to be passed to Flash Attention API
			value_states (`torch.Tensor`):
				Input value states to be passed to Flash Attention API
			attention_mask (`torch.Tensor`):
				The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
				position of padding tokens and 1 for the position of non-padding tokens.
			dropout (`int`, *optional*):
				Attention dropout
			softmax_scale (`float`, *optional*):
				The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
		"""
		if not self._flash_attn_uses_top_left_mask:
			causal = self.is_causal
		else:
			# TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in DeepseekV3FlashAttention2 __init__.
			causal = self.is_causal and query_length != 1

		# Contains at least one padding token in the sequence
		if attention_mask is not None:
			batch_size = query_states.shape[0]
			(
				query_states,
				key_states,
				value_states,
				indices_q,
				cu_seq_lens,
				max_seq_lens,
			) = self._upad_input(
				query_states, key_states, value_states, attention_mask, query_length
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
				dropout_p=dropout,
				softmax_scale=softmax_scale,
				causal=causal,
			)

			attn_output = pad_input(
				attn_output_unpad, indices_q, batch_size, query_length
			)
		else:
			attn_output = flash_attn_func(
				query_states,
				key_states,
				value_states,
				dropout,
				softmax_scale=softmax_scale,
				causal=causal,
			)

		return attn_output

	def _upad_input(
		self, query_layer, key_layer, value_layer, attention_mask, query_length
	):
		indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
		batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

		key_layer = index_first_axis(
			key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
			indices_k,
		)
		value_layer = index_first_axis(
			value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
			indices_k,
		)
		if query_length == kv_seq_len:
			query_layer = index_first_axis(
				query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
				indices_k,
			)
			cu_seqlens_q = cu_seqlens_k
			max_seqlen_in_batch_q = max_seqlen_in_batch_k
			indices_q = indices_k
		elif query_length == 1:
			max_seqlen_in_batch_q = 1
			cu_seqlens_q = torch.arange(
				batch_size + 1, dtype=torch.int32, device=query_layer.device
			)  # There is a memcpy here, that is very bad.
			indices_q = cu_seqlens_q[:-1]
			query_layer = query_layer.squeeze(1)
		else:
			# The -q_len: slice assumes left padding.
			attention_mask = attention_mask[:, -query_length:]
			query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
				query_layer, attention_mask
			)

		return (
			query_layer,
			key_layer,
			value_layer,
			indices_q,
			(cu_seqlens_q, cu_seqlens_k),
			(max_seqlen_in_batch_q, max_seqlen_in_batch_k),
		)


ATTENTION_CLASSES = {
	"eager": DeepseekV3Attention,
	"flash_attention_2": DeepseekV3FlashAttention2,
}


class DeepseekV3DecoderLayer(nn.Module):
	def __init__(self, config: DeepseekV3Config, layer_idx: int, comm):
		super().__init__()
		self.hidden_size = config.hidden_size

		self.self_attn = ATTENTION_CLASSES[config._attn_implementation](
			config=config, layer_idx=layer_idx
		)
		self.comm = comm
		if config.phase == "prefill":
			cls = DeepseekV3MoE_Prefill
		elif config.phase == "decoding":
			cls = DeepseekV3MoE_Decoding_FP8
		else:
			cls = DeepseekV3MoE_Prefill
		
		self.mlp = (
			cls(config, self.comm)
			if (
				config.n_routed_experts is not None
				and layer_idx >= config.first_k_dense_replace
				and layer_idx % config.moe_layer_freq == 0
			)
			else DeepseekV3MLP(config)
		)
		self.input_layernorm = DeepseekV3RMSNorm(
			config.hidden_size, eps=config.rms_norm_eps
		)
		self.post_attention_layernorm = DeepseekV3RMSNorm(
			config.hidden_size, eps=config.rms_norm_eps
		)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_value: Optional[Tuple[torch.Tensor]] = None,
		output_attentions: Optional[bool] = False,
		use_cache: Optional[bool] = False,
		**kwargs,
	) -> Tuple[
		torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
	]:
		"""
		Args:
			hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
			attention_mask (`torch.FloatTensor`, *optional*):
				attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
				query_sequence_length, key_sequence_length)` if default attention is used.
			output_attentions (`bool`, *optional*):
				Whether or not to return the attentions tensors of all attention layers. See `attentions` under
				returned tensors for more detail.
			use_cache (`bool`, *optional*):
				If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
				(see `past_key_values`).
			past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
		"""
		if "padding_mask" in kwargs:
			warnings.warn(
				"Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
			)
		residual = hidden_states

		# logger.warning(f"Input layernorm weight dtype: {self.input_layernorm.weight.dtype}")
		hidden_states = self.input_layernorm(hidden_states)

		# Self Attention
		hidden_states, self_attn_weights, present_key_value = self.self_attn(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_value=past_key_value,
			output_attentions=output_attentions,
			use_cache=use_cache,
			**kwargs,
		)
		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		# logger.warning(f"Post attention layernorm weight dtype: {self.post_attention_layernorm.weight.dtype}")
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if use_cache:
			outputs += (present_key_value,)

		return outputs


DeepseekV3_START_DOCSTRING = r"""
	This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
	library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
	etc.)

	This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
	Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
	and behavior.

	Parameters:
		config ([`DeepseekV3Config`]):
			Model configuration class with all the parameters of the model. Initializing with a config file does not
			load the weights associated with the model, only the configuration. Check out the
			[`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
	"The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
	DeepseekV3_START_DOCSTRING,
)
class DeepseekV3PreTrainedModel(PreTrainedModel):
	config_class = DeepseekV3Config
	base_model_prefix = "model"
	supports_gradient_checkpointing = True
	_no_split_modules = ["DeepseekV3DecoderLayer"]
	_skip_keys_device_placement = "past_key_values"
	_supports_flash_attn_2 = True
	_supports_cache_class = True

	def _init_weights(self, module):
		std = self.config.initializer_range
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()


DeepseekV3_INPUTS_DOCSTRING = r"""
	Args:
		input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
			Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
			it.

			Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
			[`PreTrainedTokenizer.__call__`] for details.

			[What are input IDs?](../glossary#input-ids)
		attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
			Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.

			[What are attention masks?](../glossary#attention-mask)

			Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
			[`PreTrainedTokenizer.__call__`] for details.

			If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
			`past_key_values`).

			If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
			and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
			information on the default strategy.

			- 1 indicates the head is **not masked**,
			- 0 indicates the head is **masked**.
		position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
			config.n_positions - 1]`.

			[What are position IDs?](../glossary#position-ids)
		past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
			Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
			blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
			returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

			Two formats are allowed:
			- a [`~cache_utils.Cache`] instance;
			- Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
			shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
			cache format.

			The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
			legacy cache format will be returned.

			If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
			have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
			of shape `(batch_size, sequence_length)`.
		inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
			is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
			model's internal embedding lookup matrix.
		use_cache (`bool`, *optional*):
			If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
			`past_key_values`).
		output_attentions (`bool`, *optional*):
			Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
			tensors for more detail.
		output_hidden_states (`bool`, *optional*):
			Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
			more detail.
		return_dict (`bool`, *optional*):
			Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
	"The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
	DeepseekV3_START_DOCSTRING,
)
class DeepseekV3Model(DeepseekV3PreTrainedModel):
	"""
	Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]

	Args:
		config: DeepseekV3Config
	"""

	def __init__(self, config: DeepseekV3Config, comm=None):
		super().__init__(config)
		self.padding_idx = config.pad_token_id
		self.vocab_size = config.vocab_size

		self.embed_tokens = nn.Embedding(
			config.vocab_size, config.hidden_size, self.padding_idx
		)
		self.comm = comm
		# if self.config.phase == "decoding":
		# 	from batchgen.distributed.utils import StatelessProcessGroup
		# 	from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
		# 	self.rank = dist.get_rank()
		# 	self.world_size = dist.get_world_size()
		# 	device = torch.device("cuda", self.rank % torch.cuda.device_count())
		# 	comm_master_addr = os.getenv("COMM_MASTER_ADDR")
		# 	try:
		# 		group = StatelessProcessGroup.create(
		# 			host=comm_master_addr,
		# 			port=20001,
		# 			rank=self.rank,
		# 			world_size=self.world_size,
		# 			data_expiration_seconds=6000,
		# 		)
		# 		self.comm = PyNcclCommunicator(
		# 			group=group,
		# 			device=device
		# 		)		
		# 	except Exception as e:
		# 		logger.error(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")
		# 		raise RuntimeError(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")
			
				
	
		# if self.config.phase == "decoding":
		# 	self.rank = dist.get_rank()
		# 	cus_group = dist.new_group(backend="nccl")
		# 	device = torch.device("cuda", self.rank % torch.cuda.device_count())
		# 	from ....distributed.device_communicators.pynccl import PyNcclCommunicator
		# 	self.comm = PyNcclCommunicator(group=cus_group, device=device)
		# 	# Check if the group size matches expected number of processes
		# 	expected_size = dist.get_world_size()
		# 	actual_size = dist.get_world_size(group=cus_group)

		# 	if actual_size == expected_size:
		# 		# print(f"Rank {self.rank}: All {actual_size} processes are in the group")
		# 		logger.info(f"Rank {self.rank}: All {actual_size} processes are in the group")
		# 	else:
		# 		# print(f"Rank {self.rank}: Only {actual_size}/{expected_size} processes in group")
		# 		logger.warning(f"Rank {self.rank}: Only {actual_size}/{expected_size} processes in group")
		# 	# Test custom communicator
		# 	try:
		# 		# Assuming your PyNcclCommunicator has similar methods
		# 		if hasattr(self.comm, 'all_reduce'):
		# 			test_tensor = torch.tensor([1.0], device=device, dtype=torch.bfloat16)
		# 			with self.comm.change_state(enable=True):
		# 				self.comm.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
		# 			self.comm.stream.synchronize()  # Ensure all-reduce is complete
		# 			expected_result = torch.tensor([dist.get_world_size()], device=device, dtype=torch.bfloat16)
					
		# 			if abs(test_tensor.item() - expected_result.item()) < 1e-2:
		# 				# print(f"Rank {self.rank}: PyNccl communicator healthy")
		# 				logger.info(f"Rank {self.rank}: PyNccl communicator healthy")
		# 			else:
		# 				# print(f"Rank {self.rank}: PyNccl communicator test failed")
		# 				logger.error(f"Rank {self.rank}: PyNccl communicator test failed - expected {expected_result}, got {test_tensor.item()}")
		# 				raise RuntimeError(
		# 					f"PyNccl communicator test failed - expected {expected_result}, got {test_tensor.item()}"
		# 				)
				
		# 		# Test barrier if available
		# 		if hasattr(self.comm, 'barrier'):
		# 			self.comm.barrier()
		# 			# print(f"Rank {self.rank}: PyNccl barrier successful")
		# 			logger.info(f"Rank {self.rank}: PyNccl barrier successful")
					
		# 	except Exception as e:
		# 		# print(f"Rank {self.rank}: PyNccl communicator test failed - {e}")
		# 		logger.error(f"Rank {self.rank}: PyNccl communicator test failed - {e}")
		self.layers = nn.ModuleList(
			[
				DeepseekV3DecoderLayer(config, layer_idx, self.comm)
				for layer_idx in range(config.num_hidden_layers)
			]
		)
		self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
		self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

		self.gradient_checkpointing = False
		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.embed_tokens

	def set_input_embeddings(self, value):
		self.embed_tokens = value

	@add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, BaseModelOutputWithPast]:
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache

		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		# retrieve input_ids and inputs_embeds
		if input_ids is not None and inputs_embeds is not None:
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time"
			)
		elif input_ids is not None:
			batch_size, seq_length = input_ids.shape[:2]
		elif inputs_embeds is not None:
			batch_size, seq_length = inputs_embeds.shape[:2]
		else:
			raise ValueError("You have to specify either input_ids or inputs_embeds")

		past_key_values_length = 0
		if use_cache:
			use_legacy_cache = not isinstance(past_key_values, Cache)
			if use_legacy_cache:
				past_key_values = DynamicCache.from_legacy_cache(past_key_values)
			past_key_values_length = past_key_values.get_usable_length(seq_length)

		if position_ids is None:
			device = input_ids.device if input_ids is not None else inputs_embeds.device
			position_ids = torch.arange(
				past_key_values_length,
				seq_length + past_key_values_length,
				dtype=torch.long,
				device=device,
			)
			position_ids = position_ids.unsqueeze(0)

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		if self._use_flash_attention_2:
			# 2d mask is passed through the layers
			attention_mask = (
				attention_mask
				if (attention_mask is not None and 0 in attention_mask)
				else None
			)
		else:
			# 4d mask is passed through the layers
			attention_mask = _prepare_4d_causal_attention_mask(
				attention_mask,
				(batch_size, seq_length),
				inputs_embeds,
				past_key_values_length,
			)

		# embed positions
		hidden_states = inputs_embeds

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = None

		for decoder_layer in self.layers:
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			layer_outputs = decoder_layer(
				hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_value=past_key_values,
				output_attentions=output_attentions,
				use_cache=use_cache,
			)

			hidden_states = layer_outputs[0]

			if use_cache:
				next_decoder_cache = layer_outputs[2 if output_attentions else 1]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

		hidden_states = self.norm(hidden_states)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		next_cache = None
		if use_cache:
			next_cache = (
				next_decoder_cache.to_legacy_cache()
				if use_legacy_cache
				else next_decoder_cache
			)
		if not return_dict:
			return tuple(
				v
				for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
				if v is not None
			)
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
	_tied_weights_keys = ["lm_head.weight"]

	def __init__(self, config, comm=None):
		super().__init__(config)
		self.model = DeepseekV3Model(config, comm)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.model.embed_tokens

	def set_input_embeddings(self, value):
		self.model.embed_tokens = value

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def set_decoder(self, decoder):
		self.model = decoder

	def get_decoder(self):
		return self.model

	@add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
	@replace_return_docstrings(
		output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
	)
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, CausalLMOutputWithPast]:
		r"""
		Args:
			labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
				Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
				config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
				(masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

		Returns:

		Example:

		```python
		>>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

		>>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
		>>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

		>>> prompt = "Hey, are you conscious? Can you talk to me?"
		>>> inputs = tokenizer(prompt, return_tensors="pt")

		>>> # Generate
		>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
		>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
		"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
		```"""
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		# decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]
		logits = self.lm_head(hidden_states)
		logits = logits.float()

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			# Flatten the tokens
			loss_fct = CrossEntropyLoss()
			shift_logits = shift_logits.view(-1, self.config.vocab_size)
			shift_labels = shift_labels.view(-1)
			# Enable model parallelism
			shift_labels = shift_labels.to(shift_logits.device)
			loss = loss_fct(shift_logits, shift_labels)

		if not return_dict:
			output = (logits,) + outputs[1:]
			return (loss,) + output if loss is not None else output

		return CausalLMOutputWithPast(
			loss=loss,
			logits=logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

	def prepare_inputs_for_generation(
		self,
		input_ids,
		past_key_values=None,
		attention_mask=None,
		inputs_embeds=None,
		**kwargs,
	):
		if past_key_values is not None:
			if isinstance(past_key_values, Cache):
				cache_length = past_key_values.get_seq_length()
				past_length = past_key_values.seen_tokens
				max_cache_length = past_key_values.get_max_length()
			else:
				cache_length = past_length = past_key_values[0][0].shape[2]
				max_cache_length = None

			# Keep only the unprocessed tokens:
			# 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
			# some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
			# input)
			if (
				attention_mask is not None
				and attention_mask.shape[1] > input_ids.shape[1]
			):
				input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
			# 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
			# input_ids based on the past_length.
			elif past_length < input_ids.shape[1]:
				input_ids = input_ids[:, past_length:]
			# 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

			# If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
			if (
				max_cache_length is not None
				and attention_mask is not None
				and cache_length + input_ids.shape[1] > max_cache_length
			):
				attention_mask = attention_mask[:, -max_cache_length:]

		position_ids = kwargs.get("position_ids", None)
		if attention_mask is not None and position_ids is None:
			# create position_ids on the fly for batch generation
			position_ids = attention_mask.long().cumsum(-1) - 1
			position_ids.masked_fill_(attention_mask == 0, 1)
			if past_key_values:
				position_ids = position_ids[:, -input_ids.shape[1] :]

		# if `inputs_embeds` are passed, we only want to use them in the 1st generation step
		if inputs_embeds is not None and past_key_values is None:
			model_inputs = {"inputs_embeds": inputs_embeds}
		else:
			model_inputs = {"input_ids": input_ids}

		model_inputs.update(
			{
				"position_ids": position_ids,
				"past_key_values": past_key_values,
				"use_cache": kwargs.get("use_cache"),
				"attention_mask": attention_mask,
			}
		)
		return model_inputs

	@staticmethod
	def _reorder_cache(past_key_values, beam_idx):
		reordered_past = ()
		for layer_past in past_key_values:
			reordered_past += (
				tuple(
					past_state.index_select(0, beam_idx.to(past_state.device))
					for past_state in layer_past
				),
			)
		return reordered_past


@add_start_docstrings(
	"""
	The DeepseekV3 Model transformer with a sequence classification head on top (linear layer).

	[`DeepseekV3ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
	(e.g. GPT-2) do.

	Since it does classification on the last token, it requires to know the position of the last token. If a
	`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
	no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
	padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
	each row of the batch).
	""",
	DeepseekV3_START_DOCSTRING,
)
class DeepseekV3ForSequenceClassification(DeepseekV3PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.model = DeepseekV3Model(config)
		self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.model.embed_tokens

	def set_input_embeddings(self, value):
		self.model.embed_tokens = value

	@add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, SequenceClassifierOutputWithPast]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
			config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
			`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
		"""
		return_dict = (
			return_dict if return_dict is not None else self.config.use_return_dict
		)

		transformer_outputs = self.model(
			input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states = transformer_outputs[0]
		logits = self.score(hidden_states)

		if input_ids is not None:
			batch_size = input_ids.shape[0]
		else:
			batch_size = inputs_embeds.shape[0]

		if self.config.pad_token_id is None and batch_size != 1:
			raise ValueError(
				"Cannot handle batch sizes > 1 if no padding token is defined."
			)
		if self.config.pad_token_id is None:
			sequence_lengths = -1
		else:
			if input_ids is not None:
				sequence_lengths = (
					torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
				).to(logits.device)
			else:
				sequence_lengths = -1

		pooled_logits = logits[
			torch.arange(batch_size, device=logits.device), sequence_lengths
		]

		loss = None
		if labels is not None:
			labels = labels.to(logits.device)
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (
					labels.dtype == torch.long or labels.dtype == torch.int
				):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				if self.num_labels == 1:
					loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
				else:
					loss = loss_fct(pooled_logits, labels)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(
					pooled_logits.view(-1, self.num_labels), labels.view(-1)
				)
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				loss = loss_fct(pooled_logits, labels)
		if not return_dict:
			output = (pooled_logits,) + transformer_outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutputWithPast(
			loss=loss,
			logits=pooled_logits,
			past_key_values=transformer_outputs.past_key_values,
			hidden_states=transformer_outputs.hidden_states,
			attentions=transformer_outputs.attentions,
		)
