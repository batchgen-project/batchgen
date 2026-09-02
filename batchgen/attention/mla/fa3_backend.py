"""
	For Hopper GPU.
	- prefill_fa3()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn_interface import flash_attn_varlen_func 
from .padding import _upad_input, pad_input
from .rotary_embedding import mla_rotary_pos_emb, rotary_pos_emb, apply_rotary_pos_emb
import deep_gemm
# from deep_gemm import get_col_major_tma_aligned_tensor
import logging
from contextlib import nullcontext
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
		MLA prefifill on hopper device.
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
	del compressed_kv

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
	del q_nope, q_pe, k_nope, k_pe, kv, normed_kv

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
	del query_states, key_states, value_states

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


# @triton.jit
# def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
# 	"""
# 	Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

# 	Args:
# 		x_ptr (triton.Pointer): Pointer to the input tensor.
# 		y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
# 		s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
# 		BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

# 	Returns:
# 		None
# 	"""
# 	pid = tl.program_id(axis=0)
# 	offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# 	x = tl.load(x_ptr + offs).to(tl.float32)
# 	s = tl.max(tl.abs(x)) / 448.
# 	y = x / s
# 	y = y.to(y_ptr.dtype.element_ty)
# 	tl.store(y_ptr + offs, y)
# 	tl.store(s_ptr + pid, s)


# def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
# 	"""
# 	Quantizes the input tensor `x` using block-wise quantization.

# 	Args:
# 		x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
# 		block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

# 	Returns:
# 		Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
# 			- The quantized tensor with dtype `torch.float8_e4m3fn`.
# 			- A tensor of scaling factors with dtype `torch.float32`.
# 	"""
# 	assert x.is_contiguous(), 'Input tensor must be contiguous'
# 	assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
# 	y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
# 	s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
# 	grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
# 	act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
# 	return y, s

""" V2 """
# @triton.jit
# def act_quant_kernel(
# 	x_ptr, 
# 	y_ptr, 
# 	scale_ptr,
# 	n_elements,
# 	eps: tl.constexpr,
# 	fp8_max: tl.constexpr,
# 	BLOCK_SIZE: tl.constexpr
# ):
# 	"""
# 	Industry standard block quantization kernel for BF16 -> FP8 E4M3.
# 	Based on NVIDIA Transformer Engine and FP8 training best practices.
# 	"""
# 	pid = tl.program_id(axis=0)
# 	block_start = pid * BLOCK_SIZE
# 	offsets = block_start + tl.arange(0, BLOCK_SIZE)
# 	mask = offsets < n_elements
	
# 	# Load input in FP32 for numerical stability
# 	x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
	
# 	# Compute absmax with epsilon for stability (industry standard)
# 	absmax = tl.max(tl.abs(x), axis=0)
# 	absmax = tl.maximum(absmax, eps)
	
# 	# Standard FP8 E4M3 scaling
# 	# FP8 E4M3 max value is 448, but use 448.0 for safety
# 	scale = absmax / fp8_max
	
# 	# Quantize and clamp
# 	x_scaled = x / scale
	
# 	# Clamp to FP8 E4M3 range - this is critical
# 	x_scaled = tl.minimum(x_scaled, fp8_max)
# 	x_scaled = tl.maximum(x_scaled, -fp8_max)
	
# 	# Convert to FP8
# 	y = x_scaled.to(y_ptr.dtype.element_ty)
	
# 	# Store outputs
# 	tl.store(y_ptr + offsets, y, mask=mask)
# 	tl.store(scale_ptr + pid, scale)


""" for 2d input """
# @triton.jit
# def act_quant_kernel_2d(
#     x_ptr,
#     y_ptr,
#     scale_ptr,
#     M, N,
#     eps: tl.constexpr,
#     fp8_max: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr
# ):
#     """
#     2D version for better efficiency with matrices.
#     Quantizes along the last dimension (row-wise).
#     """
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
	
#     # Each program handles one block in a row
#     row_start = x_ptr + pid_m * N
#     block_start = pid_n * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < N
	
#     # Load block
#     x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
	
#     # Compute scale (absmax)
#     absmax = tl.max(tl.abs(x), axis=0)
#     scale = tl.maximum(absmax, eps) / fp8_max
	
#     # Quantize
#     x_scaled = x / scale
#     x_scaled = tl.minimum(x_scaled, fp8_max)
#     x_scaled = tl.maximum(x_scaled, -fp8_max)
	
#     # Store
#     y = x_scaled.to(y_ptr.dtype.element_ty)
#     y_row_start = y_ptr + pid_m * N
#     tl.store(y_row_start + offsets, y, mask=mask)
	
#     # Store scale (one per block)
#     scale_offset = pid_m * ((N + BLOCK_SIZE - 1) // BLOCK_SIZE) + pid_n
#     tl.store(scale_ptr + scale_offset, scale)


# def act_quant(
# 	x: torch.Tensor, 
# 	block_size: int = 128,
# 	eps: float = 1e-12
# ) -> Tuple[torch.Tensor, torch.Tensor]:
# 	"""
# 	Industry standard BF16 to FP8 E4M3 block quantization.
	
# 	Args:
# 		x: Input tensor in BF16/FP16/FP32
# 		block_size: Block size for quantization (typically 128 or 256)
# 		eps: Epsilon for numerical stability (1e-12 is standard)
	
# 	Returns:
# 		y: Quantized tensor in FP8 E4M3
# 		scale: Per-block scaling factors
# 	"""
# 	assert x.is_contiguous(), 'Input must be contiguous'
	
# 	# FP8 E4M3 characteristics
# 	fp8_max = 448.0
	
# 	# Flatten all dimensions except last for block processing
# 	original_shape = x.shape
# 	x_flat = x.view(-1, x.shape[-1])
# 	M, N = x_flat.shape
	
# 	# Allocate outputs
# 	y = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
# 	num_blocks = (N + block_size - 1) // block_size
# 	scale = torch.empty((M, num_blocks), dtype=torch.float32, device=x.device)
	
# 	# Launch kernel
# 	grid = (M, num_blocks)
# 	act_quant_kernel_2d[grid](
# 		x_flat, y, scale,
# 		M, N,
# 		eps=eps,
# 		fp8_max=fp8_max,
# 		BLOCK_SIZE=block_size
# 	)
	
# 	# Restore original shape
# 	y = y.view(original_shape)
	
# 	return y, scale

@triton.jit
def act_quant_kernel_2d(
    x_ptr,
    y_ptr,
    scale_ptr,
    num_valid_tokens_ptr,
    M, N,
    eps: tl.constexpr,
    fp8_max: tl.constexpr,
    HAS_VALID_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    2D Block Quantization Kernel.
    Treats input as (Rows, Cols) -> Quantizes row segments of size BLOCK_SIZE.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate offsets
    row_start = x_ptr + pid_m * N
    block_start = pid_n * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    if HAS_VALID_TOKENS:
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        if pid_m >= num_valid_tokens:
            y_row_start = y_ptr + pid_m * N
            tl.store(y_row_start + offsets, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
            num_blocks_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
            scale_offset = pid_m * num_blocks_n + pid_n
            tl.store(scale_ptr + scale_offset, eps)
            return
    
    # Load block
    x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute scale (absmax)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(absmax, eps) / fp8_max
    
    # Quantize
    x_scaled = x / scale
    x_scaled = tl.minimum(x_scaled, fp8_max)
    x_scaled = tl.maximum(x_scaled, -fp8_max)
    
    # Store Quantized Data
    y = x_scaled.to(y_ptr.dtype.element_ty)
    y_row_start = y_ptr + pid_m * N
    tl.store(y_row_start + offsets, y, mask=mask)
    
    # Store Scale
    # Layout: (Rows, NumBlocks)
    num_blocks_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_offset = pid_m * num_blocks_n + pid_n
    tl.store(scale_ptr + scale_offset, scale)

# Maximum rows per kernel launch to avoid CUDA grid dimension limits
# and memory access issues with very long sequences
ACT_QUANT_MAX_ROWS = 32768  # 32K rows per chunk


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    eps: float = 1e-12,
    num_valid_tokens: torch.Tensor | None = None,
    scale_tma_aligned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 → FP8 blockwise quantization for the attention hot path.

    Routes bf16 block-128 inputs to `per_token_blocked_quantize_bf16_to_fp8_1d`
    (batchgen_kernels.triton.fp8_quantize). That kernel launches one CTA per
    token with a 2D [NUM_BLOCKS_P2, BLOCK_K] tile and reduces amax along
    axis=1 for all K-blocks simultaneously. Benchmarked vs the CUDA
    `act_quant_3d` kernel at M=1..100000, K=6144:
      - Bit-exact or within single-ULP rounding noise vs CUDA.
      - ~1.85x faster at M >= 4k (CUDA's warp-stride-over-K-blocks pattern
        serializes the reductions; the 2D-tile variant parallelizes them).
      - No grid-X cap concern (Triton 1D grid -> 2^31-1 on CC >= 3.0).

    Legacy Triton fallback (`act_quant_kernel_2d` below) is kept for edge
    cases where x is non-bf16 or block_size != 128.
    """
    assert x.is_contiguous(), 'Input must be contiguous'
    if num_valid_tokens is not None:
        if num_valid_tokens.device != x.device:
            raise ValueError("num_valid_tokens must be on the same device as x")
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}")
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                f"num_valid_tokens must contain one element, got {tuple(num_valid_tokens.shape)}"
            )

    original_shape = x.shape
    x_flat = x.view(-1, original_shape[-1])
    M, N = x_flat.shape

    if x.dtype == torch.bfloat16 and M > 0 and block_size == 128:
        if scale_tma_aligned and x.dim() != 2:
            raise ValueError("scale_tma_aligned act_quant currently requires a 2D input")
        from batchgen_kernels.triton.fp8_quantize import per_token_blocked_quantize_bf16_to_fp8_1d
        y_flat, scale_flat = per_token_blocked_quantize_bf16_to_fp8_1d(
            x_flat,
            block_size=block_size,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=scale_tma_aligned,
        )
        num_blocks = scale_flat.size(-1)
        y = y_flat.view(*original_shape)
        scale = scale_flat if scale_tma_aligned else scale_flat.view(*original_shape[:-1], num_blocks)
        return y, scale

    fp8_max = 448.0
    if scale_tma_aligned and x.dim() != 2:
        raise ValueError("scale_tma_aligned act_quant currently requires a 2D input")
    
    # 2. Allocate Outputs
    y = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
    num_blocks = (N + block_size - 1) // block_size
    
    # Scale shape: (Rows, NumBlocks)
    if scale_tma_aligned:
        aligned_m = ((M + 3) // 4) * 4
        scale = torch.empty((num_blocks, aligned_m), dtype=torch.float32, device=x.device).transpose(0, 1)[:M, :]
    else:
        scale = torch.empty((M, num_blocks), dtype=torch.float32, device=x.device)
    
    # 3. Launch Kernel - chunk if needed for long sequences
    if M <= ACT_QUANT_MAX_ROWS:
        # Single kernel launch for short sequences
        grid = (M, num_blocks)
        act_quant_kernel_2d[grid](
            x_flat, y, scale,
            num_valid_tokens if num_valid_tokens is not None else scale,
            M, N,
            eps=eps,
            fp8_max=fp8_max,
            HAS_VALID_TOKENS=num_valid_tokens is not None,
            BLOCK_SIZE=block_size
        )
    else:
        if num_valid_tokens is not None:
            raise ValueError("num_valid_tokens is not supported by chunked fallback act_quant")
        # Chunked processing for long sequences
        for chunk_start in range(0, M, ACT_QUANT_MAX_ROWS):
            chunk_end = min(chunk_start + ACT_QUANT_MAX_ROWS, M)
            chunk_size = chunk_end - chunk_start
            
            # Get views into the chunk
            x_chunk = x_flat[chunk_start:chunk_end]
            y_chunk = y[chunk_start:chunk_end]
            scale_chunk = scale[chunk_start:chunk_end]
            
            grid = (chunk_size, num_blocks)
            act_quant_kernel_2d[grid](
                x_chunk, y_chunk, scale_chunk,
                scale_chunk,
                chunk_size, N,
                eps=eps,
                fp8_max=fp8_max,
                HAS_VALID_TOKENS=False,
                BLOCK_SIZE=block_size
            )
    
    # 4. Restore Shapes
    # Reshape Y back to (E, T, H)
    y = y.view(original_shape)
    
    # CRITICAL: Reshape Scale back to (E, T, NumBlocks)
    # This allows downstream kernels to calculate strides correctly.
    if not scale_tma_aligned:
        scale_shape = list(original_shape[:-1]) + [num_blocks]
        scale = scale.view(*scale_shape)

    return y, scale



# @triton.jit
# def act_quant_kernel_2d(
#     x_ptr,
#     y_ptr,
#     scale_ptr,
#     M, N,
#     eps: tl.constexpr,
#     fp8_max: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr
# ):
#     """
#     Quantizes each row (token) independently in blocks.
#     This kernel is already correct!
#     """
#     pid_m = tl.program_id(axis=0)  # Token index
#     pid_n = tl.program_id(axis=1)  # Block index within token
	
#     # Each program handles one block in one token
#     row_start = x_ptr + pid_m * N
#     block_start = pid_n * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < N
	
#     # Load block
#     x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
	
#     # Compute scale for this block
#     absmax = tl.max(tl.abs(x), axis=0)
#     scale = tl.maximum(absmax, eps) / fp8_max
	
#     # Quantize
#     x_scaled = x / scale
#     x_scaled = tl.minimum(x_scaled, fp8_max)
#     x_scaled = tl.maximum(x_scaled, -fp8_max)
	
#     # Store
#     y = x_scaled.to(tl.float8e4nv)  # Fixed: explicit FP8 type
#     y_row_start = y_ptr + pid_m * N
#     tl.store(y_row_start + offsets, y, mask=mask)
	
#     # Store scale: one per block per token
#     scale_offset = pid_m * ((N + BLOCK_SIZE - 1) // BLOCK_SIZE) + pid_n
#     tl.store(scale_ptr + scale_offset, scale)


# def act_quant(
#     x: torch.Tensor,  # [bsz * seq_len, hidden_dim]
#     block_size: int = 128,
#     eps: float = 1e-12
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Quantize activations with per-token block quantization.
	
#     Args:
#         x: Input tensor [num_tokens, hidden_dim]
#         block_size: Block size for quantization (128)
#         eps: Epsilon for numerical stability
		
#     Returns:
#         y: Quantized tensor [num_tokens, hidden_dim] in FP8
#         scale: Scale factors [num_tokens, hidden_dim // block_size]
#     """
#     assert x.is_contiguous(), 'Input must be contiguous'
#     assert x.dim() == 2, 'Expected 2D tensor [num_tokens, hidden_dim]'
	
#     M, N = x.shape  # M = num_tokens, N = hidden_dim
#     fp8_max = 448.0
	
#     # Allocate outputs
#     y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
#     num_blocks = (N + block_size - 1) // block_size
#     scale = torch.empty((M, num_blocks), dtype=torch.float32, device=x.device)
	
#     # ALWAYS use 2D kernel for per-token block quantization
#     # Never use the 1D kernel as it doesn't respect token boundaries
#     grid = (M, num_blocks)
#     act_quant_kernel_2d[grid](
#         x, y, scale,
#         M, N,
#         eps=eps,
#         fp8_max=fp8_max,
#         BLOCK_SIZE=block_size,
#         num_warps=4,
#         num_stages=2
#     )
	
#     return y, scale

# from batchgen.quantization.block_quantization import act_quant_transposed_scale
def w8a16_gemm(
	weight_data_fp8: torch.Tensor,
	weight_scale_inv_fp32: torch.Tensor,
	activation_bf16: torch.Tensor,
	activation_fp8: Tuple[torch.Tensor, torch.Tensor] | None = None,
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

	# act_quant now routes bf16 inputs to the validated batchgen_kernels quant;
	# non-bf16 / empty sub-batches fall back to the legacy in-file Triton path.
	x_fp8 = activation_fp8 if activation_fp8 is not None else act_quant(x)
	# disable_ue8m0_cast removed — on Hopper (SM90) the flag is a no-op
	# (layout.hpp:22 early-exits for arch_major==9 regardless), and omitting
	# lets DeepGEMM's default handling apply (same as SGLang). This also
	# ensures Blackwell upgrade path uses UE8M0 natively when appropriate.
	deep_gemm.fp8_gemm_nt(x_fp8, y_fp8, out)
	if activation_bf16.dim() == 3:
		out = out.view(n_group, l, n)
	else:
		out = out.view(m, n)
	return out

def w8a16_gemm_dequant(
	weight_data_fp8: torch.Tensor,
	weight_scale_inv_fp32: torch.Tensor,
	activation_bf16: torch.Tensor,
) -> torch.Tensor:
	"""True W8A16: dequant FP8 weight → BF16, then BF16 matmul.

	Avoids quantizing the activation to FP8 (which w8a16_gemm does despite
	its name). Uses the validated deepseek_v3_dequantization Triton kernel
	for blocked FP8→BF16 weight dequant.
	"""
	from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization

	assert weight_data_fp8.dim() == 2
	assert activation_bf16.dim() in (2, 3)
	is_3d = activation_bf16.dim() == 3
	if is_3d:
		n_group, l, _ = activation_bf16.size()

	weight_bf16 = deepseek_v3_dequantization(
		weight_data_fp8, weight_scale_inv_fp32, block_size=128
	)
	x = activation_bf16.view(-1, activation_bf16.size(-1))
	out = torch.mm(x, weight_bf16.T)
	if is_3d:
		out = out.view(n_group, l, -1)
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
		MLA prefifill on hopper device.
		Materialize QKV and call flash_attn_varlen_func(). (flash_attn_3 backend)
	"""
	bsz, seq_len, _ = hidden_states.shape
	# Debug: Log dtypes before w8a16_gemm
	logging.info(f"[DEBUG w8a16_gemm inputs] hidden_states.dtype={hidden_states.dtype}, "
				 f"q_a_proj.weight.dtype={self.q_a_proj.weight.data.dtype}, "
				 f"scale.dtype={weight_scale['q_a_proj.weight_scale_inv'].dtype}")

	query_states = w8a16_gemm(
		self.q_a_proj.weight.data,
		weight_scale["q_a_proj.weight_scale_inv"],
		hidden_states
	)

	# Debug: Log dtype after w8a16_gemm, before layernorm
	logging.info(f"[DEBUG after w8a16_gemm] query_states.dtype={query_states.dtype}, "
				 f"q_a_layernorm.weight.dtype={self.q_a_layernorm.weight.dtype}, "
				 f"variance_epsilon={self.q_a_layernorm.variance_epsilon}")

	query_states = self.q_a_layernorm(query_states)
	query_states = w8a16_gemm(
		self.q_b_proj.weight.data,
		weight_scale["q_b_proj.weight_scale_inv"],
		query_states
	)

	query_states = query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim)
	q_nope, q_pe = torch.split(
		query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)
	compressed_kv = w8a16_gemm(
		self.kv_a_proj_with_mqa.weight.data,
		weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
		hidden_states
	)
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
	attn_output_unpad = flash_attn_varlen_func(
		query_states,
		key_states,
		value_states,
		cu_seqlens_q=cu_seqlens_q,
		cu_seqlens_k=cu_seqlens_k,
		max_seqlen_q=max_seqlen_in_batch_q,
		max_seqlen_k=max_seqlen_in_batch_k,
		# softmax_scale=self.qkv_materialized_softmax_scale,
		softmax_scale=self.softmax_scale,
		# softmax_scale=self.q_head_dim ** (-0.5),
		causal=True
	)
	# if attn_output_unpad is a tuple, we use attn_output_unpad[0]
	if isinstance(attn_output_unpad, tuple):
		attn_output_unpad = attn_output_unpad[0]		

	attn_output = pad_input(attn_output_unpad, indices_q, bsz, seq_len).view(
		bsz, seq_len, self.num_heads * self.v_head_dim
	).contiguous()
	attn_output = w8a16_gemm(
		self.o_proj.weight.data,
		weight_scale["o_proj.weight_scale_inv"],
		attn_output
	)

	return attn_output, offload_kv

# mla_prefill_w8a16_deepgemm
@torch.inference_mode()
def mla_prefill_w8a16_deepgemm(
	self,
	hidden_states: torch.Tensor,
	attention_mask: torch.Tensor,
	position_ids: torch.Tensor,
	weight_scale: dict[str, torch.Tensor],

) -> tuple[torch.Tensor, torch.Tensor]:
	"""
		MLA prefifill on hopper device.
		Materialize QKV and call flash_attn_varlen_func(). (flash_attn_3 backend)
	"""
	bsz, seq_len, _ = hidden_states.shape
	query_states = w8a16_gemm(
		self.q_a_proj.weight.data,
		weight_scale["q_a_proj.weight_scale_inv"],
		hidden_states
	)

	query_states = self.q_a_layernorm(query_states)
	query_states = w8a16_gemm(
		self.q_b_proj.weight.data,
		weight_scale["q_b_proj.weight_scale_inv"],
		query_states
	)

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

	attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
	attention_mask = torch.where(attention_mask == 0, torch.finfo(torch.bfloat16).min, torch.tensor(0.0, dtype=torch.bfloat16, device=attention_mask.device))
	attn_output = F.scaled_dot_product_attention(
		query_states.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2),
		key_states.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2),  
		value_states.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2),
		attn_mask=attention_mask,
		dropout_p=0.0,
		is_causal=True,
	).transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.v_head_dim)

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
		MLA prefifill on hopper device.
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


# ============ Prepacked Prefill Functions ============

@torch.inference_mode()
def mla_prefill_flashattention3_prepacked(
	self,
	hidden_states: torch.Tensor,
	position_ids: torch.Tensor,
	cu_seqlens: torch.Tensor,
	max_seqlen: int,
	num_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	MLA prefill on Hopper device for PREPACKED sequences.

	This function handles prepacked input where multiple sequences are packed
	together with proper cu_seqlens to separate them. No padding removal needed
	since the input is already densely packed.

	Args:
		self: The attention module
		hidden_states: [total_tokens, hidden_dim] - all tokens concatenated
		position_ids: [total_tokens] - position within each sequence
		cu_seqlens: [num_sequences + 1] - cumulative sequence lengths
		max_seqlen: Maximum sequence length in the batch
		num_sequences: Number of sequences

	Returns:
		attn_output: [total_tokens, hidden_dim]
		offload_kv: [total_tokens, kv_lora_rank + qk_rope_head_dim] for KV cache
	"""
	total_tokens = hidden_states.shape[0]

	# Project Q
	query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
	query_states = query_states.view(total_tokens, self.num_heads, self.q_head_dim)
	q_nope, q_pe = torch.split(
		query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	# Project KV
	compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
	compressed_kv, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	normed_kv = self.kv_a_layernorm(compressed_kv)
	k_pe = k_pe.view(total_tokens, 1, self.qk_rope_head_dim)

	# Apply rotary embeddings
	cos, sin = self.rotary_emb(q_pe.unsqueeze(0), seq_len=max_seqlen)
	# For prepacked, position_ids is 1D [total_tokens]
	q_pe = rotary_pos_emb(q_pe.unsqueeze(0), cos, sin, position_ids.unsqueeze(0), 2).squeeze(0)
	k_pe = rotary_pos_emb(k_pe.unsqueeze(0), cos, sin, position_ids.unsqueeze(0), 2).squeeze(0)

	k_pe_flat = k_pe.view(total_tokens, self.qk_rope_head_dim)
	offload_kv = torch.cat([normed_kv, k_pe_flat], dim=-1)
	del compressed_kv, k_pe_flat

	# Expand KV
	kv = self.kv_b_proj(normed_kv)
	kv = kv.view(total_tokens, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)

	# Assemble Q and K
	query_states = k_pe.new_empty(total_tokens, self.num_heads, self.q_head_dim)
	query_states[:, :, :self.qk_nope_head_dim] = q_nope
	query_states[:, :, self.qk_nope_head_dim:] = q_pe

	key_states = k_pe.new_empty(total_tokens, self.num_heads, self.q_head_dim)
	k_pe = k_pe.view(total_tokens, 1, self.qk_rope_head_dim)
	key_states[:, :, :self.qk_nope_head_dim] = k_nope
	key_states[:, :, self.qk_nope_head_dim:] = k_pe
	del q_nope, q_pe, k_nope, k_pe, kv, normed_kv

	query_states = query_states.contiguous()
	key_states = key_states.contiguous()
	value_states = value_states.contiguous()

	# Flash attention with varlen - input is already packed, no unpadding needed
	attn_output = flash_attn_varlen_func(
		query_states,
		key_states,
		value_states,
		cu_seqlens_q=cu_seqlens,
		cu_seqlens_k=cu_seqlens,
		max_seqlen_q=max_seqlen,
		max_seqlen_k=max_seqlen,
		softmax_scale=self.softmax_scale,
		causal=True
	)
	del query_states, key_states, value_states

	# Handle tuple return from flash_attn
	if isinstance(attn_output, tuple):
		attn_output = attn_output[0]

	attn_output = attn_output.view(total_tokens, self.num_heads * self.v_head_dim).contiguous()
	attn_output = self.o_proj(attn_output)

	return attn_output, offload_kv


@torch.inference_mode()
def mla_prefill_flashattention3_w8a16_deepgemm_prepacked(
	self,
	hidden_states: torch.Tensor,
	position_ids: torch.Tensor,
	cu_seqlens: torch.Tensor,
	max_seqlen: int,
	num_sequences: int,
	weight_scale: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	MLA prefill with W8A16 quantization for PREPACKED sequences.

	Args:
		self: The attention module
		hidden_states: [total_tokens, hidden_dim] - all tokens concatenated
		position_ids: [total_tokens] - position within each sequence
		cu_seqlens: [num_sequences + 1] - cumulative sequence lengths
		max_seqlen: Maximum sequence length in the batch
		num_sequences: Number of sequences
		weight_scale: Dictionary of weight scales for quantized weights

	Returns:
		attn_output: [total_tokens, hidden_dim]
		offload_kv: [total_tokens, kv_lora_rank + qk_rope_head_dim]
	"""
	total_tokens = hidden_states.shape[0]
	from batchgen.timing import get_prefill_timer
	_prefill_timer = get_prefill_timer()
	_layer_idx = getattr(self, "layer_idx", -1)

	def _timed(name: str):
		if _prefill_timer is None:
			return nullcontext()
		return _prefill_timer.timed(name, _layer_idx)

	# Default: FP8 act_quant + DeepGEMM fp8_gemm_nt (matches SGLang/DeepGEMM
	# blockwise FP8 semantics and the decode path's w8a8_deepgemm). Opt into
	# the dequant-to-BF16 path via BATCHGEN_W8A16_DEQUANT=1.
	import os as _os_gemm
	_w8a16_dequant_path = _os_gemm.environ.get("BATCHGEN_W8A16_DEQUANT", "0") == "1"
	_gemm = w8a16_gemm_dequant if _w8a16_dequant_path else w8a16_gemm
	_input_fp8 = None
	if not _w8a16_dequant_path:
		with _timed("attn_input_act_quant"):
			_input_fp8 = act_quant(hidden_states)

	def _input_gemm(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
		if _input_fp8 is None:
			return _gemm(weight, scale, hidden_states)
		return w8a16_gemm(
			weight, scale, hidden_states, activation_fp8=_input_fp8
		)

	# Project both fan-outs while the shared quantized input is live, then drop
	# it before the much larger Q-B output is materialized.
	with _timed("attn_q_a"):
		query_states = _input_gemm(
			self.q_a_proj.weight.data,
			weight_scale["q_a_proj.weight_scale_inv"],
		)
	with _timed("attn_kv_a"):
		compressed_kv = _input_gemm(
			self.kv_a_proj_with_mqa.weight.data,
			weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
		)
	_input_fp8 = None

	with _timed("attn_q_norm"):
		query_states = self.q_a_layernorm(query_states)
	with _timed("attn_q_b"):
		query_states = _gemm(
			self.q_b_proj.weight.data,
			weight_scale["q_b_proj.weight_scale_inv"],
			query_states
		)

	query_states = query_states.view(total_tokens, self.num_heads, self.q_head_dim)
	q_nope, q_pe = torch.split(
		query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
	)

	compressed_kv, k_pe = torch.split(
		compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
	)
	with _timed("attn_kv_norm"):
		normed_kv = self.kv_a_layernorm(compressed_kv)
	k_pe = k_pe.view(total_tokens, 1, self.qk_rope_head_dim)

	# Native interleaved RoPE (matches HF / SGLang / vLLM is_neox_style=False
	# when rope_interleave=true).
	from batchgen.attention.mla.rotary_embedding import rotary_pos_emb_interleaved_native
	with _timed("attn_rope"):
		cos, sin = self.rotary_emb(q_pe.unsqueeze(0), seq_len=max_seqlen)
		q_pe = rotary_pos_emb_interleaved_native(q_pe.unsqueeze(0), cos, sin, position_ids.unsqueeze(0), 2).squeeze(0)
		k_pe = rotary_pos_emb_interleaved_native(k_pe.unsqueeze(0), cos, sin, position_ids.unsqueeze(0), 2).squeeze(0)

	with _timed("attn_primary_kv_materialize"):
		k_pe_flat = k_pe.view(total_tokens, self.qk_rope_head_dim)
		offload_kv = torch.cat([normed_kv, k_pe_flat], dim=-1)
	del compressed_kv, k_pe_flat

	# Expand KV
	with _timed("attn_kv_b"):
		kv = _gemm(
			self.kv_b_proj.weight.data,
			weight_scale["kv_b_proj.weight_scale_inv"],
			normed_kv
		)
	kv = kv.view(total_tokens, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
	k_nope, value_states = torch.split(
		kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
	)

	# Assemble Q and K
	with _timed("attn_qkv_materialize"):
		query_states = k_pe.new_empty(total_tokens, self.num_heads, self.q_head_dim)
		query_states[:, :, :self.qk_nope_head_dim] = q_nope
		query_states[:, :, self.qk_nope_head_dim:] = q_pe

		key_states = k_pe.new_empty(total_tokens, self.num_heads, self.q_head_dim)
		k_pe = k_pe.view(total_tokens, 1, self.qk_rope_head_dim)
		key_states[:, :, :self.qk_nope_head_dim] = k_nope
		key_states[:, :, self.qk_nope_head_dim:] = k_pe
		del q_nope, q_pe, k_nope, k_pe, kv, normed_kv

		query_states = query_states.contiguous()
		key_states = key_states.contiguous()
		value_states = value_states.contiguous()

	with _timed("attn_fa3"):
		attn_output = flash_attn_varlen_func(
			query_states,
			key_states,
			value_states,
			cu_seqlens_q=cu_seqlens,
			cu_seqlens_k=cu_seqlens,
			max_seqlen_q=max_seqlen,
			max_seqlen_k=max_seqlen,
			softmax_scale=self.softmax_scale,
			causal=True
		)

	del query_states, key_states, value_states

	if isinstance(attn_output, tuple):
		attn_output = attn_output[0]

	with _timed("attn_o"):
		attn_output = attn_output.view(total_tokens, self.num_heads * self.v_head_dim).contiguous()
		attn_output = _gemm(
			self.o_proj.weight.data,
			weight_scale["o_proj.weight_scale_inv"],
			attn_output
		)

	return attn_output, offload_kv
