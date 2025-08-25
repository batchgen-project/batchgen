import torch
import triton
import triton.language as tl
import os
import logging
from typing import Tuple

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


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	device = torch.device('cuda:0')
	dtype = torch.bfloat16
	weight_dtype = torch.float8_e4m3fn
	activation_dtype = torch.float8_e4m3fn
	scale_dtype = torch.float32
	block_size = 128

	bsz = 384
	dim = 7168

	x = torch.randn(bsz, dim, dtype=dtype, device=device)
	x_warm_up = torch.randn(bsz, dim, dtype=dtype, device=device)

	#warm-up
	for _ in range(10):
		__ = act_quant(x_warm_up, block_size=block_size)

	torch.cuda.synchronize(device=device)

	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	start.record()
	quantized_x, scale = act_quant(x, block_size=block_size)
	end.record()
	torch.cuda.synchronize(device=device)
	logger.info(f"act_quant took {start.elapsed_time(end)} ms")