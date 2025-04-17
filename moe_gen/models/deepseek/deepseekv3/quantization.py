import torch
from math import ceil

_FP8_MAX = 448.0
def compressed_kv_bf16_to_fp8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Quantize a [bsz, seq, 576] BF16 tensor to FP8 per‐token.
	Returns:
	  q:   torch.Tensor[bsz, seq, 576]   dtype=float8_e4m3fn
	  s:   torch.Tensor[bsz, seq]        dtype=float32  (the scale factors)
	"""	
	assert x.dtype == torch.bfloat16
	assert x.shape[-1] == 576
	assert x.is_contiguous()
	assert x.dim == 3

	bsz, seq_len, dim = x.shape
	M = bsz * seq_len
	x_flat = x.view(M, dim).float()              # to FP32 for reduction
	amax   = x_flat.abs().amax(dim=1).clamp(min=1e-6)  # [M]
	scale  = amax / _FP8_MAX                         # [M]
	# scale & cast
	y = x_flat / scale.unsqueeze(1)                  # [M, 576]
	q = y.to(torch.float8_e4m3fn)                    # [M, 576] in FP8
	return q.view(bsz, seq_len, dim), scale.view(bsz, seq_len)	


def compressed_kv_fp8_to_bf16_per_token(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Dequantize the output of bf16_to_fp8_per_token back to BF16.
    Inputs:
      q     [bsz, seq, 576]  dtype=float8_e4m3fn
      scale [bsz, seq]       dtype=float32
    Returns:
      x_bf16 [bsz, seq, 576] dtype=bfloat16
    """
    bsz, seq_len, dim = q.shape
    M = bsz * seq_len
    # flatten
    q_flat = q.view(M, dim).float()               # upcast FP8→FP32
    x_rec = q_flat * scale.view(M, 1)             # rescale
    return x_rec.to(torch.bfloat16).view(bsz, seq_len, dim)	