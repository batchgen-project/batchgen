import torch
import triton
import triton.language as tl
import os
import logging

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	device = torch.device('cuda:0')
	dtype = torch.bfloat16
	kv_dtype = torch.float8_e4m3fn
	
	bsz = 16
	max_seq_len = 14000
	seq_len = 13000
	dim = 576
	block_size = 128

	# Inputs
	kv = torch.randn(bsz, max_seq_len, dim, dtype=dtype, device=device).to(kv_dtype)
	scale = torch.randn(bsz, max_seq_len, (dim + block_size - 1) // block_size, dtype=torch.float32, device=device)  

	from fp8e4m3 import dequant_compressed_kv_per_token_with_length, dequant_compressed_kv_per_token
	candidates_funcs = [
		dequant_compressed_kv_per_token_with_length,
		dequant_compressed_kv_per_token
	]
	for func in candidates_funcs:
		# warm up
		for _ in range(10):
			__ = func(
				kv, scale, seq_len
			)
		torch.cuda.synchronize(device=device)
		# Measure time
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()
		_ = func(
			kv, scale, seq_len
		)
		end.record()
		torch.cuda.synchronize(device=device)
		elapsed_time = start.elapsed_time(end)
		logger.info(f"Function {func.__name__} took {elapsed_time:.2f} ms for {bsz} batches of {seq_len} tokens with dim {dim}.")

	# ref_res = dequant_compressed_kv_per_token_with_length(
	# 	kv, scale, seq_len)
	# res = dequant_compressed_kv_per_token(kv, scale, seq_len)

	# if torch.allclose(ref_res, res, atol=0.1, rtol=0.1):
	# 	logger.info("Allclose check passed for dequant_compressed_kv_per_token_with_length and dequant_compressed_kv_per_token.")
	# else:
	# 	logger.error("Allclose check failed for dequant_compressed_kv_per_token_with_length and dequant_compressed_kv_per_token.")
	# 	logger.error(f"Max difference: {torch.max(torch.abs(ref_res - res))}")
	# 	logger.error(f"Mean difference: {torch.mean(torch.abs(ref_res - res))}")
	# 	logger.error(f"Reference shape: {ref_res.shape}, Result shape: {res.shape}")
