import torch
import triton
import triton.language as tl
import os
import logging
from typing import Tuple

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	device = torch.device('cuda:0')
	dtype = torch.bfloat16
	weight_dtype = torch.float8_e4m3fn
	activation_dtype = torch.float8_e4m3fn
	scale_dtype = torch.float32
	weight_scale_block_shape = [128, 128]
	tensor_scale_size = 128 # Per tensor per block quant.
	bsz = 384
	num_groups = 16

	# Weights
	w_down_list = [torch.randn(7168, 2048, dtype=dtype, device=device).to(weight_dtype) for _ in range(num_groups)]
	w_down_scale_list = [torch.randn(56, 16, dtype=scale_dtype, device=device) for _ in range(num_groups)]

	# Inputs x shape [bsz, 7168], eid: [bsz, 1] whese eids is random number from 0 to 15 and should be sorted.

	x_bf16 = torch.randn(bsz, 2048, dtype=dtype, device=device)
	x = x_bf16.to(activation_dtype)
	x_scale = torch.randn(bsz, 2048 // tensor_scale_size, dtype=scale_dtype, device=device)
	eids = torch.randint(0, num_groups, (bsz,), dtype=torch.int32, device=device)
	eids = eids.sort()[0]  # Sort the expert ids


	counts = torch.bincount(eids, minlength=num_groups)
	group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)	
	group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
	group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
	group_start_indices[0] = 0  # The first group starts at index 0	

	from fused_grouped_dequant_gemm import (
		fused_dequant_grouped_gemm_bf16_fp8_triton_v2,
		fused_dequant_grouped_gemm_fp8_fp8_triton
	)	

	candidates_func = [fused_dequant_grouped_gemm_fp8_fp8_triton]

	# for func in candidates_func:
	# 	for _ in range(10):
	# 		# Warm up
	# 		_ = func(
	# 			x_bf16, w_down_list, w_down_scale_list, group_sizes, group_start_indices
	# 		)
		
	# 	torch.cuda.synchronize(device=device)
	# 	start = torch.cuda.Event(enable_timing=True)
	# 	end = torch.cuda.Event(enable_timing=True)
	# 	start.record()
	# 	_ = func(
	# 			x_bf16, w_down_list, w_down_scale_list, group_sizes, group_start_indices
	# 	)
	# 	end.record()
	# 	torch.cuda.synchronize(device=device)
	# 	# logger.info(f"fused_fp8_moe_stage_1_optimized took {start.elapsed_time(end)} ms")
	# 	logger.info(f"{func.__name__} took {start.elapsed_time(end)} ms")
	# candidates_M = [16, 32, 64, 128, 256]
	# candidates_N = [16, 32, 64]
	# candidates_K = [32, 64, 128, 256]
	candidates_M = [64]
	candidates_N = [32]
	candidates_K = [128]
	num_stages = [1,2,3,4,5]
	num_warps = [2,4,8]

	best_time = float('inf')
	best_config = None
	for M in candidates_M:
		for N in candidates_N:
			for K in candidates_K:
				for func in candidates_func:
					for num_stage in num_stages:
						for num_warp in num_warps:
							try:
								# warm-up
								for _ in range(10):
									_ = func(
										x, x_scale, w_down_list, w_down_scale_list, group_sizes, group_start_indices,
										gemm_block_size=[M, N, K],
										num_stages=num_stage, num_warps=num_warp
									)
								torch.cuda.synchronize(device=device)
								logger.info(f"Testing config: M={M}, N={N}, K={K}, Function: {func.__name__}, num_stage={num_stage}, num_warp={num_warp}")
								start = torch.cuda.Event(enable_timing=True)
								end = torch.cuda.Event(enable_timing=True)
								start.record()
								_ = func(
									x, x_scale, w_down_list, w_down_scale_list, group_sizes, group_start_indices,
									gemm_block_size=[M, N, K],
									num_stages=num_stage, num_warps=num_warp
								)
								end.record()
								torch.cuda.synchronize(device=device)
								elapsed_time = start.elapsed_time(end)
								logger.info(f"{func.__name__} took {elapsed_time} ms")
								if elapsed_time < best_time:
									best_time = elapsed_time
									best_config = (M, N, K, func.__name__, num_stage, num_warp)
							except Exception as e:
								logger.error(f"Error testing config M={M}, N={N}, K={K}, Function: {func.__name__}, num_stage={num_stage}, num_warp={num_warp}: {e}")
								continue
	if best_config:
		logger.info(f"Best config: M={best_config[0]}, N={best_config[1]}, K={best_config[2]}, Function: {best_config[3]}, num_stage={best_config[4]}, num_warp={best_config[5]} with time {best_time} ms")

"""
	INFO:__main__:Best config: M=32, N=32, K=128, Function: fused_dequant_grouped_gemm_bf16_fp8_triton_v2, num_stage=2, num_warp=2 with time 0.6780480146408081 ms
"""