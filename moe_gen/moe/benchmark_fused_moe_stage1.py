import torch
import triton
import triton.language as tl
import os
import logging

"""
	This script is to benchmark implementations of fused grouped-gemm for MoE.

	def grouped_dequant_moe_fp8(self, x, eids):
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
		eids = eids - self.routed_expert_start_idx
		counts = torch.bincount(eids, minlength=self.experts_per_rank)
		group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)	
		group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=self.device)
		group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
		group_start_indices[0] = 0  # The first group starts at index 0	

		# Quantize the recv_x tensor to fp8_e4m3
		x, x_scale = act_quant(x)
		intermediate = fused_fp8_moe_stage_1_optimized(
			x, x_scale, gate_list, up_list, gate_scale_list, up_scale_list, group_sizes, group_start_indices
		)	
		
		res = fused_dequant_grouped_gemm_bf16_fp8_triton_v2(
			intermediate, down_list, down_scale_list, group_sizes, group_start_indices, gemm_block_size = [64, 16, 64]
		)
		return res
"""


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
	w_gate_list = [torch.randn(2048, 7168, dtype=dtype, device=device).to(weight_dtype) for _ in range(num_groups)]
	w_up_list = [torch.randn(2048, 7168, dtype=dtype, device=device).to(weight_dtype) for _ in range(num_groups)]
	w_down_list = [torch.randn(7168, 2048, dtype=dtype, device=device).to(weight_dtype) for _ in range(num_groups)]

	# Weight scale shape: [2048 // 128， 7168 // 128]
	w_gate_scale_list = [torch.randn(16, 56, dtype=scale_dtype, device=device) for _ in range(num_groups)]
	w_up_scale_list = [torch.randn(16, 56, dtype=scale_dtype, device=device) for _ in range(num_groups)]
	w_down_scale_list = [torch.randn(56, 16, dtype=scale_dtype, device=device) for _ in range(num_groups)]


	# Inputs x shape [bsz, 7168], eid: [bsz, 1] whese eids is random number from 0 to 15 and should be sorted.

	x = torch.randn(bsz, 7168, dtype=dtype, device=device).to(activation_dtype)
	x_scale = torch.randn(bsz, 7168 // tensor_scale_size, dtype=scale_dtype, device=device)
	eids = torch.randint(0, num_groups, (bsz,), dtype=torch.int32, device=device)
	eids = eids.sort()[0]  # Sort the expert ids


	counts = torch.bincount(eids, minlength=num_groups)
	group_sizes = sorted((idx, sz) for idx, sz in enumerate(counts.tolist()) if sz)	
	group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
	group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
	group_start_indices[0] = 0  # The first group starts at index 0	
	activated_group_idx = torch.arange(num_groups, device=device, dtype=torch.int32)[counts > 0]
	
	gate_ptrs_ptr = torch.tensor([w_gate_list[i].data_ptr() for i in range(num_groups)], dtype=torch.int64, device=device)
	up_ptrs_ptr = torch.tensor([w_up_list[i].data_ptr() for i in range(num_groups)], dtype=torch.int64, device=device)
	gate_scale_ptrs_ptr = torch.tensor([w_gate_scale_list[i].data_ptr() for i in range(num_groups)], dtype=torch.int64, device=device)
	up_scale_ptrs_ptr = torch.tensor([w_up_scale_list[i].data_ptr() for i in range(num_groups)], dtype=torch.int64, device=device)

											

	from fused_dequant_moe import (
		fused_fp8_moe_stage_1_optimized,
		fused_fp8_moe_stage_1
	)
	candidates_func = [fused_fp8_moe_stage_1]

	# for func in candidates_func:
	# 	for _ in range(10):
	# 		# Warm up
	# 		_ = func(
	# 			x, x_scale, w_gate_list, w_up_list, w_gate_scale_list, w_up_scale_list, group_sizes, group_start_indices
	# 		)
		
	# 	torch.cuda.synchronize(device=device)
	# 	start = torch.cuda.Event(enable_timing=True)
	# 	end = torch.cuda.Event(enable_timing=True)
	# 	start.record()
	# 	_ = func(
	# 		x, x_scale, w_gate_list, w_up_list, w_gate_scale_list, w_up_scale_list, group_sizes, group_start_indices
	# 	)
	# 	end.record()
	# 	torch.cuda.synchronize(device=device)
	# 	# logger.info(f"fused_fp8_moe_stage_1_optimized took {start.elapsed_time(end)} ms")
	# 	logger.info(f"{func.__name__} took {start.elapsed_time(end)} ms")

	# grid search tile size 
	candidates_M = [16, 32, 64, 128]
	candidates_N = [16, 32, 64]
	candidates_K = [32,64,256]
	# candidates_M = [64]
	# candidates_N = [16]
	# candidates_K = [256]
	# num_stages = [1,2,3,4,5]
	# num_warps = [2,4,8,16]
	num_stages = [1, 2, 3]
	num_warps = [2, 4, 8]
	best_time = float('inf')
	best_config = None
	for M in candidates_M:
		for N in candidates_N:
			for K in candidates_K:
				for num_stage in num_stages:
					for num_warp in num_warps:
						try:
							logger.info(f"Testing grid size: M={M}, N={N}, K={K}, num_stage={num_stage}, num_warp={num_warp}")
							for func in candidates_func:
								for _ in range(10):
									# Warm up
									_ = func(
										x, x_scale, 
										w_gate_list, gate_ptrs_ptr,
										w_up_list, up_ptrs_ptr,
										w_gate_scale_list, gate_scale_ptrs_ptr,
										w_up_scale_list, up_scale_ptrs_ptr,
										group_size, activated_group_idx, group_start_indices,
										gate_gemm_block_size=[M, N, K], num_stages=num_stage, num_warps=num_warp
									)
								"""
									def fused_fp8_moe_stage_1(
										hidden_states: torch.Tensor,
										hidden_states_scale: torch.Tensor,
										gate_weight_list: list[torch.Tensor],
										gate_ptrs_ptr: torch.Tensor,
										up_weight_list: list[torch.Tensor],
										up_ptrs_ptr: torch.Tensor,
										gate_scale_list: list[torch.Tensor],
										gate_scale_ptrs_ptr: torch.Tensor,
										up_scale_list: list[torch.Tensor],
										up_scale_ptrs_ptr: torch.Tensor,
										group_sizes: torch.Tensor,
										activated_group_idx: torch.Tensor,
										group_start_indices: torch.Tensor,
										gate_gemm_block_size=[64,16,256],
										up_gemm_block_size=[64,16,128],
										scale_block_size=[128,128],
										num_stages = 2,
										num_warps = 4
									):

								"""
								torch.cuda.synchronize(device=device)
								start = torch.cuda.Event(enable_timing=True)
								end = torch.cuda.Event(enable_timing=True)
								start.record()
								_ = func(
										x, x_scale, 
										w_gate_list, gate_ptrs_ptr,
										w_up_list, gate_ptrs_ptr,
										w_gate_scale_list, gate_scale_ptrs_ptr,
										w_up_scale_list, up_scale_ptrs_ptr,
										group_size, activated_group_idx, group_start_indices,
										gate_gemm_block_size=[M, N, K], num_stages=num_stage, num_warps=num_warp
									)
								end.record()
								torch.cuda.synchronize(device=device)
								logger.info(f"{func.__name__} with grid size M={M}, N={N}, K={K}, num_stage={num_stage}, num_warp={num_warp} took {start.elapsed_time(end)} ms")
								if start.elapsed_time(end) < best_time:
									best_time = start.elapsed_time(end)
									best_config = (M, N, K, func.__name__, num_stage, num_warp)
						except Exception as e:
							logger.error(f"Error with grid size M={M}, N={N}, K={K}, num_stage={num_stage}, num_warp={num_warp}: {e}")
							continue
	if best_config:
		logger.info(f"Best config: M={best_config[0]}, N={best_config[1]}, K={best_config[2]}, Function: {best_config[3]}, num_stage={best_config[4]}, num_warp={best_config[5]} with time {best_time} ms")



		"""
			INFO:__main__:Best config: M=64, N=16, K=256, Function: fused_fp8_moe_stage_1, num_stage=2, num_warp=4 with time 1.0309439897537231 ms

		"""

	# for M in candidates_M:
	# 	for N in candidates_N:
	# 		for K in candidates_K:
	# 			logger.info(f"Testing grid size: M={M}, N={N}, K={K}")
	# 			for func in candidates_func:
	# 				try: 
	# 					for _ in range(10):
	# 						# Warm up
	# 						_ = func(
	# 							x, x_scale, w_gate_list, w_up_list, w_gate_scale_list, w_up_scale_list, group_sizes, group_start_indices,
	# 							gate_gemm_block_size=[M, N, K]
	# 						)
						
	# 					torch.cuda.synchronize(device=device)
	# 					start = torch.cuda.Event(enable_timing=True)
	# 					end = torch.cuda.Event(enable_timing=True)
	# 					start.record()
	# 					_ = func(
	# 						x, x_scale, w_gate_list, w_up_list, w_gate_scale_list, w_up_scale_list, group_sizes, group_start_indices,
	# 						gate_gemm_block_size=[M, N, K]
	# 					)
	# 					end.record()
	# 					torch.cuda.synchronize(device=device)
	# 					logger.info(f"{func.__name__} with grid size M={M}, N={N}, K={K} took {start.elapsed_time(end)} ms")
	# 					if start.elapsed_time(end) < best_time:
	# 						best_time = start.elapsed_time(end)
	# 						best_config = (M, N, K, func.__name__)
	# 				except Exception as e:
	# 					logger.error(f"Error with grid size M={M}, N={N}, K={K}: {e}")
	# 					continue
	# if best_config:
	# 	logger.info(f"Best config: M={best_config[0]}, N={best_config[1]}, K={best_config[2]}, Function: {best_config[3]} with time {best_time} ms")


	

