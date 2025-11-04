# import torch
# import triton
# import triton.language as tl


# @triton.jit
# def fused_fp8_moe_parallel_experts_kernel(
#     lhs_ptr, lhs_scale_ptr,
#     gate_ptrs_ptr, up_ptrs_ptr,
#     gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#     group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
#     num_active_experts_ptr,
#     output_ptr,
#     M, N: tl.constexpr, K: tl.constexpr,
#     stride_lhs_m, stride_lhs_k,
#     stride_lhs_scale_m, stride_lhs_scale_k,
#     stride_gate_n, stride_gate_k,
#     stride_up_n, stride_up_k,
#     stride_output_m, stride_output_n,
#     stride_group_idx, stride_group_sizes, stride_group_start_indices,
#     stride_weight_ptrs, stride_scale_ptrs,
#     GEMM_BLOCK_SIZE_M: tl.constexpr,
#     GEMM_BLOCK_SIZE_N: tl.constexpr,
#     GEMM_BLOCK_SIZE_K: tl.constexpr,
#     SCALE_BLOCK_SIZE_N: tl.constexpr,
#     SCALE_BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     FIXED: 2D grid parallelizes over BOTH experts and N-blocks.
    
#     Grid: (num_experts, cdiv(N, BLOCK_N))
#     Each program handles ONE expert's ONE N-block.
#     NO serial loop over experts!
#     """
#     # 2D program IDs
#     group_pid = tl.program_id(axis=0)  # Which expert
#     n_pid = tl.program_id(axis=1)      # Which N-block
    
#     # Bounds check
#     num_groups = tl.load(num_active_experts_ptr)
#     if group_pid >= num_groups:
#         return
    
#     # Load THIS expert's metadata
#     gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
#     if gm == 0:
#         return
    
#     group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
#     start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
#     # Load THIS expert's weight pointers
#     gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
#     up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
#     # N-block offsets
#     offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
#     scale_n = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
#     num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
#     num_quant_blocks_per_gemm = tl.cdiv(GEMM_BLOCK_SIZE_K, SCALE_BLOCK_SIZE_K)
    
#     # Process all M-blocks for THIS expert
#     num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
#     for sub_group_idx in range(num_sub_groups):
#         sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
#         remaining_rows_in_group = start_idx + gm - sub_group_start_idx
#         valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
#         offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
#         abs_row_indices = sub_group_start_idx + offsets_m
        
#         gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
#         up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
#         # K-loop
#         num_gemm_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
#         for gemm_k_idx in range(num_gemm_k_blocks):
#             gemm_k_start = gemm_k_idx * GEMM_BLOCK_SIZE_K
            
#             for quant_sub_idx in range(num_quant_blocks_per_gemm):
#                 sub_k_start = gemm_k_start + quant_sub_idx * SCALE_BLOCK_SIZE_K
#                 sub_block_valid = sub_k_start < K
#                 offsets_k = sub_k_start + tl.arange(0, SCALE_BLOCK_SIZE_K)
                
#                 # Load scales
#                 scale_k = sub_k_start // SCALE_BLOCK_SIZE_K
#                 gate_scale = tl.load(gate_scale_base_ptr + (scale_n * num_scale_k + scale_k))
#                 up_scale = tl.load(up_scale_base_ptr + (scale_n * num_scale_k + scale_k))
                
#                 lhs_scale_k = sub_k_start // SCALE_BLOCK_SIZE_K
#                 l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
#                 lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0)
                
#                 # Load data
#                 lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
#                 gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
#                 up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
                
#                 k_mask = (offsets_k < K) & sub_block_valid
#                 lhs_mask = (abs_row_indices[:, None] < M) & k_mask[None, :] & (offsets_m[:, None] < valid_rows_this_block)
#                 rhs_mask = (offsets_n[:, None] < N) & k_mask[None, :]
                
#                 lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
#                 gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
#                 up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
                
#                 gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
#                 up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
#         # SiLU activation
#         output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
#         output = output_acc.to(tl.bfloat16)
        
#         # Store
#         offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
#         offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
#         output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
#         output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)
        
#         tl.store(output_ptrs, output, mask=output_mask)


# @torch.inference_mode()
# def fused_fp8_moe_stage_1_optimized(
#     hidden_states: torch.Tensor,
#     hidden_states_scale: torch.Tensor,
#     gate_weight_list: list[torch.Tensor],
#     gate_ptrs_ptr: torch.Tensor,
#     up_weight_list: list[torch.Tensor],
#     up_ptrs_ptr: torch.Tensor,
#     gate_scale_list: list[torch.Tensor],
#     gate_scale_ptrs_ptr: torch.Tensor,
#     up_scale_list: list[torch.Tensor],
#     up_scale_ptrs_ptr: torch.Tensor,
#     group_sizes: torch.Tensor,
#     activated_group_idx: torch.Tensor,
#     group_start_indices: torch.Tensor,
#     num_active_experts: torch.Tensor,
#     gate_gemm_block_size=[64, 16, 128],
#     scale_block_size=[128, 128],
#     num_stages=3,
#     num_warps=4
# ):
#     """
#     OPTIMIZED: 2D grid for parallel expert processing.
    
#     Old: (448,) programs, each doing 8 experts serially
#     New: (8, 448) = 3584 programs, all parallel!
#     """
#     device = hidden_states.device
#     M = hidden_states.shape[0]
#     N = gate_weight_list[0].shape[0]
#     K = hidden_states.shape[1]
    
#     output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
#     num_groups = num_active_experts.item()
    
#     # 2D GRID: (experts, N_blocks)
#     grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
    
#     # Total programs: num_groups × cdiv(N, BLOCK_N)
#     # For 8 experts, N=7168, BLOCK_N=16: 8 × 448 = 3584 programs!
    
#     fused_fp8_moe_parallel_experts_kernel[grid](
#         hidden_states, hidden_states_scale,
#         gate_ptrs_ptr, up_ptrs_ptr,
#         gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#         activated_group_idx, group_sizes, group_start_indices,
#         num_active_experts,
#         output,
#         M, N, K,
#         hidden_states.stride(0), hidden_states.stride(1),
#         hidden_states_scale.stride(0), hidden_states_scale.stride(1),
#         gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
#         up_weight_list[0].stride(0), up_weight_list[0].stride(1),
#         output.stride(0), output.stride(1),
#         activated_group_idx.stride(0),
#         group_sizes.stride(0),
#         group_start_indices.stride(0),
#         gate_ptrs_ptr.stride(0),
#         gate_scale_ptrs_ptr.stride(0),
#         GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
#         GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
#         GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
#         SCALE_BLOCK_SIZE_N=scale_block_size[0],
#         SCALE_BLOCK_SIZE_K=scale_block_size[1],
#         num_stages=num_stages,
#         num_warps=num_warps
#     )
    
#     return output


import torch
import triton
import triton.language as tl
from typing import Optional # <-- Added for allocator


# @triton.jit
# def fused_fp8_moe_parallel_experts_kernel_optimized(
#     lhs_ptr, lhs_scale_ptr,
#     gate_ptrs_ptr, up_ptrs_ptr,
#     gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#     group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
#     num_active_experts_ptr,
#     output_ptr,
#     M, N: tl.constexpr, K: tl.constexpr,
#     stride_lhs_m, stride_lhs_k,
#     stride_lhs_scale_m, stride_lhs_scale_k,
#     stride_gate_n, stride_gate_k,
#     stride_up_n, stride_up_k,
#     stride_output_m, stride_output_n,
#     stride_group_idx, stride_group_sizes, stride_group_start_indices,
#     stride_weight_ptrs, stride_scale_ptrs,
#     GEMM_BLOCK_SIZE_M: tl.constexpr,
#     GEMM_BLOCK_SIZE_N: tl.constexpr,
#     GEMM_BLOCK_SIZE_K: tl.constexpr,
#     SCALE_BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     OPTIMIZED: 2D grid parallelizes over BOTH experts and N-blocks.
#     Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    
#     Key optimizations:
#     - Removed nested quantization loop (1:1 mapping)
#     - Eliminated redundant calculations
#     - Simplified scale loading
#     - Better register utilization
#     """
#     # 2D program IDs
#     group_pid = tl.program_id(axis=0)  # Which expert
#     n_pid = tl.program_id(axis=1)      # Which N-block
    
#     # Bounds check
#     num_groups = tl.load(num_active_experts_ptr)
#     if group_pid >= num_groups:
#         return
    
#     # Load THIS expert's metadata
#     gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
#     if gm == 0:
#         return
    
#     group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
#     start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
#     # Load THIS expert's weight pointers
#     gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
#     up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
#     # N-block offsets (fixed for this program)
#     offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
#     n_mask = offsets_n < N
    
#     # Scale N index (fixed for this program)
#     num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
#     scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
#     # Process all M-blocks for THIS expert
#     num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
#     for sub_group_idx in range(num_sub_groups):
#         sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
#         remaining_rows_in_group = start_idx + gm - sub_group_start_idx
#         valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
#         offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
#         abs_row_indices = sub_group_start_idx + offsets_m
#         m_mask = abs_row_indices < M
#         valid_mask = offsets_m < valid_rows_this_block
        
#         # Initialize accumulators
#         gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
#         up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
#         # K-loop: SIMPLIFIED! One K-block = one scale block
#         num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)

#         for k_block_idx in range(num_k_blocks):
#             k_start = k_block_idx * GEMM_BLOCK_SIZE_K
#             offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
#             k_mask = offsets_k < K
            
#             # Load scales - direct mapping
#             scale_k_idx = k_block_idx
#             scale_n = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
            
#             gate_scale = tl.load(gate_scale_base_ptr + (scale_n * num_scale_k + scale_k_idx))
#             up_scale = tl.load(up_scale_base_ptr + (scale_n * num_scale_k + scale_k_idx))
            
#             lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
#                                             scale_k_idx * stride_lhs_scale_k)
#             lhs_scale = tl.load(lhs_scale_ptrs, mask=(abs_row_indices[:, None] < M), other=1.0)
            
#             # Load data (keep original pointer calculation style)
#             lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
#             gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
#             up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
#             lhs_mask = (abs_row_indices[:, None] < M) & k_mask[None, :] & (offsets_m[:, None] < valid_rows_this_block)
#             rhs_mask = (offsets_n[:, None] < N) & k_mask[None, :]
            
#             lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
#             gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
#             up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
#             # KEEP THE ORIGINAL FUSED PATTERN!
#             gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
#             up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
#         # SiLU activation: silu(x) = x / (1 + exp(-x))
#         gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
#         output_acc = gate_activated * up_acc
#         output = output_acc.to(tl.bfloat16)
        
#         # Store results
#         offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
#         offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
#         output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
#                                     offs_output_n[None, :] * stride_output_n)
#         output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
#         tl.store(output_ptrs, output, mask=output_mask)





# @torch.inference_mode()
# def fused_fp8_moe_stage_1_optimized(
#     hidden_states: torch.Tensor,
#     hidden_states_scale: torch.Tensor,
#     gate_weight_list: list[torch.Tensor],
#     gate_ptrs_ptr: torch.Tensor,
#     up_weight_list: list[torch.Tensor],
#     up_ptrs_ptr: torch.Tensor,
#     gate_scale_list: list[torch.Tensor],
#     gate_scale_ptrs_ptr: torch.Tensor,
#     up_scale_list: list[torch.Tensor],
#     up_scale_ptrs_ptr: torch.Tensor,
#     group_sizes: torch.Tensor,
#     activated_group_idx: torch.Tensor,
#     group_start_indices: torch.Tensor,
#     num_active_experts: torch.Tensor,
#     gate_gemm_block_size=[64, 16, 128],
#     scale_block_size=128,  # Simplified: single value since N and K use same size
#     num_stages=3,
#     num_warps=4
# ):
#     """
#     OPTIMIZED: 2D grid for parallel expert processing.
    
#     Key changes:
#     - Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128)
#     - Simplified scale_block_size parameter
#     - Cleaner kernel with better performance
    
#     Grid: (num_experts, cdiv(N, BLOCK_N))
#     Total programs: num_groups × cdiv(N, BLOCK_N)
#     Example: 8 experts × 448 N-blocks = 3584 parallel programs
#     """
#     device = hidden_states.device
#     M = hidden_states.shape[0]
#     N = gate_weight_list[0].shape[0]
#     K = hidden_states.shape[1]
    
#     # Validate assumption
#     assert gate_gemm_block_size[2] == scale_block_size, \
#         f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
#     output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
#     num_groups = num_active_experts.item()
    
#     # 2D GRID: (experts, N_blocks)
#     grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
    
#     fused_fp8_moe_parallel_experts_kernel_optimized[grid](
#         hidden_states, hidden_states_scale,
#         gate_ptrs_ptr, up_ptrs_ptr,
#         gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#         activated_group_idx, group_sizes, group_start_indices,
#         num_active_experts,
#         output,
#         M, N, K,
#         hidden_states.stride(0), hidden_states.stride(1),
#         hidden_states_scale.stride(0), hidden_states_scale.stride(1),
#         gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
#         up_weight_list[0].stride(0), up_weight_list[0].stride(1),
#         output.stride(0), output.stride(1),
#         activated_group_idx.stride(0),
#         group_sizes.stride(0),
#         group_start_indices.stride(0),
#         gate_ptrs_ptr.stride(0),
#         gate_scale_ptrs_ptr.stride(0),
#         GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
#         GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
#         GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
#         SCALE_BLOCK_SIZE_K=scale_block_size,
#         num_stages=num_stages,
#         num_warps=num_warps
#     )
    
#     return output



import torch
import triton
import triton.language as tl


""" Work. 1.2x """
# @triton.jit
# def fused_fp8_moe_parallel_experts_kernel_optimized(
#     lhs_ptr, lhs_scale_ptr,
#     gate_ptrs_ptr, up_ptrs_ptr,
#     gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#     group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
#     num_active_experts_ptr,
#     output_ptr,
#     M, N: tl.constexpr, K: tl.constexpr,
#     stride_lhs_m, stride_lhs_k,
#     stride_lhs_scale_m, stride_lhs_scale_k,
#     stride_gate_n, stride_gate_k,
#     stride_up_n, stride_up_k,
#     stride_output_m, stride_output_n,
#     stride_group_idx, stride_group_sizes, stride_group_start_indices,
#     stride_weight_ptrs, stride_scale_ptrs,
#     GEMM_BLOCK_SIZE_M: tl.constexpr,
#     GEMM_BLOCK_SIZE_N: tl.constexpr,
#     GEMM_BLOCK_SIZE_K: tl.constexpr,
#     SCALE_BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     OPTIMIZED: 2D grid parallelizes over BOTH experts and N-blocks.
#     Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    
#     Key optimizations:
#     - Removed nested quantization loop (1:1 mapping)
#     - Hoisted mask computations outside K-loop
#     - Eliminated redundant scale_n calculation
#     - Simplified scale loading masks
#     """
#     # 2D program IDs
#     group_pid = tl.program_id(axis=0)  # Which expert
#     n_pid = tl.program_id(axis=1)      # Which N-block
    
#     # Bounds check
#     num_groups = tl.load(num_active_experts_ptr)
#     if group_pid >= num_groups:
#         return
    
#     # Load THIS expert's metadata
#     gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
#     if gm == 0:
#         return
    
#     group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
#     start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
#     # Load THIS expert's weight pointers
#     gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
#     up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
#     # N-block offsets (fixed for this program) - HOISTED
#     offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
#     n_mask = offsets_n < N  # [N] - Computed once
    
#     # Scale N index (fixed for this program) - HOISTED
#     num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
#     scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
#     # M-dimension offsets - HOISTED
#     offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
#     # Process all M-blocks for THIS expert
#     num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
#     for sub_group_idx in range(num_sub_groups):
#         sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
#         remaining_rows_in_group = start_idx + gm - sub_group_start_idx
#         valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
#         abs_row_indices = sub_group_start_idx + offsets_m
        
#         # HOISTED: Compute M-dimension masks ONCE per M-block
#         m_mask = abs_row_indices < M  # [M]
#         valid_mask = offsets_m < valid_rows_this_block  # [M]
#         m_base_mask = m_mask & valid_mask  # [M] - Combined M-dimension mask
        
#         # Initialize accumulators
#         gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
#         up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
#         # K-loop: SIMPLIFIED! One K-block = one scale block
#         num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)

#         for k_block_idx in range(num_k_blocks):
#             k_start = k_block_idx * GEMM_BLOCK_SIZE_K
#             offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
#             k_mask = offsets_k < K  # [K]
            
#             # Load scales - direct mapping
#             scale_k_idx = k_block_idx
#             # FIXED: Use pre-computed scale_n_idx instead of recalculating
#             scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
#             gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
#             up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
#             # OPTIMIZED: Use m_mask instead of recomputing
#             lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
#                                               scale_k_idx * stride_lhs_scale_k)
#             lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
#             # Load data pointers
#             lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
#             gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
#             up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
#             # OPTIMIZED: Simplified mask computations using pre-computed base masks
#             lhs_mask = m_base_mask[:, None] & k_mask[None, :]  # [M, 1] & [1, K] = [M, K]
#             rhs_mask = n_mask[:, None] & k_mask[None, :]  # [N, 1] & [1, K] = [N, K]
            
#             # Load data
#             lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
#             gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
#             up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
#             # Fused multiply-add pattern (keep original for performance)
#             gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
#             up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
#         # SiLU activation: silu(x) = x / (1 + exp(-x))
#         gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
#         output_acc = gate_activated * up_acc
#         output = output_acc.to(tl.bfloat16)
        
#         # Store results
#         offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
#         offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
#         output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
#                                     offs_output_n[None, :] * stride_output_n)
#         # OPTIMIZED: Use pre-computed valid_mask
#         output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
#         tl.store(output_ptrs, output, mask=output_mask)


# @torch.inference_mode()
# def fused_fp8_moe_stage_1_optimized(
#     hidden_states: torch.Tensor,
#     hidden_states_scale: torch.Tensor,
#     gate_weight_list: list[torch.Tensor],
#     gate_ptrs_ptr: torch.Tensor,
#     up_weight_list: list[torch.Tensor],
#     up_ptrs_ptr: torch.Tensor,
#     gate_scale_list: list[torch.Tensor],
#     gate_scale_ptrs_ptr: torch.Tensor,
#     up_scale_list: list[torch.Tensor],
#     up_scale_ptrs_ptr: torch.Tensor,
#     group_sizes: torch.Tensor,
#     activated_group_idx: torch.Tensor,
#     group_start_indices: torch.Tensor,
#     num_active_experts: torch.Tensor,
#     num_groups: int,
#     gate_gemm_block_size=[64, 16, 128],
#     scale_block_size=128,
#     num_stages=3,
#     num_warps=4
# ):
#     """
#     OPTIMIZED: 2D grid for parallel expert processing.
#     Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
#     """
#     device = hidden_states.device
#     M = hidden_states.shape[0]
#     N = gate_weight_list[0].shape[0]
#     K = hidden_states.shape[1]
    
#     # Validate assumption
#     assert gate_gemm_block_size[2] == scale_block_size, \
#         f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
#     output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
#     # num_groups = num_active_experts.item()
#     # num_groups = 16
    
#     # 2D GRID: (experts, N_blocks)
#     grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
#     # grid = (num_groups, N // gate_gemm_block_size[1])
    
#     fused_fp8_moe_parallel_experts_kernel_optimized[grid](
#         hidden_states, hidden_states_scale,
#         gate_ptrs_ptr, up_ptrs_ptr,
#         gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#         activated_group_idx, group_sizes, group_start_indices,
#         num_active_experts,
#         output,
#         M, N, K,
#         hidden_states.stride(0), hidden_states.stride(1),
#         hidden_states_scale.stride(0), hidden_states_scale.stride(1),
#         gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
#         up_weight_list[0].stride(0), up_weight_list[0].stride(1),
#         output.stride(0), output.stride(1),
#         activated_group_idx.stride(0),
#         group_sizes.stride(0),
#         group_start_indices.stride(0),
#         gate_ptrs_ptr.stride(0),
#         gate_scale_ptrs_ptr.stride(0),
#         GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
#         GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
#         GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
#         SCALE_BLOCK_SIZE_K=scale_block_size,
#         num_stages=num_stages,
#         num_warps=num_warps
#     )
    
#     return output

from typing import Optional, List
@triton.jit
def fused_fp8_moe_baseline_optimized_v2(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    OPTIMIZED BASELINE: Same algorithm, better implementation
    
    Improvements:
    - Reduced register pressure
    - Better instruction scheduling
    - Hoisted invariants out of loops
    - Fewer memory operations
    """
    # Get 2D program IDs
    group_pid = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)
    
    # Early exit check
    num_groups = tl.load(num_active_experts_ptr)
    if group_pid >= num_groups:
        return
    
    # Load group metadata
    gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
    if gm == 0:
        return
    
    group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
    start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
    # Load weight pointers (hoisted out of M-loop)
    gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # Pre-compute N-block offsets and mask (hoisted)
    base_n = n_pid * GEMM_BLOCK_SIZE_N
    offsets_n = base_n + tl.arange(0, GEMM_BLOCK_SIZE_N)
    n_mask = offsets_n < N
    
    # Pre-compute scale indices (hoisted)
    scale_n_idx = base_n // SCALE_BLOCK_SIZE_K
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    
    # Pre-compute offsets (hoisted)
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    offsets_k = tl.arange(0, GEMM_BLOCK_SIZE_K)
    
    # Pre-compute loop bounds (hoisted)
    num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
    
    # ===== M-BLOCK LOOP =====
    for sub_group_idx in range(num_sub_groups):
        sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
        remaining_rows = start_idx + gm - sub_group_start_idx
        valid_rows = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows)
        
        # Row indices for this M-block
        abs_row_indices = sub_group_start_idx + offsets_m
        m_mask = abs_row_indices < M
        valid_mask = offsets_m < valid_rows
        m_base_mask = m_mask & valid_mask
        
        # Initialize accumulators with explicit dtype
        gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        # ===== K-BLOCK LOOP =====
        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * GEMM_BLOCK_SIZE_K
            offs_k = k_start + offsets_k
            k_mask = offs_k < K
            
            # === LOAD SCALES ===
            scale_k_idx = k_block_idx
            scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
            # Load RHS scales (single scalar per K-block)
            gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
            up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
            # Load LHS scales (per row)
            # This pointer calculation is safe because the baseline passed
            # the non-aligned tests in your previous run.
            lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                              scale_k_idx * stride_lhs_scale_k)
            lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
            # === LOAD DATA ===
            # LHS (activations)
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offs_k[None, :] * stride_lhs_k)
            lhs_mask = m_base_mask[:, None] & k_mask[None, :]
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            
            # RHS (weights) - gate and up
            gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offs_k[None, :] * stride_gate_k)
            up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offs_k[None, :] * stride_up_k)
            rhs_mask = n_mask[:, None] & k_mask[None, :]
            
            gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
            up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
            # === COMPUTE ===
            # Combined scale factor
            combined_gate_scale = lhs_scale * gate_scale
            combined_up_scale = lhs_scale * up_scale
            
            # GEMM with fused scaling
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * combined_gate_scale
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * combined_up_scale
        
        # === EPILOGUE ===
        # SiLU activation: x / (1 + exp(-x))
        gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
        
        # Element-wise multiply
        output_acc = gate_activated * up_acc
        
        # Convert to output dtype
        output = output_acc.to(tl.bfloat16)
        
        # === STORE ===
        offs_output_m = sub_group_start_idx + offsets_m
        
        # =================== START FIX ===================
        # Use offsets_n directly, which already includes base_n
        offs_output_n = offsets_n 
        # ==================== END FIX ====================
        
        output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
                                    offs_output_n[None, :] * stride_output_n)
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
        tl.store(output_ptrs, output, mask=output_mask)

@torch.inference_mode()
def fused_fp8_moe_stage_1_baseline_v2(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gate_weight_list: List[torch.Tensor],
    gate_ptrs_ptr: torch.Tensor,
    up_weight_list: List[torch.Tensor],
    up_ptrs_ptr: torch.Tensor,
    gate_scale_list: List[torch.Tensor],
    gate_scale_ptrs_ptr: torch.Tensor,
    up_scale_list: List[torch.Tensor],
    up_scale_ptrs_ptr: torch.Tensor,
    group_sizes: torch.Tensor,
    activated_group_idx: torch.Tensor,
    group_start_indices: torch.Tensor,
    num_active_experts: torch.Tensor,
    num_groups: int,
    gate_gemm_block_size=None,  # Auto-tune if None
    scale_block_size=128,
    num_stages=3,
    num_warps=4  
):
    """
    OPTIMIZED WRAPPER: Better defaults and auto-tuning
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    # === AUTO-TUNE BLOCK SIZES ===
    if gate_gemm_block_size is None:
        # Heuristics based on shape
        if N <= 2048:
            # Smaller N (Mixtral gate proj): prefer more N parallelism
            gate_gemm_block_size = [64, 32, 128]
        elif N <= 4096:
            # Medium N: balance
            gate_gemm_block_size = [64, 64, 128]
        else:
            # Large N (Mixtral up proj, Llama): larger tiles
            gate_gemm_block_size = [64, 128, 128]
        
        print(f"[Auto-tuned] block_size={gate_gemm_block_size} for shape M={M}, N={N}, K={K}")
    
    assert gate_gemm_block_size[2] == scale_block_size
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    # 2D GRID: (experts, N_blocks)
    grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
    
    fused_fp8_moe_baseline_optimized_v2[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        activated_group_idx, group_sizes, group_start_indices,
        num_active_experts,
        output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0),
        group_sizes.stride(0),
        group_start_indices.stride(0),
        gate_ptrs_ptr.stride(0),
        gate_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
        SCALE_BLOCK_SIZE_K=scale_block_size,
        num_stages=num_stages,
        num_warps=num_warps
    )
    
    return output


## Persistent + TMA

@triton.jit
def fused_fp8_moe_persistent_descriptor_kernel(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
    MAX_NUM_GROUPS: tl.constexpr
):
    """
    Persistent descriptor-based kernel for Fused FP8 MoE.
    
    - 1D persistent grid loops over flattened (group_pid, n_pid) work items.
    - Uses tl.make_tensor_descriptor for LHS, RHS (gate/up), and Output.
    - Uses tl.load for scales to avoid 16-byte TMA alignment issues.
    - NOTE: Does not use 'continue', wraps logic in 'if' blocks.
    """
    # 1D program ID
    start_pid = tl.program_id(axis=0)
    
    # --- Create Static Descriptors (constant for all work) ---
    lhs_desc = tl.make_tensor_descriptor(
        lhs_ptr,
        shape=[M, K],
        strides=[stride_lhs_m, stride_lhs_k],
        block_shape=[GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_K]
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M, N],
        strides=[stride_output_m, stride_output_n],
        block_shape=[GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N]
    )
    
    # Constants
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    total_work_items = MAX_NUM_GROUPS * NUM_N_BLOCKS
    
    # Load actual number of active experts
    actual_num_groups = tl.load(num_active_experts_ptr)
    
    # --- Persistent Loop over all work items ---
    for work_item_id in tl.range(start_pid, total_work_items, NUM_SMS):
        
        # Un-flatten 1D pid to 2D (group_pid, n_pid)
        group_pid = work_item_id // NUM_N_BLOCKS
        
        # --- FIX: Refactored 'continue' into 'if' block ---
        # Check if this group is active
        if group_pid < actual_num_groups:
            # Load this expert's metadata
            gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
            
            # Check if this group has any work
            if gm > 0:
                # This is a valid work item, calculate n_pid
                n_pid = work_item_id % NUM_N_BLOCKS
                
                group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
                start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
                
                # --- Create Dynamic Descriptors (per-expert) ---
                gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
                up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
                
                gate_desc = tl.make_tensor_descriptor(
                    gate_base_ptr,
                    shape=[N, K],
                    strides=[stride_gate_n, stride_gate_k],
                    block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K]
                )
                up_desc = tl.make_tensor_descriptor(
                    up_base_ptr,
                    shape=[N, K],
                    strides=[stride_up_n, stride_up_k],
                    block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K]
                )

                # Base pointers for scales (using tl.load)
                gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
                up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
                
                # N-block offsets for this work item
                offs_bn = n_pid * GEMM_BLOCK_SIZE_N # Base offset for descriptor
                scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
                
                # M-dimension offsets
                offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
                
                # Process all M-blocks for THIS expert and THIS N-block
                num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
                
                for sub_group_idx in range(num_sub_groups):
                    sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
                    offs_am = sub_group_start_idx # Base offset for descriptor

                    remaining_rows_in_group = start_idx + gm - sub_group_start_idx
                    valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
                    
                    abs_row_indices = sub_group_start_idx + offsets_m
                    
                    # Logical mask (for rows within this group)
                    valid_mask = (offsets_m < valid_rows_this_block)[:, None] # [M, 1]
                    
                    # Initialize accumulators
                    gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
                    up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
                    
                    num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
                    for k_block_idx in range(num_k_blocks):
                        offs_k = k_block_idx * GEMM_BLOCK_SIZE_K # Base offset for descriptor
                        
                        # --- Load Scales (No TMA) ---
                        scale_k_idx = k_block_idx
                        scale_offset = scale_n_idx * num_scale_k + scale_k_idx
                        
                        gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
                        up_scale = tl.load(up_scale_base_ptr + scale_offset)
                        
                        lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                                          scale_k_idx * stride_lhs_scale_k)
                        # Mask for tl.load needs to be boundary AND logical
                        lhs_scale_mask = (abs_row_indices[:, None] < M) & valid_mask
                        lhs_scale = tl.load(lhs_scale_ptrs, mask=lhs_scale_mask, other=1.0)
                        
                        # --- Load Data (TMA) ---
                        # Descriptors handle M, K, N boundary checks
                        lhs = lhs_desc.load([offs_am, offs_k])
                        gate_fp8 = gate_desc.load([offs_bn, offs_k])
                        up_fp8 = up_desc.load([offs_bn, offs_k])
                        
                        # Apply logical mask (zeros out rows not in this group)
                        lhs = tl.where(valid_mask, lhs, 0.0)
                        
                        # --- Compute ---
                        gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
                        up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
                    
                    # --- Epilogue ---
                    gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
                    output_acc = gate_activated * up_acc
                    output = output_acc.to(tl.bfloat16)
                    
                    # --- Store (TMA) ---
                    # Descriptor handles M, N boundary checks.
                    # Logical masking (valid_mask) was handled by zeroing `lhs`,
                    # which zeroes `acc` and `output`, so store is safe.
                    output_desc.store([offs_am, offs_bn], output)

@torch.inference_mode()
def fused_fp8_moe_stage_1_optimized(
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
    num_active_experts: torch.Tensor,
    num_groups: int, # This is the *max* number of groups
    gate_gemm_block_size=[64, 16, 128],
    scale_block_size=128,
    num_stages=3,
    num_warps=4
):
    """
    Persistent descriptor-based kernel for Fused FP8 MoE.
    Launches a 1D persistent grid.
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    assert gate_gemm_block_size[2] == scale_block_size, \
        f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    # --- ADDED: Allocator for device-side descriptors ---
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    # ---

    # --- UPDATED: Persistent 1D GRID ---
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    num_n_blocks = triton.cdiv(N, gate_gemm_block_size[1])
    
    # Total work items = (max experts) * (N blocks)
    total_work_items = num_groups * num_n_blocks
    
    # Launch at most NUM_SMS, but no more than the work we have
    grid = (min(NUM_SMS, total_work_items),)
    # ---

    fused_fp8_moe_persistent_descriptor_kernel[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        activated_group_idx, group_sizes, group_start_indices,
        num_active_experts,
        output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0),
        group_sizes.stride(0),
        group_start_indices.stride(0),
        gate_ptrs_ptr.stride(0),
        gate_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
        SCALE_BLOCK_SIZE_K=scale_block_size,
        # --- NEW ARGS ---
        NUM_SMS=NUM_SMS,
        NUM_N_BLOCKS=num_n_blocks,
        MAX_NUM_GROUPS=num_groups,
        # ---
        num_stages=num_stages,
        num_warps=num_warps
    )
    
    return output


# =============================================================================
# ALLOCATOR SETUP (Required for TMA descriptors)
# =============================================================================
_allocator_set = False

def _setup_allocator_once():
    """Set up Triton allocator for TMA descriptors (call once per process)."""
    global _allocator_set
    if not _allocator_set:
        def alloc_fn(size: int, alignment: int, stream: int):
            return torch.empty(size, device='cuda', dtype=torch.int8)
        
        triton.set_allocator(alloc_fn)
        _allocator_set = True


@triton.jit
def fused_fp8_moe_persistent_descriptor_kernel_v2(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
):
    """
    Fixed TMA Persistent Kernel - Version 2
    
    Key improvement: Better work distribution
    - Removed MAX_NUM_GROUPS parameter
    - Calculate total_work based on ACTUAL number of active experts
    - No wasted iterations checking invalid work items
    - Better load balance across SMs
    """
    # 1D program ID
    start_pid = tl.program_id(axis=0)
    
    # --- Load actual number of active experts FIRST ---
    actual_num_groups = tl.load(num_active_experts_ptr)
    
    # --- FIX 5: Calculate total work based on ACTUAL experts ---
    # OLD: total_work_items = MAX_NUM_GROUPS * NUM_N_BLOCKS  (e.g., 32 * 64 = 2048)
    # NEW: total_work_items = actual_num_groups * NUM_N_BLOCKS  (e.g., 8 * 64 = 512)
    total_work_items = actual_num_groups * NUM_N_BLOCKS
    
    # Early exit if no work for this thread block
    if start_pid >= total_work_items:
        return
    
    # --- Create Static Descriptors ONCE (outside all loops) ---
    lhs_desc = tl.make_tensor_descriptor(
        lhs_ptr,
        shape=[M, K],
        strides=[stride_lhs_m, stride_lhs_k],
        block_shape=[GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_K]
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M, N],
        strides=[stride_output_m, stride_output_n],
        block_shape=[GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N]
    )
    
    # Constants
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    
    # Pre-compute offsets (hoisted)
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    offsets_n = tl.arange(0, GEMM_BLOCK_SIZE_N)
    
    # --- Persistent Loop over all work items ---
    work_item_id = start_pid
    while work_item_id < total_work_items:
        
        # Un-flatten 1D pid to 2D (group_pid, n_pid)
        group_pid = work_item_id // NUM_N_BLOCKS
        n_pid = work_item_id % NUM_N_BLOCKS
        
        # --- FIX 5: No need to check group_pid < actual_num_groups ---
        # Since total_work_items is based on actual_num_groups, 
        # all work items are guaranteed to be valid
        
        # Load this expert's metadata
        gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
        
        # Check if this group has any work
        if gm > 0:
            group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
            start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
            
            # --- Create Dynamic Descriptors ONCE per expert ---
            gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
            up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
            
            gate_desc = tl.make_tensor_descriptor(
                gate_base_ptr,
                shape=[N, K],
                strides=[stride_gate_n, stride_gate_k],
                block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K]
            )
            up_desc = tl.make_tensor_descriptor(
                up_base_ptr,
                shape=[N, K],
                strides=[stride_up_n, stride_up_k],
                block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K]
            )

            # Base pointers for scales
            gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
            up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
            
            # N-block offsets for this work item (hoisted out of M-loop)
            offs_bn = n_pid * GEMM_BLOCK_SIZE_N
            offs_n = offs_bn + offsets_n
            n_mask = offs_n < N
            scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
            
            # Process all M-blocks for THIS expert and THIS N-block
            num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
            num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
            
            for sub_group_idx in range(num_sub_groups):
                sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
                offs_am = sub_group_start_idx

                remaining_rows_in_group = start_idx + gm - sub_group_start_idx
                valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
                
                abs_row_indices = sub_group_start_idx + offsets_m
                offs_m = abs_row_indices
                m_mask = offs_m < M
                
                # Logical mask (for rows within this group)
                valid_mask = (offsets_m < valid_rows_this_block)[:, None]
                
                # Initialize accumulators
                gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
                up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
                
                for k_block_idx in range(num_k_blocks):
                    offs_k = k_block_idx * GEMM_BLOCK_SIZE_K
                    
                    # --- Load Scales (No TMA) ---
                    scale_k_idx = k_block_idx
                    scale_offset = scale_n_idx * num_scale_k + scale_k_idx
                    
                    gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
                    up_scale = tl.load(up_scale_base_ptr + scale_offset)
                    
                    lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                                      scale_k_idx * stride_lhs_scale_k)
                    lhs_scale_mask = (abs_row_indices[:, None] < M) & valid_mask
                    lhs_scale = tl.load(lhs_scale_ptrs, mask=lhs_scale_mask, other=1.0)
                    
                    # --- Load Data (TMA) ---
                    lhs = lhs_desc.load([offs_am, offs_k])
                    gate_fp8 = gate_desc.load([offs_bn, offs_k])
                    up_fp8 = up_desc.load([offs_bn, offs_k])
                    
                    # Apply valid mask to LHS
                    lhs = tl.where(valid_mask, lhs, 0.0)
                    
                    # --- Compute ---
                    gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
                    up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
                
                # --- Epilogue ---
                gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
                output_acc = gate_activated * up_acc
                output = output_acc.to(tl.bfloat16)
                
                # --- Add proper output masking ---
                output_mask = (m_mask[:, None] & n_mask[None, :] & valid_mask)
                output_masked = tl.where(output_mask, output, 0.0)
                
                # Store (TMA)
                output_desc.store([offs_am, offs_bn], output_masked)
        
        # Increment to next work item for this thread block
        work_item_id += NUM_SMS


@torch.inference_mode()
def fused_fp8_moe_stage_1_persistent_v2(
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
    num_active_experts: torch.Tensor,
    num_groups: int,
    gate_gemm_block_size=[64, 256, 128],
    scale_block_size=128,
    num_stages=2,
    num_warps=8
):
    """
    Python wrapper for the fixed TMA persistent kernel (version 2).
    
    Key improvement: Uses actual number of experts for work calculation,
    eliminating wasted iterations and improving load balance.
    """
    # Set up allocator for TMA descriptors (once per process)
    _setup_allocator_once()
    
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    assert gate_gemm_block_size[2] == scale_block_size, \
        f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    # --- Persistent 1D GRID ---
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    num_n_blocks = triton.cdiv(N, gate_gemm_block_size[1])
    
    # --- FIX 5: Use actual number of experts ---
    # Get actual number of active experts from the tensor
    # actual_num_experts = num_active_experts.item()
    actual_num_experts = num_groups
    
    # Calculate total work based on ACTUAL experts
    total_work_items = actual_num_experts * num_n_blocks
    
    # Launch enough blocks to cover the work, but not more than we have work
    grid = (min(NUM_SMS, total_work_items),)

    fused_fp8_moe_persistent_descriptor_kernel_v2[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        activated_group_idx, group_sizes, group_start_indices,
        num_active_experts,
        output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0),
        group_sizes.stride(0),
        group_start_indices.stride(0),
        gate_ptrs_ptr.stride(0),
        gate_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
        SCALE_BLOCK_SIZE_K=scale_block_size,
        NUM_SMS=NUM_SMS,
        NUM_N_BLOCKS=num_n_blocks,
        # Note: MAX_NUM_GROUPS removed!
        num_stages=num_stages,
        num_warps=num_warps
    )
    
    return output

# @triton.jit
# def fused_fp8_moe_parallel_experts_kernel_no_activation(
#     lhs_ptr, lhs_scale_ptr,
#     gate_ptrs_ptr, up_ptrs_ptr,
#     gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#     group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
#     num_active_experts_ptr,
#     gate_output_ptr,
#     up_output_ptr,
#     M, N: tl.constexpr, K: tl.constexpr,
#     stride_lhs_m, stride_lhs_k,
#     stride_lhs_scale_m, stride_lhs_scale_k,
#     stride_gate_n, stride_gate_k,
#     stride_up_n, stride_up_k,
#     stride_gate_output_m, stride_gate_output_n,
#     stride_up_output_m, stride_up_output_n,
#     stride_group_idx, stride_group_sizes, stride_group_start_indices,
#     stride_weight_ptrs, stride_scale_ptrs,
#     GEMM_BLOCK_SIZE_M: tl.constexpr,
#     GEMM_BLOCK_SIZE_N: tl.constexpr,
#     GEMM_BLOCK_SIZE_K: tl.constexpr,
#     SCALE_BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     NO ACTIVATION FUSION: Returns separate gate and up projection accumulators in float32.
#     2D grid parallelizes over BOTH experts and N-blocks.
#     Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    
#     Key changes from optimized version:
#     - Removed SiLU activation and gating fusion
#     - Returns two separate float32 tensors: gate_output and up_output
#     - Stores float32 accumulators directly for downstream activation kernel
#     - Should reduce register pressure and improve FLOPS utilization
#     """
#     # 2D program IDs
#     group_pid = tl.program_id(axis=0)  # Which expert
#     n_pid = tl.program_id(axis=1)      # Which N-block
    
#     # Bounds check
#     num_groups = tl.load(num_active_experts_ptr)
#     if group_pid >= num_groups:
#         return
    
#     # Load THIS expert's metadata
#     gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
#     if gm == 0:
#         return
    
#     group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
#     start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
#     # Load THIS expert's weight pointers
#     gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
#     up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
#     # N-block offsets (fixed for this program) - HOISTED
#     offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
#     n_mask = offsets_n < N  # [N] - Computed once
    
#     # Scale N index (fixed for this program) - HOISTED
#     num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
#     scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
#     # M-dimension offsets - HOISTED
#     offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
#     # Process all M-blocks for THIS expert
#     num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
#     for sub_group_idx in range(num_sub_groups):
#         sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
#         remaining_rows_in_group = start_idx + gm - sub_group_start_idx
#         valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
#         abs_row_indices = sub_group_start_idx + offsets_m
        
#         # HOISTED: Compute M-dimension masks ONCE per M-block
#         m_mask = abs_row_indices < M  # [M]
#         valid_mask = offsets_m < valid_rows_this_block  # [M]
#         m_base_mask = m_mask & valid_mask  # [M] - Combined M-dimension mask
        
#         # Initialize accumulators
#         gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
#         up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
#         # K-loop: SIMPLIFIED! One K-block = one scale block
#         num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)

#         for k_block_idx in range(num_k_blocks):
#             k_start = k_block_idx * GEMM_BLOCK_SIZE_K
#             offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
#             k_mask = offsets_k < K  # [K]
            
#             # Load scales - direct mapping
#             scale_k_idx = k_block_idx
#             scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
#             gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
#             up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
#             # Load LHS scale
#             lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
#                                               scale_k_idx * stride_lhs_scale_k)
#             lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
#             # Load data pointers
#             lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
#             gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
#             up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
#             # OPTIMIZED: Simplified mask computations using pre-computed base masks
#             lhs_mask = m_base_mask[:, None] & k_mask[None, :]  # [M, 1] & [1, K] = [M, K]
#             rhs_mask = n_mask[:, None] & k_mask[None, :]  # [N, 1] & [1, K] = [N, K]
            
#             # Load data
#             lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
#             gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
#             up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
#             # Fused multiply-add pattern
#             gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
#             up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
#         # NO ACTIVATION/GATING FUSION - store float32 accumulators directly
#         # (Next kernel will handle activation and gating)
        
#         # Store results - gate output
#         offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
#         offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
#         gate_output_ptrs = gate_output_ptr + (offs_output_m[:, None] * stride_gate_output_m + 
#                                                offs_output_n[None, :] * stride_gate_output_n)
#         output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
#         tl.store(gate_output_ptrs, gate_acc, mask=output_mask)
        
#         # Store results - up output
#         up_output_ptrs = up_output_ptr + (offs_output_m[:, None] * stride_up_output_m + 
#                                            offs_output_n[None, :] * stride_up_output_n)
        
#         tl.store(up_output_ptrs, up_acc, mask=output_mask)

@triton.jit
def fused_fp8_moe_parallel_experts_kernel_no_activation(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    gate_output_ptr,
    up_output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_gate_output_m, stride_gate_output_n,
    stride_up_output_m, stride_up_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    OPTIMIZED: Manual load/compute interleaving to overlap up_fp8 load with gate compute.
    
    Key optimization:
    - Reordered operations in K-loop to hide memory latency
    - up_fp8 load happens DURING gate matmul (tensor cores busy)
    - Expected speedup: 10-15% for memory-bound cases
    """
    # 2D program IDs
    group_pid = tl.program_id(axis=0)  # Which expert
    n_pid = tl.program_id(axis=1)      # Which N-block
    
    # Bounds check
    num_groups = tl.load(num_active_experts_ptr)
    if group_pid >= num_groups:
        return
    
    # Load THIS expert's metadata
    gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
    if gm == 0:
        return
    
    group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
    start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
    # Load THIS expert's weight pointers
    gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # N-block offsets (fixed for this program) - HOISTED
    offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    n_mask = offsets_n < N  # [N] - Computed once
    
    # Scale N index (fixed for this program) - HOISTED
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
    # M-dimension offsets - HOISTED
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
    # Process all M-blocks for THIS expert
    num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
    for sub_group_idx in range(num_sub_groups):
        sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
        remaining_rows_in_group = start_idx + gm - sub_group_start_idx
        valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
        abs_row_indices = sub_group_start_idx + offsets_m
        
        # HOISTED: Compute M-dimension masks ONCE per M-block
        m_mask = abs_row_indices < M  # [M]
        valid_mask = offsets_m < valid_rows_this_block  # [M]
        m_base_mask = m_mask & valid_mask  # [M] - Combined M-dimension mask
        
        # Initialize accumulators
        gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        # K-loop: OPTIMIZED with load/compute interleaving
        num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)

        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * GEMM_BLOCK_SIZE_K
            offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
            k_mask = offsets_k < K  # [K]
            
            # Load scales - direct mapping
            scale_k_idx = k_block_idx
            scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
            gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
            up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
            # Load LHS scale
            lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                              scale_k_idx * stride_lhs_scale_k)
            lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
            # Compute data pointers
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
            up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
            # OPTIMIZED: Simplified mask computations using pre-computed base masks
            lhs_mask = m_base_mask[:, None] & k_mask[None, :]  # [M, 1] & [1, K] = [M, K]
            rhs_mask = n_mask[:, None] & k_mask[None, :]  # [N, 1] & [1, K] = [N, K]
            
            # ========================================================================
            # CRITICAL OPTIMIZATION: Interleave loads and computes for latency hiding
            # ========================================================================
            
            # Load LHS (shared by both GEMMs) and gate weights
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
            
            # START gate matmul - tensor cores are now BUSY
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
            
            # OVERLAP: Load up_fp8 WHILE gate matmul is executing
            # Memory subsystem works in parallel with tensor cores
            up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
            # Then perform up matmul
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
            
            # ========================================================================
            # Result: up_fp8 load latency is hidden behind gate compute
            # Expected improvement: 10-15% for memory-bound kernels
            # ========================================================================
        
        # NO ACTIVATION/GATING FUSION - store float32 accumulators directly
        # (Next kernel will handle activation and gating)
        
        # Store results - gate output
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
        gate_output_ptrs = gate_output_ptr + (offs_output_m[:, None] * stride_gate_output_m + 
                                               offs_output_n[None, :] * stride_gate_output_n)
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
        tl.store(gate_output_ptrs, gate_acc, mask=output_mask)
        
        # Store results - up output
        up_output_ptrs = up_output_ptr + (offs_output_m[:, None] * stride_up_output_m + 
                                           offs_output_n[None, :] * stride_up_output_n)
        
        tl.store(up_output_ptrs, up_acc, mask=output_mask)


@torch.inference_mode()
def fused_fp8_moe_stage_1_no_activation(
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
    num_active_experts: torch.Tensor,
    gate_gemm_block_size=[64, 16, 128],
    scale_block_size=128,
    num_stages=3,
    num_warps=4
):
    """
    NO ACTIVATION FUSION: Returns separate gate and up projection accumulators in float32.
    2D grid for parallel expert processing.
    Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    
    Stores float32 accumulators directly for a dedicated activation/gating kernel.
    
    Returns:
        gate_output: (M, N) float32 tensor - gate projection accumulator
        up_output: (M, N) float32 tensor - up projection accumulator
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    # Validate assumption
    assert gate_gemm_block_size[2] == scale_block_size, \
        f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
    # Allocate TWO output tensors in FLOAT32 for next kernel
    gate_output = torch.empty((M, N), dtype=torch.float32, device=device)
    up_output = torch.empty((M, N), dtype=torch.float32, device=device)
    
    num_groups = num_active_experts.item()
    
    # 2D GRID: (experts, N_blocks)
    grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
    
    fused_fp8_moe_parallel_experts_kernel_no_activation[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        activated_group_idx, group_sizes, group_start_indices,
        num_active_experts,
        gate_output,
        up_output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        gate_output.stride(0), gate_output.stride(1),
        up_output.stride(0), up_output.stride(1),
        activated_group_idx.stride(0),
        group_sizes.stride(0),
        group_start_indices.stride(0),
        gate_ptrs_ptr.stride(0),
        gate_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
        SCALE_BLOCK_SIZE_K=scale_block_size,
        num_stages=num_stages,
        num_warps=num_warps
    )
    
    return gate_output, up_output


# @triton.jit
# def fused_fp8_moe_parallel_experts_kernel_no_activation(
#     lhs_ptr, lhs_scale_ptr,
#     gate_ptrs_ptr, up_ptrs_ptr,
#     gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#     group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
#     num_active_experts_ptr,
#     gate_output_ptr,
#     up_output_ptr,
#     M, N: tl.constexpr, K: tl.constexpr,
#     stride_lhs_m, stride_lhs_k,
#     stride_lhs_scale_m, stride_lhs_scale_k,
#     stride_gate_n, stride_gate_k,
#     stride_up_n, stride_up_k,
#     stride_gate_output_m, stride_gate_output_n,
#     stride_up_output_m, stride_up_output_n,
#     stride_group_idx, stride_group_sizes, stride_group_start_indices,
#     stride_weight_ptrs, stride_scale_ptrs,
#     GEMM_BLOCK_SIZE_M: tl.constexpr,
#     GEMM_BLOCK_SIZE_N: tl.constexpr,
#     GEMM_BLOCK_SIZE_K: tl.constexpr,
#     SCALE_BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     NO ACTIVATION FUSION: Returns separate gate and up projection accumulators in float32.
#     2D grid parallelizes over BOTH experts and N-blocks.
#     Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    
#     Key optimizations:
#     - Removed SiLU activation and gating fusion
#     - Returns two separate float32 tensors: gate_output and up_output
#     - Stores float32 accumulators directly for downstream activation kernel
#     - SEPARATED gate and up GEMM loops: gate store overlaps with up computation
#     - Should reduce register pressure and improve FLOPS utilization
#     """
#     # 2D program IDs
#     group_pid = tl.program_id(axis=0)  # Which expert
#     n_pid = tl.program_id(axis=1)      # Which N-block
    
#     # Bounds check
#     num_groups = tl.load(num_active_experts_ptr)
#     if group_pid >= num_groups:
#         return
    
#     # Load THIS expert's metadata
#     gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
#     if gm == 0:
#         return
    
#     group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
#     start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
#     # Load THIS expert's weight pointers
#     gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
#     up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
#     # N-block offsets (fixed for this program) - HOISTED
#     offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
#     n_mask = offsets_n < N  # [N] - Computed once
    
#     # Scale N index (fixed for this program) - HOISTED
#     num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
#     scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
#     # M-dimension offsets - HOISTED
#     offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
#     # Process all M-blocks for THIS expert
#     num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
#     for sub_group_idx in range(num_sub_groups):
#         sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
#         remaining_rows_in_group = start_idx + gm - sub_group_start_idx
#         valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
#         abs_row_indices = sub_group_start_idx + offsets_m
        
#         # HOISTED: Compute M-dimension masks ONCE per M-block
#         m_mask = abs_row_indices < M  # [M]
#         valid_mask = offsets_m < valid_rows_this_block  # [M]
#         m_base_mask = m_mask & valid_mask  # [M] - Combined M-dimension mask
        
#         # Initialize accumulators
#         gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
#         # K-loop for GATE GEMM: Complete gate computation first
#         num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)

#         for k_block_idx in range(num_k_blocks):
#             k_start = k_block_idx * GEMM_BLOCK_SIZE_K
#             offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
#             k_mask = offsets_k < K  # [K]
            
#             # Load scales - direct mapping
#             scale_k_idx = k_block_idx
#             scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
#             gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
            
#             # Load LHS scale
#             lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
#                                               scale_k_idx * stride_lhs_scale_k)
#             lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
#             # Load data pointers
#             lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
#             gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
            
#             # OPTIMIZED: Simplified mask computations using pre-computed base masks
#             lhs_mask = m_base_mask[:, None] & k_mask[None, :]  # [M, 1] & [1, K] = [M, K]
#             rhs_mask = n_mask[:, None] & k_mask[None, :]  # [N, 1] & [1, K] = [N, K]
            
#             # Load data
#             lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
#             gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
            
#             # Gate GEMM
#             gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
        
#         # Store gate output (initiates async store)
#         offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
#         offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
#         gate_output_ptrs = gate_output_ptr + (offs_output_m[:, None] * stride_gate_output_m + 
#                                                offs_output_n[None, :] * stride_gate_output_n)
#         output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
#         tl.store(gate_output_ptrs, gate_acc, mask=output_mask)
        
#         # Now compute UP GEMM (while gate store is happening in background)
#         up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
#         for k_block_idx in range(num_k_blocks):
#             k_start = k_block_idx * GEMM_BLOCK_SIZE_K
#             offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
#             k_mask = offsets_k < K  # [K]
            
#             # Load scales - direct mapping
#             scale_k_idx = k_block_idx
#             scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
#             up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
#             # Load LHS scale
#             lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
#                                               scale_k_idx * stride_lhs_scale_k)
#             lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
#             # Load data pointers
#             lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
#             up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
#             # Masks (reuse from gate loop)
#             lhs_mask = m_base_mask[:, None] & k_mask[None, :]
#             rhs_mask = n_mask[:, None] & k_mask[None, :]
            
#             # Load data
#             lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
#             up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
#             # Up GEMM
#             up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
#         # Store up output
#         up_output_ptrs = up_output_ptr + (offs_output_m[:, None] * stride_up_output_m + 
#                                            offs_output_n[None, :] * stride_up_output_n)
        
#         tl.store(up_output_ptrs, up_acc, mask=output_mask)


# @torch.inference_mode()
# def fused_fp8_moe_stage_1_no_activation(
#     hidden_states: torch.Tensor,
#     hidden_states_scale: torch.Tensor,
#     gate_weight_list: list[torch.Tensor],
#     gate_ptrs_ptr: torch.Tensor,
#     up_weight_list: list[torch.Tensor],
#     up_ptrs_ptr: torch.Tensor,
#     gate_scale_list: list[torch.Tensor],
#     gate_scale_ptrs_ptr: torch.Tensor,
#     up_scale_list: list[torch.Tensor],
#     up_scale_ptrs_ptr: torch.Tensor,
#     group_sizes: torch.Tensor,
#     activated_group_idx: torch.Tensor,
#     group_start_indices: torch.Tensor,
#     num_active_experts: torch.Tensor,
#     gate_gemm_block_size=[64, 16, 128],
#     scale_block_size=128,
#     num_stages=3,
#     num_warps=4
# ):
#     """
#     NO ACTIVATION FUSION: Returns separate gate and up projection accumulators in float32.
#     2D grid for parallel expert processing.
#     Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    
#     Stores float32 accumulators directly for a dedicated activation/gating kernel.
    
#     Returns:
#         gate_output: (M, N) float32 tensor - gate projection accumulator
#         up_output: (M, N) float32 tensor - up projection accumulator
#     """
#     device = hidden_states.device
#     M = hidden_states.shape[0]
#     N = gate_weight_list[0].shape[0]
#     K = hidden_states.shape[1]
    
#     # Validate assumption
#     assert gate_gemm_block_size[2] == scale_block_size, \
#         f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
#     # Allocate TWO output tensors in FLOAT32 for next kernel
#     gate_output = torch.empty((M, N), dtype=torch.float32, device=device)
#     up_output = torch.empty((M, N), dtype=torch.float32, device=device)
    
#     num_groups = num_active_experts.item()
    
#     # 2D GRID: (experts, N_blocks)
#     grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
    
#     fused_fp8_moe_parallel_experts_kernel_no_activation[grid](
#         hidden_states, hidden_states_scale,
#         gate_ptrs_ptr, up_ptrs_ptr,
#         gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
#         activated_group_idx, group_sizes, group_start_indices,
#         num_active_experts,
#         gate_output,
#         up_output,
#         M, N, K,
#         hidden_states.stride(0), hidden_states.stride(1),
#         hidden_states_scale.stride(0), hidden_states_scale.stride(1),
#         gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
#         up_weight_list[0].stride(0), up_weight_list[0].stride(1),
#         gate_output.stride(0), gate_output.stride(1),
#         up_output.stride(0), up_output.stride(1),
#         activated_group_idx.stride(0),
#         group_sizes.stride(0),
#         group_start_indices.stride(0),
#         gate_ptrs_ptr.stride(0),
#         gate_scale_ptrs_ptr.stride(0),
#         GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
#         GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
#         GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
#         SCALE_BLOCK_SIZE_K=scale_block_size,
#         num_stages=num_stages,
#         num_warps=num_warps
#     )
    
#     return gate_output, up_output


@triton.jit
def fused_fp8_moe_pipelined_kernel(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    Manual pipelining: prefetch next iteration while computing current iteration.
    """
    # 2D program IDs
    group_pid = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)
    
    # Bounds check
    num_groups = tl.load(num_active_experts_ptr)
    if group_pid >= num_groups:
        return
    
    # Load THIS expert's metadata
    gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
    if gm == 0:
        return
    
    group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
    start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
    # Load THIS expert's weight pointers
    gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # N-block offsets (fixed for this program) - HOISTED
    offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    n_mask = offsets_n < N
    
    # Scale N index (fixed for this program) - HOISTED
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
    # M-dimension offsets - HOISTED
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
    # Process all M-blocks for THIS expert
    num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
    for sub_group_idx in range(num_sub_groups):
        sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
        remaining_rows_in_group = start_idx + gm - sub_group_start_idx
        valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
        abs_row_indices = sub_group_start_idx + offsets_m
        
        # HOISTED: Compute M-dimension masks ONCE per M-block
        m_mask = abs_row_indices < M
        valid_mask = offsets_m < valid_rows_this_block
        m_base_mask = m_mask & valid_mask
        
        # Initialize accumulators
        gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
        
        # ============================================================
        # MANUAL PIPELINING: 3-stage pipeline
        # Prologue -> Main Loop -> Epilogue pattern
        # ============================================================
        
        # PROLOGUE: Load first iteration
        k_block_idx = 0
        k_start = k_block_idx * GEMM_BLOCK_SIZE_K
        offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
        k_mask = offsets_k < K
        
        # Load scales for first iteration
        scale_k_idx = k_block_idx
        scale_offset = scale_n_idx * num_scale_k + scale_k_idx
        gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
        up_scale = tl.load(up_scale_base_ptr + scale_offset)
        
        lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                          scale_k_idx * stride_lhs_scale_k)
        lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
        
        # Load activation data for first iteration
        lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + 
                              offsets_k[None, :] * stride_lhs_k)
        lhs_mask = m_base_mask[:, None] & k_mask[None, :]
        lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
        
        # Load weight data for first iteration
        gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + 
                                     offsets_k[None, :] * stride_gate_k)
        up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + 
                                 offsets_k[None, :] * stride_up_k)
        rhs_mask = n_mask[:, None] & k_mask[None, :]
        
        gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
        up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
        
        # MAIN LOOP: Prefetch next while computing current
        for k_block_idx in range(num_k_blocks):
            # Save current iteration data
            curr_lhs = lhs
            curr_gate_fp8 = gate_fp8
            curr_up_fp8 = up_fp8
            curr_lhs_scale = lhs_scale
            curr_gate_scale = gate_scale
            curr_up_scale = up_scale
            
            # PREFETCH NEXT ITERATION (if not last iteration)
            if k_block_idx + 1 < num_k_blocks:
                k_start_next = (k_block_idx + 1) * GEMM_BLOCK_SIZE_K
                offsets_k_next = k_start_next + tl.arange(0, GEMM_BLOCK_SIZE_K)
                k_mask_next = offsets_k_next < K
                
                # Prefetch scales
                scale_k_idx_next = k_block_idx + 1
                scale_offset_next = scale_n_idx * num_scale_k + scale_k_idx_next
                gate_scale = tl.load(gate_scale_base_ptr + scale_offset_next)
                up_scale = tl.load(up_scale_base_ptr + scale_offset_next)
                
                lhs_scale_ptrs_next = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                                       scale_k_idx_next * stride_lhs_scale_k)
                lhs_scale = tl.load(lhs_scale_ptrs_next, mask=m_mask[:, None], other=1.0)
                
                # Prefetch activation data
                lhs_ptrs_next = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + 
                                           offsets_k_next[None, :] * stride_lhs_k)
                lhs_mask_next = m_base_mask[:, None] & k_mask_next[None, :]
                lhs = tl.load(lhs_ptrs_next, mask=lhs_mask_next, other=0.0)
                
                # Prefetch weight data
                gate_ptrs_next = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + 
                                                  offsets_k_next[None, :] * stride_gate_k)
                up_ptrs_next = up_base_ptr + (offsets_n[:, None] * stride_up_n + 
                                              offsets_k_next[None, :] * stride_up_k)
                rhs_mask_next = n_mask[:, None] & k_mask_next[None, :]
                
                gate_fp8 = tl.load(gate_ptrs_next, mask=rhs_mask_next, other=0.0)
                up_fp8 = tl.load(up_ptrs_next, mask=rhs_mask_next, other=0.0)
            
            # COMPUTE CURRENT ITERATION (overlaps with prefetch above)
            gate_acc += tl.dot(curr_lhs, tl.trans(curr_gate_fp8), out_dtype=tl.float32) * curr_lhs_scale * curr_gate_scale
            up_acc += tl.dot(curr_lhs, tl.trans(curr_up_fp8), out_dtype=tl.float32) * curr_lhs_scale * curr_up_scale
        
        # ============================================================
        # EPILOGUE - Same as original
        # ============================================================
        # SiLU activation: silu(x) = x / (1 + exp(-x))
        gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
        output_acc = gate_activated * up_acc
        output = output_acc.to(tl.bfloat16)
        
        # Store results
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
        output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
                                    offs_output_n[None, :] * stride_output_n)
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
        tl.store(output_ptrs, output, mask=output_mask)


@torch.inference_mode()
def fused_fp8_moe_stage_1_pipelined(
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
    num_active_experts: torch.Tensor,
    gate_gemm_block_size=[64, 64, 128],
    scale_block_size=128,
    num_stages=4,  # Can increase to 4-5 for more aggressive pipelining
    num_warps=4
):
    """
    Manual pipelining version - prefetch next K-block while computing current.
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    assert gate_gemm_block_size[2] == scale_block_size, \
        f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    num_groups = num_active_experts.item()
    grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
    
    fused_fp8_moe_pipelined_kernel[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        activated_group_idx, group_sizes, group_start_indices,
        num_active_experts,
        output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0),
        group_sizes.stride(0),
        group_start_indices.stride(0),
        gate_ptrs_ptr.stride(0),
        gate_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
        SCALE_BLOCK_SIZE_K=scale_block_size,
        num_stages=num_stages,
        num_warps=num_warps
    )
    
    return output


@triton.jit
def fused_fp8_moe_bf16_epilogue_kernel(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    BF16 epilogue optimization: convert to BF16 before expensive activation operations.
    """
    # 2D program IDs
    group_pid = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)
    
    # Bounds check
    num_groups = tl.load(num_active_experts_ptr)
    if group_pid >= num_groups:
        return
    
    # Load THIS expert's metadata
    gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
    if gm == 0:
        return
    
    group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
    start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
    # Load THIS expert's weight pointers
    gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # N-block offsets (fixed for this program) - HOISTED
    offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    n_mask = offsets_n < N
    
    # Scale N index (fixed for this program) - HOISTED
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
    # M-dimension offsets - HOISTED
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
    # Process all M-blocks for THIS expert
    num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
    for sub_group_idx in range(num_sub_groups):
        sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
        remaining_rows_in_group = start_idx + gm - sub_group_start_idx
        valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
        abs_row_indices = sub_group_start_idx + offsets_m
        
        # HOISTED: Compute M-dimension masks ONCE per M-block
        m_mask = abs_row_indices < M
        valid_mask = offsets_m < valid_rows_this_block
        m_base_mask = m_mask & valid_mask
        
        # Initialize accumulators
        gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        # K-loop: Same as original
        num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)

        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * GEMM_BLOCK_SIZE_K
            offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
            k_mask = offsets_k < K
            
            # Load scales
            scale_k_idx = k_block_idx
            scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
            gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
            up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
            lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                              scale_k_idx * stride_lhs_scale_k)
            lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
            # Load data pointers
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
            up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
            # Masks
            lhs_mask = m_base_mask[:, None] & k_mask[None, :]
            rhs_mask = n_mask[:, None] & k_mask[None, :]
            
            # Load data
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
            up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
            # Accumulate in FP32 (same as original)
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
        # ============================================================
        # BF16 EPILOGUE OPTIMIZATION
        # Convert to BF16 BEFORE expensive operations
        # ============================================================
        
        # Convert accumulators to BF16 immediately
        gate_acc_bf16 = gate_acc.to(tl.bfloat16)
        up_acc_bf16 = up_acc.to(tl.bfloat16)
        
        # SiLU activation in BF16: silu(x) = x / (1 + exp(-x))
        # exp() operates on BF16 values instead of FP32
        gate_activated_bf16 = gate_acc_bf16 / (1.0 + tl.exp(-gate_acc_bf16))
        
        # Element-wise multiplication in BF16
        output = gate_activated_bf16 * up_acc_bf16
        
        # Output is already BF16, no final conversion needed
        
        # ============================================================
        # Store results (same as original)
        # ============================================================
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
        output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
                                    offs_output_n[None, :] * stride_output_n)
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
        tl.store(output_ptrs, output, mask=output_mask)


@torch.inference_mode()
def fused_fp8_moe_stage_1_bf16_epilogue(
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
    num_active_experts: torch.Tensor,
    gate_gemm_block_size=[64, 16, 128],
    scale_block_size=128,
    num_stages=3,
    num_warps=4
):
    """
    BF16 epilogue optimization: convert to BF16 before activation/gating.
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    assert gate_gemm_block_size[2] == scale_block_size, \
        f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    num_groups = num_active_experts.item()
    grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
    
    fused_fp8_moe_bf16_epilogue_kernel[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        activated_group_idx, group_sizes, group_start_indices,
        num_active_experts,
        output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0),
        group_sizes.stride(0),
        group_start_indices.stride(0),
        gate_ptrs_ptr.stride(0),
        gate_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
        SCALE_BLOCK_SIZE_K=scale_block_size,
        num_stages=num_stages,
        num_warps=num_warps
    )
    
    return output


@triton.jit
def fused_fp8_moe_persistent_kernel(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    work_counter_ptr,  # Atomic counter for work distribution
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    Persistent kernel: Each CTA pulls work items from a global queue.
    Work item = (expert_id, n_tile_id) pair.
    """
    # Get CTA's persistent ID
    cta_id = tl.program_id(axis=0)
    
    # Load metadata once
    num_groups = tl.load(num_active_experts_ptr)
    num_n_tiles = tl.cdiv(N, GEMM_BLOCK_SIZE_N)
    total_work_items = num_groups * num_n_tiles
    
    # Precompute N-related constants (shared across all work items for this n_tile)
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
    # ============================================================
    # PERSISTENT LOOP: Pull work until queue is empty
    # ============================================================
    while True:
        # Atomically grab next work item
        work_idx = tl.atomic_add(work_counter_ptr, 1)
        
        if work_idx >= total_work_items:
            break  # No more work
        
        # Decode work_idx into (group_pid, n_pid)
        group_pid = work_idx // num_n_tiles
        n_pid = work_idx % num_n_tiles
        
        # Load THIS expert's metadata
        gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
        if gm == 0:
            continue  # Skip empty experts
        
        group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
        start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
        
        # Load THIS expert's weight pointers
        gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
        up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
        gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
        up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
        
        # N-block offsets for THIS work item
        offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        n_mask = offsets_n < N
        scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
        
        # Process all M-blocks for THIS expert's N-tile
        num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
        
        for sub_group_idx in range(num_sub_groups):
            sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
            remaining_rows_in_group = start_idx + gm - sub_group_start_idx
            valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
            
            abs_row_indices = sub_group_start_idx + offsets_m
            
            # Compute M-dimension masks
            m_mask = abs_row_indices < M
            valid_mask = offsets_m < valid_rows_this_block
            m_base_mask = m_mask & valid_mask
            
            # Initialize accumulators
            gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
            up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
            
            # K-loop
            num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
            
            for k_block_idx in range(num_k_blocks):
                k_start = k_block_idx * GEMM_BLOCK_SIZE_K
                offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
                k_mask = offsets_k < K
                
                # Load scales
                scale_k_idx = k_block_idx
                scale_offset = scale_n_idx * num_scale_k + scale_k_idx
                
                gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
                up_scale = tl.load(up_scale_base_ptr + scale_offset)
                
                lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                                  scale_k_idx * stride_lhs_scale_k)
                lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
                
                # Load data
                lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + 
                                      offsets_k[None, :] * stride_lhs_k)
                gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + 
                                             offsets_k[None, :] * stride_gate_k)
                up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + 
                                         offsets_k[None, :] * stride_up_k)
                
                lhs_mask = m_base_mask[:, None] & k_mask[None, :]
                rhs_mask = n_mask[:, None] & k_mask[None, :]
                
                lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
                gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
                up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
                
                # Accumulate
                gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
                up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
            
            # Epilogue
            gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
            output_acc = gate_activated * up_acc
            output = output_acc.to(tl.bfloat16)
            
            # Store results
            offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
            offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
            
            output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
                                        offs_output_n[None, :] * stride_output_n)
            output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
            
            tl.store(output_ptrs, output, mask=output_mask)


@triton.jit
def fused_fp8_moe_persistent_kernel(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    work_counter_ptr,  # Atomic counter for work distribution
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    MAX_WORK_ITEMS_PER_CTA: tl.constexpr,
):
    """
    Persistent kernel: Each CTA pulls work items from a global queue.
    Work item = (expert_id, n_tile_id) pair.
    """
    # Get CTA's persistent ID
    cta_id = tl.program_id(axis=0)
    
    # Load metadata once
    num_groups = tl.load(num_active_experts_ptr)
    num_n_tiles = tl.cdiv(N, GEMM_BLOCK_SIZE_N)
    total_work_items = num_groups * num_n_tiles
    
    # Precompute constants
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
    # ============================================================
    # PERSISTENT LOOP: Pull work until queue is empty
    # Guard entire body with condition (no return/break allowed)
    # ============================================================
    for _ in range(MAX_WORK_ITEMS_PER_CTA):
        # Atomically grab next work item
        work_idx = tl.atomic_add(work_counter_ptr, 1)
        
        # Guard entire body - if work exhausted, this iteration does nothing
        if work_idx < total_work_items:
            # Decode work_idx into (group_pid, n_pid)
            group_pid = work_idx // num_n_tiles
            n_pid = work_idx % num_n_tiles
            
            # Load THIS expert's metadata
            gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
            
            # Only process if group is non-empty
            if gm > 0:
                group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
                start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
                
                # Load THIS expert's weight pointers
                gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
                up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
                gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
                up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
                
                # N-block offsets for THIS work item
                offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
                n_mask = offsets_n < N
                scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
                
                # Process all M-blocks for THIS expert's N-tile
                num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
                
                for sub_group_idx in range(num_sub_groups):
                    sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
                    remaining_rows_in_group = start_idx + gm - sub_group_start_idx
                    valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
                    
                    abs_row_indices = sub_group_start_idx + offsets_m
                    
                    # Compute M-dimension masks
                    m_mask = abs_row_indices < M
                    valid_mask = offsets_m < valid_rows_this_block
                    m_base_mask = m_mask & valid_mask
                    
                    # Initialize accumulators
                    gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
                    up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
                    
                    # K-loop
                    num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
                    
                    for k_block_idx in range(num_k_blocks):
                        k_start = k_block_idx * GEMM_BLOCK_SIZE_K
                        offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
                        k_mask = offsets_k < K
                        
                        # Load scales
                        scale_k_idx = k_block_idx
                        scale_offset = scale_n_idx * num_scale_k + scale_k_idx
                        
                        gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
                        up_scale = tl.load(up_scale_base_ptr + scale_offset)
                        
                        lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                                          scale_k_idx * stride_lhs_scale_k)
                        lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
                        
                        # Load data
                        lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + 
                                              offsets_k[None, :] * stride_lhs_k)
                        gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + 
                                                     offsets_k[None, :] * stride_gate_k)
                        up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + 
                                                 offsets_k[None, :] * stride_up_k)
                        
                        lhs_mask = m_base_mask[:, None] & k_mask[None, :]
                        rhs_mask = n_mask[:, None] & k_mask[None, :]
                        
                        lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
                        gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
                        up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
                        
                        # Accumulate
                        gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
                        up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
                    
                    # Epilogue
                    gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
                    output_acc = gate_activated * up_acc
                    output = output_acc.to(tl.bfloat16)
                    
                    # Store results
                    offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
                    offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
                    
                    output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
                                                offs_output_n[None, :] * stride_output_n)
                    output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
                    
                    tl.store(output_ptrs, output, mask=output_mask)


@torch.inference_mode()
def fused_fp8_moe_stage_1_persistent(
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
    num_active_experts: torch.Tensor,
    gate_gemm_block_size=[64, 32, 128],
    scale_block_size=128,
    num_stages=3,
    num_warps=4,
    num_ctas=None  # If None, auto-detect SM count
):
    """
    Persistent kernel: Launch num_SMs CTAs that pull work from a queue.
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    assert gate_gemm_block_size[2] == scale_block_size, \
        f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    # Calculate total work items
    num_groups = num_active_experts.item()
    num_n_tiles = (N + gate_gemm_block_size[1] - 1) // gate_gemm_block_size[1]
    total_work_items = num_groups * num_n_tiles
    
    # Auto-detect SM count if not specified
    if num_ctas is None:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count
        # Launch 2x SMs for better occupancy, but cap at total work items
        num_ctas = min(num_sms*2, total_work_items)

    # Calculate max work items per CTA (upper bound for loop)
    # Add some buffer to handle imbalance
    max_work_per_cta = (total_work_items + num_ctas - 1) // num_ctas + 10
    
    # Initialize atomic work counter
    work_counter = torch.zeros(1, dtype=torch.int32, device=device)
    
    # 1D grid: persistent CTAs
    grid = (num_ctas,)
    
    fused_fp8_moe_persistent_kernel[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        activated_group_idx, group_sizes, group_start_indices,
        num_active_experts,
        output,
        work_counter,  # Atomic counter
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0),
        group_sizes.stride(0),
        group_start_indices.stride(0),
        gate_ptrs_ptr.stride(0),
        gate_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gate_gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gate_gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gate_gemm_block_size[2],
        SCALE_BLOCK_SIZE_K=scale_block_size,
        MAX_WORK_ITEMS_PER_CTA=max_work_per_cta,
        num_stages=num_stages,
        num_warps=num_warps
    )
    
    return output
