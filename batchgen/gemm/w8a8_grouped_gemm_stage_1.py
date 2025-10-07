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
@triton.jit
def fused_fp8_moe_parallel_experts_kernel_optimized(
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
    OPTIMIZED: 2D grid parallelizes over BOTH experts and N-blocks.
    Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    
    Key optimizations:
    - Removed nested quantization loop (1:1 mapping)
    - Hoisted mask computations outside K-loop
    - Eliminated redundant scale_n calculation
    - Simplified scale loading masks
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
        
        # K-loop: SIMPLIFIED! One K-block = one scale block
        num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)

        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * GEMM_BLOCK_SIZE_K
            offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
            k_mask = offsets_k < K  # [K]
            
            # Load scales - direct mapping
            scale_k_idx = k_block_idx
            # FIXED: Use pre-computed scale_n_idx instead of recalculating
            scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
            gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
            up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
            # OPTIMIZED: Use m_mask instead of recomputing
            lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                              scale_k_idx * stride_lhs_scale_k)
            lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
            # Load data pointers
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
            up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
            # OPTIMIZED: Simplified mask computations using pre-computed base masks
            lhs_mask = m_base_mask[:, None] & k_mask[None, :]  # [M, 1] & [1, K] = [M, K]
            rhs_mask = n_mask[:, None] & k_mask[None, :]  # [N, 1] & [1, K] = [N, K]
            
            # Load data
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
            up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
            # Fused multiply-add pattern (keep original for performance)
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
        # SiLU activation: silu(x) = x / (1 + exp(-x))
        gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
        output_acc = gate_activated * up_acc
        output = output_acc.to(tl.bfloat16)
        
        # Store results
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
        output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
                                    offs_output_n[None, :] * stride_output_n)
        # OPTIMIZED: Use pre-computed valid_mask
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
        tl.store(output_ptrs, output, mask=output_mask)


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
    gate_gemm_block_size=[64, 16, 128],
    scale_block_size=128,
    num_stages=3,
    num_warps=4
):
    """
    OPTIMIZED: 2D grid for parallel expert processing.
    Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    # Validate assumption
    assert gate_gemm_block_size[2] == scale_block_size, \
        f"GEMM_BLOCK_SIZE_K ({gate_gemm_block_size[2]}) must equal SCALE_BLOCK_SIZE_K ({scale_block_size})"
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    num_groups = num_active_experts.item()
    
    # 2D GRID: (experts, N_blocks)
    grid = (num_groups, triton.cdiv(N, gate_gemm_block_size[1]))
    
    fused_fp8_moe_parallel_experts_kernel_optimized[grid](
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
    NO ACTIVATION FUSION: Returns separate gate and up projection accumulators in float32.
    2D grid parallelizes over BOTH experts and N-blocks.
    Assumes GEMM_BLOCK_SIZE_K == SCALE_BLOCK_SIZE_K (both 128).
    
    Key optimizations:
    - Removed SiLU activation and gating fusion
    - Returns two separate float32 tensors: gate_output and up_output
    - Stores float32 accumulators directly for downstream activation kernel
    - SEPARATED gate and up GEMM loops: gate store overlaps with up computation
    - Should reduce register pressure and improve FLOPS utilization
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
        
        # K-loop for GATE GEMM: Complete gate computation first
        num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)

        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * GEMM_BLOCK_SIZE_K
            offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
            k_mask = offsets_k < K  # [K]
            
            # Load scales - direct mapping
            scale_k_idx = k_block_idx
            scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
            gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
            
            # Load LHS scale
            lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                              scale_k_idx * stride_lhs_scale_k)
            lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
            # Load data pointers
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
            
            # OPTIMIZED: Simplified mask computations using pre-computed base masks
            lhs_mask = m_base_mask[:, None] & k_mask[None, :]  # [M, 1] & [1, K] = [M, K]
            rhs_mask = n_mask[:, None] & k_mask[None, :]  # [N, 1] & [1, K] = [N, K]
            
            # Load data
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
            
            # Gate GEMM
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
        
        # Store gate output (initiates async store)
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
        gate_output_ptrs = gate_output_ptr + (offs_output_m[:, None] * stride_gate_output_m + 
                                               offs_output_n[None, :] * stride_gate_output_n)
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
        tl.store(gate_output_ptrs, gate_acc, mask=output_mask)
        
        # Now compute UP GEMM (while gate store is happening in background)
        up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * GEMM_BLOCK_SIZE_K
            offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
            k_mask = offsets_k < K  # [K]
            
            # Load scales - direct mapping
            scale_k_idx = k_block_idx
            scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
            up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
            # Load LHS scale
            lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                              scale_k_idx * stride_lhs_scale_k)
            lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
            # Load data pointers
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
            # Masks (reuse from gate loop)
            lhs_mask = m_base_mask[:, None] & k_mask[None, :]
            rhs_mask = n_mask[:, None] & k_mask[None, :]
            
            # Load data
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
            # Up GEMM
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
        # Store up output
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
