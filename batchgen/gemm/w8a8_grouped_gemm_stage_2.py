# import torch
# import triton
# import triton.language as tl


# @triton.jit
# def fused_dequant_grouped_gemm_fp8_fp8_kernel_optimized(
#     lhs_ptr, lhs_scale_ptr,
#     rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
#     group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
#     num_active_experts_ptr,
#     output_ptr,
#     M, N: tl.constexpr, K: tl.constexpr,
#     stride_lhs_m, stride_lhs_k,
#     stride_lhs_scale_m, stride_lhs_scale_k,
#     stride_rhs_n, stride_rhs_k,
#     stride_output_m, stride_output_n,
#     stride_group_idx, stride_group_sizes, stride_group_start_indices,
#     stride_rhs_ptrs, stride_rhs_scale_ptrs,
#     GEMM_BLOCK_SIZE_M: tl.constexpr,
#     GEMM_BLOCK_SIZE_N: tl.constexpr,
#     GEMM_BLOCK_SIZE_K: tl.constexpr,
#     SCALE_BLOCK_SIZE_N: tl.constexpr,
#     SCALE_BLOCK_SIZE_K: tl.constexpr,
# ):
#     """
#     OPTIMIZED: 2D grid parallelizes over BOTH experts and N-blocks.
    
#     Key optimizations:
#     - 2D grid instead of 1D (parallel experts)
#     - Hoisted mask computations
#     - Simplified K-loop (assumes BLOCK_K == SCALE_BLOCK_K)
#     - Removed serial expert loop
#     """
#     # 2D program IDs
#     group_pid = tl.program_id(axis=0)  # Which expert
#     n_pid = tl.program_id(axis=1)      # Which N-block
    
#     # Bounds check
#     num_groups = tl.load(num_active_experts_ptr)
#     if group_pid >= num_groups:
#         return
    
#     # Load expert metadata
#     gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
#     if gm == 0:
#         return
    
#     group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
#     start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
#     # Load weight pointers for this expert
#     rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(tl.float8e4nv))
#     rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))
    
#     # N-dimension setup (hoisted)
#     offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
#     n_mask = offsets_n < N
#     num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
#     scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
#     offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
#     # Process M-blocks for this expert
#     num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
#     for sub_group_idx in range(num_sub_groups):
#         sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
#         remaining_rows = start_idx + gm - sub_group_start_idx
#         valid_rows = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows)
        
#         abs_row_indices = sub_group_start_idx + offsets_m
        
#         # Hoisted M-dimension masks
#         m_mask = abs_row_indices < M
#         valid_mask = offsets_m < valid_rows
#         m_base_mask = m_mask & valid_mask
        
#         # Initialize accumulator
#         acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
#         # K-loop
#         num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
        
#         for k_block_idx in range(num_k_blocks):
#             k_start = k_block_idx * GEMM_BLOCK_SIZE_K
#             offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
#             k_mask = offsets_k < K
            
#             # Load scales (direct mapping when BLOCK_K == SCALE_BLOCK_K)
#             scale_k_idx = k_block_idx
#             scale_offset = scale_n_idx * num_scale_k + scale_k_idx
#             rhs_scale = tl.load(rhs_scale_base_ptr + scale_offset)
            
#             lhs_scale_k = k_block_idx
#             lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
#                                               lhs_scale_k * stride_lhs_scale_k)
#             lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
#             # Load data
#             lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
#             rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
            
#             # Simplified masks
#             lhs_mask = m_base_mask[:, None] & k_mask[None, :]
#             rhs_mask = n_mask[:, None] & k_mask[None, :]
            
#             lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
#             rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0)
            
#             # Compute with scales
#             acc += tl.dot(lhs, tl.trans(rhs_fp8), out_dtype=tl.float32) * lhs_scale * rhs_scale
        
#         # Store output
#         offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
#         offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
#         output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
#                                     offs_output_n[None, :] * stride_output_n)
#         output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
#         output = acc.to(tl.bfloat16)
#         tl.store(output_ptrs, output, mask=output_mask)


# @torch.inference_mode()
# def fused_dequant_grouped_gemm_fp8_fp8_triton_optimized(
#     lhs: torch.Tensor,
#     lhs_scale: torch.Tensor,
#     rhs_list: list[torch.Tensor],
#     rhs_ptrs_ptr: torch.Tensor,
#     rhs_scale_list: list[torch.Tensor],
#     rhs_scale_ptrs_ptr: torch.Tensor,
#     group_size: torch.Tensor,
#     activated_group_idx: torch.Tensor,
#     group_start_indices: torch.Tensor,
#     num_active_experts: torch.Tensor,
#     gemm_block_size=(64, 32, 128), 
#     scale_block_size=(128, 128),
#     num_stages=2,
#     num_warps=4
# ):
#     """
#     OPTIMIZED: 2D grid for parallel expert processing.
#     """
#     assert lhs.dtype == torch.float8_e4m3fn
#     assert lhs_scale.dtype == torch.float32
#     assert all(r.dtype == torch.float8_e4m3fn for r in rhs_list)
#     assert all(s.dtype == torch.float32 for s in rhs_scale_list)
    
#     device = lhs.device
#     M = lhs.shape[0]
#     N = rhs_list[0].shape[0]
#     K = lhs.shape[1]
    
#     output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
#     num_groups = num_active_experts.item()
    
#     # 2D GRID: (experts, N_blocks)
#     grid = (num_groups, triton.cdiv(N, gemm_block_size[1]))
    
#     fused_dequant_grouped_gemm_fp8_fp8_kernel_optimized[grid](
#         lhs, lhs_scale,
#         rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
#         activated_group_idx, group_size, group_start_indices,
#         num_active_experts,
#         output,
#         M, N, K,
#         lhs.stride(0), lhs.stride(1),
#         lhs_scale.stride(0), lhs_scale.stride(1),
#         rhs_list[0].stride(0), rhs_list[0].stride(1),
#         output.stride(0), output.stride(1),
#         activated_group_idx.stride(0), 
#         group_size.stride(0), group_start_indices.stride(0),
#         rhs_ptrs_ptr.stride(0), rhs_scale_ptrs_ptr.stride(0),
#         GEMM_BLOCK_SIZE_M=gemm_block_size[0],
#         GEMM_BLOCK_SIZE_N=gemm_block_size[1],
#         GEMM_BLOCK_SIZE_K=gemm_block_size[2],
#         SCALE_BLOCK_SIZE_N=scale_block_size[0],
#         SCALE_BLOCK_SIZE_K=scale_block_size[1],
#         num_warps=num_warps,
#         num_stages=num_stages
#     )
    
#     return output


import torch
import triton
import triton.language as tl


@triton.jit
def fused_dequant_grouped_gemm_fp8_fp8_kernel_optimized(
    lhs_ptr, lhs_scale_ptr,
    rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_rhs_n, stride_rhs_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_rhs_ptrs, stride_rhs_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    OPTIMIZED: 2D grid - parallelize over experts and N-blocks.
    CRITICAL: Keep close to original structure to avoid hangs.
    """
    # 2D program IDs
    group_pid = tl.program_id(axis=0)  # Which expert
    n_pid = tl.program_id(axis=1)      # Which N-block
    
    # Load num_groups and bounds check
    num_groups = tl.load(num_active_experts_ptr)
    if group_pid >= num_groups:
        return
    
    # Load THIS expert's metadata
    gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
    if gm == 0:
        return
    
    group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
    start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
    # Load weight pointers for this expert
    rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(tl.float8e4nv))
    rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # N-dimension setup - HOISTED (computed once per program)
    offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    scale_n = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    
    # Process M-blocks for this expert
    num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
    for sub_group_idx in range(num_sub_groups):
        sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
        remaining_rows_in_group = start_idx + gm - sub_group_start_idx
        valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
        offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
        abs_row_indices = sub_group_start_idx + offsets_m
        
        # Initialize accumulator
        acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        # K-loop
        num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
        for k_idx in range(num_k_blocks):
            offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
            
            # Compute pointers
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
            
            # Load RHS scale (per tile)
            scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
            rhs_scale = tl.load(scale_ptr)
            
            # Masks
            lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
            rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)
            
            # Load data
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0)
            
            # Load LHS scale - KEEP ORIGINAL HARDCODED 128 (don't assume SCALE_BLOCK_SIZE_K)
            lhs_scale_k = k_idx * GEMM_BLOCK_SIZE_K // 128
            l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
            lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0)
            
            # Compute
            acc += tl.dot(lhs, tl.trans(rhs_fp8), out_dtype=tl.float32) * lhs_scale * rhs_scale
        
        # Store output
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)
        
        output = acc.to(tl.bfloat16)
        tl.store(output_ptrs, output, mask=output_mask)


@torch.inference_mode()
def fused_dequant_grouped_gemm_fp8_fp8_triton_optimized(
    lhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_list: list[torch.Tensor],
    rhs_ptrs_ptr: torch.Tensor,
    rhs_scale_list: list[torch.Tensor],
    rhs_scale_ptrs_ptr: torch.Tensor,
    group_size: torch.Tensor,
    activated_group_idx: torch.Tensor,
    group_start_indices: torch.Tensor,
    num_active_experts: torch.Tensor,
    gemm_block_size=(64, 64, 128), 
    scale_block_size=(128, 128),
    num_stages=3,
    num_warps=4
):
    """
    OPTIMIZED: 2D grid instead of 1D + serial expert loop.
    """
    device = lhs.device
    M = lhs.shape[0]
    N = rhs_list[0].shape[0]
    K = lhs.shape[1]
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    # num_groups = num_active_experts.item()
    # TODO:
    num_groups = 16
    
    # 2D GRID: (experts, N_blocks)
    grid = (num_groups, triton.cdiv(N, gemm_block_size[1]))
    
    fused_dequant_grouped_gemm_fp8_fp8_kernel_optimized[grid](
        lhs, lhs_scale,
        rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
        activated_group_idx, group_size, group_start_indices,
        num_active_experts,
        output,
        M, N, K,
        lhs.stride(0), lhs.stride(1),
        lhs_scale.stride(0), lhs_scale.stride(1),
        rhs_list[0].stride(0), rhs_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0), 
        group_size.stride(0), group_start_indices.stride(0),
        rhs_ptrs_ptr.stride(0), rhs_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gemm_block_size[2],
        SCALE_BLOCK_SIZE_N=scale_block_size[0],
        SCALE_BLOCK_SIZE_K=scale_block_size[1],
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    return output



@triton.jit
def fused_dequant_grouped_gemm_fp8_fp8_fp32_kernel(
    lhs_ptr, lhs_scale_ptr,
    rhs_ptrs_ptr, rhs_scale_ptrs_ptr, 
    group_idx_ptr, group_sizes_ptr, group_start_indices_ptr,
    num_active_experts_ptr,
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_rhs_n, stride_rhs_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_rhs_ptrs, stride_rhs_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    OPTIMIZED: 2D grid - parallelize over experts and N-blocks.
    CRITICAL: Keep close to original structure to avoid hangs.
    """
    # 2D program IDs
    group_pid = tl.program_id(axis=0)  # Which expert
    n_pid = tl.program_id(axis=1)      # Which N-block
    
    # Load num_groups and bounds check
    num_groups = tl.load(num_active_experts_ptr)
    if group_pid >= num_groups:
        return
    
    # Load THIS expert's metadata
    gm = tl.load(group_sizes_ptr + group_pid * stride_group_sizes)
    if gm == 0:
        return
    
    group_idx = tl.load(group_idx_ptr + group_pid * stride_group_idx)
    start_idx = tl.load(group_start_indices_ptr + group_pid * stride_group_start_indices)
    
    # Load weight pointers for this expert
    rhs_base_ptr = tl.load(rhs_ptrs_ptr + group_idx * stride_rhs_ptrs).to(tl.pointer_type(tl.float8e4nv))
    rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + group_idx * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # N-dimension setup - HOISTED (computed once per program)
    offsets_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    scale_n = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    
    # Process M-blocks for this expert
    num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
    for sub_group_idx in range(num_sub_groups):
        sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
        remaining_rows_in_group = start_idx + gm - sub_group_start_idx
        valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, remaining_rows_in_group)
        
        offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
        abs_row_indices = sub_group_start_idx + offsets_m
        
        # Initialize accumulator
        acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        # K-loop
        num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
        for k_idx in range(num_k_blocks):
            offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
            
            # Compute pointers
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
            
            # Load RHS scale (per tile)
            scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
            rhs_scale = tl.load(scale_ptr)
            
            # Masks
            lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
            rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)
            
            # Load data
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0)
            
            # Load LHS scale - KEEP ORIGINAL HARDCODED 128 (don't assume SCALE_BLOCK_SIZE_K)
            lhs_scale_k = k_idx * GEMM_BLOCK_SIZE_K // 128
            l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
            lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0)
            
            # Compute
            acc += tl.dot(lhs, tl.trans(rhs_fp8), out_dtype=tl.float32) * lhs_scale * rhs_scale
        
        # Store output
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None] < valid_rows_this_block)
        
        output = acc.to(tl.bfloat16)
        tl.store(output_ptrs, acc, mask=output_mask)


@torch.inference_mode()
def fused_dequant_grouped_gemm_fp8_fp8_fp32_triton(
    lhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_list: list[torch.Tensor],
    rhs_ptrs_ptr: torch.Tensor,
    rhs_scale_list: list[torch.Tensor],
    rhs_scale_ptrs_ptr: torch.Tensor,
    group_size: torch.Tensor,
    activated_group_idx: torch.Tensor,
    group_start_indices: torch.Tensor,
    num_active_experts: torch.Tensor,
    gemm_block_size=(64, 64, 128), 
    scale_block_size=(128, 128),
    num_stages=3,
    num_warps=4
):
    """
    OPTIMIZED: 2D grid instead of 1D + serial expert loop.
    """
    device = lhs.device
    M = lhs.shape[0]
    N = rhs_list[0].shape[0]
    K = lhs.shape[1]
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    # num_groups = num_active_experts.item()
    # TODO:
    num_groups = 16
    
    # 2D GRID: (experts, N_blocks)
    grid = (num_groups, triton.cdiv(N, gemm_block_size[1]))
    
    fused_dequant_grouped_gemm_fp8_fp8_fp32_kernel[grid](
        lhs, lhs_scale,
        rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
        activated_group_idx, group_size, group_start_indices,
        num_active_experts,
        output,
        M, N, K,
        lhs.stride(0), lhs.stride(1),
        lhs_scale.stride(0), lhs_scale.stride(1),
        rhs_list[0].stride(0), rhs_list[0].stride(1),
        output.stride(0), output.stride(1),
        activated_group_idx.stride(0), 
        group_size.stride(0), group_start_indices.stride(0),
        rhs_ptrs_ptr.stride(0), rhs_scale_ptrs_ptr.stride(0),
        GEMM_BLOCK_SIZE_M=gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gemm_block_size[2],
        SCALE_BLOCK_SIZE_N=scale_block_size[0],
        SCALE_BLOCK_SIZE_K=scale_block_size[1],
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    return output
