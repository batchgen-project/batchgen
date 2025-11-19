import torch
import triton
import triton.language as tl
from typing import Optional

HAS_TENSOR_DESC = True
try:
    from triton.language.extra.cuda import libcuda
except ImportError:
    HAS_TENSOR_DESC = False

@triton.jit
def fused_fp8_moe_parallel_experts_kernel_tma_3d(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    expert_tokens_ptr,   # <--- Changed: Now holds counts (int32), not offsets
    output_ptr,
    M_max, N: tl.constexpr, K: tl.constexpr, # M_max is max tokens per expert (dim 1 size)
    stride_lhs_e, stride_lhs_m, stride_lhs_k,             # 3D Strides
    stride_lhs_scale_e, stride_lhs_scale_m, stride_lhs_scale_k, # 3D Strides
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_e, stride_output_m, stride_output_n,    # 3D Strides
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    Stage 1 Kernel (Gate + Up Proj) for 3D Input (E, T, H).
    - Grid axis 0: Physical expert index [0, experts_per_rank)
    """
    expert_idx = tl.program_id(axis=0) 
    n_pid = tl.program_id(axis=1)
    
    # Load token count for this expert
    gm = tl.load(expert_tokens_ptr + expert_idx).to(tl.int32)
    
    # Early exit if empty
    if gm == 0:
        return
    
    # ============================================================================
    # Pointer Arithmetic for 3D Layout
    # ============================================================================
    # Shift base pointers to the start of this expert's slice
    cur_lhs_ptr = lhs_ptr + expert_idx * stride_lhs_e
    cur_lhs_scale_ptr = lhs_scale_ptr + expert_idx * stride_lhs_scale_e
    cur_output_ptr = output_ptr + expert_idx * stride_output_e

    # Load weight pointers (list of pointers, indexed by expert_idx)
    gate_base_ptr = tl.load(gate_ptrs_ptr + expert_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    up_base_ptr = tl.load(up_ptrs_ptr + expert_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + expert_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + expert_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # ============================================================================
    # TMA Descriptors
    # ============================================================================
    gate_desc = tl.make_tensor_descriptor(
        gate_base_ptr,
        shape=[N, K],
        strides=[stride_gate_n, stride_gate_k],
        block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K],
    )
    
    up_desc = tl.make_tensor_descriptor(
        up_base_ptr,
        shape=[N, K],
        strides=[stride_up_n, stride_up_k],
        block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K],
    )
    
    offsets_n_base = n_pid * GEMM_BLOCK_SIZE_N
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
    # Iterate over M-blocks (tokens) for this expert
    num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
    for sub_group_idx in range(num_sub_groups):
        # Start index relative to the expert's slice (0-based)
        sub_group_start_idx = sub_group_idx * GEMM_BLOCK_SIZE_M
        
        valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, gm - sub_group_start_idx)
        
        # Indices for loading/masking
        # M_max is the allocated size of dim 1. We must guard against it.
        rel_row_indices = sub_group_start_idx + offsets_m
        
        m_mask = rel_row_indices < M_max 
        valid_mask = offsets_m < valid_rows_this_block
        m_base_mask = m_mask & valid_mask
        
        gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        # K-loop
        num_k_blocks = tl.cdiv(K, GEMM_BLOCK_SIZE_K)
        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * GEMM_BLOCK_SIZE_K
            offsets_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)
            k_mask = offsets_k < K
            
            scale_k_idx = k_block_idx
            scale_offset = scale_n_idx * num_scale_k + scale_k_idx
            
            gate_scale = tl.load(gate_scale_base_ptr + scale_offset)
            up_scale = tl.load(up_scale_base_ptr + scale_offset)
            
            # Load LHS Scale (3D addressing)
            # Ptr = Base(Expert) + Row * stride_m + K_blk * stride_k
            lhs_scale_ptrs = cur_lhs_scale_ptr + (rel_row_indices[:, None] * stride_lhs_scale_m + 
                                                  scale_k_idx * stride_lhs_scale_k)
            lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
            # Load LHS Data (3D addressing)
            lhs_ptrs = cur_lhs_ptr + (rel_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            lhs_mask = m_base_mask[:, None] & k_mask[None, :]
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            
            gate_fp8 = gate_desc.load([offsets_n_base, k_start])
            up_fp8 = up_desc.load([offsets_n_base, k_start])
            
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
        gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
        output_acc = gate_activated * up_acc
        output = output_acc.to(tl.bfloat16)
        
        # Store results (3D addressing)
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
        output_ptrs = cur_output_ptr + (offs_output_m[:, None] * stride_output_m + 
                                        offs_output_n[None, :] * stride_output_n)
        
        output_mask = (offs_output_m[:, None] < M_max) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
        tl.store(output_ptrs, output, mask=output_mask)


@triton.jit
def fused_dequant_grouped_gemm_fp8_tma_3d_kernel(
    lhs_ptr, lhs_scale_ptr,
    rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
    expert_tokens_ptr,   # <--- Counts
    num_experts_total,
    output_ptr,
    M_max, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_e, stride_lhs_m, stride_lhs_k,             # 3D Strides
    stride_lhs_scale_e, stride_lhs_scale_m, stride_lhs_scale_k,
    stride_rhs_n, stride_rhs_k,
    stride_output_e, stride_output_m, stride_output_n,    # 3D Strides
    stride_rhs_ptrs, stride_rhs_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    Stage 2 Kernel (Down Proj) for 3D Input (E, T, H).
    """
    lhs_dtype = tl.bfloat16
    rhs_dtype = tl.float8e4nv
    
    pid = tl.program_id(axis=0) # N-block ID
    
    scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    
    # Loop over local experts
    for g in range(num_experts_total):
        
        gm = tl.load(expert_tokens_ptr + g).to(tl.int32)
        
        # Handle non-empty experts
        if gm > 0:
            # Shift pointers for this expert
            cur_lhs_ptr = lhs_ptr + g * stride_lhs_e
            cur_lhs_scale_ptr = lhs_scale_ptr + g * stride_lhs_scale_e
            cur_output_ptr = output_ptr + g * stride_output_e

            # Load RHS pointers
            rhs_base_ptr = tl.load(rhs_ptrs_ptr + g * stride_rhs_ptrs).to(tl.pointer_type(rhs_dtype))
            rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + g * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))
            
            rhs_desc = tl.make_tensor_descriptor(
                rhs_base_ptr,
                shape=[N, K],
                strides=[stride_rhs_n, stride_rhs_k],
                block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K],
            )
            
            num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
            
            for sub_group_idx in range(num_sub_groups):
                sub_group_start_idx = sub_group_idx * GEMM_BLOCK_SIZE_M
                valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, gm - sub_group_start_idx)
                
                offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
                rel_row_indices = sub_group_start_idx + offsets_m
                
                acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
                
                for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
                    offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
                    
                    # Load LHS (3D)
                    lhs_ptrs = cur_lhs_ptr + (rel_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
                    lhs_mask = (rel_row_indices[:, None] < M_max) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
                    lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
                    
                    # Load RHS (TMA)
                    offs_k = k_idx * GEMM_BLOCK_SIZE_K
                    rhs_fp8 = rhs_desc.load([pid * GEMM_BLOCK_SIZE_N, offs_k])
                    
                    # Load Scales
                    lhs_scale_k = k_idx * GEMM_BLOCK_SIZE_K // 128
                    l_scale_ptr = cur_lhs_scale_ptr + (rel_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
                    lhs_scale = tl.load(l_scale_ptr, mask=(rel_row_indices[:, None] < M_max), other=1.0, cache_modifier='.cg')
                    
                    scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
                    scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
                    rhs_scale = tl.load(scale_ptr)
                    
                    acc += tl.dot(lhs, tl.trans(rhs_fp8)) * lhs_scale * rhs_scale
                
                # Store Result (3D)
                offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
                offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
                
                output_ptrs = cur_output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
                output_mask = (offs_output_m[:, None] < M_max) & (offs_output_n[None, :] < N) & (offsets_m[:, None] < valid_rows_this_block)
                
                output = tl.cast(acc, lhs_dtype)
                tl.store(output_ptrs, output, mask=output_mask)



# -----------------------------------------------------------------------------
# Helper Wrappers
# -----------------------------------------------------------------------------

def fused_fp8_moe_stage_1_tma_wrapper(
    hidden_states, hidden_states_scale,
    gate_weight_list, gate_ptrs_ptr,
    up_weight_list, up_ptrs_ptr,
    gate_scale_list, gate_scale_ptrs_ptr,
    up_scale_list, up_scale_ptrs_ptr,
    expert_token_counts,    
    num_local_experts, 
    gate_gemm_block_size=[64, 32, 128],
    scale_block_size=128,
    num_stages=3,
    num_warps=4,
    out=None
):
    device = hidden_states.device
    # Dimensions: (Experts, TokensPerExpert, Hidden)
    E_dim = hidden_states.shape[0]
    M_max = hidden_states.shape[1] 
    K = hidden_states.shape[2]
    N = gate_weight_list[0].shape[0]
    
    if out is None:
        output = torch.empty((E_dim, M_max, N), dtype=torch.bfloat16, device=device)
    else:
        output = out

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    grid = (num_local_experts, triton.cdiv(N, gate_gemm_block_size[1]))
    
    fused_fp8_moe_parallel_experts_kernel_tma_3d[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        expert_token_counts,
        output,
        M_max, N, K,
        # Pass 3D strides
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1), hidden_states_scale.stride(2),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        # Metadata strides
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

def fused_dequant_grouped_gemm_fp8_tma_wrapper(
    lhs, lhs_scale,
    rhs_list, rhs_ptrs_ptr,
    rhs_scale_list, rhs_scale_ptrs_ptr,
    expert_token_counts,     
    num_local_experts,   
    gemm_block_size=(64, 32, 128),
    scale_block_size=(128, 128),
    num_stages=3,
    num_warps=4,
    out=None
):
    device = lhs.device
    E_dim = lhs.shape[0]
    M_max = lhs.shape[1]
    K = lhs.shape[2]
    N = rhs_list[0].shape[0]
    
    if out is None:
        output = torch.zeros((E_dim, M_max, N), dtype=torch.bfloat16, device=device)
    else:
        output = out
    
    grid = (triton.cdiv(N, gemm_block_size[1]),)
    
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    fused_dequant_grouped_gemm_fp8_tma_3d_kernel[grid](
        lhs, lhs_scale,
        rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
        expert_token_counts,    
        num_local_experts, 
        output,
        M_max, N, K,
        # Pass 3D strides
        lhs.stride(0), lhs.stride(1), lhs.stride(2),
        lhs_scale.stride(0), lhs_scale.stride(1), lhs_scale.stride(2),
        rhs_list[0].stride(0), rhs_list[0].stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        # Metadata
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