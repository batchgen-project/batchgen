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
def fused_fp8_moe_parallel_experts_kernel_tma_offset(
    lhs_ptr, lhs_scale_ptr,
    gate_ptrs_ptr, up_ptrs_ptr,
    gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
    expert_offsets_ptr,  # <--- Changed from group metadata arrays
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_gate_n, stride_gate_k,
    stride_up_n, stride_up_k,
    stride_output_m, stride_output_n,
    stride_weight_ptrs, stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    Adapted Stage 1 Kernel:
    - Uses expert_offsets to locate data in the continuous input tensor.
    - Grid axis 0 represents the physical local expert index [0, experts_per_rank).
    """
    # 2D program IDs
    expert_idx = tl.program_id(axis=0)  # Physical expert index
    n_pid = tl.program_id(axis=1)       # Which N-block
    
    # 1. Load offsets to determine where this expert's data lives
    # expert_offsets is expected to be size [experts_per_rank + 1]
    start_idx = tl.load(expert_offsets_ptr + expert_idx).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + expert_idx + 1).to(tl.int32)
    
    gm = end_idx - start_idx # Number of tokens for this expert
    
    # If this expert has no tokens, exit immediately
    if gm == 0:
        return
    
    # Load THIS expert's weight pointers (assuming weights are ordered by expert_idx)
    gate_base_ptr = tl.load(gate_ptrs_ptr + expert_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    up_base_ptr = tl.load(up_ptrs_ptr + expert_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + expert_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + expert_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # ============================================================================
    # TMA OPTIMIZATION: Create descriptors for gate and up weights
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
    # ============================================================================
    
    # N-block offsets (fixed for this program)
    offsets_n_base = n_pid * GEMM_BLOCK_SIZE_N
    
    # Scale N index (fixed for this program)
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    scale_n_idx = n_pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
    # M-dimension offsets
    offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
    
    # Process all M-blocks for THIS expert
    num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    
    for sub_group_idx in range(num_sub_groups):
        # Calculate absolute row index in the global input tensor
        # start_idx comes from offsets, sub_group adds block offset
        sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
        
        # Calculate valid rows
        current_block_start_rel = sub_group_idx * GEMM_BLOCK_SIZE_M
        valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, gm - current_block_start_rel)
        
        abs_row_indices = sub_group_start_idx + offsets_m
        
        # M-dimension masks
        # Note: abs_row_indices checks against global M (total tokens) to be safe,
        # but valid_rows_this_block ensures we don't read into the next expert's data.
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
            
            # Load LHS scale (vector load)
            lhs_scale_ptrs = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + 
                                              scale_k_idx * stride_lhs_scale_k)
            lhs_scale = tl.load(lhs_scale_ptrs, mask=m_mask[:, None], other=1.0)
            
            # Load LHS data (activations)
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            lhs_mask = m_base_mask[:, None] & k_mask[None, :]
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            
            # TMA Load Weights
            gate_fp8 = gate_desc.load([offsets_n_base, k_start])
            up_fp8 = up_desc.load([offsets_n_base, k_start])
            
            # Fused multiply-add
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * lhs_scale * gate_scale
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * lhs_scale * up_scale
        
        # SiLU
        gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
        output_acc = gate_activated * up_acc
        output = output_acc.to(tl.bfloat16)
        
        # Store results - Direct write to Expert Y location
        # We use the same offsets_output_m logic as input to maintain layout
        offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
        offs_output_n = n_pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
        
        output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + 
                                    offs_output_n[None, :] * stride_output_n)
        
        # Mask ensures we don't write out of bounds
        output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & valid_mask[:, None]
        
        tl.store(output_ptrs, output, mask=output_mask)


@triton.jit
def fused_dequant_grouped_gemm_fp8_tma_offset_kernel(
    lhs_ptr, lhs_scale_ptr,
    rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
    expert_offsets_ptr, # <--- Changed
    num_experts_total,  # <--- Changed: Pass total experts (e.g., 8, 64) to loop over
    output_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_lhs_m, stride_lhs_k,
    stride_lhs_scale_m, stride_lhs_scale_k,
    stride_rhs_n, stride_rhs_k,
    stride_output_m, stride_output_n,
    stride_rhs_ptrs, stride_rhs_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    Adapted Stage 2 Kernel:
    - Loops over [0, num_experts_total) locally.
    - Uses offsets to find data chunks.
    """
    lhs_dtype = tl.bfloat16
    rhs_dtype = tl.float8e4nv
    
    pid = tl.program_id(axis=0)
    
    # Pre-calculate scale indices
    scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    
    # Iterate over ALL local experts
    # Since we don't have a compacted list of active experts, we check size > 0 inside
    for g in range(num_experts_total):
        
        # 1. Load offsets
        start_idx = tl.load(expert_offsets_ptr + g).to(tl.int32)
        end_idx = tl.load(expert_offsets_ptr + g + 1).to(tl.int32)
        gm = end_idx - start_idx
        
        # REVISION: Removed 'continue' for Triton compatibility
        if gm > 0:
            num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
            
            # Get RHS pointers for this expert (g is the index)
            rhs_base_ptr = tl.load(rhs_ptrs_ptr + g * stride_rhs_ptrs).to(tl.pointer_type(rhs_dtype))
            rhs_scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + g * stride_rhs_scale_ptrs).to(tl.pointer_type(tl.float32))
            
            # Create TMA descriptor for RHS
            rhs_desc = tl.make_tensor_descriptor(
                rhs_base_ptr,
                shape=[N, K],
                strides=[stride_rhs_n, stride_rhs_k],
                block_shape=[GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K],
            )
            
            # Process all M-tiles for this expert
            for sub_group_idx in range(num_sub_groups):
                # Absolute index calculation using offsets
                sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
                current_block_start_rel = sub_group_idx * GEMM_BLOCK_SIZE_M
                valid_rows_this_block = tl.minimum(GEMM_BLOCK_SIZE_M, gm - current_block_start_rel)
                
                offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
                abs_row_indices = sub_group_start_idx + offsets_m
                
                # Accumulator
                acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
                
                # K-dimension loop
                for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
                    offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(0, GEMM_BLOCK_SIZE_K)
                    
                    # Load LHS (standard)
                    lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
                    lhs_mask = (abs_row_indices[:, None] < M) & (offsets_k[None, :] < K) & (offsets_m[:, None] < valid_rows_this_block)
                    lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
                    
                    # Load RHS with TMA
                    offs_k = k_idx * GEMM_BLOCK_SIZE_K
                    rhs_fp8 = rhs_desc.load([pid * GEMM_BLOCK_SIZE_N, offs_k])
                    
                    # Load LHS scale
                    lhs_scale_k = k_idx * GEMM_BLOCK_SIZE_K // 128
                    l_scale_ptr = lhs_scale_ptr + (abs_row_indices[:, None] * stride_lhs_scale_m + lhs_scale_k * stride_lhs_scale_k)
                    lhs_scale = tl.load(l_scale_ptr, mask=(abs_row_indices[:, None] < M), other=1.0, cache_modifier='.cg')
                    
                    # Load RHS scale
                    scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
                    scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
                    rhs_scale = tl.load(scale_ptr)
                    
                    # Fused dequantization and matmul
                    acc += tl.dot(lhs, tl.trans(rhs_fp8)) * lhs_scale * rhs_scale
                
                # Store result
                offs_output_m = sub_group_start_idx + tl.arange(0, GEMM_BLOCK_SIZE_M)
                offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
                output_ptrs = output_ptr + (offs_output_m[:, None] * stride_output_m + offs_output_n[None, :] * stride_output_n)
                
                output_mask = (offs_output_m[:, None] < M) & (offs_output_n[None, :] < N) & (offsets_m[:, None] < valid_rows_this_block)
                
                output = tl.cast(acc, lhs_dtype)
                tl.store(output_ptrs, output, mask=output_mask)


# -----------------------------------------------------------------------------
# Python Wrapper
# -----------------------------------------------------------------------------

def grouped_dequant_moe_fp8(
    self, 
    x,                  
    expert_offsets,     
    experts_per_rank,
    out=None            # <--- ADDED: Output buffer argument
):
    """
    Optimized wrapper that writes directly to 'out' (self.expert_y) using offsets.
    """
    
    actual_num_tokens = expert_offsets[-1].item()
    
    # Dynamic slicing (views, no allocation)
    x_valid = x[:actual_num_tokens]
    x_quant, x_scale = self.act_quant(x_valid) 
    
    # Stage 1: Allocate intermediate (or use a buffer if you add one to __init__)
    # Note: Intermediate size is [tokens, intermediate_size], usually larger than hidden
    intermediate = fused_fp8_moe_stage_1_tma_wrapper(
        x_quant, x_scale, 
        self.gate_list, self.gate_ptrs_ptr,
        self.up_list, self.up_ptrs_ptr,
        self.gate_scale_list, self.gate_scale_ptrs_ptr,
        self.up_scale_list, self.up_scale_ptrs_ptr,
        expert_offsets,     
        experts_per_rank    
    )
    
    intermediate_quant, intermediate_scale = self.act_quant(intermediate)
    
    # Stage 2: Write directly to 'out' (self.expert_y)
    res = fused_dequant_grouped_gemm_fp8_tma_wrapper(
        intermediate_quant, intermediate_scale, 
        self.down_list, self.down_ptrs_ptr,
        self.down_scale_list, self.down_scale_ptrs_ptr,
        expert_offsets,     
        experts_per_rank,
        out=out             # <--- Pass to wrapper
    )
    
    return res


# -----------------------------------------------------------------------------
# Helper Wrappers
# -----------------------------------------------------------------------------

def fused_fp8_moe_stage_1_tma_wrapper(
    hidden_states, hidden_states_scale,
    gate_weight_list, gate_ptrs_ptr,
    up_weight_list, up_ptrs_ptr,
    gate_scale_list, gate_scale_ptrs_ptr,
    up_scale_list, up_scale_ptrs_ptr,
    expert_offsets,    
    num_local_experts, 
    gate_gemm_block_size=[64, 32, 128],
    scale_block_size=128,
    num_stages=3,
    num_warps=4,
    out=None           # <--- Added arg (future proofing)
):
    device = hidden_states.device
    M = hidden_states.shape[0] 
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    if out is None:
        output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    else:
        output = out[:M, :N] # View into pre-allocated buffer

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    grid = (num_local_experts, triton.cdiv(N, gate_gemm_block_size[1]))
    
    fused_fp8_moe_parallel_experts_kernel_tma_offset[grid](
        hidden_states, hidden_states_scale,
        gate_ptrs_ptr, up_ptrs_ptr,
        gate_scale_ptrs_ptr, up_scale_ptrs_ptr,
        expert_offsets,
        output,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gate_weight_list[0].stride(0), gate_weight_list[0].stride(1),
        up_weight_list[0].stride(0), up_weight_list[0].stride(1),
        output.stride(0), output.stride(1),
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
    expert_offsets,     
    num_local_experts,   
    gemm_block_size=(64, 32, 128),
    scale_block_size=(128, 128),
    num_stages=3,
    num_warps=4,
    out=None            # <--- Added arg
):
    device = lhs.device
    M = lhs.shape[0]
    N = rhs_list[0].shape[0]
    K = lhs.shape[1]
    
    if out is None:
        # Default behavior (alloc new)
        output = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    else:
        # Use pre-allocated buffer
        # IMPORTANT: Slice it to match current batch size 'M' so kernel guards work
        output = out[:M, :N]

        # We don't need to zero 'out' here because the kernel writes strictly 
        # to the rows defined by 'expert_offsets', effectively overwriting valid data.
        # Garbage data in invalid rows (beyond M) is ignored by the Combine kernel.
    
    grid = (triton.cdiv(N, gemm_block_size[1]),)
    
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    fused_dequant_grouped_gemm_fp8_tma_offset_kernel[grid](
        lhs, lhs_scale,
        rhs_ptrs_ptr, rhs_scale_ptrs_ptr,
        expert_offsets,    
        num_local_experts, 
        output,
        M, N, K,
        lhs.stride(0), lhs.stride(1),
        lhs_scale.stride(0), lhs_scale.stride(1),
        rhs_list[0].stride(0), rhs_list[0].stride(1),
        output.stride(0), output.stride(1),
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