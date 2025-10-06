import torch
import triton
import triton.language as tl
import logging

# ==================== STRATEGY 1: PERSISTENT KERNEL WITH 2D GRID ====================
@triton.jit
def fused_fp8_moe_persistent_kernel(
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
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    OPTIMIZED PERSISTENT KERNEL for FP8 MoE.
    
    Key optimizations:
    1. 2D grid: parallelize over (group, N_block) instead of just N
    2. Larger N tiles (128 vs 16) for better tensor core utilization
    3. Preload group metadata to reduce memory traffic
    4. Minimize branching in hot loops
    5. Better memory access patterns
    """
    # 2D grid: pid_group handles one group, pid_n handles one N-block
    pid_group = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Load metadata (do this once per program)
    num_groups = tl.load(num_active_experts_ptr)
    
    # Early exit if this program has no work
    if pid_group >= num_groups:
        return
    
    # Load group info
    gm = tl.load(group_sizes_ptr + pid_group * stride_group_sizes)
    
    # Early exit if group is empty
    if gm == 0:
        return
    
    group_idx = tl.load(group_idx_ptr + pid_group * stride_group_idx)
    start_idx = tl.load(group_start_indices_ptr + pid_group * stride_group_start_indices)
    
    # Load expert weight pointers
    gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
    gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
    
    # N-block offsets for this program
    offsets_n = pid_n * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    mask_n = offsets_n < N
    scale_n_idx = pid_n * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    
    # Process all M-blocks in this group
    num_m_blocks = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    
    for m_block_idx in range(num_m_blocks):
        m_start = start_idx + m_block_idx * GEMM_BLOCK_SIZE_M
        valid_rows = tl.minimum(GEMM_BLOCK_SIZE_M, start_idx + gm - m_start)
        
        offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
        abs_row_indices = m_start + offsets_m
        mask_m = (offsets_m < valid_rows) & (abs_row_indices < M)
        
        # Accumulators
        gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        # K-loop: iterate over quantization blocks directly
        num_k_blocks = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
        
        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * SCALE_BLOCK_SIZE_K
            offsets_k = k_start + tl.arange(0, SCALE_BLOCK_SIZE_K)
            mask_k = offsets_k < K
            
            # Load scales (one per quantization block)
            gate_scale_ptr = gate_scale_base_ptr + scale_n_idx * num_scale_k + k_block_idx
            up_scale_ptr = up_scale_base_ptr + scale_n_idx * num_scale_k + k_block_idx
            gate_scale = tl.load(gate_scale_ptr)
            up_scale = tl.load(up_scale_ptr)
            
            lhs_scale_ptrs = lhs_scale_ptr + abs_row_indices * stride_lhs_scale_m + k_block_idx * stride_lhs_scale_k
            lhs_scale = tl.load(lhs_scale_ptrs[:, None], mask=mask_m[:, None], other=1.0)
            
            # Load data
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
            up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
            lhs_mask = mask_m[:, None] & mask_k[None, :]
            rhs_mask = mask_n[:, None] & mask_k[None, :]
            
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
            up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
            # Compute with scales
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * (lhs_scale * gate_scale)
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * (lhs_scale * up_scale)
        
        # Apply SiLU and store
        output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
        output = output_acc.to(tl.bfloat16)
        
        output_ptrs = output_ptr + (abs_row_indices[:, None] * stride_output_m + offsets_n[None, :] * stride_output_n)
        output_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(output_ptrs, output, mask=output_mask)


# ==================== STRATEGY 3: BATCHED SMALL-M OPTIMIZATION ====================
@triton.jit
def fused_fp8_moe_batched_small_m_kernel(
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
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    BATCHED APPROACH for very small M (<32).
    
    Process multiple experts in parallel within one thread block.
    Key insight: With M<32, we can fit multiple M-blocks in registers.
    """
    pid_n = tl.program_id(axis=0)
    
    offsets_n = pid_n * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    mask_n = offsets_n < N
    
    num_groups = tl.load(num_active_experts_ptr)
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    scale_n_idx = pid_n * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_K
    
    # Process all groups (they're small, so this is fine)
    for g in range(num_groups):
        gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
        
        # Use masking instead of continue
        group_valid = gm > 0
        
        # Load group metadata (even if invalid, will be masked out)
        group_idx = tl.load(group_idx_ptr + g * stride_group_idx)
        start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)
        
        # Load expert pointers (will not be used if group_valid is False)
        gate_base_ptr = tl.load(gate_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
        up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(tl.pointer_type(tl.float8e4nv))
        gate_scale_base_ptr = tl.load(gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
        up_scale_base_ptr = tl.load(up_scale_ptrs_ptr + group_idx * stride_scale_ptrs).to(tl.pointer_type(tl.float32))
        
        # Process entire group at once (M is small)
        offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)
        abs_row_indices = start_idx + offsets_m
        mask_m = (offsets_m < gm) & (abs_row_indices < M) & group_valid
        
        gate_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        up_acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)
        
        # K-loop (only executes if group_valid)
        num_k_blocks = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
        
        for k_idx in range(num_k_blocks):
            k_start = k_idx * SCALE_BLOCK_SIZE_K
            offsets_k = k_start + tl.arange(0, SCALE_BLOCK_SIZE_K)
            mask_k = offsets_k < K
            
            # Scales
            gate_scale_ptr = gate_scale_base_ptr + scale_n_idx * num_k_blocks + k_idx
            up_scale_ptr = up_scale_base_ptr + scale_n_idx * num_k_blocks + k_idx
            gate_scale = tl.load(gate_scale_ptr)
            up_scale = tl.load(up_scale_ptr)
            
            lhs_scale_ptrs = lhs_scale_ptr + abs_row_indices * stride_lhs_scale_m + k_idx * stride_lhs_scale_k
            lhs_scale = tl.load(lhs_scale_ptrs[:, None], mask=mask_m[:, None], other=1.0)
            
            # Data
            lhs_ptrs = lhs_ptr + (abs_row_indices[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
            gate_ptrs = gate_base_ptr + (offsets_n[:, None] * stride_gate_n + offsets_k[None, :] * stride_gate_k)
            up_ptrs = up_base_ptr + (offsets_n[:, None] * stride_up_n + offsets_k[None, :] * stride_up_k)
            
            lhs_mask = mask_m[:, None] & mask_k[None, :]
            rhs_mask = mask_n[:, None] & mask_k[None, :]
            
            lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
            gate_fp8 = tl.load(gate_ptrs, mask=rhs_mask, other=0.0)
            up_fp8 = tl.load(up_ptrs, mask=rhs_mask, other=0.0)
            
            gate_acc += tl.dot(lhs, tl.trans(gate_fp8), out_dtype=tl.float32) * (lhs_scale * gate_scale)
            up_acc += tl.dot(lhs, tl.trans(up_fp8), out_dtype=tl.float32) * (lhs_scale * up_scale)
        
        # Store (only if group_valid)
        output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
        output = output_acc.to(tl.bfloat16)
        
        output_ptrs = output_ptr + (abs_row_indices[:, None] * stride_output_m + offsets_n[None, :] * stride_output_n)
        output_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(output_ptrs, output, mask=output_mask)


# ==================== PYTHON WRAPPERS ====================

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
):
    """
    Optimized FP8 MoE with automatic strategy selection.
    """
    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    
    # Calculate total work and average group size
    total_tokens = group_sizes.sum().item()
    num_groups = num_active_experts.item()
    avg_group_size = total_tokens / max(num_groups, 1) if num_groups > 0 else 0
    
    # Strategy selection based on problem characteristics
    if avg_group_size <= 8:
        # STRATEGY 3: Batched approach for very small groups
        BLOCK_M, BLOCK_N = 32, 128
        num_warps, num_stages = 4, 3
        
        grid = (triton.cdiv(N, BLOCK_N),)
        
        fused_fp8_moe_batched_small_m_kernel[grid](
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
            GEMM_BLOCK_SIZE_M=BLOCK_M,
            GEMM_BLOCK_SIZE_N=BLOCK_N,
            SCALE_BLOCK_SIZE_K=128,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    
    else:
        # STRATEGY 1: Persistent kernel with 2D grid
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 128, 128
        num_warps, num_stages = 4, 4
        
        # 2D grid: (num_groups, num_n_blocks)
        grid = (num_groups, triton.cdiv(N, BLOCK_N))
        
        fused_fp8_moe_persistent_kernel[grid](
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
            GEMM_BLOCK_SIZE_M=BLOCK_M,
            GEMM_BLOCK_SIZE_N=BLOCK_N,
            GEMM_BLOCK_SIZE_K=BLOCK_K,
            SCALE_BLOCK_SIZE_N=128,
            SCALE_BLOCK_SIZE_K=128,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    
    return output