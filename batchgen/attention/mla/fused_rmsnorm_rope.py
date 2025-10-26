import triton
import triton.language as tl
import torch
@triton.jit
def fused_rmsnorm_rope_kernel(
    data_ptr,
    cos_ptr,
    sin_ptr,
    position_ids_ptr,
    norm_weight_ptr,
    bsz,
    q_len,
    kv_lora_rank,
    qk_rope_head_dim,
    max_seq_len,
    variance_epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies:
    1. RMSNorm to the first kv_lora_rank dimensions
    2. Rotary position embedding (with rotation) to the last qk_rope_head_dim dimensions
    """
    pid = tl.program_id(0)
    
    batch_idx = pid // q_len
    seq_idx = pid % q_len
    
    total_dim = kv_lora_rank + qk_rope_head_dim
    base_offset = batch_idx * q_len * total_dim + seq_idx * total_dim
    
    # ==================== Part 1: RMSNorm with FP32 Accumulation ====================
    
    sum_sq_fp32 = 0.0
    
    # Accumulate sum of squares in fp32
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        
        data_bf16 = tl.load(data_ptr + base_offset + offsets, mask=mask, other=0.0)
        data_fp32 = data_bf16.to(tl.float32)
        sum_sq_fp32 += tl.sum(data_fp32 * data_fp32, axis=0)
    
    variance = sum_sq_fp32 / kv_lora_rank
    inv_rms = 1.0 / tl.sqrt(variance + variance_epsilon)
    
    # Apply normalization and write back
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        
        data_bf16 = tl.load(data_ptr + base_offset + offsets, mask=mask, other=0.0)
        weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=1.0)
        
        data_fp32 = data_bf16.to(tl.float32)
        normalized = data_fp32 * inv_rms * weight
        
        tl.store(data_ptr + base_offset + offsets, normalized.to(data_bf16.dtype), mask=mask)
    
    # ==================== Part 2: RoPE with Rotation ====================
    
    pos_id = tl.load(position_ids_ptr + batch_idx * q_len + seq_idx)
    
    rope_offset = base_offset + kv_lora_rank
    half_dim = qk_rope_head_dim // 2
    
    # The rotation transforms: [x0, x1, x2, x3, x4, x5, x6, x7] → [x0, x2, x4, x6, x1, x3, x5, x7]
    # We need to:
    # 1. Load from original positions (even/odd indices)
    # 2. Apply RoPE as if data is in rotated layout
    # 3. Store to rotated positions (first half, then second half)
    
    for block_start in range(0, half_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim
        
        # Step 1: Load from ORIGINAL layout
        # Even indices (x0, x2, x4, x6) will become first half of rotated layout
        even_indices = offsets * 2
        x_even = tl.load(data_ptr + rope_offset + even_indices, mask=mask, other=0.0)
        
        # Odd indices (x1, x3, x5, x7) will become second half of rotated layout
        odd_indices = offsets * 2 + 1
        x_odd = tl.load(data_ptr + rope_offset + odd_indices, mask=mask, other=0.0)
        
        # Step 2: Load cos/sin for ROTATED positions
        # Position 'offsets' in rotated layout (first half)
        # Position 'half_dim + offsets' in rotated layout (second half)
        cos_offset = pos_id * qk_rope_head_dim
        cos_first = tl.load(cos_ptr + cos_offset + offsets, mask=mask, other=1.0)
        sin_first = tl.load(sin_ptr + cos_offset + offsets, mask=mask, other=0.0)
        cos_second = tl.load(cos_ptr + cos_offset + half_dim + offsets, mask=mask, other=1.0)
        sin_second = tl.load(sin_ptr + cos_offset + half_dim + offsets, mask=mask, other=0.0)
        
        # Apply RoPE: out = x * cos + rotate_half(x) * sin
        # rotate_half swaps first and second half, negating the second half
        # For first half: out = x_even * cos + (-x_odd) * sin
        # For second half: out = x_odd * cos + x_even * sin
        out_first = x_even * cos_first + (-x_odd) * sin_first
        out_second = x_odd * cos_second + x_even * sin_second
        
        # CRITICAL: Store in ROTATED layout (not original layout)
        # The reference implementation returns data in rotated layout
        # Rotated position i gets out_first
        # Rotated position half_dim+i gets out_second
        tl.store(data_ptr + rope_offset + offsets, out_first, mask=mask)
        tl.store(data_ptr + rope_offset + half_dim + offsets, out_second, mask=mask)


