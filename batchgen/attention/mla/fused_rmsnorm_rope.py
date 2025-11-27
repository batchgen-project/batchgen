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


def fused_rmsnorm_rope(
    data: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    norm_weight: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Wrapper for fused kernel"""
    bsz, q_len, _ = data.shape
    max_seq_len = cos.shape[0]
    
    # Launch kernel
    grid = lambda meta: (bsz * q_len,)
    fused_rmsnorm_rope_kernel[grid](
        data, cos, sin, position_ids, norm_weight,
        bsz, q_len, kv_lora_rank, qk_rope_head_dim, max_seq_len,
        eps,
        BLOCK_SIZE=64,
    )
    
    return data



@triton.jit
def fused_rmsnorm_rope_cache_update_kernel(
    new_compressed_kv_ptr,     # [bsz, 1, total_dim] - input
    past_key_states_ptr,       # [bsz, max_seq_len, total_dim] - output (modified in-place)
    cos_ptr,                   # [max_seq_len, qk_rope_head_dim]
    sin_ptr,                   # [max_seq_len, qk_rope_head_dim]
    position_ids_ptr,          # [bsz, 1] - where to write in cache
    norm_weight_ptr,           # [kv_lora_rank]
    bsz,
    max_seq_len,
    kv_lora_rank,
    qk_rope_head_dim,
    variance_epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for decoding with cache update.
    Each program handles one batch element (since q_len = 1).
    
    Reads from:  new_compressed_kv[batch_idx, 0, :]
    Writes to:   past_key_states[batch_idx, position_ids[batch_idx, 0], :]
    """
    batch_idx = tl.program_id(0)
    
    total_dim = kv_lora_rank + qk_rope_head_dim
    
    # Input: new_compressed_kv[batch_idx, 0, :]
    # Shape is [bsz, 1, total_dim], so stride is [1*total_dim, total_dim, 1]
    input_offset = batch_idx * total_dim
    
    # Get the position to write to in the cache
    pos_id = tl.load(position_ids_ptr + batch_idx)
    
    # Output: past_key_states[batch_idx, pos_id, :]
    # Shape is [bsz, max_seq_len, total_dim], stride is [max_seq_len*total_dim, total_dim, 1]
    output_offset = batch_idx * max_seq_len * total_dim + pos_id * total_dim
    
    # ==================== Part 1: RMSNorm with FP32 Accumulation ====================
    
    sum_sq_fp32 = 0.0
    
    # Accumulate sum of squares
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        
        data_bf16 = tl.load(new_compressed_kv_ptr + input_offset + offsets, mask=mask, other=0.0)
        data_fp32 = data_bf16.to(tl.float32)
        sum_sq_fp32 += tl.sum(data_fp32 * data_fp32)
    
    variance = sum_sq_fp32 / kv_lora_rank
    inv_rms = 1.0 / tl.sqrt(variance + variance_epsilon)
    
    # Apply normalization and write directly to cache
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        
        data_bf16 = tl.load(new_compressed_kv_ptr + input_offset + offsets, mask=mask, other=0.0)
        weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=1.0)
        
        data_fp32 = data_bf16.to(tl.float32)
        weight_fp32 = weight.to(tl.float32)
        normalized_fp32 = data_fp32 * inv_rms * weight_fp32
        
        # Write directly to past_key_states cache
        tl.store(past_key_states_ptr + output_offset + offsets, 
                 normalized_fp32.to(data_bf16.dtype), mask=mask)
    
    # ==================== Part 2: RoPE with Rotation ====================
    
    input_rope_offset = input_offset + kv_lora_rank
    output_rope_offset = output_offset + kv_lora_rank
    half_dim = qk_rope_head_dim // 2
    
    for block_start in range(0, half_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim
        
        # Load from input (even/odd pairs)
        even_indices = offsets * 2
        x_even = tl.load(new_compressed_kv_ptr + input_rope_offset + even_indices, 
                         mask=mask, other=0.0)
        
        odd_indices = offsets * 2 + 1
        x_odd = tl.load(new_compressed_kv_ptr + input_rope_offset + odd_indices, 
                        mask=mask, other=0.0)
        
        # Load cos/sin at position pos_id
        cos_offset_base = pos_id * qk_rope_head_dim
        cos_first = tl.load(cos_ptr + cos_offset_base + offsets, mask=mask, other=1.0)
        sin_first = tl.load(sin_ptr + cos_offset_base + offsets, mask=mask, other=0.0)
        cos_second = tl.load(cos_ptr + cos_offset_base + half_dim + offsets, mask=mask, other=1.0)
        sin_second = tl.load(sin_ptr + cos_offset_base + half_dim + offsets, mask=mask, other=0.0)
        
        # Apply RoPE
        out_first = x_even * cos_first + (-x_odd) * sin_first
        out_second = x_odd * cos_second + x_even * sin_second
        
        # Write directly to cache in rotated layout
        tl.store(past_key_states_ptr + output_rope_offset + offsets, out_first, mask=mask)
        tl.store(past_key_states_ptr + output_rope_offset + half_dim + offsets, out_second, mask=mask)


def fused_rmsnorm_rope_cache_update(
    new_compressed_kv: torch.Tensor,      # [bsz, 1, total_dim]
    past_key_states: torch.Tensor,        # [bsz, max_seq_len, total_dim] - modified in-place
    cos: torch.Tensor,                    # [max_seq_len, qk_rope_head_dim]
    sin: torch.Tensor,                    # [max_seq_len, qk_rope_head_dim]
    position_ids: torch.Tensor,           # [bsz, 1]
    norm_weight: torch.Tensor,            # [kv_lora_rank]
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    eps: float = 1e-6,
) -> None:
    """
    Fused RMSNorm + RoPE with direct cache update.
    Modifies past_key_states in-place.
    
    Args:
        new_compressed_kv: New token's compressed KV [bsz, 1, total_dim]
        past_key_states: KV cache to update [bsz, max_seq_len, total_dim]
        cos: Precomputed cos values [max_seq_len, qk_rope_head_dim]
        sin: Precomputed sin values [max_seq_len, qk_rope_head_dim]
        position_ids: Cache positions to write [bsz, 1]
        norm_weight: RMSNorm weights [kv_lora_rank]
        kv_lora_rank: Size of kv lora rank
        qk_rope_head_dim: Size of rope head dimension
        eps: RMSNorm epsilon
    """
    bsz, q_len, total_dim = new_compressed_kv.shape
    assert q_len == 1, "This kernel assumes q_len = 1 for decoding"
    assert total_dim == kv_lora_rank + qk_rope_head_dim
    assert qk_rope_head_dim % 2 == 0
    assert past_key_states.is_contiguous(), "past_key_states must be contiguous"
    assert position_ids.shape == (bsz, 1), f"Expected position_ids shape {(bsz, 1)}, got {position_ids.shape}"
    
    max_seq_len = past_key_states.shape[1]
    
    # Launch one program per batch element
    grid = (bsz,)
    
    fused_rmsnorm_rope_cache_update_kernel[grid](
        new_compressed_kv,
        past_key_states,
        cos,
        sin,
        position_ids,
        norm_weight,
        bsz,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        eps,
        BLOCK_SIZE=64,
    )



@triton.jit
def fused_rmsnorm_rope_cache_update_with_q_kernel(
    new_compressed_kv_ptr,     # [bsz, 1, total_dim] - input
    past_key_states_ptr,       # [bsz, max_seq_len, total_dim] - output (modified in-place)
    q_pe_ptr,                  # [bsz, num_heads, 1, qk_rope_head_dim] - modified in-place
    cos_ptr,                   # [max_seq_len, qk_rope_head_dim]
    sin_ptr,                   # [max_seq_len, qk_rope_head_dim]
    position_ids_ptr,          # [bsz, 1] - where to write in cache
    norm_weight_ptr,           # [kv_lora_rank]
    bsz,
    num_heads,
    max_seq_len,
    kv_lora_rank,
    qk_rope_head_dim,
    variance_epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel with TRUE cos/sin reuse.
    Loads cos/sin once per block, uses for both KV and all Q heads.
    """
    batch_idx = tl.program_id(0)

    total_dim = kv_lora_rank + qk_rope_head_dim
    input_offset = batch_idx * total_dim
    pos_id = tl.load(position_ids_ptr + batch_idx)
    output_offset = batch_idx * max_seq_len * total_dim + pos_id * total_dim

    # ==================== Part 1: RMSNorm on KV ====================

    sum_sq_fp32 = 0.0

    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank

        data_bf16 = tl.load(
            new_compressed_kv_ptr + input_offset + offsets,
            mask=mask,
            other=0.0,
        )
        data_fp32 = data_bf16.to(tl.float32)
        sum_sq_fp32 += tl.sum(data_fp32 * data_fp32)

    variance = sum_sq_fp32 / kv_lora_rank
    inv_rms = 1.0 / tl.sqrt(variance + variance_epsilon)

    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank

        data_bf16 = tl.load(
            new_compressed_kv_ptr + input_offset + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=1.0)

        data_fp32 = data_bf16.to(tl.float32)
        weight_fp32 = weight.to(tl.float32)
        normalized_fp32 = data_fp32 * inv_rms * weight_fp32

        tl.store(
            past_key_states_ptr + output_offset + offsets,
            normalized_fp32.to(data_bf16.dtype),
            mask=mask,
        )

    # ==================== Part 2 & 3: RoPE on KV and Q with TRUE reuse ====================

    input_rope_offset = input_offset + kv_lora_rank
    output_rope_offset = output_offset + kv_lora_rank
    half_dim = qk_rope_head_dim // 2
    cos_sin_base = pos_id * qk_rope_head_dim

    # q_pe strides
    q_pe_batch_stride = num_heads * qk_rope_head_dim
    q_pe_head_stride = qk_rope_head_dim

    # Process in blocks - load cos/sin ONCE per block, use for KV + all Q heads
    for block_start in range(0, half_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim

        # ===== LOAD COS/SIN ONCE (into registers) =====
        cos_first = tl.load(cos_ptr + cos_sin_base + offsets, mask=mask, other=1.0)
        sin_first = tl.load(sin_ptr + cos_sin_base + offsets, mask=mask, other=0.0)
        cos_second = tl.load(
            cos_ptr + cos_sin_base + half_dim + offsets,
            mask=mask,
            other=1.0,
        )
        sin_second = tl.load(
            sin_ptr + cos_sin_base + half_dim + offsets,
            mask=mask,
            other=0.0,
        )

        # ===== Apply to KV =====
        even_indices = offsets * 2
        kv_even = tl.load(
            new_compressed_kv_ptr + input_rope_offset + even_indices,
            mask=mask,
            other=0.0,
        )

        odd_indices = offsets * 2 + 1
        kv_odd = tl.load(
            new_compressed_kv_ptr + input_rope_offset + odd_indices,
            mask=mask,
            other=0.0,
        )

        kv_out_first = kv_even * cos_first + (-kv_odd) * sin_first
        kv_out_second = kv_odd * cos_second + kv_even * sin_second

        tl.store(past_key_states_ptr + output_rope_offset + offsets, kv_out_first, mask=mask)
        tl.store(
            past_key_states_ptr + output_rope_offset + half_dim + offsets,
            kv_out_second,
            mask=mask,
        )

        # ===== Apply to ALL Q heads using SAME cos/sin (already in registers!) =====
        for head_idx in range(num_heads):
            q_pe_base_offset = batch_idx * q_pe_batch_stride + head_idx * q_pe_head_stride

            # Load Q data
            q_even = tl.load(
                q_pe_ptr + q_pe_base_offset + even_indices,
                mask=mask,
                other=0.0,
            )
            q_odd = tl.load(
                q_pe_ptr + q_pe_base_offset + odd_indices,
                mask=mask,
                other=0.0,
            )

            # Apply RoPE using cos/sin from registers (no memory load!)
            q_out_first = q_even * cos_first + (-q_odd) * sin_first
            q_out_second = q_odd * cos_second + q_even * sin_second

            # Write back
            tl.store(q_pe_ptr + q_pe_base_offset + offsets, q_out_first, mask=mask)
            tl.store(q_pe_ptr + q_pe_base_offset + half_dim + offsets, q_out_second, mask=mask)

def fused_rmsnorm_rope_cache_update_with_q(
    new_compressed_kv: torch.Tensor,      # [bsz, 1, total_dim]
    past_key_states: torch.Tensor,        # [bsz, max_seq_len, total_dim]
    q_pe: torch.Tensor,                   # [bsz, num_heads, 1, qk_rope_head_dim]
    cos: torch.Tensor,                    # [max_seq_len, qk_rope_head_dim]
    sin: torch.Tensor,                    # [max_seq_len, qk_rope_head_dim]
    position_ids: torch.Tensor,           # [bsz, 1]
    norm_weight: torch.Tensor,            # [kv_lora_rank]
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    eps: float = 1e-6,
) -> None:
    """
    Fused RMSNorm + RoPE with TRUE cos/sin reuse.
    Loads cos/sin once per block, applies to KV and all Q heads.
    """
    bsz, q_len, total_dim = new_compressed_kv.shape
    assert q_len == 1, "This kernel assumes q_len = 1 for decoding"
    assert total_dim == kv_lora_rank + qk_rope_head_dim
    assert qk_rope_head_dim % 2 == 0
    assert past_key_states.is_contiguous(), "past_key_states must be contiguous"
    assert q_pe.is_contiguous(), "q_pe must be contiguous"
    assert position_ids.shape == (bsz, 1)
    
    num_heads = q_pe.shape[1]
    max_seq_len = past_key_states.shape[1]
    
    assert q_pe.shape == (bsz, num_heads, 1, qk_rope_head_dim)
    
    grid = (bsz,)
    
    fused_rmsnorm_rope_cache_update_with_q_kernel[grid](
        new_compressed_kv,
        past_key_states,
        q_pe,
        cos,
        sin,
        position_ids,
        norm_weight,
        bsz,
        num_heads,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        eps,
        BLOCK_SIZE=64,
    )

@triton.jit
def fused_rmsnorm_rope_with_q_kernel(
    new_compressed_kv_ptr,
    processed_kv_ptr,
    q_pe_ptr,
    cos_ptr,
    sin_ptr,
    position_ids_ptr,
    norm_weight_ptr,
    bsz,
    num_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    variance_epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Standalone fused RMSNorm + RoPE kernel that omits cache writes."""
    batch_idx = tl.program_id(0)
    total_dim = kv_lora_rank + qk_rope_head_dim
    input_offset = batch_idx * total_dim
    output_offset = batch_idx * total_dim
    pos_id = tl.load(position_ids_ptr + batch_idx)

    # ==================== RMSNorm on KV Lora slice ====================
    sum_sq_fp32 = 0.0
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        data_bf16 = tl.load(
            new_compressed_kv_ptr + input_offset + offsets,
            mask=mask,
            other=0.0,
        )
        data_fp32 = data_bf16.to(tl.float32)
        sum_sq_fp32 += tl.sum(data_fp32 * data_fp32)

    variance = sum_sq_fp32 / kv_lora_rank
    inv_rms = 1.0 / tl.sqrt(variance + variance_epsilon)

    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        data_bf16 = tl.load(
            new_compressed_kv_ptr + input_offset + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=1.0)
        data_fp32 = data_bf16.to(tl.float32)
        normalized_fp32 = data_fp32 * inv_rms * weight.to(tl.float32)
        tl.store(
            processed_kv_ptr + output_offset + offsets,
            normalized_fp32.to(data_bf16.dtype),
            mask=mask,
        )

    # ==================== RoPE for KV + Q ====================
    input_rope_offset = input_offset + kv_lora_rank
    processed_rope_offset = output_offset + kv_lora_rank
    half_dim = qk_rope_head_dim // 2
    cos_sin_base = pos_id * qk_rope_head_dim
    q_pe_batch_stride = num_heads * qk_rope_head_dim
    q_pe_head_stride = qk_rope_head_dim

    for block_start in range(0, half_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim

        cos_first = tl.load(cos_ptr + cos_sin_base + offsets, mask=mask, other=1.0)
        sin_first = tl.load(sin_ptr + cos_sin_base + offsets, mask=mask, other=0.0)
        cos_second = tl.load(
            cos_ptr + cos_sin_base + half_dim + offsets,
            mask=mask,
            other=1.0,
        )
        sin_second = tl.load(
            sin_ptr + cos_sin_base + half_dim + offsets,
            mask=mask,
            other=0.0,
        )

        even_indices = offsets * 2
        odd_indices = offsets * 2 + 1
        kv_even = tl.load(
            new_compressed_kv_ptr + input_rope_offset + even_indices,
            mask=mask,
            other=0.0,
        )
        kv_odd = tl.load(
            new_compressed_kv_ptr + input_rope_offset + odd_indices,
            mask=mask,
            other=0.0,
        )

        kv_out_first = kv_even * cos_first + (-kv_odd) * sin_first
        kv_out_second = kv_odd * cos_second + kv_even * sin_second

        tl.store(
            processed_kv_ptr + processed_rope_offset + offsets,
            kv_out_first,
            mask=mask,
        )
        tl.store(
            processed_kv_ptr + processed_rope_offset + half_dim + offsets,
            kv_out_second,
            mask=mask,
        )

        for head_idx in range(num_heads):
            q_pe_base_offset = batch_idx * q_pe_batch_stride + head_idx * q_pe_head_stride
            q_even = tl.load(
                q_pe_ptr + q_pe_base_offset + even_indices,
                mask=mask,
                other=0.0,
            )
            q_odd = tl.load(
                q_pe_ptr + q_pe_base_offset + odd_indices,
                mask=mask,
                other=0.0,
            )
            q_out_first = q_even * cos_first + (-q_odd) * sin_first
            q_out_second = q_odd * cos_second + q_even * sin_second
            tl.store(
                q_pe_ptr + q_pe_base_offset + offsets,
                q_out_first,
                mask=mask,
            )
            tl.store(
                q_pe_ptr + q_pe_base_offset + half_dim + offsets,
                q_out_second,
                mask=mask,
            )


def fused_rmsnorm_rope_with_q(
    new_compressed_kv: torch.Tensor,
    q_pe: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    norm_weight: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply fused RMSNorm + RoPE to KV and Q without cache updates."""
    bsz, q_len, total_dim = new_compressed_kv.shape
    assert q_len == 1, "This kernel assumes q_len = 1 for decoding"
    assert total_dim == kv_lora_rank + qk_rope_head_dim
    assert qk_rope_head_dim % 2 == 0
    assert q_pe.is_contiguous(), "q_pe must be contiguous"
    assert position_ids.shape == (bsz, 1)

    num_heads = q_pe.shape[1]
    assert q_pe.shape == (bsz, num_heads, 1, qk_rope_head_dim)

    processed_kv = torch.empty_like(new_compressed_kv)
    grid = (bsz,)

    fused_rmsnorm_rope_with_q_kernel[grid](
        new_compressed_kv,
        processed_kv,
        q_pe,
        cos,
        sin,
        position_ids,
        norm_weight,
        bsz,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        eps,
        BLOCK_SIZE=64,
    )

    return processed_kv

@triton.jit
def fused_rmsnorm_rope_cache_update_with_q_return_new_kv_kernel(
    new_compressed_kv_ptr,     # [bsz, 1, total_dim] - input
    past_key_states_ptr,       # [bsz, max_seq_len, total_dim] - output (modified in-place)
    offload_kv_ptr,            # [bsz, 1, total_dim] - output (the processed KV)
    q_pe_ptr,                  # [bsz, num_heads, 1, qk_rope_head_dim] - modified in-place
    cos_ptr,                   # [max_seq_len, qk_rope_head_dim]
    sin_ptr,                   # [max_seq_len, qk_rope_head_dim]
    position_ids_ptr,          # [bsz, 1] - where to write in cache
    norm_weight_ptr,           # [kv_lora_rank]
    bsz,
    num_heads,
    max_seq_len,
    kv_lora_rank,
    qk_rope_head_dim,
    variance_epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel with TRUE cos/sin reuse and offload_kv output.
    Loads cos/sin once per block, uses for both KV and all Q heads.
    Writes processed KV to both past_key_states and offload_kv.
    """
    batch_idx = tl.program_id(0)
    
    total_dim = kv_lora_rank + qk_rope_head_dim
    input_offset = batch_idx * total_dim
    pos_id = tl.load(position_ids_ptr + batch_idx)
    
    # Output offsets
    cache_output_offset = batch_idx * max_seq_len * total_dim + pos_id * total_dim
    offload_output_offset = batch_idx * total_dim  # [bsz, 1, total_dim] with seq_dim=1
    
    # ==================== Part 1: RMSNorm on KV ====================
    
    sum_sq_fp32 = 0.0
    
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        
        data_bf16 = tl.load(new_compressed_kv_ptr + input_offset + offsets, mask=mask, other=0.0)
        data_fp32 = data_bf16.to(tl.float32)
        sum_sq_fp32 += tl.sum(data_fp32 * data_fp32)
    
    variance = sum_sq_fp32 / kv_lora_rank
    inv_rms = 1.0 / tl.sqrt(variance + variance_epsilon)
    
    # Apply normalization and write to BOTH cache and offload_kv
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        
        data_bf16 = tl.load(new_compressed_kv_ptr + input_offset + offsets, mask=mask, other=0.0)
        weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=1.0)
        
        data_fp32 = data_bf16.to(tl.float32)
        weight_fp32 = weight.to(tl.float32)
        normalized_fp32 = data_fp32 * inv_rms * weight_fp32
        normalized_bf16 = normalized_fp32.to(data_bf16.dtype)
        
        # Write to cache
        tl.store(past_key_states_ptr + cache_output_offset + offsets, normalized_bf16, mask=mask)
        # Write to offload_kv
        tl.store(offload_kv_ptr + offload_output_offset + offsets, normalized_bf16, mask=mask)
    
    # ==================== Part 2 & 3: RoPE on KV and Q with TRUE reuse ====================
    
    input_rope_offset = input_offset + kv_lora_rank
    cache_rope_offset = cache_output_offset + kv_lora_rank
    offload_rope_offset = offload_output_offset + kv_lora_rank
    half_dim = qk_rope_head_dim // 2
    cos_sin_base = pos_id * qk_rope_head_dim
    
    # q_pe strides
    q_pe_batch_stride = num_heads * qk_rope_head_dim
    q_pe_head_stride = qk_rope_head_dim
    
    # Process in blocks - load cos/sin ONCE per block, use for KV + all Q heads
    for block_start in range(0, half_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim
        
        # ===== LOAD COS/SIN ONCE (into registers) =====
        cos_first = tl.load(cos_ptr + cos_sin_base + offsets, mask=mask, other=1.0)
        sin_first = tl.load(sin_ptr + cos_sin_base + offsets, mask=mask, other=0.0)
        cos_second = tl.load(cos_ptr + cos_sin_base + half_dim + offsets, mask=mask, other=1.0)
        sin_second = tl.load(sin_ptr + cos_sin_base + half_dim + offsets, mask=mask, other=0.0)
        
        # ===== Apply to KV =====
        even_indices = offsets * 2
        kv_even = tl.load(new_compressed_kv_ptr + input_rope_offset + even_indices, 
                          mask=mask, other=0.0)
        
        odd_indices = offsets * 2 + 1
        kv_odd = tl.load(new_compressed_kv_ptr + input_rope_offset + odd_indices, 
                         mask=mask, other=0.0)
        
        kv_out_first = kv_even * cos_first + (-kv_odd) * sin_first
        kv_out_second = kv_odd * cos_second + kv_even * sin_second
        
        # Write to cache
        tl.store(past_key_states_ptr + cache_rope_offset + offsets, kv_out_first, mask=mask)
        tl.store(past_key_states_ptr + cache_rope_offset + half_dim + offsets, kv_out_second, mask=mask)
        
        # Write to offload_kv
        tl.store(offload_kv_ptr + offload_rope_offset + offsets, kv_out_first, mask=mask)
        tl.store(offload_kv_ptr + offload_rope_offset + half_dim + offsets, kv_out_second, mask=mask)
        
        # ===== Apply to ALL Q heads using SAME cos/sin (already in registers!) =====
        for head_idx in range(num_heads):
            q_pe_base_offset = batch_idx * q_pe_batch_stride + head_idx * q_pe_head_stride
            
            # Load Q data
            q_even = tl.load(q_pe_ptr + q_pe_base_offset + even_indices, 
                            mask=mask, other=0.0)
            q_odd = tl.load(q_pe_ptr + q_pe_base_offset + odd_indices, 
                           mask=mask, other=0.0)
            
            # Apply RoPE using cos/sin from registers (no memory load!)
            q_out_first = q_even * cos_first + (-q_odd) * sin_first
            q_out_second = q_odd * cos_second + q_even * sin_second
            
            # Write back
            tl.store(q_pe_ptr + q_pe_base_offset + offsets, q_out_first, mask=mask)
            tl.store(q_pe_ptr + q_pe_base_offset + half_dim + offsets, q_out_second, mask=mask)


def fused_rmsnorm_rope_cache_update_with_q_return_new_kv(
    new_compressed_kv: torch.Tensor,      # [bsz, 1, total_dim]
    past_key_states: torch.Tensor,        # [bsz, max_seq_len, total_dim]
    q_pe: torch.Tensor,                   # [bsz, num_heads, 1, qk_rope_head_dim]
    cos: torch.Tensor,                    # [max_seq_len, qk_rope_head_dim]
    sin: torch.Tensor,                    # [max_seq_len, qk_rope_head_dim]
    position_ids: torch.Tensor,           # [bsz, 1]
    norm_weight: torch.Tensor,            # [kv_lora_rank]
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm + RoPE with TRUE cos/sin reuse.
    Loads cos/sin once per block, applies to KV and all Q heads.
    
    Returns:
        offload_kv: Processed KV tensor [bsz, 1, total_dim] for downstream use
    """
    bsz, q_len, total_dim = new_compressed_kv.shape
    assert q_len == 1, "This kernel assumes q_len = 1 for decoding"
    assert total_dim == kv_lora_rank + qk_rope_head_dim
    assert qk_rope_head_dim % 2 == 0
    assert past_key_states.is_contiguous(), "past_key_states must be contiguous"
    assert q_pe.is_contiguous(), "q_pe must be contiguous"
    assert position_ids.shape == (bsz, 1)
    
    num_heads = q_pe.shape[1]
    max_seq_len = past_key_states.shape[1]
    
    assert q_pe.shape == (bsz, num_heads, 1, qk_rope_head_dim)
    
    # Allocate output tensor for offload_kv
    offload_kv = torch.empty(
        (bsz, 1, total_dim),
        dtype=new_compressed_kv.dtype,
        device=new_compressed_kv.device
    )
    
    grid = (bsz,)
    
    fused_rmsnorm_rope_cache_update_with_q_return_new_kv_kernel[grid](
        new_compressed_kv,
        past_key_states,
        offload_kv,
        q_pe,
        cos,
        sin,
        position_ids,
        norm_weight,
        bsz,
        num_heads,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        eps,
        BLOCK_SIZE=64,
    )
    
    return offload_kv