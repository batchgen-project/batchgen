import torch
import triton
import triton.language as tl

@triton.jit
def load_rotated_half(
    base_ptr,           # Base pointer (scalar) to the start of the vector
    offsets,            # The tl.arange(0, BLOCK_ROPE)
    stride,             # Stride for the feature dimension
    BLOCK_ROPE: tl.constexpr
):
    """
    Loads a vector from memory, applying the 'rotate_half' operation.
    This means the first half of the output vector is the *negated*
    second half of the input vector, and the second half of the output
    is the *first* half of the input.
    """
    HALF_ROPE = BLOCK_ROPE // 2
    
    # offsets are [0, 1, ..., H-1, H, ..., R-1]
    # We want to load from indices: [H, ..., R-1, 0, ..., H-1]
    rotated_indices = (offsets + HALF_ROPE) % BLOCK_ROPE
    
    # Load the vector from rotated indices with proper stride
    rotated_x = tl.load(base_ptr + rotated_indices * stride)
    
    # We need to negate the part that *was* in the second half,
    # which is now in the *first* half of our loaded `rotated_x`.
    negation_mask = offsets < HALF_ROPE # [True, ..., True, False, ..., False]
    negated_rotated_x = tl.where(negation_mask, -rotated_x, rotated_x)
    
    return negated_rotated_x


@triton.jit
def fused_kv_processing_kernel(
    # --- Inputs ---
    IN_NEW_KV,              # Pointer to new_compressed_kv, shape (bsz, kv_dim)
    POS_IDS,                # Pointer to q_position_ids, shape (bsz)
    COS_CACHE,              # Pointer to rotary_emb.cos_cached, shape (max_seqlen, rope_dim)
    SIN_CACHE,              # Pointer to rotary_emb.sin_cached, shape (max_seqlen, rope_dim)
    RMS_WEIGHT,             # Pointer to kv_a_layernorm.weight, shape (lora_rank)
    
    # --- Outputs ---
    OUT_KV_CACHE_BF16,      # Pointer to compressed_kv_ref, shape (bsz, max_seqlen_pad, kv_dim)
    OUT_KV_CACHE_FP8,       # Pointer to past_key_states, shape (bsz, max_seqlen_pad, kv_dim)
    OUT_SCALE_CACHE,        # Pointer to scale cache, shape (bsz, max_seqlen_pad, 1)

    # --- Strides ---
    stride_in_kv_b, stride_in_kv_d,
    stride_pos_b,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_rms_w,
    stride_out_bf16_b, stride_out_bf16_s, stride_out_bf16_d,
    stride_out_fp8_b, stride_out_fp8_s, stride_out_fp8_d,
    stride_out_scale_b, stride_out_scale_s, stride_out_scale_d,
    
    # --- Constants ---
    LORA_RANK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    MAX_SEQLEN_PAD: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK_LORA: tl.constexpr, # Next power of 2 >= LORA_RANK
):
    """
    Triton kernel to fuse KV processing for a single token.
    Grid: (bsz,)
    Each program handles one item in the batch.
    """
    # 1. Get program ID (batch index)
    pid = tl.program_id(0) # Batch index
    
    # 2. Get position ID for this batch item
    pos = tl.load(POS_IDS + pid * stride_pos_b) # scalar
    
    # --- 3. Load Input KV vector ---
    # Load the kv_lora part
    lora_offsets = tl.arange(0, BLOCK_LORA)
    lora_mask = lora_offsets < LORA_RANK
    in_kv_lora_ptr = IN_NEW_KV + pid * stride_in_kv_b + lora_offsets * stride_in_kv_d
    kv_lora = tl.load(in_kv_lora_ptr, mask=lora_mask, other=0.0) # shape (BLOCK_LORA,)
    
    # Load the k_pe part (ROPE_DIM is power of 2, no mask needed)
    rope_offsets = tl.arange(0, ROPE_DIM)
    in_k_pe_base_ptr = IN_NEW_KV + pid * stride_in_kv_b + LORA_RANK * stride_in_kv_d  # Base pointer (scalar)
    k_pe = tl.load(in_k_pe_base_ptr + rope_offsets * stride_in_kv_d) # shape (ROPE_DIM,)
    
    # --- 4. Apply RMSNorm to kv_lora (in float32) ---
    kv_lora_f32 = kv_lora.to(tl.float32)
    
    # RMSNorm: no mean subtraction, only RMS normalization
    # Compute mean of squares (only for valid elements)
    kv_lora_f32_masked = tl.where(lora_mask, kv_lora_f32, 0.0)
    mean_sq = tl.sum(kv_lora_f32_masked * kv_lora_f32_masked) / LORA_RANK
    
    # RMS normalization
    rrms = 1.0 / tl.sqrt(mean_sq + RMS_EPS)
    
    # Load weight (no bias for RMSNorm)
    rms_w = tl.load(RMS_WEIGHT + lora_offsets * stride_rms_w, mask=lora_mask, other=0.0)
    
    # Apply RMSNorm: x * rrms * weight
    kv_lora_normed_f32 = kv_lora_f32 * rrms * rms_w
    kv_lora_normed_bf16 = kv_lora_normed_f32.to(tl.bfloat16) # final BF16 lora part
    
    # --- 5. Apply RoPE to k_pe (in bfloat16) ---
    cos_ptr = COS_CACHE + pos * stride_cos_s + rope_offsets * stride_cos_d
    sin_ptr = SIN_CACHE + pos * stride_sin_s + rope_offsets * stride_sin_d
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)
    
    # Load the rotated version of k_pe using base pointer
    k_pe_half_rotated = load_rotated_half(in_k_pe_base_ptr, rope_offsets, stride_in_kv_d, ROPE_DIM)
    
    # k_pe_rotated = (k_pe * cos) + (rotate_half(k_pe) * sin)
    k_pe_rotated = (k_pe * cos) + (k_pe_half_rotated * sin)
    
    # --- 6. Write to BF16 Cache (compressed_kv_ref) ---
    # `offload_kv` is now `kv_lora_normed_bf16` and `k_pe_rotated`
    out_bf16_lora_ptr = OUT_KV_CACHE_BF16 + pid * stride_out_bf16_b + pos * stride_out_bf16_s + lora_offsets * stride_out_bf16_d
    tl.store(out_bf16_lora_ptr, kv_lora_normed_bf16, mask=lora_mask)
    
    out_bf16_rope_ptr = OUT_KV_CACHE_BF16 + pid * stride_out_bf16_b + pos * stride_out_bf16_s + (LORA_RANK + rope_offsets) * stride_out_bf16_d
    tl.store(out_bf16_rope_ptr, k_pe_rotated) # No mask, ROPE_DIM is pow2

    # --- 7. Quantize to FP8 (Per-Token) ---
    # We need the full vector in f32 for quantization
    k_pe_rotated_f32 = k_pe_rotated.to(tl.float32) # (ROPE_DIM,)
    
    # Find absolute max over lora part (masked)
    abs_max_lora = tl.max(tl.abs(tl.where(lora_mask, kv_lora_normed_f32, 0.0)))
    # Find absolute max over rope part (no mask needed)
    abs_max_rope = tl.max(tl.abs(k_pe_rotated_f32))
    
    # Find overall abs_max
    abs_max = tl.maximum(abs_max_lora, abs_max_rope)
    
    # Calculate scale
    sf = 127.0
    scale = abs_max / sf
    scale = tl.where(scale == 0, 1.0, scale) # Avoid division by zero
    
    # Quantize both parts
    kv_lora_f32_scaled = kv_lora_normed_f32 / scale
    k_pe_f32_scaled = k_pe_rotated_f32 / scale
    
    kv_lora_fp8 = kv_lora_f32_scaled.to(tl.int8)
    k_pe_fp8 = k_pe_f32_scaled.to(tl.int8)

    # --- 8. Write to FP8 Cache (past_key_states) ---
    # Write lora part
    out_fp8_lora_ptr = OUT_KV_CACHE_FP8 + pid * stride_out_fp8_b + pos * stride_out_fp8_s + lora_offsets * stride_out_fp8_d
    tl.store(out_fp8_lora_ptr, kv_lora_fp8, mask=lora_mask)
    
    # Write rope part
    out_fp8_rope_ptr = OUT_KV_CACHE_FP8 + pid * stride_out_fp8_b + pos * stride_out_fp8_s + (LORA_RANK + rope_offsets) * stride_out_fp8_d
    tl.store(out_fp8_rope_ptr, k_pe_fp8) # No mask, ROPE_DIM is pow2
    
    # --- 9. Write to Scale Cache (scale) ---
    out_scale_ptr = OUT_SCALE_CACHE + pid * stride_out_scale_b + pos * stride_out_scale_s # shape is (bsz, max_seqlen, 1)
    tl.store(out_scale_ptr, scale.to(tl.bfloat16))


@triton.jit
def rotate_q_pe_kernel(
    # --- In/Out ---
    IN_Q_PE,                # Pointer to q_pe, shape (bsz, num_heads, rope_dim)
    OUT_Q_PE,               # Pointer to output buffer, shape (bsz, num_heads, rope_dim)

    # --- Inputs ---
    POS_IDS,                # Pointer to q_position_ids, shape (bsz)
    COS_CACHE,              # Pointer to rotary_emb.cos_cached, shape (max_seqlen, rope_dim)
    SIN_CACHE,              # Pointer to rotary_emb.sin_cached, shape (max_seqlen, rope_dim)

    # --- Strides ---
    stride_in_q_b, stride_in_q_h, stride_in_q_d,
    stride_out_q_b, stride_out_q_h, stride_out_q_d,
    stride_pos_b,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,

    # --- Constants ---
    ROPE_DIM: tl.constexpr,
    MAX_SEQLEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    """
    Triton kernel to apply RoPE to q_pe.
    Grid: (bsz, num_heads)
    """
    # 1. Get program IDs
    pid_b = tl.program_id(0) # Batch index
    pid_h = tl.program_id(1) # Head index
    
    # 2. Get position ID
    pos = tl.load(POS_IDS + pid_b * stride_pos_b) # scalar
    
    # 3. Load q_pe vector for this batch/head
    rope_offsets = tl.arange(0, ROPE_DIM)
    in_q_pe_base_ptr = IN_Q_PE + pid_b * stride_in_q_b + pid_h * stride_in_q_h  # Base pointer (scalar)
    q_pe = tl.load(in_q_pe_base_ptr + rope_offsets * stride_in_q_d) # shape (ROPE_DIM,)
    
    # 4. Load cos/sin
    cos_ptr = COS_CACHE + pos * stride_cos_s + rope_offsets * stride_cos_d
    sin_ptr = SIN_CACHE + pos * stride_sin_s + rope_offsets * stride_sin_d
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)
    
    # 5. Apply RoPE
    # Load the rotated version of q_pe using base pointer
    q_pe_half_rotated = load_rotated_half(in_q_pe_base_ptr, rope_offsets, stride_in_q_d, ROPE_DIM)
    
    # q_pe_rotated = (q_pe * cos) + (rotate_half(q_pe) * sin)
    q_pe_rotated = (q_pe * cos) + (q_pe_half_rotated * sin)
    
    # 6. Store rotated q_pe
    out_q_pe_ptr = OUT_Q_PE + pid_b * stride_out_q_b + pid_h * stride_out_q_h + rope_offsets * stride_out_q_d
    tl.store(out_q_pe_ptr, q_pe_rotated)


# def fused_kv_update_and_rope(
#     new_compressed_kv: torch.Tensor,    # (bsz, 1, kv_dim)
#     q_pe: torch.Tensor,                 # (bsz, num_heads, 1, rope_dim)
#     q_position_ids: torch.Tensor,       # (bsz, 1)
#     rotary_emb_cos: torch.Tensor,       # (max_seqlen, rope_dim)
#     rotary_emb_sin: torch.Tensor,       # (max_seqlen, rope_dim)
#     kv_a_layernorm,                     # LayerNorm or RMSNorm module
    
#     # Caches to be updated in-place
#     compressed_kv_ref: torch.Tensor,    # (bsz, max_seqlen_pad, kv_dim)
#     past_key_states: torch.Tensor,      # (bsz, max_seqlen_pad, kv_dim)
#     scale_cache: torch.Tensor,          # (bsz, max_seqlen_pad, 1)
    
#     # Model dims
#     kv_lora_rank: int,
#     qk_rope_head_dim: int,
# ) -> torch.Tensor:
#     """
#     Python wrapper to replace the original block of PyTorch operations.
    
#     Returns:
#         rotated_q_pe: The RoPE-applied q_pe tensor, shape (bsz, num_heads, 1, rope_dim)
#     """
    
#     # --- 1. Prepare inputs and dimensions ---
#     bsz, num_heads, _, rope_dim = q_pe.shape
#     assert rope_dim == qk_rope_head_dim
    
#     _, _, kv_dim_in = new_compressed_kv.shape
#     kv_dim = kv_lora_rank + qk_rope_head_dim
#     assert kv_dim_in == kv_dim
    
#     max_seqlen = rotary_emb_cos.shape[0]
#     max_seqlen_pad = compressed_kv_ref.shape[1]
    
#     # Ensure inputs are contiguous and squeezed to expected kernel shapes
#     # (bsz, 1, kv_dim) -> (bsz, kv_dim)
#     in_kv = new_compressed_kv.squeeze(1).contiguous()
    
#     # (bsz, num_heads, 1, rope_dim) -> (bsz, num_heads, rope_dim)
#     in_q_pe = q_pe.squeeze(2).contiguous()
    
#     # (bsz, 1) -> (bsz)
#     pos_ids = q_position_ids.squeeze(1).contiguous()
    
#     # Get LN weights - handle both LayerNorm and RMSNorm
#     ln_weight = kv_a_layernorm.weight.contiguous()
#     # RMSNorm doesn't have bias, so create a zero tensor if it doesn't exist
#     if hasattr(kv_a_layernorm, 'bias') and kv_a_layernorm.bias is not None:
#         ln_bias = kv_a_layernorm.bias.contiguous()
#     else:
#         ln_bias = torch.zeros_like(ln_weight)
    
#     # Get epsilon - handle both attribute names
#     if hasattr(kv_a_layernorm, 'variance_epsilon'):
#         ln_eps = kv_a_layernorm.variance_epsilon
#     else:
#         ln_eps = kv_a_layernorm.eps
    
#     # Rotary embeddings
#     cos_cache = rotary_emb_cos.contiguous()
#     sin_cache = rotary_emb_sin.contiguous()

#     # --- 2. Launch rotate_q_pe_kernel ---
#     out_q_pe = torch.empty_like(in_q_pe)
#     grid_q = (bsz, num_heads)
    
#     rotate_q_pe_kernel[grid_q](
#         in_q_pe, out_q_pe,
#         pos_ids, cos_cache, sin_cache,
#         # Strides
#         in_q_pe.stride(0), in_q_pe.stride(1), in_q_pe.stride(2),
#         out_q_pe.stride(0), out_q_pe.stride(1), out_q_pe.stride(2),
#         pos_ids.stride(0),
#         cos_cache.stride(0), cos_cache.stride(1),
#         sin_cache.stride(0), sin_cache.stride(1),
#         # Constants
#         ROPE_DIM=qk_rope_head_dim,
#         MAX_SEQLEN=max_seqlen,
#         NUM_HEADS=num_heads,
#     )
    
#     # --- 3. Launch fused_kv_processing_kernel ---
#     grid_kv = (bsz,)
    
#     # Calculate next power of 2 for LORA_RANK
#     BLOCK_LORA = triton.next_power_of_2(kv_lora_rank)
    
#     fused_kv_processing_kernel[grid_kv](
#         in_kv, pos_ids, cos_cache, sin_cache, ln_weight, ln_bias,
#         compressed_kv_ref, past_key_states, scale_cache,
#         # Strides
#         in_kv.stride(0), in_kv.stride(1),
#         pos_ids.stride(0),
#         cos_cache.stride(0), cos_cache.stride(1),
#         sin_cache.stride(0), sin_cache.stride(1),
#         ln_weight.stride(0),
#         ln_bias.stride(0),
#         compressed_kv_ref.stride(0), compressed_kv_ref.stride(1), compressed_kv_ref.stride(2),
#         past_key_states.stride(0), past_key_states.stride(1), past_key_states.stride(2),
#         scale_cache.stride(0), scale_cache.stride(1), scale_cache.stride(2),
#         # Constants
#         LORA_RANK=kv_lora_rank,
#         ROPE_DIM=qk_rope_head_dim,
#         KV_DIM=kv_dim,
#         MAX_SEQLEN_PAD=max_seqlen_pad,
#         LN_EPS=ln_eps,
#         BLOCK_LORA=BLOCK_LORA,
#     )
    
#     # --- 4. Return rotated q_pe with original shape ---
#     return out_q_pe.unsqueeze(2)

def fused_kv_update_and_rope(
    new_compressed_kv: torch.Tensor,    # (bsz, 1, kv_dim)
    q_pe: torch.Tensor,                 # (bsz, num_heads, 1, rope_dim)
    q_position_ids: torch.Tensor,       # (bsz, 1)
    rotary_emb_cos: torch.Tensor,       # (max_seqlen, rope_dim)
    rotary_emb_sin: torch.Tensor,       # (max_seqlen, rope_dim)
    kv_a_layernorm,                     # LayerNorm or RMSNorm module
    
    # Caches to be updated in-place
    compressed_kv_ref: torch.Tensor,    # (bsz, max_seqlen_pad, kv_dim)
    past_key_states: torch.Tensor,      # (bsz, max_seqlen_pad, kv_dim)
    scale_cache: torch.Tensor,          # (bsz, max_seqlen_pad, 1)
    
    # Model dims
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    """
    Python wrapper to replace the original block of PyTorch operations.
    
    Returns:
        rotated_q_pe: The RoPE-applied q_pe tensor, shape (bsz, num_heads, 1, rope_dim)
    """
    
    # --- 1. Prepare inputs and dimensions ---
    bsz, num_heads, _, rope_dim = q_pe.shape
    assert rope_dim == qk_rope_head_dim
    
    _, _, kv_dim_in = new_compressed_kv.shape
    kv_dim = kv_lora_rank + qk_rope_head_dim
    assert kv_dim_in == kv_dim
    
    max_seqlen = rotary_emb_cos.shape[0]
    max_seqlen_pad = compressed_kv_ref.shape[1]
    
    # Ensure inputs are contiguous and squeezed to expected kernel shapes
    # (bsz, 1, kv_dim) -> (bsz, kv_dim)
    in_kv = new_compressed_kv.squeeze(1).contiguous()
    
    # (bsz, num_heads, 1, rope_dim) -> (bsz, num_heads, rope_dim)
    in_q_pe = q_pe.squeeze(2).contiguous()
    
    # (bsz, 1) -> (bsz)
    pos_ids = q_position_ids.squeeze(1).contiguous()
    
    # Get LN weights - handle both LayerNorm and RMSNorm
    ln_weight = kv_a_layernorm.weight.contiguous()
    # RMSNorm doesn't have bias, so create a zero tensor if it doesn't exist
    if hasattr(kv_a_layernorm, 'bias') and kv_a_layernorm.bias is not None:
        ln_bias = kv_a_layernorm.bias.contiguous()
    else:
        ln_bias = torch.zeros_like(ln_weight)
    
    # Get epsilon - handle both attribute names
    if hasattr(kv_a_layernorm, 'variance_epsilon'):
        ln_eps = kv_a_layernorm.variance_epsilon
    else:
        ln_eps = kv_a_layernorm.eps
    
    # Rotary embeddings
    cos_cache = rotary_emb_cos.contiguous()
    sin_cache = rotary_emb_sin.contiguous()

    # --- 2. Launch rotate_q_pe_kernel ---
    out_q_pe = torch.empty_like(in_q_pe)
    grid_q = (bsz, num_heads)
    
    rotate_q_pe_kernel[grid_q](
        in_q_pe, out_q_pe,
        pos_ids, cos_cache, sin_cache,
        # Strides
        in_q_pe.stride(0), in_q_pe.stride(1), in_q_pe.stride(2),
        out_q_pe.stride(0), out_q_pe.stride(1), out_q_pe.stride(2),
        pos_ids.stride(0),
        cos_cache.stride(0), cos_cache.stride(1),
        sin_cache.stride(0), sin_cache.stride(1),
        # Constants
        ROPE_DIM=qk_rope_head_dim,
        MAX_SEQLEN=max_seqlen,
        NUM_HEADS=num_heads,
    )
    
    # --- 3. Launch fused_kv_processing_kernel ---
    grid_kv = (bsz,)
    
    # Calculate next power of 2 for LORA_RANK
    BLOCK_LORA = triton.next_power_of_2(kv_lora_rank)
    
    fused_kv_processing_kernel[grid_kv](
        # Inputs
        in_kv, 
        pos_ids, 
        cos_cache, 
        sin_cache, 
        ln_weight, 
        ln_bias,
        # Outputs
        compressed_kv_ref, 
        past_key_states, 
        scale_cache,
        # Strides - in_kv
        in_kv.stride(0), 
        in_kv.stride(1),
        # Strides - pos_ids
        pos_ids.stride(0),
        # Strides - cos_cache
        cos_cache.stride(0), 
        cos_cache.stride(1),
        # Strides - sin_cache
        sin_cache.stride(0), 
        sin_cache.stride(1),
        # Strides - ln_weight
        ln_weight.stride(0),
        # Strides - ln_bias
        ln_bias.stride(0),
        # Strides - compressed_kv_ref
        compressed_kv_ref.stride(0), 
        compressed_kv_ref.stride(1), 
        compressed_kv_ref.stride(2),
        # Strides - past_key_states
        past_key_states.stride(0), 
        past_key_states.stride(1), 
        past_key_states.stride(2),
        # Strides - scale_cache
        scale_cache.stride(0), 
        scale_cache.stride(1), 
        scale_cache.stride(2),
        # Constants (as keyword arguments)
        LORA_RANK=kv_lora_rank,
        ROPE_DIM=qk_rope_head_dim,
        KV_DIM=kv_dim,
        MAX_SEQLEN_PAD=max_seqlen_pad,
        LN_EPS=ln_eps,
        BLOCK_LORA=BLOCK_LORA,
    )
    
    # --- 4. Return rotated q_pe with original shape ---
    return out_q_pe.unsqueeze(2)