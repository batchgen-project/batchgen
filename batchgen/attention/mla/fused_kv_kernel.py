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
    """
    HALF_ROPE = BLOCK_ROPE // 2
    rotated_indices = (offsets + HALF_ROPE) % BLOCK_ROPE
    rotated_x = tl.load(base_ptr + rotated_indices * stride)
    negation_mask = offsets < HALF_ROPE
    negated_rotated_x = tl.where(negation_mask, -rotated_x, rotated_x)
    return negated_rotated_x


@triton.jit
def fused_kv_processing_kernel(
    IN_NEW_KV, POS_IDS, COS_CACHE, SIN_CACHE, LN_WEIGHT, LN_BIAS,
    OUT_KV_CACHE_BF16, OUT_KV_CACHE_FP8, OUT_SCALE_CACHE,
    stride_in_kv_b, stride_in_kv_d,
    stride_pos_b,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    stride_ln_w, stride_ln_b,
    stride_out_bf16_b, stride_out_bf16_s, stride_out_bf16_d,
    stride_out_fp8_b, stride_out_fp8_s, stride_out_fp8_d,
    stride_out_scale_b, stride_out_scale_s, stride_out_scale_d,
    LORA_RANK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    MAX_SEQLEN_PAD: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK_LORA: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = tl.load(POS_IDS + pid * stride_pos_b)
    
    # Load lora part
    lora_offsets = tl.arange(0, BLOCK_LORA)
    lora_mask = lora_offsets < LORA_RANK
    in_kv_lora_ptr = IN_NEW_KV + pid * stride_in_kv_b + lora_offsets * stride_in_kv_d
    kv_lora = tl.load(in_kv_lora_ptr, mask=lora_mask, other=0.0)
    
    # Load rope part
    rope_offsets = tl.arange(0, ROPE_DIM)
    in_k_pe_base_ptr = IN_NEW_KV + pid * stride_in_kv_b + LORA_RANK * stride_in_kv_d
    k_pe = tl.load(in_k_pe_base_ptr + rope_offsets * stride_in_kv_d)
    
    # LayerNorm
    kv_lora_f32 = kv_lora.to(tl.float32)
    kv_lora_f32_masked = tl.where(lora_mask, kv_lora_f32, 0.0)
    mean = tl.sum(kv_lora_f32_masked) / LORA_RANK
    var_unbiased = (kv_lora_f32 - mean) * (kv_lora_f32 - mean)
    var = tl.sum(tl.where(lora_mask, var_unbiased, 0.0)) / LORA_RANK
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    
    ln_w = tl.load(LN_WEIGHT + lora_offsets * stride_ln_w, mask=lora_mask, other=0.0)
    ln_b = tl.load(LN_BIAS + lora_offsets * stride_ln_b, mask=lora_mask, other=0.0)
    
    kv_lora_normed_f32 = (kv_lora_f32 - mean) * rstd * ln_w + ln_b
    kv_lora_normed_bf16 = kv_lora_normed_f32.to(tl.bfloat16)
    
    # RoPE
    cos_ptr = COS_CACHE + pos * stride_cos_s + rope_offsets * stride_cos_d
    sin_ptr = SIN_CACHE + pos * stride_sin_s + rope_offsets * stride_sin_d
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)
    
    k_pe_half_rotated = load_rotated_half(in_k_pe_base_ptr, rope_offsets, stride_in_kv_d, ROPE_DIM)
    k_pe_rotated = (k_pe * cos) + (k_pe_half_rotated * sin)
    
    # Write BF16 cache
    out_bf16_lora_ptr = OUT_KV_CACHE_BF16 + pid * stride_out_bf16_b + pos * stride_out_bf16_s + lora_offsets * stride_out_bf16_d
    tl.store(out_bf16_lora_ptr, kv_lora_normed_bf16, mask=lora_mask)
    
    out_bf16_rope_ptr = OUT_KV_CACHE_BF16 + pid * stride_out_bf16_b + pos * stride_out_bf16_s + (LORA_RANK + rope_offsets) * stride_out_bf16_d
    tl.store(out_bf16_rope_ptr, k_pe_rotated)

    # Quantize to FP8
    k_pe_rotated_f32 = k_pe_rotated.to(tl.float32)
    abs_max_lora = tl.max(tl.abs(tl.where(lora_mask, kv_lora_normed_f32, 0.0)))
    abs_max_rope = tl.max(tl.abs(k_pe_rotated_f32))
    abs_max = tl.maximum(abs_max_lora, abs_max_rope)
    
    scale = abs_max / 127.0
    scale = tl.where(scale == 0, 1.0, scale)
    
    kv_lora_f32_scaled = kv_lora_normed_f32 / scale
    k_pe_f32_scaled = k_pe_rotated_f32 / scale
    
    kv_lora_fp8 = kv_lora_f32_scaled.to(tl.int8)
    k_pe_fp8 = k_pe_f32_scaled.to(tl.int8)

    # Write FP8 cache in chunks to avoid masked int8 store issues
    # Chunk 1: 256 elements
    o1 = tl.arange(0, 256)
    tl.store(OUT_KV_CACHE_FP8 + pid * stride_out_fp8_b + pos * stride_out_fp8_s + o1 * stride_out_fp8_d, kv_lora_fp8[o1])
    
    # Chunk 2: 128 elements
    o2 = tl.arange(0, 128) + 256
    tl.store(OUT_KV_CACHE_FP8 + pid * stride_out_fp8_b + pos * stride_out_fp8_s + o2 * stride_out_fp8_d, kv_lora_fp8[o2])
    
    # Chunk 3: 64 elements
    o3 = tl.arange(0, 64) + 384
    tl.store(OUT_KV_CACHE_FP8 + pid * stride_out_fp8_b + pos * stride_out_fp8_s + o3 * stride_out_fp8_d, kv_lora_fp8[o3])
    
    # Write rope part
    out_fp8_rope_ptr = OUT_KV_CACHE_FP8 + pid * stride_out_fp8_b + pos * stride_out_fp8_s + (LORA_RANK + rope_offsets) * stride_out_fp8_d
    tl.store(out_fp8_rope_ptr, k_pe_fp8)
    
    # Write scale
    out_scale_ptr = OUT_SCALE_CACHE + pid * stride_out_scale_b + pos * stride_out_scale_s
    tl.store(out_scale_ptr, scale.to(tl.bfloat16))


@triton.jit
def rotate_q_pe_kernel(
    IN_Q_PE, OUT_Q_PE, POS_IDS, COS_CACHE, SIN_CACHE,
    stride_in_q_b, stride_in_q_h, stride_in_q_d,
    stride_out_q_b, stride_out_q_h, stride_out_q_d,
    stride_pos_b,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    ROPE_DIM: tl.constexpr,
    MAX_SEQLEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pos = tl.load(POS_IDS + pid_b * stride_pos_b)
    
    rope_offsets = tl.arange(0, ROPE_DIM)
    in_q_pe_base_ptr = IN_Q_PE + pid_b * stride_in_q_b + pid_h * stride_in_q_h
    q_pe = tl.load(in_q_pe_base_ptr + rope_offsets * stride_in_q_d)
    
    cos_ptr = COS_CACHE + pos * stride_cos_s + rope_offsets * stride_cos_d
    sin_ptr = SIN_CACHE + pos * stride_sin_s + rope_offsets * stride_sin_d
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)
    
    q_pe_half_rotated = load_rotated_half(in_q_pe_base_ptr, rope_offsets, stride_in_q_d, ROPE_DIM)
    q_pe_rotated = (q_pe * cos) + (q_pe_half_rotated * sin)
    
    out_q_pe_ptr = OUT_Q_PE + pid_b * stride_out_q_b + pid_h * stride_out_q_h + rope_offsets * stride_out_q_d
    tl.store(out_q_pe_ptr, q_pe_rotated)


def fused_kv_update_and_rope(
    new_compressed_kv, q_pe, q_position_ids,
    rotary_emb_cos, rotary_emb_sin, kv_a_layernorm,
    compressed_kv_ref, past_key_states, scale_cache,
    kv_lora_rank, qk_rope_head_dim,
):
    bsz, num_heads, _, rope_dim = q_pe.shape
    max_seqlen = rotary_emb_cos.shape[0]
    max_seqlen_pad = compressed_kv_ref.shape[1]
    kv_dim = kv_lora_rank + qk_rope_head_dim
    
    in_kv = new_compressed_kv.squeeze(1).contiguous()
    in_q_pe = q_pe.squeeze(2).contiguous()
    pos_ids = q_position_ids.squeeze(1).contiguous()
    
    ln_weight = kv_a_layernorm.weight.contiguous()
    ln_bias = torch.zeros_like(ln_weight) if not hasattr(kv_a_layernorm, 'bias') else kv_a_layernorm.bias.contiguous()
    ln_eps = kv_a_layernorm.variance_epsilon if hasattr(kv_a_layernorm, 'variance_epsilon') else kv_a_layernorm.eps
    
    cos_cache = rotary_emb_cos.contiguous()
    sin_cache = rotary_emb_sin.contiguous()

    out_q_pe = torch.empty_like(in_q_pe)
    rotate_q_pe_kernel[(bsz, num_heads)](
        in_q_pe, out_q_pe, pos_ids, cos_cache, sin_cache,
        in_q_pe.stride(0), in_q_pe.stride(1), in_q_pe.stride(2),
        out_q_pe.stride(0), out_q_pe.stride(1), out_q_pe.stride(2),
        pos_ids.stride(0),
        cos_cache.stride(0), cos_cache.stride(1),
        sin_cache.stride(0), sin_cache.stride(1),
        ROPE_DIM=qk_rope_head_dim, MAX_SEQLEN=max_seqlen, NUM_HEADS=num_heads,
    )
    
    BLOCK_LORA = triton.next_power_of_2(kv_lora_rank)
    
    fused_kv_processing_kernel[(bsz,)](
        in_kv, pos_ids, cos_cache, sin_cache, ln_weight, ln_bias,
        compressed_kv_ref, past_key_states, scale_cache,
        in_kv.stride(0), in_kv.stride(1), pos_ids.stride(0),
        cos_cache.stride(0), cos_cache.stride(1),
        sin_cache.stride(0), sin_cache.stride(1),
        ln_weight.stride(0), ln_bias.stride(0),
        compressed_kv_ref.stride(0), compressed_kv_ref.stride(1), compressed_kv_ref.stride(2),
        past_key_states.stride(0), past_key_states.stride(1), past_key_states.stride(2),
        scale_cache.stride(0), scale_cache.stride(1), scale_cache.stride(2),
        LORA_RANK=kv_lora_rank, ROPE_DIM=qk_rope_head_dim, KV_DIM=kv_dim,
        MAX_SEQLEN_PAD=max_seqlen_pad, LN_EPS=ln_eps, BLOCK_LORA=BLOCK_LORA,
    )
    
    return out_q_pe.unsqueeze(2)