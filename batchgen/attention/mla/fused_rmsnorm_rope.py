import triton
import triton.language as tl
import torch
# @triton.jit
# def fused_rmsnorm_rope_kernel(
# 	data_ptr,
# 	cos_ptr,
# 	sin_ptr,
# 	position_ids_ptr,
# 	norm_weight_ptr,
# 	bsz,
# 	q_len,
# 	kv_lora_rank,
# 	qk_rope_head_dim,
# 	max_seq_len,
# 	variance_epsilon: tl.constexpr,
# 	BLOCK_SIZE: tl.constexpr,
# ):
# 	pid = tl.program_id(0)
	
# 	batch_idx = pid // q_len
# 	seq_idx = pid % q_len
	
# 	total_dim = kv_lora_rank + qk_rope_head_dim
# 	base_offset = batch_idx * q_len * total_dim + seq_idx * total_dim
	
# 	# ==================== Part 1: RMSNorm with FP32 Accumulation ====================
	
# 	# Initialize fp32 accumulator
# 	sum_sq_fp32 = 0.0
	
# 	# Accumulate sum of squares in fp32
# 	for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
# 		offsets = block_start + tl.arange(0, BLOCK_SIZE)
# 		mask = offsets < kv_lora_rank
		
# 		# Load bf16, convert to fp32
# 		data_bf16 = tl.load(data_ptr + base_offset + offsets, mask=mask, other=0.0)
# 		data_fp32 = data_bf16.to(tl.float32)
		
# 		# Square and accumulate in fp32
# 		sum_sq_fp32 += tl.sum(data_fp32 * data_fp32)
	
# 	# Compute variance with proper float division
# 	kv_lora_rank_float = tl.cast(kv_lora_rank, tl.float32)
# 	variance_fp32 = sum_sq_fp32 / kv_lora_rank_float
# 	inv_rms_fp32 = 1.0 / tl.sqrt(variance_fp32 + variance_epsilon)
	
# 	# Apply normalization
# 	for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
# 		offsets = block_start + tl.arange(0, BLOCK_SIZE)
# 		mask = offsets < kv_lora_rank
		
# 		data_bf16 = tl.load(data_ptr + base_offset + offsets, mask=mask, other=0.0)
# 		weight_bf16 = tl.load(norm_weight_ptr + offsets, mask=mask, other=1.0)
		
# 		# Compute in fp32
# 		data_fp32 = data_bf16.to(tl.float32)
# 		weight_fp32 = weight_bf16.to(tl.float32)
# 		normalized_fp32 = data_fp32 * inv_rms_fp32 * weight_fp32
		
# 		# Store as bf16
# 		normalized_bf16 = normalized_fp32.to(data_bf16.dtype)
# 		tl.store(data_ptr + base_offset + offsets, normalized_bf16, mask=mask)
	
# 	# ==================== Part 2: RoPE ====================
	
# 	pos_id = tl.load(position_ids_ptr + batch_idx * q_len + seq_idx)
# 	rope_offset = base_offset + kv_lora_rank
# 	half_dim = qk_rope_head_dim // 2
	
# 	for i in range(0, half_dim, BLOCK_SIZE):
# 		offsets = i + tl.arange(0, BLOCK_SIZE)
# 		mask = offsets < half_dim
		
# 		x1 = tl.load(data_ptr + rope_offset + offsets, mask=mask, other=0.0)
# 		x2 = tl.load(data_ptr + rope_offset + half_dim + offsets, mask=mask, other=0.0)
		
# 		cos_offset = pos_id * qk_rope_head_dim
# 		cos1 = tl.load(cos_ptr + cos_offset + offsets, mask=mask, other=1.0)
# 		sin1 = tl.load(sin_ptr + cos_offset + offsets, mask=mask, other=0.0)
# 		cos2 = tl.load(cos_ptr + cos_offset + half_dim + offsets, mask=mask, other=1.0)
# 		sin2 = tl.load(sin_ptr + cos_offset + half_dim + offsets, mask=mask, other=0.0)
		
# 		out1 = x1 * cos1 + (-x2) * sin1
# 		out2 = x2 * cos2 + x1 * sin2
		
# 		tl.store(data_ptr + rope_offset + offsets, out1, mask=mask)
# 		tl.store(data_ptr + rope_offset + half_dim + offsets, out2, mask=mask)

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
    pid = tl.program_id(0)
    
    batch_idx = pid // q_len
    seq_idx = pid % q_len
    
    total_dim = kv_lora_rank + qk_rope_head_dim
    base_offset = batch_idx * q_len * total_dim + seq_idx * total_dim
    
    # ==================== Part 1: RMSNorm (same as before) ====================
    
    sum_sq_fp32 = 0.0
    
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        
        data_bf16 = tl.load(data_ptr + base_offset + offsets, mask=mask, other=0.0)
        data_fp32 = data_bf16.to(tl.float32)
        sum_sq_fp32 += tl.sum(data_fp32 * data_fp32)
    
    kv_lora_rank_float = tl.cast(kv_lora_rank, tl.float32)
    variance_fp32 = sum_sq_fp32 / kv_lora_rank_float
    inv_rms_fp32 = 1.0 / tl.sqrt(variance_fp32 + variance_epsilon)
    
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        
        data_bf16 = tl.load(data_ptr + base_offset + offsets, mask=mask, other=0.0)
        weight_bf16 = tl.load(norm_weight_ptr + offsets, mask=mask, other=1.0)
        
        data_fp32 = data_bf16.to(tl.float32)
        weight_fp32 = weight_bf16.to(tl.float32)
        normalized_fp32 = data_fp32 * inv_rms_fp32 * weight_fp32
        
        normalized_bf16 = normalized_fp32.to(data_bf16.dtype)
        tl.store(data_ptr + base_offset + offsets, normalized_bf16, mask=mask)
    
    # ==================== Part 2: RoPE (NO INTERLEAVING NEEDED!) ====================
    # 
    # CRITICAL INSIGHT: The production code does interleaving in the VIEW operation,
    # but since we're working with the ORIGINAL memory layout, we DON'T need to interleave!
    # The view/transpose/reshape is just a way to express the rotation in PyTorch.
    # Our kernel can work directly on the original layout.
    
    pos_id = tl.load(position_ids_ptr + batch_idx * q_len + seq_idx)
    
    rope_offset = base_offset + kv_lora_rank
    half_dim = qk_rope_head_dim // 2
    
    # Process dimension pairs: (x0,x1), (x2,x3), ..., which map to (x0,x32), (x1,x33), ...
    # after the interleaving view in production code
    
    for i in range(0, half_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim
        
        # In original memory: [x0, x1, x2, ..., x31, x32, x33, ..., x63]
        # After production's view: [x0, x2, ..., x62, x1, x3, ..., x63]
        #                          first_half = evens, second_half = odds
        
        # So x_first[i] corresponds to original x[2*i]
        # And x_second[i] corresponds to original x[2*i+1]
        
        # Load first half (evens in original)
        x_first = tl.load(data_ptr + rope_offset + offsets, mask=mask, other=0.0)
        # Load second half (odds in original)
        x_second = tl.load(data_ptr + rope_offset + half_dim + offsets, mask=mask, other=0.0)
        
        # Load cos/sin
        cos_offset_base = pos_id * qk_rope_head_dim
        cos_first = tl.load(cos_ptr + cos_offset_base + offsets, mask=mask, other=1.0)
        sin_first = tl.load(sin_ptr + cos_offset_base + offsets, mask=mask, other=0.0)
        cos_second = tl.load(cos_ptr + cos_offset_base + half_dim + offsets, mask=mask, other=1.0)
        sin_second = tl.load(sin_ptr + cos_offset_base + half_dim + offsets, mask=mask, other=0.0)
        
        # Apply RoPE: (x * cos) + (rotate_half(x) * sin)
        # rotate_half([first, second]) = [-second, first]
        out_first = x_first * cos_first + (-x_second) * sin_first
        out_second = x_second * cos_second + x_first * sin_second
        
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
	"""Fused Triton kernel implementation"""
	bsz, q_len, total_dim = data.shape
	assert total_dim == kv_lora_rank + qk_rope_head_dim
	assert qk_rope_head_dim % 2 == 0
	
	max_seq_len = cos.shape[0]
	
	grid = (bsz * q_len,)
	BLOCK_SIZE = min(triton.next_power_of_2(kv_lora_rank), 1024)
	
	fused_rmsnorm_rope_kernel[grid](
		data,
		cos,
		sin,
		position_ids,
		norm_weight,
		bsz,
		q_len,
		kv_lora_rank,
		qk_rope_head_dim,
		max_seq_len,
		eps,
		BLOCK_SIZE=BLOCK_SIZE,
	)
	
	return data