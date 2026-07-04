# DeepSeek V4 CSA Write Path: vLLM Implementation Reference

**vLLM Commit**: 687173877781670afde318491564bab92ac353aa (Jun 2026)

## WRITE PATH ARCHITECTURE

### Phase 1: Per-Token Partial-State Staging (`save_partial_states`)
**Purpose**: Write raw KV and score tensors into compressor state cache before boundary-triggered compression.

**Function**: `save_partial_states()`
- **File**: `vllm/models/deepseek_v4/common/ops/save_partial_states.py`
- **Kernel**: `_save_partial_states_kernel` (Triton)
- **Inputs**:
  - `kv`: [num_tokens, head_dim] (bf16, from fused_wkv_wgate GEMM)
  - `score`: [num_tokens, head_dim] (bf16, from fused_wkv_wgate GEMM)
  - `ape`: [compress_ratio, coff*head_dim] (APE bias, fused into score)
  - `positions`: [num_tokens] (int64)
  - `slot_mapping`: [num_tokens] (int32, -1 for padding)
  - `state_cache`: [num_blocks, block_size, 2*state_width] (float32)
  
- **Write Pattern**:
  ```
  One program per token; skips if slot_id < 0 (padding).
  block_idx = slot_id // block_size
  pos_in_block = slot_id % block_size
  base_ptr = state_cache[block_idx, pos_in_block, :]
  
  # Write KV state (first half)
  base_ptr[0:head_dim] = kv[token_idx]
  
  # Write score state (second half) with fused APE addition
  ape_row = position % compress_ratio
  base_ptr[state_width:state_width+head_dim] = score[token_idx] + ape[ape_row]
  ```

- **RAW Hazard**: PDL disabled (`launch_pdl=False`) — this kernel reads from preceding GEMM outputs (kv/score) and writes to state_cache, which is then read by compress kernels. No PDL grid-dependency primitives emitted, causing read-after-write race if PDL enabled.

---

### Phase 2: Boundary-Triggered Compressed-Cache Materialization

#### 2a. Compress → RMSNorm → RoPE → Quantize → Store

**Dispatcher**: `DeepseekCompressor.forward()`
- **File**: `vllm/models/deepseek_v4/compressor.py` (lines 274–399)
- **Selects kernel based on**:
  - `head_dim == 512` (sparse attention, C4A/C128A) → `compress_norm_rope_store_cutedsl` (CuTe DSL, CUDA only)
  - `head_dim == 128` (indexer) → `compress_norm_rope_store_triton` (Triton, all platforms)
  - `use_fp4_cache=True` (indexer MXFP4) → `compress_norm_rope_store_triton` with FP4 kernel variant

**Kernel 1: Sparse Attention (head=512, nope=448 FP8 + rope=64 bf16)**
- **Function**: `_fused_kv_compress_norm_rope_insert_sparse_attn`
- **File**: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` (lines 113–300+)
- **Inputs**:
  - `state_cache`: [num_blocks, block_size, 2*state_width] (float32, from save_partial_states)
  - `positions`: [num_tokens] (int64)
  - `slot_mapping`: [num_tokens] (int32, KV cache slot IDs)
  - `block_table`: [num_reqs, max_blocks_per_req] (int32, paged cache block table)
  - `rms_norm_weight`: [head_dim] (float32)
  - `cos_sin_cache`: [max_pos, rope_head_dim] (cos || sin layout)
  - `kv_cache`: [num_blocks, block_size, token_stride] (uint8 for fp8_ds_mla, or bf16/fp8 for FlashInfer)

- **Write Pattern** (per-token, boundary-triggered):
  ```
  if position % compress_ratio == 0:  # Boundary token
    # Read compressed state from state_cache
    compressed_kv = state_cache[block_idx, pos_in_block, :state_width]
    
    # RMSNorm on nope dims (448)
    nope_normed = rms_norm(compressed_kv[:448])
    
    # FP8 UE8M0 quantization (nope, 448 → 7 blocks of 64)
    for i in range(7):
      block = nope_normed[i*64:(i+1)*64]
      absmax = max(abs(block))
      scale = 2^ceil(log2(absmax/6.0))
      ue8m0_scale[i] = log2(scale) + 127
      fp8_block[i] = block / scale  # packed as uint8
    
    # RoPE on rope dims (64, last dims)
    rope_rotated = apply_rope(compressed_kv[448:], position, cos_sin_cache)
    
    # Store to paged KV cache
    kv_slot = slot_mapping[token_idx]
    kv_block_idx = kv_slot // kv_cache_block_size
    kv_pos_in_block = kv_slot % kv_cache_block_size
    
    # FlashMLA layout (uint8): [fp8_nope (448) || bf16_rope (128) || ue8m0_scales (8)]
    kv_cache[kv_block_idx, kv_pos_in_block, 0:448] = fp8_nope
    kv_cache[kv_block_idx, kv_pos_in_block, 448:576] = rope_rotated (bf16)
    kv_cache[kv_block_idx, kv_pos_in_block, 576:584] = ue8m0_scales
  ```

**Kernel 2: Indexer (head=128, FP8 or MXFP4)**
- **Function**: `_fused_kv_compress_norm_rope_insert_indexer_attn` (FP8)
- **Function**: `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` (MXFP4)
- **File**: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` (lines 300+)
- **Differences from sparse**:
  - All 128 dims quantized to FP8 (no split nope/rope)
  - MXFP4 variant: 32-element blocks, 2 nibbles per byte, ue8m0 block scales
  - Single float32 scale per token (not per-block)
  - 1 block per token (no sliding window overlap)

---

#### 2b. Paged Cache Store with `slot_mapping` / `block_table`

**Metadata**:
- `slot_mapping`: [num_tokens] → global slot ID in paged cache
- `block_table`: [num_reqs, max_blocks_per_req] → block IDs for each request
- `block_size`: tokens per block (typically 16)

**Store Semantics**:
```
kv_slot = slot_mapping[token_idx]
kv_block_idx = kv_slot // block_size
kv_pos_in_block = kv_slot % block_size

# Write to paged cache
kv_cache[kv_block_idx, kv_pos_in_block, :] = quantized_kv
```

---

### Phase 3: Indexer-Cache Write (FP4 on SM100+)

**Indexer Compressor**: `DeepseekV4Indexer.forward()`
- **File**: `vllm/models/deepseek_v4/attention.py` (lines 661–800)
- **Compressor**: `DeepseekCompressor` (head_dim=128, compress_ratio=4)
  - Writes compressed KV to indexer cache via `compress_norm_rope_store_triton`
  - `use_fp4_cache=True` for Blackwell (SM100+)

**Indexer Q Quantization**: `fused_indexer_q_rope_quant()`
- **File**: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` (lines 290–438)
- **Kernel**: `_fused_indexer_q_rope_quant_kernel` (Triton)
- **Inputs**:
  - `index_q`: [num_tokens, num_heads, head_dim] (bf16)
  - `index_q_cos_sin_cache`: [max_pos, rope_head_dim] (cos || sin)
  - `index_weights`: [num_tokens, num_heads] (bf16, from weights_proj)
  
- **FP4 Path** (`use_fp4=True`):
  - MXFP4 block size: 32 elements
  - Per-block ue8m0 scale (2^(ue8m0 - 127))
  - Packed 2 nibbles per byte via inline PTX `cvt.rn.satfinite.e2m1x2.f32`
  - Output: `q_quant` [num_tokens, num_heads, head_dim // 2] (packed nibbles)
  - Output: `q_scale` [num_tokens, num_heads, head_dim // MXFP4_BLOCK_SIZE] (ue8m0 bytes)

---

## FUSED KERNELS & OPTIMIZATION

### Fused Q-Norm-RoPE-KV-Insert (SWA Path)

**Function**: `_fused_qnorm_rope_kv_insert()`
- **File**: `vllm/models/deepseek_v4/attention.py` (lines 507–594)
- **Dispatches to**:
  - `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (FlashMLA uint8 layout)
  - `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert` (FlashInfer bf16)
  - `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert` (FlashInfer per-tensor fp8)

**CUDA Kernel**: `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`
- **File**: `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` (57KB)
- **Fuses**:
  - Q side: per-head RMSNorm (no weight) + GPT-J RoPE on last 64 dims
  - KV side: GPT-J RoPE + UE8M0 FP8 quant (nope) + paged cache insert
  - One kernel, one grid; head-slot dispatch per warp
  
- **Constants** (hard-coded for DeepseekV4):
  - `HEAD_DIM = 512`
  - `ROPE_DIM = 64` (applied to dims [448, 512))
  - `NOPE_DIM = 448`
  - `QUANT_BLOCK = 64` (7 blocks per token)
  - `FP8_MAX = 224.0` (ROCm FNUZ gfx942) or `448.0` (OCP)
  - `is_neox = false` (GPT-J interleaved pairs)
  - `cos_sin_cache` layout: [max_pos, rope_dim] = cos || sin (cos first, sin second)
  
- **Cache Layout** (paged, per block):
  ```
  [0, bs*576):           token data (448 fp8 + 128 bf16 each)
  [bs*576, bs*576+bs*8): UE8M0 scales (7 real + 1 pad per token)
  ```

---

## MULTI-STREAM PARALLELIZATION

**Parallel Execution**: `maybe_execute_in_parallel()`
- **File**: `vllm/utils/multi_stream_utils.py`
- **Used in Indexer**: `DeepseekV4Indexer.forward()` (lines 770–790)
  ```python
  (q_quant, weights), k = maybe_execute_in_parallel(
      wq_b_and_q_quant,           # Q up-proj + fused_indexer_q_rope_quant
      lambda: compressor(...),     # Compressor (save_partial_states + compress_norm_rope_store)
      self.ln_events[0],           # Start event
      self.ln_events[1],           # Join event
      self.aux_stream,             # Auxiliary stream (None on ROCm)
  )
  ```
- **Overlap**: Q quantization and compressor KV write run in parallel on separate streams.

---

## RAW HAZARDS & PDL DISABLING

**Issue**: `save_partial_states` → `compress_norm_rope_store` read-after-write dependency
- `save_partial_states` writes to `state_cache`
- `compress_norm_rope_store` reads from `state_cache`
- No PDL grid-dependency primitives emitted by either kernel
- **Solution**: `launch_pdl=False` in `pdl_kwargs` (line 309, compressor.py)

**Code**:
```python
pdl_kwargs = (
    {}
    if current_platform.is_rocm() or current_platform.is_xpu()
    else {"launch_pdl": False}
)
```

---

## BLACKWELL-SPECIFIC OPTIMIZATIONS (SM100+)

### FP4 Indexer Cache (MXFP4)
- **Enabled via**: `use_fp4_cache=True` in `DeepseekV4Indexer.__init__()` (line 689)
- **Kernel**: `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`
- **Quantization**: MXFP4 block size 32, packed 2 nibbles per byte
- **PTX Inline ASM**: `cvt.rn.satfinite.e2m1x2.f32` (FP32 → FP4x2 packed)
- **Memory Savings**: ~2x vs FP8 (4 bits vs 8 bits per element)

### Fused Quant+Cache Kernels
- **CuTe DSL Kernel**: `compress_norm_rope_store_cutedsl` (head=512, CUDA only)
  - File: `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` (2164 lines)
  - Fuses compress → norm → RoPE → quant → store in single kernel
  - Better register reuse and memory coalescing on Blackwell

---

## FILE STRUCTURE & PERMALINKS

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Write Dispatcher** | `vllm/models/deepseek_v4/compressor.py` | 274–399 | Selects compress kernel, launches save_partial_states + compress_norm_rope_store |
| **Partial-State Write** | `vllm/models/deepseek_v4/common/ops/save_partial_states.py` | 9–101 | Triton kernel: writes raw KV/score to state_cache |
| **Compress+Norm+RoPE+Quant** | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | 31–666 | Triton launcher + 3 kernel variants (sparse FP8, indexer FP8, indexer MXFP4) |
| **Indexer Q Quant** | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | 290–438 | Triton kernel: Q RoPE + FP8/MXFP4 quant |
| **Fused Q-Norm-RoPE-KV-Insert** | `vllm/models/deepseek_v4/attention.py` | 507–594 | Dispatcher to CUDA ops (FlashMLA uint8, FlashInfer bf16/fp8) |
| **CUDA Kernel** | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | 1–57KB | Horizontally-fused Q/KV RoPE + quant + paged insert |
| **Indexer** | `vllm/models/deepseek_v4/attention.py` | 661–800 | Indexer forward: compressor + Q quant + sparse attention indexer |
| **CuTe DSL Kernel** | `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` | 2074+ | CuTe-based compress+norm+RoPE+quant for head=512 |

---

## GITHUB PERMALINKS (vLLM 687173877781670afde318491564bab92ac353aa)

- **save_partial_states**: https://github.com/vllm-project/vllm/blob/687173877781670afde318491564bab92ac353aa/vllm/models/deepseek_v4/common/ops/save_partial_states.py#L9-L101
- **compress_norm_rope_store_triton**: https://github.com/vllm-project/vllm/blob/687173877781670afde318491564bab92ac353aa/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py#L31-L106
- **_fused_kv_compress_norm_rope_insert_sparse_attn**: https://github.com/vllm-project/vllm/blob/687173877781670afde318491564bab92ac353aa/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py#L113-L300
- **DeepseekCompressor.forward**: https://github.com/vllm-project/vllm/blob/687173877781670afde318491564bab92ac353aa/vllm/models/deepseek_v4/compressor.py#L274-L399
- **fused_indexer_q_rope_quant**: https://github.com/vllm-project/vllm/blob/687173877781670afde318491564bab92ac353aa/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py#L290-L438
- **_fused_qnorm_rope_kv_insert**: https://github.com/vllm-project/vllm/blob/687173877781670afde318491564bab92ac353aa/vllm/models/deepseek_v4/attention.py#L507-L594
- **fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu**: https://github.com/vllm-project/vllm/blob/687173877781670afde318491564bab92ac353aa/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu#L1-L100
- **DeepseekV4Indexer.forward**: https://github.com/vllm-project/vllm/blob/687173877781670afde318491564bab92ac353aa/vllm/models/deepseek_v4/attention.py#L761-L800


---

## SGLang V4 IMPLEMENTATION (if available)

**Status**: SGLang V4 support is in development. Key test files found:
- `test/manual/kv_canary/test_self_e2e_baseline_dsv4.py` — E2E baseline test
- `test/manual/quant/test_deepseek_v32_fp4_4gpu.py` — FP4 quantization test

**Note**: SGLang's V4 implementation likely mirrors vLLM's architecture (save_partial_states → compress_norm_rope_store → paged cache store) but may use different kernel backends (e.g., SGLang's native Triton kernels vs vLLM's CuTe DSL).

---

## NVIDIA cuDNN DSA API (IndexerForward / IndexerTopK)

**Status**: NVIDIA's cuDNN DSA (Dynamic Sparse Attention) API is not directly exposed in vLLM's V4 implementation. Instead:
- vLLM uses **custom Triton/CUDA kernels** for indexer operations
- The `SparseAttnIndexer` class wraps the indexer logic
- **File**: `vllm/model_executor/layers/sparse_attn_indexer.py`

**Indexer Operations**:
1. **Top-K Selection**: `fused_indexer_q_rope_quant()` computes Q quantization + weights
2. **Sparse Attention**: Custom kernel selects top-K indices from weights
3. **KV Gather**: Gathers compressed KV from indexer cache using top-K indices

**Note**: NVIDIA's cuDNN DSA API (if used) would provide IndexerForward/IndexerTopK operations, but vLLM's current implementation uses custom kernels for tighter integration with the compressor state cache.

---

## COMPARISON: vLLM vs batchgen WRITE PATH

### vLLM Write Path (Reference)
1. **save_partial_states** (Triton): Raw KV/score → state_cache
2. **compress_norm_rope_store** (Triton/CuTe): state_cache → compress → norm → RoPE → quant → paged KV cache
3. **Paged cache store**: slot_mapping + block_table indexing
4. **Indexer Q quant** (Triton): Q → RoPE → FP8/MXFP4 quant
5. **Multi-stream overlap**: Q quant || compressor KV write

### Key Differences to Check in batchgen
- **State cache layout**: vLLM uses [num_blocks, block_size, 2*state_width] (float32)
- **Boundary triggering**: Compress only at positions where `position % compress_ratio == 0`
- **PDL disabling**: RAW hazard between save_partial_states and compress kernels
- **Paged cache indexing**: slot_mapping → block_idx / pos_in_block
- **FP8 quantization**: UE8M0 block-scaled (7 blocks of 64 for head=512)
- **MXFP4 packing**: 2 nibbles per byte via PTX inline ASM `cvt.rn.satfinite.e2m1x2.f32`
- **Multi-stream parallelization**: Separate streams for Q quant and compressor

---

## REFERENCES & SOURCES

1. **vLLM DeepSeek V4 Implementation**: https://github.com/vllm-project/vllm/tree/687173877781670afde318491564bab92ac353aa/vllm/models/deepseek_v4
2. **DeepSeek V4 Paper**: https://arxiv.org/abs/2501.12948 (if available)
3. **vLLM Attention Backends**: https://github.com/vllm-project/vllm/tree/687173877781670afde318491564bab92ac353aa/vllm/v1/attention/backends/mla
4. **Triton Documentation**: https://triton-lang.org/
5. **CuTe DSL**: https://github.com/NVIDIA/cutlass/tree/main/examples/cute

---

## SUMMARY

The DeepSeek V4 write path in vLLM consists of:

1. **Per-token partial-state staging** via `save_partial_states` (Triton)
   - Writes raw KV/score to state_cache with fused APE addition
   - PDL disabled due to RAW hazard with compress kernels

2. **Boundary-triggered compression** via `compress_norm_rope_store` (Triton/CuTe)
   - Reads state_cache at boundary positions (position % compress_ratio == 0)
   - Fuses compress → RMSNorm → RoPE → FP8/MXFP4 quant → paged cache store
   - Three kernel variants: sparse FP8 (head=512), indexer FP8 (head=128), indexer MXFP4 (head=128, Blackwell)

3. **Paged cache store** with slot_mapping / block_table
   - Converts global slot ID to block_idx / pos_in_block
   - Writes quantized KV to paged cache

4. **Indexer Q quantization** via `fused_indexer_q_rope_quant` (Triton)
   - Fuses Q RoPE + FP8/MXFP4 quant
   - MXFP4 uses PTX inline ASM for 2 nibbles per byte packing

5. **Multi-stream parallelization**
   - Q quant and compressor KV write run in parallel on separate streams
   - Joined before sparse attention indexer

**Blackwell-specific optimizations**:
- MXFP4 indexer cache (4 bits vs 8 bits, ~2x memory savings)
- CuTe DSL kernel for better register reuse and memory coalescing
- Fused quant+cache kernels for reduced kernel launch overhead

