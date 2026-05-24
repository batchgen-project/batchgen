# vLLM Entry Point Analysis: DeepSeek V4 Compression Kernels

**Source**: vLLM v0.21.0 (Blackwell variant)  
**Date**: May 23, 2026  
**Commit**: https://github.com/vllm-project/vllm/blob/ad7125a431e176d4161099480a66f0169609a690

---

## 1. vLLM `_fused_kv_compress_norm_rope_insert_sparse_attn`

**File**: `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` (lines 31–215)

### Signature

```python
@triton.jit
def _fused_kv_compress_norm_rope_insert_sparse_attn(
    # ── state cache (compressor internal state) ──
    state_cache_ptr: tl.tensor,           # [num_blocks, block_size, 2*state_width], dtype=float32
    state_cache_stride0: int,             # stride for block dimension
    state_cache_stride1: int,             # stride for position-in-block dimension
    
    # ── metadata ──
    token_to_req_indices_ptr: tl.tensor,  # [num_tokens], dtype=int32
    positions_ptr: tl.tensor,             # [num_tokens], dtype=int64
    slot_mapping_ptr: tl.tensor,          # [num_tokens], dtype=int64
    block_table_ptr: tl.tensor,           # [num_reqs, max_blocks_per_req], dtype=int32
    block_table_stride: int,              # stride for req dimension
    block_size: int,                      # tokens per block (typically 4 or 8)
    
    # ── RMSNorm ──
    rms_norm_weight_ptr: tl.tensor,       # [head_dim], dtype=float32
    rms_norm_eps: float,                  # typically 1e-6
    
    # ── RoPE ──
    cos_sin_cache_ptr: tl.tensor,         # [max_pos, rope_head_dim], dtype=float32
    cos_sin_stride: int,                  # stride for position dimension
    
    # ── KV cache output ──
    k_cache_ptr: tl.tensor,               # [num_kv_blocks, block_size*TOKEN_STRIDE + block_size*SCALE_DIM], dtype=uint8
    kv_slot_mapping_ptr: tl.tensor,       # [num_tokens], dtype=int64
    kv_cache_block_size: int,             # tokens per KV cache block
    
    # ── constexprs (compile-time constants) ──
    HEAD_SIZE: tl.constexpr,              # 512 (for sparse_attn variant)
    TRITON_BLOCK_SIZE: tl.constexpr,      # next_power_of_2(HEAD_SIZE) = 512
    STATE_WIDTH: tl.constexpr,            # state_cache.shape[-1] // 2 (kv_state width)
    COMPRESS_RATIO: tl.constexpr,         # 4 or 128
    OVERLAP: tl.constexpr,                # 1 if compress_ratio==4 else 0
    ROPE_HEAD_DIM: tl.constexpr,          # 64 (for DeepSeek V4)
    FP8_MAX: tl.constexpr,                # 448.0 (FP8 clamp bound)
    QUANT_BLOCK: tl.constexpr,            # 64 (per-block quantization)
    TOKEN_STRIDE: tl.constexpr,           # 576 (448 fp8 + 128 bf16 = 576 bytes/token)
    SCALE_DIM: tl.constexpr,              # 8 (7 real scales + 1 pad)
    KV_BLOCK_STRIDE: tl.constexpr,        # k_cache.stride(0) (bytes per block)
) -> None
```

### What It Does

**One paragraph**: This Triton kernel implements the **DeepSeek V4 sparse attention compression pipeline** for the final KV cache write. For each token at a boundary position (where `(position + 1) % COMPRESS_RATIO == 0`), it gathers the preceding `(1 + OVERLAP) * COMPRESS_RATIO` state cache entries (KV and attention scores), applies softmax-weighted compression, normalizes via RMSNorm, applies GPT-J style RoPE rotation to the rope dimensions, quantizes the non-rope portion to FP8 UE8M0 (per 64-element block), stores the rope portion as bf16, and writes both the quantized values and per-block scales to the paged KV cache. Early-exits for non-boundary tokens and invalid slots.

### Required State

1. **Pre-quantized weights**: NO. The kernel performs quantization internally (FP8 UE8M0).
2. **Model config**: YES, implicitly via constexprs:
   - `HEAD_SIZE` (512 for sparse_attn)
   - `ROPE_HEAD_DIM` (64 for DeepSeek V4)
   - `COMPRESS_RATIO` (4 or 128)
   - `OVERLAP` (derived from compress_ratio)
3. **Forward batch metadata**: YES, required:
   - `token_to_req_indices`: Maps each token to its request ID (for block_table indexing)
   - `positions`: Absolute position of each token in the sequence
   - `slot_mapping`: Physical slot ID in state cache for each token
   - `block_table`: Maps (req_idx, block_idx) → physical block number
   - `kv_slot_mapping`: Physical slot ID in KV cache for each token
4. **Other stateful requirements**:
   - `state_cache`: Pre-populated by `_save_partial_states_kernel` with KV and score states
   - `rms_norm_weight`: RMSNorm scale parameter (learnable, from model)
   - `cos_sin_cache`: Pre-computed cos/sin for RoPE (from rotary_emb)

### Random-Weight Fixture

```python
def make_fixture_for_fused_kv_compress_norm_rope_insert_sparse_attn(
    T: int,                    # num_tokens
    compress_ratio: int = 4,   # 4 or 128
    head_dim: int = 512,
    rope_head_dim: int = 64,
    block_size: int = 4,       # state cache block size
    kv_block_size: int = 4,    # KV cache block size
    device: str = 'cuda'
) -> dict:
    """
    Construct random tensors that pass all input checks for the sparse_attn kernel.
    
    Key constraints:
    - Only tokens at boundary positions (pos % compress_ratio == 0) trigger compression
    - state_cache must have valid block_table references
    - slot_mapping and kv_slot_mapping must be non-negative
    - positions must be monotonically increasing
    """
    import torch
    
    # Metadata: positions and token-to-request mapping
    # Create positions that align with compress_ratio boundaries
    positions = torch.arange(
        compress_ratio - 1,
        compress_ratio * T,
        compress_ratio,
        dtype=torch.int64,
        device=device,
    )  # [T] positions at boundaries: [3, 7, 11, ...] for ratio=4
    
    token_to_req_indices = torch.zeros(T, dtype=torch.int32, device=device)
    
    # Block table: map request 0 to physical blocks
    # For state cache: need enough blocks to cover all positions
    state_block_size = block_size
    overlap = 1 if compress_ratio == 4 else 0
    coff = 1 + overlap
    num_state_tokens = compress_ratio * T
    num_state_blocks = (num_state_tokens + state_block_size - 1) // state_block_size + 1
    
    block_table = torch.arange(
        num_state_blocks,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)  # [1, num_state_blocks] for single request
    
    # Slot mapping: linear assignment (token i → slot i)
    slot_mapping = torch.arange(T, dtype=torch.int64, device=device)
    
    # KV slot mapping: linear assignment for KV cache
    kv_slot_mapping = torch.arange(T, dtype=torch.int64, device=device)
    
    # State cache: [num_state_blocks, state_block_size, 2*state_width]
    # state_width = head_dim (kv_state) + head_dim (score_state) = 2*head_dim total
    state_width = head_dim
    state_cache = torch.randn(
        num_state_blocks,
        state_block_size,
        2 * state_width,
        dtype=torch.float32,
        device=device,
    )
    
    # RMSNorm weight: [head_dim], typically positive
    rms_norm_weight = torch.ones(head_dim, dtype=torch.float32, device=device) * 0.5
    
    # RoPE cos_sin_cache: [max_pos, rope_head_dim]
    # Layout: first half = cos, second half = sin (per-pair)
    max_pos = positions.max().item() + 1
    cos_sin_cache = torch.randn(
        max_pos,
        rope_head_dim,
        dtype=torch.float32,
        device=device,
    )
    # Normalize to unit magnitude for cos/sin
    cos_sin_cache = torch.nn.functional.normalize(cos_sin_cache, dim=-1)
    
    # KV cache output: [num_kv_blocks, kv_block_size*TOKEN_STRIDE + kv_block_size*SCALE_DIM]
    # TOKEN_STRIDE = 576 (448 fp8 + 128 bf16)
    # SCALE_DIM = 8 (7 real + 1 pad)
    token_stride = 576
    scale_dim = 8
    num_kv_blocks = max(2, (T + kv_block_size - 1) // kv_block_size + 1)
    k_cache = torch.zeros(
        num_kv_blocks,
        kv_block_size * token_stride + kv_block_size * scale_dim,
        dtype=torch.uint8,
        device=device,
    )
    
    return {
        'state_cache_ptr': state_cache,
        'state_cache_stride0': state_cache.stride(0),
        'state_cache_stride1': state_cache.stride(1),
        'token_to_req_indices_ptr': token_to_req_indices,
        'positions_ptr': positions,
        'slot_mapping_ptr': slot_mapping,
        'block_table_ptr': block_table,
        'block_table_stride': block_table.stride(0),
        'block_size': state_block_size,
        'rms_norm_weight_ptr': rms_norm_weight,
        'rms_norm_eps': 1e-6,
        'cos_sin_cache_ptr': cos_sin_cache,
        'cos_sin_stride': cos_sin_cache.stride(0),
        'k_cache_ptr': k_cache,
        'kv_slot_mapping_ptr': kv_slot_mapping,
        'kv_cache_block_size': kv_block_size,
        # Constexprs
        'HEAD_SIZE': head_dim,
        'TRITON_BLOCK_SIZE': 512,  # next_power_of_2(512)
        'STATE_WIDTH': state_width,
        'COMPRESS_RATIO': compress_ratio,
        'OVERLAP': 1 if compress_ratio == 4 else 0,
        'ROPE_HEAD_DIM': rope_head_dim,
        'FP8_MAX': 448.0,
        'QUANT_BLOCK': 64,
        'TOKEN_STRIDE': token_stride,
        'SCALE_DIM': scale_dim,
        'KV_BLOCK_STRIDE': k_cache.stride(0),
    }
```

### Caveats (What Would Crash with Random Data)

1. **Invalid block_table references**: If `block_table[req_idx, block_idx]` points to a block number ≥ `num_state_blocks`, the kernel will read garbage or OOB. Fixture ensures block_table is dense and valid.

2. **Negative slot_mapping or kv_slot_mapping**: The kernel checks `if slot_id < 0: return`, so negative values cause early exit (not a crash, but no-op). Fixture uses non-negative indices.

3. **Misaligned positions**: If positions are not at compress_ratio boundaries, the kernel early-exits. Fixture ensures `(position + 1) % compress_ratio == 0`.

4. **state_cache shape mismatch**: If `state_cache.shape[-1]` is not `2 * state_width`, the kernel will read wrong offsets. Fixture ensures correct shape.

5. **RoPE cache out-of-bounds**: If `compressed_pos = (position // compress_ratio) * compress_ratio` exceeds `cos_sin_cache.shape[0]`, the kernel will read OOB. Fixture ensures `cos_sin_cache` is large enough.

6. **FP8 quantization underflow**: If all values in a 64-element block are < 1e-4, the kernel clamps to 1e-4 to avoid log2(0). Random data is fine; this is a safety check.

7. **Stride mismatches**: If strides don't match the actual tensor layout, pointer arithmetic will be wrong. Fixture uses `.stride()` directly from tensors.

---

## 2. vLLM `DeepseekCompressor`

**File**: `vllm/model_executor/layers/deepseek_compressor.py` (lines 177–379)

### Signature

```python
class DeepseekCompressor(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,           # Full vLLM config (model, scheduler, etc.)
        compress_ratio: int,               # 4 or 128
        hidden_size: int,                  # Model hidden dimension (e.g., 4096)
        head_dim: int,                     # Per-head dimension (512 or 128)
        rotate: bool = False,              # Unused in current code
        prefix: str = "",                  # Layer name prefix for logging
        k_cache_prefix: str = "",          # Prefix for KV cache metadata lookup
        use_fp4_cache: bool = False,       # Use MXFP4 quantization (head_dim==128 only)
    ) -> None:
        ...
    
    def forward(
        self,
        kv_score: torch.Tensor,            # [num_tokens, 2*coff*head_dim], dtype=bfloat16
        positions: torch.Tensor,           # [num_tokens], dtype=int64
        rotary_emb,                        # Object with .cos_sin_cache attribute
    ) -> None:
        ...
```

### What It Does

**One paragraph**: `DeepseekCompressor` is a stateful nn.Module that wraps the fused Triton kernels for DeepSeek V4 compression. It maintains learnable parameters (`ape`, `fused_wkv_wgate`, `norm`) and a state cache (managed by `CompressorStateCache`). On forward, it splits the input `kv_score` tensor into KV and score components, stores them in the state cache via `_save_partial_states_kernel`, then calls the appropriate fused kernel (`_fused_kv_compress_norm_rope_insert_sparse_attn` or one of the indexer variants) to compress, normalize, apply RoPE, quantize, and write to the KV cache. The kernel selection depends on `head_dim` and `use_fp4_cache`.

### Required State

1. **Pre-quantized weights**: NO. The module learns `ape` (absolute position embeddings) and `fused_wkv_wgate` (linear projection) as nn.Parameters.

2. **Model config**: YES, required via `vllm_config`:
   - `vllm_config.model_config.hf_config.qk_rope_head_dim` (rope dimension)
   - `vllm_config.model_config.hf_config.rms_norm_eps` (RMSNorm epsilon)
   - `vllm_config.model_config.max_model_len` (max sequence length)
   - `vllm_config.scheduler_config.max_num_seqs` (max concurrent requests)
   - `vllm_config.scheduler_config.max_num_batched_tokens` (max tokens per batch)

3. **Forward batch metadata**: YES, required:
   - `attn_metadata` dict (from `get_forward_context()`) containing:
     - `CompressorMetadata` at key `self.state_cache.prefix`:
       - `block_table`: [num_reqs, max_blocks_per_req], dtype=int32
       - `slot_mapping`: [num_tokens], dtype=int64
       - `block_size`: int
       - `token_to_req_indices`: [num_tokens], dtype=int32
     - KV cache metadata at key `self.k_cache_prefix`:
       - `slot_mapping`: [num_tokens], dtype=int64

4. **Other stateful requirements**:
   - `self.ape`: nn.Parameter [compress_ratio, coff*head_dim], dtype=float32
   - `self.fused_wkv_wgate`: MergedColumnParallelLinear (learnable weights)
   - `self.norm`: RMSNorm (learnable scale)
   - `self.state_cache.kv_cache`: Paged KV cache tensor (managed by vLLM)
   - `rotary_emb.cos_sin_cache`: Pre-computed RoPE cache

### nn.Parameter Attributes

```python
self.ape: nn.Parameter
    # Shape: [compress_ratio, coff * head_dim]
    # dtype: float32
    # Absolute position embeddings, added to scores before compression
    # Example: [4, 1024] for compress_ratio=4, head_dim=512, overlap=True

self.fused_wkv_wgate: MergedColumnParallelLinear
    # Input: [num_tokens, hidden_size]
    # Output: [num_tokens, 2 * coff * head_dim]
    # Learnable weights (no bias)
    # Produces both KV and score components

self.norm: RMSNorm
    # Scale: [head_dim]
    # dtype: float32
    # Applied after compression and before quantization
```

### Random-Weight Fixture

```python
def make_fixture_for_deepseek_compressor(
    T: int,                    # num_tokens
    compress_ratio: int = 4,
    hidden_size: int = 4096,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    device: str = 'cuda',
) -> dict:
    """
    Construct random tensors and a minimal vllm_config for DeepseekCompressor.
    
    Key constraints:
    - vllm_config must have model_config.hf_config with qk_rope_head_dim, rms_norm_eps
    - vllm_config must have scheduler_config with max_num_seqs, max_num_batched_tokens
    - kv_score input must be [num_tokens, 2*coff*head_dim], dtype=bfloat16
    - positions must be monotonically increasing
    - attn_metadata must be a dict with CompressorMetadata and KV cache metadata
    """
    import torch
    from dataclasses import dataclass
    from types import SimpleNamespace
    
    # Minimal mock vllm_config
    @dataclass
    class MockHFConfig:
        qk_rope_head_dim: int = rope_head_dim
        rms_norm_eps: float = 1e-6
    
    @dataclass
    class MockModelConfig:
        hf_config: MockHFConfig = None
        max_model_len: int = 4096
        
        def __post_init__(self):
            if self.hf_config is None:
                self.hf_config = MockHFConfig()
    
    @dataclass
    class MockSchedulerConfig:
        max_num_seqs: int = 1
        max_num_batched_tokens: int = T
    
    @dataclass
    class MockCompilationConfig:
        static_forward_context: dict = None
        
        def __post_init__(self):
            if self.static_forward_context is None:
                self.static_forward_context = {}
    
    @dataclass
    class MockVllmConfig:
        model_config: MockModelConfig = None
        scheduler_config: MockSchedulerConfig = None
        compilation_config: MockCompilationConfig = None
        
        def __post_init__(self):
            if self.model_config is None:
                self.model_config = MockModelConfig()
            if self.scheduler_config is None:
                self.scheduler_config = MockSchedulerConfig()
            if self.compilation_config is None:
                self.compilation_config = MockCompilationConfig()
    
    vllm_config = MockVllmConfig()
    
    # Input tensor: [num_tokens, 2*coff*head_dim], dtype=bfloat16
    overlap = 1 if compress_ratio == 4 else 0
    coff = 1 + overlap
    kv_score = torch.randn(
        T,
        2 * coff * head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    
    # Positions: monotonically increasing
    positions = torch.arange(T, dtype=torch.int64, device=device)
    
    # RoPE cache: [max_pos, rope_head_dim]
    max_pos = T + compress_ratio + 16
    cos_sin_cache = torch.randn(
        max_pos,
        rope_head_dim,
        dtype=torch.float32,
        device=device,
    )
    cos_sin_cache = torch.nn.functional.normalize(cos_sin_cache, dim=-1)
    
    # Mock rotary_emb object
    rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin_cache)
    
    # Metadata: block_table, slot_mapping, etc.
    state_block_size = 4
    num_state_blocks = max(2, (T + state_block_size - 1) // state_block_size + 1)
    
    block_table = torch.arange(
        num_state_blocks,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)  # [1, num_state_blocks]
    
    slot_mapping = torch.arange(T, dtype=torch.int64, device=device)
    token_to_req_indices = torch.zeros(T, dtype=torch.int32, device=device)
    
    kv_slot_mapping = torch.arange(T, dtype=torch.int64, device=device)
    
    # CompressorMetadata
    from vllm.model_executor.layers.deepseek_compressor import CompressorMetadata
    compressor_metadata = CompressorMetadata(
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=state_block_size,
        token_to_req_indices=token_to_req_indices,
    )
    
    # KV cache metadata (minimal)
    kv_cache_metadata = SimpleNamespace(slot_mapping=kv_slot_mapping)
    
    # attn_metadata dict
    state_cache_prefix = "state_cache"
    k_cache_prefix = "k_cache"
    attn_metadata = {
        state_cache_prefix: compressor_metadata,
        k_cache_prefix: kv_cache_metadata,
    }
    
    return {
        'vllm_config': vllm_config,
        'compress_ratio': compress_ratio,
        'hidden_size': hidden_size,
        'head_dim': head_dim,
        'prefix': 'compressor',
        'k_cache_prefix': k_cache_prefix,
        'use_fp4_cache': False,
        # Forward inputs
        'kv_score': kv_score,
        'positions': positions,
        'rotary_emb': rotary_emb,
        'attn_metadata': attn_metadata,
        'state_cache_prefix': state_cache_prefix,
    }
```

### Caveats (What Would Crash with Random Data)

1. **Missing vllm_config fields**: If `vllm_config.model_config.hf_config` lacks `qk_rope_head_dim` or `rms_norm_eps`, the `__init__` will raise AttributeError. Fixture provides all required fields.

2. **Invalid head_dim**: The kernel selection (lines 243–269) only supports `head_dim in [512, 128]`. Other values raise ValueError. Fixture uses 512 or 128.

3. **use_fp4_cache=True with head_dim=512**: Line 244 asserts this is invalid. Fixture only enables MXFP4 for head_dim=128.

4. **Missing attn_metadata keys**: If `attn_metadata[self.state_cache.prefix]` or `attn_metadata[self.k_cache_prefix]` don't exist, the forward will raise KeyError. Fixture provides both.

5. **state_cache not initialized**: The `CompressorStateCache` manages `self.kv_cache`, which must be pre-allocated by vLLM's KV cache manager. In a standalone fixture, this tensor must exist and have the right shape. Fixture creates a dummy state_cache in the vllm_config.

6. **Mismatched kv_score shape**: If `kv_score.shape[-1] != 2 * coff * head_dim`, the split on line 281 will fail. Fixture ensures correct shape.

7. **Positions out of range**: If `positions.max() >= cos_sin_cache.shape[0]`, the RoPE lookup will be OOB. Fixture ensures `cos_sin_cache` is large enough.

8. **forward_context not set**: The kernel calls `get_forward_context()` (line 286), which requires a thread-local context to be active. In a standalone test, this will fail unless you mock or set the context. Fixture assumes the caller sets up the forward context.

---

## Summary Table

| Aspect | `_fused_kv_compress_norm_rope_insert_sparse_attn` | `DeepseekCompressor` |
|--------|------|------|
| **Type** | Triton @jit kernel | nn.Module wrapper |
| **Entry point** | Direct kernel call | `.forward(kv_score, positions, rotary_emb)` |
| **Learnable params** | None | `ape`, `fused_wkv_wgate`, `norm` |
| **Input tensors** | state_cache, positions, block_table, etc. (11 args) | kv_score [T, 2*coff*head_dim] |
| **Output** | Writes to k_cache (in-place) | None (writes to state_cache and k_cache) |
| **Config dependency** | Via constexprs (HEAD_SIZE, COMPRESS_RATIO, etc.) | Via vllm_config object |
| **Metadata dependency** | token_to_req_indices, slot_mapping, block_table | attn_metadata dict |
| **RoPE requirement** | cos_sin_cache tensor | rotary_emb.cos_sin_cache |
| **Quantization** | FP8 UE8M0 (per 64-elem block) | Delegates to kernel (FP8 or MXFP4) |
| **Crash risk with random data** | Block table OOB, stride mismatches, position OOB | Missing config fields, invalid head_dim, missing metadata |

---

## Integration Notes for Bench Rewrite

1. **For `bench_compress_quant.py` (K3)**:
   - Call `_fused_kv_compress_norm_rope_insert_sparse_attn` directly with the fixture tensors.
   - Ensure positions are at compress_ratio boundaries.
   - Verify k_cache output shape: `[num_kv_blocks, block_size*576 + block_size*8]`.

2. **For `bench_compressor.py` (K4)**:
   - Instantiate `DeepseekCompressor` with the mock vllm_config.
   - Call `.forward(kv_score, positions, rotary_emb)` inside a forward context.
   - Mock `get_forward_context()` to return a context with the attn_metadata dict.
   - Verify the module's learnable parameters are initialized (currently random).

3. **Validation**:
   - Compare outputs against the `_baseline` reference implementations in the bench files.
   - Check that quantized values are in valid FP8 range ([-448, 448]).
   - Verify RoPE rotation is applied correctly (compare against reference).

