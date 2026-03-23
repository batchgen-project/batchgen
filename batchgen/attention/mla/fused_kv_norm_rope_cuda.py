"""CUDA kernel: fused RMSNorm + RoPE on KV and Q + cache write.

Replaces the Triton `fused_rmsnorm_rope_cache_update_with_q_return_new_kv` kernel.

Per batch element (grid = bsz):
  1. RMSNorm on KV lora slice [512] -> normalized KV [512]
  2. RoPE on KV rope slice [64] -> rotated k_pe [64]
  3. RoPE on all 64 Q heads' rope slices [64 x 64] -- modifies q_pe in-place
  4. Write normalized+rotated KV [576] to flat cache at position
  5. Return offload_kv [bsz, 1, 576] for downstream (q_absorb etc.)

Block: 256 threads
  - Warp 0-3 (128 threads): handle RMSNorm + KV RoPE + cache write
  - Warp 4-7 (128 threads): help with Q head RoPE (64 heads x 64 dims)
"""

import torch
import batchgen_kernels.attention._C_fused_kv_norm_rope as _module


def _load():
    return _module


def fused_kv_norm_rope_cache_cuda(
    new_compressed_kv: torch.Tensor,  # [bsz, 1, 576]
    flat_cache: torch.Tensor,         # [bsz, max_seq_len, 576] -- flat KV cache
    q_pe: torch.Tensor,               # [bsz, H, 1, rope_dim] -- modified in-place
    cos: torch.Tensor,                # [max_pos, rope_dim]
    sin: torch.Tensor,                # [max_pos, rope_dim]
    position_ids: torch.Tensor,       # [bsz, 1]
    norm_weight: torch.Tensor,        # [kv_lora_rank]
    kv_lora_rank: int = 512,
    rope_dim: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CUDA fused RMSNorm + RoPE + cache write.

    Equivalent to fused_rmsnorm_rope_cache_update_with_q_return_new_kv (Triton).
    Returns offload_kv [bsz, 1, 576].
    """
    mod = _load()
    return mod.fused_kv_norm_rope_cache_forward(
        new_compressed_kv.contiguous(),
        flat_cache.contiguous(),
        q_pe.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        position_ids.contiguous(),
        norm_weight.contiguous(),
        kv_lora_rank,
        rope_dim,
        eps,
    )
