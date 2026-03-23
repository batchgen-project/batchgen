"""CUDA kernel: fused q_b output split into q_nope + q_pe (contiguous).

Replaces: q.view(bsz,1,H,q_head_dim).transpose(1,2) -> split([nope,rope]) -> q_pe.contiguous()

The q_b_proj output is [bsz, H*q_head_dim] = [bsz, 12288] where q_head_dim=192.
Each head's 192 dims split into q_nope[128] + q_pe[64].

This kernel reads the flat q_b output and writes:
  - q_nope: [bsz, H, nope_dim] contiguous  (for einsum in q_absorb)
  - q_pe:   [bsz, H, 1, rope_dim] contiguous (for RoPE kernel)

Grid: (H, bsz) -- one block per (head, batch)
Block: max(nope_dim, rope_dim) threads
"""

import torch
import batchgen_kernels.attention._C_fused_q_split as _module


def _load():
    return _module


def fused_q_split(
    q_flat: torch.Tensor,       # [bsz, H * q_head_dim] -- q_b_proj output
    num_heads: int = 64,
    nope_dim: int = 128,
    rope_dim: int = 64,
    q_nope: torch.Tensor = None,  # [bsz, H, nope_dim] pre-allocated
    q_pe: torch.Tensor = None,    # [bsz, H, 1, rope_dim] pre-allocated
):
    """Fused split of q_b output into q_nope + q_pe (both contiguous).

    Replaces:
        q = q_flat.view(bsz, 1, H, q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [nope_dim, rope_dim], dim=-1)
        q_pe = q_pe.contiguous()

    Returns:
        q_nope: [bsz, H, nope_dim] contiguous
        q_pe: [bsz, H, 1, rope_dim] contiguous
    """
    bsz = q_flat.shape[0]

    if q_nope is None:
        q_nope = torch.empty(bsz, num_heads, nope_dim, dtype=q_flat.dtype, device=q_flat.device)
    if q_pe is None:
        q_pe = torch.empty(bsz, num_heads, 1, rope_dim, dtype=q_flat.dtype, device=q_flat.device)

    mod = _load()
    mod.fused_q_split_forward(q_flat.contiguous(), q_nope, q_pe)

    return q_nope, q_pe
