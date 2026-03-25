"""CUDA implementation of fused q_absorb + query_states construction.

Replaces the Triton prototype with a CUDA kernel using warp-level reduction.

Grid: (H=64, B) -- one block per (head, batch) pair.
Each block has 512 threads, one per output C-dim element.
Each thread reduces across D=128 input elements.
Then 64 threads copy q_pe values.
"""

import torch
import batchgen_kernels.attention._C_fused_q_absorb as _module


def _load():
    return _module


def fused_q_absorb_query_states_cuda(
    q_nope: torch.Tensor,      # [bsz, H, D] squeezed
    q_absorb: torch.Tensor,    # [H, D, C]
    q_pe: torch.Tensor,        # [bsz, H, 1, R] contiguous
    output: torch.Tensor = None,
) -> torch.Tensor:
    """CUDA fused q_absorb + query_states construction.

    Returns: [bsz, 1, H, C+R] in flash_mla input layout.
    """
    B, H, D = q_nope.shape
    C = q_absorb.shape[2]
    R = q_pe.shape[-1]

    if output is None:
        output = torch.empty(B, 1, H, C + R, dtype=q_nope.dtype, device=q_nope.device)

    # Ensure contiguous
    q_nope = q_nope.contiguous()
    q_absorb = q_absorb.contiguous()
    q_pe = q_pe.contiguous()

    mod = _load()
    # The kernel writes to output with layout [B, H, C+R] (treating as [B, 1, H, C+R])
    mod.fused_q_absorb_forward(q_nope, q_absorb, q_pe, output.view(B, H, C + R))

    return output
