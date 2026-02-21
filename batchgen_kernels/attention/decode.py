"""Paged decode attention kernel — BF16, SM90+ (WGMMA + TMA).

Supports head_dim=64 (gpt-oss-120b) and head_dim=128.
Ported from hpc-ops Hunyuan decode kernel.

Usage:
    from batchgen_kernels.attention import attention_decode_bf16

    out = attention_decode_bf16(q, kcache, vcache, block_ids, num_seq_kvcache)
"""

import torch

# Import triggers compilation on first use
import batchgen_kernels.attention._C_gqa_mha_decode  # noqa: F401


def attention_decode_bf16(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    block_ids: torch.Tensor,
    num_seq_kvcache: torch.Tensor,
    new_kv_included: bool = False,
    use_splitk: bool = False,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """BF16 paged decode attention.

    Args:
        q: [num_batch, num_head_q, head_dim] BF16
        kcache: [num_kvcache_blocks, block_size, num_head_k, head_dim] BF16
        vcache: [num_kvcache_blocks, block_size, num_head_v, head_dim] BF16
        block_ids: [num_batch, num_seq_max_blocks] INT32
        num_seq_kvcache: [num_batch] INT32
        new_kv_included: whether new KV is appended in-place
        use_splitk: enable split-K for small batch sizes
        output: optional pre-allocated output tensor

    Returns:
        [num_batch, num_head_q, head_dim] BF16
    """
    return torch.ops.hpc_decode.attention_decode_bf16(
        q, kcache, vcache, block_ids, num_seq_kvcache,
        new_kv_included, use_splitk, output,
    )
