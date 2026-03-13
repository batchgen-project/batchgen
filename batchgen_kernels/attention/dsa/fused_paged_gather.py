"""Fused Paged KV Gather — H20 (SM90a) Only

Replaces pure-torch sparse_gather_from_paged_kv with Triton kernel.
Fuses: token→page/offset conversion, block_table lookup, and gather into one kernel.

Two variants:
  1. MLA gather: [B, topk] entries of dim=576 from paged MLA cache
  2. Aux gather: [B, max_seqlen] entries of dim=128 from paged indexer cache

Current baseline: sparse_gather = 9.04ms (4.9% of 183ms decode)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_gather_kernel(
    blocked_kv_ptr,     # [num_pages * page_size, D] flattened
    block_table_ptr,    # [B, max_pages_per_seq]
    token_indices_ptr,  # [B, num_tokens]
    out_ptr,            # [B, num_tokens, D]
    num_tokens,         # runtime value (not constexpr — can be large)
    D: tl.constexpr,
    page_size: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    BLOCK_T: tl.constexpr,   # tokens per program
    BLOCK_D: tl.constexpr,   # D elements per program (should cover full D)
):
    """Each program gathers BLOCK_T tokens for one batch element.

    Grid: (B, cdiv(num_tokens, BLOCK_T))
    Each program handles tokens [t_start, t_start+BLOCK_T) for batch b.
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    t_start = pid_t * BLOCK_T
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    for ti in range(BLOCK_T):
        t = t_start + ti
        if t < num_tokens:
            # Read token index
            token_idx = tl.load(token_indices_ptr + pid_b * num_tokens + t)

            # Page lookup
            logical_page = token_idx // page_size
            page_offset = token_idx % page_size
            logical_page = tl.minimum(tl.maximum(logical_page, 0), max_pages_per_seq - 1)

            physical_page = tl.load(block_table_ptr + pid_b * max_pages_per_seq + logical_page)
            physical_page = tl.maximum(physical_page, 0)

            flat_idx = physical_page * page_size + page_offset

            # Vectorized copy
            src = blocked_kv_ptr + flat_idx * D + d_offs
            vals = tl.load(src, mask=d_mask, other=0.0)

            dst = out_ptr + (pid_b * num_tokens + t) * D + d_offs
            tl.store(dst, vals, mask=d_mask)


def fused_paged_gather(
    blocked_kv: torch.Tensor,       # [num_pages, page_size, num_heads, head_dim]
    block_table: torch.Tensor,      # [B, max_pages_per_seq]
    token_indices: torch.Tensor,    # [B, num_tokens]
    page_size: int,
) -> torch.Tensor:
    """Triton fused paged KV gather."""
    token_indices = token_indices.contiguous()
    B, num_tokens = token_indices.shape
    num_heads = blocked_kv.shape[2]
    head_dim = blocked_kv.shape[3]
    D = num_heads * head_dim
    max_pages_per_seq = block_table.shape[1]

    blocked_flat = blocked_kv.reshape(-1, D)
    out = torch.empty(B, num_tokens, D, dtype=blocked_kv.dtype, device=blocked_kv.device)

    BLOCK_D = triton.next_power_of_2(D)
    # Coarser granularity: each program handles BLOCK_T tokens
    # Target ~1024-4096 programs total for good SM utilization
    total_work = B * num_tokens
    if total_work <= 4096:
        BLOCK_T = 1
    elif total_work <= 32768:
        BLOCK_T = 8
    else:
        BLOCK_T = 32

    grid = (B, triton.cdiv(num_tokens, BLOCK_T))

    _paged_gather_kernel[grid](
        blocked_flat, block_table, token_indices,
        out,
        num_tokens,
        D=D, page_size=page_size, max_pages_per_seq=max_pages_per_seq,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )

    return out.view(B, num_tokens, num_heads, head_dim)


def fused_indexer_gather(
    indexer_blocked_k: torch.Tensor,  # [num_pages, page_size, 1, 128]
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    """Gather indexer K from paged cache for all valid positions."""
    B = block_table.shape[0]
    token_indices = torch.arange(
        max_seqlen, device=block_table.device, dtype=torch.int32,
    ).unsqueeze(0).expand(B, -1).contiguous()

    return fused_paged_gather(
        indexer_blocked_k, block_table, token_indices, page_size,
    )


# ============================================================================
# PyTorch reference (for validation)
# ============================================================================

def paged_gather_reference(
    blocked_kv: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Reference: pure torch paged gather."""
    B, topk = token_indices.shape
    num_heads = blocked_kv.shape[2]
    head_dim = blocked_kv.shape[3]

    logical_page_idx = token_indices // page_size
    page_offset = token_indices % page_size
    max_pages = block_table.shape[1]
    logical_page_idx = logical_page_idx.clamp(max=max_pages - 1)
    physical_page_idx = torch.gather(block_table, 1, logical_page_idx.long())
    physical_page_idx = physical_page_idx.clamp(min=0)
    flat_idx = physical_page_idx * page_size + page_offset
    blocked_flat = blocked_kv.reshape(-1, num_heads * head_dim)
    gathered = blocked_flat[flat_idx.reshape(-1).long()]
    return gathered.view(B, topk, num_heads, head_dim)
