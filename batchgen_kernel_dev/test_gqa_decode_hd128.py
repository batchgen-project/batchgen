"""Standalone test: batchgen_gqa_decode_bf16 with head_dim=128 (MiniMax config).

Compares WGMMA decode kernel vs FlashAttention reference for correctness.
Run on H20 (SM90): python test_gqa_decode_hd128.py

MiniMax-M2.5 attention config:
- num_q_heads=48, num_kv_heads=8, head_dim=128
- GQA ratio 6:1
- Page size from gpu_kv_manager (typically 64)
"""

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gqa_decode_hd128(
    batch_sizes=[1, 4, 8],
    seq_lens=[64, 256, 1024, 4096],
    num_q_heads=48,
    num_kv_heads=8,
    head_dim=128,
    page_size=64,
):
    """Test WGMMA decode kernel with head_dim=128, GQA 6:1."""
    device = "cuda"
    dtype = torch.bfloat16

    # Import kernels
    try:
        from batchgen.attention.gqa import gqa_decode_fa
        logger.info("FlashAttention gqa_decode_fa loaded")
    except ImportError as e:
        logger.error(f"Cannot import gqa_decode_fa: {e}")
        return

    try:
        from batchgen.attention.gqa.batchgen_gqa_decode_bf16 import batchgen_gqa_decode_bf16
        logger.info("WGMMA batchgen_gqa_decode_bf16 loaded")
    except ImportError as e:
        logger.error(f"Cannot import batchgen_gqa_decode_bf16: {e}")
        return

    total_pass = 0
    total_tests = 0

    for batch in batch_sizes:
        for seq_len in seq_lens:
            total_tests += 1
            num_pages = (seq_len + page_size - 1) // page_size
            total_pages = batch * num_pages + 1  # +1 for safety

            # Create inputs
            q = torch.randn(batch, 1, num_q_heads, head_dim, device=device, dtype=dtype) * 0.1
            k_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, device=device, dtype=dtype) * 0.1
            v_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, device=device, dtype=dtype) * 0.1
            cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
            block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
            # Offset block_table per batch to avoid overlap
            for b in range(batch):
                block_table[b] = block_table[b] + b * num_pages

            # Reference: FlashAttention
            try:
                out_fa, _ = gqa_decode_fa(
                    q=q, k_cache=k_cache, v_cache=v_cache,
                    cache_seqlens=cache_seqlens, block_table=block_table,
                )
            except Exception as e:
                logger.warning(f"FA failed bsz={batch} seq={seq_len}: {e}")
                continue

            # Test: WGMMA kernel
            try:
                out_wgmma, _ = batchgen_gqa_decode_bf16(
                    q=q, k_cache=k_cache, v_cache=v_cache,
                    cache_seqlens=cache_seqlens, block_table=block_table,
                )
            except Exception as e:
                logger.error(f"WGMMA FAILED bsz={batch} seq={seq_len}: {e}")
                continue

            # Compare
            diff = (out_fa.float() - out_wgmma.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Cosine similarity
            fa_flat = out_fa.float().view(-1)
            wgmma_flat = out_wgmma.float().view(-1)
            cos_sim = torch.nn.functional.cosine_similarity(fa_flat.unsqueeze(0), wgmma_flat.unsqueeze(0)).item()

            # Check: max_diff < 0.05 and cos_sim > 0.999
            ok = max_diff < 0.05 and cos_sim > 0.999
            status = "PASS" if ok else "FAIL"
            if ok:
                total_pass += 1

            logger.info(
                f"[{status}] bsz={batch:2d} seq={seq_len:5d} | "
                f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} cos_sim={cos_sim:.6f}"
            )

            if not ok:
                # Print per-head breakdown for debugging
                per_head_diff = diff.squeeze(1).max(dim=-1)[0].mean(dim=0)  # [num_q_heads]
                worst_heads = per_head_diff.topk(3)
                logger.info(f"  Worst heads: {worst_heads.indices.tolist()} diffs: {worst_heads.values.tolist()}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {total_pass}/{total_tests} PASS")
    logger.info(f"Config: num_q_heads={num_q_heads} num_kv_heads={num_kv_heads} head_dim={head_dim} page_size={page_size}")
    logger.info(f"{'='*60}")

    return total_pass == total_tests


if __name__ == "__main__":
    ok = test_gqa_decode_hd128()
    exit(0 if ok else 1)
