"""FP8 page-split indexer cache + deep_gemm paged scoring for GLM-5 DSA.

Replaces the per-decode-step full-context BF16 gather (`fused_dense_paged_gather`) +
BF16 scoring with: a page-split FP8 indexer K cache (deep_gemm layout) written in place,
scored directly via `deep_gemm.fp8_paged_mqa_logits` (relu-gated == GLM-5 HF
GlmMoeDsaIndexer ground truth). Eliminates ~5-6x indexer-K memory traffic per layer.

Validated standalone (batchgen_kernel_dev/dsa/): byte-identity vs deep_gemm's reference
packer (kv_cache_cast_to_fp8), and >=95.6% top-2048 overlap vs the relu reference across
BSZ{1,16,128} x L{8K,32K}. See short-term/2026-06-02.md.

Cache layout (page-split / SoA), per page, uint8:
    [ page_size*head_dim  FP8 K bytes ] [ page_size*4  fp32 scale bytes ]
head_dim_with_sf = head_dim + 4 = 132 (block_size==head_dim => one fp32 scale/token).
"""

import torch

HEAD_DIM = 128
SF_BYTES = 4
HEAD_DIM_WITH_SF = HEAD_DIM + SF_BYTES  # 132


def quantize_indexer_k(k_bf16: torch.Tensor):
    """[N,128] bf16 -> (k_fp8 [N,128] e4m3, k_scale [N] fp32).

    Reciprocal-multiply form matches deep_gemm's kv_cache_cast_to_fp8 byte-for-byte.
    """
    scale = k_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4) / 448.0
    k_fp8 = (k_bf16 * (1.0 / scale)).to(torch.float8_e4m3fn)
    return k_fp8, scale.squeeze(-1)


def split_write_fp8(buf_u8, loc, k_fp8, k_scale, page_size=64, head_dim=HEAD_DIM):
    """Scatter K (page-split) into a uint8 paged buffer (port of SGLang SetK/SetS).

    buf_u8 : [num_pages, page_size*head_dim + page_size*4] uint8 (page-split layout)
    loc    : [N] int32, absolute PHYSICAL token slot = physical_page*page_size + offset
    Graph-safe: static shapes, plain index scatter.
    """
    page_bytes = buf_u8.shape[1]
    flat = buf_u8.view(-1)
    page_idx = (loc // page_size).to(torch.int64)
    off = (loc % page_size).to(torch.int64)

    k_base = page_idx * page_bytes + off * head_dim
    k_cols = torch.arange(head_dim, device=buf_u8.device)
    flat[(k_base[:, None] + k_cols[None, :]).reshape(-1)] = k_fp8.view(torch.uint8).reshape(-1)

    s_start = page_size * head_dim
    s_base = page_idx * page_bytes + s_start + off * SF_BYTES
    s_cols = torch.arange(SF_BYTES, device=buf_u8.device)
    flat[(s_base[:, None] + s_cols[None, :]).reshape(-1)] = k_scale.float().view(torch.uint8).reshape(-1)
    return buf_u8


def score_paged_fp8(q_bf16, aux_cache_u8, block_table, head_gates, cache_seqlens,
                    schedule_metadata, max_seqlen, page_size=64):
    """Relu-gated paged MQA logits via deep_gemm (no gather).

    q_bf16        : [B, n_heads, head_dim] bf16 (post wq_b + RoPE + Hadamard)
    aux_cache_u8  : [num_pages, page_size, 1, 132] uint8 page-split FP8 indexer K cache
    head_gates    : [B, n_heads] fp32 (already folds n_heads^-0.5 * softmax_scale)
    returns logits: [B, max_seqlen] fp32, clean_logits=True (-inf past cache_seqlens).

    q is act_quant'd per (token, head); q_scale folded into weights:
        relu(softmax_scale*q.k)=softmax_scale*relu(q.k) and softmax_scale lives in head_gates,
        so weights = head_gates * q_scale gives sum_h head_gates_h * relu(q_h.k).
    """
    import deep_gemm
    B, n_heads, head_dim = q_bf16.shape
    qscale = q_bf16.abs().float().amax(dim=2, keepdim=True).clamp(1e-4) / 448.0   # [B,n_heads,1]
    q_fp8 = (q_bf16 / qscale).unsqueeze(1).to(torch.float8_e4m3fn)                # [B,1,n_heads,head_dim]
    weights = (head_gates * qscale.squeeze(2)).contiguous()                      # [B,n_heads] fp32
    num_pages = aux_cache_u8.shape[0]
    kv_view = aux_cache_u8.view(num_pages, page_size, 1, HEAD_DIM_WITH_SF)
    return deep_gemm.fp8_paged_mqa_logits(
        q_fp8, kv_view, weights, cache_seqlens, block_table,
        schedule_metadata, max_seqlen, clean_logits=True)


def make_schedule_metadata(cache_seqlens, page_size=64):
    """Per-decode-step metadata (shared across all 78 layers). Compute OUTSIDE the
    captured region and copy into a persistent static buffer for graph replay
    (get_paged_mqa_logits_metadata allocates a fresh tensor each call)."""
    import deep_gemm
    return deep_gemm.get_paged_mqa_logits_metadata(
        cache_seqlens.int(), page_size, deep_gemm.get_num_sms())
