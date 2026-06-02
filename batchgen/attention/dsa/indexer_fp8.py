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


def _wrap_host_page_u8(ptr: int, page_bytes: int):
    """Wrap a raw host (CPU/SHM) page pointer as a [page_bytes] uint8 tensor.

    No copy: the returned tensor aliases the host paged-KV SHM page, so writes
    land directly in the host aux cache. `ptr` comes from the worker view's
    get_sequence_layer_page_pointers (uintptr_t)."""
    import ctypes
    buf = (ctypes.c_uint8 * page_bytes).from_address(int(ptr))
    return torch.frombuffer(buf, dtype=torch.uint8, count=page_bytes)


def write_host_indexer_fp8_pages(aux_view, layer_idx, sequence_id, k_bf16,
                                 page_size=64, head_dim=HEAD_DIM):
    """Quantize a sequence's indexer K and write FP8 page-split bytes to HOST pages.

    Mirrors the GPU-side `split_write_fp8` byte layout (per page, SoA:
    [page_size*head_dim FP8 K | page_size*4 fp32 scale]) directly into the host
    aux SHM pages, so the later host->GPU reload (verbatim per-page byte copy via
    async_load_layer_paged_kv_to_device) is FP8-coherent with what the decode
    path writes and what deep_gemm.fp8_paged_mqa_logits reads. Quantization
    happens ONCE here (prefill / decode host-append), never on the reload path.

    aux_view     : host aux worker view (DualHostKVCoordinator.auxiliary).
    layer_idx    : logical layer id (resolved to physical by the worker view).
    sequence_id  : global sequence id whose host pages are already allocated.
    k_bf16       : [T, head_dim] bf16 indexer K (post k_norm + RoPE + Hadamard),
                   on any device; moved to CPU here for the host SHM write.
    """
    k_cpu = k_bf16.detach().to("cpu", dtype=torch.bfloat16).reshape(-1, head_dim)
    num_tokens = k_cpu.shape[0]
    if num_tokens == 0:
        return
    k_fp8, k_scale = quantize_indexer_k(k_cpu)  # [T,head_dim] e4m3, [T] fp32

    page_bytes = page_size * head_dim + page_size * SF_BYTES
    # Per-page host pointers for this (sequence, layer). max_tokens=num_tokens
    # bounds the returned page list to the pages this prompt actually fills.
    k_ptrs, _ = aux_view.get_sequence_layer_page_pointers(
        sequence_id, layer_idx, num_tokens)
    num_pages = (num_tokens + page_size - 1) // page_size
    if len(k_ptrs) < num_pages:
        raise RuntimeError(
            f"write_host_indexer_fp8_pages: seq {sequence_id} layer {layer_idx} "
            f"has {len(k_ptrs)} host pages but needs {num_pages} for {num_tokens} "
            "tokens; host aux pages were not allocated before offload"
        )
    for p in range(num_pages):
        start = p * page_size
        end = min(start + page_size, num_tokens)
        n = end - start
        page_t = _wrap_host_page_u8(k_ptrs[p], page_bytes).view(1, page_bytes)
        # Page-local slots [0, n): split_write_fp8 with page_idx==0 (loc==offset).
        loc = torch.arange(n, dtype=torch.int32)
        split_write_fp8(page_t, loc, k_fp8[start:end], k_scale[start:end],
                        page_size=page_size, head_dim=head_dim)


def write_host_indexer_fp8_token(aux_view, layer_idx, sequence_id, k_bf16_1tok,
                                 write_pos, page_size=64, head_dim=HEAD_DIM):
    """Write ONE decode token's FP8 page-split indexer K into the host aux pages.

    Decode-side mirror of write_host_indexer_fp8_pages: quantizes a single
    [head_dim] (or [1,head_dim]) indexer K vector and scatters its 128 FP8 bytes
    + 4 fp32 scale bytes into the host page at slot `write_pos`
    (page = write_pos // page_size, offset = write_pos % page_size). Keeps the
    host aux cache byte-coherent with the GPU split_write_fp8 layout so a later
    host->GPU reload is a verbatim copy.
    """
    k_cpu = k_bf16_1tok.detach().to("cpu", dtype=torch.bfloat16).reshape(1, head_dim)
    k_fp8, k_scale = quantize_indexer_k(k_cpu)
    page = int(write_pos) // page_size
    offset = int(write_pos) % page_size
    page_bytes = page_size * head_dim + page_size * SF_BYTES
    k_ptrs, _ = aux_view.get_sequence_layer_page_pointers(
        sequence_id, layer_idx, int(write_pos) + 1)
    if len(k_ptrs) <= page:
        raise RuntimeError(
            f"write_host_indexer_fp8_token: seq {sequence_id} layer {layer_idx} "
            f"write_pos {write_pos} needs host page {page} but only "
            f"{len(k_ptrs)} pages allocated"
        )
    page_t = _wrap_host_page_u8(k_ptrs[page], page_bytes).view(1, page_bytes)
    loc = torch.tensor([offset], dtype=torch.int32)
    split_write_fp8(page_t, loc, k_fp8, k_scale, page_size=page_size, head_dim=head_dim)


def write_host_indexer_fp8_token_prequant(aux_view, layer_idx, sequence_id,
                                          k_fp8_1tok, k_scale_1tok, write_pos,
                                          page_size=64, head_dim=HEAD_DIM):
    """Write ONE decode token's ALREADY-QUANTIZED FP8 page-split indexer K.

    Identical to write_host_indexer_fp8_token but takes pre-quantized inputs so the
    caller can do a SINGLE batched D2H + a SINGLE vectorized quantize across many
    tokens, then scatter each token's bytes here (no per-token .to("cpu")).

    k_fp8_1tok   : [head_dim] (or [1,head_dim]) float8_e4m3fn, on CPU.
    k_scale_1tok : [] / [1] fp32, on CPU.
    """
    page = int(write_pos) // page_size
    offset = int(write_pos) % page_size
    page_bytes = page_size * head_dim + page_size * SF_BYTES
    k_ptrs, _ = aux_view.get_sequence_layer_page_pointers(
        sequence_id, layer_idx, int(write_pos) + 1)
    if len(k_ptrs) <= page:
        raise RuntimeError(
            f"write_host_indexer_fp8_token_prequant: seq {sequence_id} layer "
            f"{layer_idx} write_pos {write_pos} needs host page {page} but only "
            f"{len(k_ptrs)} pages allocated"
        )
    page_t = _wrap_host_page_u8(k_ptrs[page], page_bytes).view(1, page_bytes)
    loc = torch.tensor([offset], dtype=torch.int32)
    split_write_fp8(page_t, loc, k_fp8_1tok.reshape(1, head_dim),
                    k_scale_1tok.reshape(1), page_size=page_size, head_dim=head_dim)


def make_schedule_metadata(cache_seqlens, page_size=64):
    """Per-decode-step metadata (shared across all 78 layers). Compute OUTSIDE the
    captured region and copy into a persistent static buffer for graph replay
    (get_paged_mqa_logits_metadata allocates a fresh tensor each call).

    R3b (verified): the returned tensor shape is STATIC / batch-independent. deep_gemm
    sizes it as ``torch.empty({num_sms + 1, 2}, context_lens.options())``
    (references/DeepGEMM/csrc/apis/attention.hpp:152) — int32 (context_lens must be
    kInt), shape [num_sms+1, 2] depending ONLY on num_sms (a fixed GPU property), not
    on batch_size. fp8_paged_mqa_logits asserts ``schedule_meta_size == num_sms + 1
    and meta_info_size == 2`` (attention.hpp:196) where num_sms ==
    device_runtime->get_num_sms(), so the static buffer MUST be built with the SAME
    deep_gemm.get_num_sms() used here. Nothing in the contents depends on batch shape
    for the buffer dims, so a persistent static buffer of shape [num_sms+1, 2] int32
    is safe to copy_ into each step."""
    import deep_gemm
    return deep_gemm.get_paged_mqa_logits_metadata(
        cache_seqlens.int(), page_size, deep_gemm.get_num_sms())
