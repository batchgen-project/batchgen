"""DeepSeek-V4-Flash prefill attention, ported from the official reference.

Replicates assets/inference/model.py Attention.forward (start_pos == 0 branch)
per sequence: RoPE, fp8 QAT simulation of KV, sliding-window + compressed-KV
top-k indices (c4 learned indexer / c128 deterministic), tilelang sparse_attn
with attn_sink, and inverse RoPE on the output. Runs each sequence of a
prepacked row independently, which also prevents cross-sequence attention.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F


def _kernels():
    from batchgen.models.deepseek.deepseekv4_flash.assets.inference import (
        kernel,
    )

    return kernel


_SCALE_FMT = "ue8m0"
_SCALE_DTYPE = torch.float8_e8m0fnu
_FP4_BLOCK_SIZE = 32
_KV_QUANT_BLOCK = 64


@lru_cache(8)
def _freqs_cis_cpu(
    rope_head_dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(lo, hi, dim):
        if lo == hi:
            hi += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(linear_func, 0, 1)

    dim = rope_head_dim
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def layer_freqs_cis(
    config,
    compress_ratio: int,
    seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    rope_head_dim = int(
        getattr(
            config, "qk_rope_head_dim", getattr(config, "rope_head_dim", 64)
        )
    )
    if compress_ratio:
        original_seq_len = int(getattr(config, "original_seq_len", 65536))
        base = float(getattr(config, "compress_rope_theta", 160000.0))
    else:
        original_seq_len = 0
        base = float(getattr(config, "rope_theta", 10000.0))
    factor = float(getattr(config, "rope_factor", 16.0))
    beta_fast = float(getattr(config, "beta_fast", 32.0))
    beta_slow = float(getattr(config, "beta_slow", 1.0))
    freqs = _freqs_cis_cpu(
        rope_head_dim,
        max(seqlen, 1),
        original_seq_len,
        base,
        factor,
        beta_fast,
        beta_slow,
    )
    return freqs.to(device)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform

    return hadamard_transform(x, scale=x.size(-1) ** -0.5)


def window_topk_idxs(
    window_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    base = torch.arange(seqlen).unsqueeze(1)
    matrix = (base - window_size + 1).clamp(0) + torch.arange(
        min(seqlen, window_size)
    )
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).to(device)


def compress_topk_idxs(
    ratio: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).to(device)


def _slot_weight(slot, dtype=torch.float32) -> torch.Tensor:
    from batchgen.models.deepseek.deepseekv4_flash.model import _dequant_weight

    if slot.weight is None:
        raise RuntimeError("linear slot has no runtime weight loaded")
    return _dequant_weight(slot.weight, slot.scale, dtype)


def compressor_prefill(
    comp,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rotate: bool,
) -> Optional[torch.Tensor]:
    """Official Compressor.forward, start_pos==0 branch, single sequence.

    comp: DeepSeekV4FlashCompressor (LinearSlot wkv/wgate, ape, norm).
    x: [1, s, hidden] bf16. Returns [1, s//ratio, head_dim] bf16 or None.
    """
    kern = _kernels()
    ratio = comp.compress_ratio
    overlap = comp.overlap
    d = comp.head_dim
    rd = comp.rope_head_dim
    bsz, seqlen, _ = x.size()
    dtype = x.dtype
    if seqlen < ratio:
        return None

    xf = x.float()
    wkv_w = _slot_weight(comp.wkv)
    wgate_w = _slot_weight(comp.wgate)
    kv = F.linear(xf, wkv_w)
    score = F.linear(xf, wgate_w)

    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    kv = kv[:, :cutoff]
    score = score[:, :cutoff]
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + comp.ape.float()
    if overlap:
        kv = _overlap_transform(kv, ratio, d, 0.0)
        score = _overlap_transform(score, ratio, d, float("-inf"))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)

    kv = comp.norm(kv.to(dtype))
    apply_rotary_emb(kv[..., -rd:], freqs_cis[:cutoff:ratio])
    if rotate:
        kv = rotate_activation(kv)
        kern.fp4_act_quant(kv, _FP4_BLOCK_SIZE, True)
    else:
        kern.act_quant(
            kv[..., :-rd], _KV_QUANT_BLOCK, _SCALE_FMT, _SCALE_DTYPE, True
        )
    return kv


def _overlap_transform(
    tensor: torch.Tensor, ratio: int, d: int, value: float
) -> torch.Tensor:
    b, s, _, _ = tensor.size()
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor


def indexer_prefill_topk(
    mod,
    x: torch.Tensor,
    qr: torch.Tensor,
    freqs_cis: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    """Official Indexer.forward, start_pos==0 branch, single sequence.

    Returns compress_topk_idxs [1, s, k] (k = min(index_topk, s // ratio)),
    already offset / -1-masked for concatenation with window indices.
    """
    kern = _kernels()
    indexer = mod.indexer
    ratio = indexer.compressor.compress_ratio
    n_heads = indexer.n_heads
    head_dim = indexer.head_dim
    rd = indexer.compressor.rope_head_dim
    bsz, seqlen, _ = x.size()
    index_topk = indexer.index_topk

    kv_cache = compressor_prefill(indexer.compressor, x, freqs_cis, rotate=True)
    n_compressed = 0 if kv_cache is None else kv_cache.size(1)
    k = min(index_topk, seqlen // ratio, n_compressed)
    if k <= 0:
        return x.new_full((1, seqlen, 0), -1, dtype=torch.long)

    full = mod._prefill_full_tensors
    if "indexer.wq_b.weight" in full:
        from batchgen.models.deepseek.deepseekv4_flash.model import (
            _linear_from_weight,
        )

        q = _linear_from_weight(
            qr,
            full["indexer.wq_b.weight"],
            full.get("indexer.wq_b.scale"),
        )
        weights_w = full["indexer.weights_proj.weight"]
    else:
        q = indexer.wq_b(qr)
        weights_w = _slot_weight(indexer.weights_proj, qr.dtype)
    q = q.unflatten(-1, (n_heads, head_dim))
    apply_rotary_emb(q[..., -rd:], freqs_cis[:seqlen])
    q = rotate_activation(q)
    kern.fp4_act_quant(q, _FP4_BLOCK_SIZE, True)

    softmax_scale = head_dim**-0.5
    weights = F.linear(x, weights_w.to(x.dtype)) * (
        softmax_scale * n_heads**-0.5
    )

    index_score = torch.einsum("bshd,btd->bsht", q, kv_cache)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    mask = (
        torch.arange(n_compressed, device=x.device).repeat(seqlen, 1)
        >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
    )
    index_score = index_score + torch.where(
        mask,
        index_score.new_tensor(float("-inf")),
        index_score.new_tensor(0.0),
    )
    index_score = index_score + torch.where(
        mask, torch.float("-inf") if False else float("-inf"), 0.0
    )
    topk_idxs = index_score.topk(k, dim=-1)[1]
    invalid = topk_idxs >= (
        torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
    )
    topk_idxs = torch.where(invalid, -1, topk_idxs + offset)
    return topk_idxs


def sparse_prefill_attention_sequence(
    mod,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse prefill attention for ONE sequence x: [1, s, hidden].

    Returns (attn_output [1, s, hidden], kv_normed [1, s, head_dim]).
    kv_normed is the pre-rope kv_norm(wkv(x)) consumed by the KV-cache
    population path (which applies rope/quant itself).
    """
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        _dequant_weight,
        _linear_from_weight,
    )

    kern = _kernels()
    bsz, seqlen, _ = x.size()
    assert bsz == 1
    rd = int(getattr(mod, "rope_head_dim", 64) or 64)
    win = int(getattr(mod, "window_size", 128) or 128)
    ratio = int(mod.compress_ratio or 0)
    n_heads = mod.n_heads
    n_groups = mod.o_groups
    device = x.device

    freqs_cis = layer_freqs_cis(mod._config_ref, ratio, seqlen, device)
    full = mod._prefill_full_tensors

    qr = mod.q_norm(mod.wq_a(x))
    if "wq_b.weight" in full:
        q = _linear_from_weight(qr, full["wq_b.weight"], full.get("wq_b.scale"))
    else:
        q = mod.wq_b(qr)
    q = q.view(bsz, seqlen, n_heads, mod.head_dim)
    q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + mod.eps)
    apply_rotary_emb(q[..., -rd:], freqs_cis[:seqlen])

    kv_normed = mod.kv_norm(mod.wkv(x))
    kv = kv_normed.clone()
    apply_rotary_emb(kv[..., -rd:], freqs_cis[:seqlen])
    kern.act_quant(
        kv[..., :-rd], _KV_QUANT_BLOCK, _SCALE_FMT, _SCALE_DTYPE, True
    )

    topk_idxs = window_topk_idxs(win, seqlen, device)
    if ratio:
        offset = seqlen
        if ratio == 4 and getattr(mod, "indexer", None) is not None:
            comp_idxs = indexer_prefill_topk(mod, x, qr, freqs_cis, offset)
        else:
            comp_idxs = compress_topk_idxs(ratio, seqlen, offset, device)
        topk_idxs = torch.cat([topk_idxs, comp_idxs.to(topk_idxs)], dim=-1)
        kv_compress = compressor_prefill(
            mod.compressor, x, freqs_cis, rotate=False
        )
        if kv_compress is not None:
            kv = torch.cat([kv, kv_compress], dim=1)
    topk_idxs = topk_idxs.int()

    attn_sink = (
        (full["attn_sink"] if "attn_sink" in full else mod.attn_sink)
        .float()
        .contiguous()
    )
    # The tilelang sparse_attn kernel allocates (h+1)*head_dim shared memory;
    # 64 heads x 512 dims exceeds sm120's 100KB dynamic-smem cap. Heads are
    # independent (per-head online softmax + per-head sink), so chunking over
    # heads is mathematically exact.
    head_chunk = 16
    if n_heads <= head_chunk:
        o = kern.sparse_attn(q, kv, attn_sink, topk_idxs, mod.softmax_scale)
    else:
        o = torch.empty_like(q)
        for h0 in range(0, n_heads, head_chunk):
            h1 = min(h0 + head_chunk, n_heads)
            o[:, :, h0:h1] = kern.sparse_attn(
                q[:, :, h0:h1].contiguous(),
                kv,
                attn_sink[h0:h1].contiguous(),
                topk_idxs,
                mod.softmax_scale,
            )
    apply_rotary_emb(o[..., -rd:], freqs_cis[:seqlen], True)

    o = o.view(bsz, seqlen, n_groups, -1)
    wo_a_raw = full["wo_a.weight"] if "wo_a.weight" in full else mod.wo_a.weight
    wo_a_weight = _dequant_weight(wo_a_raw, None, x.dtype)
    wo_a = wo_a_weight.view(
        n_groups, mod.o_lora_rank, n_heads // n_groups * mod.head_dim
    )
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    if "wo_b.weight" in full:
        attn_output = _linear_from_weight(
            o.flatten(2), full["wo_b.weight"], full.get("wo_b.scale")
        )
    else:
        attn_output = mod.wo_b(o.flatten(2))
    return attn_output, kv_normed
