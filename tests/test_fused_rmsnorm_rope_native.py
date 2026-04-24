"""Unit test for ``fused_rmsnorm_rope_with_q_native``.

Verifies the new Triton decode kernel that produces NATIVE INTERLEAVED RoPE
output (pair-wise rotation stored at the original ``(2i, 2i+1)`` positions)
against:

  1. A PyTorch RMSNorm reference (FP32 internal, BF16 in/out) for the
     compressed-KV lora slice.
  2. The prefill-side ``rotary_pos_emb_interleaved_native`` helper for the
     q_pe + k_pe rotation.

Rationale
---------
The legacy ``fused_rmsnorm_rope_with_q`` stores rotated pairs in a
split-half layout (all even-rotated in the first half, all odd-rotated in
the second half). Prefill's legacy ``rotary_pos_emb`` reshape trick
produces the matching split-half layout, so the two are mutually
consistent. When prefill is switched to
``rotary_pos_emb_interleaved_native`` (true interleaved output), the decode
kernel must also write true interleaved output or the k_pe cache becomes
inconsistent between prefill-populated rows and decode-appended rows, and
``flash_mla_with_kvcache`` reads across two incompatible layouts.

This test enforces that the new ``_native`` kernel produces
element-for-element equivalent output to the PyTorch interleaved
reference.

Run on any CUDA GPU. Requires ``triton`` + ``batchgen_kernels``.
"""
import math

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires CUDA"
)


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32-internal RMSNorm reference (matches HF GlmMoeDsaRMSNorm)."""
    orig_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    normed = x_f32 * torch.rsqrt(variance + eps)
    return (weight.float() * normed).to(orig_dtype)


def _build_cos_sin_cache(max_seq_len: int, head_dim: int, base: float = 1_000_000.0, device: str = "cuda"):
    """Matches ``Glm5RotaryEmbedding._set_cos_sin_cache`` cache format."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)          # [S, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)               # [S, head_dim] duplicated
    return emb.cos(), emb.sin()


def _pairwise_interleaved_ref(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference: native interleaved RoPE.

    Pairs ``(x[2i], x[2i+1])`` are rotated by angle ``position * inv_freq[i]``
    and stored back at the same ``(2i, 2i+1)`` positions.
    """
    orig_dtype = x.dtype
    d = x.shape[-1]
    half = d // 2
    x_f32 = x.float()
    # cos_cache / sin_cache: [max_seq_len, d]; duplicated so first half ==
    # second half. We only need the first half.
    cos_half = cos_cache[..., :half]
    sin_half = sin_cache[..., :half]
    cos_pos = cos_half[position_ids]   # broadcast to x's shape
    sin_pos = sin_half[position_ids]
    # Match the broadcast dims of x_f32
    for _ in range(x_f32.dim() - cos_pos.dim() - 1):
        cos_pos = cos_pos.unsqueeze(-2)
        sin_pos = sin_pos.unsqueeze(-2)
    x_pairs = x_f32.view(*x_f32.shape[:-1], half, 2)
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]
    rot_even = x_even * cos_pos - x_odd * sin_pos
    rot_odd = x_even * sin_pos + x_odd * cos_pos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(orig_dtype)


@pytest.mark.parametrize("bsz,num_heads", [(1, 64), (2, 64), (4, 16)])
def test_fused_rmsnorm_rope_native_matches_reference(bsz: int, num_heads: int):
    """Decode kernel output == PyTorch RMSNorm ref + native-interleaved RoPE ref."""
    from batchgen_kernels.triton.fused_rmsnorm_rope import (
        fused_rmsnorm_rope_with_q_native,
    )

    device = "cuda"
    dtype = torch.bfloat16

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    total_dim = kv_lora_rank + qk_rope_head_dim  # 576
    max_seq_len = 8192
    eps = 1e-5

    torch.manual_seed(0x6C3F)
    new_compressed_kv = torch.randn(
        (bsz, 1, total_dim), dtype=dtype, device=device
    )
    q_pe = torch.randn(
        (bsz, num_heads, 1, qk_rope_head_dim), dtype=dtype, device=device
    ).contiguous()
    norm_weight = torch.randn((kv_lora_rank,), dtype=dtype, device=device) * 0.05 + 0.5
    position_ids = torch.randint(
        0, max_seq_len, (bsz, 1), dtype=torch.long, device=device
    )

    cos_cache, sin_cache = _build_cos_sin_cache(
        max_seq_len, qk_rope_head_dim, base=1_000_000.0, device=device
    )

    # --- Reference ---
    compressed_kv_lora = new_compressed_kv[..., :kv_lora_rank]
    k_pe_unrotated = new_compressed_kv[..., kv_lora_rank:]

    ref_kv_lora = _rmsnorm_ref(compressed_kv_lora, norm_weight, eps=eps)
    # Flatten position_ids for reference indexing: k_pe is [bsz, 1, rope_D]
    pos_ids_k = position_ids.view(bsz)
    pos_ids_q = position_ids.view(bsz)
    ref_k_pe = _pairwise_interleaved_ref(
        k_pe_unrotated.view(bsz, qk_rope_head_dim).float(),
        cos_cache,
        sin_cache,
        pos_ids_k,
    )
    # q_pe: [bsz, num_heads, 1, rope_D]. Broadcast cos/sin across heads.
    # The reference doesn't care about head-dim — just rotate each head.
    q_pe_ref_in = q_pe.view(bsz, num_heads, qk_rope_head_dim).float()
    # Expand position_ids to [bsz, num_heads] for broadcast.
    pos_ids_q_heads = pos_ids_q.view(bsz, 1).expand(bsz, num_heads).reshape(-1)
    q_pe_flat = q_pe_ref_in.reshape(bsz * num_heads, qk_rope_head_dim)
    ref_q_pe_flat = _pairwise_interleaved_ref(
        q_pe_flat, cos_cache, sin_cache, pos_ids_q_heads,
    )
    ref_q_pe = ref_q_pe_flat.view(bsz, num_heads, 1, qk_rope_head_dim).to(dtype)

    ref_offload = torch.empty_like(new_compressed_kv)
    ref_offload[..., :kv_lora_rank] = ref_kv_lora
    ref_offload[..., kv_lora_rank:] = ref_k_pe.view(bsz, 1, qk_rope_head_dim).to(dtype)

    # --- Kernel ---
    q_pe_kernel = q_pe.clone()  # kernel mutates in place
    kernel_offload = fused_rmsnorm_rope_with_q_native(
        new_compressed_kv.clone(),
        q_pe_kernel,
        cos_cache,
        sin_cache,
        position_ids,
        norm_weight,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        eps=eps,
    )

    # --- Checks ---
    bf16_atol, bf16_rtol = 6e-3, 6e-3

    assert torch.allclose(
        kernel_offload[..., :kv_lora_rank].float(),
        ref_offload[..., :kv_lora_rank].float(),
        atol=bf16_atol, rtol=bf16_rtol,
    ), (
        f"RMSNorm output mismatch: max|diff|="
        f"{(kernel_offload[..., :kv_lora_rank].float() - ref_offload[..., :kv_lora_rank].float()).abs().max().item():.4e}"
    )

    assert torch.allclose(
        kernel_offload[..., kv_lora_rank:].float(),
        ref_offload[..., kv_lora_rank:].float(),
        atol=bf16_atol, rtol=bf16_rtol,
    ), (
        f"k_pe RoPE output mismatch: max|diff|="
        f"{(kernel_offload[..., kv_lora_rank:].float() - ref_offload[..., kv_lora_rank:].float()).abs().max().item():.4e}"
    )

    assert torch.allclose(
        q_pe_kernel.float(), ref_q_pe.float(),
        atol=bf16_atol, rtol=bf16_rtol,
    ), (
        f"q_pe RoPE output mismatch: max|diff|="
        f"{(q_pe_kernel.float() - ref_q_pe.float()).abs().max().item():.4e}"
    )


def test_native_vs_legacy_produce_different_layouts():
    """Sanity: the new kernel and the legacy kernel disagree pointwise,
    proving we actually changed the layout (not just renamed a symbol)."""
    from batchgen_kernels.triton.fused_rmsnorm_rope import (
        fused_rmsnorm_rope_with_q,
        fused_rmsnorm_rope_with_q_native,
    )

    device = "cuda"
    dtype = torch.bfloat16
    bsz, num_heads = 2, 8
    kv_lora_rank, qk_rope_head_dim = 512, 64
    total_dim = kv_lora_rank + qk_rope_head_dim
    max_seq_len = 4096
    eps = 1e-5

    torch.manual_seed(0x1A2B)
    kv_in = torch.randn((bsz, 1, total_dim), dtype=dtype, device=device)
    q_pe = torch.randn(
        (bsz, num_heads, 1, qk_rope_head_dim), dtype=dtype, device=device
    ).contiguous()
    norm_weight = torch.randn((kv_lora_rank,), dtype=dtype, device=device) * 0.05 + 0.5
    position_ids = torch.randint(
        1, max_seq_len, (bsz, 1), dtype=torch.long, device=device
    )
    cos_cache, sin_cache = _build_cos_sin_cache(
        max_seq_len, qk_rope_head_dim, base=1_000_000.0, device=device
    )

    q_native = q_pe.clone()
    out_native = fused_rmsnorm_rope_with_q_native(
        kv_in.clone(), q_native, cos_cache, sin_cache,
        position_ids, norm_weight,
        kv_lora_rank, qk_rope_head_dim, eps=eps,
    )
    q_legacy = q_pe.clone()
    out_legacy = fused_rmsnorm_rope_with_q(
        kv_in.clone(), q_legacy, cos_cache, sin_cache,
        position_ids, norm_weight,
        kv_lora_rank, qk_rope_head_dim, eps=eps,
    )

    # RMSNorm slice should match bit-for-bit (same math, only rope output
    # differs between kernels).
    assert torch.allclose(
        out_native[..., :kv_lora_rank].float(),
        out_legacy[..., :kv_lora_rank].float(),
        atol=0.0, rtol=0.0,
    ), "RMSNorm outputs should be bit-identical between legacy and native"

    # rope slice should differ (the whole point of the new kernel).
    rope_diff = (
        out_native[..., kv_lora_rank:].float()
        - out_legacy[..., kv_lora_rank:].float()
    ).abs().max().item()
    assert rope_diff > 1e-3, (
        f"k_pe native vs legacy output should differ (layout change); "
        f"got max|diff|={rope_diff:.4e}"
    )

    q_diff = (q_native.float() - q_legacy.float()).abs().max().item()
    assert q_diff > 1e-3, (
        f"q_pe native vs legacy output should differ (layout change); "
        f"got max|diff|={q_diff:.4e}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
