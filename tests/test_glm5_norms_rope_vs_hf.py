"""
Unit tests comparing BatchGen's GLM-5 implementation against HF's ground-truth.

Hunts for the first module whose OUTPUT differs from HF beyond BF16 round-off
when fed the SAME input. Tests RMSNorm, RoPE cache construction, and rotation
formulas ELEMENT-BY-ELEMENT, not just mathematical equivalence.
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple


# ============================================================================
# HF Ground-Truth Classes (inlined for self-contained testing)
# ============================================================================

class HfGlmMoeDsaRMSNorm(nn.Module):
    """HF's GlmMoeDsaRMSNorm from modeling_glm_moe_dsa.py:47-65"""

    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """GlmMoeDsaRMSNorm is equivalent to T5LayerNorm"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def hf_rotate_half(x):
    """HF's rotate_half from modeling_glm_moe_dsa.py:67-71"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """HF's apply_rotary_pos_emb from modeling_glm_moe_dsa.py:74-102
    
    Split-half (NeoX/Llama style): (x[:d/2], x[d/2:])
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_rotated = (x * cos) + (hf_rotate_half(x) * sin)
    return x_rotated


class HfGlmMoeDsaRotaryEmbedding(nn.Module):
    """HF's GlmMoeDsaRotaryEmbedding from modeling_glm_moe_dsa.py:664-728
    
    Simplified: computes inv_freq from base and dim, then forward produces
    (cos, sin) indexed by position_ids.
    """

    def __init__(self, dim: int, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        """HF's forward: returns (cos, sin) indexed by position_ids
        
        Args:
            x: input tensor (used only for device/dtype info)
            position_ids: [batch, seq_len] or [batch, 1, seq_len] positions
        
        Returns:
            (cos, sin): each [batch, seq_len, dim] (after unsqueezing if needed)
        """
        # Ensure position_ids is 2D: [batch, seq_len]
        if position_ids.dim() == 3:
            position_ids = position_ids.squeeze(1)
        
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force FP32 for numerical stability
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # Duplicate: [B, S, dim]
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ============================================================================
# BatchGen Classes (imported)
# ============================================================================

from batchgen.models.glm.glm5.model import Glm5RMSNorm, Glm5RotaryEmbedding
from batchgen.attention.mla.rotary_embedding import rotary_pos_emb_interleaved_native


# ============================================================================
# Test: RMSNorm Element-by-Element
# ============================================================================

@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_size",
    [
        (1, 16, 64),
        (2, 64, 256),
        (1, 128, 64),
    ],
)
def test_glm5_rmsnorm_vs_hf(batch_size, seq_len, hidden_size):
    """Test Glm5RMSNorm forward matches HfGlmMoeDsaRMSNorm element-by-element."""
    
    # Use the SAME eps on both sides (matches GLM-5 config rms_norm_eps=1e-5).
    # HF's default class eps=1e-6 but at runtime it receives config.rms_norm_eps.
    eps = 1e-5
    hf_norm = HfGlmMoeDsaRMSNorm(hidden_size, eps=eps)
    batchgen_norm = Glm5RMSNorm(hidden_size, eps=eps)
    
    # Copy weight for perfect match (note: eps differs slightly)
    batchgen_norm.weight.data = hf_norm.weight.data.clone()
    
    # Create random BF16 input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    
    # Forward through both
    with torch.no_grad():
        hf_out = hf_norm(x)
        batchgen_out = batchgen_norm(x)
    
    # Compare
    assert hf_out.shape == batchgen_out.shape, f"Shape mismatch: {hf_out.shape} vs {batchgen_out.shape}"
    
    # Check element-wise with tolerance for BF16
    if torch.allclose(hf_out.float(), batchgen_out.float(), atol=1e-3, rtol=1e-3):
        print(f"✓ RMSNorm test passed: shape={hf_out.shape}")
    else:
        max_diff = (hf_out.float() - batchgen_out.float()).abs().max()
        pytest.fail(
            f"RMSNorm outputs diverge by {max_diff:.6e}\n"
            f"HF shape: {hf_out.shape}, BatchGen shape: {batchgen_out.shape}\n"
            f"This indicates a numerical difference in the normalization computation."
        )


# ============================================================================
# Test: RoPE Cache Construction (cos/sin generation)
# ============================================================================

@pytest.mark.parametrize(
    "batch_size,seq_len,head_dim",
    [
        (1, 16, 64),
        (2, 64, 256),
        (1, 128, 64),
    ],
)
def test_glm5_rope_cache_vs_hf(batch_size, seq_len, head_dim):
    """Test that Glm5RotaryEmbedding._set_cos_sin_cache produces same cos/sin as HF."""
    
    base = 1000000.0
    device = torch.device("cpu")
    
    # HF RoPE: compute cos/sin for a batch of positions. HF casts to x.dtype
    # (BF16 in practice) on output. We use an FP32 x so both engines return
    # FP32 cos/sin and the comparison reveals only true math differences, not
    # BF16 round-off from HF's output cast.
    hf_rope = HfGlmMoeDsaRotaryEmbedding(head_dim, base=base, device=device)

    # BatchGen RoPE: build cache (always FP32 internally)
    batchgen_rope = Glm5RotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len, base=base)
    batchgen_rope._set_cos_sin_cache(seq_len, device, torch.float32)

    # Create position_ids for the batch
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    # Use FP32 x so HF's `cos.to(dtype=x.dtype)` cast is a no-op on precision.
    x = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float32)

    with torch.no_grad():
        hf_cos, hf_sin = hf_rope(x, position_ids)

        # BatchGen: index the cache by position_ids
        batchgen_cos = batchgen_rope.cos_cached[position_ids]
        batchgen_sin = batchgen_rope.sin_cached[position_ids]

    # Compare shapes
    assert hf_cos.shape == batchgen_cos.shape, \
        f"cos shape mismatch: HF={hf_cos.shape}, BatchGen={batchgen_cos.shape}"
    assert hf_sin.shape == batchgen_sin.shape, \
        f"sin shape mismatch: HF={hf_sin.shape}, BatchGen={batchgen_sin.shape}"

    # Both are FP32 now; expect bit-level match (same formula: emb=cat(freqs,freqs).cos()).
    if torch.allclose(hf_cos, batchgen_cos, atol=1e-6, rtol=1e-6):
        print(f"✓ RoPE cos cache test passed: shape={hf_cos.shape}")
    else:
        max_diff = (hf_cos - batchgen_cos).abs().max()
        pytest.fail(f"RoPE cos cache diverges by {max_diff:.6e} (FP32-vs-FP32)")
    
    if torch.allclose(hf_sin, batchgen_sin, atol=1e-6, rtol=1e-6):
        print(f"✓ RoPE sin cache test passed: shape={hf_sin.shape}")
    else:
        max_diff = (hf_sin - batchgen_sin).abs().max()
        pytest.fail(f"RoPE sin cache diverges by {max_diff:.6e} (FP32-vs-FP32)")


# ============================================================================
# Test: RoPE Rotation Formula (HF split-half)
# ============================================================================

@pytest.mark.parametrize(
    "batch_size,seq_len,head_dim",
    [
        (1, 16, 64),
        (2, 64, 128),
        (1, 32, 256),
    ],
)
def test_glm5_rope_split_half_rotation(batch_size, seq_len, head_dim):
    """Test HF's split-half (NeoX/Llama) rotation formula."""
    
    base = 1000000.0
    device = torch.device("cpu")
    
    # HF RoPE
    hf_rope = HfGlmMoeDsaRotaryEmbedding(head_dim, base=base, device=device)
    
    # Create random BF16 query in BHSD format [batch, heads, seq, dim]
    torch.manual_seed(123)
    x_bhsd = torch.randn(batch_size, 8, seq_len, head_dim, dtype=torch.bfloat16)
    
    # Position IDs
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        cos, sin = hf_rope(x_bhsd[:, 0, :, :], position_ids)  # Shape [B, S, D]
        
        # Apply HF's split-half rotation
        x_rotated_hf = hf_apply_rotary_pos_emb(x_bhsd, cos, sin, unsqueeze_dim=1)
    
    # Verify output shape
    assert x_rotated_hf.shape == x_bhsd.shape, \
        f"HF rotation output shape mismatch: {x_rotated_hf.shape} vs {x_bhsd.shape}"
    
    print(f"✓ HF split-half rotation test passed: shape={x_rotated_hf.shape}")


# ============================================================================
# Test: RoPE Interleaved Rotation (BatchGen native)
# ============================================================================

@pytest.mark.parametrize(
    "batch_size,seq_len,head_dim",
    [
        (1, 16, 64),
        (2, 64, 256),
        (1, 128, 64),
    ],
)
def test_glm5_rope_interleaved_rotation(batch_size, seq_len, head_dim):
    """Test BatchGen's rotary_pos_emb_interleaved_native function."""
    
    base = 1000000.0
    device = torch.device("cpu")
    
    # Build cache via BatchGen
    rope = Glm5RotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len, base=base)
    rope._set_cos_sin_cache(seq_len, device, torch.float32)
    
    # Create random BF16 query in BHSD format
    torch.manual_seed(456)
    x_bhsd = torch.randn(batch_size, 8, seq_len, head_dim, dtype=torch.bfloat16)
    
    # Position IDs for all tokens, shaped [B, S] so cos_cached[position_ids]
    # broadcasts cleanly against x_bhsd[B, H, S, D] after unsqueeze_dim=1.
    position_ids = (
        torch.arange(seq_len, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    with torch.no_grad():
        # Apply BatchGen's native interleaved rotation
        x_rotated = rotary_pos_emb_interleaved_native(
            x_bhsd, rope.cos_cached, rope.sin_cached, position_ids, unsqueeze_dim=1
        )
    
    # Verify output shape and dtype
    assert x_rotated.shape == x_bhsd.shape, \
        f"BatchGen rotation output shape mismatch: {x_rotated.shape} vs {x_bhsd.shape}"
    assert x_rotated.dtype == x_bhsd.dtype, \
        f"BatchGen rotation output dtype mismatch: {x_rotated.dtype} vs {x_bhsd.dtype}"
    
    print(f"✓ BatchGen interleaved rotation test passed: shape={x_rotated.shape}")


# ============================================================================
# Test: RoPE Convention Difference (Split-Half vs Interleaved)
# ============================================================================

def test_rope_convention_small_vector():
    """Verify BatchGen implements INTERLEAVED RoPE, not split-half.
    
    This test constructs a small vector and applies both conventions,
    proving they produce DIFFERENT outputs (as they should).
    """
    
    # Small test vector: [1, 0, 0, 0] at position 1
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)  # [1, 1, 4]
    base = 10000.0
    dim = 4
    
    # Compute inv_freq: [0, 2] with step 2 → inv_freq[0] = 10000^(0/4) = 1, inv_freq[1] = 10000^(2/4) = 100
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    # Position 1
    pos = torch.tensor([1], dtype=torch.float32)
    
    # Compute angles: θ_i = pos * inv_freq_i
    theta_0 = (pos * inv_freq[0]).item()  # 1.0
    theta_1 = (pos * inv_freq[1]).item()  # 1.0 / 100 = 0.01
    
    # Expected INTERLEAVED output:
    #   Pairs: (x[0], x[1]), (x[2], x[3]) = (1, 0), (0, 0)
    #   Pair 0 rotated by θ_0: (1*cos(θ_0) - 0*sin(θ_0), 1*sin(θ_0) + 0*cos(θ_0)) = (cos(θ_0), sin(θ_0))
    #   Pair 1 rotated by θ_1: (0*cos(θ_1) - 0*sin(θ_1), 0*sin(θ_1) + 0*cos(θ_1)) = (0, 0)
    expected_interleaved = torch.tensor(
        [[[torch.cos(torch.tensor(theta_0)), torch.sin(torch.tensor(theta_0)), 0.0, 0.0]]],
        dtype=torch.float32
    )
    
    # BatchGen interleaved rotation
    rope = Glm5RotaryEmbedding(dim=dim, max_position_embeddings=2, base=base)
    rope._set_cos_sin_cache(2, x.device, torch.float32)
    
    position_ids = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        x_rotated_interleaved = rotary_pos_emb_interleaved_native(
            x, rope.cos_cached, rope.sin_cached, position_ids, unsqueeze_dim=1
        )
    
    # Check element-wise match with expected interleaved
    if torch.allclose(x_rotated_interleaved, expected_interleaved, atol=1e-5, rtol=1e-5):
        print(f"✓ BatchGen implements INTERLEAVED RoPE as expected")
    else:
        max_diff = (x_rotated_interleaved - expected_interleaved).abs().max()
        pytest.fail(
            f"BatchGen interleaved rotation diverges from expected by {max_diff:.6e}\n"
            f"Expected: {expected_interleaved}\n"
            f"Got: {x_rotated_interleaved}"
        )
    
    # Now test HF's split-half (should be DIFFERENT)
    hf_rope = HfGlmMoeDsaRotaryEmbedding(dim, base=base)
    position_ids_2d = torch.tensor([[1]], dtype=torch.long)
    
    with torch.no_grad():
        hf_cos, hf_sin = hf_rope(x, position_ids_2d)
        x_rotated_split = hf_apply_rotary_pos_emb(x, hf_cos, hf_sin, unsqueeze_dim=1)
    
    # HF split-half should produce DIFFERENT output
    if not torch.allclose(x_rotated_split, x_rotated_interleaved, atol=1e-5, rtol=1e-5):
        print(f"✓ HF split-half and BatchGen interleaved produce DIFFERENT outputs (as expected)")
        print(f"  BatchGen interleaved: {x_rotated_interleaved[0, 0, :]}")
        print(f"  HF split-half:        {x_rotated_split[0, 0, :]}")
    else:
        pytest.fail(
            "HF split-half and BatchGen interleaved produce SAME output — "
            "this suggests config mismatch or rope_interleave not implemented."
        )


# ============================================================================
# Test: Prefill-like Usage (multiple sequences with different positions)
# ============================================================================

@pytest.mark.parametrize(
    "batch_size,seq_len,head_dim",
    [
        (2, 32, 64),
        (4, 64, 128),
    ],
)
def test_glm5_rope_prefill_usage(batch_size, seq_len, head_dim):
    """Test RoPE in prefill scenario: batch of sequences, heterogeneous positions."""
    
    base = 1000000.0
    device = torch.device("cpu")
    
    # BatchGen setup
    rope = Glm5RotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len, base=base)
    rope._set_cos_sin_cache(seq_len, device, torch.float32)
    
    # Random query [batch, heads, seq, dim]
    torch.manual_seed(789)
    q = torch.randn(batch_size, 8, seq_len, head_dim, dtype=torch.bfloat16)
    
    # Position IDs for prefill: [0, 1, 2, ..., seq_len-1] repeated per-batch.
    # Shape [B, S] for broadcasting with unsqueeze_dim=1.
    position_ids = (
        torch.arange(seq_len, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    with torch.no_grad():
        q_rotated = rotary_pos_emb_interleaved_native(
            q, rope.cos_cached, rope.sin_cached, position_ids, unsqueeze_dim=1
        )
    
    # Sanity checks
    assert q_rotated.shape == q.shape, f"Shape mismatch: {q_rotated.shape} vs {q.shape}"
    assert q_rotated.dtype == q.dtype, f"Dtype mismatch: {q_rotated.dtype} vs {q.dtype}"
    
    # Verify no NaNs or Infs
    assert not torch.isnan(q_rotated).any(), "NaN detected in rotated output"
    assert not torch.isinf(q_rotated).any(), "Inf detected in rotated output"
    
    print(f"✓ Prefill-like RoPE test passed: batch={batch_size}, seq={seq_len}, dim={head_dim}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
