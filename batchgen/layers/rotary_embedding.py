# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""Shared YaRN Rotary Position Embedding for BatchGen models.

This module provides a single YarnRotaryEmbedding implementation used by
all models that require YaRN-extended RoPE (GPT-OSS-120B, Kimi K2.5, etc.).

Each model should create ONE instance and share it across all attention layers
to avoid duplicated cos/sin caches on GPU.

Reference: https://arxiv.org/abs/2309.00071
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def _yarn_get_mscale(scale=1, mscale=1):
    """Compute YaRN mscale factor for softmax/RoPE scaling."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class YarnRotaryEmbedding(nn.Module):
    """YaRN Rotary Position Embedding with NTK-by-parts interpolation.

    Args:
        dim: RoPE dimension (typically qk_rope_head_dim).
        max_position_embeddings: Maximum sequence length for cache.
        base: RoPE theta base frequency.
        scaling_factor: Context extension factor (1.0 = no extension).
        original_max_position_embeddings: Original context length before extension.
        beta_fast: NTK-by-parts low boundary parameter.
        beta_slow: NTK-by-parts high boundary parameter.
        device: Device for initial computation.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = None,
        mscale_all_dim: float = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        self._compute_inv_freq(device)
        # Cache in BF16 (not default FP32): every runtime consumer is BF16, so the old FP32
        # cache forced a [seq_len, dim] .to(bf16) COPY x2 (cos+sin) on EVERY forward call
        # (~122 casts / 64k prefill microbatch across 61 layers). The trig math above stays
        # FP32; this is the same single RTNE cast the per-call .to() applied — stored values
        # are bit-identical to what consumers already received.
        self._set_cos_sin_cache(max_position_embeddings, device, torch.bfloat16)

    def _compute_inv_freq(self, device: torch.device):
        """Compute inverse frequencies with YaRN NTK-by-parts interpolation."""
        freq = self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
        )

        if self.scaling_factor > 1.0:
            if self.mscale is not None and self.mscale_all_dim is not None:
                # DeepSeek/K2.5 style: mscale / mscale_all_dim ratio
                # When mscale == mscale_all_dim (e.g., both 1.0), concentration = 1.0
                concentration = float(
                    _yarn_get_mscale(self.scaling_factor, self.mscale)
                    / _yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
                )
            else:
                # GPT-OSS style: direct concentration
                concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            d_half = self.dim / 2
            low = (
                d_half
                * math.log(self.original_max_position_embeddings / (self.beta_fast * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.original_max_position_embeddings / (self.beta_slow * 2 * math.pi))
                / math.log(self.base)
            )

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
            self.concentration = concentration
        else:
            inv_freq = 1.0 / freq
            self.concentration = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build cos/sin cache for positions [0, seq_len)."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * self.concentration).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.concentration).to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) embeddings for the given sequence length.

        Args:
            x: Input tensor (used only for device/dtype).
            seq_len: Sequence length to return embeddings for.

        Returns:
            Tuple of (cos, sin) each shaped [seq_len, dim].
        """
        if seq_len is None:
            seq_len = x.size(-2)
        if seq_len > self.max_seq_len_cached:
            # Rebuild in the CACHE's dtype (not x.dtype) so an FP32 warmup/dummy caller can
            # never flip the BF16 cache back to FP32 (or reallocate it post-cuda-graph-capture).
            self._set_cos_sin_cache(seq_len, x.device, self.cos_cached.dtype)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if cos.dtype != x.dtype:
            # Fallback for non-BF16 test/eager callers; never taken in the BF16 runtime.
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor [..., dim].
        k: Key tensor [..., dim].
        cos: Cosine embeddings [seq, dim].
        sin: Sine embeddings [seq, dim].

    Returns:
        Tuple of rotated (q, k).
    """
    cos = cos.unsqueeze(1)  # [seq, 1, dim]
    sin = sin.unsqueeze(1)  # [seq, 1, dim]

    q1, q2 = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    k1, k2 = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]

    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

    return q_rot, k_rot
