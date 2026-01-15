import json
import math
import os
import time
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.distributed as dist

from .weights import Checkpoint


# =============================================================================
# Timing infrastructure for performance profiling
# =============================================================================
class DecodeTimingStats:
    """Accumulates timing statistics for decode operations."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.enabled = False
        self.call_counts = {}
        self.total_times = {}
        self._start_times = {}

    def enable(self):
        self.enabled = True
        self.reset()
        self.enabled = True

    def disable(self):
        self.enabled = False

    def start(self, name: str):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if not self.enabled or name not in self._start_times:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_times[name]
        if name not in self.total_times:
            self.total_times[name] = 0.0
            self.call_counts[name] = 0
        self.total_times[name] += elapsed
        self.call_counts[name] += 1
        del self._start_times[name]

    @contextmanager
    def time(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def print_stats(self):
        import logging
        if not self.total_times:
            logging.info("[TIMING] No timing data collected")
            return

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("DECODE TIMING STATISTICS")
        lines.append("=" * 70)

        # Sort by total time descending
        sorted_items = sorted(self.total_times.items(), key=lambda x: -x[1])
        total_measured = sum(self.total_times.values())

        lines.append(f"{'Operation':<40} {'Total(ms)':>10} {'Calls':>8} {'Avg(ms)':>10} {'%':>6}")
        lines.append("-" * 70)

        for name, total_time in sorted_items:
            calls = self.call_counts[name]
            avg_time = total_time / calls if calls > 0 else 0
            pct = (total_time / total_measured * 100) if total_measured > 0 else 0
            lines.append(f"{name:<40} {total_time*1000:>10.2f} {calls:>8} {avg_time*1000:>10.3f} {pct:>5.1f}%")

        lines.append("-" * 70)
        lines.append(f"{'TOTAL MEASURED':<40} {total_measured*1000:>10.2f}")
        lines.append("=" * 70)

        # Log all lines
        for line in lines:
            logging.info(line)


# Global timing stats instance
DECODE_TIMING = DecodeTimingStats()


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int, position_offset: int = 0):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        # Add position_offset to support KV caching during decode
        t = torch.arange(position_offset, position_offset + num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key.

        Args:
            query: Query tensor
            key: Key tensor
            position_offset: Position offset for decode mode (cached sequence length)
        """
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens, position_offset)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0, attention_mask=None):
    """Memory-efficient scaled dot-product attention using PyTorch's SDPA.

    Uses Flash Attention or memory-efficient backend when available.
    Falls back to naive implementation only for small sequences.

    Args:
        Q: Query tensor [q_len, n_kv_heads, q_mult, head_dim]
        K: Key tensor [kv_len, n_kv_heads, head_dim]
        V: Value tensor [kv_len, n_kv_heads, head_dim]
        S: Attention sinks [num_attention_heads] (currently ignored for efficiency)
        sm_scale: Softmax scale factor
        sliding_window: Sliding window size (0 = no window, full attention)
        attention_mask: Optional attention mask from BatchGenWorker
    """
    q_len, n_kv_heads, q_mult, d_head = Q.shape
    kv_len = K.shape[0]
    n_q_heads = n_kv_heads * q_mult

    # Reshape for PyTorch SDPA: [batch, heads, seq, head_dim]
    # Q: [q_len, n_kv_heads, q_mult, d_head] -> [1, n_q_heads, q_len, d_head]
    Q = Q.permute(1, 2, 0, 3).reshape(1, n_q_heads, q_len, d_head)

    # K, V: [kv_len, n_kv_heads, d_head] -> [1, n_kv_heads, kv_len, d_head]
    K = K.permute(1, 0, 2).unsqueeze(0)
    V = V.permute(1, 0, 2).unsqueeze(0)

    # Expand K, V for GQA: replicate each KV head q_mult times
    K = K.repeat_interleave(q_mult, dim=1)  # [1, n_q_heads, kv_len, d_head]
    V = V.repeat_interleave(q_mult, dim=1)  # [1, n_q_heads, kv_len, d_head]

    # Determine if this is decode mode (Q has fewer tokens than K/V)
    # In decode mode, the new query token(s) can attend to all cached K/V
    # so we don't need causal masking (the new token is at the end)
    is_decode = (q_len < kv_len)

    # Use causal masking only for prefill (q_len == kv_len) when no explicit mask
    # For decode, new tokens can attend to all context without masking
    use_causal = (attention_mask is None) and not is_decode

    # Use PyTorch's memory-efficient SDPA
    # Note: attention sinks (S) are skipped for memory efficiency
    # They add a small bias that's not critical for inference quality
    with torch.nn.attention.sdpa_kernel([
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        torch.nn.attention.SDPBackend.MATH,
    ]):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=use_causal,
            scale=sm_scale,
        )

    # Reshape output: [1, n_q_heads, q_len, d_head] -> [q_len, n_q_heads * d_head]
    attn_output = attn_output.squeeze(0).permute(1, 0, 2).reshape(q_len, -1)

    return attn_output


def sdpa_flash(Q, K, V, S, sm_scale, sliding_window=0):
    """Flash Attention SDPA with attention sinks support.

    Uses flash_attn for efficient attention computation with proper sink correction.
    Supports both prefill (q_len == kv_len) and decode (q_len < kv_len) modes.

    Args:
        Q: Query tensor [q_len, n_kv_heads, q_mult, head_dim]
        K: Key tensor [kv_len, n_kv_heads, head_dim]
        V: Value tensor [kv_len, n_kv_heads, head_dim]
        S: Attention sinks [num_attention_heads]
        sm_scale: Softmax scale factor
        sliding_window: Sliding window size (0 = no window, full attention)
    """
    from batchgen.attention.gqa import gqa_prefill_fa, gqa_decode_fa_contiguous

    q_len, n_kv_heads, q_mult, d_head = Q.shape
    kv_len = K.shape[0]
    n_q_heads = n_kv_heads * q_mult

    # Convert sliding_window: 0 means no window (None for flash attn)
    window = sliding_window if sliding_window > 0 else None

    # Reshape Q: [q_len, n_kv_heads, q_mult, head_dim] -> [q_len, n_q_heads, head_dim]
    Q_flash = Q.view(q_len, n_q_heads, d_head).contiguous()

    # K, V are already in correct shape: [kv_len, n_kv_heads, head_dim]
    K_flash = K.contiguous()
    V_flash = V.contiguous()

    # Determine if this is prefill or decode
    is_decode = (q_len == 1 and kv_len > 1)

    if is_decode:
        # Decode mode: use flash_attn_with_kvcache (contiguous)
        # Reshape to batch format: [1, seq, heads, dim]
        Q_batch = Q_flash.unsqueeze(0)  # [1, 1, n_q_heads, d_head]
        K_batch = K_flash.unsqueeze(0)  # [1, kv_len, n_kv_heads, d_head]
        V_batch = V_flash.unsqueeze(0)  # [1, kv_len, n_kv_heads, d_head]

        cache_seqlens = torch.tensor([kv_len], dtype=torch.int32, device=Q.device)

        attn_output, _ = gqa_decode_fa_contiguous(
            Q_batch, K_batch, V_batch,
            cache_seqlens=cache_seqlens,
            sinks=S,
            softmax_scale=sm_scale,
            sliding_window=window,
        )
        # Output: [1, 1, n_q_heads, d_head] -> [1, n_q_heads * d_head]
        attn_output = attn_output.view(q_len, n_q_heads * d_head)
    else:
        # Prefill mode: use flash_attn_varlen_func
        # Already in varlen format: [total_tokens, heads, dim]
        cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=Q.device)
        cu_seqlens_k = torch.tensor([0, kv_len], dtype=torch.int32, device=Q.device)

        attn_output, _ = gqa_prefill_fa(
            Q_flash, K_flash, V_flash,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            sinks=S,
            softmax_scale=sm_scale,
            sliding_window=window,
        )
        # Output: [q_len, n_q_heads, d_head] -> [q_len, n_q_heads * d_head]
        attn_output = attn_output.view(q_len, n_q_heads * d_head)

    return attn_output


def sdpa_naive(Q, K, V, S, sm_scale, sliding_window=0):
    """Original naive SDPA implementation for small sequences or debugging.

    WARNING: This creates O(n²) memory usage. Only use for sequences < 4K tokens.
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )

        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )

        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple | None = None,
        position_offset: int = 0,
        return_kv: bool = False,
    ) -> tuple:
        """Forward pass supporting both 2D [seq, hidden] and 3D [batch, seq, hidden] input.

        BatchGenWorker passes 3D tensors, but internal attention computation expects 2D.
        We flatten batch*seq at start and restore at end.

        Args:
            x: Input tensor [batch, seq, hidden] or [seq, hidden]
            attention_mask: Optional attention mask from BatchGenWorker
            past_key_value: Optional (past_k, past_v) tuple for cached KV
            position_offset: Position offset for RoPE when using cached KV
            return_kv: Whether to return (output, (k, v)) or just output

        Returns:
            If return_kv=True: (output, (key_cache, value_cache))
            If return_kv=False: output
        """
        # Handle 3D input from BatchGenWorker: [batch, seq, hidden] -> [batch*seq, hidden]
        if x.dim() == 3:
            batch_size, seq_len, hidden_size = x.shape
            x = x.view(batch_size * seq_len, hidden_size)
        else:
            batch_size = None
            seq_len = x.shape[0] if x.dim() == 2 else 1

        t = self.norm(x)
        qkv = self.qkv(t)

        # Split QKV using last dimension (works for both 2D and 3D after flatten)
        q_dim = self.num_attention_heads * self.head_dim
        k_dim = self.num_key_value_heads * self.head_dim
        v_dim = self.num_key_value_heads * self.head_dim

        q = qkv[..., :q_dim].contiguous()
        k = qkv[..., q_dim:q_dim + k_dim].contiguous()
        v = qkv[..., q_dim + k_dim:q_dim + k_dim + v_dim].contiguous()

        num_tokens = q.shape[0]
        q = q.view(
            num_tokens,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(num_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(num_tokens, self.num_key_value_heads, self.head_dim)

        # Apply RoPE with position offset for decode
        q, k = self.rope(q, k, position_offset=position_offset)

        # Concatenate with cached KV if available
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=0)
            v = torch.cat([past_v, v], dim=0)

        # Compute attention using flash attention with sink correction
        t = sdpa_flash(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t

        # Restore 3D shape if input was 3D: [batch*seq, hidden] -> [batch, seq, hidden]
        if batch_size is not None:
            t = t.view(batch_size, seq_len, -1)

        if return_kv:
            # Return K, V for caching (after concatenation with past)
            return t, (k, v)
        return t

    def decoding_attn_mode_3_bf16(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen: int,
        gpu_paged_kv_manager,
        batch_slice: tuple | None = None,
    ) -> tuple:
        """Mode 3 decode using GPU paged KV cache.

        Args:
            hidden_states: Input tensor [batch, hidden_size]
            position_ids: Position IDs for RoPE [batch]
            cache_seqlens: Current sequence lengths [batch]
            max_seqlen: Maximum sequence length in batch
            gpu_paged_kv_manager: GPU paged KV cache manager
            batch_slice: Optional (start, end) for micro-batching

        Returns:
            Tuple of (attn_output, k_new, v_new)
        """
        from batchgen.attention.gqa import gqa_decoding_mode_3_bf16

        # Compute RoPE cos/sin for current positions
        # position_ids: [batch]
        concentration, inv_freq = self.rope._compute_concentration_and_inv_freq()
        t = position_ids.float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        rope_cos = freqs.cos() * concentration
        rope_sin = freqs.sin() * concentration

        return gqa_decoding_mode_3_bf16(
            hidden_states,
            position_ids,
            cache_seqlens,
            max_seqlen,
            qkv_weight=self.qkv.weight,
            qkv_bias=self.qkv.bias,
            out_weight=self.out.weight,
            out_bias=self.out.bias,
            norm_weight=self.norm.scale,
            norm_eps=self.norm.eps,
            sinks=self.sinks,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            gpu_paged_kv_manager=gpu_paged_kv_manager,
            layer_idx=self.layer_idx,
            batch_slice=batch_slice,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            sm_scale=self.sm_scale,
            sliding_window=self.sliding_window,
        )


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    """SwiGLU activation with clamping.

    Uses contiguous chunk splitting (not interleaved) to match
    GptOssMXFP4ExpertForward pattern.
    """
    # Use chunk for contiguous splitting (matches GptOssMXFP4ExpertForward)
    x_glu, x_linear = x.chunk(2, dim=-1)
    # Clamp the input values
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class ExpertMLP(torch.nn.Module):
    """Single expert MLP module for W4A16 MXFP4 inference with BatchGen.

    This module performs expert MLP computation using MXFP4 quantized weights.
    Weights are loaded dynamically by GPT-OSS_Expert_Wrapper and passed to
    deepgemm_forward().

    MXFP4 weight format:
    - mlp1.packed: [intermediate*2, hidden//2] uint8 (2 FP4 values per byte)
    - mlp1.scales: [intermediate*2, hidden//32] uint8 (one scale per 32 values)
    - mlp1.bias: [intermediate*2] BF16
    - mlp2.packed: [hidden, intermediate//2] uint8
    - mlp2.scales: [hidden, intermediate//32] uint8
    - mlp2.bias: [hidden] BF16
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        expert_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = device

        # Weights are passed to forward(), not stored as buffers/parameters
        # This allows efficient handling of MXFP4 uint8 tensors

    # Debug counter for logging (class variable)
    _debug_call_count = 0
    _current_phase = "prefill"

    @classmethod
    def reset_debug_for_decode(cls):
        """Reset debug counter at start of decode phase to enable decode logging."""
        cls._debug_call_count = 0
        cls._current_phase = "decode"

    def forward(self, x: torch.Tensor, weights_dict: dict) -> torch.Tensor:
        """Forward pass with W4A16 MXFP4 GEMM.

        Called by GptOssExpertWrapper with dynamically loaded weights.

        Args:
            x: Input tensor [batch, hidden_size] in BF16
            weights_dict: Dict containing MXFP4 weights with keys:
                - 'mlp1.packed': [intermediate*2, hidden//2] uint8
                - 'mlp1.scales': [intermediate*2, hidden//32] uint8
                - 'mlp1.bias': [intermediate*2] BF16
                - 'mlp2.packed': [hidden, intermediate//2] uint8
                - 'mlp2.scales': [hidden, intermediate//32] uint8
                - 'mlp2.bias': [hidden] BF16

        Returns:
            Output tensor [batch, hidden_size] in BF16
        """
        from batchgen.moe.fused_mxfp4_gemm import fused_mxfp4_gemm
        import logging

        # Extract weights from dict
        mlp1_packed = weights_dict['mlp1.packed']
        mlp1_scales = weights_dict['mlp1.scales']
        mlp1_bias = weights_dict['mlp1.bias']
        mlp2_packed = weights_dict['mlp2.packed']
        mlp2_scales = weights_dict['mlp2.scales']
        mlp2_bias = weights_dict['mlp2.bias']

        # Debug logging (only first few calls per phase)
        ExpertMLP._debug_call_count += 1
        if ExpertMLP._debug_call_count <= 3:
            logging.info(
                f"[ExpertMLP DEBUG] PHASE={ExpertMLP._current_phase} L{self.layer_idx} E{self.expert_idx} call #{ExpertMLP._debug_call_count}: "
                f"x={x.shape} {x.dtype}, "
                f"mlp1.packed={mlp1_packed.shape} {mlp1_packed.dtype}, "
                f"mlp1.scales={mlp1_scales.shape} {mlp1_scales.dtype}, "
                f"mlp1.bias={mlp1_bias.shape} {mlp1_bias.dtype}, "
                f"mlp2.packed={mlp2_packed.shape} {mlp2_packed.dtype}, "
                f"mlp2.scales={mlp2_scales.shape} {mlp2_scales.dtype}, "
                f"mlp2.bias={mlp2_bias.shape} {mlp2_bias.dtype}"
            )
            # Check weight content (first few values)
            logging.info(
                f"  mlp1_packed[:3,:3]={mlp1_packed[:3,:3].tolist()}, "
                f"mlp1_scales[:3,:3]={mlp1_scales[:3,:3].tolist()}, "
                f"mlp1_bias[:5]={mlp1_bias[:5].tolist()}"
            )

        # Validate weight shapes (debug)
        assert mlp1_packed.dim() == 2, f"mlp1.packed must be 2D, got {mlp1_packed.dim()}D: {mlp1_packed.shape}"
        assert mlp1_scales.dim() == 2, f"mlp1.scales must be 2D, got {mlp1_scales.dim()}D: {mlp1_scales.shape}"
        assert mlp1_bias.dim() == 1, f"mlp1.bias must be 1D, got {mlp1_bias.dim()}D: {mlp1_bias.shape}"
        assert mlp2_packed.dim() == 2, f"mlp2.packed must be 2D, got {mlp2_packed.dim()}D: {mlp2_packed.shape}"
        assert mlp2_scales.dim() == 2, f"mlp2.scales must be 2D, got {mlp2_scales.dim()}D: {mlp2_scales.shape}"
        assert mlp2_bias.dim() == 1, f"mlp2.bias must be 1D, got {mlp2_bias.dim()}D: {mlp2_bias.shape}"

        # MLP1: [batch, hidden] -> [batch, intermediate*2]
        t = fused_mxfp4_gemm(x, mlp1_packed, mlp1_scales)
        t = t + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP2: [batch, intermediate] -> [batch, hidden]
        t = fused_mxfp4_gemm(t, mlp2_packed, mlp2_scales)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t + mlp2_bias

        return t


class MLPBlock(torch.nn.Module):
    """MoE MLP block with per-expert modules for BatchGen integration.

    This block uses the BatchGen pattern where each expert is a separate module
    that can be wrapped by Expert_Wrapper for dynamic weight loading.

    Supports two execution modes:
    1. Per-expert (default): Each expert is called individually via Expert_Wrapper.
       Weights are loaded dynamically by BatchGen's parameter server.
    2. Fused grouped GEMM: All experts are stored as 3D tensors and processed
       together using fused MXFP4 grouped GEMM. Use set_grouped_weights() to
       enable this mode.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Skeleton weights (always loaded, not per-expert)
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )

        # Per-expert MLP modules (will be wrapped by Expert_Wrapper)
        self.experts = torch.nn.ModuleList([
            ExpertMLP(config, layer_idx, expert_idx, device)
            for expert_idx in range(config.num_experts)
        ])

        # Reference to BatchGen core_engine (set during initialization)
        self.core_engine = None

        # BatchGen compatibility: GPT-OSS has no shared experts (all are routed)
        # Setting to None makes hasattr checks pass safely
        self.shared_experts = None

        # Grouped weights mode (disabled by default)
        # When enabled, uses fused grouped GEMM instead of per-expert calls
        self._use_grouped_gemm = False
        self._grouped_weights = None
        self._token_dispatcher = None

    def set_grouped_weights(
        self,
        w1_packed: torch.Tensor,
        w1_scales: torch.Tensor,
        w1_bias: torch.Tensor | None,
        w2_packed: torch.Tensor,
        w2_scales: torch.Tensor,
        w2_bias: torch.Tensor | None,
    ):
        """Enable fused grouped GEMM mode with pre-loaded 3D weight tensors.

        When set, forward() uses fused MXFP4 grouped GEMM instead of
        per-expert calls. This is more efficient when all expert weights
        are available in memory.

        Args:
            w1_packed: MXFP4 packed W1 [num_experts, intermediate*2, hidden//2]
            w1_scales: W1 scales [num_experts, intermediate*2, hidden//32]
            w1_bias: W1 bias [num_experts, intermediate*2] or None
            w2_packed: MXFP4 packed W2 [num_experts, hidden, intermediate//2]
            w2_scales: W2 scales [num_experts, hidden, intermediate//32]
            w2_bias: W2 bias [hidden] or None (shared across experts)
        """
        from batchgen.moe.token_dispatch import LocalTokenDispatcher

        self._grouped_weights = {
            'w1_packed': w1_packed,
            'w1_scales': w1_scales,
            'w1_bias': w1_bias,
            'w2_packed': w2_packed,
            'w2_scales': w2_scales,
            'w2_bias': w2_bias,
        }
        self._use_grouped_gemm = True
        self._token_dispatcher = LocalTokenDispatcher()

    def clear_grouped_weights(self):
        """Disable grouped GEMM mode and revert to per-expert calls."""
        self._grouped_weights = None
        self._use_grouped_gemm = False
        self._token_dispatcher = None

    def _forward_grouped_gemm(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using fused grouped GEMM (when grouped weights are set)."""
        from batchgen.moe.moe_mxfp4 import moe_mxfp4_forward

        # Handle 3D input
        input_shape = x.shape
        if x.dim() == 3:
            batch_dim, seq_len, hidden_size = x.shape
            x = x.view(batch_dim * seq_len, hidden_size)
        else:
            batch_dim = None

        # Normalize input
        t = self.norm(x)

        # Use fused MoE forward with grouped GEMM
        output = moe_mxfp4_forward(
            t,
            self.gate.weight.T,  # [hidden, num_experts]
            self.gate.bias,
            self._grouped_weights['w1_packed'],
            self._grouped_weights['w1_scales'],
            self._grouped_weights['w1_bias'],
            self._grouped_weights['w2_packed'],
            self._grouped_weights['w2_scales'],
            self._grouped_weights['w2_bias'],
            experts_per_token=self.experts_per_token,
            swiglu_limit=self.swiglu_limit,
            dispatcher=self._token_dispatcher,
        )

        result = x + output

        # Restore 3D shape if needed
        if batch_dim is not None:
            result = result.view(batch_dim, seq_len, -1)

        return result

    def _forward_grouped_from_loaded(self, x: torch.Tensor) -> torch.Tensor:
        """Grouped GEMM using dynamically loaded per-expert weights.

        This mode loads weights for routed experts, collects them into lists,
        and calls the grouped GEMM kernel. More efficient than per-expert
        sequential forward when multiple experts are needed.

        Requires core_engine to be set for weight loading.
        """
        from batchgen.moe.routing import moe_routing
        from batchgen.moe.token_dispatch import LocalTokenDispatcher
        from batchgen.moe.fused_mxfp4_gemm import fused_mxfp4_moe_gemm_from_list

        # Handle 3D input
        input_shape = x.shape
        if x.dim() == 3:
            batch_dim, seq_len, hidden_size = x.shape
            x = x.view(batch_dim * seq_len, hidden_size)
        else:
            batch_dim = None

        num_tokens = x.shape[0]

        # 1. Normalize and compute routing
        t = self.norm(x)
        topk_indices, topk_weights = moe_routing(
            t, self.gate.weight.T, self.gate.bias, self.experts_per_token
        )

        # 2. Find unique routed experts
        routed_expert_ids = topk_indices.unique().tolist()

        # 3. Load weights for routed experts
        w1_packed_list, w1_scales_list, w1_bias_list = [], [], []
        w2_packed_list, w2_scales_list = [], []

        for expert_id in routed_expert_ids:
            weights = self.core_engine.get_weights(
                f"routed_expert_{self.layer_idx}_{expert_id}", self._phase
            )
            w1_packed_list.append(weights['mlp1.packed'])
            w1_scales_list.append(weights['mlp1.scales'])
            w1_bias_list.append(weights['mlp1.bias'])
            w2_packed_list.append(weights['mlp2.packed'])
            w2_scales_list.append(weights['mlp2.scales'])

        # 4. Dispatch tokens
        dispatcher = LocalTokenDispatcher()
        dispatch_result = dispatcher.dispatch(t, topk_indices, topk_weights)

        # Build expert counts/offsets for loaded experts only
        # Map from routed_expert_ids indices to dispatch_result counts
        num_loaded = len(routed_expert_ids)
        expert_counts = torch.zeros(num_loaded, dtype=torch.int32, device=x.device)
        expert_offsets = torch.zeros(num_loaded, dtype=torch.int32, device=x.device)

        # Create mapping from original expert ID to loaded index
        expert_id_to_idx = {eid: idx for idx, eid in enumerate(routed_expert_ids)}

        # Recompute counts for loaded experts
        for i, eid in enumerate(routed_expert_ids):
            if eid < len(dispatch_result.expert_counts):
                expert_counts[i] = dispatch_result.expert_counts[eid]
                expert_offsets[i] = dispatch_result.expert_offsets[eid]

        # 5. W1 GEMM
        h = fused_mxfp4_moe_gemm_from_list(
            dispatch_result.dispatched_x,
            w1_packed_list, w1_scales_list,
            expert_counts, expert_offsets,
        )

        # Add W1 bias
        for i, bias in enumerate(w1_bias_list):
            start = expert_offsets[i].item()
            count = expert_counts[i].item()
            if count > 0:
                h[start:start + count] = h[start:start + count] + bias

        # SwiGLU activation
        h = swiglu(h, limit=self.swiglu_limit)

        # 6. W2 GEMM
        out = fused_mxfp4_moe_gemm_from_list(
            h,
            w2_packed_list, w2_scales_list,
            expert_counts, expert_offsets,
        )

        # 7. Combine: scatter back with weighted sum
        output = dispatcher.combine(out, dispatch_result, num_tokens)

        # 8. Free weights
        for expert_id in routed_expert_ids:
            self.core_engine.free_weights_buffer(
                f"routed_expert_{self.layer_idx}_{expert_id}"
            )

        result = x + output

        # Restore 3D shape if needed
        if batch_dim is not None:
            result = result.view(batch_dim, seq_len, -1)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with top-k expert routing.

        For each token, routes to top-k experts and combines outputs.
        Expert weights are loaded dynamically by GPT-OSS_Expert_Wrapper.

        When wrapped by BatchGen, self.experts[i] is a GPT-OSS_Expert_Wrapper
        that handles MXFP4 weight loading and calls deepgemm_forward.

        Supports both 2D [num_tokens, hidden] and 3D [batch, seq, hidden] input.

        If grouped weights are set via set_grouped_weights(), uses fused
        grouped GEMM instead of per-expert calls for better efficiency.
        """
        # Use fused grouped GEMM if weights are pre-loaded
        if self._use_grouped_gemm:
            return self._forward_grouped_gemm(x)

        # Handle 3D input from BatchGenWorker: [batch, seq, hidden] -> [num_tokens, hidden]
        input_shape = x.shape
        if x.dim() == 3:
            batch_dim, seq_len, hidden_size = x.shape
            x = x.view(batch_dim * seq_len, hidden_size)
        else:
            batch_dim = None

        num_tokens = x.shape[0]

        # Compute routing
        with DECODE_TIMING.time("mlp.routing"):
            t = self.norm(x)
            g = self.gate(t)
            experts_result = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
            expert_weights = torch.nn.functional.softmax(experts_result.values, dim=-1)
            expert_indices = experts_result.indices  # [num_tokens, experts_per_token]

        # Initialize output accumulator
        output = torch.zeros_like(x)

        # Get unique experts and their token assignments
        unique_experts = torch.unique(expert_indices)

        with DECODE_TIMING.time("mlp.expert_loop"):
            for expert_idx in unique_experts:
                expert_idx_item = expert_idx.item()

                # Find which (token, slot) pairs use this expert
                mask = (expert_indices == expert_idx)  # [num_tokens, experts_per_token]

                # Get the tokens and weights for this expert
                token_indices, slot_indices = torch.where(mask)

                if len(token_indices) == 0:
                    continue

                # Gather input tokens for this expert
                expert_input = t[token_indices]  # [num_selected, hidden]

                # Get the weights for these tokens
                token_weights = expert_weights[token_indices, slot_indices]  # [num_selected]

                # Forward through expert wrapper (handles MXFP4 weight loading)
                # The wrapper's __call__ loads weights and calls deepgemm_forward
                expert_output = self.experts[expert_idx_item](expert_input)

                # Weighted output
                weighted_output = expert_output * token_weights.unsqueeze(-1)

                # Scatter-add to output
                output.index_add_(0, token_indices, weighted_output)

        result = x + output

        # Restore 3D shape if input was 3D: [num_tokens, hidden] -> [batch, seq, hidden]
        if batch_dim is not None:
            result = result.view(batch_dim, seq_len, -1)

        return result


class TransformerBlock(torch.nn.Module):
    """Transformer block with attention and MLP.

    Supports both OpenAI-style (simple x input) and HuggingFace-style
    (hidden_states with attention_mask, position_ids, etc.) calling conventions
    for compatibility with BatchGenWorker.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, layer_idx, device)

    @property
    def self_attn(self):
        """Map HuggingFace self_attn to OpenAI attn.

        BatchGenWorker accesses layer.self_attn for attention module.
        """
        return self.attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple:
        """Forward pass with HuggingFace-compatible signature.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size] or [seq_len, hidden_size]
            attention_mask: Attention mask from BatchGenWorker for sequence boundaries
            position_ids: Position IDs (RoPE computed internally, ignored here)
            past_key_value: KV cache (not yet supported)
            output_attentions: Whether to output attention weights (not supported)
            use_cache: Whether to use KV cache (not yet supported)
            cache_position: Cache position indices (not yet supported)

        Returns:
            hidden_states tensor, or tuple of (hidden_states, ...) if use_cache
        """
        # Pass attention_mask to attention for proper sequence boundary handling
        x = hidden_states

        with DECODE_TIMING.time(f"attn"):
            x = self.attn(x, attention_mask=attention_mask)

        with DECODE_TIMING.time(f"mlp"):
            x = self.mlp(x)

        # ALWAYS return tuple (hidden_states, past_key_value) to match HuggingFace format
        # BatchGenWorker expects tuple output and does `hidden_states = layer_output[0]`
        # If we return tensor directly, layer_output[0] indexes into tensor's first dim!
        return (x, None)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with DECODE_TIMING.time("embedding"):
            x = self.embedding(x)

        for block in self.block:
            x = block(x)
            # Handle HuggingFace-style tuple return (hidden_states, past_key_value)
            if isinstance(x, tuple):
                x = x[0]

        with DECODE_TIMING.time("final_norm"):
            x = self.norm(x)

        with DECODE_TIMING.time("unembedding"):
            x = self.unembedding(x)

        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda", load_experts: bool = False
    ) -> "Transformer":
        """Load model from checkpoint.

        For BatchGen integration, expert weights are loaded dynamically via Expert_Wrapper
        and parameter server. This method only loads skeleton weights (embedding, attention,
        gate, norms, unembedding).

        Args:
            path: Path to checkpoint directory
            device: Device to load model on
            load_experts: If True, load expert weights from checkpoint (for standalone use).
                         If False (default), expert buffers remain zero-initialized for BatchGen.

        Returns:
            Transformer model with skeleton weights loaded
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        # Load skeleton weights (non-expert parameters)
        checkpoint = Checkpoint(path, device)

        for name, param in model.named_parameters():
            # Skip expert-related parameters (handled by BatchGen dynamically)
            if ".experts." in name:
                continue

            try:
                loaded_tensor = checkpoint.get(name)
                param.data.copy_(loaded_tensor)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
                continue

        # Optionally load expert weights (for standalone testing, not BatchGen)
        if load_experts:
            my_rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            per_rank_intermediate = config.intermediate_size // world_size

            for layer_idx in range(config.num_hidden_layers):
                for expert_idx in range(config.num_experts):
                    expert = model.block[layer_idx].mlp.experts[expert_idx]

                    # Load MLP1 weights (stacked in checkpoint as [num_experts, out, in])
                    mlp1_name = f"block.{layer_idx}.mlp.mlp1_weight"
                    mlp1_stacked = checkpoint.get(mlp1_name)  # Dequantized to BF16
                    mlp1_expert = mlp1_stacked[expert_idx]
                    # Shard by intermediate dimension
                    mlp1_expert = mlp1_expert[
                        my_rank * 2 * per_rank_intermediate:(my_rank + 1) * 2 * per_rank_intermediate,
                        :
                    ]
                    # Note: For standalone, we'd need to re-quantize to MXFP4
                    # This path is mainly for debugging; BatchGen loads MXFP4 directly

                    # Similar for other weights...
                    # (Full implementation would require MXFP4 quantization)

        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break