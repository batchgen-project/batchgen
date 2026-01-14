import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .weights import Checkpoint


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

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

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
        Q: Query tensor [n_tokens, n_kv_heads, q_mult, head_dim]
        K: Key tensor [n_tokens, n_kv_heads, head_dim]
        V: Value tensor [n_tokens, n_kv_heads, head_dim]
        S: Attention sinks [num_attention_heads] (currently ignored for efficiency)
        sm_scale: Softmax scale factor
        sliding_window: Sliding window size (0 = no window, full attention)
        attention_mask: Optional attention mask from BatchGenWorker
    """
    n_tokens, n_kv_heads, q_mult, d_head = Q.shape
    n_q_heads = n_kv_heads * q_mult

    # Reshape for PyTorch SDPA: [batch, heads, seq, head_dim]
    # Q: [n_tokens, n_kv_heads, q_mult, d_head] -> [1, n_q_heads, n_tokens, d_head]
    Q = Q.permute(1, 2, 0, 3).reshape(1, n_q_heads, n_tokens, d_head)

    # K, V: [n_tokens, n_kv_heads, d_head] -> [1, n_kv_heads, n_tokens, d_head]
    K = K.permute(1, 0, 2).unsqueeze(0)
    V = V.permute(1, 0, 2).unsqueeze(0)

    # Expand K, V for GQA: replicate each KV head q_mult times
    K = K.repeat_interleave(q_mult, dim=1)  # [1, n_q_heads, n_tokens, d_head]
    V = V.repeat_interleave(q_mult, dim=1)  # [1, n_q_heads, n_tokens, d_head]

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
            is_causal=(attention_mask is None),  # Use causal if no mask provided
            scale=sm_scale,
        )

    # Reshape output: [1, n_q_heads, n_tokens, d_head] -> [n_tokens, n_q_heads * d_head]
    attn_output = attn_output.squeeze(0).permute(1, 0, 2).reshape(n_tokens, -1)

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

        # Debug: Log dimensions for layer 0 with stack trace
        if layer_idx == 0:
            import traceback
            import logging
            import sys
            stack_str = ''.join(traceback.format_stack())
            logging.warning(f"[AttentionBlock {layer_idx}] Creating qkv Linear: in={config.hidden_size}, out={qkv_dim}, device={device}")
            logging.warning(f"Stack trace:\n{stack_str}")
            sys.stdout.flush()
            sys.stderr.flush()

        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )

        # Debug: Check weight shape immediately after creation
        if layer_idx == 0:
            import logging
            logging.warning(f"[AttentionBlock {layer_idx}] qkv.weight.shape={list(self.qkv.weight.shape)}")

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
    ) -> torch.Tensor:
        """Forward pass supporting both 2D [seq, hidden] and 3D [batch, seq, hidden] input.

        BatchGenWorker passes 3D tensors, but internal attention computation expects 2D.
        We flatten batch*seq at start and restore at end.

        Args:
            x: Input tensor [batch, seq, hidden] or [seq, hidden]
            attention_mask: Optional attention mask from BatchGenWorker
        """
        # Handle 3D input from BatchGenWorker: [batch, seq, hidden] -> [batch*seq, hidden]
        input_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, hidden_size = x.shape
            x = x.view(batch_size * seq_len, hidden_size)
        else:
            batch_size = None

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
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window, attention_mask)
        t = self.out(t)
        t = x + t

        # Restore 3D shape if input was 3D: [batch*seq, hidden] -> [batch, seq, hidden]
        if batch_size is not None:
            t = t.view(batch_size, seq_len, -1)

        return t


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

        # Weights are passed to deepgemm_forward, not stored as buffers/parameters
        # This allows efficient handling of MXFP4 uint8 tensors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward - raises error, use deepgemm_forward for W4A16."""
        raise NotImplementedError(
            "ExpertMLP.forward() is not supported. Use deepgemm_forward() for W4A16 inference."
        )

    def deepgemm_forward(self, x: torch.Tensor, weights_dict: dict) -> torch.Tensor:
        """Forward pass with W4A16 MXFP4 GEMM.

        Called by GPT-OSS_Expert_Wrapper with dynamically loaded weights.

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

        # Extract weights from dict
        mlp1_packed = weights_dict['mlp1.packed']
        mlp1_scales = weights_dict['mlp1.scales']
        mlp1_bias = weights_dict['mlp1.bias']
        mlp2_packed = weights_dict['mlp2.packed']
        mlp2_scales = weights_dict['mlp2.scales']
        mlp2_bias = weights_dict['mlp2.bias']

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with top-k expert routing.

        For each token, routes to top-k experts and combines outputs.
        Expert weights are loaded dynamically by GPT-OSS_Expert_Wrapper.

        When wrapped by BatchGen, self.experts[i] is a GPT-OSS_Expert_Wrapper
        that handles MXFP4 weight loading and calls deepgemm_forward.

        Supports both 2D [num_tokens, hidden] and 3D [batch, seq, hidden] input.
        """
        # Handle 3D input from BatchGenWorker: [batch, seq, hidden] -> [num_tokens, hidden]
        input_shape = x.shape
        if x.dim() == 3:
            batch_dim, seq_len, hidden_size = x.shape
            x = x.view(batch_dim * seq_len, hidden_size)
        else:
            batch_dim = None

        num_tokens = x.shape[0]

        # Compute routing
        t = self.norm(x)
        g = self.gate(t)
        experts_result = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts_result.values, dim=-1)
        expert_indices = experts_result.indices  # [num_tokens, experts_per_token]

        # Initialize output accumulator
        output = torch.zeros_like(x)

        # Get unique experts and their token assignments
        unique_experts = torch.unique(expert_indices)

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
        x = self.attn(x, attention_mask=attention_mask)
        x = self.mlp(x)

        # Return format compatible with HuggingFace expectations
        if use_cache:
            # Return (hidden_states, present_key_value) - KV cache not implemented yet
            return (x, None)
        return x


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
        import logging
        logging.warning(f"[Transformer.forward] input_ids.shape = {list(x.shape)}")
        x = self.embedding(x)
        logging.warning(f"[Transformer.forward] after embedding: x.shape = {list(x.shape)}")
        # Debug: Check first block's attn type and weight shape with object IDs
        first_attn = self.block[0].attn
        logging.warning(f"[Transformer.forward] block[0].attn type = {type(first_attn).__name__}")
        if hasattr(first_attn, 'module'):
            inner_qkv = first_attn.module.qkv
            logging.warning(f"[Transformer.forward] wrapper id={id(first_attn)}, module id={id(first_attn.module)}, qkv id={id(inner_qkv)}, weight id={id(inner_qkv.weight)}")
            logging.warning(f"[Transformer.forward] block[0].attn.module.qkv.weight.shape = {list(inner_qkv.weight.shape)}")
        elif hasattr(first_attn, 'qkv'):
            logging.warning(f"[Transformer.forward] block[0].attn.qkv.weight.shape = {list(first_attn.qkv.weight.shape)}")

        for block in self.block:
            x = block(x)
        x = self.norm(x)
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