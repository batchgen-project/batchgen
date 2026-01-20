# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""GPT-OSS-specific wrappers for BatchGen execution.

Provides wrappers for GPT-OSS-120B model with MXFP4 quantization:
- GptOssExpertWrapper: Expert wrapper with MXFP4 dequantization
- GptOssAttnWrapper: Attention wrapper with GQA and sink tokens

Optimized for single H20 GPU deployment (world_size == 1).
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.attention.gqa import gqa_attention_with_sinks, gqa_decode_fa
from batchgen.attention.sink import softmax_with_sinks


class GptOssExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with MXFP4 dequantization for GPT-OSS-120B.

    For world_size == 1 (single H20 GPU):
    - All 128 experts are local
    - No expert parallelism needed
    - Weights loaded from core_engine and dequantized on-the-fly

    MXFP4 format:
    - 32 FP4 values per scale
    - Packed as 2 values per uint8 byte
    - Scale stored as uint8, exponent = scale - 127

    Attributes:
        dequant_fn: MXFP4 dequantization function
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
    ):
        """Initialize GPT-OSS expert wrapper.

        Args:
            module: Expert FFN module (SwiGLU)
            layer_idx: Layer index in the model (0-35)
            expert_idx: Expert index (0-127)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
        """
        # GPT-OSS always loads weights (all experts local, no caching)
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent=False  # Load weights each forward (MXFP4 dequantization needed)
        )

        # Import MXFP4 dequantization
        try:
            from batchgen.quantization.mxfp4 import mxfp4_dequantize
            self.dequant_fn = mxfp4_dequantize
        except ImportError:
            logging.warning("MXFP4 dequantization not available, using identity")
            self.dequant_fn = lambda packed, scales, dtype=torch.bfloat16: packed

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize MXFP4 packed weights to BF16.

        Expected weight format in weights_dict:
        - "gate_proj.weight": packed uint8 tensor
        - "gate_proj.weight_scales": scale uint8 tensor
        - "up_proj.weight": packed uint8 tensor
        - "up_proj.weight_scales": scale uint8 tensor
        - "down_proj.weight": packed uint8 tensor
        - "down_proj.weight_scales": scale uint8 tensor
        - "gate_proj.bias", "up_proj.bias", "down_proj.bias": BF16 biases

        Args:
            weights_dict: Dict with packed weights and scales

        Returns:
            Dict with dequantized BF16 weights
        """
        result = {}

        for name, tensor in weights_dict.items():
            # Skip scale tensors - they're used with their corresponding weights
            if name.endswith("_scales"):
                continue

            # Check if this weight has a scale tensor
            scale_key = f"{name}_scales"
            if scale_key in weights_dict:
                # MXFP4 quantized weight - dequantize
                packed = tensor
                scales = weights_dict[scale_key]
                result[name] = self.dequant_fn(packed, scales, torch.bfloat16)
            else:
                # Not quantized (bias) - use as-is
                result[name] = tensor

        return result

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward with clamping.

        GPT-OSS uses clamped SwiGLU: (gate * sigmoid(a*gate)) * (up + 1)

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        return self.module(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with MXFP4 dequantization.

        Flow:
        1. Load MXFP4 packed weights from core engine
        2. Dequantize to BF16
        3. Apply to module
        4. Micro-batch forward through SwiGLU
        5. Cleanup

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"GPT-OSS expert forward. Phase: {self.phase}"
        )

        # Load MXFP4 weights from core engine
        weights = self.load_weights(self.module_key)

        # Dequantize MXFP4 to BF16
        dequant_weights = self.dequantize_weights(weights)

        # Apply to module
        self.apply_weights(dequant_weights)

        # Micro-batch forward
        result = self.micro_batch_forward(hidden_states, "expert")

        # Cleanup
        torch.cuda.current_stream(
            self.engine_config.Basic_Config.device_torch
        ).synchronize()
        self.free_weights(self.module_key)
        self.clear_weights()

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"GPT-OSS expert forward complete. Phase: {self.phase}"
        )

        return result


class GptOssAttnWrapper(AttnWrapperBase):
    """Attention wrapper for GPT-OSS-120B with GQA and sink tokens.

    GPT-OSS attention features:
    - BF16 weights (not quantized)
    - GQA with 64 query heads, 8 KV heads
    - Learned sink tokens per query head
    - Alternating sliding (128) / full attention per layer

    The wrapper delegates attention computation to GQA kernels in
    batchgen/attention/gqa/ with sink token support from batchgen/attention/sink/.

    Attributes:
        is_sliding: Whether this layer uses sliding window attention
        sliding_window: Window size for sliding attention (128 or None)
        num_heads: Number of query heads (64)
        num_kv_heads: Number of KV heads (8)
        head_dim: Dimension per head (64)
        sinks: Learned sink token parameters
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
    ):
        """Initialize GPT-OSS attention wrapper.

        Args:
            module: Attention module (GQA)
            layer_idx: Layer index (0-35)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
        """
        # GPT-OSS attention is BF16, no dequantization needed
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent=False, weight_dequant_scale=None  # Load weights each forward
        )

        # Architecture parameters
        self.num_heads = model_config.num_attention_heads  # 64
        self.num_kv_heads = model_config.num_key_value_heads  # 8
        self.head_dim = model_config.head_dim  # 64
        self.num_groups = self.num_heads // self.num_kv_heads  # 8
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Determine if this layer uses sliding window
        # GPT-OSS uses alternating: even layers = sliding, odd = full
        self.is_sliding = (layer_idx % 2 == 0)
        self.sliding_window = model_config.sliding_window if self.is_sliding else None

        # Sink token parameter will be loaded with weights
        self.sinks = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract and handle attention weights including sinks.

        Attention weights are in BF16, no dequantization needed.
        Also extracts sink tokens from the loaded weights.

        Args:
            weights_dict: Dict with attention weights and sinks

        Returns:
            Dict with weights to apply (sinks handled separately)
        """
        # Extract sinks from weights if present
        if "sinks" in weights_dict:
            self.sinks = weights_dict["sinks"]
            logging.debug(f"Layer {self.layer_idx}: Loaded sinks with shape {self.sinks.shape}")
            # Remove from dict so it's not applied as a regular parameter
            result = {k: v for k, v in weights_dict.items() if k != "sinks"}
            return result

        # If no sinks in weights, initialize with zeros
        if self.sinks is None:
            self.sinks = torch.zeros(self.num_heads, dtype=torch.bfloat16)
            logging.debug(f"Layer {self.layer_idx}: Initialized zero sinks")

        return weights_dict

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_decode: bool = False,
        cache_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute GQA attention with sinks using dedicated kernels.

        Args:
            query: [batch, num_q_heads, seq_q, head_dim]
            key: [batch, num_kv_heads, seq_k, head_dim]
            value: [batch, num_kv_heads, seq_k, head_dim]
            is_decode: Whether in decode mode (single token)
            cache_seqlens: Current sequence lengths for decode [batch]

        Returns:
            Attention output [batch, seq_q, num_heads * head_dim]
        """
        batch, num_heads, seq_q, head_dim = query.shape

        # Use cache_seqlens from parameter, fall back to class attribute
        seqlens = cache_seqlens
        if seqlens is None:
            seqlens = AttnWrapperBase.cache_seqlens

        # Use GQA attention with sinks from batchgen/attention/gqa/
        output = gqa_attention_with_sinks(
            query=query,
            key=key,
            value=value,
            sinks=self.sinks,
            scale=self.scale,
            sliding_window=self.sliding_window,
            is_decode=is_decode,
            cache_seqlens=seqlens,
        )

        # Transpose and reshape: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_q, num_heads * head_dim)

        return output

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Prefill forward with GQA and sink tokens.

        Uses module's projections but delegates attention to GQA kernels.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional KV cache from previous steps
            output_attentions: Whether to return attention weights (not supported with sinks)
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (output, attention_weights, new_kv_cache)
        """
        batch, seq_len, _ = hidden_states.shape

        # Project Q, K, V using module's projections
        query = self.module.q_proj(hidden_states)
        key = self.module.k_proj(hidden_states)
        value = self.module.v_proj(hidden_states)

        # Reshape for attention: [batch, seq, heads, head_dim]
        query = query.view(batch, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Get RoPE embeddings
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1]

        cos, sin = self.module.rotary_emb(value.transpose(1, 2), seq_len=kv_seq_len)

        # Get cos/sin for positions
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        else:
            cos = cos[:seq_len]
            sin = sin[:seq_len]

        # Apply RoPE
        query, key = self._apply_rotary(query, key, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        new_kv_cache = (key, value) if use_cache else None

        # Transpose to [batch, heads, seq, head_dim] for attention
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        # Compute attention using GQA kernels with sinks
        attn_output = self._compute_attention(query_t, key_t, value_t, is_decode=False)

        # Output projection
        attn_output = self.module.o_proj(attn_output)

        return attn_output, None, new_kv_cache

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        batch_slice: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Decode forward for single token generation using gpu_paged_kv_manager.

        For BatchGen decode, KV cache is managed by gpu_paged_kv_manager (set as
        class attribute on AttnWrapperBase). This method:
        1. Projects Q, K, V for the new token
        2. Applies RoPE
        3. Writes new K, V to paged GPU cache via gpu_paged_kv_manager
        4. Retrieves full K, V cache for attention
        5. Runs GQA attention with sinks

        Args:
            hidden_states: Input tensor [batch, 1, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Position IDs for the current token
            past_key_value: Ignored - using gpu_paged_kv_manager instead
            output_attentions: Whether to return attention weights
            use_cache: Whether to return KV cache (ignored, always uses paged)
            batch_slice: Optional (start_idx, end_idx) for micro-batching

        Returns:
            Tuple of (output, None, None) - KV cache managed externally
        """
        batch, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "Decode expects single token"

        # Get gpu_paged_kv_manager and cache_seqlens from class-level state
        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        cache_seqlens = AttnWrapperBase.cache_seqlens

        if gpu_kv_manager is None:
            # Fallback to legacy tuple-based cache if paged manager not set
            logging.warning(
                f"Layer {self.layer_idx}: gpu_paged_kv_manager not set, "
                "falling back to tuple-based KV cache"
            )
            return self._forward_decode_legacy(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs
            )

        # Project Q, K, V for the new token
        query = self.module.q_proj(hidden_states)
        key = self.module.k_proj(hidden_states)
        value = self.module.v_proj(hidden_states)

        # Reshape: [batch, 1, num_heads, head_dim]
        query = query.view(batch, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Get RoPE for current position
        # cache_seqlens contains the current position (0-indexed)
        if cache_seqlens is not None:
            # Apply batch_slice if provided
            if batch_slice is not None:
                start_idx, end_idx = batch_slice
                micro_cache_seqlens = cache_seqlens[start_idx:end_idx]
            else:
                micro_cache_seqlens = cache_seqlens

            max_seqlen = int(micro_cache_seqlens.max().item()) + 1
            cos, sin = self.module.rotary_emb(value.transpose(1, 2), seq_len=max_seqlen)

            # Apply RoPE at each sequence's current position
            if position_ids is not None:
                cos = cos[position_ids]
                sin = sin[position_ids]
            else:
                # Use cache_seqlens as position_ids (current token position)
                cos = cos[micro_cache_seqlens]
                sin = sin[micro_cache_seqlens]
        else:
            # Fallback if cache_seqlens not set
            cos, sin = self.module.rotary_emb(value.transpose(1, 2), seq_len=1)
            cos = cos[:1]
            sin = sin[:1]

        query, key = self._apply_rotary(query, key, cos, sin)

        # Write new K, V to paged GPU cache
        # Shape requirement: [batch, seq_len, num_heads, head_dim]
        gpu_kv_manager.update_layer_decode_new_token(
            k_tensor=key,
            v_tensor=value,  # GQA needs V cache (unlike MLA)
            sequence_lengths=micro_cache_seqlens if cache_seqlens is not None else torch.zeros(batch, dtype=torch.int32, device=hidden_states.device),
            layer_idx=self.layer_idx,
            batch_slice=batch_slice,
        )

        # Retrieve paged K, V cache and page table for FlashAttention
        k_cache_layer, v_cache_layer, page_table = gpu_kv_manager.get_layer_kv_with_page_table(
            self.layer_idx
        )

        # Apply batch slice to page_table for micro-batching
        if batch_slice is not None:
            start_idx, end_idx = batch_slice
            page_table = page_table[start_idx:end_idx]

        # Use FlashAttention with paged KV cache
        # gqa_decode_fa expects:
        #   q: [batch, seqlen_q, nheads, headdim]
        #   k_cache: [num_blocks, page_size, nheads_kv, headdim]
        #   v_cache: [num_blocks, page_size, nheads_kv, headdim]
        #   block_table: [batch, max_blocks_per_seq]
        #   cache_seqlens: [batch] - needs +1 because we just wrote new token
        cache_seqlens_for_attn = micro_cache_seqlens + 1 if cache_seqlens is not None else torch.ones(batch, dtype=torch.int32, device=hidden_states.device)

        attn_output, _ = gqa_decode_fa(
            q=query,  # [batch, 1, num_heads, head_dim]
            k_cache=k_cache_layer,  # [num_pages, page_size, num_kv_heads, head_dim]
            v_cache=v_cache_layer,  # [num_pages, page_size, num_kv_heads, head_dim]
            cache_seqlens=cache_seqlens_for_attn,
            block_table=page_table,
            sinks=self.sinks,
            softmax_scale=self.scale,
            sliding_window=self.sliding_window,
        )

        # Reshape output: [batch, 1, num_heads, head_dim] -> [batch, 1, hidden_size]
        attn_output = attn_output.view(batch, 1, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.module.o_proj(attn_output)

        # Return None for kv_cache since it's managed by gpu_paged_kv_manager
        return attn_output, None, None

    def _forward_decode_legacy(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Legacy decode forward using tuple-based KV cache.

        Used as fallback when gpu_paged_kv_manager is not available.
        """
        batch, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "Decode expects single token"

        # Project Q, K, V
        query = self.module.q_proj(hidden_states)
        key = self.module.k_proj(hidden_states)
        value = self.module.v_proj(hidden_states)

        # Reshape
        query = query.view(batch, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Get RoPE for current position
        cache_len = past_key_value[0].shape[1] if past_key_value is not None else 0
        total_len = cache_len + seq_len
        cos, sin = self.module.rotary_emb(value.transpose(1, 2), seq_len=total_len)

        # Apply RoPE at current position
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        else:
            cos = cos[cache_len:total_len]
            sin = sin[cache_len:total_len]

        query, key = self._apply_rotary(query, key, cos, sin)

        # Update KV cache
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        new_kv_cache = (key, value) if use_cache else None

        # Transpose for attention
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)

        # Compute attention
        attn_output = self._compute_attention(query_t, key_t, value_t, is_decode=True)

        # Output projection
        attn_output = self.module.o_proj(attn_output)

        return attn_output, None, new_kv_cache

    def _apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings.

        Args:
            query: [batch, seq, num_heads, head_dim]
            key: [batch, seq, num_kv_heads, head_dim]
            cos: [seq, head_dim]
            sin: [seq, head_dim]

        Returns:
            Tuple of rotated (query, key)
        """
        half_dim = self.head_dim // 2

        # Expand cos/sin for broadcasting: [seq, head_dim] -> [1, seq, 1, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Split heads
        q1, q2 = query[..., :half_dim], query[..., half_dim:]
        k1, k2 = key[..., :half_dim], key[..., half_dim:]

        cos_half = cos[..., :half_dim]
        sin_half = sin[..., :half_dim]

        # Apply rotation
        q_rot = torch.cat([
            q1 * cos_half - q2 * sin_half,
            q2 * cos_half + q1 * sin_half
        ], dim=-1)

        k_rot = torch.cat([
            k1 * cos_half - k2 * sin_half,
            k2 * cos_half + k1 * sin_half
        ], dim=-1)

        return q_rot, k_rot

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with weight loading and GQA attention with sinks.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional KV cache
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (output, attn_weights, kv_cache)
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"GPT-OSS attn forward. Phase: {self.phase}, "
            f"sliding={self.is_sliding}"
        )

        # Load weights (includes sinks)
        if not self.persistent:
            weights = self.load_weights(self.module_key)
            dequant_weights = self.dequantize_weights(weights)
            self.apply_weights(dequant_weights)

        # Move sinks to correct device
        if self.sinks is not None and hidden_states is not None:
            self.sinks = self.sinks.to(hidden_states.device)

        # Route to phase handler
        if self.phase == "prefill":
            result = self._forward_prefill(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:
            result = self._forward_decode(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"GPT-OSS attn forward complete. Phase: {self.phase}"
        )

        return result
