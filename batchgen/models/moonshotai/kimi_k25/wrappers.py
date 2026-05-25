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

"""Kimi K2.5-specific wrappers for BatchGen execution.

Provides wrappers for Kimi K2.5 model with INT4 W4A16 quantization:
- KimiK25ExpertWrapper: Expert wrapper with INT4 dequantization
- KimiK25AttnWrapper: Attention wrapper with BF16 MLA (no FP8)

Key differences from DeepSeek-V3:
- INT4 W4A16 weight-only quantization (vs FP8 W8A8)
- BF16 attention (no FP8 dequant scales needed)
- SiLU activation (standard, not OpenAI custom SwiGLU)
- 384 routed experts (vs 256)
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.wrappers.prefix_mla_model_adapters import (
    build_kimi_prefix_backend_context,
)
from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase


class KimiK25ExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with INT4 W4A16 dequantization for Kimi K2.5.

    INT4 W4A16 format:
    - Weights: INT4 packed (2 values per uint8, group_size=32, symmetric)
    - Activations: BF16 (untouched — no activation quantization)
    - Tensor names: .weight_packed (uint8), .weight_scale (bf16)

    Supports three modes:
    1. BF16 mode (pre-dequanted, EP with world_size >= 4): Direct BF16 GEMM
    2. INT4 mode (non-persistent): Dequant INT4→BF16, then BF16 GEMM
    3. INT4 persistent mode: INT4 weights cached on GPU, dequant per-forward

    Attributes:
        dequant_fn: INT4 dequantization function
        int4_gate_packed: Cached gate INT4 packed weights (persistent mode)
        int4_gate_scale: Cached gate INT4 scales (persistent mode)
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
    ):
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent=persistent
        )

        # Import INT4 dequantization
        try:
            from batchgen.quantization.int4 import int4_dequantize
            self.dequant_fn = int4_dequantize
        except ImportError:
            logging.warning("INT4 dequantization not available, using identity")
            self.dequant_fn = lambda packed, scales, dtype=torch.bfloat16: packed

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize INT4 packed weights to BF16.

        Expected weight format:
        - "gate_proj.weight_packed": uint8 [N, K//2]
        - "gate_proj.weight_scale": bf16 [N, K//32]
        - "up_proj.weight_packed", "up_proj.weight_scale"
        - "down_proj.weight_packed", "down_proj.weight_scale"

        Args:
            weights_dict: Dict with packed weights and scales

        Returns:
            Dict with dequantized BF16 weights (standard .weight keys)
        """
        result = {}

        for name, tensor in weights_dict.items():
            # Skip scale tensors — used with their paired weight
            if name.endswith("_scale"):
                continue

            # Check if this is a packed INT4 weight with a scale tensor
            if name.endswith("_packed"):
                scale_key = name.replace("_packed", "_scale")
                if scale_key in weights_dict:
                    # Dequantize INT4 → BF16
                    bf16_weight = self.dequant_fn(
                        tensor, weights_dict[scale_key], torch.bfloat16
                    )
                    # Map to standard weight key: gate_proj.weight_packed → gate_proj.weight
                    standard_key = name.replace("_packed", "")
                    result[standard_key] = bf16_weight
                else:
                    result[name] = tensor
            else:
                # Not quantized (e.g., bias) — use as-is
                result[name] = tensor

        return result

    def _get_stored_int4_weights(self) -> Dict[str, torch.Tensor]:
        """Get INT4 weights from the underlying expert module (persistent mode).

        Accesses weights through self.module to always get the current device
        tensors (GPU views after _move_int4_to_gpu_contiguous).

        If weights are in Marlin layout (from Marlin checkpoint), transforms
        to raw INT4 on-the-fly for WGMMA prefill. ~180 us per expert.
        """
        weights = {
            "gate_proj.weight_packed": self.module.int4_gate_packed,
            "gate_proj.weight_scale": self.module.int4_gate_scale,
            "up_proj.weight_packed": self.module.int4_up_packed,
            "up_proj.weight_scale": self.module.int4_up_scale,
            "down_proj.weight_packed": self.module.int4_down_packed,
            "down_proj.weight_scale": self.module.int4_down_scale,
        }
        # Marlin decode is default for K2.5 — transform Marlin→raw INT4 on-the-fly for WGMMA prefill
        self._transform_marlin_to_raw(weights)
        return weights

    _transform_logged = False

    def _transform_marlin_to_raw(self, weights: dict):
        """Transform Marlin-layout weights to raw INT4 on-the-fly for WGMMA prefill.

        Called for both persistent and non-persistent experts from Marlin checkpoint.
        ~180 us per expert (3 projections × 58 us), negligible vs compute.

        Core engine may reshape tensors to metadata shape [N, K//8] even though
        data is in Marlin layout [K//16, N*2]. Both have the same number of
        elements (N*K/8), so we reshape to Marlin dims before transforming.
        """
        from batchgen.moe.marlin_transform import marlin_to_wgmma_fused_gpu
        transformed = 0
        for key in list(weights.keys()):
            if not key.endswith("_packed") and not key.endswith("packed"):
                continue
            packed = weights[key]
            if not isinstance(packed, torch.Tensor) or packed.dim() != 2:
                continue
            scale_key = key.replace("_packed", "_scale").replace("packed", "scale")
            if scale_key not in weights:
                continue

            if packed.shape[0] < packed.shape[1]:
                # Already Marlin shape [K//16, N*2] (persistent experts)
                K_proj = packed.shape[0] * 16
                N_proj = packed.shape[1] // 2
            else:
                # Metadata shape [N, K//8] — reshape to Marlin [K//16, N*2]
                N_proj = packed.shape[0]
                K_proj = packed.shape[1] * 8
                marlin_rows = K_proj // 16
                marlin_cols = N_proj * 2
                packed = packed.reshape(marlin_rows, marlin_cols)
                # Also reshape scales: metadata [N, K//32] → Marlin [K//32, N]
                scale = weights[scale_key]
                scale = scale.reshape(K_proj // 32, N_proj)
                weights[scale_key] = scale

            raw_packed, raw_scale = marlin_to_wgmma_fused_gpu(
                packed, weights[scale_key], K_proj, N_proj)
            weights[key] = raw_packed
            weights[scale_key] = raw_scale
            transformed += 1
        if not self.__class__._transform_logged and transformed > 0:
            logging.info(f"[Marlin] Transformed {transformed} projections Marlin→raw "
                         f"(first shape: {list(weights.values())[0].shape})")
            self.__class__._transform_logged = True

    def _forward_bf16(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Direct BF16 GEMM path for pre-dequantized weights.

        Used when world_size >= 4 and weights have been pre-dequantized
        to BF16 during EP configuration (avoids per-forward dequant overhead).

        K2.5 uses standard SiLU gating: silu(gate) * up

        Args:
            hidden_states: Input [num_tokens, hidden_size] in BF16

        Returns:
            Output [num_tokens, hidden_size] in BF16
        """
        x = hidden_states
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Gate and Up projections (direct BF16 GEMM)
        gate_out = torch.mm(x, self.gate_weight_bf16.T)
        up_out = torch.mm(x, self.up_weight_bf16.T)

        # SiLU gating (standard DeepSeek-V3 / K2.5)
        intermediate = F.silu(gate_out) * up_out

        # Down projection
        output = torch.mm(intermediate, self.down_weight_bf16.T)

        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with INT4 dequant + BF16 GEMM.

        Routes to appropriate implementation:
        - BF16 mode: Pre-loaded BF16 weights (persistent shared/routed experts)
        - INT4 mode: Dequant + BF16 GEMM (non-persistent routed experts)

        Shared experts (expert_idx == -1) are BF16 and use the BF16 path.

        K2.5 activation: silu(gate) * up (standard, not OpenAI SwiGLU)

        Args:
            hidden_states: Input [num_tokens, hidden_size]

        Returns:
            Output [num_tokens, hidden_size]
        """
        # Fast path: pre-loaded/dequantized BF16 weights (persistent mode or shared experts)
        if getattr(self, 'use_bf16_weights', False):
            return self._forward_bf16(hidden_states)

        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"K2.5 expert forward. Phase: {self.phase}, persistent: {self.persistent}"
        )

        # Load weights from storage
        if self.persistent:
            weights = self._get_stored_int4_weights()
        else:
            weights = self.load_weights(self.module_key)
            # Marlin decode is default for K2.5 — transform Marlin→raw INT4 on-the-fly for WGMMA
            self._transform_marlin_to_raw(weights)

        # Ensure BF16 activations
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        # Shared experts (expert_idx == -1) have BF16 weights, not INT4
        if self.expert_idx == -1:
            # Load BF16 weights and assign to module
            self.module.gate_proj.weight.data = weights["gate_proj.weight"]
            self.module.up_proj.weight.data = weights["up_proj.weight"]
            self.module.down_proj.weight.data = weights["down_proj.weight"]
            result = self.module(hidden_states)
        else:
            # Routed experts: Dequant INT4 → BF16 and do GEMM
            result = self._forward_int4(hidden_states, weights)

        # Non-persistent experts use a shared GPU buffer that gets recycled.
        # Must sync before free_weights() to prevent the next expert's load_weights()
        # from overwriting the buffer while this expert's matmuls are still running.
        # Persistent experts use static module attributes — no buffer recycling, no sync needed.
        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"K2.5 expert forward complete. Phase: {self.phase}"
        )

        return result

    def _forward_int4(self, hidden_states: torch.Tensor, weights: dict) -> torch.Tensor:
        """INT4 forward using fused WGMMA kernel (preferred) or dequant+mm fallback."""
        x = hidden_states
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])

        # Try fused WGMMA kernel (stage1: gate+up+SiLU, stage2: down)
        try:
            from batchgen.moe.int4_single_expert_wgmma import single_expert_int4_forward
            return single_expert_int4_forward(
                x,
                weights["gate_proj.weight_packed"], weights["gate_proj.weight_scale"],
                weights["up_proj.weight_packed"], weights["up_proj.weight_scale"],
                weights["down_proj.weight_packed"], weights["down_proj.weight_scale"],
            )
        except Exception:
            pass

        # Fallback: dequant → BF16 matmul
        gate_weight = self.dequant_fn(
            weights["gate_proj.weight_packed"],
            weights["gate_proj.weight_scale"],
            torch.bfloat16,
        )
        up_weight = self.dequant_fn(
            weights["up_proj.weight_packed"],
            weights["up_proj.weight_scale"],
            torch.bfloat16,
        )
        down_weight = self.dequant_fn(
            weights["down_proj.weight_packed"],
            weights["down_proj.weight_scale"],
            torch.bfloat16,
        )

        gate_out = torch.mm(x, gate_weight.T)
        up_out = torch.mm(x, up_weight.T)
        intermediate = F.silu(gate_out) * up_out
        output = torch.mm(intermediate, down_weight.T)
        return output


class KimiK25AttnWrapper(AttnWrapperBase):
    """Attention wrapper with BF16 MLA for Kimi K2.5.

    K2.5 uses the same MLA (Multi-head Latent Attention) as DeepSeek-V3,
    but with BF16 weights (no FP8 quantization on attention).

    This is simpler than DeepSeek's wrapper:
    - No FP8 dequantization scales needed
    - No w8a16_gemm or deepgemm — standard BF16 GEMM
    - Weight loading follows the standard AttnWrapperBase flow

    The actual MLA computation is handled by the DeepseekV3Attention module
    (reused from the DeepSeek model code).
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
    ):
        # K2.5 attention is BF16 — no dequantization scales
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent=persistent, weight_dequant_scale=None,
        )

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """BF16 attention weights — no dequantization needed.

        Unlike DeepSeek-V3 which stores FP8 attention weights,
        K2.5 stores BF16 attention weights directly.

        Args:
            weights_dict: Dict of BF16 attention weights

        Returns:
            Same dict unchanged
        """
        return weights_dict

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward using BF16 MLA attention.

        Delegates to the DeepseekV3Attention module's standard forward.
        No FP8 handling needed — weights are BF16.

        Returns:
            Tuple of (attn_output, attn_weights, kv_cache)
        """
        if self.prepack_mode:
            # Prepacked mode: varlen flash attention
            hidden_states_2d = hidden_states.squeeze(0)
            metadata = self.prefix_cache_metadata()
            position_ids = self.position_ids.to(hidden_states_2d.device)
            prefix_context = None
            if metadata.prefix_reuse_mode:
                prefix_context = build_kimi_prefix_backend_context(
                    wrapper=self,
                    metadata=metadata,
                )

            attn_output, offload_kv = self.module.prefill_attn_bf16_prepacked(
                hidden_states_2d,
                position_ids,
                metadata.cu_seqlens.to(hidden_states_2d.device),
                metadata.max_seqlen,
                metadata.num_sequences,
                prefix_context=prefix_context,
            )
            # Offload KV cache per-sequence to host
            if offload_kv is None:
                raise RuntimeError("Kimi prepacked prefill returned no KV")
            self._offload_prepacked_kv(offload_kv)

            attn_output = attn_output.unsqueeze(0)
            return (attn_output, None, None)
        else:
            # Regular mode: standard attention with mask
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)

            attn_output, offload_kv = self.module.prefill_attn_bf16(
                hidden_states,
                attention_mask,
                position_ids,
            )
            return (attn_output, None, offload_kv)

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload KV cache per-sequence to host memory.

        For MLA, offload_kv contains [compressed_kv, k_pe] per token.

        Args:
            offload_kv: [total_tokens, kv_lora_rank + qk_rope_head_dim]
        """
        self.offload_prepacked_mla_kv(offload_kv)

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Decode forward using BF16 MLA attention.

        Routes to the appropriate decode backend based on KV cache type.
        No FP8 weight handling — uses BF16 directly.

        Args:
            hidden_states: Input tensor [batch, 1, hidden_size]

        Returns:
            Tuple of (attn_output, past_key_states, scale)
        """
        past_key_states = AttnWrapperBase.past_key_states
        attention_mask = AttnWrapperBase.attention_mask
        position_ids = AttnWrapperBase.position_ids
        scale = AttnWrapperBase.scale
        cache_seqlens = AttnWrapperBase.cache_seqlens
        max_seqlen = AttnWrapperBase.max_seqlen
        gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager

        if gpu_paged_kv_manager is not None:
            # BF16 paged KV path
            attn_output, k_tensor = self.module.decoding_attn_mode_3_bf16(
                hidden_states,
                position_ids,
                cache_seqlens,
                max_seqlen,
                None,  # No weight_dequant_scale needed (BF16 weights)
                gpu_paged_kv_manager,
                self.layer_idx,
                None,
            )

            import os as _os
            _skip_cb = _os.environ.get("BATCHGEN_SKIP_KV_CALLBACK", "0") == "1"
            if not _skip_cb and AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)

            return (attn_output, None, None)
        else:
            # Direct KV cache path
            layer_past_key = past_key_states[self.layer_idx] if past_key_states else None
            layer_scale = scale[self.layer_idx] if scale else None

            attn_output, updated_past_key, updated_scale = self.module.decoding_attn_bf16(
                hidden_states,
                layer_past_key,
                None,
                attention_mask,
                position_ids,
                layer_scale,
                cache_seqlens,
                max_seqlen,
            )

            return (attn_output, updated_past_key, updated_scale)
