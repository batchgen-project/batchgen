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

"""DeepSeek-specific wrappers for BatchGen execution.

Provides wrappers for DeepSeek-R1 and DeepSeek-V3 models with FP8 quantization:
- DeepSeekExpertWrapper: Expert wrapper with FP8 dequantization
- DeepSeekAttnWrapper: Attention wrapper with FP8 dequantization
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization


class DeepSeekExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with FP8 dequantization for DeepSeek-R1/V3.

    Handles:
    - FP8 block-wise dequantization with scale factors
    - FP8 weight caching for local experts (no loading needed)
    - deepgemm kernel for forward computation

    Attributes:
        weight_dequant_scale: Dict of scale factors for dequantization
        fp8_gate: Cached FP8 gate_proj weights (when not loading)
        fp8_up: Cached FP8 up_proj weights
        fp8_down: Cached FP8 down_proj weights
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
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize DeepSeek expert wrapper.

        Args:
            module: Expert FFN module to wrap
            layer_idx: Layer index in the model
            expert_idx: Expert index (-1 for shared expert)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
            persistent: Whether weights are pre-loaded on GPU.
                        True = pre-loaded, no buffer fetch needed.
                        False = load from buffer each forward.
            weight_dequant_scale: Dict of FP8 scale factors
        """
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}

        # FP8 weight caching for local experts
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize FP8 weights using scale factors.

        Args:
            weights_dict: Dict mapping parameter names to FP8 weights

        Returns:
            Dict mapping parameter names to BF16 weights
        """
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = deepseek_v3_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def _register_fp8_weights(self):
        """Cache FP8 weights for local experts.

        Called when persistent=True - weights stay in GPU memory.
        """
        self.fp8_gate = self.module.gate_proj.weight.data
        self.fp8_up = self.module.up_proj.weight.data
        self.fp8_down = self.module.down_proj.weight.data

    def _unregister_fp8_weights(self):
        """Clear cached FP8 weights."""
        logging.debug(
            f"Clearing expert weights. Layer idx {self.layer_idx}, "
            f"expert idx {self.expert_idx}"
        )
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using deepgemm kernel.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        # Use deepgemm forward for both prefill and decode
        return self.module.deepgemm_forward(hidden_states, self.weight_dequant_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 handling.

        If persistent=True: Use cached FP8 weights directly (pre-loaded on GPU)
        If persistent=False: Load from core engine, dequantize, forward, cleanup

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Forward pass. Phase: {self.phase}"
        )

        if not self.persistent:
            # Load weights from core engine (non-persistent experts)
            weights = self.load_weights(self.module_key)

            # Apply weights to module (dequantization happens in deepgemm_forward)
            for name, param in self.module.named_parameters():
                param.data = weights[name]

        else:
            # Use cached FP8 weights (persistent experts)
            self.module.gate_proj.weight.data = self.fp8_gate
            self.module.up_proj.weight.data = self.fp8_up
            self.module.down_proj.weight.data = self.fp8_down

        # Micro-batch forward
        result = self.micro_batch_forward(hidden_states, "expert")

        # Cleanup for non-persistent experts
        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Finish forward pass. Phase: {self.phase}"
        )

        return result

    @torch.inference_mode()
    def stream_weights_into(self, gate_dst, up_dst, down_dst):
        """Method-A offload gather: stream this non-persistent expert's FP8
        weights from the host weight-buffer into the provided 3D-stacked slots,
        then release the buffer slot.

        Honors the GPU weight-buffer copy contract: load_weights/get_weights
        blocks until the producer has copied this expert, then copy -> sync the
        compute stream (so the slot->w3d copy lands before the producer reuses
        the slot) -> free_weights -> clear. Called for EVERY offloaded expert
        each decode step, including 0-token ones, so no slot leaks and the
        producer never stalls. Scales stay resident (weight_dequant_scale), so
        only the FP8 weights stream.
        """
        weights = self.load_weights(self.module_key)
        gate_dst.copy_(weights["gate_proj.weight"])
        up_dst.copy_(weights["up_proj.weight"])
        down_dst.copy_(weights["down_proj.weight"])
        torch.cuda.current_stream(
            self.engine_config.Basic_Config.device_torch
        ).synchronize()
        self.free_weights(self.module_key)
        self.clear_weights()


class DeepSeekAttnWrapper(AttnWrapperBase):
    """Attention wrapper with FP8 dequantization for DeepSeek-R1/V3.

    This is a placeholder - the full implementation requires porting
    the extensive prefill/decode logic from the original Wrapper.py.
    For now, it inherits the base behavior.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize DeepSeek attention wrapper."""
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )

        # FP8 weight caching for MLA architecture
        # DeepSeek V3 uses: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None

    def _register_fp8_weights(self):
        """Cache FP8 attention weights for MLA architecture."""
        self.fp8_q_a_proj = self.module.q_a_proj.weight.data
        self.fp8_q_b_proj = self.module.q_b_proj.weight.data
        self.fp8_kv_a_proj = self.module.kv_a_proj_with_mqa.weight.data
        self.fp8_kv_b_proj = self.module.kv_b_proj.weight.data
        self.fp8_o_proj = self.module.o_proj.weight.data

    def _unregister_fp8_weights(self):
        """Clear cached FP8 attention weights."""
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return FP8 weights unchanged - NO dequantization.

        For DeepSeek with FP8 DeepGEMM, weights stay in FP8:
        - w8a16_gemm quantizes activations to FP8 (not weights)
        - deep_gemm.fp8_gemm_nt() handles FP8 weights directly
        - Dequantization would break w8a16_gemm which expects FP8 weights
        """
        return weights_dict

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward using FP8 DeepGEMM.

        In prepack mode (cu_seqlens-based batching), calls prepacked attention
        which doesn't need attention_mask. In regular mode, calls standard
        attention which requires attention_mask for unpadding.

        Calls prefill_attn_w8a16 which uses w8a16_gemm:
        - Weights stay in FP8 format
        - Activations are quantized to FP8 via act_quant()
        - deep_gemm.fp8_gemm_nt() outputs BF16

        Returns:
            Tuple of (attn_output, attn_weights, kv_cache) to match decoder_layer's
            expected unpacking.
        """
        if self.prepack_mode:
            # Prepack mode: hidden_states is [1, total_tokens, hidden_dim]
            # Prepacked attention expects [total_tokens, hidden_dim]
            hidden_states_2d = hidden_states.squeeze(0)

            attn_output, offload_kv = self.module.prefill_attn_w8a16_prepacked(
                hidden_states_2d,
                self.position_ids.to(hidden_states_2d.device),
                self.prepack_cu_seqlens.to(hidden_states_2d.device),
                self.prepack_max_seqlen,
                self.prepack_num_sequences,
                self.weight_dequant_scale
            )

            # Offload KV cache per-sequence to host
            # offload_kv is [total_tokens, kv_lora_rank + qk_rope_head_dim]
            self._offload_prepacked_kv(offload_kv)

            # Reshape back to [1, total_tokens, hidden_dim] for decoder_layer
            attn_output = attn_output.unsqueeze(0)

            # Return 3-tuple to match decoder_layer's expected unpacking
            return (attn_output, None, None)
        else:
            # Regular mode: use attention_mask-based unpadding
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)

            attn_output, offload_kv = self.module.prefill_attn_w8a16(
                hidden_states,
                attention_mask,
                position_ids,
                self.weight_dequant_scale  # Inverse scales for w8a16_gemm
            )

            # Return 3-tuple for consistency
            return (attn_output, None, offload_kv)

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload KV cache per-sequence to host memory.

        For MLA, offload_kv contains [compressed_kv, k_pe] for each token.
        Split by cu_seqlens and offload each sequence.

        Args:
            offload_kv: [total_tokens, kv_lora_rank + qk_rope_head_dim]
        """
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            # Extract KV for this sequence
            seq_kv = offload_kv[start_idx:end_idx]  # [seq_len, kv_dim]

            # Reshape to [1, seq_len, 1, kv_dim] for KV cache API
            seq_kv = seq_kv.unsqueeze(0).unsqueeze(2)

            seq_global_id = [global_sequence_ids[seq_idx]]

            # MLA has no V (K contains compressed KV + k_pe)
            self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Decode forward using FlashMLA backend.

        FlashMLA decode functions handle FP8 weights correctly via w8a8_deepgemm,
        unlike vanilla DeepseekV3Attention.forward() which expects BF16 weights.

        Routes to appropriate decode backend based on KV cache management:
        - If gpu_paged_kv_manager is set: Use decoding_attn_mode_3_bf16 (paged KV)
        - Otherwise: Use decoding_attn_mode_3_fp8 (FP8 KV with tensor references)

        Args:
            hidden_states: Input tensor [batch, 1, hidden_size]
            **kwargs: Unused (all state from class-level attributes)

        Returns:
            Tuple of (attn_output, past_key_states, scale) for KV cache update.
        """
        # Get class-level decode state
        past_key_states = AttnWrapperBase.past_key_states
        attention_mask = AttnWrapperBase.attention_mask
        position_ids = AttnWrapperBase.position_ids  # q_position_ids for decode
        scale = AttnWrapperBase.scale
        cache_seqlens = AttnWrapperBase.cache_seqlens
        max_seqlen = AttnWrapperBase.max_seqlen
        gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager

        # Check if using paged KV (GPUPagedKVCacheManager)
        if gpu_paged_kv_manager is not None:
            # BF16 paged KV path
            attn_output, k_tensor = self.module.decoding_attn_mode_3_bf16(
                hidden_states,
                position_ids,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale,
                gpu_paged_kv_manager,
                self.layer_idx,
                None  # batch_slice (handled by gpu_paged_kv_manager)
            )

            # Offload k_tensor via KV append callback if available
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)

            # Return tuple (attn_output is all that's needed - KV managed externally)
            return (attn_output, None, None)
        else:
            # FP8 KV cache with tensor references
            # Get layer-specific KV cache slice
            layer_past_key = past_key_states[self.layer_idx] if past_key_states else None
            layer_scale = scale[self.layer_idx] if scale else None

            attn_output, updated_past_key, updated_scale = self.module.decoding_attn_mode_3_fp8(
                hidden_states,
                layer_past_key,
                None,  # past_value_states (None for MLA - K contains compressed KV)
                attention_mask,
                position_ids,
                layer_scale,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale  # FP8 weight scales for w8a8_deepgemm
            )

            return (attn_output, updated_past_key, updated_scale)
