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

Timing instrumentation:
    Set BATCHGEN_PREFILL_TIMING=1 environment variable to enable per-layer timing.
    Or call PrefillTimingStats.enable() programmatically.
"""

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase


def log_gpu_memory(msg: str = ""):
    """Log current GPU memory usage.

    Args:
        msg: Context message to include in the log line
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logging.info(
            f"[GPU Memory] {msg}: allocated={allocated:.2f}GB, "
            f"reserved={reserved:.2f}GB, max_allocated={max_allocated:.2f}GB"
        )


from batchgen.attention.gqa import gqa_attention_with_sinks, gqa_decode_fa
from batchgen.attention.sink import softmax_with_sinks


# Global timing accumulators for profiling
class PrefillTimingStats:
    """Timing statistics for prefill phase per module type."""

    # Class-level accumulators for per-layer timing
    attn_times_ms: Dict[int, float] = {}  # layer_idx -> total time in ms
    moe_times_ms: Dict[int, float] = {}   # layer_idx -> total time in ms
    attn_call_counts: Dict[int, int] = {}
    moe_call_counts: Dict[int, int] = {}

    # Granular MoE timing breakdown
    moe_load_ms: float = 0.0      # Weight loading from core_engine
    moe_dequant_ms: float = 0.0   # MXFP4 dequantization
    moe_apply_ms: float = 0.0     # Apply weights to module
    moe_forward_ms: float = 0.0   # Actual forward pass
    moe_cleanup_ms: float = 0.0   # Sync + free + clear

    # Enable/disable timing (add overhead)
    enabled: bool = False

    @classmethod
    def reset(cls):
        """Reset all timing stats."""
        cls.attn_times_ms = {}
        cls.moe_times_ms = {}
        cls.attn_call_counts = {}
        cls.moe_call_counts = {}
        # Reset granular MoE timing
        cls.moe_load_ms = 0.0
        cls.moe_dequant_ms = 0.0
        cls.moe_apply_ms = 0.0
        cls.moe_forward_ms = 0.0
        cls.moe_cleanup_ms = 0.0

    @classmethod
    def enable(cls):
        """Enable timing instrumentation."""
        cls.enabled = True
        cls.reset()

    @classmethod
    def disable(cls):
        """Disable timing instrumentation."""
        cls.enabled = False

    @classmethod
    def record_attn(cls, layer_idx: int, time_ms: float):
        """Record attention timing for a layer."""
        if layer_idx not in cls.attn_times_ms:
            cls.attn_times_ms[layer_idx] = 0.0
            cls.attn_call_counts[layer_idx] = 0
        cls.attn_times_ms[layer_idx] += time_ms
        cls.attn_call_counts[layer_idx] += 1

    @classmethod
    def record_moe(cls, layer_idx: int, time_ms: float):
        """Record MoE timing for a layer."""
        if layer_idx not in cls.moe_times_ms:
            cls.moe_times_ms[layer_idx] = 0.0
            cls.moe_call_counts[layer_idx] = 0
        cls.moe_times_ms[layer_idx] += time_ms
        cls.moe_call_counts[layer_idx] += 1

    @classmethod
    def log_summary(cls):
        """Log timing summary to logging.info."""
        if not cls.attn_times_ms and not cls.moe_times_ms:
            return

        total_attn_ms = sum(cls.attn_times_ms.values())
        total_moe_ms = sum(cls.moe_times_ms.values())

        logging.info("=" * 60)
        logging.info("GPT-OSS Prefill Timing Summary")
        logging.info("=" * 60)
        logging.info(f"Total Attention: {total_attn_ms:.2f} ms")
        logging.info(f"Total MoE:       {total_moe_ms:.2f} ms")

        # MoE breakdown
        if cls.moe_load_ms > 0 or cls.moe_forward_ms > 0:
            logging.info("-" * 60)
            logging.info("MoE Timing Breakdown:")
            logging.info(f"  Weight Load:    {cls.moe_load_ms:10.2f} ms ({cls.moe_load_ms/total_moe_ms*100:.1f}%)")
            logging.info(f"  Dequantize:     {cls.moe_dequant_ms:10.2f} ms ({cls.moe_dequant_ms/total_moe_ms*100:.1f}%)")
            logging.info(f"  Apply Weights:  {cls.moe_apply_ms:10.2f} ms ({cls.moe_apply_ms/total_moe_ms*100:.1f}%)")
            logging.info(f"  Forward:        {cls.moe_forward_ms:10.2f} ms ({cls.moe_forward_ms/total_moe_ms*100:.1f}%)")
            logging.info(f"  Cleanup:        {cls.moe_cleanup_ms:10.2f} ms ({cls.moe_cleanup_ms/total_moe_ms*100:.1f}%)")

        logging.info("-" * 60)

        # Per-layer breakdown
        all_layers = sorted(set(cls.attn_times_ms.keys()) | set(cls.moe_times_ms.keys()))
        for layer_idx in all_layers:
            attn_ms = cls.attn_times_ms.get(layer_idx, 0.0)
            moe_ms = cls.moe_times_ms.get(layer_idx, 0.0)
            attn_calls = cls.attn_call_counts.get(layer_idx, 0)
            moe_calls = cls.moe_call_counts.get(layer_idx, 0)
            logging.info(
                f"Layer {layer_idx:2d}: attn={attn_ms:7.2f}ms ({attn_calls} calls), "
                f"moe={moe_ms:7.2f}ms ({moe_calls} calls)"
            )
        logging.info("=" * 60)

    @classmethod
    def check_env_and_enable(cls):
        """Check environment variable and enable timing if set."""
        if os.environ.get("BATCHGEN_PREFILL_TIMING", "0") == "1":
            cls.enable()
            logging.info("PrefillTimingStats enabled via BATCHGEN_PREFILL_TIMING=1")


# Auto-enable timing if environment variable is set
PrefillTimingStats.check_env_and_enable()


class DecodeTimingStats:
    """Timing statistics for decode phase per module type.

    Enable via environment variable: BATCHGEN_DECODE_TIMING=1
    """

    # Class-level accumulators for per-layer timing
    attn_times_ms: Dict[int, float] = {}  # layer_idx -> total time in ms
    moe_times_ms: Dict[int, float] = {}   # layer_idx -> total time in ms
    attn_call_counts: Dict[int, int] = {}
    moe_call_counts: Dict[int, int] = {}

    # Granular attention timing breakdown
    attn_projection_ms: float = 0.0    # Q, K, V projection
    attn_rope_ms: float = 0.0          # RoPE application
    attn_kv_update_ms: float = 0.0     # KV cache update
    attn_forward_ms: float = 0.0       # Attention forward (FA or vanilla)
    attn_output_proj_ms: float = 0.0   # Output projection

    # Granular MoE timing breakdown
    moe_routing_ms: float = 0.0        # Router forward
    moe_load_ms: float = 0.0           # Weight loading from buffer
    moe_dequant_ms: float = 0.0        # MXFP4 dequantization
    moe_apply_ms: float = 0.0          # Apply weights to module
    moe_forward_ms: float = 0.0        # Expert forward pass
    moe_cleanup_ms: float = 0.0        # Sync + free + clear

    # Enable/disable timing (adds overhead)
    enabled: bool = False

    @classmethod
    def reset(cls):
        """Reset all timing stats."""
        cls.attn_times_ms = {}
        cls.moe_times_ms = {}
        cls.attn_call_counts = {}
        cls.moe_call_counts = {}
        # Reset granular attention timing
        cls.attn_projection_ms = 0.0
        cls.attn_rope_ms = 0.0
        cls.attn_kv_update_ms = 0.0
        cls.attn_forward_ms = 0.0
        cls.attn_output_proj_ms = 0.0
        # Reset granular MoE timing
        cls.moe_routing_ms = 0.0
        cls.moe_load_ms = 0.0
        cls.moe_dequant_ms = 0.0
        cls.moe_apply_ms = 0.0
        cls.moe_forward_ms = 0.0
        cls.moe_cleanup_ms = 0.0

    @classmethod
    def enable(cls):
        """Enable timing instrumentation."""
        cls.enabled = True
        cls.reset()

    @classmethod
    def disable(cls):
        """Disable timing instrumentation."""
        cls.enabled = False

    @classmethod
    def record_attn(cls, layer_idx: int, time_ms: float):
        """Record attention timing for a layer."""
        if layer_idx not in cls.attn_times_ms:
            cls.attn_times_ms[layer_idx] = 0.0
            cls.attn_call_counts[layer_idx] = 0
        cls.attn_times_ms[layer_idx] += time_ms
        cls.attn_call_counts[layer_idx] += 1

    @classmethod
    def record_moe(cls, layer_idx: int, time_ms: float):
        """Record MoE timing for a layer."""
        if layer_idx not in cls.moe_times_ms:
            cls.moe_times_ms[layer_idx] = 0.0
            cls.moe_call_counts[layer_idx] = 0
        cls.moe_times_ms[layer_idx] += time_ms
        cls.moe_call_counts[layer_idx] += 1

    @classmethod
    def log_summary(cls):
        """Log timing summary to logging.info."""
        if not cls.attn_times_ms and not cls.moe_times_ms:
            return

        total_attn_ms = sum(cls.attn_times_ms.values())
        total_moe_ms = sum(cls.moe_times_ms.values())

        logging.info("=" * 60)
        logging.info("GPT-OSS Decode Timing Summary")
        logging.info("=" * 60)
        logging.info(f"Total Attention: {total_attn_ms:.2f} ms")
        logging.info(f"Total MoE:       {total_moe_ms:.2f} ms")

        # Attention breakdown
        if cls.attn_projection_ms > 0 or cls.attn_forward_ms > 0:
            logging.info("-" * 60)
            logging.info("Attention Timing Breakdown:")
            if total_attn_ms > 0:
                logging.info(f"  Projection:     {cls.attn_projection_ms:10.2f} ms ({cls.attn_projection_ms/total_attn_ms*100:.1f}%)")
                logging.info(f"  RoPE:           {cls.attn_rope_ms:10.2f} ms ({cls.attn_rope_ms/total_attn_ms*100:.1f}%)")
                logging.info(f"  KV Update:      {cls.attn_kv_update_ms:10.2f} ms ({cls.attn_kv_update_ms/total_attn_ms*100:.1f}%)")
                logging.info(f"  Forward:        {cls.attn_forward_ms:10.2f} ms ({cls.attn_forward_ms/total_attn_ms*100:.1f}%)")
                logging.info(f"  Output Proj:    {cls.attn_output_proj_ms:10.2f} ms ({cls.attn_output_proj_ms/total_attn_ms*100:.1f}%)")

        # MoE breakdown
        if cls.moe_load_ms > 0 or cls.moe_forward_ms > 0:
            logging.info("-" * 60)
            logging.info("MoE Timing Breakdown:")
            if total_moe_ms > 0:
                logging.info(f"  Routing:        {cls.moe_routing_ms:10.2f} ms ({cls.moe_routing_ms/total_moe_ms*100:.1f}%)")
                logging.info(f"  Weight Load:    {cls.moe_load_ms:10.2f} ms ({cls.moe_load_ms/total_moe_ms*100:.1f}%)")
                logging.info(f"  Dequantize:     {cls.moe_dequant_ms:10.2f} ms ({cls.moe_dequant_ms/total_moe_ms*100:.1f}%)")
                logging.info(f"  Apply Weights:  {cls.moe_apply_ms:10.2f} ms ({cls.moe_apply_ms/total_moe_ms*100:.1f}%)")
                logging.info(f"  Forward:        {cls.moe_forward_ms:10.2f} ms ({cls.moe_forward_ms/total_moe_ms*100:.1f}%)")
                logging.info(f"  Cleanup:        {cls.moe_cleanup_ms:10.2f} ms ({cls.moe_cleanup_ms/total_moe_ms*100:.1f}%)")

        logging.info("-" * 60)

        # Per-layer breakdown
        all_layers = sorted(set(cls.attn_times_ms.keys()) | set(cls.moe_times_ms.keys()))
        for layer_idx in all_layers:
            attn_ms = cls.attn_times_ms.get(layer_idx, 0.0)
            moe_ms = cls.moe_times_ms.get(layer_idx, 0.0)
            attn_calls = cls.attn_call_counts.get(layer_idx, 0)
            moe_calls = cls.moe_call_counts.get(layer_idx, 0)
            logging.info(
                f"Layer {layer_idx:2d}: attn={attn_ms:7.2f}ms ({attn_calls} calls), "
                f"moe={moe_ms:7.2f}ms ({moe_calls} calls)"
            )
        logging.info("=" * 60)

    @classmethod
    def check_env_and_enable(cls):
        """Check environment variable and enable timing if set."""
        if os.environ.get("BATCHGEN_DECODE_TIMING", "0") == "1":
            cls.enable()
            logging.info("DecodeTimingStats enabled via BATCHGEN_DECODE_TIMING=1")


# Auto-enable decode timing if environment variable is set
DecodeTimingStats.check_env_and_enable()


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
        persistent: bool = False,
    ):
        """Initialize GPT-OSS expert wrapper.

        Args:
            module: Expert FFN module (SwiGLU)
            layer_idx: Layer index in the model (0-35)
            expert_idx: Expert index (0-127)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
            persistent: If True, MXFP4 weights are stored on GPU and reused.
                       If False, weights are loaded from buffer each forward.
        """
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent=persistent
        )

        # Storage for MXFP4 weights in persistent mode
        self.stored_mxfp4_weights = None

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

    def _store_mxfp4_weights(self, weights_dict: Dict[str, torch.Tensor]):
        """Store MXFP4 weights on GPU for persistent mode.

        Avoids tensor duplication by checking if tensor is already on target device.

        Args:
            weights_dict: Dict containing packed weights and scales
        """
        device = self.engine_config.Basic_Config.device_torch
        result = {}
        for name, tensor in weights_dict.items():
            if tensor.device == device:
                # Already on target device - use as-is (no copy)
                result[name] = tensor
            else:
                # Move to device (creates copy)
                result[name] = tensor.to(device, non_blocking=True)
        self.stored_mxfp4_weights = result

    def _clear_stored_mxfp4_weights(self):
        """Clear stored MXFP4 weights to free GPU memory.

        Call this before reconfiguring to prevent OOM during phase transitions.
        """
        if self.stored_mxfp4_weights is not None:
            logging.debug(
                f"[Layer {self.layer_idx} Expert {self.expert_idx}] "
                f"Clearing stored MXFP4 weights"
            )
            self.stored_mxfp4_weights = None

    def _get_stored_mxfp4_weights(self) -> Dict[str, torch.Tensor]:
        """Get stored MXFP4 weights for persistent mode.

        Returns:
            Dict containing packed weights and scales (already on GPU)
        """
        return self.stored_mxfp4_weights

    def _pre_store_mxfp4_weights(self):
        """Load MXFP4 weights from core_engine and store on GPU.

        Called during initialization when persistent=True.
        Weights remain in MXFP4 format; dequantization happens during forward.
        """
        weights = self.core_engine.get_tensor(self.module_key)
        self._store_mxfp4_weights(weights)

        logging.debug(
            f"[Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Pre-stored MXFP4 weights for persistent mode"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with MXFP4 dequantization.

        Flow:
        1. Get MXFP4 weights from storage (persistent) or buffer (non-persistent)
        2. Dequantize to BF16
        3. Apply to module
        4. Micro-batch forward through SwiGLU
        5. Cleanup (free buffer only if non-persistent, always clear BF16)

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        # Start timing if enabled (prefill or decode)
        do_prefill_timing = PrefillTimingStats.enabled and self.phase == "prefill"
        do_decode_timing = DecodeTimingStats.enabled and self.phase == "decode"
        do_timing = do_prefill_timing or do_decode_timing
        timing_stats = PrefillTimingStats if do_prefill_timing else DecodeTimingStats

        if do_timing:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"GPT-OSS expert forward. Phase: {self.phase}, persistent: {self.persistent}"
        )

        # Get MXFP4 weights from appropriate source
        if do_timing:
            t0 = time.perf_counter()
        if self.persistent:
            # Persistent mode: read from stored GPU attributes
            weights = self._get_stored_mxfp4_weights()
        else:
            # Non-persistent mode: load from buffer
            weights = self.load_weights(self.module_key)
        if do_timing:
            torch.cuda.synchronize()
            timing_stats.moe_load_ms += (time.perf_counter() - t0) * 1000

        # Log dtype info for layers 0 and 1 (for debugging MXFP4 handling)
        if self.layer_idx < 2:
            logging.debug(f"[Layer {self.layer_idx} Expert {self.expert_idx}] Module: {self.module_key}")
            for name, tensor in weights.items():
                logging.debug(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

        # Dequantize MXFP4 to BF16 (always needed)
        if do_timing:
            t0 = time.perf_counter()
        dequant_weights = self.dequantize_weights(weights)
        if do_timing:
            torch.cuda.synchronize()
            timing_stats.moe_dequant_ms += (time.perf_counter() - t0) * 1000

        # Apply BF16 to module
        if do_timing:
            t0 = time.perf_counter()
        self.apply_weights(dequant_weights)
        if do_timing:
            torch.cuda.synchronize()
            timing_stats.moe_apply_ms += (time.perf_counter() - t0) * 1000

        # Micro-batch forward
        if do_timing:
            t0 = time.perf_counter()
        result = self.micro_batch_forward(hidden_states, "expert")
        if do_timing:
            torch.cuda.synchronize()
            timing_stats.moe_forward_ms += (time.perf_counter() - t0) * 1000

        # Cleanup
        if do_timing:
            t0 = time.perf_counter()
        torch.cuda.current_stream(
            self.engine_config.Basic_Config.device_torch
        ).synchronize()
        if not self.persistent:
            # Non-persistent: release buffer slot
            self.free_weights(self.module_key)
        # Always clear BF16 from module (temporary allocation)
        self.clear_weights()
        if do_timing:
            timing_stats.moe_cleanup_ms += (time.perf_counter() - t0) * 1000

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"GPT-OSS expert forward complete. Phase: {self.phase}"
        )

        # Record total timing if enabled
        if do_timing:
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            timing_stats.record_moe(self.layer_idx, elapsed_ms)

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
        persistent: bool = True,  # Default persistent; caller overrides based on weight_copy_task
    ):
        """Initialize GPT-OSS attention wrapper.

        Args:
            module: Attention module (GQA)
            layer_idx: Layer index (0-35)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
            persistent: If True, weights are pre-loaded on GPU (no buffer fetch).
                       If False, load from buffer each forward.
        """
        # GPT-OSS attention is BF16, no dequantization needed
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent=persistent, weight_dequant_scale=None
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
            # Log sink info at INFO level when BATCHGEN_DEBUG_SINKS=1
            if os.environ.get("BATCHGEN_DEBUG_SINKS", "0") == "1":
                sink_min = float(self.sinks.min().item())
                sink_max = float(self.sinks.max().item())
                sink_mean = float(self.sinks.float().mean().item())
                logging.info(
                    f"Layer {self.layer_idx}: Loaded sinks shape={self.sinks.shape}, "
                    f"range=[{sink_min:.4f}, {sink_max:.4f}], mean={sink_mean:.4f}"
                )
            else:
                logging.debug(f"Layer {self.layer_idx}: Loaded sinks with shape {self.sinks.shape}")
            # Remove from dict so it's not applied as a regular parameter
            result = {k: v for k, v in weights_dict.items() if k != "sinks"}
            return result

        # If no sinks in weights, initialize with zeros
        if self.sinks is None:
            self.sinks = torch.zeros(self.num_heads, dtype=torch.bfloat16)
            logging.warning(f"Layer {self.layer_idx}: No sinks found in weights, initialized to zeros (may cause issues)")

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
        # Decode timing instrumentation
        do_timing = DecodeTimingStats.enabled
        if do_timing:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

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

        # === Stage 1: Project Q, K, V for the new token ===
        if do_timing:
            t0 = time.perf_counter()

        query = self.module.q_proj(hidden_states)
        key = self.module.k_proj(hidden_states)
        value = self.module.v_proj(hidden_states)

        # Reshape: [batch, 1, num_heads, head_dim]
        query = query.view(batch, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        if do_timing:
            torch.cuda.synchronize()
            DecodeTimingStats.attn_projection_ms += (time.perf_counter() - t0) * 1000

        # === Stage 2: Get and apply RoPE ===
        if do_timing:
            t0 = time.perf_counter()

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

        if do_timing:
            torch.cuda.synchronize()
            DecodeTimingStats.attn_rope_ms += (time.perf_counter() - t0) * 1000

        # === Stage 3: Write new K, V to paged GPU cache ===
        if do_timing:
            t0 = time.perf_counter()

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

        if do_timing:
            torch.cuda.synchronize()
            DecodeTimingStats.attn_kv_update_ms += (time.perf_counter() - t0) * 1000

        # === Stage 4: FlashAttention with paged KV cache ===
        if do_timing:
            t0 = time.perf_counter()

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

        if do_timing:
            torch.cuda.synchronize()
            DecodeTimingStats.attn_forward_ms += (time.perf_counter() - t0) * 1000

        # === Stage 5: Output projection ===
        if do_timing:
            t0 = time.perf_counter()

        # Reshape output: [batch, 1, num_heads, head_dim] -> [batch, 1, hidden_size]
        attn_output = attn_output.view(batch, 1, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.module.o_proj(attn_output)

        if do_timing:
            torch.cuda.synchronize()
            DecodeTimingStats.attn_output_proj_ms += (time.perf_counter() - t0) * 1000

        # Record total attention timing
        if do_timing:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            DecodeTimingStats.record_attn(self.layer_idx, elapsed_ms)

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

    def _forward_prefill_prepacked(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Prepacked prefill forward using varlen flash attention.

        In prepack mode, hidden_states is [1, total_tokens, hidden_dim] (3D with batch=1),
        and we use cu_seqlens to track sequence boundaries for attention.

        Args:
            hidden_states: Input tensor [1, total_tokens, hidden_size] or [total_tokens, hidden_size]
            position_ids: Position IDs [total_tokens]

        Returns:
            Tuple of (output, None, None) - KV cache offloaded to host
        """
        # Import here to avoid circular imports
        from batchgen.attention.gqa import gqa_prefill_fa

        # Handle both 2D and 3D input
        if hidden_states.dim() == 3:
            assert hidden_states.shape[0] == 1, "Prepack mode expects batch_size=1"
            hidden_states_2d = hidden_states.squeeze(0)  # [total_tokens, hidden_dim]
            input_was_3d = True
        else:
            hidden_states_2d = hidden_states
            input_was_3d = False

        total_tokens = hidden_states_2d.shape[0]

        # Get prepack metadata from class variables
        cu_seqlens = AttnWrapperBase.prepack_cu_seqlens
        max_seqlen = AttnWrapperBase.prepack_max_seqlen
        num_sequences = AttnWrapperBase.prepack_num_sequences
        seq_lengths = AttnWrapperBase.prepack_seq_lengths

        # Project Q, K, V in varlen format
        # hidden_states_2d: [total_tokens, hidden_size]
        query = self.module.q_proj(hidden_states_2d)  # [total_tokens, num_heads * head_dim]
        key = self.module.k_proj(hidden_states_2d)    # [total_tokens, num_kv_heads * head_dim]
        value = self.module.v_proj(hidden_states_2d)  # [total_tokens, num_kv_heads * head_dim]

        # Reshape to [total_tokens, num_heads, head_dim]
        query = query.view(total_tokens, self.num_heads, self.head_dim)
        key = key.view(total_tokens, self.num_kv_heads, self.head_dim)
        value = value.view(total_tokens, self.num_kv_heads, self.head_dim)

        # Apply RoPE per sequence using position_ids
        if position_ids is not None:
            cos, sin = self.module.rotary_emb(value, seq_len=max_seqlen)
            # Get cos/sin for each position
            cos = cos[position_ids]  # [total_tokens, head_dim]
            sin = sin[position_ids]  # [total_tokens, head_dim]

            # Apply RoPE - split heads for rotation
            half_dim = self.head_dim // 2
            q1, q2 = query[..., :half_dim], query[..., half_dim:]
            k1, k2 = key[..., :half_dim], key[..., half_dim:]

            cos_half = cos[..., :half_dim].unsqueeze(1)  # [total_tokens, 1, half_dim]
            sin_half = sin[..., :half_dim].unsqueeze(1)

            query = torch.cat([
                q1 * cos_half - q2 * sin_half,
                q2 * cos_half + q1 * sin_half
            ], dim=-1)

            key = torch.cat([
                k1 * cos_half - k2 * sin_half,
                k2 * cos_half + k1 * sin_half
            ], dim=-1)

        # Use gqa_prefill_fa for varlen attention with sink correction
        # q, k, v: [total_tokens, num_heads, head_dim]
        attn_output, lse = gqa_prefill_fa(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens.to(hidden_states_2d.device),
            cu_seqlens_k=cu_seqlens.to(hidden_states_2d.device),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            sinks=self.sinks,
            softmax_scale=self.scale,
            sliding_window=self.sliding_window,
        )

        # attn_output: [total_tokens, num_heads, head_dim]
        # Reshape for output projection
        attn_output = attn_output.view(total_tokens, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.module.o_proj(attn_output)  # [total_tokens, hidden_size]

        # Offload KV cache per sequence to host
        global_sequence_ids = AttnWrapperBase.cur_batch

        torch.cuda.current_stream().synchronize()  # Make sure KV is ready

        # For GQA, we store both K and V (unlike MLA which only stores K)
        # Split by cu_seqlens and offload each sequence
        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            # Extract KV for this sequence
            seq_key = key[start_idx:end_idx]    # [seq_len, num_kv_heads, head_dim]
            seq_value = value[start_idx:end_idx]  # [seq_len, num_kv_heads, head_dim]

            # Reshape to [1, seq_len, num_kv_heads, head_dim] for KV cache API
            seq_key = seq_key.unsqueeze(0)
            seq_value = seq_value.unsqueeze(0)

            seq_global_id = [global_sequence_ids[seq_idx]]

            self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_key,
                v_tensor=seq_value,
                sequence_lengths=[seq_len],
            )

        logging.debug(
            f"[Layer {self.layer_idx}] GPT-OSS prepacked prefill complete. "
            f"total_tokens={total_tokens}, num_sequences={num_sequences}"
        )

        # Reshape output back to 3D if input was 3D
        if input_was_3d:
            attn_output = attn_output.unsqueeze(0)  # [1, total_tokens, hidden_size]

        return attn_output, None, None

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
        # Start timing if enabled
        if PrefillTimingStats.enabled and self.phase == "prefill":
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"GPT-OSS attn forward. Phase: {self.phase}, "
            f"sliding={self.is_sliding}"
        )

        # Load weights (includes sinks)
        if not self.persistent:
            weights = self.load_weights(self.module_key)

            # Log dtype info for layers 0 and 1 (for debugging)
            if self.layer_idx < 2:
                logging.debug(f"[Layer {self.layer_idx}] Module: {self.module_key}")
                for name, tensor in weights.items():
                    logging.debug(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

            dequant_weights = self.dequantize_weights(weights)
            self.apply_weights(dequant_weights)

        # Move sinks to correct device
        if self.sinks is not None and hidden_states is not None:
            self.sinks = self.sinks.to(hidden_states.device)

        # Route to phase handler
        # Check for prepack mode first (takes precedence during prefill)
        if self.phase == "prefill" and AttnWrapperBase.prepack_mode:
            result = self._forward_prefill_prepacked(
                hidden_states,
                position_ids=AttnWrapperBase.position_ids,
            )
        elif self.phase == "prefill":
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

        # Cleanup: release weight buffer after forward pass
        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"GPT-OSS attn forward complete. Phase: {self.phase}"
        )

        # Record timing if enabled
        if PrefillTimingStats.enabled and self.phase == "prefill":
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            PrefillTimingStats.record_attn(self.layer_idx, elapsed_ms)

        return result
