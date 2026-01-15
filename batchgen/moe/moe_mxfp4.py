# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
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

"""MXFP4 MoE (Mixture of Experts) forward pass.

This module provides the complete MoE forward function for GPT-OSS-120B,
combining:
- Top-k routing with softmax normalization
- Token dispatch (gather by expert)
- Fused MXFP4 dequant + GEMM
- Token combine (scatter + weighted sum)
- SwiGLU activation

Architecture (GPT-OSS-120B):
- 128 experts, top-4 routing
- hidden_size=2880, intermediate_size=2880 (with SwiGLU)
- MXFP4 (4-bit) quantized expert weights
- W1: gate_up_proj [128, 5760, 2880] -> [128, 5760, 1440] packed
- W2: down_proj [128, 2880, 2880] -> [128, 2880, 1440] packed

Usage:
    output = moe_mxfp4_forward(
        x,
        gate_weight, gate_bias,
        w1_packed, w1_scales, w1_bias,
        w2_packed, w2_scales, w2_bias,
        experts_per_token=4,
        swiglu_limit=7.0,
    )
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .token_dispatch import (
    LocalTokenDispatcher,
    DistributedTokenDispatcher,
    TokenDispatcher,
    DispatchResult,
    create_token_dispatcher,
)
from .routing import moe_routing
from .fused_mxfp4_gemm import fused_mxfp4_moe_gemm, fused_mxfp4_moe_gemm_sequential


def swiglu(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
    """SwiGLU activation with optional clamping.

    SwiGLU splits input in half: gate, up = x.chunk(2)
    output = silu(gate) * up

    Args:
        x: Input tensor. Shape: [..., 2 * intermediate_size]
        limit: Clamping value for silu activation (default: 7.0).
               Applied as clamp(-limit, limit) before silu.

    Returns:
        Output tensor. Shape: [..., intermediate_size]
    """
    gate, up = x.chunk(2, dim=-1)

    # Apply clamping to gate before silu
    if limit > 0:
        gate = gate.clamp(-limit, limit)

    return F.silu(gate) * up


def gather_bias(
    bias: torch.Tensor,
    dispatch_result: DispatchResult,
) -> torch.Tensor:
    """Gather bias values for dispatched tokens.

    Args:
        bias: Bias tensor [num_experts, N]
        dispatch_result: Dispatch result containing expert_ids

    Returns:
        Gathered bias [total_routed_tokens, N]
    """
    # Each token gets bias from its assigned expert
    return bias[dispatch_result.expert_ids]


@torch.inference_mode()
def moe_mxfp4_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w1_bias: Optional[torch.Tensor],
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    w2_bias: Optional[torch.Tensor],
    experts_per_token: int = 4,
    swiglu_limit: float = 7.0,
    dispatcher: Optional[TokenDispatcher] = None,
) -> torch.Tensor:
    """Complete MoE forward pass with MXFP4 weights.

    Computes:
        1. Gate: logits = x @ gate_weight + gate_bias
        2. Route: select top-k experts per token with softmax weights
        3. Dispatch: gather tokens by expert assignment
        4. W1 GEMM: h = dispatched_x @ W1.T + bias
        5. SwiGLU: h = silu(h[:half]) * h[half:]
        6. W2 GEMM: out = h @ W2.T + bias
        7. Combine: scatter back and weighted sum

    Args:
        x: Input hidden states. Shape: [num_tokens, hidden_size]
        gate_weight: Router projection. Shape: [hidden_size, num_experts]
        gate_bias: Router bias. Shape: [num_experts] or None
        w1_packed: MXFP4 packed W1. Shape: [num_experts, intermediate*2, hidden//2]
        w1_scales: W1 scales. Shape: [num_experts, intermediate*2, hidden//32]
        w1_bias: W1 bias. Shape: [num_experts, intermediate*2] or None
        w2_packed: MXFP4 packed W2. Shape: [num_experts, hidden, intermediate//2]
        w2_scales: W2 scales. Shape: [num_experts, hidden, intermediate//32]
        w2_bias: W2 bias. Shape: [hidden] or None (shared across experts)
        experts_per_token: Number of experts per token (default: 4)
        swiglu_limit: Clamping value for SwiGLU (default: 7.0)
        dispatcher: Token dispatcher (default: LocalTokenDispatcher)

    Returns:
        Output. Shape: [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = x.shape
    num_experts = w1_packed.shape[0]

    if dispatcher is None:
        dispatcher = LocalTokenDispatcher()

    # 1. Routing: compute gate logits and select top-k experts
    topk_indices, topk_weights = moe_routing(
        x, gate_weight, gate_bias, experts_per_token
    )

    # 2. Dispatch: gather tokens by expert
    dispatch_result = dispatcher.dispatch(x, topk_indices, topk_weights)

    # 3. W1 GEMM: [total_routed, hidden] -> [total_routed, intermediate*2]
    h = fused_mxfp4_moe_gemm(
        dispatch_result.dispatched_x,
        w1_packed,
        w1_scales,
        dispatch_result.expert_counts,
        dispatch_result.expert_offsets,
    )

    # 4. Add W1 bias (per-expert)
    if w1_bias is not None:
        h = h + gather_bias(w1_bias, dispatch_result)

    # 5. SwiGLU activation
    h = swiglu(h, limit=swiglu_limit)

    # 6. W2 GEMM: [total_routed, intermediate] -> [total_routed, hidden]
    out = fused_mxfp4_moe_gemm(
        h,
        w2_packed,
        w2_scales,
        dispatch_result.expert_counts,
        dispatch_result.expert_offsets,
    )

    # 7. Add W2 bias (shared across experts)
    if w2_bias is not None:
        out = out + w2_bias

    # 8. Combine: scatter back with weighted sum
    output = dispatcher.combine(out, dispatch_result, num_tokens)

    return output


class MoEMXFP4Layer(torch.nn.Module):
    """MXFP4 MoE layer module.

    Encapsulates all MoE weights and provides a clean forward interface.
    Supports both inference with pre-quantized weights and training
    (though training requires dequantization).

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: FFN intermediate dimension.
        num_experts: Number of experts.
        experts_per_token: Experts selected per token.
        swiglu_limit: SwiGLU clamping value.
        bias: Whether to use biases.
    """

    def __init__(
        self,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        num_experts: int = 128,
        experts_per_token: int = 4,
        swiglu_limit: float = 7.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit

        # Router (gate) - full precision
        self.gate_weight = torch.nn.Parameter(
            torch.empty(hidden_size, num_experts, device=device)
        )
        if bias:
            self.gate_bias = torch.nn.Parameter(
                torch.empty(num_experts, device=device)
            )
        else:
            self.register_parameter('gate_bias', None)

        # Expert weights - MXFP4 packed (registered as buffers)
        # W1: gate_up_proj [num_experts, intermediate*2, hidden]
        # Packed: [num_experts, intermediate*2, hidden//2]
        self.register_buffer('w1_packed', torch.empty(
            num_experts, intermediate_size * 2, hidden_size // 2,
            dtype=torch.uint8, device=device
        ))
        self.register_buffer('w1_scales', torch.empty(
            num_experts, intermediate_size * 2, hidden_size // 32,
            dtype=torch.uint8, device=device
        ))

        # W2: down_proj [num_experts, hidden, intermediate]
        # Packed: [num_experts, hidden, intermediate//2]
        self.register_buffer('w2_packed', torch.empty(
            num_experts, hidden_size, intermediate_size // 2,
            dtype=torch.uint8, device=device
        ))
        self.register_buffer('w2_scales', torch.empty(
            num_experts, hidden_size, intermediate_size // 32,
            dtype=torch.uint8, device=device
        ))

        # Biases (full precision)
        if bias:
            self.w1_bias = torch.nn.Parameter(
                torch.empty(num_experts, intermediate_size * 2, device=device)
            )
            self.w2_bias = torch.nn.Parameter(
                torch.empty(hidden_size, device=device)
            )
        else:
            self.register_parameter('w1_bias', None)
            self.register_parameter('w2_bias', None)

        # Dispatcher (created lazily based on world_size)
        self._dispatcher: Optional[TokenDispatcher] = None

    def set_dispatcher(self, dispatcher: TokenDispatcher):
        """Set the token dispatcher for distributed execution."""
        self._dispatcher = dispatcher

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input. Shape: [batch, seq_len, hidden_size] or [num_tokens, hidden_size]

        Returns:
            Output. Same shape as input.
        """
        # Flatten if needed
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x = x.view(-1, self.hidden_size)
        else:
            batch = None

        # Create default dispatcher if not set
        if self._dispatcher is None:
            self._dispatcher = LocalTokenDispatcher()

        # MoE forward
        output = moe_mxfp4_forward(
            x,
            self.gate_weight,
            self.gate_bias,
            self.w1_packed,
            self.w1_scales,
            self.w1_bias,
            self.w2_packed,
            self.w2_scales,
            self.w2_bias,
            self.experts_per_token,
            self.swiglu_limit,
            self._dispatcher,
        )

        # Restore shape if needed
        if batch is not None:
            output = output.view(batch, seq_len, self.hidden_size)

        return output

    def load_quantized_weights(
        self,
        w1_packed: torch.Tensor,
        w1_scales: torch.Tensor,
        w2_packed: torch.Tensor,
        w2_scales: torch.Tensor,
        gate_weight: Optional[torch.Tensor] = None,
        gate_bias: Optional[torch.Tensor] = None,
        w1_bias: Optional[torch.Tensor] = None,
        w2_bias: Optional[torch.Tensor] = None,
    ):
        """Load pre-quantized weights.

        Args:
            w1_packed: MXFP4 packed W1 [num_experts, intermediate*2, hidden//2]
            w1_scales: W1 scales [num_experts, intermediate*2, hidden//32]
            w2_packed: MXFP4 packed W2 [num_experts, hidden, intermediate//2]
            w2_scales: W2 scales [num_experts, hidden, intermediate//32]
            gate_weight: Router weight [hidden, num_experts]
            gate_bias: Router bias [num_experts]
            w1_bias: W1 bias [num_experts, intermediate*2]
            w2_bias: W2 bias [hidden]
        """
        self.w1_packed.copy_(w1_packed)
        self.w1_scales.copy_(w1_scales)
        self.w2_packed.copy_(w2_packed)
        self.w2_scales.copy_(w2_scales)

        if gate_weight is not None:
            self.gate_weight.data.copy_(gate_weight)
        if gate_bias is not None and self.gate_bias is not None:
            self.gate_bias.data.copy_(gate_bias)
        if w1_bias is not None and self.w1_bias is not None:
            self.w1_bias.data.copy_(w1_bias)
        if w2_bias is not None and self.w2_bias is not None:
            self.w2_bias.data.copy_(w2_bias)


def moe_mxfp4_forward_reference(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w1_bias: Optional[torch.Tensor],
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    w2_bias: Optional[torch.Tensor],
    experts_per_token: int = 4,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """Reference MoE forward using sequential per-expert GEMM.

    Same interface as moe_mxfp4_forward but uses sequential GEMM
    for easier debugging and validation.
    """
    num_tokens, hidden_size = x.shape

    dispatcher = LocalTokenDispatcher()

    # Routing
    topk_indices, topk_weights = moe_routing(
        x, gate_weight, gate_bias, experts_per_token
    )

    # Dispatch
    dispatch_result = dispatcher.dispatch(x, topk_indices, topk_weights)

    # W1 GEMM (sequential)
    h = fused_mxfp4_moe_gemm_sequential(
        dispatch_result.dispatched_x,
        w1_packed,
        w1_scales,
        dispatch_result.expert_counts,
        dispatch_result.expert_offsets,
    )

    # W1 bias
    if w1_bias is not None:
        h = h + gather_bias(w1_bias, dispatch_result)

    # SwiGLU
    h = swiglu(h, limit=swiglu_limit)

    # W2 GEMM (sequential)
    out = fused_mxfp4_moe_gemm_sequential(
        h,
        w2_packed,
        w2_scales,
        dispatch_result.expert_counts,
        dispatch_result.expert_offsets,
    )

    # W2 bias
    if w2_bias is not None:
        out = out + w2_bias

    # Combine
    output = dispatcher.combine(out, dispatch_result, num_tokens)

    return output
