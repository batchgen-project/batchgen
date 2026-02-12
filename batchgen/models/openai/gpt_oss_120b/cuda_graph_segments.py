"""CUDA Graph capturable segments for GPT-OSS-120B decode.

MINIMAL graph: only the packed QKV projection (single nn.Linear / GEMM).
Everything else runs eagerly for debugging.

  Graph segment:   packed QKV proj (nn.Linear)
  Eager:           RMSNorm → [GRAPH: QKV proj] → QKV split → RoPE →
                   KV write → FlashAttention → O_proj → residual+norm → MoE
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

from batchgen.cuda_graph.graph_manager import TensorSpec

logger = logging.getLogger(__name__)


class GptOssQkvProjSegment:
    """Capturable segment: packed QKV projection only (single GEMM).

    Input: normed hidden_states (after RMSNorm, done eagerly)
    Output: packed QKV tensor (split done eagerly)

    Args:
        attn_wrapper: The GptOssAttnWrapper for this layer.
        layer_idx: Decoder layer index (0-35).
    """

    def __init__(
        self,
        attn_wrapper,
        layer_idx: int,
    ):
        self.attn_module = attn_wrapper.module
        self.layer_idx = layer_idx

        self.hidden_size = 2880
        self.q_size = attn_wrapper.q_size                # 4096
        self.kv_size = attn_wrapper.kv_size              # 512
        self.total_qkv_size = self.q_size + 2 * self.kv_size  # 5120

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "normed": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "qkv": TensorSpec(
                ("batch_size", 1, self.total_qkv_size), torch.bfloat16
            ),
        }

    def get_weight_data_ptr(self) -> int:
        """Return the GPU data pointer of qkv_proj weight for verification."""
        return self.attn_module.qkv_proj.weight.data_ptr()

    def forward(
        self,
        normed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Packed QKV projection only.

        Args:
            normed: [bucket_size, 1, hidden_size] (already RMSNorm'd)

        Returns:
            qkv: [bucket_size, 1, total_qkv_size]
        """
        qkv = self.attn_module.qkv_proj(normed)
        return {"qkv": qkv}
