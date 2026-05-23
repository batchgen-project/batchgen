"""CUDA-graph capturable GLM-5 MoE decode segments.

The GLM-5 MoE graph captures the full decode MoE module boundary:

    padded local tokens -> all_gather -> router -> rank padding mask
      -> dispatch_scatter_3d -> FP8 blockwise S1/S3 -> reduce_weighted_scatter
      -> all_reduce -> local slice + shared expert

The decoder-layer residual add remains eager in the caller because it is owned
by ``Glm5DecoderLayer.forward()``, not by ``Glm5MoE.forward()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.moe.dispatch_scatter_3d import dispatch_scatter_3d, reduce_weighted_scatter
from batchgen.moe.grouped_fp8_blockwise_moe import (
    grouped_fp8_blockwise_fused_s1,
    grouped_fp8_blockwise_s3,
)
from batchgen.moe.routing import gate_sigmoid_topk_cuda, glm5_router_gemm_cuda

logger = logging.getLogger(__name__)


def _act_quant_3d(x: torch.Tensor, seqlens: torch.Tensor):
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d

    return act_quant_3d(x, seqlens)


def make_glm5_moe_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{layer_idx}_moe"


@dataclass
class _Glm5MoEGraphBuffers:
    padded: torch.Tensor
    all_tokens: torch.Tensor
    router_logits: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    topk_masked_indices: torch.Tensor
    topk_masked_weights: torch.Tensor
    topk_negative_ones: torch.Tensor
    topk_zero_weights: torch.Tensor
    rank_ids: torch.Tensor
    local_pos: torch.Tensor
    expert_counts: torch.Tensor
    expert_counters: torch.Tensor
    topk_pos: torch.Tensor
    dispatched_x: torch.Tensor
    intermediate: torch.Tensor
    expert_out: torch.Tensor
    routed_global_output: torch.Tensor
    local_moe_output: torch.Tensor
    cu_seqlens: torch.Tensor
    max_tokens_padded: int


class Glm5MoEGraphBufferPool:
    """Shared static buffers for all GLM-5 MoE graph segments on one rank."""

    _MTP_BLOCK = 128

    def __init__(
        self,
        *,
        world_size: int,
        hidden_size: int,
        num_experts_per_tok: int,
        num_local_experts: int,
        intermediate_size: int,
        device: torch.device,
        bucket_sizes: List[int],
        base_mtp: int,
    ) -> None:
        if not bucket_sizes:
            raise ValueError("GLM-5 MoE graph requires at least one bucket size")
        self.world_size = int(world_size)
        self.hidden_size = int(hidden_size)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_local_experts = int(num_local_experts)
        self.intermediate_size = int(intermediate_size)
        self.device = device
        self.bucket_sizes = sorted({int(b) for b in bucket_sizes})
        self.base_mtp = int(base_mtp)
        self._base: Dict[str, torch.Tensor] = {}
        self._views: Dict[int, _Glm5MoEGraphBuffers] = {}

    def setup(self) -> None:
        if self._base:
            return

        max_bucket = max(self.bucket_sizes)
        max_global = self.world_size * max_bucket
        mtp = max(self.base_mtp, self._round_up(max_global, self._MTP_BLOCK))
        rows = self.num_local_experts * mtp
        nk = max_global * self.num_experts_per_tok
        d = self.device
        h = self.hidden_size
        n = self.intermediate_size
        k = self.num_experts_per_tok

        b = self._base
        b["padded"] = torch.zeros(max_bucket, h, dtype=torch.bfloat16, device=d)
        b["all_tokens"] = torch.zeros(max_global, h, dtype=torch.bfloat16, device=d)
        b["router_logits"] = torch.empty(max_global, 256, dtype=torch.float32, device=d)
        b["topk_indices"] = torch.empty(max_global, k, dtype=torch.int32, device=d)
        b["topk_weights"] = torch.empty(max_global, k, dtype=torch.float32, device=d)
        b["topk_masked_indices"] = torch.empty(max_global, k, dtype=torch.int32, device=d)
        b["topk_masked_weights"] = torch.empty(max_global, k, dtype=torch.float32, device=d)
        b["topk_negative_ones"] = torch.full((max_global, k), -1, dtype=torch.int32, device=d)
        b["topk_zero_weights"] = torch.zeros(max_global, k, dtype=torch.float32, device=d)
        b["expert_counts"] = torch.zeros(self.num_local_experts, dtype=torch.int32, device=d)
        b["expert_counters"] = torch.zeros(self.num_local_experts, dtype=torch.int32, device=d)
        b["topk_pos"] = torch.full((nk,), -1, dtype=torch.int32, device=d)
        b["dispatched_x"] = torch.zeros(rows, h, dtype=torch.bfloat16, device=d)
        b["intermediate"] = torch.empty(rows, n, dtype=torch.bfloat16, device=d)
        b["expert_out"] = torch.empty(rows, h, dtype=torch.bfloat16, device=d)
        b["routed_global_output"] = torch.empty(max_global, h, dtype=torch.bfloat16, device=d)
        b["local_moe_output"] = torch.empty(max_bucket, h, dtype=torch.bfloat16, device=d)
        b["cu_seqlens"] = torch.arange(
            0,
            (self.num_local_experts + 1) * mtp,
            mtp,
            dtype=torch.int32,
            device=d,
        )

        total_bytes = sum(t.nelement() * t.element_size() for t in b.values())
        logger.info(
            "Glm5MoEGraphBufferPool: allocated %.2f GiB "
            "(max_bucket=%d, world_size=%d, mtp=%d, rows=%d)",
            total_bytes / (1024**3),
            max_bucket,
            self.world_size,
            mtp,
            rows,
        )

        for bucket_size in self.bucket_sizes:
            self._create_view(bucket_size, mtp)

    def get(self, bucket_size: int) -> _Glm5MoEGraphBuffers:
        self.setup()
        return self._views[int(bucket_size)]

    def _create_view(self, bucket_size: int, mtp: int) -> None:
        global_rows = self.world_size * bucket_size
        nk = global_rows * self.num_experts_per_tok
        rows = self.num_local_experts * mtp
        b = self._base
        positions = torch.arange(global_rows, dtype=torch.int64, device=self.device)
        self._views[bucket_size] = _Glm5MoEGraphBuffers(
            padded=b["padded"][:bucket_size],
            all_tokens=b["all_tokens"][:global_rows],
            router_logits=b["router_logits"][:global_rows],
            topk_indices=b["topk_indices"][:global_rows],
            topk_weights=b["topk_weights"][:global_rows],
            topk_masked_indices=b["topk_masked_indices"][:global_rows],
            topk_masked_weights=b["topk_masked_weights"][:global_rows],
            topk_negative_ones=b["topk_negative_ones"][:global_rows],
            topk_zero_weights=b["topk_zero_weights"][:global_rows],
            rank_ids=positions // bucket_size,
            local_pos=positions % bucket_size,
            expert_counts=b["expert_counts"],
            expert_counters=b["expert_counters"],
            topk_pos=b["topk_pos"][:nk],
            dispatched_x=b["dispatched_x"][:rows],
            intermediate=b["intermediate"][:rows],
            expert_out=b["expert_out"][:rows],
            routed_global_output=b["routed_global_output"][:global_rows],
            local_moe_output=b["local_moe_output"][:bucket_size],
            cu_seqlens=b["cu_seqlens"],
            max_tokens_padded=mtp,
        )

    @staticmethod
    def _round_up(value: int, block: int) -> int:
        return ((value + block - 1) // block) * block

    def release(self) -> None:
        self._views.clear()
        self._base.clear()


class Glm5MoEGraphSegment:
    """Graph-capturable full GLM-5 MoE decode module segment."""

    def __init__(
        self,
        moe,
        pool: Glm5MoEGraphBufferPool,
        comm,
        *,
        world_size: int,
        rank: int,
        device: torch.device,
    ) -> None:
        self.moe = moe
        self.pool = pool
        self.comm = comm
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.device = device
        self.hidden_size = int(moe.hidden_size)
        self.num_experts_per_tok = int(moe.num_experts_per_tok)
        self.num_local_experts = int(moe.experts_per_rank)
        self.expert_start = int(moe.routed_expert_start_idx)
        self.intermediate_size = int(moe.config.moe_intermediate_size)
        self.routed_scaling_factor = float(moe.gate.routed_scaling_factor)

        if not getattr(moe, "_fp8_blockwise_ready", False):
            raise RuntimeError(
                f"Layer {moe.layer_idx}: GLM-5 MoE graph requires FP8 blockwise weights"
            )
        if comm is None:
            raise RuntimeError(f"Layer {moe.layer_idx}: GLM-5 MoE graph requires EP communicator")
        self._validate_shared_expert_graph_safe()

        self.gate_weight_bf16 = moe.gate.weight.detach().to(torch.bfloat16).contiguous()
        self.gate_bias_fp32 = moe.gate.e_score_correction_bias.detach().float().contiguous()

    def _validate_shared_expert_graph_safe(self) -> None:
        shared = getattr(self.moe, "shared_experts", None)
        if shared is None:
            raise RuntimeError(
                f"Layer {self.moe.layer_idx}: GLM-5 MoE graph requires a shared expert module"
            )
        if getattr(shared, "persistent", True) is False:
            raise RuntimeError(
                f"Layer {self.moe.layer_idx}: GLM-5 MoE graph requires persistent "
                "shared expert weights; non-persistent shared expert forward loads "
                "and frees weights inside forward"
            )
        if getattr(shared, "is_fp8", False):
            missing = [
                name
                for name in ("cached_gate", "cached_up", "cached_down")
                if getattr(shared, name, None) is None
            ]
            if missing:
                raise RuntimeError(
                    f"Layer {self.moe.layer_idx}: GLM-5 MoE graph requires cached "
                    f"shared expert FP8 weights; missing {missing}"
                )

    def setup_static_buffers(self, bucket_size: int) -> None:
        if hasattr(self.comm, "disabled"):
            self.comm.disabled = False
        self.pool.setup()

    def release_static_buffers(self, bucket_size: int) -> None:
        self.pool.release()

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "padded": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "moe_output": TensorSpec(
                ("batch_size", self.hidden_size),
                torch.bfloat16,
            ),
        }

    def forward(
        self,
        *,
        padded: torch.Tensor,
        rank_token_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bucket_size = padded.shape[0]
        bufs = self.pool.get(bucket_size)
        global_rows = self.world_size * bucket_size

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                bufs.all_tokens,
                padded,
                stream=torch.cuda.current_stream(self.device),
            )

        glm5_router_gemm_cuda(
            bufs.all_tokens,
            self.gate_weight_bf16,
            router_logits=bufs.router_logits,
            rank_token_counts=rank_token_counts,
            bucket_size=bucket_size,
            world_size=self.world_size,
        )
        gate_sigmoid_topk_cuda(
            bufs.router_logits,
            self.gate_bias_fp32,
            k=self.num_experts_per_tok,
            routed_scaling_factor=self.routed_scaling_factor,
            topk_indices=bufs.topk_indices,
            topk_weights=bufs.topk_weights,
        )

        valid_per_row = rank_token_counts[bufs.rank_ids]
        padding_mask = bufs.local_pos >= valid_per_row
        padding_mask_2d = padding_mask.unsqueeze(1).expand_as(bufs.topk_indices)
        torch.where(
            padding_mask_2d,
            bufs.topk_negative_ones,
            bufs.topk_indices,
            out=bufs.topk_masked_indices,
        )
        torch.where(
            padding_mask_2d,
            bufs.topk_zero_weights,
            bufs.topk_weights,
            out=bufs.topk_masked_weights,
        )

        bufs.dispatched_x.zero_()
        expert_counts, topk_pos = dispatch_scatter_3d(
            bufs.all_tokens,
            bufs.topk_masked_indices,
            bufs.dispatched_x,
            self.expert_start,
            self.num_local_experts,
            bufs.max_tokens_padded,
            bufs.expert_counts,
            bufs.expert_counters,
            bufs.topk_pos,
        )

        self._fp8_blockwise_gemm_3d(bufs, expert_counts)

        bufs.routed_global_output.zero_()
        routed_global_output = reduce_weighted_scatter(
            bufs.expert_out,
            topk_pos,
            bufs.topk_masked_weights,
            global_rows,
            self.hidden_size,
            self.num_experts_per_tok,
            output=bufs.routed_global_output,
        )

        import torch.distributed as dist

        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                routed_global_output,
                op=dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        start = self.rank * bucket_size
        bufs.local_moe_output.copy_(routed_global_output[start:start + bucket_size])
        bufs.local_moe_output.add_(self.moe.shared_expert_forward(padded))
        return {"moe_output": bufs.local_moe_output}

    def _fp8_blockwise_gemm_3d(
        self,
        bufs: _Glm5MoEGraphBuffers,
        expert_counts: torch.Tensor,
    ) -> None:
        e = self.num_local_experts
        h = self.hidden_size
        n = self.intermediate_size
        mtp = bufs.max_tokens_padded
        seqlens = expert_counts[:e]
        avg = max(mtp // max(e, 1), 1)

        x_3d = bufs.dispatched_x.view(e, mtp, h)
        x_quant_3d, x_scale_3d = _act_quant_3d(x_3d, seqlens)
        x_quant = x_quant_3d.view(e * mtp, h)
        x_scale_t = x_scale_3d.view(e * mtp, -1).t().contiguous()

        s1_result = grouped_fp8_blockwise_fused_s1(
            x_quant.view(torch.float8_e4m3fn),
            x_scale_t,
            self.moe.fp8_gate_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_up_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_gate_ws3d,
            self.moe.fp8_up_ws3d,
            seqlens,
            bufs.cu_seqlens,
            avg,
            output=bufs.intermediate,
        )
        inter_quant_3d, inter_scale_3d = _act_quant_3d(
            s1_result.view(e, mtp, n),
            seqlens,
        )
        inter_quant = inter_quant_3d.view(e * mtp, n)
        inter_scale_t = inter_scale_3d.view(e * mtp, -1).t().contiguous()

        grouped_fp8_blockwise_s3(
            inter_quant.view(torch.float8_e4m3fn),
            inter_scale_t,
            self.moe.fp8_down_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_down_ws3d,
            seqlens,
            bufs.cu_seqlens,
            avg,
            output=bufs.expert_out,
        )
