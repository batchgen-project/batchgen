from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import triton

from batchgen.moe.v4_ragged_moe_sm120 import (
    _RAGGED_STAGE_CFG,
    _ragged_mxfp4_matmul_kernel,
)


def _max_ragged_block_count(num_slots: int, num_experts: int, block_m: int) -> int:
    if num_slots <= 0:
        return 0
    seeded = min(num_slots, num_experts)
    return seeded + max(num_slots - seeded, 0) // block_m


def _resolve_ragged_bundle(weight_bundle: dict[str, object]) -> dict[str, torch.Tensor]:
    bundle = weight_bundle.get("ragged_bundle", weight_bundle)
    if not isinstance(bundle, dict):
        raise TypeError(
            "weight_bundle must be a ragged bundle or a dict containing ragged_bundle"
        )
    required = ("stage1_weight", "stage1_scale", "stage2_weight", "stage2_scale")
    for name in required:
        if name not in bundle:
            raise KeyError(name)
    return bundle  # type: ignore[return-value]


def _normalize_buckets(max_batch: int, buckets: Iterable[int] | None) -> tuple[int, ...]:
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")
    if buckets is None:
        canonical = (1, 8, 16, 32, 64, 128, 256)
        selected = [bucket for bucket in canonical if bucket <= max_batch]
        if not selected or selected[-1] != max_batch:
            selected.append(max_batch)
    else:
        selected = sorted({int(bucket) for bucket in buckets})
        if not selected:
            raise ValueError("buckets must be non-empty")
        if selected[0] <= 0:
            raise ValueError(f"invalid bucket list: {selected}")
        if selected[-1] != max_batch:
            selected.append(max_batch)
    return tuple(selected)


def _launch_static_ragged_stage(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_experts: torch.Tensor,
    block_slot_starts: torch.Tensor,
    block_row_starts: torch.Tensor,
    expt_hist: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> None:
    out_features = out.shape[1]
    grid = (block_experts.numel() * triton.cdiv(out_features, block_n),)
    _ragged_mxfp4_matmul_kernel[grid](
        x,
        weight,
        scale,
        block_experts,
        block_slot_starts,
        block_row_starts,
        expt_hist,
        out,
        x.shape[0],
        out_features,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@dataclass(frozen=True)
class GraphCaptureStats:
    bucket_size: int
    capture_us: float
    warmup_iters: int
    memory_bytes: int
    slots_max: int
    blocks_max: int


class _BucketGraph:
    WARMUP_ITERS = 3

    def __init__(
        self,
        bucket_size: int,
        *,
        hidden: int,
        intermediate: int,
        topk: int,
        owned_start: int,
        owned_count: int,
        ragged: dict[str, torch.Tensor],
        device: torch.device,
        swiglu_limit: float,
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
        graph_pool: object | None,
    ) -> None:
        self.bucket_size = int(bucket_size)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.topk = int(topk)
        self.owned_start = int(owned_start)
        self.owned_count = int(owned_count)
        self.owned_end = self.owned_start + self.owned_count
        self.invalid_global_expert = self.owned_end
        self.ragged = ragged
        self.device = device
        self.swiglu_limit = float(swiglu_limit)
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.num_warps = int(num_warps)
        self.num_stages = int(num_stages)
        self.graph_pool = graph_pool

        self.slots_max = self.bucket_size * self.topk
        self.blocks_per_expert_cap = triton.cdiv(self.slots_max, self.block_m)
        self.blocks_max = _max_ragged_block_count(
            self.slots_max, self.owned_count, self.block_m
        )
        self.candidate_blocks = self.owned_count * self.blocks_per_expert_cap

        self._allocate_static_buffers()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stats = self._capture_graph()

    def _allocate_static_buffers(self) -> None:
        device = self.device
        self.static_hidden_states = torch.zeros(
            (self.bucket_size, self.hidden), device=device, dtype=torch.bfloat16
        )
        self.static_topk_indices = torch.full(
            (self.bucket_size, self.topk),
            self.invalid_global_expert,
            device=device,
            dtype=torch.int64,
        )
        self.static_topk_weights = torch.zeros(
            (self.bucket_size, self.topk), device=device, dtype=torch.float32
        )

        self._token_ids_flat = torch.arange(
            self.bucket_size, device=device, dtype=torch.int64
        ).repeat_interleave(self.topk)

        self._candidate_expert_ids_i32 = (
            torch.arange(self.owned_count, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(self.owned_count, self.blocks_per_expert_cap)
            .reshape(-1)
            .contiguous()
        )
        self._candidate_expert_ids_i64 = self._candidate_expert_ids_i32.to(torch.int64)
        self._candidate_block_ids = (
            torch.arange(self.blocks_per_expert_cap, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(self.owned_count, self.blocks_per_expert_cap)
            .reshape(-1)
            .contiguous()
        )
        self._candidate_row_starts = (
            self._candidate_block_ids * self.block_m
        ).contiguous()
        self._output_block_ids = torch.arange(
            self.blocks_max, device=device, dtype=torch.int32
        )
        self._ones_slots = torch.ones(self.slots_max, device=device, dtype=torch.int32)

        self._valid_mask = torch.empty(self.slots_max, device=device, dtype=torch.bool)
        self._local_eids_ext = torch.empty(self.slots_max, device=device, dtype=torch.int64)
        self._sorted_eids_ext = torch.empty(self.slots_max, device=device, dtype=torch.int64)
        self._sort_order = torch.empty(self.slots_max, device=device, dtype=torch.int64)
        self.sorted_token_ids = torch.empty(
            self.slots_max, device=device, dtype=torch.int64
        )
        self.sorted_weights = torch.empty(
            self.slots_max, device=device, dtype=torch.float32
        )
        self.expt_hist_ext = torch.zeros(
            self.owned_count + 1, device=device, dtype=torch.int32
        )
        self.expt_hist = self.expt_hist_ext[:-1]
        self.expt_offsets = torch.zeros(
            self.owned_count + 1, device=device, dtype=torch.int32
        )
        self.block_counts = torch.zeros(
            self.owned_count, device=device, dtype=torch.int32
        )
        self.block_offsets = torch.zeros(
            self.owned_count + 1, device=device, dtype=torch.int32
        )

        self._candidate_block_limits = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.int32
        )
        self._candidate_block_mask = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.bool
        )
        self._candidate_block_mask_i32 = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.int32
        )
        self._candidate_block_ranks = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.int64
        )
        self._safe_candidate_block_ranks = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.int64
        )
        self._candidate_slot_starts = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.int32
        )
        self._masked_candidate_experts = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.int32
        )
        self._masked_candidate_slot_starts = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.int32
        )
        self._masked_candidate_row_starts = torch.empty(
            self.candidate_blocks, device=device, dtype=torch.int32
        )
        self._valid_output_block_mask = torch.empty(
            self.blocks_max, device=device, dtype=torch.bool
        )

        self.block_experts = torch.zeros(
            self.blocks_max, device=device, dtype=torch.int32
        )
        self.block_slot_starts = torch.zeros(
            self.blocks_max, device=device, dtype=torch.int32
        )
        self.block_row_starts = torch.full(
            (self.blocks_max,), self.slots_max, device=device, dtype=torch.int32
        )

        self.sorted_hidden = torch.zeros(
            (self.slots_max, self.hidden), device=device, dtype=torch.bfloat16
        )
        self.stage1_out = torch.zeros(
            (self.slots_max, 2 * self.intermediate),
            device=device,
            dtype=torch.bfloat16,
        )
        self.gate_fp32 = torch.zeros(
            (self.slots_max, self.intermediate), device=device, dtype=torch.float32
        )
        self.up_fp32 = torch.zeros(
            (self.slots_max, self.intermediate), device=device, dtype=torch.float32
        )
        self.activated_fp32 = torch.zeros(
            (self.slots_max, self.intermediate), device=device, dtype=torch.float32
        )
        self.stage2_in = torch.zeros(
            (self.slots_max, self.intermediate), device=device, dtype=torch.bfloat16
        )
        self.stage2_out = torch.zeros(
            (self.slots_max, self.hidden), device=device, dtype=torch.bfloat16
        )
        self.stage2_out_fp32 = torch.zeros(
            (self.slots_max, self.hidden), device=device, dtype=torch.float32
        )
        self.output = torch.zeros(
            (self.bucket_size, self.hidden), device=device, dtype=torch.float32
        )
        self.memory_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._static_tensors_for_memory()
        )

    def _static_tensors_for_memory(self) -> tuple[torch.Tensor, ...]:
        return (
            self.static_hidden_states,
            self.static_topk_indices,
            self.static_topk_weights,
            self._token_ids_flat,
            self._candidate_expert_ids_i32,
            self._candidate_expert_ids_i64,
            self._candidate_block_ids,
            self._candidate_row_starts,
            self._output_block_ids,
            self._ones_slots,
            self._valid_mask,
            self._local_eids_ext,
            self._sorted_eids_ext,
            self._sort_order,
            self.sorted_token_ids,
            self.sorted_weights,
            self.expt_hist_ext,
            self.expt_offsets,
            self.block_counts,
            self.block_offsets,
            self._candidate_block_limits,
            self._candidate_block_mask,
            self._candidate_block_mask_i32,
            self._candidate_block_ranks,
            self._safe_candidate_block_ranks,
            self._candidate_slot_starts,
            self._masked_candidate_experts,
            self._masked_candidate_slot_starts,
            self._masked_candidate_row_starts,
            self._valid_output_block_mask,
            self.block_experts,
            self.block_slot_starts,
            self.block_row_starts,
            self.sorted_hidden,
            self.stage1_out,
            self.gate_fp32,
            self.up_fp32,
            self.activated_fp32,
            self.stage2_in,
            self.stage2_out,
            self.stage2_out_fp32,
            self.output,
        )

    def _build_static_routing(self) -> None:
        flat_global = self.static_topk_indices.view(-1)
        flat_weights = self.static_topk_weights.view(-1)

        torch.sub(flat_global, self.owned_start, out=self._local_eids_ext)
        torch.ge(flat_global, self.owned_start, out=self._valid_mask)
        self._valid_mask.logical_and_(flat_global < self.owned_end)
        self._local_eids_ext.masked_fill_(~self._valid_mask, self.owned_count)

        torch.sort(self._local_eids_ext, out=(self._sorted_eids_ext, self._sort_order))
        torch.index_select(
            self._token_ids_flat, 0, self._sort_order, out=self.sorted_token_ids
        )
        torch.index_select(flat_weights, 0, self._sort_order, out=self.sorted_weights)
        self.sorted_weights.masked_fill_(self._sorted_eids_ext == self.owned_count, 0.0)

        self.expt_hist_ext.zero_()
        self.expt_hist_ext.scatter_add_(0, self._sorted_eids_ext, self._ones_slots)
        self.expt_offsets.zero_()
        torch.cumsum(self.expt_hist, 0, out=self.expt_offsets[1:])

        self.block_counts.copy_(
            torch.div(
                self.expt_hist + (self.block_m - 1),
                self.block_m,
                rounding_mode="floor",
            )
        )
        self.block_offsets.zero_()
        torch.cumsum(self.block_counts, 0, out=self.block_offsets[1:])

        torch.index_select(
            self.block_counts,
            0,
            self._candidate_expert_ids_i64,
            out=self._candidate_block_limits,
        )
        torch.lt(
            self._candidate_block_ids,
            self._candidate_block_limits,
            out=self._candidate_block_mask,
        )
        self._candidate_block_mask_i32.copy_(self._candidate_block_mask)
        torch.cumsum(self._candidate_block_mask_i32, 0, out=self._candidate_block_ranks)
        self._candidate_block_ranks.sub_(1)
        self._safe_candidate_block_ranks.copy_(self._candidate_block_ranks)
        self._safe_candidate_block_ranks.masked_fill_(~self._candidate_block_mask, 0)

        torch.index_select(
            self.expt_offsets[:-1],
            0,
            self._candidate_expert_ids_i64,
            out=self._candidate_slot_starts,
        )
        self._candidate_slot_starts.add_(self._candidate_row_starts)

        self._masked_candidate_experts.copy_(self._candidate_expert_ids_i32)
        self._masked_candidate_experts.mul_(self._candidate_block_mask_i32)
        self._masked_candidate_slot_starts.copy_(self._candidate_slot_starts)
        self._masked_candidate_slot_starts.mul_(self._candidate_block_mask_i32)
        self._masked_candidate_row_starts.copy_(self._candidate_row_starts)
        self._masked_candidate_row_starts.mul_(self._candidate_block_mask_i32)

        self.block_experts.zero_()
        self.block_slot_starts.zero_()
        self.block_row_starts.zero_()
        self.block_experts.scatter_add_(
            0, self._safe_candidate_block_ranks, self._masked_candidate_experts
        )
        self.block_slot_starts.scatter_add_(
            0, self._safe_candidate_block_ranks, self._masked_candidate_slot_starts
        )
        self.block_row_starts.scatter_add_(
            0, self._safe_candidate_block_ranks, self._masked_candidate_row_starts
        )

        torch.lt(
            self._output_block_ids,
            self.block_offsets[-1],
            out=self._valid_output_block_mask,
        )
        self.block_row_starts.masked_fill_(
            ~self._valid_output_block_mask, self.slots_max
        )

    def _run_static_forward(self) -> torch.Tensor:
        self._build_static_routing()
        torch.index_select(
            self.static_hidden_states, 0, self.sorted_token_ids, out=self.sorted_hidden
        )

        self.stage1_out.zero_()
        _launch_static_ragged_stage(
            self.sorted_hidden,
            self.ragged["stage1_weight"],
            self.ragged["stage1_scale"],
            self.block_experts,
            self.block_slot_starts,
            self.block_row_starts,
            self.expt_hist,
            self.stage1_out,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        self.gate_fp32.copy_(self.stage1_out[:, : self.intermediate])
        self.up_fp32.copy_(self.stage1_out[:, self.intermediate :])
        if self.swiglu_limit > 0:
            self.gate_fp32.clamp_(max=self.swiglu_limit)
            self.up_fp32.clamp_(min=-self.swiglu_limit, max=self.swiglu_limit)
        self.activated_fp32.copy_(self.gate_fp32)
        self.activated_fp32.sigmoid_()
        self.activated_fp32.mul_(self.gate_fp32)
        self.activated_fp32.mul_(self.up_fp32)
        self.activated_fp32.mul_(self.sorted_weights.unsqueeze(-1))
        self.stage2_in.copy_(self.activated_fp32)

        self.stage2_out.zero_()
        _launch_static_ragged_stage(
            self.stage2_in,
            self.ragged["stage2_weight"],
            self.ragged["stage2_scale"],
            self.block_experts,
            self.block_slot_starts,
            self.block_row_starts,
            self.expt_hist,
            self.stage2_out,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        self.stage2_out_fp32.copy_(self.stage2_out)
        self.output.zero_()
        self.output.index_add_(0, self.sorted_token_ids, self.stage2_out_fp32)
        return self.output

    def _capture_graph(self) -> GraphCaptureStats:
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.WARMUP_ITERS):
                self._run_static_forward()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        graph_ctx: dict[str, object] = {}
        if self.graph_pool is not None:
            graph_ctx["pool"] = self.graph_pool
        with torch.cuda.graph(self.graph, **graph_ctx):
            self._run_static_forward()
        torch.cuda.synchronize()
        return GraphCaptureStats(
            bucket_size=self.bucket_size,
            capture_us=(time.perf_counter() - t0) * 1_000_000.0,
            warmup_iters=self.WARMUP_ITERS,
            memory_bytes=self.memory_bytes,
            slots_max=self.slots_max,
            blocks_max=self.blocks_max,
        )

    @torch.inference_mode()
    def copy_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        batch = int(hidden_states.shape[0])
        if batch > self.bucket_size:
            raise ValueError(
                f"batch={batch} exceeds bucket_size={self.bucket_size}"
            )
        self.static_hidden_states.zero_()
        self.static_topk_weights.zero_()
        self.static_topk_indices.fill_(self.invalid_global_expert)
        self.static_hidden_states[:batch].copy_(hidden_states, non_blocking=True)
        self.static_topk_indices[:batch].copy_(topk_indices, non_blocking=True)
        self.static_topk_weights[:batch].copy_(topk_weights, non_blocking=True)

    @torch.inference_mode()
    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.output

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(hidden_states.shape[0])
        self.copy_inputs(hidden_states, topk_indices, topk_weights)
        self.graph.replay()
        return self.output[:batch]


class V4GraphMoE:
    """Bucketed CUDA-graph wrapper for ragged MXFP4 MoE.

    Each bucket owns a separate capture sized to that bucket, so replay at B=64 can
    use a B=64 graph instead of replaying a B=256 capture over padded scratch.
    """

    def __init__(
        self,
        max_batch: int,
        config: Any,
        weight_bundle: dict[str, object],
        *,
        owned_start: int = 0,
        owned_count: int | None = None,
        device: torch.device | None = None,
        swiglu_limit: float | None = None,
        buckets: Iterable[int] | None = None,
    ) -> None:
        self.max_batch = int(max_batch)
        self.hidden = int(config.hidden_size)
        self.intermediate = int(config.moe_intermediate_size)
        self.topk = int(config.num_experts_per_tok)
        self.device = device or torch.device("cuda")
        self.owned_start = int(owned_start)
        self.ragged = _resolve_ragged_bundle(weight_bundle)
        inferred_owned_count = int(self.ragged["stage1_weight"].shape[0])
        self.owned_count = (
            inferred_owned_count if owned_count is None else int(owned_count)
        )
        if self.owned_count != inferred_owned_count:
            raise ValueError(
                f"owned_count={self.owned_count} does not match ragged bundle expert count={inferred_owned_count}"
            )
        self.swiglu_limit = float(
            config.swiglu_limit if swiglu_limit is None else swiglu_limit
        )
        self.bucket_sizes = _normalize_buckets(self.max_batch, buckets)

        cfg = _RAGGED_STAGE_CFG
        self.block_m = int(cfg["block_m"])
        self.block_n = int(cfg["block_n"])
        self.block_k = int(cfg["block_k"])
        self.num_warps = int(cfg["num_warps"])
        self.num_stages = int(cfg["num_stages"])
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._buckets = {
            bucket_size: _BucketGraph(
                bucket_size,
                hidden=self.hidden,
                intermediate=self.intermediate,
                topk=self.topk,
                owned_start=self.owned_start,
                owned_count=self.owned_count,
                ragged=self.ragged,
                device=self.device,
                swiglu_limit=self.swiglu_limit,
                block_m=self.block_m,
                block_n=self.block_n,
                block_k=self.block_k,
                num_warps=self.num_warps,
                num_stages=self.num_stages,
                graph_pool=self._graph_pool,
            )
            for bucket_size in self.bucket_sizes
        }
        self.capture_stats = {
            bucket_size: bucket_graph.capture_stats
            for bucket_size, bucket_graph in self._buckets.items()
        }
        self.total_capture_us = float(
            sum(stats.capture_us for stats in self.capture_stats.values())
        )
        self.total_memory_bytes = int(
            sum(stats.memory_bytes for stats in self.capture_stats.values())
        )

    def pick_bucket(self, batch_size: int) -> int:
        batch = int(batch_size)
        if batch <= 0:
            raise ValueError(f"batch_size must be positive, got {batch}")
        for bucket_size in self.bucket_sizes:
            if batch <= bucket_size:
                return bucket_size
        raise ValueError(
            f"batch={batch} exceeds max bucket {self.bucket_sizes[-1]}"
        )

    def bucket_memory_bytes(self, bucket_size: int) -> int:
        return int(self.capture_stats[bucket_size].memory_bytes)

    def bucket_descriptions(self) -> dict[int, dict[str, float | int]]:
        return {
            bucket_size: {
                "capture_us": stats.capture_us,
                "memory_bytes": stats.memory_bytes,
                "memory_mb": stats.memory_bytes / (1024.0 * 1024.0),
                "slots_max": stats.slots_max,
                "blocks_max": stats.blocks_max,
            }
            for bucket_size, stats in self.capture_stats.items()
        }

    @torch.inference_mode()
    def copy_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> int:
        batch = int(hidden_states.shape[0])
        if hidden_states.shape != (batch, self.hidden):
            raise ValueError(
                f"hidden_states must have shape ({batch}, {self.hidden}), got {tuple(hidden_states.shape)}"
            )
        if topk_indices.shape != (batch, self.topk):
            raise ValueError(
                f"topk_indices must have shape ({batch}, {self.topk}), got {tuple(topk_indices.shape)}"
            )
        if topk_weights.shape != (batch, self.topk):
            raise ValueError(
                f"topk_weights must have shape ({batch}, {self.topk}), got {tuple(topk_weights.shape)}"
            )
        bucket_size = self.pick_bucket(batch)
        self._buckets[bucket_size].copy_inputs(hidden_states, topk_indices, topk_weights)
        return bucket_size

    @torch.inference_mode()
    def replay(self, bucket_size: int, *, batch_size: int | None = None) -> torch.Tensor:
        bucket_graph = self._buckets[int(bucket_size)]
        out = bucket_graph.replay()
        if batch_size is None:
            return out
        return out[: int(batch_size)]

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(hidden_states.shape[0])
        bucket_size = self.copy_inputs(hidden_states, topk_indices, topk_weights)
        return self.replay(bucket_size, batch_size=batch)


__all__ = ["GraphCaptureStats", "V4GraphMoE"]
