# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #

"""CUDA-graph resident-MoE segment for Kimi-K3.

The decode model uses TP8 attention groups over a world-size EP communicator.
Each rank receives the same group rows, scatters them locally across its TP
group, and the resident MXFP4 layer gathers the resulting latent/routing rows
over EP.  This module captures that complete resident-MoE path with fixed
buffers.  It deliberately does not call ``ResidentEPMXFP4MoELayer.forward``:
that eager seam allocates its dispatch buffers and pointer tables on every
step, which is precisely the host/launch overhead this graph removes.

Padding is represented by zero rows in the fixed TP-group input.  K3's routed
and shared projections are bias-free, so zero rows produce zero output; no
host-side valid-row read is needed in the graph.  The graph is therefore
collective-safe even when the two DP groups have different batch sizes, as long
as every rank has at least one live row.  The caller disables this path for an
empty global rank set.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.moe.dispatch_scatter_3d import (
    dispatch_scatter_3d,
    reduce_weighted_scatter_fp32,
)
from batchgen.moe.marlin_grouped_moe import (
    marlin_grouped_m16_mxfp4,
    marlin_grouped_stage1_fused_mxfp4_situ,
)
from batchgen.moe.routing import gate_sigmoid_topk_cuda

logger = logging.getLogger(__name__)


def k3_moe_graph_buckets(attention_buckets: Iterable[int], tp_size: int) -> List[int]:
    """Round attention buckets to the fixed TP-group MoE input geometry.

    A K3 MoE graph input is the whole replicated TP-group batch.  Its rows are
    split evenly across ``tp_size`` ranks, so a graph bucket must be a multiple
    of that group size.  The returned set is deterministic and contains the
    exact ``tp_size * ceil(B / tp_size)`` capacity for every attention bucket.
    """
    group = int(tp_size)
    if group <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    buckets = sorted({int(b) for b in attention_buckets})
    if not buckets or any(b <= 0 for b in buckets):
        raise ValueError(f"attention_buckets must contain positive sizes, got {buckets}")
    return sorted({group * math.ceil(b / group) for b in buckets})


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


# The fused router/down-projection slab lives on the MoE block, not on the graph
# segment: decode graphs are torn down and rebuilt across bucket changes and the
# weights must be relocated exactly once per layer.
_FUSED_FRONT_ATTR = "_k3_fused_front_weight"


def fuse_router_and_down_proj(moe, down_proj) -> Optional[torch.Tensor]:
    """Merge the router GEMM and the latent down-projection into one slab.

    Both read the same ``[tokens, hidden]`` decode activation, so their weights
    can share one contiguous ``[num_experts + latent, hidden]`` BF16 tensor and
    be consumed by a single GEMM.  ``moe.gate.weight`` and ``down_proj.weight``
    are rebound to views of that slab, so the old per-weight storage is dropped
    rather than duplicated.  The slab is cached on ``moe`` and returned again on
    every later call.  Returns ``None`` when the weight dtypes or shapes make
    the fusion ineligible; the caller then keeps the unfused path.
    """
    cached = getattr(moe, _FUSED_FRONT_ATTR, None)
    if cached is not None:
        return cached

    gate_weight = moe.gate.weight
    down_weight = down_proj.weight
    if gate_weight.dtype is not torch.bfloat16 or down_weight.dtype is not torch.bfloat16:
        return None
    if (
        gate_weight.ndim != 2
        or down_weight.ndim != 2
        or gate_weight.shape[1] != down_weight.shape[1]
    ):
        return None

    num_experts = int(gate_weight.shape[0])
    with torch.no_grad():
        fused = torch.empty(
            (num_experts + int(down_weight.shape[0]), int(gate_weight.shape[1])),
            dtype=torch.bfloat16,
            device=gate_weight.device,
        )
        fused[:num_experts].copy_(gate_weight.detach())
        fused[num_experts:].copy_(down_weight.detach())
    moe.gate.weight = nn.Parameter(fused[:num_experts], requires_grad=False)
    down_proj.weight = nn.Parameter(fused[num_experts:], requires_grad=False)
    setattr(moe, _FUSED_FRONT_ATTR, fused)
    return fused


def split_fused_front(
    fused_out: torch.Tensor, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split one fused GEMM output into FP32 router logits and BF16 latent rows."""
    return (
        fused_out[:, :num_experts],
        fused_out[:, num_experts:].to(torch.bfloat16),
    )


def fused_gate_kernel_eligible(gate, *, fused_front, device) -> bool:
    """Whether ``gate_sigmoid_topk_cuda`` reproduces this gate's selection.

    The kernel implements exactly one selection recipe: sigmoid scores, an FP32
    additive correction bias, ungrouped top-k, renormalization, and a scalar
    scaling factor.  Anything else — softmax scoring, an active expert-group
    mask, a non-FP32 bias, a different ``top_k``, no renormalization — must keep
    the eager ``select_experts`` path.  The kernel also only replaces the gate on
    the fused-front CUDA graph path, where the FP32 router logits already exist
    as the leading columns of the fused GEMM output.
    """
    if getattr(device, "type", None) != "cuda":
        return False
    if fused_front is None:
        return False
    if getattr(gate, "moe_router_activation_func", None) != "sigmoid":
        return False
    num_expert_group = int(getattr(gate, "num_expert_group", 1) or 1)
    topk_group = int(getattr(gate, "topk_group", 1) or 1)
    if num_expert_group > 1 and num_expert_group > topk_group:
        return False
    if int(getattr(gate, "top_k", 0)) != 16:
        return False
    if not bool(getattr(gate, "moe_renormalize", False)):
        return False
    bias = getattr(gate, "e_score_correction_bias", None)
    if bias is None or bias.dtype is not torch.float32:
        return False
    return True


_MM_OUT_FP32_SUPPORT: Dict[str, bool] = {}


def _supports_mm_out_fp32(device: torch.device) -> bool:
    """Probe once whether ``torch.mm`` accepts ``out_dtype=torch.float32`` here.

    The BF16-input/FP32-output GEMM is what keeps the fused router logits in
    FP32.  Older torch builds reject the keyword, so probe before capture and
    fall back to the separate gate + down-projection path when unavailable.
    """
    key = str(device)
    supported = _MM_OUT_FP32_SUPPORT.get(key)
    if supported is None:
        probe = torch.zeros((1, 1), dtype=torch.bfloat16, device=device)
        try:
            torch.mm(probe, probe, out_dtype=torch.float32)
            supported = True
        except (RuntimeError, TypeError):
            supported = False
        _MM_OUT_FP32_SUPPORT[key] = supported
        logger.info(
            "K3 fused router/down front out_dtype probe: device=%s supported=%s",
            device,
            supported,
        )
    return supported


@dataclass
class _K3MoEGraphBuffers:
    all_latent: torch.Tensor
    all_indices: torch.Tensor
    all_weights: torch.Tensor
    local_indices: torch.Tensor
    local_weights: torch.Tensor
    local_latent: torch.Tensor
    dispatched: torch.Tensor
    intermediate: torch.Tensor
    expert_output: torch.Tensor
    combined: torch.Tensor
    local_combined: torch.Tensor
    expert_counts: torch.Tensor
    expert_counters: torch.Tensor
    topk_pos: torch.Tensor
    expert_starts: torch.Tensor
    s1_C_ptrs: torch.Tensor
    s3_C_ptrs: torch.Tensor
    s1_workspace: torch.Tensor
    s3_workspace: torch.Tensor
    local_tokens: int
    global_tokens: int
    max_tokens_padded: int
    max_m_tiles: int


class K3MoEGraphBufferPool:
    """Static activation/routing storage shared by all K3 MoE layer graphs."""

    def __init__(
        self,
        *,
        world_size: int,
        tp_size: int,
        num_local_experts: int,
        intermediate_size: int,
        latent_size: int,
        hidden_size: int,
        top_k: int,
        expert_buckets: Iterable[int],
        device: torch.device,
    ) -> None:
        self.world_size = int(world_size)
        self.tp_size = int(tp_size)
        self.num_local_experts = int(num_local_experts)
        self.intermediate_size = int(intermediate_size)
        self.latent_size = int(latent_size)
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.device = device
        # Mirror SGLang's KimiLinear implementation: one model-wide auxiliary
        # stream is enough to overlap the replicated shared expert with the
        # routed-expert path.  Contract tests also construct this pool on CPU,
        # where creating a CUDA stream would be invalid.
        self.shared_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.bucket_sizes = sorted({int(b) for b in expert_buckets})
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if self.num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        if not self.bucket_sizes or any(b <= 0 for b in self.bucket_sizes):
            raise ValueError(f"expert_buckets must be positive, got {self.bucket_sizes}")
        self._base: Dict[str, torch.Tensor] = {}
        self._views: Dict[int, _K3MoEGraphBuffers] = {}
        self._setup_done = False

    def setup(self) -> None:
        if self._setup_done:
            return

        max_bucket = max(self.bucket_sizes)
        # Every graph bucket is a full TP-group batch.  The TP group size is
        # inferred by the caller as bucket/local_tokens; local_tokens is always
        # the same for the corresponding graph.  The pool only needs the EP
        # global row count, so the caller supplies buckets in group geometry and
        # records the local row count at view creation below.
        # ``max_local_tokens`` is filled by the segment when it creates views.
        # K3MoEGraphBufferPool is intentionally usable in CPU contract tests,
        # hence all storage is ordinary torch allocation rather than CUDA-only
        # helpers.
        self._max_group_bucket = max_bucket
        self._max_local_tokens = 0
        self._max_global_tokens = 0
        if any(bucket % self.tp_size for bucket in self.bucket_sizes):
            raise ValueError(
                f"all K3 MoE buckets must be divisible by tp_size={self.tp_size}: "
                f"{self.bucket_sizes}"
            )
        self._max_local_tokens = max_bucket // self.tp_size
        self._max_global_tokens = self.world_size * self._max_local_tokens
        max_mtp = _round_up(self._max_global_tokens, 16)
        e = self.num_local_experts
        n = self.intermediate_size
        k = self.latent_size
        nk = self._max_global_tokens * self.top_k
        d = self.device

        b = self._base
        b["all_latent"] = torch.empty((self._max_global_tokens, k), dtype=torch.bfloat16, device=d)
        b["all_indices"] = torch.empty((self._max_global_tokens, self.top_k), dtype=torch.int32, device=d)
        b["all_weights"] = torch.empty((self._max_global_tokens, self.top_k), dtype=torch.float32, device=d)
        # Per-rank routing outputs of the fused gate kernel. Preallocating them
        # in the kernel's native int32/FP32 layout is what lets the graph drop
        # the dtype cast and the padding-mask ``torch.where`` nodes.
        b["local_indices"] = torch.empty(
            (self._max_local_tokens, self.top_k), dtype=torch.int32, device=d
        )
        b["local_weights"] = torch.empty(
            (self._max_local_tokens, self.top_k), dtype=torch.float32, device=d
        )
        # Destination for the gate kernel's latent epilogue: the fused GEMM's
        # latent columns are cast to BF16 straight into the EP all-gather
        # source, so the graph carries no separate strided FP32->BF16 copy.
        b["local_latent"] = torch.empty(
            (self._max_local_tokens, k), dtype=torch.bfloat16, device=d
        )
        b["dispatched"] = torch.empty((e * max_mtp, k), dtype=torch.bfloat16, device=d)
        b["intermediate"] = torch.empty((e * max_mtp, n), dtype=torch.bfloat16, device=d)
        b["expert_output"] = torch.empty((e * max_mtp, k), dtype=torch.bfloat16, device=d)
        b["combined"] = torch.empty((self._max_global_tokens, k), dtype=torch.float32, device=d)
        # ``combined`` is rank-major [EP, local_tokens, latent].  The
        # reduce-scatter writes one rank's chunk directly here, so the
        # post-combine norm/up projection never needs a rank-local slice or a
        # second device copy.
        b["local_combined"] = torch.empty(
            (self._max_local_tokens, k), dtype=torch.float32, device=d
        )
        b["expert_counts"] = torch.empty((e,), dtype=torch.int32, device=d)
        b["expert_counters"] = torch.empty((e,), dtype=torch.int32, device=d)
        b["topk_pos"] = torch.empty((nk,), dtype=torch.int32, device=d)
        b["s1_workspace"] = torch.empty((e * (n // 256 + 17),), dtype=torch.int32, device=d)
        b["s3_workspace"] = torch.empty((e * (k // 256 + 17),), dtype=torch.int32, device=d)

        for bucket in self.bucket_sizes:
            self._create_view(bucket, max_mtp)
        self._setup_done = True

        total_bytes = sum(t.numel() * t.element_size() for t in b.values())
        logger.info(
            "K3MoEGraphBufferPool: allocated %.2f MiB "
            "(world=%d, tp=%d, max_group=%d, max_global=%d, experts/rank=%d)",
            total_bytes / (1024**2), self.world_size, self.tp_size,
            self._max_group_bucket, self._max_global_tokens,
            self.num_local_experts,
        )

    def _create_view(self, bucket: int, max_mtp: int) -> None:
        local_tokens = int(bucket) // self.tp_size
        global_tokens = self.world_size * local_tokens
        mtp = _round_up(global_tokens, 16)
        e = self.num_local_experts
        n = self.intermediate_size
        k = self.latent_size
        b = self._base
        row_n = n * 2
        row_k = k * 2
        expert_starts = torch.arange(e, dtype=torch.int32, device=self.device) * mtp
        s1_ptrs = torch.tensor(
            [b["intermediate"].data_ptr() + i * mtp * row_n for i in range(e)],
            dtype=torch.int64,
            device=self.device,
        )
        s3_ptrs = torch.tensor(
            [b["expert_output"].data_ptr() + i * mtp * row_k for i in range(e)],
            dtype=torch.int64,
            device=self.device,
        )
        self._views[int(bucket)] = _K3MoEGraphBuffers(
            all_latent=b["all_latent"][:global_tokens],
            all_indices=b["all_indices"][:global_tokens],
            all_weights=b["all_weights"][:global_tokens],
            local_indices=b["local_indices"][:local_tokens],
            local_weights=b["local_weights"][:local_tokens],
            local_latent=b["local_latent"][:local_tokens],
            dispatched=b["dispatched"][: e * mtp],
            intermediate=b["intermediate"][: e * mtp],
            expert_output=b["expert_output"][: e * mtp],
            combined=b["combined"][:global_tokens],
            local_combined=b["local_combined"][:local_tokens],
            expert_counts=b["expert_counts"],
            expert_counters=b["expert_counters"],
            topk_pos=b["topk_pos"][: global_tokens * self.top_k],
            expert_starts=expert_starts,
            s1_C_ptrs=s1_ptrs,
            s3_C_ptrs=s3_ptrs,
            s1_workspace=b["s1_workspace"],
            s3_workspace=b["s3_workspace"],
            local_tokens=local_tokens,
            global_tokens=global_tokens,
            max_tokens_padded=mtp,
            max_m_tiles=(min(global_tokens, mtp) + 15) // 16,
        )

    def get(self, bucket: int) -> _K3MoEGraphBuffers:
        self.setup()
        try:
            return self._views[int(bucket)]
        except KeyError as exc:
            raise KeyError(
                f"K3 MoE graph bucket {bucket} was not allocated; "
                f"available={self.bucket_sizes}"
            ) from exc

    def release(self) -> None:
        self._views.clear()
        self._base.clear()
        self._setup_done = False


class K3MoEGraphSegment:
    """Graph-capturable K3 latent resident-EP MoE for one layer."""

    def __init__(self, moe, resident, pool: K3MoEGraphBufferPool, *, device):
        self.moe = moe
        self.resident = resident
        self.pool = pool
        self.device = device
        self.hidden_size = int(moe.hidden_dim)
        self.latent_size = int(resident.shard.K_latent)
        self.intermediate_size = int(resident.shard.N)
        self.top_k = int(moe.top_k)
        self.world_size = int(resident.world_size)
        self.rank = int(resident.rank)
        self.expert_start = int(resident.expert_start)
        self.num_local_experts = int(resident.shard.num_local)
        self.tp_size = int(getattr(moe, "attn_tp_size", 1))
        self.tp_rank = int(getattr(moe, "attn_tp_rank", 0))
        self.tp_group = getattr(moe, "attn_tp_group", None)
        self.shared_stream = pool.shared_stream
        self.num_experts = int(moe.gate.weight.shape[0])
        # Set by ``setup_static_buffers`` once the fusion eligibility is known.
        self.fused_front = None
        self.fused_gate_kernel = False

        if self.tp_size <= 1 or self.tp_group is None:
            raise RuntimeError("K3 resident-MoE graph requires a TP8 process group")
        if self.hidden_size <= 0 or self.latent_size <= 0:
            raise ValueError("K3 resident-MoE graph received invalid model dimensions")
        if self.top_k != 16:
            raise ValueError(f"K3 resident-MoE graph expects top_k=16, got {self.top_k}")
        shared = getattr(moe, "shared_experts", None)
        if shared is None:
            raise ValueError("K3 resident-MoE graph requires shared experts")
        # The graph folds the shared expert's row-parallel reduction into the
        # routed TP all-reduce, so it needs the unreduced FFN body and the very
        # same TP group.  A streamed/offloaded expert wrapper exposes neither;
        # refuse capture instead of emitting a different collective pattern for
        # it, which would silently change a non-K3 serving path.
        if not callable(getattr(shared, "_ffn", None)):
            raise RuntimeError(
                "K3 resident-MoE graph requires an unwrapped shared expert "
                f"exposing _ffn, got {type(shared).__name__}"
            )
        if (
            int(getattr(shared, "_tp_size", 1)) != self.tp_size
            or getattr(shared, "_tp_group", None) is not self.tp_group
        ):
            raise RuntimeError(
                "K3 resident-MoE graph requires the shared expert to reduce "
                "over the MoE TP group"
            )
        if resident.comm is None:
            raise ValueError("K3 resident-MoE graph requires an EP communicator")
        if self.tp_size != self.pool.tp_size:
            raise ValueError(
                f"K3 MoE TP mismatch: layer={self.tp_size}, pool={self.pool.tp_size}"
            )
        if self.world_size != self.pool.world_size:
            raise ValueError(
                f"K3 MoE EP mismatch: layer={self.world_size}, pool={self.pool.world_size}"
            )
        if self.hidden_size != int(self.pool.hidden_size):
            raise ValueError(
                f"K3 MoE hidden mismatch: layer={self.hidden_size}, "
                f"pool={self.pool.hidden_size}"
            )

        # The resident layer owns these pointers and keeps every weight tensor
        # alive.  Do not copy any weight into the graph pool.

    def setup_static_buffers(self, bucket_size: int) -> None:
        self.pool.setup()
        if self.fused_front is None and _supports_mm_out_fp32(self.device):
            self.fused_front = fuse_router_and_down_proj(
                self.moe, self.resident.down_proj
            )
        self.fused_gate_kernel = fused_gate_kernel_eligible(
            self.moe.gate, fused_front=self.fused_front, device=self.device
        )
        if self.fused_gate_kernel:
            # The gate kernel writes ``latent_size`` columns starting at
            # ``num_experts``.  A slab whose latent half is a different width
            # would silently truncate rather than fail, so pin it here.
            fused_latent = int(self.fused_front.shape[0]) - self.num_experts
            if fused_latent != self.latent_size:
                raise ValueError(
                    f"K3 fused front latent width {fused_latent} does not match "
                    f"the resident latent size {self.latent_size}"
                )
        if hasattr(self.resident.comm, "disabled"):
            self.resident.comm.disabled = False

    def release_static_buffers(self, bucket_size: int) -> None:
        # The pool is shared by every layer; its owner releases it once when the
        # decode graph driver is torn down.
        return None

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            # ``padded`` is the full TP-group batch.  It is needed by the
            # shared expert, which is replicated over the group and performs
            # its own row-parallel all-reduce.
            "padded": TensorSpec(
                ("batch_size", self.hidden_size), torch.bfloat16
            ),
            # ``local`` is this TP rank's balanced slice of the real group
            # rows, copied into the first rows of a fixed-size buffer.  Keeping
            # a second input avoids deriving a non-divisible balanced split
            # from a graph-time constant bucket.
            "local": TensorSpec(
                ("batch_size", self.hidden_size), torch.bfloat16
            ),
            "num_valid_tokens": TensorSpec((1,), torch.int32),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "moe_output": TensorSpec(
                ("batch_size", self.hidden_size), torch.bfloat16
            )
        }

    def _combine_fp32(self, bufs: _K3MoEGraphBuffers) -> None:
        """Combine K=16 routes into the preallocated FP32 graph buffer."""
        reduce_weighted_scatter_fp32(
            bufs.expert_output,
            bufs.topk_pos,
            bufs.all_weights,
            bufs.global_tokens,
            self.latent_size,
            self.top_k,
            bufs.combined,
        )

    def forward(
        self,
        *,
        padded: torch.Tensor,
        local: torch.Tensor,
        num_valid_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bucket = int(padded.shape[0])
        bufs = self.pool.get(bucket)
        local_tokens = bufs.local_tokens
        global_tokens = bufs.global_tokens
        local = local[:local_tokens]

        # The shared expert depends only on the replicated TP-group input.  It
        # is independent of routing, so launch it before the current stream's
        # router/down projection rather than after it.  This matches the
        # overlap schedule used by SGLang: the current stream can then run the
        # entire routed front and EP path while the auxiliary stream computes
        # the shared partial.  The wait below still keeps the merged TP
        # reduction ordered after the shared result.
        shared_partial = None
        current_stream = None
        if self.shared_stream is not None:
            current_stream = torch.cuda.current_stream(device=self.device)
            self.shared_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.shared_stream):
                shared_partial = self.moe.shared_experts._ffn(padded)

        # Router/down-projection are executed once per distinct local row.  The
        # graph input is already zero padded to the full TP-group bucket.  Both
        # read the same rows from the same activation, so when their weights are
        # fused into one slab a single BF16-in/FP32-out GEMM replaces the two
        # skinny launches; the router half stays FP32 for sigmoid/top-k.  The
        # BF16-input GEMM can differ from the eager FP32-input GEMM on near-tie
        # logits, so graph-vs-eager routing parity remains a remote gate.
        x_latent = None
        gate_out = None
        topk_idx = None
        if self.fused_front is not None:
            fused_out = torch.mm(
                local, self.fused_front.t(), out_dtype=torch.float32
            )
            if self.fused_gate_kernel:
                # One kernel does sigmoid + top-16 + renormalize + scale and
                # writes straight into the int32/FP32 EP gather sources.  It
                # reads the router half in place (a strided view of the fused
                # GEMM output) and applies the balanced-split padding mask from
                # the device-resident scalar, so the graph carries no dtype cast
                # and no ``torch.where`` after the gate.  The same kernel also
                # casts the latent half of those rows into the preallocated BF16
                # gather source, replacing the separate strided copy; padding
                # rows come out zero exactly as the zero-input GEMM made them.
                gate = self.moe.gate
                gate_sigmoid_topk_cuda(
                    fused_out[:, : self.num_experts],
                    gate.e_score_correction_bias,
                    k=self.top_k,
                    routed_scaling_factor=float(gate.routed_scaling_factor),
                    topk_indices=bufs.local_indices,
                    topk_weights=bufs.local_weights,
                    num_valid_tokens=num_valid_tokens,
                    latent_out=bufs.local_latent,
                    latent_offset=self.num_experts,
                )
                topk_idx = bufs.local_indices
                topk_weight = bufs.local_weights
                x_latent = bufs.local_latent
            else:
                router_logits, latent = split_fused_front(fused_out, self.num_experts)
                x_latent = latent.contiguous()
                gate_out = self.moe.gate.select_experts(router_logits)
        else:
            gate_out = self.moe.gate(local.view(local_tokens, 1, self.hidden_size))

        if topk_idx is None:
            # The graph always executes the fixed local-token shape.  Mask the
            # balanced-split padding before the EP all-gathers so zero rows do
            # not turn into real routed assignments (the router's zero-input
            # bias can otherwise select experts and waste grouped-GEMM work).
            # The scalar is a graph input refreshed once per replay and may
            # differ by TP rank.
            valid_rows = torch.arange(
                local_tokens, dtype=torch.int32, device=local.device
            ) < num_valid_tokens.reshape(1)
            # KimiMoEGate flattens its input internally and returns [T, K].
            # Keep this explicit because a wrapper may preserve the singleton
            # sequence dimension; the static EP gather buffers are always
            # two-dimensional.
            topk_idx = gate_out[0].reshape(local_tokens, -1).to(torch.int32)
            topk_weight = gate_out[1].reshape(local_tokens, -1)
            topk_idx = torch.where(
                valid_rows.unsqueeze(1), topk_idx, torch.full_like(topk_idx, -1)
            )
            topk_weight = torch.where(
                valid_rows.unsqueeze(1), topk_weight, torch.zeros_like(topk_weight)
            )

        if x_latent is None:
            x_latent = self.resident.down_proj(local).reshape(
                local_tokens, self.latent_size
            ).contiguous()

        # EP collectives are graph-captured on the current decode stream.  The
        # order is identical to ResidentEPMXFP4MoELayer._forward_ep.  The three
        # gathers are submitted inside one NCCL group so they are launched as a
        # single collective batch instead of three: it removes two per-collective
        # launch/handshake costs while each tensor keeps its own native buffer
        # and layout, so no packing or extra copy is introduced.
        with self.resident.comm.change_state(enable=True):
            self.resident.comm.group_start()
            self.resident.comm.all_gather(bufs.all_latent, x_latent)
            self.resident.comm.all_gather(bufs.all_indices, topk_idx)
            self.resident.comm.all_gather(bufs.all_weights, topk_weight)
            self.resident.comm.group_end()

        expert_counts, topk_pos = dispatch_scatter_3d(
            bufs.all_latent,
            bufs.all_indices,
            bufs.dispatched,
            self.expert_start,
            self.num_local_experts,
            bufs.max_tokens_padded,
            bufs.expert_counts,
            bufs.expert_counters,
            bufs.topk_pos,
        )

        # The resident layer's weight pointers and K3 SiTU kernel are reused;
        # only the activation/metadata storage is graph-owned and static.
        shard = self.resident.shard
        marlin_grouped_stage1_fused_mxfp4_situ(
            bufs.dispatched,
            bufs.intermediate,
            expert_counts,
            bufs.expert_starts,
            shard.gate_B_ptrs,
            shard.gate_scales_ptrs,
            shard.up_B_ptrs,
            shard.up_scales_ptrs,
            bufs.s1_C_ptrs,
            self.intermediate_size,
            self.latent_size,
            bufs.s1_workspace,
            bufs.max_m_tiles,
            bufs.max_tokens_padded,
            self.num_local_experts,
            global_tokens,
        )
        marlin_grouped_m16_mxfp4(
            bufs.intermediate,
            shard.down_B_ptrs,
            bufs.s3_C_ptrs,
            shard.down_scales_ptrs,
            bufs.expert_starts,
            expert_counts,
            self.num_local_experts,
            self.latent_size,
            self.intermediate_size,
            bufs.s3_workspace,
            self.num_local_experts,
            self.latent_size // 256,
            bufs.max_m_tiles,
        )
        self._combine_fp32(bufs)

        # ``combined`` is rank-major because the EP all-gathers above are
        # rank-major.  Reduce-scatter therefore returns exactly this rank's
        # contiguous local chunk, replacing the old all-reduce + rank slice.
        # It cuts the EP reduction traffic roughly in half and removes the
        # explicit slice/copy from the critical path.
        with self.resident.comm.change_state(enable=True):
            self.resident.comm.reduce_scatter(
                bufs.local_combined,
                bufs.combined,
                op=dist.ReduceOp.SUM,
            )

        local_latent = bufs.local_combined.to(torch.bfloat16)
        if self.resident.norm is not None:
            local_latent = self.resident.norm(local_latent)
        local_output = self.resident.up_proj(local_latent)

        if shared_partial is None:
            # CPU contract path, and the defensive fallback for a non-CUDA
            # device, run the same body synchronously on the current stream.
            shared_partial = self.moe.shared_experts._ffn(padded)
        else:
            # ``_ffn`` carries no collective, so this only orders the auxiliary
            # stream's local GEMMs before the fold and the TP reduction below.
            current_stream.wait_stream(self.shared_stream)

        # One TP collective instead of two.  Each TP rank owns the contiguous
        # rank-major slice ``[tp_rank * local_tokens, ...)`` of the group batch
        # — exactly the layout NCCL's all-gather produced — so adding this
        # rank's routed rows into its own shared partial makes a single
        # all-reduce deliver both sums: the shared partials add up to the full
        # replicated shared output, and every routed slice is contributed by
        # exactly one rank.  This replaces the separate routed all-gather and
        # the shared expert's own TP all-reduce.
        start = self.tp_rank * local_tokens
        shared_partial[start : start + local_tokens].add_(local_output)
        with torch.cuda.device(self.device):
            dist.all_reduce(shared_partial, group=self.tp_group)
        return {"moe_output": shared_partial}


__all__ = [
    "K3MoEGraphBufferPool",
    "K3MoEGraphSegment",
    "fuse_router_and_down_proj",
    "fused_gate_kernel_eligible",
    "k3_moe_graph_buckets",
    "split_fused_front",
]
