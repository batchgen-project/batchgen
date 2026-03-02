# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

"""MoE base abstraction: MoEBase + MoEBufferManager.

Template method pattern for MoE forward across all model families.
Zero CPU-GPU sync on the all-persistent WGMMA path (default for GLM-5).

Class hierarchy:
    MoEBufferManager          (one per model, shared across all MoE layers)
    MoEBase(nn.Module)        (abstract base, template method pattern)
      +-- Glm5FP8MoE          (FP8 grouped WGMMA, sigmoid gate)
      +-- KimiK25MoE          (INT4 grouped WGMMA, sigmoid gate)  [future]

Reference:
    batchgen_design/moe_module_template.md
    batchgen_design/moe_forward_pipeline_design.md
    K2.5 model.py KimiK25MoE._forward_decode()
"""

import logging
from abc import abstractmethod
from typing import ClassVar, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBufferManager:
    """Pre-allocated buffers for MoE decode forward. Shared across all layers.

    One instance per model, stored as class variable on MoEBase. All MoE layers
    (75 for GLM-5, 60 for K2.5) share a single MoEBufferManager.

    Semantics: pure buffer management — no model logic, no weights.
    Buffers are sized once at init, reused every forward.
    """

    def __init__(
        self,
        num_tokens_per_rank: int,
        world_size: int,
        hidden_size: int,
        device: torch.device,
    ):
        self.num_tokens_per_rank = num_tokens_per_rank
        self.world_size = world_size
        self.hidden_size = hidden_size
        self.device = device

        G = num_tokens_per_rank * world_size

        # Communication buffers (pre-allocated, zero-copy)
        self.all_tokens = torch.zeros(G, hidden_size, dtype=torch.bfloat16, device=device)
        self.padded_hidden_states = torch.zeros(
            num_tokens_per_rank, hidden_size, dtype=torch.bfloat16, device=device
        )

    def resize_if_needed(self, num_tokens_per_rank: int):
        """Resize comm buffers if num_tokens_per_rank changed."""
        if num_tokens_per_rank <= self.num_tokens_per_rank:
            return
        self.num_tokens_per_rank = num_tokens_per_rank
        G = num_tokens_per_rank * self.world_size
        self.all_tokens = torch.zeros(
            G, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        self.padded_hidden_states = torch.zeros(
            num_tokens_per_rank, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )


class MoEBase(nn.Module):
    """Abstract base class for MoE layers with template method pattern.

    Provides a single _forward_decode() that handles:
        AllGather → Gate → Dispatch → Expert Compute → Reduce → AllReduce

    Subclasses implement:
        gate()                  — model-specific gating (sigmoid/softmax, CUDA kernel)
        expert_compute()        — model-specific compute (FP8 WGMMA / INT4 WGMMA / loop)
        shared_expert_forward() — shared expert FFN (default: zeros)
    """

    # Shared across ALL MoE layer instances (one per model)
    _buffer_mgr: ClassVar[Optional[MoEBufferManager]] = None

    def __init__(self, config, comm=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.comm = comm

        import torch.distributed as dist
        if not dist.is_initialized():
            self.rank, self.world_size = 0, 1
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.experts_per_rank = config.n_routed_experts // self.world_size
        self.total_experts = self.world_size * self.experts_per_rank
        self.routed_expert_start_idx = self.rank * self.experts_per_rank
        self.routed_expert_end_idx = (self.rank + 1) * self.experts_per_rank

        self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
        self.num_tokens_per_rank = None
        self.enable_ep_offloading = False
        self.num_persistent_local_experts = self.experts_per_rank  # default: all persistent

    # ── Buffer management ──

    @classmethod
    def init_buffer_manager(cls, num_tokens_per_rank: int, world_size: int,
                            hidden_size: int, device: torch.device):
        """Initialize shared buffer manager. Called once by PSM."""
        cls._buffer_mgr = MoEBufferManager(
            num_tokens_per_rank, world_size, hidden_size, device,
        )

    def init_num_tokens(self, num_tokens_per_rank: int):
        self.num_tokens_per_rank = num_tokens_per_rank

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        self.num_tokens_per_rank = num_tokens_per_rank
        if self._buffer_mgr is not None:
            self._buffer_mgr.resize_if_needed(num_tokens_per_rank)

    # ── Abstract methods (model-specific) ──

    @abstractmethod
    def gate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gating function.

        Args:
            x: [G, H] BF16 — global tokens (all-gathered)

        Returns:
            topk_idx:    [G, K] INT32 — expert indices
            topk_weight: [G, K] FP32  — routing weights
        """
        ...

    @abstractmethod
    def expert_compute_persistent(
        self, global_x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """All-persistent expert compute. Zero CPU-GPU sync.

        Args:
            global_x:    [G, H] BF16 — all-gathered input
            topk_idx:    [G, K] INT32 — expert indices
            topk_weight: [G, K] FP32  — routing weights

        Returns:
            global_results: [G, H] BF16 — weighted expert output
        """
        ...

    @abstractmethod
    def expert_compute_mixed(
        self, global_x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Mixed persistent + non-persistent expert compute.

        One .tolist() sync is acceptable here (opt-in path, non-default).

        Same signature and return as expert_compute_persistent.
        """
        ...

    def shared_expert_forward(self, identity: torch.Tensor) -> torch.Tensor:
        """Shared expert FFN. Override in subclass. Default: zeros."""
        return torch.zeros_like(identity)

    # ── Concrete template ──

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, 'phase', 'decode') == 'decode':
            return self._forward_decode(hidden_states)
        return self._forward_prefill(hidden_states)

    @torch.inference_mode()
    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """EP decode: AllGather → Gate → Expert Compute → AllReduce → Extract + Shared.

        Zero CPU-GPU sync on all-persistent path (default).
        One .tolist() sync on mixed path (opt-in via enable_ep_offloading).
        """
        import torch.distributed as dist
        from contextlib import nullcontext as _nullctx
        from batchgen.timing import get_decode_timer

        dt = get_decode_timer()

        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states
        num_tokens, hidden_size = hidden_states.shape

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        buf = self._buffer_mgr
        ntp = self.num_tokens_per_rank

        # 1) AllGather
        with (dt.timed("allgather", 0) if dt else _nullctx()):
            G = self.world_size * ntp
            all_tokens = buf.all_tokens[:G] if buf is not None else torch.zeros(
                G, hidden_size, device=self.device, dtype=torch.bfloat16)

            if buf is not None:
                all_tokens.zero_()
                padded = buf.padded_hidden_states[:ntp]
                padded.zero_()
                padded[:num_tokens] = hidden_states
            else:
                padded = torch.zeros(ntp, hidden_size, device=self.device, dtype=hidden_states.dtype)
                padded[:num_tokens] = hidden_states

            with self.comm.change_state(enable=True):
                self.comm.all_gather(
                    all_tokens, padded,
                    stream=torch.cuda.default_stream(self.device),
                )

        # 2) Gate
        with (dt.timed("routing", 0) if dt else _nullctx()):
            global_x = all_tokens
            topk_idx, topk_weight = self.gate(global_x)

        # 3+4) Expert compute (dispatch + GEMM + reduce)
        all_persistent = (self.num_persistent_local_experts == self.experts_per_rank)
        if all_persistent and not self.enable_ep_offloading:
            with (dt.timed("wgmma_pipeline", 0) if dt else _nullctx()):
                global_results = self.expert_compute_persistent(
                    global_x, topk_idx, topk_weight)
        else:
            with (dt.timed("mixed_compute", 0) if dt else _nullctx()):
                global_results = self.expert_compute_mixed(
                    global_x, topk_idx, topk_weight)

        # 5) AllReduce
        with (dt.timed("allreduce", 0) if dt else _nullctx()):
            with self.comm.change_state(enable=True):
                self.comm.all_reduce(
                    global_results, op=dist.ReduceOp.SUM,
                    stream=torch.cuda.default_stream(self.device),
                )

        # 6) Extract local slice + shared expert
        start = self.rank * ntp
        out = global_results[start:start + num_tokens].to(hidden_states.dtype)
        out = out + self.shared_expert_forward(identity)
        return out.view(*orig_shape)

    @torch.inference_mode()
    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prefill: per-expert loop (no EP). Override in subclass if needed."""
        raise NotImplementedError("Subclass must implement _forward_prefill for prefill mode")
