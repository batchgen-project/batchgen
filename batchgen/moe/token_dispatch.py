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

"""Token dispatch/combine abstraction for MoE layers.

This module provides abstractions for token dispatch (gather) and combine (scatter)
operations in MoE layers. It supports both local execution (EP=1) and distributed
execution (EP>1) with multiple communication backends.

Usage:
    # For single GPU (EP=1):
    dispatcher = LocalTokenDispatcher()

    # For multi-GPU (EP>1):
    dispatcher = DistributedTokenDispatcher(
        world_size=8, rank=0, num_experts=128,
        backend="pplx"  # or "nccl", "allgather"
    )

    # In MoE forward:
    dispatch_result = dispatcher.dispatch(x, topk_indices, topk_weights)
    # ... expert computation on dispatch_result.dispatched_x ...
    output = dispatcher.combine(expert_output, dispatch_result, num_tokens)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DispatchResult:
    """Result of token dispatch operation.

    Attributes:
        dispatched_x: Tokens reordered/gathered for expert computation.
            Shape: [total_routed_tokens, hidden_dim] where
            total_routed_tokens = num_tokens * experts_per_token
        expert_ids: Which expert each dispatched token belongs to.
            Shape: [total_routed_tokens]
        original_indices: Original token positions before dispatch.
            Shape: [total_routed_tokens]
        topk_positions: Position within top-k selection (0..k-1) for each dispatched token.
            Shape: [total_routed_tokens]
        expert_counts: Number of tokens assigned to each expert.
            Shape: [num_experts]
        expert_offsets: Cumulative offsets for each expert in the dispatched tensor.
            Shape: [num_experts]
        routing_weights: Softmax routing weights from top-k selection.
            Shape: [num_tokens, k]
    """
    dispatched_x: torch.Tensor
    expert_ids: torch.Tensor
    original_indices: torch.Tensor
    topk_positions: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    routing_weights: torch.Tensor


class TokenDispatcher(ABC):
    """Abstract base class for token dispatch/combine operations.

    Token dispatch gathers tokens by expert assignment for efficient batched
    computation. Token combine scatters expert outputs back to original positions
    with weighted summation.
    """

    @abstractmethod
    def dispatch(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchResult:
        """Dispatch tokens to experts (gather operation).

        Args:
            x: Input tokens. Shape: [num_tokens, hidden_dim]
            topk_indices: Expert indices from top-k selection. Shape: [num_tokens, k]
            topk_weights: Routing weights from softmax. Shape: [num_tokens, k]

        Returns:
            DispatchResult containing dispatched tokens and metadata for combine.
        """
        pass

    @abstractmethod
    def combine(
        self,
        expert_output: torch.Tensor,
        dispatch_result: DispatchResult,
        num_tokens: int,
    ) -> torch.Tensor:
        """Combine expert outputs back to original token order (scatter operation).

        Args:
            expert_output: Output from expert computation.
                Shape: [total_routed_tokens, output_dim]
            dispatch_result: Result from dispatch() containing scatter metadata.
            num_tokens: Original number of tokens.

        Returns:
            Combined output with weighted sum from all selected experts.
            Shape: [num_tokens, output_dim]
        """
        pass


class LocalTokenDispatcher(TokenDispatcher):
    """Local dispatcher for EP=1 (single GPU, no inter-GPU communication).

    Tokens are gathered by sorting by expert assignment, creating contiguous
    groups for efficient batched GEMM. Combine scatters results back with
    weighted summation.
    """

    def dispatch(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchResult:
        """Local gather - sort tokens by expert assignment.

        Creates contiguous groups of tokens for each expert by sorting.

        Args:
            x: Input tokens. Shape: [num_tokens, hidden_dim]
            topk_indices: Expert indices. Shape: [num_tokens, k]
            topk_weights: Routing weights. Shape: [num_tokens, k]

        Returns:
            DispatchResult with tokens sorted by expert ID.
        """
        num_tokens, k = topk_indices.shape
        num_experts = topk_indices.max().item() + 1
        device = x.device

        # Flatten to [num_tokens * k]
        flat_indices = topk_indices.view(-1)

        # Sort by expert ID for contiguous groups
        sorted_order = torch.argsort(flat_indices, stable=True)
        expert_ids = flat_indices[sorted_order]

        # Compute expert counts and offsets
        expert_counts = torch.bincount(expert_ids, minlength=num_experts).to(torch.int32)
        expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
        expert_offsets[1:] = torch.cumsum(expert_counts[:-1], dim=0)

        # Map sorted positions back to original tokens
        token_ids = sorted_order // k
        topk_positions = sorted_order % k

        # Gather input tokens
        dispatched_x = x[token_ids]

        return DispatchResult(
            dispatched_x=dispatched_x,
            expert_ids=expert_ids,
            original_indices=token_ids,
            topk_positions=topk_positions,
            expert_counts=expert_counts,
            expert_offsets=expert_offsets,
            routing_weights=topk_weights,
        )

    def combine(
        self,
        expert_output: torch.Tensor,
        dispatch_result: DispatchResult,
        num_tokens: int,
    ) -> torch.Tensor:
        """Local scatter - weighted sum back to original positions.

        Args:
            expert_output: Expert computation results. Shape: [total_routed_tokens, output_dim]
            dispatch_result: Dispatch metadata.
            num_tokens: Original token count.

        Returns:
            Combined output. Shape: [num_tokens, output_dim]
        """
        output_dim = expert_output.shape[-1]
        device = expert_output.device
        dtype = expert_output.dtype

        result = torch.zeros(num_tokens, output_dim, dtype=dtype, device=device)

        # Get weights for each routed token
        weights = dispatch_result.routing_weights[
            dispatch_result.original_indices,
            dispatch_result.topk_positions
        ].unsqueeze(-1)

        # Scatter-add weighted outputs
        result.index_add_(0, dispatch_result.original_indices, expert_output * weights)
        return result


class DistributedTokenDispatcher(TokenDispatcher):
    """Distributed dispatcher for EP>1 (multi-GPU with communication).

    Supports multiple communication backends:
    - "pplx": PPLX All-to-All (FP8 optimized, recommended for Hopper)
    - "nccl": NCCL all_to_all_single (standard distributed)
    - "allgather": AllGather + AllReduce (simple fallback)

    Expert parallelism distributes experts across GPUs. Each GPU owns a subset
    of experts (experts_per_rank = num_experts // world_size). Tokens must be
    communicated to the GPU owning each selected expert.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        num_experts: int,
        backend: str = "pplx",
        device: Optional[torch.device] = None,
    ):
        """Initialize distributed dispatcher.

        Args:
            world_size: Total number of GPUs in expert parallel group.
            rank: This GPU's rank in the expert parallel group.
            num_experts: Total number of experts across all GPUs.
            backend: Communication backend ("pplx", "nccl", "allgather").
            device: CUDA device to use (default: current device).
        """
        assert num_experts % world_size == 0, \
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"

        self.world_size = world_size
        self.rank = rank
        self.num_experts = num_experts
        self.experts_per_rank = num_experts // world_size
        self.backend = backend
        self.device = device if device is not None else torch.cuda.current_device()

        # Local experts owned by this rank
        self.local_expert_start = rank * self.experts_per_rank
        self.local_expert_end = (rank + 1) * self.experts_per_rank

        # Initialize backend
        self._init_backend()

    def _init_backend(self):
        """Initialize the communication backend."""
        if self.backend == "pplx":
            self._init_pplx()
        elif self.backend == "nccl":
            self._init_nccl()
        elif self.backend == "allgather":
            self._init_allgather()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_pplx(self):
        """Initialize PPLX All-to-All backend."""
        try:
            from pplx_kernels import AllToAll
            # PPLX initialization would go here
            # self.ata = AllToAll(...)
            self._pplx_available = True
        except ImportError:
            raise ImportError(
                "PPLX kernels not available. Install pplx_kernels or use 'nccl' backend."
            )

    def _init_nccl(self):
        """Initialize NCCL backend."""
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before using NCCL backend"
            )
        self._dist = dist

    def _init_allgather(self):
        """Initialize AllGather+AllReduce fallback backend."""
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before using allgather backend"
            )
        self._dist = dist

    def dispatch(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchResult:
        """Distributed gather via all-to-all communication.

        Sends tokens to the GPU that owns each selected expert.
        """
        if self.backend == "pplx":
            return self._dispatch_pplx(x, topk_indices, topk_weights)
        elif self.backend == "nccl":
            return self._dispatch_nccl(x, topk_indices, topk_weights)
        else:
            return self._dispatch_allgather(x, topk_indices, topk_weights)

    def combine(
        self,
        expert_output: torch.Tensor,
        dispatch_result: DispatchResult,
        num_tokens: int,
    ) -> torch.Tensor:
        """Distributed scatter via all-to-all communication.

        Receives expert outputs and combines them back to original positions.
        """
        if self.backend == "pplx":
            return self._combine_pplx(expert_output, dispatch_result, num_tokens)
        elif self.backend == "nccl":
            return self._combine_nccl(expert_output, dispatch_result, num_tokens)
        else:
            return self._combine_allgather(expert_output, dispatch_result, num_tokens)

    def _dispatch_pplx(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchResult:
        """PPLX All-to-All dispatch with FP8 quantization."""
        # TODO: Implement PPLX dispatch
        # 1. Quantize input to FP8 for efficient communication
        # 2. Compute send counts per rank based on expert assignments
        # 3. Call PPLX all-to-all dispatch
        # 4. Return DispatchResult with received tokens
        raise NotImplementedError("PPLX dispatch not yet implemented")

    def _combine_pplx(
        self,
        expert_output: torch.Tensor,
        dispatch_result: DispatchResult,
        num_tokens: int,
    ) -> torch.Tensor:
        """PPLX All-to-All combine with FP8 quantization."""
        # TODO: Implement PPLX combine
        # 1. Quantize output for communication
        # 2. Call PPLX all-to-all combine with routing weights
        # 3. Dequantize and return combined output
        raise NotImplementedError("PPLX combine not yet implemented")

    def _dispatch_nccl(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchResult:
        """NCCL all_to_all_single dispatch."""
        num_tokens, k = topk_indices.shape
        hidden_dim = x.shape[-1]

        # Determine which rank each expert belongs to
        expert_to_rank = topk_indices // self.experts_per_rank  # [num_tokens, k]

        # Count tokens to send to each rank
        send_counts = torch.zeros(self.world_size, dtype=torch.int64, device=x.device)
        for r in range(self.world_size):
            send_counts[r] = (expert_to_rank == r).sum()

        # Gather all send counts to compute receive counts
        recv_counts = torch.zeros_like(send_counts)
        self._dist.all_to_all_single(recv_counts, send_counts)

        # Sort tokens by destination rank, then by expert within rank
        flat_indices = topk_indices.view(-1)
        flat_ranks = expert_to_rank.view(-1)

        # Sort by (rank, expert_id) for contiguous sends
        sort_keys = flat_ranks * self.num_experts + flat_indices
        sorted_order = torch.argsort(sort_keys, stable=True)

        token_ids = sorted_order // k
        topk_positions = sorted_order % k

        # Prepare send buffer (sorted by destination)
        send_x = x[token_ids]  # [num_tokens * k, hidden_dim]

        # All-to-all communication
        total_recv = recv_counts.sum().item()
        recv_x = torch.empty(total_recv, hidden_dim, dtype=x.dtype, device=x.device)

        self._dist.all_to_all_single(
            recv_x, send_x,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

        # Compute local expert counts and offsets
        recv_expert_ids = flat_indices[sorted_order]
        # Filter to local experts only
        local_mask = (recv_expert_ids >= self.local_expert_start) & \
                     (recv_expert_ids < self.local_expert_end)
        local_expert_ids = recv_expert_ids[local_mask] - self.local_expert_start

        expert_counts = torch.bincount(
            local_expert_ids, minlength=self.experts_per_rank
        ).to(torch.int32)
        expert_offsets = torch.zeros(self.experts_per_rank, dtype=torch.int32, device=x.device)
        expert_offsets[1:] = torch.cumsum(expert_counts[:-1], dim=0)

        return DispatchResult(
            dispatched_x=recv_x,
            expert_ids=local_expert_ids,
            original_indices=token_ids,
            topk_positions=topk_positions,
            expert_counts=expert_counts,
            expert_offsets=expert_offsets,
            routing_weights=topk_weights,
        )

    def _combine_nccl(
        self,
        expert_output: torch.Tensor,
        dispatch_result: DispatchResult,
        num_tokens: int,
    ) -> torch.Tensor:
        """NCCL all_to_all_single combine."""
        # TODO: Implement reverse all-to-all and weighted summation
        # 1. Send expert outputs back to originating ranks
        # 2. Apply routing weights
        # 3. Sum contributions from all selected experts
        raise NotImplementedError("NCCL combine not yet implemented")

    def _dispatch_allgather(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchResult:
        """AllGather-based dispatch (simple but less efficient)."""
        num_tokens, k = topk_indices.shape
        hidden_dim = x.shape[-1]

        # AllGather inputs from all ranks
        all_x_list = [torch.empty_like(x) for _ in range(self.world_size)]
        self._dist.all_gather(all_x_list, x)
        all_x = torch.cat(all_x_list, dim=0)  # [total_tokens, hidden_dim]

        # AllGather routing info
        all_indices_list = [torch.empty_like(topk_indices) for _ in range(self.world_size)]
        self._dist.all_gather(all_indices_list, topk_indices)
        all_indices = torch.cat(all_indices_list, dim=0)  # [total_tokens, k]

        all_weights_list = [torch.empty_like(topk_weights) for _ in range(self.world_size)]
        self._dist.all_gather(all_weights_list, topk_weights)
        all_weights = torch.cat(all_weights_list, dim=0)  # [total_tokens, k]

        # Filter to tokens assigned to local experts
        total_tokens = all_x.shape[0]
        flat_indices = all_indices.view(-1)

        local_mask = (flat_indices >= self.local_expert_start) & \
                     (flat_indices < self.local_expert_end)

        local_flat_indices = flat_indices[local_mask]
        local_token_ids = torch.arange(total_tokens * k, device=x.device)[local_mask] // k
        local_topk_positions = torch.arange(total_tokens * k, device=x.device)[local_mask] % k

        dispatched_x = all_x[local_token_ids]
        local_expert_ids = local_flat_indices - self.local_expert_start

        # Sort by expert for contiguous groups
        sorted_order = torch.argsort(local_expert_ids, stable=True)
        dispatched_x = dispatched_x[sorted_order]
        local_expert_ids = local_expert_ids[sorted_order]
        local_token_ids = local_token_ids[sorted_order]
        local_topk_positions = local_topk_positions[sorted_order]

        expert_counts = torch.bincount(
            local_expert_ids, minlength=self.experts_per_rank
        ).to(torch.int32)
        expert_offsets = torch.zeros(self.experts_per_rank, dtype=torch.int32, device=x.device)
        expert_offsets[1:] = torch.cumsum(expert_counts[:-1], dim=0)

        return DispatchResult(
            dispatched_x=dispatched_x,
            expert_ids=local_expert_ids,
            original_indices=local_token_ids,
            topk_positions=local_topk_positions,
            expert_counts=expert_counts,
            expert_offsets=expert_offsets,
            routing_weights=all_weights,
        )

    def _combine_allgather(
        self,
        expert_output: torch.Tensor,
        dispatch_result: DispatchResult,
        num_tokens: int,
    ) -> torch.Tensor:
        """AllReduce-based combine (simple but less efficient)."""
        output_dim = expert_output.shape[-1]

        # Each rank computes partial output for its local experts
        # Then AllReduce to sum contributions
        partial_output = torch.zeros(
            num_tokens * self.world_size, output_dim,
            dtype=expert_output.dtype, device=expert_output.device
        )

        # Scatter local expert outputs to global positions with weights
        weights = dispatch_result.routing_weights[
            dispatch_result.original_indices,
            dispatch_result.topk_positions
        ].unsqueeze(-1)

        partial_output.index_add_(
            0, dispatch_result.original_indices,
            expert_output * weights
        )

        # AllReduce to sum across all ranks
        self._dist.all_reduce(partial_output)

        # Extract this rank's portion
        start = self.rank * num_tokens
        end = (self.rank + 1) * num_tokens
        return partial_output[start:end]


def create_token_dispatcher(
    world_size: int = 1,
    rank: int = 0,
    num_experts: int = 128,
    backend: str = "auto",
    device: Optional[torch.device] = None,
) -> TokenDispatcher:
    """Factory function to create appropriate token dispatcher.

    Args:
        world_size: Number of GPUs in expert parallel group.
        rank: This GPU's rank.
        num_experts: Total number of experts.
        backend: Communication backend ("auto", "pplx", "nccl", "allgather").
            "auto" selects PPLX if available, else NCCL.
        device: CUDA device.

    Returns:
        LocalTokenDispatcher for EP=1, DistributedTokenDispatcher for EP>1.
    """
    if world_size == 1:
        return LocalTokenDispatcher()

    if backend == "auto":
        try:
            import pplx_kernels
            backend = "pplx"
        except ImportError:
            backend = "nccl"

    return DistributedTokenDispatcher(
        world_size=world_size,
        rank=rank,
        num_experts=num_experts,
        backend=backend,
        device=device,
    )
