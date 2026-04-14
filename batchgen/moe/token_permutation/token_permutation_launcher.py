import torch
import torch.nn as nn
import warnings

import batchgen_kernels.moe._C_fused_moe_token_permutation as _fused_moe_cuda


def get_fused_moe_cuda():
    """Get the AOT-compiled CUDA extension."""
    return _fused_moe_cuda


class FusedMoETokenPermutation:
    """
    Fused MoE token permutation that replaces the manual expand, sort, permute pipeline.

    This class provides both optimized CUDA and fallback PyTorch implementations.

    Supported data types:
    - torch.float32 (fp32)
    - torch.float16 (fp16)
    - torch.bfloat16 (bf16)
    """

    def __init__(self, use_cuda_if_available=True):
        self.use_cuda = use_cuda_if_available
        self.cuda_ext = _fused_moe_cuda if use_cuda_if_available else None

    def __call__(self,
                 global_x: torch.Tensor,
                 topk_idx: torch.Tensor,
                 token_idx: torch.Tensor,
                 topk_pos: torch.Tensor,
                 routed_expert_start_idx: int,
                 routed_expert_end_idx: int):
        """
        Args:
            global_x: [num_tokens, hidden_size] - Hidden states from all ranks
            topk_idx: [num_tokens, K] - Expert assignments for each token
            token_idx: [num_tokens * K] - Flattened token indices (for compatibility)
            topk_pos: [num_tokens * K] - Flattened topk positions (for compatibility)
            routed_expert_start_idx: Start index of local experts
            routed_expert_end_idx: End index of local experts

        Returns:
            tuple: (input_x, input_eids, global_indices, token_topk_pos)
                - input_x: [local_tokens, hidden_size] - Tokens for local experts
                - input_eids: [local_tokens] - Expert IDs for each token
                - global_indices: [local_tokens] - Original token indices
                - token_topk_pos: [local_tokens] - TopK positions
        """

        # Use CUDA implementation if available
        if self.cuda_ext is not None and global_x.is_cuda:
            return self._fused_cuda_permutation(
                global_x, topk_idx, token_idx, topk_pos,
                routed_expert_start_idx, routed_expert_end_idx
            )
        else:
            # Fallback to original PyTorch implementation
            return self._pytorch_fallback(
                global_x, topk_idx, token_idx, topk_pos,
                routed_expert_start_idx, routed_expert_end_idx
            )

    def _fused_cuda_permutation(self, global_x, topk_idx, token_idx, topk_pos,
                           routed_expert_start_idx, routed_expert_end_idx):
        """Optimized CUDA implementation."""

        # Ensure contiguous tensors
        global_x = global_x.contiguous()
        topk_idx = topk_idx.contiguous()

        # Call the CUDA extension
        results = self.cuda_ext.fused_moe_token_permutation(
            global_x,
            topk_idx,
            token_idx,  # Not used in CUDA version, but kept for interface compatibility
            topk_pos,   # Not used in CUDA version, but kept for interface compatibility
            routed_expert_start_idx,
            routed_expert_end_idx
        )

        input_x, input_eids, global_indices, token_topk_pos, expert_counts = results

        return input_x, input_eids, global_indices, token_topk_pos

    def _pytorch_fallback(self, global_x, topk_idx, token_idx, topk_pos,
                         routed_expert_start_idx, routed_expert_end_idx):
        """Original PyTorch implementation as fallback."""

        K = topk_idx.size(1)

        # Flatten and expand (original implementation)
        flat_eids = topk_idx.flatten()
        expanded_x = global_x.repeat_interleave(K, dim=0)

        # Sort by expert ID
        sorted_eids, sort_idx = flat_eids.sort()
        sorted_x = expanded_x[sort_idx]
        sorted_tok = token_idx[sort_idx]
        sorted_pos = topk_pos[sort_idx]

        # Filter for local experts
        local_token_expanded_x_indices = (
            (sorted_eids >= routed_expert_start_idx) &
            (sorted_eids < routed_expert_end_idx)
        )

        input_x = sorted_x[local_token_expanded_x_indices]
        input_eids = sorted_eids[local_token_expanded_x_indices]
        global_indices = sorted_tok[local_token_expanded_x_indices]
        token_topk_pos = sorted_pos[local_token_expanded_x_indices]

        return input_x, input_eids, global_indices, token_topk_pos
