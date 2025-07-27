import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import warnings

# Compile the CUDA extension on-the-fly
def load_fused_moe_cuda():
    """Load the fused MoE CUDA extension."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Try to load the extension
        fused_moe_cuda = load(
            name="fused_moe_cuda",
            sources=[
                os.path.join(current_dir, "fused_moe_dispatch.cu")
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-Xptxas=-O3",
                "-Xcompiler=-O3",
            ],
            verbose=False
        )
        return fused_moe_cuda
    except Exception as e:
        warnings.warn(f"Failed to compile CUDA extension: {e}. Falling back to PyTorch implementation.")
        return None

# Global variable to cache the compiled extension
_fused_moe_cuda = None

def get_fused_moe_cuda():
    """Get the compiled CUDA extension, compiling it if necessary."""
    global _fused_moe_cuda
    if _fused_moe_cuda is None:
        _fused_moe_cuda = load_fused_moe_cuda()
    return _fused_moe_cuda


class FusedMoETokenDispatch:
    """
    Fused MoE token dispatch that replaces the manual expand, sort, permute pipeline.
    
    This class provides both optimized CUDA and fallback PyTorch implementations.
    """
    
    def __init__(self, use_cuda_if_available=True):
        self.use_cuda = use_cuda_if_available
        self.cuda_ext = get_fused_moe_cuda() if use_cuda_if_available else None
        
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
            return self._fused_cuda_dispatch(
                global_x, topk_idx, token_idx, topk_pos,
                routed_expert_start_idx, routed_expert_end_idx
            )
        else:
            # Fallback to original PyTorch implementation
            return self._pytorch_fallback(
                global_x, topk_idx, token_idx, topk_pos,
                routed_expert_start_idx, routed_expert_end_idx
            )
    
    def _fused_cuda_dispatch(self, global_x, topk_idx, token_idx, topk_pos,
                           routed_expert_start_idx, routed_expert_end_idx):
        """Optimized CUDA implementation."""
        
        # Ensure contiguous tensors
        global_x = global_x.contiguous()
        topk_idx = topk_idx.contiguous()
        
        # Call the CUDA extension
        results = self.cuda_ext.fused_moe_token_dispatch(
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


def replace_moe_dispatch_in_code(original_code_section):
    """
    Helper function to show how to replace the original code section.
    
    Replace:
        flat_eids   = topk_idx.flatten()
        expanded_x  = global_x.repeat_interleave(K, dim=0)
        sorted_eids, sort_idx = flat_eids.sort()
        sorted_x   = expanded_x[sort_idx]
        sorted_tok = self.token_idx[sort_idx]
        sorted_pos = self.topk_pos[sort_idx]
        
        local_token_expanded_x_indices = (sorted_eids >= self.routed_expert_start_idx) & (sorted_eids < self.routed_expert_end_idx)
        input_x = sorted_x[local_token_expanded_x_indices]
        input_eids = sorted_eids[local_token_expanded_x_indices]
        global_indices = sorted_tok[local_token_expanded_x_indices]
        token_topk_pos = sorted_pos[local_token_expanded_x_indices]
    
    With:
        dispatcher = FusedMoETokenDispatch()
        input_x, input_eids, global_indices, token_topk_pos = dispatcher(
            global_x, topk_idx, self.token_idx, self.topk_pos,
            self.routed_expert_start_idx, self.routed_expert_end_idx
        )
    """
    pass


# Example usage and benchmark
def benchmark_dispatch(num_tokens=8192, hidden_size=4096, K=2, num_experts=64, num_local_experts=8):
    """Benchmark the fused vs original implementation."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    global_x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    topk_idx = torch.randint(0, num_experts, (num_tokens, K), device=device, dtype=torch.int32)
    
    # Create token_idx and topk_pos (as in original implementation)
    token_idx = torch.arange(num_tokens, device=device, dtype=torch.int32).repeat_interleave(K)
    topk_pos = torch.arange(K, device=device, dtype=torch.int32).repeat(num_tokens)
    
    routed_expert_start_idx = 0
    routed_expert_end_idx = num_local_experts
    
    # Initialize dispatcher
    dispatcher = FusedMoETokenDispatch(use_cuda_if_available=True)
    
    # Warm up
    for _ in range(10):
        result = dispatcher(global_x, topk_idx, token_idx, topk_pos,
                          routed_expert_start_idx, routed_expert_end_idx)
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        result = dispatcher(global_x, topk_idx, token_idx, topk_pos,
                          routed_expert_start_idx, routed_expert_end_idx)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Average time per dispatch: {(end_time - start_time) / 100 * 1000:.2f} ms")
    print(f"Result shapes: input_x={result[0].shape}, input_eids={result[1].shape}")
    
    return result


if __name__ == "__main__":
    # Run benchmark
    benchmark_dispatch()