import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import warnings

# Compile the CUDA extension on-the-fly
def load_expert_bincount_cuda():
    """Load the expert bincount CUDA extension."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Try to load the extension
        expert_bincount_cuda = load(
            name="expert_bincount_cuda",
            sources=[
                os.path.join(current_dir, "expert_bincount.cu")
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-Xptxas=-O3",
                "-Xcompiler=-O3",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
            verbose=False
        )
        return expert_bincount_cuda
    except Exception as e:
        warnings.warn(f"Failed to compile expert bincount CUDA extension: {e}. Falling back to PyTorch implementation.")
        return None

# Global variable to cache the compiled extension
_expert_bincount_cuda = None

def get_expert_bincount_cuda():
    """Get the compiled CUDA extension, compiling it if necessary."""
    global _expert_bincount_cuda
    if _expert_bincount_cuda is None:
        _expert_bincount_cuda = load_expert_bincount_cuda()
    return _expert_bincount_cuda


class FusedExpertBincount:
    """
    Fused expert bincount that replaces the PyTorch bincount + nonzero + cumsum pipeline.
    
    This optimizes the expert token counting and grouping used in MoE models.
    """
    
    def __init__(self, use_cuda_if_available=True):
        self.use_cuda = use_cuda_if_available
        self.cuda_ext = get_expert_bincount_cuda() if use_cuda_if_available else None
        
    def __call__(self, 
                 eids: torch.Tensor,
                 routed_expert_start_idx: int,
                 experts_per_rank: int,
                 device: torch.device = None):
        """
        Optimized expert bincount with active expert compaction.
        
        Args:
            eids: [num_tokens] - Expert IDs for each token
            routed_expert_start_idx: Start index of local experts
            experts_per_rank: Number of experts per rank/device
            device: Target device (inferred from eids if None)
            
        Returns:
            tuple: (group_size, activated_group_idx, group_start_indices)
                - group_size: [num_active_experts] - Number of tokens per active expert
                - activated_group_idx: [num_active_experts] - Indices of active experts
                - group_start_indices: [num_active_experts] - Start positions for each expert
        """
        
        if device is None:
            device = eids.device
            
        # Ensure eids is the right shape and type
        if eids.dim() == 2 and eids.size(1) == 1:
            eids = eids.squeeze(1)  # [num_tokens, 1] -> [num_tokens]
        
        # Use CUDA implementation if available
        if self.cuda_ext is not None and eids.is_cuda:
            return self._fused_cuda_bincount(eids, routed_expert_start_idx, experts_per_rank, device)
        else:
            # Fallback to original PyTorch implementation
            return self._pytorch_fallback(eids, routed_expert_start_idx, experts_per_rank, device)
    
    def _fused_cuda_bincount(self, eids, routed_expert_start_idx, experts_per_rank, device):
        """Optimized CUDA implementation."""
        
        # Ensure contiguous tensor
        eids = eids.contiguous()
        
        # Call the CUDA extension
        results = self.cuda_ext.expert_bincount(
            eids,
            routed_expert_start_idx,
            experts_per_rank,
            device
        )
        
        group_size, activated_group_idx, group_start_indices = results
        
        return group_size, activated_group_idx, group_start_indices
    
    def _pytorch_fallback(self, eids, routed_expert_start_idx, experts_per_rank, device):
        """Original PyTorch implementation as fallback."""
        
        # Original implementation
        eids_adjusted = eids - routed_expert_start_idx  
        counts = torch.bincount(eids_adjusted, minlength=experts_per_rank)
        
        nonzero_mask = counts > 0
        activated_group_idx = torch.nonzero(nonzero_mask, as_tuple=True)[0].to(torch.int32)
        group_size = counts[nonzero_mask].to(torch.int32)
        
        group_start_indices = torch.zeros_like(group_size)
        if group_size.numel() > 1:
            group_start_indices[1:] = torch.cumsum(group_size[:-1], dim=0)
        
        return group_size, activated_group_idx, group_start_indices


def replace_expert_bincount_in_code():
    """
    Helper function to show how to replace the original expert_bincount method.
    
    Replace:
        def expert_bincount(self, eids, routed_expert_start_idx, experts_per_rank, device):
            eids_adjusted = eids - routed_expert_start_idx  
            counts = torch.bincount(eids_adjusted, minlength=experts_per_rank)
            
            nonzero_mask = counts > 0
            activated_group_idx = torch.nonzero(nonzero_mask, as_tuple=True)[0].to(torch.int32)
            group_size = counts[nonzero_mask].to(torch.int32)
            
            group_start_indices = torch.zeros_like(group_size)
            if group_size.numel() > 1:
                group_start_indices[1:] = torch.cumsum(group_size[:-1], dim=0)
            
            return group_size, activated_group_idx, group_start_indices
    
    With:
        def __init__(self):
            self.expert_bincounter = FusedExpertBincount()
            
        def expert_bincount(self, eids, routed_expert_start_idx, experts_per_rank, device):
            return self.expert_bincounter(eids, routed_expert_start_idx, experts_per_rank, device)
    """
    pass


# Example usage and benchmark
def benchmark_expert_bincount(num_tokens=8192, experts_per_rank=64, num_active_experts=8):
    """Benchmark the fused vs original expert bincount."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data - simulate tokens assigned to a subset of experts
    # Most tokens go to first few experts (realistic MoE load balancing)
    active_expert_ids = torch.randint(0, num_active_experts, (num_tokens,), device=device, dtype=torch.int32)
    # Add some tokens to other experts to test edge cases
    random_expert_ids = torch.randint(0, experts_per_rank, (num_tokens // 10,), device=device, dtype=torch.int32)
    eids = torch.cat([active_expert_ids, random_expert_ids])
    
    routed_expert_start_idx = 0
    
    print(f"Testing with {eids.size(0)} tokens, {experts_per_rank} experts per rank")
    print(f"Expert ID range: {eids.min().item()} to {eids.max().item()}")
    
    # Initialize bincount functions
    fused_bincounter = FusedExpertBincount(use_cuda_if_available=True)
    
    # Warm up
    for _ in range(10):
        result = fused_bincounter(eids, routed_expert_start_idx, experts_per_rank, device)
    
    # Benchmark fused implementation
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        group_size, activated_group_idx, group_start_indices = fused_bincounter(
            eids, routed_expert_start_idx, experts_per_rank, device
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    fused_time = (end_time - start_time) / 100 * 1000
    
    # Benchmark original implementation  
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        # Original implementation
        eids_adjusted = eids - routed_expert_start_idx  
        counts = torch.bincount(eids_adjusted, minlength=experts_per_rank)
        nonzero_mask = counts > 0
        activated_group_idx_orig = torch.nonzero(nonzero_mask, as_tuple=True)[0].to(torch.int32)
        group_size_orig = counts[nonzero_mask].to(torch.int32)
        group_start_indices_orig = torch.zeros_like(group_size_orig)
        if group_size_orig.numel() > 1:
            group_start_indices_orig[1:] = torch.cumsum(group_size_orig[:-1], dim=0)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    original_time = (end_time - start_time) / 100 * 1000
    
    # Verify correctness
    try:
        torch.testing.assert_close(group_size, group_size_orig)
        torch.testing.assert_close(activated_group_idx, activated_group_idx_orig)
        torch.testing.assert_close(group_start_indices, group_start_indices_orig)
        correctness = "✅ PASSED"
    except Exception as e:
        correctness = f"❌ FAILED: {e}"
    
    print(f"\n--- Expert Bincount Benchmark ---")
    print(f"Original implementation: {original_time:.2f} ms")
    print(f"Fused implementation:    {fused_time:.2f} ms")
    print(f"Speedup:                 {original_time / fused_time:.2f}x")
    print(f"Correctness:             {correctness}")
    print(f"Active experts:          {len(activated_group_idx)}/{experts_per_rank}")
    print(f"Group sizes:             {group_size.tolist()}")
    
    return group_size, activated_group_idx, group_start_indices


def stress_test_edge_cases():
    """Test edge cases like empty inputs, single expert, etc."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bincounter = FusedExpertBincount()
    
    print("\n--- Edge Case Testing ---")
    
    # Test 1: Empty input
    try:
        empty_eids = torch.empty(0, dtype=torch.int32, device=device)
        result = bincounter(empty_eids, 0, 8, device)
        print("✅ Empty input test passed")
    except Exception as e:
        print(f"❌ Empty input test failed: {e}")
    
    # Test 2: Single expert
    try:
        single_eids = torch.zeros(100, dtype=torch.int32, device=device)
        result = bincounter(single_eids, 0, 8, device)
        assert result[0].numel() == 1 and result[0][0] == 100
        print("✅ Single expert test passed")
    except Exception as e:
        print(f"❌ Single expert test failed: {e}")
    
    # Test 3: All experts active
    try:
        all_eids = torch.arange(8, dtype=torch.int32, device=device).repeat(10)
        result = bincounter(all_eids, 0, 8, device)
        assert result[0].numel() == 8
        print("✅ All experts active test passed")
    except Exception as e:
        print(f"❌ All experts active test failed: {e}")
    
    # Test 4: Out of range expert IDs (should be ignored)
    try:
        mixed_eids = torch.tensor([0, 1, 2, 100, 200], dtype=torch.int32, device=device)
        result = bincounter(mixed_eids, 0, 8, device)
        assert result[0].sum() == 3  # Only first 3 should be counted
        print("✅ Out of range test passed")
    except Exception as e:
        print(f"❌ Out of range test failed: {e}")


if __name__ == "__main__":
    print("Fused Expert Bincount Benchmark")
    print("=" * 40)
    
    # Run main benchmark
    benchmark_expert_bincount()
    
    # Run edge case tests
    stress_test_edge_cases()
    
    print("\nKey Benefits:")
    print("- Single kernel pass instead of bincount + nonzero + cumsum")
    print("- Atomic operations for efficient counting")
    print("- Direct compaction without intermediate tensors")
    print("- Handles edge cases robustly")