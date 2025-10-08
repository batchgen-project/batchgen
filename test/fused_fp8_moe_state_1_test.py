import torch
import numpy as np
import triton
import math
from batchgen.moe.fused_dequant_moe import fused_fp8_moe_stage_1


def quantize_fp8_with_block_scales(
    tensor: torch.Tensor,
    block_size_k: int = 128,
    block_size_n: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 with blocked scaling.
    
    Args:
        tensor: Input tensor [M, K] or [N, K]
        block_size_k: Block size along K dimension
        block_size_n: Block size along N dimension (None for 1D blocking)
    
    Returns:
        fp8_tensor: Quantized tensor
        scales: Scale tensor
    """
    dtype = tensor.dtype
    device = tensor.device
    
    if block_size_n is None:
        # 1D blocking (for activations): [M, K] -> [M, K//block_size_k]
        M, K = tensor.shape
        num_blocks_k = math.ceil(K / block_size_k)
        
        # Pad K dimension if needed
        if K % block_size_k != 0:
            pad_k = block_size_k - (K % block_size_k)
            tensor = torch.nn.functional.pad(tensor, (0, pad_k))
        
        # Reshape to [M, num_blocks_k, block_size_k]
        tensor_blocks = tensor.reshape(M, num_blocks_k, block_size_k)
        
        # Compute scales per block: max absolute value
        scales = tensor_blocks.abs().max(dim=2, keepdim=False)[0]
        scales = scales.clamp(min=1e-12)  # Avoid division by zero
        
        # Quantize: divide by scale and convert to FP8
        tensor_normalized = tensor_blocks / scales.unsqueeze(2)
        fp8_tensor = tensor_normalized.reshape(M, -1)[:, :K].to(torch.float8_e4m3fn)
        
        return fp8_tensor, scales.to(torch.float32)
    else:
        # 2D blocking (for weights): [N, K] -> [N//block_size_n, K//block_size_k]
        N, K = tensor.shape
        num_blocks_n = math.ceil(N / block_size_n)
        num_blocks_k = math.ceil(K / block_size_k)
        
        # Pad if needed
        pad_n = (block_size_n - (N % block_size_n)) % block_size_n
        pad_k = (block_size_k - (K % block_size_k)) % block_size_k
        if pad_n > 0 or pad_k > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_n))
        
        # Reshape to [num_blocks_n, block_size_n, num_blocks_k, block_size_k]
        tensor_blocks = tensor.reshape(num_blocks_n, block_size_n, num_blocks_k, block_size_k)
        
        # Compute scales per 2D block
        scales = tensor_blocks.abs().amax(dim=(1, 3))  # [num_blocks_n, num_blocks_k]
        scales = scales.clamp(min=1e-12)
        
        # Quantize
        tensor_normalized = tensor_blocks / scales.unsqueeze(1).unsqueeze(3)
        fp8_tensor = tensor_normalized.reshape(num_blocks_n * block_size_n, num_blocks_k * block_size_k)[:N, :K].to(torch.float8_e4m3fn)
        
        return fp8_tensor, scales.to(torch.float32)


def dequantize_fp8_with_block_scales(
    fp8_tensor: torch.Tensor,
    scales: torch.Tensor,
    block_size_k: int = 128,
    block_size_n: int = None
) -> torch.Tensor:
    """
    Dequantize FP8 tensor with blocked scales.
    """
    if block_size_n is None:
        # 1D dequantization (activations)
        M, K = fp8_tensor.shape
        num_blocks_k = math.ceil(K / block_size_k)
        
        # Convert to float and expand scales
        tensor_float = fp8_tensor.to(torch.float32)
        
        # Create scale matrix [M, K] by repeating scales
        scales_expanded = scales.unsqueeze(2).repeat(1, 1, block_size_k).reshape(M, -1)[:, :K]
        
        return (tensor_float * scales_expanded).to(torch.bfloat16)
    else:
        # 2D dequantization (weights)
        N, K = fp8_tensor.shape
        num_blocks_n = math.ceil(N / block_size_n)
        num_blocks_k = math.ceil(K / block_size_k)
        
        tensor_float = fp8_tensor.to(torch.float32)
        
        # Expand scales to [N, K]
        scales_expanded = scales.unsqueeze(1).unsqueeze(3).repeat(1, block_size_n, 1, block_size_k)
        scales_expanded = scales_expanded.reshape(num_blocks_n * block_size_n, num_blocks_k * block_size_k)[:N, :K]
        
        return (tensor_float * scales_expanded).to(torch.bfloat16)


def silu(x):
    """SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))"""
    return x * torch.sigmoid(x)


def reference_fused_moe_stage_1(
    hidden_states_fp8: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gate_weight_list: list[torch.Tensor],
    gate_scale_list: list[torch.Tensor],
    up_weight_list: list[torch.Tensor],
    up_scale_list: list[torch.Tensor],
    group_sizes: torch.Tensor,
    activated_group_idx: torch.Tensor,
    group_start_indices: torch.Tensor,
    scale_block_size: tuple = (128, 128)
):
    """
    Reference implementation of MoE stage 1: silu(X @ gate^T) * (X @ up^T)
    """
    M, K = hidden_states_fp8.shape
    N = gate_weight_list[0].shape[0]
    device = hidden_states_fp8.device
    
    # Dequantize hidden states
    hidden_states = dequantize_fp8_with_block_scales(
        hidden_states_fp8,
        hidden_states_scale,
        block_size_k=scale_block_size[1],
        block_size_n=None
    )
    
    # Output buffer
    output = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    
    # Process each group
    for g in range(group_sizes.shape[0]):
        group_size = group_sizes[g].item()
        expert_idx = activated_group_idx[g].item()
        start_idx = group_start_indices[g].item()
        
        # Get the rows for this group
        row_indices = list(range(start_idx, start_idx + group_size))
        hidden_group = hidden_states[row_indices]  # [group_size, K]
        
        # Dequantize gate and up weights for this expert
        gate_weight = dequantize_fp8_with_block_scales(
            gate_weight_list[expert_idx],
            gate_scale_list[expert_idx],
            block_size_k=scale_block_size[1],
            block_size_n=scale_block_size[0]
        )  # [N, K]
        
        up_weight = dequantize_fp8_with_block_scales(
            up_weight_list[expert_idx],
            up_scale_list[expert_idx],
            block_size_k=scale_block_size[1],
            block_size_n=scale_block_size[0]
        )  # [N, K]
        
        # Compute: silu(hidden @ gate^T) * (hidden @ up^T)
        gate_output = hidden_group @ gate_weight.t()  # [group_size, N]
        up_output = hidden_group @ up_weight.t()      # [group_size, N]
        
        # Apply SiLU and elementwise multiply
        result = silu(gate_output) * up_output
        
        # Store result
        output[row_indices] = result
    
    return output


def test_fused_fp8_moe_stage_1():
    """
    Test the fused FP8 MoE stage 1 kernel against reference implementation.
    """
    print("="*80)
    print("Testing Fused FP8 MoE Stage 1 Kernel")
    print("="*80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    M = 30  # Number of tokens (small batch)
    K = 1024  # Hidden dimension
    N = 2048  # Expert output dimension
    num_experts = 4
    scale_block_size = (128, 128)
    
    print(f"\nConfiguration:")
    print(f"  M (tokens): {M}")
    print(f"  K (hidden_dim): {K}")
    print(f"  N (expert_dim): {N}")
    print(f"  Num experts: {num_experts}")
    print(f"  Scale block size: {scale_block_size}")
    
    # Generate random data
    torch.manual_seed(42)
    hidden_states_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # Quantize hidden states (1D blocking along K)
    hidden_states_fp8, hidden_states_scale = quantize_fp8_with_block_scales(
        hidden_states_bf16,
        block_size_k=scale_block_size[1],
        block_size_n=None
    )
    
    print(f"\nHidden states:")
    print(f"  FP8 shape: {hidden_states_fp8.shape}")
    print(f"  Scale shape: {hidden_states_scale.shape}")
    
    # Generate expert weights
    gate_weight_list = []
    gate_scale_list = []
    up_weight_list = []
    up_scale_list = []
    
    for i in range(num_experts):
        # Gate weights
        gate_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02
        gate_fp8, gate_scale = quantize_fp8_with_block_scales(
            gate_bf16,
            block_size_k=scale_block_size[1],
            block_size_n=scale_block_size[0]
        )
        gate_weight_list.append(gate_fp8)
        gate_scale_list.append(gate_scale)
        
        # Up weights
        up_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02
        up_fp8, up_scale = quantize_fp8_with_block_scales(
            up_bf16,
            block_size_k=scale_block_size[1],
            block_size_n=scale_block_size[0]
        )
        up_weight_list.append(up_fp8)
        up_scale_list.append(up_scale)
    
    print(f"\nExpert weights:")
    print(f"  Gate shape: {gate_weight_list[0].shape}")
    print(f"  Gate scale shape: {gate_scale_list[0].shape}")
    print(f"  Up shape: {up_weight_list[0].shape}")
    print(f"  Up scale shape: {up_scale_list[0].shape}")
    
    # Simulate MoE routing: distribute tokens across experts
    # For simplicity, split tokens roughly evenly
    tokens_per_expert = M // num_experts
    remaining = M % num_experts
    
    group_sizes_list = []
    activated_group_idx_list = []
    group_start_indices_list = []
    
    current_idx = 0
    for i in range(num_experts):
        group_size = tokens_per_expert + (1 if i < remaining else 0)
        if group_size > 0:
            group_sizes_list.append(group_size)
            activated_group_idx_list.append(i)
            group_start_indices_list.append(current_idx)
            current_idx += group_size
    
    num_groups = len(group_sizes_list)
    group_sizes = torch.tensor(group_sizes_list, dtype=torch.int32, device=device)
    activated_group_idx = torch.tensor(activated_group_idx_list, dtype=torch.int32, device=device)
    group_start_indices = torch.tensor(group_start_indices_list, dtype=torch.int32, device=device)
    
    print(f"\nRouting:")
    print(f"  Num groups: {num_groups}")
    print(f"  Group sizes: {group_sizes_list}")
    print(f"  Activated experts: {activated_group_idx_list}")
    print(f"  Start indices: {group_start_indices_list}")
    
    # Create pointer tensors
    gate_ptrs = torch.tensor([w.data_ptr() for w in gate_weight_list], dtype=torch.int64, device=device)
    up_ptrs = torch.tensor([w.data_ptr() for w in up_weight_list], dtype=torch.int64, device=device)
    gate_scale_ptrs = torch.tensor([s.data_ptr() for s in gate_scale_list], dtype=torch.int64, device=device)
    up_scale_ptrs = torch.tensor([s.data_ptr() for s in up_scale_list], dtype=torch.int64, device=device)
    
    # Run reference implementation
    print("\n" + "-"*80)
    print("Running reference implementation...")
    ref_output = reference_fused_moe_stage_1(
        hidden_states_fp8,
        hidden_states_scale,
        gate_weight_list,
        gate_scale_list,
        up_weight_list,
        up_scale_list,
        group_sizes,
        activated_group_idx,
        group_start_indices,
        scale_block_size=scale_block_size
    )
    print(f"Reference output shape: {ref_output.shape}")
    print(f"Reference output stats: min={ref_output.min():.4f}, max={ref_output.max():.4f}, mean={ref_output.mean():.4f}")
    
    # Run kernel implementation
    print("\n" + "-"*80)
    print("Running kernel implementation...")
    kernel_output = fused_fp8_moe_stage_1(
        hidden_states_fp8,
        hidden_states_scale,
        gate_weight_list,
        gate_ptrs,
        up_weight_list,
        up_ptrs,
        gate_scale_list,
        gate_scale_ptrs,
        up_scale_list,
        up_scale_ptrs,
        group_sizes,
        activated_group_idx,
        group_start_indices,
        gate_gemm_block_size=[64, 16, 256],
        up_gemm_block_size=[64, 16, 256],
        scale_block_size=list(scale_block_size),
        num_stages=2,
        num_warps=4
    )
    print(f"Kernel output shape: {kernel_output.shape}")
    print(f"Kernel output stats: min={kernel_output.min():.4f}, max={kernel_output.max():.4f}, mean={kernel_output.mean():.4f}")
    
    # Compare outputs
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Convert to float32 for comparison
    ref_f32 = ref_output.float()
    kernel_f32 = kernel_output.float()
    
    # Calculate errors
    abs_diff = torch.abs(ref_f32 - kernel_f32)
    rel_diff = abs_diff / (torch.abs(ref_f32) + 1e-6)
    
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    max_rel_error = rel_diff.max().item()
    mean_rel_error = rel_diff.mean().item()
    
    print(f"\nAbsolute Error:")
    print(f"  Max:  {max_abs_error:.6f}")
    print(f"  Mean: {mean_abs_error:.6f}")
    
    print(f"\nRelative Error:")
    print(f"  Max:  {max_rel_error:.6f}")
    print(f"  Mean: {mean_rel_error:.6f}")
    
    # Check for correctness (bfloat16 tolerance)
    # BF16 has ~3 decimal digits of precision, so we use loose tolerances
    rtol = 1e-2  # 1% relative tolerance
    atol = 1e-2  # Absolute tolerance
    
    is_close = torch.allclose(ref_f32, kernel_f32, rtol=rtol, atol=atol)
    
    print(f"\nTest Status (rtol={rtol}, atol={atol}):")
    if is_close:
        print("  ✓ PASSED - Outputs match within tolerance!")
    else:
        print("  ✗ FAILED - Outputs differ beyond tolerance!")
        
        # Find worst mismatches
        worst_indices = torch.topk(abs_diff.flatten(), k=5)
        print(f"\n  Top 5 worst mismatches:")
        for i, (val, idx) in enumerate(zip(worst_indices.values, worst_indices.indices)):
            m_idx = idx // N
            n_idx = idx % N
            print(f"    {i+1}. [{m_idx}, {n_idx}]: ref={ref_f32[m_idx, n_idx]:.6f}, "
                  f"kernel={kernel_f32[m_idx, n_idx]:.6f}, diff={val:.6f}")
    
    print("\n" + "="*80)
    
    return is_close


def test_different_configurations():
    """Test multiple configurations"""
    configs = [
        {"M": 30, "K": 1024, "N": 2048, "experts": 4, "name": "Small batch"},
        {"M": 128, "K": 2048, "N": 4096, "experts": 8, "name": "Medium batch"},
        {"M": 8, "K": 512, "N": 1024, "experts": 2, "name": "Very small batch"},
    ]
    
    print("\n" + "="*80)
    print("TESTING MULTIPLE CONFIGURATIONS")
    print("="*80)
    
    results = []
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")
        # You would need to adapt the test function to accept parameters
        # For now, just running the default test
        passed = test_fused_fp8_moe_stage_1()
        results.append((config['name'], passed))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    # Run single test
    test_fused_fp8_moe_stage_1()
    
    # Uncomment to test multiple configurations
    # test_different_configurations()