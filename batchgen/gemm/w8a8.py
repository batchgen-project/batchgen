
# import torch
# import triton
# import triton.language as tl
# from typing import Optional, Tuple
# import math

# @triton.autotune(
#     configs=[
#         # Small tiles for small matrices
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        
#         # Medium tiles
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        
#         # Large tiles for large matrices
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#     ],
#     key=['M', 'N', 'K'],
# )
# @triton.jit
# def w8a8_gemm_kernel(
#     # Pointers to matrices
#     a_ptr, w_ptr, c_ptr,
#     # Pointers to scales
#     a_scale_ptr, w_scale_ptr,
#     # Matrix dimensions
#     M, N: tl.constexpr, K: tl.constexpr,
#     # Quantization block sizes
#     a_block_size, w_block_size_k, w_block_size_n,
#     # The stride variables represent how much to increase the ptr by when moving by 1
#     # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
#     # by to get the element one row down (A has M rows).
#     stride_am, stride_ak,  # A is (M, K)
#     stride_wn, stride_wk,  # W is (N, K) 
#     stride_cm, stride_cn,  # C is (M, N)
#     stride_a_scale_m, stride_a_scale_k,  # a_scale is (M, ceil(K/a_block_size))
#     stride_w_scale_n, stride_w_scale_k,  # w_scale is (ceil(N/w_block_size_n), ceil(K/w_block_size_k))
#     # Meta-parameters
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     """Kernel for computing the matmul C = A @ W^T with quantized inputs.
    
#     A has shape (M, K) with per-row quantization
#     W has shape (N, K) with 2D block quantization
#     C has shape (M, N) and will be in bf16
#     """
#     # -----------------------------------------------------------
#     # Map program ids `pid` to the block of C it should compute.
#     # This is done in a grouped ordering to promote L2 data reuse.
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # ----------------------------------------------------------
#     # Create pointers for the first blocks of A, W, and their scales.
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
    
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)
    
#     # Initialize the accumulator.
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
#     # -----------------------------------------------------------
#     # Iterate to compute a block of the C matrix.
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         # Load the next block of A and W.
#         k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
#         a_fp8 = tl.load(a_ptrs, mask=k_mask, other=0.0)
        
#         w_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
#         w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
#         # Load quantization scales
#         # For A: per-row quantization, scale varies across K dimension in blocks
#         k_start = k * BLOCK_SIZE_K
#         a_k_block_idx = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + (offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k)
#         a_scales = tl.load(a_scale_ptrs, mask=offs_am < M, other=1.0)
        
#         # For W: 2D block quantization
#         # Calculate which blocks we're in for both N and K dimensions
#         w_k_block_idx = k_start // w_block_size_k
        
#         # For each element in offs_bn, calculate which N block it belongs to
#         n_block_indices = offs_bn // w_block_size_n
        
#         # Load the appropriate scales for each N block
#         w_scale_ptrs = w_scale_ptr + (n_block_indices * stride_w_scale_n + w_k_block_idx * stride_w_scale_k)
#         w_scales = tl.load(w_scale_ptrs, mask=offs_bn < N, other=1.0)

#         # Apply scales and accumulate
#         accumulator += tl.dot(a_fp8 * tl.trans(w_fp8)) * a_scales * w_scales\
        
#         # Advance the ptrs to the next K block.
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         w_ptrs += BLOCK_SIZE_K * stride_wk

#     # Convert accumulator from fp32 to bf16
#     c = accumulator.to(tl.bfloat16)

#     # -----------------------------------------------------------
#     # Write back the block of the output matrix C with masks.
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# def w8a8_gemm(
#     a: torch.Tensor,           # [M, K] - activations in fp8e4m3
#     a_scale: torch.Tensor,     # [M, ceil(K / a_block_size)] - per-row quantization scales
#     w: torch.Tensor,           # [N, K] - weights in fp8e4m3  
#     w_scale: torch.Tensor,     # [ceil(N/w_block_size_n), ceil(K/w_block_size_k)] - 2D block scales
#     a_block_size: int = 128,   # Block size for A quantization
#     w_block_size_k: int = 128, # Block size for W quantization along K
#     w_block_size_n: int = 128, # Block size for W quantization along N
# ) -> torch.Tensor:
#     """
#     Performs quantized GEMM: C = A @ W^T
#     This kernel assumes:
#     - A is quantized with per-token (row) scales with block size `a_block_size`
#     - W is quantized with 2D block scales with block sizes `w_block_size_k` and `w_block_size_n`
#     - W is [N, K] 
#     - Scales are in fp32 format.
#     - Both A and W are in fp8e4m3 format
#     - Output C is in bfloat16 format
    
#     Args:
#         a: Activations tensor [M, K] in fp8e4m3
#         a_scale: Per-row quantization scales [M, ceil(K/a_block_size)]
#         w: Weight tensor [N, K] in fp8e4m3
#         w_scale: 2D block quantization scales [ceil(N/w_block_size_n), ceil(K/w_block_size_k)]
#         a_block_size: Block size for activation quantization
#         w_block_size_k: Block size for weight quantization along K dimension
#         w_block_size_n: Block size for weight quantization along N dimension
    
#     Returns:
#         torch.Tensor: Output tensor [M, N] in bfloat16
#     """
#     # Check input shapes and dtypes
#     assert a.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3 for activations, got {a.dtype}"
#     assert w.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3 for weights, got {w.dtype}"
#     assert a_scale.dtype == torch.float32, f"Expected float32 for a_scale, got {a_scale.dtype}"
#     assert w_scale.dtype == torch.float32, f"Expected float32 for w_scale, got {w_scale.dtype}"
    
#     # Get dimensions
#     M, K = a.shape
#     N, K_w = w.shape
#     assert K == K_w, f"Dimension mismatch: A has {K} columns, W has {K_w} columns"
        
#     # Validate scale tensor shapes
#     expected_a_scale_shape = (M, math.ceil(K / a_block_size))
#     expected_w_scale_shape = (math.ceil(N / w_block_size_n), math.ceil(K / w_block_size_k))
    
#     assert a_scale.shape == expected_a_scale_shape, \
#         f"a_scale shape mismatch: expected {expected_a_scale_shape}, got {a_scale.shape}"
#     assert w_scale.shape == expected_w_scale_shape, \
#         f"w_scale shape mismatch: expected {expected_w_scale_shape}, got {w_scale.shape}"
    
#     # Allocate output tensor in bfloat16
#     c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        
#     # Launch kernel with auto-tuning
#     grid = lambda META: (
#         triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#     )
    
#     w8a8_gemm_kernel[grid](
#         a, w, c,
#         a_scale, w_scale,
#         M, N, K,
#         a_block_size, w_block_size_k, w_block_size_n,
#         a.stride(0), a.stride(1),
#         w.stride(0), w.stride(1),
#         c.stride(0), c.stride(1),
#         a_scale.stride(0), a_scale.stride(1),
#         w_scale.stride(0), w_scale.stride(1),
#         # BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=4
#     )
    
#     return c


# def test_w8a8_gemm():
#     # Example test case
#     M, K, N = 128, 256, 64
#     a = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
#     w = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn)
#     a_scale = torch.randn(M, math.ceil(K / 128), dtype=torch.float32, device='cuda')
#     w_scale = torch.randn(math.ceil(N / 128), math.ceil(K / 128), dtype=torch.float32, device='cuda')
    
#     c = w8a8_gemm(a, a_scale, w, w_scale)
#     assert c.dtype == torch.bfloat16
#     assert c.shape == (M, N)
#     print("Test passed! Output shape:", c.shape)



# if __name__ == "__main__":
#     test_w8a8_gemm()
#     print("w8a8_gemm kernel is ready for use!")


# import torch
# import triton
# import triton.language as tl
# import math


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=4),
#     ],
#     key=['M', 'N', 'K'],
# )
# @triton.jit
# def w8a8_gemm_kernel(
#     # Pointers to matrices
#     a_ptr, w_ptr, c_ptr,
#     # Pointers to scales
#     a_scale_ptr, w_scale_ptr,
#     # Matrix dimensions
#     M, N, K,
#     # Quantization block sizes
#     a_block_size, w_block_size_k, w_block_size_n,
#     # Strides
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     # Meta-parameters
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     """
#     Optimized kernel for computing C = A @ W^T with blocked quantized FP8 inputs.
    
#     A: [M, K] in fp8e4m3 with per-row block quantization
#     W: [N, K] in fp8e4m3 with 2D block quantization
#     C: [M, N] in bfloat16
#     """
#     # Program ID and block mapping with swizzling for better L2 cache reuse
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # Compute offsets
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)

#     # Initialize pointers for A and W
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)

#     # Initialize accumulator in fp32 for numerical stability
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     # Main computation loop over K dimension
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         k_start = k * BLOCK_SIZE_K
#         k_remaining = K - k_start
        
#         # Compute masks for boundary conditions
#         k_mask = offs_k < k_remaining
#         a_mask = (offs_am[:, None] < M) & (k_mask[None, :])
#         w_mask = (offs_bn[:, None] < N) & (k_mask[None, :])
        
#         # Load FP8 data
#         a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
#         # Compute FP8 matmul with Hopper tensor cores
#         # Use input_precision for FP8 accumulation on Hopper
#         acc_block = tl.dot(a_fp8, tl.trans(w_fp8), input_precision="tf32", out_dtype=tl.float32)
        
#         # Load quantization scales for this K block
#         # A scales: per-row, indexed by K block
#         a_k_block_idx = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + (offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k)
#         a_scales = tl.load(a_scale_ptrs, mask=offs_am < M, other=1.0)
        
#         # W scales: 2D block indexed by (N block, K block)
#         w_k_block_idx = k_start // w_block_size_k
#         n_block_indices = offs_bn // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + (n_block_indices * stride_w_scale_n + w_k_block_idx * stride_w_scale_k)
#         w_scales = tl.load(w_scale_ptrs, mask=offs_bn < N, other=1.0)
        
#         # Apply scales with proper broadcasting
#         # a_scales: (BLOCK_SIZE_M,) -> (BLOCK_SIZE_M, 1)
#         # w_scales: (BLOCK_SIZE_N,) -> (1, BLOCK_SIZE_N)
#         # Result: (BLOCK_SIZE_M, BLOCK_SIZE_N)
#         scale_factor = a_scales[:, None] * w_scales[None, :]
#         accumulator += acc_block * scale_factor
        
#         # Advance pointers for next K block
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         w_ptrs += BLOCK_SIZE_K * stride_wk

#     # Convert to bfloat16 for output
#     c = accumulator.to(tl.bfloat16)

#     # Write output with boundary checking
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# def w8a8_gemm(
#     a: torch.Tensor,
#     a_scale: torch.Tensor,
#     w: torch.Tensor,
#     w_scale: torch.Tensor,
#     a_block_size: int = 128,
#     w_block_size_k: int = 128,
#     w_block_size_n: int = 128,
# ) -> torch.Tensor:
#     """
#     Performs quantized GEMM: C = A @ W^T with blocked quantization.
    
#     Args:
#         a: Activations [M, K] in fp8e4m3fn
#         a_scale: Scales [M, ceil(K/a_block_size)] in fp32
#         w: Weights [N, K] in fp8e4m3fn
#         w_scale: Scales [ceil(N/w_block_size_n), ceil(K/w_block_size_k)] in fp32
#         a_block_size: Block size for A quantization along K
#         w_block_size_k: Block size for W quantization along K
#         w_block_size_n: Block size for W quantization along N
        
#     Returns:
#         Output [M, N] in bfloat16
#     """
#     # Validate inputs
#     assert a.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3fn for a, got {a.dtype}"
#     assert w.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3fn for w, got {w.dtype}"
#     assert a_scale.dtype == torch.float32, f"Expected fp32 for a_scale, got {a_scale.dtype}"
#     assert w_scale.dtype == torch.float32, f"Expected fp32 for w_scale, got {w_scale.dtype}"
#     assert a.is_contiguous(), "a must be contiguous"
#     assert w.is_contiguous(), "w must be contiguous"
    
#     M, K = a.shape
#     N, K_w = w.shape
#     assert K == K_w, f"K dimension mismatch: A has {K}, W has {K_w}"
    
#     # Validate scale shapes
#     expected_a_scale_shape = (M, math.ceil(K / a_block_size))
#     expected_w_scale_shape = (math.ceil(N / w_block_size_n), math.ceil(K / w_block_size_k))
#     assert a_scale.shape == expected_a_scale_shape, \
#         f"a_scale shape mismatch: expected {expected_a_scale_shape}, got {a_scale.shape}"
#     assert w_scale.shape == expected_w_scale_shape, \
#         f"w_scale shape mismatch: expected {expected_w_scale_shape}, got {w_scale.shape}"
    
#     # Allocate output
#     c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
#     # Launch kernel
#     grid = lambda META: (
#         triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#     )
    
#     w8a8_gemm_kernel[grid](
#         a, w, c,
#         a_scale, w_scale,
#         M, N, K,
#         a_block_size, w_block_size_k, w_block_size_n,
#         a.stride(0), a.stride(1),
#         w.stride(0), w.stride(1),
#         c.stride(0), c.stride(1),
#         a_scale.stride(0), a_scale.stride(1),
#         w_scale.stride(0), w_scale.stride(1),
#     )
    
#     return c


# # Test and benchmark function
# def test_w8a8_gemm():
#     """Test the kernel with sample data."""
#     torch.manual_seed(0)
#     device = torch.device('cuda')
    
#     # Problem size
#     M, N, K = 4096, 4096, 4096
#     a_block_size = 128
#     w_block_size_k = 128
#     w_block_size_n = 128
    
#     # Create random FP8 tensors
#     a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#     w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    
#     # Create scales
#     a_scale = torch.rand(M, math.ceil(K / a_block_size), device=device, dtype=torch.float32) * 0.1
#     w_scale = torch.rand(
#         math.ceil(N / w_block_size_n),
#         math.ceil(K / w_block_size_k),
#         device=device,
#         dtype=torch.float32
#     ) * 0.1
    
#     # Run kernel
#     c = w8a8_gemm(a, a_scale, w, w_scale, a_block_size, w_block_size_k, w_block_size_n)
    
#     print(f"Output shape: {c.shape}")
#     print(f"Output dtype: {c.dtype}")
#     print(f"Output range: [{c.min().item():.4f}, {c.max().item():.4f}]")
    
#     return c


# if __name__ == "__main__":
#     test_w8a8_gemm()


import torch
import triton
import triton.language as tl
import math


@triton.jit
def w8a8_gemm_kernel(
    # Pointers to matrices
    a_ptr, w_ptr, c_ptr,
    # Pointers to scales
    a_scale_ptr, w_scale_ptr,
    # Matrix dimensions
    M, N, K,
    # Quantization block sizes
    a_block_size, w_block_size_k, w_block_size_n,
    # Strides
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k,
    stride_w_scale_n, stride_w_scale_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized kernel for computing C = A @ W^T with blocked quantized FP8 inputs.
    
    A: [M, K] in fp8e4m3 with per-row block quantization
    W: [N, K] in fp8e4m3 with 2D block quantization
    C: [M, N] in bfloat16
    """
    # Program ID and block mapping with swizzling for better L2 cache reuse
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers for A and W
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    # Initialize accumulator in fp32 for numerical stability
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main computation loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_remaining = K - k_start
        
        # Compute masks for boundary conditions
        k_mask = offs_k < k_remaining
        a_mask = (offs_am[:, None] < M) & (k_mask[None, :])
        w_mask = (offs_bn[:, None] < N) & (k_mask[None, :])
        
        # Load FP8 data
        a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Compute FP8 matmul with Hopper tensor cores
        # Use input_precision for FP8 accumulation on Hopper
        acc_block = tl.dot(a_fp8, tl.trans(w_fp8), input_precision="tf32", out_dtype=tl.float32)
        
        # Load quantization scales for this K block
        # A scales: per-row, indexed by K block
        a_k_block_idx = k_start // a_block_size
        a_scale_ptrs = a_scale_ptr + (offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k)
        a_scales = tl.load(a_scale_ptrs, mask=offs_am < M, other=1.0)
        
        # W scales: 2D block indexed by (N block, K block)
        w_k_block_idx = k_start // w_block_size_k
        n_block_indices = offs_bn // w_block_size_n
        w_scale_ptrs = w_scale_ptr + (n_block_indices * stride_w_scale_n + w_k_block_idx * stride_w_scale_k)
        w_scales = tl.load(w_scale_ptrs, mask=offs_bn < N, other=1.0)
        
        # Apply scales with proper broadcasting
        # a_scales: (BLOCK_SIZE_M,) -> (BLOCK_SIZE_M, 1)
        # w_scales: (BLOCK_SIZE_N,) -> (1, BLOCK_SIZE_N)
        # Result: (BLOCK_SIZE_M, BLOCK_SIZE_N)
        scale_factor = a_scales[:, None] * w_scales[None, :]
        accumulator += acc_block * scale_factor
        
        # Advance pointers for next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # Convert to bfloat16 for output
    c = accumulator.to(tl.bfloat16)

    # Write output with boundary checking
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    a_block_size: int = 128,
    w_block_size_k: int = 128,
    w_block_size_n: int = 128,
) -> torch.Tensor:
    """
    Performs quantized GEMM: C = A @ W^T with blocked quantization.
    
    Args:
        a: Activations [M, K] in fp8e4m3fn
        a_scale: Scales [M, ceil(K/a_block_size)] in fp32
        w: Weights [N, K] in fp8e4m3fn
        w_scale: Scales [ceil(N/w_block_size_n), ceil(K/w_block_size_k)] in fp32
        a_block_size: Block size for A quantization along K
        w_block_size_k: Block size for W quantization along K
        w_block_size_n: Block size for W quantization along N
        
    Returns:
        Output [M, N] in bfloat16
    """
    # Validate inputs
    assert a.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3fn for a, got {a.dtype}"
    assert w.dtype == torch.float8_e4m3fn, f"Expected fp8e4m3fn for w, got {w.dtype}"
    assert a_scale.dtype == torch.float32, f"Expected fp32 for a_scale, got {a_scale.dtype}"
    assert w_scale.dtype == torch.float32, f"Expected fp32 for w_scale, got {w_scale.dtype}"
    assert a.is_contiguous(), "a must be contiguous"
    assert w.is_contiguous(), "w must be contiguous"
    
    M, K = a.shape
    N, K_w = w.shape
    assert K == K_w, f"K dimension mismatch: A has {K}, W has {K_w}"
    
    # Validate scale shapes
    expected_a_scale_shape = (M, math.ceil(K / a_block_size))
    expected_w_scale_shape = (math.ceil(N / w_block_size_n), math.ceil(K / w_block_size_k))
    assert a_scale.shape == expected_a_scale_shape, \
        f"a_scale shape mismatch: expected {expected_a_scale_shape}, got {a_scale.shape}"
    assert w_scale.shape == expected_w_scale_shape, \
        f"w_scale shape mismatch: expected {expected_w_scale_shape}, got {w_scale.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
    # Define grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel with optimized defaults for Hopper
    # BLOCK_SIZE_M=128, BLOCK_SIZE_N=256: Good balance for Hopper's register file
    # BLOCK_SIZE_K=128: Matches typical quantization block sizes
    # GROUP_SIZE_M=8: Promotes L2 cache reuse
    # num_stages=3: Pipeline depth for latency hiding
    # num_warps=8: Fully utilizes Hopper's warp schedulers
    w8a8_gemm_kernel[grid](
        a, w, c,
        a_scale, w_scale,
        M, N, K,
        a_block_size, w_block_size_k, w_block_size_n,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=256,
        GROUP_SIZE_M=4
    )
    
    return c


# Test and benchmark function
def test_w8a8_gemm():
    """Test the kernel with sample data."""
    torch.manual_seed(0)
    device = torch.device('cuda')
    
    # Problem size
    M, N, K = 4096, 4096, 4096
    a_block_size = 128
    w_block_size_k = 128
    w_block_size_n = 128
    
    # Create random FP8 tensors
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    
    # Create scales
    a_scale = torch.rand(M, math.ceil(K / a_block_size), device=device, dtype=torch.float32) * 0.1
    w_scale = torch.rand(
        math.ceil(N / w_block_size_n),
        math.ceil(K / w_block_size_k),
        device=device,
        dtype=torch.float32
    ) * 0.1
    
    # Run kernel
    c = w8a8_gemm(a, a_scale, w, w_scale, a_block_size, w_block_size_k, w_block_size_n)
    
    print(f"Output shape: {c.shape}")
    print(f"Output dtype: {c.dtype}")
    print(f"Output range: [{c.min().item():.4f}, {c.max().item():.4f}]")
    
    return c


if __name__ == "__main__":
    test_w8a8_gemm()


# import torch
# import triton
# import triton.language as tl
# import math
# import time
# from typing import Dict, List, Tuple
# import pandas as pd


# @triton.jit
# def w8a8_gemm_kernel(
#     # Pointers to matrices
#     a_ptr, w_ptr, c_ptr,
#     # Pointers to scales
#     a_scale_ptr, w_scale_ptr,
#     # Matrix dimensions
#     M, N, K,
#     # Quantization block sizes
#     a_block_size, w_block_size_k, w_block_size_n,
#     # Strides
#     stride_am, stride_ak,
#     stride_wn, stride_wk,
#     stride_cm, stride_cn,
#     stride_a_scale_m, stride_a_scale_k,
#     stride_w_scale_n, stride_w_scale_k,
#     # Meta-parameters
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
# ):
#     """Optimized kernel for computing C = A @ W^T with blocked quantized FP8 inputs."""
#     # Program ID and block mapping
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # Compute offsets
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)

#     # Initialize pointers
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)

#     # Initialize accumulator
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

#     # Main computation loop
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         k_start = k * BLOCK_SIZE_K
#         k_remaining = K - k_start
        
#         # Compute masks
#         k_mask = offs_k < k_remaining
#         a_mask = (offs_am[:, None] < M) & (k_mask[None, :])
#         w_mask = (offs_bn[:, None] < N) & (k_mask[None, :])
        
#         # Load FP8 data
#         a_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
#         w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
#         # Compute matmul
#         acc_block = tl.dot(a_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
        
#         # Load scales
#         a_k_block_idx = k_start // a_block_size
#         a_scale_ptrs = a_scale_ptr + (offs_am * stride_a_scale_m + a_k_block_idx * stride_a_scale_k)
#         a_scales = tl.load(a_scale_ptrs, mask=offs_am < M, other=1.0)
        
#         w_k_block_idx = k_start // w_block_size_k
#         n_block_indices = offs_bn // w_block_size_n
#         w_scale_ptrs = w_scale_ptr + (n_block_indices * stride_w_scale_n + w_k_block_idx * stride_w_scale_k)
#         w_scales = tl.load(w_scale_ptrs, mask=offs_bn < N, other=1.0)
        
#         # Apply scales
#         scale_factor = a_scales[:, None] * w_scales[None, :]
#         accumulator += acc_block * scale_factor
        
#         # Advance pointers
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         w_ptrs += BLOCK_SIZE_K * stride_wk

#     # Convert to bfloat16
#     c = accumulator.to(tl.bfloat16)

#     # Write output
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# def w8a8_gemm(
#     a: torch.Tensor,
#     a_scale: torch.Tensor,
#     w: torch.Tensor,
#     w_scale: torch.Tensor,
#     a_block_size: int,
#     w_block_size_k: int,
#     w_block_size_n: int,
#     block_size_m: int,
#     block_size_n: int,
#     block_size_k: int,
#     group_size_m: int,
#     num_warps: int,
#     num_stages: int,
# ) -> torch.Tensor:
#     """GEMM with configurable block sizes."""
#     M, K = a.shape
#     N, K_w = w.shape
    
#     c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
#     grid = lambda META: (
#         triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#     )
    
#     w8a8_gemm_kernel[grid](
#         a, w, c,
#         a_scale, w_scale,
#         M, N, K,
#         a_block_size, w_block_size_k, w_block_size_n,
#         a.stride(0), a.stride(1),
#         w.stride(0), w.stride(1),
#         c.stride(0), c.stride(1),
#         a_scale.stride(0), a_scale.stride(1),
#         w_scale.stride(0), w_scale.stride(1),
#         BLOCK_SIZE_M=block_size_m,
#         BLOCK_SIZE_N=block_size_n,
#         BLOCK_SIZE_K=block_size_k,
#         GROUP_SIZE_M=group_size_m,
#         num_warps=num_warps,
#         num_stages=num_stages,
#     )
    
#     return c


# def benchmark_config(
#     a: torch.Tensor,
#     a_scale: torch.Tensor,
#     w: torch.Tensor,
#     w_scale: torch.Tensor,
#     a_block_size: int,
#     w_block_size_k: int,
#     w_block_size_n: int,
#     config: Dict,
#     warmup: int = 10,
#     repetitions: int = 100,
# ) -> Tuple[float, bool]:
#     """Benchmark a specific configuration."""
#     try:
#         # Warmup
#         for _ in range(warmup):
#             c = w8a8_gemm(
#                 a, a_scale, w, w_scale,
#                 a_block_size, w_block_size_k, w_block_size_n,
#                 config['BLOCK_SIZE_M'],
#                 config['BLOCK_SIZE_N'],
#                 config['BLOCK_SIZE_K'],
#                 config['GROUP_SIZE_M'],
#                 config['num_warps'],
#                 config['num_stages'],
#             )
#         torch.cuda.synchronize()
        
#         # Benchmark
#         start = time.time()
#         for _ in range(repetitions):
#             c = w8a8_gemm(
#                 a, a_scale, w, w_scale,
#                 a_block_size, w_block_size_k, w_block_size_n,
#                 config['BLOCK_SIZE_M'],
#                 config['BLOCK_SIZE_N'],
#                 config['BLOCK_SIZE_K'],
#                 config['GROUP_SIZE_M'],
#                 config['num_warps'],
#                 config['num_stages'],
#             )
#         torch.cuda.synchronize()
#         end = time.time()
        
#         avg_time_ms = (end - start) / repetitions * 1000
#         return avg_time_ms, True
    
#     except Exception as e:
#         print(f"Config failed: {config}, Error: {e}")
#         return float('inf'), False


# def grid_search(
#     M: int,
#     N: int,
#     K: int = 7168,
#     a_block_size: int = 128,
#     w_block_size_k: int = 128,
#     w_block_size_n: int = 128,
#     warmup: int = 10,
#     repetitions: int = 100,
# ) -> pd.DataFrame:
#     """
#     Perform grid search over block size configurations.
    
#     Args:
#         M, N, K: Matrix dimensions
#         a_block_size, w_block_size_k, w_block_size_n: Quantization block sizes
#         warmup: Number of warmup iterations
#         repetitions: Number of benchmark iterations
#     """
#     device = torch.device('cuda')
#     torch.manual_seed(42)
    
#     # Create test data
#     print(f"Creating test data: M={M}, N={N}, K={K}")
#     a = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#     w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
#     a_scale = torch.rand(M, math.ceil(K / a_block_size), device=device, dtype=torch.float32) * 0.1
#     w_scale = torch.rand(
#         math.ceil(N / w_block_size_n),
#         math.ceil(K / w_block_size_k),
#         device=device,
#         dtype=torch.float32
#     ) * 0.1
    
#     # Define search space
#     # For K=7168, good BLOCK_SIZE_K candidates: 64, 128, 256
#     # These divide K reasonably: 7168/64=112, 7168/128=56, 7168/256=28
#     configs = []
    
#     # BLOCK_SIZE_M options
#     block_m_options = [32, 64, 128, 256]
#     # BLOCK_SIZE_N options
#     block_n_options = [32, 64, 128, 256]
#     # BLOCK_SIZE_K options - important for K=7168
#     block_k_options = [32, 64, 128, 256]
#     # GROUP_SIZE_M options
#     group_m_options = [4]
#     # num_warps options
#     warp_options = [4]
#     # num_stages options
#     # stage_options = [2, 3, 4, 5]
#     stage_options = [2]
    
#     print("Generating configurations...")
#     for bm in block_m_options:
#         for bn in block_n_options:
#             for bk in block_k_options:
#                 for gm in group_m_options:
#                     for nw in warp_options:
#                         for ns in stage_options:
#                             # Filter out invalid combinations
#                             # Ensure blocks aren't too large for register pressure
#                             if bm * bn > 65536:  # Too large
#                                 continue
#                             # Ensure reasonable warp/stage combinations
#                             if nw == 8 and ns > 4:
#                                 continue
                            
#                             configs.append({
#                                 'BLOCK_SIZE_M': bm,
#                                 'BLOCK_SIZE_N': bn,
#                                 'BLOCK_SIZE_K': bk,
#                                 'GROUP_SIZE_M': gm,
#                                 'num_warps': nw,
#                                 'num_stages': ns,
#                             })
    
#     print(f"Testing {len(configs)} configurations...")
    
#     results = []
#     for i, config in enumerate(configs):
#         if i % 10 == 0:
#             print(f"Progress: {i}/{len(configs)}")
        
#         avg_time_ms, success = benchmark_config(
#             a, a_scale, w, w_scale,
#             a_block_size, w_block_size_k, w_block_size_n,
#             config,
#             warmup=warmup,
#             repetitions=repetitions,
#         )
        
#         if success:
#             # Calculate TFLOPS
#             # For FP8 matmul: 2*M*N*K operations (multiply-add)
#             flops = 2 * M * N * K
#             tflops = (flops / (avg_time_ms / 1000)) / 1e12
            
#             results.append({
#                 'BLOCK_SIZE_M': config['BLOCK_SIZE_M'],
#                 'BLOCK_SIZE_N': config['BLOCK_SIZE_N'],
#                 'BLOCK_SIZE_K': config['BLOCK_SIZE_K'],
#                 'GROUP_SIZE_M': config['GROUP_SIZE_M'],
#                 'num_warps': config['num_warps'],
#                 'num_stages': config['num_stages'],
#                 'time_ms': avg_time_ms,
#                 'TFLOPS': tflops,
#             })
    
#     # Create DataFrame and sort by performance
#     df = pd.DataFrame(results)
#     df = df.sort_values('TFLOPS', ascending=False)
    
#     return df


# def run_grid_search():
#     """Run grid search with typical LLM dimensions."""
#     print("="*80)
#     print("FP8-FP8 W8A8 GEMM Grid Search (K=7168)")
#     print("="*80)
    
#     # Common LLM dimensions
#     test_cases = [
#         # (M, N, K)
#         (32, 1536, 7168),
#         (32, 24576, 1536),

#         # (8192, 8192, 7168),
#         # (2048, 2048, 7168),
#     ]
    
#     all_results = {}
    
#     for M, N, K in test_cases:
#         print(f"\n{'='*80}")
#         print(f"Testing M={M}, N={N}, K={K}")
#         print(f"{'='*80}\n")
        
#         df = grid_search(M, N, K, warmup=5, repetitions=50)
#         all_results[f"M{M}_N{N}_K{K}"] = df
        
#         print(f"\n{'='*80}")
#         print(f"Top 10 configurations for M={M}, N={N}, K={K}:")
#         print(f"{'='*80}")
#         print(df.head(10).to_string(index=False))
#         print(f"\nBest config: {df.iloc[0]['TFLOPS']:.2f} TFLOPS")
#         print(f"Worst config: {df.iloc[-1]['TFLOPS']:.2f} TFLOPS")
#         print(f"Speedup: {df.iloc[0]['TFLOPS'] / df.iloc[-1]['TFLOPS']:.2f}x")
    
#     # Find most consistent best config across all test cases
#     print(f"\n{'='*80}")
#     print("Finding best overall configuration...")
#     print(f"{'='*80}\n")
    
#     # Combine rankings
#     config_scores = {}
#     for test_name, df in all_results.items():
#         for idx, row in df.iterrows():
#             config_key = (
#                 row['BLOCK_SIZE_M'],
#                 row['BLOCK_SIZE_N'],
#                 row['BLOCK_SIZE_K'],
#                 row['GROUP_SIZE_M'],
#                 row['num_warps'],
#                 row['num_stages'],
#             )
#             if config_key not in config_scores:
#                 config_scores[config_key] = []
#             config_scores[config_key].append(row['TFLOPS'])
    
#     # Find config with best average performance
#     best_config = max(config_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))
    
#     print("Best overall configuration:")
#     print(f"  BLOCK_SIZE_M: {best_config[0][0]}")
#     print(f"  BLOCK_SIZE_N: {best_config[0][1]}")
#     print(f"  BLOCK_SIZE_K: {best_config[0][2]}")
#     print(f"  GROUP_SIZE_M: {best_config[0][3]}")
#     print(f"  num_warps: {best_config[0][4]}")
#     print(f"  num_stages: {best_config[0][5]}")
#     print(f"  Average TFLOPS: {sum(best_config[1]) / len(best_config[1]):.2f}")
    
#     return all_results, best_config


# if __name__ == "__main__":
#     results, best_config = run_grid_search()