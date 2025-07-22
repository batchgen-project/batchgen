import math
import os
import time

import torch
import triton
import triton.language as tl

"""
	Step 1:
		torch.einsum('hdc,bhd->bhc', self.q_absorb, q_nope)
	Step 2:
		Store result to C[:, :, : self.kv_lora_rank]
	Step 3:
		Fuse weight dequant.
"""


@triton.jit
def fused_bhd_hdc_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    B,
    H,
    D: tl.constexpr,
    C: tl.constexpr,
    # Strides for A, B, and C
    stride_a_b,
    stride_a_h,
    stride_a_d,
    stride_b_h,
    stride_b_d,
    stride_b_c,
    stride_c_b,
    stride_c_h,
    stride_c_c,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    Fused kernel for matrix multiplication with BHD and HDC formats.
    BHC = BHD @ HDC
    C[:, h, :] = A[:, h, :] @ B[h, :, :] -> [b,d] @ [d,c]
    The workload is split along the h dimension to thread-blocks.
    Args:
            a_ptr (tl.pointer): Pointer to the first matrix (BHD layout).
            b_ptr (tl.pointer): Pointer to the second matrix (HDC layout).
            c_ptr (tl.pointer): Pointer to the output matrix (BHC layout).
    """
    pid = tl.program_id(axis=0)  # Manage index h.
    num_b_blocks = tl.cdiv(
        B, BLOCK_SIZE_B
    )  # Number of blocks along h dimension.
    offsets_b = (
        pid * stride_b_h
        + tl.arange(0, D)[:, None] * stride_b_d
        + tl.arange(0, C)[None, :] * stride_b_c
    )
    # Load B
    b = tl.load(b_ptr + offsets_b)
    for block in range(num_b_blocks):
        # Load the b-th block of A.
        offsets_a = (
            block * BLOCK_SIZE_B
            + tl.arange(0, BLOCK_SIZE_B)[:, None] * stride_a_b
            + pid * stride_a_h
            + tl.arange(0, D)[None, :] * stride_a_d
        )
        a = tl.load(
            a_ptr + offsets_a,
            mask=(
                block * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[:, None] < B
            ),
            other=0.0,
        )
        # Perform the dot product
        c = tl.dot(a, b)
        # Store the result in C
        offsets_c = (
            block * BLOCK_SIZE_B
            + tl.arange(0, BLOCK_SIZE_B)[:, None] * stride_c_b
            + pid * stride_c_h
            + tl.arange(0, C)[None, :] * stride_c_c
        )
        tl.store(
            c_ptr + offsets_c,
            c,
            mask=(
                block * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[:, None] < B
            ),
        )


def fused_bhd_hdc(
    LHS: torch.Tensor,  # BHD format
    RHS: torch.Tensor,  # HDC format
):
    assert LHS.dim() == 3, "A must be a 3D tensor in BHD format"
    assert RHS.dim() == 3, "B must be a 3D tensor in HDC format"
    B, H, D = LHS.shape
    H_, D_, C = RHS.shape
    assert H == H_, "The second dimension of A and B must match"
    assert D == D_, "The third dimension of A and B must match"

    result = torch.empty((B, H, C), dtype=LHS.dtype, device=LHS.device)

    # Launch H programs, each handling one h dimension
    fused_bhd_hdc_kernel[(H,)](
        LHS,
        RHS,
        result,
        B,
        H,
        D,
        C,
        LHS.stride(0),
        LHS.stride(1),
        LHS.stride(2),
        RHS.stride(0),
        RHS.stride(1),
        RHS.stride(2),
        result.stride(0),
        result.stride(1),
        result.stride(2),
        BLOCK_SIZE_B=16,  # Adjust block size as needed
    )
    return result


@triton.jit
def fused_bhd_hdc_kernel_with_c_offset(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    B,
    H,
    D: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    # Strides for A, B, and C
    stride_a_b,
    stride_a_h,
    stride_a_d,
    stride_b_h,
    stride_b_d,
    stride_b_c,
    stride_c_b,
    stride_c_h,
    stride_c_c,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    Fused kernel for matrix multiplication with BHD and HDC formats.
    BHC = BHD @ HDC
    C[:, h, :] = A[:, h, :] @ B[h, :, :] -> [b,d] @ [d,c]
    The workload is split along the h dimension to thread-blocks.
    Args:
            a_ptr (tl.pointer): Pointer to the first matrix (BHD layout).
            b_ptr (tl.pointer): Pointer to the second matrix (HDC layout).
            c_ptr (tl.pointer): Pointer to the output matrix (BHC layout).
    """
    pid = tl.program_id(axis=0)  # Manage index h.
    num_b_blocks = tl.cdiv(
        B, BLOCK_SIZE_B
    )  # Number of blocks along h dimension.
    offsets_b = (
        pid * stride_b_h
        + tl.arange(0, D)[:, None] * stride_b_d
        + tl.arange(0, C)[None, :] * stride_b_c
    )
    # Load B
    b = tl.load(b_ptr + offsets_b)
    for block in range(num_b_blocks):
        # Load the b-th block of A.
        offsets_a = (
            block * BLOCK_SIZE_B
            + tl.arange(0, BLOCK_SIZE_B)[:, None] * stride_a_b
            + pid * stride_a_h
            + tl.arange(0, D)[None, :] * stride_a_d
        )
        a = tl.load(
            a_ptr + offsets_a,
            mask=(
                block * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[:, None] < B
            ),
            other=0.0,
        )
        # Perform the dot product
        c = tl.dot(a, b)
        # Store the result in C
        offsets_c = (
            block * BLOCK_SIZE_B
            + tl.arange(0, BLOCK_SIZE_B)[:, None] * stride_c_b
            + pid * stride_c_h
            + (K + tl.arange(0, C))[None, :] * stride_c_c
        )
        tl.store(
            c_ptr + offsets_c,
            c,
            mask=(
                block * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)[:, None] < B
            ),
        )


def fused_bhd_hdc_inplace(
    LHS: torch.Tensor,  # BHD format
    RHS: torch.Tensor,  # HDC format
    result: torch.Tensor,  # BH(C+K) format with offset
):
    assert LHS.dim() == 3, "A must be a 3D tensor in BHD format"
    assert RHS.dim() == 3, "B must be a 3D tensor in HDC format"
    B, H, D = LHS.shape
    H_, D_, C = RHS.shape
    assert H == H_, "The second dimension of A and B must match"
    assert D == D_, "The third dimension of A and B must match"

    # result = torch.empty((B, H, C), dtype=LHS.dtype, device=LHS.device)

    # Launch H programs, each handling one h dimension
    fused_bhd_hdc_kernel[(H,)](
        LHS,
        RHS,
        result,
        B,
        H,
        D,
        C,
        LHS.stride(0),
        LHS.stride(1),
        LHS.stride(2),
        RHS.stride(0),
        RHS.stride(1),
        RHS.stride(2),
        result.stride(0),
        result.stride(1),
        result.stride(2),
        BLOCK_SIZE_B=16,  # Adjust block size as needed
    )
    # return result
