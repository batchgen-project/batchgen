import logging
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

# os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.triton/cache")
# os.environ["TRITON_CACHE_MANAGER"] = "1"


# @triton.autotune(
# 	configs=[
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 64, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 64}),
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 64, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 128}),
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 64, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 32}),
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 32, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 64}),
# 		triton.Config({'GEMM_BLOCK_SIZE_M': 32, 'GEMM_BLOCK_SIZE_N': 16, 'GEMM_BLOCK_SIZE_K': 32}),
# 	],
# 	key=['N', 'K'],  # Autotune based on these input shapes
# 	# warmup=25,            # Number of warmup iterations
# 	# rep=100,              # Number of measurement iterations
# 	use_cuda_graph=True   # Use CUDA graphs for more accurate timing
# )
@triton.jit
def fused_dequant_weighted_moe_stage_1_kernel(
    lhs_ptr,
    gate_ptrs_ptr,
    up_ptrs_ptr,
    gate_scale_ptrs_ptr,
    up_scale_ptrs_ptr,
    group_idx_ptr,
    group_sizes_ptr,
    group_start_indices_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    num_groups,
    stride_lhs_m,
    stride_lhs_k,
    stride_gate_n,
    stride_gate_k,
    stride_up_n,
    stride_up_k,
    stride_output_m,
    stride_output_n,
    stride_group_idx,
    stride_group_sizes,
    stride_group_start_indices,
    stride_weight_ptrs,
    stride_scale_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    """
    # (act(hidden_states @ deq(gate)) * (hidden_states @ deq(up))
    Note: act - silu. And we assume the gate and up weights have the same shape which is common in MoE models.
    """
    pid = tl.program_id(axis=0)
    lhs_dtype = tl.bfloat16
    rhs_dtype = tl.float8e4nv
    scale_dtype = tl.float32
    acc_dtype = tl.float32

    offsets_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
    scale_n = pid * GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    for g in range(num_groups):
        # Get group size: gm
        gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
        group_idx = tl.load(
            group_idx_ptr + g * stride_group_idx
        )  # Which group we are working on.
        # Get row indices for the current group.
        start_idx = tl.load(
            group_start_indices_ptr + g * stride_group_start_indices
        )
        num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)

        # We have determined the rhs. So we do the base pointer calculation here.
        gate_base_ptr = tl.load(
            gate_ptrs_ptr + group_idx * stride_weight_ptrs
        ).to(tl.pointer_type(rhs_dtype))
        up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(
            tl.pointer_type(rhs_dtype)
        )
        gate_scale_base_ptr = tl.load(
            gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs
        ).to(tl.pointer_type(scale_dtype))
        up_scale_base_ptr = tl.load(
            up_scale_ptrs_ptr + group_idx * stride_scale_ptrs
        ).to(tl.pointer_type(scale_dtype))

        for sub_group_idx in range(num_sub_groups):
            # Calculate the base pointer for the current sub-group
            sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
            # Remaining rows in the group:
            remaining_rows_in_group = start_idx + gm - sub_group_start_idx
            valid_rows_this_block = tl.minimum(
                GEMM_BLOCK_SIZE_M, remaining_rows_in_group
            )

            # base_ptr_sub_g = lhs_ptr + sub_group_start_idx * stride_lhs_m
            # # Process the associated tile
            offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)

            # Loop along K dimension
            gate_acc = tl.zeros(
                (GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype
            )
            up_acc = tl.zeros(
                (GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype
            )
            for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
                offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(
                    0, GEMM_BLOCK_SIZE_K
                )
                # Create pointers for lhs and rhs
                abs_row_indices = sub_group_start_idx + offsets_m
                # lhs_ptrs = base_ptr_sub_g + (offsets_m[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
                lhs_ptrs = lhs_ptr + (
                    abs_row_indices[:, None] * stride_lhs_m
                    + offsets_k[None, :] * stride_lhs_k
                )
                # rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
                gate_ptrs = gate_base_ptr + (
                    offsets_n[:, None] * stride_gate_n
                    + offsets_k[None, :] * stride_gate_k
                )
                up_ptrs = up_base_ptr + (
                    offsets_n[:, None] * stride_up_n
                    + offsets_k[None, :] * stride_up_k
                )

                # Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
                # Find out which scale block this tile is on:
                scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
                # scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
                gate_scale_ptr = gate_scale_base_ptr + (
                    scale_n * num_scale_k + scale_k
                )
                up_scale_ptr = up_scale_base_ptr + (
                    scale_n * num_scale_k + scale_k
                )
                # Load the scale for this tile
                gate_scale = tl.load(gate_scale_ptr)
                up_scale = tl.load(up_scale_ptr)

                # Create masks for lhs and rhs
                lhs_mask = (
                    (abs_row_indices[:, None] < M)
                    & (offsets_k[None, :] < K)
                    & (offsets_m[:, None] < valid_rows_this_block)
                )
                rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

                # Load rhs tile:
                gate_fp8 = tl.load(
                    gate_ptrs, mask=rhs_mask, other=0.0, cache_modifier=".cg"
                )
                up_fp8 = tl.load(
                    up_ptrs, mask=rhs_mask, other=0.0, cache_modifier=".cg"
                )
                lhs = tl.load(
                    lhs_ptrs, mask=lhs_mask, other=0.0, cache_modifier=".cg"
                )

                gate_fp32 = tl.cast(gate_fp8, tl.float32)
                gate_scaled = gate_fp32 * gate_scale
                gate_bf16 = tl.cast(gate_scaled, lhs_dtype)
                gate_acc += tl.dot(lhs, tl.trans(gate_bf16))
                # gate_acc = tl.dot(lhs, tl.trans(gate_bf16), acc=gate_acc)

                up_fp32 = tl.cast(up_fp8, tl.float32)
                up_scaled = up_fp32 * up_scale
                up_bf16 = tl.cast(up_scaled, lhs_dtype)
                up_acc += tl.dot(lhs, tl.trans(up_bf16))
                # up_acc = tl.dot(lhs, tl.trans(up_bf16), acc=up_acc)

            # Store the result
            offs_output_m = sub_group_start_idx + tl.arange(
                0, GEMM_BLOCK_SIZE_M
            )
            offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(
                0, GEMM_BLOCK_SIZE_N
            )
            output_ptrs = output_ptr + (
                offs_output_m[:, None] * stride_output_m
                + offs_output_n[None, :] * stride_output_n
            )
            # output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
            output_mask = (
                (offs_output_m[:, None] < M)
                & (offs_output_n[None, :] < N)
                & (
                    tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None]
                    < valid_rows_this_block
                )
            )

            # Convert to bf16 before storing
            output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
            output = tl.cast(output_acc, lhs_dtype)
            tl.store(output_ptrs, output, mask=output_mask)


@torch.inference_mode()
def fused_dequant_weighted_moe_stage_1(
    hidden_states: torch.Tensor,
    gate_weight_list: list[torch.Tensor],
    up_weight_list: list[torch.Tensor],
    gate_scale_list: list[torch.Tensor],
    up_scale_list: list[torch.Tensor],
    group_sizes: tuple[int, int],
    group_start_indices: torch.Tensor,
    gate_gemm_block_size=[64, 16, 128],
    up_gemm_block_size=[64, 16, 128],
    scale_block_size=[128, 128],
):
    assert (
        hidden_states.dtype == torch.bfloat16
    ), "hidden_states must be of dtype bfloat16"
    assert all(
        r.dtype == torch.float8_e4m3fn for r in gate_weight_list
    ), "All gate weights must be of dtype float8_e4m3fn"
    assert all(
        r.dtype == torch.float8_e4m3fn for r in up_weight_list
    ), "All up weights must be of dtype float8_e4m3fn"
    assert all(
        s.dtype == torch.float32 for s in gate_scale_list
    ), "All gate scales must be of dtype float32"
    assert all(
        s.dtype == torch.float32 for s in up_scale_list
    ), "All up scales must be of dtype float32"
    assert len(gate_weight_list) == len(
        gate_scale_list
    ), "gate_weight_list and gate_scale_list must have the same length"
    assert len(up_weight_list) == len(
        up_scale_list
    ), "up_weight_list and up_scale_list must have the same length"

    device = hidden_states.device
    M = hidden_states.shape[0]
    N = gate_weight_list[0].shape[0]
    K = hidden_states.shape[1]

    gate_ptrs_ptr = torch.tensor(
        [r.data_ptr() for r in gate_weight_list],
        dtype=torch.int64,
        device=device,
    )
    up_ptrs_ptr = torch.tensor(
        [r.data_ptr() for r in up_weight_list], dtype=torch.int64, device=device
    )
    gate_scale_ptrs_ptr = torch.tensor(
        [s.data_ptr() for s in gate_scale_list],
        dtype=torch.int64,
        device=device,
    )
    up_scale_ptrs_ptr = torch.tensor(
        [s.data_ptr() for s in up_scale_list], dtype=torch.int64, device=device
    )
    group_size = torch.tensor(
        [size for _, size in group_sizes], dtype=torch.int32, device=device
    )
    activated_group_idx = torch.tensor(
        [idx for idx, _ in group_sizes], dtype=torch.int32, device=device
    )
    num_groups = len(group_sizes)

    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    grid = lambda META: (triton.cdiv(N, META["GEMM_BLOCK_SIZE_N"]),)
    # Launch the kernel
    try:
        fused_dequant_weighted_moe_stage_1_kernel[grid](
            hidden_states,
            gate_ptrs_ptr,
            up_ptrs_ptr,
            gate_scale_ptrs_ptr,
            up_scale_ptrs_ptr,
            activated_group_idx,
            group_size,
            group_start_indices,
            output,
            M,
            N,
            K,
            num_groups,
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_weight_list[0].stride(0),
            gate_weight_list[0].stride(1),
            up_weight_list[0].stride(0),
            up_weight_list[0].stride(1),
            output.stride(0),
            output.stride(1),
            activated_group_idx.stride(0),
            group_size.stride(0),
            group_start_indices.stride(0),
            gate_ptrs_ptr.stride(0),
            gate_scale_ptrs_ptr.stride(0),
            gate_gemm_block_size[0],
            gate_gemm_block_size[1],
            gate_gemm_block_size[2],
            SCALE_BLOCK_SIZE_N=scale_block_size[0],
            SCALE_BLOCK_SIZE_K=scale_block_size[1],
        )
    except Exception as e:
        logging.error(f"Error in fused_dequant_weighted_moe_stage_1: {e}")
        raise
    return output


@triton.jit
def fused_dequant_weighted_moe_kernel(
    lhs_ptr,
    gate_ptrs_ptr,
    up_ptrs_ptr,
    down_ptrs_ptr,
    gate_scale_ptrs_ptr,
    up_scale_ptrs_ptr,
    down_scale_ptrs_ptr,
    group_idx_ptr,
    group_sizes_ptr,
    group_start_indices_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    num_groups,
    stride_lhs_m,
    stride_lhs_k,
    stride_gate_n,
    stride_gate_k,
    stride_up_n,
    stride_up_k,
    stride_down_n,
    stride_down_k,
    stride_output_m,
    stride_output_n,
    stride_grouped_idx,
    stride_group_sizes,
    stride_group_start_indices,
    stride_weight_ptrs,
    stride_scale_ptrs,
    GATE_GEMM_BLOCK_SIZE_M: tl.constexpr,
    GATE_GEMM_BLOCK_SIZE_N: tl.constexpr,
    GATE_GEMM_BLOCK_SIZE_K: tl.constexpr,
    UP_GEMM_BLOCK_SIZE_M: tl.constexpr,
    UP_GEMM_BLOCK_SIZE_N: tl.constexpr,
    UP_GEMM_BLOCK_SIZE_K: tl.constexpr,
    DOWN_GEMM_BLOCK_SIZE_M: tl.constexpr,
    DOWN_GEMM_BLOCK_SIZE_N: tl.constexpr,
    DOWN_GEMM_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    lhs_dtype = tl.bfloat16
    rhs_dtype = tl.float8e4nv
    scale_dtype = tl.float32
    acc_dtype = tl.float32

    # ----1) (act(hidden_states @ deq(gate)) * (hidden_states @ deq(up))
    offsets_n = pid * GATE_GEMM_BLOCK_SIZE_M + tl.arange(
        0, GATE_GEMM_BLOCK_SIZE_N
    )
    scale_n = pid * GATE_GEMM_BLOCK_SIZE_N // SCALE_BLOCK_SIZE_N
    num_scale_k = tl.cdiv(K, SCALE_BLOCK_SIZE_K)
    for g in range(num_groups):
        # Get group size: gm
        gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
        group_idx = tl.load(
            group_idx_ptr + g * stride_group_idx
        )  # Which group we are working on.
        # Get row indices for the current group.
        start_idx = tl.load(
            group_start_indices_ptr + g * stride_group_start_indices
        )
        num_sub_groups = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)

        # We have determined the rhs. So we do the base pointer calculation here.
        gate_base_ptr = tl.load(
            gate_ptrs_ptr + group_idx * stride_weight_ptrs
        ).to(tl.pointer_type(rhs_dtype))
        up_base_ptr = tl.load(up_ptrs_ptr + group_idx * stride_weight_ptrs).to(
            tl.pointer_type(rhs_dtype)
        )
        gate_scale_base_ptr = tl.load(
            gate_scale_ptrs_ptr + group_idx * stride_scale_ptrs
        ).to(tl.pointer_type(scale_dtype))
        up_scale_base_ptr = tl.load(
            up_scale_ptrs_ptr + group_idx * stride_scale_ptrs
        ).to(tl.pointer_type(scale_dtype))

        for sub_group_idx in range(num_sub_groups):
            # Calculate the base pointer for the current sub-group
            sub_group_start_idx = start_idx + sub_group_idx * GEMM_BLOCK_SIZE_M
            # Remaining rows in the group:
            remaining_rows_in_group = start_idx + gm - sub_group_start_idx
            valid_rows_this_block = tl.minimum(
                GEMM_BLOCK_SIZE_M, remaining_rows_in_group
            )

            # base_ptr_sub_g = lhs_ptr + sub_group_start_idx * stride_lhs_m
            # # Process the associated tile
            offsets_m = tl.arange(0, GEMM_BLOCK_SIZE_M)

            # Loop along K dimension
            gate_acc = tl.zeros(
                (GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype
            )
            up_acc = tl.zeros(
                (GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=acc_dtype
            )
            for k_idx in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
                offsets_k = k_idx * GEMM_BLOCK_SIZE_K + tl.arange(
                    0, GEMM_BLOCK_SIZE_K
                )
                # Create pointers for lhs and rhs
                abs_row_indices = sub_group_start_idx + offsets_m
                # lhs_ptrs = base_ptr_sub_g + (offsets_m[:, None] * stride_lhs_m + offsets_k[None, :] * stride_lhs_k)
                lhs_ptrs = lhs_ptr + (
                    abs_row_indices[:, None] * stride_lhs_m
                    + offsets_k[None, :] * stride_lhs_k
                )
                # rhs_ptrs = rhs_base_ptr + (offsets_n[:, None] * stride_rhs_n + offsets_k[None, :] * stride_rhs_k)
                gate_ptrs = gate_base_ptr + (
                    offsets_n[:, None] * stride_gate_n
                    + offsets_k[None, :] * stride_gate_k
                )
                up_ptrs = up_base_ptr + (
                    offsets_n[:, None] * stride_up_n
                    + offsets_k[None, :] * stride_up_k
                )

                # Note the N,K tile size are <= scale_block_size. So there would not be a tile crossing the scale block boundary.
                # Find out which scale block this tile is on:
                scale_k = k_idx * GEMM_BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
                # scale_ptr = rhs_scale_base_ptr + (scale_n * num_scale_k + scale_k)
                gate_scale_ptr = gate_scale_base_ptr + (
                    scale_n * num_scale_k + scale_k
                )
                up_scale_ptr = up_scale_base_ptr + (
                    scale_n * num_scale_k + scale_k
                )
                # Load the scale for this tile
                # scale = tl.load(scale_ptr)
                gate_scale = tl.load(gate_scale_ptr)
                up_scale = tl.load(up_scale_ptr)

                # Create masks for lhs and rhs
                # lhs_mask = (offsets_m[:, None] < valid_rows_this_block) & (offsets_k[None, :] < K)
                lhs_mask = (
                    (abs_row_indices[:, None] < M)
                    & (offsets_k[None, :] < K)
                    & (offsets_m[:, None] < valid_rows_this_block)
                )
                rhs_mask = (offsets_n[:, None] < N) & (offsets_k[None, :] < K)

                # Load rhs tile:
                # rhs_fp8 = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0, cache_modifier='.cg')
                gate_fp8 = tl.load(
                    gate_ptrs, mask=rhs_mask, other=0.0, cache_modifier=".cg"
                )
                up_fp8 = tl.load(
                    up_ptrs, mask=rhs_mask, other=0.0, cache_modifier=".cg"
                )

                gate_fp32 = tl.cast(gate_fp8, tl.float32)
                gate_scaled = gate_fp32 * gate_scale
                gate_bf16 = tl.cast(gate_scaled, lhs_dtype)

                lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
                gate_acc += tl.dot(lhs, tl.trans(gate_bf16))
                # act_out = intermediate / (1.0 + tl.exp(-intermediate))  # Silu activation

                up_fp32 = tl.cast(up_fp8, tl.float32)
                up_scaled = up_fp32 * up_scale
                up_bf16 = tl.cast(up_scaled, lhs_dtype)
                up_acc += tl.dot(lhs, tl.trans(up_bf16))

                # acc += act_out * up_out

            # Store the result
            offs_output_m = sub_group_start_idx + tl.arange(
                0, GEMM_BLOCK_SIZE_M
            )
            offs_output_n = pid * GEMM_BLOCK_SIZE_N + tl.arange(
                0, GEMM_BLOCK_SIZE_N
            )
            output_ptrs = output_ptr + (
                offs_output_m[:, None] * stride_output_m
                + offs_output_n[None, :] * stride_output_n
            )
            # output_mask = (offs_output_m[:, None] < sub_group_start_idx + valid_rows_this_block) & (offs_output_n[None, :] < N)
            output_mask = (
                (offs_output_m[:, None] < M)
                & (offs_output_n[None, :] < N)
                & (
                    tl.arange(0, GEMM_BLOCK_SIZE_M)[:, None]
                    < valid_rows_this_block
                )
            )

            # Convert to bf16 before storing
            output_acc = gate_acc / (1.0 + tl.exp(-gate_acc)) * up_acc
            output = tl.cast(output_acc, lhs_dtype)
            tl.store(output_ptrs, output, mask=output_mask)
