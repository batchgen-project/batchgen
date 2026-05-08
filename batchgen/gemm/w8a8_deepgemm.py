import torch
import deep_gemm


def _tma_aligned_m(rows: int, element_size: int) -> int:
    alignment = 16 // element_size
    return ((rows + alignment - 1) // alignment) * alignment


def _as_one_group_lhs_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    if scale.dim() == 3:
        grouped = scale
    elif scale.dim() == 2:
        aligned_m = _tma_aligned_m(rows, scale.element_size())
        if scale.stride(0) != 1 or scale.stride(1) != aligned_m:
            raise ValueError(
                "masked DeepGEMM requires TMA-aligned column-major lhs scales; "
                "call act_quant(..., scale_tma_aligned=True)"
            )
        grouped = scale.as_strided(
            (1, scale.shape[0], scale.shape[1]),
            (aligned_m * scale.shape[1], 1, aligned_m),
        )
    else:
        raise ValueError(
            f"masked DeepGEMM lhs scales must be rank 2 or 3, got {scale.dim()}"
        )
    aligned_m = _tma_aligned_m(rows, scale.element_size())
    if grouped.shape[0] != 1 or grouped.shape[1] != rows:
        raise ValueError(f"lhs scale shape must be [1,{rows},Kb], got {tuple(grouped.shape)}")
    if grouped.stride(1) != 1 or grouped.stride(2) != aligned_m:
        raise ValueError(
            "masked DeepGEMM lhs scale is not TMA-aligned column-major; "
            f"shape={tuple(grouped.shape)} stride={grouped.stride()} expected stride[1]=1 stride[2]={aligned_m}"
        )
    return grouped


def _as_one_group_rhs_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 3:
        if scale.shape[0] != 1:
            raise ValueError(f"rhs scale group dimension must be 1, got {tuple(scale.shape)}")
        return scale
    if scale.dim() == 2:
        return scale.unsqueeze(0)
    raise ValueError(f"masked DeepGEMM rhs scales must be rank 2 or 3, got {scale.dim()}")


def _masked_fp8_grouped_gemm():
    for name in (
        "m_grouped_gemm_fp8_fp8_bf16_nt_masked",
        "m_grouped_fp8_gemm_nt_masked",
        "fp8_m_grouped_gemm_nt_masked",
    ):
        fn = getattr(deep_gemm, name, None)
        if fn is not None:
            return fn
    raise RuntimeError(
        "installed deep_gemm does not expose a masked grouped FP8 NT GEMM"
    )


def w8a8_deepgemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    c: torch.Tensor = None,  # Optional accumulation tensor
    disable_ue8m0_cast: bool = True,  # UE8M0 optimization control
    recipe: tuple = None,  # Optional recipe for kernel config
    out: torch.Tensor = None,  # Optional output tensor
    num_valid_tokens: torch.Tensor = None,  # Optional masked-M device scalar
    expected_m: int = None,  # Optional masked DeepGEMM CPU performance hint
) -> torch.Tensor:
    """
    Advanced W8A8 GEMM with optional accumulation and settings.
    """
    M, K = a.shape
    N = w.shape[0]

    # Allocate output
    if out is None:
        output = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    else:
        output = out

    if num_valid_tokens is not None:
        if c is not None:
            raise ValueError("masked DeepGEMM path does not support accumulation tensor c")
        if recipe is not None:
            raise ValueError("masked DeepGEMM path does not support recipe override")
        masked_gemm = _masked_fp8_grouped_gemm()
        if num_valid_tokens.device != a.device:
            raise ValueError("num_valid_tokens must be on the same device as a")
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}")
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                f"num_valid_tokens must contain one element, got {tuple(num_valid_tokens.shape)}"
            )
        if not a.is_contiguous() or not w.is_contiguous() or not output.is_contiguous():
            raise ValueError("masked DeepGEMM requires contiguous a, w, and output tensors")
        if not num_valid_tokens.is_contiguous():
            raise ValueError("num_valid_tokens must be contiguous")
        lhs_scale = _as_one_group_lhs_scale(a_scale, M)
        rhs_scale = _as_one_group_rhs_scale(w_scale)
        masked_gemm(
            (a.view(1, M, K), lhs_scale),
            (w.view(1, N, K), rhs_scale),
            output.view(1, M, N),
            num_valid_tokens,
            int(expected_m or M),
        )
        return output if out is None else None

    # Ensure contiguity
    a = a.contiguous()
    w = w.contiguous()
    a_scale = a_scale.contiguous()
    w_scale = w_scale.contiguous()

    lhs = (a, a_scale)
    rhs = (w, w_scale)

    # Call with optional parameters
    deep_gemm.fp8_gemm_nt(
        lhs,
        rhs,
        output,
        disable_ue8m0_cast=disable_ue8m0_cast,
        recipe=recipe
    )

    return output if out is None else None
