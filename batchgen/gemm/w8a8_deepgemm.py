import torch
import deep_gemm
import logging

def w8a8_deepgemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    c: torch.Tensor = None,  # Optional accumulation tensor
    disable_ue8m0_cast: bool = True,  # UE8M0 optimization control
    recipe: tuple = None,  # Optional recipe for kernel config
    out: torch.Tensor = None  # Optional output tensor
) -> torch.Tensor:
    """
    Advanced W8A8 GEMM with optional accumulation and settings.
    """
    M, K = a.shape
    N = w.shape[0]

    # DEBUG: Log tensor dimensions to diagnose deep_gemm assertion errors
    logging.info(
        f"[w8a8_deepgemm] a.shape={a.shape}, a.dim={a.dim()}, a.dtype={a.dtype}, "
        f"a_scale.shape={a_scale.shape}, a_scale.dim={a_scale.dim()}, "
        f"w.shape={w.shape}, w.dim={w.dim()}, w.dtype={w.dtype}, "
        f"w_scale.shape={w_scale.shape}, w_scale.dim={w_scale.dim()}"
    )

    # Ensure contiguity
    a = a.contiguous()
    w = w.contiguous()
    a_scale = a_scale.contiguous()
    w_scale = w_scale.contiguous()

    lhs = (a, a_scale)
    rhs = (w, w_scale)

    # Allocate output
    if out is None:
        output = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    else:
        output = out

    # Call with optional parameters
    deep_gemm.fp8_gemm_nt(
        lhs,
        rhs,
        output,
        disable_ue8m0_cast=disable_ue8m0_cast,
        recipe=recipe
    )

    return output if out is None else None