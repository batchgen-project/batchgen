"""Marlin grouped MoE Stage 1 kernel wrapper — zero per-step overhead.

Two kernel variants:
- M8 (v12c): mma_trans, MBLOCK=8, decode M<=8. 80 regs, 32% occ, ~179us.
- M16 (v14): standard mma, MBLOCK=16, CTA M-tiling for any M. 130 regs, ~318us.
  Grid: num_matrices × max_m_tiles × n_tiles. GPU-side expert_counts dispatch.

Both use GROUP_BLOCKS=2 (gs=32, K2.5 native).
All buffers and pointer arrays pre-computed at init time.
Per-step forward: 2 kernel launches (GEMM + SiLU), zero Python loops or allocations.

K2.5 wrappers above keep that zero-overhead contract. The K3 MXFP4 wrappers
below add host-side hard-fail contract checks per the 2026-08-04 ledger —
cheap (a handful of attribute reads per launch, no device work) but not zero;
if the model-side integration calls them per-step in eager mode, hoisting the
static checks (L1/L4) to plan build is a named integration follow-up.
"""

import logging

import torch

from batchgen_kernels import load_extension as _load_extension

try:
    _module = _load_extension("batchgen_kernels.moe._C_marlin_grouped_gemm")
except Exception as _e:  # kernel unavailable (AOT-only env without a build)
    logging.warning("Marlin grouped GEMM kernel unavailable: %s", _e)
    _module = None

_warned_m8 = False
_warned_m16 = False


def _load_module():
    """Return pre-compiled Marlin grouped GEMM kernel from batchgen_kernels."""
    return _module


def is_marlin_available() -> bool:
    return _module is not None


def marlin_grouped_stage1_3d_inplace(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    B_ptrs: torch.Tensor,
    scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    gate_buf: torch.Tensor,
    up_buf: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    compact_stride: int = 16,
) -> None:
    """M8 path: Marlin S1 for decode (M<=8 per expert).

    Writes gate+up to compact buffers, then scatter SiLU to intermediate.
    """
    global _warned_m8
    if not _warned_m8:
        logging.info("[Marlin] Using M8 Marlin W4A16 grouped GEMM for decode S1")
        _warned_m8 = True

    mod = _load_module()
    E = expert_counts.shape[0]
    n_tiles = N // 256
    mtp = intermediate_3d.shape[0] // E

    # Launch 1: Grouped GEMM — all 2E matrices in one kernel
    mod.grouped_marlin_gemm(
        dispatched_x_3d, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, 2 * E, n_tiles,
    )

    # Launch 2: SiLU with scatter — compact gate/up → mtp-strided intermediate
    mod.silu_mul_scatter(
        gate_buf, up_buf, intermediate_3d, expert_counts,
        E, compact_stride, mtp, N,
    )


def marlin_grouped_stage1_fused(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    gate_B_ptrs: torch.Tensor,
    gate_scales_ptrs: torch.Tensor,
    up_B_ptrs: torch.Tensor,
    up_scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    max_m_tiles: int,
    mtp: int,
    num_experts: int,
) -> None:
    """Fused S1: gate+up+SiLU in single kernel. No temp buffer.

    Each CTA does two sequential K-reductions (gate then up) for the same
    (expert, m_tile, n_tile). Gate result stored in SMEM, fused with SiLU
    in the write-back.

    Args:
        dispatched_x_3d: [E*mtp, K] BF16 input
        intermediate_3d: [E*mtp, N] BF16 output (SiLU(gate) * up written here)
        expert_counts: [E] int32 GPU
        expert_starts: [E] int32 GPU (= arange(E) * mtp)
        gate_B_ptrs: [E] int64 gate weight pointers
        gate_scales_ptrs: [E] int64 gate scale pointers
        up_B_ptrs: [E] int64 up weight pointers
        up_scales_ptrs: [E] int64 up scale pointers
        C_ptrs: [E] int64 output pointers (into intermediate at mtp stride)
        N, K: dimensions
        workspace: [locks] int32
        max_m_tiles: ceil(min(num_global, mtp) / 16)
        mtp: max tokens padded per expert
        num_experts: E
    """
    global _warned_m16
    if not _warned_m16:
        logging.info("[Marlin] Using fused M16 Marlin S1 (gate+up+SiLU, single kernel)")
        _warned_m16 = True

    mod = _load_module()
    n_tiles = N // 256

    mod.grouped_marlin_gemm_m16_s1(
        dispatched_x_3d,
        gate_B_ptrs, up_B_ptrs, C_ptrs,
        gate_scales_ptrs, up_scales_ptrs,
        expert_starts, expert_counts,
        num_experts, N, K, workspace, n_tiles, max_m_tiles,
    )


def single_expert_marlin_decode(
    x: torch.Tensor,
    gate_qw: torch.Tensor, gate_scale: torch.Tensor,
    up_qw: torch.Tensor, up_scale: torch.Tensor,
    down_qw: torch.Tensor, down_scale: torch.Tensor,
    N: int, K: int,
) -> torch.Tensor:
    """Grouped-Marlin W4A16 decode for ONE expert (E=1), for streamed/offloaded experts.

    The fused decode kernel is the same one the persistent path uses; here it runs over a
    single expert with `expert_starts=[0]`, `expert_counts=[t]`. The weight tensors come
    straight from the weight-buffer slot in **Marlin layout** — the kernel only reads their
    `.data_ptr()` (tensor .shape is ignored), so no Marlin->raw transform / reshape is needed.

    Args:
        x: [t, K] BF16 gathered tokens routed to this expert.
        *_qw / *_scale: the expert's Marlin MXFP4 packed weights (int32) +
            uint8 E8M0 scales.
        N: moe_intermediate_size; K: hidden_size.
    Returns: [t, K] BF16.
    """
    mod = _load_module()
    device = x.device
    t = x.shape[0]
    x = x.contiguous()

    def _p(tensor):
        return torch.tensor([tensor.data_ptr()], dtype=torch.int64, device=device)

    gate_B, up_B, down_B = _p(gate_qw), _p(up_qw), _p(down_qw)
    gate_sB, up_sB, down_sB = _p(gate_scale), _p(up_scale), _p(down_scale)
    expert_starts = torch.zeros(1, dtype=torch.int32, device=device)
    expert_counts = torch.tensor([t], dtype=torch.int32, device=device)
    intermediate = torch.empty(t, N, dtype=torch.bfloat16, device=device)
    expert_out = torch.empty(t, K, dtype=torch.bfloat16, device=device)
    s1_C, s3_C = _p(intermediate), _p(expert_out)
    s1_ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=device)
    s3_ws = torch.zeros(K // 256 + 17, dtype=torch.int32, device=device)
    max_m_tiles = (t + 15) // 16

    # Stage 1: fused gate + up + SiLU -> intermediate [t, N]
    mod.grouped_marlin_gemm_m16_s1(
        x, gate_B, up_B, s1_C, gate_sB, up_sB,
        expert_starts, expert_counts, 1, N, K, s1_ws, N // 256, max_m_tiles)
    # Stage 3: down -> expert_out [t, K]
    mod.grouped_marlin_gemm_m16(
        intermediate, down_B, s3_C, down_sB, expert_starts, expert_counts,
        1, K, N, s3_ws, 1, K // 256, max_m_tiles)
    return expert_out


def marlin_grouped_stage1_unified(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    up_buf: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    B_ptrs: torch.Tensor,
    scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    max_m_tiles: int,
    compact_stride: int,
    mtp: int,
    num_experts: int,
) -> None:
    """Unified M16 path: Marlin S1 for any M via CTA M-tiling.

    Gate output writes directly to intermediate (mtp stride).
    Up output writes to compact up_buf (compact_stride).
    Dual-stride SiLU fuses: intermediate = SiLU(gate_in_intermediate) * up_from_buf.

    Args:
        dispatched_x_3d: [E*mtp, K] BF16 input
        intermediate_3d: [E*mtp, N] BF16 output (gate writes here, SiLU in-place)
        up_buf: [E*compact_stride, N] BF16 temp buffer for up projection
        expert_counts: [E] int32 GPU — actual tokens per expert
        expert_starts: [E] int32 GPU — input row offsets (arange(E)*mtp)
        B_ptrs: [2E] int64 — gate+up weight pointers
        scales_ptrs: [2E] int64 — gate+up scale pointers
        C_ptrs: [2E] int64 — gate ptrs into intermediate, up ptrs into up_buf
        N, K: dimensions
        workspace: [locks] int32
        max_m_tiles: pigeonhole upper bound on M-tiles per expert
        compact_stride: up_buf rows per expert (= max_m_tiles * 16)
        mtp: gate stride in intermediate (max_tokens_padded)
        num_experts: E
    """
    global _warned_m16
    if not _warned_m16:
        logging.info("[Marlin] Using M16 Marlin W4A16 grouped GEMM (CTA M-tiling, any M)")
        _warned_m16 = True

    mod = _load_module()
    n_tiles = N // 256
    num_matrices = 2 * num_experts

    # Launch 1: M16 grouped GEMM with CTA M-tiling
    # Grid: num_matrices × max_m_tiles × n_tiles
    # Each CTA processes 16 rows, early-exits if beyond expert_counts[expert]
    mod.grouped_marlin_gemm_m16(
        dispatched_x_3d, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        num_experts, N, K, workspace, num_matrices, n_tiles, max_m_tiles,
    )

    # Launch 2: Dual-stride SiLU
    # Reads gate from intermediate (mtp stride), up from up_buf (compact_stride)
    # Writes SiLU(gate) * up in-place to intermediate
    mod.silu_mul_dual_stride(
        intermediate_3d, up_buf, expert_counts,
        num_experts, mtp, compact_stride, N,
    )


# ============================================================================
# Kimi-K3 MXFP4 (E2M1 + E8M0) wrappers — task #34. HARD-FAIL policy:
# every contract violation RAISES; there is no warn-and-degrade. The unfused
# reference path survives only behind the explicit batchgen_debug opt-in
# (`k3_moe_reference`) as the parity oracle — it is not a fallback. NOTE the
# opt-in's consumer is model/server-side wiring (outside this kernel PR's
# allowlist) and is a NAMED FOLLOW-UP; until it lands there is no reference
# forward path at all — the flag name below is forward-declared, not live.
# ============================================================================

_MXFP4_KERNEL_ENTRIES = (
    "grouped_marlin_gemm_m16_mxfp4",
    "grouped_marlin_gemm_m16_s1_mxfp4_situ",
    "grouped_marlin_gemm_m16_mxfp4_v2",
    "grouped_marlin_gemm_m16_s1_mxfp4_situ_v2",
    "grouped_marlin_gemm_m16_mxfp4_v3",
    "grouped_marlin_gemm_m16_s1_mxfp4_situ_v3",
    "marlin_mxfp4_v3_expert_order",
    "marlin_mxfp4_v3_launch_config",
)

_warned_mxfp4 = False
_warned_mxfp4_v2 = False
_warned_mxfp4_v3 = False

# v2 variant -> (TM 16-row m-blocks per CTA, TN 16-column n-blocks per CTA).
MXFP4_V2_VARIANTS = {0: (1, 16), 1: (2, 16), 2: (2, 8), 3: (1, 8)}
# The v2 kernels fold the E2M1 2^126 rebias into the scale (bf16 2^(e-1));
# byte 129 would reach the bf16 Inf/NaN exponent.
MXFP4_V2_MAX_SCALE_BYTE = 128
# v3 variant -> (JW A fragments per warp per 16-K step, NCAP rows per CTA
# chunk). A warpgroup covers 64*JW output columns; v3 reuses the v2 decode, so
# MXFP4_V2_MAX_SCALE_BYTE applies to it as well. Variant 3 has variant 0's
# tiling but stages the dequantised A operand in shared memory (wgmma reads
# both operands from shared memory) with the same MMA inputs and K order, so
# its outputs must be bit-identical to variant 0's. Variant 4 has variant 0's
# MMA inputs and K order with the shared-memory ring filled by TMA and bulk
# copies under mbarriers; variant 5 is variant 4 with liveness fences on the
# A register sets. Both must be bit-identical to variant 0 as well.
MXFP4_V3_VARIANTS = {0: (2, 64), 1: (2, 48), 2: (4, 32), 3: (2, 64), 4: (2, 64),
                     5: (2, 64)}


def is_marlin_mxfp4_available() -> bool:
    return _module is not None and all(
        hasattr(_module, k) for k in _MXFP4_KERNEL_ENTRIES)


def _require_mxfp4_kernels():
    """L1: the marlin MXFP4 entries must exist — K3 refuses to run otherwise."""
    missing = [k for k in _MXFP4_KERNEL_ENTRIES if not hasattr(_module, k)]
    if missing:
        # Name the .so actually loaded. A stale copy installed in
        # site-packages shadows the repo's in-tree build whenever the repo
        # root is not on sys.path (e.g. `python path/to/script.py`, whose
        # sys.path[0] is the SCRIPT's directory) — in which case the fix is
        # the import path, not a rebuild.
        loaded = getattr(_module, "__file__", "<unknown>")
        raise RuntimeError(
            f"Marlin MXFP4 kernel entries missing from "
            f"batchgen_kernels.moe._C_marlin_grouped_gemm: {missing}. "
            f"The extension actually loaded is {loaded} — if that path is "
            f"not inside this repo, a stale installed copy is shadowing the "
            f"in-tree build; fix sys.path/PYTHONPATH rather than rebuilding. "
            f"K3 refuses to run. "
            f"The designated parity-debug opt-in is batchgen_debug."
            f"k3_moe_reference; its model-side wiring is a named follow-up "
            f"of task #34 — if it is not wired yet there is NO alternative "
            f"path and rebuilding is the only fix.")


def _check_ptr_array(name: str, t: torch.Tensor, length: int):
    if t.dtype != torch.int64 or not t.is_cuda or t.numel() != length:
        raise ValueError(
            f"group metadata contract violated: {name} must be int64 CUDA "
            f"[{length}], got {t.dtype} {t.device} numel={t.numel()}")


def _check_counts(name: str, t: torch.Tensor, length: int):
    if t.dtype != torch.int32 or not t.is_cuda or t.numel() != length:
        raise ValueError(
            f"group metadata contract violated: {name} must be int32 CUDA "
            f"[{length}], got {t.dtype} {t.device} numel={t.numel()}")


def _check_activation(name: str, x: torch.Tensor, last_dim: int,
                      require_contiguous: bool = True):
    """Activation-tensor contract at the hard-fail seams. An fp16 (or fp32)
    activation is byte-compatible with the kernel's bf16 reinterpret and
    produces finite silent garbage — exactly the class the HARD-FAIL ledger
    targets — so dtype/device/shape are checked, not assumed."""
    if (x.dtype != torch.bfloat16 or not x.is_cuda or x.shape[-1] != last_dim
            or (require_contiguous and not x.is_contiguous())):
        raise ValueError(
            f"activation contract violated: {name} must be contiguous bf16 "
            f"CUDA [..., {last_dim}], got {x.dtype} {x.device} "
            f"{tuple(x.shape)} contiguous={x.is_contiguous()}")


def _check_m16_shapes(prob_n: int, prob_k: int):
    """L4: kernel tiling constraints (prob_n%256 for n_tiles, prob_k%128 per
    pipeline stage). K3 shapes 3072/3584/6144 all pass; anything else is a
    wiring bug."""
    if prob_n % 256 != 0 or prob_k % 128 != 0:
        raise ValueError(
            f"marlin M16 kernel constraint violated: prob_n%256==0 and "
            f"prob_k%128==0 required, got prob_n={prob_n}, prob_k={prob_k}")


def _check_m_tile_bound(max_m_tiles: int, mtp: int, total_rows: int):
    """L5 (host-static, plan-build): CTAs beyond max_m_tiles*16 rows would be
    silently dropped. The dispatcher guarantees counts[e] <= min(mtp,
    total_rows), so this bound makes drops impossible."""
    admissible = min(int(mtp), int(total_rows))
    if max_m_tiles * 16 < admissible:
        raise ValueError(
            f"M-tile bound below admissible per-expert tokens: "
            f"max_m_tiles={max_m_tiles} covers {max_m_tiles * 16} rows < "
            f"min(mtp={mtp}, total_rows={total_rows})={admissible} — CTAs "
            f"would silently drop rows")


def _check_marlin_mxfp4_tensors(name: str, qw: torch.Tensor, scale: torch.Tensor,
                                prob_n: int, prob_k: int):
    """L2 (tensor-visible call sites only): marlin layout + raw E8M0 scales."""
    if qw.dtype != torch.int32:
        raise ValueError(
            f"{name}: marlin_qw must be int32 marlin-packed, got {qw.dtype}")
    if scale.dtype != torch.uint8:
        raise ValueError(
            f"{name}: scale dtype != uint8 E8M0 at kernel boundary "
            f"(got {scale.dtype})")
    if tuple(qw.reshape(-1, qw.shape[-1]).shape) != (prob_k // 16, prob_n * 2):
        raise ValueError(
            f"{name}: marlin_qw shape {tuple(qw.shape)} != "
            f"[{prob_k // 16}, {prob_n * 2}] for prob_k={prob_k}, prob_n={prob_n}")
    if tuple(scale.reshape(-1, scale.shape[-1]).shape) != (prob_k // 32, prob_n):
        raise ValueError(
            f"{name}: marlin scale shape {tuple(scale.shape)} != "
            f"[{prob_k // 32}, {prob_n}] for prob_k={prob_k}, prob_n={prob_n}")


def marlin_grouped_stage1_fused_mxfp4_situ(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    gate_B_ptrs: torch.Tensor,
    gate_scales_ptrs: torch.Tensor,
    up_B_ptrs: torch.Tensor,
    up_scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    max_m_tiles: int,
    mtp: int,
    num_experts: int,
    total_rows: int,
) -> None:
    """K3 fused S1: gate(w1)+up(w3) MXFP4 GEMM + SiTU in a single kernel.

    Same seam as marlin_grouped_stage1_fused (K2.5 production), with:
    - E2M1 in-kernel weight decode (dequant_e2m1) instead of (q-8),
    - SiTU epilogue (beta=4, linear_beta=25) instead of SiLU,
    - hard-fail contract checks L1/L3/L4/L5 + activation contract (dtype/
      device/shape/contiguity of both activation tensors; no warn-and-degrade),
    - `total_rows` REQUIRED so the L5 M-tile bound is enforceable here
      (K2.5 computes it at plan build; the K3 seam must not trust it).

    Pointer arrays must point at tensors produced by
    marlin_weight_prep.repack_mxfp4_to_marlin_gs32 with uint8 E8M0 scales at the
    kernel boundary (L2 is checked at the tensor-visible call sites; the
    checkpoint stamp check L6 is model-side).

    Zero-token experts are handled by the kernel's per-CTA early-exit — the
    caller must NOT filter empty experts (graph-static pointer arrays).
    """
    global _warned_mxfp4
    _require_mxfp4_kernels()
    _check_activation("dispatched_x_3d", dispatched_x_3d, K)
    _check_activation("intermediate_3d", intermediate_3d, N)
    E = int(num_experts)
    _check_counts("expert_counts", expert_counts, E)
    _check_counts("expert_starts", expert_starts, E)
    for name, t in (("gate_B_ptrs", gate_B_ptrs), ("up_B_ptrs", up_B_ptrs),
                    ("gate_scales_ptrs", gate_scales_ptrs),
                    ("up_scales_ptrs", up_scales_ptrs), ("C_ptrs", C_ptrs)):
        _check_ptr_array(name, t, E)
    _check_m16_shapes(N, K)
    _check_m_tile_bound(max_m_tiles, mtp, total_rows)

    if not _warned_mxfp4:
        logging.info("[Marlin] Using fused M16 Marlin MXFP4 S1 (gate+up+SiTU, K3)")
        _warned_mxfp4 = True

    mod = _load_module()
    n_tiles = N // 256
    mod.grouped_marlin_gemm_m16_s1_mxfp4_situ(
        dispatched_x_3d,
        gate_B_ptrs, up_B_ptrs, C_ptrs,
        gate_scales_ptrs, up_scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, n_tiles, max_m_tiles,
    )


def marlin_grouped_m16_mxfp4(
    A: torch.Tensor,
    B_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    scales_ptrs: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    num_experts: int,
    prob_n: int,
    prob_k: int,
    workspace: torch.Tensor,
    num_matrices: int,
    n_tiles: int,
    max_m_tiles: int,
) -> None:
    """K3 M16 MXFP4 grouped GEMM (S3 down projection / standalone)."""
    _require_mxfp4_kernels()
    _check_activation("A", A, prob_k)
    E = int(num_experts)
    _check_counts("expert_counts", expert_counts, E)
    _check_counts("expert_starts", expert_starts, E)
    _check_ptr_array("B_ptrs", B_ptrs, num_matrices)
    _check_ptr_array("C_ptrs", C_ptrs, num_matrices)
    _check_ptr_array("scales_ptrs", scales_ptrs, num_matrices)
    _check_m16_shapes(prob_n, prob_k)
    if n_tiles != prob_n // 256:
        raise ValueError(f"n_tiles={n_tiles} != prob_n//256={prob_n // 256}")

    mod = _load_module()
    mod.grouped_marlin_gemm_m16_mxfp4(
        A, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, prob_n, prob_k, workspace, num_matrices, n_tiles, max_m_tiles,
    )


def check_mxfp4_scales_v2_compatible(scales: torch.Tensor) -> None:
    """Raise ValueError unless every E8M0 scale byte is <= 128.

    The v2 MXFP4 kernels expand byte e to bf16 2^(e-1) (the E2M1 rebias
    folded into the scale), which is only finite for e <= 128. One device max
    plus a host sync: call once per weight tensor at weight-build time, never
    per launch.
    """
    if scales.dtype != torch.uint8:
        raise ValueError(
            f"mxfp4 v2 scale check: expected uint8 E8M0 scales, got {scales.dtype}")
    if scales.numel() == 0:
        return
    hi = int(scales.max())
    if hi > MXFP4_V2_MAX_SCALE_BYTE:
        raise ValueError(
            f"mxfp4 v2 scale check: max E8M0 byte {hi} > "
            f"{MXFP4_V2_MAX_SCALE_BYTE}; the v2 kernels' folded scale 2^(e-1) "
            f"would overflow bf16. Use the v1 entries for these weights.")


def _check_v2_variant(variant: int, prob_n: int):
    if variant not in MXFP4_V2_VARIANTS:
        raise ValueError(
            f"mxfp4 v2 variant must be one of {sorted(MXFP4_V2_VARIANTS)}, "
            f"got {variant}")
    tn = MXFP4_V2_VARIANTS[variant][1]
    if prob_n % (16 * tn) != 0:
        raise ValueError(
            f"mxfp4 v2 variant {variant}: prob_n={prob_n} not a multiple of "
            f"{16 * tn}")


def marlin_grouped_stage1_fused_mxfp4_situ_v2(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    gate_B_ptrs: torch.Tensor,
    gate_scales_ptrs: torch.Tensor,
    up_B_ptrs: torch.Tensor,
    up_scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    max_m_tiles: int,
    mtp: int,
    num_experts: int,
    total_rows: int,
    *,
    variant: int = 2,
) -> None:
    """K3 fused S1 v2: same contract and checks as
    marlin_grouped_stage1_fused_mxfp4_situ, plus `variant` (see
    MXFP4_V2_VARIANTS).

    The grid is num_experts x (N / (16*TN)) and each CTA walks all rows of
    its expert, so max_m_tiles does not limit the rows processed; it is still
    validated (L5) so v1 and v2 stay drop-in interchangeable.

    Precondition beyond v1: every E8M0 scale byte behind the pointer arrays
    is <= 128 (check_mxfp4_scales_v2_compatible, at weight-build time).
    """
    global _warned_mxfp4_v2
    _require_mxfp4_kernels()
    _check_activation("dispatched_x_3d", dispatched_x_3d, K)
    _check_activation("intermediate_3d", intermediate_3d, N)
    E = int(num_experts)
    _check_counts("expert_counts", expert_counts, E)
    _check_counts("expert_starts", expert_starts, E)
    for name, t in (("gate_B_ptrs", gate_B_ptrs), ("up_B_ptrs", up_B_ptrs),
                    ("gate_scales_ptrs", gate_scales_ptrs),
                    ("up_scales_ptrs", up_scales_ptrs), ("C_ptrs", C_ptrs)):
        _check_ptr_array(name, t, E)
    _check_m16_shapes(N, K)
    _check_m_tile_bound(max_m_tiles, mtp, total_rows)
    _check_v2_variant(variant, N)

    if not _warned_mxfp4_v2:
        logging.info("[Marlin] Using fused Marlin MXFP4 S1 v2 (gate+up+SiTU), "
                     "variant %d", variant)
        _warned_mxfp4_v2 = True

    mod = _load_module()
    n_tiles = N // 256
    mod.grouped_marlin_gemm_m16_s1_mxfp4_situ_v2(
        dispatched_x_3d,
        gate_B_ptrs, up_B_ptrs, C_ptrs,
        gate_scales_ptrs, up_scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, n_tiles, max_m_tiles, variant,
    )


def marlin_grouped_m16_mxfp4_v2(
    A: torch.Tensor,
    B_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    scales_ptrs: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    num_experts: int,
    prob_n: int,
    prob_k: int,
    workspace: torch.Tensor,
    num_matrices: int,
    n_tiles: int,
    max_m_tiles: int,
    *,
    variant: int = 2,
) -> None:
    """K3 MXFP4 grouped GEMM v2 (S3 down projection / standalone): same
    contract and checks as marlin_grouped_m16_mxfp4, plus `variant`.
    max_m_tiles is accepted for interchangeability but does not limit rows.
    Scale bytes must be <= 128 (check_mxfp4_scales_v2_compatible)."""
    _require_mxfp4_kernels()
    _check_activation("A", A, prob_k)
    E = int(num_experts)
    _check_counts("expert_counts", expert_counts, E)
    _check_counts("expert_starts", expert_starts, E)
    _check_ptr_array("B_ptrs", B_ptrs, num_matrices)
    _check_ptr_array("C_ptrs", C_ptrs, num_matrices)
    _check_ptr_array("scales_ptrs", scales_ptrs, num_matrices)
    _check_m16_shapes(prob_n, prob_k)
    if n_tiles != prob_n // 256:
        raise ValueError(f"n_tiles={n_tiles} != prob_n//256={prob_n // 256}")
    _check_v2_variant(variant, prob_n)

    mod = _load_module()
    mod.grouped_marlin_gemm_m16_mxfp4_v2(
        A, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, prob_n, prob_k, workspace, num_matrices, n_tiles, max_m_tiles,
        variant,
    )


def mxfp4_v3_cta_cols(variant: int, fused: bool) -> int:
    """Output columns per CTA of a v3 variant (fused S1: one warpgroup tile;
    grouped: two)."""
    jw = MXFP4_V3_VARIANTS[variant][0]
    return 64 * jw if fused else 128 * jw


def _check_v3_variant(variant: int, prob_n: int, fused: bool):
    if variant not in MXFP4_V3_VARIANTS:
        raise ValueError(
            f"mxfp4 v3 variant must be one of {sorted(MXFP4_V3_VARIANTS)}, "
            f"got {variant}")
    cols = mxfp4_v3_cta_cols(variant, fused)
    if prob_n % cols != 0:
        raise ValueError(
            f"mxfp4 v3 variant {variant}: prob_n={prob_n} not a multiple of "
            f"{cols}")


def _check_v3_order(expert_order, num_experts: int):
    if expert_order is not None:
        _check_counts("expert_order", expert_order, num_experts)
        if not expert_order.is_contiguous():
            raise ValueError("group metadata contract violated: expert_order "
                             "must be contiguous")


def compute_mxfp4_v3_expert_order(
    expert_counts: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Expert indices by descending row count (ties by index) for the v3
    `expert_order` argument, computed on the device without a host sync.

    Pass a preallocated int32 `out` of the same length when capturing a CUDA
    graph; the kernel is capturable together with the v3 launches.
    """
    _require_mxfp4_kernels()
    E = expert_counts.numel()
    _check_counts("expert_counts", expert_counts, E)
    if out is None:
        out = torch.empty(E, dtype=torch.int32, device=expert_counts.device)
    _check_counts("expert_order", out, E)
    _load_module().marlin_mxfp4_v3_expert_order(expert_counts, out)
    return out


def marlin_grouped_stage1_fused_mxfp4_situ_v3(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    gate_B_ptrs: torch.Tensor,
    gate_scales_ptrs: torch.Tensor,
    up_B_ptrs: torch.Tensor,
    up_scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    max_m_tiles: int,
    mtp: int,
    num_experts: int,
    total_rows: int,
    *,
    variant: int = 0,
    expert_order: torch.Tensor = None,
) -> None:
    """Fused S1 v3 (swap-AB wgmma, sm_90a): same contract and checks as
    marlin_grouped_stage1_fused_mxfp4_situ, plus `variant` (see
    MXFP4_V3_VARIANTS) and an optional int32 [num_experts] `expert_order`
    (compute_mxfp4_v3_expert_order; None = expert index order). The order only
    changes CTA scheduling; outputs are bit-identical with and without it.

    As in v1, the grid is shaped by max_m_tiles, so the L5 bound applies.
    Precondition beyond v1: every E8M0 scale byte behind the pointer arrays
    is <= 128 (check_mxfp4_scales_v2_compatible, at weight-build time).
    """
    global _warned_mxfp4_v3
    _require_mxfp4_kernels()
    _check_activation("dispatched_x_3d", dispatched_x_3d, K)
    _check_activation("intermediate_3d", intermediate_3d, N)
    E = int(num_experts)
    _check_counts("expert_counts", expert_counts, E)
    _check_counts("expert_starts", expert_starts, E)
    for name, t in (("gate_B_ptrs", gate_B_ptrs), ("up_B_ptrs", up_B_ptrs),
                    ("gate_scales_ptrs", gate_scales_ptrs),
                    ("up_scales_ptrs", up_scales_ptrs), ("C_ptrs", C_ptrs)):
        _check_ptr_array(name, t, E)
    _check_m16_shapes(N, K)
    _check_m_tile_bound(max_m_tiles, mtp, total_rows)
    _check_v3_variant(variant, N, fused=True)
    _check_v3_order(expert_order, E)

    if not _warned_mxfp4_v3:
        logging.info("[Marlin] Using fused Marlin MXFP4 S1 v3 (gate+up+SiTU, "
                     "wgmma), variant %d", variant)
        _warned_mxfp4_v3 = True

    mod = _load_module()
    n_tiles = N // 256
    mod.grouped_marlin_gemm_m16_s1_mxfp4_situ_v3(
        dispatched_x_3d,
        gate_B_ptrs, up_B_ptrs, C_ptrs,
        gate_scales_ptrs, up_scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, n_tiles, max_m_tiles, variant, expert_order,
    )


def marlin_grouped_m16_mxfp4_v3(
    A: torch.Tensor,
    B_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    scales_ptrs: torch.Tensor,
    expert_starts: torch.Tensor,
    expert_counts: torch.Tensor,
    num_experts: int,
    prob_n: int,
    prob_k: int,
    workspace: torch.Tensor,
    num_matrices: int,
    n_tiles: int,
    max_m_tiles: int,
    *,
    variant: int = 0,
    expert_order: torch.Tensor = None,
) -> None:
    """MXFP4 grouped GEMM v3 (S3 down projection / standalone, swap-AB
    wgmma, sm_90a): same contract and checks as marlin_grouped_m16_mxfp4,
    plus `variant` and an optional `expert_order` (requires num_matrices ==
    num_experts). Rows per expert are bounded by max_m_tiles as in v1. Scale
    bytes must be <= 128 (check_mxfp4_scales_v2_compatible)."""
    _require_mxfp4_kernels()
    _check_activation("A", A, prob_k)
    E = int(num_experts)
    _check_counts("expert_counts", expert_counts, E)
    _check_counts("expert_starts", expert_starts, E)
    _check_ptr_array("B_ptrs", B_ptrs, num_matrices)
    _check_ptr_array("C_ptrs", C_ptrs, num_matrices)
    _check_ptr_array("scales_ptrs", scales_ptrs, num_matrices)
    _check_m16_shapes(prob_n, prob_k)
    if n_tiles != prob_n // 256:
        raise ValueError(f"n_tiles={n_tiles} != prob_n//256={prob_n // 256}")
    _check_v3_variant(variant, prob_n, fused=False)
    _check_v3_order(expert_order, E)
    if expert_order is not None and num_matrices != E:
        raise ValueError(
            f"expert_order needs num_matrices == num_experts, got "
            f"{num_matrices} vs {E}")

    mod = _load_module()
    mod.grouped_marlin_gemm_m16_mxfp4_v3(
        A, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, prob_n, prob_k, workspace, num_matrices, n_tiles, max_m_tiles,
        variant, expert_order,
    )


def select_mxfp4_kernel_version(version: int):
    """(fused S1 wrapper, grouped S3 wrapper) of MXFP4 kernel version 1, 2
    or 3. The pairs share the positional arguments of the v1 wrappers; v2 and
    v3 add keyword-only arguments."""
    pairs = {
        1: (marlin_grouped_stage1_fused_mxfp4_situ, marlin_grouped_m16_mxfp4),
        2: (marlin_grouped_stage1_fused_mxfp4_situ_v2, marlin_grouped_m16_mxfp4_v2),
        3: (marlin_grouped_stage1_fused_mxfp4_situ_v3, marlin_grouped_m16_mxfp4_v3),
    }
    if version not in pairs:
        raise ValueError(
            f"mxfp4 kernel version must be one of {sorted(pairs)}, got {version}")
    return pairs[version]


def single_expert_marlin_mxfp4_decode(
    x: torch.Tensor,
    gate_qw: torch.Tensor, gate_scale: torch.Tensor,
    up_qw: torch.Tensor, up_scale: torch.Tensor,
    down_qw: torch.Tensor, down_scale: torch.Tensor,
    N: int, K: int,
) -> torch.Tensor:
    """K3 W4A16 MXFP4 decode for ONE expert (streamed/offloaded experts).

    Mirror of single_expert_marlin_decode with the _mxfp4 + SiTU kernel
    entries and full tensor-level contract checks (this is the one seam where
    the real tensors — not just pointers — are visible, so L2 is enforced).

    Args:
        x: [t, K] BF16 gathered tokens routed to this expert.
        gate_qw/up_qw: [K//16, N*2] int32 marlin MXFP4 (from
            repack_mxfp4_to_marlin_gs32); gate = w1, up = w3 (gate-first —
            swapped branches are silent, pinned by the GPU mutation test).
        gate_scale/up_scale: [K//32, N] uint8 E8M0.
        down_qw: [N//16, K*2] int32; down_scale: [N//32, K] uint8 E8M0.
        N: moe_intermediate_size (K3: 3072); K: hidden_size (K3: 3584).
    Returns: [t, K] BF16.
    """
    _require_mxfp4_kernels()
    _check_activation("x", x, K, require_contiguous=False)  # .contiguous() below
    _check_m16_shapes(N, K)   # S1: prob_n=N, prob_k=K
    _check_m16_shapes(K, N)   # S3: prob_n=K, prob_k=N
    _check_marlin_mxfp4_tensors("gate(w1)", gate_qw, gate_scale, N, K)
    _check_marlin_mxfp4_tensors("up(w3)", up_qw, up_scale, N, K)
    _check_marlin_mxfp4_tensors("down(w2)", down_qw, down_scale, K, N)

    mod = _load_module()
    device = x.device
    t = x.shape[0]
    x = x.contiguous()

    def _p(tensor):
        return torch.tensor([tensor.data_ptr()], dtype=torch.int64, device=device)

    gate_B, up_B, down_B = _p(gate_qw), _p(up_qw), _p(down_qw)
    gate_sB, up_sB, down_sB = _p(gate_scale), _p(up_scale), _p(down_scale)
    expert_starts = torch.zeros(1, dtype=torch.int32, device=device)
    expert_counts = torch.tensor([t], dtype=torch.int32, device=device)
    intermediate = torch.empty(t, N, dtype=torch.bfloat16, device=device)
    expert_out = torch.empty(t, K, dtype=torch.bfloat16, device=device)
    s1_C, s3_C = _p(intermediate), _p(expert_out)
    s1_ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=device)
    s3_ws = torch.zeros(K // 256 + 17, dtype=torch.int32, device=device)
    max_m_tiles = (t + 15) // 16

    # Stage 1: fused gate + up + SiTU -> intermediate [t, N]
    mod.grouped_marlin_gemm_m16_s1_mxfp4_situ(
        x, gate_B, up_B, s1_C, gate_sB, up_sB,
        expert_starts, expert_counts, 1, N, K, s1_ws, N // 256, max_m_tiles)
    # Stage 3: down -> expert_out [t, K]
    mod.grouped_marlin_gemm_m16_mxfp4(
        intermediate, down_B, s3_C, down_sB, expert_starts, expert_counts,
        1, K, N, s3_ws, 1, K // 256, max_m_tiles)
    return expert_out
