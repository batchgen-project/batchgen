"""QKV projection + split (+ optional RoPE) Triton kernel for SM100 (Blackwell).

SM100 port of the SM90a WGMMA kernel `batchgen_kernels/src/attention/qkv_wgmma.cu`.
That kernel uses Hopper-only `wgmma.*` + warpgroup TMA, which do not exist on
sm_100a, so on Blackwell we replace it with a pure-Triton GEMM that Triton JIT-compiles
to UMMA. No `torch.matmul` / `F.linear` is used.

Operation (decode/prefill):

    x[M, K] @ W_qkv[N, K]^T  ->  [M, N],   N = q_size + kv_size + kv_size

split column-wise into:

    Q[M, q_size]   (cols [0, q_size))
    K[M, kv_size]  (cols [q_size, q_size + kv_size))
    V[M, kv_size]  (cols [q_size + kv_size, N))

RoPE (optional) is applied to Q and K only, never V (see `apply_rope_qk`).

Design note (see batchgen-agent-metadata/.../incremental_compilation_contract.md and
blackwell-kernel-port-v1.md sub-task 5): RoPE is applied in a *separate* validated pass
(`apply_rope_qk`) rather than inline in the GEMM epilogue. At decode M the GEMM is
memory-bound on the [N, K] weight read, so the extra Q/K reread is negligible, and the
two-pass structure keeps each step independently testable — matching how BatchGen already
structures RoPE as dedicated kernels.
"""

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pass 1 — GEMM + column split into Q / K / V
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def _qkv_proj_split_kernel(
    x_ptr, w_ptr, bias_ptr,
    q_ptr, k_ptr, v_ptr,
    M, K, N, q_size, kv_size,
    stride_xm, stride_xk,
    stride_wn, stride_wk,          # w is [N, K] row-major: stride_wn=K, stride_wk=1
    stride_qm, stride_qn,
    stride_km, stride_kn,
    stride_vm, stride_vn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # x[BLOCK_M, K] @ w[N, K]^T  ->  acc[BLOCK_M, BLOCK_N], accumulate in fp32
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # w is [N, K]; we need w[n, k] for the n-tile, contracted over k -> shape [BLOCK_K, BLOCK_N]
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    out = acc.to(q_ptr.dtype.element_ty)

    kv_end = q_size + kv_size

    # Region masks (per-column). Robust to any BLOCK_N / boundary alignment.
    is_q = offs_n < q_size
    is_k = (offs_n >= q_size) & (offs_n < kv_end)
    is_v = offs_n >= kv_end

    # Q store: local col = offs_n
    q_cols = offs_n
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + q_cols[None, :] * stride_qn)
    tl.store(q_ptrs, out, mask=m_mask[:, None] & (is_q & n_mask)[None, :])

    # K store: local col = offs_n - q_size
    k_cols = offs_n - q_size
    k_ptrs = k_ptr + (offs_m[:, None] * stride_km + k_cols[None, :] * stride_kn)
    tl.store(k_ptrs, out, mask=m_mask[:, None] & (is_k & n_mask)[None, :])

    # V store: local col = offs_n - kv_end
    v_cols = offs_n - kv_end
    v_ptrs = v_ptr + (offs_m[:, None] * stride_vm + v_cols[None, :] * stride_vn)
    tl.store(v_ptrs, out, mask=m_mask[:, None] & (is_v & n_mask)[None, :])


def qkv_proj_split(
    x: torch.Tensor,          # [M, K] BF16
    w_qkv: torch.Tensor,      # [N, K] BF16,  N = q_size + 2*kv_size
    bias: torch.Tensor | None,
    q_size: int,
    kv_size: int,
):
    """Fused QKV GEMM + column split. Returns (Q, K, V) BF16, no RoPE."""
    assert x.dim() == 2 and w_qkv.dim() == 2
    M, K = x.shape
    N, Kw = w_qkv.shape
    assert Kw == K, f"K mismatch: x K={K}, w K={Kw}"
    assert N == q_size + 2 * kv_size, f"N={N} != q_size({q_size}) + 2*kv_size({kv_size})"

    Q = torch.empty((M, q_size),  dtype=x.dtype, device=x.device)
    Kt = torch.empty((M, kv_size), dtype=x.dtype, device=x.device)
    V = torch.empty((M, kv_size),  dtype=x.dtype, device=x.device)

    HAS_BIAS = bias is not None
    bias_arg = bias if HAS_BIAS else x  # dummy ptr when unused

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _qkv_proj_split_kernel[grid](
        x, w_qkv, bias_arg,
        Q, Kt, V,
        M, K, N, q_size, kv_size,
        x.stride(0), x.stride(1),
        w_qkv.stride(0), w_qkv.stride(1),
        Q.stride(0), Q.stride(1),
        Kt.stride(0), Kt.stride(1),
        V.stride(0), V.stride(1),
        HAS_BIAS=HAS_BIAS,
    )
    return Q, Kt, V


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2 — RoPE for Q and K (standard rotate_half / contiguous-half convention)
# ──────────────────────────────────────────────────────────────────────────────
# cos/sin are [M, head_dim] (already gathered per-row by position_id upstream).
# rotate_half: for head split into [x1 | x2] (each head_dim/2 wide),
#     out = [x1*cos1 - x2*sin1 ,  x2*cos2 + x1*sin2]
# with cos = [cos1 | cos2], sin = [sin1 | sin2] and cos1==cos2, sin1==sin2 in the
# standard HF layout. This matches torch:
#     (t * cos) + (rotate_half(t) * sin),  rotate_half(t) = cat(-t2, t1)

@triton.jit
def _rope_inplace_kernel(
    t_ptr,                 # [M, n_heads * head_dim]
    cos_ptr, sin_ptr,      # [M, head_dim]
    M, n_heads,
    head_dim: tl.constexpr,
    half: tl.constexpr,
    stride_tm, stride_td,
    stride_cm, stride_cd,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)        # head index
    if pid_m >= M:
        return

    h = pid_h
    if h >= n_heads:
        return

    half_idx = tl.arange(0, half)
    base = pid_m * stride_tm + h * head_dim * stride_td

    # x1 = first half, x2 = second half
    x1 = tl.load(t_ptr + base + half_idx * stride_td)
    x2 = tl.load(t_ptr + base + (half + half_idx) * stride_td)

    cos1 = tl.load(cos_ptr + pid_m * stride_cm + half_idx * stride_cd)
    sin1 = tl.load(sin_ptr + pid_m * stride_cm + half_idx * stride_cd)
    cos2 = tl.load(cos_ptr + pid_m * stride_cm + (half + half_idx) * stride_cd)
    sin2 = tl.load(sin_ptr + pid_m * stride_cm + (half + half_idx) * stride_cd)

    x1f = x1.to(tl.float32)
    x2f = x2.to(tl.float32)
    out1 = x1f * cos1.to(tl.float32) - x2f * sin1.to(tl.float32)
    out2 = x2f * cos2.to(tl.float32) + x1f * sin2.to(tl.float32)

    tl.store(t_ptr + base + half_idx * stride_td, out1.to(t_ptr.dtype.element_ty))
    tl.store(t_ptr + base + (half + half_idx) * stride_td, out2.to(t_ptr.dtype.element_ty))


def apply_rope_qk(
    t: torch.Tensor,        # [M, n_heads * head_dim] BF16, modified in place
    cos: torch.Tensor,      # [M, head_dim]
    sin: torch.Tensor,      # [M, head_dim]
    head_dim: int,
):
    """Apply standard rotate_half RoPE in place to a [M, n_heads*head_dim] tensor."""
    M, total = t.shape
    assert total % head_dim == 0
    n_heads = total // head_dim
    half = head_dim // 2
    grid = (M, n_heads)
    _rope_inplace_kernel[grid](
        t, cos, sin,
        M, n_heads, head_dim, half,
        t.stride(0), t.stride(1),
        cos.stride(0), cos.stride(1),
        BLOCK_H=1,
    )
    return t


def qkv_proj_rope(
    x: torch.Tensor,          # [M, K] BF16
    w_qkv: torch.Tensor,      # [N, K] BF16
    bias: torch.Tensor | None,
    q_size: int,
    kv_size: int,
    head_dim: int,
    rope_cos: torch.Tensor | None,   # [M, head_dim]
    rope_sin: torch.Tensor | None,   # [M, head_dim]
):
    """QKV projection + split, with optional standard rotate_half RoPE on Q and K.

    Returns (Q, K, V), each BF16. RoPE is applied to Q and K only when
    rope_cos/rope_sin are provided. V is never rotated.
    """
    Q, Kt, V = qkv_proj_split(x, w_qkv, bias, q_size, kv_size)
    if rope_cos is not None and rope_sin is not None:
        apply_rope_qk(Q, rope_cos, rope_sin, head_dim)
        apply_rope_qk(Kt, rope_cos, rope_sin, head_dim)
    return Q, Kt, V
