"""
CUDA routing kernels for GPT-OSS-120B MoE.

Pre-compiled via pip install -e batchgen_kernels/. Loaded lazily on first use.

Usage:
    from batchgen.moe.routing import (
        gate_topk_softmax_cuda,
        dispatch_count_gather_cuda,
        reduce_weighted_scatter_cuda,
    )
"""

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Pre-compiled at pip install time, loaded lazily on first kernel call
# ──────────────────────────────────────────────────────────────────────────────

_cuda_ext = None


def _get_ext():
    global _cuda_ext
    if _cuda_ext is None:
        import batchgen_kernels
        _cuda_ext = batchgen_kernels.load_extension("batchgen_kernels.moe._C_routing")
    return _cuda_ext


# ──────────────────────────────────────────────────────────────────────────────
# Python wrappers (matching Triton kernel signatures)
# ──────────────────────────────────────────────────────────────────────────────

def gate_topk_softmax_cuda(router_logits, topk_indices=None, topk_weights=None, k=4,
                           num_valid_tokens=-1):
    """
    CUDA gate kernel: fused top-k selection + softmax.

    Args:
        router_logits: [N, E] FP32
        topk_indices: [N, K] int32 pre-allocated output (optional)
        topk_weights: [N, K] FP32 pre-allocated output (optional)
        k: top-k (default 4)
        num_valid_tokens: only process first num_valid_tokens tokens (-1 = all)

    Returns:
        topk_indices: [N, K] int32
        topk_weights: [N, K] FP32
    """
    ext = _get_ext()
    N = router_logits.shape[0]
    device = router_logits.device

    # Kernel requires FP32 for numerical stability in softmax
    if router_logits.dtype != torch.float32:
        router_logits = router_logits.float()

    if topk_indices is None:
        topk_indices = torch.empty(N, k, dtype=torch.int32, device=device)
    if topk_weights is None:
        topk_weights = torch.empty(N, k, dtype=torch.float32, device=device)

    result = ext.gate_topk_softmax(router_logits, k, topk_indices, topk_weights,
                                   num_valid_tokens)
    return result[0], result[1]


def gate_sigmoid_topk_cuda(
    router_logits, e_score_correction,
    k=8, routed_scaling_factor=2.5,
    topk_indices=None, topk_weights=None,
    num_valid_tokens=None,
    latent_out=None, latent_offset=None,
):
    """
    CUDA gate kernel: fused sigmoid + top-k + normalize + scale (K2.5, K3).

    Algorithm:
        1. sigmoid(logits) → scores
        2. scores + e_score_correction → biased (for selection only)
        3. topk(biased, k) → indices
        4. gather raw sigmoid scores at indices → weights
        5. normalize weights, multiply by routed_scaling_factor

    ``router_logits`` rows may be strided (e.g. the leading expert columns of a
    fused router/down-projection GEMM output); only each row itself has to be
    contiguous.

    Args:
        router_logits: [N, E] FP32 (row stride may exceed E)
        e_score_correction: [E] FP32
        k: top-k (default 8; supported 2, 4, 8, 16)
        routed_scaling_factor: scaling factor (default 2.5)
        topk_indices: [N, K] int32 pre-allocated output (optional)
        topk_weights: [N, K] FP32 pre-allocated output (optional)
        num_valid_tokens: device int32 scalar (optional). Rows at or beyond it
            are written as index -1 / weight 0 without a host read, so a
            captured CUDA graph can vary the live row count across replays.
        latent_out: [N, L] BF16 pre-allocated output (optional, K3). When given,
            the kernel also casts columns ``[latent_offset, latent_offset + L)``
            of the same ``router_logits`` rows to BF16, which removes the
            separate strided FP32->BF16 contiguous copy. Rows at or beyond
            ``num_valid_tokens`` are written as zero.
        latent_offset: first latent column within the ``router_logits`` row
            stride. Required whenever ``latent_out`` is given.

    Returns:
        topk_indices: [N, K] int32
        topk_weights: [N, K] FP32
    """
    ext = _get_ext()
    N = router_logits.shape[0]
    device = router_logits.device

    if router_logits.dtype != torch.float32:
        if latent_out is not None:
            # The cast would repack the router columns into their own buffer,
            # detaching them from the fused row the latent suffix lives in.
            raise ValueError("latent_out requires FP32 router_logits")
        router_logits = router_logits.float()
    if e_score_correction.dtype != torch.float32:
        e_score_correction = e_score_correction.float()

    if topk_indices is None:
        topk_indices = torch.empty(N, k, dtype=torch.int32, device=device)
    if topk_weights is None:
        topk_weights = torch.empty(N, k, dtype=torch.float32, device=device)
    args = (
        router_logits, e_score_correction, k, routed_scaling_factor,
        topk_indices, topk_weights,
    )
    # Preserve the legacy K2.5/GLM call exactly: the extension's trailing
    # arguments are Python optionals, so callers without a live-row scalar or a
    # latent epilogue should not create even a zero-sized CUDA tensor inside
    # graph capture.
    if latent_out is not None:
        if latent_offset is None:
            raise ValueError("latent_offset is required when latent_out is given")
        result = ext.gate_sigmoid_topk(
            *args, num_valid_tokens, latent_out, int(latent_offset)
        )
    elif num_valid_tokens is not None:
        result = ext.gate_sigmoid_topk(*args, num_valid_tokens)
    else:
        result = ext.gate_sigmoid_topk(*args)
    return result[0], result[1]


def router_bias_cast_cuda(logits, bias, output):
    """Fused router epilogue: BF16 bias add + BF16→FP32 cast.

    Args:
        logits: [N, E] BF16 (router matmul output)
        bias: [E] BF16 (router bias, or empty tensor if no bias)
        output: [N, E] FP32 (pre-allocated output for gate kernel)
    """
    _get_ext().router_bias_cast(logits, bias, output)


def glm5_router_gemm_cuda(
    hidden_states,
    router_weight,
    router_logits=None,
    rank_token_counts=None,
    bucket_size=0,
    world_size=1,
):
    """GLM-5 graph-safe router GEMM.

    Computes ``hidden_states @ router_weight.T`` into FP32 logits from BF16
    inputs. If ``rank_token_counts`` is provided, rows are interpreted as
    rank-major ``[world_size, bucket_size]`` and invalid padding rows are
    zeroed inside the CUDA kernel using device-side counts.
    """
    ext = _get_ext()
    if hidden_states.dtype != torch.bfloat16:
        hidden_states = hidden_states.to(torch.bfloat16)
    if router_weight.dtype != torch.bfloat16:
        router_weight = router_weight.to(torch.bfloat16)
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not router_weight.is_contiguous():
        router_weight = router_weight.contiguous()

    N = hidden_states.shape[0]
    E = router_weight.shape[0]
    device = hidden_states.device
    if router_logits is None:
        router_logits = torch.empty(N, E, dtype=torch.float32, device=device)
    elif not router_logits.is_contiguous():
        raise ValueError("router_logits must be contiguous")

    if rank_token_counts is None:
        rank_token_counts = torch.empty(0, dtype=torch.int64, device=device)
    else:
        if rank_token_counts.dtype != torch.int64:
            rank_token_counts = rank_token_counts.to(torch.int64)
        if not rank_token_counts.is_contiguous():
            rank_token_counts = rank_token_counts.contiguous()

    return ext.glm5_router_gemm(
        hidden_states,
        router_weight,
        rank_token_counts,
        router_logits,
        int(bucket_size),
        int(world_size),
    )


def dispatch_count_gather_cuda(
    x, topk_indices,
    expert_start, num_local_experts,
    expert_counts=None, expert_offsets=None,
    expert_counters=None, dispatched_x=None, topk_pos=None,
    num_valid_tokens=-1,
):
    """
    CUDA dispatch kernel: count+prefix_sum + gather (2 fused kernels).

    Args:
        x: [N, H] BF16 token activations
        topk_indices: [N, K] int32 expert assignments
        expert_start: first local expert index
        num_local_experts: number of local experts
        expert_counts/offsets/counters/dispatched_x/topk_pos: pre-allocated (optional)
        num_valid_tokens: only process first num_valid_tokens tokens (-1 = all)

    Returns:
        dispatched_x: [max_dispatched, H] BF16
        expert_counts: [E_local] int32
        expert_offsets: [E_local+1] int32
        topk_pos: [N*K] int32
    """
    ext = _get_ext()

    # Kernel requires int32 indices
    if topk_indices.dtype != torch.int32:
        topk_indices = topk_indices.to(torch.int32)

    N, K = topk_indices.shape
    H = x.shape[1]
    NK = N * K
    device = x.device
    E_local = num_local_experts

    # Allocate outputs if not pre-allocated
    if expert_counts is None:
        expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
    else:
        expert_counts.zero_()

    if expert_offsets is None:
        expert_offsets = torch.empty(E_local + 1, dtype=torch.int32, device=device)

    if expert_counters is None:
        expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
    else:
        expert_counters.zero_()

    if topk_pos is None:
        topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)

    if dispatched_x is None:
        dispatched_x = torch.empty(NK, H, dtype=x.dtype, device=device)

    result = ext.dispatch_count_gather(
        x, topk_indices,
        expert_start, num_local_experts,
        expert_counts, expert_offsets,
        expert_counters, dispatched_x, topk_pos,
        num_valid_tokens,
    )
    return result[0], result[1], result[2], result[3]


class FusedGateContext:
    """Cached context for fused WGMMA GEMM + bias + TopK + Softmax (SM90a).

    Caches weight transpose, TMA descriptor for B, encode_func, and
    cudaFuncSetAttribute at init time. Per-call only creates TMA desc
    for A (input changes) and launches 2 kernels.

    Must be created at model init (once per MoE layer) and kept alive
    for the lifetime of the model. The weight tensor must not be
    reallocated after context creation (TMA descriptor points to it).

    Usage:
        ctx = FusedGateContext(router_weight, router_bias, topk=4)
        topk_idx, topk_wt = ctx.forward(hidden_states)
        # or with pre-allocated outputs and num_valid_tokens:
        topk_idx, topk_wt = ctx.forward(
            hidden_states, logits=buf_logits,
            topk_indices=buf_idx, topk_weights=buf_wt,
            num_valid_tokens=actual_B)
    """

    def __init__(self, router_weight, router_bias, topk=4):
        """Create fused gate context.

        Args:
            router_weight: [E, K_dim] BF16 (nn.Linear weight, kept as-is)
            router_bias: [E] BF16 (or empty tensor if no bias)
            topk: number of top experts (2, 4, or 8)
        """
        ext = _get_ext()
        bias = router_bias if router_bias is not None else torch.empty(
            0, dtype=torch.bfloat16, device=router_weight.device)
        self._ctx = ext.create_fused_gate_context(router_weight, bias, topk)
        self._ext = ext

    def warmup(self, base_buffer):
        """Create TMA descriptor for input against the full base buffer.

        Call once with the base buffer (not per-bucket views) before CUDA
        graph capture. All per-bucket views share the same data_ptr, so
        one TMA desc covers all bucket sizes. The kernel grid controls
        which rows are actually processed per bucket.

        Args:
            base_buffer: [WB_max, H] BF16 — the full allocated buffer
        """
        self._ext.fused_gate_warmup(self._ctx, base_buffer)

    def forward(self, hidden_states, logits=None, topk_indices=None,
                topk_weights=None, num_valid_tokens=-1):
        """Run fused gate: WGMMA GEMM + bias + TopK + Softmax.

        Args:
            hidden_states: [N, K_dim] BF16
            logits: [N, E] FP32 pre-allocated (optional)
            topk_indices: [N, K] int32 pre-allocated (optional)
            topk_weights: [N, K] FP32 pre-allocated (optional)
            num_valid_tokens: only process first N tokens (-1 = all)

        Returns:
            topk_indices: [N, K] int32
            topk_weights: [N, K] FP32
        """
        _empty = torch.empty(0)
        result = self._ext.fused_gate_forward(
            self._ctx,
            hidden_states,
            logits if logits is not None else _empty,
            topk_indices if topk_indices is not None else _empty,
            topk_weights if topk_weights is not None else _empty,
            num_valid_tokens,
        )
        return result[0], result[1]

    def __del__(self):
        if hasattr(self, '_ctx') and hasattr(self, '_ext'):
            self._ext.destroy_fused_gate_context(self._ctx)


def reduce_weighted_scatter_cuda(
    expert_output, topk_pos, topk_weights, N, H=None, K=4,
    output=None, num_valid_tokens=-1,
):
    """
    CUDA reduce kernel: weighted scatter-add.

    Args:
        expert_output: [total_dispatched, H] BF16
        topk_pos: [N*K] int32 (-1 for non-local)
        topk_weights: [N, K] FP32
        N: number of original tokens
        H: hidden size (auto-detected if None)
        K: top-k (default 4)
        output: [N, H] BF16 pre-allocated output (optional)
        num_valid_tokens: only process first num_valid_tokens tokens (-1 = all)

    Returns:
        output: [N, H] BF16
    """
    ext = _get_ext()

    # Kernel requires FP32 weights for accumulation precision
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.float()

    if H is None:
        H = expert_output.shape[1]
    device = expert_output.device

    if output is None:
        output = torch.empty(N, H, dtype=torch.bfloat16, device=device)

    return ext.reduce_weighted_scatter(
        expert_output, topk_pos, topk_weights,
        N, H, K, output, num_valid_tokens,
    )
