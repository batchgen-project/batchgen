"""Packed GLM-5.2 DSA prefill with sparse absorbed MLA.

The existing GLM prefill path expands KV-B into full per-head K/V and runs
dense causal FA3 on every layer.  This module keeps the checkpoint's DSA
semantics instead:

* full indexer layers score every causal key and select ``index_topk``;
* shared layers consume the previous full layer's selected indices;
* every layer runs FlashMLA sparse prefill on compressed latent KV.

Inputs are the worker's packed (variable-length) prompt tensors.  Selected
indices are packed-global offsets, which is the layout consumed by
``flash_mla_sparse_fwd``.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_VALIDATED_RUNTIME_DEVICES: set[int] = set()


@dataclass(frozen=True)
class Glm52SparsePrefillResult:
    attn_output: torch.Tensor
    primary_kv: torch.Tensor
    indexer_kv: torch.Tensor | None
    topk_indices: torch.Tensor


def should_use_glm52_sparse_prefill(
    model_type: str | None,
    max_seqlen: int,
    index_topk: int,
) -> bool:
    return model_type == "glm_moe_dsa_5_2" and max_seqlen > index_topk


def build_packed_causal_ranges(
    cu_seqlens: torch.Tensor,
    position_ids: torch.Tensor,
    total_tokens: int,
    sequence_lengths: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed-global ``[start, end)`` key ranges for every query token."""

    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1-D tensor with at least two entries")
    if position_ids.shape != (total_tokens,):
        raise ValueError(
            f"position_ids must have shape {(total_tokens,)}, got {tuple(position_ids.shape)}"
        )
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
    if position_ids.dtype != torch.int64:
        raise TypeError(f"position_ids must be int64, got {position_ids.dtype}")
    if cu_seqlens.device != position_ids.device:
        raise ValueError("cu_seqlens and position_ids must be on the same device")
    if sequence_lengths is not None:
        if len(sequence_lengths) != cu_seqlens.numel() - 1:
            raise ValueError("sequence_lengths and cu_seqlens batch size mismatch")
        if any(length <= 0 for length in sequence_lengths):
            raise ValueError("sequence_lengths must be positive")
        if sum(sequence_lengths) != total_tokens:
            raise ValueError(
                "sequence_lengths must span the complete packed token tensor"
            )
    elif not cu_seqlens.is_cuda:
        cu_cpu = cu_seqlens.tolist()
        if cu_cpu[0] != 0 or cu_cpu[-1] != total_tokens:
            raise ValueError(
                "cu_seqlens must span the complete packed token tensor: "
                f"first={cu_cpu[0]}, last={cu_cpu[-1]}, total_tokens={total_tokens}"
            )
        if any(end <= start for start, end in zip(cu_cpu, cu_cpu[1:])):
            raise ValueError(
                "cu_seqlens must contain strictly increasing sequence spans"
            )

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    starts = torch.repeat_interleave(
        cu_seqlens[:-1],
        lengths,
        output_size=total_tokens,
    )
    positions_i32 = position_ids.to(dtype=torch.int32)
    if not position_ids.is_cuda:
        expected_positions = torch.arange(
            total_tokens,
            dtype=torch.int32,
            device=position_ids.device,
        ) - starts
        if not torch.equal(positions_i32, expected_positions):
            raise ValueError(
                "position_ids must restart at zero and increase by one inside each "
                "packed sequence"
            )
    ends = starts + positions_i32 + 1
    return starts.contiguous(), ends.contiguous()


@triton.jit
def _offset_topk_indices_kernel(
    indices_ptr,
    starts_ptr,
    n_elements: tl.constexpr,
    topk: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    values = tl.load(indices_ptr + offsets, mask=mask, other=-1)
    rows = offsets // topk
    starts = tl.load(starts_ptr + rows, mask=mask, other=0)
    values = tl.where(values >= 0, values + starts, values)
    tl.store(indices_ptr + offsets, values, mask=mask)


def offset_packed_topk_indices_(
    indices: torch.Tensor,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    """Convert row-relative top-k indices to packed-global offsets in place."""

    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("indices must be a contiguous int32 [tokens, topk] tensor")
    if row_starts.shape != (indices.shape[0],) or row_starts.dtype != torch.int32:
        raise ValueError("row_starts must be int32 with one entry per index row")
    if not indices.is_contiguous() or not row_starts.is_contiguous():
        raise ValueError("indices and row_starts must be contiguous")

    n_elements = indices.numel()
    block = 256
    _offset_topk_indices_kernel[(triton.cdiv(n_elements, block),)](
        indices,
        row_starts,
        n_elements=n_elements,
        topk=indices.shape[1],
        block=block,
    )
    return indices


def validate_carried_topk(
    carried_topk_indices: torch.Tensor | None,
    total_tokens: int,
    index_topk: int,
) -> torch.Tensor:
    if carried_topk_indices is None:
        raise RuntimeError(
            "GLM-5.2 shared prefill layer has no carried top-k indices"
        )
    if carried_topk_indices.shape != (total_tokens, index_topk):
        raise RuntimeError(
            "GLM-5.2 shared prefill top-k shape mismatch: "
            f"expected {(total_tokens, index_topk)}, "
            f"got {tuple(carried_topk_indices.shape)}"
        )
    if carried_topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "GLM-5.2 shared prefill top-k must be int32, "
            f"got {carried_topk_indices.dtype}"
        )
    return carried_topk_indices


def _score_chunk_rows(max_seqlen: int) -> int:
    """Keep the temporary FP32 logits matrix at or below two GiB."""

    score_width = ((max_seqlen + 255) // 256) * 256
    bytes_per_row = score_width * 4
    rows = min(8192, (2 * 1024**3) // max(bytes_per_row, 1))
    # DeepGEMM rounds the query dimension to four rows internally.
    rows = rows // 4 * 4
    return max(4, rows)


def _fp8_linear_from_quantized(
    weight_data_fp8: torch.Tensor,
    weight_scale_inv_fp32: torch.Tensor,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
) -> torch.Tensor:
    """DeepGEMM linear using a caller-owned activation quantization."""

    import deep_gemm

    if x_fp8.ndim != 2 or x_scale.ndim != 2:
        raise ValueError("prequantized activation must be two-dimensional")
    output = torch.empty(
        x_fp8.shape[0],
        weight_data_fp8.shape[0],
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    deep_gemm.fp8_gemm_nt(
        (x_fp8, x_scale),
        (weight_data_fp8, weight_scale_inv_fp32),
        output,
    )
    return output


def _fused_rope_hadamard_q(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply the corrected fused RoPE/Hadamard kernel to multi-head Q."""

    from batchgen.other_kernels.hadamard_transform import fused_rope_hadamard

    batch_size, num_heads, head_dim = q.shape
    positions_expanded = position_ids.reshape(batch_size).repeat_interleave(num_heads)
    return fused_rope_hadamard(
        q.reshape(batch_size * num_heads, head_dim),
        cos.float(),
        sin.float(),
        positions_expanded,
        scale=head_dim**-0.5,
    ).reshape(batch_size, num_heads, head_dim)


def _compute_indexer_kv_from_quantized_hidden(
    *,
    indexer,
    hidden_fp8: torch.Tensor,
    hidden_scale: torch.Tensor,
    position_ids: torch.Tensor,
    max_seqlen: int,
    timed,
) -> torch.Tensor:
    if not hasattr(indexer, "wk_scale"):
        raise RuntimeError("GLM-5.2 sparse prefill requires indexer.wk FP8 scales")
    with timed("indexer_wk"):
        k = _fp8_linear_from_quantized(
            indexer.wk.weight.data,
            indexer.wk_scale,
            hidden_fp8,
            hidden_scale,
        )
    with timed("indexer_norm"):
        k = indexer.k_norm(k)
    with timed("indexer_rope_hadamard"):
        try:
            from batchgen.other_kernels.hadamard_transform import fused_rope_hadamard
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "GLM-5.2 sparse prefill requires the fused RoPE/Hadamard kernel"
            ) from exc
        cos, sin = indexer.rotary_emb(k, max_seqlen)
        k = fused_rope_hadamard(
            k.to(torch.bfloat16),
            cos.float(),
            sin.float(),
            position_ids.reshape(-1),
            scale=indexer.index_head_dim**-0.5,
        )
        indexer.record_prefill_rope_hadamard_path("fused", indexer.layer_idx)
    return k.unsqueeze(0).unsqueeze(2)


def select_packed_glm52_topk(
    *,
    indexer,
    hidden_states: torch.Tensor,
    q_a_fp8: torch.Tensor,
    q_a_scale: torch.Tensor,
    indexer_kv: torch.Tensor,
    position_ids: torch.Tensor,
    causal_starts: torch.Tensor,
    causal_ends: torch.Tensor,
    max_seqlen: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run GLM-5.2's causal FP8 indexer over packed prompt tokens."""

    import deep_gemm

    from batchgen.attention.mla.fa3_backend import act_quant, w8a16_gemm
    from batchgen_kernels.attention.dsa.fast_topk_cuda import fast_topk_2048_out

    if not hasattr(deep_gemm, "fp8_mqa_logits"):
        raise RuntimeError("GLM-5.2 sparse prefill requires deep_gemm.fp8_mqa_logits")
    if not hasattr(indexer, "wq_b_scale"):
        raise RuntimeError("GLM-5.2 sparse prefill requires indexer.wq_b FP8 scales")
    if indexer.index_topk != 2048:
        raise RuntimeError(
            f"GLM-5.2 sparse prefill currently requires index_topk=2048, "
            f"got {indexer.index_topk}"
        )

    total_tokens = hidden_states.shape[0]
    if causal_starts.shape != (total_tokens,) or causal_ends.shape != (total_tokens,):
        raise ValueError("causal starts/ends must have one entry per packed token")
    causal_lengths = (causal_ends - causal_starts).contiguous()

    cos, sin = indexer.rotary_emb(
        indexer_kv[:1, :1, :1, : indexer.rope_head_dim],
        max_seqlen,
    )

    packed_k = indexer_kv[0, :, 0, :].contiguous()
    k_fp8, k_scale = act_quant(packed_k)
    k_fp8_with_scale = (k_fp8, k_scale.squeeze(-1).contiguous())

    if output is None:
        topk_indices = torch.empty(
            total_tokens,
            indexer.index_topk,
            dtype=torch.int32,
            device=hidden_states.device,
        )
    else:
        topk_indices = validate_carried_topk(
            output,
            total_tokens,
            indexer.index_topk,
        )
        if topk_indices.device != hidden_states.device:
            raise RuntimeError(
                "reused GLM-5.2 prefill top-k buffer is on the wrong device"
            )
    chunk_rows = _score_chunk_rows(max_seqlen)
    for start in range(0, total_tokens, chunk_rows):
        end = min(start + chunk_rows, total_tokens)
        q = _fp8_linear_from_quantized(
            indexer.wq_b.weight.data,
            indexer.wq_b_scale,
            q_a_fp8[start:end],
            q_a_scale[start:end],
        ).view(end - start, indexer.index_n_heads, indexer.index_head_dim)
        q = _fused_rope_hadamard_q(
            q,
            cos,
            sin,
            position_ids[start:end],
        )
        q_fp8, q_scale = act_quant(q)
        head_gates = torch.mm(
            hidden_states[start:end],
            indexer.weights_proj.weight.data.t(),
            out_dtype=torch.float32,
        )
        head_gates.mul_(indexer.index_n_heads**-0.5)
        score_weights = (
            head_gates.unsqueeze(-1) * q_scale * indexer.softmax_scale
        ).squeeze(-1).contiguous()
        _score_packed_indexer_topk_chunk(
            q_fp8=q_fp8,
            k_fp8_with_scale=k_fp8_with_scale,
            score_weights=score_weights,
            causal_starts=causal_starts[start:end],
            causal_ends=causal_ends[start:end],
            causal_lengths=causal_lengths[start:end],
            output=topk_indices[start:end],
            max_seqlen=max_seqlen,
        )
        del q, q_fp8, q_scale, head_gates, score_weights

    del packed_k, k_fp8_with_scale
    return topk_indices


def _score_packed_indexer_topk_chunk(
    *,
    q_fp8: torch.Tensor,
    k_fp8_with_scale: tuple[torch.Tensor, torch.Tensor],
    score_weights: torch.Tensor,
    causal_starts: torch.Tensor,
    causal_ends: torch.Tensor,
    causal_lengths: torch.Tensor,
    output: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Score one packed query chunk and emit packed-global top-k indices."""

    import deep_gemm

    from batchgen_kernels.attention.dsa.fast_topk_cuda import fast_topk_2048_out

    score_width = ((max_seqlen + 255) // 256) * 256
    logits = deep_gemm.fp8_mqa_logits(
        q_fp8,
        k_fp8_with_scale,
        score_weights,
        causal_starts,
        causal_ends,
        max_seqlen_k=score_width,
        clean_logits=False,
    )
    if not logits.is_contiguous():
        raise RuntimeError(
            "DeepGEMM compressed indexer logits must be contiguous when "
            f"score_width={score_width}, got stride={tuple(logits.stride())}"
        )
    fast_topk_2048_out(logits, causal_lengths, output)
    offset_packed_topk_indices_(output, causal_starts)
    return output


def _reference_rope_hadamard(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Small independent reference used only by the startup smoke gate."""

    original_shape = x.shape
    transformed = x.reshape(-1, 128).float().clone()
    positions = position_ids.reshape(-1).long()
    cos_rows = cos.index_select(0, positions)[:, :32].float()
    sin_rows = sin.index_select(0, positions)[:, :32].float()
    even = transformed[:, :64:2].clone()
    odd = transformed[:, 1:64:2].clone()
    transformed[:, :64:2] = even * cos_rows - odd * sin_rows
    transformed[:, 1:64:2] = odd * cos_rows + even * sin_rows

    stride = 1
    while stride < 128:
        groups = transformed.reshape(-1, 128 // (2 * stride), 2, stride)
        low = groups[:, :, 0].clone()
        high = groups[:, :, 1].clone()
        transformed = torch.stack((low + high, low - high), dim=2).reshape(-1, 128)
        stride *= 2
    return (transformed * scale).to(x.dtype).reshape(original_shape)


def validate_glm52_sparse_prefill_runtime() -> None:
    """Fail at prefill model setup if the required sparse runtime is absent."""

    device_index = torch.cuda.current_device()
    if device_index in _VALIDATED_RUNTIME_DEVICES:
        return

    try:
        import deep_gemm
    except ImportError as exc:
        raise RuntimeError(
            "GLM-5.2 sparse prefill requires the pinned DeepGEMM runtime"
        ) from exc
    if not hasattr(deep_gemm, "fp8_mqa_logits"):
        raise RuntimeError("GLM-5.2 sparse prefill requires deep_gemm.fp8_mqa_logits")
    try:
        from flash_mla import flash_mla_sparse_fwd
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "GLM-5.2 sparse prefill requires FlashMLA "
            "flash_mla_sparse_fwd from the pinned dependency"
        ) from exc
    if not callable(flash_mla_sparse_fwd):
        raise RuntimeError("flash_mla.flash_mla_sparse_fwd is not callable")
    try:
        from batchgen.other_kernels.hadamard_transform import fused_rope_hadamard
        from batchgen_kernels.attention.dsa.fast_topk_cuda import _get_module
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "GLM-5.2 sparse prefill requires fused top-k and "
            "RoPE/Hadamard kernels"
        ) from exc
    if not callable(fused_rope_hadamard):
        raise RuntimeError("fused_rope_hadamard is not callable")
    topk_module = _get_module()

    # Exercise the exact sparse-prefill ABI before a long model forward.
    device = torch.device("cuda", device_index)
    angles = (
        torch.arange(4 * 64, dtype=torch.float32, device=device).reshape(4, 64)
        * 0.003
    )
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rope_input = (
        torch.arange(2 * 128, dtype=torch.float32, device=device).reshape(2, 128)
        .remainder(31)
        .sub_(15)
        .mul_(0.03125)
        .to(torch.bfloat16)
    )
    rope_positions = torch.tensor([0, 3], dtype=torch.int64, device=device)
    rope_out = fused_rope_hadamard(
        rope_input,
        cos,
        sin,
        rope_positions,
    )
    rope_reference = _reference_rope_hadamard(
        rope_input,
        cos,
        sin,
        rope_positions,
        128**-0.5,
    )
    if (
        rope_out.shape != rope_input.shape
        or rope_out.dtype != torch.bfloat16
        or not rope_out.is_contiguous()
        or not torch.allclose(
            rope_out.float(),
            rope_reference.float(),
            atol=2e-2,
            rtol=2e-2,
        )
    ):
        raise RuntimeError(
            "fused_rope_hadamard failed its nonzero numerical smoke gate: "
            f"shape={tuple(rope_out.shape)}, dtype={rope_out.dtype}, "
            f"contiguous={rope_out.is_contiguous()}"
        )
    q_input = (
        torch.arange(
            2 * 32 * 128,
            dtype=torch.float32,
            device=device,
        )
        .reshape(2, 32, 128)
        .remainder(29)
        .sub_(14)
        .mul_(0.03125)
        .to(torch.bfloat16)
    )
    q_positions = torch.tensor([1, 3], dtype=torch.int64, device=device)
    q_rope_out = _fused_rope_hadamard_q(
        q_input,
        cos,
        sin,
        q_positions,
    )
    q_reference = _reference_rope_hadamard(
        q_input,
        cos,
        sin,
        q_positions.repeat_interleave(32),
        128**-0.5,
    )
    if (
        q_rope_out.shape != q_input.shape
        or q_rope_out.dtype != torch.bfloat16
        or not q_rope_out.is_contiguous()
        or not torch.allclose(
            q_rope_out.float(),
            q_reference.float(),
            atol=2e-2,
            rtol=2e-2,
        )
    ):
        raise RuntimeError(
            "multi-head fused_rope_hadamard failed its nonzero numerical smoke "
            f"gate: shape={tuple(q_rope_out.shape)}, dtype={q_rope_out.dtype}, "
            f"contiguous={q_rope_out.is_contiguous()}"
        )

    score = torch.zeros(1, 2048, dtype=torch.float32, device=device)
    lengths = torch.ones(1, dtype=torch.int32, device=device)
    selected = torch.empty(1, 2048, dtype=torch.int32, device=device)
    topk_module.fast_topk_2048_out(score, lengths, selected, None)
    if selected[0, 0].item() != 0:
        raise RuntimeError("fast_topk_2048_out failed its dense-prefix smoke gate")
    offset_packed_topk_indices_(
        selected,
        torch.tensor([7], dtype=torch.int32, device=device),
    )
    if selected[0, 0].item() != 7:
        raise RuntimeError("packed top-k offset smoke gate failed")

    q_index = torch.zeros(
        1, 32, 128, dtype=torch.float8_e4m3fn, device=device
    )
    k_index = torch.zeros(1, 128, dtype=torch.float8_e4m3fn, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    weights = torch.zeros(1, 32, dtype=torch.float32, device=device)
    starts = torch.zeros(1, dtype=torch.int32, device=device)
    ends = torch.ones(1, dtype=torch.int32, device=device)
    logits = deep_gemm.fp8_mqa_logits(
        q_index,
        (k_index, k_scale),
        weights,
        starts,
        ends,
        max_seqlen_k=256,
        clean_logits=False,
    )
    if logits.shape != (1, 256) or not logits.is_contiguous():
        raise RuntimeError(
            "deep_gemm.fp8_mqa_logits returned incompatible compressed logits "
            f"shape/stride {tuple(logits.shape)}/{tuple(logits.stride())}"
        )

    q = torch.zeros(1, 64, 576, dtype=torch.bfloat16, device=device)
    kv = torch.zeros(32, 1, 576, dtype=torch.bfloat16, device=device)
    indices = torch.full((1, 1, 2048), -1, dtype=torch.int32, device=device)
    indices[0, 0, 0] = 0
    result = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale=256**-0.5,
        d_v=512,
    )
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise RuntimeError(
            "flash_mla_sparse_fwd must return (output, max_logits, lse)"
        )
    if result[0].shape != (1, 64, 512):
        raise RuntimeError(
            "flash_mla_sparse_fwd returned incompatible output shape "
            f"{tuple(result[0].shape)}"
        )
    torch.cuda.current_stream(device).synchronize()
    _VALIDATED_RUNTIME_DEVICES.add(device_index)


@torch.inference_mode()
def glm52_sparse_prefill_prepacked(
    *,
    attn,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    max_seqlen: int,
    weight_scale: dict[str, torch.Tensor],
    indexer,
    carried_topk_indices: torch.Tensor | None,
    reusable_topk_indices: torch.Tensor | None,
    causal_starts: torch.Tensor | None,
    causal_ends: torch.Tensor | None,
) -> Glm52SparsePrefillResult:
    """Execute one GLM-5.2 packed sparse-attention prefill layer."""

    try:
        from flash_mla import flash_mla_sparse_fwd
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "GLM-5.2 sparse prefill requires FlashMLA "
            "flash_mla_sparse_fwd from the pinned dependency"
        ) from exc

    from batchgen.attention.mla.fa3_backend import act_quant, w8a16_gemm
    from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization
    from batchgen.attention.mla.rotary_embedding import (
        rotary_pos_emb_interleaved_native,
    )

    total_tokens = hidden_states.shape[0]
    from batchgen.timing import get_prefill_timer

    timer = get_prefill_timer()

    def timed(name: str):
        if timer is None:
            return nullcontext()
        return timer.timed(name, attn.layer_idx)

    with timed("attn_input_quant"):
        hidden_fp8, hidden_scale = act_quant(hidden_states)
    with timed("attn_q_a"):
        q_a = _fp8_linear_from_quantized(
            attn.q_a_proj.weight.data,
            weight_scale["q_a_proj.weight_scale_inv"],
            hidden_fp8,
            hidden_scale,
        )
    with timed("attn_q_norm"):
        q_a_normed = attn.q_a_layernorm(q_a)
    del q_a
    with timed("attn_q_a_quant"):
        q_a_fp8, q_a_scale = act_quant(q_a_normed)
    del q_a_normed
    with timed("attn_kv_a"):
        compressed_kv = _fp8_linear_from_quantized(
            attn.kv_a_proj_with_mqa.weight.data,
            weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
            hidden_fp8,
            hidden_scale,
        )
    compressed_kv, k_pe = torch.split(
        compressed_kv,
        [attn.kv_lora_rank, attn.qk_rope_head_dim],
        dim=-1,
    )
    with timed("attn_kv_norm"):
        normed_kv = attn.kv_a_layernorm(compressed_kv)
    k_pe = k_pe.view(total_tokens, 1, attn.qk_rope_head_dim)

    with timed("attn_rope"):
        cos, sin = attn.rotary_emb(k_pe, seq_len=max_seqlen)
        k_pe = rotary_pos_emb_interleaved_native(
            k_pe.unsqueeze(0),
            cos,
            sin,
            position_ids.unsqueeze(0),
            2,
        ).squeeze(0)

    with timed("attn_primary_kv_materialize"):
        primary_kv = torch.cat(
            [normed_kv, k_pe.view(total_tokens, attn.qk_rope_head_dim)],
            dim=-1,
        ).contiguous()
    del compressed_kv, normed_kv, k_pe

    indexer_kv = None
    if indexer is not None:
        if causal_starts is None or causal_ends is None:
            raise RuntimeError(
                "GLM-5.2 full sparse-prefill layer requires packed causal ranges"
            )
        indexer_kv = _compute_indexer_kv_from_quantized_hidden(
            indexer=indexer,
            hidden_fp8=hidden_fp8,
            hidden_scale=hidden_scale,
            position_ids=position_ids,
            max_seqlen=max_seqlen,
            timed=timed,
        )
        with timed("indexer_prefill_score"):
            topk_indices = select_packed_glm52_topk(
                indexer=indexer,
                hidden_states=hidden_states,
                q_a_fp8=q_a_fp8,
                q_a_scale=q_a_scale,
                indexer_kv=indexer_kv,
                position_ids=position_ids,
                causal_starts=causal_starts,
                causal_ends=causal_ends,
                max_seqlen=max_seqlen,
                output=reusable_topk_indices,
            )
    else:
        topk_indices = validate_carried_topk(
            carried_topk_indices,
            total_tokens,
            attn.config.index_topk,
        )
    del hidden_fp8, hidden_scale

    with timed("attn_kv_b_dequant"):
        kv_b = deepseek_v3_dequantization(
            attn.kv_b_proj.weight.data,
            weight_scale["kv_b_proj.weight_scale_inv"],
        ).view(
            attn.num_heads,
            attn.qk_nope_head_dim + attn.v_head_dim,
            attn.kv_lora_rank,
        )
    q_absorb = kv_b[:, : attn.qk_nope_head_dim, :]
    out_absorb = kv_b[:, attn.qk_nope_head_dim :, :].transpose(1, 2)
    kv_sparse = primary_kv.unsqueeze(1)
    attn_output = torch.empty(
        total_tokens,
        attn.hidden_size,
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    query_chunk_rows = 8192
    for start in range(0, total_tokens, query_chunk_rows):
        end = min(start + query_chunk_rows, total_tokens)
        with timed("attn_q_b"):
            q = _fp8_linear_from_quantized(
                attn.q_b_proj.weight.data,
                weight_scale["q_b_proj.weight_scale_inv"],
                q_a_fp8[start:end],
                q_a_scale[start:end],
            ).view(end - start, attn.num_heads, attn.q_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [attn.qk_nope_head_dim, attn.qk_rope_head_dim],
            dim=-1,
        )
        with timed("attn_rope"):
            q_pe = rotary_pos_emb_interleaved_native(
                q_pe.unsqueeze(0),
                cos,
                sin,
                position_ids[start:end].unsqueeze(0),
                2,
            ).squeeze(0)
        with timed("attn_sparse_q_absorb"):
            q_latent = torch.bmm(
                q_nope.transpose(0, 1),
                q_absorb,
            ).transpose(0, 1)
            q_sparse = torch.cat([q_latent, q_pe], dim=-1).contiguous()
        with timed("attn_sparse_flashmla"):
            attn_latent, _, _ = flash_mla_sparse_fwd(
                q_sparse,
                kv_sparse,
                topk_indices[start:end].unsqueeze(1),
                sm_scale=attn.softmax_scale,
                d_v=attn.kv_lora_rank,
            )
        with timed("attn_sparse_out_absorb"):
            attn_heads = torch.bmm(
                attn_latent.transpose(0, 1),
                out_absorb,
            ).transpose(0, 1)
        with timed("attn_o"):
            attn_output[start:end] = w8a16_gemm(
                attn.o_proj.weight.data,
                weight_scale["o_proj.weight_scale_inv"],
                attn_heads.reshape(
                    end - start,
                    attn.num_heads * attn.v_head_dim,
                ).contiguous(),
            )
        del q, q_nope, q_pe, q_latent, q_sparse, attn_latent, attn_heads
    del q_a_fp8, q_a_scale, q_absorb, out_absorb, kv_b, kv_sparse

    return Glm52SparsePrefillResult(
        attn_output=attn_output,
        primary_kv=primary_kv,
        indexer_kv=indexer_kv,
        topk_indices=topk_indices,
    )
