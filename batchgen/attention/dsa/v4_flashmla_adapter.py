from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any, Optional

import torch

from batchgen.attention.dsa.v4_indexer_metadata import (
    init_compressed_attention_metadata,
)
from batchgen.attention.dsa.v4_mla_torch_ref import (
    flashmla_decode_torch_reference,
)
from batchgen.attention.v4_backend import DSV4AttnMetadata
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator
from batchgen.timing import get_decode_timer

# Env-gated diagnostic (default OFF).
_V4_ATTN_PROBE = os.environ.get("BATCHGEN_V4_ATTN_PROBE", "0") == "1"
_V4_ATTN_PROBE_STEPS = int(os.environ.get("BATCHGEN_V4_ATTN_PROBE_STEPS", "1"))
# Tensor-level dump for offline diff vs the official reference's sparse_attn
# inputs (set to a directory path).
_V4_ATTN_TENSOR_DUMP = os.environ.get("BATCHGEN_V4_ATTN_TENSOR_DUMP", "")
_V4_ATTN_TENSOR_DUMP_LAYERS = 4
_v4_attn_tensor_dump_calls: dict[int, int] = {}


def _v4_dump_attn_tensors(
    *,
    layer_idx: int,
    q_roped: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    extra_indices: Optional[torch.Tensor],
    extra_lengths: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    attn_out: torch.Tensor,
    coordinator: Any,
) -> None:
    if layer_idx >= _V4_ATTN_TENSOR_DUMP_LAYERS:
        return
    call_no = _v4_attn_tensor_dump_calls.get(layer_idx, 0)
    if call_no >= 1:
        return
    _v4_attn_tensor_dump_calls[layer_idx] = call_no + 1
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )

    def _gather_rows(idx, lengths, cache):
        row = idx[0, 0]
        n = int(lengths[0].item())
        valid = row[:n].to(torch.long)
        flat_bytes = cache.reshape(-1, cache.shape[-1])
        sel = flat_bytes.index_select(0, valid.clamp_min(0))
        return valid.cpu(), sel.cpu()

    payload = {
        "layer_idx": layer_idx,
        "q_roped": q_roped.float().cpu(),
        "attn_sink": None if attn_sink is None else attn_sink.float().cpu(),
        "attn_out": attn_out.float().cpu(),
        "main_lengths": main_lengths.cpu(),
    }
    payload["main_idx"], payload["main_kv_bytes"] = _gather_rows(
        main_indices, main_lengths, k_cache
    )
    payload["main_kv_decoded"] = (
        coordinator.swa.debug_read_kv(
            layer_idx=coordinator.get_layer_routing(layer_idx).swa_layer_idx,
            token_slots=payload["main_idx"].to(k_cache.device),
        )
        .float()
        .cpu()
    )
    if extra_indices is not None and extra_k_cache is not None:
        payload["extra_idx"], payload["extra_kv_bytes"] = _gather_rows(
            extra_indices, extra_lengths, extra_k_cache
        )
    torch.save(
        payload,
        os.path.join(
            _V4_ATTN_TENSOR_DUMP,
            f"attn_dump_layer{layer_idx}_rank{rank}.pt",
        ),
    )


def _v4_mla_torch_default() -> bool:
    env = os.environ.get("BATCHGEN_V4_MLA_TORCH")
    if env is not None:
        return env == "1"
    # Auto-enable on Blackwell sm120 (no wgmma => custom FlashMLA cannot run there).
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()[0] == 12
    except Exception:
        return False
    return False


_V4_MLA_TORCH = _v4_mla_torch_default()


def _v4_mla_sm120_triton_default() -> bool:
    env = os.environ.get("BATCHGEN_V4_MLA_SM120_TRITON")
    if env is not None:
        return env == "1"
    # Default ON for the sm120 Triton path (replaces the slow torch reference).
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()[0] == 12
    except Exception:
        return False
    return False


_V4_MLA_SM120_TRITON = _v4_mla_sm120_triton_default()
_v4_attn_probe_calls: dict[int, int] = {}

_ROPE_DIM = 64
_HEAD_DIM = 512
_TOPK_ALIGN = 64
_SOFTMAX_SCALE = 512**-0.5
_SWA_WINDOW = 128
_V4_MLA_VALIDATE_INDICES = (
    os.environ.get("BATCHGEN_V4_MLA_VALIDATE_INDICES", "0") == "1"
)
_V4_FAST_PREFIX_INDICES = (
    os.environ.get("BATCHGEN_V4_FAST_PREFIX_INDICES", "1") == "1"
)


def _select_v4_mla_backend() -> str:
    if _V4_MLA_SM120_TRITON:
        return "sm120_triton"
    if _V4_MLA_TORCH:
        return "torch_ref"
    return "flashmla"


def build_v4_rope_cache(
    *,
    max_pos: int,
    theta: float,
    rope_head_dim: int = _ROPE_DIM,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device | str | int = "cpu",
) -> torch.Tensor:
    """Port of assets precompute_freqs_cis: complex [max_pos, rope_head_dim/2].

    Dense/SWA layers use original_seq_len=0 (YaRN off, base theta). Compressed
    layers use YaRN with original_seq_len>0 and compress_rope_theta.
    """
    freqs = _v4_rope_freqs(
        max_pos=max_pos,
        theta=theta,
        rope_head_dim=rope_head_dim,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    t = torch.arange(max_pos)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(device=device)


def _v4_rope_freqs(
    *,
    max_pos: int,
    theta: float,
    rope_head_dim: int,
    original_seq_len: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    dim = rope_head_dim
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    if original_seq_len > 0:
        low = math.floor(
            dim
            * math.log(original_seq_len / (beta_fast * 2 * math.pi))
            / (2 * math.log(theta))
        )
        high = math.ceil(
            dim
            * math.log(original_seq_len / (beta_slow * 2 * math.pi))
            / (2 * math.log(theta))
        )
        low = max(low, 0)
        high = min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low),
            0,
            1,
        )
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def build_v4_compress_cos_sin_cache(
    *,
    max_pos: int,
    theta: float,
    rope_head_dim: int = _ROPE_DIM,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device | str | int = "cpu",
) -> torch.Tensor:
    """[max_pos, rope_head_dim] = cat(cos, sin) for the kernel compressor _apply_rope."""
    freqs = _v4_rope_freqs(
        max_pos=max_pos,
        theta=theta,
        rope_head_dim=rope_head_dim,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(device=device)


def build_v4_rope_tables(
    *,
    max_pos: int,
    theta: float,
    rope_head_dim: int = _ROPE_DIM,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device | str | int = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [max_pos, rope_head_dim] for rope_hadamard_q (cos/sin repeated x2)."""
    freqs = _v4_rope_freqs(
        max_pos=max_pos,
        theta=theta,
        rope_head_dim=rope_head_dim,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    angles = t[:, None] * freqs[None, :]
    cos_table = torch.cos(angles).repeat(1, 2).to(device=device, dtype=dtype)
    sin_table = torch.sin(angles).repeat(1, 2).to(device=device, dtype=dtype)
    return cos_table, sin_table


def _to_int32_tensor(
    values: torch.Tensor | Sequence[int], *, device: torch.device
) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=device)
    return tensor.to(dtype=torch.int32, device=device)


def _normalize_rope_cache(rope_cache: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(rope_cache):
        return torch.cat((rope_cache.real, rope_cache.imag), dim=-1).to(
            dtype=torch.float32
        )
    if rope_cache.ndim == 3 and rope_cache.shape[1:] == (_ROPE_DIM, 2):
        half = _ROPE_DIM // 2
        cos = rope_cache[:, :, 0][:, :half]
        sin = rope_cache[:, :, 1][:, :half]
        return torch.cat((cos, sin), dim=-1).to(dtype=torch.float32)
    if rope_cache.ndim == 2 and rope_cache.shape[-1] == _ROPE_DIM:
        return rope_cache.to(dtype=torch.float32)
    raise ValueError(
        "rope_cache must be complex [max_pos, 32], real [max_pos, 64], "
        "or real [max_pos, 64, 2]"
    )


def _apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_cache: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if x.shape[-1] != _HEAD_DIM:
        raise ValueError(
            f"expected hidden dim {_HEAD_DIM}, got {tuple(x.shape)}"
        )
    cache = _normalize_rope_cache(rope_cache).to(device=x.device)
    positions = positions.to(device=x.device, dtype=torch.long)
    out = x.clone()
    half = _ROPE_DIM // 2
    rope = out[..., -_ROPE_DIM:].float().view(*out.shape[:-1], half, 2)
    pos_cache = cache.index_select(0, positions)
    view_shape = (positions.shape[0],) + (1,) * (rope.ndim - 3) + (half,)
    cos = pos_cache[:, :half].view(view_shape)
    sin = pos_cache[:, half:].view(view_shape)

    even = rope[..., 0]
    odd = rope[..., 1]
    if inverse:
        rot_even = even * cos + odd * sin
        rot_odd = odd * cos - even * sin
    else:
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
    out[..., -_ROPE_DIM:] = (
        torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(x.dtype)
    )
    return out


def _resolve_sequence_ids(
    sequence_ids: Optional[Sequence[int] | torch.Tensor],
    *,
    batch_size: int,
) -> tuple[int, ...]:
    if sequence_ids is None:
        raise ValueError(
            "sequence_ids are required to build/consume DeepSeek-V4 decode metadata"
        )
    if isinstance(sequence_ids, torch.Tensor):
        seqs = tuple(int(item) for item in sequence_ids.view(-1).tolist())
    else:
        seqs = tuple(int(item) for item in sequence_ids)
    if len(seqs) != batch_size:
        raise ValueError(f"expected {batch_size} sequence_ids, got {len(seqs)}")
    return seqs


def _resolve_swa_token_slots_slow(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    positions: torch.Tensor,
) -> torch.Tensor:
    slots = [
        coordinator.swa.sequence_token_slots(seq_id, [int(position.item())])[0]
        for seq_id, position in zip(sequence_ids, positions)
    ]
    return torch.stack(slots).to(dtype=torch.int32, device=positions.device)


def _resolve_swa_token_slots_fast(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    positions: torch.Tensor,
) -> Optional[torch.Tensor]:
    pool = coordinator.swa
    page_table = getattr(pool, "_page_table", None)
    if page_table is None or not _pool_active_order_matches(pool, sequence_ids):
        return None
    device = positions.device
    pos = positions.to(device=device, dtype=torch.long)
    page_size = int(pool.page_size_tokens)
    page_table = page_table.to(device=device)
    page_offsets = torch.div(pos, page_size, rounding_mode="floor")
    if int(page_offsets.max().item()) >= page_table.shape[1]:
        return None
    rows = torch.arange(pos.shape[0], device=device, dtype=torch.long)
    pages = page_table[rows, page_offsets].to(torch.long)
    if bool((pages < 0).any().item()):
        return None
    slots = pages * page_size + torch.remainder(pos, page_size)
    return slots.to(dtype=torch.int32, device=device)


def _resolve_swa_token_slots(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    positions: torch.Tensor,
) -> torch.Tensor:
    if _V4_MLA_SM120_TRITON and _V4_FAST_PREFIX_INDICES:
        fast = _resolve_swa_token_slots_fast(
            coordinator, sequence_ids, positions
        )
        if fast is not None:
            return fast
    return _resolve_swa_token_slots_slow(coordinator, sequence_ids, positions)


def _aligned_topk(length: int) -> int:
    return ((length + _TOPK_ALIGN - 1) // _TOPK_ALIGN) * _TOPK_ALIGN


def _build_slot_indices_from_positions(
    pool: Any,
    sequence_ids: Sequence[int],
    logical_positions: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(
        [int(pos.numel()) for pos in logical_positions],
        dtype=torch.int32,
        device=device,
    )
    padded_topk = (
        _aligned_topk(int(lengths.max().item())) if lengths.numel() else 0
    )
    indices = torch.full(
        (len(sequence_ids), 1, padded_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for batch_idx, (seq_id, positions) in enumerate(
        zip(sequence_ids, logical_positions, strict=False)
    ):
        if positions.numel() == 0:
            continue
        seq_slots = pool.sequence_token_slots(seq_id, positions).to(
            dtype=torch.int32, device=device
        )
        indices[batch_idx, 0, : seq_slots.numel()] = seq_slots
    return indices, lengths


def _pool_active_order_matches(pool: Any, sequence_ids: Sequence[int]) -> bool:
    active = getattr(pool, "_active_sequence_ids", None)
    if active is None:
        return False
    return tuple(int(seq_id) for seq_id in sequence_ids) == tuple(active)


def _physicalize_positions_with_page_table(
    pool: Any,
    sequence_ids: Sequence[int],
    logical_positions: torch.Tensor,
    *,
    device: torch.device,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    # Disabled: the page-table gather fast path produces slot values that crash
    # FlashMLA sparse decode (cudaErrorMisalignedAddress, splitkv_mla.cuh:779) on
    # the first decode step it activates. Opt in only for benchmarking.
    if os.environ.get("BATCHGEN_V4_FAST_PHYS", "0") != "1":
        return None
    page_table = getattr(pool, "_page_table", None)
    if page_table is None or not _pool_active_order_matches(pool, sequence_ids):
        return None
    if logical_positions.ndim == 1:
        logical_positions = logical_positions.unsqueeze(1)
    if logical_positions.ndim != 2:
        raise ValueError(
            f"logical_positions must have shape [B,T], got {tuple(logical_positions.shape)}"
        )
    logical_positions = logical_positions.to(device=device, dtype=torch.long)
    valid = logical_positions >= 0
    lengths = valid.sum(dim=1).to(dtype=torch.int32)
    padded_topk = logical_positions.shape[1]
    if padded_topk == 0:
        out = torch.empty(
            logical_positions.shape[0], 1, 0, dtype=torch.int32, device=device
        )
        return out, lengths

    page_size = int(pool.page_size_tokens)
    page_table = page_table.to(device=device)
    page_offsets = torch.div(
        torch.clamp_min(logical_positions, 0), page_size, rounding_mode="floor"
    )
    token_offsets = torch.remainder(
        torch.clamp_min(logical_positions, 0), page_size
    )
    in_page_table = page_offsets < page_table.shape[1]
    safe_page_offsets = page_offsets.clamp(max=max(page_table.shape[1] - 1, 0))
    pages = torch.gather(page_table, 1, safe_page_offsets)
    pages_i64 = pages.to(torch.long)
    slot_valid = valid & in_page_table & (pages_i64 >= 0)
    slots = pages_i64 * page_size + token_offsets
    slots = torch.where(slot_valid, slots, torch.full_like(slots, -1))
    return slots.unsqueeze(1).to(dtype=torch.int32), lengths


def _build_full_prefix_indices_slow(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    cache_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logical_positions = [
        torch.arange(
            int(seq_len.item()), device=cache_seqlens.device, dtype=torch.long
        )
        for seq_len in cache_seqlens
    ]
    return _build_slot_indices_from_positions(
        coordinator.swa,
        sequence_ids,
        logical_positions,
        device=cache_seqlens.device,
    )


def _physicalize_tail_padded_contiguous(
    pool: Any,
    sequence_ids: Sequence[int],
    starts: torch.Tensor,
    lengths: torch.Tensor,
    padded_topk: int,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Vectorized slot gather for tail-padded contiguous logical positions.

    Each row covers logical positions [start, start+length) with -1 only in the
    tail (no interior holes), so the compacted slow-path output equals this
    gather directly. Returns None when the page table is unavailable/stale.
    """
    page_table = getattr(pool, "_page_table", None)
    if page_table is None or not _pool_active_order_matches(pool, sequence_ids):
        return None
    batch = starts.shape[0]
    if padded_topk == 0:
        return torch.empty(batch, 1, 0, dtype=torch.int32, device=device)

    page_size = int(pool.page_size_tokens)
    page_table = page_table.to(device=device)
    offsets = torch.arange(padded_topk, device=device, dtype=torch.long)
    valid = offsets[None, :] < lengths[:, None]
    logical = starts[:, None] + offsets[None, :]
    page_offsets = torch.div(logical, page_size, rounding_mode="floor")
    token_offsets = torch.remainder(logical, page_size)
    in_page_table = page_offsets < page_table.shape[1]
    safe_page_offsets = page_offsets.clamp(max=max(page_table.shape[1] - 1, 0))
    pages = torch.gather(page_table, 1, safe_page_offsets).to(torch.long)
    slot_valid = valid & in_page_table & (pages >= 0)
    slots = pages * page_size + token_offsets
    slots = torch.where(slot_valid, slots, torch.full_like(slots, -1))
    return slots.unsqueeze(1).to(dtype=torch.int32)


def _build_full_prefix_indices_fast(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    cache_seqlens: torch.Tensor,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    device = cache_seqlens.device
    lengths = cache_seqlens.to(device=device, dtype=torch.int32)
    seqlens_long = cache_seqlens.to(device=device, dtype=torch.long)
    padded_topk = (
        _aligned_topk(int(seqlens_long.max().item()))
        if seqlens_long.numel()
        else 0
    )
    starts = torch.zeros_like(seqlens_long)
    indices = _physicalize_tail_padded_contiguous(
        coordinator.swa,
        sequence_ids,
        starts,
        seqlens_long,
        padded_topk,
        device=device,
    )
    return indices, lengths


def _build_full_prefix_indices(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    cache_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _V4_MLA_SM120_TRITON and _V4_FAST_PREFIX_INDICES:
        fast_indices, fast_lengths = _build_full_prefix_indices_fast(
            coordinator, sequence_ids, cache_seqlens
        )
        if fast_indices is not None:
            return fast_indices, fast_lengths
    return _build_full_prefix_indices_slow(
        coordinator, sequence_ids, cache_seqlens
    )


def _swa_window_geometry(
    cache_seqlens: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    lengths = torch.minimum(
        cache_seqlens.to(dtype=torch.long),
        torch.full_like(cache_seqlens.to(dtype=torch.long), int(window)),
    )
    padded_topk = (
        _aligned_topk(int(lengths.max().item())) if lengths.numel() else 0
    )
    starts = (cache_seqlens.to(dtype=torch.long) - lengths).clamp_min(0)
    return starts, lengths, padded_topk


def _build_swa_window_indices_fast(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    cache_seqlens: torch.Tensor,
    *,
    window: int = _SWA_WINDOW,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    device = cache_seqlens.device
    starts, lengths, padded_topk = _swa_window_geometry(cache_seqlens, window)
    indices = _physicalize_tail_padded_contiguous(
        coordinator.swa,
        sequence_ids,
        starts,
        lengths,
        padded_topk,
        device=device,
    )
    return indices, lengths.to(dtype=torch.int32)


def _build_swa_window_indices_slow(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    cache_seqlens: torch.Tensor,
    *,
    window: int = _SWA_WINDOW,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = cache_seqlens.device
    starts, lengths, padded_topk = _swa_window_geometry(cache_seqlens, window)
    offsets = torch.arange(padded_topk, device=device, dtype=torch.long)
    logical_positions = starts[:, None] + offsets[None, :]
    logical_positions = torch.where(
        offsets[None, :] < lengths[:, None],
        logical_positions,
        torch.full_like(logical_positions, -1),
    )
    fast = _physicalize_positions_with_page_table(
        coordinator.swa,
        sequence_ids,
        logical_positions,
        device=device,
    )
    if fast is not None:
        return fast
    fallback_positions = [
        row[row >= 0].to(dtype=torch.long, device=device)
        for row in logical_positions
    ]
    return _build_slot_indices_from_positions(
        coordinator.swa,
        sequence_ids,
        fallback_positions,
        device=device,
    )


def _build_swa_window_indices(
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int],
    cache_seqlens: torch.Tensor,
    *,
    window: int = _SWA_WINDOW,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _V4_MLA_SM120_TRITON and _V4_FAST_PREFIX_INDICES:
        fast_indices, fast_lengths = _build_swa_window_indices_fast(
            coordinator, sequence_ids, cache_seqlens, window=window
        )
        if fast_indices is not None:
            return fast_indices, fast_lengths
    return _build_swa_window_indices_slow(
        coordinator, sequence_ids, cache_seqlens, window=window
    )


def _build_extra_indices_from_logical_positions(
    pool: Any,
    sequence_ids: Sequence[int],
    logical_positions: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logical_positions.ndim == 1:
        logical_positions = logical_positions.unsqueeze(1)
    fast = _physicalize_positions_with_page_table(
        pool, sequence_ids, logical_positions, device=device
    )
    if fast is not None:
        return fast
    lengths = torch.tensor(
        [int((row >= 0).sum().item()) for row in logical_positions],
        dtype=torch.int32,
        device=device,
    )
    padded_topk = (
        _aligned_topk(int(lengths.max().item())) if lengths.numel() else 0
    )
    indices = torch.full(
        (logical_positions.shape[0], 1, padded_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for batch_idx, (seq_id, row) in enumerate(
        zip(sequence_ids, logical_positions, strict=False)
    ):
        valid = row[row >= 0].to(dtype=torch.long, device=device)
        if valid.numel() == 0:
            continue
        slots = pool.sequence_token_slots(seq_id, valid).to(
            dtype=torch.int32, device=device
        )
        indices[batch_idx, 0, : slots.numel()] = slots
    return indices, lengths


def _physicalize_existing_indices(
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.ndim == 2:
        indices = indices.unsqueeze(1)
    if indices.ndim != 3 or indices.shape[1] != 1:
        raise ValueError(
            f"expected indices [B,1,T] or [B,T], got {tuple(indices.shape)}"
        )
    indices = indices.to(device=device, dtype=torch.int32)
    valid = indices[:, 0, :] >= 0
    lengths = valid.sum(dim=1).to(dtype=torch.int32)
    padded_topk = _aligned_topk(indices.shape[2]) if indices.shape[2] else 0
    if padded_topk == indices.shape[2]:
        return indices, lengths
    pad = torch.full(
        (indices.shape[0], 1, padded_topk - indices.shape[2]),
        -1,
        dtype=torch.int32,
        device=device,
    )
    return torch.cat([indices, pad], dim=2), lengths


def _v4_emit_attn_probe(
    *,
    layer_idx: int,
    sequence_ids: Sequence[int],
    cache_seqlens: torch.Tensor,
    q_roped: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    extra_indices: Optional[torch.Tensor],
    extra_lengths: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    coordinator: Any,
) -> None:
    call_no = _v4_attn_probe_calls.get(layer_idx, 0)
    if call_no >= _V4_ATTN_PROBE_STEPS:
        return
    _v4_attn_probe_calls[layer_idx] = call_no + 1

    def _row_fp(idx_row: torch.Tensor, length: int, cache: torch.Tensor) -> str:
        valid = idx_row[idx_row >= 0][:length]
        if valid.numel() == 0:
            return "EMPTY"
        flat = cache.reshape(cache.shape[0] * cache.shape[1], -1)
        sel = flat.index_select(0, valid.to(torch.long).clamp_min(0))
        body = sel.to(torch.float32)
        return (
            f"n={int(valid.numel())} idx[:8]={valid[:8].tolist()} "
            f"absum={float(body.abs().sum().item()):.3e} "
            f"first4={body[0, :4].tolist()}"
        )

    bsz = q_roped.shape[0]
    print(
        f"[V4_ATTN_PROBE] layer={layer_idx} call={call_no} bsz={bsz}",
        flush=True,
    )
    print(
        f"[V4_ATTN_PROBE]   sequence_ids={list(sequence_ids)} "
        f"cache_seqlens={cache_seqlens.tolist()}",
        flush=True,
    )
    if attn_sink is not None:
        s = attn_sink.to(torch.float32)
        print(
            f"[V4_ATTN_PROBE]   attn_sink shape={tuple(attn_sink.shape)} "
            f"min={float(s.min()):.3e} max={float(s.max()):.3e} "
            f"mean={float(s.mean()):.3e} finite={bool(torch.isfinite(s).all())}",
            flush=True,
        )
    else:
        print("[V4_ATTN_PROBE]   attn_sink=None", flush=True)
    for b in range(bsz):
        q_fp = q_roped[b].reshape(-1)[:8].to(torch.float32).tolist()
        ml = int(main_lengths[b].item())
        main_str = _row_fp(main_indices[b, 0], ml, k_cache)
        line = (
            f"[V4_ATTN_PROBE]   b={b} q[:8]={[round(v, 4) for v in q_fp]} "
            f"main_len={ml} main_KV[{main_str}]"
        )
        if extra_indices is not None and extra_lengths is not None:
            el = int(extra_lengths[b].item())
            extra_str = _row_fp(extra_indices[b, 0], el, extra_k_cache)
            line += f" extra_len={el} extra_KV[{extra_str}]"
        print(line, flush=True)


def _validate_sparse_indices(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    capacity: int,
    name: str,
) -> None:
    if indices.dtype != torch.int32:
        raise AssertionError(f"{name} must be int32")
    if indices.numel() and indices.min().item() < -1:
        raise AssertionError(f"{name} sentinel must be -1")
    valid = indices[indices >= 0]
    if valid.numel() and valid.max().item() >= capacity:
        raise AssertionError(f"{name} exceed physical slot capacity")
    for batch_idx, seq_len in enumerate(lengths.tolist()):
        if seq_len and (indices[batch_idx, 0, :seq_len] < 0).any():
            raise AssertionError(
                f"{name} entries inside valid length must be non-negative"
            )
        if (indices[batch_idx, 0, seq_len:] != -1).any():
            raise AssertionError(
                f"{name} entries after valid length must be -1"
            )


def _resolve_attention_q(
    q: torch.Tensor,
    *,
    q_attn: Optional[torch.Tensor],
) -> torch.Tensor:
    return q_attn if q_attn is not None else q


def build_v4_decode_attn_metadata(
    *,
    coordinator: DeepSeekV4KVCoordinator,
    sequence_ids: Sequence[int] | torch.Tensor,
    cache_seqlens: torch.Tensor | Sequence[int],
    positions: Optional[torch.Tensor | Sequence[int]] = None,
    page_tables: Optional[Mapping[str, object]] = None,
    rope_cache: Optional[torch.Tensor] = None,
    extras: Optional[dict[str, Any]] = None,
) -> DSV4AttnMetadata:
    """Build per-step V4 decode metadata from the Phase-1 coordinator.

    Phase 2 only needs the dense/SWA path. c4/c128 fields are populated from
    ``init_compressed_attention_metadata`` when cheap, but the dense adapter only
    relies on the SWA/base fields plus ``extras``.
    """

    device = coordinator.swa.device
    cache_seqlens_t = _to_int32_tensor(cache_seqlens, device=device)
    batch_size = int(cache_seqlens_t.shape[0])
    sequence_ids_t = _resolve_sequence_ids(sequence_ids, batch_size=batch_size)

    if positions is None:
        positions_t = (cache_seqlens_t - 1).clamp_min(0)
    else:
        positions_t = _to_int32_tensor(positions, device=device)
    if positions_t.shape != cache_seqlens_t.shape:
        raise ValueError(
            "positions and cache_seqlens must have the same shape; got "
            f"{tuple(positions_t.shape)} vs {tuple(cache_seqlens_t.shape)}"
        )

    page_tables = page_tables or coordinator.rebuild_page_table(sequence_ids_t)
    swa_page_table_obj = page_tables.get("swa")
    c128_page_table_obj = page_tables.get("c128")
    if not isinstance(swa_page_table_obj, torch.Tensor):
        raise TypeError("page_tables['swa'] must be a torch.Tensor")
    swa_page_table = swa_page_table_obj.to(dtype=torch.int32, device=device)
    if not isinstance(c128_page_table_obj, torch.Tensor):
        raise TypeError("page_tables['c128'] must be a torch.Tensor")
    c128_page_table = c128_page_table_obj.to(dtype=torch.int32, device=device)
    raw_out_loc = positions_t.clone()
    swa_token_slots = _resolve_swa_token_slots(
        coordinator, sequence_ids_t, positions_t
    )

    (
        c4_out_loc,
        _c4_positions,
        c4_topk_lengths_raw,
        c4_topk_lengths_clamp1,
        c128_out_loc,
        _c128_positions,
        c128_topk_lengths_clamp1,
        c128_page_indices,
    ) = init_compressed_attention_metadata(
        seq_lens=cache_seqlens_t,
        positions=positions_t,
        raw_out_loc=raw_out_loc,
        page_table=c128_page_table,
        page_size=coordinator.base_page_size,
        compute_page_indices=True,
    )

    metadata_extras = dict(extras or {})
    metadata_extras.setdefault("sequence_ids", sequence_ids_t)
    metadata_extras.setdefault("page_tables", page_tables)
    metadata_extras.setdefault("coordinator", coordinator)
    metadata_extras.setdefault("swa_token_slots", swa_token_slots)
    if rope_cache is not None:
        metadata_extras.setdefault("rope_cache", rope_cache)

    return DSV4AttnMetadata(
        page_size=coordinator.swa.page_size_tokens,
        page_table=swa_page_table,
        raw_out_loc=raw_out_loc,
        seq_lens_casual=cache_seqlens_t,
        positions_casual=positions_t,
        swa_page_indices=swa_token_slots.unsqueeze(1),
        swa_topk_lengths=cache_seqlens_t.clamp_min(1),
        c4_out_loc=c4_out_loc,
        c4_topk_lengths_raw=c4_topk_lengths_raw,
        c4_topk_lengths_clamp1=c4_topk_lengths_clamp1,
        c128_out_loc=c128_out_loc,
        c128_page_indices=c128_page_indices,
        c128_topk_lengths_clamp1=c128_topk_lengths_clamp1,
        extras=metadata_extras,
    )


class DeepSeekV4FlashMLADecodeAdapter:
    """Dense decode adapter backed by FlashMLA V4 fp8 paged-KV API."""

    def __init__(
        self,
        coordinator: DeepSeekV4KVCoordinator,
        *,
        flashmla_impl: Any = None,
        get_mla_metadata_impl: Any = None,
    ) -> None:
        self.coordinator = coordinator
        self._flashmla_impl = flashmla_impl
        self._get_mla_metadata_impl = get_mla_metadata_impl
        self._c128_decode_state: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @property
    def flashmla_impl(self):
        if self._flashmla_impl is None:
            from flash_mla import flash_mla_with_kvcache

            self._flashmla_impl = flash_mla_with_kvcache
        return self._flashmla_impl

    @property
    def get_mla_metadata_impl(self):
        if self._get_mla_metadata_impl is None:
            from flash_mla import get_mla_metadata

            self._get_mla_metadata_impl = get_mla_metadata
        return self._get_mla_metadata_impl

    def _get_c128_state(
        self,
        *,
        layer_idx: int,
        sequence_id: int,
        compressor: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_idx, sequence_id)
        state = self._c128_decode_state.get(key)
        if state is None:
            kv_state = torch.zeros(
                compressor.compress_ratio,
                compressor.coeff * compressor.head_dim,
                dtype=torch.float32,
                device=device,
            )
            score_state = torch.zeros_like(kv_state)
            state = (kv_state, score_state)
            self._c128_decode_state[key] = state
        return state

    def seed_c128_decode_state(
        self,
        *,
        c128_layer_idx: int,
        sequence_id: int,
        compressor: Any,
        remainder_hidden: torch.Tensor,
        remainder_positions: torch.Tensor,
    ) -> None:
        if remainder_hidden.shape[0] == 0:
            return
        kv_state, score_state = self._get_c128_state(
            layer_idx=c128_layer_idx,
            sequence_id=sequence_id,
            compressor=compressor,
            device=remainder_hidden.device,
        )
        kv_state, score_state = compressor.seed_decode_state(
            remainder_hidden, kv_state, score_state, remainder_positions
        )
        self._c128_decode_state[(c128_layer_idx, sequence_id)] = (
            kv_state,
            score_state,
        )

    def _maybe_store_c128_emission(
        self,
        *,
        route: Any,
        sequence_ids: Sequence[int],
        positions: torch.Tensor,
        metadata: DSV4AttnMetadata,
        rope_cache: torch.Tensor,
        compress_hidden_states: Optional[torch.Tensor],
        compressor: Any,
    ) -> None:
        if (
            route.c128_layer_idx is None
            or compress_hidden_states is None
            or compressor is None
        ):
            return
        if compress_hidden_states.ndim != 2 or compress_hidden_states.shape[
            0
        ] != len(sequence_ids):
            raise ValueError(
                "compress_hidden_states must have shape [B, hidden_size] for c128 decode"
            )
        states = [
            self._get_c128_state(
                layer_idx=route.c128_layer_idx,
                sequence_id=seq_id,
                compressor=compressor,
                device=compress_hidden_states.device,
            )
            for seq_id in sequence_ids
        ]
        kv_state = torch.stack([state[0] for state in states], dim=0)
        score_state = torch.stack([state[1] for state in states], dim=0)
        emitted, kv_state, score_state, emit_mask = (
            compressor.forward_decode_batch(
                compress_hidden_states,
                kv_state,
                score_state,
                positions,
                rope_cache,
            )
        )
        for batch_idx, seq_id in enumerate(sequence_ids):
            self._c128_decode_state[(route.c128_layer_idx, seq_id)] = (
                kv_state[batch_idx],
                score_state[batch_idx],
            )
        if emitted.numel() == 0:
            return
        out_locs = metadata.c128_out_loc.to(
            device=compress_hidden_states.device, dtype=torch.long
        ).unsqueeze(1)
        slot_indices, _slot_lengths = _physicalize_positions_with_page_table(
            self.coordinator.c128,
            sequence_ids,
            out_locs,
            device=compress_hidden_states.device,
        ) or _build_extra_indices_from_logical_positions(
            self.coordinator.c128,
            sequence_ids,
            out_locs,
            device=compress_hidden_states.device,
        )
        token_slots = slot_indices[:, 0, 0][emit_mask].to(dtype=torch.int64)
        self.coordinator.c128.store_kv(
            layer_idx=route.c128_layer_idx,
            token_slots=token_slots,
            kv_processed=emitted.to(torch.bfloat16),
        )

    def __call__(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
        metadata: DSV4AttnMetadata,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError(f"expected q=[B,H,D], got {tuple(q.shape)}")

        rope_cache = kwargs.pop("rope_cache", None)
        if rope_cache is None:
            rope_cache = metadata.extras.get("rope_cache")
        if rope_cache is None:
            raise ValueError(
                "dense V4 decode requires rope_cache via metadata.extras['rope_cache'] or kwargs['rope_cache']"
            )

        sequence_ids = _resolve_sequence_ids(
            kwargs.pop("sequence_ids", metadata.extras.get("sequence_ids")),
            batch_size=q.shape[0],
        )
        positions = metadata.positions_casual.to(
            device=q.device, dtype=torch.long
        )
        cache_seqlens = metadata.seq_lens_casual.to(
            device=q.device, dtype=torch.int32
        )
        _dt = get_decode_timer()

        q_attn = kwargs.pop("q_attn", None)
        attn_q = _resolve_attention_q(q, q_attn=q_attn)
        if attn_q.ndim != 3 or attn_q.shape[-1] != _HEAD_DIM:
            raise ValueError(
                f"attention q must have shape [B,H,{_HEAD_DIM}], got {tuple(attn_q.shape)}"
            )

        with _dt.timed("attn_q_rope", layer_idx) if _dt else nullcontext():
            q_roped = _apply_rope(attn_q, positions, rope_cache)

        route = self.coordinator.get_layer_routing(layer_idx)
        current_kv = kwargs.pop("current_kv", None)
        if current_kv is None and kv.ndim == 2 and kv.shape[-1] == _HEAD_DIM:
            current_kv = kv
        if current_kv is not None:
            with (
                _dt.timed("attn_kv_store", layer_idx) if _dt else nullcontext()
            ):
                current_kv = current_kv.to(device=q.device)
                if current_kv.shape != (q.shape[0], _HEAD_DIM):
                    raise ValueError(
                        f"current_kv must have shape {(q.shape[0], _HEAD_DIM)}, got {tuple(current_kv.shape)}"
                    )
                kv_roped = _apply_rope(current_kv, positions, rope_cache)
                token_slots = metadata.extras.get("swa_token_slots")
                if token_slots is None:
                    token_slots = _resolve_swa_token_slots(
                        self.coordinator, sequence_ids, positions
                    )
                token_slots = token_slots.to(device=q.device, dtype=torch.int32)
                self.coordinator.swa.store_kv(
                    layer_idx=route.swa_layer_idx,
                    token_slots=token_slots,
                    kv_processed=kv_roped.contiguous(),
                )

        with _dt.timed("attn_kv_fetch", layer_idx) if _dt else nullcontext():
            k_cache, _, _block_table = (
                self.coordinator.swa.get_layer_kv_with_page_table(
                    route.swa_layer_idx
                )
            )
        del _block_table
        softmax_scale = kwargs.pop("softmax_scale", _SOFTMAX_SCALE)

        sparse_indices = kwargs.pop("sparse_indices", None)
        compressed_page_indices = kwargs.pop("compressed_page_indices", None)
        compressed_lengths = kwargs.pop("compressed_lengths", None)
        compress_hidden_states = kwargs.pop("compress_hidden_states", None)
        compressor = kwargs.pop("compressor", None)

        extra_k_cache = None
        extra_indices = None
        extra_lengths = None
        if sparse_indices is not None:
            if route.c4_layer_idx is None:
                raise RuntimeError("c4 sparse path requires c4 routing")
            with (
                _dt.timed("attn_build_main_indices", layer_idx)
                if _dt
                else nullcontext()
            ):
                main_indices, main_lengths = _build_swa_window_indices(
                    self.coordinator, sequence_ids, cache_seqlens
                )
            with (
                _dt.timed("attn_build_extra_indices", layer_idx)
                if _dt
                else nullcontext()
            ):
                extra_indices, extra_lengths = (
                    _build_extra_indices_from_logical_positions(
                        self.coordinator.c4,
                        sequence_ids,
                        sparse_indices.to(device=q.device),
                        device=q.device,
                    )
                )
            with (
                _dt.timed("attn_extra_kv_fetch", layer_idx)
                if _dt
                else nullcontext()
            ):
                extra_k_cache, _, _ = (
                    self.coordinator.c4.get_layer_kv_with_page_table(
                        route.c4_layer_idx
                    )
                )
        elif compressed_page_indices is not None:
            if route.c128_layer_idx is None:
                raise RuntimeError("c128 compressed path requires c128 routing")
            with (
                _dt.timed("attn_c128_store", layer_idx)
                if _dt
                else nullcontext()
            ):
                self._maybe_store_c128_emission(
                    route=route,
                    sequence_ids=sequence_ids,
                    positions=positions,
                    metadata=metadata,
                    rope_cache=rope_cache,
                    compress_hidden_states=compress_hidden_states,
                    compressor=compressor,
                )
            with (
                _dt.timed("attn_build_main_indices", layer_idx)
                if _dt
                else nullcontext()
            ):
                main_indices, main_lengths = _build_swa_window_indices(
                    self.coordinator, sequence_ids, cache_seqlens
                )
            if compressed_lengths is None:
                raise ValueError(
                    "compressed_lengths are required with compressed_page_indices"
                )
            with (
                _dt.timed("attn_build_extra_indices", layer_idx)
                if _dt
                else nullcontext()
            ):
                extra_indices, extra_lengths = _physicalize_existing_indices(
                    compressed_page_indices.to(device=q.device),
                    device=q.device,
                )
                expected_lengths = compressed_lengths.to(
                    device=q.device, dtype=torch.int32
                )
                extra_lengths = torch.minimum(extra_lengths, expected_lengths)
            with (
                _dt.timed("attn_extra_kv_fetch", layer_idx)
                if _dt
                else nullcontext()
            ):
                extra_k_cache, _, _ = (
                    self.coordinator.c128.get_layer_kv_with_page_table(
                        route.c128_layer_idx
                    )
                )
            if extra_lengths.numel() and int(extra_lengths.max().item()) == 0:
                extra_k_cache = None
                extra_indices = None
                extra_lengths = None
        else:
            with (
                _dt.timed("attn_build_main_indices", layer_idx)
                if _dt
                else nullcontext()
            ):
                main_indices, main_lengths = _build_full_prefix_indices(
                    self.coordinator, sequence_ids, cache_seqlens
                )

        valid_indices = main_indices[main_indices >= 0]

        if q_roped.unsqueeze(1).shape[:3] != (q.shape[0], 1, q.shape[1]):
            raise AssertionError("q must be shaped [B, 1, H, D] for FlashMLA")
        if k_cache.shape != (
            self.coordinator.swa.num_pages,
            self.coordinator.swa.page_size_tokens,
            1,
            self.coordinator.swa.bytes_per_token,
        ):
            raise AssertionError(
                f"unexpected k_cache shape: {tuple(k_cache.shape)}"
            )
        if k_cache.stride(0) % 576 != 0:
            raise AssertionError(
                f"page stride must be 576-byte aligned, got {k_cache.stride(0)}"
            )
        if _V4_MLA_VALIDATE_INDICES:
            with (
                _dt.timed("attn_validate", layer_idx) if _dt else nullcontext()
            ):
                _validate_sparse_indices(
                    main_indices,
                    main_lengths,
                    capacity=self.coordinator.swa.num_pages
                    * self.coordinator.swa.page_size_tokens,
                    name="indices_in_kvcache",
                )
        if (
            attn_sink is not None
            and torch.isfinite(attn_sink).logical_not().any()
        ):
            raise AssertionError("attn_sink must be finite")
        if extra_k_cache is not None:
            if extra_indices is None or extra_lengths is None:
                raise AssertionError(
                    "extra_k_cache requires extra indices/lengths"
                )
            if _V4_MLA_VALIDATE_INDICES:
                with (
                    _dt.timed("attn_validate_extra", layer_idx)
                    if _dt
                    else nullcontext()
                ):
                    _validate_sparse_indices(
                        extra_indices,
                        extra_lengths,
                        capacity=extra_k_cache.shape[0]
                        * extra_k_cache.shape[1],
                        name="extra_indices_in_kvcache",
                    )

        if _V4_ATTN_PROBE:
            _v4_emit_attn_probe(
                layer_idx=layer_idx,
                sequence_ids=sequence_ids,
                cache_seqlens=cache_seqlens,
                q_roped=q_roped,
                main_indices=main_indices,
                main_lengths=main_lengths,
                extra_indices=extra_indices,
                extra_lengths=extra_lengths,
                k_cache=k_cache,
                extra_k_cache=extra_k_cache,
                attn_sink=attn_sink,
                coordinator=self.coordinator,
            )

        backend_name = _select_v4_mla_backend()
        if backend_name == "sm120_triton":
            from batchgen.attention.dsa.v4_mla_sm120_triton import (
                flash_mla_sparse_decode_sm120,
                maybe_warmup_sm120_sparse_decode,
            )

            maybe_warmup_sm120_sparse_decode(
                num_heads=q.shape[1], head_dim=q.shape[-1], device=q.device
            )

            with (
                _dt.timed("attn_sm120_kernel_total", layer_idx)
                if _dt
                else nullcontext()
            ):
                attn_out = flash_mla_sparse_decode_sm120(
                    q=q_roped.unsqueeze(1).contiguous(),
                    k_cache=k_cache,
                    indices=main_indices,
                    topk_length=main_lengths,
                    attn_sink=attn_sink,
                    head_dim_v=q_roped.shape[-1],
                    softmax_scale=softmax_scale,
                    extra_k_cache=extra_k_cache,
                    extra_indices=extra_indices,
                    extra_topk_length=extra_lengths,
                    layer_idx=layer_idx,
                )
        elif backend_name == "torch_ref":
            attn_out = flashmla_decode_torch_reference(
                q=q_roped.unsqueeze(1).contiguous(),
                k_cache=k_cache,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=q_roped.shape[-1],
                tile_scheduler_metadata=None,
                num_splits=None,
                softmax_scale=softmax_scale,
                causal=False,
                is_fp8_kvcache=True,
                indices=main_indices,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices,
                topk_length=main_lengths,
                extra_topk_length=extra_lengths,
            )
        else:
            tile_scheduler_metadata, num_splits = self.get_mla_metadata_impl()
            attn_out, _ = self.flashmla_impl(
                q=q_roped.unsqueeze(1).contiguous(),
                k_cache=k_cache,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=q_roped.shape[-1],
                tile_scheduler_metadata=tile_scheduler_metadata,
                num_splits=num_splits,
                softmax_scale=softmax_scale,
                causal=False,
                is_fp8_kvcache=True,
                indices=main_indices,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices,
                topk_length=main_lengths,
                extra_topk_length=extra_lengths,
            )
        if _V4_ATTN_TENSOR_DUMP:
            _v4_dump_attn_tensors(
                layer_idx=layer_idx,
                q_roped=q_roped,
                main_indices=main_indices,
                main_lengths=main_lengths,
                extra_indices=extra_indices,
                extra_lengths=extra_lengths,
                k_cache=k_cache,
                extra_k_cache=extra_k_cache,
                attn_sink=attn_sink,
                attn_out=attn_out,
                coordinator=self.coordinator,
            )
        with (
            _dt.timed("attn_inverse_rope", layer_idx) if _dt else nullcontext()
        ):
            return _apply_rope(
                attn_out.squeeze(1), positions, rope_cache, inverse=True
            )


__all__ = [
    "DeepSeekV4FlashMLADecodeAdapter",
    "build_v4_decode_attn_metadata",
]
