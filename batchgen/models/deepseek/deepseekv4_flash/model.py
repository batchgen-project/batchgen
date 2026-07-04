# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash model definition.

This file is intentionally self-contained.  It mirrors the V4 tensor names from
``assets/inference/model.py`` while exposing the BatchGen worker contract:
``ForCausalLM.model``, ``ForCausalLM.lm_head``, ``model.embed_tokens``,
``model.layers``, ``model.norm``, ``layer.self_attn`` and ``layer.mlp``.

The structure is DP-attention + EP-MoE oriented: attention modules hold full
head projections, and MoE layers expose global expert slots that the parallel
strategy manager assigns to per-rank expert-parallel ranges.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.wrappers.attention import AttnWrapperBase
from batchgen.timing import get_decode_timer, init_decode_timer
from batchgen_kernels.common.v4_hyper_connections import hc_post, hc_pre
from batchgen_kernels.moe.v4_hash_routing import hash_routing
from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

# Per-op decode timing (BATCHGEN_DECODE_TIMING=1). Initializes the shared
# decode-timer singleton consumed here and in wrappers.py via
# get_decode_timer(). Categories only control summary display ordering;
# any op_name is accepted at record time. Disabled timer => pure no-op.
_V4_DECODE_TIMER_CATEGORIES = [
    "self_attn",
    "moe",
    "attn_q_proj",
    "attn_kv_proj",
    "attn_indexer",
    "attn_backend",
    "attn_o_proj",
    "moe_allgather",
    "moe_gate",
    "moe_expert_loop",
    "moe_allreduce",
    "moe_shared",
]
init_decode_timer("DeepSeek-V4-Flash", _V4_DECODE_TIMER_CATEGORIES)

_DDL_TRACE = os.environ.get("BATCHGEN_DECODE_DEADLOCK_TRACE", "0") == "1"


def _ddl_trace(rank, tag: str) -> None:
    if not _DDL_TRACE:
        return
    os.write(2, f"[DDL] pid={os.getpid()} rank={rank} {tag}\n".encode())


_FP4_E2M1_TABLE_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_V4_LAYER_BARRIER = os.environ.get("BATCHGEN_V4_LAYER_BARRIER", "1") == "1"


def _v4_layer_barrier_enabled() -> bool:
    # On by default: required for streamed (offloaded) experts to bound the
    # per-layer host drift that deadlocks the EP collective. Fully-resident
    # experts cannot drift, so set BATCHGEN_V4_LAYER_BARRIER=0 to drop the
    # 43-barriers/token cost when BATCHGEN_V4_RESIDENT_EXPERTS=1.
    return _V4_LAYER_BARRIER


# Use PyNcclCommunicator for EP-decode collectives instead of torch.distributed
# (default ON; set 0 to fall back to dist.*). See _ep_all_gather.
_V4_PYNCCL_COMM = os.environ.get("BATCHGEN_V4_PYNCCL_COMM", "1") == "1"

# Env-gated diagnostic (default OFF).
_V4_DIVTRACE = os.environ.get("BATCHGEN_V4_DIVTRACE", "0") == "1"
# Prefill mode: trace the PREFILL forward (q_len > 1) instead of decode tokens,
# dumping only the last prompt position for comparison with the official
# reference's prefill activations.
_V4_DIVTRACE_PREFILL = (
    os.environ.get("BATCHGEN_V4_DIVTRACE_PREFILL", "0") == "1"
)
_V4_DIVTRACE_DUMP_PATH = os.environ.get(
    "BATCHGEN_V4_DIVTRACE_DUMP_PATH", "/data3/leyangxue/v4-repro-artifacts"
)
_V4_DIVTRACE_FFN_ATTRIB_LAYERS = {4, 5, 6}
_V4_DIVTRACE_MOE_INTERNALS_LAYERS = {4, 5, 6}
_v4_divtrace_calls: dict[int, int] = {}
_v4_divtrace_active_layers: set[int] = set()
_v4_divtrace_final_calls = 0
_v4_divtrace_batch_note_emitted = False
_v4_divtrace_records: list[dict[str, Any]] = []
_v4_divtrace_dump_written = False


def vocab_parallel_embedding(
    embed: nn.Embedding,
    input_ids: torch.Tensor,
    full_vocab_size: int,
) -> torch.Tensor:
    """Embedding lookup that tolerates a vocab-parallel sharded table.

    The V4 checkpoint shards embed_tokens by vocab across TP ranks: each rank
    holds rows [rank*local, (rank+1)*local) of the full vocab. A naive
    ``embed(global_id)`` then indexes out of bounds. When the loaded table is a
    shard (rows < full_vocab_size), restrict to this rank's id range, look up
    locally, zero out-of-range rows, and all-reduce the partial embeddings.
    When the table is full (rows == full_vocab_size), this is a plain lookup.
    """
    local_vocab = embed.weight.shape[0]
    if local_vocab >= full_vocab_size or not dist.is_initialized():
        return embed(input_ids)
    world_size = dist.get_world_size()
    if local_vocab * world_size < full_vocab_size:
        return embed(input_ids)

    original_shape = input_ids.shape
    flat_ids = input_ids.reshape(-1).contiguous()
    local_rows = torch.tensor(
        [flat_ids.shape[0]], dtype=torch.int64, device=input_ids.device
    )
    row_counts = [torch.empty_like(local_rows) for _ in range(world_size)]
    dist.all_gather(row_counts, local_rows)
    row_counts_int = [int(count.item()) for count in row_counts]
    max_rows = max(row_counts_int)
    if max_rows == 0:
        return embed.weight.new_empty(*original_shape, embed.weight.shape[1])

    if flat_ids.shape[0] < max_rows:
        pad = flat_ids.new_zeros(max_rows - flat_ids.shape[0])
        padded_ids = torch.cat([flat_ids, pad], dim=0)
    else:
        padded_ids = flat_ids

    gathered_ids = [torch.empty_like(padded_ids) for _ in range(world_size)]
    dist.all_gather(gathered_ids, padded_ids)
    global_ids = torch.cat(
        [ids[:count] for ids, count in zip(gathered_ids, row_counts_int)],
        dim=0,
    )

    start = dist.get_rank() * local_vocab
    mask = (global_ids >= start) & (global_ids < start + local_vocab)
    local_ids = torch.where(
        mask, global_ids - start, torch.zeros_like(global_ids)
    )
    out = embed(local_ids)
    out = out * mask.unsqueeze(-1).to(out.dtype)
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    rank = dist.get_rank()
    row_start = sum(row_counts_int[:rank])
    row_end = row_start + row_counts_int[rank]
    return out[row_start:row_end].reshape(
        *original_shape, embed.weight.shape[1]
    )


def vocab_parallel_lm_head(
    lm_head: nn.Linear,
    hidden_states: torch.Tensor,
    full_vocab_size: int,
    force_fp32: bool = False,
) -> torch.Tensor:
    """LM-head projection for vocab-parallel V4 checkpoint shards.

    V4 shards ``head.weight`` across the vocab dimension.  Each rank computes
    local logits for its shard, then all ranks gather those local logits in rank
    order so downstream sampling sees global token ids.
    """
    local_vocab = lm_head.weight.shape[0]
    if local_vocab == full_vocab_size:
        if force_fp32:
            bias = lm_head.bias.float() if lm_head.bias is not None else None
            return F.linear(hidden_states.float(), lm_head.weight.float(), bias)
        return F.linear(hidden_states, lm_head.weight, lm_head.bias)

    if local_vocab > full_vocab_size:
        raise RuntimeError(
            f"lm_head rows {local_vocab} exceed full vocab {full_vocab_size}"
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "vocab-parallel lm_head requires torch.distributed to be initialized"
        )

    world_size = dist.get_world_size()
    if local_vocab * world_size != full_vocab_size:
        raise RuntimeError(
            "invalid vocab-parallel lm_head layout: "
            f"local={local_vocab}, world_size={world_size}, "
            f"full_vocab={full_vocab_size}"
        )

    original_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    local_hidden = hidden_states.reshape(-1, hidden_size).contiguous()
    local_rows = torch.tensor(
        [local_hidden.shape[0]], dtype=torch.int64, device=hidden_states.device
    )
    row_counts = [torch.empty_like(local_rows) for _ in range(world_size)]
    dist.all_gather(row_counts, local_rows)
    row_counts_int = [int(count.item()) for count in row_counts]
    max_rows = max(row_counts_int)
    if max_rows == 0:
        return hidden_states.new_empty(*original_shape, full_vocab_size)

    if local_hidden.shape[0] < max_rows:
        pad = local_hidden.new_zeros(
            max_rows - local_hidden.shape[0], hidden_size
        )
        padded_hidden = torch.cat([local_hidden, pad], dim=0)
    else:
        padded_hidden = local_hidden

    gathered_hidden = [
        torch.empty_like(padded_hidden) for _ in range(world_size)
    ]
    dist.all_gather(gathered_hidden, padded_hidden)
    global_hidden = torch.cat(
        [rows[:count] for rows, count in zip(gathered_hidden, row_counts_int)],
        dim=0,
    )

    if force_fp32:
        bias = lm_head.bias.float() if lm_head.bias is not None else None
        local_logits = F.linear(
            global_hidden.float(), lm_head.weight.float(), bias
        )
    else:
        local_logits = F.linear(global_hidden, lm_head.weight, lm_head.bias)
    gathered_logits = [
        torch.empty_like(local_logits) for _ in range(world_size)
    ]
    dist.all_gather(gathered_logits, local_logits.contiguous())
    global_logits = torch.cat(gathered_logits, dim=-1)

    rank = dist.get_rank()
    start = sum(row_counts_int[:rank])
    end = start + row_counts_int[rank]
    return global_logits[start:end].reshape(*original_shape, full_vocab_size)


def _v4_divtrace_note(message: str) -> None:
    print(f"[V4_DIVTRACE] {message}", flush=True)


def _v4_divtrace_note_batch_skip(batch_size: int) -> None:
    global _v4_divtrace_batch_note_emitted
    if _v4_divtrace_batch_note_emitted:
        return
    _v4_divtrace_batch_note_emitted = True
    _v4_divtrace_note(f"skip trace: batch size {batch_size} < 1")


def _v4_divtrace_rank() -> int:
    if dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _v4_divtrace_sequence_ids() -> Optional[list[int]]:
    cur_batch = getattr(AttnWrapperBase, "cur_batch", None)
    if cur_batch is None:
        return None
    if isinstance(cur_batch, torch.Tensor):
        return [int(v) for v in cur_batch.detach().cpu().tolist()]
    try:
        return [int(v) for v in cur_batch]
    except TypeError:
        return None


def _v4_divtrace_cache_seqlens(
    cache_seqlens: Optional[torch.Tensor],
) -> Optional[list[int]]:
    source = cache_seqlens
    if source is None:
        source = getattr(AttnWrapperBase, "cache_seqlens", None)
    if source is None:
        return None
    if isinstance(source, torch.Tensor):
        return [int(v) for v in source.detach().cpu().tolist()]
    try:
        return [int(v) for v in source]
    except TypeError:
        return None


def _v4_divtrace_metadata(
    cache_seqlens: Optional[torch.Tensor],
    batch_idx: int = 0,
) -> dict[str, Optional[int]]:
    seq_id = None
    sequence_ids = _v4_divtrace_sequence_ids()
    if sequence_ids is not None and batch_idx < len(sequence_ids):
        seq_id = int(sequence_ids[batch_idx])
    cache_seqlen = None
    cache_seqlens_list = _v4_divtrace_cache_seqlens(cache_seqlens)
    if cache_seqlens_list is not None and batch_idx < len(cache_seqlens_list):
        cache_seqlen = int(cache_seqlens_list[batch_idx])
    return {"seq_id": seq_id, "cache_seqlen": cache_seqlen}


def _v4_divtrace_append(record: dict[str, Any]) -> None:
    _v4_divtrace_records.append(record)


def _v4_divtrace_dump_tensor(
    layer_idx: int,
    name: str,
    tensor: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor],
) -> None:
    meta = _v4_divtrace_metadata(cache_seqlens)
    payload = tensor[:1]
    if _V4_DIVTRACE_PREFILL and payload.dim() >= 2 and payload.size(1) > 1:
        # Keep only the last prompt position to match the official
        # reference dump (last-token activations).
        payload = payload[:, -1:]
    _v4_divtrace_append(
        {
            "kind": "boundary",
            "rank": _v4_divtrace_rank(),
            "layer_idx": int(layer_idx),
            "name": name,
            "seq_id": meta["seq_id"],
            "cache_seqlen": meta["cache_seqlen"],
            "tensor": payload.detach().to(torch.float32).cpu().clone(),
        }
    )


def _v4_divtrace_flush() -> None:
    global _v4_divtrace_dump_written
    if _v4_divtrace_dump_written:
        return
    os.makedirs(_V4_DIVTRACE_DUMP_PATH, exist_ok=True)
    rank = _v4_divtrace_rank()
    path = os.path.join(_V4_DIVTRACE_DUMP_PATH, f"divtrace_rank{rank}.pt")
    torch.save(_v4_divtrace_records, path)
    _v4_divtrace_dump_written = True
    _v4_divtrace_note(
        f"wrote {len(_v4_divtrace_records)} trace records to {path}"
    )


def _v4_divtrace_is_decode_token(
    tensor: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor, ...]],
) -> bool:
    del past_key_value
    if _V4_DIVTRACE_PREFILL:
        return tensor.dim() >= 2 and tensor.size(1) > 1
    return tensor.dim() >= 2 and tensor.size(1) == 1


def _v4_divtrace_begin_layer(
    layer_idx: int,
    hidden_states: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor, ...]],
) -> bool:
    if not _v4_divtrace_is_decode_token(hidden_states, past_key_value):
        return False
    if hidden_states.size(0) < 1:
        _v4_divtrace_note_batch_skip(hidden_states.size(0))
        return False
    if _v4_divtrace_calls.get(layer_idx, 0) > 0:
        return False
    _v4_divtrace_active_layers.add(layer_idx)
    return True


def _v4_divtrace_end_layer(layer_idx: int) -> None:
    _v4_divtrace_active_layers.discard(layer_idx)
    _v4_divtrace_calls[layer_idx] = _v4_divtrace_calls.get(layer_idx, 0) + 1
    if _V4_DIVTRACE_PREFILL and layer_idx >= 42:
        # The batchgen prefill path bypasses ForCausalLM.forward (where the
        # normal flush lives), so flush after the last decoder layer.
        _v4_divtrace_flush()


def _v4_divtrace_should_trace_final(
    hidden_states: torch.Tensor,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]],
) -> bool:
    global _v4_divtrace_final_calls
    if not _v4_divtrace_is_decode_token(hidden_states, past_key_values):
        return False
    if hidden_states.size(0) < 1:
        _v4_divtrace_note_batch_skip(hidden_states.size(0))
        return False
    if _v4_divtrace_final_calls > 0:
        return False
    _v4_divtrace_final_calls += 1
    return True


def _v4_divtrace_tensor_summary(
    tensor: torch.Tensor,
) -> tuple[float, float, float, bool]:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    abs_mean = flat.abs().mean().item()
    rms = flat.square().mean().sqrt().item()
    max_abs = flat.abs().max().item()
    finite = bool(torch.isfinite(flat).all().item())
    return abs_mean, rms, max_abs, finite


def _v4_divtrace_l2_rms_max_abs(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    return {
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "max_abs": float(flat.abs().max().item()),
    }


def _v4_divtrace_stats(tensor: torch.Tensor) -> dict[str, float | int]:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return {
            "l2": 0.0,
            "rms": 0.0,
            "max_abs": 0.0,
            "mean": 0.0,
            "nan_count": 0,
            "inf_count": 0,
        }

    finite = torch.isfinite(flat)
    finite_flat = flat[finite]
    if finite_flat.numel() == 0:
        mean = float("nan")
        rms = float("nan")
        max_abs = float("nan")
        l2 = float("nan")
    else:
        mean = float(finite_flat.mean().item())
        rms = float(finite_flat.square().mean().sqrt().item())
        max_abs = float(finite_flat.abs().max().item())
        l2 = float(torch.linalg.vector_norm(finite_flat).item())
    return {
        "l2": l2,
        "rms": rms,
        "max_abs": max_abs,
        "mean": mean,
        "nan_count": int(torch.isnan(flat).sum().item()),
        "inf_count": int(torch.isinf(flat).sum().item()),
    }


def _v4_divtrace_first_row(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.detach().to(torch.float32).reshape(1, 1).cpu().clone()
    last_dim = tensor.shape[-1]
    return (
        tensor.detach()
        .to(torch.float32)
        .reshape(-1, last_dim)[:1]
        .cpu()
        .clone()
    )


def _v4_divtrace_segment_norms(
    tensor: torch.Tensor, segment_size: int
) -> list[dict[str, float | int]]:
    rows = tensor.detach().to(torch.float32).reshape(-1, tensor.shape[-1])
    if segment_size <= 0:
        return []
    out: list[dict[str, float | int]] = []
    for segment_idx, start in enumerate(range(0, rows.size(0), segment_size)):
        segment = rows[start : start + segment_size]
        stats = _v4_divtrace_stats(segment)
        out.append(
            {
                "segment_idx": int(segment_idx),
                "rows": int(segment.size(0)),
                "l2": float(stats["l2"]),
                "rms": float(stats["rms"]),
                "max_abs": float(stats["max_abs"]),
            }
        )
    return out


def _v4_divtrace_cross_summary(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
) -> tuple[float, float]:
    a = tensor_a.detach().to(torch.float32).reshape(-1)
    b = tensor_b.detach().to(torch.float32).reshape(-1)
    diff = a - b
    rel_l2 = (
        torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(a) + 1e-6)
    ).item()
    cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
    return rel_l2, cosine


def _v4_divtrace_emit_boundary(
    layer_idx: int,
    name: str,
    tensor: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> None:
    first = tensor[0]
    abs_mean, rms, max_abs, finite = _v4_divtrace_tensor_summary(first)
    meta = _v4_divtrace_metadata(cache_seqlens)
    _v4_divtrace_dump_tensor(layer_idx, name, tensor, cache_seqlens)
    _v4_divtrace_note(
        f"layer={layer_idx:02d} {name} "
        f"rank={_v4_divtrace_rank()} "
        f"seq_id={meta['seq_id']} cache_seqlen={meta['cache_seqlen']} "
        f"abs_mean={abs_mean:.6e},rms={rms:.6e},max_abs={max_abs:.6e},finite={int(finite)}"
    )


def _v4_divtrace_emit_router(
    layer_idx: int,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> None:
    meta = _v4_divtrace_metadata(cache_seqlens)
    ids = topk_indices[0].detach().to(torch.int64).cpu().tolist()
    weights = [
        float(v)
        for v in topk_weights[0].detach().to(torch.float32).cpu().tolist()
    ]
    _v4_divtrace_append(
        {
            "kind": "router",
            "rank": _v4_divtrace_rank(),
            "layer_idx": int(layer_idx),
            "name": "router",
            "seq_id": meta["seq_id"],
            "cache_seqlen": meta["cache_seqlen"],
            "ids": ids,
            "weights": weights,
        }
    )
    _v4_divtrace_note(
        f"layer={layer_idx:02d} router "
        f"rank={_v4_divtrace_rank()} "
        f"seq_id={meta['seq_id']} cache_seqlen={meta['cache_seqlen']} "
        f"ids={ids},weights={[round(v, 6) for v in weights]}"
    )


def _v4_divtrace_emit_ffn_attrib(
    layer_idx: int,
    residual: torch.Tensor,
    mlp_out: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> None:
    meta = _v4_divtrace_metadata(cache_seqlens)
    residual0 = residual[:1].detach().to(torch.float32)
    mlp_out0 = mlp_out[:1].detach().to(torch.float32)
    post0 = post[:1].detach().to(torch.float32)
    comb0 = comb[:1].detach().to(torch.float32)
    post_term = post0.unsqueeze(-1) * mlp_out0.unsqueeze(-2)
    comb_term = torch.sum(comb0.unsqueeze(-1) * residual0.unsqueeze(-2), dim=2)
    y = post_term + comb_term

    token_comb = comb0[0, 0]
    row_sums = token_comb.sum(dim=-1)
    col_sums = token_comb.sum(dim=-2)
    svdvals = torch.linalg.svdvals(token_comb)

    _v4_divtrace_append(
        {
            "kind": "ffn_attrib",
            "rank": _v4_divtrace_rank(),
            "layer_idx": int(layer_idx),
            "name": "ffn_attrib",
            "seq_id": meta["seq_id"],
            "cache_seqlen": meta["cache_seqlen"],
            "residual": residual0.cpu().clone(),
            "mlp_out": mlp_out0.cpu().clone(),
            "post": post0.cpu().clone(),
            "comb": comb0.cpu().clone(),
            "post_term": post_term.cpu().clone(),
            "comb_term": comb_term.cpu().clone(),
            "stats": {
                "residual": _v4_divtrace_l2_rms_max_abs(residual0),
                "mlp_out": _v4_divtrace_l2_rms_max_abs(mlp_out0),
                "post_term": _v4_divtrace_l2_rms_max_abs(post_term),
                "comb_term": _v4_divtrace_l2_rms_max_abs(comb_term),
                "y": _v4_divtrace_l2_rms_max_abs(y),
            },
            "comb_diag": {
                "row_sums": row_sums.cpu().tolist(),
                "col_sums": col_sums.cpu().tolist(),
                "min": float(token_comb.min().item()),
                "max": float(token_comb.max().item()),
                "num_negatives": int((token_comb < 0).sum().item()),
                "max_row_sum_err": float((row_sums - 1).abs().max().item()),
                "max_col_sum_err": float((col_sums - 1).abs().max().item()),
                "max_singular": float(svdvals.max().item()),
            },
            "post_diag": {
                "min": float(post0.min().item()),
                "max": float(post0.max().item()),
                "mean": float(post0.mean().item()),
            },
        }
    )
    _v4_divtrace_note(
        f"layer={layer_idx:02d} ffn_attrib "
        f"rank={_v4_divtrace_rank()} "
        f"seq_id={meta['seq_id']} cache_seqlen={meta['cache_seqlen']} "
        f"||R||={_v4_divtrace_l2_rms_max_abs(residual0)['l2']:.6e},"
        f"||U||={_v4_divtrace_l2_rms_max_abs(mlp_out0)['l2']:.6e},"
        f"||post_term||={_v4_divtrace_l2_rms_max_abs(post_term)['l2']:.6e},"
        f"||comb_term||={_v4_divtrace_l2_rms_max_abs(comb_term)['l2']:.6e},"
        f"||Y||={_v4_divtrace_l2_rms_max_abs(y)['l2']:.6e}"
    )


def _v4_divtrace_emit_moe_internals(
    layer_idx: int,
    tensors: Dict[str, torch.Tensor],
    cache_seqlens: Optional[torch.Tensor] = None,
    extras: Optional[dict[str, Any]] = None,
) -> None:
    meta = _v4_divtrace_metadata(cache_seqlens)
    captured = {
        name: _v4_divtrace_first_row(tensor) for name, tensor in tensors.items()
    }
    stats = {
        name: _v4_divtrace_stats(tensor) for name, tensor in captured.items()
    }
    record: dict[str, Any] = {
        "kind": "moe_internals",
        "rank": _v4_divtrace_rank(),
        "layer_idx": int(layer_idx),
        "name": "moe_internals",
        "seq_id": meta["seq_id"],
        "cache_seqlen": meta["cache_seqlen"],
        "stats": stats,
    }
    record.update(captured)
    if extras is not None:
        record["extras"] = extras
    _v4_divtrace_append(record)

    summary_names = [
        "reduced",
        "mlp_input",
        "routed_before_allreduce",
        "routed_after_allreduce",
        "shared",
        "mlp_out",
    ]
    summary = ",".join(
        f"{name}.l2={float(stats[name]['l2']):.6e}"
        for name in summary_names
        if name in stats
    )
    _v4_divtrace_note(
        f"layer={layer_idx:02d} moe_internals "
        f"rank={_v4_divtrace_rank()} "
        f"seq_id={meta['seq_id']} cache_seqlen={meta['cache_seqlen']} "
        f"{summary}"
    )


def _v4_divtrace_emit_final(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> None:
    meta = _v4_divtrace_metadata(cache_seqlens)
    _v4_divtrace_emit_boundary(-1, "final_norm", hidden_states, cache_seqlens)
    topk = min(20, logits.size(-1))
    vals, idx = torch.topk(logits[0, -1].detach().to(torch.float32), k=topk)
    _v4_divtrace_append(
        {
            "kind": "final_topk",
            "rank": _v4_divtrace_rank(),
            "layer_idx": -1,
            "name": "logits_topk",
            "seq_id": meta["seq_id"],
            "cache_seqlen": meta["cache_seqlen"],
            "ids": idx.to(torch.int64).cpu().tolist(),
            "values": [float(v) for v in vals.cpu().tolist()],
        }
    )
    _v4_divtrace_note(
        f"final logits_top{topk} "
        f"rank={_v4_divtrace_rank()} "
        f"seq_id={meta['seq_id']} cache_seqlen={meta['cache_seqlen']} "
        f"ids={idx.to(torch.int64).cpu().tolist()},values={[round(float(v), 6) for v in vals.cpu().tolist()]}"
    )
    _v4_divtrace_flush()


@dataclass
class _CausalLMOutput:
    """Minimal output container with ``.logits`` for BatchGen workers."""

    logits: torch.Tensor


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


# QAT-faithful linear (opt-in): quantize activations to fp8 (block 128,
# ue8m0) and run the official tilelang fp8/fp4 GEMM, exactly like the
# reference `linear()`. Verified bit-exact vs official in
# tests/integration/test_v4_linear_numerics_parity.py, but the tilelang
# kernel-launch storm (256 experts x 43 layers x 3 GEMMs per rank) wedges
# the multi-process server, so it stays off until batched per-layer.
_V4_QAT_LINEAR = os.environ.get("BATCHGEN_V4_QAT_LINEAR", "0") == "1"


def _v4_official_kernels():
    from batchgen.models.deepseek.deepseekv4_flash.assets.inference import (
        kernel,
    )

    return kernel


_v4_qat_linear_logged = {"on": False, "fail": False}


def _qat_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> Optional[torch.Tensor]:
    kern = _v4_official_kernels()
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    k = x.shape[-1]
    if k % 128 != 0:
        return None
    # Parameter-server runtime tensors may arrive as raw uint8 views of the
    # quantized checkpoint bytes; recover the logical dtype from the scale
    # layout (fp4: scale rows == weight rows; fp8: scale rows == ceil(N/128)).
    if weight.dtype in (torch.uint8, torch.int8):
        n = weight.shape[0]
        if scale.shape[0] == n and fp4_dtype is not None:
            weight = weight.view(fp4_dtype)
        elif scale.shape[0] == (n + 127) // 128:
            weight = weight.view(torch.float8_e4m3fn)
        else:
            return None
    is_fp4 = fp4_dtype is not None and weight.dtype == fp4_dtype
    is_fp8 = weight.dtype == torch.float8_e4m3fn
    if not (is_fp4 or is_fp8):
        return None
    x2d = x.reshape(-1, k)
    if x2d.dtype != torch.bfloat16:
        x2d = x2d.to(torch.bfloat16)
    xq, xs = kern.act_quant(x2d, 128, "ue8m0", torch.float8_e8m0fnu)
    wscale = (
        scale
        if scale.dtype == torch.float8_e8m0fnu
        else scale.view(torch.float8_e8m0fnu)
        if scale.dtype == torch.uint8
        else scale.to(torch.float32).to(torch.float8_e8m0fnu)
    )
    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        if is_fp4:
            out = kern.fp4_gemm(xq, xs, weight, wscale, torch.float8_e8m0fnu)
        else:
            out = kern.fp8_gemm(xq, xs, weight, wscale, torch.float8_e8m0fnu)
    finally:
        torch.set_default_dtype(prev_default_dtype)
    if not _v4_qat_linear_logged["on"]:
        _v4_qat_linear_logged["on"] = True
        print(
            f"[V4_QAT_LINEAR] active (first GEMM: fp4={is_fp4}, "
            f"x={tuple(x.shape)}, w={tuple(weight.shape)})",
            flush=True,
        )
    return out.reshape(*x.shape[:-1], out.shape[-1]).to(x.dtype)


def _linear_from_weight(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linear helper for V4 slots.

    FP8 checkpoint tensors are block scaled.  This fallback dequantizes them in
    PyTorch for correctness-oriented bring-up; optimized wrappers/kernels replace
    this path in production.
    """

    if (
        _V4_QAT_LINEAR
        and scale is not None
        and bias is None
        and x.is_cuda
        and x.numel() > 0
    ):
        try:
            out = _qat_linear(x, weight, scale)
        except Exception as exc:
            out = None
            if not _v4_qat_linear_logged["fail"]:
                _v4_qat_linear_logged["fail"] = True
                print(
                    f"[V4_QAT_LINEAR] FAILED first call: {type(exc).__name__}: "
                    f"{exc} (x={tuple(x.shape)},{x.dtype} "
                    f"w={tuple(weight.shape)},{weight.dtype} "
                    f"s={tuple(scale.shape)},{scale.dtype})",
                    flush=True,
                )
        if out is not None:
            return out
        if (
            not _v4_qat_linear_logged["fail"]
            and not _v4_qat_linear_logged["on"]
        ):
            _v4_qat_linear_logged["fail"] = True
            print(
                f"[V4_QAT_LINEAR] SKIPPED first call (returned None): "
                f"x={tuple(x.shape)},{x.dtype} "
                f"w={tuple(weight.shape)},{weight.dtype} "
                f"s={tuple(scale.shape)},{scale.dtype}",
                flush=True,
            )

    raw_weight_shape = tuple(weight.shape)
    weight = _dequant_weight(weight, scale, x.dtype)
    if x.shape[-1] != weight.shape[-1]:
        scale_shape = None if scale is None else tuple(scale.shape)
        raise RuntimeError(
            "DeepSeek-V4 linear shape mismatch: "
            f"input={tuple(x.shape)}, weight={tuple(weight.shape)}, "
            f"raw_weight={raw_weight_shape}, scale={scale_shape}"
        )
    return F.linear(x, weight, bias)


def _dequant_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if _is_fp4_e2m1_weight(weight, scale):
        return _dequant_fp4_e2m1_weight(weight, scale, dtype)
    if scale is not None and scale.ndim == 2 and weight.ndim == 2:
        row_block = max(weight.shape[0] // scale.shape[0], 1)
        col_block = max(weight.shape[1] // scale.shape[1], 1)
        expanded_scale = (
            scale.to(torch.float32)
            .repeat_interleave(row_block, dim=0)
            .repeat_interleave(col_block, dim=1)
        )
        expanded_scale = expanded_scale[: weight.shape[0], : weight.shape[1]]
        return (weight.to(torch.float32) * expanded_scale).to(dtype)
    return weight.to(dtype)


def _is_fp4_e2m1_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
) -> bool:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None and weight.dtype == fp4_dtype:
        return True
    if weight.dtype in (torch.int8, torch.uint8):
        return True
    return (
        scale is not None
        and weight.ndim == 2
        and scale.ndim == 2
        and weight.shape[0] == scale.shape[0]
        and weight.shape[1] == scale.shape[1] * 16
    )


def _fp4_packed_bytes(weight: torch.Tensor) -> torch.Tensor:
    if weight.element_size() == 1:
        return weight.contiguous().view(torch.uint8)
    return weight.contiguous().to(torch.uint8)


def _dequant_fp4_e2m1_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    if scale is None:
        raise RuntimeError(
            "DeepSeek-V4 FP4 weight is missing its E8M0 scale tensor."
        )
    packed = _fp4_packed_bytes(weight)
    table = torch.tensor(
        _FP4_E2M1_TABLE_VALUES,
        dtype=torch.float32,
        device=packed.device,
    )
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    unpacked = torch.empty(
        unpacked_shape, dtype=torch.float32, device=packed.device
    )
    unpacked[..., 0::2] = table[low.long()]
    unpacked[..., 1::2] = table[high.long()]

    expanded_scale = (
        scale.to(torch.float32)
        .unsqueeze(-1)
        .expand(*scale.shape, 32)
        .reshape(*scale.shape[:-1], scale.shape[-1] * 32)
    )
    expanded_scale = expanded_scale[..., : unpacked.shape[-1]]
    return (unpacked * expanded_scale).to(dtype)


class DeepSeekV4FlashLinearSlot(nn.Module):
    """Runtime-loaded linear slot.

    Attention and expert bundle tensors are owned by the BatchGen parameter
    server/wrappers, not by the skeleton state dict.  The slot records shape
    metadata and receives tensors at wrapper execution time.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def set_runtime_tensors(
        self, tensors: Dict[str, torch.Tensor], prefix: str
    ) -> None:
        self.weight = tensors.get(f"{prefix}.weight")
        self.scale = tensors.get(f"{prefix}.scale")

    def clear_runtime_tensors(self) -> None:
        self.weight = None
        self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError(
                f"DeepSeek-V4 linear slot ({self.out_features}, {self.in_features}) "
                "has no runtime weight loaded."
            )
        return _linear_from_weight(x, self.weight, self.scale, self.bias)


class DeepSeekV4FlashRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight.float()).to(dtype)


class DeepSeekV4FlashCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        eps: float,
        overlap: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        coeff = 2 if overlap else 1
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coeff * head_dim, dtype=torch.float32)
        )
        self.wkv = DeepSeekV4FlashLinearSlot(hidden_size, coeff * head_dim)
        self.wgate = DeepSeekV4FlashLinearSlot(hidden_size, coeff * head_dim)
        self.norm = DeepSeekV4FlashRMSNorm(head_dim, eps)


class DeepSeekV4FlashIndexer(nn.Module):
    def __init__(self, config: Any, compress_ratio: int):
        super().__init__()
        hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        q_lora_rank = int(_cfg(config, "q_lora_rank", 1024))
        head_dim = int(_cfg(config, "index_head_dim", 128))
        n_heads = int(_cfg(config, "index_n_heads", 64))
        rope_head_dim = int(
            _cfg(config, "qk_rope_head_dim", _cfg(config, "rope_head_dim", 64))
        )
        eps = float(
            _cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6))
        )

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = int(_cfg(config, "index_topk", 512))
        self.wq_b = DeepSeekV4FlashLinearSlot(q_lora_rank, n_heads * head_dim)
        self.weights_proj = DeepSeekV4FlashLinearSlot(hidden_size, n_heads)
        self.compressor = DeepSeekV4FlashCompressor(
            hidden_size,
            head_dim,
            rope_head_dim,
            compress_ratio,
            eps,
            overlap=True,
        )


class DeepSeekV4FlashAttention(nn.Module):
    """DP attention surface for V4.

    All V4 projection tensors use their checkpoint names as attributes.  The
    optimized sparse/compressed attention implementation is attached by the V4
    attention wrapper; this module also carries a small PyTorch fallback for
    early smoke tests on short prompts.
    """

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.n_heads = int(
            _cfg(config, "num_attention_heads", _cfg(config, "n_heads", 64))
        )
        self.head_dim = int(_cfg(config, "head_dim", 512))
        self.q_lora_rank = int(_cfg(config, "q_lora_rank", 1024))
        self.o_groups = int(_cfg(config, "o_groups", 8))
        self.world_size = int(
            _cfg(
                config,
                "world_size",
                dist.get_world_size() if dist.is_initialized() else 1,
            )
        )
        if self.n_heads % self.world_size != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by world_size "
                f"({self.world_size}) for tensor-parallel attention"
            )
        if self.o_groups % self.world_size != 0:
            raise ValueError(
                f"o_groups ({self.o_groups}) must be divisible by world_size "
                f"({self.world_size}) for tensor-parallel output projection"
            )
        self.n_local_heads = self.n_heads // self.world_size
        self.n_local_groups = self.o_groups // self.world_size
        self.o_lora_rank = int(_cfg(config, "o_lora_rank", 1024))
        self.eps = float(
            _cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6))
        )
        self.softmax_scale = self.head_dim**-0.5

        ratios = list(_cfg(config, "compress_ratios", []))
        self.compress_ratio = (
            int(ratios[layer_idx]) if layer_idx < len(ratios) else 0
        )
        self.window_size = int(_cfg(config, "window_size", 128))
        self.rope_head_dim = int(
            _cfg(config, "qk_rope_head_dim", _cfg(config, "rope_head_dim", 64))
        )
        self._config_ref = config

        self.runtime_phase = "prefill"
        self._prefill_full_tensors: Dict[str, torch.Tensor] = {}

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = DeepSeekV4FlashLinearSlot(
            self.hidden_size, self.q_lora_rank
        )
        self.q_norm = DeepSeekV4FlashRMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = DeepSeekV4FlashLinearSlot(
            self.q_lora_rank, self.n_heads * self.head_dim
        )
        self.wkv = DeepSeekV4FlashLinearSlot(self.hidden_size, self.head_dim)
        self.kv_norm = DeepSeekV4FlashRMSNorm(self.head_dim, self.eps)
        self.wo_a = DeepSeekV4FlashLinearSlot(
            self.n_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
        )
        self.wo_b = DeepSeekV4FlashLinearSlot(
            self.o_groups * self.o_lora_rank, self.hidden_size
        )

        if self.compress_ratio:
            rope_head_dim = int(
                _cfg(
                    config,
                    "qk_rope_head_dim",
                    _cfg(config, "rope_head_dim", 64),
                )
            )
            self.compressor = DeepSeekV4FlashCompressor(
                self.hidden_size,
                self.head_dim,
                rope_head_dim,
                self.compress_ratio,
                self.eps,
                overlap=self.compress_ratio == 4,
            )
            self.indexer = (
                DeepSeekV4FlashIndexer(config, self.compress_ratio)
                if self.compress_ratio == 4
                else None
            )
        else:
            self.compressor = None
            self.indexer = None

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            getattr(self, name).set_runtime_tensors(tensors, name)
        if "attn_sink" in tensors:
            self.attn_sink.data = tensors["attn_sink"].to(self.attn_sink.device)
        if "q_norm.weight" in tensors:
            self.q_norm.weight.data = tensors["q_norm.weight"].to(
                self.q_norm.weight.device
            )
        if "kv_norm.weight" in tensors:
            self.kv_norm.weight.data = tensors["kv_norm.weight"].to(
                self.kv_norm.weight.device
            )
        self._set_compressor_runtime(self.compressor, tensors, "compressor")
        if self.indexer is not None:
            self.indexer.wq_b.set_runtime_tensors(tensors, "indexer.wq_b")
            self.indexer.weights_proj.set_runtime_tensors(
                tensors, "indexer.weights_proj"
            )
            self._set_compressor_runtime(
                self.indexer.compressor, tensors, "indexer.compressor"
            )

    @staticmethod
    def _set_compressor_runtime(comp, tensors, prefix: str) -> None:
        if comp is None:
            return
        ape_key = f"{prefix}.ape"
        norm_key = f"{prefix}.norm.weight"
        if ape_key in tensors:
            comp.ape.data = tensors[ape_key].to(comp.ape.device)
        if norm_key in tensors:
            comp.norm.weight.data = tensors[norm_key].to(
                comp.norm.weight.device
            )
        comp.wkv.set_runtime_tensors(tensors, f"{prefix}.wkv")
        comp.wgate.set_runtime_tensors(tensors, f"{prefix}.wgate")

    def set_prefill_full_tensors(
        self, tensors: Dict[str, torch.Tensor]
    ) -> None:
        self._prefill_full_tensors = tensors

    def clear_prefill_full_tensors(self) -> None:
        self._prefill_full_tensors = {}

    def _get_prefill_full_tensor(self, name: str) -> torch.Tensor:
        tensor = self._prefill_full_tensors.get(name)
        if tensor is None:
            raise RuntimeError(
                f"DeepSeek-V4 prefill requires full replicated tensor "
                f"'{name}' for layer {self.layer_idx}"
            )
        return tensor

    def clear_runtime_tensors(self) -> None:
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            getattr(self, name).clear_runtime_tensors()
        if self.compressor is not None:
            self.compressor.wkv.clear_runtime_tensors()
            self.compressor.wgate.clear_runtime_tensors()
        if self.indexer is not None:
            self.indexer.wq_b.clear_runtime_tensors()
            self.indexer.weights_proj.clear_runtime_tensors()
            self.indexer.compressor.wkv.clear_runtime_tensors()
            self.indexer.compressor.wgate.clear_runtime_tensors()

    def _forward_prefill_sparse(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        from batchgen.models.deepseek.deepseekv4_flash.v4_prefill_sparse import (
            sparse_prefill_attention_sequence,
        )

        bsz, q_len, _ = hidden_states.shape
        prepack = bool(getattr(AttnWrapperBase, "prepack_mode", False))
        cu = getattr(AttnWrapperBase, "prepack_cu_seqlens", None)
        if prepack and cu is not None and bsz == 1:
            bounds = cu.tolist()
            spans = [
                (int(bounds[i]), int(bounds[i + 1]))
                for i in range(len(bounds) - 1)
                if bounds[i + 1] > bounds[i]
            ]
        else:
            spans = [(0, q_len)]

        attn_out = torch.empty_like(hidden_states)
        kv_out = hidden_states.new_empty(bsz, q_len, self.head_dim)
        for b in range(bsz):
            row = hidden_states[b : b + 1]
            row_spans = spans if bsz == 1 else [(0, q_len)]
            for start, end in row_spans:
                seq_x = row[:, start:end]
                seq_attn, seq_kv = sparse_prefill_attention_sequence(
                    self, seq_x
                )
                attn_out[b : b + 1, start:end] = seq_attn
                kv_out[b : b + 1, start:end] = seq_kv
        return attn_out, None, kv_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]
    ]:
        del position_ids, use_cache
        bsz, q_len, _ = hidden_states.shape

        prefill_dp = self.runtime_phase == "prefill" and self.world_size > 1
        if (
            self.runtime_phase == "prefill"
            and q_len > 1
            and os.environ.get("BATCHGEN_V4_SPARSE_PREFILL", "1") == "1"
        ):
            return self._forward_prefill_sparse(hidden_states)
        if prefill_dp:
            n_heads = self.n_heads
            n_groups = self.o_groups
        else:
            n_heads = self.n_local_heads
            n_groups = self.n_local_groups

        q_low = self.q_norm(self.wq_a(hidden_states))
        if prefill_dp:
            q = _linear_from_weight(
                q_low,
                self._get_prefill_full_tensor("wq_b.weight"),
                self._prefill_full_tensors.get("wq_b.scale"),
            )
        else:
            q = self.wq_b(q_low)
        q = q.view(bsz, q_len, n_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + self.eps)

        kv = self.kv_norm(self.wkv(hidden_states))
        kv_for_attn = kv
        if past_key_value is not None:
            kv_for_attn = self._normalize_past_kv(past_key_value)
            if q_len == 1 and cache_seqlens is not None:
                self._write_current_kv(kv_for_attn, kv, cache_seqlens)
        k = kv_for_attn.unsqueeze(2).expand(-1, -1, n_heads, -1)
        v = k
        attn_scores = torch.einsum("bshd,bthd->bhst", q, k) * self.softmax_scale
        attn_scores = self._apply_fallback_masks(
            attn_scores,
            attention_mask,
            cache_seqlens,
            q_len,
            kv_for_attn.size(1),
            past_key_value is not None,
        )
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_output = torch.einsum("bhst,bthd->bshd", attn_weights, v)

        attn_output = attn_output.reshape(
            bsz,
            q_len,
            n_groups,
            n_heads // n_groups * self.head_dim,
        )
        if prefill_dp:
            wo_a_weight = _dequant_weight(
                self._get_prefill_full_tensor("wo_a.weight"),
                None,
                hidden_states.dtype,
            )
            wo_a = wo_a_weight.view(
                n_groups,
                self.o_lora_rank,
                n_heads // n_groups * self.head_dim,
            )
            attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
            attn_output = _linear_from_weight(
                attn_output.flatten(2),
                self._get_prefill_full_tensor("wo_b.weight"),
                self._prefill_full_tensors.get("wo_b.scale"),
            )
            return attn_output, None, kv

        wo_a_weight = self.wo_a.weight
        if wo_a_weight is None:
            raise RuntimeError(
                "DeepSeek-V4 attention wo_a weight is not loaded."
            )
        wo_a_weight = _dequant_weight(
            wo_a_weight,
            self.wo_a.scale,
            hidden_states.dtype,
        )
        wo_a = wo_a_weight.view(
            n_groups,
            self.o_lora_rank,
            n_heads // n_groups * self.head_dim,
        )
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
        attn_output = self.wo_b(attn_output.flatten(2))
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(attn_output)
        return attn_output, None, kv

    @staticmethod
    def _normalize_past_kv(past_key_value: torch.Tensor) -> torch.Tensor:
        if past_key_value.dim() == 4 and past_key_value.size(2) == 1:
            return past_key_value.squeeze(2)
        if past_key_value.dim() == 3:
            return past_key_value
        raise RuntimeError(
            "DeepSeek-V4 fallback attention expected past KV with shape "
            f"[B, T, D] or [B, T, 1, D], got {tuple(past_key_value.shape)}"
        )

    @staticmethod
    def _write_current_kv(
        past_kv: torch.Tensor,
        current_kv: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        positions = (cache_seqlens.to(current_kv.device).long() - 1).clamp_min(
            0
        )
        batch_idx = torch.arange(current_kv.size(0), device=current_kv.device)
        valid = positions < past_kv.size(1)
        if valid.any():
            past_kv[batch_idx[valid], positions[valid]] = current_kv[valid, 0]

    @staticmethod
    def _apply_fallback_masks(
        attn_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_seqlens: Optional[torch.Tensor],
        q_len: int,
        kv_len: int,
        using_past: bool,
    ) -> torch.Tensor:
        neg_inf = torch.finfo(attn_scores.dtype).min
        device = attn_scores.device
        if cache_seqlens is not None:
            valid_lens = cache_seqlens.to(device).long().clamp(max=kv_len)
            key_pos = torch.arange(kv_len, device=device).unsqueeze(0)
            mask = key_pos >= valid_lens.unsqueeze(1)
            return attn_scores.masked_fill(mask[:, None, None, :], neg_inf)

        if attention_mask is not None and attention_mask.dim() == 2:
            key_mask = attention_mask[:, -kv_len:].to(device) == 0
            attn_scores = attn_scores.masked_fill(
                key_mask[:, None, None, :], neg_inf
            )
        elif attention_mask is not None and attention_mask.dim() == 4:
            attn_scores = attn_scores + attention_mask.to(device)

        if not using_past and q_len > 1:
            causal = torch.triu(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal[None, None, :, :], neg_inf
            )
        return attn_scores


class DeepSeekV4FlashGate(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.num_experts = int(
            _cfg(
                config,
                "n_routed_experts",
                _cfg(config, "num_local_experts", 256),
            )
        )
        self.topk = int(
            _cfg(
                config,
                "num_experts_per_tok",
                _cfg(config, "n_activated_experts", 6),
            )
        )
        self.score_func = str(
            _cfg(
                config,
                "scoring_func",
                _cfg(config, "score_func", "sqrtsoftplus"),
            )
        )
        self.route_scale = float(
            _cfg(
                config,
                "routed_scaling_factor",
                _cfg(config, "route_scale", 1.5),
            )
        )
        self.norm_topk_prob = bool(_cfg(config, "norm_topk_prob", True))
        self.is_hash_layer = layer_idx < int(
            _cfg(config, "num_hash_layers", _cfg(config, "n_hash_layers", 3))
        )

        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size)
        )
        if self.is_hash_layer:
            vocab_size = int(_cfg(config, "vocab_size", 129280))
            self.tid2eid = nn.Parameter(
                torch.empty(vocab_size, self.topk, dtype=torch.long),
                requires_grad=False,
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                torch.empty(self.num_experts, dtype=torch.float32)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_hash_layer:
            return hash_routing(
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                hidden_states=hidden_states,
                gate_weight=self.weight,
                topk=self.topk,
                route_scale=self.route_scale,
                score_func=self.score_func,
                norm_topk_prob=self.norm_topk_prob,
            )
        return sqrtsoftplus_topk(
            hidden_states=hidden_states,
            gate_weight=self.weight,
            bias=self.bias,
            topk=self.topk,
            route_scale=self.route_scale,
            norm_topk_prob=self.norm_topk_prob,
        )


class DeepSeekV4FlashExpertPlaceholder(nn.Module):
    """Lightweight expert slot replaced/configured by V4 expert wrappers."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, swiglu_limit: float
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        self.runtime_weights: Optional[Dict[str, torch.Tensor]] = None

    def set_runtime_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        self.runtime_weights = tensors

    def clear_runtime_tensors(self) -> None:
        self.runtime_weights = None

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if self.runtime_weights is None:
            raise RuntimeError("DeepSeek-V4 expert weights are not loaded.")
        return _linear_from_weight(
            x,
            self.runtime_weights[f"{name}.weight"],
            self.runtime_weights.get(f"{name}.scale"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate = self._linear(hidden_states, "w1")
        up = self._linear(hidden_states, "w3")
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate.float(), max=self.swiglu_limit).to(
                gate.dtype
            )
            up = torch.clamp(
                up.float(), min=-self.swiglu_limit, max=self.swiglu_limit
            ).to(up.dtype)
        if _V4_QAT_LINEAR:
            # Official Expert.forward: silu*up in fp32, cast to bf16, then
            # w2's linear act-quants ONCE (block 128). The fused silu-quant
            # kernel would add a second, different fp8 quantization.
            activated = F.silu(gate.float()) * up.float()
        else:
            try:
                from batchgen_kernels.moe.silu_mul_quant import (
                    fused_silu_mul_quant_cuda,
                )

                activated_fp8, _scales = fused_silu_mul_quant_cuda(
                    gate.to(torch.bfloat16), up.to(torch.bfloat16)
                )
                activated = activated_fp8.float() * _scales.unsqueeze(-1)
            except (ImportError, RuntimeError):
                activated = F.silu(gate.float()) * up.float()
        if weights is not None:
            activated = activated * weights
        return self._linear(
            activated.to(
                weights.dtype if weights is not None else hidden_states.dtype
            ),
            "w2",
        )


class DeepSeekV4FlashMoE(nn.Module):
    """V4 EP-MoE surface with global expert slots."""

    _grouped_scratch = None
    _grouped_scratch_key = None

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.runtime_phase = str(_cfg(config, "phase", "prefill"))
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.intermediate_size = int(
            _cfg(
                config,
                "moe_intermediate_size",
                _cfg(config, "moe_inter_dim", 2048),
            )
        )
        self.total_experts = int(
            _cfg(
                config,
                "n_routed_experts",
                _cfg(config, "num_local_experts", 256),
            )
        )
        self.num_experts_per_tok = int(
            _cfg(
                config,
                "num_experts_per_tok",
                _cfg(config, "n_activated_experts", 6),
            )
        )
        self.swiglu_limit = float(_cfg(config, "swiglu_limit", 10.0))
        self.gate = DeepSeekV4FlashGate(config, layer_idx)
        self.experts = nn.ModuleList(
            [
                DeepSeekV4FlashExpertPlaceholder(
                    self.hidden_size, self.intermediate_size, self.swiglu_limit
                )
                for _ in range(self.total_experts)
            ]
        )
        self.shared_experts = DeepSeekV4FlashExpertPlaceholder(
            self.hidden_size, self.intermediate_size, self.swiglu_limit
        )
        self.comm = None
        self.rank = 0
        self.world_size = 1
        self.routed_expert_start_idx = 0
        self.routed_expert_end_idx = self.total_experts
        self.experts_per_rank = self.total_experts
        self.enable_ep_offloading = False
        self.num_tokens_per_rank = None
        self.max_num_tokens_per_rank = None
        self.pad_token_id = int(_cfg(config, "pad_token_id", 0))
        self._grouped_staged = None
        self._divtrace_pending_moe: Optional[dict[str, torch.Tensor]] = None

    def configure_ep(self, rank: int, world_size: int, comm=None) -> None:
        self.comm = comm
        self.rank = rank
        self.world_size = world_size
        self.experts_per_rank = math.ceil(self.total_experts / world_size)
        self.routed_expert_start_idx = min(
            rank * self.experts_per_rank, self.total_experts
        )
        self.routed_expert_end_idx = min(
            (rank + 1) * self.experts_per_rank, self.total_experts
        )
        self.enable_ep_offloading = world_size > 1

    def _use_pynccl(self) -> bool:
        # PyNCCL collectives are validated only for single-token EP decode (PR#2).
        # Prepacked prefill and multi-sequence batches use a different/larger
        # all-gather that deadlocks via PyNcclCommunicator, so fall back to
        # torch.distributed there.
        #
        # The backend choice MUST be globally uniform across ranks: all ranks
        # all-gather a tensor padded to the GLOBAL num_tokens_per_rank, so the
        # gate must use that global value, not this rank's local token count.
        # Using local _cur_real_tokens caused an uneven decode tail (e.g. per-rank
        # counts [2,2,2,1]) to mix PyNCCL on the 1-token rank with
        # torch.distributed on the others -> collective backend mismatch -> hang.
        if AttnWrapperBase is not None and getattr(
            AttnWrapperBase, "prepack_mode", False
        ):
            return False
        if int(getattr(self, "num_tokens_per_rank", 1) or 1) != 1:
            return False
        return _V4_PYNCCL_COMM and getattr(self, "comm", None) is not None

    def _ep_all_gather(self, output: torch.Tensor, inp: torch.Tensor) -> None:
        # torch.distributed.all_gather_into_tensor adds ~8ms/call CPU launch+sync
        # overhead (measured: 340ms/token across 43 layers for a ~7MB transfer).
        # PyNcclCommunicator submits NCCL directly on the current stream (GLM5
        # pattern), avoiding that overhead and staying CUDA-graph-safe.
        if self._use_pynccl():
            with self.comm.change_state(enable=True):
                self.comm.all_gather(
                    output, inp, stream=torch.cuda.current_stream()
                )
        else:
            dist.all_gather_into_tensor(output, inp)

    def _ep_all_reduce(self, tensor: torch.Tensor) -> None:
        if self._use_pynccl():
            with self.comm.change_state(enable=True):
                self.comm.all_reduce(
                    tensor,
                    op=dist.ReduceOp.SUM,
                    stream=torch.cuda.current_stream(),
                )
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def init_num_tokens(self, num_tokens_per_rank: int) -> None:
        self.num_tokens_per_rank = int(num_tokens_per_rank)
        self.max_num_tokens_per_rank = int(num_tokens_per_rank)

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int) -> None:
        num_tokens_per_rank = int(num_tokens_per_rank)
        if (
            self.max_num_tokens_per_rank is None
            or num_tokens_per_rank > self.max_num_tokens_per_rank
        ):
            self.max_num_tokens_per_rank = num_tokens_per_rank
        self.num_tokens_per_rank = num_tokens_per_rank

    def _run_owned_experts(
        self,
        token_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self._should_stream_prefill_owned_experts():
            return self._run_owned_experts_prefill_eager(
                token_states, topk_weights, topk_indices
            )
        return self._run_owned_experts_grouped(
            token_states, topk_weights, topk_indices
        )

    def _should_stream_prefill_owned_experts(self) -> bool:
        # Real first-request prefill runs with world_size=1, so rank0 owns all 256
        # experts for every MoE layer. A full grouped bundle for those 256 experts
        # is ~4x the decode shard (~3.19 GiB/layer vs ~0.80 GiB/layer) and does
        # not fit on rank0/GPU0 once the rest of the model is resident. Decode
        # stays on the grouped path; only the all-owned prefill path falls back to
        # streamed eager expert execution.
        return (
            self.runtime_phase == "prefill"
            and self.world_size == 1
            and self.routed_expert_start_idx == 0
            and self.routed_expert_end_idx == self.total_experts
        )

    def _run_owned_experts_prefill_eager(
        self,
        token_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden = token_states.shape
        out = torch.zeros(
            (num_tokens, hidden),
            dtype=torch.float32,
            device=token_states.device,
        )
        for expert_idx in range(
            self.routed_expert_start_idx, self.routed_expert_end_idx
        ):
            token_ids, topk_pos = torch.where(topk_indices == expert_idx)
            if token_ids.numel() == 0:
                continue
            local_hidden = token_states.index_select(0, token_ids)
            local_weights = topk_weights[token_ids, topk_pos].unsqueeze(-1)
            expert_out = self.experts[expert_idx](local_hidden, local_weights)
            out.index_add_(0, token_ids, expert_out.float())
        return out

    def _expert_weight_dict(self, expert_idx: int):
        wrapper = self.experts[expert_idx]
        module = getattr(wrapper, "module", wrapper)
        rw = getattr(module, "runtime_weights", None)
        if rw is not None:
            return rw
        load = getattr(wrapper, "load_weights", None)
        key = getattr(wrapper, "module_key", None)
        if load is None or key is None:
            return None
        return load(key)

    def _owned_expert_module(self, expert_idx: int) -> nn.Module:
        wrapper = self.experts[expert_idx]
        return getattr(wrapper, "module", wrapper)

    def _owned_expert_runtime_bytes(self) -> int:
        resident_bytes = 0
        for expert_idx in range(
            self.routed_expert_start_idx, self.routed_expert_end_idx
        ):
            module = self._owned_expert_module(expert_idx)
            runtime_weights = getattr(module, "runtime_weights", None)
            if runtime_weights is None:
                continue
            for tensor in runtime_weights.values():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    resident_bytes += tensor.numel() * tensor.element_size()
        return resident_bytes

    def _grouped_staged_bundle_bytes(self) -> int:
        staged = self._grouped_staged
        if staged is None:
            return 0
        seen: set[tuple[torch.device, int, int]] = set()
        resident_bytes = 0

        def _visit(value) -> None:
            nonlocal resident_bytes
            if isinstance(value, torch.Tensor):
                if not value.is_cuda:
                    return
                storage = value.untyped_storage()
                key = (value.device, storage.data_ptr(), storage.nbytes())
                if key in seen:
                    return
                seen.add(key)
                resident_bytes += storage.nbytes()
                return
            if isinstance(value, dict):
                for child in value.values():
                    _visit(child)
                return
            if isinstance(value, (list, tuple)):
                for child in value:
                    _visit(child)

        _visit(staged)
        return resident_bytes

    def _release_owned_expert_runtime_tensors(self) -> int:
        released_bytes = 0
        for expert_idx in range(
            self.routed_expert_start_idx, self.routed_expert_end_idx
        ):
            module = self._owned_expert_module(expert_idx)
            runtime_weights = getattr(module, "runtime_weights", None)
            if runtime_weights is None:
                continue
            for tensor in runtime_weights.values():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    released_bytes += tensor.numel() * tensor.element_size()
            clear_runtime_tensors = getattr(module, "clear_runtime_tensors", None)
            if clear_runtime_tensors is not None:
                clear_runtime_tensors()
        return released_bytes

    def _collect_owned_expert_weight_dicts(
        self,
    ) -> Optional[list[dict[str, torch.Tensor]]]:
        dicts: list[dict[str, torch.Tensor]] = []
        for expert_idx in range(
            self.routed_expert_start_idx, self.routed_expert_end_idx
        ):
            runtime_weights = self._expert_weight_dict(expert_idx)
            if (
                runtime_weights is None
                or "w1.weight" not in runtime_weights
            ):
                return None
            dicts.append(runtime_weights)
        return dicts

    def _stage_owned_expert_weights(
        self, *, release_runtime_tensors: bool = False
    ) -> bool:
        # Canonicalize this layer's owned expert weights into the grouped MoE
        # bundle once, then reuse that bundle across forwards. Resident decode
        # now prebuilds this during model load; streaming decode keeps the lazy
        # first-forward path because its source expert buffers are recyclable.
        if self._grouped_staged is not None:
            if release_runtime_tensors:
                self._release_owned_expert_runtime_tensors()
            return True
        owned_count = self.routed_expert_end_idx - self.routed_expert_start_idx
        if owned_count <= 0:
            return False

        dicts = self._collect_owned_expert_weight_dicts()
        if dicts is None:
            return False

        try:
            from batchgen.moe.v4_slot_moe_sm120 import (
                setup_v4_expert_weight_pointers,
            )

            self._grouped_staged = setup_v4_expert_weight_pointers(
                dicts,
                global_expert_count=self.total_experts,
            )
        except (KeyError, ValueError):
            return False
        finally:
            dicts = None

        if release_runtime_tensors:
            self._release_owned_expert_runtime_tensors()
        return True

    def _run_owned_experts_grouped(
        self,
        token_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self._stage_owned_expert_weights():
            raise RuntimeError(
                "DeepSeek-V4 ragged grouped MoE could not stage owned expert weights"
            )
        from batchgen.moe.v4_slot_moe_sm120 import (
            v4_grouped_mxfp4_moe_forward_3d_ptrs,
        )

        owned_count = self.routed_expert_end_idx - self.routed_expert_start_idx
        return v4_grouped_mxfp4_moe_forward_3d_ptrs(
            token_states,
            topk_weights,
            topk_indices,
            self._grouped_staged,
            self.routed_expert_start_idx,
            owned_count,
            self.swiglu_limit,
        )

    def _forward_local_routed(
        self, flat_states: torch.Tensor, flat_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        topk_weights, topk_indices = self.gate(flat_states, flat_ids)
        return self._run_owned_experts(flat_states, topk_weights, topk_indices)

    def _forward_ep_decode_routed(
        self, flat_states: torch.Tensor, flat_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.num_tokens_per_rank is None:
            raise RuntimeError(
                "DeepSeek-V4 MoE num_tokens_per_rank is not initialized; "
                "configure_decoding must call init_num_tokens before EP decode."
            )
        real_tokens = flat_states.shape[0]
        self._cur_real_tokens = real_tokens
        ntpr = int(self.num_tokens_per_rank)
        if real_tokens > ntpr:
            raise RuntimeError(
                f"DeepSeek-V4 MoE buffer overflow: real_tokens={real_tokens} > "
                f"num_tokens_per_rank={ntpr}"
            )

        padded = flat_states.new_zeros((ntpr, self.hidden_size))
        if real_tokens > 0:
            padded[:real_tokens] = flat_states
        global_states = flat_states.new_empty(
            (self.world_size * ntpr, self.hidden_size)
        )
        self._ep_all_gather(global_states, padded)

        global_ids = None
        if flat_ids is not None:
            padded_ids = torch.full(
                (ntpr,),
                self.pad_token_id,
                dtype=flat_ids.dtype,
                device=flat_ids.device,
            )
            if real_tokens > 0:
                padded_ids[:real_tokens] = flat_ids
            global_ids = torch.empty(
                (self.world_size * ntpr,),
                dtype=flat_ids.dtype,
                device=flat_ids.device,
            )
            self._ep_all_gather(global_ids, padded_ids)
        elif getattr(self.gate, "is_hash_layer", False):
            raise RuntimeError(
                "DeepSeek-V4 hash-routing MoE requires input_ids during EP decode."
            )

        topk_weights, topk_indices = self.gate(global_states, global_ids)
        global_routed = self._run_owned_experts(
            global_states, topk_weights, topk_indices
        )
        self._ep_all_reduce(global_routed)

        start = self.rank * ntpr
        return global_routed[start : start + real_tokens]

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        flat_ids = input_ids.reshape(-1) if input_ids is not None else None
        trace_moe = (
            _V4_DIVTRACE
            and self.layer_idx in _v4_divtrace_active_layers
            and self.layer_idx in _V4_DIVTRACE_MOE_INTERNALS_LAYERS
        )
        pending = self._divtrace_pending_moe if trace_moe else None

        _dt = get_decode_timer()
        topk_weights = None
        topk_indices = None
        if _V4_DIVTRACE and self.layer_idx in _v4_divtrace_active_layers:
            topk_weights, topk_indices = self.gate(flat_states, flat_ids)
            _v4_divtrace_emit_router(
                self.layer_idx,
                topk_indices,
                topk_weights,
                getattr(AttnWrapperBase, "cache_seqlens", None),
            )

        if self.enable_ep_offloading and dist.is_initialized():
            if self.num_tokens_per_rank is None:
                raise RuntimeError(
                    "DeepSeek-V4 MoE num_tokens_per_rank is not initialized; "
                    "configure_decoding must call init_num_tokens before EP decode."
                )
            real_tokens = flat_states.shape[0]
            self._cur_real_tokens = real_tokens
            ntpr = int(self.num_tokens_per_rank)
            if real_tokens > ntpr:
                raise RuntimeError(
                    f"DeepSeek-V4 MoE buffer overflow: real_tokens={real_tokens} > "
                    f"num_tokens_per_rank={ntpr}"
                )

            # Per-layer host rendezvous (bounds inter-rank layer drift to <=1).
            # Streamed (offloaded) experts make each rank do a data-dependent
            # number of load/cudaStreamSynchronize/free ops per layer (the loop
            # skips zero-token experts), so ranks drift apart across layers and
            # the per-layer EP collective deadlocks (7R+1futex) instead of
            # re-aligning. A 1-element all_reduce whose .item() forces a host D2H
            # wait makes every rank block here until all arrive, regardless of
            # routed tokens. Skipped when experts are fully resident (no drift).
            if _v4_layer_barrier_enabled():
                _bar = flat_states.new_ones((), dtype=torch.int32)
                dist.all_reduce(_bar, op=dist.ReduceOp.SUM)
                if int(_bar.item()) != self.world_size:
                    raise RuntimeError(
                        f"DeepSeek-V4 MoE layer barrier desync at layer "
                        f"{self.layer_idx}: saw {int(_bar.item())} of "
                        f"{self.world_size} ranks."
                    )

            padded = flat_states.new_zeros((ntpr, self.hidden_size))
            if real_tokens > 0:
                padded[:real_tokens] = flat_states
            global_states = flat_states.new_empty(
                (self.world_size * ntpr, self.hidden_size)
            )
            with (
                _dt.timed("moe_allgather", self.layer_idx)
                if _dt
                else nullcontext()
            ):
                with (
                    _dt.timed("mc_states_ag", self.layer_idx)
                    if _dt
                    else nullcontext()
                ):
                    _ddl_trace(
                        self.rank,
                        f"moe:before_states_ag L={self.layer_idx} "
                        f"real={real_tokens} ntpr={ntpr} ws={self.world_size}",
                    )
                    self._ep_all_gather(global_states, padded)
                    _ddl_trace(
                        self.rank,
                        f"moe:after_states_ag L={self.layer_idx}",
                    )

                global_ids = None
                if flat_ids is not None:
                    padded_ids = torch.full(
                        (ntpr,),
                        self.pad_token_id,
                        dtype=flat_ids.dtype,
                        device=flat_ids.device,
                    )
                    if real_tokens > 0:
                        padded_ids[:real_tokens] = flat_ids
                    global_ids = torch.empty(
                        (self.world_size * ntpr,),
                        dtype=flat_ids.dtype,
                        device=flat_ids.device,
                    )
                    with (
                        _dt.timed("mc_ids_ag", self.layer_idx)
                        if _dt
                        else nullcontext()
                    ):
                        _ddl_trace(
                            self.rank,
                            f"moe:before_ids_ag L={self.layer_idx}",
                        )
                        self._ep_all_gather(global_ids, padded_ids)
                        _ddl_trace(
                            self.rank,
                            f"moe:after_ids_ag L={self.layer_idx}",
                        )
                elif getattr(self.gate, "is_hash_layer", False):
                    raise RuntimeError(
                        "DeepSeek-V4 hash-routing MoE requires input_ids during EP decode."
                    )

            with (
                _dt.timed("moe_gate", self.layer_idx) if _dt else nullcontext()
            ):
                topk_weights, topk_indices = self.gate(
                    global_states, global_ids
                )
            routed_before_allreduce = None
            routed_after_allreduce = None
            routed_extras: dict[str, Any] = {
                "ep_mode": True,
                "real_tokens": int(real_tokens),
                "num_tokens_per_rank": int(ntpr),
            }
            with (
                _dt.timed("moe_expert_loop", self.layer_idx)
                if _dt
                else nullcontext()
            ):
                routed = self._run_owned_experts(
                    global_states, topk_weights, topk_indices
                )
            if trace_moe:
                start = self.rank * ntpr
                routed_before_allreduce = routed[
                    start : start + real_tokens
                ].clone()
                routed_extras["routed_before_allreduce_global"] = (
                    _v4_divtrace_stats(routed)
                )
                routed_extras["routed_before_allreduce_segments"] = (
                    _v4_divtrace_segment_norms(routed, ntpr)
                )
            with (
                _dt.timed("moe_allreduce", self.layer_idx)
                if _dt
                else nullcontext()
            ):
                _ddl_trace(
                    self.rank,
                    f"moe:before_allreduce L={self.layer_idx}",
                )
                self._ep_all_reduce(routed)
                _ddl_trace(
                    self.rank,
                    f"moe:after_allreduce L={self.layer_idx}",
                )
            if trace_moe:
                routed_extras["routed_after_allreduce_global"] = (
                    _v4_divtrace_stats(routed)
                )
                routed_extras["routed_after_allreduce_segments"] = (
                    _v4_divtrace_segment_norms(routed, ntpr)
                )
            start = self.rank * ntpr
            routed = routed[start : start + real_tokens]
            if trace_moe:
                routed_after_allreduce = routed.clone()
        else:
            routed_before_allreduce = None
            routed_after_allreduce = None
            routed_extras = {
                "ep_mode": False,
                "real_tokens": int(flat_states.shape[0]),
            }
            with (
                _dt.timed("moe_gate", self.layer_idx) if _dt else nullcontext()
            ):
                if topk_weights is None or topk_indices is None:
                    topk_weights, topk_indices = self.gate(
                        flat_states, flat_ids
                    )
            with (
                _dt.timed("moe_expert_loop", self.layer_idx)
                if _dt
                else nullcontext()
            ):
                routed = self._run_owned_experts(
                    flat_states, topk_weights, topk_indices
                )
            if trace_moe:
                routed_before_allreduce = routed.clone()
                routed_after_allreduce = routed.clone()
                routed_extras["routed_before_allreduce_global"] = (
                    _v4_divtrace_stats(routed)
                )
                routed_extras["routed_before_allreduce_segments"] = [
                    {
                        "segment_idx": 0,
                        "rows": int(routed.size(0)),
                        "l2": float(_v4_divtrace_stats(routed)["l2"]),
                        "rms": float(_v4_divtrace_stats(routed)["rms"]),
                        "max_abs": float(_v4_divtrace_stats(routed)["max_abs"]),
                    }
                ]
                routed_extras["routed_after_allreduce_global"] = (
                    _v4_divtrace_stats(routed)
                )
                routed_extras["routed_after_allreduce_segments"] = [
                    {
                        "segment_idx": 0,
                        "rows": int(routed.size(0)),
                        "l2": float(_v4_divtrace_stats(routed)["l2"]),
                        "rms": float(_v4_divtrace_stats(routed)["rms"]),
                        "max_abs": float(_v4_divtrace_stats(routed)["max_abs"]),
                    }
                ]

        with _dt.timed("moe_shared", self.layer_idx) if _dt else nullcontext():
            shared = self.shared_experts(flat_states).float()
        mlp_out = routed + shared
        if trace_moe:
            tensors: Dict[str, torch.Tensor] = {
                "flat_states": flat_states,
                "routed_before_allreduce": routed_before_allreduce,
                "routed_after_allreduce": routed_after_allreduce,
                "shared": shared,
                "mlp_out": mlp_out,
            }
            if pending is not None:
                tensors.update(pending)
            _v4_divtrace_emit_moe_internals(
                self.layer_idx,
                tensors,
                getattr(AttnWrapperBase, "cache_seqlens", None),
                routed_extras,
            )
        return mlp_out.to(hidden_states.dtype).view(shape)


class DeepSeekV4FlashDecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.hc_mult = int(_cfg(config, "hc_mult", 4))
        self.hc_eps = float(_cfg(config, "hc_eps", 1e-6))
        self.hc_sinkhorn_iters = int(_cfg(config, "hc_sinkhorn_iters", 20))
        self.rms_norm_eps = float(
            _cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6))
        )
        hc_dim = self.hc_mult * self.hidden_size
        mix_hc = (2 + self.hc_mult) * self.hc_mult

        self.self_attn = DeepSeekV4FlashAttention(config, layer_idx)
        self.attn = self.self_attn
        self.mlp = DeepSeekV4FlashMoE(config, layer_idx)
        self.ffn = self.mlp
        self.attn_norm = DeepSeekV4FlashRMSNorm(
            self.hidden_size, self.rms_norm_eps
        )
        self.ffn_norm = DeepSeekV4FlashRMSNorm(
            self.hidden_size, self.rms_norm_eps
        )
        self.input_layernorm = self.attn_norm
        self.post_attention_layernorm = self.ffn_norm

        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32)
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32)
        )
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]
    ]:
        del output_attentions, kwargs
        collapse_hc_state = hidden_states.dim() == 3
        if collapse_hc_state:
            hidden_states = (
                hidden_states.unsqueeze(2)
                .expand(-1, -1, self.hc_mult, -1)
                .contiguous()
            )

        trace_layer = False
        if _V4_DIVTRACE:
            trace_layer = _v4_divtrace_begin_layer(
                self.layer_idx, hidden_states, past_key_value
            )
            if trace_layer:
                _v4_divtrace_emit_boundary(
                    self.layer_idx, "h_in", hidden_states, cache_seqlens
                )

        try:
            residual = hidden_states
            attn_input, post, comb = hc_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.hc_eps,
                self.rms_norm_eps,
            )
            attn_input = self.attn_norm(attn_input)
            _dt = get_decode_timer()
            with (
                _dt.timed("self_attn", self.layer_idx) if _dt else nullcontext()
            ):
                attn_out, attn_weights, present = self.self_attn(
                    attn_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    cache_seqlens=cache_seqlens,
                    use_cache=use_cache,
                )
            if trace_layer:
                _v4_divtrace_emit_boundary(
                    self.layer_idx, "attn_out", attn_out, cache_seqlens
                )
            hidden_states = hc_post(attn_out, residual, post, comb)
            if trace_layer:
                _v4_divtrace_emit_boundary(
                    self.layer_idx,
                    "h_after_attn",
                    hidden_states,
                    cache_seqlens,
                )

            residual = hidden_states
            mlp_input, post, comb = hc_pre(
                hidden_states,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.hc_eps,
                self.rms_norm_eps,
            )
            mlp_reduced = mlp_input
            mlp_input = self.ffn_norm(mlp_input)
            trace_moe_internals = (
                trace_layer
                and self.layer_idx in _V4_DIVTRACE_MOE_INTERNALS_LAYERS
            )
            if trace_moe_internals:
                self.mlp._divtrace_pending_moe = {
                    "reduced": mlp_reduced,
                    "mlp_input": mlp_input,
                }
            try:
                with _dt.timed("moe", self.layer_idx) if _dt else nullcontext():
                    mlp_out = self.mlp(mlp_input, input_ids)
            finally:
                if trace_moe_internals:
                    self.mlp._divtrace_pending_moe = None
            hidden_states = hc_post(mlp_out, residual, post, comb)
            if trace_layer:
                if self.layer_idx in _V4_DIVTRACE_FFN_ATTRIB_LAYERS:
                    _v4_divtrace_emit_ffn_attrib(
                        self.layer_idx,
                        residual,
                        mlp_out,
                        post,
                        comb,
                        cache_seqlens,
                    )
                _v4_divtrace_emit_boundary(
                    self.layer_idx,
                    "h_after_ffn",
                    hidden_states,
                    cache_seqlens,
                )
            if collapse_hc_state:
                hidden_states = hidden_states.mean(dim=2)
            return hidden_states, attn_weights, present
        finally:
            if trace_layer:
                _v4_divtrace_end_layer(self.layer_idx)


class DeepSeekV4FlashModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.vocab_size = int(_cfg(config, "vocab_size", 129280))
        self.hc_mult = int(_cfg(config, "hc_mult", 4))
        self.hc_eps = float(_cfg(config, "hc_eps", 1e-6))
        self.rms_norm_eps = float(
            _cfg(config, "rms_norm_eps", _cfg(config, "norm_eps", 1e-6))
        )
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            int(_cfg(config, "pad_token_id", 1)),
        )
        self.embed = self.embed_tokens
        self.layers = nn.ModuleList(
            [
                DeepSeekV4FlashDecoderLayer(config, layer_idx)
                for layer_idx in range(
                    int(
                        _cfg(
                            config,
                            "num_hidden_layers",
                            _cfg(config, "n_layers", 43),
                        )
                    )
                )
            ]
        )
        self.norm = DeepSeekV4FlashRMSNorm(self.hidden_size, self.rms_norm_eps)
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def _hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
        )
        mixes = F.linear(flat, self.hc_head_fn) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base)
            + self.hc_eps
        )
        return torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2).to(
            hidden_states.dtype
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        del return_dict, kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = vocab_parallel_embedding(
                self.embed_tokens, input_ids, self.vocab_size
            )

        hidden_states = (
            inputs_embeds.unsqueeze(2)
            .expand(-1, -1, self.hc_mult, -1)
            .contiguous()
        )
        presents = []
        for idx, layer in enumerate(self.layers):
            past_kv = (
                past_key_values[idx] if past_key_values is not None else None
            )
            hidden_states, _, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_ids=input_ids,
                past_key_value=past_kv,
                output_attentions=bool(output_attentions),
                use_cache=bool(use_cache),
            )
            if use_cache:
                presents.append(present)
        hidden_states = self.norm(self._hc_head(hidden_states))
        if use_cache:
            return hidden_states, tuple(presents)
        return (hidden_states,)


class DeepSeekV4FlashForCausalLM(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.model = DeepSeekV4FlashModel(config)
        self.vocab_size = int(_cfg(config, "vocab_size", 129280))
        hidden_size = int(
            _cfg(config, "hidden_size", _cfg(config, "dim", 4096))
        )
        self.lm_head = nn.Linear(hidden_size, self.vocab_size, bias=False)
        self.head = self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> _CausalLMOutput:
        del kwargs
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = vocab_parallel_lm_head(
            self.lm_head, hidden_states, self.vocab_size
        )
        if _V4_DIVTRACE and _v4_divtrace_should_trace_final(
            hidden_states, past_key_values
        ):
            _v4_divtrace_emit_final(
                hidden_states,
                logits,
                getattr(AttnWrapperBase, "cache_seqlens", None),
            )
        return _CausalLMOutput(logits=logits)
