# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 DSA, prefill side: compute the index keys and write them to Host.

Decode scores every cached token against an index key, so prefill must leave
one per prompt token in the auxiliary (indexer) Host-KV region. That region is
the ``kimi_k31_indexer`` profile: DeepGEMM's page-split FP8 layout, one MQA
key per token per DSA layer (128 e4m3 bytes + one fp32 scale), on the same 24
physical rows as the latent cache.

* :func:`prefill_index_keys` — ``k_norm(wk(hidden))`` for every packed prompt
  token, bf16, through the same identity-rotation gate the decode writer and
  the Q producer pass (``dsa_index_rotation``). These bf16 keys are also what
  the prefill indexer scores.
* :func:`quantize_index_keys` — the shipped per-token e4m3 recipe
  (``quantize_indexer_k_fp8``: amax floor, reciprocal multiply), so the bytes
  prefill stores are the bytes the decode writer's epilogue would have stored.
* :func:`offload_packed_index_keys` — one packed asynchronous Host write per
  layer WITH the scales the page-split region requires, pinned, tracked by
  the worker's end-of-prefill retire, and waited on before returning.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from .dsa_index_rotation import KERNEL_INDEX_HEAD_DIM, prefill_index_k_rows


def _quantize_indexer_k_fp8():
    """The shipped page-split quantizer, imported on first use (its package
    ``__init__`` imports triton)."""
    from batchgen_kernels.attention.dsa.indexer_k_quant import quantize_indexer_k_fp8

    return quantize_indexer_k_fp8


def prefill_index_keys(indexer, hidden_2d: torch.Tensor) -> torch.Tensor:
    """``[T, hidden]`` post-``input_layernorm`` hidden -> ``[T, 128]`` bf16 keys.

    ``index_keys`` is ``k_norm(wk(hidden))``: on bf16 weights the LayerNorm
    runs in fp32 and returns bf16, which is the value the fused decode writer
    rounds to before quantizing.
    """
    if hidden_2d.dim() != 2:
        raise ValueError(
            f"prefill index keys need packed [T, hidden], got {tuple(hidden_2d.shape)}"
        )
    keys = indexer.index_keys(hidden_2d)
    if keys.shape != (hidden_2d.shape[0], KERNEL_INDEX_HEAD_DIM):
        raise RuntimeError(
            f"indexer produced {tuple(keys.shape)} index keys for "
            f"{hidden_2d.shape[0]} tokens; expected "
            f"({hidden_2d.shape[0]}, {KERNEL_INDEX_HEAD_DIM})"
        )
    return prefill_index_k_rows(keys.to(torch.bfloat16)).contiguous()


def quantize_index_keys(keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[T, 128]`` bf16 -> ``([T, 1, 128] e4m3, [T, 1] fp32)`` page-split pair.

    The trailing-1 head axis is the aux region's single MQA head
    (``num_k_heads=1``) the packed Host write expects.
    """
    k_fp8, scales = _quantize_indexer_k_fp8()(keys.unsqueeze(1))
    return k_fp8.contiguous(), scales.contiguous()


def offload_packed_index_keys(
    *,
    aux_view,
    layer_idx: int,
    keys_fp8: torch.Tensor,
    scales: torch.Tensor,
    sequence_ids: Sequence[int],
    sequence_lengths: Sequence[int],
    pin,
    track,
) -> None:
    """Write one layer's packed page-split index keys to the aux Host region.

    ``pin(tensor, layer_idx)`` / ``track(task, layer_idx)`` are the wrapper's
    ``AttnWrapperBase.pin_prefill_offload_tensor`` /
    ``track_prefill_offload_task``: both sources must stay alive until the d2h
    copy lands, and the worker's end-of-prefill retire waits on every task.
    """
    if aux_view is None:
        raise RuntimeError(
            f"Kimi-K3.1 DSA layer {layer_idx}: no auxiliary Host KV worker view "
            "is bound, so the prompt's index keys would be dropped and every "
            "decode step would score against stale or zeroed keys"
        )
    if not getattr(aux_view, "is_page_split", False):
        raise RuntimeError(
            f"Kimi-K3.1 DSA layer {layer_idx}: the auxiliary Host region is not "
            "page-split FP8; K3.1 writes e4m3 keys + fp32 scales for the "
            "ReLU-gated paged scorer, so the kimi_k31_indexer profile must be "
            "the one allocated"
        )
    if keys_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"page-split index keys must be e4m3, got {keys_fp8.dtype}")
    if scales.dtype != torch.float32:
        raise TypeError(f"page-split index-key scales must be fp32, got {scales.dtype}")
    if keys_fp8.dim() != 3 or keys_fp8.shape[1:] != (1, KERNEL_INDEX_HEAD_DIM):
        raise ValueError(
            f"packed index keys must be [T, 1, {KERNEL_INDEX_HEAD_DIM}], got "
            f"{tuple(keys_fp8.shape)}"
        )
    if scales.shape != keys_fp8.shape[:-1]:
        raise ValueError(
            f"scales must be {tuple(keys_fp8.shape[:-1])} (one per key), got "
            f"{tuple(scales.shape)}"
        )
    lengths = [int(n) for n in sequence_lengths]
    ids = list(sequence_ids)
    if len(lengths) != len(ids):
        raise RuntimeError(
            f"Kimi-K3.1 DSA layer {layer_idx}: {len(ids)} sequence ids but "
            f"{len(lengths)} lengths for the packed index-key write"
        )
    if sum(lengths) != keys_fp8.shape[0]:
        raise RuntimeError(
            f"Kimi-K3.1 DSA layer {layer_idx}: packed index keys cover "
            f"{keys_fp8.shape[0]} tokens, sequence lengths sum to {sum(lengths)}"
        )
    if keys_fp8.shape[0] == 0:
        return
    task = aux_view.async_offload_packed_layer_kv_to_host(
        layer_idx=layer_idx,
        sequence_ids=ids,
        k_tensor=keys_fp8,
        v_tensor=None,
        sequence_lengths=lengths,
        scale_tensor=scales,
    )
    pin(keys_fp8, layer_idx)
    pin(scales, layer_idx)
    track(task, layer_idx)
    # Same as the latent offload next to it: complete this copy before the
    # next layer issues its own.
    task.wait()
