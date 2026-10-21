"""CPU contract for the sigmoid top-k gates' padding-shape arguments.

Both ``gate_sigmoid_topk_cuda`` (one block per row) and
``gate_sigmoid_topk_warp_cuda`` (one warp per row) follow the same policy.

The kernel accepts two mutually exclusive descriptions of which rows are live:

  * ``num_valid_tokens`` — one device scalar; the live rows are a prefix of N.
  * ``rank_token_counts`` + ``rows_per_rank`` — the EP all-gathered layout, R
    contiguous blocks of ``rows_per_rank`` rows each live up to its own count,
    so the live rows are NOT a prefix of N.

Combining them, or giving half of the per-rank pair, is a caller bug. These
cases must raise a named Python error BEFORE the CUDA extension is loaded, so
they are reachable on a machine with no GPU and they cannot surface as a
CUDA-side abort in the middle of a graph capture.
"""

import pytest
import torch

import batchgen.moe.routing.cuda_routing as cuda_routing
from batchgen.moe.routing import (
    GATE_SIGMOID_TOPK_WARP_MAX_EXPERTS,
    gate_sigmoid_topk_cuda,
    gate_sigmoid_topk_warp_cuda,
)


_SENTINEL = object()  # never dereferenced: every case must fail before the ext loads

_GATES = pytest.mark.parametrize(
    "gate", [gate_sigmoid_topk_cuda, gate_sigmoid_topk_warp_cuda],
    ids=["block", "warp"])


def _call(gate=gate_sigmoid_topk_cuda, **kwargs):
    return gate(
        _SENTINEL,  # router_logits
        _SENTINEL,  # e_score_correction
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _no_extension(monkeypatch):
    """Every case here must be decided before the CUDA extension loads."""
    def refuse():
        raise AssertionError("the extension was loaded before the argument check")
    monkeypatch.setattr(cuda_routing, "_get_ext", refuse)


@_GATES
def test_rank_counts_and_num_valid_tokens_are_mutually_exclusive(gate):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _call(gate, num_valid_tokens=_SENTINEL, rank_token_counts=_SENTINEL,
              rows_per_rank=8)


@_GATES
def test_rank_counts_without_rows_per_rank_fails(gate):
    with pytest.raises(ValueError, match="rows_per_rank is required"):
        _call(gate, rank_token_counts=_SENTINEL)


@_GATES
@pytest.mark.parametrize("rows_per_rank", [0, -1])
def test_non_positive_rows_per_rank_fails(gate, rows_per_rank):
    with pytest.raises(ValueError, match="rows_per_rank must be positive"):
        _call(gate, rank_token_counts=_SENTINEL, rows_per_rank=rows_per_rank)


@_GATES
def test_rows_per_rank_without_rank_counts_fails(gate):
    with pytest.raises(ValueError, match="only meaningful together"):
        _call(gate, rows_per_rank=8)


def test_latent_out_without_offset_fails():
    with pytest.raises(ValueError, match="latent_offset is required"):
        _call(latent_out=_SENTINEL)


# --- one-warp-per-row gate: inputs it cannot route are named errors ---------
def test_warp_gate_refuses_more_experts_than_one_warp_holds():
    logits = torch.zeros(4, GATE_SIGMOID_TOPK_WARP_MAX_EXPERTS + 1)
    bias = torch.zeros(GATE_SIGMOID_TOPK_WARP_MAX_EXPERTS + 1)
    with pytest.raises(ValueError, match="use gate_sigmoid_topk_cuda"):
        gate_sigmoid_topk_warp_cuda(logits, bias, k=8)


@pytest.mark.parametrize("which", ["logits", "bias"])
def test_warp_gate_refuses_non_fp32_inputs_instead_of_casting(which):
    logits = torch.zeros(4, 256)
    bias = torch.zeros(256)
    if which == "logits":
        logits = logits.to(torch.bfloat16)
    else:
        bias = bias.to(torch.bfloat16)
    with pytest.raises(TypeError, match="FP32"):
        gate_sigmoid_topk_warp_cuda(logits, bias, k=8)


def test_warp_gate_refuses_an_index_dtype_other_than_int32_or_int64():
    with pytest.raises(TypeError, match="int32 or int64"):
        gate_sigmoid_topk_warp_cuda(
            torch.zeros(4, 256), torch.zeros(256), k=8,
            topk_indices=torch.empty(4, 8, dtype=torch.int16))


def test_warp_gate_refuses_an_extension_built_without_it(monkeypatch):
    """A stale _C_routing build is a named error, not an AttributeError."""
    monkeypatch.setattr(cuda_routing, "_get_ext", lambda: object())
    with pytest.raises(RuntimeError, match="rebuild batchgen_kernels"):
        gate_sigmoid_topk_warp_cuda(torch.zeros(4, 256), torch.zeros(256), k=8)
