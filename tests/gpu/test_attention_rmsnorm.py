"""Staged GPU validation for the production attention RMSNorm kernels.

Run explicitly on a remote GPU machine:

    BATCHGEN_RMSNORM_GPU_STAGE=1 python -m pytest \
        tests/gpu/test_attention_rmsnorm.py -x -q -rA

An explicitly staged run fails during collection when CUDA is unavailable;
ordinary CPU test collection skips this module.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import pytest
import torch


STAGE_ENV = "BATCHGEN_RMSNORM_GPU_STAGE"
EPS = 1e-5
ATOL = 1e-5
RTOL = 1.6e-2
MAX_ABS = 0.03125
MIN_COSINE = 0.9999
MAX_GATE_FAIL_FRACTION = 1e-4
HIDDEN = 6144

if os.environ.get(STAGE_ENV) == "1" and not torch.cuda.is_available():
    raise RuntimeError(
        f"{STAGE_ENV}=1 but CUDA is unavailable; the staged run must not skip")

pytestmark = pytest.mark.skipif(
    os.environ.get(STAGE_ENV) != "1",
    reason=f"staged GPU validation; set {STAGE_ENV}=1 explicitly",
)


@pytest.fixture(scope="module")
def ops() -> tuple[Callable, Callable]:
    from batchgen.attention.fused_kernels.ops import (
        cuda_add_rmsnorm,
        cuda_rmsnorm,
    )

    return cuda_rmsnorm, cuda_add_rmsnorm


def _inputs(
    batch: int,
    hidden: int = HIDDEN,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 20260824,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + batch + hidden)
    x = (torch.randn((batch, hidden), device="cuda", generator=generator) * 0.1).to(dtype)
    residual = (
        torch.randn((batch, hidden), device="cuda", generator=generator) * 0.1
    ).to(dtype)
    hidden_input = (
        torch.randn((batch, hidden), device="cuda", generator=generator) * 0.1
    ).to(dtype)
    weight = (
        1.0 + torch.randn((hidden,), device="cuda", generator=generator) * 0.02
    ).to(dtype)
    return x, residual, hidden_input, weight


def _fp32_rmsnorm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value_fp32 = value.float()
    inv_rms = torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + EPS)
    return value_fp32 * inv_rms * weight.float()


def _assert_numeric(actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual_fp32 = actual.float()
    error = (actual_fp32 - reference).abs()
    max_abs = float(error.max().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_fp32.reshape(-1), reference.reshape(-1), dim=0
        ).item()
    )
    failed = error > (ATOL + RTOL * reference.abs())
    fail_count = int(failed.sum().item())
    fail_fraction = fail_count / reference.numel()
    assert max_abs <= MAX_ABS, f"max_abs={max_abs}"
    assert cosine >= MIN_COSINE, f"cosine={cosine}"
    assert fail_fraction < MAX_GATE_FAIL_FRACTION, (
        f"WGMMA gate failures={fail_count}/{reference.numel()} "
        f"({fail_fraction:.6e})"
    )


def _assert_add_result(
    output: torch.Tensor,
    returned_residual: torch.Tensor,
    residual_input: torch.Tensor,
    residual_reference: torch.Tensor,
    output_reference: torch.Tensor,
) -> None:
    _assert_numeric(output, output_reference)
    assert returned_residual.data_ptr() == residual_input.data_ptr()
    assert torch.equal(residual_input, residual_reference)
    assert torch.equal(returned_residual, residual_reference)


@pytest.mark.parametrize(
    "batch",
    [1, 2, 3, 4, 7, 8, 11, 16, 32, 48, 61, 64, 128, 256, 512, 1024],
)
def test_bf16_6144_matches_fp32_oracle(batch: int, ops: tuple[Callable, Callable]) -> None:
    rmsnorm, add_rmsnorm = ops
    x, residual, hidden_input, weight = _inputs(batch)

    output = rmsnorm(x, weight, EPS)
    _assert_numeric(output, _fp32_rmsnorm(x, weight))

    residual_input = residual.clone()
    unrounded_sum = residual.float() + hidden_input.float()
    output_reference = _fp32_rmsnorm(unrounded_sum, weight)
    residual_reference = unrounded_sum.to(torch.bfloat16)
    add_output, returned_residual = add_rmsnorm(
        residual_input, hidden_input, weight, EPS
    )
    _assert_add_result(
        add_output,
        returned_residual,
        residual_input,
        residual_reference,
        output_reference,
    )


def test_num_valid_tokens_preserves_residual_padding(
    ops: tuple[Callable, Callable],
) -> None:
    rmsnorm, add_rmsnorm = ops
    batch, valid_rows = 7, 3
    x, residual, hidden_input, weight = _inputs(batch, seed=20260825)
    num_valid = torch.tensor([valid_rows], device="cuda", dtype=torch.int32)

    output = rmsnorm(x, weight, EPS, num_valid)
    _assert_numeric(output[:valid_rows], _fp32_rmsnorm(x[:valid_rows], weight))

    residual_input = residual.clone()
    residual_padding = residual_input[valid_rows:].clone()
    unrounded_sum = residual[:valid_rows].float() + hidden_input[:valid_rows].float()
    output_reference = _fp32_rmsnorm(unrounded_sum, weight)
    residual_reference = unrounded_sum.to(torch.bfloat16)
    add_output, returned_residual = add_rmsnorm(
        residual_input, hidden_input, weight, EPS, num_valid
    )

    _assert_numeric(add_output[:valid_rows], output_reference)
    assert returned_residual.data_ptr() == residual_input.data_ptr()
    assert torch.equal(residual_input[:valid_rows], residual_reference)
    assert torch.equal(returned_residual[:valid_rows], residual_reference)
    assert torch.equal(residual_input[valid_rows:], residual_padding)
    assert torch.equal(returned_residual[valid_rows:], residual_padding)
    # Invalid output rows intentionally remain unchecked: empty_like leaves them unspecified.


def _capture(call: Callable[[], object]) -> tuple[torch.cuda.CUDAGraph, object]:
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        assert torch.cuda.is_current_stream_capturing()
        result = call()
    return graph, result


def test_cuda_graph_capture_and_replay(ops: tuple[Callable, Callable]) -> None:
    rmsnorm, add_rmsnorm = ops
    x, residual, hidden_input, weight = _inputs(7, seed=20260826)

    # Load the extension and populate allocator state before capture.
    rmsnorm(x, weight, EPS)
    add_rmsnorm(residual.clone(), hidden_input, weight, EPS)
    torch.cuda.synchronize()

    rms_graph, rms_output = _capture(lambda: rmsnorm(x, weight, EPS))
    rms_output.fill_(float("nan"))
    rms_graph.replay()
    torch.cuda.synchronize()
    _assert_numeric(rms_output, _fp32_rmsnorm(x, weight))

    residual_input = residual.clone()
    add_graph, add_result = _capture(
        lambda: add_rmsnorm(residual_input, hidden_input, weight, EPS)
    )
    add_output, returned_residual = add_result
    add_output.fill_(float("nan"))
    residual_input.copy_(residual)
    add_graph.replay()
    torch.cuda.synchronize()

    unrounded_sum = residual.float() + hidden_input.float()
    _assert_add_result(
        add_output,
        returned_residual,
        residual_input,
        unrounded_sum.to(torch.bfloat16),
        _fp32_rmsnorm(unrounded_sum, weight),
    )


def test_cuda_graph_replay_honors_dynamic_num_valid_tokens(
    ops: tuple[Callable, Callable],
) -> None:
    rmsnorm, add_rmsnorm = ops
    batch, valid_rows = 7, 3
    x, residual, hidden_input, weight = _inputs(batch, seed=20260828)
    num_valid = torch.tensor([batch], device="cuda", dtype=torch.int32)

    # Warm extension and allocator state before capture.
    rmsnorm(x, weight, EPS, num_valid)
    add_rmsnorm(residual.clone(), hidden_input, weight, EPS, num_valid)
    torch.cuda.synchronize()

    rms_graph, rms_output = _capture(
        lambda: rmsnorm(x, weight, EPS, num_valid)
    )
    rms_output.fill_(float("nan"))
    num_valid.fill_(valid_rows)
    rms_graph.replay()
    torch.cuda.synchronize()
    _assert_numeric(
        rms_output[:valid_rows],
        _fp32_rmsnorm(x[:valid_rows], weight),
    )
    assert torch.isnan(rms_output[valid_rows:].float()).all()

    num_valid.fill_(batch)
    residual_input = residual.clone()
    add_graph, add_result = _capture(
        lambda: add_rmsnorm(
            residual_input,
            hidden_input,
            weight,
            EPS,
            num_valid,
        )
    )
    add_output, returned_residual = add_result
    residual_input.copy_(residual)
    residual_padding = residual_input[valid_rows:].clone()
    add_output.fill_(float("nan"))
    num_valid.fill_(valid_rows)
    add_graph.replay()
    torch.cuda.synchronize()

    unrounded_sum = (
        residual[:valid_rows].float() + hidden_input[:valid_rows].float()
    )
    _assert_numeric(
        add_output[:valid_rows],
        _fp32_rmsnorm(unrounded_sum, weight),
    )
    assert returned_residual.data_ptr() == residual_input.data_ptr()
    assert torch.equal(
        residual_input[:valid_rows],
        unrounded_sum.to(torch.bfloat16),
    )
    assert torch.equal(residual_input[valid_rows:], residual_padding)
    assert torch.isnan(add_output[valid_rows:].float()).all()


@pytest.mark.parametrize(
    ("dtype", "hidden"),
    [(torch.bfloat16, 4096), (torch.float16, HIDDEN)],
)
def test_scalar_fallbacks_match_fp32_oracle(
    dtype: torch.dtype,
    hidden: int,
    ops: tuple[Callable, Callable],
) -> None:
    rmsnorm, add_rmsnorm = ops
    x, residual, hidden_input, weight = _inputs(3, hidden, dtype, seed=20260827)

    output = rmsnorm(x, weight, EPS)
    _assert_numeric(output, _fp32_rmsnorm(x, weight))

    residual_input = residual.clone()
    unrounded_sum = residual.float() + hidden_input.float()
    add_output, returned_residual = add_rmsnorm(
        residual_input, hidden_input, weight, EPS
    )
    _assert_add_result(
        add_output,
        returned_residual,
        residual_input,
        unrounded_sum.to(dtype),
        _fp32_rmsnorm(unrounded_sum, weight),
    )


def _unaligned_random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    elements = math.prod(shape)
    storage = (torch.randn(elements + 1, device="cuda", generator=generator) * 0.1).to(
        torch.bfloat16
    )
    tensor = storage[1:].view(shape)
    assert tensor.is_contiguous()
    assert tensor.data_ptr() % 16 != 0
    return tensor


def test_unaligned_bf16_6144_falls_back_safely(
    ops: tuple[Callable, Callable],
) -> None:
    rmsnorm, add_rmsnorm = ops
    batch = 3
    x = _unaligned_random((batch, HIDDEN), 1)
    residual = _unaligned_random((batch, HIDDEN), 2)
    hidden_input = _unaligned_random((batch, HIDDEN), 3)
    weight = _unaligned_random((HIDDEN,), 4)
    weight.add_(1.0)
    assert weight.data_ptr() % 16 != 0

    output = rmsnorm(x, weight, EPS)
    _assert_numeric(output, _fp32_rmsnorm(x, weight))

    unrounded_sum = residual.float() + hidden_input.float()
    add_output, returned_residual = add_rmsnorm(
        residual, hidden_input, weight, EPS
    )
    _assert_add_result(
        add_output,
        returned_residual,
        residual,
        unrounded_sum.to(torch.bfloat16),
        _fp32_rmsnorm(unrounded_sum, weight),
    )
