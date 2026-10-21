"""GPU parity for the ported indexer-K norm+rope+quantize+write kernel.

``fused_indexer_k_norm_rope_write_fp8`` is a port of SGLang's
``fused_k_indexer_norm_rope_store`` (sgl-project/sglang @ ``e00930c5``, tag
``v0.5.21``). It folds three launches of the GLM-5.2 captured decode K tail --
ATen's ``k_norm`` LayerNorm, the fused RoPE+Hadamard and the fused
quantize+write -- into one.

Unlike the Q-side fp8 fold, this is NOT bit-identical and cannot be: the warp
mean/variance reductions and the 7-stage warp butterfly accumulate in a
different fp32 order than ATen's LayerNorm and the 16-thread RoPE/Hadamard
kernel. The CPU contract file already proves the two orderings realize the
SAME transform (``test_indexer_k_norm_rope_write_contract.py``:
``test_hadamard_matrix_is_the_natural_order_sylvester_transform`` and
``test_reference_matches_the_unfused_chain``), so what is left for a device is
the numeric gate from ``knowledge/agent-memory/rules/kernel-validation.md``:

1. the fp8 keys and fp32 scales agree with the oracle's quantization of the
   oracle's keys, DeepGEMM-style (``calc_diff < 1e-3``), over the production
   decode buckets 1/8/32/64/128 and with and without the rotation;
2. the page bytes, the row gating and the host-append sources are the SAME
   bytes the un-fused writer produces -- that part of the contract is shared
   layout code, not a re-derivation, so only e4m3 rounding ties may differ;
3. adversarial rows (NaN, 1e4, all-zero) stay row-local: one warp owns one
   token, so no reduction may cross tokens, and an all-zero row must take the
   amax floor rather than divide by zero;
4. CUDA-graph bucket padding writes no page bytes while still producing every
   host-append staging row.

The oracle is the torch-only reference chain in
``batchgen_kernels.attention.dsa.indexer_k_quant``, which the CPU file pins
against an independent ``nn.LayerNorm``-based chain.

Run on an SM90 node:
    pytest tests/gpu/test_glm5_indexer_k_norm_rope_parity.py -v
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the indexer-K norm/rope port needs an SM90 CUDA device",
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_QUANT_PATH = (
    _REPO_ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer_k_quant.py"
)


def _load_quant_module():
    name = "_indexer_k_norm_rope_parity_quant"
    spec = importlib.util.spec_from_file_location(name, _QUANT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_quant = _load_quant_module()

AMAX_FLOOR = _quant.AMAX_FLOOR
FP8_MAX = _quant.FP8_MAX
DATA_DIM = _quant.NORM_ROPE_HEAD_DIM
ROPE_DIM = _quant.NORM_ROPE_ROPE_DIM
indexer_k_norm_rope_reference = _quant.indexer_k_norm_rope_reference
quantize_indexer_k_fp8 = _quant.quantize_indexer_k_fp8

PAGE_SIZE = 64
MAX_SEQ = 256
LN_EPS = 1e-5


def _rope_tables(generator: torch.Generator) -> tuple:
    angle = torch.rand(MAX_SEQ, ROPE_DIM, generator=generator) * (2 * math.pi)
    return angle.cos().contiguous(), angle.sin().contiguous()


def _norm_params(dtype: torch.dtype = torch.bfloat16) -> tuple:
    g = torch.Generator().manual_seed(7)
    weight = (1.0 + 0.1 * torch.randn(DATA_DIM, generator=g)).to(dtype)
    bias = (0.05 * torch.randn(DATA_DIM, generator=g)).to(dtype)
    return weight.contiguous(), bias.contiguous()


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """DeepGEMM's fp8 similarity metric (knowledge/.../kernel-validation.md)."""

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denominator)


def _device_case(batch: int, seed: int, device: torch.device) -> dict:
    g = torch.Generator().manual_seed(seed)
    cos, sin = _rope_tables(g)
    weight, bias = _norm_params()
    k_raw = (0.5 * torch.randn(batch, DATA_DIM, generator=g)).to(torch.bfloat16)
    positions = torch.randint(
        0, MAX_SEQ, (batch,), generator=g, dtype=torch.int64
    )
    return {
        "k_raw": k_raw,
        "ln_weight": weight,
        "ln_bias": bias,
        "cos": cos,
        "sin": sin,
        "positions": positions,
        "device": device,
    }


def _run_fused(case: dict, *, slots, token_indices, page_table_rows,
               num_valid, apply_hadamard, num_pages=8):
    from batchgen_kernels.attention.dsa.indexer.indexer_k_norm_rope_write_fp8 import (
        fused_indexer_k_norm_rope_write_fp8,
    )

    device = case["device"]
    cache = torch.zeros(
        num_pages, PAGE_SIZE, 1, DATA_DIM + 4,
        dtype=torch.float8_e4m3fn, device=device,
    )
    k_fp8, scales = fused_indexer_k_norm_rope_write_fp8(
        aux_cache=cache,
        k_raw=case["k_raw"].to(device),
        ln_weight=case["ln_weight"].to(device),
        ln_bias=case["ln_bias"].to(device),
        ln_eps=LN_EPS,
        cos_cache=case["cos"].to(device),
        sin_cache=case["sin"].to(device),
        positions=case["positions"].to(device),
        page_table=torch.tensor(page_table_rows, dtype=torch.int32, device=device),
        slot_indices=torch.tensor(slots, dtype=torch.int32, device=device),
        token_indices=torch.tensor(token_indices, dtype=torch.int32, device=device),
        page_size_tokens=PAGE_SIZE,
        apply_hadamard=apply_hadamard,
        num_valid_tokens=(
            None if num_valid is None
            else torch.tensor([num_valid], dtype=torch.int32, device=device)
        ),
    )
    torch.cuda.synchronize()
    return cache, k_fp8, scales


@pytest.mark.parametrize("apply_hadamard", (True, False))
@pytest.mark.parametrize("batch", (1, 8, 32, 64, 128))
def test_cuda_fused_keys_and_scales_match_the_oracle(batch, apply_hadamard):
    device = torch.device("cuda")
    case = _device_case(batch, 200 + batch, device)
    slots = [0] * batch
    token_indices = list(range(batch))
    _, k_fp8, scales = _run_fused(
        case,
        slots=slots,
        token_indices=token_indices,
        page_table_rows=[[0, 1, 2, 3]],
        num_valid=batch,
        apply_hadamard=apply_hadamard,
    )

    want_keys = indexer_k_norm_rope_reference(
        case["k_raw"], case["ln_weight"], case["ln_bias"], LN_EPS,
        case["cos"], case["sin"], case["positions"], apply_hadamard,
    )
    want_fp8, want_scales = quantize_indexer_k_fp8(want_keys)

    diff = _calc_diff(
        k_fp8.view(batch, DATA_DIM).float().cpu(), want_fp8.float()
    )
    assert diff < 1e-3, f"fp8 keys calc_diff {diff:.3e}"
    torch.testing.assert_close(
        scales.cpu(), want_scales, atol=0.0, rtol=2 ** -7
    )


@pytest.mark.parametrize("apply_hadamard", (True, False))
def test_cuda_dequantized_keys_match_the_oracle(apply_hadamard):
    """Dequantize the kernel's own bytes (keys * scale) and compare against
    the oracle dequantized the same way, at the full production bucket."""

    device = torch.device("cuda")
    batch = 128
    case = _device_case(batch, 777, device)
    _, k_fp8, scales = _run_fused(
        case,
        slots=[0] * batch,
        token_indices=list(range(batch)),
        page_table_rows=[[0, 1, 2, 3]],
        num_valid=batch,
        apply_hadamard=apply_hadamard,
    )
    got = k_fp8.view(batch, DATA_DIM).float().cpu() * scales.cpu()[:, None]

    want_keys = indexer_k_norm_rope_reference(
        case["k_raw"], case["ln_weight"], case["ln_bias"], LN_EPS,
        case["cos"], case["sin"], case["positions"], apply_hadamard,
    )
    want_fp8, want_scales = quantize_indexer_k_fp8(want_keys)
    want = want_fp8.float() * want_scales[:, None]
    diff = _calc_diff(got, want)
    assert diff < 1e-3, f"dequantized keys calc_diff {diff:.3e}"


def test_cuda_fused_pages_match_the_unfused_writer_byte_for_byte():
    """Same page bytes, same row gating, same host-append sources.

    Feeds the un-fused writer the oracle's bf16 keys and the fused kernel the
    raw wk output, then compares the two caches. This is the test that proves
    the port writes OUR layout: page tail scales, -1 slot sentinel,
    ``num_valid_tokens`` bound and the unallocated-page skip.
    """

    from batchgen_kernels.attention.dsa.indexer.indexer_k_write_fp8 import (
        fused_indexer_k_write_fp8,
    )

    device = torch.device("cuda")
    batch = 8
    case = _device_case(batch, 31, device)
    # Page-boundary and gating coverage in one launch: first/last slot of a
    # page, a page crossing, the -1 sentinel, an unallocated page-table entry
    # and rows past num_valid.
    page_table_rows = [[0, 1], [2, 3], [4, -1], [5, 6]]
    slots = [0, 0, 1, -1, 2, 2, 3, 1]
    token_indices = [0, PAGE_SIZE - 1, PAGE_SIZE, 5, 0, PAGE_SIZE, 2, 3]
    num_valid = 7

    want_keys = indexer_k_norm_rope_reference(
        case["k_raw"], case["ln_weight"], case["ln_bias"], LN_EPS,
        case["cos"], case["sin"], case["positions"], apply_hadamard=True,
    )

    unfused_cache = torch.zeros(
        8, PAGE_SIZE, 1, DATA_DIM + 4, dtype=torch.float8_e4m3fn, device=device
    )
    unfused_k_fp8, unfused_scales = fused_indexer_k_write_fp8(
        aux_cache=unfused_cache,
        k_src=want_keys.to(device),
        page_table=torch.tensor(page_table_rows, dtype=torch.int32, device=device),
        slot_indices=torch.tensor(slots, dtype=torch.int32, device=device),
        token_indices=torch.tensor(token_indices, dtype=torch.int32, device=device),
        page_size_tokens=PAGE_SIZE,
        num_valid_tokens=torch.tensor([num_valid], dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize()

    fused_cache, fused_k_fp8, fused_scales = _run_fused(
        case,
        slots=slots,
        token_indices=token_indices,
        page_table_rows=page_table_rows,
        num_valid=num_valid,
        apply_hadamard=True,
    )

    fused_bytes = fused_cache.view(torch.uint8).cpu()
    unfused_bytes = unfused_cache.view(torch.uint8).cpu()
    # Exactly the same bytes are TOUCHED (the zero init makes this a layout,
    # gating and addressing check, independent of the arithmetic).
    assert torch.equal(fused_bytes == 0, unfused_bytes == 0), (
        "the fused and un-fused writers disagree about which bytes they write"
    )
    # And the written bytes are the same bytes, bar e4m3 rounding ties: a
    # stage-order difference of one bf16 ulp can flip at most the last e4m3
    # mantissa bit of a few values.
    differing = int(torch.count_nonzero(fused_bytes != unfused_bytes).item())
    touched = int(torch.count_nonzero(unfused_bytes).item())
    assert differing <= max(1, touched // 100), (
        f"{differing} of {touched} written bytes differ between the fused and "
        f"un-fused writers, far above the e4m3 rounding-tie rate"
    )
    diff = _calc_diff(
        fused_k_fp8.view(batch, DATA_DIM).float().cpu(),
        unfused_k_fp8.view(batch, DATA_DIM).float().cpu(),
    )
    assert diff < 1e-3, f"host-append keys calc_diff {diff:.3e}"
    torch.testing.assert_close(
        fused_scales.cpu(), unfused_scales.cpu(), atol=0.0, rtol=2 ** -7
    )


def test_cuda_adversarial_rows_do_not_cross_tokens():
    device = torch.device("cuda")
    batch = 8
    case = _device_case(batch, 41, device)
    case["k_raw"][2] = float("nan")
    case["k_raw"][3] = 1e4
    case["k_raw"][4] = 0.0
    _, _, scales = _run_fused(
        case,
        slots=[0] * batch,
        token_indices=list(range(batch)),
        page_table_rows=[[0, 1]],
        num_valid=batch,
        apply_hadamard=True,
    )
    clean_rows = [0, 1, 5, 6, 7]
    clean = _device_case(batch, 41, device)
    clean["k_raw"] = case["k_raw"][clean_rows].contiguous()
    clean["positions"] = case["positions"][clean_rows].contiguous()
    _, _, clean_scales = _run_fused(
        clean,
        slots=[0] * len(clean_rows),
        token_indices=list(range(len(clean_rows))),
        page_table_rows=[[0, 1]],
        num_valid=len(clean_rows),
        apply_hadamard=True,
    )
    torch.testing.assert_close(
        scales[clean_rows].cpu(), clean_scales.cpu(), atol=0.0, rtol=0.0
    )
    # The all-zero row takes the amax floor, not a NaN or an infinity.
    assert scales[4].item() == pytest.approx(AMAX_FLOOR / FP8_MAX, rel=1e-6)


@pytest.mark.parametrize("batch", (8, 32, 64, 128))
def test_cuda_bucket_padding_rows_write_no_page_bytes(batch):
    """A captured graph launches the whole bucket; only the valid prefix may
    reach the cache, while the host-append staging covers every row."""

    device = torch.device("cuda")
    case = _device_case(batch, 900 + batch, device)
    num_valid = batch // 2
    slots = [0] * num_valid + [-1] * (batch - num_valid)
    cache, k_fp8, scales = _run_fused(
        case,
        slots=slots,
        token_indices=list(range(batch)),
        page_table_rows=[[0, 1, 2, 3]],
        num_valid=num_valid,
        apply_hadamard=True,
    )
    pages = cache.view(torch.uint8).cpu()
    # Page 0 holds the valid prefix; the scale tail of a padded offset stays 0.
    tail = PAGE_SIZE * DATA_DIM
    for offset in range(num_valid, min(batch, PAGE_SIZE)):
        base = tail + offset * 4
        assert torch.count_nonzero(pages[0, base:base + 4]) == 0, (
            f"padded row {offset} wrote a scale"
        )
    assert torch.count_nonzero(scales) == batch, (
        "every row must still produce a host-append scale"
    )
    assert k_fp8.numel() == batch * DATA_DIM
