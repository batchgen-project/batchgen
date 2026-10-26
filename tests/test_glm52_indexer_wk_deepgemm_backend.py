"""Contract for the DeepGEMM version of the GLM-5 DSA indexer wk projection.

Two tiers, both CPU-only:

* The REAL ``IndexerWkDeepGemmWeights`` / ``indexer_wk_proj_deepgemm_out`` are
  exercised directly. Only DeepGEMM itself is stubbed -- at the library
  boundary, the one thing that cannot run on a CPU box -- and the stub is a
  RECORDER that also computes the dequantized reference product, so the
  operands the real wrapper hands DeepGEMM are asserted by value, not by
  having been passed through a fake of our own code.
* The switch module and the source-level invariant the dense GEMM relies on
  (the shared quantize zeroes the rows at or past ``num_valid_tokens``) are
  pinned by reading the real modules.

Why the boundary stub and not the real library: ``deep_gemm`` is CUDA-only, so
the module under test is loaded from its file with a stub standing in for that
import. Nothing else about it is replaced.
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_WKDG_SRC = (
    _REPO_ROOT / "batchgen_kernels/attention/dsa/indexer_wk_proj_deepgemm.py"
)
_QUANT_SRC = _REPO_ROOT / "batchgen_kernels/triton/fp8_quantize.py"

_BLOCK = 128


def _load_module_with_stubbed_deep_gemm():
    """Load the real module, with ``deep_gemm`` stubbed for its import only."""
    stub = types.ModuleType("deep_gemm")
    stub.fp8_gemm_nt = lambda *a, **kw: None
    spec = importlib.util.spec_from_file_location(
        "_indexer_wk_proj_deepgemm_under_test", _WKDG_SRC
    )
    module = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("deep_gemm")
    sys.modules["deep_gemm"] = stub
    try:
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("deep_gemm", None)
        else:
            sys.modules["deep_gemm"] = saved
    return module


wkdg = _load_module_with_stubbed_deep_gemm()


# --------------------------------------------------------------------------- #
# Helpers: the real operand layouts, built the way production builds them
# --------------------------------------------------------------------------- #

def _blocked_quantize(x: torch.Tensor, *, valid_rows: int | None = None,
                      tma_aligned: bool = True):
    """A CPU stand-in for ``act_quant(..., scale_tma_aligned=True)``.

    Same contract as the production Triton kernel: per-128-column-block fp32
    scales, and rows at or past ``valid_rows`` written as EXACT ZEROS with a
    1e-12 scale. ``tma_aligned`` reproduces the MN-major scale layout
    (``stride == (1, align4(M))``) the production kernel allocates, which is
    what this path hands DeepGEMM without any per-layer copy.
    """
    M, K = x.shape
    blocks = K // _BLOCK
    out = torch.empty(M, K, dtype=torch.float8_e4m3fn)
    if tma_aligned:
        aligned_m = ((M + 3) // 4) * 4
        scale = torch.empty(blocks, aligned_m, dtype=torch.float32).transpose(0, 1)[:M, :]
    else:
        scale = torch.empty(M, blocks, dtype=torch.float32)
    for m in range(M):
        if valid_rows is not None and m >= valid_rows:
            out[m] = torch.zeros(K).to(torch.float8_e4m3fn)
            scale[m] = 1e-12
            continue
        for b in range(blocks):
            chunk = x[m, b * _BLOCK:(b + 1) * _BLOCK].float()
            s = max(chunk.abs().max().item(), 1.52587890625e-05) / 448.0
            scale[m, b] = s
            out[m, b * _BLOCK:(b + 1) * _BLOCK] = (chunk / s).to(torch.float8_e4m3fn)
    return out, scale


def _dequant(fp8: torch.Tensor, scale: torch.Tensor, row_block: int) -> torch.Tensor:
    """Dequantize a block-scaled FP8 operand to fp32."""
    rows, K = fp8.shape
    blocks = K // _BLOCK
    out = fp8.float()
    for r in range(rows):
        for b in range(blocks):
            out[r, b * _BLOCK:(b + 1) * _BLOCK] *= scale[r // row_block, b]
    return out


def _recorder(monkeypatch):
    """Replace DeepGEMM with a recorder that also computes the real product."""
    seen = []

    def fp8_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                    compiled_dims="", disable_ue8m0_cast=False):
        a_fp8, a_scale = a
        w_fp8, w_scale = b
        seen.append({
            "a_fp8": a_fp8, "a_scale": a_scale,
            "w_fp8": w_fp8, "w_scale": w_scale,
            "out": d, "c": c, "recipe": recipe,
            "disable_ue8m0_cast": disable_ue8m0_cast,
        })
        lhs = _dequant(a_fp8, a_scale, row_block=1)
        rhs = _dequant(w_fp8, w_scale, row_block=_BLOCK)
        d.copy_((lhs @ rhs.t()).to(d.dtype))

    stub = types.ModuleType("deep_gemm")
    stub.fp8_gemm_nt = fp8_gemm_nt
    monkeypatch.setattr(wkdg, "deep_gemm", stub)
    return seen


def _weights(N=128, K=6144):
    torch.manual_seed(7)
    w = (torch.randn(N, K) * 0.1).to(torch.float8_e4m3fn)
    scale = torch.full(((N + _BLOCK - 1) // _BLOCK, K // _BLOCK), 0.03125)
    return wkdg.IndexerWkDeepGemmWeights(w, scale)


# --------------------------------------------------------------------------- #
# Weight construction: the checkpoint's own FP8 operand, validated
# --------------------------------------------------------------------------- #

def test_the_weights_are_the_checkpoint_operand_untouched():
    N, K = 128, 6144
    w = (torch.randn(N, K) * 0.1).to(torch.float8_e4m3fn)
    scale = torch.rand(1, K // _BLOCK, dtype=torch.float32)
    weights = wkdg.IndexerWkDeepGemmWeights(w, scale)
    # No dequantize/requantize round trip: the GEMM reads the checkpoint bytes.
    assert weights.w_fp8.data_ptr() == w.data_ptr()
    assert weights.w_scale.data_ptr() == scale.data_ptr()
    assert weights.N == N and weights.K == K


def test_a_non_fp8_weight_is_named_not_silently_accepted():
    with pytest.raises(TypeError, match="float8_e4m3fn"):
        wkdg.IndexerWkDeepGemmWeights(
            torch.randn(128, 6144, dtype=torch.bfloat16),
            torch.rand(1, 48, dtype=torch.float32),
        )


def test_a_wrong_block_scale_shape_is_named():
    w = torch.randn(128, 6144).to(torch.float8_e4m3fn)
    # The custom WGMMA kernel's layout is 32-row tiles -> (4, 48); DeepGEMM's
    # is 128-row blocks -> (1, 48). Handing over the wrong one must not be
    # quietly reinterpreted.
    with pytest.raises(ValueError, match=r"\(1, 48\)"):
        wkdg.IndexerWkDeepGemmWeights(w, torch.rand(4, 48, dtype=torch.float32))


def test_non_fp32_block_scales_are_named():
    w = torch.randn(128, 6144).to(torch.float8_e4m3fn)
    with pytest.raises(TypeError, match="fp32"):
        wkdg.IndexerWkDeepGemmWeights(w, torch.rand(1, 48, dtype=torch.float16))


# --------------------------------------------------------------------------- #
# The call: operands, output-buffer identity, no per-layer scale copy
# --------------------------------------------------------------------------- #

def test_the_shared_activation_reaches_deep_gemm_unmodified(monkeypatch):
    seen = _recorder(monkeypatch)
    weights = _weights()
    hidden = torch.randn(64, weights.K, dtype=torch.bfloat16) * 0.1
    hidden_fp8, hidden_scale = _blocked_quantize(hidden)
    out = torch.empty(64, weights.N, dtype=torch.bfloat16)

    returned = wkdg.indexer_wk_proj_deepgemm_out(
        hidden_fp8, hidden_scale, weights, out
    )

    assert len(seen) == 1, "exactly one GEMM launch per layer"
    call = seen[0]
    # The SHARED quantize's own tensors, not copies: a copy here would be an
    # extra per-layer kernel and would mean the activation is not shared.
    assert call["a_fp8"] is hidden_fp8
    assert call["a_scale"] is hidden_scale
    assert call["w_fp8"] is weights.w_fp8
    assert call["w_scale"] is weights.w_scale
    # The MN-major TMA-aligned scale is consumed in that layout, untransposed.
    assert call["a_scale"].stride() == (1, 64)
    assert not call["a_scale"].is_contiguous()
    # Output-buffer identity: written in place, exactly as the custom kernel's
    # ``_out`` variant writes ``buffers.indexer_k_raw``.
    assert call["out"] is out
    assert returned is out
    assert call["c"] is None and call["recipe"] is None
    # The configuration the probe timed and validated, and the one this repo's
    # unmasked DeepGEMM path always uses.
    assert call["disable_ue8m0_cast"] is True


def test_the_dense_gemm_produces_the_projection(monkeypatch):
    """Value check against an fp32 reference of the same quantized operands:
    the wrapper must not transpose, slice or rescale anything."""
    _recorder(monkeypatch)
    weights = _weights(N=128, K=256)
    hidden = torch.randn(8, weights.K, dtype=torch.bfloat16) * 0.1
    hidden_fp8, hidden_scale = _blocked_quantize(hidden)
    out = torch.empty(8, weights.N, dtype=torch.bfloat16)
    wkdg.indexer_wk_proj_deepgemm_out(hidden_fp8, hidden_scale, weights, out)

    ref = (
        _dequant(hidden_fp8, hidden_scale, row_block=1)
        @ _dequant(weights.w_fp8, weights.w_scale, row_block=_BLOCK).t()
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_rows_past_num_valid_tokens_come_out_exactly_zero(monkeypatch):
    """Why the dense GEMM needs no ``valid_m`` guard of its own.

    The shared quantize writes exact-zero FP8 rows at and past
    ``num_valid_tokens``, so the dense product's extra rows are exact zeros --
    the same bytes the custom kernel's ``valid_m`` epilogue stored there. No
    extra masking launch is introduced anywhere.
    """
    _recorder(monkeypatch)
    weights = _weights(N=128, K=256)
    hidden = torch.randn(8, weights.K, dtype=torch.bfloat16) * 0.1
    hidden_fp8, hidden_scale = _blocked_quantize(hidden, valid_rows=5)
    out = torch.full((8, weights.N), 7.0, dtype=torch.bfloat16)

    wkdg.indexer_wk_proj_deepgemm_out(hidden_fp8, hidden_scale, weights, out)

    assert (out[5:] == 0).all(), "padded rows must be overwritten with zeros"
    assert (out[:5] != 0).any(), "valid rows must carry the projection"


def test_a_mismatched_output_buffer_is_named(monkeypatch):
    _recorder(monkeypatch)
    weights = _weights(N=128, K=256)
    hidden_fp8, hidden_scale = _blocked_quantize(
        torch.randn(8, weights.K, dtype=torch.bfloat16)
    )
    with pytest.raises(ValueError, match=r"bf16 with shape \(8, 128\)"):
        wkdg.indexer_wk_proj_deepgemm_out(
            hidden_fp8, hidden_scale, weights,
            torch.empty(8, weights.N, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="does not match the weight"):
        wkdg.indexer_wk_proj_deepgemm_out(
            hidden_fp8[:, :_BLOCK], hidden_scale[:, :1], weights,
            torch.empty(8, weights.N, dtype=torch.bfloat16),
        )


def test_a_bf16_activation_is_rejected(monkeypatch):
    _recorder(monkeypatch)
    weights = _weights(N=128, K=256)
    with pytest.raises(TypeError, match="float8_e4m3fn"):
        wkdg.indexer_wk_proj_deepgemm_out(
            torch.randn(8, weights.K, dtype=torch.bfloat16),
            torch.rand(8, 2),
            weights,
            torch.empty(8, weights.N, dtype=torch.bfloat16),
        )


# --------------------------------------------------------------------------- #
# The premise the dense GEMM rests on, pinned in the producing kernel
# --------------------------------------------------------------------------- #

def test_the_shared_quantize_still_zeroes_the_padded_rows():
    """``indexer_wk_proj_deepgemm_out`` drops the ``valid_m`` guard because the
    producer zeroes those rows. That is a GPU Triton kernel, so the invariant
    is pinned here at the source level; its runtime behaviour belongs to the
    GPU tier."""
    src = _QUANT_SRC.read_text()
    head = src.split("def per_token_blocked_quantize_bf16_to_fp8_1d")[0]
    assert "if pid_m >= num_valid_tokens:" in head
    zeroing = head.split("if pid_m >= num_valid_tokens:")[1].split("return")[0]
    assert "tl.store(out_ptr + out_offsets, tl.zeros(" in zeroing, (
        "the padded FP8 rows are no longer written as exact zeros; the dense "
        "wk GEMM would then propagate stale activations into indexer_k_raw"
    )


# --------------------------------------------------------------------------- #
# The switch itself
# --------------------------------------------------------------------------- #

def test_the_switch_selects_the_deepgemm_version_on_this_branch():
    from batchgen.models.glm.glm5 import indexer_wk_backend as backend

    assert backend.GLM5_INDEXER_WK_BACKEND == "deepgemm"
    assert backend.glm5_indexer_wk_backend() == "deepgemm"
    assert backend.glm5_indexer_wk_is_deepgemm() is True


def test_the_old_version_stays_selectable(monkeypatch):
    from batchgen.models.glm.glm5 import indexer_wk_backend as backend

    monkeypatch.setattr(backend, "GLM5_INDEXER_WK_BACKEND", "fused_cuda")
    assert backend.glm5_indexer_wk_backend() == "fused_cuda"
    assert backend.glm5_indexer_wk_is_deepgemm() is False


def test_an_unknown_backend_fails_loud(monkeypatch):
    from batchgen.models.glm.glm5 import indexer_wk_backend as backend

    monkeypatch.setattr(backend, "GLM5_INDEXER_WK_BACKEND", "cutlass")
    with pytest.raises(ValueError, match="GLM5_INDEXER_WK_BACKEND"):
        backend.glm5_indexer_wk_is_deepgemm()
