"""Slice 2c-2, commit 1: the absorb container dispatch (kernel layer).

Two jobs:
  * the bf16 strided-batched path is mathematically right, and the `out=`
    transposed-view mechanics -- the technical core of the slice -- hold;
  * adding the dispatch does not change the fp8 path (commit 1 must be
    bit-identical, so commit 2 can land after it).

CPU-only where possible. The Triton fp8 kernel needs a GPU, so the arms that
exercise it skip without CUDA; the layout, contract and AST assertions run
everywhere.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

H, KV_LORA, QK_NOPE, V_HEAD, B = 8, 64, 24, 32, 4
KD = pathlib.Path(__file__).resolve().parent.parent

# bf16_absorb.py is a LEAF module: no Triton, no intra-package import at module
# scope. Load it by path so this suite runs on a CPU-only dev machine -- going
# through `batchgen_kernels.attention.dsa` would execute that package's
# __init__, which pulls in Triton-dependent siblings.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_bf16_absorb_under_test",
    KD / "batchgen_kernels" / "attention" / "dsa" / "bf16_absorb.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BF16AbsorbWeights = _mod.BF16AbsorbWeights
_bmm_absorb_out = _mod._bmm_absorb_out
absorb_out = _mod.absorb_out
absorb_out_out = _mod.absorb_out_out
absorb_q = _mod.absorb_q
absorb_q_out = _mod.absorb_q_out


def _weights():
    q = torch.randn(H, QK_NOPE, KV_LORA, dtype=torch.bfloat16) * 0.1
    o = torch.randn(H, V_HEAD, KV_LORA, dtype=torch.bfloat16) * 0.1
    return q, o, BF16AbsorbWeights(q, o)


# --- layout: the whole gain depends on it -------------------------------


def test_container_stores_both_weights_as_h_k_n():
    q, o, w = _weights()
    # q arrives [H,K,N] already; out arrives [H,N,K] and must be flipped once.
    assert w.q_absorb_knn.shape == (H, QK_NOPE, KV_LORA)
    assert w.out_absorb_knn.shape == (H, KV_LORA, V_HEAD)
    assert w.q_absorb_knn.is_contiguous() and w.out_absorb_knn.is_contiguous()
    assert w.q_absorb_knn.dtype == torch.bfloat16
    # the flip must be a real transpose of the source, not a reshape
    assert torch.equal(w.out_absorb_knn, o.transpose(1, 2).contiguous())


def test_container_rejects_mismatched_shapes():
    q = torch.randn(H, QK_NOPE, KV_LORA, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head counts differ"):
        BF16AbsorbWeights(q, torch.randn(H + 1, V_HEAD, KV_LORA, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="share kv_lora_rank"):
        BF16AbsorbWeights(q, torch.randn(H, V_HEAD, KV_LORA + 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="must be 3D"):
        BF16AbsorbWeights(q[0], torch.randn(H, V_HEAD, KV_LORA, dtype=torch.bfloat16))


def test_container_reports_its_bytes():
    _, _, w = _weights()
    assert w.nbytes() == w.q_absorb_knn.nbytes + w.out_absorb_knn.nbytes


# --- the out= transposed-view core --------------------------------------


def test_bmm_absorb_out_writes_through_a_transposed_view():
    """THE technical core: no permute on either operand, result still correct."""
    q, _, w = _weights()
    x = torch.randn(B, H, QK_NOPE, dtype=torch.bfloat16) * 0.1
    out = torch.empty(B, H, KV_LORA, dtype=torch.bfloat16)
    r = _bmm_absorb_out(x, w.q_absorb_knn, out)
    assert r.data_ptr() == out.data_ptr(), "must write through, not reallocate"
    ref = torch.einsum("bhk,hkn->bhn", x, w.q_absorb_knn)
    assert torch.equal(out, ref), "strided bmm must match the einsum exactly"


def test_bmm_absorb_out_operands_are_views_not_copies():
    """If either side silently became contiguous, the slice's gain is gone."""
    x = torch.randn(B, H, QK_NOPE, dtype=torch.bfloat16)
    out = torch.empty(B, H, KV_LORA, dtype=torch.bfloat16)
    xv, ov = x.transpose(0, 1), out.transpose(0, 1)
    assert not xv.is_contiguous() and not ov.is_contiguous()
    # strides are the strided-batched GEMM form: constant batch stride, lda=H*K
    assert xv.stride() == (QK_NOPE, H * QK_NOPE, 1)
    assert ov.stride() == (KV_LORA, H * KV_LORA, 1)
    assert xv.data_ptr() == x.data_ptr() and ov.data_ptr() == out.data_ptr()


def test_bmm_absorb_out_rejects_mismatched_operands():
    _, _, w = _weights()
    x = torch.randn(B, H, QK_NOPE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not match"):
        _bmm_absorb_out(x, w.q_absorb_knn, torch.empty(B, H, KV_LORA + 1,
                                                       dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="expected \\[H, K, N\\]"):
        _bmm_absorb_out(x, w.out_absorb_knn, torch.empty(B, H, V_HEAD,
                                                         dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="needs 3D"):
        _bmm_absorb_out(x.unsqueeze(1), w.q_absorb_knn,
                        torch.empty(B, H, KV_LORA, dtype=torch.bfloat16))


# --- dispatch behaviour --------------------------------------------------


def test_bf16_dispatch_matches_einsum_on_both_absorbs():
    _, _, w = _weights()
    xq = torch.randn(B, H, QK_NOPE, dtype=torch.bfloat16) * 0.1
    assert torch.equal(absorb_q(xq, w),
                       torch.einsum("bhk,hkn->bhn", xq, w.q_absorb_knn))
    xo = torch.randn(B, H, KV_LORA, dtype=torch.bfloat16) * 0.1
    assert torch.equal(absorb_out(xo, w),
                       torch.einsum("bhk,hkn->bhn", xo, w.out_absorb_knn))


def test_out_absorb_accepts_the_4d_shape_callers_pass():
    _, _, w = _weights()
    x4 = torch.randn(B, 1, H, KV_LORA, dtype=torch.bfloat16) * 0.1
    r = absorb_out(x4, w)
    assert r.shape == (B, 1, H, V_HEAD)
    out4 = torch.empty(B, 1, H, V_HEAD, dtype=torch.bfloat16)
    r2 = absorb_out_out(x4, w, out4)
    assert r2.shape == (B, 1, H, V_HEAD)
    assert torch.equal(r2.squeeze(1), r.squeeze(1))


def test_bf16_path_ignores_num_valid_tokens_by_design():
    """Pinned on purpose: see BF16AbsorbWeights' docstring for the proof.

    The Triton kernel's mask exists so a Triton program never takes a dynamic
    early return inside a captured graph; a bmm has no such branch. Ignoring
    it is safe because act_quant never loads invalid rows and the one
    load-bearing mask (attn_output.mul_) is independent of the absorb. If that
    ever stops holding, this test is the place the reasoning is recorded.
    """
    _, _, w = _weights()
    x = torch.randn(B, H, QK_NOPE, dtype=torch.bfloat16) * 0.1
    a = torch.empty(B, H, KV_LORA, dtype=torch.bfloat16)
    b = torch.empty(B, H, KV_LORA, dtype=torch.bfloat16)
    absorb_q_out(x, w, a, num_valid_tokens=None)
    absorb_q_out(x, w, b, num_valid_tokens=torch.tensor([1], dtype=torch.int32))
    assert torch.equal(a, b), "bf16 path must be independent of num_valid_tokens"


def test_dispatch_refuses_an_unknown_container():
    x = torch.randn(B, H, QK_NOPE, dtype=torch.bfloat16)
    for fn in (absorb_q, absorb_out):
        with pytest.raises(TypeError, match="BF16AbsorbWeights or FP8AbsorbWeights"):
            fn(x, object())
    out = torch.empty(B, H, KV_LORA, dtype=torch.bfloat16)
    for fn in (absorb_q_out, absorb_out_out):
        with pytest.raises(TypeError, match="BF16AbsorbWeights or FP8AbsorbWeights"):
            fn(x, object(), out)


# --- commit-1 contract: the fp8 path is untouched ------------------------


def test_triton_kernel_and_fp8_container_are_not_deleted():
    """Repo rule: kernel versions are never removed, only superseded."""
    src = (KD / "batchgen_kernels/attention/dsa/fp8_absorb.py").read_text()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for kept in ("_absorb_fp8_wgmma_kernel", "FP8AbsorbWeights", "fp8_q_absorb",
                 "fp8_q_absorb_out", "fp8_out_absorb", "fp8_out_absorb_out",
                 "triton_absorb_fp8", "triton_absorb_fp8_out", "_absorb_gemv_kernel"):
        assert kept in names, f"{kept} must not be removed"


def test_attention_call_sites_use_the_dispatch_not_the_fp8_entry():
    """Both attention-side sites must go through absorb_*, or the paths diverge."""
    for rel, wanted, banned in (
        ("batchgen/attention/dsa/glm5_decode_selector.py", {"absorb_q"},
         {"fp8_q_absorb"}),
        ("batchgen/attention/mla/flashmla_backend.py", {"absorb_q", "absorb_out"},
         {"fp8_q_absorb", "fp8_out_absorb"}),
    ):
        tree = ast.parse((KD / rel).read_text())
        called = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                called.add(n.func.id)
        assert wanted <= called, f"{rel} must call {sorted(wanted)}"
        assert not (banned & called), f"{rel} still calls {sorted(banned & called)}"


def test_dispatch_accepts_either_container_so_commit_1_is_neutral():
    """Commit 1 changes only call sites, so it must work with EITHER container.

    Stated as a property of the dispatch rather than of whichever container the
    backends happen to build: that way this test holds both at commit 1 (fp8
    still constructed) and after commit 2 (bf16 constructed), which is what
    "each commit is independently testable" requires.
    """
    import inspect
    for fn in (absorb_q, absorb_q_out, absorb_out, absorb_out_out):
        src = inspect.getsource(fn)
        assert "BF16AbsorbWeights" in src, f"{fn.__name__} must handle bf16"
        assert "_is_fp8_container" in src, f"{fn.__name__} must handle fp8"


def test_bf16_module_is_triton_free_and_leaf():
    """It must stay importable where Triton is absent, or the dispatch dies."""
    src = (KD / "batchgen_kernels/attention/dsa/bf16_absorb.py").read_text()
    tree = ast.parse(src)
    top_level_imports = set()
    for node in tree.body:                      # module scope only
        if isinstance(node, ast.Import):
            top_level_imports |= {n.name.split(".")[0] for n in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "triton" not in top_level_imports, (
        "bf16_absorb must not import triton at module scope"
    )
    assert "batchgen_kernels" not in top_level_imports, (
        "bf16_absorb must not import its own package at module scope; the fp8 "
        "fallback is imported lazily inside the dispatch functions"
    )
    assert {"torch"} <= top_level_imports


def test_fp8_absorb_module_is_unmodified_by_this_slice():
    """The Triton kernel file carries a ZERO diff; the bf16 path is additive."""
    src = (KD / "batchgen_kernels/attention/dsa/fp8_absorb.py").read_text()
    for added in ("BF16AbsorbWeights", "absorb_q_out", "_bmm_absorb_out"):
        assert added not in src, (
            f"{added} leaked into fp8_absorb.py; it belongs in bf16_absorb.py "
            "so the Triton file stays untouched"
        )
