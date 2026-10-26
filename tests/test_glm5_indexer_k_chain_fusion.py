"""The GLM-5.2 indexer K tail is ONE launch on a page-split FP8 cache.

Ported from SGLang's ``fused_k_indexer_norm_rope_store`` (sgl-project/sglang @
``e00930c5``, tag ``v0.5.21``), the fused kernel absorbs three launches of the
captured decode K path -- ``indexer.k_norm`` (ATen LayerNorm),
``indexer._fused_rope_hadamard_or_fallback`` and the fused quantize+write --
leaving the ``wk`` GEMM and the head-gate projection beside it.

Two things can silently undo that, and both are pinned here:

* the un-fused triple creeping back into the page-split branch (a launch count
  regression that no numeric test would catch), while the bf16 AoS fallback
  must keep it, because AoS records are not quantized;
* a wrong argument at the one remaining call -- the fused kernel now owns the
  LayerNorm parameters and the RoPE tables, so a stale table or a dropped
  ``num_valid_tokens`` would corrupt pages without raising.

The kernel source itself cannot be compiled on a dev box, so its
load-bearing invariants are checked structurally: the upstream attribution it
is licensed under, the warp-uniform early return the full-mask shuffles
depend on, the round-to-bf16 that keeps the fp8 scale equal to the un-fused
path's, and the byte contract shared with
``fused_indexer_k_write_fp8.cu``. Both build registrations are checked too: an
extension that exists in ``setup.py`` but not in ``_jit_registry.py`` cannot
be iterated on with ``BATCHGEN_KERNELS_DEV=1``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GLM5 = _REPO_ROOT / "batchgen" / "models" / "glm" / "glm5"
_SEGMENTS_SRC = _GLM5 / "cuda_graph_segments.py"
_SELECTOR_SRC = _REPO_ROOT / "batchgen" / "attention" / "dsa" / "glm5_decode_selector.py"
_KERNEL_SRC = (
    _REPO_ROOT / "batchgen_kernels" / "src" / "attention"
    / "fused_indexer_k_norm_rope_write_fp8.cu"
)
_SETUP_SRC = _REPO_ROOT / "batchgen_kernels" / "setup.py"
_JIT_SRC = _REPO_ROOT / "batchgen_kernels" / "_jit_registry.py"

_EXTENSION = "batchgen_kernels.attention._C_fused_indexer_k_norm_rope_write_fp8"
_KERNEL_RELPATH = "src/attention/fused_indexer_k_norm_rope_write_fp8.cu"

# The launches the fused kernel replaced; none of them may run on a page-split
# cache any more.
_UNFUSED_TAIL = (
    "indexer.k_norm",
    "indexer._fused_rope_hadamard_or_fallback",
    "write_aux_indexer_k",
    "outputs.indexer_k_tensor.copy_",
)


def _tree(path):
    return ast.parse(path.read_text())


def _method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _call_name(node):
    return ast.unparse(node.func) if isinstance(node, ast.Call) else None


def _calls(node, name):
    return [n for n in ast.walk(node) if _call_name(n) == name]


def _block(stmts):
    """A walkable node for one branch's statements only (never its sibling)."""

    return ast.Module(body=list(stmts), type_ignores=[])


def _calls_in(stmts, name):
    return _calls(_block(stmts), name)


def _forward():
    return _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")


def _page_split_branch():
    fwd = _forward()
    guards = [
        n for n in ast.walk(fwd)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "self.aux_is_page_split"
        and _calls(n, "write_aux_indexer_k_norm_rope")
    ]
    assert len(guards) == 1, (
        f"expected exactly one `if self.aux_is_page_split:` around the fused "
        f"K write, got {len(guards)}"
    )
    return guards[0]


def _fused_call():
    calls = _calls(_page_split_branch(), "write_aux_indexer_k_norm_rope")
    assert len(calls) == 1
    return calls[0]


def _kwargs(call):
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords}


# --------------------------------------------------------------------------
# Launch count: one on page-split, the triple only on AoS
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", _UNFUSED_TAIL)
def test_the_unfused_tail_is_gone_from_the_page_split_branch(name):
    assert not _calls_in(_page_split_branch().body, name), (
        f"{name} is still launched on a page-split FP8 indexer cache"
    )


@pytest.mark.parametrize("name", _UNFUSED_TAIL)
def test_the_aos_fallback_keeps_the_unfused_tail(name):
    branch = _page_split_branch()
    assert branch.orelse, "the bf16 AoS fallback branch disappeared"
    assert _calls_in(branch.orelse, name), (
        f"{name} is missing from the bf16 AoS fallback, which stores "
        f"unquantized records and has no fused path"
    )


def test_the_fused_write_is_the_only_k_tail_launch_on_page_split():
    branch = _page_split_branch()
    names = {
        _call_name(n) for n in ast.walk(_block(branch.body))
        if isinstance(n, ast.Call)
    }
    # ``position_ids.view`` is a free reshape of an argument, not a launch.
    assert names - {"position_ids.view"} == {"write_aux_indexer_k_norm_rope"}, (
        f"the page-split branch issues {sorted(names)}; the whole K tail must "
        f"be the one fused launch"
    )


def test_the_eager_selector_deliberately_keeps_the_unfused_writer():
    """Scope boundary: only the captured decode path is fused.

    Both paths must rotate, which is why the fused call asks for the Hadamard;
    if this ever changes, the two writers would disagree on cache content.
    """

    src = _SELECTOR_SRC.read_text()
    assert "write_aux_indexer_k(" in src
    assert "write_aux_indexer_k_norm_rope" not in src


# --------------------------------------------------------------------------
# The arguments ARE the contract
# --------------------------------------------------------------------------


def test_the_fused_call_takes_the_unnormalized_wk_output():
    assert _kwargs(_fused_call())["k_raw"] == "buffers.indexer_k_raw"


@pytest.mark.parametrize(
    "arg,expected",
    (
        ("ln_weight", "indexer.k_norm.weight"),
        ("ln_bias", "indexer.k_norm.bias"),
        ("ln_eps", "indexer.k_norm.eps"),
    ),
)
def test_the_fused_call_takes_the_models_own_layernorm(arg, expected):
    """The kernel owns the LayerNorm now; its parameters must come from the
    module that used to apply it, bias and epsilon included."""

    assert _kwargs(_fused_call())[arg] == expected


@pytest.mark.parametrize(
    "arg,expected", (("cos_cache", "self.cos_table"), ("sin_cache", "self.sin_table"))
)
def test_the_fused_call_uses_the_hoisted_rope_tables(arg, expected):
    """The same fp32 tables the Q path ropes with, built once per segment --
    not a per-forward ``rotary_emb`` call inside the captured region."""

    assert _kwargs(_fused_call())[arg] == expected


def test_the_fused_call_ropes_by_absolute_position():
    assert _kwargs(_fused_call())["positions"] == "position_ids.view(batch_size)"


def test_the_fused_call_keeps_the_row_gating_and_host_staging():
    kwargs = _kwargs(_fused_call())
    assert kwargs["num_valid_tokens"] == "num_valid_tokens"
    assert kwargs["slot_indices"] == "buffers.kv_aux_slot_indices"
    assert kwargs["token_indices"] == "token_indices"
    assert kwargs["page_table"] == "self.aux_page_table"
    assert kwargs["page_size_tokens"] == "self.aux_page_size"
    assert kwargs["k_fp8_out"] == "outputs.indexer_k_tensor"
    assert kwargs["scales_out"] == "outputs.indexer_k_scale"


def test_the_fused_call_takes_the_segments_one_rotation_predicate():
    """The rotation is never a literal here: it is the same expression the Q
    epilogue in this forward uses, so Q and K can only move together. On this
    page-split FP8 route it resolves to False (upstream v3.2 semantics, which
    preserves the indexer logit and only moves fp8 quantization error).

    The whole producer/consumer set is enumerated in
    ``test_glm52_indexer_no_hadamard_consistency.py``."""

    assert _kwargs(_fused_call())["apply_hadamard"] == "self.aux_applies_hadamard"


def test_the_head_gate_projection_stays_a_separate_launch():
    """Upstream folds it into the wk GEMM; that is not part of this port."""

    assert _calls(_forward(), "head_gates_out")


# --------------------------------------------------------------------------
# Kernel source invariants (no nvcc on a dev box)
# --------------------------------------------------------------------------


def test_the_kernel_keeps_its_upstream_attribution():
    src = _KERNEL_SRC.read_text()
    assert "Apache License, Version 2.0" in src
    assert "Copyright 2025 SGLang Team" in src
    # The exact commit the port was read from, so a later diff is possible.
    assert "e00930c5489053f26d86b179cee0d087f846acbb" in src
    assert "deepseek_v32/indexer_k.cuh" in src
    assert "fused_k_indexer_norm_rope_store" in src


def test_the_early_return_is_warp_uniform_before_any_shuffle():
    """The butterfly and both reductions use the full 32-lane mask, which is
    only defined if no lane of a live warp has exited."""

    src = _KERNEL_SRC.read_text()
    # ``token`` is derived from blockIdx and warp_id only -- never from lane --
    # so the guard retires a whole warp or none of it.
    assert "const int token = blockIdx.x * kWarpsPerBlock + warp_id;" in src
    assert "constexpr unsigned kFullMask = 0xffffffffu;" in src

    body = src[
        src.index("void fused_indexer_k_norm_rope_write_fp8_kernel("):
        src.index("void CheckCudaInt32(")
    ]
    guard = body.index("if (token >= num_tokens)")
    for after in (
        "warp_reduce_sum(sum)",       # LayerNorm mean
        "warp_reduce_sum(var)",       # LayerNorm variance
        "hadamard_128(data, lane",    # the butterfly
        "warp_reduce_max(local_max)",  # the fp8 amax
    ):
        assert guard < body.index(after), (
            f"{after} is reachable before the warp-uniform early return"
        )
    # And nothing exits between them: the only other returns are in the
    # paged-store tail, after every shuffle.
    tail = body.index("if (num_valid_tokens != nullptr")
    assert body.index("warp_reduce_max(local_max)") < tail
    assert body.count("return;") == 5  # guard + 4 store-tail row skips


def test_the_fp8_scale_is_taken_on_bf16_rounded_values():
    """Byte contract with the chain this replaces: the un-fused writer read a
    bf16 tensor, so the amax has to see bf16-rounded numbers."""

    src = _KERNEL_SRC.read_text()
    body = src[src.index("void fused_indexer_k_norm_rope_write_fp8_kernel("):]
    rounds = [
        i for i in range(len(body))
        if body.startswith("__bfloat162float(__float2bfloat16_rn(", i)
    ]
    # Once after the LayerNorm (the tensor the RoPE kernel used to read) and
    # once before the amax (the tensor the writer used to read).
    assert len(rounds) == 2, f"expected two bf16 rounds, found {len(rounds)}"
    assert rounds[1] < body.index("local_max = fmaxf(local_max, fabsf(data[i]))")


def test_the_kernel_keeps_the_quantization_byte_contract():
    src = _KERNEL_SRC.read_text()
    assert "constexpr float kAmaxFloor = 1e-4f;" in src
    assert "constexpr float kFp8Max = 448.0f;" in src
    # Reciprocal MULTIPLY, never a divide, and the SATFINITE cast.
    assert "const float inv = 1.0f / scale;" in src
    assert "data[i] * inv, __NV_SATFINITE,\n" in src or (
        "__nv_cvt_float_to_fp8(data[i] * inv, __NV_SATFINITE," in src
    )
    # NaN in a padded row degenerates to the floor instead of poisoning it.
    assert "float local_max = 0.0f;" in src


def test_the_kernel_writes_the_page_split_layout():
    src = _KERNEL_SRC.read_text()
    assert "constexpr int kScaleBytesPerToken = 4;" in src
    assert "static_cast<long long>(page_size) * (kHeadDim + kScaleBytesPerToken)" in src
    assert "static_cast<long long>(page_size) * kHeadDim +" in src
    # Rows the un-fused writer skipped must still be skipped.
    assert "if (num_valid_tokens != nullptr && token >= *num_valid_tokens)" in src
    assert "if (slot < 0)" in src
    assert "if (page < 0 || page >= num_pages)" in src


def test_the_kernel_is_specialized_and_fails_loud_off_geometry():
    src = _KERNEL_SRC.read_text()
    assert "constexpr int kHeadDim = 128;" in src
    assert "constexpr int kRopeDim = 64;" in src
    assert "k_raw.size(1) == kHeadDim" in src
    assert "specialized for a " in src
    assert 'static_assert(kHeadDim == kWarpSize * kVecSize' in src


# --------------------------------------------------------------------------
# Build registration
# --------------------------------------------------------------------------


def test_the_extension_is_registered_for_aot_and_for_jit_dev_mode():
    setup_src = _SETUP_SRC.read_text()
    jit_src = _JIT_SRC.read_text()
    for name, src in (("setup.py", setup_src), ("_jit_registry.py", jit_src)):
        assert _EXTENSION in src, f"{_EXTENSION} is missing from {name}"
        assert _KERNEL_RELPATH in src, f"{_KERNEL_RELPATH} is missing from {name}"


def test_the_extension_is_not_built_with_fast_math():
    """Approximate reciprocals move values across an e4m3 rounding boundary,
    which is the whole point of the byte contract above."""

    for src_path in (_SETUP_SRC, _JIT_SRC):
        text = src_path.read_text()
        start = text.index(_KERNEL_RELPATH)
        # Stop at this entry's own closing brace so a neighbour's flags cannot
        # leak into (or out of) the window.
        window = text[start:text.index("},", start)]
        assert "--use_fast_math" not in window, (
            f"{src_path.name} builds the fused K writer with fast math"
        )
