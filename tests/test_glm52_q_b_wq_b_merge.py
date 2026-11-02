"""Contract for the merged ``[q_b | wq_b]`` GLM-5 DSA decode GEMM (Group B).

Both projections consume ``q_a_normed``. With ``GLM5_QB_WQB_MERGE`` on, an
indexer layer runs them as ONE 128-block-scaled FP8 DeepGEMM over a load-time
concatenated ``[q_b (16384) | wq_b (4096)] x 2048`` operand; the q_b consumers
read the leading column slice of the one output and the indexer RoPE reads the
trailing slice in place.

Everything here runs on a CPU-only box:

* The REAL torch-only pieces are exercised directly: the switch module, the
  concatenation (``q_b_wq_b_merge.py``) at the real GLM-5.2 shapes, the
  wrapper's load-time seam (its real method body, executed against real
  tensors), and the real ``rope_hadamard_q_out`` / ``fused_score_pipeline``
  wrappers -- loaded from their file with only Triton and the CUDA extension
  modules stubbed at the import boundary.
* The block-scaled GEMM is emulated in fp32 so the scale-to-row map of the
  merged operand is checked by value against two separate GEMMs, with a
  negative control proving the check can see a wrong map.
* The graph segment, eager selector and kernel sources cannot import here
  (FA3 / Triton / DeepGEMM / nvcc), so their contracts are pinned from source.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import pathlib
import sys
import types

import pytest
import torch

from batchgen.models.glm.glm5 import qb_wqb_merge_policy as policy
from batchgen.models.glm.glm5.q_b_wq_b_merge import (
    FP8_WEIGHT_BLOCK,
    MergedGemmAlignmentError,
    MergedGemmOperandError,
    build_q_b_wq_b_merged,
    concat_block_scaled_rows,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GLM5 = _ROOT / "batchgen" / "models" / "glm" / "glm5"
_POLICY_SRC = _GLM5 / "qb_wqb_merge_policy.py"
_SEGMENTS_SRC = _GLM5 / "cuda_graph_segments.py"
_REUSE_SRC = _GLM5 / "reuse_topk_segment.py"
_WRAPPERS_SRC = _GLM5 / "wrappers.py"
_MODEL_SRC = _GLM5 / "model.py"
_SELECTOR_SRC = _ROOT / "batchgen" / "attention" / "dsa" / "glm5_decode_selector.py"
_SCORE_SRC = _ROOT / "batchgen_kernels" / "attention" / "dsa" / "fused_indexer_score.py"
_CSRC = _ROOT / "batchgen_kernels" / "src" / "attention" / "hadamard"
_INDEXER_INIT = _ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer" / "__init__.py"

# GLM-5.2 (batchgen/models/glm/glm5/config.json): 64 heads x 256, 32 index
# heads x 128, q_lora_rank 2048.
NUM_HEADS, Q_HEAD_DIM = 64, 256
INDEX_HEADS, INDEX_HEAD_DIM = 32, 128
Q_LORA_RANK = 2048
Q_B_ROWS = NUM_HEADS * Q_HEAD_DIM          # 16384
WQ_B_ROWS = INDEX_HEADS * INDEX_HEAD_DIM   # 4096
MERGED_ROWS = Q_B_ROWS + WQ_B_ROWS         # 20480
K_BLOCKS = Q_LORA_RANK // FP8_WEIGHT_BLOCK  # 16


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"no class `{name}`")


def _method(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    for node in _class(tree, class_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no `{class_name}.{name}`")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level `{name}`")


def _calls(node: ast.AST, callee: str) -> list[ast.Call]:
    return [
        sub for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == callee
    ]


def _if(node: ast.AST, test_src: str) -> ast.If:
    found = [
        sub for sub in ast.walk(node)
        if isinstance(sub, ast.If) and ast.unparse(sub.test) == test_src
    ]
    assert found, f"no `if {test_src}:`"
    return found[0]


def _block(stmts) -> ast.Module:
    return ast.Module(body=list(stmts), type_ignores=[])


def _fp8(rows: int, cols: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=gen) * 64).to(torch.float8_e4m3fn)


def _scale(rows: int, cols: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(rows, cols, generator=gen, dtype=torch.float32) * 1e-3 + 1e-4


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _strides(t: torch.Tensor) -> tuple:
    """Strides with size-1 dims masked: torch may recompute the stride of a
    singleton dim on view/unflatten (MEASURED: a [1, 16384] column slice of a
    [1, 20480] buffer unflattens to stride 16384, not 20480). A singleton dim
    is never stepped, so its stride carries no layout information."""
    return tuple(None if size == 1 else stride for size, stride in zip(t.shape, t.stride()))


# --------------------------------------------------------------------------- #
# 1. The switch
# --------------------------------------------------------------------------- #

def test_switch_is_on_by_default_on_this_branch():
    assert policy.GLM5_QB_WQB_MERGE is True
    assert policy.glm5_qb_wqb_merge_enabled() is True


@pytest.mark.parametrize("value", ("separate", 1, 0, None, "True"))
def test_switch_fails_loud_on_anything_but_a_bool(monkeypatch, value):
    monkeypatch.setattr(policy, "GLM5_QB_WQB_MERGE", value)
    with pytest.raises(TypeError, match="GLM5_QB_WQB_MERGE must be True or False"):
        policy.glm5_qb_wqb_merge_enabled()


def test_switch_off_is_read_back_as_false(monkeypatch):
    monkeypatch.setattr(policy, "GLM5_QB_WQB_MERGE", False)
    assert policy.glm5_qb_wqb_merge_enabled() is False


def test_switch_module_is_dependency_free():
    """The graph segment imports it at top level; it must not drag in torch,
    Triton, FA3 or DeepGEMM (the 2cb44458 lesson)."""
    imports = [
        node for node in ast.walk(_tree(_POLICY_SRC))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imports == [], [ast.unparse(n) for n in imports]


@pytest.mark.parametrize("path", (_SEGMENTS_SRC, _SELECTOR_SRC, _WRAPPERS_SRC))
def test_every_consumer_imports_the_switch_from_the_policy_module(path):
    found = [
        node for node in _tree(path).body
        if isinstance(node, ast.ImportFrom)
        and node.module == "batchgen.models.glm.glm5.qb_wqb_merge_policy"
        and [a.name for a in node.names] == ["glm5_qb_wqb_merge_enabled"]
    ]
    assert len(found) == 1, path.name


# --------------------------------------------------------------------------- #
# 2. Concatenation algebra at the real shapes
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_members():
    q_b_w = _fp8(Q_B_ROWS, Q_LORA_RANK, seed=1)
    wq_b_w = _fp8(WQ_B_ROWS, Q_LORA_RANK, seed=2)
    q_b_s = _scale(Q_B_ROWS // 128, K_BLOCKS, seed=3)
    wq_b_s = _scale(WQ_B_ROWS // 128, K_BLOCKS, seed=4)
    return q_b_w, q_b_s, wq_b_w, wq_b_s


def test_merged_operand_shapes_and_bytes(real_members):
    q_b_w, q_b_s, wq_b_w, wq_b_s = real_members
    merged = build_q_b_wq_b_merged(q_b_w, q_b_s, wq_b_w, wq_b_s)
    assert tuple(merged.weight.shape) == (MERGED_ROWS, Q_LORA_RANK)
    assert merged.weight.dtype == torch.float8_e4m3fn
    assert merged.weight.is_contiguous()
    assert tuple(merged.scale.shape) == (160, 16)
    assert merged.scale.dtype == torch.float32 and merged.scale.is_contiguous()
    assert merged.q_b_rows == Q_B_ROWS and merged.wq_b_rows == WQ_B_ROWS
    assert merged.q_b_scale_rows == 128
    # Checkpoint bytes, untouched, in [q_b | wq_b] order.
    assert _same_bytes(merged.weight[:Q_B_ROWS], q_b_w)
    assert _same_bytes(merged.weight[Q_B_ROWS:], wq_b_w)
    assert torch.equal(merged.scale[:128], q_b_s)
    assert torch.equal(merged.scale[128:], wq_b_s)


def test_scale_to_row_map_is_exact_for_every_merged_row(real_members):
    """For every merged row n: merged_scale[n // 128] is the scale row its
    member's own GEMM would use. This is the check a wrong member order or a
    misaligned member would fail."""
    q_b_w, q_b_s, wq_b_w, wq_b_s = real_members
    merged = build_q_b_wq_b_merged(q_b_w, q_b_s, wq_b_w, wq_b_s)
    rows = torch.arange(MERGED_ROWS)
    via_merged = merged.scale[rows // 128]
    in_q_b = rows < Q_B_ROWS
    via_member = torch.where(
        in_q_b.view(-1, 1),
        q_b_s[(rows.clamp(max=Q_B_ROWS - 1)) // 128],
        wq_b_s[((rows - Q_B_ROWS).clamp(min=0)) // 128],
    )
    assert torch.equal(via_merged, via_member)


def test_non_last_misaligned_member_is_a_named_error():
    k = Q_LORA_RANK
    with pytest.raises(MergedGemmAlignmentError, match="kv_like: 576 rows"):
        concat_block_scaled_rows(
            (
                ("kv_like", _fp8(576, k, 5), _scale(5, K_BLOCKS, 6)),
                ("tail", _fp8(128, k, 7), _scale(1, K_BLOCKS, 8)),
            )
        )


def test_misaligned_member_is_legal_only_when_last():
    """The Group A order ``[q_a | wk | kv_a]`` (N = 2752, a half-block tail)."""
    k = 6144
    kb = k // 128
    weight, scale, row_offsets, scale_row_offsets = concat_block_scaled_rows(
        (
            ("q_a", _fp8(2048, k, 9), _scale(16, kb, 10)),
            ("wk", _fp8(128, k, 11), _scale(1, kb, 12)),
            ("kv_a", _fp8(576, k, 13), _scale(5, kb, 14)),
        )
    )
    assert tuple(weight.shape) == (2752, k)
    assert tuple(scale.shape) == (22, kb)
    assert row_offsets == [0, 2048, 2176]
    assert scale_row_offsets == [0, 16, 17]


@pytest.mark.parametrize(
    "case",
    ("k_mismatch", "scale_rows", "scale_cols", "scale_dtype", "weight_dtype", "single"),
)
def test_operand_disagreements_are_named_errors(case):
    k = Q_LORA_RANK
    a = ("a", _fp8(256, k, 1), _scale(2, K_BLOCKS, 2))
    b = ("b", _fp8(128, k, 3), _scale(1, K_BLOCKS, 4))
    if case == "k_mismatch":
        b = ("b", _fp8(128, k - 128, 3), _scale(1, K_BLOCKS - 1, 4))
    elif case == "scale_rows":
        b = ("b", b[1], _scale(2, K_BLOCKS, 4))
    elif case == "scale_cols":
        b = ("b", b[1], _scale(1, K_BLOCKS + 1, 4))
    elif case == "scale_dtype":
        b = ("b", b[1], b[2].to(torch.bfloat16))
    elif case == "weight_dtype":
        b = ("b", b[1].to(torch.bfloat16), b[2])
    members = (a,) if case == "single" else (a, b)
    with pytest.raises(MergedGemmOperandError):
        concat_block_scaled_rows(members)


# --------------------------------------------------------------------------- #
# 3. fp32 emulation: merged GEMM == two separate GEMMs, exactly
# --------------------------------------------------------------------------- #

def _emulated_block_scaled_gemm(a_fp8, a_scale, w_fp8, w_scale) -> torch.Tensor:
    """DeepGEMM's 1D2D recipe in fp32: per 128-wide K block, a raw product of
    the fp8 operands, promoted by ``scale_a[m, kb] * scale_b[n // 128, kb]``
    and accumulated in k-block order. N does not appear in the k order."""
    m, k = a_fp8.shape
    n = w_fp8.shape[0]
    a = a_fp8.float()
    w = w_fp8.float()
    out = torch.zeros(m, n, dtype=torch.float32)
    row_block = torch.arange(n) // 128
    for kb in range(k // 128):
        cols = slice(kb * 128, (kb + 1) * 128)
        raw = a[:, cols] @ w[:, cols].t()
        out += (a_scale[:, kb].view(m, 1) * w_scale[row_block, kb].view(1, n)) * raw
    return out


def test_emulated_merged_gemm_equals_the_two_member_gemms(real_members):
    q_b_w, q_b_s, wq_b_w, wq_b_s = real_members
    merged = build_q_b_wq_b_merged(q_b_w, q_b_s, wq_b_w, wq_b_s)
    m = 4
    a = _fp8(m, Q_LORA_RANK, seed=21)
    a_scale = _scale(m, K_BLOCKS, seed=22)

    out_merged = _emulated_block_scaled_gemm(a, a_scale, merged.weight, merged.scale)
    out_q_b = _emulated_block_scaled_gemm(a, a_scale, q_b_w, q_b_s)
    out_wq_b = _emulated_block_scaled_gemm(a, a_scale, wq_b_w, wq_b_s)
    assert torch.equal(out_merged[:, :Q_B_ROWS], out_q_b)
    assert torch.equal(out_merged[:, Q_B_ROWS:], out_wq_b)

    # Negative control: the same check sees a scale table that is off by one
    # block row for the wq_b member (what a misaligned boundary would do).
    shifted = merged.scale.clone()
    shifted[128:] = torch.roll(shifted[128:], shifts=1, dims=0)
    out_bad = _emulated_block_scaled_gemm(a, a_scale, merged.weight, shifted)
    assert torch.equal(out_bad[:, :Q_B_ROWS], out_q_b)
    assert not torch.equal(out_bad[:, Q_B_ROWS:], out_wq_b)


# --------------------------------------------------------------------------- #
# 4. Output slices: legal views with the expected strides
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("batch", (1, 3, 64))
def test_merged_output_slices_are_strided_views(batch):
    q_flat_merged = torch.empty(batch, MERGED_ROWS, dtype=torch.bfloat16)
    q_flat = q_flat_merged[:, :Q_B_ROWS]
    q_flat_indexer = q_flat_merged[:, Q_B_ROWS:]

    row = None if batch == 1 else MERGED_ROWS
    q_heads = q_flat.unflatten(-1, (NUM_HEADS, Q_HEAD_DIM))
    assert _strides(q_heads) == (row, Q_HEAD_DIM, 1)
    assert q_heads.data_ptr() == q_flat_merged.data_ptr()
    q_index = q_flat_indexer.unflatten(-1, (INDEX_HEADS, INDEX_HEAD_DIM))
    assert _strides(q_index) == (row, INDEX_HEAD_DIM, 1)
    assert q_index.data_ptr() == q_flat_merged.data_ptr() + Q_B_ROWS * 2
    # 16-byte aligned start for the strided RoPE kernel's vector loads.
    assert (Q_B_ROWS * 2) % 16 == 0 and MERGED_ROWS % 8 == 0

    # MEASURED on this torch: `.view(batch, H, D)` on the column slice is a
    # legal view too (it leaves the batch axis alone) and equals the
    # `.unflatten` the segment uses -- the design doc's "`.view` raises" claim
    # does not hold. The real hazard is flattening batch and heads together:
    # the flat view is refused and `reshape` silently COPIES, which is why the
    # indexer RoPE takes the strided source through its own entry point.
    as_view = q_flat.view(batch, NUM_HEADS, Q_HEAD_DIM)
    assert (as_view.data_ptr(), _strides(as_view)) == (q_heads.data_ptr(), _strides(q_heads))
    if batch > 1:
        with pytest.raises(RuntimeError):
            q_index.view(batch * INDEX_HEADS, INDEX_HEAD_DIM)
        flat = q_index.reshape(batch * INDEX_HEADS, INDEX_HEAD_DIM)
        assert flat.data_ptr() != q_index.data_ptr(), "reshape must copy here"

    # A smaller bucket's leading-dim view keeps the same row stride.
    small_rows = max(1, batch // 2)
    small = q_flat_merged[:small_rows]
    assert _strides(small[:, :Q_B_ROWS].unflatten(-1, (NUM_HEADS, Q_HEAD_DIM))) == (
        None if small_rows == 1 else MERGED_ROWS, Q_HEAD_DIM, 1
    )


def test_unflatten_is_the_same_view_on_the_separate_version():
    """OFF path: `q_flat` is contiguous and `.unflatten` produces exactly the
    tensor `.view` produced before (same storage, shape, strides)."""
    q_flat = torch.empty(5, Q_B_ROWS, dtype=torch.bfloat16)
    a = q_flat.view(5, NUM_HEADS, Q_HEAD_DIM)
    b = q_flat.unflatten(-1, (NUM_HEADS, Q_HEAD_DIM))
    assert (a.data_ptr(), a.shape, a.stride()) == (b.data_ptr(), b.shape, b.stride())


# --------------------------------------------------------------------------- #
# 5. The wrapper's load-time seam, real method body, real tensors
# --------------------------------------------------------------------------- #

def _wrapper_method(name: str):
    """Compile the real ``GLM5AttnWrapper.<name>`` body (wrappers.py cannot
    import on a CPU box) with its module-level names bound to the real ones."""
    fn = _method(_tree(_WRAPPERS_SRC), "GLM5AttnWrapper", name)
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {
        "build_q_b_wq_b_merged": build_q_b_wq_b_merged,
        "logging": logging,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(_WRAPPERS_SRC), "exec"), namespace)
    return namespace[name]


def _fake_wrapper(q_b_w, q_b_s, *, registered=True):
    param = torch.nn.Parameter(q_b_w, requires_grad=False)
    module = types.SimpleNamespace(q_b_proj=types.SimpleNamespace(weight=param))
    return types.SimpleNamespace(
        layer_idx=7,
        module=module,
        weight_dequant_scale={"q_b_proj.weight_scale_inv": q_b_s},
        fp8_q_b_proj=param.data if registered else None,
        _fp8_q_b_wq_b_proj=None,
        _fp8_q_b_wq_b_scale=None,
    )


def test_wrapper_merge_aliases_q_b_and_leaves_the_skeleton_wq_b_alone(real_members):
    q_b_w, q_b_s, wq_b_w, wq_b_s = real_members
    register = _wrapper_method("_register_q_b_wq_b_merged")
    wrapper = _fake_wrapper(q_b_w.clone(), q_b_s.clone())
    wq_b_param = torch.nn.Parameter(wq_b_w.clone(), requires_grad=False)
    indexer = types.SimpleNamespace(
        wq_b=types.SimpleNamespace(weight=wq_b_param), wq_b_scale=wq_b_s.clone()
    )
    skeleton_ptr = wq_b_param.data_ptr()
    skeleton_scale = indexer.wq_b_scale

    register(wrapper, indexer)

    merged_w, merged_s = wrapper._fp8_q_b_wq_b_proj, wrapper._fp8_q_b_wq_b_scale
    assert tuple(merged_w.shape) == (MERGED_ROWS, Q_LORA_RANK)
    assert tuple(merged_s.shape) == (160, 16)
    # q_b (phase-managed) now lives IN the merged buffer: no second copy.
    q_b_now = wrapper.module.q_b_proj.weight
    assert q_b_now.data_ptr() == merged_w.data_ptr()
    assert tuple(q_b_now.shape) == (Q_B_ROWS, Q_LORA_RANK) and q_b_now.is_contiguous()
    assert wrapper.fp8_q_b_proj.data_ptr() == merged_w.data_ptr()
    scale_now = wrapper.weight_dequant_scale["q_b_proj.weight_scale_inv"]
    assert scale_now.data_ptr() == merged_s.data_ptr()
    assert tuple(scale_now.shape) == (128, 16)
    assert _same_bytes(q_b_now, q_b_w)
    # wq_b (shared device skeleton) is NOT rebound: rebinding would trip
    # _assert_skeleton_still_shared. The merged buffer holds a copy.
    assert wq_b_param.data_ptr() == skeleton_ptr
    assert indexer.wq_b_scale is skeleton_scale
    assert _same_bytes(merged_w[Q_B_ROWS:], wq_b_param.data)


def test_wrapper_merge_skips_an_unregistered_wrapper(real_members):
    q_b_w, q_b_s, wq_b_w, wq_b_s = real_members
    register = _wrapper_method("_register_q_b_wq_b_merged")
    wrapper = _fake_wrapper(q_b_w, q_b_s, registered=False)
    before = wrapper.module.q_b_proj.weight.data_ptr()
    indexer = types.SimpleNamespace(
        wq_b=types.SimpleNamespace(weight=wq_b_w), wq_b_scale=wq_b_s
    )
    register(wrapper, indexer)
    assert wrapper._fp8_q_b_wq_b_proj is None and wrapper._fp8_q_b_wq_b_scale is None
    assert wrapper.module.q_b_proj.weight.data_ptr() == before


def test_wrapper_merge_without_wq_b_scale_fails_loud(real_members):
    q_b_w, q_b_s, wq_b_w, _ = real_members
    register = _wrapper_method("_register_q_b_wq_b_merged")
    wrapper = _fake_wrapper(q_b_w, q_b_s)
    indexer = types.SimpleNamespace(wq_b=types.SimpleNamespace(weight=wq_b_w))
    with pytest.raises(RuntimeError, match="indexer.wq_b_scale"):
        register(wrapper, indexer)


def test_wrapper_builds_the_operand_only_under_the_switch():
    fn = _method(_tree(_WRAPPERS_SRC), "GLM5AttnWrapper", "initialize_fused_kernels")
    calls = _calls(fn, "self._register_q_b_wq_b_merged")
    assert len(calls) == 1
    guard = _if(fn, "glm5_qb_wqb_merge_enabled()")
    assert _calls(_block(guard.body), "self._register_q_b_wq_b_merged")
    # It sits after the no-indexer early return, i.e. only indexer layers.
    early = [
        i for i, stmt in enumerate(fn.body)
        if isinstance(stmt, ast.If)
        and "indexer" in ast.unparse(stmt.test)
        and any(isinstance(s, ast.Return) for s in stmt.body)
    ]
    at = next(i for i, stmt in enumerate(fn.body) if stmt is guard)
    assert early and early[0] < at
    # The WP4 container is still built: prefill and the separate version read it.
    assert "FP8WqbWeightsCUDA(" in ast.unparse(fn)


@pytest.mark.parametrize("method", ("__init__", "_unregister_fp8_weights"))
def test_wrapper_resets_the_merged_operand(method):
    src = ast.unparse(_method(_tree(_WRAPPERS_SRC), "GLM5AttnWrapper", method))
    assert "self._fp8_q_b_wq_b_proj = None" in src
    assert "self._fp8_q_b_wq_b_scale = None" in src


# --------------------------------------------------------------------------- #
# 6. The real RoPE wrapper: the strided entry point reads q in place
# --------------------------------------------------------------------------- #

class _AnyAttr:
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


def _load_fused_indexer_score(monkeypatch):
    """Load the real fused_indexer_score.py with Triton and the CUDA extension
    modules stubbed for its import only; returns (module, recorder)."""
    seen: list[tuple[str, tuple, dict]] = []

    def recorder(name):
        def shim(*args, **kwargs):
            seen.append((name, args, kwargs))
            return args[4] if len(args) > 4 else None
        return shim

    triton = types.ModuleType("triton")
    triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    tl = types.ModuleType("triton.language")
    tl.__getattr__ = lambda name: _AnyAttr()
    triton.language = tl
    topk = types.ModuleType("batchgen_kernels.attention.dsa.fast_topk_cuda")
    topk.fast_topk_2048 = topk.fast_topk_2048_out = recorder("fast_topk")
    kvproj = types.ModuleType("batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda")
    kvproj.build_module = recorder("build_module")
    kvproj.FP8IndexerWeightsCUDA = object
    kvproj._validate_projection_out_buffers = recorder("validate")
    indexer = types.ModuleType("batchgen_kernels.attention.dsa.indexer")
    indexer.fused_rope_hadamard = recorder("fused_rope_hadamard")
    indexer.fused_rope_hadamard_out = recorder("fused_rope_hadamard_out")
    indexer.fused_rope_hadamard_strided_out = recorder("fused_rope_hadamard_strided_out")
    for name, mod in (
        ("triton", triton),
        ("triton.language", tl),
        ("batchgen_kernels.attention.dsa.fast_topk_cuda", topk),
        ("batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda", kvproj),
        ("batchgen_kernels.attention.dsa.indexer", indexer),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("_fused_indexer_score_under_test", _SCORE_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, seen


def _rope_inputs(batch):
    cos = torch.rand(64, 64, dtype=torch.float32)
    sin = torch.rand(64, 64, dtype=torch.float32)
    positions = torch.arange(batch * INDEX_HEADS, dtype=torch.int64) % 64
    out = torch.empty(batch, INDEX_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    out_fp8 = torch.empty(batch, 1, INDEX_HEADS, INDEX_HEAD_DIM, dtype=torch.float8_e4m3fn)
    return cos, sin, positions, out, out_fp8


@pytest.mark.parametrize("batch", (1, 4))
def test_strided_q_reaches_the_kernel_shim_without_a_copy(monkeypatch, batch):
    score, seen = _load_fused_indexer_score(monkeypatch)
    merged = torch.randn(batch, MERGED_ROWS).to(torch.bfloat16)
    q = merged[:, Q_B_ROWS:].unflatten(-1, (INDEX_HEADS, INDEX_HEAD_DIM))
    cos, sin, positions, out, out_fp8 = _rope_inputs(batch)

    score.rope_hadamard_q_out(
        q, cos, sin, positions, out, out_fp8=out_fp8, apply_hadamard=False,
        q_row_stride=q.stride(0),
    )

    assert [name for name, _, _ in seen] == ["fused_rope_hadamard_strided_out"]
    _, args, _ = seen[0]
    x = args[0]
    # The ORIGINAL strided tensor: same bytes, same strides -- no reshape copy.
    assert x is q
    assert x.data_ptr() == merged.data_ptr() + Q_B_ROWS * 2
    assert _strides(x) == (None if batch == 1 else MERGED_ROWS, INDEX_HEAD_DIM, 1)
    assert args[4] is out
    # The declared stride is the tensor's own (for batch 1 torch may report
    # a recomputed singleton stride; the kernel never steps that axis).
    assert args[5] == q.stride(0)
    assert args[7] is out_fp8 and args[8] is False


def test_contiguous_q_keeps_the_original_entry_point(monkeypatch):
    score, seen = _load_fused_indexer_score(monkeypatch)
    batch = 3
    q = torch.randn(batch, INDEX_HEADS, INDEX_HEAD_DIM).to(torch.bfloat16)
    cos, sin, positions, out, _ = _rope_inputs(batch)
    score.rope_hadamard_q_out(q, cos, sin, positions, out)
    assert [name for name, _, _ in seen] == ["fused_rope_hadamard_out"]
    _, args, _ = seen[0]
    assert args[0].data_ptr() == q.data_ptr()
    assert tuple(args[0].shape) == (batch * INDEX_HEADS, INDEX_HEAD_DIM)
    assert args[7] is True  # apply_hadamard default unchanged


def test_strided_q_on_the_contiguous_entry_point_is_a_named_error(monkeypatch):
    score, seen = _load_fused_indexer_score(monkeypatch)
    merged = torch.randn(2, MERGED_ROWS).to(torch.bfloat16)
    q = merged[:, Q_B_ROWS:].unflatten(-1, (INDEX_HEADS, INDEX_HEAD_DIM))
    cos, sin, positions, out, _ = _rope_inputs(2)
    with pytest.raises(score.RopeHadamardStridedInputError, match="not contiguous"):
        score.rope_hadamard_q_out(q, cos, sin, positions, out)
    with pytest.raises(score.RopeHadamardStridedInputError, match="declares q strides"):
        score.rope_hadamard_q_out(q, cos, sin, positions, out, q_row_stride=4096)
    assert seen == []


@pytest.mark.parametrize("apply_hadamard", (False, True))
def test_rope_reference_is_layout_invariant(monkeypatch, apply_hadamard):
    """The in-repo PyTorch reference gives the same values on the strided
    slice and on its contiguous copy (the kernel must too, on the node)."""
    score, _ = _load_fused_indexer_score(monkeypatch)
    batch = 3
    merged = torch.randn(batch, MERGED_ROWS).to(torch.bfloat16)
    q = merged[:, Q_B_ROWS:].unflatten(-1, (INDEX_HEADS, INDEX_HEAD_DIM))
    cos = torch.rand(16, 64, dtype=torch.float32)
    sin = torch.rand(16, 64, dtype=torch.float32)
    pos = torch.tensor([1, 5, 9])
    strided = score.rope_hadamard_q_pytorch(q, cos, sin, pos, apply_hadamard=apply_hadamard)
    dense = score.rope_hadamard_q_pytorch(
        q.contiguous(), cos, sin, pos, apply_hadamard=apply_hadamard
    )
    assert torch.equal(strided, dense)


def test_score_pipeline_skips_its_wq_b_gemm_for_a_precomputed_q(monkeypatch):
    score, _ = _load_fused_indexer_score(monkeypatch)
    batch = 2
    q_raw = torch.randn(batch, INDEX_HEADS, INDEX_HEAD_DIM).to(torch.bfloat16)
    handed = {}

    def no_wq_b(*a, **k):
        raise AssertionError("cuda_wq_b_proj must not run for a precomputed q")

    def rope(q, *a, **k):
        handed["rope_q"] = q
        return q

    monkeypatch.setattr(score, "cuda_wq_b_proj", no_wq_b)
    monkeypatch.setattr(score, "rope_hadamard_q", rope)
    monkeypatch.setattr(score, "compute_head_gates", lambda *a, **k: None)
    monkeypatch.setattr(score, "fused_score_and_topk", lambda *a, **k: "topk")
    top_k, q = score.fused_score_pipeline(
        q_a=torch.zeros(batch, Q_LORA_RANK), hidden_states=None, cached_k=None,
        cache_seqlens=None, wq_b_weights=None, weights_proj_weight=None,
        cos_table=None, sin_table=None, positions=None, module=None,
        q_index_raw=q_raw,
    )
    assert top_k == "topk" and handed["rope_q"] is q_raw and q is q_raw
    with pytest.raises(ValueError, match="q_index_raw must be"):
        score.fused_score_pipeline(
            q_a=torch.zeros(batch, Q_LORA_RANK), hidden_states=None, cached_k=None,
            cache_seqlens=None, wq_b_weights=None, weights_proj_weight=None,
            cos_table=None, sin_table=None, positions=None, module=None,
            q_index_raw=q_raw[:1],
        )


# --------------------------------------------------------------------------- #
# 7. Kernel sources: a NEW strided entry point; the old one is untouched
# --------------------------------------------------------------------------- #

def test_strided_kernel_is_a_separate_instantiation_pair():
    text = (_CSRC / "fused_rope_hadamard.cu").read_text()
    # The original kernel and launch keep their exact signatures.
    assert "void fused_rope_hadamard_kernel(FusedRopeHadamardParams params) {" in text
    assert text.count("fused_rope_hadamard_kernel<true><<<grid, 16, 0, stream>>>(params);") == 1
    assert text.count("fused_rope_hadamard_kernel<false><<<grid, 16, 0, stream>>>(params);") == 1
    # The new one: same rotation switch, strided source addressing.
    assert "fused_rope_hadamard_strided_kernel<true><<<" in text
    assert "fused_rope_hadamard_strided_kernel<false><<<" in text
    assert "x_row * x_row_stride + x_head * 128" in text
    strided = text.split("void fused_rope_hadamard_strided_kernel(", 1)[1]
    assert "if constexpr (kApplyHadamard) {" in strided
    assert "store_scale = params.scale;" in strided.split("if constexpr (kApplyHadamard) {", 1)[1].split("}", 1)[0]


def test_strided_binding_validates_the_layout_it_reads():
    text = (_CSRC / "fused_rope_hadamard_binding.cpp").read_text()
    body = text.split("void fused_rope_hadamard_strided_out(", 1)[1].split("PYBIND11_MODULE", 1)[0]
    for check in (
        "x.stride(2) == 1 && x.stride(1) == 128",
        "x.stride(0) == x_row_stride",
        "x_row_stride % 8 == 0",
        "out.is_contiguous()",
        "positions.numel() == flat_rows",
    ):
        assert check in body, check
    assert "reshape" not in body and "contiguous()" not in body.replace("is_contiguous()", "")
    assert 'm.def("fused_rope_hadamard_strided_out"' in text
    assert 'pybind11::arg("x_row_stride")' in text


def test_python_kernel_wrapper_forwards_the_row_stride():
    fn = _function(_tree(_INDEXER_INIT), "fused_rope_hadamard_strided_out")
    (call,) = _calls(fn, "_fused_rope_hadamard_cuda.fused_rope_hadamard_strided_out")
    assert [ast.unparse(a) for a in call.args] == [
        "x", "cos_cache", "sin_cache", "positions", "out", "x_row_stride",
        "scale", "out_fp8", "apply_hadamard",
    ]


# --------------------------------------------------------------------------- #
# 8. Captured segment: per-layer choice, resolved once, both versions intact
# --------------------------------------------------------------------------- #

def test_switch_is_read_once_in_init_and_never_in_forward():
    tree = _tree(_SEGMENTS_SRC)
    init = _method(tree, "Glm5FullDsaAttnSegment", "__init__")
    assert len(_calls(init, "glm5_qb_wqb_merge_enabled")) == 1
    assert "self._q_b_wq_b_merged = self._q_b_wq_b_merge_selected and (not self.all_short)" in ast.unparse(init)
    for name in ("forward", "setup_static_buffers", "_make_bucket_view_buffers"):
        assert not _calls(_method(tree, "Glm5FullDsaAttnSegment", name), "glm5_qb_wqb_merge_enabled"), name
    # Fail-loud on a missing operand, named.
    guard = _if(init, "self._q_b_wq_b_merge_selected")
    assert "GLM5_QB_WQB_MERGE selects the merged" in ast.unparse(guard)


def test_skip_layers_never_merge():
    """The reuse (skip_topk) segment does not run the parent __init__, never
    reads the switch, and keeps its plain q_b GEMM into its own q_flat."""
    tree = _tree(_REUSE_SRC)
    cls_src = ast.unparse(_class(tree, "Glm5ReuseTopkAttnSegment"))
    assert "glm5_qb_wqb_merge_enabled" not in cls_src
    assert "_q_b_wq_b" not in cls_src
    fwd = _method(tree, "Glm5ReuseTopkAttnSegment", "forward")
    (q_b_gemm,) = [
        c for c in _calls(fwd, "w8a8_deepgemm") if "q_b_proj" in ast.unparse(c)
    ]
    assert ast.unparse(q_b_gemm.args[2]) == "attn.q_b_proj.weight"
    assert {kw.arg: ast.unparse(kw.value) for kw in q_b_gemm.keywords}["out"] == "buffers.q_flat"
    init = _method(tree, "Glm5ReuseTopkAttnSegment", "__init__")
    assert not _calls(init, "super().__init__")


def test_forward_q_b_gemm_branches_on_the_resolved_version():
    fwd = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    branch = _if(fwd, "self._q_b_wq_b_merged")
    merged_src = ast.unparse(_block(branch.body))
    separate_src = ast.unparse(_block(branch.orelse))
    assert "q_b_out = buffers.q_flat_merged" in merged_src
    assert "q_b_weight = self._q_b_wq_b_weight" in merged_src
    # The separate version's operands are the original call's, verbatim.
    assert "q_b_weight = attn.q_b_proj.weight" in separate_src
    assert "q_b_scale = self.wrapper.weight_dequant_scale['q_b_proj.weight_scale_inv']" in separate_src
    assert "q_b_out = buffers.q_flat" in separate_src
    gemms = [c for c in _calls(fwd, "w8a8_deepgemm") if ast.unparse(c.args[2]) == "q_b_weight"]
    assert len(gemms) == 1
    assert ast.unparse(gemms[0].args[3]) == "q_b_scale"
    assert {kw.arg: ast.unparse(kw.value) for kw in gemms[0].keywords}["out"] == "q_b_out"


def test_forward_drops_the_custom_wq_b_only_on_the_merged_version():
    fwd = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    scoring = _if(fwd, "not self.all_short")
    inner = [
        s for s in scoring.body
        if isinstance(s, ast.If) and ast.unparse(s.test) == "self._q_b_wq_b_merged"
    ]
    assert len(inner) == 1
    merged_src = ast.unparse(_block(inner[0].body))
    assert "cuda_wq_b_proj_out" not in merged_src
    assert "q_index_row_stride = q_index_src.stride(0)" in merged_src
    assert "buffers.q_flat_indexer.unflatten(" in merged_src
    separate = _block(inner[0].orelse)
    assert len(_calls(separate, "cuda_wq_b_proj_out")) == 1
    assert "q_index_row_stride = None" in ast.unparse(separate)
    assert len(_calls(fwd, "cuda_wq_b_proj_out")) == 1
    (rope,) = _calls(fwd, "rope_hadamard_q_out")
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in rope.keywords}
    assert kwargs["q_row_stride"] == "q_index_row_stride"
    assert ast.unparse(rope.args[0]) == "q_index_src"


def test_buffer_layout_follows_the_version():
    tree = _tree(_SEGMENTS_SRC)
    setup = _method(tree, "Glm5FullDsaAttnSegment", "setup_static_buffers")
    branch = _if(setup, "self._q_b_wq_b_merged")
    merged_src = ast.unparse(_block(branch.body))
    assert "q_flat_merged = torch.empty(" in merged_src
    assert "_merged_wq_b_scratch_placeholders(device)" in merged_src
    assert "make_fp8_activation_scratch" not in merged_src
    separate_src = ast.unparse(_block(branch.orelse))
    assert "q_flat_merged = None" in separate_src
    assert (
        "q_x_fp8, q_x_scale, q_tma_desc = make_fp8_activation_scratch(bucket_size, "
        "attn.q_lora_rank, self.cuda_module, device=device)"
    ) in separate_src
    src = ast.unparse(setup)
    assert "q_flat_merged=q_flat_merged" in src
    assert "if q_flat_merged is None else q_flat_merged[:, :q_b_cols]" in src
    assert "if q_flat_merged is None else q_flat_merged[:, q_b_cols:]" in src
    # Bucket views follow the BASE set's layout, and a mismatch is named.
    view = _method(tree, "Glm5FullDsaAttnSegment", "_make_bucket_view_buffers")
    assert _calls(view, "_check_q_b_wq_b_layout")
    assert _if(view, "base.q_flat_merged is not None")
    fwd = _method(tree, "Glm5FullDsaAttnSegment", "forward")
    assert _calls(fwd, "_check_q_b_wq_b_layout")
    check = _function(tree, "_check_q_b_wq_b_layout")
    assert "must agree on GLM5_QB_WQB_MERGE" in ast.unparse(check)


# --------------------------------------------------------------------------- #
# 9. Eager mirror
# --------------------------------------------------------------------------- #

def test_eager_selector_mirrors_the_merge():
    tree = _tree(_SELECTOR_SRC)
    operand = ast.unparse(_function(tree, "_q_b_wq_b_merged_operand"))
    # Skip layers (no indexer) and switch-off are the separate version; an
    # indexer layer with the switch on and no operand fails loud.
    assert "if getattr(wrapper.module, 'indexer', None) is None:\n        return None" in operand
    assert "if not glm5_qb_wqb_merge_enabled():\n        return None" in operand
    assert "raise RuntimeError" in operand

    builder = _function(tree, "build_glm5_dsa_flashmla_inputs")
    assert _calls(builder, "_q_b_wq_b_merged_operand")
    branch = _if(builder, "q_b_wq_b is not None")
    merged_src = ast.unparse(_block(branch.body))
    assert "w8a8_deepgemm(q_a_fp8, q_a_scale, *q_b_wq_b)" in merged_src
    assert "q_index_raw = q_merged[:, q_b_cols:].unflatten(" in merged_src
    separate_src = ast.unparse(_block(branch.orelse))
    assert "attn.q_b_proj.weight" in separate_src
    assert "q.view(bsz, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)" in separate_src
    (select,) = _calls(builder, "_select_glm5_dsa_indices")
    assert {kw.arg: ast.unparse(kw.value) for kw in select.keywords}["q_index_raw"] == "q_index_raw"

    selector = _function(tree, "_select_glm5_dsa_indices")
    kwonly = [a.arg for a in selector.args.kwonlyargs]
    assert "q_index_raw" in kwonly
    assert ast.unparse(selector.args.kw_defaults[kwonly.index("q_index_raw")]) == "None"
    # The separate version issues the scorer calls exactly as before (no new
    # keyword), so every existing stub/caller signature still binds.
    src = ast.unparse(selector)
    assert src.count("{} if q_index_raw is None else") == 2


def test_eager_fp8_scorer_skips_its_wq_b_for_a_precomputed_q():
    fn = _method(_tree(_MODEL_SRC), "Glm5Indexer", "_score_and_select_paged_fp8")
    branch = _if(fn, "q_index_raw is not None")
    assert not _calls(_block(branch.body), "cuda_wq_b_proj")
    assert len(_calls(_block(branch.orelse), "cuda_wq_b_proj")) == 1
    assert len(_calls(fn, "cuda_wq_b_proj")) == 1
    dispatch = _method(_tree(_MODEL_SRC), "Glm5Indexer", "score_and_select_paged")
    for callee in ("self._score_and_select_paged_fp8", "fused_score_pipeline"):
        (call,) = _calls(dispatch, callee)
        assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords}["q_index_raw"] == "q_index_raw"
