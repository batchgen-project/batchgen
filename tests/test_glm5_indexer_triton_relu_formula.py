"""Pin the bf16 Triton indexer scorers to the GLM-5 reference formula.

GLM-5 and GLM-5.1 keep a bf16 AoS indexer cache, so their decode selection runs
the Triton scorers in ``batchgen_kernels/attention/dsa/fused_indexer_score.py``
instead of ``deep_gemm.fp8_paged_mqa_logits``. Both must compute the HF
reference aggregate ``agg[t] = sum_h g_h * relu(q_h . k_t)`` (masked ``-inf``
past the row's length). Before this test the Triton scorers summed
``g_h * (q_h . k_t)`` without the per-head ReLU.

The dev machine has no Triton, so the kernels are executed by a small CPU
interpreter for the subset of ``triton.language`` they use: every
``@triton.jit`` function in the module is compiled from source with ``tl``
bound to the shim below and launched over its grid. That runs the real kernel
bodies (pointer arithmetic, masks, the head loop), so a formula change in any
of the three scorers is caught here without a GPU. The GPU twin of this check
is ``tests/gpu/test_glm5_indexer_triton_relu_formula_gpu.py``.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCORER_SRC = (
    _REPO_ROOT / "batchgen_kernels" / "attention" / "dsa" / "fused_indexer_score.py"
)
_SCORER_KERNELS = (
    "_fused_score_kernel",
    "_fused_paged_score_kernel",
    "_fused_paged_score_with_slots_kernel",
)


# --------------------------------------------------------------------------
# Minimal CPU interpreter for the triton.language subset the scorers use
# --------------------------------------------------------------------------


class _Ptr:
    """A flat tensor plus an element offset (int or index tensor)."""

    def __init__(self, flat: torch.Tensor, off=0):
        self.flat = flat
        self.off = off

    def __add__(self, other):
        return _Ptr(self.flat, self.off + other)

    __radd__ = __add__


def _as_index(off) -> torch.Tensor:
    return torch.as_tensor(off, dtype=torch.long)


class _TL(types.SimpleNamespace):
    float32 = torch.float32
    int64 = torch.int64
    int32 = torch.int32
    constexpr = object

    def __init__(self):
        super().__init__()
        self.pid = (0, 0)

    def program_id(self, axis):
        return self.pid[axis]

    @staticmethod
    def arange(start, end):
        return torch.arange(start, end)

    @staticmethod
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def sum(x, axis):
        return x.sum(dim=axis)

    @staticmethod
    def where(cond, a, b):
        return torch.where(torch.as_tensor(cond, dtype=torch.bool), a, b)

    @staticmethod
    def maximum(a, b):
        a = torch.as_tensor(a)
        return torch.maximum(a, torch.as_tensor(b, dtype=a.dtype))

    @staticmethod
    def minimum(a, b):
        a = torch.as_tensor(a)
        return torch.minimum(a, torch.as_tensor(b, dtype=a.dtype))

    @staticmethod
    def load(ptr: _Ptr, mask=None, other=None):
        off = _as_index(ptr.off)
        if mask is None:
            return ptr.flat[off]
        off, mask = torch.broadcast_tensors(off, torch.as_tensor(mask, dtype=torch.bool))
        values = ptr.flat[torch.where(mask, off, 0)]
        fill = torch.as_tensor(0 if other is None else other, dtype=values.dtype)
        return torch.where(mask, values, fill)

    @staticmethod
    def store(ptr: _Ptr, value, mask=None):
        off = _as_index(ptr.off)
        value = torch.as_tensor(value)
        if mask is None:
            mask = torch.ones((), dtype=torch.bool)
        off, value, mask = torch.broadcast_tensors(
            off, value, torch.as_tensor(mask, dtype=torch.bool)
        )
        ptr.flat[off[mask]] = value[mask].to(ptr.flat.dtype)


def _is_triton_jit(node: ast.FunctionDef) -> bool:
    return any(ast.unparse(d) == "triton.jit" for d in node.decorator_list)


def _load_kernels():
    """Compile every ``@triton.jit`` function of the scorer module on the shim."""

    tree = ast.parse(_SCORER_SRC.read_text(encoding="utf-8"))
    jitted = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and _is_triton_jit(node)
    ]
    for node in jitted:
        node.decorator_list = []
    module = ast.Module(body=jitted, type_ignores=[])
    tl = _TL()
    namespace = {"tl": tl, "float": float, "range": range}
    exec(compile(module, str(_SCORER_SRC), "exec"), namespace)
    missing = [name for name in _SCORER_KERNELS if name not in namespace]
    assert not missing, f"scorer kernels not found: {missing}"
    return tl, namespace


def _launch(tl: _TL, kernel, grid, *args, **kwargs):
    for pid_b in range(grid[1]):
        for pid_s in range(grid[0]):
            tl.pid = (pid_s, pid_b)
            kernel(*args, **kwargs)


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------


def _reference(q, k_rows, gates, lens, *, relu: bool) -> torch.Tensor:
    """fp64 ``sum_h g_h * relu(q_h . k_t)`` per row, ``-inf`` past ``lens``.

    ``k_rows`` is the per-row dense K ``[B, max_seqlen, D]`` the paged layouts
    resolve to.
    """

    B, _, _ = q.shape
    max_seqlen = k_rows.shape[1]
    out = torch.full((B, max_seqlen), float("-inf"), dtype=torch.float64)
    for b in range(B):
        n = int(lens[b])
        per_head = q[b].double() @ k_rows[b, :n].double().T  # [H, n]
        if relu:
            per_head = torch.relu(per_head)
        out[b, :n] = (gates[b].double()[:, None] * per_head).sum(dim=0)
    return out


def _inputs(seed: int, B: int, H: int, D: int, max_seqlen: int):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, D, generator=gen).to(torch.bfloat16)
    k = torch.randn(B, max_seqlen, D, generator=gen).to(torch.bfloat16)
    # Real head gates are signed (``weights_proj`` output times positive scales).
    gates = torch.randn(B, H, generator=gen, dtype=torch.float32)
    return q, k, gates


def _assert_matches_relu_reference(agg, q, k_rows, gates, lens):
    expected = _reference(q, k_rows, gates, lens, relu=True)
    no_relu = _reference(q, k_rows, gates, lens, relu=False)
    finite = torch.isfinite(expected)
    # The inputs must discriminate the two formulas, otherwise this test could
    # not fail on the retired no-ReLU kernels.
    assert (expected[finite] - no_relu[finite]).abs().max() > 1.0
    assert torch.equal(torch.isfinite(agg), finite)
    torch.testing.assert_close(
        agg.double()[finite], expected[finite], rtol=1e-4, atol=1e-4
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_dense_scorer_kernel_computes_the_relu_gated_formula():
    tl, ns = _load_kernels()
    B, H, D, max_seqlen, block_s = 3, 4, 16, 40, 16
    q, k, gates = _inputs(0, B, H, D, max_seqlen)
    lens = torch.tensor([40, 23, 1], dtype=torch.int32)
    agg = torch.empty(B, max_seqlen, dtype=torch.float32)

    _launch(
        tl, ns["_fused_score_kernel"], ((max_seqlen + block_s - 1) // block_s, B),
        _Ptr(q.reshape(-1)), _Ptr(k.reshape(-1)), _Ptr(gates.reshape(-1)),
        _Ptr(lens), _Ptr(agg.view(-1)), max_seqlen, B,
        n_heads=H, head_dim=D, BLOCK_S=block_s, BLOCK_D=D,
    )

    _assert_matches_relu_reference(agg, q, k, gates, lens)


def _paged_layout(k_rows: torch.Tensor, page_size: int, seed: int):
    """Scatter dense per-row K into a shuffled paged cache + block table."""

    B, max_seqlen, D = k_rows.shape
    pages_per_seq = (max_seqlen + page_size - 1) // page_size
    num_pages = B * pages_per_seq + 3
    gen = torch.Generator().manual_seed(seed)
    physical = torch.randperm(num_pages, generator=gen)[: B * pages_per_seq]
    block_table = physical.view(B, pages_per_seq).to(torch.int32)
    cache = torch.zeros(num_pages, page_size, 1, D, dtype=k_rows.dtype)
    for b in range(B):
        for p in range(pages_per_seq):
            chunk = k_rows[b, p * page_size:(p + 1) * page_size]
            cache[int(block_table[b, p]), : chunk.shape[0], 0] = chunk
    return cache, block_table, pages_per_seq


def test_paged_scorer_kernel_computes_the_relu_gated_formula():
    tl, ns = _load_kernels()
    B, H, D, max_seqlen, page_size, block_s = 2, 4, 16, 40, 8, 16
    q, k, gates = _inputs(1, B, H, D, max_seqlen)
    lens = torch.tensor([37, 40], dtype=torch.int32)
    cache, block_table, pages_per_seq = _paged_layout(k, page_size, seed=11)
    agg = torch.empty(B, max_seqlen, dtype=torch.float32)

    _launch(
        tl, ns["_fused_paged_score_kernel"], ((max_seqlen + block_s - 1) // block_s, B),
        _Ptr(q.reshape(-1)), _Ptr(cache.reshape(-1)), _Ptr(block_table.reshape(-1)),
        _Ptr(gates.reshape(-1)), _Ptr(lens), _Ptr(agg.view(-1)), max_seqlen, B,
        n_heads=H, head_dim=D, page_size=page_size,
        max_pages_per_seq=pages_per_seq, BLOCK_S=block_s, BLOCK_D=D,
    )

    _assert_matches_relu_reference(agg, q, k, gates, lens)


@pytest.mark.parametrize("with_valid_tokens", [False, True])
def test_graph_slot_scorer_kernel_computes_the_relu_gated_formula(with_valid_tokens):
    """The captured decode graph path (``fused_paged_score_and_topk_with_slots_out``)."""

    tl, ns = _load_kernels()
    B, H, D, max_seqlen, page_size, block_s = 3, 4, 16, 40, 8, 16
    q, k, gates = _inputs(2, B, H, D, max_seqlen)
    lens = torch.tensor([33, 40, 12], dtype=torch.int32)
    cache, block_table, pages_per_seq = _paged_layout(k, page_size, seed=12)
    # Slot table holds the rows in reverse order plus an unused slot.
    slot_table = torch.full((B + 1, pages_per_seq), -1, dtype=torch.int32)
    slot_indices = torch.tensor([3, 1, 0], dtype=torch.int32)
    for b in range(B):
        slot_table[int(slot_indices[b])] = block_table[b]
    num_valid = torch.tensor([B], dtype=torch.int32)
    agg = torch.empty(B, max_seqlen, dtype=torch.float32)

    _launch(
        tl, ns["_fused_paged_score_with_slots_kernel"],
        ((max_seqlen + block_s - 1) // block_s, B),
        _Ptr(q.reshape(-1)), _Ptr(cache.reshape(-1)), _Ptr(slot_table.reshape(-1)),
        _Ptr(slot_indices), _Ptr(num_valid), _Ptr(gates.reshape(-1)), _Ptr(lens),
        _Ptr(agg.view(-1)), max_seqlen,
        topk=8, skip_dense_prefix=False, B=B, n_heads=H, head_dim=D,
        page_size=page_size, max_pages_per_seq=pages_per_seq,
        HAS_VALID_TOKENS=with_valid_tokens, BLOCK_S=block_s, BLOCK_D=D,
    )

    _assert_matches_relu_reference(agg, q, k, gates, lens)


def test_all_scorer_kernels_share_one_head_formula():
    """One formula, one place: every scorer accumulates through the helper and
    no scorer multiplies a head gate itself."""

    tree = ast.parse(_SCORER_SRC.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    helper = functions["_relu_gated_head_sum"]
    assert "tl.maximum(tl.sum(k_tile * q_vec[None, :], axis=1), 0.0)" in ast.unparse(helper)
    for name in _SCORER_KERNELS:
        src = ast.unparse(functions[name])
        assert "_relu_gated_head_sum(" in src, name
        assert "GATES_ptr + gates_base + h" not in src, name


def test_shim_detects_the_retired_no_relu_formula():
    """Self-check of the interpreter: the same harness on a no-ReLU copy of the
    helper must FAIL the reference comparison."""

    tl, ns = _load_kernels()
    retired = ast.parse(_SCORER_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(retired):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "tl.maximum"
            and ast.unparse(node.args[0]).startswith("tl.sum(k_tile")
        ):
            node.func, node.args = ast.Name("_identity", ast.Load()), [node.args[0]]
    jitted = [
        node for node in retired.body
        if isinstance(node, ast.FunctionDef) and _is_triton_jit(node)
    ]
    for node in jitted:
        node.decorator_list = []
    namespace = {"tl": tl, "float": float, "_identity": lambda x: x}
    exec(compile(ast.fix_missing_locations(ast.Module(jitted, [])), "retired", "exec"), namespace)

    B, H, D, max_seqlen, block_s = 2, 4, 16, 32, 16
    q, k, gates = _inputs(3, B, H, D, max_seqlen)
    lens = torch.tensor([32, 20], dtype=torch.int32)
    agg = torch.empty(B, max_seqlen, dtype=torch.float32)
    _launch(
        tl, namespace["_fused_score_kernel"], (2, B),
        _Ptr(q.reshape(-1)), _Ptr(k.reshape(-1)), _Ptr(gates.reshape(-1)),
        _Ptr(lens), _Ptr(agg.view(-1)), max_seqlen, B,
        n_heads=H, head_dim=D, BLOCK_S=block_s, BLOCK_D=D,
    )
    with pytest.raises(AssertionError):
        _assert_matches_relu_reference(agg, q, k, gates, lens)
