"""Regression: the attn-residual views must survive a ZERO-row decode batch."""
import ast
from pathlib import Path

MODEL = (Path(__file__).resolve().parents[1] / "batchgen" / "models" / "moonshotai"
         / "kimi_linear" / "model.py")


def _fn(name):
    tree = ast.parse(MODEL.read_text())
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def test_attn_residual_views_do_not_infer_from_empty_batch():
    """`view(batch_size, -1, hidden)` is undefined when the batch has 0 rows.

    Regression for a 512-request decode run that died at 506/512 with
      RuntimeError: cannot reshape tensor of 0 elements into shape
      [0, -1, 7168] because the unspecified dimension size -1 ... is ambiguous
    once the final sequences completed and left an empty decode batch. The
    middle dimension must be explicit. Deliberately NOT fixed with an early
    return: under G>1 every rank must keep reaching the collectives below.
    """
    fn = _fn("_forward_attn_residual")
    bad = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "view" or len(node.args) != 3:
            continue
        mid = node.args[1]
        is_neg_one = (
            isinstance(mid, ast.UnaryOp)
            and isinstance(mid.op, ast.USub)
            and isinstance(mid.operand, ast.Constant)
            and mid.operand.value == 1
        )
        if is_neg_one:
            bad.append(node.lineno)
    assert not bad, (
        f"3-arg view(..., -1, ...) at model.py lines {bad} cannot infer the "
        "middle dimension when the decode batch drains to zero rows; pass the "
        "sequence length explicitly"
    )


def _load_view_helper():
    """Exec ONLY `_view_rows_as_bsh` from model.py.

    Importing the module pulls fla/einops and the whole K3 stack; the helper
    is a pure tensor-shape function, so run its real source in isolation.
    """
    import pytest
    pytest.importorskip("torch")
    import torch

    tree = ast.parse(MODEL.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_view_rows_as_bsh"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), str(MODEL), "exec"), ns)
    return torch, ns["_view_rows_as_bsh"]


def test_empty_decode_batch_view_matches_the_surviving_residual_shape():
    """The empty view must reuse the ENTRY seq_len, not 0.

    Regression for a 512-request decode run that died at 480/512 with
      RuntimeError: output with shape [0, 1, 7168] doesn't match the
      broadcast shape [0, 0, 7168]
    `_forward_attn_residual` keeps `prefix_sum` at the entry shape (B, S, H)
    while re-viewing `hidden_states` through `_view_rows_as_bsh`. Emitting
    (B, 0, H) for the empty batch desynchronises the two, and the in-place
    residual merge `prefix_sum.add_(hidden_states)` then fails to broadcast.
    """
    torch, view_rows_as_bsh = _load_view_helper()

    hidden = 8
    entry = torch.zeros(0, 1, hidden)      # 0 sequences, decode seq-dim 1
    prefix_sum = entry                     # residual that survives the layer
    sharded_rows = torch.zeros(0, hidden)  # _apply_attn_res / scatter_rows out

    viewed = view_rows_as_bsh(
        sharded_rows, entry.shape[0], entry.shape[1], hidden
    )
    assert viewed.shape == prefix_sum.shape, (
        f"empty view {tuple(viewed.shape)} must equal the entry residual "
        f"shape {tuple(prefix_sum.shape)}"
    )
    prefix_sum.add_(viewed)  # the crash site


def test_non_empty_view_still_infers_rows_from_the_tensor():
    """A NON-empty tensor must never take the hoisted seq_len.

    Guards the `b46b4813` regression: `scatter_rows` shards rows across the
    TP group, so the caller's entry seq_len is NOT the row count here. Using
    it broke prefill with "shape '[1, N, H]' is invalid for input of size ...".
    """
    torch, view_rows_as_bsh = _load_view_helper()

    hidden = 8
    # 8 entry rows sharded across a group of 8 -> 1 row on this rank.
    viewed = view_rows_as_bsh(torch.zeros(1, hidden), 1, 8, hidden)
    assert tuple(viewed.shape) == (1, 1, hidden), (
        "row count must come from the tensor, not the entry seq_len"
    )


def test_empty_branch_returns_a_parameter_not_a_constant_middle_dim():
    """The empty branch must forward a parameter, never a literal.

    `4cce16a1` returned `view(batch_size, 0, hidden_size)`, which is a 3-arg
    view whose middle dim is the constant 0 -- invisible to the `-1` check
    above, yet it is exactly what desynchronised the residual pair. Pin the
    defect class directly: the 0-element branch must forward a name.
    """
    fn = _fn("_view_rows_as_bsh")
    empty_branch = next(n for n in fn.body if isinstance(n, ast.If))
    ret = next(n for n in ast.walk(empty_branch) if isinstance(n, ast.Return))
    mid = ret.value.args[1]
    assert isinstance(mid, ast.Name), (
        "the empty-batch view must forward the caller's seq_len parameter; a "
        f"literal middle dim ({ast.dump(mid)}) breaks prefix_sum.add_()"
    )
