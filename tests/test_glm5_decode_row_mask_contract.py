"""CPU contract for GLM-5 decode row masking (invalid / padded bucket rows).

A captured decode graph always runs the full bucket, so every segment carries
rows that are not live tokens: rows at or beyond ``num_valid_tokens``, and rows
inside it whose sequence has no cache slot. Each masked DeepGEMM leaves the
first kind unwritten (stale), and computes the second kind normally.

The invariant this file pins is WHERE those rows have to be neutralized:

  * Inside the attention segment nothing has to be masked. Every op between the
    GEMMs is row-local (fused_rmsnorm, act_quant, RoPE, the absorb GEMVs, FA3),
    every FP8 activation quantization hard-zeroes rows beyond
    ``num_valid_tokens`` before a GEMM can see them, and every write to state
    shared across rows is already gated on the same row validity: KV-cache
    writes return on ``slot < 0`` and on the row count, the indexer scorer and
    top-k return on the row count, and the selected-page-table transform stores
    ``selected_lengths = 0`` for an invalid row.
  * ``attn_output`` must be masked. It leaves the segment into the decoder
    layer's residual add and from there into the MoE router, whose gate top-k
    and ragged dispatch mix rows.
  * ``primary_k_tensor`` also leaves the segment (the whole-model graph stages
    it for host KV offload) and must be written by a SELECT, not a multiply:
    bf16 ``NaN * 0`` is ``NaN``, and after the upstream masks are gone a stale
    row can be non-finite.
  * The MoE segment's rank padding is the gate kernel's epilogue, not a
    gather + compare + two ``torch.where`` launches.

These are source-level assertions on purpose: the property is about which
launches exist in the captured graph, so a test that only checks numbers would
still pass if a well-meaning change put a full-buffer mask back.
"""

import ast
from pathlib import Path

import pytest


_GLM5 = Path(__file__).parents[1] / "batchgen" / "models" / "glm" / "glm5"
ATTENTION_SEGMENTS = ("cuda_graph_segments.py", "reuse_topk_segment.py")


def _tree(name):
    return ast.parse((_GLM5 / name).read_text())


def _calls(tree, attr):
    """Every ``<something>.<attr>(...)`` call, as (target_source, node)."""
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
        ):
            out.append((ast.unparse(node.func.value), node))
    return out


def _torch_where_out_targets(tree):
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) != "torch.where":
            continue
        for kw in node.keywords:
            if kw.arg == "out":
                targets.append(ast.unparse(kw.value))
    return targets


@pytest.mark.parametrize("segment", ATTENTION_SEGMENTS)
def test_attn_output_is_the_only_valid_row_mask(segment):
    tree = _tree(segment)
    masked = [
        target
        for target, node in _calls(tree, "mul_")
        if "valid_rows_bf16" in ast.unparse(node)
    ]
    assert masked == ["outputs.attn_output"], (
        f"{segment}: the valid-row mask must be applied exactly once, on the "
        f"attn_output that leaves the segment, got {masked}. Masking an "
        "intermediate buffer costs one launch per layer per step and is "
        "redundant: the consumers are row-local and the shared-state writes "
        "are already row-gated."
    )


@pytest.mark.parametrize("segment", ATTENTION_SEGMENTS)
def test_primary_k_tensor_is_written_by_a_masked_select(segment):
    tree = _tree(segment)
    assert "outputs.primary_k_tensor" in _torch_where_out_targets(tree), (
        f"{segment}: primary_k_tensor is staged for host KV offload, so an "
        "invalid row's bytes leave the segment through it; it must be written "
        "by torch.where(valid_mask, ..., zero) rather than copied."
    )
    copied = [t for t, _ in _calls(tree, "copy_")]
    assert "outputs.primary_k_tensor" not in copied, (
        f"{segment}: a plain copy_ into primary_k_tensor would carry a stale "
        "row through, and a follow-up multiply cannot clean it (bf16 NaN*0)."
    )


@pytest.mark.parametrize("segment", ATTENTION_SEGMENTS)
def test_no_full_buffer_mask_on_intermediate_activations(segment):
    tree = _tree(segment)
    for target, node in _calls(tree, "mul_"):
        if "valid_rows_bf16" not in ast.unparse(node):
            continue
        assert target == "outputs.attn_output", (
            f"{segment}: {target} is an intermediate activation; its invalid "
            "rows are neutralized by the num_valid_tokens gate in its "
            "consumer, so masking the whole buffer here is a per-layer launch "
            "for nothing."
        )


def _identifiers(tree):
    """Every name, attribute and string literal the module actually uses.

    Comments are deliberately excluded: the point is what the module does, and
    a comment that says what it no longer does is not a violation.
    """
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            seen.add(node.value)
        elif isinstance(node, ast.arg):
            seen.add(node.arg)
    return seen


def test_moe_rank_padding_is_folded_into_the_gate():
    tree = ast.parse((_GLM5 / "moe_cuda_graph_segments.py").read_text())

    gate_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "gate_sigmoid_topk_cuda"
    ]
    assert len(gate_calls) == 1
    passed = {kw.arg for kw in gate_calls[0].keywords}
    assert {"rank_token_counts", "rows_per_rank"} <= passed, (
        "the MoE graph must hand the gate its EP padding shape so padding rows "
        f"come out as expert -1 / weight 0 directly, got {sorted(passed)}"
    )

    # The four launches that shape replaces must be gone, buffers included.
    used = _identifiers(tree)
    for dead in (
        "topk_masked_indices",
        "topk_masked_weights",
        "topk_negative_ones",
        "topk_zero_weights",
        "rank_ids",
        "local_pos",
    ):
        assert dead not in used, (
            f"{dead} belongs to the separate-launch rank mask the gate epilogue "
            "replaced; leaving it allocates graph-resident memory for nothing"
        )
    assert not _torch_where_out_targets(tree), (
        "the MoE graph segment should run no torch.where masking launches"
    )


def test_moe_dispatch_consumes_the_gate_output_directly():
    tree = ast.parse((_GLM5 / "moe_cuda_graph_segments.py").read_text())
    dispatch = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "dispatch_scatter_ragged"
    ]
    assert len(dispatch) == 1
    # Second positional argument is the routed expert-id table.
    assert ast.unparse(dispatch[0].args[1]) == "bufs.topk_indices"
