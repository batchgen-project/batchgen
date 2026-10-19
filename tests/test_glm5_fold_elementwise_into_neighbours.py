"""CPU contract for folding three GLM-5 decode elementwise ops into neighbours.

The captured decode graph spent one launch per layer on three single-purpose
elementwise kernels, each sitting next to a kernel that can absorb it:

  (a) ``outputs.attn_output.mul_(valid_rows_bf16)`` -- the row-validity mask on
      the attention output, immediately before the decoder layer's residual
      add consumes it.
  (b) ``torch.where(valid_mask, offload_kv, zeros, out=primary_k_tensor)`` --
      immediately after the fused rmsnorm+RoPE kernel that produces
      ``offload_kv``.
  (c) ``residual + mlp_output`` at the end of every layer, immediately before
      the next layer's ``input_layernorm``.

All three are folded here, and the point of this file is that the two
LOAD-BEARING invariants survive the fold:

  * (a) must still zero rows that are inside ``num_valid_tokens`` but have no
    cache slot, and it must still do so BEFORE the MoE router mixes rows. The
    fold moves it into the add+RMSNorm launch that already consumes
    ``attn_output``, which is upstream of the MoE block in the same layer.
  * (b) must remain a SELECT. bf16 ``NaN * 0`` is ``NaN``, so a masked-out row
    has to receive the literal zero, not the product of a stale row and zero.
    The fold keeps a ``tl.where`` inside the producing kernel.

Both invariants are pinned behaviourally where CPU-testable and structurally
where the kernel itself is CUDA-only. The structural half is not decoration: a
numeric test alone would still pass if a fold had been reverted into separate
launches, which is the entire quantity being optimized.

The two attention-segment modules cannot be imported on a CPU box (FA3 /
triton), so they are asserted on with ``ast`` -- the same approach as
``test_glm5_decode_row_mask_contract.py`` and
``test_glm5_row_validity_prologue_hoist.py``. The layer segment and the
whole-model driver DO import on CPU and are tested by running them.
"""

import ast
import types
from pathlib import Path

import pytest
import torch

from batchgen.models.glm.glm5.layer_cuda_graph_segments import (
    Glm5DecoderLayerGraphSegment,
)


_GLM5 = Path(__file__).parents[1] / "batchgen" / "models" / "glm" / "glm5"
ATTENTION_SEGMENTS = ("cuda_graph_segments.py", "reuse_topk_segment.py")
HIDDEN = 32
KV_DIM = 576


# ---------------------------------------------------------------------------
# ast helpers
# ---------------------------------------------------------------------------


def _tree(name):
    return ast.parse((_GLM5 / name).read_text())


def _calls(tree, func_source):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == func_source
    ]


def _attr_calls(tree, attr):
    return [
        (ast.unparse(node.func.value), node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def _kwargs(call):
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def _enclosing_if_tests(tree, target):
    """Source of every ``if`` test whose body (transitively) contains target."""
    tests = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if any(inner is target for stmt in node.body for inner in ast.walk(stmt)):
            tests.append(ast.unparse(node.test))
    return tests


# ---------------------------------------------------------------------------
# (b) the primary_k_tensor select moved INTO the producing kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("segment", ATTENTION_SEGMENTS)
def test_primary_k_select_is_no_longer_a_separate_launch(segment):
    tree = _tree(segment)
    where_outs = [
        _kwargs(call).get("out")
        for call in _calls(tree, "torch.where")
    ]
    offenders = [out for out in where_outs if out and "primary_k_tensor" in out]
    assert not offenders, (
        f"{segment}: the primary_k_tensor gate is folded into the fused "
        f"rmsnorm+RoPE kernel that produces the value; a standalone "
        f"torch.where is the launch this change removes, got {offenders}"
    )


@pytest.mark.parametrize("segment", ATTENTION_SEGMENTS)
def test_primary_k_is_written_by_the_producer_under_a_select_mask(segment):
    tree = _tree(segment)
    producers = _calls(tree, "_fused_rmsnorm_rope")
    assert producers, f"{segment}: expected a fused rmsnorm+RoPE call"
    for call in producers:
        kw = _kwargs(call)
        assert "out" in kw and "primary_k_tensor" in kw["out"], (
            f"{segment}: the producer must write straight into the static "
            f"primary_k_tensor output, got out={kw.get('out')!r}"
        )
        # The mask must be the BOOLEAN row-validity mask, which the kernel
        # consumes as a tl.where predicate. Handing it the bf16 1.0/0.0 mask
        # would invite a future multiply, and bf16 NaN * 0 is NaN.
        assert kw.get("row_valid_mask") == "buffers.valid_mask", (
            f"{segment}: primary_k_tensor's gate must stay a select driven by "
            f"the boolean valid_mask, got row_valid_mask="
            f"{kw.get('row_valid_mask')!r}"
        )


@pytest.mark.parametrize("segment", ATTENTION_SEGMENTS)
def test_primary_k_is_never_written_by_a_plain_copy_or_a_multiply(segment):
    tree = _tree(segment)
    for attr in ("copy_", "mul_"):
        targets = [t for t, _ in _attr_calls(tree, attr)]
        assert "outputs.primary_k_tensor" not in targets, (
            f"{segment}: primary_k_tensor is staged for host KV offload, so a "
            f"stale row's bytes would leave the segment through it. A plain "
            f"{attr} cannot gate it and a follow-up multiply cannot clean it "
            f"(bf16 NaN * 0 is NaN)."
        )


# ---------------------------------------------------------------------------
# (a) the attn_output mask moved INTO the residual add
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("segment", ATTENTION_SEGMENTS)
def test_attn_output_mask_is_still_applied_when_nobody_absorbs_it(segment):
    """The standalone per-layer DSA graph has no consumer inside the segment."""
    tree = _tree(segment)
    masked = [
        (target, node)
        for target, node in _attr_calls(tree, "mul_")
        if "valid_rows_bf16" in ast.unparse(node)
    ]
    assert [target for target, _ in masked] == ["outputs.attn_output"], (
        f"{segment}: the row mask must still exist for callers that do not "
        f"absorb it, and only on the attn_output that leaves the segment"
    )
    tests = _enclosing_if_tests(tree, masked[0][1])
    assert "not attn_output_mask_deferred" in tests, (
        f"{segment}: the mask must be skipped ONLY under an explicit caller "
        f"declaration that it was absorbed; got guards {tests}"
    )


@pytest.mark.parametrize("segment", ATTENTION_SEGMENTS)
def test_segment_exposes_the_mask_it_lets_a_caller_defer(segment):
    tree = _tree(segment)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    args = {
        arg.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
        for arg in list(node.args.args) + list(node.args.kwonlyargs)
    }
    assert "attn_output_mask_deferred" in args, (
        f"{segment}: forward must take the deferral declaration explicitly; a "
        f"caller that silently drops the mask would feed stale rows to the "
        f"MoE router"
    )
    if segment == "cuda_graph_segments.py":
        assert "attn_output_row_mask" in names, (
            "the full segment must publish the mask a deferring caller has to "
            "apply, otherwise 'deferred' means 'lost'"
        )


# ---------------------------------------------------------------------------
# Row-select reference: what the folded kernels must compute
# ---------------------------------------------------------------------------


def _add_rmsnorm_ref(residual, hidden, weight, eps, row_valid_mask=None):
    """Reference for the folded add+RMSNorm, mirroring rmsnorm.cu.

    The gate is a SELECT on the loaded value, so a masked row contributes an
    exact 0.0 regardless of what ``hidden`` holds. The reduction uses the
    unrounded fp32 sum while the normalized numerator re-reads the rounded
    residual, which is what the CUDA kernel documents.
    """
    r = residual.to(torch.float32)
    h = hidden.to(torch.float32)
    if row_valid_mask is not None:
        keep = (row_valid_mask.to(torch.float32) != 0.0).view(-1, 1)
        h = torch.where(keep, h, torch.zeros_like(h))
    s = r + h
    rounded = s.to(residual.dtype)
    var = (s * s).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    normed = rounded.to(torch.float32) * inv_rms * weight.to(torch.float32)
    return normed.to(residual.dtype), rounded


def test_masked_row_leaves_the_residual_untouched():
    torch.manual_seed(0)
    residual = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    hidden = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    weight = torch.ones(HIDDEN, dtype=torch.bfloat16)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.bfloat16)

    before = residual.clone()
    _, updated = _add_rmsnorm_ref(residual, hidden, weight, 1e-5, mask)

    assert torch.equal(updated[1], before[1])
    assert torch.equal(updated[3], before[3])
    assert not torch.equal(updated[0], before[0])


def test_masked_row_is_inert_even_when_hidden_is_non_finite():
    """The select-not-multiply invariant, stated as a number.

    This is the property the fold must not lose: once the upstream per-op
    masks are gone, a row with no cache slot can carry non-finite bytes, and
    ``NaN * 0`` is ``NaN``.
    """
    residual = torch.ones(2, HIDDEN, dtype=torch.bfloat16)
    hidden = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    hidden[1, 0] = float("nan")
    hidden[1, 1] = float("inf")
    weight = torch.ones(HIDDEN, dtype=torch.bfloat16)
    mask = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)

    normed, updated = _add_rmsnorm_ref(residual, hidden, weight, 1e-5, mask)
    assert torch.isfinite(normed).all()
    assert torch.equal(updated[1], torch.ones(HIDDEN, dtype=torch.bfloat16))

    # The same input through a MULTIPLY gate, which is what the invariant
    # forbids: the masked row is poisoned.
    poisoned = (hidden.to(torch.float32) * mask.to(torch.float32).view(-1, 1))
    assert not torch.isfinite(poisoned[1]).all()


def test_mask_of_all_ones_is_the_unmasked_result():
    torch.manual_seed(1)
    residual = torch.randn(3, HIDDEN, dtype=torch.bfloat16)
    hidden = torch.randn(3, HIDDEN, dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN, dtype=torch.bfloat16)
    ones = torch.ones(3, dtype=torch.bfloat16)

    a = _add_rmsnorm_ref(residual.clone(), hidden, weight, 1e-5, None)
    b = _add_rmsnorm_ref(residual.clone(), hidden, weight, 1e-5, ones)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


# ---------------------------------------------------------------------------
# (c) the residual add folded into the next layer's input_layernorm
# ---------------------------------------------------------------------------


class _RecordingNorm:
    def __init__(self, tag, log):
        self.tag = tag
        self.log = log
        self.weight = torch.full((HIDDEN,), 1.0, dtype=torch.bfloat16)
        self.eps = 1e-5

    def __call__(self, x):
        self.log.append(("rmsnorm", self.tag))
        return x.clone()


class _FakeDsaSegment:
    def __init__(self, log):
        self.log = log
        self.max_seqlen = 8
        self.index_topk = 4
        self.primary_blocked_k = torch.zeros(1, 1, 1, KV_DIM)
        self.aux_blocked_k = None
        self.calls = []
        self.mask = None

    def attn_output_row_mask(self, bucket_size):
        if self.mask is None:
            self.mask = torch.ones(bucket_size, dtype=torch.bfloat16)
        return self.mask

    def forward(self, **kwargs):
        self.log.append(("dsa", kwargs["attn_output_mask_deferred"]))
        self.calls.append(kwargs)
        batch = kwargs["hidden_states"].shape[0]
        return {
            "attn_output": torch.full(
                (batch, 1, HIDDEN), 0.5, dtype=torch.bfloat16
            ),
            "primary_k_tensor": torch.zeros(batch, 1, 1, KV_DIM),
        }


class _FakeMoeSegment:
    """Writes into ONE shared buffer, like the real per-bucket MoE pool."""

    def __init__(self, log, buffer):
        self.log = log
        self.buffer = buffer
        self.generation = 0

    def forward(self, *, padded, rank_token_counts):
        self.generation += 1
        self.log.append(("moe_write", self.generation))
        self.buffer.fill_(float(self.generation))
        return {"moe_output": self.buffer}


def _make_layer(log, *, moe_buffer=None, layer_idx=0):
    norm_in = _RecordingNorm("input", log)
    norm_post = _RecordingNorm("post_attn", log)
    layer = types.SimpleNamespace(
        layer_idx=layer_idx,
        hidden_size=HIDDEN,
        input_layernorm=norm_in,
        post_attention_layernorm=norm_post,
        mlp=lambda x: torch.full_like(x, 0.25),
    )
    dsa = _FakeDsaSegment(log)
    moe = None if moe_buffer is None else _FakeMoeSegment(log, moe_buffer)
    segment = Glm5DecoderLayerGraphSegment(
        layer=layer,
        dsa_segment=dsa,
        moe_segment=moe,
        device=torch.device("cpu"),
        world_size=1,
    )
    return segment, dsa, moe


def _fake_add_rmsnorm(log):
    def impl(residual, hidden, weight, eps, num_valid_tokens=None, row_valid_mask=None):
        log.append(("add_rmsnorm", row_valid_mask is not None))
        normed, updated = _add_rmsnorm_ref(
            residual.reshape(-1, HIDDEN),
            hidden.reshape(-1, HIDDEN),
            weight,
            eps,
            row_valid_mask,
        )
        residual.reshape(-1, HIDDEN).copy_(updated)
        return normed.reshape(residual.shape), residual

    return impl


@pytest.fixture
def add_rmsnorm_log(monkeypatch):
    log = []
    import batchgen.attention.fused_kernels as fused

    monkeypatch.setattr(fused, "cuda_add_rmsnorm", _fake_add_rmsnorm(log))
    return log


def _forward_kwargs(batch):
    return dict(
        position_ids=torch.zeros(batch, 1, dtype=torch.int64),
        cache_seqlens=torch.ones(batch, dtype=torch.int32),
        primary_slot_indices=torch.zeros(batch, dtype=torch.int32),
        aux_slot_indices=torch.zeros(batch, dtype=torch.int32),
        num_valid_tokens=torch.tensor([batch], dtype=torch.int32),
        rank_token_counts=torch.tensor([batch], dtype=torch.int64),
    )


def test_first_layer_normalizes_without_an_add(add_rmsnorm_log):
    segment, dsa, _ = _make_layer(add_rmsnorm_log)
    hidden = torch.zeros(2, 1, HIDDEN, dtype=torch.bfloat16)

    segment.forward(
        hidden_states=hidden,
        residual=None,
        defer_residual_add=True,
        **_forward_kwargs(2),
    )

    # Exactly one plain rmsnorm (the input norm) and one add+rmsnorm (post-attn).
    assert ("rmsnorm", "input") in add_rmsnorm_log
    assert ("rmsnorm", "post_attn") not in add_rmsnorm_log
    assert [e for e in add_rmsnorm_log if e[0] == "add_rmsnorm"] == [
        ("add_rmsnorm", True)
    ]


def test_chained_layer_folds_the_previous_add_into_its_input_norm(add_rmsnorm_log):
    segment, _, _ = _make_layer(add_rmsnorm_log)
    carried = torch.full((2, 1, HIDDEN), 2.0, dtype=torch.bfloat16)
    previous_mlp = torch.full((2, 1, HIDDEN), 3.0, dtype=torch.bfloat16)

    out = segment.forward(
        hidden_states=previous_mlp,
        residual=carried,
        defer_residual_add=True,
        **_forward_kwargs(2),
    )

    # No plain input rmsnorm survives: the add absorbed it.
    assert ("rmsnorm", "input") not in add_rmsnorm_log
    adds = [e for e in add_rmsnorm_log if e[0] == "add_rmsnorm"]
    assert adds == [("add_rmsnorm", False), ("add_rmsnorm", True)], (
        "the chained layer must run exactly two add+rmsnorms: the folded input "
        "norm (unmasked) and the post-attention norm (masked)"
    )
    # The carried residual really absorbed the previous layer's mlp_output.
    assert torch.equal(
        carried.reshape(-1, HIDDEN)[0, 0],
        torch.tensor(5.5, dtype=torch.bfloat16),
    ), "2.0 + 3.0 (input fold) + 0.5 (attn_output) = 5.5"
    assert out["residual"] is carried


def test_dsa_mask_is_deferred_and_handed_to_the_add(add_rmsnorm_log):
    segment, dsa, _ = _make_layer(add_rmsnorm_log)
    segment.forward(
        hidden_states=torch.zeros(3, 1, HIDDEN, dtype=torch.bfloat16),
        residual=None,
        defer_residual_add=True,
        **_forward_kwargs(3),
    )
    assert ("dsa", True) in add_rmsnorm_log, (
        "the layer absorbs the mask, so the segment must be told to skip it"
    )
    assert ("add_rmsnorm", True) in add_rmsnorm_log, (
        "and the absorbing launch must actually receive a row mask -- "
        "'deferred' without a mask argument is a dropped invariant"
    )
    assert dsa.mask is not None and dsa.mask.numel() == 3


def test_residual_without_deferral_is_a_named_error(add_rmsnorm_log):
    segment, _, _ = _make_layer(add_rmsnorm_log)
    with pytest.raises(ValueError, match="one convention for the whole layer chain"):
        segment.forward(
            hidden_states=torch.zeros(2, 1, HIDDEN, dtype=torch.bfloat16),
            residual=torch.zeros(2, 1, HIDDEN, dtype=torch.bfloat16),
            defer_residual_add=False,
            **_forward_kwargs(2),
        )


def test_deferred_output_is_the_unsummed_pair(add_rmsnorm_log):
    buffer = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    segment, _, moe = _make_layer(add_rmsnorm_log, moe_buffer=buffer)
    out = segment.forward(
        hidden_states=torch.zeros(2, 1, HIDDEN, dtype=torch.bfloat16),
        residual=None,
        defer_residual_add=True,
        **_forward_kwargs(2),
    )
    assert set(out) >= {"hidden_states", "residual", "primary_k_tensor"}
    # hidden_states IS the mlp output, not a sum: that is the removed launch.
    assert out["hidden_states"].data_ptr() == buffer.data_ptr()


def test_materialized_and_deferred_chains_agree(add_rmsnorm_log):
    """The fold is an associativity move, so the two conventions must match."""
    kwargs = _forward_kwargs(2)

    seg_a, _, _ = _make_layer(add_rmsnorm_log)
    hidden = torch.full((2, 1, HIDDEN), 1.0, dtype=torch.bfloat16)
    legacy = seg_a.forward(
        hidden_states=hidden.clone(), residual=None, defer_residual_add=False, **kwargs
    )["hidden_states"]

    seg_b, _, _ = _make_layer(add_rmsnorm_log)
    deferred = seg_b.forward(
        hidden_states=hidden.clone(), residual=None, defer_residual_add=True, **kwargs
    )
    folded = deferred["residual"] + deferred["hidden_states"]

    assert torch.equal(legacy, folded)


def test_previous_mlp_output_is_read_before_the_next_moe_overwrites_it(add_rmsnorm_log):
    """Pins the aliasing argument the deferral relies on.

    ``mlp_output`` is the MoE pool's shared buffer, so the next layer's MoE
    block overwrites it. That is safe only because the next layer's folded
    input add READS it to produce the very input that MoE block consumes --
    a true data dependency, not launch-order luck. If a future change hoisted
    MoE work above the input norm, this ordering would break silently.
    """
    buffer = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    kwargs = _forward_kwargs(2)

    seg0, _, _ = _make_layer(add_rmsnorm_log, moe_buffer=buffer, layer_idx=0)
    seg1, _, _ = _make_layer(add_rmsnorm_log, moe_buffer=buffer, layer_idx=1)

    out0 = seg0.forward(
        hidden_states=torch.zeros(2, 1, HIDDEN, dtype=torch.bfloat16),
        residual=None,
        defer_residual_add=True,
        **kwargs,
    )
    marker = len(add_rmsnorm_log)
    seg1.forward(
        hidden_states=out0["hidden_states"],
        residual=out0["residual"],
        defer_residual_add=True,
        **kwargs,
    )

    tail = add_rmsnorm_log[marker:]
    first_add = next(i for i, e in enumerate(tail) if e[0] == "add_rmsnorm")
    first_moe = next(i for i, e in enumerate(tail) if e[0] == "moe_write")
    assert first_add < first_moe, (
        f"the folded add must consume the previous layer's mlp_output before "
        f"this layer's MoE block overwrites the shared buffer, got {tail}"
    )


# ---------------------------------------------------------------------------
# Driver: exactly one standalone add survives
# ---------------------------------------------------------------------------


def test_driver_defers_every_layer_and_materializes_one_add():
    source = (_GLM5 / "whole_model_cuda_graph_segments.py").read_text()
    tree = ast.parse(source)

    forwards = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "layer_segment.forward"
    ]
    assert len(forwards) == 1
    kw = _kwargs(forwards[0])
    assert kw.get("defer_residual_add") == "True"
    assert kw.get("residual") == "residual"

    # One add outside the (debug-only) probe branch.
    run = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_model_with_probes"
    )
    adds = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and "residual" in ast.unparse(node)
    ]
    assert len(adds) == 2, (
        "expected exactly two residual adds in the driver: the surviving "
        "final add and the debug probe's materialization, got "
        f"{[ast.unparse(node) for node in adds]}"
    )


# ---------------------------------------------------------------------------
# Kernel-side gates. These MUST run on the node; they are the only evidence
# that the folded CUDA / triton paths compute what the references above say.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_add_rmsnorm_row_mask_matches_the_reference():
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm

    torch.manual_seed(7)
    rows, hidden = 8, 6144
    residual = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    hidden_t = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
    mask = torch.tensor(
        [1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.bfloat16, device="cuda"
    )

    want_normed, want_residual = _add_rmsnorm_ref(
        residual.clone(), hidden_t, weight, 1e-5, mask
    )
    got_normed, got_residual = cuda_add_rmsnorm(
        residual.clone(), hidden_t, weight, 1e-5, None, mask
    )
    assert torch.equal(got_residual, want_residual.to("cuda"))
    assert torch.equal(got_normed, want_normed.to("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_add_rmsnorm_masked_row_rejects_non_finite_hidden():
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm

    rows, hidden = 4, 6144
    residual = torch.ones(rows, hidden, dtype=torch.bfloat16, device="cuda")
    hidden_t = torch.zeros(rows, hidden, dtype=torch.bfloat16, device="cuda")
    hidden_t[1].fill_(float("nan"))
    hidden_t[3].fill_(float("inf"))
    weight = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")
    mask = torch.tensor([1, 0, 1, 0], dtype=torch.bfloat16, device="cuda")

    normed, updated = cuda_add_rmsnorm(
        residual.clone(), hidden_t, weight, 1e-5, None, mask
    )
    assert torch.isfinite(normed).all(), (
        "a masked row must contribute an exact 0, not NaN * 0"
    )
    assert torch.equal(updated[1], torch.ones(hidden, dtype=torch.bfloat16, device="cuda"))
    assert torch.equal(updated[3], torch.ones(hidden, dtype=torch.bfloat16, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_add_rmsnorm_without_mask_is_unchanged():
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm

    torch.manual_seed(11)
    rows, hidden = 6, 6144
    residual = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    hidden_t = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
    ones = torch.ones(rows, dtype=torch.bfloat16, device="cuda")

    bare = cuda_add_rmsnorm(residual.clone(), hidden_t, weight, 1e-5)
    ones_masked = cuda_add_rmsnorm(
        residual.clone(), hidden_t, weight, 1e-5, None, ones
    )
    assert torch.equal(bare[0], ones_masked[0])
    assert torch.equal(bare[1], ones_masked[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_add_rmsnorm_rejects_a_mismatched_mask():
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm

    rows, hidden = 4, 6144
    residual = torch.zeros(rows, hidden, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")
    bad = torch.ones(rows + 1, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="one entry per row"):
        cuda_add_rmsnorm(residual, residual.clone(), weight, 1e-5, None, bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_rmsnorm_rope_out_and_mask_match_the_two_step_form():
    """The (b) fold must be bit-identical to produce-then-select."""
    from batchgen_kernels.triton.fused_rmsnorm_rope import (
        fused_rmsnorm_rope_with_q_native,
    )

    torch.manual_seed(3)
    bsz, heads, lora, rope = 8, 4, 512, 64
    total = lora + rope
    kv = torch.randn(bsz, 1, total, dtype=torch.bfloat16, device="cuda")
    kv[2].fill_(float("nan"))  # an invalid row with non-finite bytes
    q_pe = torch.randn(bsz, heads, 1, rope, dtype=torch.bfloat16, device="cuda")
    cos = torch.randn(4096, rope, dtype=torch.bfloat16, device="cuda")
    sin = torch.randn(4096, rope, dtype=torch.bfloat16, device="cuda")
    pos = torch.zeros(bsz, 1, dtype=torch.int64, device="cuda")
    w = torch.randn(lora, dtype=torch.bfloat16, device="cuda")
    mask = torch.tensor(
        [1, 1, 0, 1, 0, 1, 1, 0], dtype=torch.bool, device="cuda"
    )

    unmasked = fused_rmsnorm_rope_with_q_native(
        kv, q_pe.clone(), cos, sin, pos, w, lora, rope
    )
    want = torch.where(
        mask.view(bsz, 1, 1), unmasked.view(bsz, 1, -1), torch.zeros_like(unmasked)
    )

    got = torch.empty(bsz, 1, total, dtype=torch.bfloat16, device="cuda")
    fused_rmsnorm_rope_with_q_native(
        kv, q_pe.clone(), cos, sin, pos, w, lora, rope, out=got, row_valid_mask=mask
    )
    assert torch.equal(got, want)
    assert torch.isfinite(got[2]).all(), "the gate must write exact zeros"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_rmsnorm_rope_mask_requires_out():
    from batchgen_kernels.triton.fused_rmsnorm_rope import (
        fused_rmsnorm_rope_with_q_native,
    )

    bsz, heads, lora, rope = 2, 2, 512, 64
    kv = torch.zeros(bsz, 1, lora + rope, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.zeros(bsz, heads, 1, rope, dtype=torch.bfloat16, device="cuda")
    cos = torch.zeros(16, rope, dtype=torch.bfloat16, device="cuda")
    sin = torch.zeros(16, rope, dtype=torch.bfloat16, device="cuda")
    pos = torch.zeros(bsz, 1, dtype=torch.int64, device="cuda")
    w = torch.ones(lora, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="row_valid_mask requires out"):
        fused_rmsnorm_rope_with_q_native(
            kv,
            q_pe,
            cos,
            sin,
            pos,
            w,
            lora,
            rope,
            row_valid_mask=torch.ones(bsz, dtype=torch.bool, device="cuda"),
        )
