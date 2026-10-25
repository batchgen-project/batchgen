"""Source contract for the DSA indexer K path running on a side stream.

``Glm5FullDsaAttnSegment.forward`` forks the indexer K path (wk quant + GEMM,
k LayerNorm, K RoPE+Hadamard, aux K write, head gates) onto a side stream and
joins it before the first main-stream reader. These tests parse the source, so
they run on CPU-only machines where the segment module cannot be imported
(FA3 / Triton / DeepGEMM); the value-level race test and the stream-resource
tests need the real module and live with the GPU tier.
"""

import ast
import pathlib

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GLM5 = _REPO_ROOT / "batchgen/models/glm/glm5"
_SEGMENTS_SRC = _GLM5 / "cuda_graph_segments.py"
_REUSE_SRC = _GLM5 / "reuse_topk_segment.py"

# The K path, exactly: every kernel it issues, by call name. The wk
# projection appears twice because both of its versions live in the tree and
# stay selectable (GLM5_INDEXER_WK_BACKEND); both are issued on the side
# stream, and exactly one of them runs per capture.
_K_PATH_CALLS = (
    "cuda_wk_proj_gemm_only_out",
    "indexer_wk_proj_deepgemm_out",
    "indexer.k_norm",
    "indexer._fused_rope_hadamard_or_fallback",
    "write_aux_indexer_k",
    "write_aux_indexer_k_norm_rope",
    "outputs.indexer_k_tensor.copy_",
    "head_gates_out",
)
# Main-stream work that must NOT be pulled onto the side stream.
_MAIN_PATH_CALLS = (
    "act_quant",
    "w8a8_deepgemm",
    "fused_rmsnorm",
    "_fused_rmsnorm_rope",
    "run_paged_kv_token_update_fused",
    "cuda_wq_b_proj_out",
    "rope_hadamard_q_out",
    "self._score_and_topk",
    "absorb_q_out",
    "transform_selected_positions_out",
    "self._run_selected_fa3",
    "self._run_all_short_fa3",
    "absorb_out_out",
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


def _forward():
    return _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")


def _fork_block(fwd):
    blocks = [
        n for n in ast.walk(fwd)
        if isinstance(n, ast.With)
        and len(n.items) == 1
        and ast.unparse(n.items[0].context_expr) == "k_path.fork()"
    ]
    assert len(blocks) == 1, f"expected exactly one `with k_path.fork():`, got {len(blocks)}"
    return blocks[0]


def _top_level_index(fwd, node):
    for i, stmt in enumerate(fwd.body):
        if any(sub is node for sub in ast.walk(stmt)):
            return i
    raise AssertionError("node not in forward body")


# --------------------------------------------------------------------------- #
# The wk-projection backend switch (kernel version management)
# --------------------------------------------------------------------------- #

# The switch, read once in __init__ and never re-evaluated per step.
_SWITCH = "self._indexer_wk_deepgemm"
_OLD_SWITCH = "not self._indexer_wk_deepgemm"

# The layer's blocked-FP8 cast of its hidden states. Source-identical in both
# positions: the hoist moves WHERE it is issued, never what it computes.
_HIDDEN_QUANT_SRC = (
    "act_quant(hidden_flat, num_valid_tokens=num_valid_tokens, "
    "scale_tma_aligned=num_valid_tokens is not None)"
)

# The pre-existing wk call, byte for byte (d1cc70e9). The `fused_cuda` backend
# must keep issuing exactly this, with exactly these operands.
_OLD_WK_CALL_SRC = (
    "cuda_wk_proj_gemm_only_out(hidden_flat, self.wrapper._indexer_cuda_weights, "
    "self.cuda_module, buffers.indexer_k_x_fp8, buffers.indexer_k_x_scale, "
    "buffers.indexer_k_tma_desc, buffers.indexer_k_raw, "
    "num_valid_tokens=num_valid_tokens)"
)

# The DeepGEMM version: the shared hoisted activation in, the SAME static
# output buffer the `_out` variant above writes out.
_NEW_WK_CALL_SRC = (
    "indexer_wk_proj_deepgemm_out(hidden_fp8, hidden_scale, "
    "self._indexer_wk_dg_weights, buffers.indexer_k_raw)"
)


def _hidden_quantize_calls(fwd):
    """The layer's quantize of ``hidden_flat``, in program order.

    Deliberately matched on the full call source: ``forward`` also quantizes
    ``q_a_normed`` and ``attn_heads_flat``, and those are not hoistable.
    """
    return [c for c in _calls(fwd, "act_quant")
            if ast.unparse(c) == _HIDDEN_QUANT_SRC]


def _hoisted_quantize_calls(fwd):
    fork_at = _top_level_index(fwd, _fork_block(fwd))
    return [c for c in _hidden_quantize_calls(fwd)
            if _top_level_index(fwd, c) < fork_at]


def _guarded_by(fwd, node, test_src):
    """True when ``node`` sits under an ``if <test_src>:`` inside ``forward``."""
    for n in ast.walk(fwd):
        if not isinstance(n, ast.If) or ast.unparse(n.test) != test_src:
            continue
        if any(sub is node for stmt in n.body for sub in ast.walk(stmt)):
            return True
    return False


def test_the_shared_quantize_is_hoisted_above_the_fork_for_the_deepgemm_backend():
    """One quantize of the hidden states, issued in one of two positions.

    On the DeepGEMM wk version the side stream reads its output, so it has to
    be written on the main stream BEFORE the fork's ``wait_stream`` -- the same
    reason ``token_indices`` is written before the fork. On the `fused_cuda`
    version the side stream quantizes for itself, so this one stays exactly
    where it was, after the fork.
    """
    fwd = _forward()
    quantizes = _hidden_quantize_calls(fwd)
    assert len(quantizes) == 2, (
        "expected exactly two positions for the hidden quantize (hoisted for "
        f"the DeepGEMM wk version, unhoisted for the custom one), got {len(quantizes)}"
    )
    fork_at = _top_level_index(fwd, _fork_block(fwd))
    hoisted, unhoisted = quantizes
    assert _top_level_index(fwd, hoisted) < fork_at, "the hoisted quantize is not above the fork"
    assert _top_level_index(fwd, unhoisted) > fork_at, "the unhoisted quantize is not below the fork"
    # Coherent gating: each position belongs to exactly one backend, so no
    # build issues the quantize twice and none issues it zero times.
    assert _guarded_by(fwd, hoisted, _SWITCH)
    assert _guarded_by(fwd, unhoisted, _OLD_SWITCH)
    # Both write the same two names, which is what the wk GEMM and the qkv_a
    # GEMM then read.
    for call in quantizes:
        assign = next(
            n for n in ast.walk(fwd)
            if isinstance(n, ast.Assign) and any(sub is call for sub in ast.walk(n))
        )
        assert [ast.unparse(t) for t in assign.targets] == ["(hidden_fp8, hidden_scale)"]


def test_both_wk_versions_are_on_the_side_stream_and_switch_coherently():
    """The switch selects the version INSIDE the fork, so neither version can
    drift onto the main stream and the two cannot both run."""
    fwd = _forward()
    fork = _fork_block(fwd)
    switches = [
        n for n in ast.walk(fork)
        if isinstance(n, ast.If) and ast.unparse(n.test) == _SWITCH
    ]
    assert len(switches) == 1, f"expected one wk-version switch in the fork, got {len(switches)}"
    switch = switches[0]
    new_calls = _calls(switch, "indexer_wk_proj_deepgemm_out")
    old_calls = _calls(switch, "cuda_wk_proj_gemm_only_out")
    assert len(new_calls) == 1 and len(old_calls) == 1
    assert any(sub is new_calls[0] for stmt in switch.body for sub in ast.walk(stmt)), (
        "the DeepGEMM wk version must be the switch's taken branch"
    )
    assert any(sub is old_calls[0] for stmt in switch.orelse for sub in ast.walk(stmt)), (
        "the custom wk kernel must be the switch's fallback branch"
    )
    assert ast.unparse(new_calls[0]) == _NEW_WK_CALL_SRC


def test_the_custom_wk_version_keeps_its_original_call_and_order():
    """``fused_cuda`` must be the pre-change path: same kernel, same operands,
    same stream, and the quantize back where it was (immediately before the
    qkv_a GEMM that consumes it)."""
    fwd = _forward()
    old_calls = _calls(fwd, "cuda_wk_proj_gemm_only_out")
    assert len(old_calls) == 1
    assert ast.unparse(old_calls[0]) == _OLD_WK_CALL_SRC
    # The custom kernel still quantizes for itself: it reads the raw hidden,
    # never the hoisted FP8 pair.
    old_src = ast.unparse(old_calls[0])
    assert "hidden_fp8" not in old_src and "hidden_scale" not in old_src
    # Unhoisted quantize -> qkv_a GEMM, adjacent and in that order, as before.
    _, unhoisted = _hidden_quantize_calls(fwd)
    quant_at = _top_level_index(fwd, unhoisted)
    qkv_a = _calls(fwd, "w8a8_deepgemm")[0]
    assert _top_level_index(fwd, qkv_a) == quant_at + 1


def test_the_wk_backend_is_resolved_once_at_construction():
    """A per-step decision cannot be captured: the graph records whichever
    kernels the capturing forward issued."""
    tree = _tree(_SEGMENTS_SRC)
    init = ast.unparse(_method(tree, "Glm5FullDsaAttnSegment", "__init__"))
    assert "self._indexer_wk_deepgemm = glm5_indexer_wk_is_deepgemm()" in init
    assert "_indexer_wk_deepgemm_weights" in init, (
        "__init__ must bind (and fail loud on) the DeepGEMM wk weights"
    )
    fwd = ast.unparse(_forward())
    assert "glm5_indexer_wk_is_deepgemm" not in fwd, (
        "forward must read the resolved attribute, not re-evaluate the switch"
    )
    assert "GLM5_INDEXER_WK_BACKEND" not in fwd


def test_every_k_path_kernel_is_issued_inside_the_fork_and_nowhere_else():
    fwd = _forward()
    fork = _fork_block(fwd)
    for name in _K_PATH_CALLS:
        inside, total = len(_calls(fork, name)), len(_calls(fwd, name))
        assert inside >= 1, f"{name} is not on the side stream"
        assert inside == total, f"{name} is also issued outside the fork"
    # One writer per aux-cache format, both on the side stream: a page-split
    # FP8 cache takes the fused norm+rope+quantize+write launch, a bf16 AoS
    # cache the LayerNorm/RoPE pair plus the plain paged update.
    assert len(_calls(fork, "write_aux_indexer_k_norm_rope")) == 1
    assert len(_calls(fork, "write_aux_indexer_k")) == 1


def test_no_main_path_kernel_is_pulled_onto_the_side_stream():
    fork = _fork_block(_forward())
    for name in _MAIN_PATH_CALLS:
        assert not _calls(fork, name), f"{name} was moved onto the side stream"


def test_the_fork_precedes_every_main_stream_launch():
    """The side stream gets the whole Q path to overlap with, and everything
    it reads was issued on the main stream before ``wait_stream``.

    The one main-stream launch allowed to precede the fork is the HOISTED
    blocked-FP8 quantize of ``hidden_flat`` -- the side stream's wk GEMM reads
    its output, so like ``token_indices`` its write has to be on the main
    stream's timeline when ``wait_stream`` runs. It is pinned separately by
    ``test_the_shared_quantize_is_hoisted_above_the_fork_for_the_deepgemm_backend``.
    """
    fwd = _forward()
    fork_at = _top_level_index(fwd, _fork_block(fwd))
    hoisted = _hoisted_quantize_calls(fwd)
    for name in _MAIN_PATH_CALLS:
        for call in _calls(fwd, name):
            if any(call is h for h in hoisted):
                continue
            assert _top_level_index(fwd, call) > fork_at, f"{name} precedes the fork"
    # token_indices is read by both streams, so its VALUE must already exist
    # when the side stream's wait_stream runs. It is now a step-level static
    # the prologue writes before the layer loop (compute_step_statics), so
    # this forward only binds the buffer -- once, and before the fork, which
    # keeps the binding on the main stream's timeline and leaves no
    # per-layer narrowing launch anywhere in the method.
    assigns = [
        (i, stmt) for i, stmt in enumerate(fwd.body)
        if isinstance(stmt, ast.Assign)
        and [ast.unparse(t) for t in stmt.targets] == ["token_indices"]
    ]
    assert len(assigns) == 1
    assert assigns[0][0] < fork_at
    assert ast.unparse(assigns[0][1]) == "token_indices = buffers.token_indices"
    assert "position_ids.view(batch_size).to(" not in ast.unparse(fwd), (
        "the int32 narrowing of position_ids is step-level; a per-layer "
        "launch must not come back into forward"
    )
    hidden = [
        i for i, stmt in enumerate(fwd.body)
        if isinstance(stmt, ast.Assign)
        and [ast.unparse(t) for t in stmt.targets] == ["hidden_flat"]
    ]
    assert hidden and hidden[0] < fork_at


def _guard(fwd, test_src):
    found = [
        n for n in fwd.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == test_src
    ]
    return found


def _stmt_index_of_call(body, name):
    for i, stmt in enumerate(body):
        if _calls(stmt, name):
            return i
    raise AssertionError(f"no {name} in block")


def test_scoring_graphs_join_after_the_q_path_and_before_the_scorer():
    fwd = _forward()
    guards = _guard(fwd, "not self.all_short")
    assert len(guards) == 1, "expected one top-level `if not self.all_short:`"
    body = guards[0].body
    join_at = _stmt_index_of_call(body, "k_path.join")
    assert _stmt_index_of_call(body, "rope_hadamard_q_out") < join_at
    assert _stmt_index_of_call(body, "cuda_wq_b_proj_out") < join_at
    assert join_at < _stmt_index_of_call(body, "self._score_and_topk")


def test_all_short_graphs_join_once_at_the_end_of_the_segment():
    fwd = _forward()
    tail_guards = [
        g for g in _guard(fwd, "self.all_short") if _calls(g, "k_path.join")
    ]
    assert len(tail_guards) == 1
    at = fwd.body.index(tail_guards[0])
    assert not tail_guards[0].orelse
    # After the o_proj GEMM (the last main-stream kernel) and before the
    # outputs are handed out.
    last_gemm = max(_top_level_index(fwd, c) for c in _calls(fwd, "w8a8_deepgemm"))
    assert last_gemm < at
    outputs_at = next(
        i for i, stmt in enumerate(fwd.body)
        if isinstance(stmt, ast.Assign)
        and [ast.unparse(t) for t in stmt.targets] == ["graph_outputs"]
    )
    assert at < outputs_at


def test_exactly_one_join_is_reachable_per_forward():
    fwd = _forward()
    joins = _calls(fwd, "k_path.join")
    assert len(joins) == 2
    scoring = _guard(fwd, "not self.all_short")[0]
    short = [g for g in _guard(fwd, "self.all_short") if _calls(g, "k_path.join")][0]
    assert len(_calls(scoring, "k_path.join")) == 1
    assert len(_calls(short, "k_path.join")) == 1
    # The K path's own head-gate guard lives inside the fork and holds no join.
    fork = _fork_block(fwd)
    assert not _calls(fork, "k_path.join")


def test_the_overlap_is_built_at_construction_not_per_forward():
    tree = _tree(_SEGMENTS_SRC)
    init = ast.unparse(_method(tree, "Glm5FullDsaAttnSegment", "__init__"))
    assert "_IndexerKPathOverlap(" in init
    assert "_InlineIndexerKPath()" in init
    fwd = ast.unparse(_forward())
    for needle in ("torch.cuda.Stream(", "torch.cuda.Event(", "_IndexerKPathOverlap("):
        assert needle not in fwd, needle
    assert "k_path = self._indexer_k_overlap" in fwd


def test_the_reuse_segment_has_no_k_path_to_fork():
    """Skip layers have no indexer: nothing to overlap, and the reuse segment
    deliberately does not run the parent constructor."""
    reuse = ast.unparse(_method(_tree(_REUSE_SRC), "Glm5ReuseTopkAttnSegment", "forward"))
    assert "k_path" not in reuse
    assert "_indexer_k_overlap" not in reuse
    for name in _K_PATH_CALLS:
        assert f"{name}(" not in reuse, name
    # No wk projection of either version => nothing to hoist a shared quantize
    # above, so the wk backend switch does not reach this segment at all. Its
    # own quantize of ``hidden_flat`` feeds only its qkv_a GEMM, on the main
    # stream, and stays where it is under both backends.
    assert "_indexer_wk_deepgemm" not in reuse
    assert "GLM5_INDEXER_WK_BACKEND" not in reuse
    reuse_quantizes = _hidden_quantize_calls(
        _method(_tree(_REUSE_SRC), "Glm5ReuseTopkAttnSegment", "forward")
    )
    assert len(reuse_quantizes) == 1, (
        "the reuse segment has one unconditional hidden quantize; a second "
        "position would mean the wk switch leaked into a skip layer"
    )


# --------------------------------------------------------------------------- #
# Stream / event resources (need the importable module)
# --------------------------------------------------------------------------- #

try:
    from batchgen.models.glm.glm5 import cuda_graph_segments as segments

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - CPU dev boxes have no FA3/Triton
    segments = None
    _IMPORT_ERROR = exc

requires_module = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"segment module needs its GPU-only dependencies ({_IMPORT_ERROR})",
)


class _FakeStream:
    def __init__(self, device):
        self.device = device


def _fake_cuda(monkeypatch, *, capturing=False, current_device=0):
    monkeypatch.setattr(segments, "_INDEXER_K_STREAMS", {})
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **kw: object())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)


@requires_module
def test_the_side_stream_is_one_process_resource_per_device(monkeypatch):
    _fake_cuda(monkeypatch)
    first = segments._glm5_indexer_k_stream(torch.device("cuda", 0))
    assert segments._glm5_indexer_k_stream(torch.device("cuda", 0)) is first
    assert segments._glm5_indexer_k_stream(torch.device("cuda")) is first
    assert segments._glm5_indexer_k_stream(torch.device("cuda", 1)) is not first


@requires_module
def test_the_side_stream_refuses_a_non_cuda_device(monkeypatch):
    _fake_cuda(monkeypatch)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        segments._glm5_indexer_k_stream(torch.device("cpu"))


@requires_module
def test_the_overlap_resources_are_never_minted_inside_a_capture(monkeypatch):
    _fake_cuda(monkeypatch, capturing=True)
    with pytest.raises(RuntimeError, match="must exist before CUDA"):
        segments._glm5_indexer_k_stream(torch.device("cuda", 0))
    _fake_cuda(monkeypatch, capturing=False)
    segments._glm5_indexer_k_stream(torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="must be created before CUDA"):
        segments._IndexerKPathOverlap(torch.device("cuda", 0), 3)


@requires_module
def test_the_indexer_stream_is_not_the_moe_side_stream(monkeypatch):
    """Separate resources: the MoE shared-expert fork keeps its own stream."""
    from batchgen.models.glm.glm5 import moe_cuda_graph_segments as moe

    _fake_cuda(monkeypatch)
    monkeypatch.setattr(moe, "_SIDE_STREAMS", {})
    assert segments._glm5_indexer_k_stream(torch.device("cuda", 0)) is not (
        moe._glm5_moe_side_stream(torch.device("cuda", 0))
    )
