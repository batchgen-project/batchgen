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

# The K path, exactly: every kernel it issues, by call name.
_K_PATH_CALLS = (
    "cuda_wk_proj_gemm_only_out",
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
    "fp8_q_absorb_out",
    "transform_selected_positions_out",
    "self._run_selected_fa3",
    "self._run_all_short_fa3",
    "fp8_out_absorb_out",
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
    it reads was issued on the main stream before ``wait_stream``."""
    fwd = _forward()
    fork_at = _top_level_index(fwd, _fork_block(fwd))
    for name in _MAIN_PATH_CALLS:
        for call in _calls(fwd, name):
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
