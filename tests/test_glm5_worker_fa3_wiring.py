"""CPU contract tests for the worker side of the GLM-5 direct-index FA3 graph.

The worker cannot be imported on a CPU dev box (Triton/FA3/DeepGEMM), so these
tests parse ``batchgen_worker.py`` instead. They pin:

* the budget-based ``all_short`` graph-lifetime decision and its wiring into
  both DSA segment constructors;
* the ``all_short_exceeded`` eager return in ``_glm5_whole_graph_path_state``;
* the removal of the dead FlashMLA scheduler metadata from the whole graph
  chain (worker, whole-model / decoder-layer segments, adapter, reuse
  segment, and the full-DSA segment class).

``_glm5_dsa_graph_required_tokens`` is additionally exercised as a unit by
compiling its AST against stub globals and calling it with a stub ``self``.
"""

import ast
import math
import pathlib
import types
import typing

import pytest

from batchgen.sequence import SequenceStatus


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_WORKER_SRC = _REPO_ROOT / "batchgen/batchgen_worker.py"
_GLM5_DIR = _REPO_ROOT / "batchgen/models/glm/glm5"
_SEGMENTS_SRC = _GLM5_DIR / "cuda_graph_segments.py"

# Files in which the FlashMLA graph metadata must be gone entirely.
_FLASHMLA_FREE_FILES = (
    _WORKER_SRC,
    _GLM5_DIR / "whole_model_cuda_graph_segments.py",
    _GLM5_DIR / "layer_cuda_graph_segments.py",
    _GLM5_DIR / "cuda_graph_adapter.py",
    _GLM5_DIR / "reuse_topk_segment.py",
)


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level class `{name}`")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            return stmt
    raise AssertionError(f"no `{cls.name}.{name}`")


def _worker_method(name: str) -> ast.FunctionDef:
    return _method(_class(_tree(_WORKER_SRC), "BatchGenWorker"), name)


def _calls_to(node: ast.AST, callee: str) -> list[ast.Call]:
    return [
        sub
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == callee
    ]


def _kwargs(call: ast.Call) -> dict[str, str]:
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def _function_containing_call(callee: str) -> ast.FunctionDef:
    cls = _class(_tree(_WORKER_SRC), "BatchGenWorker")
    owners = [
        stmt
        for stmt in cls.body
        if isinstance(stmt, ast.FunctionDef) and _calls_to(stmt, callee)
    ]
    assert len(owners) == 1, [fn.name for fn in owners]
    return owners[0]


# --------------------------------------------------------------------------- #
# all_short decision + constructor wiring
# --------------------------------------------------------------------------- #

def test_required_tokens_helper_signature():
    fn = _worker_method("_glm5_dsa_graph_required_tokens")
    assert [a.arg for a in fn.args.args] == ["self", "active_sequence_ids"]
    assert [a.arg for a in fn.args.kwonlyargs] == ["page_size"]


def test_build_block_derives_all_short_from_required_tokens():
    build = _function_containing_call("self._glm5_dsa_graph_score_capacity_tokens")
    src = ast.unparse(build)

    (required_call,) = _calls_to(build, "self._glm5_dsa_graph_required_tokens")
    assert [ast.unparse(a) for a in required_call.args] == ["active_sequence_ids"]
    assert _kwargs(required_call) == {"page_size": "primary_page_size"}

    assert "required_seqlen = self._glm5_dsa_graph_required_tokens(" in src
    # index_topk must come from the live indexer module (the value the
    # scoring kernels use), never from model_config: a local-checkpoint
    # config may not carry the field.
    assert "index_topk_cfg = int(getattr(_first_indexer, 'index_topk', 0) or 0)" in src
    assert "self._glm5_whole_model_index_topk = index_topk_cfg" in src
    assert "model_config, 'index_topk'" not in src
    assert (
        "all_short = bool(index_topk_cfg) and required_seqlen <= index_topk_cfg"
        in src
    )
    assert "self._glm5_whole_model_all_short = all_short" in src
    # graph_max_seqlen keeps the capacity-based sizing.
    assert "graph_max_seqlen = int(capacity_seqlen)" in src


@pytest.mark.parametrize("ctor", ["Glm5FullDsaAttnSegment", "Glm5ReuseTopkAttnSegment"])
def test_both_segment_constructors_receive_all_short(ctor):
    build = _function_containing_call("self._glm5_dsa_graph_score_capacity_tokens")
    calls = _calls_to(build, ctor)
    assert len(calls) == 1, ctor
    assert _kwargs(calls[0]).get("all_short") == "all_short"


def test_all_short_flag_is_reset_with_the_whole_model_segment():
    tree = _tree(_WORKER_SRC)
    resets = 0
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for body in (
            getattr(node, field)
            for node in ast.walk(fn)
            for field in ("body", "orelse", "handlers", "finalbody")
            if isinstance(getattr(node, field, None), list)
        ):
            for idx, stmt in enumerate(body):
                if ast.unparse(stmt) != "self._whole_model_segment = None":
                    continue
                following = [ast.unparse(s) for s in body[idx + 1 : idx + 2]]
                assert following == ["self._glm5_whole_model_all_short = False"], (
                    fn.name,
                    following,
                )
                resets += 1
    # Release-state helper, capture-OOM rollback, and model unload.
    assert resets >= 3


# --------------------------------------------------------------------------- #
# Path state: an all_short graph must not serve rows past index_topk
# --------------------------------------------------------------------------- #

def test_whole_graph_path_state_has_all_short_exceeded_eager_return():
    fn = _worker_method("_glm5_whole_graph_path_state")
    returns = [ast.unparse(node.value) for node in ast.walk(fn) if isinstance(node, ast.Return)]
    assert "('eager', bucket, 'all_short_exceeded')" in returns
    guard = next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(s, ast.Return) and "all_short_exceeded" in ast.unparse(s)
            for s in node.body
        )
    )
    test_src = ast.unparse(guard.test)
    assert "_glm5_whole_model_all_short" in test_src
    assert "current_max_seqlen > index_topk_cfg" in test_src
    # The check runs after the captured-max_seqlen guard.
    order = [r for r in returns if "max_seqlen_exceeds_capture" in r or "all_short_exceeded" in r]
    assert order == [
        "('eager', bucket, 'max_seqlen_exceeds_capture')",
        "('eager', bucket, 'all_short_exceeded')",
    ]


# --------------------------------------------------------------------------- #
# Dead FlashMLA metadata is gone from the whole graph chain
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", _FLASHMLA_FREE_FILES, ids=lambda p: p.name)
def test_no_flashmla_token_in_graph_chain_file(path):
    hits = [
        f"{lineno}: {line.strip()}"
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if "flashmla" in line
    ]
    assert hits == []


def test_full_dsa_segment_class_has_no_flashmla():
    src = _SEGMENTS_SRC.read_text()
    cls = _class(ast.parse(src), "Glm5FullDsaAttnSegment")
    assert "flashmla" not in ast.get_source_segment(src, cls).lower()


def test_worker_replay_and_capture_inputs_carry_no_scheduler_metadata():
    for name in ("_prepare_glm5_layer_graph_inputs", "_make_glm5_whole_model_capture_inputs"):
        fn = _worker_method(name)
        assert not _calls_to(fn, "prepare_sparse_flash_mla_decode_tensor_metadata"), name
        returned = [n.value for n in ast.walk(fn) if isinstance(n, ast.Return)]
        keys = {
            k.value
            for r in returned
            if isinstance(r, ast.Dict)
            for k in r.keys
            if isinstance(k, ast.Constant)
        }
        assert "num_valid_tokens" in keys, name
        assert not {k for k in keys if "flashmla" in k}, (name, keys)


# --------------------------------------------------------------------------- #
# _glm5_dsa_graph_required_tokens as a unit (AST-compiled against stubs)
# --------------------------------------------------------------------------- #

class _StubAttnWrapperBase:
    max_seqlen = 0


def _required_tokens_fn():
    fn = _worker_method("_glm5_dsa_graph_required_tokens")
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {
        "math": math,
        "Sequence": typing.Sequence,
        "SequenceStatus": SequenceStatus,
        "AttnWrapperBase": _StubAttnWrapperBase,
    }
    exec(compile(module, str(_WORKER_SRC), "exec"), namespace)
    return namespace["_glm5_dsa_graph_required_tokens"]


def _seq(gid: int, budget: int, status=SequenceStatus.IN_DECODE):
    return types.SimpleNamespace(global_idx=gid, kv_token_budget=budget, status=status)


def _stub_worker(global_batch, *, max_input=0, max_decoding=0):
    return types.SimpleNamespace(
        global_batch=global_batch,
        max_input_length=max_input,
        max_decoding_length=max_decoding,
    )


@pytest.fixture
def required_tokens(monkeypatch):
    monkeypatch.setattr(_StubAttnWrapperBase, "max_seqlen", 0)
    return _required_tokens_fn()


def test_required_tokens_is_page_aligned_max_budget(required_tokens):
    worker = _stub_worker([_seq(0, 1000), _seq(1, 3000)])
    assert required_tokens(worker, [0, 1], page_size=64) == 3008


def test_required_tokens_ignores_completed_sequences(required_tokens):
    worker = _stub_worker(
        [_seq(0, 1000), _seq(1, 9000, status=SequenceStatus.COMPLETED)]
    )
    assert required_tokens(worker, [0], page_size=64) == 1024


def test_required_tokens_falls_back_to_active_ids_when_all_completed(required_tokens):
    # Every sequence is COMPLETED (empty non-completed budget list): the
    # active-id lookup decides, ignoring ids that are not in the batch.
    worker = _stub_worker(
        [
            _seq(3, 500, status=SequenceStatus.COMPLETED),
            _seq(4, 2100, status=SequenceStatus.COMPLETED),
            _seq(5, 7000, status=SequenceStatus.COMPLETED),
        ]
    )
    assert required_tokens(worker, [3, 4, 99], page_size=64) == 2112


def test_required_tokens_empty_batch_uses_max_seqlen_fallback(required_tokens, monkeypatch):
    monkeypatch.setattr(_StubAttnWrapperBase, "max_seqlen", 1500)
    worker = _stub_worker([], max_input=100, max_decoding=200)
    assert required_tokens(worker, [7, 8], page_size=64) == 1536


def test_required_tokens_no_batch_uses_input_plus_decoding_fallback(required_tokens):
    worker = _stub_worker(None, max_input=1000, max_decoding=1100)
    assert required_tokens(worker, [], page_size=64) == 2112


def test_required_tokens_floor_is_one_page(required_tokens):
    worker = _stub_worker(None)
    assert required_tokens(worker, [], page_size=64) == 64


def test_inline_replay_gate_blocks_an_exceeded_all_short_graph():
    """The decode loop's inline gate (not just the advisory path-state) must
    refuse to replay an all_short capture once the longest context passes
    index_topk; under a required graph that surfaces as reason=all_short_exceeded."""
    source = _WORKER_SRC.read_text()
    gate = source[source.index("_glm5_whole_graph_active = bool("):]
    gate = gate[: gate.index("_use_graph = (")]
    assert "_glm5_all_short_exceeded" in gate
    assert "not _glm5_all_short_exceeded" in gate
    assert "_glm5_whole_model_all_short" in gate


def test_admission_refuses_over_budget_rows_in_an_all_short_lifetime():
    """The decode-batch admission must reject budgets beyond index_topk while
    an all_short capture is live, identically on every rank, so no rank can
    diverge from the collective mid-replay."""
    source = _WORKER_SRC.read_text()
    anchor = source.index("DecodeScheduler.select_decode_batch")
    window = source[anchor: anchor + 2500]
    assert "_glm5_whole_model_all_short" in window
    assert "kv_token_budget" in window
    assert "raise RuntimeError" in window


def test_guards_read_the_stored_index_topk():
    source = _WORKER_SRC.read_text()
    assert source.count("_glm5_whole_model_index_topk", 0) >= 5
    # No guard reads index_topk off model_config any more (the step-state
    # hint helper at module level is a separate, pre-existing consumer).
    gate = source[source.index("_glm5_whole_graph_active = bool("):]
    gate = gate[: gate.index("_use_graph = (")]
    assert "_glm5_whole_model_index_topk" in gate
    admission = source[source.index("DecodeScheduler.select_decode_batch"):][:2500]
    assert "_glm5_whole_model_index_topk" in admission
