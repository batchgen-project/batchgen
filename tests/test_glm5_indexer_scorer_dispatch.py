"""The indexer scorer dispatches on CACHE FORMAT, with no fallback edge.

GLM-5.2 stores indexer keys in DeepGEMM's page-split FP8 layout and must score
them with ``deep_gemm.fp8_paged_mqa_logits`` (the ReLU-gated GLM-5
aggregation). GLM-5, 5.1 and 5.3 keep bf16 AoS pages and the retired no-ReLU
Triton scorer. Picking the wrong one is silent -- a misread cache still
produces 2048 plausible indices -- so the selector is pinned here from source:

* all three production scorer sites (one eager, two graph segments) branch on
  the cache format, not on a model name or an SM version;
* the page-split arm reaches DeepGEMM with no ``try``/``except`` and no
  gather, so GLM-5.2 cannot silently fall back to the bf16 scorer;
* the retired scorer survives only on the AoS arm;
* the DeepGEMM JIT is warmed before any capture can need it.

These are source-contract tests: importing the segment or model modules pulls
in Triton, FA3 and DeepGEMM, so they parse the files instead and run anywhere.
"""

from __future__ import annotations

import ast
import collections
import pathlib

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SEGMENTS_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/cuda_graph_segments.py"
_MODEL_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/model.py"
_SCORER_SRC = _REPO_ROOT / "batchgen_kernels/attention/dsa/paged_mqa_scorer.py"


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _top_level(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no top-level `{name}`")


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    cls = _top_level(tree, class_name)
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == method_name:
            return stmt
    raise AssertionError(f"no `{class_name}.{method_name}`")


def _branch(fn: ast.FunctionDef, test_src: str) -> ast.If:
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ast.unparse(node.test) == test_src:
            return node
    raise AssertionError(f"no `if {test_src}:` in `{fn.name}`")


def _call_names(node) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            counts[ast.unparse(sub.func)] += 1
    return counts


def _call_kwargs(node, callee: str) -> dict[str, str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == callee:
            return {kw.arg: ast.unparse(kw.value) for kw in sub.keywords if kw.arg}
    raise AssertionError(f"no call to `{callee}`")


_DEEPGEMM_SCORER = "paged_mqa_score_and_topk_out"
_TRITON_SCORER = "fused_paged_score_and_topk_with_slots_out"
_SEGMENT_CLASSES = ("Glm5DsaAttnSegment", "Glm5FullDsaAttnSegment")


# --------------------------------------------------------------------------- #
# The two graph segments
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("class_name", _SEGMENT_CLASSES)
def test_graph_segment_scorer_branches_on_the_cache_format(class_name):
    fn = _method(_tree(_SEGMENTS_SRC), class_name, "_score_and_topk")
    branch = _branch(fn, "self.aux_is_page_split")
    page_split_calls = _call_names(branch)
    assert page_split_calls[_DEEPGEMM_SCORER] == 1
    assert page_split_calls[_TRITON_SCORER] == 0
    # The retired scorer is reachable ONLY outside the page-split branch.
    total = _call_names(fn)
    assert total[_DEEPGEMM_SCORER] == 1
    assert total[_TRITON_SCORER] == 1


@pytest.mark.parametrize("class_name", _SEGMENT_CLASSES)
def test_graph_segment_forward_only_scores_through_the_dispatcher(class_name):
    """No forward may call either scorer directly: that would pin one format."""
    fwd = _method(_tree(_SEGMENTS_SRC), class_name, "forward")
    calls = _call_names(fwd)
    assert calls["self._score_and_topk"] == 1
    assert calls[_DEEPGEMM_SCORER] == 0
    assert calls[_TRITON_SCORER] == 0


@pytest.mark.parametrize("class_name", _SEGMENT_CLASSES)
def test_graph_segment_page_split_arm_has_no_fallback(class_name):
    """A page-split cache is unreadable by the Triton scorer, so a swallowed
    DeepGEMM import/launch error must not be able to degrade into it."""
    fn = _method(_tree(_SEGMENTS_SRC), class_name, "_score_and_topk")
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)]


@pytest.mark.parametrize("class_name", _SEGMENT_CLASSES)
def test_graph_segment_warms_the_deepgemm_jit_before_touching_buffers(class_name):
    """The metadata kernel is templated on align(bucket, 32), so each bucket
    is its own JIT instantiation -- including buckets that are built as views
    of a larger base and return early."""
    fn = _method(_tree(_SEGMENTS_SRC), class_name, "setup_static_buffers")
    first = fn.body[0]
    assert isinstance(first, ast.Expr)
    assert ast.unparse(first.value) == "self._warm_scorer_jit(bucket_size)"


@pytest.mark.parametrize("class_name", _SEGMENT_CLASSES)
def test_graph_segment_passes_the_static_deepgemm_buffers(class_name):
    """Every tensor the captured call needs must come from the segment's own
    static set, or replay would read a freed graph-private allocation."""
    fn = _method(_tree(_SEGMENTS_SRC), class_name, "_score_and_topk")
    kwargs = _call_kwargs(fn, _DEEPGEMM_SCORER)
    assert kwargs["q_fp8"] == "buffers.q_index_fp8"
    assert kwargs["block_table"] == "buffers.aux_block_table"
    assert kwargs["context_lens"] == "buffers.aux_context_lens"
    assert kwargs["top_k_indices"] == "buffers.top_k_indices"
    assert kwargs["max_seqlen"] == "self.max_seqlen"


def test_full_segment_scores_the_masked_slot_and_seqlen_views():
    """The full segment derives slot/seqlen views that zero invalid rows; the
    DeepGEMM path must consume those, not the raw static inputs."""
    fwd = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    kwargs = _call_kwargs(fwd, "self._score_and_topk")
    assert kwargs["aux_slot_indices"] == "buffers.safe_aux_slot_indices"
    assert kwargs["cache_seqlens"] == "buffers.safe_cache_seqlens"
    assert kwargs["num_valid_tokens"] == "num_valid_tokens"


# --------------------------------------------------------------------------- #
# The eager path
# --------------------------------------------------------------------------- #

def test_eager_scorer_branches_on_the_cache_format():
    fn = _method(_tree(_MODEL_SRC), "Glm5Indexer", "score_and_select_paged")
    branch = _branch(fn, "cache_is_page_split(indexer_blocked_k, self.index_head_dim)")
    calls = _call_names(branch)
    assert calls["self._score_and_select_paged_fp8"] == 1


def test_eager_page_split_path_does_not_gather_the_cache():
    """The whole point of the swap: DeepGEMM reads the paged cache, so the
    dense [batch, max_seqlen, 128] dequantizing gather is gone."""
    fn = _method(_tree(_MODEL_SRC), "Glm5Indexer", "_score_and_select_paged_fp8")
    src = ast.unparse(fn)
    assert "fused_dense_paged_gather" not in src
    assert "fused_score_pipeline" not in src
    calls = _call_names(fn)
    assert calls["paged_mqa_score_and_topk"] == 1


def test_eager_page_split_path_has_no_fallback():
    fn = _method(_tree(_MODEL_SRC), "Glm5Indexer", "_score_and_select_paged_fp8")
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)]


def test_eager_aos_path_keeps_the_retired_triton_scorer():
    fn = _method(_tree(_MODEL_SRC), "Glm5Indexer", "score_and_select_paged")
    src = ast.unparse(fn)
    assert "fused_score_pipeline" in src
    assert "fused_dense_paged_gather" in src
    # ...and the page-split dequantizing gather has no production caller left.
    assert "fused_dense_paged_gather_fp8_pagesplit" not in src


def test_eager_path_folds_the_same_head_gate_scale_as_the_graph():
    """`weights = head_gates` is only exact because the gates already carry
    n_heads**-0.5 * head_dim**-0.5 and relu(s*x) == s*relu(x) for s > 0. Both
    production paths must therefore build the gates the same way."""
    eager = _method(_tree(_MODEL_SRC), "Glm5Indexer", "_score_and_select_paged_fp8")
    for sub in ast.walk(eager):
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == "compute_head_gates":
            args = [ast.unparse(a) for a in sub.args]
            assert args[2] == "self.index_n_heads"
            assert args[3] == "self.index_head_dim"
            break
    else:
        raise AssertionError("no compute_head_gates call")

    graph = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    graph_scale = _call_kwargs(graph, "head_gates_out")["scale"].replace(" ", "")
    assert graph_scale in (
        "indexer.index_n_heads**-0.5*indexer.index_head_dim**-0.5",
        "indexer.index_n_heads**(-0.5)*indexer.index_head_dim**(-0.5)",
    )


def test_relu_gated_reference_names_the_production_implementation():
    fn = _method(_tree(_MODEL_SRC), "Glm5Indexer", "score_and_select_relu_gated")
    doc = ast.get_docstring(fn) or ""
    assert "REFERENCE ONLY" in doc
    assert "fp8_paged_mqa_logits" in doc
    assert "_score_and_select_paged_fp8" in doc


def test_relu_gated_reference_still_has_no_callers():
    """It is the fp32 oracle, not a decode path. Any call site would be a
    second, differently-quantized ReLU implementation able to drift."""
    for src in (_MODEL_SRC, _SEGMENTS_SRC):
        tree = _tree(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert "score_and_select_relu_gated" not in ast.unparse(node.func)


# --------------------------------------------------------------------------- #
# The scorer wrapper
# --------------------------------------------------------------------------- #

def test_scorer_passes_the_two_dimensional_context_lens_and_no_clean_logits():
    """Both corrections from the item-2a node probe: the serving build
    hard-asserts on the 1-D ``context_lens`` form, and ``clean_logits=True``
    asserts against the 2-D form it also requires."""
    fn = _top_level(_tree(_SCORER_SRC), "deep_gemm_paged_mqa_logits")
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call) and ast.unparse(sub.func).endswith(
            "fp8_paged_mqa_logits"
        ):
            args = [ast.unparse(a) for a in sub.args]
            assert args[3] == "deep_gemm_context_lens"
            assert args[6] == "plan.max_context_len"
            assert args[7] == "False", "clean_logits must stay False"
            break
    else:
        raise AssertionError("no fp8_paged_mqa_logits call")
    # context_lens is built 2-D by the preparation pass, which the plan's
    # next_n pins at 1.
    assert "PAGED_MQA_NEXT_N" in _SCORER_SRC.read_text()


def test_scorer_rechecks_the_logits_row_stride_after_every_call():
    fn = _top_level(_tree(_SCORER_SRC), "deep_gemm_paged_mqa_logits")
    assert _call_names(fn)["check_logits_layout"] == 1


def test_scorer_bounds_the_topk_by_the_same_context_lens():
    """DeepGEMM leaves the tail unmasked, so the top-k's lengths MUST be the
    TRUE per-row context_lens -- not max_seqlen, not the raw cache_seqlens, and
    not the guarded copy DeepGEMM was handed (which is elementwise >= them)."""
    fn = _top_level(_tree(_SCORER_SRC), "_topk_from_logits")
    src = ast.unparse(fn)
    assert "lengths = context_lens.view(plan.batch)" in src
    kwargs = _call_kwargs(fn, "_topk_from_scores_kernel[plan.batch,]")
    assert kwargs["MASK_BY_SEQLEN"] == "True"


def _positional_calls(fn, callee: str) -> list[list[str]]:
    return [
        [ast.unparse(a) for a in sub.args]
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == callee
    ]


def test_graph_scorer_hands_deepgemm_the_guarded_lens_and_the_topk_the_true_ones():
    """The all-zero-context_lens guard (DeepGEMM sm90 metadata OOB): the
    Triton pass writes BOTH lengths in one launch; DeepGEMM reads only the
    guarded one, the top-k only the true one.

    The graph scorer is split in two -- a step-level prologue and the
    per-layer half -- so the guard spans both halves and each is pinned where
    it now lives.
    """
    tree = _tree(_SCORER_SRC)
    prologue = _top_level(tree, "paged_mqa_step_prologue_out")
    prep = _call_kwargs(prologue, "prepare_paged_mqa_inputs_out")
    assert prep["deep_gemm_context_lens_out"] == "deep_gemm_context_lens"
    # The schedule is derived from the GUARDED lengths, never the true ones.
    (meta_args,) = _positional_calls(prologue, "paged_mqa_logits_schedule_meta")
    assert meta_args == ["deep_gemm_context_lens"]

    per_layer = _top_level(tree, "paged_mqa_score_and_topk_prepared_out")
    (dg_args,) = _positional_calls(per_layer, "deep_gemm_paged_mqa_logits")
    assert dg_args[3] == "deep_gemm_context_lens"
    (topk_args,) = _positional_calls(per_layer, "_topk_from_logits")
    assert topk_args[1] == "context_lens"


def test_the_whole_layer_scorer_is_the_prologue_plus_the_per_layer_half():
    """``paged_mqa_score_and_topk_out`` must stay exactly the composition of
    the two halves, so a single-indexer-layer caller and a 21-layer caller
    cannot drift apart."""
    fn = _top_level(_tree(_SCORER_SRC), "paged_mqa_score_and_topk_out")
    prologue = _call_kwargs(fn, "paged_mqa_step_prologue_out")
    per_layer = _call_kwargs(fn, "paged_mqa_score_and_topk_prepared_out")
    # The guarded lengths are this call's own, and the same tensor is written
    # by the prologue and read by the per-layer half.
    src = ast.unparse(fn)
    assert "deep_gemm_context_lens = torch.empty_like(context_lens)" in src
    assert prologue["deep_gemm_context_lens"] == "deep_gemm_context_lens"
    assert per_layer["deep_gemm_context_lens"] == "deep_gemm_context_lens"
    assert per_layer["schedule_meta"] == "schedule_meta"
    for shared in ("block_table", "context_lens", "cache_seqlens"):
        assert prologue[shared] == shared
        assert per_layer[shared] == shared
    # No prep or metadata launch may survive in the per-layer half (names in
    # its docstring and its named errors are fine -- CALLS are not).
    called = _call_names(
        _top_level(_tree(_SCORER_SRC), "paged_mqa_score_and_topk_prepared_out")
    )
    for hoisted in (
        "prepare_paged_mqa_inputs_out",
        "paged_mqa_logits_schedule_meta",
        "paged_mqa_step_prologue_out",
    ):
        assert hoisted not in called, hoisted


def test_eager_scorer_hands_deepgemm_the_guarded_lens_and_the_topk_the_true_ones():
    fn = _top_level(_tree(_SCORER_SRC), "paged_mqa_score_and_topk")
    src = ast.unparse(fn)
    assert (
        "deep_gemm_context_lens = guard_deep_gemm_context_lens(context_lens)" in src
    )
    (dg_args,) = _positional_calls(fn, "deep_gemm_paged_mqa_logits")
    assert dg_args[3] == "deep_gemm_context_lens"
    (topk_args,) = _positional_calls(fn, "_topk_from_logits")
    assert topk_args[1] == "context_lens"


def test_every_deepgemm_launch_gets_guarded_or_positive_lengths():
    """No call site may hand DeepGEMM unguarded lengths. The JIT warm-up is
    the one exception, and only because its lengths are a positive fill."""
    tree = _tree(_SCORER_SRC)
    sites = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for args in _positional_calls(node, "deep_gemm_paged_mqa_logits"):
                sites[node.name] = args[3]
    assert sites == {
        "paged_mqa_score_and_topk_prepared_out": "deep_gemm_context_lens",
        "paged_mqa_score_and_topk": "deep_gemm_context_lens",
        "warm_paged_mqa_logits": "context_lens",
    }
    warm = ast.unparse(_top_level(tree, "warm_paged_mqa_logits"))
    assert "min(int(page_size), int(max_seqlen))" in warm


def test_every_schedule_metadata_launch_gets_guarded_or_positive_lengths():
    """Same rule for the metadata kernel, which is the one that faults: it is
    derived ONLY from lengths that are already guarded or a positive fill."""
    tree = _tree(_SCORER_SRC)
    sites = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for args in _positional_calls(node, "paged_mqa_logits_schedule_meta"):
                sites[node.name] = args[0]
    assert sites == {
        "paged_mqa_step_prologue_out": "deep_gemm_context_lens",
        "paged_mqa_score_and_topk": "deep_gemm_context_lens",
        "warm_paged_mqa_logits": "context_lens",
    }
    # And the raw DeepGEMM entry point is CALLED from exactly one place.
    raw = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            name.endswith("get_paged_mqa_logits_metadata")
            for name in _call_names(node)
        )
    ]
    assert raw == ["paged_mqa_logits_schedule_meta"]


def test_triton_prepare_fuses_the_row0_guard():
    src = _SCORER_SRC.read_text()
    kernel = ast.get_source_segment(
        src, _top_level(_tree(_SCORER_SRC), "_prepare_paged_mqa_inputs_kernel")
    )
    assert "tl.where(pid_b == 0, tl.maximum(context_len, 1), context_len)" in kernel
    assert "tl.store(DG_CONTEXT_LENS_ptr + pid_b, dg_len)" in kernel
    assert "tl.store(CONTEXT_LENS_ptr + pid_b, context_len)" in kernel
    # Rows that read nothing never dereference their slot and get page 0.
    assert "mask=mask & reads" in kernel
    assert "other=0," in kernel


def test_deepgemm_call_names_the_upstream_defect():
    """A future DeepGEMM pin bump must find WHY the guard exists at the call.

    The call that faults is the metadata one, so the note lives with it.
    """
    src = _SCORER_SRC.read_text()
    fn = ast.get_source_segment(
        src, _top_level(_tree(_SCORER_SRC), "paged_mqa_logits_schedule_meta")
    )
    assert "0.1.5.post3" in fn
    assert "sm90_fp8_mqa_logits.hpp:188" in fn
    assert "batch % 32 == 0" in fn
    assert "2026-10-17" in fn
    # And the logits call still states which form of the lengths it takes.
    logits = ast.get_source_segment(
        src, _top_level(_tree(_SCORER_SRC), "deep_gemm_paged_mqa_logits")
    )
    assert "GUARDED form" in logits


def test_scorer_reinterprets_the_cache_as_bytes_without_copying():
    fn = _top_level(_tree(_SCORER_SRC), "deep_gemm_paged_mqa_logits")
    src = ast.unparse(fn)
    assert "aux_blocked_k.view(torch.uint8)" in src


def test_deep_gemm_accessor_fails_loud_on_a_missing_api():
    fn = _top_level(_tree(_SCORER_SRC), "_deep_gemm")
    src = ast.unparse(fn)
    assert "fp8_paged_mqa_logits" in src
    assert "get_paged_mqa_logits_metadata" in src
    assert "get_num_sms" in src
    assert "raise RuntimeError" in src
