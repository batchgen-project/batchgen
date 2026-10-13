"""CPU contract tests for the GLM-5 PURE EAGER direct-index FA3 decode path.

The eager selector cannot be imported on a CPU dev box (it pulls FA3, Triton
and DeepGEMM at module scope), so these tests parse
``batchgen/attention/dsa/glm5_decode_selector.py`` and
``batchgen/models/glm/glm5/wrappers.py`` instead. They pin:

* the gather + FlashMLA tail is gone (no ``select_mla_kv_for_flashmla_bf16``,
  no ``prepare_sparse_flash_mla_decode_inputs``, no padded KV slab, no packed
  ``query_states``);
* sparse attention runs on ``flash_attn_with_kvcache`` with the same call
  shape as the whole-model graph segment, over physical token IDs produced by
  ``transform_selected_positions_out``;
* the batch-level all-short fast path reads the page-size-N cache directly;
* the dense-index build survives exactly where the top-k carry needs it;
* the pure-eager consumer in ``wrappers.py`` takes ``attn_out`` from the
  builder instead of calling FlashMLA itself.

The last test asserts the LEGACY standalone captured-DSA-graph path is still
wired, so this conversion cannot quietly delete it.
"""

import ast
import pathlib

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SELECTOR_SRC = _REPO_ROOT / "batchgen/attention/dsa/glm5_decode_selector.py"
_WRAPPERS_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/wrappers.py"
_SPARSE_MLA_SRC = _REPO_ROOT / "batchgen/attention/dsa/sparse_decode_mla.py"
_UNIFIED_SELECTOR_SRC = _REPO_ROOT / "batchgen/attention/dsa/unified_selector.py"

# Names that only existed to serve the retired gather + FlashMLA tail.
_GATHER_ERA_NAMES = (
    "select_mla_kv_for_flashmla_bf16",
    "select_mla_kv_for_flashmla_bf16_out",
    "prepare_sparse_flash_mla_decode_inputs",
    "prepare_sparse_flash_mla_decode_tensor_metadata",
    "run_prepared_sparse_flash_mla_decode",
    "PreparedSparseFlashMlaDecode",
)


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(node: ast.AST, name: str) -> ast.FunctionDef:
    for sub in ast.walk(node):
        if isinstance(sub, ast.FunctionDef) and sub.name == name:
            return sub
    raise AssertionError(f"no function `{name}`")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level class `{name}`")


def _calls_to(node: ast.AST, callee: str) -> list[ast.Call]:
    return [
        sub
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == callee
    ]


def _called_names(node: ast.AST) -> set[str]:
    return {
        ast.unparse(sub.func)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
    }


def _imported_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            names.update(alias.name for alias in sub.names)
    return names


def _imported_modules(node: ast.AST) -> set[str]:
    mods: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.ImportFrom) and sub.module:
            mods.add(sub.module)
        elif isinstance(sub, ast.Import):
            mods.update(alias.name for alias in sub.names)
    return mods


def _kwargs(call: ast.Call) -> dict[str, str]:
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def _selector() -> ast.Module:
    return _tree(_SELECTOR_SRC)


def _builder() -> ast.FunctionDef:
    return _function(_selector(), "build_glm5_dsa_flashmla_inputs")


def _eager_consumer() -> ast.FunctionDef:
    return _function(
        _class(_tree(_WRAPPERS_SRC), "GLM5AttnWrapper"),
        "_forward_decode_dsa_eager",
    )


# --------------------------------------------------------------------------- #
# (a) the gather + FlashMLA tail is gone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", _GATHER_ERA_NAMES)
def test_selector_no_longer_imports_gather_era_names(name: str) -> None:
    assert name not in _imported_names(_selector()), (
        f"`{name}` is still imported by the eager selector; the eager path "
        "must run FA3 over the resident paged KV"
    )


@pytest.mark.parametrize("name", _GATHER_ERA_NAMES)
def test_selector_no_longer_calls_gather_era_names(name: str) -> None:
    assert not _calls_to(_selector(), name), f"`{name}` is still called"


def test_selector_no_longer_imports_gather_modules() -> None:
    mods = _imported_modules(_selector())
    assert "batchgen.attention.dsa.unified_selector" not in mods
    assert "batchgen.attention.dsa.sparse_decode_mla" not in mods


def test_builder_has_no_sparse_gather_stage() -> None:
    src = ast.unparse(_builder())
    assert "sparse_gather" not in src, "the gather timer/stage must be gone"


def test_build_query_states_and_gather_log_are_deleted() -> None:
    tree = _selector()
    for dead in ("_build_query_states", "_log_gather_bounds"):
        with pytest.raises(AssertionError):
            _function(tree, dead)


def test_bounds_logging_is_retargeted_to_physical_ids() -> None:
    tree = _selector()
    fn = _function(tree, "_log_selected_token_bounds")
    src = ast.unparse(fn)
    # Validates physical ids against the page-size-1 cache extent and reports
    # the -1 padding sentinel.
    assert "total_tokens" in src
    assert "selected_token_ids < 0" in src
    assert "max_physical_id" in src
    # Still reachable from the builder under the verify-indices env flag.
    assert _calls_to(_builder(), "_log_selected_token_bounds")
    # `_log_dsa_bounds` (slot/page-table bounds) is unrelated and stays.
    _function(tree, "_log_dsa_bounds")
    assert _calls_to(_builder(), "_log_dsa_bounds")


def test_dataclass_drops_gather_fields_and_carries_attn_out() -> None:
    cls = _class(_selector(), "Glm5DsaFlashMlaInputs")
    fields = [
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]
    assert fields == [
        "attn_out",
        "q_nope",
        "q_rope",
        "selected_lengths",
        "selected_token_ids",
        "row_modes",
        "primary_k_tensor",
        "indexer_k_tensor",
        "branch_label",
    ]
    for dead in ("flashmla", "query_states", "selected_mla_kv", "selected_indices"):
        assert dead not in fields


# --------------------------------------------------------------------------- #
# (b) the FA3 call shape
# --------------------------------------------------------------------------- #


def test_selector_imports_fa3_and_the_transform_kernel() -> None:
    tree = _selector()
    fa3_imports = [
        sub
        for sub in ast.walk(tree)
        if isinstance(sub, ast.ImportFrom) and sub.module == "flash_attn_interface"
    ]
    assert len(fa3_imports) == 1
    alias = fa3_imports[0].names[0]
    assert alias.name == "flash_attn_with_kvcache"
    assert alias.asname == "_fa3_with_kvcache"
    # Guarded so a CPU box without FA3 fails at the call site, not at import.
    assert any(
        isinstance(sub, ast.Try)
        and any(
            isinstance(stmt, ast.ImportFrom)
            and stmt.module == "flash_attn_interface"
            for stmt in sub.body
        )
        for sub in ast.walk(tree)
    )
    assert "transform_selected_positions_out" in _imported_names(tree)
    assert (
        "batchgen_kernels.attention.dsa.selected_page_table"
        in _imported_modules(tree)
    )


def test_missing_fa3_fails_loud() -> None:
    fn = _function(_selector(), "_require_fa3")
    src = ast.unparse(fn)
    assert "_fa3_with_kvcache is None" in src
    assert "RuntimeError" in src
    assert "flash_attn_interface" in src
    assert "flash_attn_with_kvcache is unavailable" in src
    # Both FA3 branches route through the fail-loud accessor.
    for runner in ("_run_selected_fa3", "_run_all_short_fa3"):
        assert _calls_to(_function(_selector(), runner), "_require_fa3")


def test_selected_fa3_call_matches_the_graph_segment_shape() -> None:
    fn = _function(_selector(), "_run_selected_fa3")
    calls = _calls_to(fn, "fa3")
    assert len(calls) == 1
    kw = _kwargs(calls[0])
    assert kw["page_table"] == "selected_token_ids"
    assert kw["cache_seqlens"] == "selected_lengths"
    assert kw["num_splits"] == "0"
    assert kw["causal"] == "True"
    assert kw["return_softmax_lse"] == "False"
    assert kw["softmax_scale"] == "float(attn.softmax_scale)"
    assert kw["q"] == "q_pe.squeeze(2).unsqueeze(1)"
    assert kw["qv"] == "absorbed_q.unsqueeze(1)"
    # Page-size-1 view of the resident cache; KV is never copied.
    assert "mla_blocked_k.view(-1, 1, 1," in ast.unparse(fn)
    assert kw["k_cache"] == "flat_kv[..., attn.kv_lora_rank:]"
    assert kw["v_cache"] == "flat_kv[..., :attn.kv_lora_rank]"
    # Absolute token IDs: no slot indirection on this branch.
    assert "cache_batch_idx" not in kw


def test_eager_fa3_shape_matches_the_whole_model_graph_segment() -> None:
    segment_src = _REPO_ROOT / "batchgen/models/glm/glm5/cuda_graph_segments.py"
    seg = _function(_tree(segment_src), "_run_selected_fa3")
    seg_kw = _kwargs(_calls_to(seg, "_fa3_with_kvcache")[0])
    eager_kw = _kwargs(
        _calls_to(_function(_selector(), "_run_selected_fa3"), "fa3")[0]
    )
    # Shared keyword surface: graph and eager must use one backend contract.
    assert set(seg_kw) == set(eager_kw)
    for scalar in ("softmax_scale", "causal", "num_splits", "return_softmax_lse"):
        assert seg_kw[scalar] == eager_kw[scalar]


def test_q_absorb_feeds_fa3_qv_directly() -> None:
    fn = _function(_selector(), "_absorb_q_nope")
    src = ast.unparse(fn)
    assert "fp8_q_absorb" in src
    assert "_fp8_absorb_weights is None" in src
    assert "RuntimeError" in src
    # The builder absorbs once, under the q_absorb timer, and hands the result
    # to FA3 as `qv` rather than packing it into a query_states buffer.
    builder = _builder()
    assert _calls_to(builder, "_absorb_q_nope")
    assert "pack_flashmla_query" not in ast.unparse(builder)


# --------------------------------------------------------------------------- #
# (c) the all-short page-size-N fast path
# --------------------------------------------------------------------------- #


def _dense_short_circuit_branch() -> ast.If:
    for sub in ast.walk(_builder()):
        if isinstance(sub, ast.If) and "dense-short-circuit" in ast.unparse(sub.test):
            return sub
    raise AssertionError("no `dense-short-circuit` branch in the builder")


def test_dense_short_circuit_uses_the_page_size_cache_fast_path() -> None:
    branch = _dense_short_circuit_branch()
    fast = ast.Module(body=branch.body, type_ignores=[])
    slow = ast.Module(body=branch.orelse, type_ignores=[])

    assert _calls_to(fast, "_run_all_short_fa3")
    assert not _calls_to(fast, "transform_selected_positions_out")
    assert not _calls_to(fast, "_run_selected_fa3")
    assert "torch.empty" not in ast.unparse(fast), (
        "the all-short fast path must not allocate a selection table"
    )

    assert _calls_to(slow, "transform_selected_positions_out")
    assert _calls_to(slow, "_run_selected_fa3")
    assert not _calls_to(slow, "_run_all_short_fa3")


def test_all_short_fa3_reads_the_real_page_table() -> None:
    fn = _function(_selector(), "_run_all_short_fa3")
    calls = _calls_to(fn, "fa3")
    assert len(calls) == 1
    kw = _kwargs(calls[0])
    assert kw["page_table"] == "page_table"
    assert kw["cache_batch_idx"] == "cache_batch_idx"
    assert kw["cache_seqlens"] == "cache_seqlens"
    assert kw["num_splits"] == "0"
    assert kw["causal"] == "True"
    # Dense over the resident page-size-N cache: no page-size-1 reinterpret.
    assert kw["k_cache"] == "mla_blocked_k[..., attn.kv_lora_rank:]"
    assert kw["v_cache"] == "mla_blocked_k[..., :attn.kv_lora_rank]"
    assert ".view(-1, 1, 1," not in ast.unparse(fn)

    call = _calls_to(_dense_short_circuit_branch(), "_run_all_short_fa3")[0]
    fast_kw = _kwargs(call)
    assert fast_kw["page_table"] == "mla_block_table"
    assert fast_kw["cache_batch_idx"] == "primary_selector_slots"
    assert fast_kw["cache_seqlens"] == "safe_cache_seqlens"


def test_slot_semantics_match_the_retired_gather() -> None:
    """`slot_override_active` decides table-vs-slot indirection, as before."""
    builder = _builder()
    src = ast.unparse(builder)
    # Override active: storage-ordered table + explicit slot indices.
    assert "primary_selector_slots = primary_slot_indices" in src
    # Otherwise: table pre-reordered into batch order, no slot indirection.
    assert _calls_to(builder, "reorder_block_table_to_batch_slots")
    transform = _calls_to(builder, "transform_selected_positions_out")
    assert len(transform) == 1
    assert _kwargs(transform[0])["primary_slot_indices"] == "primary_selector_slots"
    assert _kwargs(transform[0])["page_size"] == "mla_page_size"


def test_dense_index_build_is_gated_on_the_topk_carry() -> None:
    """`build_clamped_dense_token_indices` survives only for the carry.

    The all-short FA3 fast path never reads logical top-k, but a layer with
    `next_skip_topk` must still publish `_dsa_prev_topk_indices` for the
    top-k-reusing layers that follow.
    """
    selector = _function(_selector(), "_select_glm5_dsa_indices")
    kwonly = [arg.arg for arg in selector.args.kwonlyargs]
    assert "need_dense_indices" in kwonly
    # Default preserves the historical contract for legacy callers/tests.
    default = selector.args.kw_defaults[kwonly.index("need_dense_indices")]
    assert ast.unparse(default) == "True"

    # The dense build in the `not any_long` short circuit is now conditional.
    guards = [
        sub
        for sub in ast.walk(selector)
        if isinstance(sub, ast.If)
        and ast.unparse(sub.test) == "need_dense_indices"
        and _calls_to(sub, "build_clamped_dense_token_indices")
    ]
    assert len(guards) == 1

    builder = _builder()
    call = _calls_to(builder, "_select_glm5_dsa_indices")[0]
    assert (
        _kwargs(call)["need_dense_indices"]
        == "bool(wrapper.module.next_skip_topk)"
    )
    # The carry itself is untouched: still published when next_skip_topk.
    assert "type(wrapper)._dsa_prev_topk_indices = top_k_indices" in ast.unparse(
        builder
    )
    # ...and the reuse-shared branch still consumes it.
    assert "_dsa_prev_topk_indices" in ast.unparse(builder)


def test_sparse_attn_timer_moved_into_the_builder() -> None:
    """Decode timing stays comparable: the label is still `sparse_attn`."""
    builder_src = ast.unparse(_builder())
    assert builder_src.count("dt.timed('sparse_attn', li)") == 2, (
        "both FA3 branches must run under the sparse_attn timer"
    )
    assert "dt.timed('q_absorb', li)" in builder_src
    assert "sparse_attn" not in ast.unparse(_eager_consumer())


# --------------------------------------------------------------------------- #
# (d) the pure-eager consumer in wrappers.py
# --------------------------------------------------------------------------- #


def test_eager_consumer_no_longer_runs_flashmla() -> None:
    fn = _eager_consumer()
    assert "run_prepared_sparse_flash_mla_decode" not in _imported_names(fn)
    assert not _calls_to(fn, "run_prepared_sparse_flash_mla_decode")
    assert "batchgen.attention.dsa.sparse_decode_mla" not in _imported_modules(fn)
    src = ast.unparse(fn)
    assert "attn_out = selector_inputs.attn_out" in src
    assert ".flashmla" not in src


def test_eager_consumer_keeps_absorb_and_debug_contract() -> None:
    fn = _eager_consumer()
    called = _called_names(fn)
    # out-absorb / o_proj stages unchanged.
    assert "fp8_out_absorb" in called
    assert "w8a8_deepgemm" in called
    assert "act_quant" in called
    assert _calls_to(fn, "build_glm5_dsa_flashmla_inputs")
    # return_debug keys preserved; raw_attn_out now carries the FA3 output.
    debug_dicts = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Dict)
        and {ast.unparse(k).strip("'\"") for k in sub.keys if k is not None}
        == {"selector_inputs", "raw_attn_out", "attn_heads"}
    ]
    assert len(debug_dicts) == 1
    mapping = {
        ast.unparse(k).strip("'\""): ast.unparse(v)
        for k, v in zip(debug_dicts[0].keys, debug_dicts[0].values)
    }
    assert mapping["raw_attn_out"] == "attn_out"


# --------------------------------------------------------------------------- #
# over-deletion guard: the legacy standalone captured DSA graph stays wired
# --------------------------------------------------------------------------- #


def test_legacy_standalone_dsa_graph_path_is_intact() -> None:
    selector_tree = _selector()
    # The graph-route prefix and its dataclass are untouched.
    _function(selector_tree, "build_glm5_dsa_graph_segment_inputs")
    _class(selector_tree, "Glm5DsaGraphSegmentInputs")

    wrappers_tree = _tree(_WRAPPERS_SRC)
    wrappers_src = _WRAPPERS_SRC.read_text()
    assert "build_glm5_dsa_graph_segment_inputs" in _imported_names(wrappers_tree)
    assert "self._dsa_cuda_graph_manager.replay" in wrappers_src

    # The FlashMLA helpers themselves are still provided for the graph chain.
    sparse_mla = _SPARSE_MLA_SRC.read_text()
    assert "def run_prepared_sparse_flash_mla_decode" in sparse_mla
    assert "def prepare_sparse_flash_mla_decode_inputs" in sparse_mla
    unified = _UNIFIED_SELECTOR_SRC.read_text()
    assert "def select_mla_kv_for_flashmla_bf16" in unified
