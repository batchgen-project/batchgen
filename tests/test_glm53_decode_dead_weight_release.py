"""GLM-5.3 decode dead-weight recovery: the three releases and their wiring.

Decode-entry residency is sized BEFORE the GPU KV pool, so every byte a
decode instance keeps without reading it is a byte the pool does not get.
Three such blocks are released here, each independently:

1. the FP8 ``kv_b_proj`` weight, dead once ``initialize_decode_absorb`` has
   turned it into the BF16 absorb pair (prefill's readers run on the prefill
   instance, which streams that weight per forward);
2. the eager ``Glm5MoE3DBuffers``, dead once the decode CUDA graphs are
   captured and active (the graph has its own buffer pool) and re-created on
   demand at eager entry;
3. the WP4 container's re-quantized wq_b copy, dead once the merged
   ``[q_b | wq_b]`` operand is live.

A release that frees something still reachable is silent corruption, and a
release that forgets to re-create is a silent capability loss, so both halves
are pinned here. The behavior tests exec the released classes/methods out of
source: importing the GLM-5 model or wrapper modules pulls in Triton, FA3 and
DeepGEMM, and these invariants are pure Python plus torch allocation.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest
import torch


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODEL_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/model.py"
_WRAPPERS_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/wrappers.py"
_PSM_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/Parallel_Strategy_Manager.py"
_SEGMENTS_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/cuda_graph_segments.py"
_SELECTOR_SRC = _REPO_ROOT / "batchgen/attention/dsa/glm5_decode_selector.py"
_WORKER_SRC = _REPO_ROOT / "batchgen/batchgen_worker.py"
_WQB_SRC = _REPO_ROOT / "batchgen_kernels/attention/dsa/fused_indexer_score.py"


# ---------------------------------------------------------------------------
# source helpers
# ---------------------------------------------------------------------------

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


def _branch(node, test_src: str) -> ast.If:
    for sub in ast.walk(node):
        if isinstance(sub, ast.If) and ast.unparse(sub.test) == test_src:
            return sub
    raise AssertionError(f"no `if {test_src}:` in {getattr(node, 'name', node)}")


def _calls(node) -> list[str]:
    body = node.body if isinstance(node, ast.If) else [node]
    return [
        ast.unparse(sub.func)
        for stmt in body
        for sub in ast.walk(stmt)
        if isinstance(sub, ast.Call)
    ]


def _calls_in(statements) -> list[str]:
    return [
        ast.unparse(sub.func)
        for stmt in statements
        for sub in ast.walk(stmt)
        if isinstance(sub, ast.Call)
    ]


def _exec_classdef(path: pathlib.Path, class_name: str, namespace: dict):
    """Compile one ClassDef out of ``path`` into ``namespace`` and return it."""
    cls = _top_level(_tree(path), class_name)
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace[class_name]


def _exec_method(path: pathlib.Path, class_name: str, method: str, namespace: dict):
    """Compile one method out of ``path`` as a plain function (self is explicit)."""
    fn = _method(_tree(path), class_name, method)
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace[method]


# ===========================================================================
# 1. decode FP8 kv_b_proj
# ===========================================================================

def _release_kv_b():
    return _exec_method(
        _WRAPPERS_SRC,
        "GLM5AttnWrapper",
        "release_decode_kv_b_proj",
        {"torch": torch},
    )


def _fake_attn_wrapper(rows: int = 64, cols: int = 16, absorb=object()):
    weight = torch.nn.Parameter(
        torch.zeros(rows, cols, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    return types.SimpleNamespace(
        layer_idx=7,
        _fp8_absorb_weights=absorb,
        _decode_kv_b_proj_released=False,
        fp8_kv_b_proj=weight.data,
        module=types.SimpleNamespace(
            kv_b_proj=types.SimpleNamespace(weight=weight),
        ),
    )


def test_kv_b_release_frees_the_weight_and_keeps_the_absorb_pair():
    release = _release_kv_b()
    wrapper = _fake_attn_wrapper(rows=64, cols=16)
    absorb = wrapper._fp8_absorb_weights

    freed = release(wrapper)

    assert freed == 64 * 16, "one e4m3 byte per element must be reported"
    assert wrapper.fp8_kv_b_proj is None
    assert wrapper.module.kv_b_proj.weight.data.numel() == 0
    assert wrapper.module.kv_b_proj.weight.data.dtype is torch.float8_e4m3fn
    assert wrapper._decode_kv_b_proj_released is True
    # The absorb pair is the decode authority and must be untouched.
    assert wrapper._fp8_absorb_weights is absorb


def test_kv_b_release_is_idempotent():
    release = _release_kv_b()
    wrapper = _fake_attn_wrapper()
    assert release(wrapper) > 0
    assert release(wrapper) == 0


def test_kv_b_release_refuses_to_run_before_the_absorb_pair_exists():
    release = _release_kv_b()
    wrapper = _fake_attn_wrapper(absorb=None)
    with pytest.raises(RuntimeError, match="initialize_decode_absorb must run"):
        release(wrapper)
    # Nothing may have been dropped on the failing path.
    assert wrapper.module.kv_b_proj.weight.data.numel() == 64 * 16
    assert wrapper.fp8_kv_b_proj is not None


def test_absorb_init_fails_loud_after_the_weight_was_released():
    fn = _method(_tree(_WRAPPERS_SRC), "GLM5AttnWrapper", "initialize_decode_absorb")
    guard = _branch(fn, "self._decode_kv_b_proj_released")
    raised = [n for n in ast.walk(guard) if isinstance(n, ast.Raise)]
    assert raised, "the released state must fail loud, not silently rebuild"
    message = ast.unparse(raised[0])
    assert "release_decode_kv_b_proj" in message
    assert "configure_decoding" in message, "the message must name the recovery"
    # The guard runs before anything reads the weight.
    weight_reads = [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and ast.unparse(n).endswith("kv_b_proj.weight.data")
    ]
    assert weight_reads, "absorb init still dequantizes from kv_b_proj"
    assert fn.body.index(guard) < len(fn.body) - 1


def test_only_the_decode_configure_releases_kv_b_proj():
    fn = _method(_tree(_PSM_SRC), "GLM5ParallelStrategyManager", "_config_attn_module")
    # The phase is read explicitly, not inferred from `persistent` alone.
    assigns = [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign) and "decode_phase" in ast.unparse(n.targets[0])
    ]
    assert any("'decode'" in a or '"decode"' in a for a in assigns), assigns

    persistent = _branch(fn, "persistent")
    decode_only = _branch(persistent, "decode_phase")
    assert "wrapper.release_decode_kv_b_proj" in _calls(decode_only)
    # ... and only there.
    outside = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("release_decode_kv_b_proj")
    ]
    assert len(outside) == 1
    # Absorb init must precede the release in the same persistent block.
    persistent_calls = _calls(persistent)
    assert persistent_calls.index(
        "wrapper.initialize_decode_absorb"
    ) < persistent_calls.index("wrapper.release_decode_kv_b_proj")


def test_kv_b_proj_is_phase_managed_so_the_skeleton_can_never_alias_it():
    """The release is only safe because kv_b_proj is not shared with prefill."""
    tree = _tree(_PSM_SRC)
    for configure in ("configure_prefill", "configure_decoding"):
        fn = _method(tree, "GLM5ParallelStrategyManager", configure)
        src = ast.unparse(fn)
        assert "self_attn.named_parameters()" in src
        # Only the indexer and the two attention norms are routed to the
        # skeleton; every other self_attn parameter -- kv_b_proj included --
        # lands in state_dict_name_map.
        assert "name.startswith('indexer.')" in src
        assert "('q_a_layernorm.weight', 'kv_a_layernorm.weight')" in src
        assert "self.state_dict_name_map[tensor_full_name]" in src

    binding = _method(
        tree, "GLM5ParallelStrategyManager", "_check_shared_skeleton_binding"
    )
    overlap = ast.unparse(binding)
    assert "set(bound) & set(self.state_dict_name_map)" in overlap
    assert "phase-managed keys" in overlap, (
        "binding a phase-managed key to the shared skeleton must stay fatal"
    )


# ===========================================================================
# 2. eager Glm5MoE3DBuffers
# ===========================================================================

def _buffers_class():
    from batchgen.models.glm.glm5.moe_ragged import (
        make_quant_buffers,
        ragged_row_capacity,
    )

    namespace = {
        "torch": torch,
        "logging": __import__("logging"),
        "_glm5_ragged_row_capacity": ragged_row_capacity,
        "_glm5_make_quant_buffers": make_quant_buffers,
    }
    return _exec_classdef(_MODEL_SRC, "Glm5MoE3DBuffers", namespace)


def _make_buffers(cls, max_global_bsz: int = 256):
    return cls(
        E_local=4,
        max_global_bsz=max_global_bsz,
        H=256,
        N_inter=128,
        topk=8,
        num_tokens_per_rank=16,
        device=torch.device("cpu"),
    )


def _snapshot(buf, attrs):
    return {
        name: (tuple(getattr(buf, name).shape), getattr(buf, name).dtype)
        for name in attrs
    }


def test_eager_buffers_release_then_recreate_bit_identically():
    cls = _buffers_class()
    buf = _make_buffers(cls)
    attrs = cls._BUFFER_ATTRS

    before = _snapshot(buf, attrs)
    held = buf.nbytes()
    assert held > 0 and not buf.released

    freed = buf.release()
    assert freed == held
    assert buf.released and buf.nbytes() == 0
    assert all(getattr(buf, name) is None for name in attrs)
    assert buf.release() == 0, "release must be idempotent"
    # The shape state survives: it is what the re-create is derived from.
    assert buf.capacity > 0 and buf.max_global_bsz == 256

    allocated = buf.ensure_resident()
    assert allocated == held
    assert not buf.released
    assert _snapshot(buf, attrs) == before
    assert buf.ensure_resident() == 0, "ensure_resident must be idempotent"

    # The documented initializers are reproduced, not left as garbage.
    assert float(buf.x_scale.abs().sum()) == 0.0
    assert float(buf.inter_scale.abs().sum()) == 0.0
    assert int(buf.topk_pos.min()) == -1 and int(buf.topk_pos.max()) == -1
    assert float(buf.dispatched_x.abs().sum()) == 0.0
    assert float(buf.expert_out.abs().sum()) == 0.0
    assert int(buf.cu_seqlens.abs().sum()) == 0
    assert int(buf.expert_counts.abs().sum()) == 0


def test_released_buffers_grow_on_the_next_recreate_not_in_place():
    cls = _buffers_class()
    buf = _make_buffers(cls, max_global_bsz=256)
    small_capacity = buf.capacity
    buf.release()

    buf.resize_if_needed(1024)
    assert buf.released, "a released object must not allocate inside resize"
    assert buf.max_global_bsz == 1024
    assert buf.capacity > small_capacity

    buf.ensure_resident()
    assert buf.all_tokens.shape[0] == 1024
    assert buf.dispatched_x.shape[0] == buf.capacity
    assert buf.topk_pos.shape[0] == 1024 * buf.topk


def test_buffer_attrs_covers_exactly_what_the_allocators_assign():
    tree = _tree(_MODEL_SRC)
    assigned = set()
    for helper in ("_alloc_comm_buffers", "_alloc_row_buffers"):
        fn = _method(tree, "Glm5MoE3DBuffers", helper)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                for leaf in ([target] if not isinstance(target, ast.Tuple) else target.elts):
                    text = ast.unparse(leaf)
                    if text.startswith("self."):
                        assigned.add(text[len("self."):])

    cls = _buffers_class()
    assert set(cls._BUFFER_ATTRS) == assigned, (
        "release()/ensure_resident() must cover every allocated buffer"
    )


def test_eager_entry_recreates_before_touching_the_buffers():
    fn = _method(_tree(_MODEL_SRC), "Glm5MoE", "_forward_decode_3d")
    calls = _calls(fn)
    assert "buf.ensure_resident" in calls
    assert calls.index("buf.ensure_resident") < calls.index("buf.resize_if_needed")
    # No re-creation inside a loop: the allocation sits at eager entry.
    for loop in ast.walk(fn):
        if isinstance(loop, (ast.For, ast.While)):
            assert "buf.ensure_resident" not in _calls(loop)


def test_release_keeps_the_3d_path_selectable():
    tree = _tree(_MODEL_SRC)
    fn = _method(tree, "Glm5MoE", "release_eager_decode_buffers")
    src = ast.unparse(fn)
    assert "buf.release()" in src
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            assert "_3d_buf" not in ast.unparse(node.targets[0]), (
                "nulling Glm5MoE._3d_buf would demote every decode step to the "
                "non-3D fallback, graph replay included"
            )
    # ... because the dispatch gate reads it.
    gate = ast.unparse(_method(tree, "Glm5MoE", "_forward_decode"))
    assert "Glm5MoE._3d_buf is not None" in gate


def test_set_num_tokens_per_rank_is_released_safe():
    fn = _method(_tree(_MODEL_SRC), "Glm5MoE", "set_num_tokens_per_rank")
    guard = _branch(fn, "buf.released")
    src = ast.unparse(guard)
    assert "buf.num_tokens_per_rank = num_tokens_per_rank" in src
    assert isinstance(guard.body[-1], ast.Return), (
        "a released buffer set must not be indexed for its padded shape"
    )
    # The in-place rebind that follows must stay behind the guard.
    body = ast.unparse(fn)
    assert body.index("if buf.released") < body.index("buf.padded.shape[0]")


def test_the_engine_releases_the_eager_buffers_only_after_capture():
    tree = _tree(_WORKER_SRC)
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("release_eager_decode_buffers")
    ]
    assert len(hits) == 1, "exactly one engine-side release site"
    src = _WORKER_SRC.read_text()
    call_at = src.index("release_eager_decode_buffers")
    capture_at = src.index("manager.warmup_and_capture_buckets")
    ready_at = src.index("whole-model CUDA graph ready in")
    assert capture_at < call_at < ready_at, (
        "the release belongs between the capture loop and the ready log, so a "
        "capture that OOMs returns before it and keeps the eager buffers"
    )
    # It iterates the MoE layers the capture path already collected, so the
    # engine gains no new model-specific name (layer-boundary ratchet).
    loop_src = src[src.rindex("for ", 0, call_at):call_at]
    assert "moe_layers" in loop_src


# ===========================================================================
# 3. the third resident wq_b copy
# ===========================================================================

def _wqb_class():
    namespace = {"torch": torch, "FP8IndexerWeightsCUDA": None}
    return _exec_classdef(_WQB_SRC, "FP8WqbWeightsCUDA", namespace)


def _fake_wqb(cls, n: int = 64, k: int = 128):
    obj = object.__new__(cls)
    obj.inner = types.SimpleNamespace(
        w_fp8=torch.zeros(n, k, dtype=torch.float8_e4m3fn),
        w_scale=torch.zeros(n // 32, k // 128, dtype=torch.float32),
        tma_desc=object(),
        N=n,
        K=k,
        block_k=128,
    )
    obj._shape = (n, k, 128)
    return obj


def test_wqb_release_frees_the_bytes_and_fails_loud_afterwards():
    cls = _wqb_class()
    obj = _fake_wqb(cls)
    expected = 64 * 128 + (64 // 32) * (128 // 128) * 4

    assert not obj.released
    assert obj.release() == expected
    assert obj.released and obj.inner is None
    assert obj.release() == 0

    for name in ("w_fp8", "w_scale", "tma_desc"):
        with pytest.raises(RuntimeError, match="released"):
            getattr(obj, name)
    # Static geometry still answers: callers that only size buffers do not
    # need the bytes, and the error must name the real problem.
    assert (obj.N, obj.K, obj.block_k) == (64, 128, 128)


def test_merge_init_releases_the_separate_operand():
    tree = _tree(_WRAPPERS_SRC)
    init = _method(tree, "GLM5AttnWrapper", "initialize_fused_kernels")
    merged = _branch(init, "glm5_qb_wqb_merge_enabled()")
    calls = _calls(merged)
    assert calls == [
        "self._register_q_b_wq_b_merged",
        "self._release_separate_wq_b_operand",
    ], calls

    release = _method(tree, "GLM5AttnWrapper", "_release_separate_wq_b_operand")
    src = ast.unparse(release)
    # A wrapper without a merged operand (a streaming prefill wrapper) keeps
    # its container: there the separate version IS the wq_b authority.
    assert "self._fp8_q_b_wq_b_proj is None or self._fused_wqb_weights is None" in src
    assert "return 0" in src
    assert "self._fused_wqb_weights.release()" in src
    for node in ast.walk(release):
        if isinstance(node, ast.Assign):
            assert "_fused_wqb_weights" not in ast.unparse(node.targets[0]), (
                "the container object must survive: _init_fused_kernels and the "
                "engine both require it to be non-None"
            )


def test_the_captured_segment_cannot_select_the_separate_wq_b_gemm():
    tree = _tree(_SEGMENTS_SRC)
    init = _method(tree, "Glm5FullDsaAttnSegment", "__init__")
    selected = [
        ast.unparse(n.value)
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "self._q_b_wq_b_merged"
    ]
    assert selected == ["self._q_b_wq_b_merge_selected and (not self.all_short)"], selected

    forward = _method(tree, "Glm5FullDsaAttnSegment", "forward")
    all_short_block = _branch(forward, "not self.all_short")
    merged_branch = _branch(all_short_block, "self._q_b_wq_b_merged")
    # The wq_b GEMM over the container lives ONLY in the else of that branch,
    # and that branch is True whenever this block runs with the merge on.
    assert "cuda_wq_b_proj_out" in _calls_in(merged_branch.orelse)
    assert "cuda_wq_b_proj_out" not in _calls_in(merged_branch.body)
    gemm_sites = [
        n for n in ast.walk(forward)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "cuda_wq_b_proj_out"
    ]
    assert len(gemm_sites) == 1, "one non-merged site only, inside that else"

    # setup_static_buffers already relies on the same unreachability: it hands
    # the merged layout 1-element placeholders for that version's scratch.
    scratch = _top_level(tree, "_merged_wq_b_scratch_placeholders")
    doc = " ".join((ast.get_docstring(scratch) or "").split())
    assert "has no reader once wq_b runs inside the merged DeepGEMM" in doc
    assert "1-element placeholders" in doc


def test_the_eager_scorer_always_gets_the_merged_wq_b_slice():
    tree = _tree(_SELECTOR_SRC)
    fn = _top_level(tree, "build_glm5_dsa_flashmla_inputs")
    merged = _branch(fn, "q_b_wq_b is not None")
    assigned = [
        ast.unparse(n.targets[0])
        for n in ast.walk(merged)
        if isinstance(n, ast.Assign)
    ]
    assert "q_index_raw" in assigned, (
        "an indexer layer with the merge on must produce q_index_raw, or the "
        "scorer falls back to the container's own wq_b GEMM"
    )
    # Both scorer call sites forward it when it exists.
    select = _top_level(tree, "_select_glm5_dsa_indices")
    sites = [
        n for n in ast.walk(select)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func).endswith("score_and_select_paged")
    ]
    assert len(sites) == 2
    for site in sites:
        assert "q_index_raw" in ast.unparse(site)


# ===========================================================================
# 4. the blocker that keeps the full expert scale upload
# ===========================================================================

def test_grouped_prefill_reads_every_expert_scale_from_the_device():
    """Why the FP8 expert ``weight_scale_inv`` upload cannot be narrowed.

    The decode consumer (``Glm5MoE.init``) indexes only this rank's resident
    experts, but grouped PREFILL stages a DEVICE POINTER for every expert of
    every layer, and both phase instances share one ``_device_scales`` cache.
    Filtering the upload to the local shard therefore breaks prefill with a
    KeyError, not a slow path. Pinned so the recovery attempt is not repeated
    without first splitting that cache per phase.
    """
    tree = _tree(_MODEL_SRC)
    prep = _method(tree, "Glm5MoE", "_prefill_prepare_weights")
    loops = [
        n for n in ast.walk(prep)
        if isinstance(n, ast.For) and "self.experts" in ast.unparse(n.iter)
    ]
    assert loops, "grouped prefill still walks the expert list"
    body = ast.unparse(loops[0])
    assert "enumerate(self.experts)" in ast.unparse(loops[0].iter), (
        "the loop must still cover ALL experts, not a local slice"
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert f"'{name}.weight_scale_inv'" in body
    assert "data_ptr()" in body, "the scales are consumed as device pointers"

    decode = _method(tree, "Glm5MoE", "init")
    decode_src = ast.unparse(decode)
    assert "range(self.routed_expert_start_idx, persistent_end)" in decode_src

    psm = _method(
        _tree(_PSM_SRC), "GLM5ParallelStrategyManager", "_config_expert_module"
    )
    assert "self._device_scale(" in ast.unparse(psm), (
        "both phases take expert scales from the one shared device cache"
    )
