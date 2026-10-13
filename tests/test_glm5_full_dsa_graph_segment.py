"""Contract tests for the GLM-5 whole-model DSA graph segments.

Two tiers live in this file:

* **Source-contract tests** run everywhere, including CPU-only dev machines.
  They parse ``cuda_graph_segments.py`` / ``reuse_topk_segment.py`` instead of
  importing them, because importing pulls in FA3 (``flash_attn_interface``),
  Triton and DeepGEMM. They pin the direct-index FA3 contract: the buffer
  set, the dropped FlashMLA plumbing, and the two ``all_short`` branches.
* **Runtime tests** need a GPU (they capture real CUDA graphs over stubbed
  kernels) and are skipped otherwise, exactly as before.
"""

import ast
import collections
import pathlib
import types

import pytest
import torch


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SEGMENTS_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/cuda_graph_segments.py"
_REUSE_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/reuse_topk_segment.py"

try:
    from batchgen.cuda_graph.graph_manager import BatchSizeBucketing, CUDAGraphManager
    from batchgen.models.glm.glm5 import cuda_graph_segments as segments
    from batchgen.models.glm.glm5.cuda_graph_segments import (
        Glm5FullDsaAttnSegment,
        make_glm5_full_dsa_graph_segment_name,
    )
    from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - CPU dev boxes have no FA3/Triton
    BatchSizeBucketing = CUDAGraphManager = segments = None
    Glm5FullDsaAttnSegment = make_glm5_full_dsa_graph_segment_name = None
    GLM5AttnWrapper = None
    _IMPORT_ERROR = exc


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or _IMPORT_ERROR is not None,
    reason=f"CUDA and the GPU-only segment dependencies are required ({_IMPORT_ERROR})",
)


# --------------------------------------------------------------------------- #
# Source-contract helpers (no import of the segment modules)
# --------------------------------------------------------------------------- #

def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _top_level(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no top-level `{name}`")


def _dataclass_field_names(tree: ast.Module, class_name: str) -> list[str]:
    cls = _top_level(tree, class_name)
    return [
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]


def _name_set_literal(tree: ast.Module, name: str) -> set[str]:
    """Collect the string elements of a top-level ``frozenset({...})`` binding."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name not in targets:
            continue
        return {
            elt.value
            for elt in ast.walk(node.value)
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
    raise AssertionError(f"no top-level set binding `{name}`")


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    cls = _top_level(tree, class_name)
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == method_name:
            return stmt
    raise AssertionError(f"no `{class_name}.{method_name}`")


def _param_defaults(fn: ast.FunctionDef) -> dict[str, str]:
    """Map each parameter that has a default to the source of that default."""
    args = fn.args
    out: dict[str, str] = {}
    positional = args.posonlyargs + args.args
    for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        out[arg.arg] = ast.unparse(default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            out[arg.arg] = ast.unparse(default)
    return out


def _call_names(node: ast.AST) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            counts[ast.unparse(sub.func)] += 1
    return counts


def _assigned_value(node: ast.AST, name: str) -> str:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in sub.targets
        ):
            return ast.unparse(sub.value)
    raise AssertionError(f"no assignment to `{name}`")


def _call_kwargs(node: ast.AST, callee: str) -> dict[str, str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == callee:
            return {kw.arg: ast.unparse(kw.value) for kw in sub.keywords if kw.arg}
    raise AssertionError(f"no call to `{callee}`")


def _branch(fn: ast.FunctionDef, test_src: str) -> ast.If:
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ast.unparse(node.test) == test_src:
            return node
    raise AssertionError(f"no `if {test_src}:` in `{fn.name}`")


_SCORING_CALLS = (
    "head_gates_out",
    "cuda_wq_b_proj_out",
    "rope_hadamard_q_out",
    "fused_paged_score_and_topk_with_slots_out",
)
_DROPPED_FLASHMLA_CALLS = (
    "pack_flashmla_query_out",
    "select_mla_kv_for_flashmla_bf16_out",
    "run_prepared_sparse_flash_mla_decode",
    "prepare_sparse_flash_mla_decode_inputs",
)


# --------------------------------------------------------------------------- #
# Source-contract tests: buffers
# --------------------------------------------------------------------------- #

def test_full_dsa_buffers_hold_selected_token_ids_not_a_gathered_slab():
    fields = _dataclass_field_names(_tree(_SEGMENTS_SRC), "_Glm5FullDsaSegmentBuffers")
    assert "selected_token_ids" in fields
    assert "selected_lengths" in fields
    # The BF16 selected-KV slab, the FlashMLA query pack and its prepared
    # handle are gone with the gather; row_modes had no consumer left once
    # the transform decided dense-vs-topk internally.
    for gone in ("selected_mla_kv", "query_states", "prepared_flashmla", "row_modes"):
        assert gone not in fields, gone


@pytest.mark.parametrize(
    "src,class_name",
    [(_SEGMENTS_SRC, "Glm5FullDsaAttnSegment"), (_REUSE_SRC, "Glm5ReuseTopkAttnSegment")],
)
def test_selected_token_ids_is_allocated_as_a_bucket_by_topk_int32_table(src, class_name):
    alloc = _method(_tree(src), class_name, "setup_static_buffers")
    expr = _assigned_value(alloc, "selected_token_ids")
    assert expr.startswith("torch.empty(bucket_size, self.index_topk")
    assert "dtype=torch.int32" in expr
    # The gathered BF16 slab must not be allocated anywhere any more.
    body = ast.unparse(alloc)
    for gone in ("selected_mla_kv", "query_states", "prepared_flashmla", "row_modes"):
        assert gone not in body, gone


def test_full_dsa_field_lists_exactly_cover_the_buffer_dataclass():
    """The view/rebuild split must stay total — the module asserts this at
    import time; assert it from source too so a CPU run catches drift."""
    tree = _tree(_SEGMENTS_SRC)
    fields = set(_dataclass_field_names(tree, "_Glm5FullDsaSegmentBuffers"))
    split = (
        _name_set_literal(tree, "_GLM5_DSA_VIEW_FIELDS")
        | _name_set_literal(tree, "_GLM5_DSA_REBUILT_FIELDS")
    )
    assert fields ^ split == set()
    assert "prepared_flashmla" not in _name_set_literal(tree, "_GLM5_DSA_REBUILT_FIELDS")
    assert "selected_token_ids" in _name_set_literal(tree, "_GLM5_DSA_VIEW_FIELDS")


def test_reuse_field_lists_exactly_cover_the_buffer_dataclass():
    fields = set(_dataclass_field_names(_tree(_SEGMENTS_SRC), "_Glm5FullDsaSegmentBuffers"))
    reuse = _tree(_REUSE_SRC)
    split = (
        _name_set_literal(reuse, "_GLM5_REUSE_VIEW_FIELDS")
        | _name_set_literal(reuse, "_GLM5_REUSE_PLACEHOLDER_FIELDS")
        | _name_set_literal(reuse, "_GLM5_REUSE_SPECIAL_FIELDS")
    )
    assert fields ^ split == set()
    assert _name_set_literal(reuse, "_GLM5_REUSE_SPECIAL_FIELDS") == {"top_k_indices"}
    assert "selected_token_ids" in _name_set_literal(reuse, "_GLM5_REUSE_VIEW_FIELDS")


# --------------------------------------------------------------------------- #
# Source-contract tests: static input specs / dead FlashMLA metadata
# --------------------------------------------------------------------------- #

def test_full_dsa_static_input_specs_drop_the_flashmla_metadata():
    specs = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "get_static_input_specs")
    src = ast.unparse(specs)
    assert "flashmla" not in src
    for required in ("hidden_states", "position_ids", "cache_seqlens", "num_valid_tokens"):
        assert required in src, required


def test_full_dsa_initialize_static_inputs_no_longer_builds_metadata():
    init = _method(
        _tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "initialize_static_inputs"
    )
    assert "flashmla" not in ast.unparse(init)


@pytest.mark.parametrize(
    "src,class_name",
    [(_SEGMENTS_SRC, "Glm5FullDsaAttnSegment"), (_REUSE_SRC, "Glm5ReuseTopkAttnSegment")],
)
def test_segment_forward_has_no_flashmla_metadata(src, class_name):
    """The FlashMLA scheduler metadata is dead through the whole graph chain:
    neither forward accepts it, and no FlashMLA kernel is called."""
    fwd = _method(_tree(src), class_name, "forward")
    params = {a.arg for a in fwd.args.args + fwd.args.kwonlyargs}
    assert not {p for p in params if "flashmla" in p}, params
    assert not any(isinstance(stmt, ast.Delete) for stmt in fwd.body)
    # No transitional helper, cache, or comment survives on the class either.
    cls_src = ast.get_source_segment(src.read_text(), _top_level(_tree(src), class_name))
    assert "flashmla" not in cls_src.lower()
    calls = _call_names(fwd)
    for dropped in _DROPPED_FLASHMLA_CALLS:
        assert dropped not in calls, dropped


# --------------------------------------------------------------------------- #
# Source-contract tests: the two FA3 branches
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "src,class_name",
    [(_SEGMENTS_SRC, "Glm5FullDsaAttnSegment"), (_REUSE_SRC, "Glm5ReuseTopkAttnSegment")],
)
def test_all_short_is_an_opt_in_constructor_flag(src, class_name):
    init = _method(_tree(src), class_name, "__init__")
    assert _param_defaults(init).get("all_short") == "False"


@pytest.mark.parametrize(
    "src,class_name",
    [(_SEGMENTS_SRC, "Glm5FullDsaAttnSegment"), (_REUSE_SRC, "Glm5ReuseTopkAttnSegment")],
)
def test_attention_branches_on_all_short(src, class_name):
    fwd = _method(_tree(src), class_name, "forward")
    branch = _branch(fwd, "self.all_short")
    short_calls = _call_names(ast.Module(body=branch.body, type_ignores=[]))
    long_calls = _call_names(ast.Module(body=branch.orelse, type_ignores=[]))

    assert "self._run_all_short_fa3" in short_calls
    assert "transform_selected_positions_out" not in short_calls
    assert "self._run_selected_fa3" not in short_calls

    assert "transform_selected_positions_out" in long_calls
    assert "self._run_selected_fa3" in long_calls
    assert "self._run_all_short_fa3" not in long_calls

    # Neither helper may be reachable outside the branch.
    total = _call_names(fwd)
    assert total["self._run_all_short_fa3"] == short_calls["self._run_all_short_fa3"]
    assert total["self._run_selected_fa3"] == long_calls["self._run_selected_fa3"]
    assert total["transform_selected_positions_out"] == (
        long_calls["transform_selected_positions_out"]
    )


def test_all_short_skips_every_scoring_stage():
    fwd = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    guarded = _branch(fwd, "not self.all_short")
    guarded_calls = _call_names(guarded)
    total_calls = _call_names(fwd)
    for stage in _SCORING_CALLS:
        assert guarded_calls[stage] >= 1, stage
        # Same count inside the guard as in the whole forward => unreachable
        # when all_short is True.
        assert total_calls[stage] == guarded_calls[stage], stage
    # positions_expanded is only materialized for the indexer query rope.
    assert "buffers.positions_expanded" in ast.unparse(guarded)
    assert ast.unparse(fwd).count("buffers.positions_expanded") == (
        ast.unparse(guarded).count("buffers.positions_expanded")
    )


def test_selected_fa3_call_shape_is_page_size_one_over_selected_token_ids():
    helper = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "_run_selected_fa3")
    kwargs = _call_kwargs(helper, "_fa3_with_kvcache")
    assert kwargs["q"] == "buffers.q_rope_4d.squeeze(2).unsqueeze(1)"
    assert kwargs["qv"] == "buffers.absorbed_q.unsqueeze(1)"
    assert kwargs["page_table"] == "buffers.selected_token_ids"
    assert kwargs["cache_seqlens"] == "buffers.selected_lengths"
    # Selected token IDs are absolute: no slot indirection may be passed.
    assert "cache_batch_idx" not in kwargs
    assert kwargs["k_cache"] == "flat_kv[..., attn.kv_lora_rank:]"
    assert kwargs["v_cache"] == "flat_kv[..., :attn.kv_lora_rank]"
    assert kwargs["causal"] == "True"
    assert kwargs["num_splits"] == "0"
    assert kwargs["return_softmax_lse"] == "False"
    assert kwargs["softmax_scale"] == "float(attn.softmax_scale)"
    assert "self.primary_blocked_k.view(-1, 1, 1, self.primary_blocked_k.shape[-1])" in (
        ast.unparse(helper)
    )


def test_all_short_fa3_call_reads_the_resident_paged_cache():
    helper = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "_run_all_short_fa3")
    kwargs = _call_kwargs(helper, "_fa3_with_kvcache")
    assert kwargs["q"] == "buffers.q_rope_4d.squeeze(2).unsqueeze(1)"
    assert kwargs["qv"] == "buffers.absorbed_q.unsqueeze(1)"
    assert kwargs["page_table"] == "self.primary_page_table"
    assert kwargs["cache_batch_idx"] == "buffers.safe_primary_slot_indices"
    assert kwargs["cache_seqlens"] == "buffers.safe_cache_seqlens"
    assert kwargs["k_cache"] == "self.primary_blocked_k[..., attn.kv_lora_rank:]"
    assert kwargs["v_cache"] == "self.primary_blocked_k[..., :attn.kv_lora_rank]"
    assert kwargs["causal"] == "True"
    assert kwargs["num_splits"] == "0"


@pytest.mark.parametrize(
    "src,class_name",
    [(_SEGMENTS_SRC, "Glm5FullDsaAttnSegment"), (_REUSE_SRC, "Glm5ReuseTopkAttnSegment")],
)
def test_segment_init_fails_loud_without_fa3(src, class_name):
    init = _method(_tree(src), class_name, "__init__")
    src_text = ast.unparse(init)
    assert "_fa3_with_kvcache is None" in src_text
    assert "raise RuntimeError" in src_text


def test_fa3_symbol_is_an_optional_module_level_import():
    tree = _tree(_SEGMENTS_SRC)
    handlers = [
        node for node in tree.body
        if isinstance(node, ast.Try)
        and "flash_attn_interface" in ast.unparse(node)
    ]
    assert handlers, "FA3 must be imported once at module scope"
    body = ast.unparse(handlers[0])
    assert "flash_attn_with_kvcache as _fa3_with_kvcache" in body
    assert "_fa3_with_kvcache = None" in body


def test_transform_is_called_with_the_kernel_contract():
    fwd = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    kwargs = _call_kwargs(fwd, "transform_selected_positions_out")
    assert kwargs["page_size"] == "self.page_size"
    assert kwargs["primary_slot_indices"] == "buffers.safe_primary_slot_indices"
    assert kwargs["num_valid_tokens"] == "num_valid_tokens"
    reuse_fwd = _method(_tree(_REUSE_SRC), "Glm5ReuseTopkAttnSegment", "forward")
    reuse_kwargs = _call_kwargs(reuse_fwd, "transform_selected_positions_out")
    assert reuse_kwargs["primary_slot_indices"] == "buffers.safe_primary_slot_indices"
    # The reuse layer borrows ONLY the producer's top-k and transforms itself.
    assert "buffers.top_k_indices" in ast.unparse(reuse_fwd)


# --------------------------------------------------------------------------- #
# Runtime tests (GPU)
# --------------------------------------------------------------------------- #

class _Identity(torch.nn.Module):
    def __init__(self, size=None):
        super().__init__()
        if size is not None:
            self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = 1e-5

    def forward(self, x):
        return x


def _fake_linear_weight(out_features, in_features, device):
    values = torch.arange(
        out_features * in_features,
        device=device,
        dtype=torch.float32,
    ).view(out_features, in_features)
    return (values.remainder(7).sub(3).mul_(0.03)).to(torch.bfloat16)


def _build_fake_wrapper(device):
    hidden_size = 8
    q_lora_rank = 4
    num_heads = 2
    qk_nope_head_dim = 2
    qk_rope_head_dim = 2
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_lora_rank = 2
    index_head_dim = 3
    index_n_heads = 2

    indexer = types.SimpleNamespace(
        index_topk=3,
        index_head_dim=index_head_dim,
        index_n_heads=index_n_heads,
        k_norm=_Identity().to(device),
        weights_proj=torch.nn.Linear(hidden_size, index_n_heads, bias=False).to(device),
    )
    with torch.no_grad():
        indexer.weights_proj.weight.copy_(
            torch.arange(
                index_n_heads * hidden_size,
                device=device,
                dtype=torch.float32,
            ).view(index_n_heads, hidden_size).mul_(0.01)
        )

    def _fused_rope_hadamard_or_fallback(k_normed, positions, max_seqlen):
        del positions, max_seqlen
        return k_normed + 0.125

    indexer._fused_rope_hadamard_or_fallback = _fused_rope_hadamard_or_fallback

    attn = types.SimpleNamespace(
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        num_heads=num_heads,
        q_head_dim=q_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=kv_lora_rank,
        softmax_scale=0.25,
        indexer=indexer,
        q_a_layernorm=_Identity(q_lora_rank).to(device),
        kv_a_layernorm=types.SimpleNamespace(
            weight=torch.ones(kv_lora_rank, device=device, dtype=torch.bfloat16),
            eps=1e-5,
        ),
    )
    attn.q_a_proj = types.SimpleNamespace(
        weight=_fake_linear_weight(q_lora_rank, hidden_size, device)
    )
    attn.q_b_proj = types.SimpleNamespace(
        weight=_fake_linear_weight(num_heads * q_head_dim, q_lora_rank, device)
    )
    attn.kv_a_proj_with_mqa = types.SimpleNamespace(
        weight=_fake_linear_weight(kv_lora_rank + qk_rope_head_dim, hidden_size, device)
    )
    attn.o_proj = types.SimpleNamespace(
        weight=_fake_linear_weight(hidden_size, num_heads * kv_lora_rank, device)
    )

    weight_dequant_scale = {
        "q_a_proj.weight_scale_inv": torch.ones(1, device=device, dtype=torch.float32),
        "q_b_proj.weight_scale_inv": torch.ones(1, device=device, dtype=torch.float32),
        "kv_a_proj_with_mqa.weight_scale_inv": torch.ones(1, device=device, dtype=torch.float32),
        "o_proj.weight_scale_inv": torch.ones(1, device=device, dtype=torch.float32),
    }
    return types.SimpleNamespace(
        module=attn,
        layer_idx=0,
        weight_dequant_scale=weight_dequant_scale,
        _fp8_qkv_a_proj=torch.cat(
            (attn.q_a_proj.weight, attn.kv_a_proj_with_mqa.weight),
            dim=0,
        ).contiguous(),
        _fp8_qkv_a_scale=torch.ones(1, device=device, dtype=torch.float32),
        _indexer_cuda_weights=object(),
        _indexer_cuda_module=object(),
    )


def _patch_full_dsa_dependencies(
    monkeypatch, *, bucket_size, index_topk, v_dim, device, recorder=None
):
    """Stub every kernel the segment calls.

    ``recorder`` (optional) collects the stub invocations so a test can assert
    which stages ran and with which arguments.
    """
    calls = recorder if recorder is not None else {}
    calls.setdefault("fa3", [])
    calls.setdefault("transform", [])
    calls.setdefault("scoring", [])

    topk_template = torch.arange(index_topk, device=device, dtype=torch.int32).view(1, index_topk)
    topk_template = topk_template.expand(bucket_size, index_topk).contiguous()

    def fake_act_quant(x, num_valid_tokens=None, scale_tma_aligned=False):
        del num_valid_tokens, scale_tma_aligned
        return x.contiguous(), torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32)

    def fake_w8a8_deepgemm(a, a_scale, w, w_scale, c=None, disable_ue8m0_cast=True, recipe=None, out=None, num_valid_tokens=None, expected_m=None):
        del a_scale, w_scale, c, disable_ue8m0_cast, recipe, expected_m, num_valid_tokens
        result = a.float().matmul(w.float().t()).to(torch.bfloat16)
        if out is None:
            return result
        out.copy_(result)
        return out

    def fake_rmsnorm_rope(new_compressed_kv, q_rope, cos, sin, position_ids, weight, kv_lora_rank, qk_rope_head_dim, eps):
        del cos, sin, position_ids, weight, kv_lora_rank, qk_rope_head_dim, eps
        q_rope.add_(0.0625)
        return new_compressed_kv + 0.25

    def fake_rmsnorm(x, weight, eps, out=None):
        del weight, eps
        if out is None:
            return x.clone()
        out.copy_(x)
        return out

    def fake_make_scratch(batch, cols, cuda_module, device):
        del cuda_module
        return (
            torch.empty(batch, cols, dtype=torch.bfloat16, device=device),
            torch.empty(batch, 1, dtype=torch.float32, device=device),
            torch.empty(1, dtype=torch.uint8, device=device),
        )

    def fake_wk_proj(hidden_flat, weights, cuda_module, x_fp8, x_scale, tma_desc, out, num_valid_tokens=None):
        del weights, cuda_module, x_fp8, x_scale, tma_desc, num_valid_tokens
        out.copy_(hidden_flat[:, : out.shape[1]] + 0.5)
        return out

    def fake_head_gates(hidden_flat, weight, out, *, scale, num_valid_tokens=None):
        del weight, scale, num_valid_tokens
        calls["scoring"].append("head_gates_out")
        out.copy_(hidden_flat[:, : out.shape[1]].float() + 0.25)
        return out

    def fake_wq_b_proj(q_a_normed, weights, cuda_module, x_fp8, x_scale, tma_desc, out, num_valid_tokens=None):
        del weights, cuda_module, x_fp8, x_scale, tma_desc, num_valid_tokens
        calls["scoring"].append("cuda_wq_b_proj_out")
        out.copy_(q_a_normed[:, :1].expand_as(out) + 0.25)
        return out

    def fake_rope_hadamard_q(q_flat, cos, sin, positions, out):
        del cos, sin, positions
        calls["scoring"].append("rope_hadamard_q_out")
        out.copy_(q_flat + 0.03125)
        return out

    def fake_score_topk(q_index, aux_blocked_k, aux_page_table, aux_slot_indices, head_gates, cache_seqlens, agg_scores, top_k_indices, *, topk, page_size, max_seqlen, num_valid_tokens=None):
        del q_index, aux_blocked_k, aux_page_table, aux_slot_indices, head_gates, cache_seqlens, agg_scores, page_size, max_seqlen, num_valid_tokens
        assert topk == index_topk
        calls["scoring"].append("fused_paged_score_and_topk_with_slots_out")
        top_k_indices.copy_(topk_template[: top_k_indices.shape[0]])
        return top_k_indices

    def fake_transform(primary_page_table, cache_seqlens, long_topk_indices, physical_token_ids, selected_lengths, *, page_size, primary_slot_indices=None, num_valid_tokens=None):
        del primary_page_table, page_size, num_valid_tokens
        calls["transform"].append(
            {
                "physical_token_ids": physical_token_ids,
                "selected_lengths": selected_lengths,
                "primary_slot_indices": primary_slot_indices,
            }
        )
        # Deterministic stand-in for the logical->physical mapping: every
        # valid row maps to top-k + 1, padded rows (seqlen 0) select nothing.
        physical_token_ids.copy_(long_topk_indices + 1)
        selected_lengths.copy_(torch.clamp(cache_seqlens, max=index_topk))
        return physical_token_ids, selected_lengths

    def fake_fa3(*, q, k_cache, v_cache, qv, page_table, cache_seqlens, softmax_scale, causal, num_splits, return_softmax_lse, cache_batch_idx=None):
        del v_cache, softmax_scale, causal, num_splits, return_softmax_lse
        calls["fa3"].append(
            {
                "q": q,
                "k_cache": k_cache,
                "qv": qv,
                "page_table": page_table,
                "cache_seqlens": cache_seqlens,
                "cache_batch_idx": cache_batch_idx,
            }
        )
        base = qv[..., :v_dim]
        lengths = cache_seqlens.to(base.dtype).view(-1, 1, 1, 1)
        first_entry = page_table[:, :1].to(base.dtype).view(-1, 1, 1, 1)
        if cache_batch_idx is not None:
            # Dense branch: the page table is slot-major, not row-major.
            return base + lengths + k_cache.sum(dim=(0, 1, 2, 3))
        return base + lengths + first_entry

    def fake_q_absorb(q_nope, weights, absorbed_q, num_valid_tokens=None):
        del weights, num_valid_tokens
        absorbed_q.copy_(q_nope[..., : absorbed_q.shape[-1]])
        return absorbed_q

    def fake_out_absorb(attn_out, weights, attn_heads, num_valid_tokens=None):
        del weights, num_valid_tokens
        attn_heads.copy_(attn_out)
        return attn_heads

    monkeypatch.setattr(segments, "act_quant", fake_act_quant)
    monkeypatch.setattr(segments, "w8a8_deepgemm", fake_w8a8_deepgemm)
    monkeypatch.setattr(segments, "fused_rmsnorm", fake_rmsnorm)
    monkeypatch.setattr(segments, "_fused_rmsnorm_rope", fake_rmsnorm_rope)
    monkeypatch.setattr(segments, "make_fp8_activation_scratch", fake_make_scratch)
    monkeypatch.setattr(segments, "cuda_wk_proj_gemm_only_out", fake_wk_proj)
    monkeypatch.setattr(segments, "head_gates_out", fake_head_gates)
    monkeypatch.setattr(segments, "cuda_wq_b_proj_out", fake_wq_b_proj)
    monkeypatch.setattr(segments, "rope_hadamard_q_out", fake_rope_hadamard_q)
    monkeypatch.setattr(segments, "fused_paged_score_and_topk_with_slots_out", fake_score_topk)
    monkeypatch.setattr(segments, "transform_selected_positions_out", fake_transform)
    monkeypatch.setattr(segments, "_fa3_with_kvcache", fake_fa3)
    monkeypatch.setattr(segments, "fp8_q_absorb_out", fake_q_absorb)
    monkeypatch.setattr(segments, "fp8_out_absorb_out", fake_out_absorb)
    return calls


@requires_cuda
def test_glm5_full_dsa_segment_graph_replay_matches_eager_and_writes_kv(monkeypatch):
    device = torch.device("cuda")
    torch.manual_seed(0)
    bucket_size = 4
    actual_bsz = 2
    page_size = 4
    max_seqlen = 8
    wrapper = _build_fake_wrapper(device)
    attn = wrapper.module
    kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    index_dim = attn.indexer.index_head_dim
    _patch_full_dsa_dependencies(
        monkeypatch,
        bucket_size=bucket_size,
        index_topk=attn.indexer.index_topk,
        v_dim=attn.v_head_dim,
        device=device,
    )

    primary_cache = torch.zeros(4, page_size, 1, kv_dim, dtype=torch.bfloat16, device=device)
    aux_cache = torch.zeros(4, page_size, 1, index_dim, dtype=torch.bfloat16, device=device)
    primary_page_table = torch.tensor([[0, -1], [1, -1]], dtype=torch.int32, device=device)
    aux_page_table = torch.tensor([[2, -1], [3, -1]], dtype=torch.int32, device=device)
    cos = torch.ones(max_seqlen, attn.qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.zeros_like(cos)
    shared_buffers = {}
    segment = Glm5FullDsaAttnSegment(
        wrapper=wrapper,
        primary_blocked_k=primary_cache,
        aux_blocked_k=aux_cache,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=object(),
        absorb_weights=object(),
        cuda_module=object(),
        cos_table=cos,
        sin_table=sin,
        max_seqlen=max_seqlen,
        index_topk=attn.indexer.index_topk,
        page_size=page_size,
        aux_page_size=page_size,
        shared_buffers=shared_buffers,
    )
    assert segment.all_short is False

    hidden = torch.randn(actual_bsz, 1, attn.hidden_size, dtype=torch.bfloat16, device=device)
    position_ids = torch.tensor([[1], [2]], dtype=torch.int64, device=device)
    cache_seqlens = torch.tensor([2, 3], dtype=torch.int32, device=device)
    primary_slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    aux_slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    num_valid_tokens = torch.tensor([actual_bsz], dtype=torch.int32, device=device)

    def run_eager():
        primary_cache.zero_()
        aux_cache.zero_()
        outputs = segment.forward(
            hidden_states=hidden,
            position_ids=position_ids,
            cache_seqlens=cache_seqlens,
            primary_slot_indices=primary_slots,
            aux_slot_indices=aux_slots,
            num_valid_tokens=num_valid_tokens,
        )
        torch.cuda.synchronize()
        return (
            {key: value.detach().clone() for key, value in outputs.items()},
            primary_cache.detach().clone(),
            aux_cache.detach().clone(),
        )

    eager_outputs, eager_primary_cache, eager_aux_cache = run_eager()

    manager = CUDAGraphManager(BatchSizeBucketing([bucket_size]), device=device)
    manager.WARMUP_ITERATIONS = 1
    name = make_glm5_full_dsa_graph_segment_name(0)
    manager.register_segment(name, segment)
    manager.warmup_and_capture_buckets([bucket_size])
    assert bucket_size in shared_buffers
    assert bucket_size in segment._outputs
    captured = manager._graphs[name][bucket_size]
    # The dead FlashMLA metadata is no longer a static input of this segment.
    assert "flashmla_tile_scheduler_metadata" not in captured.static_inputs
    assert "flashmla_num_splits" not in captured.static_inputs
    assert torch.equal(
        captured.static_inputs["num_valid_tokens"],
        torch.ones(1, dtype=torch.int32, device=device),
    )
    assert torch.equal(
        captured.static_inputs["cache_seqlens"],
        torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        captured.static_inputs["primary_slot_indices"],
        torch.tensor([0, -1, -1, -1], dtype=torch.int32, device=device),
    )

    primary_cache.zero_()
    aux_cache.zero_()
    graph_outputs = manager.replay(
        name,
        actual_bsz,
        hidden_states=hidden,
        position_ids=position_ids,
        cache_seqlens=cache_seqlens,
        primary_slot_indices=primary_slots,
        aux_slot_indices=aux_slots,
    )
    torch.cuda.synchronize()
    graph_primary_cache = primary_cache.detach().clone()
    graph_aux_cache = aux_cache.detach().clone()

    for key in ("attn_output", "primary_k_tensor", "indexer_k_tensor"):
        assert torch.equal(graph_outputs[key], eager_outputs[key]), key
    assert torch.equal(graph_primary_cache, eager_primary_cache)
    assert torch.equal(graph_aux_cache, eager_aux_cache)
    assert torch.count_nonzero(graph_primary_cache[2:]).item() == 0
    assert torch.count_nonzero(graph_aux_cache[:2]).item() == 0
    buffers = shared_buffers[bucket_size]
    static_outputs = segment._outputs[bucket_size]
    expected_safe_slots = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device=device)
    expected_kv_slots = torch.tensor([0, 1, -1, -1], dtype=torch.int32, device=device)
    expected_safe_seqlens = torch.tensor([2, 3, 0, 0], dtype=torch.int32, device=device)
    assert torch.equal(buffers.safe_primary_slot_indices, expected_safe_slots)
    assert torch.equal(buffers.safe_aux_slot_indices, expected_safe_slots)
    assert torch.equal(buffers.kv_primary_slot_indices, expected_kv_slots)
    assert torch.equal(buffers.kv_aux_slot_indices, expected_kv_slots)
    assert torch.equal(buffers.safe_cache_seqlens, expected_safe_seqlens)
    assert torch.equal(captured.static_inputs["num_valid_tokens"], num_valid_tokens)
    # Padded rows select nothing and contribute nothing downstream.
    assert buffers.selected_token_ids.dtype == torch.int32
    assert tuple(buffers.selected_token_ids.shape) == (
        bucket_size,
        attn.indexer.index_topk,
    )
    assert torch.count_nonzero(buffers.selected_lengths[actual_bsz:]).item() == 0
    assert torch.count_nonzero(buffers.attn_heads[actual_bsz:]).item() == 0
    assert torch.count_nonzero(static_outputs.attn_output[actual_bsz:]).item() == 0

    manager.drop_bucket(bucket_size)
    assert bucket_size not in shared_buffers
    assert bucket_size not in segment._outputs


def _build_full_segment(monkeypatch, device, shared_buffers, *, all_short=False, recorder=None):
    wrapper = _build_fake_wrapper(device)
    attn = wrapper.module
    kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    _patch_full_dsa_dependencies(
        monkeypatch,
        bucket_size=4,
        index_topk=attn.indexer.index_topk,
        v_dim=attn.v_head_dim,
        device=device,
        recorder=recorder,
    )
    page_size = 4
    primary_cache = torch.zeros(4, page_size, 1, kv_dim, dtype=torch.bfloat16, device=device)
    aux_cache = torch.zeros(
        4, page_size, 1, attn.indexer.index_head_dim, dtype=torch.bfloat16, device=device
    )
    primary_page_table = torch.tensor([[0, -1], [1, -1]], dtype=torch.int32, device=device)
    aux_page_table = torch.tensor([[2, -1], [3, -1]], dtype=torch.int32, device=device)
    cos = torch.ones(8, attn.qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.zeros_like(cos)
    segment = Glm5FullDsaAttnSegment(
        wrapper=wrapper,
        primary_blocked_k=primary_cache,
        aux_blocked_k=aux_cache,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=object(),
        absorb_weights=object(),
        cuda_module=object(),
        cos_table=cos,
        sin_table=sin,
        max_seqlen=8,
        index_topk=attn.indexer.index_topk,
        page_size=page_size,
        aux_page_size=page_size,
        all_short=all_short,
        shared_buffers=shared_buffers,
    )
    return segment, primary_cache, aux_cache


@requires_cuda
def test_full_dsa_smaller_buckets_are_views_of_largest(monkeypatch):
    device = torch.device("cuda")
    shared_buffers = {}
    segment, _, _ = _build_full_segment(monkeypatch, device, shared_buffers)

    segment.setup_static_buffers(4)
    segment.setup_static_buffers(2)
    base, small = shared_buffers[4], shared_buffers[2]

    for name in sorted(segments._GLM5_DSA_VIEW_FIELDS):
        base_t, small_t = getattr(base, name), getattr(small, name)
        assert small_t.data_ptr() == base_t.data_ptr(), name
        assert small_t.shape[0] == 2, name
    # Rebuilt fields: fresh scratch tensors only; there is no prepared
    # FlashMLA handle to rebuild any more.
    assert small.q_x_fp8.data_ptr() != base.q_x_fp8.data_ptr()
    assert small.indexer_k_x_fp8.data_ptr() != base.indexer_k_x_fp8.data_ptr()
    assert not hasattr(base, "prepared_flashmla")
    assert small.selected_token_ids.data_ptr() == base.selected_token_ids.data_ptr()
    assert small.selected_token_ids.shape == (2, segment.index_topk)
    # Outputs follow the same slicing.
    out_base, out_small = segment._outputs[4], segment._outputs[2]
    assert out_small.attn_output.data_ptr() == out_base.attn_output.data_ptr()
    assert out_small.primary_k_tensor.data_ptr() == out_base.primary_k_tensor.data_ptr()
    assert out_small.attn_output.shape[0] == 2


@requires_cuda
def test_full_dsa_smaller_first_order_stays_correct(monkeypatch):
    # Not the production order (capture is largest-first): a smaller bucket
    # arriving first must not break anything — the larger bucket allocates a
    # fresh full set (with a warning), and later smaller buckets view the
    # largest base.
    device = torch.device("cuda")
    shared_buffers = {}
    segment, _, _ = _build_full_segment(monkeypatch, device, shared_buffers)
    segment.setup_static_buffers(2)
    segment.setup_static_buffers(4)
    assert (
        shared_buffers[4].q_a.data_ptr() != shared_buffers[2].q_a.data_ptr()
    )
    assert shared_buffers[4].q_a.shape[0] == 4
    segment.setup_static_buffers(3)
    assert shared_buffers[3].q_a.data_ptr() == shared_buffers[4].q_a.data_ptr()
    assert shared_buffers[3].q_a.shape[0] == 3


@requires_cuda
def test_reuse_segment_views_borrow_topk_and_placeholders(monkeypatch):
    from batchgen.models.glm.glm5 import reuse_topk_segment as reuse_mod
    from batchgen.models.glm.glm5.reuse_topk_segment import Glm5ReuseTopkAttnSegment

    device = torch.device("cuda")
    shared_full = {}
    full_segment, primary_cache, _ = _build_full_segment(monkeypatch, device, shared_full)
    # reuse_topk_segment bound these names at ITS import; patch them there too.
    monkeypatch.setattr(reuse_mod, "_fa3_with_kvcache", segments._fa3_with_kvcache)
    monkeypatch.setattr(
        reuse_mod,
        "transform_selected_positions_out",
        segments.transform_selected_positions_out,
    )
    full_segment.setup_static_buffers(4)
    full_segment.setup_static_buffers(2)

    attn = full_segment.attn
    shared_reuse = {}
    reuse_segment = Glm5ReuseTopkAttnSegment(
        wrapper=full_segment.wrapper,
        primary_blocked_k=primary_cache,
        primary_page_table=full_segment.primary_page_table,
        absorb_weights=object(),
        cos_table=full_segment.cos_table,
        sin_table=full_segment.sin_table,
        max_seqlen=8,
        index_topk=attn.indexer.index_topk,
        page_size=full_segment.page_size,
        topk_source=full_segment,
        shared_buffers=shared_reuse,
    )
    assert reuse_segment.all_short is False
    reuse_segment.setup_static_buffers(4)
    reuse_segment.setup_static_buffers(2)
    base, small = shared_reuse[4], shared_reuse[2]

    for name in sorted(reuse_mod._GLM5_REUSE_VIEW_FIELDS):
        base_t, small_t = getattr(base, name), getattr(small, name)
        assert small_t.data_ptr() == base_t.data_ptr(), name
        assert small_t.shape[0] == 2, name
    for name in sorted(reuse_mod._GLM5_REUSE_PLACEHOLDER_FIELDS):
        assert getattr(small, name) is getattr(base, name), name
    # The borrowed top-k is the PRODUCER's per-bucket view (which itself
    # aliases the producer's base buffer).
    assert small.top_k_indices.data_ptr() == shared_full[4].top_k_indices.data_ptr()
    assert small.top_k_indices.shape[0] == 2
    # Each reuse layer owns its own selected-token-ID buffer and re-runs the
    # transform; nothing is borrowed except the top-k.
    assert small.selected_token_ids.data_ptr() != shared_full[4].selected_token_ids.data_ptr()
    assert small.selected_token_ids.dtype == torch.int32
    out_base, out_small = reuse_segment._outputs[4], reuse_segment._outputs[2]
    assert out_small.attn_output.data_ptr() == out_base.attn_output.data_ptr()
    assert out_small.indexer_k_tensor is out_base.indexer_k_tensor


@requires_cuda
def test_full_dsa_two_bucket_replay_parity_with_aliased_buffers(monkeypatch):
    device = torch.device("cuda")
    torch.manual_seed(0)
    shared_buffers = {}
    segment, primary_cache, aux_cache = _build_full_segment(
        monkeypatch, device, shared_buffers
    )
    attn = segment.attn

    manager = CUDAGraphManager(BatchSizeBucketing([4, 2]), device=device)
    manager.WARMUP_ITERATIONS = 1
    name = make_glm5_full_dsa_graph_segment_name(0)
    manager.register_segment(name, segment)
    # Largest-first: bucket 4 allocates the base, bucket 2 captures on views.
    manager.warmup_and_capture_buckets([4, 2])
    assert shared_buffers[2].q_a.data_ptr() == shared_buffers[4].q_a.data_ptr()

    def inputs_for(bsz, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return {
            "hidden_states": torch.randn(
                bsz, 1, attn.hidden_size, generator=gen, dtype=torch.float32
            ).to(torch.bfloat16).to(device),
            "position_ids": torch.arange(1, bsz + 1, dtype=torch.int64, device=device).view(bsz, 1),
            "cache_seqlens": torch.full((bsz,), 2, dtype=torch.int32, device=device),
            "primary_slot_indices": torch.arange(bsz, dtype=torch.int32, device=device) % 2,
            "aux_slot_indices": torch.arange(bsz, dtype=torch.int32, device=device) % 2,
        }

    def eager(bsz, seed):
        inp = inputs_for(bsz, seed)
        primary_cache.zero_()
        aux_cache.zero_()
        out = segment.forward(
            num_valid_tokens=torch.tensor([bsz], dtype=torch.int32, device=device),
            **inp,
        )
        torch.cuda.synchronize()
        return {k: v.detach().clone() for k, v in out.items()}

    def replay(bsz, seed):
        inp = inputs_for(bsz, seed)
        primary_cache.zero_()
        aux_cache.zero_()
        out = manager.replay(name, bsz, **inp)
        torch.cuda.synchronize()
        return {k: v.detach().clone() for k, v in out.items()}

    # Alternate buckets so each graph replays over buffer rows the other
    # bucket's replay has just rewritten through the aliased base storage.
    for bsz, seed in ((2, 11), (3, 12), (2, 13), (4, 14)):
        expected = eager(bsz, seed)
        got = replay(bsz, seed)
        for key in expected:
            assert torch.equal(got[key], expected[key]), (bsz, key)


@requires_cuda
def test_selected_branch_fa3_kwargs(monkeypatch):
    """The page-size-1 FA3 call takes the selected token IDs as its table."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    recorder: dict = {}
    shared_buffers = {}
    segment, primary_cache, _ = _build_full_segment(
        monkeypatch, device, shared_buffers, all_short=False, recorder=recorder
    )
    attn = segment.attn
    bsz = 2
    segment.forward(
        hidden_states=torch.randn(bsz, 1, attn.hidden_size, dtype=torch.bfloat16, device=device),
        position_ids=torch.tensor([[1], [2]], dtype=torch.int64, device=device),
        cache_seqlens=torch.tensor([2, 3], dtype=torch.int32, device=device),
        primary_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        aux_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        num_valid_tokens=torch.tensor([bsz], dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize()
    buffers = shared_buffers[bsz]

    assert [stage for stage in recorder["scoring"]] == [
        "head_gates_out",
        "cuda_wq_b_proj_out",
        "rope_hadamard_q_out",
        "fused_paged_score_and_topk_with_slots_out",
    ]
    assert len(recorder["transform"]) == 1
    transform = recorder["transform"][0]
    assert transform["physical_token_ids"] is buffers.selected_token_ids
    assert transform["selected_lengths"] is buffers.selected_lengths
    assert transform["primary_slot_indices"] is buffers.safe_primary_slot_indices

    assert len(recorder["fa3"]) == 1
    kw = recorder["fa3"][0]
    assert kw["page_table"] is buffers.selected_token_ids
    assert kw["cache_seqlens"] is buffers.selected_lengths
    assert kw["cache_batch_idx"] is None
    assert tuple(kw["q"].shape) == (bsz, 1, attn.num_heads, attn.qk_rope_head_dim)
    assert tuple(kw["qv"].shape) == (bsz, 1, attn.num_heads, attn.kv_lora_rank)
    # Page-size-1 view of the SAME storage as the real paged cache.
    assert kw["k_cache"].shape[0] == primary_cache.shape[0] * primary_cache.shape[1]
    assert kw["k_cache"].shape[1:3] == (1, 1)
    assert kw["k_cache"].shape[-1] == attn.qk_rope_head_dim
    assert kw["k_cache"].untyped_storage().data_ptr() == (
        primary_cache.untyped_storage().data_ptr()
    )


@requires_cuda
def test_all_short_branch_skips_scoring_and_reads_the_paged_cache(monkeypatch):
    device = torch.device("cuda")
    torch.manual_seed(0)
    recorder: dict = {}
    shared_buffers = {}
    segment, primary_cache, _ = _build_full_segment(
        monkeypatch, device, shared_buffers, all_short=True, recorder=recorder
    )
    attn = segment.attn
    assert segment.all_short is True
    bsz = 2
    segment.forward(
        hidden_states=torch.randn(bsz, 1, attn.hidden_size, dtype=torch.bfloat16, device=device),
        position_ids=torch.tensor([[1], [2]], dtype=torch.int64, device=device),
        cache_seqlens=torch.tensor([2, 3], dtype=torch.int32, device=device),
        primary_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        aux_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        num_valid_tokens=torch.tensor([bsz], dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize()
    buffers = shared_buffers[bsz]

    # No scoring, no transform: an all_short graph never selects.
    assert recorder["scoring"] == []
    assert recorder["transform"] == []
    assert len(recorder["fa3"]) == 1
    kw = recorder["fa3"][0]
    assert kw["page_table"] is segment.primary_page_table
    assert kw["cache_batch_idx"] is buffers.safe_primary_slot_indices
    assert kw["cache_seqlens"] is buffers.safe_cache_seqlens
    assert tuple(kw["q"].shape) == (bsz, 1, attn.num_heads, attn.qk_rope_head_dim)
    assert tuple(kw["qv"].shape) == (bsz, 1, attn.num_heads, attn.kv_lora_rank)
    # Resident page-size-N cache, not a flattened view.
    assert kw["k_cache"].shape[:3] == primary_cache.shape[:3]
    assert kw["k_cache"].shape[-1] == attn.qk_rope_head_dim


@requires_cuda
def test_glm5_registers_fused_qkv_a_storage_views():
    device = torch.device("cuda")
    q_weight = torch.arange(
        12,
        device=device,
        dtype=torch.float32,
    ).view(3, 4).to(torch.bfloat16)
    kv_weight = torch.arange(
        8,
        device=device,
        dtype=torch.float32,
    ).view(2, 4).add_(100).to(torch.bfloat16)
    q_scale = torch.tensor([[1.0, 2.0]], device=device)
    kv_scale = torch.tensor([[3.0, 4.0]], device=device)
    wrapper = types.SimpleNamespace(
        layer_idx=0,
        module=types.SimpleNamespace(
            q_a_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=q_weight)),
            q_b_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=q_weight)),
            kv_a_proj_with_mqa=types.SimpleNamespace(
                weight=types.SimpleNamespace(data=kv_weight)
            ),
            kv_b_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=kv_weight)),
            o_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=q_weight)),
        ),
        weight_dequant_scale={
            "q_a_proj.weight_scale_inv": q_scale,
            "kv_a_proj_with_mqa.weight_scale_inv": kv_scale,
        },
    )

    GLM5AttnWrapper._register_fp8_weights(wrapper)

    assert torch.equal(
        wrapper._fp8_qkv_a_proj,
        torch.cat((q_weight, kv_weight), dim=0),
    )
    assert torch.equal(
        wrapper._fp8_qkv_a_scale,
        torch.cat((q_scale, kv_scale), dim=0),
    )
    assert (
        wrapper.module.q_a_proj.weight.data.untyped_storage().data_ptr()
        == wrapper._fp8_qkv_a_proj.untyped_storage().data_ptr()
    )
    assert (
        wrapper.module.kv_a_proj_with_mqa.weight.data.untyped_storage().data_ptr()
        == wrapper._fp8_qkv_a_proj.untyped_storage().data_ptr()
    )
