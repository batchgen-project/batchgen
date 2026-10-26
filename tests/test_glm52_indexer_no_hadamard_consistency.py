"""The DSA indexer Hadamard rotation is ONE predicate across its whole set.

The incoherence rotation ``H`` is orthonormal, so applying it to both operands
of the indexer logit leaves ``q @ k.T`` unchanged in exact arithmetic -- it only
redistributes fp8 quantization error. Upstream DeepSeek-V3.2 therefore drops it
on BOTH q and k wherever its fused indexer writers are on, and BatchGen follows
that on the page-split FP8 route (GLM-5.2's indexer cache).

Dropping it on one side only would NOT preserve the logit, and a cache written
under one setting must never be read under the other. So the rule lives in one
function, ``indexer_applies_hadamard``, and this file enumerates the set that
has to agree with it:

PRODUCERS of the fp8 index-K cache
  1. the fused K writer                 (graph decode, page-split)
  2. the un-fused K chain               (eager selector, and the AoS branch of
                                         the captured segment)
  3. the sparse-prefill K chain         (GLM-5.2/5.3 packed prefill)
  4. the prepack-prefill K chain        (``compute_indexer_kv``)

CONSUMERS -- producers of the Q index vector scored against that cache
  5. the captured Q epilogue            (both graph segments)
  6. the eager paged FP8 scorer         (``_score_and_select_paged_fp8``)
  7. the sparse-prefill Q chain         (same packed prefill layer)

Nothing here needs a GPU: the predicate is torch-only, and every call site is
checked in the AST so a literal ``True``/``False`` cannot creep back in.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types

import pytest
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]

_QUANT_SRC = ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer_k_quant.py"
_WRITER_SRC = ROOT / "batchgen" / "attention" / "dsa" / "aux_indexer_k_writer.py"
_SEGMENTS_SRC = ROOT / "batchgen" / "models" / "glm" / "glm5" / "cuda_graph_segments.py"
_SELECTOR_SRC = ROOT / "batchgen" / "attention" / "dsa" / "glm5_decode_selector.py"
_MODEL_SRC = ROOT / "batchgen" / "models" / "glm" / "glm5" / "model.py"
_WRAPPERS_SRC = ROOT / "batchgen" / "models" / "glm" / "glm5" / "wrappers.py"
_PREFILL_SRC = ROOT / "batchgen" / "models" / "glm" / "glm5" / "sparse_prefill.py"
_Q_KERNEL_SRC = (
    ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer" / "csrc"
    / "fused_rope_hadamard.cu"
)
_Q_KERNEL_BINDING = (
    ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer" / "csrc"
    / "fused_rope_hadamard_binding.cpp"
)
_MODEL_Q_KERNEL_SRC = (
    ROOT / "batchgen" / "other_kernels" / "hadamard_transform" / "csrc"
    / "fused_rope_hadamard.cu"
)
_MODEL_Q_KERNEL_BINDING = (
    ROOT / "batchgen" / "other_kernels" / "hadamard_transform" / "csrc"
    / "fused_rope_hadamard_binding.cpp"
)
_K_KERNEL_SRC = (
    ROOT / "batchgen_kernels" / "src" / "attention"
    / "fused_indexer_k_norm_rope_write_fp8.cu"
)

_SEGMENT_CLASSES = ("Glm5DsaAttnSegment", "Glm5FullDsaAttnSegment")

# The predicate, by name, wherever a call site is allowed to get it from.
_PREDICATE_EXPRESSIONS = frozenset(
    {
        "self.aux_applies_hadamard",
        "aux_manager_applies_hadamard(gpu_paged_kv_manager_aux, indexer.index_head_dim)",
        "cache_applies_hadamard(indexer_blocked_k, self.index_head_dim)",
        "self._indexer_applies_hadamard()",
        "indexer_applies_hadamard",
        "apply_hadamard",
    }
)


# ──────────────────────────────────────────────────────────────────────────────
# Loading: the quant module is torch-only, the rest is read as source
# ──────────────────────────────────────────────────────────────────────────────

def _load_quant_module():
    """Import ``indexer_k_quant`` by path -- its package pulls in triton."""

    name = "_nohad_consistency_indexer_k_quant"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _QUANT_SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_quant = _load_quant_module()

SCALE_BYTES = _quant.SCALE_BYTES_PER_TOKEN
DATA_DIM = _quant.NORM_ROPE_HEAD_DIM


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _top_level(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level function {name}")


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return sub
            raise AssertionError(f"{cls} has no method {name}")
    raise AssertionError(f"no class {cls}")


def _calls(scope: ast.AST, callee: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
    ]


def _kwargs(call: ast.Call) -> dict[str, str]:
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def _rotation_kwarg(call: ast.Call, key: str = "apply_hadamard") -> str:
    kwargs = _kwargs(call)
    assert key in kwargs, (
        f"call `{ast.unparse(call.func)}` does not pass `{key}`; the rotation "
        "must be explicit at every producer and consumer"
    )
    return kwargs[key]


def _aux_cache(*, page_split: bool, data_dim: int = DATA_DIM) -> torch.Tensor:
    """A 2-page aux indexer cache in the layout the KV manager allocates."""

    if page_split:
        return torch.zeros(
            2, 64, 1, data_dim + SCALE_BYTES, dtype=torch.float8_e4m3fn
        )
    return torch.zeros(2, 64, 1, data_dim, dtype=torch.bfloat16)


# ──────────────────────────────────────────────────────────────────────────────
# 1. The predicate itself
# ──────────────────────────────────────────────────────────────────────────────

def test_page_split_fp8_does_not_rotate_and_bf16_aos_does():
    """The whole semantic change, in one assertion pair."""

    assert _quant.indexer_applies_hadamard(page_split=True) is False
    assert _quant.indexer_applies_hadamard(page_split=False) is True


def test_the_cache_resolves_the_same_answer_as_the_bare_predicate():
    page_split = _aux_cache(page_split=True)
    aos = _aux_cache(page_split=False)

    assert _quant.cache_is_page_split(page_split, DATA_DIM) is True
    assert _quant.cache_applies_hadamard(page_split, DATA_DIM) is False

    assert _quant.cache_is_page_split(aos, DATA_DIM) is False
    assert _quant.cache_applies_hadamard(aos, DATA_DIM) is True


def test_a_mis_sized_cache_is_named_instead_of_picking_a_rotation():
    """``cache_is_page_split`` owns the failure, so the rotation inherits it."""

    bad_fp8 = torch.zeros(2, 64, 1, DATA_DIM, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="must be page-split"):
        _quant.cache_applies_hadamard(bad_fp8, DATA_DIM)

    bad_aos = torch.zeros(2, 64, 1, DATA_DIM + 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must carry"):
        _quant.cache_applies_hadamard(bad_aos, DATA_DIM)


def test_the_reference_chain_without_rotation_is_rope_only():
    """``apply_hadamard=False`` must remove H, not merely rescale it."""

    gen = torch.Generator().manual_seed(11)
    k_raw = (0.5 * torch.randn(6, DATA_DIM, generator=gen)).to(torch.bfloat16)
    weight = torch.ones(DATA_DIM, dtype=torch.bfloat16)
    bias = torch.zeros(DATA_DIM, dtype=torch.bfloat16)
    cos = torch.ones(8, _quant.NORM_ROPE_ROPE_DIM)
    sin = torch.zeros(8, _quant.NORM_ROPE_ROPE_DIM)
    positions = torch.zeros(6, dtype=torch.int64)

    rotated = _quant.indexer_k_norm_rope_reference(
        k_raw, weight, bias, 1e-5, cos, sin, positions, True
    )
    plain = _quant.indexer_k_norm_rope_reference(
        k_raw, weight, bias, 1e-5, cos, sin, positions, False
    )
    # cos=1/sin=0 makes RoPE the identity, so `plain` is the normalized input
    # and `rotated` is that input times H/sqrt(128): a real rotation, and the
    # two are not proportional.
    h = _quant.hadamard_matrix_128()
    want = (plain.float() @ h) * (DATA_DIM ** -0.5)
    torch.testing.assert_close(rotated.float(), want.to(torch.bfloat16).float())
    assert not torch.allclose(rotated.float(), plain.float())


# ──────────────────────────────────────────────────────────────────────────────
# 2. The manager-config resolver answers the same as the cache tensor
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("page_split", (True, False))
def test_the_manager_config_resolver_matches_the_allocated_cache(page_split):
    """Two carriers of ONE fact: the config that allocated the cache, and the
    cache. The eager selector must rotate (or not) before it holds the tensor,
    so both routes are pinned against each other here."""

    writer = _load_writer_module()
    cache = _aux_cache(page_split=page_split)
    manager = types.SimpleNamespace(
        config=types.SimpleNamespace(
            k_head_dim=int(cache.shape[-1]),
            kv_dtype=cache.dtype,
        )
    )
    assert (
        writer.aux_manager_applies_hadamard(manager, DATA_DIM)
        is _quant.cache_applies_hadamard(cache, DATA_DIM)
    )


def test_the_manager_config_resolver_names_a_mis_sized_configuration():
    writer = _load_writer_module()
    manager = types.SimpleNamespace(
        config=types.SimpleNamespace(
            k_head_dim=DATA_DIM, kv_dtype=torch.float8_e4m3fn
        )
    )
    with pytest.raises(ValueError, match="must be page-split"):
        writer.aux_manager_applies_hadamard(manager, DATA_DIM)


def _load_writer_module():
    """``aux_indexer_k_writer`` with its triton dependency stubbed out."""

    name = "_nohad_consistency_aux_writer"
    if name in sys.modules:
        return sys.modules[name]

    pkg = "batchgen_kernels.triton.kv_cache"
    stub = types.ModuleType(pkg)
    stub.run_paged_kv_token_update_fused = lambda **_: None
    quant_name = "batchgen_kernels.attention.dsa.indexer_k_quant"
    injected = {
        pkg: stub,
        quant_name: _quant,
    }
    saved = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        spec = importlib.util.spec_from_file_location(name, _WRITER_SRC)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


# ──────────────────────────────────────────────────────────────────────────────
# 3. PRODUCERS: every writer of the fp8 index-K cache takes the predicate
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("segment", _SEGMENT_CLASSES)
def test_both_graph_segments_derive_one_rotation_from_the_aux_cache(segment):
    """``self.aux_applies_hadamard`` is assigned from the shared resolver, in
    the same place the page format is read."""

    tree = _tree(_SEGMENTS_SRC)
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == segment
    )
    assigns = [
        node for node in ast.walk(cls)
        if isinstance(node, ast.Assign)
        and any(
            ast.unparse(t) == "self.aux_applies_hadamard" for t in node.targets
        )
    ]
    assert len(assigns) == 1, (
        f"{segment} must assign self.aux_applies_hadamard exactly once"
    )
    assert ast.unparse(assigns[0].value).startswith("aux_cache_applies_hadamard("), (
        "the segment must take the rotation from the shared cache resolver, "
        f"got {ast.unparse(assigns[0].value)}"
    )


def test_the_fused_k_writer_call_site_takes_the_segment_predicate():
    """Producer 1. The literal ``True`` this call used to carry IS the change."""

    fn = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    call = _calls(fn, "write_aux_indexer_k_norm_rope")
    assert len(call) == 1
    assert _rotation_kwarg(call[0]) == "self.aux_applies_hadamard"


def test_the_captured_unfused_k_chain_takes_the_segment_predicate():
    """Producer 2a: the bf16 AoS branch of the captured segment."""

    fn = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    call = _calls(fn, "indexer._fused_rope_hadamard_or_fallback")
    assert len(call) == 1
    assert _rotation_kwarg(call[0]) == "self.aux_applies_hadamard"


def test_the_eager_selector_k_chain_resolves_the_rotation_itself():
    """Producer 2b. Both selector entry points rope+rotate indexer K before
    they resolve the page table, so both read the manager config."""

    tree = _tree(_SELECTOR_SRC)
    calls = _calls(tree, "indexer._fused_rope_hadamard_or_fallback")
    assert len(calls) == 2, (
        f"expected the two selector K chains, found {len(calls)}"
    )
    for call in calls:
        assert _rotation_kwarg(call) == (
            "aux_manager_applies_hadamard(gpu_paged_kv_manager_aux, "
            "indexer.index_head_dim)"
        )


def test_the_sparse_prefill_k_chain_takes_the_layer_predicate():
    """Producer 3. These keys ARE the cache decode reads back."""

    tree = _tree(_PREFILL_SRC)
    fn = _top_level(tree, "_compute_indexer_kv_from_quantized_hidden")
    assert "apply_hadamard" in [a.arg for a in fn.args.kwonlyargs] + [
        a.arg for a in fn.args.args
    ]
    call = _calls(fn, "fused_rope_hadamard")
    assert len(call) == 1
    assert _rotation_kwarg(call[0]) == "apply_hadamard"

    layer = _top_level(tree, "glm52_sparse_prefill_prepacked")
    assert "indexer_applies_hadamard" in [a.arg for a in layer.args.kwonlyargs]
    k_call = _calls(layer, "_compute_indexer_kv_from_quantized_hidden")
    assert len(k_call) == 1
    assert _rotation_kwarg(k_call[0]) == "indexer_applies_hadamard"


def test_the_prepack_prefill_k_chain_takes_the_wrapper_predicate():
    """Producer 4. GLM-5.2 prompts at or below index_topk skip sparse prefill
    and still populate the page-split cache through ``compute_indexer_kv``."""

    model = _tree(_MODEL_SRC)
    fn = _method(model, "Glm5Indexer", "compute_indexer_kv")
    assert "apply_hadamard" in [a.arg for a in fn.args.args]
    forwarded = _calls(fn, "self._fused_rope_hadamard_or_fallback")
    assert len(forwarded) == 1
    assert _rotation_kwarg(forwarded[0]) == "apply_hadamard"

    wrappers = _tree(_WRAPPERS_SRC)
    calls = _calls(wrappers, "self.module.indexer.compute_indexer_kv")
    assert len(calls) == 1
    assert _rotation_kwarg(calls[0]) == "self._indexer_applies_hadamard()"


def test_the_wrapper_predicate_comes_from_the_aux_host_view_page_format():
    """Prefill's carrier of the page-split fact is the aux host KV view, and it
    is turned into the rotation by the same shared function."""

    fn = _method(_tree(_WRAPPERS_SRC), "GLM5AttnWrapper", "_indexer_applies_hadamard")
    src = ast.unparse(fn)
    assert "host_paged_kv_worker_view_aux" in src
    assert "is_page_split" in src
    call = _calls(fn, "indexer_applies_hadamard")
    assert len(call) == 1
    assert "page_split" in _kwargs(call[0])


# ──────────────────────────────────────────────────────────────────────────────
# 4. CONSUMERS: every producer of the Q index vector takes the SAME predicate
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("segment", _SEGMENT_CLASSES)
def test_the_captured_q_epilogue_takes_the_segment_predicate(segment):
    """Consumer 5. The same ``self.aux_applies_hadamard`` the K writer uses --
    that identity is what makes the change logit-preserving."""

    fn = _method(_tree(_SEGMENTS_SRC), segment, "forward")
    call = _calls(fn, "rope_hadamard_q_out")
    assert len(call) == 1
    assert _rotation_kwarg(call[0]) == "self.aux_applies_hadamard"


def test_the_eager_paged_fp8_scorer_reads_the_rotation_off_the_cache():
    """Consumer 6: scored straight out of the page-split cache, so the cache
    it is handed is the right carrier."""

    fn = _method(_tree(_MODEL_SRC), "Glm5Indexer", "_score_and_select_paged_fp8")
    call = _calls(fn, "rope_hadamard_q")
    assert len(call) == 1
    assert _rotation_kwarg(call[0]) == (
        "cache_applies_hadamard(indexer_blocked_k, self.index_head_dim)"
    )


def test_the_sparse_prefill_q_chain_shares_the_k_chain_predicate():
    """Consumer 7. One name, passed to both halves of the same layer."""

    tree = _tree(_PREFILL_SRC)
    scorer = _top_level(tree, "select_packed_glm52_topk")
    assert "apply_hadamard" in [a.arg for a in scorer.args.kwonlyargs]
    q_call = _calls(scorer, "_fused_rope_hadamard_q")
    assert len(q_call) == 1
    assert _rotation_kwarg(q_call[0]) == "apply_hadamard"

    layer = _top_level(tree, "glm52_sparse_prefill_prepacked")
    scorer_call = _calls(layer, "select_packed_glm52_topk")
    assert len(scorer_call) == 1
    assert _rotation_kwarg(scorer_call[0]) == "indexer_applies_hadamard"


def test_the_captured_segments_pair_q_and_k_on_one_name():
    """The set-as-a-unit claim, asserted directly: inside the one forward that
    writes K and builds Q, both carry the identical expression."""

    fn = _method(_tree(_SEGMENTS_SRC), "Glm5FullDsaAttnSegment", "forward")
    rotations = {
        ast.unparse(call.func): _rotation_kwarg(call)
        for callee in (
            "write_aux_indexer_k_norm_rope",
            "indexer._fused_rope_hadamard_or_fallback",
            "rope_hadamard_q_out",
        )
        for call in _calls(fn, callee)
    }
    assert len(rotations) == 3, rotations
    assert len(set(rotations.values())) == 1, (
        f"Q and K must rotate by one expression, got {rotations}"
    )


def test_no_call_site_hardcodes_the_rotation():
    """A literal would silently desynchronize one member of the set."""

    offenders = []
    for path in (_SEGMENTS_SRC, _SELECTOR_SRC, _MODEL_SRC, _WRAPPERS_SRC):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "apply_hadamard":
                    continue
                if isinstance(kw.value, ast.Constant):
                    offenders.append(
                        f"{path.name}: {ast.unparse(node.func)}"
                        f"(apply_hadamard={ast.unparse(kw.value)})"
                    )
                elif ast.unparse(kw.value) not in _PREDICATE_EXPRESSIONS:
                    offenders.append(
                        f"{path.name}: {ast.unparse(node.func)}"
                        f"(apply_hadamard={ast.unparse(kw.value)})"
                    )
    assert not offenders, (
        "these call sites do not take the shared predicate: " + "; ".join(offenders)
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Both kernel instantiations stay buildable (structural -- no nvcc here)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path", (_Q_KERNEL_SRC, _MODEL_Q_KERNEL_SRC), ids=("kernels", "model")
)
def test_the_q_kernel_keeps_both_rotation_instantiations(path):
    """Mirrors upstream's ``kHadamard`` template parameter: the rotation is a
    compile-time switch and NEITHER arm is deleted."""

    text = path.read_text()
    assert "template <bool kApplyHadamard>" in text
    assert "if constexpr (kApplyHadamard) {" in text
    assert "fused_rope_hadamard_kernel<true><<<" in text
    assert "fused_rope_hadamard_kernel<false><<<" in text


@pytest.mark.parametrize(
    "path", (_Q_KERNEL_SRC, _MODEL_Q_KERNEL_SRC), ids=("kernels", "model")
)
def test_the_q_kernel_scales_only_when_it_rotates(path):
    """1/sqrt(128) normalizes H, so the no-rotation arm must not apply it."""

    text = path.read_text()
    assert "float store_scale = 1.0f;" in text
    assert "store_scale = params.scale;" in text
    body = text.split("if constexpr (kApplyHadamard) {", 1)[1]
    guarded = body.split("\n    }", 1)[0]
    assert "store_scale = params.scale;" in guarded, (
        "the Hadamard normalization must be inside the rotation guard"
    )


@pytest.mark.parametrize(
    "path", (_Q_KERNEL_BINDING, _MODEL_Q_KERNEL_BINDING), ids=("kernels", "model")
)
def test_the_q_binding_defaults_to_rotating(path):
    """The bf16 AoS models keep their semantics even at an un-updated call."""

    text = path.read_text()
    assert "bool apply_hadamard = true" in text
    assert 'pybind11::arg("apply_hadamard") = true' in text
    assert "fused_rope_hadamard_launch(params, apply_hadamard, stream)" in text


def test_the_fused_k_writer_keeps_both_rotation_instantiations():
    """The K port already carried the switch; both arms must stay compiled."""

    text = _K_KERNEL_SRC.read_text()
    assert "bool kApplyHadamard" in text
    assert "if constexpr (kApplyHadamard) {" in text
    assert "BATCHGEN_DISPATCH_K_NORM_ROPE(true);" in text
    assert "BATCHGEN_DISPATCH_K_NORM_ROPE(false);" in text


@pytest.mark.parametrize(
    "name", ("fused_rope_hadamard", "fused_rope_hadamard_out")
)
def test_the_python_kernel_wrappers_forward_the_switch(name):
    init = ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer" / "__init__.py"
    fn = _top_level(_tree(init), name)
    args = [a.arg for a in fn.args.args]
    assert "apply_hadamard" in args
    defaults = dict(zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults))
    assert ast.unparse(defaults["apply_hadamard"]) == "True"
    assert "apply_hadamard" in ast.unparse(
        _calls(fn, f"_fused_rope_hadamard_cuda.{name}")[0]
    )


@pytest.mark.parametrize("name", ("rope_hadamard_q", "rope_hadamard_q_out"))
def test_the_q_wrappers_forward_the_switch(name):
    src = ROOT / "batchgen_kernels" / "attention" / "dsa" / "fused_indexer_score.py"
    fn = _top_level(_tree(src), name)
    args = [a.arg for a in fn.args.args]
    assert "apply_hadamard" in args
    defaults = dict(zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults))
    assert ast.unparse(defaults["apply_hadamard"]) == "True"


def test_the_indexer_fallback_defaults_to_rotating():
    """``_fused_rope_hadamard_or_fallback`` serves the AoS models too."""

    fn = _method(
        _tree(_MODEL_SRC), "Glm5Indexer", "_fused_rope_hadamard_or_fallback"
    )
    args = [a.arg for a in fn.args.args]
    assert "apply_hadamard" in args
    defaults = dict(zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults))
    assert ast.unparse(defaults["apply_hadamard"]) == "True"


# ──────────────────────────────────────────────────────────────────────────────
# 6. The startup gate exercises BOTH arms on the device
# ──────────────────────────────────────────────────────────────────────────────

def test_the_sparse_prefill_startup_gate_checks_both_arms():
    """No nvcc here, so the proof that the False instantiation actually runs is
    the runtime gate; it must cover both."""

    fn = _top_level(_tree(_PREFILL_SRC), "validate_glm52_sparse_prefill_runtime")
    loops = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "apply_hadamard"
        and ast.unparse(node.iter) == "(True, False)"
    ]
    assert len(loops) == 2, (
        "both the single-head K and the multi-head Q smoke checks must run "
        f"with and without the rotation, found {len(loops)} loops"
    )
