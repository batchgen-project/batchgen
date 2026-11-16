# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 JIT prewarm: the DSA indexer's kernels and the KDA gated norm.

Under test: ``batchgen/models/moonshotai/kimi_linear/dsa_prewarm.py``,
``jit_coverage.py`` and the K3.1 hooks of ``triton_prewarm.py`` and the
strategy manager. CPU only: no CUDA, Triton or DeepGEMM.

  * the shapes come from a K3.1 ``config.json`` fixture (K3's released
    config plus the DSA delta): the indexer geometry from ``dsa_config``, the
    gated-norm width and rows from ``linear_attn_config``;
  * every kernel of ``K31_DSA_JIT_KERNELS`` has prewarm entries, and every
    entry's site is declared there;
  * the witness sets reach every key the serving calls can produce
    (act_quant row classes, the scorer prologue's width classes, the
    transform's widths, the paged-MQA batch alignments);
  * the runners hand the serving functions the serving dtypes and layouts
    (recording fakes; the prefill scorer is the REAL
    ``_score_and_select_prefill`` over stubbed kernels);
  * the coverage registry carries the kimi_linear deep_gemm part;
  * the strategy manager's prefill row bound and DSA deployment.

Run: python -m pytest tests/unit/test_kimi_k31_dsa_prewarm.py -q
"""

import ast
import copy
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batchgen.config.model_registry import load_config  # noqa: E402
from batchgen.models.moonshotai.kimi_linear import (  # noqa: E402
    dsa_prewarm as DP,
    jit_coverage as COV,
)
from batchgen.models.moonshotai.kimi_linear import (  # noqa: E402
    triton_prewarm as TP,
)
from batchgen_kernels import jit_site_registry as REG  # noqa: E402

KL_DIR = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
K3_CONFIG_JSON = (ROOT / "batchgen" / "models" / "moonshotai" / "kimi_k3"
                  / "assets" / "config.json")
DSA_DIR = ROOT / "batchgen_kernels" / "attention" / "dsa"

K31_DSA_CONFIG = {
    "dsa_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
                   64, 68, 72, 76, 80, 84, 88, 92, 93],
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}

#: The kernels the 4-node fresh-cache probe compiled after readiness.
LATE_ON_PROBE = {
    "layer_norm_gated_fwd_kernel",
    "_per_token_blocked_quantize_1d_kernel",
    "_transform_selected_positions_kernel",
    "_prepare_paged_mqa_inputs_kernel",
    "sm90_fp8_mqa_logits",
    "sm90_paged_mqa_logits_metadata",
    "sm90_fp8_paged_mqa_logits",
}

ATTN_TP = 8
PAGE = 64
MAX_ROWS = 128


@pytest.fixture(scope="module")
def k31_config(tmp_path_factory):
    raw = json.loads(K3_CONFIG_JSON.read_text())
    raw = copy.deepcopy(raw)
    raw["model_type"] = "kimi_k31"
    raw["text_config"]["dsa_config"] = copy.deepcopy(K31_DSA_CONFIG)
    path = tmp_path_factory.mktemp("k31") / "config.json"
    path.write_text(json.dumps(raw))
    cfg = load_config(str(path.parent))
    assert cfg.model_type == "kimi_k31" and cfg.dsa() is not None
    return cfg


@pytest.fixture(scope="module")
def k3_config():
    cfg = load_config(str(K3_CONFIG_JSON.parent))
    assert cfg.dsa() is None
    return cfg


def _shapes(cfg, max_model_len=262_144, max_rows=MAX_ROWS):
    return DP.KimiDsaPrewarmShapes.from_config(
        cfg, page_size=PAGE, max_model_len=max_model_len,
        max_decode_rows=max_rows)


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


# --------------------------------------------------------------------------- #
#  Shapes from the K3.1 config                                               #
# --------------------------------------------------------------------------- #
def test_shapes_derive_from_the_k31_config_fields(k31_config):
    shapes = _shapes(k31_config)
    dsa = k31_config.dsa_config
    assert shapes.index_n_heads == dsa["index_num_heads"] == 32
    assert shapes.index_head_dim == dsa["index_head_dim"] == 128
    assert shapes.index_topk == dsa["index_topk"] == 2048
    assert shapes.max_aux_pages == 262_144 // PAGE
    assert shapes.aux_carrier_bytes == 128 + 4


def test_gated_norm_width_and_rows_derive_from_linear_attn_config(k31_config):
    lac = k31_config.linear_attn_config
    sites = TP.gated_norm_sites(k31_config, ATTN_TP)
    assert sites == [TP.GatedNormSite(
        "kda_o_norm", width=lac["head_dim"],
        rows_per_token=lac["num_heads"] // ATTN_TP)]
    assert (sites[0].width, sites[0].rows_per_token) == (128, 12)
    with pytest.raises(TP.K3PrewarmGeometryError, match="do not shard"):
        TP.gated_norm_sites(k31_config, 7)


def test_live_kda_module_is_checked_against_the_config(k31_config):
    sites = TP.gated_norm_sites(k31_config, ATTN_TP)
    TP._check_o_norm_site(SimpleNamespace(head_dim=128, num_heads=12), sites)
    with pytest.raises(TP.K3PrewarmGeometryError, match="o_norm"):
        TP._check_o_norm_site(SimpleNamespace(head_dim=128, num_heads=96),
                              sites)


def test_unservable_page_size_and_dense_config_are_refused(k31_config,
                                                           k3_config):
    with pytest.raises(DP.KimiDsaPrewarmError, match="64-token pages"):
        DP.KimiDsaPrewarmShapes.from_config(
            k31_config, page_size=128, max_model_len=4096, max_decode_rows=8)
    with pytest.raises(DP.KimiDsaPrewarmError, match="max_decode_rows"):
        DP.KimiDsaPrewarmShapes.from_config(
            k31_config, page_size=PAGE, max_model_len=4096, max_decode_rows=0)
    with pytest.raises(DP.KimiDsaPrewarmError, match="dsa_config"):
        DP.KimiDsaPrewarmShapes.from_config(
            k3_config, page_size=PAGE, max_model_len=4096, max_decode_rows=8)


def test_constants_match_the_kernel_modules():
    """The Triton-importing modules are read, not imported."""
    def assigned(path, name):
        for node in ast.parse(path.read_text()).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None) == name):
                return ast.literal_eval(node.value)
        raise AssertionError(f"{name} not assigned in {path}")

    logits = DSA_DIR / "paged_mqa_logits.py"
    assert assigned(logits, "PAGED_MQA_BLOCK_KV") == DP.PAGED_MQA_BLOCK_KV
    assert assigned(logits, "PAGED_MQA_NEXT_N") == DP.PAGED_MQA_NEXT_N
    # Read, never import: this module's package pulls in Triton, which a
    # CPU-only test environment does not have.
    quant = DSA_DIR / "indexer_k_quant.py"
    assert assigned(quant, "SCALE_BYTES_PER_TOKEN") == DP.PAGE_SPLIT_SCALE_BYTES
    scorer = (DSA_DIR / "paged_mqa_scorer.py").read_text()
    assert "block_p = min(256, triton.next_power_of_2(max_pages))" in scorer
    assert "(batch + 31) // 32 * 32" in scorer


# --------------------------------------------------------------------------- #
#  Coverage: every listed kernel has entries, every entry is declared         #
# --------------------------------------------------------------------------- #
def test_every_late_kernel_is_declared_with_its_sites():
    declared = {slot.kernel for slot in COV.K31_DSA_JIT_KERNELS}
    assert declared == LATE_ON_PROBE
    assert {slot.backend for slot in COV.K31_DSA_JIT_KERNELS} == {
        "triton", "deep_gemm"}


def _o_norm_module():
    weight = torch.zeros(4, 16, dtype=torch.bfloat16)
    return SimpleNamespace(q_proj=SimpleNamespace(weight=weight), num_heads=12,
                           head_dim=128, o_norm=lambda o, g: o)


def test_every_declared_site_has_entries_and_no_entry_is_undeclared(
        k31_config):
    entries = DP.dsa_entries(_shapes(k31_config), "cpu")
    entries += [e for e in TP.kda_prefill_entries(
        _o_norm_module(), None, 4096, shapes=())
        if e.name.startswith(COV.SITE_KDA_O_NORM)]
    sites = set(DP.planned_kernel_sites(entries))
    declared = {site for slot in COV.K31_DSA_JIT_KERNELS for site in slot.sites}
    assert sites == declared
    names = [e.name for e in entries]
    assert len(names) == len(set(names))


def test_dsa_entries_join_the_k3_list_only_on_k31(k31_config, k3_config):
    deployment = DP.KimiDsaDeployment(page_size=PAGE, max_model_len=131_072,
                                      max_decode_rows=MAX_ROWS)
    assert TP.dsa_prewarm_entries(k3_config, deployment, "cpu") == []
    planned = TP.dsa_prewarm_entries(k31_config, deployment, "cpu")
    assert [e.name for e in planned] == [
        e.name for e in DP.dsa_entries(_shapes(k31_config, 131_072), "cpu")]
    with pytest.raises(DP.KimiDsaPrewarmError, match="decode deployment"):
        TP.dsa_prewarm_entries(k31_config, None, "cpu")


def test_entries_are_identical_across_ranks(k31_config):
    a = [e.name for e in DP.dsa_entries(_shapes(k31_config), "cpu")]
    b = [e.name for e in DP.dsa_entries(_shapes(k31_config), "cpu")]
    assert a == b


# --------------------------------------------------------------------------- #
#  Witness sets reach every serving key                                       #
# --------------------------------------------------------------------------- #
def test_prefill_scorer_rows_reach_every_act_quant_key(k31_config):
    shapes = _shapes(k31_config)
    planned = set()
    for rows in DP.PREFILL_ROW_WITNESSES:
        planned |= DP.prefill_select_quant_keys(shapes, rows)
    # Any packed (or pooled) chunk of 1..N query tokens.
    reachable = set()
    for rows in range(1, 4097):
        reachable |= DP.prefill_select_quant_keys(shapes, rows)
    assert reachable == planned
    # Both casts are 128 wide with one scale block: the 2 binaries the
    # probe compiled are among these.
    assert {key[0] for key in planned} == {128}


@pytest.mark.parametrize("max_model_len", [2_048, 8_192, 131_072, 1_048_576])
def test_prepare_witnesses_reach_every_width_class(k31_config, max_model_len):
    shapes = _shapes(k31_config, max_model_len)
    witnesses = DP.aux_width_witnesses(shapes.max_aux_pages, PAGE)
    every = {DP.prepare_key(w, PAGE)
             for w in range(1, shapes.max_aux_pages + 1)}
    assert {DP.prepare_key(w, PAGE) for w in witnesses} == every
    assert len(witnesses) == len(every)


@pytest.mark.parametrize("max_model_len", [1_000, 2_048, 131_072])
def test_graph_score_widths_follow_the_graph_step(k31_config, max_model_len):
    from batchgen.models.moonshotai.kimi_linear.dsa_decode import (
        graph_score_width,
    )

    shapes = _shapes(k31_config, max_model_len)
    expected = {graph_score_width(shapes.index_topk, w * PAGE)
                for w in range(1, shapes.max_aux_pages + 1)}
    assert set(DP.graph_score_widths(shapes.index_topk, PAGE,
                                     shapes.max_aux_pages)) == expected
    assert max(expected) == min(shapes.index_topk,
                                shapes.max_aux_pages * PAGE)


def test_paged_mqa_batches_cover_every_alignment():
    batches = DP.paged_mqa_batches(MAX_ROWS)
    assert batches == [32, 64, 96, 128]
    assert {(b + 31) // 32 * 32 for b in range(1, MAX_ROWS + 1)} == set(batches)
    assert DP.paged_mqa_batches(97) == [32, 64, 96, 128]


# --------------------------------------------------------------------------- #
#  Runners: the serving functions, the serving layouts                         #
# --------------------------------------------------------------------------- #
def test_prefill_runner_drives_the_real_serving_scorer(k31_config,
                                                       monkeypatch):
    """``_score_and_select_prefill`` itself, with its three kernels stubbed:
    the act_quant inputs it builds from the runner's tensors are the serving
    casts, and fp8_mqa_logits gets the compressed-logits call."""
    quant_inputs, logits_calls = [], []

    def act_quant(x, block_size=128, **kwargs):
        quant_inputs.append((tuple(x.shape), x.dtype, x.is_contiguous(),
                             kwargs))
        flat = x.reshape(-1, x.shape[-1])
        return (torch.zeros(x.shape, dtype=torch.float8_e4m3fn),
                torch.ones(*x.shape[:-1], 1))

    def fp8_mqa_logits(q, kv, weights, starts, ends, **kwargs):
        logits_calls.append((q.shape, kv[0].shape, kv[1].shape, weights.shape,
                             starts.dtype, kwargs))
        return torch.zeros(q.shape[0], kwargs["max_seqlen_k"])

    monkeypatch.setitem(sys.modules, "deep_gemm", types.SimpleNamespace(
        fp8_mqa_logits=fp8_mqa_logits))
    monkeypatch.setitem(sys.modules, "batchgen.quantization.act_quant",
                        types.SimpleNamespace(act_quant=act_quant))
    monkeypatch.setitem(
        sys.modules, "batchgen_kernels.attention.dsa.fast_topk_cuda",
        types.SimpleNamespace(fast_topk_2048_out=lambda *a, **k: None))
    from batchgen.models.moonshotai.kimi_linear.dsa_sparse_prefill import (
        _prefill_index_topk,
    )

    shapes = _shapes(k31_config)
    for rows in DP.PREFILL_ROW_WITNESSES:
        DP._run_prefill_select(shapes, rows, "cpu", fn=_prefill_index_topk())
    seen = set()
    for shape, dtype, contiguous, kwargs in quant_inputs:
        assert dtype == torch.bfloat16 and contiguous and kwargs == {}
        rows = 1
        for d in shape[:-1]:
            rows *= d
        seen.add(DP.act_quant_key(rows, shape[-1]))
    planned = set()
    for rows in DP.PREFILL_ROW_WITNESSES:
        planned |= DP.prefill_select_quant_keys(shapes, rows)
    assert seen == planned
    assert len(logits_calls) == len(DP.PREFILL_ROW_WITNESSES)
    for q_shape, k_shape, ks_shape, w_shape, start_dtype, kwargs in logits_calls:
        assert q_shape[1:] == (32, 128) and k_shape[1:] == (128,)
        assert ks_shape == k_shape[:1] and w_shape == (q_shape[0], 32)
        assert start_dtype == torch.int32
        assert kwargs["clean_logits"] is False
        assert kwargs["max_seqlen_k"] % 256 == 0


def test_runner_kwargs_match_the_serving_signature():
    tree = ast.parse((KL_DIR / "dsa_sparse_prefill.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_score_and_select_prefill")
    assert [a.arg for a in fn.args.kwonlyargs] == [
        "q_idx", "index_keys", "head_gates", "causal_starts", "causal_ends",
        "max_seqlen", "topk"]


@pytest.mark.parametrize("has_valid", [False, True])
def test_selected_positions_runner_is_the_kimi_step_call(has_valid):
    fn = _Recorder()
    DP._run_selected_positions(2048, PAGE, has_valid, "cpu", fn=fn)
    (table, seqlens, topk, physical, lengths), kwargs = fn.calls[0]
    assert table.dtype == seqlens.dtype == torch.int32
    # Both Kimi steps hand int64 positions (clamp_to_last_token / the
    # eager torch.full(long)), never slot indices.
    assert topk.dtype == torch.int64 and topk.shape == (1, 2048)
    assert physical.dtype == lengths.dtype == torch.int32
    assert set(kwargs) == {"page_size", "num_valid_tokens"}
    assert kwargs["page_size"] == PAGE
    assert (kwargs["num_valid_tokens"] is not None) == has_valid


def test_dsa_decode_passes_what_the_transform_runner_models():
    source = (KL_DIR / "dsa_decode.py").read_text()
    assert "primary_slot_indices" not in source
    assert "torch.minimum(indices.to(torch.long)" in source
    assert "dtype=torch.long, device=hidden_flat.device" in source


def test_prepare_runner_is_the_graph_prologue_layout():
    fn = _Recorder()
    DP._run_paged_mqa_prepare(4096, PAGE, "cpu", fn=fn)
    (table, slots, seqlens, block_table, ctx), kwargs = fn.calls[0]
    assert table.shape == block_table.shape == (1, 4096)
    assert table.dtype == block_table.dtype == torch.int32
    assert slots.dtype == seqlens.dtype == ctx.dtype == torch.int32
    assert ctx.shape == (1, DP.PAGED_MQA_NEXT_N)
    assert kwargs["num_valid_tokens"] is not None
    assert kwargs["deep_gemm_context_lens_out"].data_ptr() != ctx.data_ptr()
    assert kwargs["page_size"] == PAGE


def test_paged_mqa_runner_uses_a_page_split_scratch_cache(k31_config):
    fn = _Recorder()
    DP._run_paged_mqa(_shapes(k31_config), 96, "cpu", fn=fn)
    _, kwargs = fn.calls[0]
    cache = kwargs.pop("aux_blocked_k")
    assert cache.shape == (1, PAGE, 1, 132) and cache.dtype == torch.uint8
    assert kwargs == {"batch": 96, "num_heads": 32, "index_head_dim": 128,
                      "max_pages_per_seq": 1, "max_seqlen": PAGE,
                      "page_size": PAGE}


# --------------------------------------------------------------------------- #
#  Coverage registry                                                          #
# --------------------------------------------------------------------------- #
def test_registry_carries_the_kimi_linear_deep_gemm_part():
    import batchgen.models.glm.glm5.jit_coverage as glm5_cov

    parts = {(p.backend, p.family): p for p in REG.backend_inventories()}
    assert parts[("deep_gemm", "kimi_linear")] is COV.DEEP_GEMM_INVENTORY
    assert parts[("deep_gemm", "glm5")] is glm5_cov.DEEP_GEMM_INVENTORY
    (count, reaches), = COV.DEEP_GEMM_INVENTORY.call_sites.values()
    assert count == 1 and reaches == (COV.SITE_DSA_PREFILL_SELECT,)
    merged = REG.backend_inventory("deep_gemm")
    assert set(merged.call_sites) == (
        set(glm5_cov.DEEP_GEMM_INVENTORY.call_sites)
        | set(COV.DEEP_GEMM_INVENTORY.call_sites))
    assert "batchgen/models/moonshotai/kimi_linear" in merged.paths
    # The paged-MQA scorer the K3.1 decode reaches is pinned by the part that
    # owns its scan root.
    assert any(path == "batchgen_kernels/attention/dsa/paged_mqa_scorer.py"
               for path, _, _ in glm5_cov.DEEP_GEMM_INVENTORY.call_sites)


def test_overlapping_or_unnamed_parts_are_refused():
    overlap = REG.BackendInventory(
        backend="deep_gemm", paths=("batchgen/models/glm/glm5",), callees=(),
        non_kernels=(), call_sites={}, family="intruder")
    with pytest.raises(REG.JitCoverageError, match="overlap"):
        REG.register_backend_inventory(overlap)
    unnamed = REG.BackendInventory(
        backend="deep_gemm", paths=("elsewhere",), callees=(),
        non_kernels=(), call_sites={})
    with pytest.raises(REG.JitCoverageError, match="different inventory"):
        REG.register_backend_inventory(unnamed)
    changed = REG.BackendInventory(
        backend="deep_gemm", paths=COV.DEEP_GEMM_INVENTORY.paths, callees=(),
        non_kernels=(), call_sites={}, family="kimi_linear")
    with pytest.raises(REG.JitCoverageError, match="different inventory"):
        REG.register_backend_inventory(changed)
    # An identical re-registration (a re-import) is a no-op.
    assert (REG.register_backend_inventory(COV.DEEP_GEMM_INVENTORY)
            is COV.DEEP_GEMM_INVENTORY)


# --------------------------------------------------------------------------- #
#  Strategy manager: prefill row bound and the DSA deployment                 #
# --------------------------------------------------------------------------- #
PSM = KL_DIR / "Parallel_Strategy_Manager.py"


def _psm_methods(names, namespace):
    from tests.unit.test_warmup_geometry_prewarm import load_class_methods

    return load_class_methods(PSM, "KimiLinearParallelStrategyManager",
                              names, namespace)


def _psm(config, *, cap, prompt, padding_bsz=MAX_ROWS, buckets=(1, 8, 64),
         manager=None):
    basic = SimpleNamespace(
        decode_graph_buckets=list(buckets), device_torch="cpu",
        max_model_len=131_072, get_max_prompt_length=lambda: prompt)
    return SimpleNamespace(
        engine_config=SimpleNamespace(
            Basic_Config=basic,
            Module_Batching_Config=SimpleNamespace(
                prefill_micro_batch_token_cap=cap),
            Device_Paged_KV_Config=SimpleNamespace(page_size=PAGE)),
        loaded_model_config=config, model="model", padding_bsz=padding_bsz,
        _attn_tp_size=1, _attn_tp_group=None, _attn_tp_rank=0,
        global_rank=0,
    )


def test_prefill_row_bound_is_the_longest_prompt(k31_config):
    calls = []

    def prewarm(model, config, **kwargs):
        calls.append(kwargs)
        return False

    fake_prewarm = types.ModuleType("kimi_prewarm_fake")
    fake_prewarm.prewarm_k3_triton_kernels = prewarm
    namespace = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        "time": __import__("time"), "logging": __import__("logging"),
        "AttnWrapperBase": SimpleNamespace(gpu_paged_kv_manager=None),
    }
    methods = _psm_methods(["_prewarm_triton_kernels",
                            "_dsa_prewarm_deployment"], namespace)
    monkey = {".triton_prewarm": fake_prewarm}
    # Relative imports in the lifted method resolve through __package__.
    namespace_pkg = "batchgen.models.moonshotai.kimi_linear"
    for fn in methods.values():
        fn.__globals__["__package__"] = namespace_pkg
    saved = sys.modules.get(namespace_pkg + ".triton_prewarm")
    sys.modules[namespace_pkg + ".triton_prewarm"] = fake_prewarm
    try:
        psm = _psm(k31_config, cap=16_384, prompt=61_440)
        psm._dsa_prewarm_deployment = (
            lambda buckets: methods["_dsa_prewarm_deployment"](psm, buckets))
        methods["_prewarm_triton_kernels"](psm)
        psm_cap_only = _psm(k31_config, cap=16_384, prompt=None)
        psm_cap_only._dsa_prewarm_deployment = lambda buckets: None
        methods["_prewarm_triton_kernels"](psm_cap_only)
    finally:
        if saved is None:
            del sys.modules[namespace_pkg + ".triton_prewarm"]
        else:
            sys.modules[namespace_pkg + ".triton_prewarm"] = saved
    del monkey
    first, second = calls
    # The o_norm row bound is the serving row-block constant, not a
    # deployment knob, so the prewarm call carries no prefill token bound.
    assert "max_prefill_tokens" not in first
    assert "max_prefill_tokens" not in second
    assert first["attn_tp_size"] == 1
    deployment = first["dsa_deployment"]
    assert deployment == DP.KimiDsaDeployment(
        page_size=PAGE,
        max_model_len=max(131_072, int(k31_config.max_position_embeddings)),
        max_decode_rows=MAX_ROWS)


def test_dsa_deployment_prefers_the_live_page_size(k31_config, k3_config):
    live = SimpleNamespace(config=SimpleNamespace(page_size_tokens=32))
    methods = _psm_methods(["_dsa_prewarm_deployment"], {
        "AttnWrapperBase": SimpleNamespace(gpu_paged_kv_manager=live)})
    fn = methods["_dsa_prewarm_deployment"]
    fn.__globals__["__package__"] = "batchgen.models.moonshotai.kimi_linear"
    psm = _psm(k31_config, cap=0, prompt=0, padding_bsz=None,
               buckets=(1, 256))
    deployment = fn(psm, [1, 256])
    assert deployment.page_size == 32 and deployment.max_decode_rows == 256
    assert fn(_psm(k3_config, cap=0, prompt=0), [1]) is None


# --------------------------------------------------------------------------- #
#  o_norm floor: the site is warmed even without a prefill token cap          #
# --------------------------------------------------------------------------- #
def test_o_norm_minimal_counts_cover_both_alignment_classes():
    from batchgen.models.moonshotai.kimi_linear.triton_prewarm import (
        o_norm_minimal_token_counts,
    )
    for heads in (1, 4, 16, 17, 34):
        counts = o_norm_minimal_token_counts(heads)
        rows = [t * heads for t in counts]
        assert any(r % 16 == 0 for r in rows), (heads, rows)
        if heads % 16:
            # The unaligned class exists only when heads is not itself a
            # multiple of 16 (observed missing key: 17 heads per rank).
            assert any(r % 16 for r in rows), (heads, rows)
        assert 1 in counts


def test_kda_prefill_entries_always_include_the_o_norm_site():
    from batchgen.models.moonshotai.kimi_linear import triton_prewarm as tp

    class _Proj:
        weight = type("W", (), {"dtype": None, "device": None})()

    module = type("M", (), {
        "num_heads": 17, "head_dim": 128, "q_proj": _Proj(), "o_norm": None,
    })()
    entries = tp.kda_prefill_entries(module, template=None, segment_tokens=8192,
                                     shapes=())
    names = [e.name for e in entries]
    assert any(name.startswith("kda_o_norm/") for name in names), names


def test_o_norm_coverage_reaches_every_serving_block_count():
    """Serving cuts the o_norm into fixed row blocks, so the prewarmed token
    counts must reach every autotune block-count bucket a <=block call can
    key on, in both row-alignment classes."""
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        KDA_PREFILL_SEGMENT_TOKENS,
    )
    from batchgen.models.moonshotai.kimi_linear.triton_prewarm import (
        O_NORM_ROWS_PER_BLOCK,
        o_norm_minimal_token_counts,
        o_norm_token_counts,
    )

    for heads in (16, 17, 34):
        counts = set(o_norm_minimal_token_counts(heads)) | set(
            o_norm_token_counts(heads, KDA_PREFILL_SEGMENT_TOKENS))
        covered = {}
        for tokens in counts:
            rows = tokens * heads
            nb = -(-rows // O_NORM_ROWS_PER_BLOCK)
            covered.setdefault(nb, set()).add(rows % 16 == 0)
        max_nb = -(-KDA_PREFILL_SEGMENT_TOKENS * heads
                   // O_NORM_ROWS_PER_BLOCK)
        for nb in range(1, max_nb + 1):
            lo = (nb - 1) * O_NORM_ROWS_PER_BLOCK
            hi = nb * O_NORM_ROWS_PER_BLOCK
            reachable = {
                (tokens * heads) % 16 == 0
                for tokens in range(1, KDA_PREFILL_SEGMENT_TOKENS + 1)
                if lo < tokens * heads <= hi
            }
            assert covered.get(nb, set()) >= reachable, (heads, nb)


def test_serving_o_norm_runs_in_fixed_row_blocks():
    """_o_norm_segments cuts at the constant, reproduces the uncut call,
    and the packed KDA prefill o_norm goes through it."""
    import re

    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        _o_norm_segments as seg,
    )

    src = (KL_DIR / "serving_modules.py").read_text()
    assert re.search(
        r"_o_norm_segments\(self\.o_norm,\s*o\.reshape\(", src), (
        "the packed KDA prefill o_norm call must go through _o_norm_segments")

    calls = []

    def fake_norm(o, z):
        calls.append(int(o.shape[0]))
        return o * 2 + z

    o = torch.arange(150 * 3 * 4, dtype=torch.float32).reshape(150, 3, 4)
    z = torch.ones_like(o)
    out = seg(fake_norm, o, z, segment_tokens=64)
    assert calls == [64, 64, 22]
    assert torch.equal(out, o * 2 + z)
