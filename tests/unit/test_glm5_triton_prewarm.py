# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                             #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The GLM-5 Triton prewarm's class model and entry lists, on CPU.

Under test: ``batchgen/models/glm/glm5/triton_prewarm.py``.

  * the row witnesses {1, 13, 16, 17} reach every (rows, align4(rows)) class
    pair of the blocked FP8 casts, for every width and flag combination;
  * one head-gate and paged-score entry per bucket; the split-K router and
    the DeepEP masked SiLU*up cast on bucket-independent witnesses;
  * the geometry-to-key derivation (width classes, BLOCK_P, BLOCK_S, the
    dense gather's BLOCK_T thresholds, the router row range) against
    hand-computed values;
  * the entry list is a function of the configuration only, identical on
    every rank;
  * the section-D predicates are logged, at WARNING when one makes an
    unprewarmed kernel reachable;
  * ``_offset_topk_indices_kernel`` takes ``n_elements`` as a runtime int;
  * the runners hand each wrapper the serving call's dtypes and strides.

Loaded by file path (no ``import batchgen``); every wrapper is replaced by a
recording fake, so neither Triton nor CUDA is needed.

Run: python -m pytest tests/unit/test_glm5_triton_prewarm.py -q
"""

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_GLM5_DIR = _ROOT / "batchgen" / "models" / "glm" / "glm5"
_ROUTER_CONFIG = _ROOT / "batchgen_kernels" / "triton" / "glm5_router_gemm_config.py"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_prewarm(name="_glm5_triton_prewarm_under_test"):
    return _load(_GLM5_DIR / "triton_prewarm.py", name)


TP = _load_prewarm()
ROUTER = _load(_ROUTER_CONFIG, "_glm5_router_gemm_config_under_test")
ROUTER_RANGE = (ROUTER.ROUTER_SPLIT_K_MIN_ROWS, ROUTER.ROUTER_SPLIT_K_MAX_ROWS)


def _shapes(sparse_prefill=True):
    """GLM-5.2 dimensions (``config.py``)."""
    return TP.Glm5PrewarmShapes(
        hidden_size=6144, q_lora_rank=2048, kv_lora_rank=512,
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
        num_heads=64, intermediate_size=12288, shared_intermediate_size=2048,
        expert_intermediate_size=2048, index_n_heads=32, index_head_dim=128, index_topk=2048,
        num_routed_experts=256, max_position_embeddings=202752,
        sparse_prefill=sparse_prefill,
    )


def _weights():
    return TP.Glm5DecodeWeights(
        q_a_norm_weight=torch.ones(2048, dtype=torch.bfloat16),
        q_a_norm_eps=1e-5,
        kv_a_norm_weight=torch.ones(512, dtype=torch.bfloat16),
        kv_a_norm_eps=1e-5,
        indexer_weights_proj=torch.zeros(32, 6144, dtype=torch.bfloat16),
    )


def _geometry(aux_page_split=False, primary_width=3169, aux_width=3169):
    return TP.Glm5DecodeGeometry(
        primary_width=primary_width, aux_width=aux_width,
        primary_page_size=64, aux_page_size=64,
        aux_page_split=aux_page_split, max_position_embeddings=202752,
    )


class _Recorder:
    """A fake wrapper that keeps every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
def test_shapes_from_the_glm5_config_fields():
    config = SimpleNamespace(
        hidden_size=6144, q_lora_rank=2048, kv_lora_rank=512,
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
        num_attention_heads=64, intermediate_size=12288,
        moe_intermediate_size=2048, n_shared_experts=1, index_n_heads=32,
        index_head_dim=128, index_topk=2048, n_routed_experts=256,
        max_position_embeddings=202752, model_type="glm_moe_dsa_5_2",
    )
    shapes = TP.Glm5PrewarmShapes.from_config(config)
    assert shapes == _shapes()
    assert (shapes.qkv_a_width, shapes.kv_dim, shapes.attn_out_dim) == (
        2624, 576, 16384)
    assert (shapes.q_b_cols, shapes.wq_b_cols, shapes.kv_b_rows) == (
        16384, 4096, 28672)
    config.model_type = "glm_moe_dsa"
    assert not TP.Glm5PrewarmShapes.from_config(config).sparse_prefill
    del config.index_topk
    with pytest.raises(ValueError, match="index_topk"):
        TP.Glm5PrewarmShapes.from_config(config)


def test_checkpoint_fp8_dtype_follows_the_quantization_config():
    fp8 = SimpleNamespace(quantization_config={"quant_method": "fp8",
                                               "fmt": "e4m3"})
    assert TP.checkpoint_fp8_dtype(fp8) is torch.float8_e4m3fn
    assert TP.checkpoint_fp8_dtype(SimpleNamespace()) is None
    assert TP.checkpoint_fp8_dtype(SimpleNamespace(
        quantization_config={"quant_method": "fp8", "fmt": "e5m2"})) is None


# --------------------------------------------------------------------------- #
#  Blocked FP8 cast classes
# --------------------------------------------------------------------------- #
def test_align4_is_the_wrappers_tma_scale_layout():
    for rows in range(1, 70):
        scale = torch.empty((16, TP.align4(rows))).transpose(0, 1)[:rows, :]
        assert scale.stride() == TP.quant_scale_strides(rows, 2048, True)
    assert TP.quant_scale_strides(5, 6144, False) == (48, 1)
    assert TP.quant_scale_strides(5, 128, False) == (1, 1)


@pytest.mark.parametrize("phase", [TP.PHASE_PREFILL, TP.PHASE_DECODE])
def test_row_witnesses_cover_every_quant_class(phase):
    for combo in TP.quant_combos(_shapes(), phase):
        witnessed = {TP.quant_key(r, combo.dim, combo.has_valid, combo.tma)
                     for r in TP.ROW_WITNESSES}
        reached = {TP.quant_key(r, combo.dim, combo.has_valid, combo.tma)
                   for r in range(1, 4097)}
        assert reached == witnessed, combo


def test_every_row_witness_is_needed_for_the_tma_layout():
    for dropped in TP.ROW_WITNESSES:
        kept = [r for r in TP.ROW_WITNESSES if r != dropped]
        witnessed = {TP.quant_key(r, 6144, True, True) for r in kept}
        assert TP.quant_key(dropped, 6144, True, True) not in witnessed


def test_row_witnesses_cover_the_fused_norm_and_silu_casts():
    for key, args in ((TP.silu_mul_quant_key, (2048,)),
                      (TP.rmsnorm_quant_key, (2048, True, True))):
        witnessed = {key(r, *args) for r in TP.ROW_WITNESSES}
        assert {key(r, *args) for r in range(1, 4097)} == witnessed


def test_quant_combos_per_phase():
    def dims(combos, valid, tma):
        return sorted(c.dim for c in combos
                      if (c.has_valid, c.tma) == (valid, tma))

    prefill = TP.quant_combos(_shapes(), TP.PHASE_PREFILL)
    assert dims(prefill, False, False) == [128, 512, 2048, 6144, 12288, 16384]
    assert dims(prefill, False, True) == [6144]
    assert dims(prefill, True, True) == []
    dense = TP.quant_combos(_shapes(sparse_prefill=False), TP.PHASE_PREFILL)
    assert dims(dense, False, False) == [512, 2048, 6144, 12288, 16384]
    decode = TP.quant_combos(_shapes(), TP.PHASE_DECODE)
    assert dims(decode, False, False) == [2048, 6144, 12288, 16384]
    assert dims(decode, False, True) == [6144]
    assert dims(decode, True, True) == [2048, 6144, 16384]
    assert len({c.name for c in decode}) == len(decode)


# --------------------------------------------------------------------------- #
#  Entries per bucket
# --------------------------------------------------------------------------- #
def _decode_names(buckets, deepep=False, fuse=True):
    return [e.name for e in TP.decode_entries(
        _shapes(), _weights(), buckets=buckets,
        deepep=deepep, router_row_range=ROUTER_RANGE,
        fuse_q_a_norm_quant=fuse, device="cpu",
    )]


def _geometry_names(buckets, aux_page_split=False):
    return [e.name for e in TP.geometry_entries(
        _shapes(), _weights(), _geometry(aux_page_split), buckets=buckets,
        q_row_widths={16384, 20480}, primary_dtype=torch.bfloat16,
        aux_dtype=torch.bfloat16, device="cpu",
    )]


def test_one_head_gate_entry_per_bucket_and_bucket_free_router():
    names = _decode_names(BUCKETS)
    assert [n for n in names if n.startswith("head_gates/")] == [
        f"head_gates/b{b}" for b in BUCKETS]
    # The router's keys do not depend on the bucket: one entry per row class
    # of the split-K range, with or without the bucket list or DeepEP.
    router = ["router_split_k/rows64", "router_split_k/rows65"]
    assert [n for n in names if n.startswith("router_split_k/")] == router
    for other in (_decode_names(None), _decode_names(BUCKETS, deepep=True)):
        assert [n for n in other if n.startswith("router_split_k/")] == router
    assert not [n for n in names if n.startswith("masked_act_quant/")]
    deepep = _decode_names(BUCKETS, deepep=True)
    assert [n for n in deepep if n.startswith("masked_act_quant/")] == [
        f"masked_act_quant/i2048/s{s}" for s in TP.MASKED_ROW_STRIDE_WITNESSES]
    assert "rmsnorm_fp8_quant/q_a" in names
    assert "rmsnorm/q_a" in _decode_names(BUCKETS, fuse=False)


def test_one_paged_score_entry_per_bucket_on_the_aos_cache():
    names = _geometry_names(BUCKETS)
    scores = [n for n in names if n.startswith("paged_score/")]
    assert scores == [f"paged_score/b{b}/w3169/s202752" for b in BUCKETS]
    assert "paged_mqa_prepare/w3169" not in names
    split = _geometry_names(BUCKETS, aux_page_split=True)
    assert not [n for n in split if n.startswith(("paged_score/",
                                                  "kv_write/aux",
                                                  "eager_score"))]
    assert "paged_mqa_prepare/w3169" in split
    assert {"rmsnorm_rope/graph/q16384/w3169", "rmsnorm_rope/graph/q20480/w3169",
            "selected_positions/graph/w3169"} <= set(split)


def test_without_buckets_row_classes_cover_class_keyed_entries():
    names = _decode_names(None)
    assert [n for n in names if n.startswith("head_gates/")] == [
        f"head_gates/rows{r}" for r in TP.ROW_WITNESSES]
    scores = [n for n in _geometry_names(None) if n.startswith("paged_score/")]
    assert scores == [f"paged_score/rows{r}/w3169/s202752"
                      for r in TP.ROW_WITNESSES]


def test_prefill_entries():
    names = [e.name for e in TP.prefill_entries(
        _shapes(), "cpu", kv_b_dtype=torch.float8_e4m3fn)]
    assert {"silu_mul_quant/d2048", "offset_topk/k2048",
            "kv_b_dequant/28672x512", "act_quant/plain/d128",
            "act_quant/tma/d6144"} <= set(names)
    dense = [e.name for e in TP.prefill_entries(_shapes(False), "cpu")]
    assert not [n for n in dense if n.startswith(("offset_topk",
                                                  "kv_b_dequant"))]


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #
def test_router_row_range_and_rows():
    assert ROUTER_RANGE == (64, 512)
    assert TP.router_row_witnesses(ROUTER_RANGE) == (64, 65)
    # One witness per class the range contains, the first row of each.
    assert TP.router_row_witnesses((1, 512)) == (1, 2, 16)
    assert TP.router_row_witnesses((65, 70)) == (65,)
    assert TP.router_row_witnesses((60, 64)) == (60, 64)
    # The masked cast's row extents: multiples of the 4-float TMA alignment,
    # one divisible by 16 and one not.
    assert all(s % 4 == 0 for s in TP.MASKED_ROW_STRIDE_WITNESSES)
    assert sorted(TP.int_class(s) for s in TP.MASKED_ROW_STRIDE_WITNESSES) == [
        "any", "d16"]


def test_geometry_derivation_matches_hand_computed_values():
    g = _geometry()
    # min(3169 * 64, 3169 * 64, 202752) = 202752
    assert g.graph_max_seqlen == 202752
    assert TP.int_class(g.primary_width) == "any"
    assert g.paged_mqa_block_p == 256          # next_pow2(3169) = 4096 > 256
    assert g.score_block_s == 128
    small = TP.Glm5DecodeGeometry(primary_width=16, aux_width=5,
                                  primary_page_size=64, aux_page_size=64,
                                  aux_page_split=True)
    assert small.graph_max_seqlen == 320       # min(16 * 64, 5 * 64)
    assert small.paged_mqa_block_p == 8
    assert TP.int_class(small.primary_width) == "d16"
    tiny = TP.Glm5DecodeGeometry(primary_width=1, aux_width=1,
                                 primary_page_size=64, aux_page_size=64,
                                 aux_page_split=False)
    assert (tiny.graph_max_seqlen, tiny.score_block_s,
            tiny.paged_mqa_block_p) == (64, 64, 1)
    assert TP.int_class(tiny.aux_width) == "1"
    with pytest.raises(TP.Glm5PrewarmGeometryError):
        TP.Glm5DecodeGeometry(0, 1, 64, 64, False).graph_max_seqlen
    assert g.key() != _geometry(primary_width=3170).key()


def test_dense_gather_block_t_thresholds():
    assert TP.gather_block_t(1, 4096) == 1
    assert TP.gather_block_t(1, 4097) == 8
    assert TP.gather_block_t(8, 4096) == 8      # 32768
    assert TP.gather_block_t(8, 4097) == 32
    assert TP.eager_score_seqlens(2048) == (2049, 2064)
    assert [TP.int_class(s) for s in TP.eager_score_seqlens(2048)] == [
        "any", "d16"]


def test_next_power_of_2():
    assert [TP.next_power_of_2(v) for v in (1, 2, 3, 64, 65, 3169)] == [
        1, 2, 4, 64, 128, 4096]
    with pytest.raises(ValueError):
        TP.next_power_of_2(0)


def test_geometry_runners_bake_the_live_widths():
    g = _geometry(primary_width=3169, aux_width=1584)
    shapes = _shapes()

    rec = _Recorder()
    TP._run_selected_positions(g, 2048, "cpu", fn=rec)
    (args, kwargs), = rec.calls
    assert args[0].shape == (1, 3169) and args[0].dtype == torch.int32
    assert args[2].dtype == torch.int32
    assert kwargs["primary_slot_indices"] is not None
    assert int(kwargs["num_valid_tokens"]) == 1

    rec = _Recorder()
    TP._run_paged_mqa_prepare(g, "cpu", fn=rec, next_n=1)
    (args, kwargs), = rec.calls
    assert args[0].shape == (1, 1584) and args[3].shape == (1, 1584)
    assert args[0].stride(0) == args[3].stride(0) == 1584

    rec = _Recorder()
    TP._run_paged_score(shapes, g, 16, torch.bfloat16, "cpu", fn=rec)
    (args, kwargs), = rec.calls
    assert args[2].shape == (1, 1584)
    assert args[6].shape == (16, g.graph_max_seqlen)
    assert args[7].dtype == torch.int32
    assert kwargs["max_seqlen"] == g.graph_max_seqlen
    assert kwargs["topk"] == 2048 and int(kwargs["num_valid_tokens"]) == 16

    rec = _Recorder()
    TP._run_rope_graph(shapes, _weights().kv_a_norm_weight, 1e-5, 20480,
                       3169, 64, torch.bfloat16, "cpu", rows_list=(13,),
                       fn=rec)
    (args, kwargs), = rec.calls
    assert args[0].shape == (13, 1, 576) and args[0].stride(0) == 2624
    assert args[1].is_contiguous() and args[2].dtype == torch.float32
    assert args[4].dtype == torch.int64
    assert kwargs["q_pe_src"].shape == (13, 64, 1, 64)
    assert kwargs["q_pe_src"].stride()[0] == 20480
    assert kwargs["q_pe_src"].stride()[1] == 256
    assert kwargs["q_pe_src"].stride()[-1] == 1
    assert kwargs["page_table"].shape[1] == 3169
    assert kwargs["cache_tensor"].stride()[:2] == (64 * 576, 576)
    assert kwargs["row_valid_mask"].dtype == torch.bool


# --------------------------------------------------------------------------- #
#  Runners: serving layouts
# --------------------------------------------------------------------------- #
def test_act_quant_runner_flags_and_rows():
    for combo in (TP.QuantCombo(6144, True, True), TP.QuantCombo(512, False,
                                                                  False)):
        rec = _Recorder()
        TP._run_act_quant(combo, "cpu", fn=rec)
        assert [a[0].shape[0] for a, _ in rec.calls] == list(TP.ROW_WITNESSES)
        for (x,), kwargs in rec.calls:
            assert x.dtype == torch.bfloat16 and x.is_contiguous()
            assert x.shape[1] == combo.dim
            assert kwargs["scale_tma_aligned"] is combo.tma
            valid = kwargs["num_valid_tokens"]
            if combo.has_valid:
                assert valid.dtype == torch.int32 and int(valid) == x.shape[0]
            else:
                assert valid is None


def test_q_a_norm_reads_the_qkv_a_column_slice():
    rec = _Recorder()
    w = _weights()
    TP._run_q_a_norm(_shapes(), w.q_a_norm_weight, 1e-5, True, "cpu", fn=rec)
    for (x, weight, eps), kwargs in rec.calls:
        assert x.stride() == (2624, 1) and x.shape[1] == 2048
        assert kwargs["out"].stride() == (2048, 1)
        assert kwargs["scale_tma_aligned"] is True
        assert int(kwargs["num_valid_tokens"]) == x.shape[0]
    rec = _Recorder()
    TP._run_q_a_norm(_shapes(), w.q_a_norm_weight, 1e-5, False, "cpu", fn=rec)
    assert all(set(kw) == {"out"} for _, kw in rec.calls)


def test_eager_rope_is_the_plain_form():
    rec = _Recorder()
    TP._run_rope_eager(_shapes(), _weights().kv_a_norm_weight, 1e-5, "cpu",
                       fn=rec)
    for args, kwargs in rec.calls:
        assert args[0].stride(0) == 576
        assert args[2].dtype == torch.float32
        assert set(kwargs) == {"eps"}


def test_kv_write_covers_every_runtime_int_class():
    rec = _Recorder()
    TP._run_kv_write(128, 64, TP.PAGE_TABLE_COL_WITNESSES,
                     TP.PAGE_TABLE_COL_WITNESSES, False, torch.bfloat16, "cpu",
                     fn=rec)
    seen = {(TP.int_class(kw["k_tokens"].shape[0]),
             TP.int_class(kw["page_table"].shape[1])) for _, kw in rec.calls}
    assert seen == {(r, c) for r in ("1", "d16", "any")
                    for c in ("1", "d16", "any")}
    for _, kw in rec.calls:
        assert kw["k_cache"].stride()[:2] == (64 * 128, 128)
        assert kw["num_valid_tokens"] is None
    rec = _Recorder()
    TP._run_kv_write(576, 64, (16,), (3169,), True, torch.bfloat16, "cpu",
                     fn=rec)
    (_, kw), = rec.calls
    assert kw["page_table"].shape[1] == 3169
    assert int(kw["num_valid_tokens"]) == 16


def test_head_gates_router_and_eager_score_runners():
    shapes = _shapes()
    rec = _Recorder()
    TP._run_head_gates(shapes, _weights().indexer_weights_proj, 32, "cpu",
                       fn=rec)
    (args, kwargs), = rec.calls
    assert args[0].shape == (32, 6144) and args[2].dtype == torch.float32
    assert kwargs["scale"] == (32 ** -0.5) * (128 ** -0.5)

    rec = _Recorder()
    TP._run_router(shapes, 64, "cpu", fn=rec)
    (args, _), = rec.calls
    assert [tuple(a.shape) for a in args] == [(64, 6144), (256, 6144),
                                              (64, 256)]
    assert args[2].dtype == torch.float32

    # The masked cast as run_deepgemm_masked_moe hands it: gate/up are the
    # column halves of one [E, S, 2I] buffer, the scales the (1, 2)
    # transpose of a contiguous [E, I/128, S] block.
    rec = _Recorder()
    TP._run_masked_act_quant(2048, 20, "cpu", fn=rec)
    (args, kwargs), = rec.calls
    gate, up, out, scales, masked_m = args
    assert gate.shape == up.shape == out.shape == (1, 20, 2048)
    assert gate.stride() == up.stride() == (20 * 4096, 4096, 1)
    assert up.data_ptr() - gate.data_ptr() == 2048 * gate.element_size()
    assert out.dtype == torch.float8_e4m3fn and out.stride() == (
        20 * 2048, 2048, 1)
    assert scales.shape == (1, 20, 16) and scales.stride() == (16 * 20, 1, 20)
    assert masked_m.dtype == torch.int32 and kwargs == {"group_size": 128}

    rec = _Recorder()
    TP._run_eager_score(shapes, "cpu", rows_list=(1,), fn=rec)
    spans = [args[1].shape[1] for args, _ in rec.calls]
    assert spans == [2049, 2064] and all(s > 2048 for s in spans)
    assert all(args[3].dtype == torch.int32 and args[4] == 2048
               for args, _ in rec.calls)


def test_prefill_runners():
    rec = _Recorder()
    TP._run_offset_topk(2048, "cpu", fn=rec)
    (args, _), = rec.calls
    assert args[0].shape == (1, 2048) and args[0].dtype == torch.int32
    assert TP.int_class(args[0].numel()) == "d16"

    rec = _Recorder()
    TP._run_weight_dequant(28672, 512, torch.float8_e4m3fn, "cpu", fn=rec)
    (args, _), = rec.calls
    assert args[0].dtype == torch.float8_e4m3fn
    assert args[1].shape == (224, 4) and args[1].stride() == (4, 1)

    rec = _Recorder()
    TP._run_silu_mul_quant(2048, "cpu", fn=rec)
    assert [a[0].shape for a, _ in rec.calls] == [
        (r, 4096) for r in TP.ROW_WITNESSES]
    assert all(kw == {"scale_tma_aligned": True} for _, kw in rec.calls)


# --------------------------------------------------------------------------- #
#  Determinism across ranks
# --------------------------------------------------------------------------- #
def _all_names(buckets):
    shapes = _shapes()
    entries = TP.prefill_entries(shapes, "cpu", kv_b_dtype=torch.float8_e4m3fn)
    entries += TP.decode_entries(
        shapes, _weights(), buckets=buckets, deepep=True,
        router_row_range=ROUTER_RANGE, fuse_q_a_norm_quant=True, device="cpu")
    entries += TP.geometry_entries(
        shapes, _weights(), _geometry(), buckets=buckets,
        q_row_widths=[20480, 16384], primary_dtype=torch.bfloat16,
        aux_dtype=torch.bfloat16, device="cpu")
    return [e.name for e in entries]


def test_entry_list_is_deterministic_across_ranks():
    first = _all_names(list(BUCKETS))
    # Another rank: same configuration, buckets in another order.
    second = _all_names(list(reversed(BUCKETS)))
    assert first == second
    for phase_names in (
        [e.name for e in TP.prefill_entries(_shapes(), "cpu")],
        _decode_names(BUCKETS), _geometry_names(BUCKETS),
    ):
        assert len(phase_names) == len(set(phase_names))
    shares = [TP._core.partition(first, i, 8) for i in range(8)]
    assert sorted(n for share in shares for n in share) == sorted(first)


# --------------------------------------------------------------------------- #
#  Section D
# --------------------------------------------------------------------------- #
def _fake_model(absorb, moe_flags):
    class FP8AbsorbWeights:
        pass

    class BF16AbsorbWeights:
        pass

    containers = {"bf16": BF16AbsorbWeights(), "fp8": FP8AbsorbWeights(),
                  None: None}
    layers = [SimpleNamespace(
        self_attn=SimpleNamespace(_fp8_absorb_weights=containers[absorb]),
        mlp=SimpleNamespace(),
    )]
    for use_3d, ready in moe_flags:
        layers.append(SimpleNamespace(
            self_attn=SimpleNamespace(_fp8_absorb_weights=containers[absorb]),
            mlp=SimpleNamespace(use_3d_moe=use_3d, _fp8_blockwise_ready=ready),
        ))
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_section_d_predicates_from_the_model():
    config = SimpleNamespace(use_dense_mla=False, index_topk=2048,
                             hidden_size=6144, enable_deepep=False)
    preds = TP.decode_section_d_predicates(
        _fake_model("bf16", [(True, True), (True, True)]), config)
    assert preds == {"absorb_container": "BF16AbsorbWeights",
                     "moe_3d_ready": True, "use_dense_mla": False,
                     "index_topk": 2048, "hidden_size": 6144,
                     "enable_deepep": False}
    drift = TP.decode_section_d_predicates(
        _fake_model("fp8", [(True, True), (True, False)]), config)
    assert drift["absorb_container"] == "FP8AbsorbWeights"
    assert drift["moe_3d_ready"] is False
    assert TP.decode_section_d_predicates(
        _fake_model(None, []), config)["moe_3d_ready"] is None


def test_section_d_logging(caplog):
    expected = {"absorb_container": "BF16AbsorbWeights", "moe_3d_ready": True,
                "use_dense_mla": False, "index_topk": 2048,
                "hidden_size": 6144, "enable_deepep": False,
                "aux_page_split": True, "eager_width_constexpr": True}
    with caplog.at_level(logging.INFO, logger=TP.logger.name):
        TP.log_section_d(expected, rank=3)
    messages = [r.getMessage() for r in caplog.records]
    assert messages and all(m.startswith("[GLM5_TRITON_WARMUP] rank=3 ")
                            for m in messages)
    text = "\n".join(messages)
    for part in ("_pack_flashmla_query_kernel", "_absorb_fp8_wgmma_kernel",
                 "_scatter_weight_reduce_kernel", "flashmla_backend",
                 "_topk_from_scores_kernel", "_rmsnorm_kernel_large",
                 "page-split", "_dense_paged_gather_kernel"):
        assert part in text
    # The production predicates all hold; the unbounded eager widths are an
    # INFO note, not a configuration drift.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    caplog.clear()
    drift = dict(expected, absorb_container="FP8AbsorbWeights",
                 moe_3d_ready=False, use_dense_mla=True, index_topk=1024,
                 enable_deepep=True)
    with caplog.at_level(logging.INFO, logger=TP.logger.name):
        TP.log_section_d(drift, rank=0)
    warned = "\n".join(r.getMessage() for r in caplog.records
                       if r.levelno == logging.WARNING)
    for part in ("FP8AbsorbWeights", "_triton_compute", "use_dense_mla=True",
                 "index_topk=1024"):
        assert part in warned
    # DeepEP's masked cast is prewarmed: an INFO line, not a drift warning.
    assert "_silu_mul_masked_group_quant_kernel" not in warned
    assert "_silu_mul_masked_group_quant_kernel" in "\n".join(
        r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
#  Kernel signature and strict key
# --------------------------------------------------------------------------- #
def test_offset_topk_kernel_takes_n_elements_at_runtime():
    tree = ast.parse((_GLM5_DIR / "sparse_prefill.py").read_text())
    kernel, = [node for node in tree.body
               if isinstance(node, ast.FunctionDef)
               and node.name == "_offset_topk_indices_kernel"]
    annotations = {arg.arg: ast.unparse(arg.annotation) if arg.annotation
                   else None for arg in kernel.args.args}
    assert annotations["n_elements"] is None
    assert annotations["topk"] == "tl.constexpr"
    assert annotations["block"] == "tl.constexpr"


def test_router_split_k_kernels_take_the_split_stride_at_runtime():
    """``stride_ps`` is ``rows * n_experts``; as a constexpr every row count
    in the split-K range would be its own key."""
    tree = ast.parse((_ROOT / "batchgen_kernels" / "triton"
                      / "glm5_router_gemm.py").read_text())
    for name in ("_glm5_router_gemm_split_k_kernel",
                 "_glm5_router_split_k_reduce_kernel"):
        kernel, = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        annotations = {arg.arg: ast.unparse(arg.annotation) if arg.annotation
                       else None for arg in kernel.args.args}
        assert annotations["m"] is None and annotations["stride_ps"] is None
        assert annotations["stride_pm"] == "tl.constexpr"


def test_strict_key_is_glm5_specific():
    assert TP.STRICT_DEBUG_KEY == "glm5_triton_strict"
    assert TP.LOG_TAG == "[GLM5_TRITON_WARMUP]"
    assert TP.strict_requested({"glm5_triton_strict": True})
    assert not TP.strict_requested({"k3_triton_strict": True})
    debug = {}
    guard = TP._core.LateCompileGuard(debug_source=lambda: debug,
                                      strict_keys=(TP.STRICT_DEBUG_KEY,))
    guard.arm()
    assert guard.state() == TP.LOG
    debug[TP.STRICT_DEBUG_KEY] = "on"
    assert guard.state() == TP.STRICT


def test_guard_restore_round_trips_every_mode():
    guard = TP._core.LateCompileGuard(debug_source=dict)
    for mode in (TP.DISARMED, TP.LOG, TP.STRICT):
        TP._restore_guard(guard, mode)
        assert guard.armed_mode == mode


# --------------------------------------------------------------------------- #
#  Import hygiene
# --------------------------------------------------------------------------- #
def test_imports_without_triton(monkeypatch):
    for name in ("triton", "triton.language", "fla"):
        monkeypatch.setitem(sys.modules, name, None)
    module = _load_prewarm("_glm5_triton_prewarm_no_gpu_deps")
    assert callable(module.prewarm_glm5_prefill)
    assert callable(module.prewarm_glm5_geometry)
