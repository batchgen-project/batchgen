# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                             #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""After the GLM-5 Triton prewarm, random serving shapes compile nothing.

The prewarm's prefill, decode and geometry entries run at the GLM-5.2
dimensions on a fresh ``TRITON_CACHE_DIR`` (synthetic weights; the keys
depend on shapes, dtypes and strides only). The late-compile guard is then
armed strict and random shapes run through the same serving wrappers:
decode rows 1..128 including non-bucket values, prefill token counts up to
the micro-batch cap, eager page-table widths of any class, the graph's page
widths, and both aux cache formats (page-split FP8 as on GLM-5.2, bf16 AoS
as on GLM-5.3). A ``LateTritonCompileError`` names the kernel and the key
the witness sets missed. The test also asserts that the recorder saw no
compile after the guard was armed and that the cache directory gained no
file.

Not covered here: the eager decode's constexpr page-table widths
(``_transform_selected_positions_kernel`` and ``_dense_paged_gather_kernel``
over the ACTIVE table) and the DeepEP masked activation, which the prewarm
logs as unbounded / not prewarmed.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
import torch


if os.environ.get("GLM5_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError("GLM5_GPU_STAGE=1 requires a visible CUDA device")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="staged GLM-5 CUDA test"
)

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
WORLD_SIZE = 8
MAX_PREFILL_TOKENS = 131_072  # ModuleBatchingConfig.prefill_micro_batch_token_cap default
# Keep one prefill cast under 2**30 elements (a 2 GiB bf16 input).
MAX_PREFILL_ELEMENTS = 1 << 30


def _shapes():
    from batchgen.models.glm.glm5 import triton_prewarm as tp

    return tp.Glm5PrewarmShapes(
        hidden_size=6144, q_lora_rank=2048, kv_lora_rank=512,
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
        num_heads=64, intermediate_size=12288, shared_intermediate_size=2048,
        index_n_heads=32, index_head_dim=128, index_topk=2048,
        num_routed_experts=256, max_position_embeddings=202752,
        sparse_prefill=True,
    )


def _weights():
    from batchgen.models.glm.glm5 import triton_prewarm as tp

    torch.manual_seed(20261101)
    return tp.Glm5DecodeWeights(
        q_a_norm_weight=torch.randn(2048, dtype=torch.bfloat16, device="cuda"),
        q_a_norm_eps=1e-5,
        kv_a_norm_weight=torch.randn(512, dtype=torch.bfloat16, device="cuda"),
        kv_a_norm_eps=1e-5,
        indexer_weights_proj=torch.randn(32, 6144, dtype=torch.bfloat16,
                                         device="cuda"),
    )


def _cache_files(root: Path) -> set:
    return {p for p in root.rglob("*") if p.is_file()} if root.exists() else set()


@pytest.mark.parametrize("aux_page_split", [True, False],
                         ids=["page_split_fp8", "bf16_aos"])
def test_random_serving_shapes_compile_nothing_after_prewarm(
        tmp_path, monkeypatch, aux_page_split):
    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen.models.glm.glm5 import triton_prewarm as tp
    from batchgen.models.glm.glm5.norm_quant_fuse_policy import (
        glm5_norm_quant_fuse_site_enabled,
    )
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        fused_score_and_topk,
    )
    from batchgen_kernels.triton.glm5_router_gemm_config import (
        ROUTER_SPLIT_K_MAX_ROWS,
        ROUTER_SPLIT_K_MIN_ROWS,
    )

    cache_dir = tmp_path / "triton"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir))
    recorder = tp.get_recorder()
    if not recorder.install():
        pytest.skip("this Triton exposes no compile hooks")

    device = torch.device("cuda", torch.cuda.current_device())
    shapes, weights = _shapes(), _weights()
    geometry = tp.Glm5DecodeGeometry(
        primary_width=3169, aux_width=3169, primary_page_size=64,
        aux_page_size=64, aux_page_split=aux_page_split,
        max_position_embeddings=shapes.max_position_embeddings,
    )
    aux_dtype = torch.uint8 if aux_page_split else torch.bfloat16
    fuse_q_a = glm5_norm_quant_fuse_site_enabled("q_a")
    row_range = (ROUTER_SPLIT_K_MIN_ROWS, ROUTER_SPLIT_K_MAX_ROWS)
    entries = tp.prefill_entries(shapes, device,
                                 kv_b_dtype=torch.float8_e4m3fn)
    entries += tp.decode_entries(
        shapes, weights, buckets=BUCKETS, world_size=WORLD_SIZE, deepep=False,
        router_row_range=row_range, fuse_q_a_norm_quant=fuse_q_a,
        device=device,
    )
    entries += tp.geometry_entries(
        shapes, weights, geometry, buckets=BUCKETS,
        q_row_widths={shapes.q_b_cols, shapes.q_b_cols + shapes.wq_b_cols},
        primary_dtype=torch.bfloat16, aux_dtype=aux_dtype, device=device,
    )
    with torch.inference_mode():
        reports = tp._core.run_prewarm(entries, recorder=recorder)
    assert [r.failed for r in reports] == [0]

    rng = random.Random(20261101 + int(aux_page_split))
    before = _cache_files(cache_dir)
    mark = recorder.mark()
    recorder.guard.arm(strict=True)
    try:
        with torch.inference_mode():
            for _ in range(48):
                # Decode casts at any row count, buckets or not.
                rows = rng.randint(1, 128)
                for combo in tp.quant_combos(shapes, tp.PHASE_DECODE):
                    tp._run_act_quant(combo, device, rows_list=(rows,))
                tp._run_silu_mul_quant(shapes.shared_intermediate_size,
                                       device, rows_list=(rows,))
                tp._run_q_a_norm(shapes, weights.q_a_norm_weight, 1e-5,
                                 fuse_q_a, device, rows_list=(rows,))
                tp._run_rope_eager(shapes, weights.kv_a_norm_weight, 1e-5,
                                   device, rows_list=(rows,))
                # Prefill casts up to the micro-batch cap.
                tokens = rng.choice((rng.randint(1, 64),
                                     rng.randint(1, 8192),
                                     rng.randint(1, MAX_PREFILL_TOKENS)))
                for combo in tp.quant_combos(shapes, tp.PHASE_PREFILL):
                    capped = min(tokens, MAX_PREFILL_ELEMENTS // combo.dim)
                    tp._run_act_quant(combo, device, rows_list=(capped,))
                # The sparse prefill's top-k offset over one score chunk.
                chunk = rng.randint(1, 8192)
                tp._run_offset_topk(shapes.index_topk, device, rows=chunk)
                # Graph-only kernels at the captured buckets and widths.
                bucket = rng.choice(BUCKETS)
                tp._run_head_gates(shapes, weights.indexer_weights_proj,
                                   bucket, device)
                if row_range[0] <= WORLD_SIZE * bucket <= row_range[1]:
                    tp._run_router(shapes, WORLD_SIZE * bucket, device)
                for width in (shapes.q_b_cols,
                              shapes.q_b_cols + shapes.wq_b_cols):
                    tp._run_rope_graph(
                        shapes, weights.kv_a_norm_weight, 1e-5, width,
                        geometry.primary_width, geometry.primary_page_size,
                        torch.bfloat16, device, rows_list=(bucket,),
                    )
                tp._run_selected_positions(geometry, shapes.index_topk,
                                           device)
                # Eager writes into an active table of any width.
                eager_width = rng.randint(1, geometry.primary_width)
                tp._run_kv_write(shapes.kv_dim, 64, (rows,), (eager_width,),
                                 False, torch.bfloat16, device)
                if aux_page_split:
                    tp._run_paged_mqa_prepare(geometry, device)
                    continue
                tp._run_paged_score(shapes, geometry, bucket, aux_dtype,
                                    device)
                tp._run_kv_write(shapes.index_head_dim, 64, (bucket,),
                                 (geometry.aux_width,), True, aux_dtype,
                                 device)
                tp._run_kv_write(shapes.index_head_dim, 64, (rows,),
                                 (eager_width,), False, aux_dtype, device)
                span = rng.randint(shapes.index_topk + 1, 8192)
                fused_score_and_topk(
                    torch.zeros(rows, 32, 128, dtype=torch.bfloat16,
                                device=device),
                    torch.zeros(rows, span, 128, dtype=torch.bfloat16,
                                device=device),
                    torch.zeros(rows, 32, dtype=torch.float32, device=device),
                    torch.zeros(rows, dtype=torch.int32, device=device),
                    shapes.index_topk,
                )
            # The packed sparse prefill's per-token activation cast.
            act_quant(torch.zeros(4 * 32, 128, dtype=torch.bfloat16,
                                  device=device).view(4, 32, 128))
            torch.cuda.synchronize()
    finally:
        recorder.guard.disarm()
    assert recorder.counts(mark)["compiled"] == 0
    assert _cache_files(cache_dir) - before == set()
