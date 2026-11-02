# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                             #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""After the GLM-5 DeepGEMM prewarm, random serving shapes compile nothing.

The prewarm's DeepGEMM entries for all three phases run at the GLM-5.2
dimensions (DeepEP MoE with 32 local experts, page-split indexer cache,
merged q_b/wq_b, DeepGEMM wk) on an EMPTY ``DG_JIT_CACHE_DIR``. Random
serving-shaped calls then go through the same serving functions: every
Normal site at row counts up to the prefill micro-batch cap (capped per call
by memory), the masked one-group GEMMs and the MoE masked GEMMs at random
buckets up to the slot capacity, the paged-MQA scorer at random batch sizes
and the sparse prefill's ``fp8_mqa_logits`` at random row counts and score
widths. The cache directory must gain no kernel, which the prewarm's own
late-compile check confirms in strict mode.

DeepGEMM resolves its cache directory per process, so run this file in its
own pytest process: ``python -m pytest tests/gpu/test_glm5_deepgemm_prewarm.py``.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

# Before anything imports deep_gemm: an empty cache, so every kernel the
# serving calls need must come from the prewarm.
os.environ["DG_JIT_CACHE_DIR"] = os.environ.get(
    "GLM5_DEEPGEMM_TEST_CACHE_DIR"
) or tempfile.mkdtemp(prefix="glm5_deepgemm_prewarm_")

import pytest  # noqa: E402
import torch  # noqa: E402


if os.environ.get("GLM5_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError("GLM5_GPU_STAGE=1 requires a visible CUDA device")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="staged GLM-5 CUDA test"
)

MAX_ROWS = 131_072  # ModuleBatchingConfig.prefill_micro_batch_token_cap default
SLOTS = 128         # the worker's default decode graph bucket cap
WORLD_SIZE = 8
TOPK = 8
LOCAL_EXPERTS = 32
# Keep one random activation under 2**29 elements (a 1 GiB bf16 tensor).
MAX_ELEMENTS = 1 << 29


def _shapes():
    from batchgen.models.glm.glm5 import triton_prewarm as tp

    return tp.Glm5PrewarmShapes(
        hidden_size=6144, q_lora_rank=2048, kv_lora_rank=512,
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
        num_heads=64, intermediate_size=12288, shared_intermediate_size=2048,
        expert_intermediate_size=2048, index_n_heads=32, index_head_dim=128,
        index_topk=2048, num_routed_experts=256,
        max_position_embeddings=1048576, sparse_prefill=True,
    )


def _moe_stacks(device):
    from batchgen.models.glm.glm5 import deepgemm_prewarm as dg

    e, h, i = LOCAL_EXPERTS, 6144, 2048
    fp8 = torch.float8_e4m3fn
    return dg.MoeStacks(
        gate_up=torch.zeros((e, 2 * i, h), dtype=fp8, device=device),
        gate_up_scale=torch.ones((e, 2 * i // 128, h // 128), device=device),
        down=torch.zeros((e, h, i), dtype=fp8, device=device),
        down_scale=torch.ones((e, h // 128, i // 128), device=device),
    )


def _kernel_dirs(cache: Path) -> set:
    if not cache.exists():
        return set()
    return {p.name for p in cache.iterdir()
            if p.name.startswith("kernel.") and p.is_dir()}


def _paged_mqa_call(batch: int, device) -> None:
    """The eager scorer's DeepGEMM half at an arbitrary batch size."""
    from batchgen_kernels.attention.dsa.paged_mqa_logits import (
        PAGED_MQA_NEXT_N,
        PagedMqaLogitsPlan,
        align_context_len,
    )
    from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
        deep_gemm_paged_mqa_logits,
        paged_mqa_logits_schedule_meta,
    )

    cache = torch.zeros((2, 64, 1, 132), dtype=torch.uint8, device=device)
    q = torch.zeros((batch, PAGED_MQA_NEXT_N, 32, 128),
                    dtype=torch.float8_e4m3fn, device=device)
    gates = torch.zeros((batch, 32), dtype=torch.float32, device=device)
    lens = torch.full((batch, PAGED_MQA_NEXT_N), 100, dtype=torch.int32,
                      device=device)
    table = torch.zeros((batch, 2), dtype=torch.int32, device=device)
    plan = PagedMqaLogitsPlan(
        batch=batch, next_n=PAGED_MQA_NEXT_N, num_heads=32, head_dim=128,
        num_pages=2, page_size=64, max_pages_per_seq=2, topk=1,
        max_seqlen=128, max_context_len=align_context_len(128),
    )
    deep_gemm_paged_mqa_logits(q, cache, gates, lens, table, plan,
                               paged_mqa_logits_schedule_meta(lens))


def _mqa_logits_call(rows: int, width: int, device) -> None:
    import deep_gemm

    q = torch.zeros((rows, 32, 128), dtype=torch.float8_e4m3fn, device=device)
    k = torch.zeros((width, 128), dtype=torch.float8_e4m3fn, device=device)
    deep_gemm.fp8_mqa_logits(
        q, (k, torch.ones((width,), device=device)),
        torch.zeros((rows, 32), device=device),
        torch.zeros((rows,), dtype=torch.int32, device=device),
        torch.ones((rows,), dtype=torch.int32, device=device),
        max_seqlen_k=width, clean_logits=False,
    )


def test_random_serving_shapes_compile_nothing_after_prewarm():
    from batchgen.models.glm.glm5 import deepgemm_prewarm as dg
    from batchgen.models.glm.glm5 import triton_prewarm as tp

    cache = dg.jit_cache_dir()
    assert _kernel_dirs(cache) == set(), (
        f"{cache} is not empty; point GLM5_DEEPGEMM_TEST_CACHE_DIR at an "
        "empty directory")
    device = torch.device("cuda", torch.cuda.current_device())
    shapes = _shapes()
    num_sms = dg.deepgemm_num_sms()
    stacks = _moe_stacks(device)
    entries = dg.prefill_entries(shapes, device, max_rows=MAX_ROWS,
                                 num_sms=num_sms)
    entries += dg.decode_entries(shapes, device, max_rows=MAX_ROWS,
                                 num_sms=num_sms, merge_q_b_wq_b=True,
                                 wk_deepgemm=True)
    entries += dg.geometry_entries(
        shapes, device, max_decode_rows=SLOTS, num_sms=num_sms,
        merge_q_b_wq_b=True, page_split=True, moe_stacks=stacks,
        world_size=WORLD_SIZE, topk=TOPK,
    )
    with torch.inference_mode():
        reports = tp._core.run_prewarm(entries)
    assert [r.failed for r in reports] == [0]
    prewarmed = _kernel_dirs(cache)
    assert prewarmed, (
        "the prewarm compiled nothing into DG_JIT_CACHE_DIR; DeepGEMM was "
        "imported before this module set it: run the file on its own")
    dg.record_baseline(0)

    rng = random.Random(20261102)
    dense = {(s.via, s.n, s.k): s
             for phase in (dg.PHASE_PREFILL, dg.PHASE_DECODE)
             for s in dg.dense_sites(shapes, phase)}
    masked = dg.masked_dense_sites(shapes)
    moe = ((stacks.gate_up, stacks.gate_up_scale),
           (stacks.down, stacks.down_scale))
    with torch.inference_mode():
        for _ in range(16):
            for site in dense.values():
                cap = min(MAX_ROWS, MAX_ELEMENTS // max(site.k, site.n))
                rows = rng.choice((rng.randint(1, 64), rng.randint(1, 8192),
                                   rng.randint(1, cap)))
                dg._run_dense(site, rows, device)
            bucket = rng.randint(1, SLOTS)
            for site in masked:
                dg._run_masked_dense(site, bucket, device)
            expected = dg.expected_m_per_expert(
                local_rows=bucket, world_size=WORLD_SIZE, topk=TOPK,
                num_experts_global=LOCAL_EXPERTS * WORLD_SIZE)
            for weight, scale in moe:
                dg._run_masked_moe(weight, scale, expected, device)
            _paged_mqa_call(rng.randint(1, SLOTS), device)
            _mqa_logits_call(rng.randint(1, 64), 256 * rng.randint(1, 32),
                             device)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    assert dg.check_late_compiles(0, "random serving shapes", strict=True) == 0
    assert _kernel_dirs(cache) == prewarmed
