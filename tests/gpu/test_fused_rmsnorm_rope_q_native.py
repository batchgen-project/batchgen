# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                            #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""REMOTE-ONLY GPU exactness gate for ``fused_rmsnorm_rope_with_q_native``.

Run on a GPU machine (one device is enough), never on the dev box:

    pytest tests/gpu/test_fused_rmsnorm_rope_q_native.py -q -rA

Set ``BATCHGEN_FUSED_ROPE_GPU_STAGE=1`` to fail closed instead of skipping
when no CUDA device is visible.

The kernel's head axis moved from a serial in-program loop to the launch
grid.  That is a pure repartitioning: every output element is still produced
by one program, from the same inputs, with the same expression in the same
dtypes, so the requirement here is **bitwise** equality with the pre-change
kernel -- not a tolerance.  ``_reference_native`` below is a frozen,
verbatim copy of the pre-change kernel (``grid = (bsz,)``, default
``num_warps``), kept in the test so the oracle cannot drift with the
production file.

The production launch later widened the kv-lora reduction from the original
``BLOCK_SIZE=64`` serial loop to a single pass (``min(512,
next_pow2(kv_lora_rank))``).  The ORACLE KERNEL CODE is unchanged by that:
its value is pinning the arithmetic, so ``_reference_native`` now launches
the frozen code at the SAME block width as production (``_mirror_block``),
which keeps every bitwise assertion meaningful -- identical arithmetic at
identical width must agree exactly.  The one place the old 64-wide launch is
still compared directly is ``test_glm52_shape_new_width_matches_old_64wide``,
which pins the measured property that at GLM-5.2 geometry the bf16 STORED
output is unchanged by the width change (fp32 inv_rms reassociation is
absorbed by bf16 rounding there).

What this file pins:

  A  Bitwise agreement with the pre-change kernel across the decode buckets
     at GLM-5.2 geometry (64 q heads, kv_lora_rank 512, qk_rope_head_dim 64),
     on both the rotated q_pe and the returned processed KV.
  B  Bitwise agreement on awkward geometry: head counts that do not divide
     the tile, a kv_lora_rank that is not a multiple of BLOCK_SIZE, and a
     tiny rope dim.
  C  The strided-kv-input contract (a non-contiguous ``new_compressed_kv``
     must give the same answer as its contiguous clone) still holds.
  D  Tiling independence: forcing each ``BLOCK_H`` the policy can emit for
     one geometry gives bit-identical results, which is what makes the result
     independent of the CUDA-graph bucket the launch was sized for.
  E  Row isolation: a row's output depends only on that row, so padded rows
     in a capture bucket cannot perturb the live rows.
  F  CUDA-graph capture/replay reproduces the eager result bitwise.

Why bitwise and not a ULP band: the RoPE arithmetic is element-wise, and the
RMSNorm reduction is compared at the SAME block width on both sides (oracle
launch mirrors the production width), so ``inv_rms`` is computed by the same
instruction sequence.  A mismatch here is a real finding, not reassociation
noise, and should be reported rather than absorbed into a tolerance.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

# Module level so a CPU-only box skips the file at collection instead of
# failing to import it; the frozen oracle below is a real @triton.jit kernel.
triton = pytest.importorskip("triton", reason="frozen Triton oracle kernel")
tl = pytest.importorskip(
    "triton.language", reason="frozen Triton oracle kernel"
)

STAGE_ENV = "BATCHGEN_FUSED_ROPE_GPU_STAGE"

# GLM-5.2 decode geometry (config.json: 64 attention heads, kv_lora_rank 512,
# qk_rope_head_dim 64; attention is data-parallel so a rank sees whole rows).
GLM52_HEADS = 64
GLM52_KV_LORA_RANK = 512
GLM52_ROPE_DIM = 64
EPS = 1e-6

if os.environ.get(STAGE_ENV) == "1" and not torch.cuda.is_available():
    raise RuntimeError(f"{STAGE_ENV}=1 requires a visible CUDA device")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="staged fused RMSNorm+RoPE CUDA exactness test",
)


# --------------------------------------------------------------------------- #
#  Frozen pre-change kernel (the oracle).  Do not refactor: its value is that
#  it is byte-for-byte the code this change replaced.
# --------------------------------------------------------------------------- #
@triton.jit
def _reference_native_kernel(
    new_compressed_kv_ptr,
    processed_kv_ptr,
    q_pe_ptr,
    cos_ptr,
    sin_ptr,
    position_ids_ptr,
    norm_weight_ptr,
    input_batch_stride,
    bsz,
    num_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    variance_epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    total_dim = kv_lora_rank + qk_rope_head_dim
    input_offset = batch_idx * input_batch_stride
    output_offset = batch_idx * total_dim
    pos_id = tl.load(position_ids_ptr + batch_idx)

    sum_sq_fp32 = 0.0
    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        data_bf16 = tl.load(
            new_compressed_kv_ptr + input_offset + offsets,
            mask=mask,
            other=0.0,
        )
        data_fp32 = data_bf16.to(tl.float32)
        sum_sq_fp32 += tl.sum(data_fp32 * data_fp32)

    variance = sum_sq_fp32 / kv_lora_rank
    inv_rms = 1.0 / tl.sqrt(variance + variance_epsilon)

    for block_start in range(0, kv_lora_rank, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < kv_lora_rank
        data_bf16 = tl.load(
            new_compressed_kv_ptr + input_offset + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=1.0)
        data_fp32 = data_bf16.to(tl.float32)
        normalized_fp32 = data_fp32 * inv_rms * weight.to(tl.float32)
        tl.store(
            processed_kv_ptr + output_offset + offsets,
            normalized_fp32.to(data_bf16.dtype),
            mask=mask,
        )

    input_rope_offset = input_offset + kv_lora_rank
    processed_rope_offset = output_offset + kv_lora_rank
    half_dim = qk_rope_head_dim // 2
    cos_sin_base = pos_id * qk_rope_head_dim
    q_pe_batch_stride = num_heads * qk_rope_head_dim
    q_pe_head_stride = qk_rope_head_dim

    for block_start in range(0, half_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim

        cos_i = tl.load(cos_ptr + cos_sin_base + offsets, mask=mask, other=1.0)
        sin_i = tl.load(sin_ptr + cos_sin_base + offsets, mask=mask, other=0.0)

        even_indices = offsets * 2
        odd_indices = offsets * 2 + 1
        kv_even = tl.load(
            new_compressed_kv_ptr + input_rope_offset + even_indices,
            mask=mask,
            other=0.0,
        )
        kv_odd = tl.load(
            new_compressed_kv_ptr + input_rope_offset + odd_indices,
            mask=mask,
            other=0.0,
        )

        kv_rot_even = kv_even * cos_i - kv_odd * sin_i
        kv_rot_odd = kv_even * sin_i + kv_odd * cos_i

        tl.store(
            processed_kv_ptr + processed_rope_offset + even_indices,
            kv_rot_even,
            mask=mask,
        )
        tl.store(
            processed_kv_ptr + processed_rope_offset + odd_indices,
            kv_rot_odd,
            mask=mask,
        )

        for head_idx in range(num_heads):
            q_pe_base_offset = (
                batch_idx * q_pe_batch_stride + head_idx * q_pe_head_stride
            )
            q_even = tl.load(
                q_pe_ptr + q_pe_base_offset + even_indices,
                mask=mask,
                other=0.0,
            )
            q_odd = tl.load(
                q_pe_ptr + q_pe_base_offset + odd_indices,
                mask=mask,
                other=0.0,
            )
            q_rot_even = q_even * cos_i - q_odd * sin_i
            q_rot_odd = q_even * sin_i + q_odd * cos_i
            tl.store(
                q_pe_ptr + q_pe_base_offset + even_indices,
                q_rot_even,
                mask=mask,
            )
            tl.store(
                q_pe_ptr + q_pe_base_offset + odd_indices,
                q_rot_odd,
                mask=mask,
            )


def _mirror_block(kv_lora_rank: int) -> int:
    """The production launch's kv-lora reduction width (keep in step with
    ``fused_rmsnorm_rope_with_q_native``)."""
    return min(512, triton.next_power_of_2(kv_lora_rank))


def _reference_native(
    new_compressed_kv,
    q_pe,
    cos,
    sin,
    position_ids,
    norm_weight,
    kv_lora_rank,
    qk_rope_head_dim,
    eps=1e-6,
    block_size=None,
):
    """Frozen pre-change wrapper: one program per row, default num_warps.

    ``block_size`` defaults to the production launch's width so bitwise
    comparison stays width-matched; pass 64 explicitly to reproduce the
    original pre-change launch.
    """
    bsz, q_len, total_dim = new_compressed_kv.shape
    assert q_len == 1
    assert total_dim == kv_lora_rank + qk_rope_head_dim
    num_heads = q_pe.shape[1]

    processed_kv = torch.empty(
        new_compressed_kv.shape,
        dtype=new_compressed_kv.dtype,
        device=new_compressed_kv.device,
    )
    _reference_native_kernel[(bsz,)](
        new_compressed_kv,
        processed_kv,
        q_pe,
        cos,
        sin,
        position_ids,
        norm_weight,
        new_compressed_kv.stride(0),
        bsz,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        eps,
        BLOCK_SIZE=_mirror_block(kv_lora_rank) if block_size is None else block_size,
    )
    return processed_kv


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def production():
    pytest.importorskip("batchgen_kernels")
    from batchgen_kernels.triton.fused_rmsnorm_rope import (
        fused_rmsnorm_rope_with_q_native,
    )

    return fused_rmsnorm_rope_with_q_native


def _inputs(
    bsz: int,
    num_heads: int = GLM52_HEADS,
    kv_lora_rank: int = GLM52_KV_LORA_RANK,
    rope_dim: int = GLM52_ROPE_DIM,
    max_pos: int = 4096,
    seed: int = 20261016,
):
    """Random decode-shaped inputs.  cos/sin are fp32 as graph capture requires."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + bsz * 1000003 + num_heads * 1009 + rope_dim)
    total_dim = kv_lora_rank + rope_dim

    kv = torch.randn(
        (bsz, 1, total_dim), device="cuda", generator=generator
    ).to(torch.bfloat16)
    q_pe = torch.randn(
        (bsz, num_heads, 1, rope_dim), device="cuda", generator=generator
    ).to(torch.bfloat16)
    cos = torch.randn((max_pos, rope_dim), device="cuda", generator=generator)
    sin = torch.randn((max_pos, rope_dim), device="cuda", generator=generator)
    positions = torch.randint(
        0, max_pos, (bsz, 1), device="cuda", generator=generator, dtype=torch.int64
    )
    weight = torch.randn(
        (kv_lora_rank,), device="cuda", generator=generator
    ).to(torch.bfloat16)
    return kv, q_pe, cos, sin, positions, weight


def _assert_bitwise(got_kv, got_q, want_kv, want_q, label: str) -> None:
    kv_diff = int((got_kv != want_kv).sum().item())
    q_diff = int((got_q != want_q).sum().item())
    assert kv_diff == 0 and q_diff == 0, (
        f"{label}: not bitwise equal to the pre-change kernel -- "
        f"{kv_diff}/{want_kv.numel()} processed-KV elements and "
        f"{q_diff}/{want_q.numel()} q_pe elements differ "
        f"(max |dKV|={(got_kv.float() - want_kv.float()).abs().max().item():.3e}, "
        f"max |dQ|={(got_q.float() - want_q.float()).abs().max().item():.3e})"
    )


def _run_both(production, kv, q_pe, cos, sin, positions, weight,
              kv_lora_rank, rope_dim):
    q_new = q_pe.clone()
    q_ref = q_pe.clone()
    kv_new = production(
        kv, q_new, cos, sin, positions, weight, kv_lora_rank, rope_dim, eps=EPS
    )
    kv_ref = _reference_native(
        kv, q_ref, cos, sin, positions, weight, kv_lora_rank, rope_dim, eps=EPS
    )
    torch.cuda.synchronize()
    return kv_new, q_new, kv_ref, q_ref


# --------------------------------------------------------------------------- #
#  A  Decode buckets at GLM-5.2 geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bsz", [1, 2, 7, 8, 16, 32, 64, 128, 192, 256, 384, 512])
def test_bitwise_against_pre_change_kernel_glm52(production, bsz):
    kv, q_pe, cos, sin, positions, weight = _inputs(bsz)
    kv_new, q_new, kv_ref, q_ref = _run_both(
        production, kv, q_pe, cos, sin, positions, weight,
        GLM52_KV_LORA_RANK, GLM52_ROPE_DIM,
    )
    _assert_bitwise(kv_new, q_new, kv_ref, q_ref, f"glm5.2 bucket bsz={bsz}")


# --------------------------------------------------------------------------- #
#  A2  The width change itself: bf16 output unchanged at GLM-5.2 geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bsz", [64, 512])
def test_glm52_shape_new_width_matches_old_64wide(production, bsz):
    """Production (single-pass reduction) vs the ORIGINAL 64-wide launch.

    fp32 ``inv_rms`` reassociation from the wider block is absorbed by bf16
    rounding at this geometry (probe 2026-10-23 measured exactly 0 diff);
    this pins that property so a future input/geometry drift that breaks it
    is a reported finding, not a silent numerics shift.
    """
    kv, q_pe, cos, sin, positions, weight = _inputs(bsz)
    q_new = q_pe.clone()
    q_old = q_pe.clone()
    kv_new = production(
        kv, q_new, cos, sin, positions, weight,
        GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS,
    )
    kv_old = _reference_native(
        kv, q_old, cos, sin, positions, weight,
        GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS, block_size=64,
    )
    torch.cuda.synchronize()
    _assert_bitwise(
        kv_new, q_new, kv_old, q_old,
        f"glm5.2 bsz={bsz}: single-pass width vs original 64-wide launch",
    )


# --------------------------------------------------------------------------- #
#  B  Awkward geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("bsz", "num_heads", "kv_lora_rank", "rope_dim"),
    [
        (3, 2, 8, 4),  # the shape tests/test_dsa_valid_token_kernels.py uses
        (5, 7, 100, 16),  # head count and lora rank both indivisible
        (64, 3, 512, 64),  # fewer heads than one tile
        (64, 96, 512, 64),  # more heads than GLM-5.2
        (17, 64, 513, 64),  # lora rank one past a BLOCK_SIZE multiple
        (64, 64, 512, 2),  # half_dim 1: a single pair per head
    ],
)
def test_bitwise_on_awkward_geometry(
    production, bsz, num_heads, kv_lora_rank, rope_dim
):
    kv, q_pe, cos, sin, positions, weight = _inputs(
        bsz, num_heads=num_heads, kv_lora_rank=kv_lora_rank, rope_dim=rope_dim
    )
    kv_new, q_new, kv_ref, q_ref = _run_both(
        production, kv, q_pe, cos, sin, positions, weight, kv_lora_rank, rope_dim
    )
    _assert_bitwise(
        kv_new, q_new, kv_ref, q_ref,
        f"bsz={bsz} heads={num_heads} lora={kv_lora_rank} rope={rope_dim}",
    )


# --------------------------------------------------------------------------- #
#  C  Strided kv input contract
# --------------------------------------------------------------------------- #
def test_strided_kv_input_matches_contiguous_and_reference(production):
    bsz, num_heads, kv_lora_rank, rope_dim = 13, GLM52_HEADS, 512, 64
    total_dim = kv_lora_rank + rope_dim
    _, q_pe, cos, sin, positions, weight = _inputs(
        bsz, num_heads=num_heads, kv_lora_rank=kv_lora_rank, rope_dim=rope_dim
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261016)
    base = torch.randn(
        (bsz, total_dim + 5), device="cuda", generator=generator
    ).to(torch.bfloat16)
    strided_kv = base[:, :total_dim].view(bsz, 1, total_dim)
    assert not strided_kv.is_contiguous()
    contiguous_kv = strided_kv.contiguous()

    q_strided = q_pe.clone()
    q_contig = q_pe.clone()
    q_ref = q_pe.clone()
    out_strided = production(
        strided_kv, q_strided, cos, sin, positions, weight,
        kv_lora_rank, rope_dim, eps=EPS,
    )
    out_contig = production(
        contiguous_kv, q_contig, cos, sin, positions, weight,
        kv_lora_rank, rope_dim, eps=EPS,
    )
    out_ref = _reference_native(
        strided_kv, q_ref, cos, sin, positions, weight,
        kv_lora_rank, rope_dim, eps=EPS,
    )
    torch.cuda.synchronize()

    _assert_bitwise(out_strided, q_strided, out_contig, q_contig,
                    "strided vs contiguous kv input")
    _assert_bitwise(out_strided, q_strided, out_ref, q_ref,
                    "strided kv input vs pre-change kernel")


# --------------------------------------------------------------------------- #
#  D  Tiling independence
# --------------------------------------------------------------------------- #
def test_every_legal_block_h_gives_identical_bits():
    """Forcing each BLOCK_H must not move a single bit.

    This is the invariant behind CUDA-graph bucket independence, and it is
    also what lets the policy pick its head tile from measurement alone: the
    width changes only how the heads are partitioned, never a result bit.
    """
    pytest.importorskip("batchgen_kernels")
    # ``batchgen_kernels.triton.__init__`` re-exports a function named
    # ``fused_rmsnorm_rope``, which shadows the submodule of the same name, so
    # ``from batchgen_kernels.triton import fused_rmsnorm_rope`` hands back the
    # function, not the module. Resolve the module by its dotted name, which
    # always comes from ``sys.modules``.
    frr = importlib.import_module("batchgen_kernels.triton.fused_rmsnorm_rope")

    bsz, num_heads, kv_lora_rank, rope_dim = 64, GLM52_HEADS, 512, 64
    kv, q_pe, cos, sin, positions, weight = _inputs(
        bsz, num_heads=num_heads, kv_lora_rank=kv_lora_rank, rope_dim=rope_dim
    )
    _, block_pairs, _ = frr.select_q_rope_tiling(bsz, num_heads, rope_dim // 2)
    fused_rmsnorm_rope_with_q_native_kernel = (
        frr.fused_rmsnorm_rope_with_q_native_kernel
    )

    # Every head-tile width that is legal at this geometry, not just the one
    # the policy currently selects: one head per program up to the widest tile
    # the per-program element budget allows.  A retune of Q_ROPE_BLOCK_H can
    # then only pick a width this test has already proven bit-identical.
    widths = [1]
    while (
        widths[-1] * 2 <= num_heads
        and widths[-1] * 2 * block_pairs <= frr.Q_ROPE_MAX_TILE_ELEMS
    ):
        widths.append(widths[-1] * 2)
    assert frr.Q_ROPE_BLOCK_H in widths, f"policy width not covered: {widths}"
    assert len(widths) >= 2, f"nothing to compare: {widths}"

    baseline_kv = None
    baseline_q = None
    for block_h in widths:
        num_tiles = -(-num_heads // block_h)
        q_run = q_pe.clone()
        out = torch.empty_like(kv)
        fused_rmsnorm_rope_with_q_native_kernel[(bsz, 1 + num_tiles)](
            kv, out, q_run, cos, sin, positions, weight,
            kv.stride(0), bsz, num_heads, kv_lora_rank, rope_dim,
            EPS,
            BLOCK_SIZE=_mirror_block(kv_lora_rank),
            BLOCK_H=block_h,
            BLOCK_PAIRS=block_pairs,
            num_warps=4,
        )
        torch.cuda.synchronize()
        if baseline_kv is None:
            baseline_kv, baseline_q = out, q_run
        else:
            _assert_bitwise(out, q_run, baseline_kv, baseline_q,
                            f"BLOCK_H={block_h} vs the narrowest BLOCK_H")


def test_bucket_width_does_not_change_a_row(production):
    """A 64-row result must survive being computed inside a 512-row bucket."""
    big = 512
    small = 64
    kv, q_pe, cos, sin, positions, weight = _inputs(big)

    q_big = q_pe.clone()
    kv_big = production(
        kv, q_big, cos, sin, positions, weight,
        GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS,
    )

    kv_small_in = kv[:small].clone()
    q_small = q_pe[:small].clone()
    kv_small = production(
        kv_small_in, q_small, cos, sin, positions[:small].contiguous(), weight,
        GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS,
    )
    torch.cuda.synchronize()

    _assert_bitwise(
        kv_small, q_small, kv_big[:small], q_big[:small],
        f"bucket {small} vs the same rows inside bucket {big}",
    )


# --------------------------------------------------------------------------- #
#  E  Row isolation (padded rows inert)
# --------------------------------------------------------------------------- #
def test_padded_rows_cannot_perturb_live_rows(production):
    """Rewriting the tail rows must leave the head rows bit-identical.

    The kernel has no valid-token count: a capture bucket's padded rows are
    computed from whatever is in the buffer.  That is only safe if rows do
    not interact, which is what this pins.
    """
    bsz = 128
    live = 40
    kv, q_pe, cos, sin, positions, weight = _inputs(bsz)

    q_first = q_pe.clone()
    kv_first = production(
        kv.clone(), q_first, cos, sin, positions, weight,
        GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(7)
    kv_perturbed = kv.clone()
    kv_perturbed[live:] = (
        torch.randn(kv[live:].shape, device="cuda", generator=generator) * 17.0
    ).to(torch.bfloat16)
    q_perturbed = q_pe.clone()
    q_perturbed[live:] = (
        torch.randn(q_pe[live:].shape, device="cuda", generator=generator) * 17.0
    ).to(torch.bfloat16)
    positions_perturbed = positions.clone()
    positions_perturbed[live:] = 0

    kv_second = production(
        kv_perturbed, q_perturbed, cos, sin, positions_perturbed, weight,
        GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS,
    )
    torch.cuda.synchronize()

    _assert_bitwise(
        kv_second[:live], q_perturbed[:live],
        kv_first[:live], q_first[:live],
        f"rows 0..{live - 1} after rewriting rows {live}..{bsz - 1}",
    )


# --------------------------------------------------------------------------- #
#  F  CUDA-graph capture / replay
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bsz", [64, 512])
def test_cuda_graph_replay_matches_eager(production, bsz):
    kv, q_pe, cos, sin, positions, weight = _inputs(bsz)

    q_eager = q_pe.clone()
    kv_eager = production(
        kv, q_eager, cos, sin, positions, weight,
        GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS,
    )
    torch.cuda.synchronize()
    kv_eager = kv_eager.clone()

    kv_static = kv.clone()
    q_static = q_pe.clone()
    pos_static = positions.clone()

    # Warm up on a side stream, then capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            q_static.copy_(q_pe)
            production(
                kv_static, q_static, cos, sin, pos_static, weight,
                GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS,
            )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    q_static.copy_(q_pe)
    with torch.cuda.graph(graph):
        out_static = production(
            kv_static, q_static, cos, sin, pos_static, weight,
            GLM52_KV_LORA_RANK, GLM52_ROPE_DIM, eps=EPS,
        )

    for replay in range(3):
        q_static.copy_(q_pe)
        kv_static.copy_(kv)
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(
            out_static, q_static, kv_eager, q_eager,
            f"graph replay {replay} at bsz={bsz}",
        )
