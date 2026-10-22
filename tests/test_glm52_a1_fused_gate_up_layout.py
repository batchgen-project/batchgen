"""A1: the fused [E, 2I, H] gate/up layout and its enable_deepep condition.

The DeepEP decode path runs gate+up as ONE masked m-grouped GEMM, which needs
the two projections fused per expert. The CuTe grouped kernels cannot read a
half of that block -- they derive each expert's weight address from a hardcoded
``N * K`` group stride (``fp8_blockwise_gemm.cu:60``, and the scales at ``:67``)
-- so the layout is chosen ONCE at load time and only when ``enable_deepep`` is
set. With it off, the per-projection stacks and every code path that reads them
are unchanged.

These are the tests that keep the condition from being lost:

* the layout follows the flag, both ways;
* the fused halves are individually contiguous [N, K] views with the addresses
  the pointer-form CuTe kernels and the masked GEMM both consume;
* the per-projection attributes are ``None`` under the fused layout, so a
  contiguous-stack reader fails loudly instead of striding a half-sized block;
* the masked bind refuses a layout that disagrees with the configuration.

CPU-only: this is a statement about layout, addresses and branch selection, not
about GEMM values. The numerical gate for the fused chain lives in
``test_glm52_deepgemm_masked_parity_gpu.py``.
"""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5.moe_deepgemm_masked import (
    DeepGemmMaskedExpertWeights,
    SCALE_BLOCK,
)

# H is chosen so that K // 128 is itself a multiple of 4: the CuTe scale stacks
# pad that trailing block axis to 4, while the masked GEMM needs the EXACT
# extent, so the fused layout is only legal where the pad is the identity.
# H = 512 -> k_blocks = 4 = pad4. GLM-5.2's real K = 6144 -> 48, likewise.
# A geometry where they disagree must be REFUSED, which
# ``test_the_fused_layout_refuses_a_geometry_whose_scale_axis_is_padded``
# pins directly -- H = 256 (k_blocks 2 -> pad4 4) is that case.
E, H, I = 2, 512, 128
H_PADDED_SCALE_AXIS = 256
FP8 = torch.float8_e4m3fn


class _FakeWrapper:
    def __init__(self):
        self.module = SimpleNamespace()
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None


def _stub_moe(deepep: bool, prealloc=None, h=None, i=None):
    """The minimum ``Glm5MoE`` surface ``_init_fp8_blockwise_weights`` touches."""
    h = H if h is None else h
    i = I if i is None else i
    nb_n, nb_k = i // SCALE_BLOCK, h // SCALE_BLOCK
    moe = SimpleNamespace(
        experts_per_rank=E,
        hidden_size=h,
        config=SimpleNamespace(moe_intermediate_size=i),
        device=torch.device("cpu"),
        layer_idx=3,
        routed_expert_start_idx=0,
        glm5_deepep_ll=deepep,
        experts=[_FakeWrapper() for _ in range(E)],
        gate_list=[torch.zeros(i, h, dtype=torch.uint8) for _ in range(E)],
        up_list=[torch.ones(i, h, dtype=torch.uint8) for _ in range(E)],
        down_list=[torch.zeros(h, i, dtype=torch.uint8) for _ in range(E)],
        gate_scale_list=[torch.full((nb_n, nb_k), 2.0) for _ in range(E)],
        up_scale_list=[torch.full((nb_n, nb_k), 3.0) for _ in range(E)],
        down_scale_list=[torch.full((nb_k, nb_n), 4.0) for _ in range(E)],
    )
    if prealloc is not None:
        moe._w3d_prealloc = prealloc
    return moe


def _stack(moe):
    """Run the real loader against the stub.

    ``batchgen.models.glm.glm5.model`` imports Triton at module scope, so the
    loader tests skip where Triton is absent (they run on the GPU hosts and in
    CI). The tests below that do NOT need the loader -- the masked bind's
    refusal and the weight contract -- import nothing heavy and always run.
    """
    pytest.importorskip(
        "triton", reason="glm5.model imports triton at module scope")
    from batchgen.models.glm.glm5.model import Glm5MoE

    Glm5MoE._init_fp8_blockwise_weights(moe)
    return moe


# --------------------------------------------------------------------------
# The layout follows the flag
# --------------------------------------------------------------------------
def test_deepep_off_keeps_the_per_projection_layout():
    """The non-DeepEP configuration must be byte-for-byte what it was: two
    separate [E, I, H] stacks, no fused block, and the CuTe contiguous kernel
    still readable (group stride N * K == I * H)."""
    moe = _stack(_stub_moe(deepep=False))

    assert moe.fp8_gate_up_fused is False
    assert moe.fp8_gate_up_w3d is None and moe.fp8_gate_up_ws3d is None
    assert tuple(moe.fp8_gate_w3d.shape) == (E, I, H)
    assert tuple(moe.fp8_up_w3d.shape) == (E, I, H)
    assert moe.fp8_gate_w3d.is_contiguous() and moe.fp8_up_w3d.is_contiguous()
    # the stride the contiguous CuTe kernel assumes, per expert
    assert moe.fp8_gate_w3d.stride(0) == I * H
    assert moe.fp8_up_w3d.stride(0) == I * H
    # and no pointer arrays are built, because nothing needs them
    assert moe.fused_gate_w_ptrs is None and moe.fused_up_w_ptrs is None


def test_deepep_on_fuses_gate_and_up_into_one_block():
    moe = _stack(_stub_moe(deepep=True))

    assert moe.fp8_gate_up_fused is True
    assert tuple(moe.fp8_gate_up_w3d.shape) == (E, 2 * I, H)
    assert moe.fp8_gate_up_w3d.is_contiguous()
    assert tuple(moe.fp8_gate_up_ws3d.shape) == (
        E, (2 * I) // SCALE_BLOCK, H // SCALE_BLOCK)
    # the per-projection stacks are gone, on purpose: a contiguous-stack reader
    # must fail loudly rather than read this block at half the group stride
    assert moe.fp8_gate_w3d is None and moe.fp8_up_w3d is None
    assert moe.fp8_gate_ws3d is None and moe.fp8_up_ws3d is None


def test_the_fused_layout_refuses_a_geometry_whose_scale_axis_is_padded():
    """The fused layout is only legal where ``pad4`` is the identity.

    The CuTe stacks pad a weight-scale stack's trailing block axis up to 4
    floats; the masked m-grouped GEMM reads the EXACT ``K // 128`` extent and
    cannot see past it (``DeepGemmMaskedExpertWeights.validate`` says so). So on
    a geometry where ``K // 128`` is not already a multiple of 4, fusing would
    hand the masked GEMM a padded scale stack it misreads -- and the loader must
    refuse at load time with the padding named, not produce it.

    This is pinned on its own because it is the one branch that fires only on
    geometries the production model never uses: GLM-5.2's K = 6144 gives 48,
    which divides by 4, so nothing else in this file reaches it.
    """
    k_blocks = H_PADDED_SCALE_AXIS // SCALE_BLOCK
    assert (k_blocks + 3) // 4 * 4 != k_blocks, (
        "this test needs a geometry whose scale axis really is padded")
    moe = _stub_moe(deepep=True, h=H_PADDED_SCALE_AXIS)
    with pytest.raises(RuntimeError, match=r"exact K//128 trailing extent"):
        _stack(moe)

    # ... and the same geometry is fine with the fused layout off, because the
    # CuTe kernels are the ones the padding was for.
    ok = _stack(_stub_moe(deepep=False, h=H_PADDED_SCALE_AXIS))
    assert ok.fp8_gate_up_w3d is None
    assert tuple(ok.fp8_gate_ws3d.shape) == (
        E, I // SCALE_BLOCK, (k_blocks + 3) // 4 * 4)


def test_the_fused_halves_carry_the_values_of_their_own_projection():
    """Gate in rows [0, I), up in [I, 2I) -- not interleaved, not swapped."""
    moe = _stack(_stub_moe(deepep=True))
    block = moe.fp8_gate_up_w3d
    for e in range(E):
        assert torch.all(block[e, :I, :] == 0), e      # gate_list was zeros
        assert torch.all(block[e, I:, :] == 1), e      # up_list was ones
    scales = moe.fp8_gate_up_ws3d
    nb = I // SCALE_BLOCK
    for e in range(E):
        assert torch.all(scales[e, :nb, :] == 2.0), e
        assert torch.all(scales[e, nb:, :] == 3.0), e


def test_the_fused_halves_are_contiguous_views_at_the_pointer_addresses():
    """What makes the pointer-form CuTe kernel immune to the hardcoded stride:
    each half is its own contiguous [I, H] view, and the pointer arrays hold
    exactly those addresses."""
    moe = _stack(_stub_moe(deepep=True))
    block = moe.fp8_gate_up_w3d
    gate_ptrs = moe.fused_gate_w_ptrs.tolist()
    up_ptrs = moe.fused_up_w_ptrs.tolist()
    assert len(gate_ptrs) == len(up_ptrs) == E
    for e in range(E):
        gate, up = block[e, :I, :], block[e, I:, :]
        assert gate.is_contiguous() and up.is_contiguous()
        assert tuple(gate.shape) == tuple(up.shape) == (I, H)
        assert gate_ptrs[e] == gate.data_ptr()
        assert up_ptrs[e] == up.data_ptr()
        # the two halves of one expert are exactly I * H elements apart
        assert up.data_ptr() - gate.data_ptr() == I * H * block.element_size()
    # the resident expert rows alias the fused block, so the weight pointer
    # array rebuilt from them agrees
    assert moe.gate_ptrs_ptr.tolist() == gate_ptrs
    assert moe.up_ptrs_ptr.tolist() == up_ptrs


def test_the_fused_scale_pointers_address_the_fused_scale_stack():
    """The scale pointers must come from the fused stack, never from
    ``gate_scale_list``: those tensors are the pre-stack sources and are NOT
    the bytes the GEMM reads."""
    moe = _stack(_stub_moe(deepep=True))
    scales = moe.fp8_gate_up_ws3d
    nb = I // SCALE_BLOCK
    for e in range(E):
        assert moe.fused_gate_ws_ptrs[e].item() == scales[e, :nb, :].data_ptr()
        assert moe.fused_up_ws_ptrs[e].item() == scales[e, nb:, :].data_ptr()
    sources = {s.data_ptr() for s in moe.gate_scale_list}
    assert not sources & set(moe.fused_gate_ws_ptrs.tolist())


# --------------------------------------------------------------------------
# The preallocated (persistent-decode) path
# --------------------------------------------------------------------------
def test_the_prealloc_path_adopts_a_fused_two_tuple():
    gate_up = torch.empty(E, 2 * I, H, dtype=torch.uint8)
    down = torch.empty(E, H, I, dtype=torch.uint8)
    moe = _stub_moe(deepep=True, prealloc=(gate_up, down))
    # the PSM points each expert at its row of the carve
    moe.gate_list = [gate_up[e, :I, :] for e in range(E)]
    moe.up_list = [gate_up[e, I:, :] for e in range(E)]
    moe.down_list = [down[e] for e in range(E)]
    _stack(moe)

    assert moe.fp8_gate_up_w3d is gate_up and moe.fp8_down_w3d is down
    assert moe.fp8_gate_w3d is None and moe.fp8_up_w3d is None


def test_the_prealloc_path_rejects_the_wrong_tuple_arity():
    """A 3-tuple carve under the fused layout is a PSM/loader disagreement and
    must be named, not silently unpacked."""
    stacks = (
        torch.empty(E, I, H, dtype=torch.uint8),
        torch.empty(E, I, H, dtype=torch.uint8),
        torch.empty(E, H, I, dtype=torch.uint8),
    )
    moe = _stub_moe(deepep=True, prealloc=stacks)
    with pytest.raises(RuntimeError, match="needs a .gate_up, down."):
        _stack(moe)


def test_the_prealloc_path_rejects_rows_that_do_not_alias_the_carve():
    gate_up = torch.empty(E, 2 * I, H, dtype=torch.uint8)
    down = torch.empty(E, H, I, dtype=torch.uint8)
    moe = _stub_moe(deepep=True, prealloc=(gate_up, down))
    moe.gate_list = [gate_up[e, :I, :] for e in range(E)]
    moe.up_list = [torch.empty(I, H, dtype=torch.uint8) for _ in range(E)]
    moe.down_list = [down[e] for e in range(E)]
    with pytest.raises(RuntimeError, match="does not alias its preallocated"):
        _stack(moe)


# --------------------------------------------------------------------------
# The masked bind refuses a layout that disagrees with the configuration
# --------------------------------------------------------------------------
def test_the_masked_bind_refuses_the_per_projection_layout(monkeypatch):
    """The DeepEP segment's bind is the seam where a configuration/layout
    mismatch would otherwise become a half-sized read at the wrong stride.

    The bind resolves the masked kernel before it checks the layout, so the
    resolver is stubbed here: this test is about the layout gate, not about
    whether deep_gemm is installed.
    """
    from batchgen.models.glm.glm5 import moe_cuda_graph_segments as seg

    monkeypatch.setattr(seg, "masked_gemm_fn", lambda: (lambda *a, **k: None))

    segment = SimpleNamespace(
        moe=SimpleNamespace(
            layer_idx=7,
            fp8_gate_up_fused=False,
            fp8_gate_up_w3d=None,
            fp8_gate_up_ws3d=None,
        ),
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
    )
    with pytest.raises(RuntimeError, match="per-projection layout"):
        seg.Glm5MoEGraphSegment._bind_masked_gemm_weights(segment)


def test_the_fused_weight_contract_names_a_half_sized_stack():
    """``validate`` is the whole contract, so a per-projection stack handed to
    the fused field must be named by shape, not accepted."""
    halved = DeepGemmMaskedExpertWeights(
        gate_up_w=torch.zeros(E, I, H, dtype=torch.uint8).view(FP8),
        gate_up_scale=torch.ones(
            E, (2 * I) // SCALE_BLOCK, H // SCALE_BLOCK),
        down_w=torch.zeros(E, H, I, dtype=torch.uint8).view(FP8),
        down_scale=torch.ones(E, H // SCALE_BLOCK, I // SCALE_BLOCK),
    )
    with pytest.raises(RuntimeError, match=r"gate_up_w.*\[E, N, K\]"):
        halved.validate(num_experts=E, hidden_size=H, intermediate_size=I)


# --------------------------------------------------------------------------
# Item 5: the recv_count histogram diagnostic
#
# recv_count is a device tensor's VALUES, and a chrome/torch profile carries
# only launch metadata (grid, block, registers, occupancy, bytes) -- never
# tensor contents. So the per-expert row distribution, and with it the rate at
# which an expert needs more than one 64-row m-tile, is not recoverable from
# any existing trace. This batch-level diagnostic is the only route to it.
# --------------------------------------------------------------------------
from batchgen.models.glm.glm5.moe_ragged import (  # noqa: E402
    accumulate_recv_count_histogram,
    format_recv_count_histogram,
    recv_count_histogram_enabled,
    recv_count_histogram_width,
)

BLOCK_M = 64


def test_the_histogram_width_covers_every_reachable_tile_count():
    # stride = num_ranks * tokens/rank bounds an expert's rows, so the tile
    # count cannot exceed ceil(stride / block_m)
    assert recv_count_histogram_width(512, 64) == 2 + 8 + 1
    assert recv_count_histogram_width(64, 64) == 2 + 1 + 1
    # a partial last tile still needs its own slot
    assert recv_count_histogram_width(65, 64) == 2 + 2 + 1
    for bad in ((0, 64), (512, 0), (-1, 64)):
        with pytest.raises(ValueError, match="positive stride/block_m"):
            recv_count_histogram_width(*bad)


def test_the_histogram_counts_tiles_rows_and_layer_steps():
    hist = torch.zeros(recv_count_histogram_width(512, BLOCK_M),
                       dtype=torch.int64)
    # 0 rows -> 0 tiles, 1..64 -> 1 tile, 65 -> 2 tiles, 200 -> 4 tiles
    counts = torch.tensor([0, 1, 64, 65, 200], dtype=torch.int32)
    accumulate_recv_count_histogram(hist, counts, BLOCK_M)

    assert hist[0].item() == 1                     # one layer-step
    assert hist[1].item() == 0 + 1 + 64 + 65 + 200  # routed rows
    assert hist[2 + 0].item() == 1                 # the empty expert
    assert hist[2 + 1].item() == 2                 # 1 row and 64 rows
    assert hist[2 + 2].item() == 1                 # 65 rows crosses one tile
    assert hist[2 + 4].item() == 1                 # 200 rows -> 4 tiles


def test_the_histogram_accumulates_across_steps():
    hist = torch.zeros(recv_count_histogram_width(512, BLOCK_M),
                       dtype=torch.int64)
    counts = torch.tensor([16, 16], dtype=torch.int32)
    for _ in range(5):
        accumulate_recv_count_histogram(hist, counts, BLOCK_M)
    assert hist[0].item() == 5
    assert hist[1].item() == 5 * 32
    assert hist[2 + 1].item() == 10
    # cumulative, which is what makes the readback a distribution rather than
    # one step's snapshot: nothing in the accumulate zeroes it
    assert hist[2 + 2].item() == 0


def test_the_histogram_clamps_rather_than_writing_out_of_bounds():
    """An expert cannot exceed the stride, but the accumulator must not be the
    thing that corrupts memory if routing ever did."""
    hist = torch.zeros(recv_count_histogram_width(64, BLOCK_M),
                       dtype=torch.int64)
    accumulate_recv_count_histogram(
        hist, torch.tensor([10_000], dtype=torch.int32), BLOCK_M)
    assert hist.sum().item() > 0
    assert hist[-1].item() == 1


def test_the_histogram_refuses_the_wrong_dtype_and_rank():
    hist32 = torch.zeros(8, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="must be int64"):
        accumulate_recv_count_histogram(
            hist32, torch.tensor([1], dtype=torch.int32), BLOCK_M)
    hist = torch.zeros(8, dtype=torch.int64)
    with pytest.raises(RuntimeError, match=r"per-expert \[E\] row counts"):
        accumulate_recv_count_histogram(
            hist, torch.zeros(2, 2, dtype=torch.int32), BLOCK_M)


def test_the_histogram_report_states_the_over_one_tile_rate():
    """The rate that decides whether capping to one tile is even viable."""
    hist = torch.zeros(recv_count_histogram_width(512, BLOCK_M),
                       dtype=torch.int64)
    # 19 experts inside one tile, 1 over it: a 5% over-one-tile rate
    counts = torch.tensor([16] * 19 + [100], dtype=torch.int32)
    accumulate_recv_count_histogram(hist, counts, BLOCK_M)
    line = format_recv_count_histogram(hist, BLOCK_M)
    assert "expert-slots=20" in line
    assert "over_one_tile=1(5.000%)" in line
    assert "block_m=64" in line
    assert "2t=1" in line and "1t=19" in line


def test_the_empty_histogram_reports_no_accumulation():
    hist = torch.zeros(recv_count_histogram_width(512, BLOCK_M),
                       dtype=torch.int64)
    assert "no accumulation yet" in format_recv_count_histogram(hist, BLOCK_M)


def test_the_diagnostic_is_off_unless_the_batch_flag_sets_it(monkeypatch):
    """Off by default and off for a malformed flag: a diagnostic that defaulted
    on would add a launch per routed layer to the measured path."""
    from batchgen.models.wrappers import attention as attn

    monkeypatch.setattr(attn.AttnWrapperBase, "batchgen_debug", None,
                        raising=False)
    assert recv_count_histogram_enabled() is False
    monkeypatch.setattr(attn.AttnWrapperBase, "batchgen_debug", {},
                        raising=False)
    assert recv_count_histogram_enabled() is False
    monkeypatch.setattr(attn.AttnWrapperBase, "batchgen_debug",
                        "not-a-dict", raising=False)
    assert recv_count_histogram_enabled() is False
    monkeypatch.setattr(attn.AttnWrapperBase, "batchgen_debug",
                        {"glm52_recv_count_hist": True}, raising=False)
    assert recv_count_histogram_enabled() is True


def test_the_histogram_readback_refuses_to_run_inside_a_capture(monkeypatch):
    """``hist.tolist()`` is a D2H copy: issuing it during a CUDA graph capture
    fails the capture, and only ever with the diagnostic ON -- the one
    configuration that uses it. The split is therefore part of the contract:
    accumulate inside the capture, read back only outside it.
    """
    hist = torch.zeros(recv_count_histogram_width(512, BLOCK_M),
                       dtype=torch.int64)
    accumulate_recv_count_histogram(
        hist, torch.tensor([16, 16], dtype=torch.int32), BLOCK_M)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="must not be called inside a CUDA"):
        format_recv_count_histogram(hist, BLOCK_M)

    # and it works again the moment the capture is over
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert "expert-slots=2" in format_recv_count_histogram(hist, BLOCK_M)


def test_the_accumulate_stays_usable_inside_a_capture(monkeypatch):
    """The other half of the same contract: the device-side accumulate must
    NOT grow a capture guard, or the diagnostic could never collect anything."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    hist = torch.zeros(recv_count_histogram_width(512, BLOCK_M),
                       dtype=torch.int64)
    accumulate_recv_count_histogram(
        hist, torch.tensor([16, 65], dtype=torch.int32), BLOCK_M)
    assert hist[0].item() == 1
    assert hist[2 + 1].item() == 1 and hist[2 + 2].item() == 1


def test_the_captured_forward_does_not_read_the_histogram_back():
    """Structural: the segment's captured forward may accumulate but must never
    format. A regression here is a capture failure on a GPU host, which is an
    expensive way to learn it, so it is pinned statically here.
    """
    import ast
    import inspect

    from batchgen.models.glm.glm5 import moe_cuda_graph_segments as seg

    src = inspect.getsource(seg.Glm5MoEGraphSegment._forward_deepep)
    tree = ast.parse(textwrap.dedent(src))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "accumulate_recv_count_histogram" in called
    assert "format_recv_count_histogram" not in called

    # and the readback lives on the pre-capture host path
    setup_src = inspect.getsource(
        seg.Glm5MoEGraphSegment.setup_static_buffers)
    assert "format_recv_count_histogram" in setup_src
