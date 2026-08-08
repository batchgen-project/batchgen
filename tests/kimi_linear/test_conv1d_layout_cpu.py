"""CPU check: varlen ``causal_conv1d_fwd`` output layout (``overwrite_x``).

Runs without a GPU and without the compiled extension — a stub stands in for
the CUDA op, so only the wrapper's staging/transpose bookkeeping is exercised.

What it pins:
  1. ``overwrite_x=True`` returns the caller's own storage, token-major and
     CONTIGUOUS.
  2. Its values are bit-identical to the default (strided-view) path, and the
     conv-state pool writes are identical too. The flag is a layout change
     only.
  3. After ``rearrange("l (h d) -> 1 l h d")`` the new path ``is_contiguous()``
     and ``.contiguous()`` returns the *identical object* — i.e. fla's
     ``@input_guard`` no-ops instead of allocating a third full copy of q/k/v.
     The default path is non-contiguous and does allocate.

Run: python -m batchgen_kernels.tests.kimi_linear.test_conv1d_layout_cpu
"""

import logging
import sys

import torch

_LOG = logging.getLogger("batchgen_kernels.tests.kimi_linear.conv1d_layout_cpu")
from einops import rearrange

import batchgen_kernels.conv1d as conv1d_mod
from batchgen_kernels.conv1d import causal_conv1d_fwd

DTYPE = torch.bfloat16
DIM = 24
W = 4
HEADS = 3  # DIM = HEADS * 8


class _StubExt:
    """Reference varlen causal conv1d over channel-major (dim, total) input.

    Mirrors the CUDA op's contract: writes the output in place into ``x_cm``
    and the last W-1 raw inputs of each sequence into ``conv_states``.
    """

    @staticmethod
    def causal_conv1d_fwd(x_cm, weight, bias, conv_states, query_start_loc,
                          cache_indices, has_initial_state, silu_activation,
                          pad_slot_id):
        assert x_cm.stride(-1) == 1, "kernel requires x.stride(-1) == 1"
        dim = x_cm.shape[0]
        cu = query_start_loc.tolist()
        out = torch.empty_like(x_cm)
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            seq = x_cm[:, s:e].float()                       # (dim, T)
            init = torch.zeros(dim, W - 1, dtype=torch.float32)
            if has_initial_state is not None and bool(has_initial_state[i]):
                init = conv_states[cache_indices[i]].float()
            padded = torch.cat([init, seq], dim=1)           # (dim, W-1+T)
            acc = torch.zeros(dim, e - s, dtype=torch.float32)
            for w in range(W):
                acc += padded[:, w:w + (e - s)] * weight[:, w].float()[:, None]
            if bias is not None:
                acc += bias.float()[:, None]
            if silu_activation:
                acc = acc * torch.sigmoid(acc)
            out[:, s:e] = acc.to(x_cm.dtype)
            if conv_states is not None:
                conv_states[cache_indices[i]] = padded[:, -(W - 1):].to(
                    conv_states.dtype)
        x_cm.copy_(out)


def _run(x, weight, bias, cu, slots, pool, overwrite_x):
    return causal_conv1d_fwd(
        x, weight, bias=bias, conv_states=pool, query_start_loc=cu,
        cache_indices=slots, has_initial_state=None, overwrite_x=overwrite_x,
    )


def main():
    conv1d_mod._ext = _StubExt()  # bypass the CUDA extension load

    g = torch.Generator().manual_seed(7)
    lens = [37, 5, 1, 64]
    total = sum(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()],
                      dtype=torch.int32)
    slots = torch.tensor([3, 0, 2, 1], dtype=torch.int32)

    x0 = torch.randn(total, DIM, generator=g).mul_(0.5).to(DTYPE)
    weight = torch.randn(DIM, W, generator=g).mul_(0.3).to(DTYPE)
    bias = torch.randn(DIM, generator=g).mul_(0.1).to(DTYPE)

    fails = []

    def check(name, ok, detail=""):
        _LOG.info(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    # --- default path (strided view over the staging buffer) ---
    x_a = x0.clone()
    pool_a = torch.zeros(4, DIM, W - 1, dtype=DTYPE)
    y_a = _run(x_a, weight, bias, cu, slots, pool_a, overwrite_x=False)

    # --- overwrite_x path ---
    x_b = x0.clone()
    pool_b = torch.zeros(4, DIM, W - 1, dtype=DTYPE)
    y_b = _run(x_b, weight, bias, cu, slots, pool_b, overwrite_x=True)

    check("default path returns a NON-contiguous (total, dim) view",
          y_a.shape == (total, DIM) and not y_a.is_contiguous(),
          f"strides={tuple(y_a.stride())}")
    check("overwrite_x returns the caller's own storage",
          y_b.data_ptr() == x_b.data_ptr() and y_b is x_b)
    check("overwrite_x result is token-major CONTIGUOUS",
          y_b.shape == (total, DIM) and y_b.is_contiguous(),
          f"strides={tuple(y_b.stride())}")
    check("values BIT-IDENTICAL across the two paths",
          torch.equal(y_a, y_b))
    check("conv-state pool writes bit-identical",
          torch.equal(pool_a, pool_b))
    check("default path leaves the input untouched",
          torch.equal(x_a, x0))
    check("overwrite_x consumes the input (x now holds the output)",
          torch.equal(x_b, y_a))

    # --- what fla's @input_guard sees: arg.contiguous() on the rearranged q ---
    qa = rearrange(y_a, "l (h d) -> 1 l h d", h=HEADS)
    qb = rearrange(y_b, "l (h d) -> 1 l h d", h=HEADS)
    check("rearrange is a view on BOTH paths (no hidden copy there)",
          qa.data_ptr() == y_a.data_ptr() and qb.data_ptr() == y_b.data_ptr())
    check("OLD: input_guard's .contiguous() ALLOCATES a full copy",
          not qa.is_contiguous() and qa.contiguous().data_ptr() != qa.data_ptr())
    check("NEW: input_guard's .contiguous() is a no-op (returns self)",
          qa.shape == qb.shape and qb.is_contiguous() and qb.contiguous() is qb)
    check("no-op .contiguous() preserves values exactly",
          torch.equal(qa.contiguous(), qb.contiguous()))

    # --- what the SEGMENTED KDA sweep sees (kimi_linear fix 5) ---
    # This is where the saving is actually collected once chunk_kda runs in
    # token segments: fla copies whatever slice it is handed, so the win is
    # per-SEGMENT, not per-sequence. 1.125 GiB/layer at K3's T=16,384, not the
    # 9.00 GiB the unsegmented sweep would have saved.
    seg = slice(8, 40)
    seg_a, seg_b = qa[:, seg], qb[:, seg]
    check("SEGMENT slice of the OLD path is non-contiguous (fla copies it)",
          not seg_a.is_contiguous())
    check("SEGMENT slice of the NEW path IS contiguous (fla copies nothing)",
          seg_b.is_contiguous() and seg_b.contiguous() is seg_b)
    check("segment values identical across the two paths",
          torch.equal(seg_a.contiguous(), seg_b))

    # --- the contiguity precondition, and that it fires BEFORE the kernel ---
    # The kernel mutates conv_states in place, so a late assert would leave the
    # pool half-updated. x here is a last-dim slice of a fused qkv buffer.
    x_bad = torch.randn(total, 2 * DIM, generator=g).to(DTYPE)[:, :DIM]
    pool_c = torch.zeros(4, DIM, W - 1, dtype=DTYPE)
    pool_c0 = pool_c.clone()
    raised = False
    try:
        _run(x_bad, weight, bias, cu, slots, pool_c, overwrite_x=True)
    except AssertionError:
        raised = True
    check("overwrite_x REFUSES non-contiguous x", raised)
    check("...and refuses before the kernel touches conv_states",
          torch.equal(pool_c, pool_c0))

    _LOG.info("")
    if fails:
        _LOG.info(f"{len(fails)} FAILED: {fails}")
        return 1
    _LOG.info("all checks passed")
    return 0


def test_conv1d_layout_cpu():
    """pytest entry point, so the check actually gates in a suite run as well
    as under ``python -m``."""
    assert main() == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
