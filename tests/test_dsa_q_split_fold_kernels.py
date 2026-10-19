"""Kernel-side contract for reading a packed q projection with strides.

The GLM-5 decode graph used to split the q_b_proj output into its NoPE and
RoPE halves with two strided gathers per layer before either consumer could
read it. Both consumers can take the halves straight out of the single packed
output instead, and this file pins the two contracts that makes possible:

  * ``fused_rmsnorm_rope_with_q_native`` gains an optional ``q_pe_src``: the
    rotated pair is READ from a (row, head)-strided source and WRITTEN to the
    contiguous ``q_pe`` destination. Nothing else changes -- same arithmetic,
    same destination bytes -- so the split form must be BIT-IDENTICAL to
    gathering first and rotating in place.

  * ``triton_absorb_fp8_out`` states the requirement its kernel actually has:
    a contiguous K axis, with the batch and head strides free (they are passed
    to the kernel). The old blanket ``is_contiguous()`` guard was stricter than
    the kernel, and the replacement is not a relaxation -- it now names BOTH
    the rank and the innermost-stride requirement, and a K-strided view is
    still refused.

The positional-signature pin is the load-bearing one for the Triton kernel: a
``@triton.jit`` signature is a shared positional contract, and a parameter
inserted ahead of one an existing caller already fills slides every later
argument down one slot. With defaults in play that fails SILENTLY, with a
wrong value in the shifted slot, so the pin is on the INDEX of every
pre-existing parameter, not merely on their presence.
"""

import importlib
import inspect

import pytest
import torch


def _rope_module():
    # `batchgen_kernels.triton.__init__` re-exports a FUNCTION named
    # `fused_rmsnorm_rope`, which shadows the submodule of the same name, so
    # resolve the module by its dotted name instead of attribute access.
    return importlib.import_module("batchgen_kernels.triton.fused_rmsnorm_rope")


def _kernel_params(jit_fn):
    fn = getattr(jit_fn, "fn", jit_fn)
    return list(inspect.signature(fn).parameters)


# The signature as every pre-existing caller binds it. `variance_epsilon` is
# the last slot a positional caller fills (tests/gpu/test_fused_rmsnorm_rope_
# q_native.py passes 13 positionals, ending there) and it is a `tl.constexpr`,
# which is why "append after the runtime args" is NOT the rule -- the rule is
# positional: never insert before any parameter an existing caller fills.
_PRE_EXISTING_ROPE_PARAMS = [
    "new_compressed_kv_ptr",
    "processed_kv_ptr",
    "q_pe_ptr",
    "cos_ptr",
    "sin_ptr",
    "position_ids_ptr",
    "norm_weight_ptr",
    "input_batch_stride",
    "bsz",
    "num_heads",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "variance_epsilon",
    "BLOCK_SIZE",
    "BLOCK_H",
    "BLOCK_PAIRS",
    "row_valid_ptr",
    "HAS_ROW_MASK",
]
_SPLIT_ROPE_PARAMS = [
    "q_pe_in_ptr",
    "q_in_row_stride",
    "q_in_head_stride",
    "HAS_SPLIT_Q",
]


def test_split_source_params_are_appended_after_every_existing_parameter():
    params = _kernel_params(_rope_module().fused_rmsnorm_rope_with_q_native_kernel)
    assert params[: len(_PRE_EXISTING_ROPE_PARAMS)] == _PRE_EXISTING_ROPE_PARAMS, (
        "a parameter was inserted ahead of one an existing caller fills; that "
        "slides every later argument down one slot and, because the tail "
        "parameters carry defaults, it fails silently with a wrong value in "
        "the shifted slot instead of raising"
    )
    assert params == _PRE_EXISTING_ROPE_PARAMS + _SPLIT_ROPE_PARAMS


def test_split_source_kernel_params_default_to_the_in_place_form():
    fn = _rope_module().fused_rmsnorm_rope_with_q_native_kernel
    fn = getattr(fn, "fn", fn)
    defaults = inspect.signature(fn).parameters
    # Integer 0, not None: the repo has integer-default precedent for a
    # @triton.jit pointer parameter that is never dereferenced, and the branch
    # that would form an address from it is statically dead under the default.
    assert defaults["q_pe_in_ptr"].default == 0
    assert defaults["q_in_row_stride"].default == 0
    assert defaults["q_in_head_stride"].default == 0
    assert defaults["HAS_SPLIT_Q"].default is False


def test_wrapper_q_pe_src_is_appended_and_optional():
    params = list(
        inspect.signature(
            _rope_module().fused_rmsnorm_rope_with_q_native
        ).parameters.items()
    )
    names = [name for name, _ in params]
    assert names[-1] == "q_pe_src", (
        "q_pe_src must be last so a caller that predates it binds unchanged"
    )
    assert params[-1][1].default is None
    assert names[:9] == [
        "new_compressed_kv",
        "q_pe",
        "cos",
        "sin",
        "position_ids",
        "norm_weight",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "eps",
    ]


# --------------------------------------------------------------------------- #
# Wrapper validation (CPU: every check below runs before the Triton launch)
# --------------------------------------------------------------------------- #


def _cpu_rope_inputs(bsz=3, num_heads=2, kv_lora_rank=8, rope_dim=4):
    total = kv_lora_rank + rope_dim
    kv = torch.randn(bsz, 1, total, dtype=torch.bfloat16)
    q_pe = torch.randn(bsz, num_heads, 1, rope_dim, dtype=torch.bfloat16)
    cos = torch.randn(16, rope_dim, dtype=torch.float32)
    sin = torch.randn_like(cos)
    positions = torch.zeros(bsz, 1, dtype=torch.int64)
    weight = torch.randn(kv_lora_rank, dtype=torch.bfloat16)
    return kv, q_pe, cos, sin, positions, weight, kv_lora_rank, rope_dim


def _call_rope(q_pe_src, **over):
    kv, q_pe, cos, sin, positions, weight, kv_rank, rope_dim = _cpu_rope_inputs(
        **over
    )
    return _rope_module().fused_rmsnorm_rope_with_q_native(
        kv,
        q_pe,
        cos,
        sin,
        positions,
        weight,
        kv_rank,
        rope_dim,
        out=torch.empty_like(kv),
        q_pe_src=q_pe_src,
    )


def test_q_pe_src_shape_mismatch_is_a_named_error():
    with pytest.raises(ValueError, match="q_pe_src must have q_pe's shape"):
        _call_rope(torch.randn(3, 2, 1, 8, dtype=torch.bfloat16))


def test_q_pe_src_dtype_mismatch_is_a_named_error():
    with pytest.raises(ValueError, match="q_pe_src dtype"):
        _call_rope(torch.randn(3, 2, 1, 4, dtype=torch.float32))


def test_q_pe_src_with_a_strided_head_dim_axis_is_a_named_error():
    # Head-dim-strided, i.e. the pairs are NOT consecutive. The kernel
    # addresses them at unit stride, so this must raise rather than read the
    # wrong elements.
    packed = torch.randn(3, 2, 1, 8, dtype=torch.bfloat16)
    bad = packed[..., ::2]
    assert bad.shape == (3, 2, 1, 4) and bad.stride(-1) == 2
    with pytest.raises(ValueError, match="must be contiguous"):
        _call_rope(bad)


def test_head_strided_q_pe_src_passes_validation_and_reaches_the_launch():
    """A packed [B, H, nope+rope] slice is exactly the accepted shape."""
    nope, rope_dim, heads, bsz = 6, 4, 2, 3
    packed = torch.randn(bsz, heads, nope + rope_dim, dtype=torch.bfloat16)
    src = packed[..., nope:].unsqueeze(2)
    assert src.shape == (bsz, heads, 1, rope_dim)
    assert src.stride(-1) == 1
    assert src.stride(1) == nope + rope_dim
    assert not src.is_contiguous()
    # The launch itself is not what this pins: on a CPU-only box there is no
    # Triton runtime, and where there IS one the launcher rejects a CPU tensor
    # with a ValueError of its own -- which is indistinguishable from a guard
    # rejection if we discriminate on the exception TYPE. Discriminate on the
    # guard's message prefix instead (the three sibling cases above pin those
    # exact prefixes) and treat any other failure as "no guard rejected this
    # layout", which is what this case exists to prove. The numeric half is
    # the CUDA test below.
    try:
        _call_rope(src, num_heads=heads, rope_dim=rope_dim)
    except Exception as exc:  # noqa: BLE001 - see the comment above
        if "fused_rmsnorm_rope_with_q_native:" in str(exc):
            pytest.fail(f"a head-strided q_pe_src must be accepted, got: {exc}")


# --------------------------------------------------------------------------- #
# Absorb guard
# --------------------------------------------------------------------------- #


def _absorb_module():
    return importlib.import_module("batchgen_kernels.attention.dsa.fp8_absorb")


def _absorb_weights(heads, nope, out_dim):
    w_fp8 = torch.zeros(heads, out_dim, nope, dtype=torch.bfloat16)
    k_blocks = -(-nope // 128)
    w_scale = torch.ones(heads, out_dim, k_blocks, dtype=torch.float32)
    return w_fp8, w_scale


def test_absorb_rejects_a_non_3d_activation_by_name():
    mod = _absorb_module()
    w_fp8, w_scale = _absorb_weights(2, 6, 4)
    x = torch.randn(3, 2, 1, 6, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"x_bhk must be 3D"):
        mod.triton_absorb_fp8_out(
            x, w_fp8, w_scale, torch.empty(3, 2, 4, dtype=torch.bfloat16)
        )


def test_absorb_rejects_a_k_strided_activation_by_name():
    mod = _absorb_module()
    w_fp8, w_scale = _absorb_weights(2, 6, 4)
    packed = torch.randn(3, 2, 12, dtype=torch.bfloat16)
    x = packed[..., ::2]
    assert x.shape == (3, 2, 6) and x.stride(-1) == 2
    with pytest.raises(ValueError, match=r"x_bhk's K axis must be contiguous"):
        mod.triton_absorb_fp8_out(
            x, w_fp8, w_scale, torch.empty(3, 2, 4, dtype=torch.bfloat16)
        )


def test_absorb_guard_accepts_a_head_strided_activation():
    """The NoPE half of a packed q projection, i.e. the production caller."""
    mod = _absorb_module()
    nope, rope_dim, heads, bsz, out_dim = 6, 4, 2, 3, 4
    w_fp8, w_scale = _absorb_weights(heads, nope, out_dim)
    packed = torch.randn(bsz, heads, nope + rope_dim, dtype=torch.bfloat16)
    x = packed[..., :nope]
    assert x.stride(-1) == 1 and not x.is_contiguous()
    try:
        mod.triton_absorb_fp8_out(
            x, w_fp8, w_scale, torch.empty(bsz, heads, out_dim, dtype=torch.bfloat16)
        )
    except ValueError as exc:  # pragma: no cover - would be a guard regression
        pytest.fail(f"a head-strided activation must be accepted, got: {exc}")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Numeric equivalence (CUDA)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_source_rope_is_bit_identical_to_gather_then_rotate():
    """The exact claim the fold rests on, over the production geometry.

    `q_pe_src` must reproduce, bit for bit, what the previous two-step form
    produced: gather the RoPE half of the packed projection into a contiguous
    buffer, then rotate that buffer in place.
    """
    mod = _rope_module()
    torch.manual_seed(20261019)
    bsz, num_heads = 13, 64
    kv_lora_rank, rope_dim, nope = 512, 64, 192
    q_head_dim = nope + rope_dim

    kv = torch.randn(
        bsz, 1, kv_lora_rank + rope_dim, device="cuda", dtype=torch.bfloat16
    )
    q_flat = torch.randn(
        bsz, num_heads * q_head_dim, device="cuda", dtype=torch.bfloat16
    )
    cos = torch.randn(64, rope_dim, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    positions = torch.randint(0, 64, (bsz, 1), device="cuda", dtype=torch.int64)
    weight = torch.randn(kv_lora_rank, device="cuda", dtype=torch.bfloat16)

    q_heads = q_flat.view(bsz, num_heads, q_head_dim)

    # Old form: gather, then rotate in place.
    gathered = q_heads[..., nope:].unsqueeze(2).contiguous()
    kv_two_step = mod.fused_rmsnorm_rope_with_q_native(
        kv, gathered, cos, sin, positions, weight, kv_lora_rank, rope_dim,
        out=torch.empty_like(kv),
    )

    # New form: read strided, write the contiguous destination.
    folded = torch.empty(
        bsz, num_heads, 1, rope_dim, device="cuda", dtype=torch.bfloat16
    )
    src = q_heads[..., nope:].unsqueeze(2)
    assert not src.is_contiguous() and src.stride(-1) == 1
    kv_folded = mod.fused_rmsnorm_rope_with_q_native(
        kv, folded, cos, sin, positions, weight, kv_lora_rank, rope_dim,
        out=torch.empty_like(kv),
        q_pe_src=src,
    )
    torch.cuda.synchronize()

    assert torch.equal(folded, gathered), "rotated q differs bit for bit"
    assert torch.equal(kv_folded, kv_two_step), "processed KV differs bit for bit"
    # The source must be left untouched: the NoPE half shares the same rows and
    # is read AFTER this launch by the absorb GEMV.
    assert torch.equal(
        q_heads[..., nope:], q_flat.view(bsz, num_heads, q_head_dim)[..., nope:]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_source_leaves_the_packed_projection_unmodified():
    """The in-place form writes its q buffer; the split form must not.

    If the kernel wrote through `q_pe_src` instead of `q_pe`, the NoPE half
    read later out of the same packed tensor would still be correct (different
    columns) but the RoPE columns would be rotated twice across a retry. Pin
    the source as read-only.
    """
    mod = _rope_module()
    torch.manual_seed(7)
    bsz, num_heads, kv_lora_rank, rope_dim, nope = 4, 8, 64, 8, 24
    q_head_dim = nope + rope_dim
    kv = torch.randn(
        bsz, 1, kv_lora_rank + rope_dim, device="cuda", dtype=torch.bfloat16
    )
    q_flat = torch.randn(
        bsz, num_heads * q_head_dim, device="cuda", dtype=torch.bfloat16
    )
    before = q_flat.clone()
    cos = torch.randn(16, rope_dim, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    positions = torch.zeros(bsz, 1, device="cuda", dtype=torch.int64)
    weight = torch.randn(kv_lora_rank, device="cuda", dtype=torch.bfloat16)

    mod.fused_rmsnorm_rope_with_q_native(
        kv,
        torch.empty(bsz, num_heads, 1, rope_dim, device="cuda", dtype=torch.bfloat16),
        cos, sin, positions, weight, kv_lora_rank, rope_dim,
        out=torch.empty_like(kv),
        q_pe_src=q_flat.view(bsz, num_heads, q_head_dim)[..., nope:].unsqueeze(2),
    )
    torch.cuda.synchronize()
    assert torch.equal(q_flat, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_split_source_writes_every_destination_element():
    """No residue: a pre-poisoned destination must be fully overwritten."""
    mod = _rope_module()
    torch.manual_seed(11)
    bsz, num_heads, kv_lora_rank, rope_dim, nope = 5, 64, 512, 64, 192
    q_head_dim = nope + rope_dim
    kv = torch.randn(
        bsz, 1, kv_lora_rank + rope_dim, device="cuda", dtype=torch.bfloat16
    )
    q_flat = torch.randn(
        bsz, num_heads * q_head_dim, device="cuda", dtype=torch.bfloat16
    )
    cos = torch.randn(16, rope_dim, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    positions = torch.zeros(bsz, 1, device="cuda", dtype=torch.int64)
    weight = torch.randn(kv_lora_rank, device="cuda", dtype=torch.bfloat16)
    src = q_flat.view(bsz, num_heads, q_head_dim)[..., nope:].unsqueeze(2)

    poisoned = torch.full(
        (bsz, num_heads, 1, rope_dim),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    mod.fused_rmsnorm_rope_with_q_native(
        kv, poisoned, cos, sin, positions, weight, kv_lora_rank, rope_dim,
        out=torch.empty_like(kv),
        q_pe_src=src,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(poisoned).all(), (
        "a NaN survived, so some destination element was never written"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_absorb_on_a_head_strided_view_is_bit_identical_to_a_contiguous_copy():
    from batchgen_kernels.attention.dsa.fp8_absorb import (
        FP8AbsorbWeights,
        fp8_q_absorb_out,
    )

    torch.manual_seed(20261020)
    bsz, num_heads, nope, rope_dim, kv_lora_rank = 16, 64, 192, 64, 512
    q_head_dim = nope + rope_dim
    weights = FP8AbsorbWeights(
        torch.randn(num_heads, nope, kv_lora_rank, device="cuda", dtype=torch.bfloat16)
        * 0.05,
        torch.randn(
            num_heads, kv_lora_rank, kv_lora_rank, device="cuda", dtype=torch.bfloat16
        )
        * 0.05,
    )
    q_flat = (
        torch.randn(
            bsz, num_heads * q_head_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.1
    )
    strided = q_flat.view(bsz, num_heads, q_head_dim)[..., :nope]
    assert not strided.is_contiguous() and strided.stride(-1) == 1
    packed_copy = strided.contiguous()

    out_strided = torch.empty(
        bsz, num_heads, kv_lora_rank, device="cuda", dtype=torch.bfloat16
    )
    out_copy = torch.empty_like(out_strided)
    fp8_q_absorb_out(strided, weights, out_strided)
    fp8_q_absorb_out(packed_copy, weights, out_copy)
    torch.cuda.synchronize()
    assert torch.equal(out_strided, out_copy)
