"""CPU suite for the K3 MXFP4 → Marlin repack (task #34).

Proves, on CPU with no compiled kernels:
  1. The frozen compressed-tensors oracle vector pins the dequant convention
     (and catches the known mutant conventions).
  2. repack_mxfp4_to_marlin_gs32 is PURE REARRANGEMENT: marlin → inverse is
     BYTE-IDENTICAL to the source (weights and E8M0 scale bytes), at the real
     K3 shapes including the w1/w3 branch shape.
  3. E8M0 → bf16 scale expansion is exact over the whole legal window and
     hard-fails on the forbidden edge bytes.
  4. Every contract check (R1–R7) raises.
  5. Deliberately broken variants (wrong nibble order, transposed/wrong perm,
     off-by-one scale group) are CAUGHT by these tests.
  6. The w1‖w3 commutation theorem: marlin(w1 ‖_N w3) == hcat of per-branch
     marlin tensors; the fused repack is gate-first storage adjacency.

What CPU cannot prove (GPU-staged, see gpu_parity_mxfp4_marlin.py): the
in-kernel E2M1 decode (incl. the bf16-subnormal 0.5 path), the SiTU epilogue,
and GEMM parity under the project tolerance gate.
"""

import hashlib
import os
from pathlib import Path

import pytest
import torch

from tests.moe._loader import load_moe_modules

oracle, mwp = load_moe_modules()

# (K, N) per projection; K3 real shapes: w1/w3 branch and w2
K3_W13 = (3584, 3072)
K3_W2 = (3072, 3584)
SMALL = (128, 256)


def _rand_mxfp4(K, N, seed, scale_lo=112, scale_hi=134):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (N, K // 2), generator=g, dtype=torch.int16).to(torch.uint8)
    scale = torch.randint(scale_lo, scale_hi + 1, (N, K // 32), generator=g,
                          dtype=torch.int16).to(torch.uint8)
    return packed, scale


# ---------------------------------------------------------------------------
# 1. Frozen oracle vector
# ---------------------------------------------------------------------------

def test_frozen_vector_pins_oracle_dequant():
    oracle.check_dequant_fn(oracle.mxfp4_dequantize_oracle)


def test_format_id_pinned():
    """The R9-R11 follow-up (ckpt_converter stamps this id; model-side L6
    matches on it) is a cross-PR contract — pin the string so a silent edit
    breaks loudly instead of desynchronizing converter and loader."""
    assert oracle.MXFP4_MARLIN_FORMAT_ID == "mxfp4_marlin_gs32_v1"


def test_frozen_vector_pins_repack_unpack_path():
    # R8's exact self-check: the dequant built on the repack module's own
    # nibble unpack must match the frozen real-checkpoint vector.
    oracle.check_dequant_fn(mwp._mxfp4_dequant_via_unpack)


def test_frozen_vector_catches_mutant_dequants():
    lut = torch.tensor(list(oracle.MXFP4_E2M1_LUT)
                       + [-v for v in oracle.MXFP4_E2M1_LUT], dtype=torch.float32)

    def _dequant(packed, scales, *, swap=False, bias=oracle.MXFP4_E8M0_BIAS,
                 int4_style=False):
        lo = (packed & 0x0F).to(torch.long)
        hi = (packed >> 4).to(torch.long)
        if swap:
            lo, hi = hi, lo
        out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.float32)
        if int4_style:
            out[:, 0::2] = lo.float() - 8.0
            out[:, 1::2] = hi.float() - 8.0
        else:
            out[:, 0::2] = lut[lo]
            out[:, 1::2] = lut[hi]
        exp = (scales.to(torch.int32) - bias).repeat_interleave(32, dim=-1)
        return torch.ldexp(out, exp).to(torch.bfloat16)

    mutants = {
        "swapped nibbles": lambda p, s: _dequant(p, s, swap=True),
        "bias off-by-one (126)": lambda p, s: _dequant(p, s, bias=126),
        "INT4 (q-8)*scale instead of E2M1 LUT": lambda p, s: _dequant(p, s, int4_style=True),
    }
    caught = 0
    for name, fn in mutants.items():
        try:
            oracle.check_dequant_fn(fn)
        except AssertionError:
            caught += 1
        else:
            pytest.fail(f"mutant dequant NOT caught by frozen vector: {name}")
    assert caught == 3


# ---------------------------------------------------------------------------
# 2. Round-trip byte identity (pure-rearrangement proof)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K,N", [K3_W13, K3_W2, SMALL])
@pytest.mark.parametrize("emit_scale", ["e8m0", "bf16"])
def test_repack_roundtrip_byte_identical(K, N, emit_scale):
    packed, scale = _rand_mxfp4(K, N, seed=K + N)
    qw, s = mwp.repack_mxfp4_to_marlin_gs32(packed, scale, K, N, emit_scale=emit_scale)
    assert qw.dtype == torch.int32 and tuple(qw.shape) == (K // 16, N * 2)
    expected_s_dtype = torch.uint8 if emit_scale == "e8m0" else torch.bfloat16
    assert s.dtype == expected_s_dtype and tuple(s.shape) == (K // 32, N)
    # byte count of the nibble payload is unchanged
    assert qw.numel() * 4 == packed.numel()

    rp, rs = mwp.marlin_mxfp4_to_raw_cpu(qw, s, K, N)
    assert torch.equal(rp, packed), "marlin round-trip weights not BYTE-IDENTICAL"
    assert torch.equal(rs, scale), "marlin round-trip E8M0 scale bytes not BYTE-IDENTICAL"


@pytest.mark.parametrize("K,N", [SMALL, K3_W13])
def test_dequant_of_repacked_matches_oracle(K, N):
    packed, scale = _rand_mxfp4(K, N, seed=7)
    qw, s = mwp.repack_mxfp4_to_marlin_gs32(packed, scale, K, N, emit_scale="e8m0")
    rp, rs = mwp.marlin_mxfp4_to_raw_cpu(qw, s, K, N)
    a = oracle.mxfp4_dequantize_oracle(rp, rs)
    b = oracle.mxfp4_dequantize_oracle(packed, scale)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), \
        "dequant of repacked-then-unpacked diverges from oracle dequant of source"


# ---------------------------------------------------------------------------
# 3. Exact E8M0 -> bf16 expansion
# ---------------------------------------------------------------------------

def test_scale_e8m0_to_bf16_exact_full_window():
    e8 = torch.arange(1, 255, dtype=torch.int16).to(torch.uint8)
    got = mwp.mxfp4_scale_e8m0_to_bf16(e8).float()
    ref = torch.ldexp(torch.ones(254), e8.to(torch.int32) - 127)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), \
        "E8M0->bf16 expansion is not exact over [1, 254]"


@pytest.mark.parametrize("edge", [0x00, 0xFF])
def test_scale_e8m0_edge_bytes_raise(edge):
    s = torch.full((4,), edge, dtype=torch.uint8)
    with pytest.raises(ValueError, match="edge byte"):
        mwp.mxfp4_scale_e8m0_to_bf16(s)


def test_bf16_scale_roundtrip_lossless():
    _, scale = _rand_mxfp4(*SMALL, seed=11, scale_lo=1, scale_hi=254)
    K, N = SMALL
    packed, _ = _rand_mxfp4(K, N, seed=12)
    qw, s_bf16 = mwp.repack_mxfp4_to_marlin_gs32(packed, scale, K, N, emit_scale="bf16")
    _, rs = mwp.marlin_mxfp4_to_raw_cpu(qw, s_bf16, K, N)
    assert torch.equal(rs, scale)


# ---------------------------------------------------------------------------
# 4. Contract checks R1-R7 (hard-fail negatives)
# ---------------------------------------------------------------------------

def test_contract_checks_raise():
    K, N = SMALL
    packed, scale = _rand_mxfp4(K, N, seed=3)

    # R1: wrong packed dtype
    with pytest.raises(ValueError, match="must be uint8"):
        mwp.repack_mxfp4_to_marlin_gs32(packed.to(torch.int32), scale, K, N)
    # R2: bf16 scales = INT4 checkpoint
    with pytest.raises(ValueError, match="INT4 checkpoint"):
        mwp.repack_mxfp4_to_marlin_gs32(packed, scale.to(torch.bfloat16), K, N)
    # R3: packed dim != K//2
    with pytest.raises(ValueError, match="K//2"):
        mwp.repack_mxfp4_to_marlin_gs32(packed[:, :-1], scale, K, N)
    # R4: scale groups != K/32
    with pytest.raises(ValueError, match="K/32"):
        mwp.repack_mxfp4_to_marlin_gs32(packed, scale[:, :-1], K, N)
    # R5: N mismatch packed vs scale
    with pytest.raises(ValueError, match="N mismatch"):
        mwp.repack_mxfp4_to_marlin_gs32(packed, scale[:-1], K, N)
    # R6: N % 64 != 0
    p32, s32 = _rand_mxfp4(K, 32, seed=4)
    with pytest.raises(ValueError, match="N%64"):
        mwp.repack_mxfp4_to_marlin_gs32(p32, s32, K, 32)
    # K-divisibility leg: K % 32 != 0 is caught by R4 (which subsumes R6's
    # K%16 clause — that clause is defense in depth and unreachable while R4
    # runs first). K=48: packed [N, 24] passes R3, then R4 raises.
    p48 = torch.randint(0, 256, (N, 24), dtype=torch.int16).to(torch.uint8)
    s48 = torch.full((N, 1), 120, dtype=torch.uint8)
    with pytest.raises(ValueError, match="K/32"):
        mwp.repack_mxfp4_to_marlin_gs32(p48, s48, 48, N)
    # R7: forbidden E8M0 edge bytes
    for edge in (0x00, 0xFF):
        bad = scale.clone()
        bad[0, 0] = edge
        with pytest.raises(ValueError, match="edge byte"):
            mwp.repack_mxfp4_to_marlin_gs32(packed, bad, K, N)
    # emit_scale validation
    with pytest.raises(ValueError, match="emit_scale"):
        mwp.repack_mxfp4_to_marlin_gs32(packed, scale, K, N, emit_scale="fp16")


# ---------------------------------------------------------------------------
# 5. Mutation arms — each deliberately broken variant must be CAUGHT
# ---------------------------------------------------------------------------

def test_mutation_swapped_nibble_order_caught():
    """A hi-nibble-first unpack must be caught (a) by the frozen vector and
    (b) by byte-identity against the source."""
    K, N = SMALL
    packed, scale = _rand_mxfp4(K, N, seed=21)

    swapped = ((packed & 0x0F) << 4) | (packed >> 4)  # mutant source unpack order

    # (a) dequant-level: swapping changes values on real-vector data
    def mutant_dequant(p, s):
        return oracle.mxfp4_dequantize_oracle(((p & 0x0F) << 4) | (p >> 4), s)
    with pytest.raises(AssertionError):
        oracle.check_dequant_fn(mutant_dequant)

    # (b) byte-level: repacking the swapped bytes cannot round-trip to the
    # original source
    qw, s = mwp.repack_mxfp4_to_marlin_gs32(swapped, scale, K, N)
    rp, _ = mwp.marlin_mxfp4_to_raw_cpu(qw, s, K, N)
    assert not torch.equal(rp, packed), \
        "swapped-nibble mutation NOT caught by byte-identity"


def test_mutation_wrong_perm_caught():
    """Packing with a wrong (inverse-instead-of-forward) tile permutation must
    break the byte-identity round trip."""
    K, N = SMALL
    packed, scale = _rand_mxfp4(K, N, seed=22)
    q_w_nk = mwp._unpack_mxfp4_nibbles(packed, K, N)
    q_w = q_w_nk.t().contiguous()

    wrong_perm = mwp._inverse_weight_perm(4)  # mutant: inverse used as forward
    qw_mut = mwp._marlin_pack_weights(q_w, K, N, wrong_perm)

    rp, _ = mwp.marlin_mxfp4_to_raw_cpu(
        qw_mut, mwp._marlin_permute_scales(scale.t().contiguous(), K, N, 32), K, N)
    assert not torch.equal(rp, packed), "wrong-perm mutation NOT caught"


def test_mutation_transposed_tiles_caught():
    """Skipping the [K,N] transpose (packing N-major) must break round-trip."""
    K, N = (256, 256)  # square so shapes still line up — hardest case
    packed, scale = _rand_mxfp4(K, N, seed=23)
    q_w_nk = mwp._unpack_mxfp4_nibbles(packed, K, N)

    qw_mut = mwp._marlin_pack_weights(q_w_nk.contiguous(), K, N,
                                      mwp.get_weight_perm(4))  # mutant: no .t()
    rp, _ = mwp.marlin_mxfp4_to_raw_cpu(
        qw_mut, mwp._marlin_permute_scales(scale.t().contiguous(), K, N, 32), K, N)
    assert not torch.equal(rp, packed), "transposed-tiles mutation NOT caught"


def test_mutation_off_by_one_scale_group_caught():
    K, N = SMALL
    packed, scale = _rand_mxfp4(K, N, seed=24, scale_lo=100, scale_hi=140)
    qw, s = mwp.repack_mxfp4_to_marlin_gs32(packed, scale, K, N)
    s_mut = torch.roll(s, shifts=1, dims=0)  # off-by-one K-group
    rp, rs = mwp.marlin_mxfp4_to_raw_cpu(qw, s_mut, K, N)
    a = oracle.mxfp4_dequantize_oracle(rp, rs)
    b = oracle.mxfp4_dequantize_oracle(packed, scale)
    assert not torch.equal(a.view(torch.int16), b.view(torch.int16)), \
        "off-by-one scale-group mutation NOT caught"


# ---------------------------------------------------------------------------
# 6. w1 ‖ w3 fusion
# ---------------------------------------------------------------------------

def test_w13_commutation_theorem():
    """marlin(w1 ‖_N w3) == hcat(marlin(w1), marlin(w3)) — the permutation
    acts within 64-N-column blocks, and 3072 % 64 == 0, so no permuted unit
    crosses the branch boundary. Verified here at a small N multiple of 64."""
    K, N = (128, 128)
    w1_p, w1_s = _rand_mxfp4(K, N, seed=31)
    w3_p, w3_s = _rand_mxfp4(K, N, seed=32)

    cat_p = torch.cat([w1_p, w3_p], dim=0)   # [2N, K//2] source-coordinate concat
    cat_s = torch.cat([w1_s, w3_s], dim=0)
    qw_cat, s_cat = mwp.repack_mxfp4_to_marlin_gs32(cat_p, cat_s, K, 2 * N)

    qw1, s1 = mwp.repack_mxfp4_to_marlin_gs32(w1_p, w1_s, K, N)
    qw3, s3 = mwp.repack_mxfp4_to_marlin_gs32(w3_p, w3_s, K, N)

    assert torch.equal(qw_cat, torch.cat([qw1, qw3], dim=1)), \
        "w13 commutation theorem violated (weights)"
    assert torch.equal(s_cat, torch.cat([s1, s3], dim=1)), \
        "w13 commutation theorem violated (scales)"


def test_w13_fused_repack_gate_first_adjacency():
    K, N = SMALL
    w1_p, w1_s = _rand_mxfp4(K, N, seed=41)
    w3_p, w3_s = _rand_mxfp4(K, N, seed=42)

    qw, s = mwp.repack_mxfp4_w13_to_marlin_gs32(w1_p, w1_s, w3_p, w3_s, K, N)
    qw1, s1 = mwp.repack_mxfp4_to_marlin_gs32(w1_p, w1_s, K, N)
    qw3, s3 = mwp.repack_mxfp4_to_marlin_gs32(w3_p, w3_s, K, N)

    assert tuple(qw.shape) == (2, K // 16, N * 2) and qw.is_contiguous()
    assert torch.equal(qw[0], qw1) and torch.equal(qw[1], qw3), \
        "fused repack is not gate(w1)-first / up(w3)-second"
    assert torch.equal(s[0], s1) and torch.equal(s[1], s3)

    # storage adjacency: up tensor starts exactly one branch after gate
    assert qw[1].data_ptr() == qw.data_ptr() + qw[0].numel() * qw.element_size()

    # gate/up swap is a DIFFERENT blob (silent at kernel level — pinned here
    # and by the GPU SiTU mutation test)
    qw_sw, _ = mwp.repack_mxfp4_w13_to_marlin_gs32(w3_p, w3_s, w1_p, w1_s, K, N)
    assert not torch.equal(qw_sw, qw)

    # branch shape mismatch raises
    with pytest.raises(ValueError, match="mismatch"):
        mwp.repack_mxfp4_w13_to_marlin_gs32(w1_p, w1_s, w3_p[:-1], w3_s[:-1], K, N)


# ---------------------------------------------------------------------------
# 7. Optional heavier oracles
# ---------------------------------------------------------------------------

def test_oracle_matches_compressed_tensors_package():
    """Independent cross-check against the installed compressed-tensors
    package (the checkpoint's own format implementation). Skipped when the
    package is not installed; the frozen vector pins the convention anyway."""
    pytest.importorskip("compressed_tensors")
    try:
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
        from compressed_tensors.compressors.mx_utils import decompress_mx_scale
    except ImportError:
        pytest.skip("compressed-tensors version lacks nvfp4/mx helpers")

    K, N = (256, 128)
    packed, scale = _rand_mxfp4(K, N, seed=51, scale_lo=100, scale_hi=140)
    vals = unpack_fp4_from_uint8(packed, N, K, dtype=torch.float32)
    ref = (vals.unflatten(-1, (K // 32, 32))
           * decompress_mx_scale(scale).to(torch.float32).unsqueeze(-1)
           ).flatten(-2)
    ours = oracle.mxfp4_dequantize_oracle(packed, scale, dtype=torch.float32)
    assert torch.equal(ours.view(torch.int32), ref.view(torch.int32)), \
        "mxfp4_dequantize_oracle diverges from compressed-tensors"


REAL_DIR = os.environ.get("MXFP4_W1_REAL_DIR", "")


@pytest.mark.skipif(
    not (REAL_DIR and Path(REAL_DIR, "w1_packed.bin").exists()),
    reason="set MXFP4_W1_REAL_DIR to a dir with w1_packed.bin/w1_scale.bin "
           "(real K3 layer-4 expert-0 w1 bytes) to run the real-tensor pin")
def test_real_w1_tensor_pins_and_roundtrip():
    N, K = 3072, 3584
    packed = torch.frombuffer(
        bytearray(Path(REAL_DIR, "w1_packed.bin").read_bytes()),
        dtype=torch.uint8).view(N, K // 2)
    scale = torch.frombuffer(
        bytearray(Path(REAL_DIR, "w1_scale.bin").read_bytes()),
        dtype=torch.uint8).view(N, K // 32)

    assert hashlib.sha256(packed.numpy().tobytes()).hexdigest() == \
        "cf822517403f5ccb418150e10b303568f617f3099bea6dcc9af3a7b6a48e3501"
    assert hashlib.sha256(scale.numpy().tobytes()).hexdigest() == \
        "b1cd3499f23097edbbc3f4dc3304c573dbdc49602404b90bdfb1b3dbb0b4ea92"

    deq = oracle.mxfp4_dequantize_oracle(packed, scale)
    assert hashlib.sha256(deq.view(torch.int16).numpy().tobytes()).hexdigest() == \
        oracle.VEC_FULL_W1_DEQUANT_BF16_SHA256

    qw, s = mwp.repack_mxfp4_to_marlin_gs32(packed, scale, K, N)
    rp, rs = mwp.marlin_mxfp4_to_raw_cpu(qw, s, K, N)
    assert torch.equal(rp, packed) and torch.equal(rs, scale)
