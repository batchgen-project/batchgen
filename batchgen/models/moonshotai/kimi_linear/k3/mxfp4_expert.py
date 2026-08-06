# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-K3 MXFP4 routed expert — the module the engine's tensors actually fit.

WHY THIS MODULE EXISTS
----------------------
The K3 checkpoint ships NO ``.weight`` for any routed expert: it ships
``w{1,2,3}.weight_packed`` (uint8 E2M1 nibbles) + ``w{1,2,3}.weight_scale``
(uint8 E8M0), and the parameter server serves exactly those six names per
expert (``k3/tensor_map.py:428-436``).  ``KimiBlockSparseMLP`` declares three
``nn.Linear`` ``.weight`` parameters instead, and
``apply_weights`` (``models/wrappers/base.py:167-170``) skips a served name
that is not a module parameter — with no ``else``.  The BF16 expert module is
therefore not "unquantized"; it is SILENTLY EMPTY: every routed expert would
compute on ``torch.empty(0)``.

``K3MXFP4Expert`` declares the six served names verbatim, so the name sets are
equal and nothing can be skipped.  ``KimiK3MXFP4ExpertWrapper`` then validates
the ring slot before every use and hard-fails on any drift.

WHY MARLIN AND NOT THE WGMMA MXFP4 KERNELS
------------------------------------------
``batchgen_kernels.moe._C_expert_mxfp4_wgmma`` imports, but its epilogue is
gpt-oss's OpenAI SwiGLU (``gate * sigmoid(1.702*gate) * (up+1)`` with +-7
clipping, ``fused_wgmma_expert.py:15-16``) — a different activation from K3's
SiTU, hardcoded in the kernel, so it cannot produce K3 numerics at all.  Its
numerics are also ungated and its host wrapper still carries debug ``print``
statements.  The Triton grouped MXFP4 surface is a tombstone
(``mxfp4_grouped_gemm.py:1-24``: 12 confirmed defects, zero callers).

The marlin path is the one with a SiTU epilogue compiled in
(``marlin_grouped_gemm.cu:195-206``, beta 4 / linear_beta 25 — asserted against
the config below), an E2M1 in-kernel decode, hard-fail host contracts, and a
green 9/9 GPU parity ladder (``tests/moe/gpu_parity_mxfp4_marlin.py``).  This
also matches the 2026-08-04 decision ledger recorded in
``marlin_grouped_moe.py:253-261``.

THE ONE COST, STATED PLAINLY
----------------------------
The marlin kernels read weights in marlin TILE order; the engine streams them
in CHECKPOINT order, because ``kimi_parameter_server.py:242-243`` converts with
``marlin=False`` (and ``ckpt_converter._apply_marlin_repack`` explicitly
REFUSES uint8/E8M0 scales — it is a uniform-INT4 path).  So this module repacks
per forward, on device.  That is a pure static rearrangement of bytes that does
not depend on the activations, so it does not belong here: the end state is the
converter emitting marlin layout (byte-identical in size — ``[K//16, N*2]``
int32 == ``[N, K//2]`` uint8, ``[K//32, N]`` uint8 == ``[N, K//32]`` uint8), at
which point ``_marlin_from_slot`` collapses to two ``.view()``s.  NAMED
FOLLOW-UP; it is a converter change (``batchgen/ckpt_converter/``), which is
outside a model PR's allowlist.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..wrappers import KimiLinearExpertWrapper
from .mxfp4_layout import (
    MXFP4_DTYPE,
    MXFP4_GROUP_SIZE,
    MXFP4_PACK_FACTOR,
    K3QuantContractError,
    routed_expert_module_shapes,
    validate_quantization_config,
    validate_routed_expert_slot,
)

#: ``marlin_grouped_gemm.cu:196-206`` COMPILES these two constants into the
#: SiTU epilogue.  A config that disagrees would run K3's own activation
#: parameters nowhere — the kernel would silently keep using 4/25.
KERNEL_SITU_BETA = 4.0
KERNEL_SITU_LINEAR_BETA = 25.0

#: Marlin's weight permutation acts on 1024-nibble blocks and its scale
#: permutation on 64-column blocks.
_MARLIN_TILE = 16


def is_mxfp4_quantized(config) -> bool:
    """True iff ``config`` declares the MXFP4 routed-expert format we implement.

    False ONLY for a config with no quantization declaration at all — the
    Kimi-Linear-48B, whose routed experts really are BF16.  Every OTHER
    declaration raises out of :func:`validate_quantization_config` instead of
    quietly selecting the BF16 expert module, which would then find no
    ``.weight`` in the checkpoint and compute on empty tensors.
    """
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None:
        return False
    validate_quantization_config(quantization_config)
    return True


# --------------------------------------------------------------------------- #
#  Checkpoint layout -> marlin tile layout, on device                          #
# --------------------------------------------------------------------------- #

_PERM_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def _marlin_perms(device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Device-resident (weight_perm, scale_perm), built once per device.

    Imported lazily from ``marlin_weight_prep`` so this module — and therefore
    ``model.py`` — stays importable on a host with no compiled kernels; the
    permutations themselves are pure numpy/torch.
    """
    key = str(device)
    cached = _PERM_CACHE.get(key)
    if cached is None:
        from batchgen.moe.marlin_weight_prep import _get_scale_perms, get_weight_perm

        weight_perm = get_weight_perm(4).to(device=device, dtype=torch.long)
        scale_perm, _ = _get_scale_perms()
        cached = (
            weight_perm,
            torch.tensor(scale_perm, device=device, dtype=torch.long),
        )
        _PERM_CACHE[key] = cached
    return cached


def repack_mxfp4_to_marlin_device(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    K: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Device-resident twin of ``repack_mxfp4_to_marlin_gs32(..., "bf16")``.

    Step for step the same rearrangement as the CPU function that the frozen
    oracle gates (``marlin_weight_prep.py:384-440``) — unpack low-nibble-first,
    transpose to [K, N], marlin tile permute, repack 8 nibbles per int32,
    transpose + index-permute the E8M0 bytes, expand them EXACTLY to bf16.  It
    exists only because the CPU one round-trips through ``.cpu().numpy()``,
    which would sync and stall the copy engine on every streamed expert.

    ``tests/gpu/verify_k3_mxfp4_expert.py --gpu`` asserts bit-identity against
    the CPU function at both K3 shapes; nothing here may drift from it
    independently.

    Args:
        weight_packed: [N, K//2] uint8, low nibble = even K index.
        weight_scale: [N, K//32] uint8 E8M0.
    Returns:
        (marlin_qw [K//16, N*2] int32, marlin_s [K//32, N] bf16)
    """
    from batchgen.moe.marlin_weight_prep import mxfp4_scale_e8m0_to_bf16

    weight_perm, scale_perm = _marlin_perms(weight_packed.device)

    # 1. unpack nibbles -> [N, K] raw E2M1 codes (NOT decoded).
    codes = torch.stack(
        (weight_packed & 0x0F, weight_packed >> 4), dim=-1
    ).reshape(N, K)

    # 2-3. transpose to [K, N], marlin tile permute.
    q = codes.t().contiguous()
    q = q.reshape(K // _MARLIN_TILE, _MARLIN_TILE, N // _MARLIN_TILE, _MARLIN_TILE)
    q = q.permute(0, 2, 1, 3).reshape(K // _MARLIN_TILE, N * _MARLIN_TILE)
    q = q.reshape(-1, weight_perm.numel())[:, weight_perm]

    # 4. pack. Two nibbles per byte (low first) then a bitwise int32 view is
    #    identical to the CPU path's `|= q[:, i::8] << 4*i`: int32 word w is
    #    bytes 4w..4w+3 little-endian, so its nibble i is source column 8w+i.
    q = q.contiguous().view(K // _MARLIN_TILE, N * 8, 2)
    marlin_qw = (q[..., 0] | (q[..., 1] << 4)).contiguous().view(torch.int32)

    # 5. scales: [N, K//32] -> [K//32, N], index-permute, EXACT bf16 expansion
    #    (a bit shift; mxfp4_scale_e8m0_to_bf16 also rejects the 0x00/0xFF
    #    edge bytes rather than clamping them).
    s = weight_scale.t().contiguous()
    s = s.reshape(-1, scale_perm.numel())[:, scale_perm].reshape(-1, N).contiguous()
    return marlin_qw, mxfp4_scale_e8m0_to_bf16(s)


# --------------------------------------------------------------------------- #
#  The module                                                                  #
# --------------------------------------------------------------------------- #

class K3MXFP4Projection(nn.Module):
    """One MXFP4 projection: EXACTLY the two tensors the engine serves.

    The parameter names are the checkpoint suffixes, so
    ``module.named_parameters()`` and the served ``tensor_key`` set are equal
    and ``apply_weights`` cannot skip either member of the couple.
    """

    def __init__(self, n_out: int, k_in: int):
        super().__init__()
        if int(k_in) % MXFP4_GROUP_SIZE != 0:
            raise K3QuantContractError(
                f"K={k_in} is not a multiple of group_size={MXFP4_GROUP_SIZE}."
            )
        self.n_out = int(n_out)
        self.k_in = int(k_in)
        self.weight_packed = nn.Parameter(
            torch.empty(self.n_out, self.k_in // MXFP4_PACK_FACTOR,
                        dtype=MXFP4_DTYPE),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(self.n_out, self.k_in // MXFP4_GROUP_SIZE,
                        dtype=MXFP4_DTYPE),
            requires_grad=False,
        )

    def marlin(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return repack_mxfp4_to_marlin_device(
            self.weight_packed.data, self.weight_scale.data,
            self.k_in, self.n_out,
        )

    def extra_repr(self) -> str:
        return f"in={self.k_in}, out={self.n_out}, mxfp4_gs{MXFP4_GROUP_SIZE}"


class K3MXFP4Expert(nn.Module):
    """K3 routed expert, MXFP4-packed end to end.

    Shape-for-shape the replacement of ``KimiBlockSparseMLP`` for a quantized
    config: same ``w1``(gate) / ``w2``(down) / ``w3``(up) names in the same
    latent space, no ``act_fn`` submodule because SiTU is fused into the S1
    kernel's epilogue.
    """

    def __init__(self, config, hidden_size: int, intermediate_size: int):
        super().__init__()
        beta = getattr(config, "activation_situ_beta", None)
        linear_beta = getattr(config, "activation_situ_linear_beta", None)
        if beta != KERNEL_SITU_BETA or linear_beta != KERNEL_SITU_LINEAR_BETA:
            raise K3QuantContractError(
                f"activation_situ_beta/linear_beta = {beta}/{linear_beta}, but "
                f"the fused MXFP4 S1 kernel COMPILES IN "
                f"{KERNEL_SITU_BETA}/{KERNEL_SITU_LINEAR_BETA} "
                f"(marlin_grouped_gemm.cu:196-206). The kernel would ignore the "
                f"config values silently; change the kernel, not this check."
            )
        self.hidden_size = int(hidden_size)          # latent, 3584
        self.intermediate_size = int(intermediate_size)  # 3072
        self.w1 = K3MXFP4Projection(self.intermediate_size, self.hidden_size)
        self.w2 = K3MXFP4Projection(self.hidden_size, self.intermediate_size)
        self.w3 = K3MXFP4Projection(self.intermediate_size, self.hidden_size)

    def expected_slot_shapes(self) -> Dict[str, list]:
        """The ring-slot geometry this module accepts — same declaration the
        parameter server sizes the slot from, so drift on either side shows up
        as a mismatch rather than an overrun."""
        return routed_expert_module_shapes(self.intermediate_size,
                                           self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 0-token experts are DRIVEN, not skipped (moe_forward_serving keeps the
        # copy engine in lockstep by forwarding every expert). Return early:
        # the marlin grid is undefined at M=0 and the repack would be pure waste.
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)

        from batchgen.moe.marlin_grouped_moe import (
            single_expert_marlin_mxfp4_decode,
        )

        gate_qw, gate_s = self.w1.marlin()
        up_qw, up_s = self.w3.marlin()
        down_qw, down_s = self.w2.marlin()
        # gate=w1 BEFORE up=w3: the HF reference is cat([w1(x), w3(x)]) and a
        # swap is numerically silent (pinned by the GPU mutation test m2).
        return single_expert_marlin_mxfp4_decode(
            hidden_states,
            gate_qw, gate_s,
            up_qw, up_s,
            down_qw, down_s,
            N=self.intermediate_size,
            K=self.hidden_size,
        )


# --------------------------------------------------------------------------- #
#  The wrapper                                                                 #
# --------------------------------------------------------------------------- #

class KimiK3MXFP4ExpertWrapper(KimiLinearExpertWrapper):
    """Streamed MXFP4 routed expert.

    Identical lifecycle to the BF16 wrapper (load -> apply -> compute -> free);
    the two differences are both hard-fails, because every failure mode here is
    of the silent-wrong-weights class:

      * ``dequantize_weights`` VALIDATES the slot instead of waving it through
        (it still does not dequantize — K3 stays packed into the K-loop);
      * ``apply_weights`` asserts that every served tensor landed on a
        parameter, closing ``models/wrappers/base.py:167-170``'s missing
        ``else`` for K3 without touching the shared base class.
    """

    def __init__(self, module, layer_idx, expert_idx, core_engine,
                 engine_config, model_config, persistent: bool = False):
        if persistent:
            raise K3QuantContractError(
                "KimiK3MXFP4ExpertWrapper requires persistent=False. A "
                "persistent expert never calls load_weights, so its packed "
                "parameters would stay at the empty(0) the PSM leaves them at "
                "and the kernel would read a null slot."
            )
        super().__init__(module, layer_idx, expert_idx, core_engine,
                         engine_config, model_config, persistent=False)

    def dequantize_weights(self, weights_dict):
        validate_routed_expert_slot(
            self.module_key, weights_dict, self.module.expected_slot_shapes()
        )
        return weights_dict

    def apply_weights(self, weights_dict):
        super().apply_weights(weights_dict)
        applied = self._applied_param_keys
        if applied != set(weights_dict):
            raise K3QuantContractError(
                f"{self.module_key}: apply_weights matched {sorted(applied)} "
                f"but the slot served {sorted(weights_dict)}. Unmatched names "
                f"are DROPPED by the base class, leaving the expert on "
                f"empty(0) — refusing rather than computing on nothing."
            )
