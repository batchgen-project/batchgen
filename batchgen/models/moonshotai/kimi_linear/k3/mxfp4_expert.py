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

CURRENT OFFLINE-MARLIN CONTRACT
-------------------------------
The converter now emits routed experts in Marlin tile order:

  * ``weight_packed`` metadata is int32 Marlin tiles;
  * ``weight_scale`` remains byte-neutral uint8 E8M0 in Marlin order.

Decode's host ``get_tensor`` exposes the true int32 shape. Prefill's fixed GPU
ring presents the same linear bytes through its historical uint8
``[N, K//2]`` slot shape; :meth:`K3MXFP4Projection.marlin` reinterprets that
view back to Marlin without permuting the packed weights.

There is no Marlin→WGMMA transform in K3 prefill. Both phases call the Marlin
MXFP4 kernels with the SiTU epilogue. The remaining per-forward format work is
the exact E8M0 uint8→BF16 scale expansion (a device tensor bit shift + view).
Storing BF16 scales would remove it but adds ~85.1 GB across 82,432 experts.
"""

import logging
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..wrappers import KimiLinearExpertWrapper
from .mxfp4_layout import (
    K3_ROUTED_EXPERTS_MARLIN,
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

    False ONLY for a config with no quantization declaration at all.  Every
    OTHER declaration raises out of :func:`validate_quantization_config` instead
    of quietly selecting the BF16 expert module, which would then find no
    ``.weight`` in the checkpoint and compute on empty tensors.

    On the "no declaration" branch a ``kimi_k3`` config is a fallback and is
    WARNED about, per the project's no-silent-fallback rule — but it is not
    refused here, for two measured reasons:

      * the real load path ALREADY refuses it, three times over:
        ``k3/tensor_map.py::validate_k3_config`` collects
        ``validate_quantization_config(cfg.quantization_config)`` (tensor_map
        :361) and is called by ``kimi_initializer`` :205, by
        ``build_k3_state_dict_name_map`` :392 and by ``load_k3_config`` :685.
        No real K3 serving build can reach this function with a ``None``;
      * an UNQUANTIZED K3 is a supported, tested shape.  The M2 eager ground
        truth (``kimi_k3/model.py``) builds BF16 experts for exactly these
        configs, and the synthetic K3-SYN-25 / K3-SKEW-10 harness configs
        (``tests/kimi_k3_harness.py``) declare ``model_type='kimi_k3'`` with no
        ``quantization_config`` on purpose, so that the serving MoE can be
        compared against it.  Raising here would delete that comparison.
    """
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is not None:
        validate_quantization_config(quantization_config)
        return True

    if str(getattr(config, "quantization", "none")).lower() == "mxfp4":
        raise K3QuantContractError(
            "config.quantization == 'mxfp4' but quantization_config is None. "
            "The label and the block must agree; the packed layout (group "
            "size, dtypes, ignore list) is read from the block, so there is "
            "nothing to build MXFP4 experts from."
        )

    if getattr(config, "model_type", None) == "kimi_k3":
        logging.warning(
            "Kimi-K3 config carries NO quantization_config: building BF16 "
            "routed experts (KimiBlockSparseMLP, w{1,2,3}.weight). The RELEASED "
            "checkpoint ships no .weight for any routed expert — only "
            "w{1,2,3}.weight_{packed,scale} — so against real weights this "
            "expert computes on empty(0). Legitimate only for a synthetic "
            "unquantized K3 (tests/kimi_k3_harness.py). A real load cannot "
            "reach here: validate_k3_config raises first."
        )
    return False


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
    scale_bf16: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Device-resident twin of ``repack_mxfp4_to_marlin_gs32(..., "bf16")``.

    Step for step the same rearrangement as the CPU function that the frozen
    oracle gates (``marlin_weight_prep.py:384-440``) — unpack low-nibble-first,
    transpose to [K, N], marlin tile permute, repack 8 nibbles per int32, then
    transpose + index-permute the E8M0 bytes. Optional BF16 expansion is kept
    for parity fixtures; production passes ``scale_bf16=False`` because the
    K3 Marlin kernel consumes E8M0 directly. It
    exists only because the CPU one round-trips through ``.cpu().numpy()``,
    which would sync and stall the copy engine on every streamed expert.

    ``tests/gpu/verify_k3_mxfp4_expert.py --gpu`` asserts bit-identity against
    the CPU function at both K3 shapes; nothing here may drift from it
    independently.

    Args:
        weight_packed: [N, K//2] uint8, low nibble = even K index.
        weight_scale: [N, K//32] uint8 E8M0.
    Returns:
        (marlin_qw [K//16, N*2] int32,
         marlin_s [K//32, N] uint8 when ``scale_bf16=False`` else bf16)
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

    # 5. scales: [N, K//32] -> [K//32, N], index-permute. The optional exact
    #    BF16 expansion is a bit shift; mxfp4_scale_e8m0_to_bf16 also rejects
    #    the 0x00/0xFF edge bytes rather than clamping them.
    s = weight_scale.t().contiguous()
    s = s.reshape(-1, scale_perm.numel())[:, scale_perm].reshape(-1, N).contiguous()
    if not scale_bf16:
        # Return Marlin-order uint8 E8M0 for the production kernel. Expanding
        # this tensor with mxfp4_scale_e8m0_to_bf16 remains bit-identical to
        # the legacy BF16 branch by construction (same s).
        return marlin_qw, s
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
        if K3_ROUTED_EXPERTS_MARLIN:
            # Offline marlin (task #53): the streamed slot already holds
            # marlin bytes, presented under the packed uint8 module_shape
            # [n_out, k_in//2]. Reinterpret both tensors to Marlin dims; the
            # kernel consumes E8M0 scale bytes directly. The slot bytes are
            # the exact linear bytes repack_mxfp4_to_marlin_device emits, so
            # the same-linear-order reshape recovers its marlin_qw/scale
            # bit-for-bit (validated: verify_k3_mxfp4_expert repack identity
            # + the gate below). marlin_qw [k_in//16, n_out*2] int32; marlin
            # scale [k_in//32, n_out] uint8 E8M0.
            qw = self.weight_packed.data.contiguous().view(
                torch.int32).reshape(self.k_in // _MARLIN_TILE, self.n_out * 2)
            s = self.weight_scale.data.contiguous().reshape(
                self.k_in // MXFP4_GROUP_SIZE, self.n_out)
            return qw, s
        return repack_mxfp4_to_marlin_device(
            self.weight_packed.data, self.weight_scale.data,
            self.k_in, self.n_out, scale_bf16=False,
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

    _prefill_profile_enabled = False
    _prefill_profile_calls = 0
    _prefill_profile_active_calls = 0
    _prefill_profile_token_rows = 0
    _prefill_profile_wall_s = 0.0

    @classmethod
    def reset_prefill_profile(cls, enabled: bool) -> None:
        cls._prefill_profile_enabled = bool(enabled)
        cls._prefill_profile_calls = 0
        cls._prefill_profile_active_calls = 0
        cls._prefill_profile_token_rows = 0
        cls._prefill_profile_wall_s = 0.0

    @classmethod
    def prefill_profile_snapshot(cls) -> dict:
        return {
            "enabled": cls._prefill_profile_enabled,
            "calls": cls._prefill_profile_calls,
            "active_calls": cls._prefill_profile_active_calls,
            "token_rows": cls._prefill_profile_token_rows,
            "wall_s": cls._prefill_profile_wall_s,
        }

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not type(self)._prefill_profile_enabled:
            return super().forward(hidden_states)
        start = time.perf_counter()
        result = super().forward(hidden_states)
        cls = type(self)
        cls._prefill_profile_calls += 1
        rows = int(hidden_states.shape[0])
        if rows:
            cls._prefill_profile_active_calls += 1
            cls._prefill_profile_token_rows += rows
        cls._prefill_profile_wall_s += time.perf_counter() - start
        return result

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
