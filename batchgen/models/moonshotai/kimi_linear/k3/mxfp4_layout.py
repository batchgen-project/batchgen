# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-K3 MXFP4 routed-expert ingest contract.

Single source of truth for how a routed expert's ``(weight_packed, weight_scale)``
couples are named, shaped, typed and validated between the checkpoint and the
fused MXFP4-dequant grouped GEMM.

NOTHING HERE DEQUANTIZES.  K3 keeps routed-expert weights MXFP4-packed end to
end (PREFILL_PLAN.md §0.1b): checkpoint bytes -> converted .bin -> host SHM ->
H2D memcpy -> GPU ring slot -> fused dequant inside the GEMM's K-loop.  Host-side
dequantization is 3.765x the bytes (17.55 MB -> 66.06 MB per expert) and puts the
model 2.6x over the 2.147 TB host ceiling.

The packed/scale couple is kept together BY CONSTRUCTION, not by new machinery:
both members carry the same ``module_key``, so ``Parameter_Server.cpp:357-366``
files them into one host module map, ``HtoD_Engine.cu:439-452`` copies them into
one GPU slot, and ``GPU_Weight_Buffer.cpp:294-306`` publishes that slot exactly
once.  No consumer can observe packed-without-scale.

Format facts, read out of the released checkpoint (not from the modeling files,
which contain zero dequant code):

  * ``quant_method`` "compressed-tensors", ``format`` "mxfp4-pack-quantized",
    ``group_size`` 32, ``num_bits`` 4, ``type`` "float", ``symmetric`` true,
    ``scale_dtype`` "torch.uint8" (E8M0), ``input_activations`` **null**.
    Weight-only W4A16 — activations stay BF16.
  * The declaration is NESTED at ``config["text_config"]["quantization_config"]``;
    ``KimiLinearConfig.from_hf_dict`` flattens ``text_config``, so a config built
    any other way arrives here with ``quantization_config is None`` and is
    rejected rather than treated as unquantized.
  * Per expert: ``w1``/``w3`` packed U8[3072,1792] + scale U8[3072,112] (K=3584);
    ``w2`` packed U8[3584,1536] + scale U8[3584,96] (K=3072).  17,547,264 B.
"""

from typing import Any, Dict, List, Optional, Sequence

import torch


class K3QuantContractError(RuntimeError):
    """Raised whenever the checkpoint or a GPU slot violates the contract.

    Never downgraded to a warning: a violation here is the silent-wrong-weights
    class (in-repo postmortem: ``models/glm/glm5/glm5_initializer.py:141-146``).
    """


# --------------------------------------------------------------------------- #
#  Declared format                                                             #
# --------------------------------------------------------------------------- #

#: The fused GEMM hardcodes ``num_k_blocks = K // 32``
#: (``batchgen/moe/mxfp4_grouped_gemm.py:578``); any other group size is a
#: different kernel.
MXFP4_GROUP_SIZE = 32
#: Two E2M1 codes per uint8.
MXFP4_PACK_FACTOR = 2
MXFP4_QUANT_METHOD = "compressed-tensors"
MXFP4_FORMAT_SUBSTR = "mxfp4"
#: Packed codes AND E8M0 scales are both uint8 in K3 — unlike K2.5, where the
#: scale is BF16.
MXFP4_DTYPE = torch.uint8

#: task #53 (V1 offline marlin). When True, the converter emits routed-
#: expert weights in marlin tile order (int32 qw + marlin-order uint8 E8M0
#: scale, +0 host bytes) and the serving paths consume marlin directly:
#: DECODE direct-copies (get_tensor exposes the true int32 dtype), PREFILL
#: reinterprets the packed-shaped uint8 GPU slot bytes as marlin and SKIPS
#: the per-forward repack. The converter flag (kimi_parameter_server) reads
#: THIS constant, so a fresh conversion always matches. FLIPPING THIS
#: REQUIRES RE-CONVERTING (a stale converted_ckpt is reused if present).
K3_ROUTED_EXPERTS_MARLIN = True

PACKED_SUFFIX = ".weight_packed"
SCALE_SUFFIX = ".weight_scale"
#: Checkpoint names: w1 = gate, w3 = up, w2 = down.
ROUTED_EXPERT_PROJECTIONS = ("w1", "w3", "w2")

#: The six checkpoint patterns the released `quantization_config["ignore"]`
#: declares NOT quantized.  This is the checkpoint stating, in its own words,
#: which modules are BF16 — the same question ``module_shapes`` answers, and the
#: only free evidence for it.  Pinned to EXACT equality in both directions:
#: an entry REMOVED means a module became MXFP4 while we still declare it BF16
#: (attention, shared experts, lm_head); an entry ADDED means a module became
#: BF16 while we may still declare it uint8.  If a future revision only adds a
#: module already declared BF16, add it here after checking `module_shapes`.
#: NOTE the converse does NOT hold: `block_sparse_moe.gate` and
#: `routed_expert_{down,up}_proj` are absent from this list yet ship BF16
#: (verified from the shard headers), so "not ignored" does not imply packed.
K3_EXPECTED_QUANT_IGNORE = frozenset({
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
    "re:.*lm_head.*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
})

# Which nibble of a packed byte holds the LOWER K index.  UNVERIFIED — this is
# an ASSUMPTION carried in code so it is reviewable, not a measured constant.
# "lo_first" is what every existing BatchGen consumer assumes
# (``batchgen/quantization/mxfp4.py:61`` and
# ``batchgen/moe/mxfp4_grouped_gemm.py:591``).  A wrong value swaps adjacent K
# pairs in every expert weight in the model: plausible weights, a different
# model, invisible to every per-group statistic AND to the packed-expert HF
# oracle (which shares this decode by construction, PREFILL_PLAN.md:263).
#
# EVIDENCE — see batchgen_design/model_support/kimi_k3/NIBBLE_ORDER.md:
#   N0 writer source ......... <pin: package@version, file:line>     UNFILLED
#   N1 group-max-code check .. <pin: N groups, % max magnitude in {4, 6}> UNFILLED
#   N2 latent-channel pairing  <pin: rho_lo, rho_hi, null spread>    UNFILLED
# Nothing reads this yet.  The dequant consumer MUST NOT be written against it
# until N0-N2 are filled in; re-run all three before editing the value.
K3_MXFP4_NIBBLE_ORDER_ASSUMED = "lo_first"


def routed_expert_tensor_names() -> List[str]:
    """The six ``tensor_key``s of one routed-expert module.

    Order is for human reading only (packed next to its scale).  It has no
    runtime effect: the H2D producer iterates
    ``Weights_Storage::get_module_weights_storage``, which returns a
    ``std::unordered_map`` (``Weights_Storage.h:70``), so the copy order inside
    a module is hash order.
    """
    names: List[str] = []
    for projection in ROUTED_EXPERT_PROJECTIONS:
        names.append(projection + PACKED_SUFFIX)
        names.append(projection + SCALE_SUFFIX)
    return names


def _packed_scale_shapes(n_out: int, k_in: int):
    if k_in % MXFP4_GROUP_SIZE != 0:
        raise K3QuantContractError(
            f"K={k_in} is not a multiple of group_size={MXFP4_GROUP_SIZE}. "
            "K3 ships no padded K; a padded K needs a different kernel."
        )
    return ([n_out, k_in // MXFP4_PACK_FACTOR],
            [n_out, k_in // MXFP4_GROUP_SIZE])


def routed_expert_module_shapes(moe_intermediate_size: int,
                                routed_expert_hidden_size: int
                                ) -> Dict[str, List[int]]:
    """``module_shapes["routed_expert"]`` for K3.

    K3 experts live in the MoE LATENT space (``routed_expert_hidden_size`` 3584),
    not hidden (7168) — ``routed_expert_{down,up}_proj`` sit on the token stream
    outside dispatch.  With F=3072, L=3584 this yields the checkpoint's shapes
    exactly: w1/w3 [3072,1792]+[3072,112], w2 [3584,1536]+[3584,96].
    """
    ffn = int(moe_intermediate_size)
    latent = int(routed_expert_hidden_size)
    gate_up_packed, gate_up_scale = _packed_scale_shapes(ffn, latent)
    down_packed, down_scale = _packed_scale_shapes(latent, ffn)
    return {
        "w1" + PACKED_SUFFIX: gate_up_packed,
        "w1" + SCALE_SUFFIX: gate_up_scale,
        "w3" + PACKED_SUFFIX: gate_up_packed,
        "w3" + SCALE_SUFFIX: gate_up_scale,
        "w2" + PACKED_SUFFIX: down_packed,
        "w2" + SCALE_SUFFIX: down_scale,
    }


def routed_expert_tensor_dtypes() -> Dict[str, torch.dtype]:
    """Every K3 routed-expert tensor is uint8 — packed AND E8M0 scale alike.

    Declared per tensor rather than left to the ``weight_dtypes`` fallback
    (``GPU_Weight_Buffer.cpp:59-82``) because an unresolved dtype string silently
    becomes fp8 at ``Weights_Storage.cpp:233-239``.
    """
    return {name: MXFP4_DTYPE for name in routed_expert_tensor_names()}


# --------------------------------------------------------------------------- #
#  Validation — every unsupported declaration hard-fails                        #
# --------------------------------------------------------------------------- #

def validate_quantization_config(quantization_config: Optional[Dict[str, Any]]
                                 ) -> Dict[str, Any]:
    """Reject every quantization declaration this load path does not implement.

    Takes the flattened ``KimiLinearConfig.quantization_config``.  ``None`` is an
    error, not "unquantized": K3 nests the block under ``text_config``, so a
    config read from the top level of ``config.json`` arrives here empty and
    would otherwise load 1.45 TB of packed nibbles as if they were weights.
    """
    if quantization_config is None:
        raise K3QuantContractError(
            "quantization_config is None. K3 NESTS it at "
            "config['text_config']['quantization_config'] "
            "(configuration_kimi_k3.py:282-283 hoists it only at HF runtime); "
            "build the config with KimiLinearConfig.from_json, which flattens "
            "text_config. Refusing to load K3 as an unquantized model."
        )

    method = quantization_config.get("quant_method")
    if method != MXFP4_QUANT_METHOD:
        raise K3QuantContractError(
            f"quant_method={method!r}, expected {MXFP4_QUANT_METHOD!r}."
        )

    groups = quantization_config.get("config_groups") or {}
    if not groups:
        raise K3QuantContractError(
            "quantization_config has no config_groups; nothing declares the "
            "weight format."
        )

    for group_name, group in groups.items():
        fmt = group.get("format") or quantization_config.get("format") or ""
        if MXFP4_FORMAT_SUBSTR not in str(fmt):
            raise K3QuantContractError(
                f"config_group {group_name!r}: format={fmt!r} does not contain "
                f"{MXFP4_FORMAT_SUBSTR!r}."
            )
        if group.get("input_activations") is not None:
            raise K3QuantContractError(
                f"config_group {group_name!r} declares input_activations. K3 "
                "prefill is weight-only W4A16 with BF16 activations; the fused "
                "MXFP4 GEMM takes a BF16 A-operand and there is no activation-"
                "quant path (PREFILL_PLAN.md §0.1c)."
            )
        weights = group.get("weights") or {}
        expectations = (
            ("group_size", MXFP4_GROUP_SIZE),
            ("num_bits", 4),
            ("type", "float"),
            ("symmetric", True),
            ("strategy", "group"),
            ("scale_dtype", "torch.uint8"),
            ("dynamic", False),
        )
        for field_name, expected in expectations:
            got = weights.get(field_name)
            if got != expected:
                raise K3QuantContractError(
                    f"config_group {group_name!r}: weights.{field_name}={got!r}, "
                    f"expected {expected!r}."
                )
        for field_name in ("actorder", "block_structure"):
            if weights.get(field_name) is not None:
                raise K3QuantContractError(
                    f"config_group {group_name!r}: weights.{field_name}="
                    f"{weights.get(field_name)!r}; permuted/blocked layouts are "
                    "not implemented."
                )

    status = quantization_config.get("quantization_status")
    if status != "compressed":
        raise K3QuantContractError(
            f"quantization_status={status!r}, expected 'compressed'. Any other "
            "status ('frozen', 'calibration') means the shipped tensors are not "
            "the packed form this load path streams."
        )

    ignored = frozenset(quantization_config.get("ignore") or ())
    if ignored != K3_EXPECTED_QUANT_IGNORE:
        raise K3QuantContractError(
            "quantization_config['ignore'] changed: "
            f"missing={sorted(K3_EXPECTED_QUANT_IGNORE - ignored)}, "
            f"added={sorted(ignored - K3_EXPECTED_QUANT_IGNORE)}. This list is "
            "the checkpoint's own statement of which modules are NOT MXFP4, and "
            "module_shapes declares attention / shared experts / lm_head / the "
            "dense MLP as BF16 on the strength of it. Re-derive the affected "
            "module_shapes entries before updating K3_EXPECTED_QUANT_IGNORE."
        )
    return quantization_config


def validate_routed_expert_slot(module_key: str,
                                weights: Dict[str, torch.Tensor],
                                expected_shapes: Dict[str, Sequence[int]]
                                ) -> None:
    """Validate one GPU ring slot before it is handed to the GEMM.

    Catches, in order: name-set drift, dtype drift, shape drift and a broken
    packed/scale pairing.  Cheap (six metadata reads, no device sync) and it is
    the last line before the kernel reads the pointers.
    """
    got, want = set(weights), set(expected_shapes)
    if got != want:
        raise K3QuantContractError(
            f"{module_key}: slot tensors {sorted(got)} != declared "
            f"module_shapes {sorted(want)} (missing={sorted(want - got)}, "
            f"extra={sorted(got - want)})."
        )
    for name, tensor in weights.items():
        if tensor.dtype is not MXFP4_DTYPE:
            raise K3QuantContractError(
                f"{module_key}.{name}: dtype={tensor.dtype}, expected uint8."
            )
        if list(tensor.shape) != [int(s) for s in expected_shapes[name]]:
            raise K3QuantContractError(
                f"{module_key}.{name}: shape={list(tensor.shape)}, expected "
                f"{[int(s) for s in expected_shapes[name]]}."
            )
        if not tensor.is_contiguous():
            raise K3QuantContractError(
                f"{module_key}.{name}: non-contiguous slot tensor; the grouped "
                "GEMM takes strides from a reference tensor and assumes every "
                "expert has an identical layout."
            )
    for projection in ROUTED_EXPERT_PROJECTIONS:
        packed = weights[projection + PACKED_SUFFIX]
        scale = weights[projection + SCALE_SUFFIX]
        if packed.shape[0] != scale.shape[0]:
            raise K3QuantContractError(
                f"{module_key}.{projection}: N mismatch — packed "
                f"{packed.shape[0]} vs scale {scale.shape[0]}."
            )
        k_from_packed = packed.shape[1] * MXFP4_PACK_FACTOR
        k_from_scale = scale.shape[1] * MXFP4_GROUP_SIZE
        if k_from_packed != k_from_scale:
            raise K3QuantContractError(
                f"{module_key}.{projection}: K disagreement — packed implies "
                f"{k_from_packed}, scale implies {k_from_scale}."
            )
