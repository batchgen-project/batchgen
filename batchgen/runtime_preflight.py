"""Fail-closed runtime dependency checks for the production server.

The installer validates the environment once, but a running server can still
resolve a different worktree or discover a lazy native dependency only after
the first request.  This module is deliberately small and import-only: it
must run before worker processes, SHM, or model weights are created.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import site
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from batchgen.kernel_compat import check_kernels_version

logger = logging.getLogger(__name__)


# First case-insensitive substring match wins, so a more specific pattern must
# precede any pattern it contains when the two resolve differently ("Kimi-K3"
# is a substring of "Kimi-K3.1-Pro"). An entry whose type equals that of a
# LATER pattern it contains is dead and is not listed: the later one already
# resolves every identifier it would (tests/test_runtime_preflight.py).
_KNOWN_IDENTIFIER_PATTERNS = (
    ("DeepSeek-V4-Flash", "deepseek_v4"),
    ("DeepSeek-V4-Pro", "deepseek_v4"),
    ("MiniMax-M2.5", "minimax_m25"),
    ("DeepSeek-R1", "deepseek_v3"),
    ("DeepSeek-V3", "deepseek_v3"),
    ("DeepSeek-V2", "deepseek_v2"),
    ("Mixtral-8x22B", "mixtral"),
    ("Mixtral-8x7B", "mixtral"),
    ("gpt-oss", "gpt_oss"),
    ("GLM-5.3", "glm_moe_dsa_5_3"),
    ("GLM-5.2", "glm_moe_dsa_5_2"),
    ("GLM-5", "glm_moe_dsa"),
    ("Kimi-Linear", "kimi_linear"),
    ("Kimi-K3.1", "kimi_k31"),
    ("Kimi-K3", "kimi_k3"),
)


def _detect_model_type_from_identifier(model_identifier: str):
    """Resolve common IDs without importing every model package.

    The full registry eagerly imports model parameter servers.  That is too
    broad for a startup gate: an unrelated model's native dependency must not
    prevent the gate from reporting the selected model's own contract.
    """

    for pattern, model_type in _KNOWN_IDENTIFIER_PATTERNS:
        if pattern.lower() in model_identifier.lower():
            return model_type
    from batchgen.config.model_registry import _detect_model_type_from_identifier as detect

    return detect(model_identifier)


def load_config(model_identifier: str):
    """Load model config lazily so missing native deps are reported by preflight."""

    from batchgen.config.model_registry import load_config as load

    return load(model_identifier)


class RuntimePreflightError(RuntimeError):
    """Raised when the selected model cannot run in the current environment."""


@dataclass(frozen=True)
class RuntimeContract:
    """Native/runtime capabilities required by one exact model type."""

    modules: tuple[str, ...] = ()
    # Required on every device.
    extensions: tuple[str, ...] = ()
    # sm_90a-only builds the generation path loads on a Hopper (sm_90) device;
    # other architectures do not build them and the model takes its
    # non-WGMMA kernels there.
    hopper_extensions: tuple[str, ...] = ()
    require_deepgemm: bool = False


_COMMON_EXTENSIONS = (
    "batchgen_kernels.attention._C_fused_ops",
    "batchgen_kernels.attention._C_gqa_mha_decode_bf16",
)

# AOT extensions of the page-split FP8 DSA indexer itself: the fused index-key
# writer and the top-k both scorers end in. Every DSA contract carries them.
_DSA_PAGE_SPLIT_INDEXER_EXTENSIONS = (
    "batchgen_kernels.attention._C_fused_indexer_k_norm_rope_write_fp8",
    "batchgen_kernels.attention._C_dsa_fast_topk",
)

# AOT extensions the GLM-5 DSA decode path loads on top of the common set.
# They are listed per contract rather than in _COMMON_EXTENSIONS because a
# model that never takes the DSA decode path must not be refused for them.
_GLM5_DSA_EXTENSIONS = _DSA_PAGE_SPLIT_INDEXER_EXTENSIONS + (
    "batchgen_kernels.attention._C_add_rmsnorm_fp8_quant",
    "batchgen_kernels.attention._C_fast_hadamard_transform",
    "batchgen_kernels.attention._C_fused_rope_hadamard",
    "batchgen_kernels.attention._C_indexer_kv_proj_wgmma",
    # The decode criticality probe's spin kernel. Loaded only when a batch
    # requests the probe, but checked here for the same reason as the opt-in
    # WGMMA MoE set below: it ships in the same kernel wheel, so a partial
    # build is refused at startup, not at the first probed capture.
    "batchgen_kernels.common._C_spin_delay",
)

# The FP8 WGMMA MoE pipeline the DSA models can select. Checked for every
# DSA contract even though the pipeline is opt-in: the extensions ship in
# the same kernel wheel, so a partial build is refused at startup rather than
# at the first MoE forward that selects the pipeline.
_DSA_MOE_WGMMA_EXTENSIONS = (
    "batchgen_kernels.moe._C_fp8_wgmma_grouped_gemm",
    "batchgen_kernels.moe._C_fp8_wgmma_quant_ops",
    "batchgen_kernels.moe._C_dispatch_scatter_3d",
)

# Every extension a contract lists is one its model package can load
# (tests/test_runtime_preflight_extension_coverage.py derives the set from the
# source and fails when a contract misses one), so a partial kernel build is
# refused here rather than minutes into decode.
_K = "batchgen_kernels."


def _ext(*names: str) -> tuple[str, ...]:
    return tuple(_K + name for name in names)


# GLM-5.x on Hopper: FP8 blockwise MoE GEMMs and the FP8 DSA index-key write.
_DSA_FP8_HOPPER_EXTENSIONS = _ext(
    "moe._C_fp8_blockwise_gemm",
    "moe._C_fp8_blockwise_ops",
    "moe._C_routing",
    "moe._C_routing_wgmma",
    "attention._C_fused_indexer_k_write_fp8",
    "attention._C_fused_q_split",
    "attention._C_qkv_wgmma",
)

# KDA models: Kimi-Linear and Kimi-K3 (one model package; Kimi-K3.1 adds the
# DSA indexer).
_KDA_EXTENSIONS = _COMMON_EXTENSIONS + _ext("conv1d._C_causal_conv1d")
_KDA_HOPPER_EXTENSIONS = _ext(
    "attention._C_fused_q_split",
    "attention._C_kda_fused_decode",
    "moe._C_dispatch_scatter_3d",
    "moe._C_marlin_grouped_gemm",
    "moe._C_routing",
    "moe._C_routing_wgmma",
)

# Generic CUDA extensions selected by BUILD_ARCH=sm103a.  The Blackwell Kimi
# contracts use this set so a partial lane build fails before model startup.
# _C_gqa_mha_decode_bf16 is excluded: it is a sourceless prebuilt binary for
# the Hopper decode path (its consumer falls back on any non-(9,0) device),
# so it cannot be rebuilt for this lane and is not reachable on it.
_SM103A_EXTENSIONS = (
    tuple(
        extension
        for extension in _COMMON_EXTENSIONS
        if extension != "batchgen_kernels.attention._C_gqa_mha_decode_bf16"
    )
    + (
        "batchgen_kernels.conv1d._C_causal_conv1d",
        "batchgen_kernels.attention._C_add_rmsnorm_fp8_quant",
        "batchgen_kernels.moe._C_mxfp4_dequant_cute",
        "batchgen_kernels.moe._C_mxfp4_dequant",
        "batchgen_kernels.common._C_rmsnorm",
        "batchgen_kernels.common._C_cuda_rmsnorm",
        "batchgen_kernels.common._C_spin_delay",
        "batchgen_kernels.common._C_mgn_ops",
        "batchgen_kernels.moe._C_fused_moe_token_permutation",
        "batchgen_kernels.attention._C_dsa_fast_topk",
        "batchgen_kernels.attention._C_fast_hadamard_transform",
        "batchgen_kernels.attention._C_fused_rope_hadamard",
        "batchgen_kernels.moe._C_marlin_grouped_gemm",
        "batchgen_kernels.moe._C_routing",
        "batchgen_kernels.moe._C_dispatch_scatter_3d",
        "batchgen_kernels.attention._C_fused_kv_norm_rope",
        "batchgen_kernels.attention._C_fused_q_absorb",
        "batchgen_kernels.attention._C_fused_q_split",
        "batchgen_kernels.attention._C_fused_indexer_k_write_fp8",
        "batchgen_kernels.attention._C_fused_indexer_k_norm_rope_write_fp8",
        "batchgen_kernels.attention._C_kda_fused_decode",
    )
)

# These are exact model-type contracts, not family-wide fallback rules.  A new
# model type must be added here before it can enter the production server path.
_CONTRACTS: dict[str, RuntimeContract] = {
    "gpt_oss": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_COMMON_EXTENSIONS + _ext("moe._C_mxfp4_dequant_cute"),
        hopper_extensions=_ext(
            "moe._C_expert_mxfp4_wgmma",
            "moe._C_grouped_mxfp4_wgmma",
            "moe._C_routing",
            "moe._C_routing_wgmma",
            "attention._C_qkv_wgmma",
        ),
        require_deepgemm=True,
    ),
    "glm_moe_dsa": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=(_COMMON_EXTENSIONS + _GLM5_DSA_EXTENSIONS
                    + _DSA_MOE_WGMMA_EXTENSIONS),
        hopper_extensions=_DSA_FP8_HOPPER_EXTENSIONS,
        require_deepgemm=True,
    ),
    "glm_moe_dsa_5_2": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=(_COMMON_EXTENSIONS + _GLM5_DSA_EXTENSIONS
                    + _DSA_MOE_WGMMA_EXTENSIONS),
        hopper_extensions=_DSA_FP8_HOPPER_EXTENSIONS,
        require_deepgemm=True,
    ),
    "glm_moe_dsa_5_3": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=(_COMMON_EXTENSIONS + _GLM5_DSA_EXTENSIONS
                    + _DSA_MOE_WGMMA_EXTENSIONS),
        hopper_extensions=_DSA_FP8_HOPPER_EXTENSIONS,
        require_deepgemm=True,
    ),
    "deepseek_v2": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS,
    ),
    "deepseek_v3": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS + _ext(
            "common._C_mgn_ops", "moe._C_fused_moe_token_permutation"
        ),
        hopper_extensions=_ext("attention._C_fused_q_split"),
    ),
    "deepseek_v4": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS,
    ),
    "mixtral": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS,
    ),
    "minimax_m25": RuntimeContract(
        modules=("flash_attn_interface", "libucx"),
        extensions=_COMMON_EXTENSIONS + _ext("common._C_mgn_ops"),
        hopper_extensions=_ext(
            "moe._C_dispatch_scatter_3d",
            "moe._C_fp8_blockwise_gemm",
            "moe._C_fp8_blockwise_ops",
            "moe._C_routing",
            "moe._C_routing_wgmma",
            "attention._C_fused_q_split",
            "attention._C_qkv_wgmma",
        ),
    ),
    "kimi_linear": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_KDA_EXTENSIONS,
        hopper_extensions=_KDA_HOPPER_EXTENSIONS,
    ),
    "kimi_k3": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_KDA_EXTENSIONS,
        hopper_extensions=_KDA_HOPPER_EXTENSIONS,
    ),
    # The dense contract plus the page-split DSA indexer: DeepGEMM scores the
    # index keys (paged on decode, causal on prefill).
    "kimi_k31": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_KDA_EXTENSIONS + _DSA_PAGE_SPLIT_INDEXER_EXTENSIONS,
        hopper_extensions=_KDA_HOPPER_EXTENSIONS,
        require_deepgemm=True,
    ),
    "kimi_k25": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS + _ext(
            "common._C_cuda_rmsnorm",
            "common._C_mgn_ops",
            "moe._C_fused_moe_token_permutation",
        ),
        hopper_extensions=_ext(
            "moe._C_dispatch_scatter_3d",
            "moe._C_fused_int4_wgmma_grouped",
            "moe._C_marlin_grouped_gemm",
            "moe._C_marlin_transform",
            "moe._C_routing",
            "moe._C_routing_wgmma",
            "moe._C_single_expert_int4_wgmma",
            "attention._C_fused_q_split",
        ),
    ),
}

_KIMI_MODEL_TYPES = frozenset(
    model_type
    for identifier, model_type in _KNOWN_IDENTIFIER_PATTERNS
    if identifier.startswith("Kimi")
)

_BLACKWELL_KIMI_CONTRACTS = {
    model_type: RuntimeContract(
        # flash_mla: graph/legacy entry points; flashinfer: the Blackwell
        # dense paged MLA decode and varlen prefill kernels.
        modules=("flash_mla", "flashinfer"),
        extensions=_SM103A_EXTENSIONS,
        require_deepgemm=contract.require_deepgemm,
    )
    for model_type, contract in _CONTRACTS.items()
    if model_type in _KIMI_MODEL_TYPES
}


def _detect_runtime_capability() -> tuple[tuple[int, int], int | None]:
    """Return one capability snapshot for backend and contract selection."""
    if not torch.cuda.is_available():
        return (9, 0), None
    capability = tuple(torch.cuda.get_device_capability())
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return capability, properties.multi_processor_count


def _decode_backend_for_capability(
    capability: tuple[int, int], sm_count: int | None
) -> str:
    """Map the detected capability to the single decode backend table."""
    if capability == (9, 0):
        return "wgmma" if sm_count == 78 else "fa3"
    if capability[0] == 10:
        return "flashmla"
    raise RuntimePreflightError(
        f"unsupported CUDA compute capability {capability}; no decode backend"
    )


def _contract_for_capability(
    model_type: str, capability: tuple[int, int]
) -> RuntimeContract:
    """Select the exact model contract from the one capability snapshot."""
    if capability[0] == 10 and model_type in _BLACKWELL_KIMI_CONTRACTS:
        return _BLACKWELL_KIMI_CONTRACTS[model_type]
    return _CONTRACTS[model_type]


def _site_roots() -> tuple[Path, ...]:
    roots = {
        *site.getsitepackages(),
        site.getusersitepackages(),
        sysconfig.get_paths()["purelib"],
    }
    # Deepest first, so a nested site dir wins deterministically.
    resolved = {Path(root).resolve() for root in roots if root}
    return tuple(sorted(resolved, key=lambda p: len(p.parts), reverse=True))


def _module_path(module_name: str, module: object) -> Path:
    module_path = getattr(module, "__file__", None)
    if not module_path:
        raise RuntimePreflightError(
            f"{module_name} has no __file__; refusing unverifiable runtime module"
        )
    return Path(module_path).resolve()


def _under(path: Path, roots: Iterable[Path]) -> bool:
    return any(root == path or root in path.parents for root in roots)


def _batchgen_root() -> Path:
    """Root that owns the running ``batchgen`` package.

    Installed mode: a site-packages directory.  Source mode (pyroot/PYTHONPATH):
    the worktree that ``batchgen`` resolves to after following symlinks.
    """

    import batchgen

    package_dir = _module_path("batchgen", batchgen).parent
    for root in _site_roots():
        if _under(package_dir, (root,)):
            return root
    return package_dir.parent


def _is_source_root(root: Path) -> bool:
    return root not in _site_roots()


# Where each required module may come from:
#   "site"     - third-party native package, must be installed in site-packages
#   "batchgen" - BatchGen-owned native code, must share the batchgen root so a
#                worktree never mixes its Python with another checkout's kernels
_ORIGIN_SITE = "site"
_ORIGIN_BATCHGEN = "batchgen"


def _require_origin(module_name: str, module: object, origin: str) -> Path:
    resolved = _module_path(module_name, module)
    if origin == _ORIGIN_SITE:
        if not _under(resolved, _site_roots()):
            raise RuntimePreflightError(
                f"{module_name} resolves outside site-packages: {resolved}; "
                "third-party native packages must be installed, not shadowed"
            )
    elif origin == _ORIGIN_BATCHGEN:
        root = _batchgen_root()
        if not _under(resolved, (root,)):
            raise RuntimePreflightError(
                f"{module_name} resolves to {resolved}, but batchgen runs from "
                f"{root}; batchgen, batchgen_kernels, and core_engine must come "
                "from the same install or worktree"
            )
    return resolved


def _import_required(module_name: str, *, origin: str | None = None) -> object:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - preserve the native import cause
        raise RuntimePreflightError(
            f"required runtime module {module_name!r} failed to import: {exc}"
        ) from exc
    if origin is not None:
        _manifest[module_name] = str(_require_origin(module_name, module, origin))
    else:
        _manifest[module_name] = str(getattr(module, "__file__", "<built-in>"))
    logger.info("[runtime-preflight] %s=%s", module_name, _manifest[module_name])
    return module


_manifest: dict[str, str] = {}

_CORE_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}


def _check_core_engine() -> None:
    core_engine = _import_required("batchgen.core_engine", origin=_ORIGIN_BATCHGEN)
    core_path = Path(getattr(core_engine, "__file__", ""))
    if core_path.suffix not in {".so", ".pyd", ".dylib"}:
        raise RuntimePreflightError(
            f"batchgen.core_engine is not an AOT native module: {str(core_path)!r}"
        )
    root = _batchgen_root()
    if not _is_source_root(root):
        return
    # Source worktree: the in-place build must be newer than every build input,
    # otherwise the server would silently run an old engine.
    if not (root / "core").is_dir():
        raise RuntimePreflightError(
            f"source worktree {root} has no core/ directory; cannot verify "
            "that batchgen.core_engine matches its sources"
        )
    inputs = [
        p
        for d in ("core", "external")
        for p in (root / d).rglob("*")
        if p.is_file() and p.suffix in _CORE_SOURCE_SUFFIXES
    ]
    inputs.append(root / "op_builder" / "core_engine.py")
    inputs = [p for p in inputs if p.is_file()]
    newest = max(inputs, key=lambda p: p.stat().st_mtime, default=None)
    if newest is not None and newest.stat().st_mtime > core_path.stat().st_mtime:
        raise RuntimePreflightError(
            f"batchgen.core_engine {core_path} is older than {newest}; rebuild "
            "with `BUILD_OPS=1 python setup.py build_ext --inplace`"
        )


def _check_torch() -> None:
    version = torch.__version__
    if not version.startswith("2.9.0"):
        raise RuntimePreflightError(
            f"Torch {version!r} is unsupported; expected 2.9.0 with the pinned CUDA ABI"
        )
    expected_channel = os.environ.get("TORCH_CUDA_CHANNEL", "cu128")
    expected_cuda = (
        f"{expected_channel[2:-1]}.{expected_channel[-1]}"
        if expected_channel.startswith("cu") and expected_channel[2:].isdigit()
        else None
    )
    if expected_cuda and torch.version.cuda != expected_cuda:
        raise RuntimePreflightError(
            f"Torch CUDA runtime {torch.version.cuda!r} does not match "
            f"TORCH_CUDA_CHANNEL={expected_channel!r} ({expected_cuda})"
        )
    logger.info(
        "[runtime-preflight] torch=%s cuda=%s",
        version,
        torch.version.cuda,
    )


def _check_ucx(module: object) -> None:
    loader = getattr(module, "load_library", None)
    if not callable(loader):
        raise RuntimePreflightError("libucx has no load_library() runtime contract")
    try:
        loader()
    except Exception as exc:  # noqa: BLE001 - preserve loader diagnostics
        raise RuntimePreflightError(f"libucx.load_library() failed: {exc}") from exc


def _check_deepgemm() -> None:
    module = _import_required("deep_gemm")
    try:
        version = importlib.metadata.version("sgl-deep-gemm")
        signature = inspect.signature(module.fp8_mqa_logits)
    except Exception as exc:  # noqa: BLE001 - report the exact contract miss
        raise RuntimePreflightError(
            f"DeepGEMM runtime contract is incomplete: {exc}"
        ) from exc
    if "max_seqlen_k" not in signature.parameters:
        raise RuntimePreflightError(
            "DeepGEMM fp8_mqa_logits() lacks required max_seqlen_k parameter"
        )
    logger.info("[runtime-preflight] sgl-deep-gemm=%s", version)


def _check_tokenizer(model: str) -> None:
    try:
        from batchgen.config.tokenizer_registry import load_tokenizer

        tokenizer = load_tokenizer(model)
    except Exception as exc:  # noqa: BLE001 - preserve tokenizer contract cause
        raise RuntimePreflightError(
            f"tokenizer contract failed for {model!r}: {exc}"
        ) from exc
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is None:
        # Single-EOS tokenizers (including GPT-OSS) intentionally expose the
        # singular HuggingFace-compatible attribute.  The worker normalizes it
        # to a set; preflight must validate the same contract rather than
        # rejecting a valid tokenizer for lacking an optional plural alias.
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_ids = set() if eos_id is None else {eos_id}
    if not eos_ids:
        raise RuntimePreflightError(
            f"tokenizer contract for {model!r} does not define eos_token_ids"
        )
    renderer = getattr(tokenizer, "apply_chat_template", None)
    if callable(renderer):
        try:
            rendered = renderer(
                [{"role": "user", "content": "runtime preflight"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:  # noqa: BLE001 - template is a runtime contract
            raise RuntimePreflightError(
                f"chat-template contract failed for {model!r}: {exc}"
            ) from exc
        if not isinstance(rendered, str) or not rendered:
            raise RuntimePreflightError(
                f"chat-template contract for {model!r} returned no rendered text"
            )
        logger.info("[runtime-preflight] chat_template_chars=%d", len(rendered))
    logger.info(
        "[runtime-preflight] tokenizer=%s eos_token_ids=%s",
        type(tokenizer).__name__,
        sorted(eos_ids),
    )


def _resolve_model_type(model: str) -> str:
    model_type = _detect_model_type_from_identifier(model)
    if model_type is None:
        try:
            model_type = load_config(model).model_type
        except Exception as exc:  # noqa: BLE001 - preserve model resolution cause
            raise RuntimePreflightError(
                f"cannot resolve an exact runtime contract for model {model!r}: {exc}"
            ) from exc
    if model_type not in _CONTRACTS:
        raise RuntimePreflightError(
            f"model type {model_type!r} has no declared runtime contract; "
            "refusing an implicit dependency/fallback path"
        )
    return model_type


def run_runtime_preflight(server_args: object) -> str:
    """Validate the exact model runtime before server resources are allocated."""

    model = str(getattr(server_args, "model", ""))
    _manifest.clear()
    if not model:
        raise RuntimePreflightError("server model is empty")

    if os.environ.get("BATCHGEN_KERNELS_DEV") == "1":  # noqa: hygiene - rejects dev JIT
        raise RuntimePreflightError(
            "BATCHGEN_KERNELS_DEV=1 enables runtime JIT compilation; unset it and "
            "build the kernels ahead of time"
        )
    _check_torch()
    model_type = _resolve_model_type(model)
    capability, sm_count = _detect_runtime_capability()
    contract = _contract_for_capability(model_type, capability)
    decode_backend = _decode_backend_for_capability(capability, sm_count)
    _check_tokenizer(model)

    for module_name in contract.modules:
        module = _import_required(
            module_name,
            origin=_ORIGIN_SITE
            if module_name in {"flash_attn_interface", "flash_attn", "flash_mla"}
            else None,
        )
        if module_name == "libucx":
            _check_ucx(module)

    hopper = capability == (9, 0) and sm_count is not None
    extensions = contract.extensions + (contract.hopper_extensions if hopper else ())
    for extension_name in extensions:
        _import_required(extension_name, origin=_ORIGIN_BATCHGEN)

    if decode_backend == "wgmma":
        decode_module = _import_required(
            "batchgen_kernels.attention.decode", origin=_ORIGIN_BATCHGEN
        )
        if not callable(getattr(decode_module, "attention_decode_bf16", None)):
            raise RuntimePreflightError(
                "selected WGMMA decode backend has no attention_decode_bf16()"
            )

    _check_core_engine()

    if contract.require_deepgemm:
        _check_deepgemm()

    logger.info(
        "[runtime-preflight] passed model=%s model_type=%s root=%s manifest=%s",
        model,
        model_type,
        _batchgen_root(),
        json.dumps(_manifest, sort_keys=True),
    )
    return model_type


def main(argv: list[str] | None = None) -> int:
    """Run the startup gate without launching a server (agent/debug entry)."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_kernels_version()
    try:
        model_type = run_runtime_preflight(args)
    except RuntimePreflightError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)  # noqa: hygiene - CLI output
        return 1
    print(json.dumps(  # noqa: hygiene - CLI output
        {"model_type": model_type, "root": str(_batchgen_root()), "modules": _manifest},
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RuntimePreflightError", "RuntimeContract", "run_runtime_preflight"]
