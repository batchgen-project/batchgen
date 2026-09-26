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
import logging
import os
import site
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from batchgen.config.model_registry import (
    _detect_model_type_from_identifier,
    load_config,
)
from batchgen.runtime_policy import configure_runtime_policy

logger = logging.getLogger(__name__)


class RuntimePreflightError(RuntimeError):
    """Raised when the selected model cannot run in the current environment."""


@dataclass(frozen=True)
class RuntimeContract:
    """Native/runtime capabilities required by one exact model type."""

    modules: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    require_deepgemm: bool = False
    flash_backend: str = "fa3"


_COMMON_EXTENSIONS = (
    "batchgen_kernels.attention._C_fused_ops",
    "batchgen_kernels.attention._C_gqa_mha_decode_bf16",
)

# These are exact model-type contracts, not family-wide fallback rules.  A new
# model type must be added here before it can enter the production server path.
_CONTRACTS: dict[str, RuntimeContract] = {
    "gpt_oss": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        require_deepgemm=True,
    ),
    "glm_moe_dsa": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        require_deepgemm=True,
    ),
    "glm_moe_dsa_5_2": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        require_deepgemm=True,
    ),
    "glm_moe_dsa_5_3": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        require_deepgemm=True,
    ),
    "deepseek_v2": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        flash_backend="fa2",
    ),
    "deepseek_v3": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        flash_backend="fa2",
    ),
    "deepseek_v4": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        flash_backend="fa2",
    ),
    "mixtral": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        flash_backend="fa2",
    ),
    "minimax_m25": RuntimeContract(
        modules=("flash_attn_interface", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        flash_backend="fa3",
    ),
    "kimi_linear": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_COMMON_EXTENSIONS,
    ),
    "kimi_k3": RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=_COMMON_EXTENSIONS,
    ),
    "kimi_k25": RuntimeContract(
        modules=("flash_attn", "libucx"),
        extensions=_COMMON_EXTENSIONS,
        flash_backend="fa2",
    ),
}


def _site_roots() -> tuple[Path, ...]:
    roots = {*site.getsitepackages(), sysconfig.get_paths()["purelib"]}
    return tuple(Path(root).resolve() for root in roots if root)


def _require_site_package(module_name: str, module: object) -> None:
    module_path = getattr(module, "__file__", None)
    if not module_path:
        raise RuntimePreflightError(
            f"{module_name} has no __file__; refusing unverifiable runtime module"
        )
    resolved = Path(module_path).resolve()
    if not any(root == resolved or root in resolved.parents for root in _site_roots()):
        raise RuntimePreflightError(
            f"{module_name} resolves outside site-packages: {resolved}; "
            "editable/source worktree shadowing is not allowed"
        )


def _import_required(module_name: str, *, site_package: bool = False) -> object:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - preserve the native import cause
        raise RuntimePreflightError(
            f"required runtime module {module_name!r} failed to import: {exc}"
        ) from exc
    if site_package:
        _require_site_package(module_name, module)
    logger.info(
        "[runtime-preflight] %s=%s",
        module_name,
        getattr(module, "__file__", "<built-in>"),
    )
    return module


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
    if not eos_ids:
        raise RuntimePreflightError(
            f"tokenizer contract for {model!r} does not define eos_token_ids"
        )
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
    if not model:
        raise RuntimePreflightError("server model is empty")

    _check_torch()
    model_type = _resolve_model_type(model)
    contract = _CONTRACTS[model_type]
    decode_backend = (
        "wgmma"
        if torch.cuda.is_available() and "H20" in torch.cuda.get_device_name()
        else "fa3"
    )
    configure_runtime_policy(
        flash_backend=contract.flash_backend,
        decode_backend=decode_backend,
        allow_fallback=bool(getattr(server_args, "allow_runtime_fallback", False)),
    )
    _check_tokenizer(model)

    for module_name in contract.modules:
        module = _import_required(
            module_name,
            site_package=module_name in {"flash_attn_interface", "flash_attn", "flash_mla"},
        )
        if module_name == "libucx":
            _check_ucx(module)

    for extension_name in contract.extensions:
        _import_required(extension_name, site_package=True)

    if decode_backend == "wgmma":
        decode_module = _import_required(
            "batchgen_kernels.attention.decode", site_package=True
        )
        if not callable(getattr(decode_module, "attention_decode_bf16", None)):
            raise RuntimePreflightError(
                "selected WGMMA decode backend has no attention_decode_bf16()"
            )

    core_engine = _import_required("batchgen.core_engine", site_package=False)
    core_path = getattr(core_engine, "__file__", "")
    if Path(core_path).suffix not in {".so", ".pyd", ".dylib"}:
        raise RuntimePreflightError(
            f"batchgen.core_engine is not an AOT native module: {core_path!r}"
        )

    if contract.require_deepgemm:
        _check_deepgemm()

    logger.info(
        "[runtime-preflight] passed model=%s model_type=%s",
        model,
        model_type,
    )
    return model_type


__all__ = ["RuntimePreflightError", "RuntimeContract", "run_runtime_preflight"]
