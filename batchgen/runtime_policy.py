"""Process-inherited runtime backend policy set by the server preflight."""

from __future__ import annotations

import os


def configure_runtime_policy(*, flash_backend: str, decode_backend: str) -> None:
    if flash_backend not in {"fa2", "fa3"}:
        raise ValueError(f"unsupported FlashAttention backend: {flash_backend!r}")
    if decode_backend not in {"fa3", "wgmma"}:
        raise ValueError(f"unsupported decode backend: {decode_backend!r}")
    os.environ["BATCHGEN_FLASH_ATTN_BACKEND"] = flash_backend
    os.environ["BATCHGEN_DECODE_BACKEND"] = decode_backend


def selected_flash_backend() -> str | None:
    backend = os.environ.get("BATCHGEN_FLASH_ATTN_BACKEND")
    return backend if backend in {"fa2", "fa3"} else None


__all__ = [
    "configure_runtime_policy",
    "selected_flash_backend",
]
