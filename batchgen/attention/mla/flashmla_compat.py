"""Version adaptation for the installed flash_mla package.

Two flash_mla generations exist in our environments:

- the original package (Hopper lanes): dense BF16 MLA decoding is the default
  and ``flash_mla_with_kvcache`` has no ``use_legacy`` parameter;
- the current upstream wheel (Blackwell lanes): the default kernel set is the
  sparse one, dense decoding lives on the "legacy" side (built for
  sm_90a/100a/103a/107a) and both ``get_mla_metadata`` and
  ``flash_mla_with_kvcache`` must be told ``use_legacy=True`` to reach it.

``dense_decode_kwargs()`` returns the extra keyword arguments a dense-decode
call site must pass for the installed package, resolved once per process.
"""

from __future__ import annotations

import inspect
from typing import Optional

_DENSE_KWARGS: Optional[dict] = None


def dense_decode_kwargs() -> dict:
    global _DENSE_KWARGS
    if _DENSE_KWARGS is None:
        from flash_mla import flash_mla_with_kvcache

        parameters = inspect.signature(flash_mla_with_kvcache).parameters
        _DENSE_KWARGS = {"use_legacy": True} if "use_legacy" in parameters else {}
    return _DENSE_KWARGS
