"""TorchTokenizerBackend — wraps the HuggingFace tokenizer used in production.

Main instantiates a tokenizer via the model loader and keeps it on
``self.tokenizer``. The adapter exposes just the three methods the
handlers need:

  - ``encode(text) -> list[int]``
  - ``decode(ids) -> str``
  - ``eos_token_ids: set[int]`` — MUST be plural per conventions.md.
    If the underlying tokenizer only defines ``eos_token_id``
    (singular), the adapter wraps it in a 1-element set.
"""

from __future__ import annotations

from typing import Any


class TorchTokenizerBackend:
    """Production adapter for :class:`TokenizerBackend`."""

    def __init__(self, tokenizer: Any) -> None:
        self._t = tokenizer
        # Plural EOS convention: honor eos_token_ids if present, else
        # wrap the singular eos_token_id in a set.
        eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if eos_ids is None:
            singular = getattr(tokenizer, "eos_token_id", None)
            if singular is None:
                eos_ids = set()
            else:
                eos_ids = {int(singular)}
        self.eos_token_ids: set[int] = set(eos_ids)

    def encode(self, text: str) -> list[int]:
        # HF tokenizers return tensors by default — force a plain list.
        ids = self._t.encode(text, add_special_tokens=False)
        if hasattr(ids, "tolist"):
            return list(ids.tolist())
        return list(ids)

    def decode(self, ids: list[int]) -> str:
        return self._t.decode(ids, skip_special_tokens=True)


__all__ = ["TorchTokenizerBackend"]
