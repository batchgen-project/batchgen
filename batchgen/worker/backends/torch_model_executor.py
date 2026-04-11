"""TorchModelExecutorBackend — wraps the existing model forward path.

**Stub**: in production the model forward pass is orchestrated by the
legacy ``BatchGenWorker`` via dozens of sites (attention masks, KV
page-table binding, sampling). The Protocol's ``forward_prefill`` /
``forward_decode`` are a conceptual wrapper that the orchestrator
calls once per prefill-round / decode-iteration.

The adapter delegates to the underlying ``BatchGenWorker`` through
two free-function callbacks so the orchestrator doesn't need to know
anything about the existing worker's method names.
"""

from __future__ import annotations

from typing import Any, Callable


class TorchModelExecutorBackend:
    """Production adapter for :class:`ModelExecutorBackend`.

    Constructed with two callbacks the entry point prepares at startup:

      - ``prefill_fn(batch_dict) -> Any`` — wraps main's
        ``BatchGenWorker.prefill`` / ``prefill_prepacked`` path.
      - ``decode_fn(batch_dict) -> Any`` — wraps main's
        ``BatchGenWorker.decoding_continuous`` single-step path.

    The orchestrator currently does not read the returned value, so
    the callbacks may return ``None`` (useful when the existing
    worker methods return ``None`` too).
    """

    def __init__(
        self,
        prefill_fn: Callable[[dict[str, Any]], Any],
        decode_fn: Callable[[dict[str, Any]], Any],
    ) -> None:
        self._prefill_fn = prefill_fn
        self._decode_fn = decode_fn

    def forward_prefill(self, batch: dict[str, Any]) -> Any:
        return self._prefill_fn(batch)

    def forward_decode(self, batch: dict[str, Any]) -> Any:
        return self._decode_fn(batch)


__all__ = ["TorchModelExecutorBackend"]
