"""Unit tests for decode/bind.py (Phase 2.8.2a)."""

from __future__ import annotations

from typing import Any

from batchgen.worker.decode.bind import bind_decode_context
from tests.unit.worker.fakes import FakeLegacyBackend


class TestBindDecodeContext:
    def test_forwards_all_kwargs_to_adapter(self) -> None:
        legacy = FakeLegacyBackend()
        manager = object()
        view = object()
        legacy._bind_gpu_manager = manager  # type: ignore[attr-defined]
        legacy._bind_worker_view = view     # type: ignore[attr-defined]
        past_k = object()
        past_v = object()
        scale = {"q": 0.125}

        gpu_manager, worker_view = bind_decode_context(
            legacy,
            batch=[0, 1, 2],
            past_key_states=past_k,
            past_value_states=past_v,
            scale_dict=scale,
        )

        assert gpu_manager is manager
        assert worker_view is view
        call = next(c for c in legacy.calls if c[0] == "bind_decode_context")
        assert call[2] == {
            "batch": [0, 1, 2],
            "past_key_states": past_k,
            "past_value_states": past_v,
            "scale_dict": scale,
        }

    def test_none_defaults(self) -> None:
        """Past-KV states are optional in legacy for models that don't
        use the K/V caching tensors directly; bind must accept None."""
        legacy = FakeLegacyBackend()
        gpu_manager, worker_view = bind_decode_context(
            legacy,
            batch=[0],
        )
        # Default returns None for both since the fake wasn't preset.
        assert gpu_manager is None
        assert worker_view is None
        call = next(c for c in legacy.calls if c[0] == "bind_decode_context")
        assert call[2]["past_key_states"] is None
        assert call[2]["past_value_states"] is None
        assert call[2]["scale_dict"] is None
