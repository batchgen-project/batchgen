"""Unit tests for decode/forward_step.py (Phase 2.8.2d)."""

from __future__ import annotations

from batchgen.worker.decode.forward_step import forward_decode_step
from tests.unit.worker.fakes import FakeLegacyBackend


class TestForwardDecodeStepWrapper:
    def test_forwards_kwargs_to_adapter(self) -> None:
        legacy = FakeLegacyBackend()
        gpu = object()
        new_tokens_in = object()
        out = forward_decode_step(
            legacy,
            batch=[0, 1],
            new_tokens=new_tokens_in,
            gpu_manager=gpu,
            page_table_verified=True,
            local_iteration=3,
        )
        # Default fake behaviour: echo input tokens out.
        assert out is new_tokens_in
        call = next(c for c in legacy.calls if c[0] == "forward_decode_step")
        assert call[2] == {
            "batch": [0, 1],
            "new_tokens": new_tokens_in,
            "gpu_manager": gpu,
            "page_table_verified": True,
            "local_iteration": 3,
        }

    def test_uses_injected_output(self) -> None:
        """Tests that drive multiple iterations rely on the fake
        returning a preset tensor from each call."""
        legacy = FakeLegacyBackend()
        sentinel = object()
        legacy._forward_step_output = sentinel  # type: ignore[attr-defined]
        out = forward_decode_step(
            legacy,
            batch=[],
            new_tokens=None,
            gpu_manager=None,
            page_table_verified=False,
            local_iteration=0,
        )
        assert out is sentinel
