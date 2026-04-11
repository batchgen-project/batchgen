"""Smoke test: validates the worker re-extract test harness end-to-end.

This file exists only to prove the jazz1 → origin → wechat_87 container →
`worker-refactor-test` conda env → pytest loop works. It is replaced by real
unit tests as M1 slices land.
"""


def test_worker_package_importable() -> None:
    import batchgen.worker  # noqa: F401


def test_harness_smoke() -> None:
    assert 1 + 1 == 2
