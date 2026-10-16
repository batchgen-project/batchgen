"""The KDA state pool's layer map must give every logical layer its own row.

``_normalize_gpu_layer_mapping`` validates range and non-emptiness, never
injectivity, and a map that sends two logical KDA layers to one physical row
makes those layers share all four recurrent/conv states. Nothing downstream
can tell that apart from a correct run, so the manager reports the resolved
map once at construction and logs an error on the collision. These tests pin
that report: an injective map is announced and nothing else, a colliding map
names the shared rows, and ``-1`` (an unmapped logical layer) is excluded from
the duplicate search instead of colliding with itself.

Pure CPU. The method under test only reads the config and the resolved map, so
it runs unbound against a stub rather than by building a real state pool.
``batchgen.kv_cache`` pulls in a Triton kernel module on import, so -- as in
``tests/test_fused_rmsnorm_rope_q_tiling_policy.py`` -- Triton is stubbed, but
ONLY when it is genuinely absent, so CI keeps importing the real one.
"""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import pytest


def _import_manager():
    """``KDAStateGPUManager``, stubbing Triton only on a CPU-only box."""
    try:
        from batchgen.kv_cache.kda_state_gpu_manager import KDAStateGPUManager

        return KDAStateGPUManager
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda fn: fn
    triton_stub.autotune = lambda *args, **kwargs: (lambda fn: fn)
    triton_stub.Config = lambda *args, **kwargs: None
    triton_stub.cdiv = lambda a, b: -(-a // b)
    language_stub = types.ModuleType("triton.language")
    language_stub.constexpr = object
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub
    from batchgen.kv_cache.kda_state_gpu_manager import KDAStateGPUManager

    return KDAStateGPUManager


_MANAGER = _import_manager()


def _report(mapping, num_kda_layers, caplog):
    """Run the construction-time report on a stub and return its records."""
    stub = SimpleNamespace(
        _logical_to_physical_layer=mapping,
        config=SimpleNamespace(num_kda_layers=num_kda_layers),
    )
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _MANAGER._log_layer_mapping_isolation(stub)
    return caplog.records


def test_identity_map_is_announced_without_an_error(caplog):
    records = _report(None, 6, caplog)
    assert [record.levelno for record in records] == [logging.INFO]
    assert "identity layer map over 6 KDA layers" in records[0].getMessage()


@pytest.mark.parametrize(
    "mapping",
    [(0, 1, 2, 3), (3, 2, 1, 0), (0,)],
)
def test_injective_map_reports_no_duplicate_rows(mapping, caplog):
    records = _report(mapping, len(mapping), caplog)
    assert [record.levelno for record in records] == [logging.INFO]
    message = records[0].getMessage()
    assert "0 duplicate rows" in message
    assert f"{len(mapping)} mapped" in message


def test_collision_is_logged_as_an_error_naming_the_shared_rows(caplog):
    records = _report((0, 1, 1, 0, 2), 5, caplog)
    assert [record.levelno for record in records] == [logging.INFO, logging.ERROR]
    assert "2 duplicate rows" in records[0].getMessage()
    error = records[1].getMessage()
    assert "layer map is NOT injective" in error
    # Both shared rows are named, ascending; row 2 is claimed once and stays out.
    assert "physical rows [0, 1] are" in error


def test_unmapped_layers_are_excluded_from_the_duplicate_search(caplog):
    """``-1`` means the logical layer owns no row, so two of them do not
    collide: the pool never indexes a negative row."""
    records = _report((-1, 0, -1, 1), 4, caplog)
    assert [record.levelno for record in records] == [logging.INFO]
    message = records[0].getMessage()
    assert "2 mapped" in message
    assert "0 duplicate rows" in message
