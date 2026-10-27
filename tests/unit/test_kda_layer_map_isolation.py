"""The KDA state pool's layer map must give every logical layer its own row.

``_normalize_gpu_layer_mapping`` validates range and non-emptiness, never
injectivity, and a map that sends two logical KDA layers to one physical row
makes those layers share all four recurrent/conv states. Nothing downstream can
tell that apart from a correct run - the states simply overwrite each other -
so the map is checked once, at construction, and a non-injective one is REFUSED
as the configuration error it is. Logging it and continuing (the earlier
behaviour) left the state-sharing run to proceed on a line nobody reads in a
serving log.

The healthy case is one INFO line of counts; the resolved map, which is
hundreds of entries per rank on a real model, goes to DEBUG.

These tests pin that contract twice: unbound against a stub, for the cases
worth enumerating, and through a REAL ``KDAStateGPUManager`` construction on
CPU, because the stub cannot show that ``__init__`` actually reaches the check
before it allocates anything.

Pure CPU. ``batchgen.kv_cache`` pulls in a Triton kernel module on import, so
-- as in ``tests/test_fused_rmsnorm_rope_q_tiling_policy.py`` -- Triton is
stubbed, but ONLY when it is genuinely absent, so CI keeps importing the real
one.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _import_manager():
    """``KDAStateGPUManager`` and its config, stubbing Triton on a CPU-only box."""
    try:
        from batchgen.kv_cache.kda_state_gpu_manager import (
            KDAStateGPUConfig,
            KDAStateGPUManager,
        )

        return KDAStateGPUManager, KDAStateGPUConfig
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
    extra_stub = types.ModuleType("triton.language.extra")
    extra_stub.libdevice = types.ModuleType("triton.language.extra.libdevice")
    language_stub.extra = extra_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub
    sys.modules["triton.language.extra"] = extra_stub
    sys.modules["triton.language.extra.libdevice"] = extra_stub.libdevice
    from batchgen.kv_cache.kda_state_gpu_manager import (
        KDAStateGPUConfig,
        KDAStateGPUManager,
    )

    return KDAStateGPUManager, KDAStateGPUConfig


_MANAGER, _CONFIG = _import_manager()


def _check(mapping, num_kda_layers, caplog):
    """Run the construction-time check on a stub and return its records."""
    stub = SimpleNamespace(
        _logical_to_physical_layer=mapping,
        config=SimpleNamespace(num_kda_layers=num_kda_layers),
    )
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _MANAGER._validate_layer_mapping_isolation(stub)
    return caplog.records


# --------------------------------------------------------------------------- #
#  The check itself
# --------------------------------------------------------------------------- #
def test_identity_map_is_announced_in_one_line(caplog):
    records = _check(None, 6, caplog)
    assert [record.levelno for record in records] == [logging.INFO]
    assert "identity layer map over 6 KDA layers" in records[0].getMessage()


@pytest.mark.parametrize("mapping", [(0, 1, 2, 3), (3, 2, 1, 0), (0,)])
def test_an_injective_map_reports_counts_and_keeps_the_map_out_of_the_line(
    mapping, caplog
):
    records = _check(mapping, len(mapping), caplog)

    assert [record.levelno for record in records] == [logging.INFO]
    message = records[0].getMessage()
    assert f"over {len(mapping)} KDA layers" in message
    assert f"{len(mapping)} mapped to their own row" in message
    assert "0 unmapped" in message
    # The resolved map is DEBUG: one INFO line per rank, not one map per rank.
    assert str(list(mapping)) not in message


def test_the_resolved_map_is_available_at_debug(caplog):
    stub = SimpleNamespace(
        _logical_to_physical_layer=(0, 1, 2),
        config=SimpleNamespace(num_kda_layers=3),
    )
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        _MANAGER._validate_layer_mapping_isolation(stub)

    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug) == 1
    assert "logical->physical [0, 1, 2]" in debug[0].getMessage()


def test_a_collision_is_refused_and_names_the_shared_rows(caplog):
    """Two logical layers on one row share all four states; nothing
    downstream can distinguish that from a correct run, so it never runs."""

    with pytest.raises(ValueError) as excinfo:
        _check((0, 1, 1, 0, 2), 5, caplog)

    message = str(excinfo.value)
    assert "layer map is NOT injective" in message
    # Both shared rows are named, ascending; row 2 is claimed once and stays out.
    assert "physical rows [0, 1] are" in message
    # The map IS in the exception, where it is the actionable detail.
    assert "[0, 1, 1, 0, 2]" in message
    # Nothing was announced as healthy on the way out.
    assert [r for r in caplog.records if r.levelno == logging.INFO] == []


def test_unmapped_layers_are_excluded_from_the_duplicate_search(caplog):
    """``-1`` means the logical layer owns no row, so two of them do not
    collide: the pool never indexes a negative row."""
    records = _check((-1, 0, -1, 1), 4, caplog)

    assert [record.levelno for record in records] == [logging.INFO]
    message = records[0].getMessage()
    assert "2 mapped to their own row" in message
    assert "2 unmapped" in message


# --------------------------------------------------------------------------- #
#  The construction seam: __init__ really runs the check, before allocating
# --------------------------------------------------------------------------- #
def _config(mapping, *, num_kda_layers=4):
    return _CONFIG(
        num_kda_layers=num_kda_layers,
        num_state_items=2,
        num_heads=1,
        head_dim=8,
        conv_width=4,
        logical_to_physical_layer=mapping,
    )


def test_constructing_a_manager_emits_the_layer_map_report(caplog):
    """The real ``__init__``, not a transcription of it."""

    caplog.clear()
    with caplog.at_level(logging.INFO):
        manager = _MANAGER(config=_config((0, 1, 2, 3)), device="cpu")

    assert manager._logical_to_physical_layer == (0, 1, 2, 3)
    lines = [
        record.getMessage() for record in caplog.records
        if record.getMessage().startswith("[KDA_STATE] layer map resolved")
    ]
    assert len(lines) == 1, caplog.records
    assert "4 mapped to their own row" in lines[0] and "0 unmapped" in lines[0]


def test_constructing_a_manager_with_no_mapping_announces_the_identity_map(
    caplog,
):
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _MANAGER(config=_config(None), device="cpu")

    assert any(
        "identity layer map over 4 KDA layers" in record.getMessage()
        for record in caplog.records
    )


def test_constructing_a_manager_on_a_shared_layer_map_raises():
    """The configuration error is refused before a state row exists."""

    with pytest.raises(ValueError, match="NOT injective"):
        _MANAGER(config=_config((0, 1, 1, 3)), device="cpu")


def test_the_refusal_precedes_the_runtime_state(caplog):
    """A refused manager must not be left half built.

    ``_reset_runtime_state`` runs after the check, so a manager that raised
    never acquired the free list a caller could go on to allocate from.
    """

    with pytest.raises(ValueError, match="NOT injective"):
        _MANAGER(config=_config((2, 2, 2, 2)), device="cpu")

    healthy = _MANAGER(config=_config((0, 1, 2, 3)), device="cpu")
    # The check sits between the mapping resolution and _reset_runtime_state,
    # so the healthy manager still reaches its uninitialized runtime state.
    assert healthy._is_initialized is False
    assert healthy._sequence_state_items == {}


def test_the_construction_seam_is_a_refusal_and_not_a_log_line():
    """Moved here from the epilogue-trace suite, which has no business
    asserting this module's source: the check is this file's subject."""

    source = _ROOT.joinpath(
        "batchgen", "kv_cache", "kda_state_gpu_manager.py"
    ).read_text()

    assert "def _validate_layer_mapping_isolation(self)" in source
    assert "self._validate_layer_mapping_isolation()" in source
    body = source.split("def _validate_layer_mapping_isolation")[1]
    body = body.split("\n    def ")[0]
    assert "is NOT injective" in body
    assert "physical.count(row) > 1" in body
    assert "raise ValueError" in body
    assert "logging.error" not in body
