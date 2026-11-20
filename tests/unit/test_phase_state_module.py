# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Unit tests for the phase-state writer module's binding semantics."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

PHASE_STATE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "batchgen"
    / "models"
    / "wrappers"
    / "phase_state.py"
)

spec = importlib.util.spec_from_file_location(
    "_phase_state_under_test", PHASE_STATE_PATH
)
phase_state = importlib.util.module_from_spec(spec)
# dataclass processing resolves the defining module through sys.modules.
sys.modules[spec.name] = phase_state
spec.loader.exec_module(phase_state)


class _Recording(type):
    """Metaclass that records every class-attribute assignment in order."""

    def __setattr__(cls, name, value):
        cls.__dict__["_log"].append((name, value))
        super().__setattr__(name, value)


def _target():
    return _Recording("Target", (), {"_log": []})


def test_subset_assigned_in_caller_order():
    target = _target()
    phase_state.bind_step(target, position_ids="p", attention_mask="a")
    assert [name for name, _ in target._log] == [
        "position_ids",
        "attention_mask",
    ]
    target2 = _target()
    phase_state.bind_step(target2, attention_mask="a", position_ids="p")
    assert [name for name, _ in target2._log] == [
        "attention_mask",
        "position_ids",
    ]


def test_assigned_object_identity():
    target = _target()
    value = object()
    phase_state.bind_decode_entry(target, gpu_paged_kv_manager=value)
    assert target.gpu_paged_kv_manager is value


def test_unknown_field_raises_and_assigns_nothing():
    target = _target()
    with pytest.raises(TypeError):
        phase_state.bind_step(target, cur_batch=[], not_a_field=1)
    assert target._log == []


def test_field_of_another_group_rejected():
    target = _target()
    with pytest.raises(TypeError):
        phase_state.bind_step(target, kv_append_callback=None)


def test_restore_follows_mapping_iteration_order():
    target = _target()
    saved = {"max_seqlen": 7, "cur_batch": [1], "attention_mask": None}
    phase_state.restore_saved_state(target, saved)
    assert target._log == [
        ("max_seqlen", 7),
        ("cur_batch", [1]),
        ("attention_mask", None),
    ]


def test_restore_rejects_non_phase_field():
    target = _target()
    with pytest.raises(TypeError):
        phase_state.restore_saved_state(target, {"not_phase_state": 1})
    assert target._log == []


def test_groups_are_disjoint_and_cover_the_field_set():
    import dataclasses

    groups = [
        phase_state.PhaseFlag,
        phase_state.DebugState,
        phase_state.StepMetadata,
        phase_state.DecodeEntryBinding,
        phase_state.KVAppendBinding,
        phase_state.PrepackMetadata,
        phase_state.PrefillPoolBinding,
    ]
    seen = []
    for group in groups:
        seen.extend(field.name for field in dataclasses.fields(group))
    assert len(seen) == len(set(seen)), "a field appears in two groups"
    assert frozenset(seen) == phase_state.PHASE_STATE_FIELDS
