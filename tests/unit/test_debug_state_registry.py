"""The server-level debug state: registry self-check, validation, generations."""
import json

import pytest

from batchgen import debug_state as ds
from batchgen.config.model_registry import DEBUG_KEY_REGISTRY as REGISTRY

EMPTY = REGISTRY.empty


def test_every_key_declares_owner_phase_tier_and_summary():
    assert len(REGISTRY.keys) == 35
    for name, key in REGISTRY.keys.items():
        assert key.name == name
        assert key.owner in ("core", "kimi_linear", "glm5"), name
        assert key.phase in ds.PHASES, name
        assert key.tier in ds.TIERS, name
        assert key.summary and callable(key.parse), name


def test_the_rewarm_tier_is_exactly_the_six_keys_baked_into_graphs_or_instances():
    rewarm = {name for name, key in REGISTRY.keys.items() if key.tier == ds.TIER_REWARM}
    assert rewarm == {
        "k3_prefill_profile", "kimi_decode_graph_mode", "glm5_moe_tilem_avg",
        "glm52_recv_count_hist", "glm5_whole_model_graph_probe_layers",
        "glm5_criticality_probe",
    }


def test_retired_keys_are_not_registered_and_are_refused_with_their_replacement():
    assert not set(REGISTRY.retired) & set(REGISTRY.keys)
    with pytest.raises(ds.DebugStateError, match="--prefix-rank-assignment-mode"):
        EMPTY.changed(set={"prefix_rank_assignment_mode": "global_idx"})
    with pytest.raises(ds.DebugStateError, match="glm5_moe_mode is retired"):
        EMPTY.changed(clear=["glm5_moe_mode"])


def test_an_unknown_key_is_refused_by_name_on_set_clear_and_read():
    with pytest.raises(ds.DebugStateError, match="unknown debug key 'glm5_grpah_path_log'"):
        EMPTY.changed(set={"glm5_grpah_path_log": True})
    with pytest.raises(ds.DebugStateError, match="unknown debug key 'nope'"):
        EMPTY.changed(clear=["nope"])
    with pytest.raises(ds.DebugStateError, match="unknown debug key 'nope'"):
        EMPTY.get("nope")


@pytest.mark.parametrize("name,value,stored", [
    ("glm5_graph_path_log", "on", True),
    ("glm5_graph_path_log", 0, False),
    ("kimi_decode_graph_mode", " Eager ", "eager"),
    ("decode_profile_steps", "8", 8),
    ("decode_profile_ranks", [0, 3], (0, 3)),
    ("prefix_pool_dump_kernel_args", True, "sync"),
    ("prefix_pool_dump_kernel_args", "nosync", "nosync"),
    ("glm5_whole_model_graph_compare_atol", "0.5", 0.5),
    ("glm5_whole_model_graph_probe_layers", "0, 3,7", (0, 3, 7)),
    ("glm5_whole_model_graph_probe_layers", "ALL", "all"),
])
def test_values_are_normalised(name, value, stored):
    assert EMPTY.changed(set={name: value}).get(name) == stored


@pytest.mark.parametrize("name,value", [
    ("glm5_graph_path_log", "maybe"),
    ("glm5_graph_path_log", 2),
    ("kimi_decode_graph_mode", "fast"),
    ("decode_profile_steps", 0),
    ("decode_profile_steps", True),
    ("decode_profile_ranks", []),
    ("glm5_whole_model_graph_compare_atol", float("nan")),
    ("glm5_whole_model_graph_compare_rtol", -1),
    ("weight_pass_end_delay", {"rank": 0, "seconds": 61}),
    ("weight_pass_end_delay", {"rank": 0}),
    ("glm5_criticality_probe", {}),
    ("prefix_pool_dump_dir", ""),
])
def test_a_refused_value_names_the_key(name, value):
    with pytest.raises(ds.DebugStateError, match=f"debug key {name} "):
        EMPTY.changed(set={name: value})


def test_a_change_is_a_new_generation_and_leaves_the_old_state_untouched():
    one = EMPTY.changed(set={"glm5_graph_path_log": True, "triton_strict": True})
    two = one.changed(clear=["triton_strict"], set={"k3_prefill_profile": True})
    assert (EMPTY.generation, one.generation, two.generation) == (0, 1, 2)
    assert not EMPTY and dict(one.values) == {"glm5_graph_path_log": True, "triton_strict": True}
    assert dict(two.values) == {"glm5_graph_path_log": True, "k3_prefill_profile": True}
    with pytest.raises(TypeError):
        one.values["x"] = 1


def test_a_failed_change_changes_nothing():
    one = EMPTY.changed(set={"triton_strict": True})
    with pytest.raises(ds.DebugStateError):
        one.changed(set={"glm5_graph_path_log": True, "kimi_decode_graph_mode": "fast"})
    assert dict(one.values) == {"triton_strict": True} and one.generation == 1


def test_rewarm_keys_changed_lists_only_rewarm_keys_that_differ():
    one = EMPTY.changed(set={"glm5_graph_path_log": True})
    two = one.changed(set={"glm52_recv_count_hist": True, "kimi_decode_graph_mode": "eager"})
    assert EMPTY.rewarm_keys_changed(one) == ()
    assert one.rewarm_keys_changed(two) == ("glm52_recv_count_hist", "kimi_decode_graph_mode")
    assert two.rewarm_keys_changed(two) == ()


def test_the_wire_form_round_trips_and_is_plain_json():
    state = EMPTY.changed(set={
        "weight_pass_end_delay": {"rank": 1, "seconds": 2},
        "decode_profile_ranks": [0, 1],
        "glm5_criticality_probe": {"sites": ["after_router"]},
    })
    wire = json.loads(json.dumps(state.to_wire()))
    back = REGISTRY.from_wire(wire)
    assert back.generation == 1 and back.to_wire() == state.to_wire()
    assert REGISTRY.from_wire(None) is EMPTY


def test_the_log_line_is_stable_and_marks_an_empty_state():
    assert ds.format_log_line(EMPTY, node=0) == "[DEBUG_STATE] generation=0 node=0 keys=none"
    state = EMPTY.changed(set={"triton_strict": True, "kimi_decode_graph_mode": "eager"})
    assert ds.format_log_line(state, node=1) == (
        "[DEBUG_STATE] generation=1 node=1 keys=kimi_decode_graph_mode='eager',triton_strict=True")


def test_the_request_field_refusal_names_the_admin_endpoint():
    assert "/v1/admin/debug-state" in ds.request_field_refusal()


def test_core_declares_no_model_key_and_each_family_declares_only_its_own():
    assert {key.owner for key in ds.CORE_KEYS} == {"core"}
    from batchgen.models.glm.glm5 import debug_keys as glm5
    from batchgen.models.moonshotai.kimi_linear import debug_keys as kimi
    assert {key.owner for key in glm5.DEBUG_KEYS} == {"glm5"}
    assert {key.owner for key in kimi.DEBUG_KEYS} == {"kimi_linear"}


def test_a_key_declared_twice_or_also_retired_is_refused_when_the_registry_is_built():
    one = ds.key("x", "core", "both", ds.parse_bool, "x")
    with pytest.raises(ValueError, match="declared twice"):
        ds.Registry((one, one))
    with pytest.raises(ValueError, match="both declared and retired"):
        ds.Registry((one,), retired={"x": "gone"})


def test_a_renamed_key_is_refused_with_its_new_name():
    with pytest.raises(ds.DebugStateError, match="renamed decode_profile_steps"):
        EMPTY.changed(set={"k3_decode_profile_steps": 4})
