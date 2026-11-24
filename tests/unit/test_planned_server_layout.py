"""The server layout is resolved by the family's planner before the model
manager is built, checked then, and the built manager must match it."""

import ast
import pathlib
from types import SimpleNamespace

import pytest

from batchgen.parallel_declaration import (
    GROUP_WORLD,
    LayoutSupport,
    ModuleSupport,
    ParallelDeclarationError,
    UnsupportedLayoutError,
    check_built_layout_is_planned,
    check_planned_server_layout,
    replicated_prefill_world_ep_decode,
    server_layout_from_declaration,
)
from batchgen.parallel_declaration import (
    planned_resident_experts,
    replicated_prefill_world_ep_decode_layout,
)
from tests.worker_method_source import load_worker_methods

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "batchgen" / "models"


def _ep(offloading=False, ratio=0.0, local=None):
    return SimpleNamespace(
        enable_offloading=offloading,
        offloading_ratio=ratio,
        num_local_expert_per_layer=local,
    )


def _render(layout):
    return (
        f"{layout.prefill.render()} | {layout.decode.render()}"
    )


@pytest.mark.parametrize(
    "ep, world, expected",
    [
        (_ep(), 8, (False, 32)),
        (_ep(offloading=True, ratio=0.5), 8, (True, 16)),
        (_ep(offloading=True, ratio=0.5), 16, (False, 16)),  # multi-node: resident
        (_ep(local=20), 8, (False, 20)),
        (_ep(local=0), 4, (False, 64)),
    ],
)
def test_the_resident_expert_rule(ep, world, expected):
    assert planned_resident_experts(ep, world, 256) == expected


def test_the_common_layout_is_resident_or_partial_in_decode():
    full = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=True)
    part = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=False)
    assert _render(full) == (
        "attention=dp:streamed moe=dp:streamed | attention=dp:resident moe=ep8:resident"
    )
    assert _render(part) == (
        "attention=dp:streamed moe=dp:streamed | attention=dp:resident moe=ep8:partial"
    )


def test_glm5_layout_follows_the_expert_split():
    from batchgen.models.glm.glm5.planner import glm5_server_layout

    resident = glm5_server_layout(SimpleNamespace(EP_Config=_ep()), 8)
    partial = glm5_server_layout(
        SimpleNamespace(EP_Config=_ep(offloading=True, ratio=0.5)), 8
    )
    assert resident.decode.moe.render() == "ep8:resident"
    assert partial.decode.moe.render() == "ep8:partial"
    assert resident.prefill.moe.render() == "dp:streamed"


def test_gpt_oss_layout_streams_prefill_experts_only_with_offloading():
    from batchgen.models.openai.gpt_oss_120b.planner import (
        gpt_oss_local_expert_split,
        gpt_oss_server_layout,
    )

    plain = gpt_oss_server_layout(SimpleNamespace(EP_Config=_ep()), 8, 128)
    offload = gpt_oss_server_layout(
        SimpleNamespace(EP_Config=_ep(offloading=True, ratio=0.5)), 8, 128
    )
    assert _render(plain) == (
        "attention=dp:streamed moe=dp:resident | attention=dp:resident moe=ep8:resident"
    )
    assert _render(offload) == (
        "attention=dp:streamed moe=dp:streamed | attention=dp:resident moe=ep8:partial"
    )
    # A single-GPU deployment keeps every expert resident.
    assert gpt_oss_local_expert_split(_ep(offloading=True, ratio=0.5), 1, 128) == (
        False, 128, 128,
    )


def test_deepseek_v4_flash_streams_both_phases():
    from batchgen.models.deepseek.deepseekv4_flash.set_basic_config import (
        deepseekv4_flash_server_layout,
    )

    assert _render(deepseekv4_flash_server_layout(8)) == (
        "attention=dp:streamed moe=dp:streamed | attention=dp:streamed moe=ep8:streamed"
    )


def _kimi_config(*, group=1, stream_all=False, distributed=False, mode=None, shard_latent=False):
    return SimpleNamespace(
        Basic_Config=SimpleNamespace(
            attention_group_size=group,
            stream_all_modules=stream_all,
            distributed_weight_sharded=distributed,
            prefill_moe_mode=mode,
        ),
        Module_Batching_Config=SimpleNamespace(k3_shard_latent_projections=shard_latent),
        EP_Config=SimpleNamespace(),
    )


@pytest.mark.parametrize(
    "config, world, expected",
    [
        (
            _kimi_config(group=1, stream_all=True),
            8,
            "attention=dp:streamed moe=dp:streamed | attention=dp:resident moe=ep8:resident",
        ),
        (
            _kimi_config(group=8),
            16,
            "attention=tp8:resident moe=dp:streamed | attention=tp8:resident moe=ep16:resident",
        ),
        (
            _kimi_config(group=8, shard_latent=True),
            16,
            "attention=tp8:resident moe=ep8:streamed | attention=tp8:resident moe=ep16:resident",
        ),
        (
            _kimi_config(group=8, distributed=True),
            32,
            "attention=tp8:resident moe=ep8:streamed | attention=tp8:resident moe=ep32:resident",
        ),
        (
            _kimi_config(group=8, mode="resident_ep"),
            16,
            "attention=tp8:resident moe=ep16:resident | attention=tp8:resident moe=ep16:resident",
        ),
    ],
)
def test_kimi_layout_is_resolved_from_the_planned_config(config, world, expected):
    from batchgen.models.moonshotai.kimi_linear.planner import kimi_linear_server_layout

    assert _render(kimi_linear_server_layout(config, world, is_k3=True)) == expected


@pytest.mark.parametrize(
    "config, message",
    [
        (_kimi_config(group=8, shard_latent=True, mode="streamed"), "unsharded LatentMoE"),
        (_kimi_config(group=8, distributed=True, mode="streamed"), "require prefill MoE mode"),
        (_kimi_config(group=1, mode="resident_ep"), "resident TP"),
        (_kimi_config(group=4, mode="streamed_sp8"), "resident TP8"),
        (_kimi_config(group=8, mode="bogus"), "must be 'streamed'"),
    ],
)
def test_kimi_refuses_a_mode_the_topology_cannot_run_before_any_model(config, message):
    from batchgen.models.moonshotai.kimi_linear.planner import kimi_linear_server_layout

    with pytest.raises(ValueError, match=message):
        kimi_linear_server_layout(config, 16, is_k3=True)


def test_kimi_non_k3_refuses_the_k3_modes():
    from batchgen.models.moonshotai.kimi_linear.planner import kimi_linear_server_layout

    with pytest.raises(ValueError, match="only for Kimi-K3"):
        kimi_linear_server_layout(
            _kimi_config(group=8, mode="resident_ep"), 16, is_k3=False
        )


class _Manager:
    SUPPORTED_LAYOUTS = replicated_prefill_world_ep_decode()


def test_a_planned_layout_is_checked_before_the_manager_is_built():
    planned = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=True)
    check_planned_server_layout(_Manager, planned, 8)
    with pytest.raises(ParallelDeclarationError, match="over 8 ranks but the server runs 16"):
        check_planned_server_layout(_Manager, planned, 16)
    with pytest.raises(ParallelDeclarationError, match="must be a ServerLayout"):
        check_planned_server_layout(_Manager, "tp8", 8)


def test_a_planned_layout_outside_the_table_is_refused_by_name():
    class Narrow:
        SUPPORTED_LAYOUTS = {
            "prefill": (
                LayoutSupport(
                    ModuleSupport("dp", ("streamed",)),
                    ModuleSupport("ep", ("streamed",), GROUP_WORLD),
                ),
            ),
            "decode": replicated_prefill_world_ep_decode()["decode"],
        }

    planned = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=True)
    with pytest.raises(UnsupportedLayoutError, match="not implemented for Narrow"):
        check_planned_server_layout(Narrow, planned, 8)


def test_a_manager_built_for_another_layout_is_refused_with_both_named():
    planned = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=True)
    other = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=False)
    check_built_layout_is_planned("M", planned, planned)
    with pytest.raises(ParallelDeclarationError) as error:
        check_built_layout_is_planned("M", planned, other)
    assert "planned" in str(error.value) and "ep8:partial" in str(error.value)
    with pytest.raises(ParallelDeclarationError, match="built no layout"):
        check_built_layout_is_planned("M", planned, None)


def _worker(planned, declared):
    return SimpleNamespace(
        world_size=8,
        engine_config=SimpleNamespace(
            Basic_Config=SimpleNamespace(server_layout=planned)
        ),
        parallel_manager=SimpleNamespace(
            parallel_declaration=lambda graph_state=None: declared.declaration()
        ),
        _server_layout=None,
    )


def test_the_worker_takes_the_planned_layout_before_the_manager_exists():
    methods = load_worker_methods(
        ["_check_planned_server_layout", "_resolve_server_layout"], {}
    )
    planned = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=True)
    worker = _worker(planned, planned)
    del worker.parallel_manager  # nothing is built yet
    methods["_check_planned_server_layout"](worker, _Manager)
    assert worker._server_layout is planned


def test_the_worker_refuses_a_manager_built_for_another_layout():
    methods = load_worker_methods(
        ["_check_planned_server_layout", "_resolve_server_layout"], {}
    )
    planned = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=True)
    other = replicated_prefill_world_ep_decode_layout(8, decode_moe_fully_resident=False)
    worker = _worker(planned, other)
    methods["_check_planned_server_layout"](worker, _Manager)
    with pytest.raises(ParallelDeclarationError, match="another layout than the planned"):
        methods["_resolve_server_layout"](worker)
    worker = _worker(planned, planned)
    methods["_check_planned_server_layout"](worker, _Manager)
    methods["_resolve_server_layout"](worker)
    assert worker._server_layout == server_layout_from_declaration(
        planned.declaration(), 8
    )


def _assigns_server_layout(path: pathlib.Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "server_layout":
                    return True
    return False


def test_every_family_with_an_implemented_table_plans_its_layout():
    """A manager that declares SUPPORTED_LAYOUTS belongs to a family whose
    initializer resolves the layout before the manager is built."""
    families = sorted(
        path.parent
        for path in MODELS.rglob("Parallel_Strategy_Manager.py")
        if "SUPPORTED_LAYOUTS" in path.read_text()
    )
    assert len(families) == 7
    missing = [
        str(family.relative_to(MODELS))
        for family in families
        if not any(
            _assigns_server_layout(path) for path in family.glob("*initializer*.py")
        )
    ]
    assert missing == []


@pytest.mark.parametrize(
    "family, constant",
    [
        ("glm/glm5", "GLM5_NUM_ROUTED_EXPERTS"),
        ("openai/gpt_oss_120b", "GPT_OSS_NUM_ROUTED_EXPERTS"),
        ("moonshotai/kimi_k25", "KIMI_K25_NUM_ROUTED_EXPERTS"),
        ("minimax/minimax_m25", "MINIMAX_M25_NUM_ROUTED_EXPERTS"),
        ("deepseek/deepseekv3", "DEEPSEEKV3_NUM_ROUTED_EXPERTS"),
    ],
)
def test_the_planner_and_the_manager_count_the_same_routed_experts(family, constant):
    import re

    def literal(path, name):
        match = re.search(rf"^\s*{name}\s*=\s*(\d+)", path.read_text(), re.M)
        assert match, (path, name)
        return int(match.group(1))

    planned = literal(MODELS / family / "planner.py", constant)
    built = literal(MODELS / family / "Parallel_Strategy_Manager.py", "NUM_TOTAL_EXPERTS")
    assert planned == built


def test_the_kimi_manager_and_planner_name_the_same_prefill_modes():
    def modes(path):
        node = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "PREFILL_MOE_MODES" for t in node.targets)
        )
        return ast.literal_eval(node.value)

    family = MODELS / "moonshotai" / "kimi_linear"
    assert modes(family / "planner.py") == modes(family / "Parallel_Strategy_Manager.py")
