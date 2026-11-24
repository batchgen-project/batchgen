"""Implemented layouts are declared as data, and a resolved layout outside a
family's table is refused by name (batchgen.parallel_declaration)."""
import ast
import pathlib

import pytest

from batchgen.parallel_declaration import (
    GROUP_ATTENTION,
    GROUP_DIVISOR,
    GROUP_WORLD,
    LAYOUT_PHASES,
    LayoutSupport,
    ModuleParallelism,
    ModuleSupport,
    ParallelDeclaration,
    ParallelDeclarationError,
    PHASE_DECODE,
    PHASE_PREFILL,
    UnsupportedLayoutError,
    check_layout_supported,
    read_supported_layouts,
    render_supported_layouts,
    replicated_prefill_world_ep_decode,
    resolve_server_layout,
    server_layout_from_declaration,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
MANAGERS = sorted((REPO / "batchgen" / "models").glob("*/*/Parallel_Strategy_Manager.py"))


def _declaration(prefill_attn, prefill_moe, decode_attn, decode_moe):
    return ParallelDeclaration(
        prefill_attention=ModuleParallelism(*prefill_attn),
        prefill_moe=ModuleParallelism(*prefill_moe),
        decode_attention=ModuleParallelism(*decode_attn),
        decode_moe=ModuleParallelism(*decode_moe),
    )


def _kimi_table():
    module = ast.parse(
        (REPO / "batchgen/models/moonshotai/kimi_linear/Parallel_Strategy_Manager.py").read_text()
    )
    cls = next(
        n for n in module.body
        if isinstance(n, ast.ClassDef) and n.name == "KimiLinearParallelStrategyManager"
    )
    assign = next(
        n for n in cls.body
        if isinstance(n, ast.Assign) and n.targets[0].id == "SUPPORTED_LAYOUTS"
    )
    import batchgen.parallel_declaration as pd

    return eval(compile(ast.Expression(assign.value), "kimi-table", "eval"), vars(pd))


# --- the table types ------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    dict(scheme="dp", residencies=("resident",), group=GROUP_WORLD),
    dict(scheme="ep", residencies=("resident",)),
    dict(scheme="tp", residencies=(), group=GROUP_DIVISOR),
    dict(scheme="tp", residencies=("warm",), group=GROUP_DIVISOR),
    dict(scheme="ep", residencies=("resident",), group=1),
    dict(scheme="ep", residencies=("resident",), group="node"),
    dict(scheme="none", residencies=("resident",)),
])
def test_an_ill_formed_module_support_is_refused(kwargs):
    with pytest.raises(ParallelDeclarationError):
        ModuleSupport(**kwargs)


def test_the_attention_group_cannot_be_defined_by_itself():
    with pytest.raises(ParallelDeclarationError):
        LayoutSupport(
            ModuleSupport("tp", ("resident",), GROUP_ATTENTION),
            ModuleSupport("dp", ("streamed",)),
        )


@pytest.mark.parametrize("group, size, world, attention, admitted", [
    (GROUP_WORLD, 16, 16, 8, True),
    (GROUP_WORLD, 8, 16, 8, False),
    (GROUP_DIVISOR, 4, 16, 4, True),
    (GROUP_DIVISOR, 3, 16, 3, False),
    (GROUP_ATTENTION, 8, 16, 8, True),
    (GROUP_ATTENTION, 16, 16, 8, False),
    (8, 8, 16, 8, True),
    (8, 4, 16, 4, False),
])
def test_group_rules(group, size, world, attention, admitted):
    support = ModuleSupport("ep", ("resident",), group)
    module = ModuleParallelism("ep", "resident", group_size=size)
    assert support.admits(module, world, attention) is admitted


def test_scheme_and_residency_must_match():
    support = ModuleSupport("dp", ("streamed",))
    assert support.admits(ModuleParallelism("dp", "streamed"), 8, 1)
    assert not support.admits(ModuleParallelism("dp", "resident"), 8, 1)
    assert not support.admits(ModuleParallelism("tp", "streamed", group_size=2), 8, 2)


# --- today's layouts are admitted, others refused by name ------------------ #
def test_the_common_table_admits_the_replicated_prefill_world_ep_decode_layout():
    table = replicated_prefill_world_ep_decode()
    for decode_moe in ("resident", "partial"):
        layout = server_layout_from_declaration(
            _declaration(("dp", "streamed"), ("dp", "streamed"),
                         ("dp", "resident"), ("ep", decode_moe, 8)), 8,
        )
        check_layout_supported("family", table, layout)


def test_an_unimplemented_layout_is_refused_by_name_with_the_implemented_list():
    table = replicated_prefill_world_ep_decode()
    layout = server_layout_from_declaration(
        _declaration(("tp", "resident", 2), ("dp", "streamed"),
                     ("dp", "resident"), ("ep", "resident", 8)), 8,
    )
    with pytest.raises(UnsupportedLayoutError) as error:
        check_layout_supported("GLM5ParallelStrategyManager", table, layout)
    message = str(error.value)
    assert "prefill attention=tp2:resident moe=dp:streamed" in message
    assert "GLM5ParallelStrategyManager" in message
    assert "implemented prefill layouts: attention=dp:streamed moe=dp:streamed" in message


def test_the_decode_phase_is_checked_too():
    table = replicated_prefill_world_ep_decode()
    layout = server_layout_from_declaration(
        _declaration(("dp", "streamed"), ("dp", "streamed"),
                     ("dp", "resident"), ("ep", "resident", 4)), 8,
    )
    with pytest.raises(UnsupportedLayoutError, match="layout decode"):
        check_layout_supported("family", table, layout)


@pytest.mark.parametrize("declaration, world", [
    # K3 on two 8-GPU nodes: TP8 attention, streamed experts over the group.
    ((("tp", "resident", 8), ("ep", "streamed", 8), ("tp", "resident", 8), ("ep", "resident", 16)), 16),
    # K3 resident-EP prefill control.
    ((("tp", "resident", 8), ("ep", "resident", 16), ("tp", "resident", 8), ("ep", "resident", 16)), 16),
    # K3 at group size 1: everything streamed in prefill.
    ((("dp", "streamed"), ("dp", "streamed"), ("dp", "resident"), ("ep", "resident", 8)), 8),
    # The 48B: resident attention, streamed experts.
    ((("dp", "resident"), ("dp", "streamed"), ("dp", "resident"), ("ep", "resident", 8)), 8),
    ((("tp", "resident", 4), ("dp", "streamed"), ("tp", "resident", 4), ("ep", "resident", 8)), 8),
])
def test_kimi_table_admits_the_layouts_the_manager_can_declare(declaration, world):
    layout = server_layout_from_declaration(_declaration(*declaration), world)
    check_layout_supported("KimiLinearParallelStrategyManager", _kimi_table(), layout)


def test_kimi_table_refuses_streamed_group_experts_outside_the_tp8_group():
    layout = server_layout_from_declaration(
        _declaration(("tp", "resident", 4), ("ep", "streamed", 4),
                     ("tp", "resident", 4), ("ep", "resident", 16)), 16,
    )
    with pytest.raises(UnsupportedLayoutError, match="moe=ep4:streamed"):
        check_layout_supported("KimiLinearParallelStrategyManager", _kimi_table(), layout)


# --- the manager seam ------------------------------------------------------ #
class _Manager:
    SUPPORTED_LAYOUTS = replicated_prefill_world_ep_decode()

    def __init__(self, prefill_attention=("dp", "streamed")):
        self._prefill_attention = prefill_attention

    def parallel_declaration(self, decode_graph_state=None):
        return _declaration(self._prefill_attention, ("dp", "streamed"),
                            ("dp", "resident"), ("ep", "resident", 8))


def test_resolve_returns_the_layout_the_report_is_rendered_from():
    layout, declaration = resolve_server_layout(_Manager(), 8)
    assert layout.declaration(declaration.decode_graph).rendered_slots() == declaration.rendered_slots()
    assert layout.decode.moe_groups == (tuple(range(8)),)


def test_resolve_refuses_an_unimplemented_layout():
    with pytest.raises(UnsupportedLayoutError, match="_Manager"):
        resolve_server_layout(_Manager(prefill_attention=("tp", "resident", 2)), 8)


def test_a_manager_without_a_declaration_is_refused_by_name():
    from batchgen.parallel_declaration import ParallelDeclarationError

    with pytest.raises(ParallelDeclarationError, match="object declares no parallel layout"):
        resolve_server_layout(object(), 8)


@pytest.mark.parametrize("table", [
    {PHASE_PREFILL: ()},
    {PHASE_PREFILL: (), PHASE_DECODE: ()},
    {PHASE_PREFILL: ["x"], PHASE_DECODE: ["y"]},
    "prefill",
])
def test_an_ill_formed_table_is_a_hard_error(table):
    manager = type("M", (), {"SUPPORTED_LAYOUTS": table})()
    with pytest.raises(ParallelDeclarationError):
        read_supported_layouts(manager)


# --- every family declares its table --------------------------------------- #
def test_every_manager_that_declares_a_layout_declares_its_implemented_table():
    assert len(MANAGERS) >= 7
    missing = []
    for path in MANAGERS:
        tree = ast.parse(path.read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            names = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
            if "parallel_declaration" not in names:
                continue
            tables = [
                n for n in cls.body
                if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "SUPPORTED_LAYOUTS"
            ]
            if len(tables) != 1:
                missing.append(f"{path.relative_to(REPO)}::{cls.name}")
    assert not missing, f"managers without one SUPPORTED_LAYOUTS table: {missing}"


def test_the_tables_render_into_one_page():
    page = render_supported_layouts(
        {"kimi-linear": _kimi_table(), "common": replicated_prefill_world_ep_decode()}
    )
    lines = page.splitlines()
    assert lines[0] == "| family | phase | attention | MoE |"
    assert "| common | decode | dp:resident | ep<world>:resident|partial |" in lines
    assert "| kimi-linear | prefill | tp<8>:resident | ep<attn>:streamed |" in lines
    assert {line.split(" | ")[1] for line in lines[2:]} == set(LAYOUT_PHASES)


# --- launch flags: honoured or refused, never ignored ---------------------- #
def _glm_like_layout():
    return server_layout_from_declaration(
        _declaration(("dp", "streamed"), ("dp", "streamed"),
                     ("dp", "resident"), ("ep", "resident", 8)), 8,
    )


def test_a_requested_group_size_the_model_resolved_to_is_accepted():
    from batchgen.parallel_declaration import check_requested_group_sizes

    check_requested_group_sizes("M", _glm_like_layout(), {
        ("prefill", "attention"): 1, ("prefill", "moe"): 1,
        ("decode", "attention"): None, ("decode", "moe"): 8,
    })
    check_requested_group_sizes("M", _glm_like_layout(), {})


@pytest.mark.parametrize("slot, value, flag, resolved", [
    (("decode", "attention"), 2, "--decode-attention-tp 2", "decode attention=dp:resident"),
    (("prefill", "attention"), 8, "--prefill-attention-tp 8", "prefill attention=dp:streamed"),
    (("decode", "moe"), 4, "--decode-moe-ep 4", "decode moe=ep8:resident"),
    (("prefill", "moe"), 8, "--prefill-moe-ep 8", "prefill moe=dp:streamed"),
])
def test_a_requested_group_size_the_model_did_not_resolve_to_is_refused_by_name(
    slot, value, flag, resolved
):
    from batchgen.parallel_declaration import check_requested_group_sizes

    with pytest.raises(UnsupportedLayoutError) as error:
        check_requested_group_sizes("GLM5ParallelStrategyManager", _glm_like_layout(), {slot: value})
    message = str(error.value)
    assert flag in message and resolved in message
    assert "GLM5ParallelStrategyManager" in message


# --- Kimi: one attention group for both phases, taken from the flags ------- #
def _kimi_group(**kwargs):
    from batchgen.models.moonshotai.kimi_linear.planner import (
        requested_attention_group_size,
    )

    kwargs.setdefault("environ", {})
    kwargs.setdefault("prefill_tp", None)
    kwargs.setdefault("decode_tp", None)
    return requested_attention_group_size(**kwargs)


def test_kimi_group_size_comes_from_either_flag_and_defaults_to_one():
    assert _kimi_group() == 1
    assert _kimi_group(decode_tp=8) == 8
    assert _kimi_group(prefill_tp=4) == 4
    assert _kimi_group(prefill_tp=8, decode_tp=8) == 8


def test_kimi_refuses_a_different_group_per_phase_by_name():
    with pytest.raises(ValueError, match="--prefill-attention-tp 4 and --decode-attention-tp 8 differ"):
        _kimi_group(prefill_tp=4, decode_tp=8)


def test_kimi_refuses_the_retired_environment_variable_by_name():
    with pytest.raises(ValueError, match="BATCHGEN_KIMI_ATTENTION_GROUP_SIZE is retired"):
        _kimi_group(decode_tp=8, environ={"BATCHGEN_KIMI_ATTENTION_GROUP_SIZE": "8"})


def test_kimi_refuses_a_non_positive_group_size():
    with pytest.raises(ValueError, match="must be positive"):
        _kimi_group(decode_tp=0)

