"""Every planning entry must normalize before executing any planning code."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V4_PATH = "batchgen/models/deepseek/deepseekv4_flash/deepseekv4_flash_initializer.py"
V4_ENTRY = "_set_batching_and_buffer_config"
EXPECTED = {
    ("batchgen/planner/base_planner.py", "BasePlanner", "generate_config"):
        ("config.Basic_Config", "model_type", "self.FORCED_DECODE_MODE"),
    ("batchgen/models/minimax/minimax_m25/planner.py", "MiniMaxM25Planner", "generate_config"):
        ("engine_config.Basic_Config", "model_type"),
    (V4_PATH, "DeepSeekV4FlashInitializer", V4_ENTRY):
        ("self.engine_config.Basic_Config", "self.model_config.model_type"),
}


def test_every_planning_entry_normalizes_as_its_first_statement():
    entries = {}
    call_sites = []
    for path in sorted((ROOT / "batchgen").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text())
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "apply_normalized_attn_mode"):
                call_sites.append((rel, node.lineno))
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "generate_config" and not (rel == V4_PATH and node.name == V4_ENTRY):
                continue
            parent = parents[node]
            key = (rel, parent.name if isinstance(parent, ast.ClassDef) else "<module>", node.name)
            assert key not in entries, f"Duplicate planning entry: {key}"
            first = node.body[0]
            assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), key
            call = first.value
            assert isinstance(call.func, ast.Name) and call.func.id == "apply_normalized_attn_mode", key
            assert not call.keywords, key
            entries[key] = tuple(ast.unparse(arg) for arg in call.args)
    # This is an exact inventory, not an exemption list: every generate_config
    # above is checked, including definitions added by future planners.
    assert entries == EXPECTED
    assert len(call_sites) == 3


def test_initializer_planning_calls_pass_the_parsed_model_type():
    calls = []
    for path in (ROOT / "batchgen").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "generate_config"):
                assert [ast.unparse(arg) for arg in node.args] == [
                    "self.engine_config", "self.model_config.model_type",
                ], str(path)
                assert not node.keywords
                calls.append(path.relative_to(ROOT).as_posix())
    assert len(calls) == 6
