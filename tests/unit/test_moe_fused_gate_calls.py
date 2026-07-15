import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _moe_fused_gate_calls(relative_path: str) -> list[ast.Call]:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "moe_fused_gate":
                calls.append(node)
    return calls


def _assert_moe_fused_gate_wrapper_signature(relative_path: str) -> None:
    calls = _moe_fused_gate_calls(relative_path)
    assert calls, f"expected at least one moe_fused_gate call in {relative_path}"
    for call in calls:
        positional = [ast.unparse(arg) for arg in call.args]
        keywords = {keyword.arg for keyword in call.keywords}

        assert len(positional) == 5
        assert not any("n_routed_experts" in arg for arg in positional)
        assert "routed_scaling_factor" in keywords


def test_deepseek_moe_fused_gate_uses_python_wrapper_signature():
    _assert_moe_fused_gate_wrapper_signature(
        "batchgen/models/deepseek/deepseekv3/modeling_deepseek_v3.py"
    )


def test_kimi_asset_moe_fused_gate_uses_python_wrapper_signature():
    _assert_moe_fused_gate_wrapper_signature(
        "batchgen/models/moonshotai/kimi_k25/assets/modeling_deepseek.py"
    )
