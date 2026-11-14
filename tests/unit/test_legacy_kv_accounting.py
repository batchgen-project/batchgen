import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.config.legacy_kv_accounting import legacy_past_kv_byte_size


def test_deepseek_formula_uses_input_length_when_larger():
    config = SimpleNamespace(model_type="deepseek", compressed_kv_dim=7)
    # max(10 + 2 + 1, 5) * 7 = 13 * 7 = 91 bytes.
    assert legacy_past_kv_byte_size(
        config, max_input_length=10, token_idx=2, max_cache_len=5
    ) == 91


def test_deepseek_formula_uses_cache_length_when_larger():
    config = SimpleNamespace(model_type="deepseek", compressed_kv_dim=7)
    # max(3 + 1 + 1, 10) * 7 = 10 * 7 = 70 bytes.
    assert legacy_past_kv_byte_size(
        config, max_input_length=3, token_idx=1, max_cache_len=10
    ) == 70


def test_mixtral_formula_uses_input_length_when_larger():
    config = SimpleNamespace(
        model_type="mixtral", num_key_value_heads=2, head_dim=3
    )
    # max(4 + 2, 1) * 2 * 3 * 2 = 6 * 12 = 72 bytes.
    assert legacy_past_kv_byte_size(
        config, max_input_length=4, token_idx=2, max_cache_len=1
    ) == 72


def test_mixtral_formula_uses_cache_length_when_larger():
    config = SimpleNamespace(
        model_type="mixtral", num_key_value_heads=2, head_dim=3
    )
    # max(2 + 1, 10) * 2 * 3 * 2 = 10 * 12 = 120 bytes.
    assert legacy_past_kv_byte_size(
        config, max_input_length=2, token_idx=1, max_cache_len=10
    ) == 120


def test_unsupported_model_message_is_exact():
    config = SimpleNamespace(model_type="unknown-arch")
    with pytest.raises(
        ValueError, match=r"^Model architecture unknown-arch not supported yet\.$"
    ):
        legacy_past_kv_byte_size(
            config, max_input_length=1, token_idx=0, max_cache_len=1
        )


def test_worker_mode_one_delegates_byte_accounting_to_owner():
    source = Path("batchgen/batchgen_worker.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_decoding_legacy_modes"
    )
    branch = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and ast.unparse(node.test) == "strategy is DecodeStrategy.LEGACY_MICROBATCH_GPU"
    )
    producers = [
        node
        for node in ast.walk(branch)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "past_kv_byte_size"
            for target in node.targets
        )
    ]
    assert len(producers) == 1
    value = producers[0].value
    assert isinstance(value, ast.Call)
    assert ast.unparse(value.func) == "legacy_past_kv_byte_size"
    branch_source = ast.get_source_segment(source, branch)
    assert "compressed_kv_dim" not in branch_source
    assert "num_key_value_heads" not in branch_source
