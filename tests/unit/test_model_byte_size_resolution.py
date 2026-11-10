"""The host weight envelope names the model ``--model`` names, or fails loud.

``get_model_byte_size`` used to fall back to a case-insensitive substring
match over ``MODEL_BYTE_SIZES`` and then to a 700 GiB default, so an unlisted
model (``zai-org/GLM-5.4-FP8``, the Kimi-K3 family) sized the hugepage pool,
the parameter-server region and the host-memory plan with a number that
describes another model.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


@pytest.fixture(scope="module")
def pu():
    # By file path: batchgen.server's package init imports the GPU stack.
    spec = importlib.util.spec_from_file_location(
        "_byte_size_process_utils_for_test",
        ROOT / "batchgen" / "server" / "process_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Served identifiers and the envelope the previous lookup returned for each
# (captured from it before the substring and default fallbacks were removed);
# every one resolved by exact match or by model_identity, and still must.
UNCHANGED = [
    ("openai/gpt-oss-120b", 70 * GIB),
    ("/models/gpt-oss-120b", 70 * GIB),
    ("deepseek-ai/DeepSeek-R1", 675 * GIB),
    ("deepseek-ai/deepseek-r1", 675 * GIB),
    ("deepseek-ai/DeepSeek-V3", 675 * GIB),
    ("deepseek-ai/deepseek-v3", 675 * GIB),
    ("deepseek-ai/DeepSeek-V3-0324", 675 * GIB),
    ("deepseek-ai/DeepSeek-V4-Flash", 180 * GIB),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", 180 * GIB),
    ("deepseek-ai/deepseek-v4-flash", 180 * GIB),
    ("deepseek-ai/DeepSeek-V4-Pro", 700 * GIB),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", 700 * GIB),
    ("deepseek-ai/deepseek-v4-pro", 700 * GIB),
    ("deepseek-ai/DeepSeek-V2", 472 * GIB),
    ("deepseek-ai/DeepSeek-V2-Lite", 32 * GIB),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 32 * GIB),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", 96 * GIB),
    ("mistralai/Mixtral-8x22B-Instruct-v0.1", 286 * GIB),
    ("moonshotai/Kimi-K2.5", 650 * GIB),
    ("moonshotai/Kimi-K2.6", 650 * GIB),
    ("moonshotai/kimi-k2.5", 650 * GIB),
    ("moonshotai/kimi-k2.6", 650 * GIB),
    ("path/to/Kimi-K2.5/", 650 * GIB),
    ("MiniMaxAI/MiniMax-M2.5", 250 * GIB),
    ("minimaxai/minimax-m2.5", 250 * GIB),
    ("/models/MiniMax-M2.5", 250 * GIB),
    ("zai-org/GLM-5", 1400 * GIB),
    ("zai-org/GLM-5-FP8", 760 * GIB),
    ("zai-org/GLM-5.1", 1400 * GIB),
    ("zai-org/GLM-5.1-FP8", 760 * GIB),
    ("zai-org/GLM-5.2", 1400 * GIB),
    ("zai-org/GLM-5.2-FP8", 760 * GIB),
    ("zai-org/GLM-5.3", 1400 * GIB),
    ("zai-org/GLM-5.3-FP8", 760 * GIB),
    ("zai-org/glm-5.3-fp8", 760 * GIB),
    ("/data/zai-org/GLM-5.3-FP8", 760 * GIB),
    ("/staged/GLM-5.2", 1400 * GIB),
]


@pytest.mark.parametrize("model, expected", UNCHANGED)
def test_served_models_keep_their_envelope(pu, model, expected):
    assert pu.get_model_byte_size(model) == expected


# Released Kimi-K3: metadata.total_size plus one 4 KiB unit per shard (96).
K3_BYTES = 1_560_860_324_864 + 96 * 4096
K31_BYTES = 1_561_217_377_280 + 96 * 4096


@pytest.mark.parametrize(
    "model, expected",
    [
        ("moonshotai/Kimi-K3", K3_BYTES),
        ("moonshotai/kimi-k3", K3_BYTES),
        ("moonshotai/Kimi-K3-Instruct", K3_BYTES),
        ("/shared/models/Kimi-K3", K3_BYTES),
        ("moonshotai/Kimi-K3.1-Pro", K31_BYTES),
        ("/shared/models/Kimi-K3.1-Pro", K31_BYTES),
        ("moonshotai/Kimi-Linear-48B-A3B-Instruct", 110 * GIB),
        ("moonshotai/kimi-linear-48b-a3b-instruct", 110 * GIB),
        ("moonshotai/Kimi-Linear-48B-A3B", 110 * GIB),
        ("moonshotai/Kimi-Linear-48B-A3B-Base", 110 * GIB),
        ("/shared/models/Kimi-Linear-48B-A3B-Instruct", 110 * GIB),
    ],
)
def test_kimi_linear_family_resolves_to_its_own_envelope(pu, model, expected):
    # These fell through to the 700 GiB default before they had entries.
    assert pu.get_model_byte_size(model) == expected


def test_kimi_linear_entry_matches_its_parameter_server_reservation(pu):
    path = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "kimi_parameter_server.py"
    tree = ast.parse(path.read_text())
    node = next(
        stmt for stmt in tree.body
        if isinstance(stmt, ast.Assign)
        and any(getattr(t, "id", None) == "_KIMI_LINEAR_48B_BYTE_SIZE" for t in stmt.targets)
    )
    reserved = eval(compile(ast.Expression(node.value), str(path), "eval"), {})
    assert pu.MODEL_BYTE_SIZES["moonshotai/Kimi-Linear-48B-A3B-Instruct"] == reserved


@pytest.mark.parametrize(
    "model",
    [
        "moonshotai/Kimi-K3.2",        # unlisted release: never its neighbour
        "zai-org/GLM-5.4-FP8",         # was the BF16 GLM-5 entry by substring
        "deepseek-ai/DeepSeek-V3.2",   # was the V3 entry by substring
        "deepseek-ai/DeepSeek-V2.5",   # was the V2 entry by substring
        "openai/gpt_oss_120b",         # was the 700 GiB default
        "/data/converted",             # a directory that names no release
        "",
    ],
)
def test_a_model_without_an_envelope_fails_loud(pu, model):
    with pytest.raises(pu.UnknownModelByteSizeError, match="MODEL_BYTE_SIZES"):
        pu.get_model_byte_size(model)


def test_the_default_envelope_is_gone(pu):
    assert not hasattr(pu, "DEFAULT_MODEL_BYTE_SIZE")
    assert issubclass(pu.UnknownModelByteSizeError, ValueError)
