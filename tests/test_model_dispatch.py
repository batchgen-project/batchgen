"""M1 regression: the consolidated registry resolves every model name to the
same class the former get_initializer / get_parallel_strategy_manager if/elif
chains did.

GPU-free: asserts on the entry key, so it does not import any model package.
"""
import pytest

from batchgen.model_dispatch import resolve_model


# (input model name, expected entry key) -- one per branch of the old if/elif,
# incl. the canonical HF id the GLM-5 server launches with.
CASES = [
    ("zai-org/GLM-5.1-FP8", "glm5"),
    ("GLM-5", "glm5"),
    ("glm-5.1-fp8", "glm5"),
    ("moonshotai/Kimi-K2.5", "kimi_k25"),
    ("moonshotai/Kimi-K2.6", "kimi_k25"),
    ("kimi_k25", "kimi_k25"),
    ("deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4_flash"),
    ("deepseek-ai/DeepSeek-R1", "deepseek_v3"),
    ("deepseek-ai/DeepSeek-V3", "deepseek_v3"),
    ("openai/gpt-oss-120b", "gpt_oss"),
    ("MiniMax-M2.5", "minimax_m25"),
    ("minimax", "minimax_m25"),
]


@pytest.mark.parametrize("name,expected_key", CASES)
def test_resolve_model_key(name, expected_key):
    assert resolve_model(name).key == expected_key


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        resolve_model("not-a-real-model")


def test_every_entry_has_loaders():
    for name, _ in CASES:
        entry = resolve_model(name)
        assert entry.initializer_loader is not None
        assert entry.psm_loader is not None
