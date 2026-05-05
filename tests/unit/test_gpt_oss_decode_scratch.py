import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_decode_scratch():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "batchgen"
        / "models"
        / "openai"
        / "gpt_oss_120b"
        / "decode_scratch.py"
    )
    spec = importlib.util.spec_from_file_location("decode_scratch", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_gpt_oss_model_raises_for_gpt_oss_estimator():
    decode_scratch = _load_decode_scratch()
    config = SimpleNamespace(model_type="glm5")

    with pytest.raises(RuntimeError, match="unsupported"):
        decode_scratch.estimate_gpt_oss_decode_scratch_reserve_gb(
            model_config=config,
            world_size=8,
            max_num_seq_per_rank=32,
        )


def test_gpt_oss_model_reserves_at_least_two_gb():
    decode_scratch = _load_decode_scratch()
    config = SimpleNamespace(
        model_type="gpt_oss",
        hidden_size=2880,
        intermediate_size=2880,
        num_experts_per_tok=4,
        num_local_experts=128,
        vocab_size=201088,
    )

    reserve = decode_scratch.estimate_gpt_oss_decode_scratch_reserve_gb(
        model_config=config,
        world_size=2,
        max_num_seq_per_rank=1,
    )

    assert reserve >= 2.0
