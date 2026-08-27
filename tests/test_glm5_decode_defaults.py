from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ROOT / "batchgen/models/glm/glm5/wrappers.py"
FLASHMLA_BACKEND = ROOT / "batchgen/attention/mla/flashmla_backend.py"


def test_retired_glm5_decode_backend_env_switches_are_absent():
    source = WRAPPERS.read_text()
    for retired in (
        "BATCHGEN_GLM5_USE_DSA_V2",
        "BATCHGEN_GLM5_USE_KIMI_MLA",
        "BATCHGEN_GLM5_USE_SHARED_PAGEKV_DENSE",
    ):
        assert retired not in source


def test_glm5_shared_pagekv_backend_passes_kv_lora_rank_to_flashmla():
    source = FLASHMLA_BACKEND.read_text()
    assert "def mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv(" in source
    assert "cache_seqlens,\n\t\t\tself.kv_lora_rank," in source
