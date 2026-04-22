from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ROOT / "batchgen/models/glm/glm5/wrappers.py"
FLASHMLA_BACKEND = ROOT / "batchgen/attention/mla/flashmla_backend.py"


def test_glm5_dsa_v2_remains_opt_in():
    source = WRAPPERS.read_text()
    assert (
        '_GLM5_USE_DSA_V2 = os.environ.get("BATCHGEN_GLM5_USE_DSA_V2", "0") == "1"'
        in source
    )


def test_glm5_dense_mla_uses_kimi_path_by_default():
    source = WRAPPERS.read_text()
    assert (
        '_use_kimi_mla = _os.environ.get("BATCHGEN_GLM5_USE_KIMI_MLA", "1") == "1"'
        in source
    )


def test_glm5_dense_decode_defaults_to_shared_pagekv_backend():
    source = WRAPPERS.read_text()
    assert (
        '_GLM5_USE_SHARED_PAGEKV_DENSE = ('
        in source
    )
    assert (
        'os.environ.get("BATCHGEN_GLM5_USE_SHARED_PAGEKV_DENSE", "1") == "1"'
        in source
    )


def test_glm5_shared_pagekv_backend_passes_kv_lora_rank_to_flashmla():
    source = FLASHMLA_BACKEND.read_text()
    assert "def mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv(" in source
    assert "cache_seqlens,\n\t\t\tself.kv_lora_rank," in source
