from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ROOT / "batchgen/models/glm/glm5/wrappers.py"


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
