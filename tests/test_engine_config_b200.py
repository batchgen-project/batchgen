"""Regression tests for the Blackwell (B200) DeepSeek-R1 engine config.

These run without a GPU or torch import: they only parse the JSON config files
shipped in ``configurations/`` and assert the B200 config is well-formed and
selects the ``blackwell`` architecture.
"""

import json
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configurations" / "DeepSeek-R1"
_B200 = _CONFIG_DIR / "engine_config_B200_8.json"
_H20 = _CONFIG_DIR / "engine_config_H20_8.json"


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_b200_config_exists_and_parses():
    assert _B200.is_file(), "engine_config_B200_8.json is missing"
    _load(_B200)  # raises if invalid JSON


def test_b200_config_selects_blackwell_arch():
    config = _load(_B200)
    assert config["Basic_Config"]["gpu_arch"] == "blackwell"


def test_b200_config_matches_h20_structure():
    """The B200 config is derived from H20_8; it must keep the same shape so
    the engine's config loader treats it identically (only ``gpu_arch`` differs)."""
    b200 = _load(_B200)
    h20 = _load(_H20)

    assert b200.keys() == h20.keys()
    for section in h20:
        assert b200[section].keys() == h20[section].keys(), section

    # The only intended difference is the architecture selector.
    assert h20["Basic_Config"]["gpu_arch"] == "hopper"
    assert b200["Basic_Config"]["gpu_arch"] == "blackwell"
