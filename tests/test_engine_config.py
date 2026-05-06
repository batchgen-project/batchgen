import importlib.util
from pathlib import Path


def _load_config_module():
    config_path = (
        Path(__file__).resolve().parents[1] / "batchgen" / "config" / "config.py"
    )
    spec = importlib.util.spec_from_file_location("_batchgen_config_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_config_module = _load_config_module()
BasicConfig = _config_module.BasicConfig


def test_basic_config_prints_max_prompt_length_not_padding_length():
    config = BasicConfig(max_prompt_length=3233, padding_length=8192)

    rendered = str(config)

    assert "max_prompt_length: 3233" in rendered
    assert "padding_length" not in rendered


def test_basic_config_getter_accepts_deprecated_padding_length_alias():
    config = BasicConfig(padding_length=4096)

    assert config.get_max_prompt_length() == 4096


def test_basic_config_setter_keeps_deprecated_alias_in_sync():
    config = BasicConfig(padding_length=8192)
    config.set_max_prompt_length(3233)

    assert config.get_max_prompt_length() == 3233
    assert config.padding_length == 3233
