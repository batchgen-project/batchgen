from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
import torch

from batchgen.ckpt_converter.ckpt_converter import ckpt_converter
from batchgen.config.tokenizer_registry import load_tokenizer


FIXTURES_DIR = Path(__file__).parent / "fixtures"
TOKENIZER_FIXTURE = FIXTURES_DIR / "tokenizer.json"
KIMI_FIXTURE_DIR = FIXTURES_DIR / "kimi_k25"
STANDARD_TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer_config.json")
GPT_OSS_TOKENIZER_ASSETS = ("tokenizer_config.json", "chat_template.jinja")
KIMI_TOKENIZER_ASSETS = ("tiktoken.model", "tokenizer_config.json", "chat_template.jinja")
MODEL_IDENTIFIERS = [
    "deepseek-ai/DeepSeek-R1",
    "THUDM/GLM-5",
    "MiniMaxAI/MiniMax-M2.5",
    "openai/gpt-oss-120b",
    "moonshotai/Kimi-K2.5",
]


def _write_checkpoint(source_dir: Path) -> Path:
    ckpt_path = source_dir / "model.pt"
    torch.save({"weight": torch.tensor([1.0, 2.0, 3.0])}, ckpt_path)
    return ckpt_path


def _asset_files_for_model(model_identifier: str | None = None) -> tuple[str, ...]:
    if model_identifier and "Kimi-K2.5" in model_identifier:
        return KIMI_TOKENIZER_ASSETS
    if model_identifier == "openai/gpt-oss-120b":
        return GPT_OSS_TOKENIZER_ASSETS
    return STANDARD_TOKENIZER_ASSETS


def _write_standard_tokenizer_assets(tokenizer_dir: Path) -> None:
    tokenizer_dir.mkdir(exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_bytes(TOKENIZER_FIXTURE.read_bytes())
    (tokenizer_dir / "tokenizer_config.json").write_bytes(
        b'{"chat_template":"{{ messages }}"}\n'
    )


def _write_gpt_oss_tokenizer_assets(tokenizer_dir: Path) -> None:
    tokenizer_dir.mkdir(exist_ok=True)
    (tokenizer_dir / "tokenizer_config.json").write_text(
        '{"bos_token":"<|startoftext|>","eos_token":"<|return|>","pad_token":"<|endoftext|>"}\n',
        encoding="utf-8",
    )
    (tokenizer_dir / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}<|start|>assistant{% endif %}",
        encoding="utf-8",
    )


def _write_kimi_tokenizer_assets(tokenizer_dir: Path) -> None:
    tokenizer_dir.mkdir(exist_ok=True)
    for file_name in KIMI_TOKENIZER_ASSETS:
        shutil.copyfile(KIMI_FIXTURE_DIR / file_name, tokenizer_dir / file_name)


def _write_tokenizer_assets(tokenizer_dir: Path, model_identifier: str | None = None) -> None:
    if model_identifier and "Kimi-K2.5" in model_identifier:
        _write_kimi_tokenizer_assets(tokenizer_dir)
        return
    if model_identifier == "openai/gpt-oss-120b":
        _write_gpt_oss_tokenizer_assets(tokenizer_dir)
        return
    _write_standard_tokenizer_assets(tokenizer_dir)


@pytest.mark.parametrize("model_identifier", MODEL_IDENTIFIERS)
def test_load_tokenizer_from_converted_checkpoint_dir(
    tmp_path: Path,
    model_identifier: str,
) -> None:
    converted_ckpt_dir = tmp_path / "converted_ckpt"
    _write_tokenizer_assets(converted_ckpt_dir, model_identifier)

    tokenizer = load_tokenizer(model_identifier, converted_ckpt_dir)
    token_ids = tokenizer.encode("hello world")

    assert token_ids
    if model_identifier != "openai/gpt-oss-120b":
        assert tokenizer.decode(token_ids) == "hello world"

    if model_identifier == "moonshotai/Kimi-K2.5":
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert rendered == "hello[GEN]"
    elif model_identifier == "openai/gpt-oss-120b":
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert rendered == "hello<|start|>assistant"


@pytest.mark.parametrize(
    "model_identifier",
    ["deepseek-ai/DeepSeek-R1", "moonshotai/Kimi-K2.5"],
)
def test_converter_copies_tokenizer_assets_byte_identically(
    tmp_path: Path,
    model_identifier: str,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "converted_ckpt"
    source_dir.mkdir()
    _write_checkpoint(source_dir)

    _write_tokenizer_assets(source_dir, model_identifier)

    converter = ckpt_converter()
    result_dir = Path(converter.convert_model_directory(str(source_dir), str(output_dir), model_identifier=model_identifier))

    assert result_dir == output_dir
    for file_name in _asset_files_for_model(model_identifier):
        assert (result_dir / file_name).read_bytes() == (source_dir / file_name).read_bytes()


@pytest.mark.parametrize(
    "model_identifier, expected_missing_assets",
    [
        ("deepseek-ai/DeepSeek-R1", ["tokenizer.json", "tokenizer_config.json"]),
        ("THUDM/GLM-5", ["tokenizer.json", "tokenizer_config.json"]),
        ("MiniMaxAI/MiniMax-M2.5", ["tokenizer.json", "tokenizer_config.json"]),
        ("openai/gpt-oss-120b", ["tokenizer_config.json", "chat_template.jinja"]),
        ("moonshotai/Kimi-K2.5", ["chat_template.jinja", "tiktoken.model", "tokenizer_config.json"]),
    ],
)
def test_converter_warns_when_required_tokenizer_assets_missing(
    tmp_path: Path,
    caplog,
    model_identifier: str,
    expected_missing_assets: list[str],
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "converted_ckpt"
    source_dir.mkdir()
    _write_checkpoint(source_dir)

    converter = ckpt_converter()

    with caplog.at_level(logging.WARNING):
        result_dir = Path(
            converter.convert_model_directory(
                str(source_dir),
                str(output_dir),
                model_identifier=model_identifier,
            )
        )

    assert result_dir == output_dir
    assert str(source_dir) in caplog.text
    for asset_name in expected_missing_assets:
        assert asset_name in caplog.text


@pytest.mark.parametrize(
    "model_identifier, required_assets_to_remove",
    [
        ("deepseek-ai/DeepSeek-R1", ["tokenizer.json", "tokenizer_config.json"]),
        ("THUDM/GLM-5", ["tokenizer.json", "tokenizer_config.json"]),
        ("MiniMaxAI/MiniMax-M2.5", ["tokenizer.json", "tokenizer_config.json"]),
        ("openai/gpt-oss-120b", ["tokenizer_config.json", "chat_template.jinja"]),
        ("moonshotai/Kimi-K2.5", ["tiktoken.model", "tokenizer_config.json", "chat_template.jinja"]),
    ],
)
def test_validation_requires_model_specific_tokenizer_assets(
    tmp_path: Path,
    model_identifier: str,
    required_assets_to_remove: list[str],
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "converted_ckpt"
    source_dir.mkdir()
    _write_checkpoint(source_dir)
    _write_tokenizer_assets(source_dir, model_identifier)

    converter = ckpt_converter()
    converter.convert_model_directory(
        str(source_dir),
        str(output_dir),
        model_identifier=model_identifier,
    )

    for file_name in required_assets_to_remove:
        (output_dir / file_name).unlink()

    is_valid, error_msg = converter.validate_converted_directory(
        str(source_dir),
        str(output_dir),
        model_identifier=model_identifier,
    )
    assert not is_valid
    assert "Required tokenizer asset" in error_msg


@pytest.mark.parametrize(
    "model_identifier",
    ["deepseek-ai/DeepSeek-R1", "moonshotai/Kimi-K2.5"],
)
def test_validation_allows_tokenizer_assets_but_rejects_dirty_output_dir(
    tmp_path: Path,
    model_identifier: str,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "converted_ckpt"
    source_dir.mkdir()
    _write_checkpoint(source_dir)
    _write_tokenizer_assets(source_dir, model_identifier)

    converter = ckpt_converter()
    converter.convert_model_directory(
        str(source_dir), str(output_dir), model_identifier=model_identifier
    )

    is_valid, error_msg = converter.validate_converted_directory(
        str(source_dir), str(output_dir), model_identifier=model_identifier
    )
    assert is_valid
    assert error_msg is None

    (output_dir / "stale.json").write_text("{}\n")
    is_valid, error_msg = converter.validate_converted_directory(
        str(source_dir), str(output_dir), model_identifier=model_identifier
    )
    assert not is_valid
    assert "Unexpected files or directories" in error_msg
    assert "stale.json" in error_msg


@pytest.mark.parametrize(
    "model_identifier",
    ["deepseek-ai/DeepSeek-R1", "moonshotai/Kimi-K2.5"],
)
def test_existing_converted_checkpoint_backfills_missing_tokenizer_assets(
    tmp_path: Path,
    model_identifier: str,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "converted_ckpt"
    source_dir.mkdir()
    ckpt_path = _write_checkpoint(source_dir)
    _write_tokenizer_assets(source_dir, model_identifier)

    converter = ckpt_converter()
    converter.convert(str(ckpt_path), str(output_dir))
    for file_name in _asset_files_for_model(model_identifier):
        (output_dir / file_name).unlink()

    result_dir = Path(
        converter.convert_model_directory(
            str(source_dir), str(output_dir), model_identifier=model_identifier
        )
    )

    assert result_dir == output_dir
    for file_name in _asset_files_for_model(model_identifier):
        assert (output_dir / file_name).read_bytes() == (source_dir / file_name).read_bytes()


def test_gpt_oss_tokenizer_requires_chat_template_file(tmp_path: Path) -> None:
    converted_ckpt_dir = tmp_path / "converted_ckpt"
    converted_ckpt_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="chat_template.jinja"):
        load_tokenizer("openai/gpt-oss-120b", converted_ckpt_dir)
