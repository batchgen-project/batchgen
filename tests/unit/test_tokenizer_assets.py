"""GPT-OSS reads its o200k_base encoding from a staged, verified file and
never from the network (sub-project tokenizer-runtime-network-dependency)."""

from __future__ import annotations

import base64
import hashlib
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from batchgen.config import tokenizer_assets as ta

_TOKENS = [b"h", b"e", b"l", b"o", b" ", b"he", b"ll", b"hell", b"hello"]


def _ranks_file(path: Path) -> bytes:
    data = b"\n".join(
        base64.b64encode(token) + b" " + str(rank).encode()
        for rank, token in enumerate(_TOKENS)
    ) + b"\n"
    path.write_bytes(data)
    return data


def _synthetic_asset(tmp_path):
    source = tmp_path / "src" / "o200k_base.tiktoken"
    source.parent.mkdir()
    data = _ranks_file(source)
    asset = replace(ta.O200K_BASE, sha256=hashlib.sha256(data).hexdigest())
    return asset, source


@pytest.fixture
def no_network(monkeypatch):
    """Any socket connection is a test failure."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_pinned_asset_matches_tiktoken_upstream_definition(monkeypatch):
    """URL, digest, split regex and special tokens equal tiktoken's own
    o200k_base, captured without letting it download anything."""

    from tiktoken_ext import openai_public

    seen = {}

    def capture(url, expected_hash=None):
        seen.update(url=url, expected_hash=expected_hash)
        return {}

    monkeypatch.setattr(openai_public, "load_tiktoken_bpe", capture)
    upstream = openai_public.o200k_base()
    assert seen == {"url": ta.O200K_BASE.url, "expected_hash": ta.O200K_BASE.sha256}
    assert upstream["pat_str"] == ta.O200K_BASE.pat_str
    assert upstream["special_tokens"] == ta.O200K_BASE.special_tokens


def test_stage_from_a_local_copy_verifies_and_is_idempotent(tmp_path, no_network):
    asset, source = _synthetic_asset(tmp_path)
    root = tmp_path / "assets"
    path = ta.stage(asset, source=source, root=root)
    assert path == root / "o200k_base.tiktoken"
    assert path.read_bytes() == source.read_bytes()
    assert [p.name for p in root.iterdir()] == ["o200k_base.tiktoken"]
    # A verified file in place is kept without reading the source again.
    source.unlink()
    assert ta.stage(asset, source=source, root=root) == path


def test_stage_refuses_a_file_with_the_wrong_digest(tmp_path):
    _, source = _synthetic_asset(tmp_path)
    root = tmp_path / "assets"
    with pytest.raises(ta.TokenizerAssetError, match="refusing to stage"):
        ta.stage(ta.O200K_BASE, source=source, root=root)
    assert not root.exists() or list(root.iterdir()) == []


def test_stage_downloads_through_the_injected_fetch(tmp_path):
    asset, source = _synthetic_asset(tmp_path)
    calls = []

    def fetch(url):
        calls.append(url)
        return source.read_bytes()

    path = ta.stage(asset, root=tmp_path / "assets", fetch=fetch)
    assert calls == [ta.O200K_BASE.url]
    assert ta.load_tiktoken_ranks(asset, path.parent) == {
        token: rank for rank, token in enumerate(_TOKENS)}


def test_missing_asset_names_the_staging_command(tmp_path, no_network):
    with pytest.raises(ta.TokenizerAssetError) as info:
        ta.load_tiktoken_ranks(ta.O200K_BASE, tmp_path)
    assert ta.STAGE_COMMAND in str(info.value)
    assert "--from" in str(info.value)


def test_altered_staged_asset_is_refused(tmp_path):
    asset, source = _synthetic_asset(tmp_path)
    path = ta.stage(asset, source=source, root=tmp_path / "assets")
    path.write_bytes(path.read_bytes() + b"aGk= 99\n")
    with pytest.raises(ta.TokenizerAssetError, match="expected"):
        ta.load_tiktoken_ranks(asset, path.parent)


def test_default_location_is_inside_the_environment():
    import sys

    assert ta.assets_dir() == Path(sys.prefix) / "share" / "batchgen" / "tokenizer-assets"


def test_cli_stage_and_check(tmp_path, monkeypatch, capsys):
    asset, source = _synthetic_asset(tmp_path)
    monkeypatch.setattr(ta, "ASSETS", {"o200k_base": asset})
    monkeypatch.setattr(ta, "assets_dir", lambda: tmp_path / "assets")
    assert ta.main(["check"]) == 1
    assert ta.STAGE_COMMAND in capsys.readouterr().err
    assert ta.main(["stage", "--from", str(source)]) == 0
    assert ta.main(["check"]) == 0


# --------------------------------------------------------------------------- #
# The GPT-OSS tokenizer and the runtime preflight
# --------------------------------------------------------------------------- #


def _gpt_oss_tokenizer_module():
    from batchgen.config.tokenizer_registry import load_tokenizer  # noqa: F401
    import importlib

    return importlib.import_module("batchgen.models.openai.gpt_oss_120b.tokenizer")


def test_gpt_oss_tokenizer_without_a_staged_encoding_fails_offline(
        tmp_path, monkeypatch, no_network):
    """Before the fix this reached tiktoken.get_encoding, which downloads."""

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(ta, "assets_dir", lambda: tmp_path / "assets")
    module = _gpt_oss_tokenizer_module()
    with pytest.raises(ta.TokenizerAssetError, match="is not staged"):
        module.GPTOssTokenizer()


def test_gpt_oss_tokenizer_builds_from_the_staged_encoding(tmp_path, monkeypatch, no_network):
    asset, source = _synthetic_asset(tmp_path)
    ta.stage(asset, source=source, root=tmp_path / "assets")
    monkeypatch.setattr(ta, "assets_dir", lambda: tmp_path / "assets")
    module = _gpt_oss_tokenizer_module()
    monkeypatch.setattr(module, "O200K_BASE", asset)
    tokenizer = module.GPTOssTokenizer()
    assert tokenizer.encode("hello") == [_TOKENS.index(b"hello")]
    assert tokenizer.encode("<|return|>") == [200002]
    assert tokenizer.decode([_TOKENS.index(b"hello")]) == "hello"


def test_runtime_preflight_reports_the_staging_step(tmp_path, monkeypatch, no_network):
    import batchgen.runtime_preflight as preflight

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(ta, "assets_dir", lambda: tmp_path / "assets")
    with pytest.raises(preflight.RuntimePreflightError) as info:
        preflight._check_tokenizer("openai/gpt-oss-120b")
    assert ta.STAGE_COMMAND in str(info.value)
