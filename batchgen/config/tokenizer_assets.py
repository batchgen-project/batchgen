"""Tokenizer encoding files as staged, hash-verified install assets.

``tiktoken.get_encoding("o200k_base")`` downloads the BPE ranks from a public
blob store the first time a process asks for them (or reads a per-user cache
that a new node does not have), so a GPT-OSS server on an offline node failed
inside its runtime preflight with a connection timeout. The ranks are now an
install asset:

* the installer stages each file once per environment under
  ``<sys.prefix>/share/batchgen/tokenizer-assets`` with
  ``python -m batchgen.config.tokenizer_assets stage`` (``--from FILE`` stages
  a copy obtained elsewhere, for offline nodes); staging refuses a file whose
  SHA-256 differs from the pinned digest;
* the tokenizer reads only the staged file, verifies the digest again and
  never touches the network; a missing or altered file raises
  :class:`TokenizerAssetError`, which names the staging command. The runtime
  preflight constructs the tokenizer, so a launch fails there.

Importable without torch.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

STAGE_COMMAND = "python -m batchgen.config.tokenizer_assets stage"


class TokenizerAssetError(RuntimeError):
    """A tokenizer encoding file is not staged, or not the pinned file."""


@dataclass(frozen=True)
class EncodingAsset:
    """One tiktoken BPE ranks file and the encoding metadata it pairs with."""

    name: str
    filename: str
    url: str
    sha256: str
    pat_str: str
    special_tokens: Dict[str, int] = field(default_factory=dict)


# Upstream definition: tiktoken_ext.openai_public.o200k_base (pinned against
# it by tests/unit/test_tokenizer_assets.py).
O200K_BASE = EncodingAsset(
    name="o200k_base",
    filename="o200k_base.tiktoken",
    url="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    sha256="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    pat_str="|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    ),
    special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
)

ASSETS: Dict[str, EncodingAsset] = {asset.name: asset for asset in (O200K_BASE,)}


def assets_dir() -> Path:
    """Where the installer stages tokenizer assets for this environment."""

    return Path(sys.prefix) / "share" / "batchgen" / "tokenizer-assets"


def asset_path(asset: EncodingAsset, root: Optional[Path] = None) -> Path:
    return (assets_dir() if root is None else Path(root)) / asset.filename


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_verified(asset: EncodingAsset, root: Optional[Path] = None) -> bytes:
    """The staged file's bytes; raise when it is missing or not the pinned file."""

    path = asset_path(asset, root)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise TokenizerAssetError(
            f"tokenizer encoding {asset.name!r} is not staged at {path}; the "
            f"runtime never downloads it. Run `{STAGE_COMMAND}` in this "
            f"environment (on an offline node: `{STAGE_COMMAND} --from "
            f"<copy of {asset.filename}>`)"
        ) from None
    digest = _sha256(data)
    if digest != asset.sha256:
        raise TokenizerAssetError(
            f"tokenizer encoding {path} has SHA-256 {digest}, expected "
            f"{asset.sha256}; re-stage it with `{STAGE_COMMAND}`"
        )
    return data


def parse_tiktoken_ranks(data: bytes, origin: str) -> Dict[bytes, int]:
    """tiktoken's BPE file format: one ``<base64 token> <rank>`` per line."""

    ranks: Dict[bytes, int] = {}
    for line in data.splitlines():
        if not line:
            continue
        try:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
        except Exception as exc:  # noqa: BLE001 - name the offending line
            raise TokenizerAssetError(
                f"cannot parse line {line[:80]!r} of {origin}"
            ) from exc
    return ranks


def load_tiktoken_ranks(
    asset: EncodingAsset, root: Optional[Path] = None
) -> Dict[bytes, int]:
    """Mergeable ranks of a staged, verified encoding. No network I/O."""

    return parse_tiktoken_ranks(read_verified(asset, root), str(asset_path(asset, root)))


def _download(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - pinned https URL
        return response.read()


def stage(
    asset: EncodingAsset,
    *,
    source: Optional[Path] = None,
    root: Optional[Path] = None,
    fetch: Callable[[str], bytes] = _download,
) -> Path:
    """Install the verified file; a correct file already in place is kept."""

    target = asset_path(asset, root)
    if target.is_file() and _sha256(target.read_bytes()) == asset.sha256:
        return target
    data = Path(source).read_bytes() if source is not None else fetch(asset.url)
    digest = _sha256(data)
    if digest != asset.sha256:
        origin = source if source is not None else asset.url
        raise TokenizerAssetError(
            f"{origin} has SHA-256 {digest}, expected {asset.sha256} for "
            f"{asset.name}; refusing to stage it"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{asset.filename}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    stage_cmd = sub.add_parser("stage", help="stage the encodings, verified")
    stage_cmd.add_argument("--from", dest="source", type=Path, default=None,
                           help="stage this local copy instead of downloading "
                                "(only with one --name)")
    stage_cmd.add_argument("--name", action="append", choices=sorted(ASSETS),
                           help="encoding to stage (default: all)")
    check_cmd = sub.add_parser("check", help="verify the staged encodings")
    check_cmd.add_argument("--name", action="append", choices=sorted(ASSETS))
    args = parser.parse_args(argv)
    names = args.name or sorted(ASSETS)
    try:
        if args.command == "stage":
            if args.source is not None and len(names) != 1:
                parser.error("--from needs exactly one --name")
            for name in names:
                path = stage(ASSETS[name], source=args.source)
                print(f"staged {name}: {path}")  # noqa: hygiene - CLI output
        else:
            for name in names:
                read_verified(ASSETS[name])
                print(f"verified {name}: {asset_path(ASSETS[name])}")  # noqa: hygiene - CLI output
    except (TokenizerAssetError, OSError) as exc:
        print(f"tokenizer assets: {exc}", file=sys.stderr)  # noqa: hygiene - CLI output
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSETS",
    "EncodingAsset",
    "O200K_BASE",
    "STAGE_COMMAND",
    "TokenizerAssetError",
    "asset_path",
    "assets_dir",
    "load_tiktoken_ranks",
    "parse_tiktoken_ranks",
    "read_verified",
    "stage",
]
