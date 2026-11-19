"""Every file a registered tokenizer loads at import time ships in the package.

A source checkout imports modules from directories that package discovery
skips (no ``__init__.py``, e.g. vendored upstream trees imported as implicit
namespace packages) and reads data files next to the code. Neither reaches an
installed package unless ``setup.py``'s package/package-data rules list it, so
a gap only surfaces on an install, as an ImportError or a missing asset.

The test compares two file sets, each computed in a fresh interpreter:

* built: the files ``setup.py build_py`` writes, taken from setuptools'
  ``build_py.get_outputs()`` after parsing ``setup.py``. That is the list
  ``build_py.run()`` copies (Python modules of the discovered packages plus the
  package-data / ``MANIFEST.in`` matches), evaluated without copying the
  package tree and without the copy step that replaces the in-tree
  ``core``/``external``/``op_builder`` symlinks.
* required: every file under ``batchgen/`` that importing each module in the
  tokenizer registry loads, i.e. the ``__file__`` of each imported
  ``batchgen`` module plus each file opened under ``batchgen/``.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "batchgen"

# The regression this test was written for: the DeepSeek-V4 tokenizer imports
# the vendored encoder from a directory that is not a package.
DSV4_ENCODER = "batchgen/models/deepseek/deepseekv4_flash/assets/encoding/encoding_dsv4.py"


def _write_built_files(out_dir: str) -> None:
    """Child-process body: write the build_py output set as JSON."""
    import setuptools  # noqa: F401 - selects setuptools' distutils
    from distutils.core import run_setup

    build_lib = os.path.join(out_dir, "build")
    dist = run_setup(
        "setup.py",
        script_args=["build_py", "--build-lib", build_lib],
        stop_after="commandline",
    )
    # include_package_data runs egg_info to read MANIFEST.in; keep its output
    # out of the source tree.
    dist.get_command_obj("egg_info").egg_base = out_dir
    build_py = dist.get_command_obj("build_py")
    build_py.ensure_finalized()
    built = sorted(
        Path(os.path.relpath(path, build_lib)).as_posix()
        for path in build_py.get_outputs(include_bytecode=False)
    )
    Path(out_dir, "built.json").write_text(json.dumps(built))


def _write_tokenizer_import_files(out_dir: str) -> None:
    """Child-process body: write the files tokenizer imports load as JSON."""
    sys.path.insert(0, str(REPO_ROOT))
    package_prefix = str(PACKAGE_DIR) + os.sep
    opened: set[str] = set()

    def _record_open(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(args[0]))
            if path.startswith(package_prefix):
                opened.add(path)

    sys.addaudithook(_record_open)

    import batchgen
    from batchgen.config.tokenizer_registry import _TOKENIZER_MODULES

    assert os.path.abspath(batchgen.__file__).startswith(package_prefix), (
        f"batchgen resolved to {batchgen.__file__}, not {PACKAGE_DIR}"
    )
    failures = {}
    for module_path in dict.fromkeys(_TOKENIZER_MODULES.values()):
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - reported by the parent test
            failures[module_path] = f"{type(exc).__name__}: {exc}"
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and os.path.abspath(path).startswith(package_prefix):
            opened.add(os.path.abspath(path))
    required = sorted(
        Path(os.path.relpath(path, REPO_ROOT)).as_posix()
        for path in opened
        if "__pycache__" not in Path(path).parts
    )
    Path(out_dir, "required.json").write_text(
        json.dumps(
            {
                "modules": list(dict.fromkeys(_TOKENIZER_MODULES.values())),
                "failures": failures,
                "required": required,
            }
        )
    )


def _run_child(mode: str, out_dir: Path) -> None:
    env = dict(os.environ, BUILD_OPS="0")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, str(out_dir)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{mode} child failed (exit {result.returncode}):\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    )


def test_tokenizer_import_files_are_in_build_output(tmp_path):
    _run_child("built", tmp_path)
    _run_child("required", tmp_path)
    built = set(json.loads((tmp_path / "built.json").read_text()))
    trace = json.loads((tmp_path / "required.json").read_text())

    assert not trace["failures"], (
        "registered tokenizer modules failed to import from the source tree: "
        f"{trace['failures']}"
    )
    required = set(trace["required"])
    for module_path in trace["modules"]:
        module_file = module_path.replace(".", "/") + ".py"
        assert module_file in required, f"import trace missed {module_file}"
        assert module_file in built, f"build output lacks {module_file}"
    assert DSV4_ENCODER in required, f"import trace missed {DSV4_ENCODER}"

    missing = sorted(required - built)
    assert not missing, (
        "files loaded when importing registered tokenizers are absent from "
        "the setup.py build_py output (add a package or package-data rule):\n  "
        + "\n  ".join(missing)
    )


if __name__ == "__main__":
    _MODES = {
        "built": _write_built_files,
        "required": _write_tokenizer_import_files,
    }
    _MODES[sys.argv[1]](sys.argv[2])
