"""The legacy offline entry modules stay deleted.

``batchgen/entrypoint.py`` and ``batchgen/batchgen_.py`` went with the legacy
offline runner (#539); ``batchgen/launch_server.py`` (a pre-rename argparse
stub whose server class was never implemented) and the empty
``batchgen/client_optimized.py`` follow here. The one way to launch BatchGen
is the HTTP server (``batchgen/launch_http_server.py`` ->
``batchgen.server.http_server.launch_server``); requests go through the batch
API. A reappearing stub would silently re-grow a second entry path, so the
absence is pinned by file and by import spec, without importing batchgen.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEGACY_MODULES = (
    "entrypoint",
    "batchgen_",
    "launch_server",
    "client_optimized",
)


def test_legacy_entry_files_are_absent():
    present = [
        name for name in LEGACY_MODULES
        if (ROOT / "batchgen" / f"{name}.py").exists()
    ]
    assert not present, present


def test_legacy_entry_modules_are_not_importable():
    importable = [
        name for name in LEGACY_MODULES
        if importlib.util.find_spec(f"batchgen.{name}") is not None
    ]
    assert not importable, importable


def test_the_live_launcher_exists():
    # The replacement path this deletion leans on: the HTTP server launcher.
    assert (ROOT / "batchgen" / "launch_http_server.py").exists()
