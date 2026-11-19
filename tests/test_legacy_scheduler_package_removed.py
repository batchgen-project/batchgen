"""The legacy ``batchgen/scheduler`` package stays deleted.

Its ``Scheduler`` (``scheduler.py``) went first; ``device_mem.py`` had no
user, and the one live helper, the ``/proc/meminfo`` reader in
``host_mem.py``, moved to ``batchgen/hardware/meminfo.py``. Scheduling lives
in ``batchgen/planner/`` and ``batchgen/server/batch_scheduler.py``. A
reappearing module or import would re-grow a second, dead scheduling home,
so the absence is pinned by file, by import spec and by import statement,
without importing batchgen.
"""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "batchgen"

LEGACY_MODULES = (
    "batchgen.scheduler.scheduler",
    "batchgen.scheduler.device_mem",
    "batchgen.scheduler.host_mem",
)


def _is_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _imported_modules(path: Path):
    """Absolute names of the modules one source file imports."""
    package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                name = f"{base}.{node.module}" if node.module else base
            else:
                name = node.module or ""
            yield name
            yield from (f"{name}.{alias.name}" for alias in node.names)


def test_legacy_scheduler_files_are_absent():
    assert not (PACKAGE / "scheduler.py").exists()
    present = sorted(
        str(p.relative_to(ROOT)) for p in (PACKAGE / "scheduler").glob("*.py")
    )
    assert not present, present


def test_legacy_scheduler_modules_are_not_importable():
    importable = [name for name in LEGACY_MODULES if _is_importable(name)]
    assert not importable, importable


def test_no_source_imports_the_legacy_scheduler_package():
    importers = sorted(
        str(path.relative_to(ROOT))
        for path in PACKAGE.rglob("*.py")
        if any(
            name == "batchgen.scheduler" or name.startswith("batchgen.scheduler.")
            for name in _imported_modules(path)
        )
    )
    assert not importers, importers


def test_the_meminfo_reader_has_its_generic_home():
    # The replacement the host-memory preflight leans on.
    assert (PACKAGE / "hardware" / "meminfo.py").exists()
    source = (PACKAGE / "host_memory_preflight.py").read_text(encoding="utf-8")
    assert "from batchgen.hardware.meminfo import get_physical_memory_info" in source
