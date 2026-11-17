"""The WGMMA loader modules never call the root logging convenience API.

``logging.info(...)`` on a bare root logger runs an implicit
``logging.basicConfig()`` (WARNING, bare format). These two modules log at
import time — reached from ``import batchgen.server.server_args`` through the
GPT-OSS model's ``is_wgmma_available`` — so a root call here used to install
the root handler before the entry point configured logging, silently eating
every later INFO line of the server process (sub-project
server-logging-handlers-lost, second cause). Module loggers never
auto-configure, so the pin is: no ``logging.<level>()`` root convenience
calls in these files.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILES = (
    "batchgen/moe/fused_wgmma_expert.py",
    "batchgen/moe/fused_wgmma_grouped.py",
)

_ROOT_CONVENIENCE = {"debug", "info", "warning", "error", "critical", "exception", "log"}


def test_wgmma_modules_use_module_loggers_only():
    offenders = []
    for rel in FILES:
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logging"
                and node.func.attr in _ROOT_CONVENIENCE
            ):
                offenders.append(f"{rel}:{node.lineno}: logging.{node.func.attr}")
    assert not offenders, offenders
