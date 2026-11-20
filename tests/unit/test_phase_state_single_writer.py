# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""``phase_state`` is the only module that writes the per-phase class state.

Fails if any file under ``batchgen/`` other than
``batchgen/models/wrappers/phase_state.py`` assigns, augments, deletes or
``setattr``/``delattr``s one of the phase attributes on one of the wrapper
base classes (or a module-local alias of one). The attribute list comes from
the module itself (``PHASE_STATE_FIELDS``), so a group added there is guarded
automatically.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PHASE_STATE_PATH = REPO / "batchgen" / "models" / "wrappers" / "phase_state.py"

# The wrapper base classes (and their canonical module-level aliases) whose
# per-phase attributes the module owns. Subclass-private attributes (for
# example GLM-5 dispatch-trace state on GLM5AttnWrapper) are not phase state
# and are out of scope.
WRAPPER_BASE_NAMES = {
    "AttnWrapperBase",
    "Attn_Wrapper",
    "ExpertWrapperBase",
    "Expert_Wrapper",
    "BaseModuleWrapper",
}


def _phase_state_fields() -> frozenset:
    spec = importlib.util.spec_from_file_location(
        "_phase_state_single_writer_probe", PHASE_STATE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # dataclass processing resolves the defining module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PHASE_STATE_FIELDS


def _local_names(tree: ast.AST) -> set:
    """WRAPPER_BASE_NAMES plus aliases bound to them in this module."""
    names = set(WRAPPER_BASE_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            if node.value.id in WRAPPER_BASE_NAMES:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in WRAPPER_BASE_NAMES and alias.asname:
                    names.add(alias.asname)
    return names


def test_phase_state_is_the_single_writer():
    fields = _phase_state_fields()
    assert fields, "phase_state.PHASE_STATE_FIELDS is empty"
    offenders = []
    for path in sorted((REPO / "batchgen").rglob("*.py")):
        if path == PHASE_STATE_PATH:
            continue
        tree = ast.parse(path.read_text())
        names = _local_names(tree)
        rel = path.relative_to(REPO)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and isinstance(node.value, ast.Name)
                and node.value.id in names
                and node.attr in fields
            ):
                offenders.append(f"{rel}:{node.lineno} {ast.unparse(node)}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("setattr", "delattr")
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in names
            ):
                offenders.append(
                    f"{rel}:{node.lineno} {ast.unparse(node)[:80]}"
                )
    assert not offenders, (
        "phase-state class attributes must be written through "
        "batchgen/models/wrappers/phase_state.py only; found:\n"
        + "\n".join(offenders)
    )
