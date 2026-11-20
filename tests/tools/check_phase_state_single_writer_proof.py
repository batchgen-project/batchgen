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

"""Mechanical before/after proof for the phase-state single-writer slice.

For every function that, at the base revision, assigned a per-phase class
attribute on a wrapper class, this extracts the ordered list of
``(target class, attribute, unparsed value expression)`` the base body
performed and the same list reconstructed from the new body's
``phase_state.bind_*`` / ``restore_saved_state`` calls, and asserts the lists
are equal. Three rules make the reconstruction mechanical:

* a ``bind_*(target, a=..., b=...)`` call expands to one entry per keyword,
  in keyword order (keyword order is assignment order);
* a base tuple assignment ``C.a, C.b = EXPR`` expands to entries whose value
  is ``(EXPR, index)``; on the new side, a tuple unpacking of the identical
  ``EXPR`` into local names followed by keywords passing those names matches
  element-wise;
* a base ``for name, value in M.items(): setattr(C, name, value)`` restore
  loop and a new ``restore_saved_state(C, M)`` call both record ``(C,
  "__restore__", M)``.

Attributes deliberately deleted by the slice (reader-less, confirmed dead)
are dropped from the base list and reported.

usage::

    python tests/tools/check_phase_state_single_writer_proof.py \
        --repo <worktree> --base <git revision>
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import subprocess
import sys

# Class names whose attribute stores are in scope, mapped to the class object
# they denote (Attn_Wrapper / Expert_Wrapper are module-level aliases).
CANON = {
    "AttnWrapperBase": "AttnWrapperBase",
    "Attn_Wrapper": "AttnWrapperBase",
    "ExpertWrapperBase": "ExpertWrapperBase",
    "Expert_Wrapper": "ExpertWrapperBase",
    "BaseModuleWrapper": "BaseModuleWrapper",
    "KimiLinearExpertWrapper": "KimiLinearExpertWrapper",
}

# Attributes the slice deleted outright (no reader of any kind was found).
DELETED_ATTRS = {
    "async_kv_load_active",
    "async_kv_load_task",
    "past_value_states",
}

# The files that carried write sites at the base revision.
FILES = [
    "batchgen/batchgen_worker.py",
    "batchgen/worker/graph_runtime.py",
    "batchgen/models/glm/glm5/Parallel_Strategy_Manager.py",
    "batchgen/models/glm/glm5/whole_model_cuda_graph_segments.py",
    "batchgen/models/moonshotai/kimi_linear/Parallel_Strategy_Manager.py",
    "batchgen/models/openai/gpt_oss_120b/Parallel_Strategy_Manager.py",
]

BIND_FUNCS = {
    "bind_phase",
    "bind_debug",
    "bind_step",
    "bind_decode_entry",
    "bind_kv_append",
    "bind_prepack",
    "bind_prefill_pool",
}

# Worker binder methods that engine modules call instead of importing
# phase_state, keeping the worker-to-models import direction.
# ``self._name(arg0, arg1, ...)`` expands to these (target, attr) pairs in
# positional-argument order; the binder's own body is verified to perform
# exactly these assignments with its parameters in the same order.
DELEGATED_BINDERS = {
    "_bind_graph_capture_kv_managers": [
        ("AttnWrapperBase", "gpu_paged_kv_manager"),
        ("AttnWrapperBase", "gpu_paged_kv_manager_aux"),
    ],
    "_bind_graph_capture_kv_manager": [
        ("AttnWrapperBase", "gpu_paged_kv_manager"),
    ],
}


def known_names(tree: ast.AST) -> dict:
    """CANON plus module/function-local aliases bound to one of its classes."""
    names = dict(CANON)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            if node.value.id in CANON:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id not in CANON:
                        names[tgt.id] = CANON[node.value.id]
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in CANON and alias.asname:
                    names[alias.asname] = CANON[alias.name]
    return names


def functions_with_writes(tree: ast.AST, names: dict) -> dict:
    """{qualified function name: FunctionDef} for functions that store on the
    classes (or run a restore loop on them) anywhere in their body."""
    out = {}

    def writes_in(node) -> bool:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.ctx, (ast.Store, ast.Del))
                and isinstance(sub.value, ast.Name)
                and sub.value.id in names
            ):
                return True
            if restore_loop_target(sub, names) is not None:
                return True
            if new_call_kind(sub) is not None:
                return True
            if delegated_call(sub) is not None:
                return True
        return False

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}.{child.name}" if prefix else child.name
                if writes_in(child):
                    out[qual] = child
                visit(child, qual)
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}.{child.name}" if prefix else child.name)
            else:
                visit(child, prefix)

    visit(tree, "")
    return out


def restore_loop_target(node, names: dict):
    """``for k, v in M.items(): setattr(C, k, v)`` -> (canonical C, M-unparse)."""
    if not isinstance(node, ast.For) or node.orelse:
        return None
    if not (
        isinstance(node.target, ast.Tuple)
        and len(node.target.elts) == 2
        and all(isinstance(e, ast.Name) for e in node.target.elts)
    ):
        return None
    if not (
        isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "items"
        and not node.iter.args
    ):
        return None
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
        return None
    call = node.body[0].value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "setattr"
        and len(call.args) == 3
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id in names
        and isinstance(call.args[1], ast.Name)
        and isinstance(call.args[2], ast.Name)
        and call.args[1].id == node.target.elts[0].id
        and call.args[2].id == node.target.elts[1].id
    ):
        return None
    return names[call.args[0].id], ast.unparse(node.iter.func.value)


def delegated_call(node):
    """``self._binder(arg, ...)`` for a registered delegated binder."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr in DELEGATED_BINDERS
    ):
        return node.func.attr
    return None


def new_call_kind(node):
    """phase_state.bind_*(...) / restore_saved_state(...) -> function name."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "phase_state"
        and (func.attr in BIND_FUNCS or func.attr == "restore_saved_state")
    ):
        return func.attr
    return None


def iter_statements(node):
    """Every statement node under ``node`` in source order (recursive DFS)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            yield child
        yield from iter_statements(child)


def extract_base(fn: ast.AST, names: dict) -> list:
    entries = []
    for stmt in iter_statements(fn):
        restore = restore_loop_target(stmt, names)
        if restore is not None:
            entries.append((restore[0], "__restore__", restore[1]))
            continue
        if isinstance(stmt, ast.Assign):
            for tgt_group in stmt.targets:
                elts = (
                    list(tgt_group.elts)
                    if isinstance(tgt_group, (ast.Tuple, ast.List))
                    else [tgt_group]
                )
                tuple_rhs = (
                    ast.unparse(stmt.value) if len(elts) > 1 else None
                )
                for idx, tgt in enumerate(elts):
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id in names
                    ):
                        if tuple_rhs is not None:
                            value = ("tuple", tuple_rhs, idx)
                        else:
                            value = ast.unparse(stmt.value)
                        entries.append((names[tgt.value.id], tgt.attr, value))
        elif isinstance(stmt, (ast.AugAssign, ast.Delete)):
            targets = (
                stmt.targets if isinstance(stmt, ast.Delete) else [stmt.target]
            )
            for tgt in targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id in names
                ):
                    raise SystemExit(
                        f"unsupported base statement kind at line {stmt.lineno}"
                    )
    return [e for e in entries if e[1] not in DELETED_ATTRS]


def extract_new(fn: ast.AST, names: dict, base_tuple_rhs: set) -> list:
    entries = []
    temp_bindings = {}
    problems = []
    for stmt in iter_statements(fn):
        # A tuple unpack of a base tuple RHS into locals: record the locals.
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], (ast.Tuple, ast.List))
            and all(isinstance(e, ast.Name) for e in stmt.targets[0].elts)
            and ast.unparse(stmt.value) in base_tuple_rhs
        ):
            for idx, elt in enumerate(stmt.targets[0].elts):
                temp_bindings[elt.id] = ("tuple", ast.unparse(stmt.value), idx)
            continue
        # Any remaining direct store on the classes is a conversion failure.
        if isinstance(stmt, ast.Assign):
            for tgt_group in stmt.targets:
                elts = (
                    list(tgt_group.elts)
                    if isinstance(tgt_group, (ast.Tuple, ast.List))
                    else [tgt_group]
                )
                for tgt in elts:
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id in names
                    ):
                        problems.append(
                            f"direct class-attribute store survives at line "
                            f"{stmt.lineno}: {ast.unparse(stmt)[:80]}"
                        )
        restore = restore_loop_target(stmt, names)
        if restore is not None:
            problems.append(
                f"setattr restore loop survives at line {stmt.lineno}"
            )
        if not isinstance(stmt, ast.Expr):
            continue
        binder = delegated_call(stmt.value)
        if binder is not None:
            pairs = DELEGATED_BINDERS[binder]
            args = stmt.value.args
            if len(args) != len(pairs) or stmt.value.keywords:
                problems.append(
                    f"delegated binder call shape at line {stmt.lineno}"
                )
                continue
            for (target, attr), arg in zip(pairs, args):
                entries.append((target, attr, ast.unparse(arg)))
            continue
        kind = new_call_kind(stmt.value)
        if kind is None:
            continue
        call = stmt.value
        if not call.args or not isinstance(call.args[0], ast.Name):
            problems.append(f"call without a named target at line {stmt.lineno}")
            continue
        target = names.get(call.args[0].id)
        if target is None:
            problems.append(
                f"unknown target {call.args[0].id!r} at line {stmt.lineno}"
            )
            continue
        if kind == "restore_saved_state":
            if len(call.args) != 2:
                problems.append(f"restore call shape at line {stmt.lineno}")
                continue
            entries.append((target, "__restore__", ast.unparse(call.args[1])))
            continue
        for kw in call.keywords:
            if kw.arg is None:
                problems.append(f"**kwargs call at line {stmt.lineno}")
                continue
            if (
                isinstance(kw.value, ast.Name)
                and kw.value.id in temp_bindings
            ):
                value = temp_bindings[kw.value.id]
            else:
                value = ast.unparse(kw.value)
            entries.append((target, kw.arg, value))
    if problems:
        raise SystemExit("\n".join(problems))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    repo = pathlib.Path(args.repo)

    total_funcs = 0
    total_base_entries = 0
    total_new_entries = 0
    failures = []

    for rel in FILES:
        base_src = subprocess.run(
            ["git", "-C", str(repo), "show", f"{args.base}:{rel}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        new_src = (repo / rel).read_text()
        base_tree = ast.parse(base_src)
        new_tree = ast.parse(new_src)
        base_names = known_names(base_tree)
        new_names = known_names(new_tree)
        base_funcs = functions_with_writes(base_tree, base_names)
        new_funcs = functions_with_writes(new_tree, new_names)

        for qual, base_fn in sorted(base_funcs.items()):
            base_entries = extract_base(base_fn, base_names)
            base_tuple_rhs = {
                v[1] for (_, _, v) in base_entries if isinstance(v, tuple)
            }
            if qual not in new_funcs:
                if base_entries:
                    failures.append(
                        f"{rel}::{qual}: function gone but base still has "
                        f"{len(base_entries)} undeleted write(s)"
                    )
                else:
                    print(f"OK   {rel}::{qual}  (deleted; all writes were "
                          f"to deleted attributes)")
                continue
            new_entries = extract_new(
                new_funcs[qual], new_names, base_tuple_rhs
            )
            total_funcs += 1
            total_base_entries += len(base_entries)
            total_new_entries += len(new_entries)
            if base_entries != new_entries:
                lines = [f"{rel}::{qual}: MISMATCH"]
                for i, (b, n) in enumerate(
                    zip(
                        base_entries + [None] * len(new_entries),
                        new_entries + [None] * len(base_entries),
                    )
                ):
                    if b != n:
                        lines.append(f"  [{i}] base={b!r}")
                        lines.append(f"      new ={n!r}")
                failures.append("\n".join(lines))
            else:
                print(f"OK   {rel}::{qual}  ({len(base_entries)} writes)")

        # A function that gained phase_state calls without base writes would
        # be a write the base never performed -- except a registered
        # delegated binder, whose body must perform exactly its declared
        # (target, attr) pairs with its own parameters in order.
        for qual in sorted(set(new_funcs) - set(base_funcs)):
            entries = extract_new(new_funcs[qual], new_names, set())
            name = qual.rsplit(".", 1)[-1]
            if name in DELEGATED_BINDERS:
                fn = new_funcs[qual]
                params = [a.arg for a in fn.args.args if a.arg != "self"]
                expected = [
                    (target, attr, param)
                    for (target, attr), param in zip(
                        DELEGATED_BINDERS[name], params
                    )
                ]
                if entries != expected or len(params) != len(
                    DELEGATED_BINDERS[name]
                ):
                    failures.append(
                        f"{rel}::{qual}: delegated binder body does not "
                        f"match its declaration: {entries!r} != {expected!r}"
                    )
                else:
                    print(
                        f"OK   {rel}::{qual}  (delegated binder, "
                        f"{len(entries)} writes)"
                    )
                continue
            if entries:
                failures.append(
                    f"{rel}::{qual}: new writes with no base counterpart: "
                    f"{entries!r}"
                )

    print(
        f"\nchecked {total_funcs} functions: "
        f"{total_base_entries} base writes == {total_new_entries} new writes"
        if not failures
        else f"\nchecked {total_funcs} functions; {len(failures)} FAILURE(S)"
    )
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
