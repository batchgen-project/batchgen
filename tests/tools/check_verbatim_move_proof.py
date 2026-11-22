"""Prove engine-body moves declared by a generic context rewrite map."""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import subprocess
import sys
from typing import Iterable, List, Optional


def _functions(path: pathlib.Path, class_name: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _base_functions(
    repo: pathlib.Path, revision: str, relative: str, class_name: str
) -> dict:
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{relative}"],
        cwd=repo,
        text=True,
    )
    tree = ast.parse(source, filename=relative)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _expr(text: str) -> ast.AST:
    return ast.parse(text, mode="eval").body


class _Rewrite(ast.NodeTransformer):
    def __init__(self, spec: dict, function: str):
        self.spec = spec
        self.function = function
        self.used: list[tuple[str, str, str]] = []

    def _record(self, kind: str, old: str, new: str) -> None:
        self.used.append((kind, old, new))

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        old = ast.unparse(node)
        new = self.spec.get("attribute_rewrites", {}).get(old)
        if new is None:
            return node
        self._record("attribute", old, new)
        replacement = _expr(new)
        if hasattr(replacement, "ctx") and hasattr(node, "ctx"):
            replacement.ctx = node.ctx
        return ast.copy_location(replacement, node)

    def visit_Call(self, node):
        old = ast.unparse(node)
        replacement = self.spec.get("call_rewrites", {}).get(old)
        if replacement is not None:
            self._record("call", old, replacement)
            return ast.copy_location(_expr(replacement), node)
        special = self.spec.get("special_rewrites", {}).get(old)
        if special is not None:
            self._record("special", old, special)
            return ast.copy_location(_expr(special), node)
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in self.spec.get("converted_calls", [])
            and not (node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "ctx")
        ):
            node.args.insert(0, ast.Name(id="ctx", ctx=ast.Load()))
            self._record("call", f"self.{node.func.attr}(...)", f"self.{node.func.attr}(ctx, ...)")
        return node

    def visit_BoolOp(self, node):
        old = ast.unparse(node)
        special = self.spec.get("special_rewrites", {}).get(old)
        if special is not None:
            self._record("special", old, special)
            return ast.copy_location(_expr(special), node)
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        old = ast.unparse(node)
        special = self.spec.get("special_rewrites", {}).get(old)
        if special is not None:
            self._record("special", old, special)
            return ast.copy_location(_expr(special), node)
        return self.generic_visit(node)


def _body(node: ast.FunctionDef, *, base: bool) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    if base:
        body = [
            statement
            for statement in body
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "self"
                and isinstance(statement.value, ast.Name)
                and statement.value.id == "worker"
            )
        ]
    return body


def _has_worker_access(node: ast.AST) -> bool:
    return any(isinstance(item, ast.Name) and item.id == "worker" for item in ast.walk(node))


def _map_entries(spec: dict) -> Iterable[dict]:
    entries = spec.get("entries")
    if entries is not None:
        return entries
    return (spec,)


def _filtered_base_body(node: ast.FunctionDef, spec: dict) -> list[ast.stmt]:
    body = _body(node, base=True)
    statements = (spec.get("base") or {}).get("statements")
    if statements is not None:
        # A split step moves a slice of the base body: indices are into the
        # docstring-stripped (and `self = worker`-stripped) base body, before
        # any omission filter, so they stay stable and auditable.
        body = body[statements[0]:statements[1]]
    omitted = set(spec.get("omitted_statements", []))
    return [
        statement
        for statement in body
        if ast.unparse(statement) not in omitted
    ]


def _filtered_new_body(node: ast.FunctionDef, spec: dict) -> list[ast.stmt]:
    omitted = set(spec.get("new_omitted_statements", []))
    return [
        statement
        for statement in _body(node, base=False)
        if ast.unparse(statement) not in omitted
    ]


class _StatementRewrite(ast.NodeTransformer):
    def __init__(self, rewrites: dict[str, str]):
        self.rewrites = rewrites

    def _rewrite(self, node):
        replacement = self.rewrites.get(ast.unparse(node))
        if replacement is None:
            return node
        return ast.copy_location(ast.parse(replacement).body[0], node)

    def visit_Return(self, node):
        return self._rewrite(self.generic_visit(node))

    def visit_Assign(self, node):
        return self._rewrite(self.generic_visit(node))


def prove(repo: pathlib.Path, base_revision: str, map_path: pathlib.Path) -> None:
    spec = json.loads(map_path.read_text())
    failures: List[str] = []
    passed = 0
    for entry in _map_entries(spec):
        relative = entry["file"]
        class_name = entry.get("class", "KVCapacityEngine")
        new = _functions(repo / relative, class_name)
        # A cross-file (or split) move declares where its base body lives:
        # `base` names the file/class/function the statements came from, and
        # optionally the statement slice. Default: same file/class/name.
        base_spec = entry.get("base") or {}
        base_relative = base_spec.get("file", relative)
        base_class = base_spec.get("class", class_name)
        old = _base_functions(repo, base_revision, base_relative, base_class)
        for name in entry["functions"]:
            base_name = base_spec.get("function", name)
            if base_name not in old or name not in new:
                failures.append(f"{relative}:{name}: function missing in base or working tree")
                continue
            old_body = _filtered_base_body(old[base_name], entry)
            rewritten = _Rewrite(entry, name)
            old_body = [rewritten.visit(statement) for statement in old_body]
            statement_rewrites = entry.get("statement_rewrites", {})
            old_body = [
                _StatementRewrite(statement_rewrites).visit(statement)
                for statement in old_body
            ]
            new_body = _filtered_new_body(new[name], entry)
            if _has_worker_access(new[name]):
                failures.append(f"{relative}:{name}: residual worker access")
            if len(old_body) != len(new_body):
                failures.append(
                    f"{relative}:{name}: statement count {len(old_body)} != {len(new_body)}"
                )
            else:
                for index, (expected, actual) in enumerate(zip(old_body, new_body)):
                    if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
                        failures.append(f"{relative}:{name}: statement {index} differs")
                        break
            for kind, before, after in rewritten.used:
                print(f"RULE {relative}:{name} [{kind}] {before} -> {after}")
            if not any(f.startswith(f"{relative}:{name}:") for f in failures):
                passed += 1
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)
    print(f"PASS {passed} functions")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base git revision")
    parser.add_argument("--map", required=True, type=pathlib.Path, help="rewrite map JSON")
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args(argv)
    prove(args.repo.resolve(), args.base, args.map.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
