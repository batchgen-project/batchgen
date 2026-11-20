"""Prove that the AS3.1 engine bodies moved by a declared context rewrite."""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import subprocess
import sys
from typing import List, Optional


def _functions(path: pathlib.Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCapacityEngine"
    )
    return {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _base_functions(repo: pathlib.Path, revision: str, relative: str) -> dict:
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{relative}"],
        cwd=repo,
        text=True,
    )
    tree = ast.parse(source, filename=relative)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCapacityEngine"
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
        new = self.spec["attribute_rewrites"].get(old)
        if new is None:
            return node
        self._record("attribute", old, new)
        return ast.copy_location(_expr(new), node)

    def visit_Call(self, node):
        old = ast.unparse(node)
        special = self.spec["special_rewrites"].get(old)
        if special is not None:
            self._record("special", old, special)
            return ast.copy_location(_expr(special), node)
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in self.spec["converted_calls"]
            and not (node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "ctx")
        ):
            node.args.insert(0, ast.Name(id="ctx", ctx=ast.Load()))
            self._record("call", f"self.{node.func.attr}(...)", f"self.{node.func.attr}(ctx, ...)")
        return node

    def visit_BoolOp(self, node):
        old = ast.unparse(node)
        special = self.spec["special_rewrites"].get(old)
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


def prove(repo: pathlib.Path, base_revision: str, map_path: pathlib.Path) -> None:
    spec = json.loads(map_path.read_text())
    relative = spec["file"]
    new = _functions(repo / relative)
    old = _base_functions(repo, base_revision, relative)
    failures: List[str] = []
    for name in spec["functions"]:
        if name not in old or name not in new:
            failures.append(f"{name}: function missing in base or working tree")
            continue
        old_body = _body(old[name], base=True)
        rewritten = _Rewrite(spec, name)
        old_body = [rewritten.visit(statement) for statement in old_body]
        new_body = _body(new[name], base=False)
        if _has_worker_access(new[name]):
            failures.append(f"{name}: residual worker access")
        if len(old_body) != len(new_body):
            failures.append(f"{name}: statement count {len(old_body)} != {len(new_body)}")
        else:
            for index, (expected, actual) in enumerate(zip(old_body, new_body)):
                if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
                    failures.append(f"{name}: statement {index} differs")
                    break
        for kind, before, after in rewritten.used:
            print(f"RULE {name} [{kind}] {before} -> {after}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)
    print(f"PASS {len(spec['functions'])} functions")


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
