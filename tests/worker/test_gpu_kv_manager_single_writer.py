"""Source guard: the GPU paged KV manager is the only writer of its state.

``GPUPagedKVCacheManager`` and its module-private page-table component
``_GPUPageTableManager`` (``batchgen/kv_cache/gpu_paged_kv_manager.py``) own
the GPU page pool, refcounts, page tables and slot maps; everything else must
go through their methods.  This test fails when any module other than the
owner touches that private storage, or mutates one of the public fields
(rebinding, ``del``, an item write, or a container mutator call).  Reads of
public fields and calls of owner methods are allowed.

The field names are derived from the owner classes' AST (every attribute they
assign on ``self``), so new fields are guarded automatically.  AST carries no
types, so a field name that some other definition in the package also uses (a
``self`` assignment in another class, a method, a class-level or module-level
name) cannot be attributed to the manager by spelling alone — ``config``,
``device``, ``_is_initialized`` and ``_gpu_page_table_manager`` are such
names.  Ambiguous names are checked only when the receiver expression is
recognisably the manager: a bare name or an attribute chain ending in one of
the holder spellings the code base uses (``gpu_paged_kv_cache_manager``,
``gpu_kv_manager``, ``mla_manager``, the generic ``gpu_manager``, or the
page-table hop ``_gpu_page_table_manager``).  Unambiguous names (most of the
private storage, ``seq_id_to_slot``, ``gpu_table``, ...) are checked on every
receiver.  Not caught: aliasing through a local variable whose name is not a
holder spelling (e.g. the ``primary``/``auxiliary`` halves inside the dual
GPU coordinator), and mutation via ``getattr``/``setattr`` strings.

The tree currently contains known outside accesses of the private storage
(no mutator/rebind/item writes of public fields): subclasses and the KV
coordinators reach ``_sequences``/``_free_pages``/``_geometry``/``_k_cache``/
``_v_cache``/``_shared_page_key_to_gpu_page``, and the worker, engines and
model adapters reach ``_gpu_page_table_manager``/``_slot_index_tensor``.
They are pinned in ``KNOWN_OFFENDERS`` (one entry per file and kind of
access, no line numbers so the list does not churn); the test asserts exact
equality, so the list can shrink deliberately but never grow silently.
"""
import ast
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
OWNER = PACKAGE / "kv_cache" / "gpu_paged_kv_manager.py"
OWNER_CLASSES = {"GPUPagedKVCacheManager", "_GPUPageTableManager"}
HOLDER_NAMES = {
    "gpu_paged_kv_cache_manager",
    "gpu_kv_manager",
    "gpu_manager",
    "mla_manager",
    "_gpu_page_table_manager",
}
MUTATORS = {
    "add", "append", "appendleft", "clear", "difference_update", "discard",
    "extend", "extendleft", "insert", "intersection_update", "pop", "popitem",
    "popleft", "remove", "reverse", "setdefault", "sort",
    "symmetric_difference_update", "update",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


@cache
def _package_trees():
    return [(path, _parse(path)) for path in sorted(PACKAGE.rglob("*.py"))]


def _self_assigned_names(classdef: ast.ClassDef) -> set:
    """Attribute names the class assigns on ``self``."""
    names = set()
    for sub in ast.walk(classdef):
        if isinstance(sub, ast.Assign):
            targets = sub.targets
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
            targets = [sub.target]
        else:
            continue
        for target in targets:
            for node in ast.walk(target):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    names.add(node.attr)
    return names


@cache
def _owner_fields():
    """``(private, public)`` field-name sets of the owner class(es)."""
    private, public = set(), set()
    for node in ast.walk(_parse(OWNER)):
        if isinstance(node, ast.ClassDef) and node.name in OWNER_CLASSES:
            for name in _self_assigned_names(node):
                (private if name.startswith("_") else public).add(name)
    return private, public


@cache
def _ambiguous_names():
    """Owner field names that some non-owner definition in the package also
    uses, so an access cannot be attributed to the owner by spelling alone."""
    private, public = _owner_fields()
    fields = private | public
    foreign = set()
    for path, tree in _package_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if path == OWNER and node.name in OWNER_CLASSES:
                continue
            foreign |= _self_assigned_names(node)
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    foreign.add(stmt.name)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    foreign.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    foreign.update(
                        t.id for t in stmt.targets if isinstance(t, ast.Name)
                    )
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                foreign.add(stmt.name)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(
                stmt.target, ast.Name
            ):
                foreign.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                foreign.update(
                    t.id for t in stmt.targets if isinstance(t, ast.Name)
                )
    return fields & foreign


def _is_recognised_owner(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in HOLDER_NAMES
    if isinstance(expr, ast.Attribute):
        return expr.attr in HOLDER_NAMES
    return False


def _checked(receiver: ast.AST, name: str, ambiguous: set) -> bool:
    return name not in ambiguous or _is_recognised_owner(receiver)


def _offences(tree: ast.Module, private: set, public: set, ambiguous: set):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = node.attr
            if name in private:
                if _checked(node.value, name, ambiguous):
                    found.append((node.lineno, f"private storage .{name}"))
            elif name in public and isinstance(node.ctx, (ast.Store, ast.Del)):
                if _checked(node.value, name, ambiguous):
                    found.append((node.lineno, f"rebinding .{name}"))
        elif isinstance(node, ast.Subscript) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr in public
                and _checked(value.value, value.attr, ambiguous)
            ):
                found.append((node.lineno, f"item write on .{value.attr}"))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in MUTATORS
                and isinstance(func.value, ast.Attribute)
                and func.value.attr in public
                and _checked(func.value.value, func.value.attr, ambiguous)
            ):
                found.append(
                    (node.lineno, f".{func.value.attr}.{func.attr}(")
                )
    return found


# The exact outside accesses that exist today.  Entries may only be removed
# (by moving the access behind an owner method or property), never added.
KNOWN_OFFENDERS = [
    "batchgen/batchgen_worker.py: private storage ._gpu_page_table_manager",
    "batchgen/batchgen_worker.py: private storage ._slot_index_tensor",
    "batchgen/kv_cache/compressed_ratio_gpu_paged_kv_manager.py: private storage ._k_cache",
    "batchgen/kv_cache/compressed_ratio_gpu_paged_kv_manager.py: private storage ._v_cache",
    "batchgen/kv_cache/dual_kv_cache_coordinator.py: private storage ._free_pages",
    "batchgen/kv_cache/dual_kv_cache_coordinator.py: private storage ._geometry",
    "batchgen/kv_cache/dual_kv_cache_coordinator.py: private storage ._shared_page_key_to_gpu_page",
    "batchgen/kv_cache/kimi_linear_kv_coordinator.py: private storage ._sequences",
    "batchgen/kv_cache/position_mapped_gpu_paged_kv_manager.py: private storage ._slot_index_tensor",
    "batchgen/models/glm/glm5/cuda_graph_adapter.py: private storage ._slot_index_tensor",
    "batchgen/models/moonshotai/kimi_k25/cuda_graph_adapter.py: private storage ._slot_index_tensor",
    "batchgen/models/moonshotai/kimi_k25/model.py: private storage ._gpu_page_table_manager",
    "batchgen/models/moonshotai/kimi_k25/model.py: private storage ._slot_index_tensor",
    "batchgen/models/openai/gpt_oss_120b/model.py: private storage ._gpu_page_table_manager",
    "batchgen/models/openai/gpt_oss_120b/model.py: private storage ._slot_index_tensor",
    "batchgen/worker/decode_engine.py: private storage ._gpu_page_table_manager",
    "batchgen/worker/decode_engine.py: private storage ._slot_index_tensor",
]


def test_only_the_gpu_kv_manager_writes_its_state():
    private, public = _owner_fields()
    ambiguous = _ambiguous_names()
    offenders = sorted(
        {
            f"{path.relative_to(ROOT)}: {what}"
            for path, tree in _package_trees()
            if path != OWNER
            for _line, what in _offences(tree, private, public, ambiguous)
        }
    )
    assert offenders == KNOWN_OFFENDERS


def test_derived_fields_describe_the_owner():
    """The guard is not vacuous: the derived sets match a manual read."""
    private, public = _owner_fields()
    assert len(private) >= 30
    assert {"_free_pages", "_sequences", "_gpu_page_refcounts"} <= private
    assert {"seq_id_to_slot", "gpu_table", "config", "device"} <= public
    ambiguous = _ambiguous_names()
    # ``config``/``device`` are assigned on ``self`` by dozens of other
    # classes and ``_gpu_page_table_manager`` by the model parallel strategy
    # managers, so those spellings are receiver-restricted; the pool and
    # slot-map names are unique in the package and stay checked everywhere.
    assert {"config", "device", "_gpu_page_table_manager"} <= ambiguous
    assert "_free_pages" not in ambiguous
    assert "seq_id_to_slot" not in ambiguous


def test_the_guard_sees_each_kind_of_write(tmp_path):
    """The guard is not vacuous: every kind of write it names is detected."""
    private, public = _owner_fields()
    ambiguous = _ambiguous_names()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(ctx, mgr, gpu_paged_kv_cache_manager):\n"
        "    mgr._free_pages = []\n"
        "    ctx.gpu_kv_manager._gpu_page_table_manager.rebuild()\n"
        "    ctx.gpu_kv_manager.config = None\n"
        "    mgr.seq_id_to_slot[1] = 2\n"
        "    gpu_paged_kv_cache_manager.device = 0\n"
        "    mgr.gpu_table.clear()\n"
        "    return ctx.gpu_kv_manager.gpu_table\n"
    )
    lines = sorted(
        line
        for line, _ in _offences(_parse(sample), private, public, ambiguous)
    )
    assert lines == [2, 3, 4, 5, 6, 7]
