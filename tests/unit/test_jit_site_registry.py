"""The JIT coverage registry: the repo-wide call-site pin and the operator
slots.

The scanner walks every registered backend inventory's scan roots and counts
calls of ``<backend>.<kernel>`` (direct attribute calls AND
``getattr(<backend>, "<kernel>")`` resolutions — a dynamic resolution is the
one static anchor of a dynamically dispatched kernel) and of the inventory's
wrapper callees; the pinned call-site table must match exactly. A newly
merged call site under the roots therefore cannot escape coverage silently.
"""

import ast
import collections
import pathlib

import pytest

import batchgen.models.glm.glm5.jit_coverage  # noqa: F401 - registers glm5
import batchgen.models.moonshotai.kimi_linear.jit_coverage  # noqa: F401
from batchgen_kernels import jit_site_registry as REG

_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Modules whose own bodies run the prewarm (their calls are the coverage).
_PREWARM_MODULES = ("triton_prewarm.py", "deepgemm_prewarm.py")


def _scan_calls(inventory, root=_ROOT):
    """``(path, enclosing scope, callee) -> count`` over the inventory."""
    files = set()
    for rel in inventory.paths:
        path = root / rel
        files.update([path] if path.is_file() else path.rglob("*.py"))
    callees = set(inventory.callees)
    backend = inventory.backend
    non_kernels = set(inventory.non_kernels)
    found = collections.Counter()

    class Visitor(ast.NodeVisitor):
        def __init__(self, rel):
            self.rel, self.stack = rel, []

        def _scope(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _scope

        def visit_Call(self, node):
            func, callee = node.func, None
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == backend
                    and func.attr not in non_kernels):
                callee = f"{backend}.{func.attr}"
            elif (isinstance(func, ast.Name) and func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == backend
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                    and node.args[1].value not in non_kernels):
                callee = f"{backend}.{node.args[1].value}"
            elif isinstance(func, ast.Name) and func.id in callees:
                callee = func.id
            if callee is not None:
                scope = ".".join(self.stack) or "<module>"
                found[(self.rel, scope, callee)] += 1
            self.generic_visit(node)

    for path in sorted(files):
        if path.name in _PREWARM_MODULES:
            continue
        rel = path.relative_to(root).as_posix()
        Visitor(rel).visit(ast.parse(path.read_text()))
    return found


@pytest.mark.parametrize("inventory", REG.backend_inventories(),
                         ids=lambda inv: f"{inv.backend}-{inv.family}")
def test_inventory_names_every_backend_call(inventory):
    found = _scan_calls(inventory)
    pinned = {key: count for key, (count, _) in inventory.call_sites.items()}
    missing = {k: v for k, v in found.items() if k not in pinned}
    stale = {k: v for k, v in pinned.items() if k not in found}
    changed = {k: (pinned[k], found[k]) for k in pinned.keys() & found.keys()
               if pinned[k] != found[k]}
    assert missing == {}, f"{inventory.backend} calls the table does not name"
    assert stale == {}, "table calls that no longer exist"
    assert changed == {}, "call counts changed (pinned, found)"


def test_scan_reaches_a_direct_call_under_batchgen_moe(tmp_path):
    inv = REG.backend_inventory("deep_gemm")
    call = ("import deep_gemm\n"
            "def grouped(a, b, out, layout):\n"
            "    deep_gemm.planted_kernel(a, b, out, layout)\n")
    for rel in ("batchgen/moe/planted_direct.py",
                "batchgen/elsewhere/planted.py"):
        (tmp_path / rel).parent.mkdir(parents=True)
        (tmp_path / rel).write_text(call)
    for rel in inv.paths:
        if not (tmp_path / rel).exists():
            (tmp_path / rel).mkdir(parents=True)
    assert "batchgen/moe" in inv.paths
    assert dict(_scan_calls(inv, tmp_path)) == {
        ("batchgen/moe/planted_direct.py", "grouped",
         "deep_gemm.planted_kernel"): 1,
    }


def test_scan_reaches_a_getattr_resolved_kernel(tmp_path):
    """``getattr(deep_gemm, "<kernel>")`` is a call site; non-kernel
    attributes stay out."""
    inv = REG.backend_inventory("deep_gemm")
    call = (
        "import deep_gemm\n"
        "def resolve():\n"
        "    fn = getattr(deep_gemm, 'planted_kernel', None)\n"
        "    align = getattr(deep_gemm,"
        " 'get_m_alignment_for_contiguous_layout', None)\n"
        "    return fn, align\n"
    )
    rel = "batchgen/moe/planted_getattr.py"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text(call)
    for other in inv.paths:
        if not (tmp_path / other).exists():
            (tmp_path / other).mkdir(parents=True)
    assert dict(_scan_calls(inv, tmp_path)) == {
        (rel, "resolve", "deep_gemm.planted_kernel"): 1,
    }


# --------------------------------------------------------------------------- #
#  Operator slots
# --------------------------------------------------------------------------- #
def test_glm5_prefill_grouped_gemm_slot():
    slot = REG.check_slot_backend("glm5", "prefill_grouped_gemm", "deepgemm")
    assert slot.default == "deepgemm"
    assert REG.default_backend("glm5", "prefill_grouped_gemm") == "deepgemm"
    REG.check_slot_backend("glm5", "prefill_grouped_gemm", "cute_ptrs")


def test_unvalidated_backend_is_refused_by_name():
    with pytest.raises(REG.JitCoverageError, match="not validated"):
        REG.check_slot_backend("glm5", "prefill_grouped_gemm", "trtllm")


def test_undeclared_slot_is_refused_by_name():
    with pytest.raises(REG.JitCoverageError, match="declares no operator slot"):
        REG.check_slot_backend("glm5", "decode_attention", "deepgemm")
    with pytest.raises(REG.JitCoverageError, match="declares no operator slot"):
        REG.default_backend("glm5", "decode_attention")


def test_jit_backend_without_inventory_is_refused():
    slot = REG.OperatorSlot(
        family="t", slot="s",
        backends={"x": ("jit", "no_such_backend")}, default="x",
    )
    REG._SLOTS[("t", "s")] = slot
    try:
        with pytest.raises(REG.JitCoverageError, match="no registered"):
            REG.check_slot_backend("t", "s", "x")
    finally:
        del REG._SLOTS[("t", "s")]


def test_slot_default_must_be_validated():
    with pytest.raises(ValueError, match="default backend"):
        REG.OperatorSlot(family="t", slot="s",
                         backends={"a": ("aot",)}, default="b")


def test_model_coverage_check():
    assert REG.family_for_model_type("glm_moe_dsa_5_3") == "glm5"
    assert REG.family_for_model_type("gpt_oss") is None
    assert REG.check_model_type_coverage("glm_moe_dsa_5_3") == (
        "glm5.prefill_grouped_gemm=deepgemm",)
    # A model type with no registered family is, truthfully, unchecked.
    assert REG.check_model_type_coverage("gpt_oss") == ()


def test_reregistration_is_idempotent_and_conflicts_are_refused():
    import batchgen.models.glm.glm5.jit_coverage as glm5_cov

    # Re-registering the identical contract (a re-import) is a no-op.
    REG.register_backend_inventory(glm5_cov.DEEP_GEMM_INVENTORY)
    conflicting = REG.BackendInventory(
        backend="deep_gemm", paths=("x",), callees=(), non_kernels=(),
        call_sites={})
    with pytest.raises(REG.JitCoverageError, match="different inventory"):
        REG.register_backend_inventory(conflicting)
