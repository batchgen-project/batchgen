"""Each admission module declares its contract, and the source must match it.

Completion criterion P2 asks that every module state what it reads, what it
writes and which orderings it relies on, in a few lines, and that a test check
that statement against the source.  Each module under
``batchgen/worker/admission/`` now carries a ``MODULE_CONTRACT`` dict, and this
test derives the same three sets from the module's own syntax tree and demands
equality -- not containment, so a declaration can neither miss a real read nor
claim one the module does not make.

What is derived, and therefore what the declaration means:

* ``reads`` -- every ``<param>.<name>`` loaded anywhere in the module, where
  ``<param>`` is a parameter of the function doing the loading.  That is the
  module's input surface: what it needs from the objects it is handed.  A
  method call on a parameter reads a name too (``gpu_manager.get_stats``), and
  that is deliberate: the module depends on that name existing.
* ``writes`` -- every ``<param>.<name>`` stored to or deleted.  These are the
  mutations the module performs on something it was given, as opposed to on an
  object it built itself.
* ``mutates`` -- every container-mutator call (``append``, ``update``, ...)
  whose receiver chains back to a parameter.  An empty set is the strong claim:
  the module takes its inputs and returns its answer without reaching into
  them.

``orderings`` is prose and is not derived.  It is the part a reader needs and a
script cannot supply, so the test only requires that it be present and
non-empty for a module with any behaviour at all.

Deliberately out of scope, so the claim is not read as wider than it is: reads
and writes through a local alias of a parameter, a parameter handed to another
function that mutates it, module-level globals, and anything reached by
``getattr``/``setattr`` with a computed name.  None of those exists in this
package today; a module that grows one needs this test extended, with the
reason, rather than its declaration quietly widened.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "batchgen" / "worker" / "admission"

MUTATORS = {
    "add", "append", "clear", "discard", "extend", "insert", "pop", "popitem",
    "remove", "reverse", "setdefault", "sort", "update",
}


def _derive(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    reads, writes, mutates = set(), set(), set()
    for fn in [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        params = {
            arg.arg
            for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        }
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in params
            ):
                name = f"{node.value.id}.{node.attr}"
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    writes.add(name)
                else:
                    reads.add(name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATORS
            ):
                base = node.func.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in params:
                    mutates.add(
                        f"{ast.unparse(node.func.value)}.{node.func.attr}()"
                    )
    return reads, writes, mutates


def _modules():
    return sorted(
        path for path in PACKAGE_DIR.glob("*.py")
    )


def _contract(path: Path):
    """Read the declaration from the source, not by importing the module.

    These modules sit under a package whose import pulls in native extensions,
    so a source read keeps the guard runnable wherever the repository is
    checked out -- and a contract that only holds once the native stack loads
    would not be much of a contract.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "MODULE_CONTRACT":
                return ast.literal_eval(node.value)
    return None


def test_every_admission_module_declares_a_contract():
    missing = [
        path.name for path in _modules() if _contract(path) is None
    ]
    assert not missing, (
        "these admission modules declare no MODULE_CONTRACT: "
        + ", ".join(missing)
    )


def test_each_declared_read_set_equals_what_the_source_reads():
    wrong = []
    for path in _modules():
        declared = set(_contract(path)["reads"])
        actual, _, _ = _derive(path)
        for extra in sorted(actual - declared):
            wrong.append(f"{path.name}: reads {extra}, not declared")
        for stale in sorted(declared - actual):
            wrong.append(f"{path.name}: declares {stale}, no longer read")
    assert not wrong, "\n  ".join(["read sets disagree with the source:"] + wrong)


def test_each_declared_write_set_equals_what_the_source_writes():
    wrong = []
    for path in _modules():
        declared = set(_contract(path)["writes"])
        _, actual, _ = _derive(path)
        for extra in sorted(actual - declared):
            wrong.append(f"{path.name}: writes {extra}, not declared")
        for stale in sorted(declared - actual):
            wrong.append(f"{path.name}: declares writing {stale}, it does not")
    assert not wrong, "\n  ".join(["write sets disagree with the source:"] + wrong)


def test_each_declared_mutation_set_equals_what_the_source_mutates():
    wrong = []
    for path in _modules():
        declared = set(_contract(path)["mutates"])
        _, _, actual = _derive(path)
        for extra in sorted(actual - declared):
            wrong.append(f"{path.name}: mutates {extra}, not declared")
        for stale in sorted(declared - actual):
            wrong.append(f"{path.name}: declares mutating {stale}, it does not")
    assert not wrong, (
        "\n  ".join(["mutation sets disagree with the source:"] + wrong)
    )


def test_a_module_with_behaviour_says_what_it_relies_on_being_ordered():
    """Prose, so it is only checked for presence -- but a blank is a gap."""
    silent = []
    for path in _modules():
        contract = _contract(path)
        reads, writes, mutates = _derive(path)
        has_behaviour = bool(reads or writes or mutates)
        if has_behaviour and not contract["orderings"]:
            silent.append(path.name)
    assert not silent, (
        "these modules do work but say nothing about the orderings they rely "
        "on; '" + "', '".join(silent) + "' need a line each (write 'nothing: "
        "<why>' when there genuinely is no ordering)"
    )


def test_only_the_boundary_timing_record_is_written_through_the_seam():
    """The package's whole write surface, in one assertion.

    Everything the admission modules are handed is read and returned, with one
    exception: the two boundary steps fill in the timing record they are given.
    This is the claim worth protecting -- a new write anywhere else in the
    package means a decision started keeping state where it was meant to
    produce a plan.
    """
    offenders = []
    for path in _modules():
        _, writes, mutates = _derive(path)
        for name in sorted(writes):
            if not name.startswith("timing."):
                offenders.append(f"{path.name}: writes {name}")
        for name in sorted(mutates):
            offenders.append(f"{path.name}: mutates {name}")
    assert not offenders, (
        "\n  ".join(
            ["the admission package writes outside the timing record:"]
            + offenders
        )
    )
