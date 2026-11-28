"""Each admission module declares its contract, and the source must match it.

Completion criterion P2 asks that every module state what it reads, what it
writes and which orderings it relies on, in a few lines, and that a test check
that statement against the source.  Each module under
``batchgen/worker/admission/`` carries a ``MODULE_CONTRACT`` dict, and this test
derives the first four keys from the module's own syntax tree and demands
equality -- not containment, so a declaration can neither miss a real read nor
claim one the module does not make.

What is derived, and therefore what each key means:

* ``reads`` -- every ``<param>.<name>`` loaded, where ``<param>`` is a parameter
  of the function doing the loading.  A method call on a parameter reads a name
  too (``gpu_manager.get_stats``), deliberately: the module depends on that name
  existing.
* ``writes`` -- every ``<param>.<name>`` stored to or deleted: a mutation of
  something handed in, on the object itself.
* ``writes_via_local`` -- the same, one step further out: a store on a local
  that was bound from an expression mentioning a parameter, so
  ``seq = ctx.sched.pool.get_sequence(u)`` followed by ``seq.eos_reached = ...``
  counts.  **This key is why the first version of this guard was wrong.**  It
  derived only the two above, which made the admission package look as if four
  of its six modules never touched their inputs; the planner in fact writes
  eight fields on sequence mirrors it reaches through its context.  A guard that
  stops at the first dereference flatters the code it checks.
* ``mutates`` -- every container-mutator call (``append``, ``sort``, ...) whose
  receiver chains back to a parameter or to such a local.

``self`` is excluded from all four: an object initialising or updating its own
attributes is not reaching into something it was handed, and single ownership of
an object's own state is a different criterion row with its own guard.

``orderings`` is prose and is not derived.  It is the part a reader needs and a
script cannot supply, so the test only requires it to be present and non-empty
for a module with any behaviour at all.

Still out of scope, and stated so the claim is not read as wider than it is: a
parameter handed on to another function that mutates it, aliasing through a
container element, and anything reached by ``getattr``/``setattr`` with a
computed name.  A module that grows one of those needs this test extended, with
the reason, rather than its declaration quietly widened.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "batchgen" / "worker" / "admission"

MUTATORS = {
    "add", "append", "clear", "discard", "extend", "insert", "pop", "popitem",
    "remove", "reverse", "setdefault", "sort", "update",
}


def _tainted_locals(fn, params):
    """Locals bound from an expression that mentions a parameter."""
    tainted = set()
    for _ in range(3):  # settle through chained binds
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.For):
                targets = [node.target]
            elif isinstance(node, ast.comprehension):
                targets = [node.target]
            if not targets:
                continue
            source = getattr(node, "value", None) or getattr(node, "iter", None)
            if source is None:
                continue
            names = {x.id for x in ast.walk(source) if isinstance(x, ast.Name)}
            if names & (params | tainted):
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            tainted.add(name.id)
    return tainted


def _derive(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    reads, writes, via_local, mutates = set(), set(), set(), set()
    for fn in [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        params = {
            arg.arg
            for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        }
        params.discard("self")
        tainted = _tainted_locals(fn, params)
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    if node.value.id in params:
                        writes.add(f"{node.value.id}.{node.attr}")
                    elif node.value.id in tainted:
                        via_local.add(f"{node.value.id}.{node.attr}")
                elif node.value.id in params:
                    reads.add(f"{node.value.id}.{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATORS
            ):
                base = node.func.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and (
                    base.id in params or base.id in tainted
                ):
                    mutates.add(f"{ast.unparse(node.func.value)}.{node.func.attr}()")
    return {
        "reads": reads,
        "writes": writes,
        "writes_via_local": via_local,
        "mutates": mutates,
    }


def _modules():
    return sorted(PACKAGE_DIR.glob("*.py"))


def _contract(path: Path):
    """Read the declaration from the source, not by importing the module.

    This package's import pulls in native extensions, and a contract that only
    held once the native stack loaded would not be much of a contract.
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
    missing = [path.name for path in _modules() if _contract(path) is None]
    assert not missing, (
        "these admission modules declare no MODULE_CONTRACT: "
        + ", ".join(missing)
    )


def _compare(key, label):
    wrong = []
    for path in _modules():
        declared = set(_contract(path)[key])
        actual = _derive(path)[key]
        for extra in sorted(actual - declared):
            wrong.append(f"{path.name}: {label} {extra}, not declared")
        for stale in sorted(declared - actual):
            wrong.append(f"{path.name}: declares {label} {stale}, it does not")
    return wrong


def test_each_declared_read_set_equals_what_the_source_reads():
    wrong = _compare("reads", "reads")
    assert not wrong, "\n  ".join(["read sets disagree with the source:"] + wrong)


def test_each_declared_write_set_equals_what_the_source_writes():
    wrong = _compare("writes", "writes")
    assert not wrong, "\n  ".join(["write sets disagree with the source:"] + wrong)


def test_each_declared_write_through_a_local_equals_the_source():
    wrong = _compare("writes_via_local", "writes through a local")
    assert not wrong, "\n  ".join(
        ["writes through a local disagree with the source:"] + wrong
    )


def test_each_declared_mutation_set_equals_what_the_source_mutates():
    wrong = _compare("mutates", "mutates")
    assert not wrong, "\n  ".join(
        ["mutation sets disagree with the source:"] + wrong
    )


def test_a_module_with_behaviour_says_what_it_relies_on_being_ordered():
    """Prose, so only presence is checked -- but a blank is a gap."""
    silent = []
    for path in _modules():
        contract = _contract(path)
        derived = _derive(path)
        if any(derived[k] for k in ("reads", "writes", "writes_via_local", "mutates")):
            if not contract["orderings"]:
                silent.append(path.name)
    assert not silent, (
        "these modules do work but say nothing about the orderings they rely "
        "on; '" + "', '".join(silent) + "' need a line each (write 'nothing: "
        "<why>' when there genuinely is no ordering)"
    )


def test_the_packages_whole_write_surface_is_three_named_things():
    """The claim worth protecting, stated correctly this time.

    The first version of this test asserted that the package wrote nothing but
    the boundary timing record. That was false, and false in the flattering
    direction: it only looked at stores on a parameter itself. The honest
    surface is three things, and nothing else:

    * the two boundary steps fill in the ``timing`` record they are handed;
    * the planner mirrors another rank's decode state onto the local
      ``SequenceEntry`` for sequences this rank does not own, writing eight
      fields directly beside the owner method it calls for three others;
    * the decode-entry selection sorts the candidate list it was given.

    A write outside those three means a decision started keeping state where it
    was meant to produce a plan, or a module began reaching further into what it
    was handed.
    """
    offenders = []
    for path in _modules():
        derived = _derive(path)
        for name in sorted(derived["writes"]):
            if not name.startswith("timing."):
                offenders.append(f"{path.name}: writes {name}")
        for name in sorted(derived["writes_via_local"]):
            if not (path.name == "boundary_planner.py" and name.startswith("seq.")):
                offenders.append(f"{path.name}: writes through a local {name}")
        for name in sorted(derived["mutates"]):
            if not (path.name == "decode_entry.py" and name == "all_candidates.sort()"):
                offenders.append(f"{path.name}: mutates {name}")
    assert not offenders, "\n  ".join(
        ["the admission package writes outside its three declared surfaces:"]
        + offenders
    )
