"""Source guard: per-sequence resource scalars are assigned only where pinned.

``SequenceEntry`` (``batchgen/sequence.py``) carries the per-sequence
resource scalars this guard protects: the page/capacity counters
(``gpu_pages_allocated``, ``host_pages_allocated``, ``host_owned_pages``,
``host_token_capacity``), the length fields (``current_context_length``,
``decoded_length``) and the DP-group pair (``decode_dp_group``,
``evicted_from_dp_group``).  The acceptance target is one owner per scalar;
today the worker and the engines still assign them directly.  This test pins
TODAY's exact set of outside assignment sites as a list that can only shrink:
it fails when a NEW module/kind of write appears, and it fails when a listed
offender no longer exists (stale entry), so every removal is deliberate.

Covered patterns: plain attribute assignment, augmented assignment
(``seq.decoded_length += 1`` — the target has Store context) and ``del``, on
ANY receiver expression for field names that are unique to the sequence
entry (``gpu_pages_allocated``, ``host_owned_pages``,
``evicted_from_dp_group``).  A guarded name that some other definition in
the package also uses (``decoded_length`` is also a field of the lifespan
event record; ``current_context_length``, ``host_token_capacity`` and
``host_pages_allocated`` are also fields of the boundary metadata records;
``decode_dp_group`` is also a module/parameter spelling) cannot be
attributed to the sequence by spelling alone; such ambiguous names are
checked only when the receiver is recognisably a sequence (a bare name or an
attribute chain ending in ``seq``/``sequence``, the spellings the code base
uses).  NOT caught: writes to an ambiguous name through a receiver with any
other spelling (e.g. ``entry.decoded_length = 0`` on a real sequence),
``setattr`` with a dynamic name, and writes made by C extensions.  The
scalars are ints/None, so item writes and container-mutator calls do not
apply to them and are not checked.
"""
import ast
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
OWNER = PACKAGE / "sequence.py"
OWNER_CLASSES = {"SequenceEntry"}
HOLDER_NAMES = {"seq", "sequence"}
GUARDED_FIELDS = {
    "gpu_pages_allocated",
    "host_pages_allocated",
    "host_owned_pages",
    "host_token_capacity",
    "current_context_length",
    "decoded_length",
    "decode_dp_group",
    "evicted_from_dp_group",
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
def _ambiguous_names():
    """Guarded field names that some non-owner definition in the package also
    uses, so an access cannot be attributed to the sequence by spelling alone."""
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
    return GUARDED_FIELDS & foreign


def _is_recognised_sequence(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in HOLDER_NAMES
    if isinstance(expr, ast.Attribute):
        return expr.attr in HOLDER_NAMES
    return False


def _checked(receiver: ast.AST, name: str, ambiguous: set) -> bool:
    return name not in ambiguous or _is_recognised_sequence(receiver)


def _offences(tree: ast.Module, ambiguous: set):
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in GUARDED_FIELDS
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and _checked(node.value, node.attr, ambiguous)
        ):
            found.append((node.lineno, f"rebinding .{node.attr}"))
    return found


# The exact outside writes that exist today (one entry per file and kind of
# write, no line numbers so the list does not churn).  Entries may only be
# removed (by moving the write behind the owner's methods), never added.
KNOWN_OFFENDERS = [
    "batchgen/batchgen_worker.py: rebinding .current_context_length",
    "batchgen/batchgen_worker.py: rebinding .decode_dp_group",
    "batchgen/batchgen_worker.py: rebinding .decoded_length",
    "batchgen/batchgen_worker.py: rebinding .evicted_from_dp_group",
    "batchgen/batchgen_worker.py: rebinding .gpu_pages_allocated",
    "batchgen/batchgen_worker.py: rebinding .host_owned_pages",
    "batchgen/batchgen_worker.py: rebinding .host_pages_allocated",
    "batchgen/batchgen_worker.py: rebinding .host_token_capacity",
    "batchgen/worker/admission/boundary_planner.py: rebinding .current_context_length",
    "batchgen/worker/admission/boundary_planner.py: rebinding .decoded_length",
    "batchgen/worker/admission/boundary_planner.py: rebinding .gpu_pages_allocated",
    "batchgen/worker/admission/boundary_planner.py: rebinding .host_owned_pages",
    "batchgen/worker/admission/boundary_planner.py: rebinding .host_pages_allocated",
    "batchgen/worker/admission/boundary_planner.py: rebinding .host_token_capacity",
    "batchgen/worker/decode_engine.py: rebinding .current_context_length",
    "batchgen/worker/decode_engine.py: rebinding .decoded_length",
    "batchgen/worker/prefill_engine.py: rebinding .current_context_length",
    "batchgen/worker/prefill_engine.py: rebinding .decoded_length",
]


def test_only_pinned_modules_assign_sequence_resource_scalars():
    ambiguous = _ambiguous_names()
    offenders = sorted(
        {
            f"{path.relative_to(ROOT)}: {what}"
            for path, tree in _package_trees()
            if path != OWNER
            for _line, what in _offences(tree, ambiguous)
        }
    )
    assert offenders == KNOWN_OFFENDERS


def test_guarded_fields_exist_on_the_owner():
    """The guard is not vacuous: every guarded name is a SequenceEntry slot."""
    for node in ast.walk(_parse(OWNER)):
        if isinstance(node, ast.ClassDef) and node.name == "SequenceEntry":
            assert GUARDED_FIELDS <= _self_assigned_names(node)
            break
    else:
        raise AssertionError("SequenceEntry not found in batchgen/sequence.py")
    # The lifespan event record assigns ``decoded_length`` on ``self`` and
    # the boundary metadata records declare ``current_context_length``,
    # ``host_token_capacity``, ``host_pages_allocated`` and
    # ``decode_dp_group`` spellings of their own, so those five names are
    # receiver-restricted; the other three are unique in the package and
    # stay checked on every receiver.
    assert _ambiguous_names() == {
        "current_context_length",
        "decode_dp_group",
        "decoded_length",
        "host_pages_allocated",
        "host_token_capacity",
    }


def test_the_guard_sees_each_kind_of_write(tmp_path):
    """The guard is not vacuous: every kind of write it names is detected."""
    ambiguous = _ambiguous_names()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(self, seq, other):\n"
        "    seq.gpu_pages_allocated = 3\n"
        "    other.host_owned_pages = 0\n"
        "    seq.decoded_length += 1\n"
        "    del seq.decode_dp_group\n"
        "    self.decoded_length = 0\n"
        "    other.host_token_capacity = 0\n"
        "    return seq.gpu_pages_allocated\n"
    )
    lines = sorted(line for line, _ in _offences(_parse(sample), ambiguous))
    # Lines 6 and 7 write ambiguous names on receivers that are not a
    # recognised sequence spelling, so they are not flagged (a documented
    # blind spot); the unambiguous counter on line 3 is flagged on any
    # receiver.
    assert lines == [2, 3, 4, 5]
