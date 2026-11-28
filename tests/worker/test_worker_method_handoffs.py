"""Source guard: where the worker hands a bound method of itself to an object.

Completion criterion P3 has two halves.  The first -- that imports point one
way -- has its checker.  The second asks for a test that a context holds no
worker, **no bound worker method, and no object that itself holds one**, and
the criteria recorded only that two indirect holders were known.  This is that
test's missing half, measured rather than described: every place the worker
passes ``self.<method>`` into a constructor or factory, which is how a bound
method -- and through it the whole worker -- ends up living inside another
object.

A hand-off counts when, inside ``batchgen/batchgen_worker.py``, a call whose
callee looks like a constructor or a factory (a capitalised final name, or one
starting ``build_``/``new_``/``make_``/``create_``) receives ``self.<name>``
where ``<name>`` is a method of the worker class.

``@property`` is deliberately excluded, and that exclusion is the difference
between this guard and a padded one: ``self._local_to_uuid_map`` reads like a
hand-off but the name is a property, so the call receives the dict it returns,
not a bound method, and the receiving object holds no worker.  Two call sites
look like offenders for exactly that reason and are not.

NOT caught, and stated so the claim is not read as wider than it is: a bound
method passed through a local variable, one reached by ``getattr``, a hand-off
written outside the worker file, and a closure that captures the worker instead
of being handed a method.  None of the first three exists today; a closure
would need a different analysis, and the list below is the attribute-passing
half.

The list may only shrink.  A new entry is a regression -- give the receiving
object the value it needs, or a small owner object, instead of a door back to
the worker -- and an entry that no longer matches must be removed, so a
removed hand-off cannot quietly return.

Measured on ``8ef48633``: 6 call sites, 11 hand-offs counted as call site and
method, 6 distinct worker methods, and 3 receiving objects -- the three
``_traj.build_*`` entries are factories on the trajectory store, so the object
that ends up holding those methods is reached through it.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"

# (callee as written, the method names handed to it) -- only shrinks.
ALLOWED = {
    ("TrajectoryStore", ("_recurrent_state_owner",)),
    ("self._traj.build_committer", ("_host_views_by_group", "_next_pool_segment_sid")),
    (
        "self._traj.build_kda_record_store",
        ("_host_views_by_group", "_kda_record_shared_map", "_next_pool_segment_sid"),
    ),
    ("HostKVSnapshotArenaAllocator", ("_next_pool_segment_sid",)),
    ("self._traj.build_restorer", ("_host_views_by_group", "_next_pool_segment_sid")),
    ("CompletionIO", ("_gather_completed_tokens", "_report_completion")),
}

_FACTORY_PREFIXES = ("build_", "new_", "make_", "create_")


def _looks_like_a_factory(callee: str) -> bool:
    last = callee.split(".")[-1]
    return bool(re.match(r"^[A-Z]", last)) or last.startswith(_FACTORY_PREFIXES)


def _worker_methods(tree: ast.AST):
    """Method names, and separately the ones that are properties."""
    methods, properties = set(), set()
    for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            methods.add(fn.name)
            for dec in fn.decorator_list:
                text = ast.unparse(dec)
                if text == "property" or text.endswith(".setter"):
                    properties.add(fn.name)
    return methods, properties


def _handoffs():
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    methods, properties = _worker_methods(tree)
    bound = methods - properties
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        if not _looks_like_a_factory(callee):
            continue
        handed = sorted(
            {
                arg.attr
                for arg in list(node.args) + [kw.value for kw in node.keywords]
                if isinstance(arg, ast.Attribute)
                and isinstance(arg.value, ast.Name)
                and arg.value.id == "self"
                and arg.attr in bound
            }
        )
        if handed:
            found.append((node.lineno, callee, tuple(handed)))
    return found


def test_no_unlisted_object_is_handed_a_bound_worker_method():
    allowed = {entry for entry in ALLOWED}
    new = [
        f"line {lineno}: {callee} <- "
        + ", ".join(f"self.{name}" for name in handed)
        for lineno, callee, handed in _handoffs()
        if (callee, handed) not in allowed
    ]
    assert not new, "\n  ".join(
        [
            "the worker hands a bound method of itself to an object that is "
            "not on the only-shrinks list; pass the value, or a small owner "
            "object, instead of a door back to the worker:"
        ]
        + new
    )


def test_the_list_is_not_padded_with_handoffs_that_are_already_gone():
    """The half that makes this a ratchet: a removed hand-off must be removed here."""
    present = {(callee, handed) for _, callee, handed in _handoffs()}
    stale = [
        f"{callee} <- " + ", ".join(f"self.{name}" for name in handed)
        for callee, handed in sorted(ALLOWED)
        if (callee, handed) not in present
    ]
    assert not stale, "\n  ".join(
        ["these listed hand-offs no longer exist; remove them so a regression "
         "cannot hide behind them:"] + stale
    )


def test_a_property_read_is_not_counted_as_a_hand_off():
    """Guards the guard: the exclusion that keeps this list honest.

    ``self._local_to_uuid_map`` and ``self._uuid_to_local_map`` are properties,
    so a call that takes them receives the dict, not a bound method.  If either
    ever stopped being a property, it would start handing the worker out and the
    list above would have to say so -- which is why this is asserted rather than
    assumed.
    """
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    _, properties = _worker_methods(tree)
    for name in ("_local_to_uuid_map", "_uuid_to_local_map"):
        assert name in properties, (
            f"{name} is no longer a property, so passing self.{name} now hands "
            "out a bound worker method; re-measure the hand-off list"
        )
