"""Source guard: closures defined in the worker that capture it, and where they go.

This is the other half of criterion P3's second question.  The hand-off guard
beside this one covers the case where the worker passes a bound method of itself
to an object.  A closure is the other route: a function defined inside a worker
method, referring to the enclosing ``self``, carries the whole worker with it
wherever it is handed.  The criteria named one such holder ("the KV stats
backend closes over the worker") and tested none.

Every worker-capturing closure is listed below with **where it goes**, and the
list is exact in both directions: a new one fails, and a listed one that has
moved or gone fails too.  The destination is part of the entry on purpose --
the same closure handed to a sort key and handed to a constructor are entirely
different facts, and a guard that recorded only "a closure exists here" would
not notice the change.

Three of the destinations keep the closure **beyond the call**, and those are the
ones that matter:

* ``DecodeProfileCapture`` -- a constructor; the object is then stored on the
  worker, so the lambda lives as long as the capture does.
* ``phase_state.bind_kv_append`` (twice) -- assigns the callback onto the
  attention wrapper *type*, so the closure outlives every batch and sits in the
  decode hot path.

The remaining eight are transient: a sort key, or a callable the receiving
method invokes while it runs and does not keep (``prefix_page_digests``,
``admission_shared_prefix_segments`` and ``restore_payload`` each take it as a
parameter and return data), or one used only inside the defining method.  They
are listed anyway, because "transient" is a fact about today's callee that a
future change to that callee could falsify quietly.

What this does NOT decide: whether a given capture should be removed.  Some of
these are the honest shape for a callback the model layer must call back into.
The guard's job is that the set cannot grow, and cannot change destination,
without someone saying so.

NOT caught, stated so the claim is not read wider than it is: a closure defined
outside ``batchgen/batchgen_worker.py`` that captures a worker passed into it, a
closure that captures a *bound method* taken from ``self`` earlier in the same
method (the hand-off guard covers the method, not its later capture), and
``functools.partial`` over a bound method.  None of the three exists in this
file today.

Measured on ``8ef48633``: 11 worker-capturing closures, 3 of them kept beyond
the call.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"

# (defining method, closure name, destination) -- exact, only shrinks.
ALLOWED = {
    ("_admission_shared_prefix_segments", "<lambda>", "self._prefix.admission_shared_prefix_segments"),
    ("_assign_ranks_for_prefix_sharing", "prompt_tokens", "local"),
    ("_decode_profile_capture_state", "<lambda>", "DecodeProfileCapture"),
    ("_glm5_decode_model_forward", "_forward", "local"),
    ("_install_decode_kv_append_callbacks", "kv_append_callback", "phase_state.bind_kv_append"),
    ("_install_decode_kv_append_callbacks", "kv_append_callback_aux", "phase_state.bind_kv_append"),
    ("_prefix_page_digests", "<lambda>", "self._prefix.prefix_page_digests"),
    ("_prepare_glm5_layer_graph_inputs", "_graph_slots", "local"),
    ("_stamp_prefix_prefill_order", "<lambda>", "sorted"),
    ("_trajectory_restore_payload", "_planned_boundary", "self._traj.restore_payload"),
    ("generate", "<lambda>", "decode_uuids.sort"),
}

# The destinations that keep the closure after the call returns.
KEPT_BEYOND_THE_CALL = {"DecodeProfileCapture", "phase_state.bind_kv_append"}


def _parents(tree):
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _captures_self(fn) -> bool:
    own = {
        arg.arg
        for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    }
    if "self" in own:
        return False
    return any(
        isinstance(node, ast.Name) and node.id == "self"
        for node in ast.walk(fn)
    )


def _destination(parents, method, node, name) -> str:
    """The call this closure is handed to, 'returned', or 'local'."""
    cursor = node
    while cursor is not None and cursor is not method:
        parent = parents.get(cursor)
        if isinstance(parent, ast.Return):
            return "returned"
        if isinstance(parent, ast.Call) and cursor is not parent.func:
            return ast.unparse(parent.func)
        cursor = parent
    if name == "<lambda>":
        return "local"
    for node2 in ast.walk(method):
        if isinstance(node2, ast.Call):
            for arg in list(node2.args) + [kw.value for kw in node2.keywords]:
                if isinstance(arg, ast.Name) and arg.id == name:
                    return ast.unparse(node2.func)
        if (
            isinstance(node2, ast.Return)
            and isinstance(node2.value, ast.Name)
            and node2.value.id == name
        ):
            return "returned"
    return "local"


def _closures():
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    parents = _parents(tree)
    found = set()
    for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(method):
                if node is method:
                    continue
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                ) and _captures_self(node):
                    name = getattr(node, "name", "<lambda>")
                    found.add(
                        (
                            method.name,
                            name,
                            _destination(parents, method, node, name),
                        )
                    )
    return found


def test_no_unlisted_closure_captures_the_worker():
    new = sorted(_closures() - ALLOWED)
    assert not new, "\n  ".join(
        [
            "a closure in the worker captures it and is not on the "
            "only-shrinks list (or an existing one changed destination); give "
            "the closure the values it needs as arguments, or add it here with "
            "the reason:"
        ]
        + [f"{method}: {name} -> {dest}" for method, name, dest in new]
    )


def test_the_list_is_not_padded_with_closures_that_are_already_gone():
    stale = sorted(ALLOWED - _closures())
    assert not stale, "\n  ".join(
        [
            "these listed closures no longer exist, or no longer go where the "
            "list says; update the entry so a regression cannot hide behind it:"
        ]
        + [f"{method}: {name} -> {dest}" for method, name, dest in stale]
    )


def test_only_three_captures_outlive_the_call_that_receives_them():
    """The claim worth protecting, separate from the inventory.

    A transient capture costs nothing: the callee uses it and drops it.  A
    capture that is stored keeps the whole worker reachable from wherever it
    was stored, which is what criterion P3 is about.  Three do that today, and
    a fourth would be a real change in the dependency direction.
    """
    kept = sorted(
        entry for entry in _closures() if entry[2] in KEPT_BEYOND_THE_CALL
    )
    assert len(kept) == 3, "\n  ".join(
        [
            f"expected 3 captures kept beyond their call, found {len(kept)}:"
        ]
        + [f"{method}: {name} -> {dest}" for method, name, dest in kept]
    )
