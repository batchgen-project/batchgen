"""Admission decisions and the status transition plans they produce."""

# Declared contract, checked against this module's source by
# tests/worker/test_admission_module_contracts.py: what this module
# reads off the objects it is given, what it writes on them, what it
# mutates through them, and the orderings it relies on. A change here
# that the source does not match, or a source change this does not
# cover, fails that test.
MODULE_CONTRACT = {
    "reads": (),
    "writes": (),
    "mutates": (),
    "orderings": (),
}

from batchgen.worker.admission.plan import (
    RejectedTransition,
    StatusEdge,
    Transition,
    TransitionPlan,
    apply_or_raise,
)

__all__ = [
    "RejectedTransition",
    "StatusEdge",
    "Transition",
    "TransitionPlan",
    "apply_or_raise",
]
