"""Regression: prefill capacity must exclude the decode-graph scratch slot."""
import ast
from pathlib import Path

PSM = (Path(__file__).resolve().parents[1] / "batchgen" / "models" / "moonshotai"
       / "kimi_linear" / "Parallel_Strategy_Manager.py")


def test_prefill_sequence_limits_reserves_graph_scratch_slot():
    """The KDA pool is sized `sequence_slots + 1`; the +1 is graph scratch.

    Regression for `RuntimeError: Insufficient free KDA state items`. Nothing
    reserves that extra slot, so reporting the whole pool as prefill capacity
    let admission fill it: a 193-slot pool reported "193 free", the scheduler
    selected 193 sequences for the node, and prefill then had no scratch slot.
    """
    tree = ast.parse(PSM.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "prefill_sequence_limits"
    )
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "_KDA_GRAPH_SCRATCH_SLOTS" in names, (
        "prefill_sequence_limits must subtract the reserved graph-scratch slot "
        "from the reported capacity, or admission consumes it and prefill dies "
        "with 'Insufficient free KDA state items'"
    )
