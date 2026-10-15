"""Re-entry history length must follow the request's own decode budget.

In pool mode the worker-wide max_decoding_length is the budget of the first
batch a server saw. A later request that decoded more than that before
eviction used to re-enter with its history truncated to that budget, so
decoded_length fell below total_decoded_before_eviction and metadata
validation killed every rank.
"""

import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


def _load_helper():
    tree = ast.parse(WORKER.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_reentry_history_length")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace["_reentry_history_length"]


reentry_history_length = _load_helper()


def test_history_is_not_truncated_by_a_small_first_batch_budget():
    # 128 tokens decoded before eviction by a request allowed 1024 tokens.
    assert reentry_history_length(3000 + 128, 3000, 1024) == 128


def test_history_is_capped_by_the_request_budget():
    assert reentry_history_length(3000 + 700, 3000, 512) == 512


def test_history_is_never_negative():
    assert reentry_history_length(2990, 3000, 1024) == 0
