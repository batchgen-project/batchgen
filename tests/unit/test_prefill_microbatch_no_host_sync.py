"""The prepacked prefill micro-batch loop must not force a host-device sync.

Two per-micro-batch syncs used to sit in ``prefill_prepacked``: a pageable
``torch.tensor(..., device=cuda)`` H2D for the cu_seqlens table, and a rank-0
``.tolist()`` of the sampled tokens for a log line. Both stall the issuing
host thread once per micro-batch. The table now goes through a pinned buffer
with a non-blocking copy, and the sampled-ids line is emitted once per wave
after the existing completion boundary. This test pins both at source level.
"""

import ast
import pathlib

# S5b: prefill_prepacked's body moved verbatim to PrefillEngine.
ENGINE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "batchgen" / "worker" / "prefill_engine.py"
)


def _prefill_prepacked():
    tree = ast.parse(ENGINE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "prefill_prepacked":
            return node
    raise AssertionError("prefill_prepacked not found")


def _loops(fn):
    return [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))]


def test_cu_seqlens_copy_is_pinned_and_non_blocking():
    src = ast.unparse(_prefill_prepacked())
    assert "prefill_sequence_spans_to_cu_seqlens(batch_spans)" in src
    idx = src.index("prefill_sequence_spans_to_cu_seqlens(batch_spans)")
    window = src[idx: idx + 300]
    assert ".pin_memory()" in window
    assert "non_blocking=True" in window


def test_no_sampled_token_tolist_inside_any_loop():
    for loop in _loops(_prefill_prepacked()):
        for call in ast.walk(loop):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "tolist"
            ):
                text = ast.unparse(call)
                assert "batch_new_tokens" not in text, (
                    f"per-micro-batch sampled-token readback: {text}"
                )


def test_sampled_ids_line_is_emitted_after_the_completion_boundary():
    src = ast.unparse(_prefill_prepacked())
    sync = src.index("current_stream(self.torch_device).synchronize()")
    log = src.index("[PREFILL] first sampled token ids")
    assert sync < log
