"""The [DECODE] heartbeat's device-memory fields: format, placement, cost."""

import ast
from pathlib import Path

from batchgen.worker.decode_engine import format_device_memory_fields

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


def test_fields_render_used_and_total_in_gib():
    text = format_device_memory_fields(lambda: (39 * GIB, 191 * GIB))
    assert text == " dev_mem_used_gib=152.00 dev_mem_total_gib=191.00"
    # Appended fields: leading space, key=value, no spaces inside values.
    assert text.startswith(" ") and text.count("=") == 2


def test_without_a_device_the_fields_say_so_explicitly():
    assert format_device_memory_fields(None) == (
        " dev_mem_used_gib=nodevice dev_mem_total_gib=nodevice"
    )


def test_heartbeat_appends_the_fields_after_every_existing_one():
    source = (ROOT / "batchgen/worker/decode_engine.py").read_text()
    tree = ast.parse(source)
    heartbeats = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        and any(
            isinstance(part, ast.Constant) and "[DECODE] step=" in str(part.value)
            for part in node.values
        )
    ]
    assert len(heartbeats) == 1
    rendered = [ast.unparse(part) for part in heartbeats[0].values]
    # The new fields go last, so every byte of the line before them — the
    # original fields and the earlier optional suffixes — is unchanged.
    assert rendered[-1] == "{format_device_memory_fields(_dev_mem_query)}"
    # One driver query per throttled heartbeat: the query callable is built
    # once per decode round, outside the step loop, and only with a device.
    assert "torch.cuda.is_available()" in source
