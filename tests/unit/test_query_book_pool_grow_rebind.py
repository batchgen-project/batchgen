"""CPU unit test for the QueryBook buffer-pool GROW + REBIND-with-live-sequences path.

This exercises the branch that GPU runs have never hit. Wave admission serialises
batches, so a pool grow has never happened while a live, mid-decode sequence still
occupies a row -- every observed grow rebound 0 sequences. That copy-live-rows /
re-point-views branch is therefore untested code that could corrupt on a real
concurrent grow. Here we force exactly that and assert the live row survives the
grow byte-for-byte.

The code under test is the REAL shipping source of
  - ``QueryBookBufferPool`` (including ``.adopt`` -- the live-row copy)
  - ``BatchGenWorker._ensure_buffer_pool``      (the grow orchestration)
  - ``BatchGenWorker._rebind_buffer_pool_views``(re-point seq + query_book views)
  - ``BatchGenWorker._retire_buffer_pool``
extracted verbatim from ``batchgen/batchgen_worker.py`` by AST and exec'd against a
fake worker ``self``. We extract instead of importing because importing
``batchgen.batchgen_worker`` pulls in the JIT-compiled ``core_engine`` (see its
module-level ``from batchgen.models.engine_loader import core_engine``), which is
not available on a CPU box. Extraction keeps the test bound to the real source:
any edit to adopt/_ensure_buffer_pool/_rebind flows straight into these asserts.

The node-shared input_ids segment is allocated through the real
``allocate_node_shared_int64`` (POSIX /dev/shm) with a no-op barrier, exactly as its
own docstring prescribes for tests.
"""

import ast
import logging
import os
import textwrap
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

import batchgen
from batchgen.query_book import QueryBookEntry
from batchgen.sequence import SequenceEntry

WORKER_PATH = os.path.join(os.path.dirname(batchgen.__file__), "batchgen_worker.py")

# The single line inside QueryBookBufferPool.adopt that copies live input_ids rows
# from the superseded pool into the grown one. The mutation test neuters exactly
# this line to prove the positive assertions are load-bearing.
COPY_LINE = "self.input_ids_buffer[:rows, :cols] = old.input_ids_buffer[:rows, :cols]"


def _extract_segments():
    """Pull the exact source text of the symbols under test from the real file."""
    src = open(WORKER_PATH).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)

    def grab(node):
        # Slice full physical lines (they keep their leading tabs) then dedent, so a
        # tab-indented method becomes a top-level function.
        return textwrap.dedent("".join(lines[node.lineno - 1 : node.end_lineno]))

    wanted_top = {"allocate_node_shared_int64", "QueryBookPoolCapacityError", "QueryBookBufferPool"}
    wanted_methods = {
        "_node_shared_tag",
        "_ensure_buffer_pool",
        "_rebind_buffer_pool_views",
        "_retire_buffer_pool",
    }
    seg = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted_top:
            seg[node.name] = grab(node)
        elif isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in wanted_methods:
                    seg[m.name] = grab(m)
    missing = (wanted_top | wanted_methods) - set(seg)
    assert not missing, f"failed to extract from real source: {missing}"
    return seg


# Definitions must land in this order: exception -> shm helper -> pool -> methods
# (_retire_buffer_pool annotates a param with QueryBookBufferPool, so the class must
# already exist when its def executes).
_EXEC_ORDER = [
    "QueryBookPoolCapacityError",
    "allocate_node_shared_int64",
    "QueryBookBufferPool",
    "_node_shared_tag",
    "_ensure_buffer_pool",
    "_rebind_buffer_pool_views",
    "_retire_buffer_pool",
]


def _build_worker(mutate_adopt=False):
    """Return (FakeWorker class, QueryBookBufferPool, QueryBookPoolCapacityError).

    All functions share one globals dict ``g`` so their cross-references resolve.
    ``mutate_adopt`` neuters the live-row input_ids copy inside ``adopt``.
    """
    seg = _extract_segments()
    g = {
        "torch": torch,
        "os": os,
        "logging": logging,
        "dist": SimpleNamespace(barrier=lambda *a, **k: None),
        "NUM_GPUS_PER_NODE": 8,
        "Tuple": Tuple,
        "Optional": Optional,
        "Dict": Dict,
        "List": List,
        "Callable": Callable,
        "Sequence": Sequence,
        "Set": Set,
    }
    for name in _EXEC_ORDER:
        code = seg[name]
        if name == "QueryBookBufferPool" and mutate_adopt:
            assert COPY_LINE in code, "adopt row-copy line not found -- source drifted"
            code = code.replace(COPY_LINE, "pass  # MUTATION: live-row input_ids copy skipped")
        exec(compile(code, WORKER_PATH, "exec"), g)

    method_names = (
        "_node_shared_tag",
        "_ensure_buffer_pool",
        "_rebind_buffer_pool_views",
        "_retire_buffer_pool",
    )
    FakeWorker = type("FakeWorker", (), {m: g[m] for m in method_names})
    return FakeWorker, g["QueryBookBufferPool"], g["QueryBookPoolCapacityError"]


def _setup_live_pool(Pool):
    """A small (2 x 8) pool with row 0 occupied by a live, mid-decode sequence."""
    pool = Pool(num_sequences=2, input_ids_width=8, max_decoding_length=4, pad_token_id=0)
    slot = pool.allocate_slot()  # -> 0
    seq = SequenceEntry("seq-live", global_idx=0, prompt_length=5, max_decode_length=3, text="live")
    # kv_token_budget = 5 + 3 = 8 == input_ids_width
    prompt = torch.tensor([11, 12, 13, 14, 15], dtype=torch.long)
    iv = pool.get_input_ids_view(slot, seq.kv_token_budget)  # (1, 8)
    iv[0, :5] = prompt
    dv = pool.get_decoded_tokens_view(slot)  # (1, 4)
    dv[0, :2] = torch.tensor([901, 902], dtype=torch.int64)
    seq.decoded_length = 2
    seq._buffer_slot = slot
    seq.input_ids = iv
    seq.decoded_tokens = dv
    entry = QueryBookEntry(
        encoded={"input_ids": iv}, decoded_tokens=dv, kv_token_budget=seq.kv_token_budget
    )
    return pool, seq, entry, prompt


def _make_worker(FakeWorker, pool, seq, entry):
    w = FakeWorker()
    w._buffer_pool = pool
    w._buffer_pool_generation = 1
    w.rank = 0
    w.pad_token_id = 0
    w._retired_buffer_pools = []
    # short unique tag -> POSIX shm name stays under the macOS 31-char limit
    w._shared_buffer_tag = os.urandom(2).hex()
    w.global_batch = [seq]
    w._uuid_to_local_map = {seq.uuid: 0}
    w.query_book = {0: entry}
    return w


def _cleanup(pool):
    shm = getattr(pool, "input_ids_shm", None)
    if shm is not None:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


def test_extraction_covers_real_source():
    """Guard: the harness really pulled the grow/rebind code, not empty stubs."""
    seg = _extract_segments()
    assert COPY_LINE in seg["QueryBookBufferPool"]
    assert "def adopt(self" in seg["QueryBookBufferPool"]
    assert "new_pool.adopt(old)" in seg["_ensure_buffer_pool"]
    assert "self._rebind_buffer_pool_views()" in seg["_ensure_buffer_pool"]
    assert "rebound += 1" in seg["_rebind_buffer_pool_views"]


def test_grow_copies_live_row_rebinds_and_admits_second():
    FakeWorker, Pool, _ = _build_worker()
    pool, seq, entry, prompt = _setup_live_pool(Pool)

    old_buf_ptr = pool.input_ids_buffer.data_ptr()
    prompt_snap = seq.input_ids[0, :5].clone()
    dec_snap = seq.decoded_tokens[0, :2].clone()

    w = _make_worker(FakeWorker, pool, seq, entry)
    try:
        # Force a grow: width 8 -> 16 (also rows 2 -> 4, decode 4 -> 8).
        w._ensure_buffer_pool(
            required_rows=4,
            required_input_width=16,
            required_decode_width=8,
            reason="unit test forced grow with a live sequence",
        )
        grown = w._buffer_pool

        # A real grow occurred into a fresh, larger allocation.
        assert grown is not pool
        assert (grown.num_sequences, grown.input_ids_width, grown.max_decoding_length) == (4, 16, 8)
        assert grown.input_ids_buffer.data_ptr() != old_buf_ptr

        # (1) live row copied byte-identically into the grown buffer
        assert torch.equal(grown.input_ids_buffer[0, :5], prompt_snap)
        assert grown.input_ids_buffer[0, 5:].sum() == 0  # remainder is padding
        assert torch.equal(grown.decoded_tokens_buffer[0, :2], dec_snap)

        # (2) slot mapping intact; seq + query_book rebound onto the grown buffer
        assert seq._buffer_slot == 0
        assert seq.input_ids.shape == (1, seq.kv_token_budget)  # (1, 8)
        assert torch.equal(seq.input_ids[0, :5], prompt_snap)
        # the rebound view actually aliases the grown buffer (not a stale mapping)
        grown.input_ids_buffer[0, 7] = 4242
        assert seq.input_ids[0, 7] == 4242
        grown.input_ids_buffer[0, 7] = 0
        # query_book entry rebound to the SAME grown view object as the sequence
        assert entry.encoded["input_ids"].data_ptr() == seq.input_ids.data_ptr()
        assert torch.equal(entry.decoded_tokens[0, :2], dec_snap)

        # (3) a second sequence admits into the grown pool without clobbering the first
        slot2 = grown.allocate_slot()
        assert slot2 == 1  # _next_slot carried over from the old pool (1 row used)
        iv2 = grown.get_input_ids_view(slot2, 16)
        iv2[0, :10] = torch.full((10,), 777, dtype=torch.long)
        assert torch.equal(grown.input_ids_buffer[0, :5], prompt_snap)  # row 0 untouched
        assert torch.equal(seq.input_ids[0, :5], prompt_snap)
    finally:
        _cleanup(w._buffer_pool)


def test_mutation_row_copy_removed_is_detected():
    """Falsifiability: with the live-row copy skipped, the survival assertion fails.

    Proves test_grow_copies_live_row_rebinds_and_admits_second is not vacuous.
    """
    FakeWorker, Pool, _ = _build_worker(mutate_adopt=True)
    pool, seq, entry, prompt = _setup_live_pool(Pool)
    prompt_snap = seq.input_ids[0, :5].clone()

    w = _make_worker(FakeWorker, pool, seq, entry)
    try:
        w._ensure_buffer_pool(
            required_rows=4,
            required_input_width=16,
            required_decode_width=8,
            reason="unit test forced grow (mutated adopt)",
        )
        grown = w._buffer_pool
        # The live row was NOT carried over -> grown row 0 is the zero-filled segment.
        assert not torch.equal(grown.input_ids_buffer[0, :5], prompt_snap)
        assert grown.input_ids_buffer[0, :5].sum() == 0
        # and the rebound live view now reads zeros -- the corruption the copy prevents.
        assert seq.input_ids[0, :5].sum() == 0
    finally:
        _cleanup(w._buffer_pool)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
