"""Buffer slots must not depend on the order completions freed them.

input_ids is one node-shared buffer, so every rank must give a sequence the
same slot. Allocation used set.pop(), whose result depends on the order slots
were freed; ranks free completed sequences in different orders, so the same
new sequence could get different slots on different ranks, and allocate_slot
zeroes the shared row it takes -- corrupting another rank's prompt.
"""

import ast
import heapq
import random
from pathlib import Path
from typing import Optional

import torch

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


class QueryBookPoolCapacityError(RuntimeError):
    pass


def _load_pool_class():
    tree = ast.parse(WORKER.read_text())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "QueryBookBufferPool")
    namespace = {"torch": torch, "heapq": heapq, "Optional": Optional,
                 "QueryBookPoolCapacityError": QueryBookPoolCapacityError}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace["QueryBookBufferPool"]


QueryBookBufferPool = _load_pool_class()


def _pool():
    return QueryBookBufferPool(num_sequences=64, input_ids_width=8,
                               max_decoding_length=4, pad_token_id=0)


def test_slots_do_not_depend_on_free_order():
    # Realistic pool: thousands of slots, completions free a sparse subset.
    pool_rows = 10240
    freed = random.Random(1).sample(range(pool_rows), 300)
    a = QueryBookBufferPool(num_sequences=pool_rows, input_ids_width=2,
                            max_decoding_length=1, pad_token_id=0)
    b = QueryBookBufferPool(num_sequences=pool_rows, input_ids_width=2,
                            max_decoding_length=1, pad_token_id=0)
    for pool in (a, b):
        pool._next_slot = pool_rows  # every slot handed out once already
    for slot in freed:
        a.free_slot(slot)
    for slot in reversed(freed):
        b.free_slot(slot)
    assert [a.allocate_slot() for _ in range(50)] == [b.allocate_slot() for _ in range(50)]


def test_lowest_free_slot_is_reused_first():
    pool = _pool()
    slots = [pool.allocate_slot() for _ in range(5)]
    pool.free_slot(slots[3])
    pool.free_slot(slots[1])
    assert pool.allocate_slot() == slots[1]
    assert pool.allocate_slot() == slots[3]
    assert pool.allocate_slot() == 5
