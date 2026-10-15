"""Prefix-grouped wave order: queueing candidates admit in ``prefill_order``.

Two halves, both CPU-only:
  - ``PrefillScheduler.select_prefill_batch`` honours ``prefill_order`` and
    falls back to arrival ``global_idx`` when it is unset.
  - ``BatchGenWorker._stamp_prefix_prefill_order`` stamps that order from the
    prompt token ids, and only for a ``packing_v2`` batch.

Capacity math (page_size=64, initial_gpu_page_buffer=32, chunk=128): a
prompt_length=100 candidate reserves ceil(max(100+128, (ceil(101/64)+32)*64)
/ 64) = 34 pages, so a 102-page node admits exactly three of them.
"""

import ast
import functools
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch


_MODULE_PATH = Path(__file__).resolve().parents[2] / "batchgen" / "worker" / "prefill.py"
_SPEC = importlib.util.spec_from_file_location("test_prefill_wave_order_mod", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PrefillCandidate = _MODULE.PrefillCandidate
PrefillScheduler = _MODULE.PrefillScheduler
PrefillSelectionRequest = _MODULE.PrefillSelectionRequest

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"

_PAGE = 64
_BUF = 32
_PAGES_PER_SEQ = 34


def _cand(uuid, *, gidx, order=None, evicted=False, decoded=0):
    return PrefillCandidate(
        uuid=uuid,
        assigned_rank=0,
        node_id=0,
        is_evicted=evicted,
        global_idx=gidx,
        total_decoded_before_eviction=decoded,
        prompt_length=100,
        kv_token_budget=100000,
        page_size=_PAGE,
        prefill_order=order,
    )


def _select(candidates, *, seats):
    return PrefillScheduler.select_prefill_batch(PrefillSelectionRequest(
        candidates=tuple(candidates),
        per_node_host_free=(seats * _PAGES_PER_SEQ,),
        chunk_size=128,
        num_nodes=1,
        gpus_per_node=8,
        initial_gpu_page_buffer=_BUF,
    ))


# Arrival interleaves two prefix families; the grouped order separates them.
_ARRIVAL = ["a0", "b0", "a1", "b1", "a2", "b2"]
_GROUPED = {"a0": 0, "a1": 1, "a2": 2, "b0": 3, "b1": 4, "b2": 5}


def test_unset_prefill_order_keeps_arrival_order():
    candidates = [_cand(uuid, gidx=gidx) for gidx, uuid in enumerate(_ARRIVAL)]
    # Shuffled input: selection sorts, so the snapshot order must not matter.
    assert _select(candidates[::-1], seats=6) == _ARRIVAL
    assert _select(candidates, seats=3) == _ARRIVAL[:3]


def test_prefill_order_fills_the_first_wave_with_one_prefix_family():
    candidates = [
        _cand(uuid, gidx=gidx, order=_GROUPED[uuid])
        for gidx, uuid in enumerate(_ARRIVAL)
    ]
    assert _select(candidates, seats=3) == ["a0", "a1", "a2"]
    assert _select(candidates, seats=6) == ["a0", "a1", "a2", "b0", "b1", "b2"]


def test_evicted_candidates_still_precede_ordered_queueing():
    # The evicted member arrived last and would sort last by any queue order.
    evicted = _cand("e", gidx=9, decoded=5, evicted=True)
    queued = [
        _cand(uuid, gidx=gidx, order=_GROUPED[uuid])
        for gidx, uuid in enumerate(_ARRIVAL)
    ]
    assert _select([*queued, evicted], seats=3) == ["e", "a0", "a1"]


@functools.lru_cache(maxsize=1)
def _worker_methods():
    """Exec the stamping method (plus the mode predicate it calls) alone."""
    tree = ast.parse(WORKER.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "BatchGenWorker")
    names = ("_stamp_prefix_prefill_order", "_prefix_rank_packing_mode")
    wanted = [node for node in cls.body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(wanted) == len(names), [node.name for node in wanted]
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {"List": List, "logging": logging}
    exec(compile(module, str(WORKER), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _seq(uuid, gidx, tokens):
    return SimpleNamespace(
        uuid=uuid, global_idx=gidx, prompt_length=len(tokens),
        input_ids=torch.tensor([tokens]),
    )


def _stamping_worker(seqs, *, mode):
    methods = _worker_methods()
    worker_state = SimpleNamespace(
        rank=0,
        global_batch=SimpleNamespace(get_sequence=seqs.get),
        _active_batchgen_debug_for_sequences=lambda batch_sequences: (
            {"prefix_rank_assignment_mode": mode} if mode else None
        ),
    )
    worker_state._prefix_rank_packing_mode = types.MethodType(
        methods["_prefix_rank_packing_mode"], worker_state
    )
    worker_state._stamp_prefix_prefill_order = types.MethodType(
        methods["_stamp_prefix_prefill_order"], worker_state
    )
    # Prompts are served through the trajectory-book contract.
    worker_state._trajectory_tokens = (
        lambda seq, length=None, **_: seq.input_ids[0, :length]
    )
    return worker_state


def _family_admission():
    """Two prefix families admitted interleaved, one duplicate prompt."""
    return {
        "a0": _seq("a0", 10, [1, 1, 7]),
        "b0": _seq("b0", 11, [2, 4]),
        "a1": _seq("a1", 12, [1, 1, 8]),
        "b1": _seq("b1", 13, [2, 4]),  # ties with b0 on tokens
        "a2": _seq("a2", 14, [1, 0]),  # shorter prefix sibling sorts first
    }


def test_stamping_groups_by_prompt_prefix_and_ties_by_global_idx():
    seqs = _family_admission()
    worker_state = _stamping_worker(seqs, mode="packing_v2")
    worker_state._stamp_prefix_prefill_order(list(seqs))
    # Lexicographic on token ids: [1,0] < [1,1,7] < [1,1,8] < [2,4] (b0, b1).
    assert [seq.uuid for seq in sorted(seqs.values(), key=lambda s: s.prefill_order)] \
        == ["a2", "a0", "a1", "b0", "b1"]
    # Base is this admission's smallest global_idx, so a later admission
    # (larger global_idx) still orders after it.
    assert min(seq.prefill_order for seq in seqs.values()) == 10


def test_stamping_is_skipped_outside_packing_mode():
    for mode in (None, "legacy"):
        seqs = _family_admission()
        _stamping_worker(seqs, mode=mode)._stamp_prefix_prefill_order(list(seqs))
        assert all(not hasattr(seq, "prefill_order") for seq in seqs.values())


def test_stamping_ignores_uuids_with_no_sequence():
    seqs = _family_admission()
    worker_state = _stamping_worker(seqs, mode="packing_v2")
    worker_state._stamp_prefix_prefill_order([*seqs, "gone"])
    assert sorted(seq.prefill_order for seq in seqs.values()) == [10, 11, 12, 13, 14]


def test_sequence_entry_carries_the_stamped_order():
    # The stamp lands on a real SequenceEntry, which is __slots__-based: the
    # slot must exist or the admission path raises AttributeError. Unset reads
    # stay AttributeError, which is what the candidate's getattr default takes.
    from batchgen.sequence import SequenceEntry

    seq = SequenceEntry(uuid="s", global_idx=0, prompt_length=3,
                        max_decode_length=1, text="s")
    assert getattr(seq, "prefill_order", None) is None
    seq.prefill_order = 7
    assert seq.prefill_order == 7
