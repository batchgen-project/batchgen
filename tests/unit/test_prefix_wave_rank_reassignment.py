"""A selected wave may change owners after admission bound local query slots."""

import ast
import functools
import hashlib
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch

from batchgen.query_book import bind_local_sequence_to_query_book, release_local_query_slot


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"

# The worker methods are exec'd on their own, so every global they touch is
# supplied here: a fake collective (the digest check), a fake status enum, and
# the real slot release.
SequenceStatus = SimpleNamespace(QUEUEING="QUEUEING", IN_PREFILL="IN_PREFILL",
                                 COMPLETED="COMPLETED", EVICTED="EVICTED")


@functools.lru_cache(maxsize=1)
def _worker_class():
    tree = ast.parse(WORKER.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "BatchGenWorker")


def _worker_methods(*names):
    """Exec the named methods on a fresh namespace of fakes."""
    wanted = [node for node in _worker_class().body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(wanted) == len(names), [node.name for node in wanted]
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {
        "List": List,
        "hashlib": hashlib,
        "logging": logging,
        "torch": torch,
        "dist": SimpleNamespace(
            all_gather=lambda out, value: [item.copy_(value) for item in out]
        ),
        "SequenceStatus": SequenceStatus,
        "release_local_query_slot": release_local_query_slot,
    }
    exec(compile(module, str(WORKER), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _assignment_method():
    name = "_assign_ranks_for_prefix_sharing"
    return _worker_methods(name)[name]


def test_selected_wave_releases_old_owner_slots_before_prefill(monkeypatch):
    # Admission bound one sequence to each rank. The selected wave swaps them.
    monkeypatch.setitem(sys.modules, "batchgen.prefix_reuse.wave_plan",
                        SimpleNamespace(assign_ranks_for_sharing=lambda *a, **k: (1, 0)))
    assign = _assignment_method()
    for rank in (0, 1):
        seqs = {
            "a": SimpleNamespace(uuid="a", assigned_rank=0, prompt_length=3,
                                 input_ids=None, text="a",
                                 decoded_tokens=None, kv_token_budget=4),
            "b": SimpleNamespace(uuid="b", assigned_rank=1, prompt_length=3,
                                 input_ids=None, text="b",
                                 decoded_tokens=None, kv_token_budget=4),
        }
        prompts = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
        batch = SimpleNamespace(
            get_sequence=seqs.get,
            assign_rank=lambda uuid, owner: setattr(seqs[uuid], "assigned_rank", owner),
        )
        old = "a" if rank == 0 else "b"
        new = "b" if rank == 0 else "a"
        worker_state = SimpleNamespace(
            rank=rank, world_size=2, global_batch=batch,
            prefix_cache_runtime_config=SimpleNamespace(
                group_specs=[SimpleNamespace(raw_page_tokens=2)]),
            _uuid_to_local_map={old: 4}, _local_to_uuid_map={4: old},
            query_book={4: object()}, _free_local_indices=set(),
        )
        # Prompts live in the trajectory book after admission.
        worker_state._trajectory_tokens = (
            lambda seq, length=None, **_: prompts[seq.uuid][:length]
        )
        worker_state._prefix_rank_packing_mode = types.MethodType(
            _worker_methods("_prefix_rank_packing_mode")["_prefix_rank_packing_mode"],
            worker_state,
        )
        assign(worker_state, ["a", "b"], current_wave=True)
        assert worker_state._uuid_to_local_map == {}
        assert worker_state._local_to_uuid_map == {}
        assert worker_state.query_book == {}
        assert worker_state._free_local_indices == {4}
        assert seqs[new].assigned_rank == rank

        # Config prefill binds the new owner into the released local slot.
        idx, next_idx = bind_local_sequence_to_query_book(
            new, seqs[new], query_book=worker_state.query_book,
            local_to_uuid_map=worker_state._local_to_uuid_map,
            uuid_to_local_map=worker_state._uuid_to_local_map,
            free_local_indices=worker_state._free_local_indices,
            next_local_idx=5,
        )
        assert (idx, next_idx) == (4, 5)
        assert worker_state._local_to_uuid_map == {4: new}


def _legacy_must_not_run(*a, **k):
    raise AssertionError("packing mode must not call the legacy assigner")


def _install_packer(monkeypatch, ranks, recorder):
    """Stub the prefix modules: record the call and return ``ranks``.

    ``prompt_score_from_length`` is the pinned-load helper, stubbed as the
    prompt's length so a pinned entry names the length the worker scored.
    """

    def pack_wave_ranks(prompts, *, world_size, block_tokens, chunk_tokens,
                        profile, fixed=()):
        recorder["prompts"] = [list(prompt) for prompt in prompts]
        recorder["fixed"] = list(fixed)
        recorder["block_tokens"] = block_tokens
        recorder["chunk_tokens"] = chunk_tokens
        return SimpleNamespace(
            ranks=tuple(ranks), predicted=(0.0,) * world_size, round_used=3
        )

    monkeypatch.setitem(
        sys.modules, "batchgen.prefix_reuse.rank_packing",
        SimpleNamespace(
            pack_wave_ranks=pack_wave_ranks,
            prompt_score=lambda tokens, **kwargs: float(len(tokens)),
            prompt_score_from_length=lambda length, **kwargs: float(length),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "batchgen.prefix_reuse.executor_cost",
        SimpleNamespace(find_executor_cost_profile=lambda *a: None),
    )
    monkeypatch.setitem(
        sys.modules, "batchgen.prefix_reuse.wave_plan",
        SimpleNamespace(assign_ranks_for_sharing=_legacy_must_not_run),
    )


def _packing_worker(rank, seqs, slots, *, world_size=2):
    batch = SimpleNamespace(
        get_sequence=seqs.get,
        assign_rank=lambda uuid, owner: setattr(seqs[uuid], "assigned_rank", owner),
    )
    worker_state = SimpleNamespace(
        rank=rank, world_size=world_size, global_batch=batch, local_rank=rank,
        model_name="gpt-oss-120b", torch_device=torch.device("cpu"),
        # A cached profile keeps the CUDA device-name lookup out of this test.
        _prefix_rank_cost_profile=None,
        prefix_cache_runtime_config=SimpleNamespace(
            group_specs=[SimpleNamespace(raw_page_tokens=2)]),
        engine_config=SimpleNamespace(
            Module_Batching_Config=SimpleNamespace(prefill_micro_batch_token_cap=64)),
        _active_batchgen_debug_for_sequences=lambda batch_sequences: {
            "prefix_rank_assignment_mode": "packing_v2"
        },
        _uuid_to_local_map=dict(slots), _local_to_uuid_map={v: k for k, v in slots.items()},
        query_book={index: object() for index in slots.values()},
        _free_local_indices=set(),
    )
    worker_state._prefix_rank_packing_mode = types.MethodType(
        _worker_methods("_prefix_rank_packing_mode")["_prefix_rank_packing_mode"],
        worker_state,
    )
    worker_state._cached_prefix_rank_cost_profile = types.MethodType(
        _worker_methods("_cached_prefix_rank_cost_profile")[
            "_cached_prefix_rank_cost_profile"
        ],
        worker_state,
    )
    # Prompts live in the trajectory book after admission; these stubs keep
    # the tokens on input_ids and serve them through the book contract.
    worker_state._trajectory_tokens = (
        lambda seq, length=None, **_: seq.input_ids[0, :length]
    )
    return worker_state


def _mixed_wave(rank):
    """One re-entering member on rank 1 plus one queued member per rank.

    The reconstructed prompt lives in the node-shared trajectory book, so
    every rank reads the same tokens for the pinned load.
    """
    evicted = SimpleNamespace(
        uuid="e", assigned_rank=1, prompt_length=4, status=SequenceStatus.EVICTED,
        input_ids=torch.tensor([[1, 2, 3, 9]]),
        evicted_token_ids=torch.tensor([1, 2, 3, 9]) if rank == 1 else None,
    )
    return {
        "e": evicted,
        "q0": SimpleNamespace(uuid="q0", assigned_rank=0, prompt_length=3,
                              status=SequenceStatus.QUEUEING,
                              input_ids=torch.tensor([[4, 5, 6]]),
                              evicted_token_ids=None),
        "q1": SimpleNamespace(uuid="q1", assigned_rank=1, prompt_length=2,
                              status=SequenceStatus.QUEUEING,
                              input_ids=torch.tensor([[7, 8]]),
                              evicted_token_ids=None),
    }


def test_packing_mode_pins_the_evicted_member_and_repacks_only_queued(monkeypatch):
    assign = _assignment_method()
    for rank in (0, 1):
        seqs = _mixed_wave(rank)
        # q0 and q1 swap ranks; the evicted member keeps rank 1.
        recorder = {}
        _install_packer(monkeypatch, (1, 0), recorder)
        moved_away = "q0" if rank == 0 else "q1"
        worker_state = _packing_worker(rank, seqs, {moved_away: 4})
        assign(worker_state, ["e", "q0", "q1"], current_wave=True)

        assert recorder["prompts"] == [[4, 5, 6], [7, 8]]
        assert recorder["fixed"] == [(1, 4.0)]
        assert recorder["block_tokens"] == 2 and recorder["chunk_tokens"] == 64
        assert seqs["e"].assigned_rank == 1
        assert (seqs["q0"].assigned_rank, seqs["q1"].assigned_rank) == (1, 0)
        # Only the queued member that left this rank gives its slot back.
        assert worker_state._uuid_to_local_map == {}
        assert worker_state._local_to_uuid_map == {}
        assert worker_state.query_book == {}
        assert worker_state._free_local_indices == {4}


def test_packing_mode_keeps_the_evicted_members_slot(monkeypatch):
    # The evicted member owns rank 1's slot and is never reassigned, so the
    # slot must survive a wave in which a queued member also stays put.
    recorder = {}
    _install_packer(monkeypatch, (1, 1), recorder)
    seqs = _mixed_wave(1)
    worker_state = _packing_worker(1, seqs, {"e": 6, "q1": 7})
    _assignment_method()(worker_state, ["e", "q0", "q1"], current_wave=True)
    assert seqs["e"].assigned_rank == 1 and seqs["q1"].assigned_rank == 1
    assert worker_state._uuid_to_local_map == {"e": 6, "q1": 7}
    assert worker_state._free_local_indices == set()


def test_packing_mode_raises_when_ranks_disagree(monkeypatch):
    recorder = {}
    _install_packer(monkeypatch, (1, 0), recorder)
    seqs = _mixed_wave(0)
    worker_state = _packing_worker(0, seqs, {"q0": 4})
    assign = _assignment_method()

    def diverging_all_gather(out, value):
        out[0].copy_(value)
        out[1].fill_(7)  # one peer reports a different digest for this wave

    # Each extraction owns its exec namespace, so this replaces only this call's
    # collective.
    assign.__globals__["dist"] = SimpleNamespace(all_gather=diverging_all_gather)
    with pytest.raises(RuntimeError, match="diverged across ranks.*inputs differ"):
        assign(worker_state, ["e", "q0", "q1"], current_wave=True)


def test_packing_mode_names_packer_non_determinism(monkeypatch):
    # Same inputs everywhere but a peer computed another layout: the error must
    # blame the packer, not the synced metadata.
    _install_packer(monkeypatch, (1, 0), {})
    seqs = _mixed_wave(0)
    worker_state = _packing_worker(0, seqs, {"q0": 4})
    assign = _assignment_method()

    def digest_only_all_gather(out, value):
        out[0].copy_(value)
        out[1].copy_(value)
        out[1][1] += 1  # identical input fingerprint, different digest

    assign.__globals__["dist"] = SimpleNamespace(all_gather=digest_only_all_gather)
    with pytest.raises(RuntimeError, match="identical inputs: packer non-determinism"):
        assign(worker_state, ["e", "q0", "q1"], current_wave=True)


def test_receipt_packing_pins_only_sequences_that_owe_a_prefill(monkeypatch):
    # An earlier batch left a completed sequence on rank 0 and a queued one on
    # rank 1. Only the queued one still owes rank 1 a prefill.
    recorder = {}
    _install_packer(monkeypatch, (0,), recorder)
    seqs = {
        "done": SimpleNamespace(uuid="done", assigned_rank=0, prompt_length=5,
                                status=SequenceStatus.COMPLETED,
                                input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                                evicted_token_ids=None),
        "wait": SimpleNamespace(uuid="wait", assigned_rank=1, prompt_length=2,
                                status=SequenceStatus.QUEUEING,
                                input_ids=torch.tensor([[6, 7]]),
                                evicted_token_ids=None),
        "new": SimpleNamespace(uuid="new", assigned_rank=None, prompt_length=3,
                               status=SequenceStatus.QUEUEING,
                               input_ids=torch.tensor([[8, 9, 10]]),
                               evicted_token_ids=None),
    }
    worker_state = _packing_worker(0, seqs, {})
    iterable = type("Batch", (SimpleNamespace,), {"__iter__": lambda self: iter(seqs.values())})
    worker_state.global_batch = iterable(**vars(worker_state.global_batch))
    _assignment_method()(worker_state, ["new"])
    assert recorder["fixed"] == [(1, 2.0)]
    assert recorder["prompts"] == [[8, 9, 10]]


def test_default_mode_still_uses_the_legacy_assigner(monkeypatch):
    calls = []

    def assign_ranks_for_sharing(prompts, **kwargs):
        calls.append((list(prompts), kwargs))
        return (0, 1)

    recorder = {}
    _install_packer(monkeypatch, (1, 1), recorder)
    monkeypatch.setitem(sys.modules, "batchgen.prefix_reuse.wave_plan",
                        SimpleNamespace(assign_ranks_for_sharing=assign_ranks_for_sharing))
    seqs = _mixed_wave(0)
    worker_state = _packing_worker(0, seqs, {"q0": 4})
    worker_state._active_batchgen_debug_for_sequences = lambda batch_sequences: None
    _assignment_method()(worker_state, ["q0", "q1"], current_wave=True)
    assert len(calls) == 1 and calls[0][0] == [[4, 5, 6], [7, 8]]
    assert calls[0][1]["initial_loads"] == [0, 0]
    assert recorder == {}  # the packer was never consulted
    assert (seqs["q0"].assigned_rank, seqs["q1"].assigned_rank) == (0, 1)
