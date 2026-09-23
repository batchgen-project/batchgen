"""A selected wave may change owners after admission bound local query slots."""

import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch

from batchgen.query_book import bind_local_sequence_to_query_book, release_local_query_slot


WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


def _assignment_method():
    tree = ast.parse(WORKER.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "BatchGenWorker")
    method = next(node for node in worker.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_assign_ranks_for_prefix_sharing")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"List": List, "logging": logging,
                 "release_local_query_slot": release_local_query_slot}
    exec(compile(module, str(WORKER), "exec"), namespace)
    return namespace[method.name]


def test_selected_wave_releases_old_owner_slots_before_prefill(monkeypatch):
    # Admission bound one sequence to each rank. The selected wave swaps them.
    monkeypatch.setitem(sys.modules, "batchgen.prefix_reuse.wave_plan",
                        SimpleNamespace(assign_ranks_for_sharing=lambda *a, **k: (1, 0)))
    assign = _assignment_method()
    for rank in (0, 1):
        seqs = {
            "a": SimpleNamespace(uuid="a", assigned_rank=0, prompt_length=3,
                                 input_ids=torch.tensor([[1, 2, 3]]), text="a",
                                 decoded_tokens=torch.empty((1, 1)), kv_token_budget=4),
            "b": SimpleNamespace(uuid="b", assigned_rank=1, prompt_length=3,
                                 input_ids=torch.tensor([[4, 5, 6]]), text="b",
                                 decoded_tokens=torch.empty((1, 1)), kv_token_budget=4),
        }
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
