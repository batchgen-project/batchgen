"""Worker admission wiring for the node-local prompt arena."""

import ast
import types
from pathlib import Path
from typing import Dict, Sequence, Set

import torch

from batchgen.token_store import PromptTokenArena


WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _load_publish_method():
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker")
    method = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "_publish_prompt_arena")
    namespace = {
        "Dict": Dict,
        "Sequence": Sequence,
        "Set": Set,
        "PromptTokenArena": PromptTokenArena,
        "torch": torch,
        "dist": types.SimpleNamespace(),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(WORKER), "exec"), namespace)
    return namespace["_publish_prompt_arena"], namespace["dist"]


PUBLISH, DIST = _load_publish_method()


class _Sequence:
    def __init__(self, uuid):
        self.uuid = uuid


def _tokenized(*pairs):
    return {
        index: {"input_ids": torch.tensor(tokens, dtype=torch.int64), "length": len(tokens)}
        for index, tokens in enumerate(pairs)
    }


def test_leader_publishes_int32_handles_and_reader_attaches():
    leader = types.SimpleNamespace(
        local_rank=0,
        rank=0,
        world_size=2,
        local_world_size=2,
        _query_book_shm_prefix="test_prompt_arena",
        _prompt_arena=None,
        _prompt_handles={},
        _prompt_tensors={},
    )
    sequences = [_Sequence("a"), _Sequence("b")]
    tokenized = _tokenized([1, 2, 3], [7, 8])
    published = []

    def gather_leader(outputs, payload):
        outputs[:] = [payload, None]
        published.append(payload)

    DIST.all_gather_object = gather_leader
    try:
        PUBLISH(leader, sequences, tokenized, set())
        endpoint = published[0][0]
        handles = published[0][1]
        assert leader._prompt_arena.is_creator
        assert leader._prompt_tensors["a"].dtype == torch.int32
        assert leader._prompt_tensors["a"].tolist() == [[1, 2, 3]]

        reader = types.SimpleNamespace(
            local_rank=1,
            rank=1,
            world_size=2,
            local_world_size=2,
            _query_book_shm_prefix="test_prompt_arena",
            _prompt_arena=None,
            _prompt_handles={},
            _prompt_tensors={},
        )
        reader._owns_local_sequence = lambda seq: seq.uuid == "a"

        def gather_reader(outputs, payload):
            outputs[:] = [(endpoint, handles), None]

        DIST.all_gather_object = gather_reader
        PUBLISH(reader, sequences, tokenized, set())
        assert not reader._prompt_arena.is_creator
        assert reader._prompt_tensors["a"].tolist() == [[1, 2, 3]]
        assert "b" not in reader._prompt_tensors
    finally:
        if getattr(reader, "_prompt_arena", None) is not None:
            reader._prompt_arena.close()
        if getattr(leader, "_prompt_arena", None) is not None:
            leader._prompt_arena.unlink()
