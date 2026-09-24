"""Pool-admission tokenization must move compact token tensors, and must
serialize tokenizer execution across ranks for very large admissions."""

import ast
import logging
import pickle
import types
import weakref
from pathlib import Path
from typing import List

import pytest
import torch


WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
WORKER_TREE = ast.parse(WORKER.read_text(), filename=str(WORKER))


def _module_nodes(*names):
    """Module-level constants and functions, in source order."""
    return [
        node
        for node in WORKER_TREE.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            )
        )
    ]


def _exec_nodes(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace


def _load_tokenize_method():
    """Load the real method without importing the full inference engine."""
    worker_class = next(
        node
        for node in WORKER_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node
        for node in worker_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_tokenize_admitted_sequences"
    )
    namespace = _exec_nodes(
        _module_nodes("_TOKENIZER_SERIALIZE_CHARS_PER_RANK", "_malloc_trim") + [method],
        {
            "List": List,
            "ctypes": types.SimpleNamespace(),
            "dist": types.SimpleNamespace(),
            "logging": logging,
            "torch": torch,
        },
    )
    return namespace["_tokenize_admitted_sequences"], namespace


TOKENIZE_ADMITTED, TOKENIZE_NAMESPACE = _load_tokenize_method()
FAKE_DIST = TOKENIZE_NAMESPACE["dist"]


def _load_module_function(name):
    """Load a module-level helper without importing the full inference engine."""
    namespace = _exec_nodes(
        _module_nodes("_TOKENIZER_TRIM_TOKENS_PER_RANK", "_malloc_trim", name),
        {"ctypes": types.SimpleNamespace()},
    )
    return namespace[name], namespace


RELEASE_ARENAS, ARENAS_NAMESPACE = _load_module_function("_release_tokenizer_arenas")


def install_fake_ctypes(monkeypatch, cdll):
    monkeypatch.setitem(ARENAS_NAMESPACE, "ctypes", types.SimpleNamespace(CDLL=cdll))


def encode(text):
    return [ord(ch) for ch in text]


class FakeTokenizer:
    def __init__(self):
        self.kwargs_seen = []

    def __call__(self, texts, **kwargs):
        self.kwargs_seen.append(kwargs)
        return {"input_ids": [encode(text) for text in texts]}


class FakeSequence:
    def __init__(self, uuid, text, max_decode_length):
        self.uuid = uuid
        self.text = text
        self.max_decode_length = max_decode_length
        self.prompt_length = 0
        self.batch_id = None
        self._buffer_slot = -1


class FakeSequenceBatch:
    def __init__(self, sequences):
        self.sequences = {seq.uuid: seq for seq in sequences}

    def get_sequence(self, uuid):
        return self.sequences.get(uuid)

    def remove_sequence(self, uuid):
        self.sequences.pop(uuid, None)


class FakeBufferPool:
    def __init__(self, rows, input_width, decode_width):
        self.input_ids = torch.zeros((rows, input_width), dtype=torch.int64)
        self.decoded_tokens = torch.zeros((rows, decode_width), dtype=torch.int64)
        self.next_slot = 0
        self.freed_slots = []

    def allocate_slot(self):
        slot = self.next_slot
        self.next_slot += 1
        return slot

    def free_slot(self, slot):
        self.freed_slots.append(slot)

    def get_input_ids_view(self, slot, width):
        return self.input_ids[slot : slot + 1, :width]

    def get_decoded_tokens_view(self, slot):
        return self.decoded_tokens[slot : slot + 1]


class FakeResponseQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def make_worker(
    texts,
    *,
    rank=0,
    world_size=1,
    model_context_length=4096,
    max_decode_length=8,
    pool_width=None,
):
    sequences = [
        FakeSequence(f"req-{index}", text, max_decode_length)
        for index, text in enumerate(texts)
    ]
    if pool_width is None:
        pool_width = max(len(text) for text in texts) + max_decode_length
    worker = types.SimpleNamespace(
        rank=rank,
        world_size=world_size,
        model_context_length=model_context_length,
        tokenizer=FakeTokenizer(),
        _response_queue=FakeResponseQueue(),
        _max_pool_size=len(texts),
        global_batch=FakeSequenceBatch(sequences),
        _buffer_pool=FakeBufferPool(len(texts), pool_width, max_decode_length),
        _ensure_buffer_pool=lambda **kwargs: None,
    )
    uuids = [seq.uuid for seq in sequences]
    return worker, uuids


def install_fake_collective(monkeypatch, worker, sent, other_rank_payloads=None):
    other_rank_payloads = other_rank_payloads or {}

    def fake_all_gather_object(out_list, obj):
        sent.append(obj)
        for rank in range(worker.world_size):
            payload = obj if rank == worker.rank else other_rank_payloads.get(rank, [])
            out_list[rank] = pickle.loads(pickle.dumps(payload))

    monkeypatch.setattr(
        FAKE_DIST, "all_gather_object", fake_all_gather_object, raising=False
    )


def strided_payload(texts, rank, world_size):
    """The compact payload a remote rank owns under the rank-strided split."""
    return [
        {
            "idx": index,
            "input_ids": torch.tensor(encode(texts[index]), dtype=torch.int64),
            "length": len(texts[index]),
        }
        for index in range(rank, len(texts), world_size)
    ]


def assert_compact_payload(payload):
    for item in payload:
        token_ids = item["input_ids"]
        assert isinstance(token_ids, torch.Tensor)
        assert token_ids.dtype == torch.int64
        assert token_ids.device.type == "cpu"
        assert token_ids.is_contiguous()
        assert token_ids.dim() == 1
        assert token_ids.numel() == item["length"]


def run_tokenize(worker, uuids):
    TOKENIZE_ADMITTED(worker, uuids)


def test_gathered_payload_is_cpu_int64_tensors_and_fills_pool(monkeypatch):
    texts = ["hello", "hi", "batchgen"]
    worker, uuids = make_worker(texts)
    sent = []
    install_fake_collective(monkeypatch, worker, sent)

    run_tokenize(worker, uuids)

    assert len(sent) == 1
    assert_compact_payload(sent[0])
    assert [item["idx"] for item in sent[0]] == [0, 1, 2]
    for uuid, text in zip(uuids, texts):
        seq = worker.global_batch.get_sequence(uuid)
        expected = encode(text)
        assert seq.prompt_length == len(expected)
        assert seq.original_prompt_length == len(expected)
        assert seq.current_context_length == len(expected)
        assert seq.kv_token_budget == len(expected) + seq.max_decode_length
        assert seq.input_ids[0, : len(expected)].tolist() == expected
        assert seq.input_ids[0, len(expected) :].tolist() == [0] * seq.max_decode_length
    kwargs = worker.tokenizer.kwargs_seen[0]
    assert kwargs["return_tensors"] is None
    assert kwargs["padding"] is False


@pytest.mark.parametrize(
    ("rank", "local_indices"),
    [(0, [0, 2, 4]), (1, [1, 3])],
)
def test_multi_rank_gather_preserves_global_ordering(
    monkeypatch, rank, local_indices
):
    texts = ["aaaa", "bb", "ccc", "d", "eeeee"]
    worker, uuids = make_worker(texts, rank=rank, world_size=2)
    remote_rank = 1 - rank
    remote_indices = [index for index in range(len(texts)) if index not in local_indices]
    remote_payload = [
        {
            "idx": index,
            "input_ids": torch.tensor(encode(texts[index]), dtype=torch.int64),
            "length": len(texts[index]),
        }
        for index in remote_indices
    ]
    sent = []
    install_fake_collective(monkeypatch, worker, sent, {remote_rank: remote_payload})

    run_tokenize(worker, uuids)

    assert [item["idx"] for item in sent[0]] == local_indices
    assert_compact_payload(sent[0])
    for uuid, text in zip(uuids, texts):
        seq = worker.global_batch.get_sequence(uuid)
        assert seq.input_ids[0, : len(text)].tolist() == encode(text)
    slots = [worker.global_batch.get_sequence(uuid)._buffer_slot for uuid in uuids]
    assert slots == list(range(len(uuids)))


def test_over_context_sequence_is_rejected_and_others_still_fill(monkeypatch):
    texts = ["abc", "toolongprompt", "de"]
    worker, uuids = make_worker(
        texts, model_context_length=8, max_decode_length=4, pool_width=8
    )
    sent = []
    install_fake_collective(monkeypatch, worker, sent)

    run_tokenize(worker, uuids)

    assert_compact_payload(sent[0])
    assert worker.global_batch.get_sequence(uuids[1]) is None
    errors = [item for item in worker._response_queue.items if item.get("error")]
    assert len(errors) == 1
    assert errors[0]["request_id"] == uuids[1]
    assert errors[0]["error"]["code"] == "context_length_exceeded"
    for uuid, text in ((uuids[0], texts[0]), (uuids[2], texts[2])):
        seq = worker.global_batch.get_sequence(uuid)
        assert seq.input_ids[0, : len(text)].tolist() == encode(text)


def test_tokenizer_lists_are_released_before_collective(monkeypatch):
    worker, uuids = make_worker(["alpha", "beta"])
    raw_refs = []

    class TrackedList(list):
        pass

    class TrackingTokenizer(FakeTokenizer):
        def __call__(self, texts, **kwargs):
            result = super().__call__(texts, **kwargs)
            result["input_ids"] = [
                TrackedList(token_ids) for token_ids in result["input_ids"]
            ]
            raw_refs.extend(weakref.ref(token_ids) for token_ids in result["input_ids"])
            return result

    worker.tokenizer = TrackingTokenizer()
    alive_at_gather = []

    def record(out_list, obj):
        alive_at_gather.append([ref() is not None for ref in raw_refs])
        out_list[0] = obj

    monkeypatch.setattr(FAKE_DIST, "all_gather_object", record, raising=False)

    run_tokenize(worker, uuids)

    assert raw_refs
    assert alive_at_gather == [[False] * len(raw_refs)]


@pytest.mark.parametrize(
    ("rank", "expected_events"),
    [
        (0, ["tokenize", "trim", "broadcast-0", "broadcast-1"]),
        (1, ["broadcast-0", "tokenize", "trim", "broadcast-1"]),
    ],
)
def test_large_admission_tokenizes_one_source_rank_at_a_time(
    monkeypatch, rank, expected_events
):
    texts = ["aaaa", "bb", "ccc", "d", "eeeee"]
    worker, uuids = make_worker(texts, rank=rank, world_size=2)
    # 15 chars over 2 ranks: forces the serialized path in a small test.
    monkeypatch.setitem(TOKENIZE_NAMESPACE, "_TOKENIZER_SERIALIZE_CHARS_PER_RANK", 7)
    events = []
    sent = []

    class RecordingTokenizer(FakeTokenizer):
        def __call__(self, texts, **kwargs):
            events.append("tokenize")
            return super().__call__(texts, **kwargs)

    worker.tokenizer = RecordingTokenizer()

    def fake_broadcast_object_list(object_list, src):
        assert len(object_list) == 1
        assert (object_list[0] is not None) == (src == worker.rank)
        events.append(f"broadcast-{src}")
        if src == worker.rank:
            sent.append(object_list[0])
        else:
            object_list[0] = pickle.loads(
                pickle.dumps(strided_payload(texts, src, worker.world_size))
            )

    def forbidden_all_gather(out_list, obj):
        raise AssertionError("the serialized path must not all_gather")

    monkeypatch.setattr(
        FAKE_DIST, "broadcast_object_list", fake_broadcast_object_list, raising=False
    )
    monkeypatch.setattr(
        FAKE_DIST, "all_gather_object", forbidden_all_gather, raising=False
    )
    monkeypatch.setitem(
        TOKENIZE_NAMESPACE, "_malloc_trim", lambda: events.append("trim")
    )

    run_tokenize(worker, uuids)

    # Identical ascending source order on both ranks, each rank tokenizes only
    # at its own turn, and trims its freed arenas before the next source runs.
    assert events == expected_events
    assert len(sent) == 1
    assert_compact_payload(sent[0])
    assert [item["idx"] for item in sent[0]] == list(range(rank, len(texts), 2))
    for uuid, text in zip(uuids, texts):
        seq = worker.global_batch.get_sequence(uuid)
        assert seq.prompt_length == len(text)
        assert seq.input_ids[0, : len(text)].tolist() == encode(text)
    slots = [worker.global_batch.get_sequence(uuid)._buffer_slot for uuid in uuids]
    assert slots == list(range(len(uuids)))


@pytest.mark.parametrize(
    ("world_size", "expect_serialized"),
    [(1, True), (2, False)],
)
def test_serialize_gate_is_inclusive_and_scales_with_world_size(
    monkeypatch, world_size, expect_serialized
):
    texts = ["aaaa", "bb", "ccc", "d", "eeeee"]  # 15 chars
    worker, uuids = make_worker(texts, world_size=world_size)
    # world_size=1 assigns all 15 chars to one rank; world_size=2 assigns at
    # most 12, so only world_size=1 reaches the per-rank threshold.
    monkeypatch.setitem(TOKENIZE_NAMESPACE, "_TOKENIZER_SERIALIZE_CHARS_PER_RANK", 15)
    used = []

    def fake_broadcast_object_list(object_list, src):
        used.append("broadcast")

    def fake_all_gather_object(out_list, obj):
        used.append("all_gather")
        for other in range(world_size):
            out_list[other] = (
                obj if other == worker.rank else strided_payload(texts, other, world_size)
            )

    monkeypatch.setattr(
        FAKE_DIST, "broadcast_object_list", fake_broadcast_object_list, raising=False
    )
    monkeypatch.setattr(
        FAKE_DIST, "all_gather_object", fake_all_gather_object, raising=False
    )

    run_tokenize(worker, uuids)

    assert used == (["broadcast"] if expect_serialized else ["all_gather"])
    for uuid, text in zip(uuids, texts):
        seq = worker.global_batch.get_sequence(uuid)
        assert seq.input_ids[0, : len(text)].tolist() == encode(text)


def test_serialized_path_stops_sources_at_sequence_count(monkeypatch):
    texts = ["alpha", "beta"]
    worker, uuids = make_worker(texts, rank=3, world_size=4)
    monkeypatch.setitem(TOKENIZE_NAMESPACE, "_TOKENIZER_SERIALIZE_CHARS_PER_RANK", 1)
    sources = []

    def fake_broadcast_object_list(object_list, src):
        assert object_list == [None]
        sources.append(src)
        object_list[0] = strided_payload(texts, src, worker.world_size)

    monkeypatch.setattr(
        FAKE_DIST, "broadcast_object_list", fake_broadcast_object_list, raising=False
    )
    monkeypatch.setattr(
        FAKE_DIST,
        "all_gather_object",
        lambda *_: (_ for _ in ()).throw(AssertionError("unexpected all_gather")),
        raising=False,
    )

    run_tokenize(worker, uuids)

    assert sources == [0, 1]
    assert worker.tokenizer.kwargs_seen == []
    for uuid, text in zip(uuids, texts):
        seq = worker.global_batch.get_sequence(uuid)
        assert seq.input_ids[0, : len(text)].tolist() == encode(text)


def test_tokenizer_lists_are_released_before_broadcast(monkeypatch):
    worker, uuids = make_worker(["alpha", "beta"], world_size=1)
    monkeypatch.setitem(TOKENIZE_NAMESPACE, "_TOKENIZER_SERIALIZE_CHARS_PER_RANK", 1)
    raw_refs = []

    class TrackedList(list):
        pass

    class TrackingTokenizer(FakeTokenizer):
        def __call__(self, texts, **kwargs):
            result = super().__call__(texts, **kwargs)
            result["input_ids"] = [
                TrackedList(token_ids) for token_ids in result["input_ids"]
            ]
            raw_refs.extend(weakref.ref(token_ids) for token_ids in result["input_ids"])
            return result

    worker.tokenizer = TrackingTokenizer()
    alive_at_broadcast = []

    def record(object_list, src):
        alive_at_broadcast.append([ref() is not None for ref in raw_refs])

    monkeypatch.setattr(FAKE_DIST, "broadcast_object_list", record, raising=False)

    run_tokenize(worker, uuids)

    assert raw_refs
    assert alive_at_broadcast == [[False] * len(raw_refs)]


def test_release_tokenizer_arenas_trims_the_process_heap(monkeypatch):
    handles = []

    class FakeMallocTrim:
        def __init__(self):
            self.args = []
            self.argtypes = None
            self.restype = None

        def __call__(self, pad):
            self.args.append(pad)
            return 1

    class FakeLibc:
        def __init__(self):
            self.malloc_trim = FakeMallocTrim()

    libc = FakeLibc()

    def fake_cdll(handle):
        handles.append(handle)
        return libc

    fake_ctypes = types.SimpleNamespace(
        CDLL=fake_cdll,
        c_size_t=object(),
        c_int=object(),
    )
    monkeypatch.setitem(ARENAS_NAMESPACE, "ctypes", fake_ctypes)

    RELEASE_ARENAS(8 * (1 << 20), 8)

    assert handles == [None]
    assert libc.malloc_trim.args == [0]
    assert libc.malloc_trim.argtypes == [fake_ctypes.c_size_t]
    assert libc.malloc_trim.restype is fake_ctypes.c_int


def test_release_tokenizer_arenas_skips_small_admission(monkeypatch):
    calls = []
    install_fake_ctypes(monkeypatch, lambda handle: calls.append(handle))

    RELEASE_ARENAS(8 * (1 << 20) - 1, 8)

    assert calls == []


def test_release_tokenizer_arenas_is_noop_without_malloc_trim(monkeypatch):
    install_fake_ctypes(monkeypatch, lambda handle: types.SimpleNamespace())

    RELEASE_ARENAS(1 << 20, 1)


def test_release_tokenizer_arenas_is_noop_when_libc_cannot_be_opened(monkeypatch):
    def failing_cdll(handle):
        raise OSError("no such library")

    install_fake_ctypes(monkeypatch, failing_cdll)

    RELEASE_ARENAS(1 << 20, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
