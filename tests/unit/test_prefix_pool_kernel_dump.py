"""``batchgen_debug.prefix_pool_dump_kernel_args``: the dump and its schema.

The probe records, at the one seam where the pooled prefill both writes the
pool and reads it back (``pool_prefill_attention``), the kernel arguments plus
a digest of every pool page the page table addresses and of every row's source
K, and - schema 2 - the same source split at the pool's PAGE boundaries so
write->read byte identity is checkable per page instead of only on a row that
happens to own exactly one full page. Pinned here: the mode parser, the dump
directory rule, the two-layer limit, the file schema round-tripped through
``read_dump`` with every digest recomputed independently, the slot-run split
itself, that a clean write matches on every page and a wrong write is caught
on exactly the pages it damaged, and that nothing happens with the flag off.

CPU only: the pool and the executor are the real classes, loaded by file path,
and ``gqa_extend_fa`` is stubbed because the paged kernel is GPU work.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch


def _load(name, *relative):
    path = Path(__file__).resolve().parents[2].joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLAN = _load("test_kdump_plan_mod", "batchgen", "prefix_reuse", "wave_plan.py")
_POOL = _load("test_kdump_pool_mod", "batchgen", "prefix_reuse", "pool.py")
_EXEC = _load("test_kdump_exec_mod", "batchgen", "prefix_reuse", "executor.py")
_DUMP = _load("test_kdump_mod", "batchgen", "prefix_reuse", "kernel_dump.py")

B = 4  # block tokens == page tokens
HEADS = 2
K_DIM = 6
V_DIM = 3
GLOBAL_IDS = [500, 501, 502, 503, 504]


def blk(index: int) -> list[int]:
    return [index * 10 + offset for offset in range(B)]


# A nested chain: five members, so the chunk holds segment rows AND tail rows,
# ancestor counts differ, and the deepest row reads several pages.
PROMPTS = [
    [token for index in range(depth + 1) for token in blk(index)]
    for depth in range(1, 6)
]

# The same chain with a per-member TEMPLATE SUFFIX, the shape of the live
# gapped wave: no prompt length is a multiple of the page, so a member
# diverges inside a block and every tail ends on a PARTIALLY filled page. In
# the live wave exactly one row out of 126 owned a whole page, which is why a
# whole-row digest was comparable almost nowhere.
TEMPLATE = [9000, 9001, 9002]
PROMPTS_RAGGED = [prompt + TEMPLATE for prompt in PROMPTS]


def _digest(tensor: torch.Tensor) -> str:
    """The oracle digest, recomputed here instead of trusting the module."""

    return hashlib.blake2b(
        tensor.detach().contiguous().numpy().tobytes(),
        digest_size=_DUMP.DIGEST_BYTES,
    ).hexdigest()


@pytest.fixture
def stub_kernel(monkeypatch):
    """Serve the paged FA3 entry point; it is GPU work."""

    calls = []

    def fake_extend(**kwargs):
        calls.append(kwargs)
        rows = kwargs["q"].shape[0]
        return kwargs["q"].new_zeros(rows, HEADS, V_DIM), None

    module = types.ModuleType("batchgen.attention.gqa")
    module.gqa_extend_fa = fake_extend
    monkeypatch.setitem(sys.modules, "batchgen.attention.gqa", module)
    return calls


def _wave(num_layers=1, prompts=PROMPTS):
    plan = _PLAN.plan_wave_prefix_sharing(
        prompts, block_tokens=B, pool_pages=64, chunk_tokens=4096,
        same_chunk_ancestors=True,
    )
    _PLAN.validate_wave_prefix_plan(plan, prompts)
    pool = _POOL.PrefillPrefixPool(
        num_layers=num_layers, num_pages=plan.pool_pages + 32, page_tokens=B,
        num_kv_heads=HEADS, head_dim=K_DIM, dtype=torch.float32, device="cpu",
        v_head_dim=V_DIM,
    )
    pool.k.zero_()
    pool.v.zero_()
    return plan, pool


def _attend(pool, batch, layer_idx, seed=0):
    """Run the production write+read seam with deterministic K/V."""

    tokens = int(batch.input_ids.numel())
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(tokens, HEADS, K_DIM, generator=generator)
    key = torch.randn(tokens, HEADS, K_DIM, generator=generator)
    value = torch.randn(tokens, HEADS, V_DIM, generator=generator)
    _EXEC.pool_prefill_attention(
        pool=pool, layer_idx=layer_idx, batch=batch, query=query, key=key,
        value=value, sinks=None, softmax_scale=0.5, sliding_window=None,
    )
    return key


# --------------------------------------------------------------------------- #
#  (i) The flag: both modes, off, and a refused value
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    ("sync", "sync"),
    ("SYNC", "sync"),
    (" nosync ", "nosync"),
    ("nosync", "nosync"),
    (True, "sync"),
    ("1", "sync"),
    ("on", "sync"),
    (None, None),
    (False, None),
    ("", None),
    ("0", None),
    ("off", None),
])
def test_the_mode_parser_accepts_exactly_the_documented_values(value, expected):
    assert _DUMP.parse_dump_mode(value) == expected


@pytest.mark.parametrize("value", ["synch", "async", "yes please", "2sync", []])
def test_an_unsupported_mode_is_refused_by_name(value):
    """A mistyped mode must not read as a clean run."""

    with pytest.raises(ValueError, match="prefix_pool_dump_kernel_args"):
        _DUMP.parse_dump_mode(value)


def test_the_dump_directory_is_per_rank_under_dev_shm_by_default():
    assert _DUMP.dump_directory(None, 3) == "/dev/shm/pool_dump_rank3"
    assert _DUMP.dump_directory("", 0) == "/dev/shm/pool_dump_rank0"
    assert _DUMP.dump_directory("/data0/probe", 11) == \
        "/data0/probe/pool_dump_rank11"


@pytest.mark.parametrize("layers,expected", [
    (1, (0,)), (2, (0, 1)), (7, (0, 3)), (24, (0, 12)), (27, (0, 13)),
])
def test_only_the_first_and_a_middle_pool_layer_are_dumped(layers, expected):
    assert _DUMP.dumped_pool_layers(layers) == expected


def test_a_pool_with_no_layers_is_refused():
    with pytest.raises(ValueError, match="0 layers"):
        _DUMP.dumped_pool_layers(0)


# --------------------------------------------------------------------------- #
#  (ii) Off is the default and costs nothing
# --------------------------------------------------------------------------- #
def test_a_fresh_pool_carries_no_recorder():
    _, pool = _wave()

    assert pool.kernel_dump is None


def test_nothing_is_written_when_the_flag_is_off(tmp_path, stub_kernel):
    plan, pool = _wave()
    executor = _EXEC.PoolChunkExecutor(plan, PROMPTS, pool)
    chunk, = plan.chunks

    batch = executor.build(chunk)
    _attend(pool, batch, 0)

    assert pool.kernel_dump is None
    assert list(tmp_path.iterdir()) == []
    assert len(stub_kernel) == 1, "the kernel still runs with the probe off"


# --------------------------------------------------------------------------- #
#  (iii) The schema, every digest recomputed independently
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["sync", "nosync"])
def test_the_dump_round_trips_with_independently_recomputed_digests(
    tmp_path, stub_kernel, mode
):
    plan, pool = _wave()
    recorder = _DUMP.PoolKernelDumpRecorder(
        rank=7, mode=mode, directory=str(tmp_path), page_tokens=B,
        global_ids=GLOBAL_IDS,
    )
    pool.kernel_dump = recorder
    executor = _EXEC.PoolChunkExecutor(plan, PROMPTS, pool)
    chunk, = plan.chunks

    batch = executor.build(chunk)
    key = _attend(pool, batch, 0)

    assert len(recorder.files) == 1
    record = _DUMP.read_dump(recorder.files[0])

    # Identity and mode.
    assert record["schema"] == _DUMP.SCHEMA_VERSION
    assert (record["rank"], record["mode"], record["chunk"]) == (7, mode, 0)
    assert record["synced"] is (mode == "sync")
    assert record["pool_layer"] == 0 and record["num_pool_layers"] == 1
    assert record["page_tokens"] == B
    assert record["num_pool_pages"] == pool.num_pages

    # The kernel arguments, as the kernel got them.
    assert record["page_table"] == batch.page_table.tolist()
    assert record["cache_seqlens"] == batch.cache_seqlens.tolist()
    assert record["cu_seqlens_q"] == batch.cu_seqlens_q.tolist()
    assert record["max_seqlen_q"] == batch.max_seqlen_q
    kernel_args, = stub_kernel
    assert record["page_table"] == kernel_args["page_table"].tolist()
    assert record["cache_seqlens"] == kernel_args["cache_seqlens"].tolist()

    # Row metadata: enough to map a row back to a live member.
    rows = len(batch.items)
    assert record["row_table_row"] == list(range(rows))
    assert record["kind"] == [item.kind for item in batch.items]
    assert record["row_ref"] == [item.ref for item in batch.items]
    assert record["row_token_start"] == [i.token_start for i in batch.items]
    assert record["row_token_end"] == [i.token_end for i in batch.items]
    assert record["row_global_idx"] == [
        GLOBAL_IDS[item.ref] if item.kind == "tail" else -1
        for item in batch.items
    ]

    # Read-side digests: every page the table addresses, hashed from the pool.
    expected_pages = sorted({page for row in record["page_table"] for page in row})
    assert record["read_pages"] == expected_pages
    assert record["read_digest"] == [
        _digest(pool.k[0][page]) for page in expected_pages
    ]
    # A page that no row addresses is absent, so the dump stays small.
    assert len(expected_pages) < pool.num_pages

    # Write-side digests: each row's source K, hashed before the index_copy_.
    assert record["write_digest"] == [
        _digest(key[batch.row_bounds[row]:batch.row_bounds[row + 1]])
        for row in range(rows)
    ]
    assert record["bytes_hashed"] > 0

    # Schema 2: the same source split at the pool's page boundaries, with the
    # read side taken over exactly the slots each piece claims to have filled.
    runs = list(zip(
        record["write_run_row"], record["write_run_page"],
        record["write_run_offset"], record["write_run_length"],
        record["write_run_source_start"], record["write_run_token_start"],
    ))
    expected_runs = []
    for row, item in enumerate(batch.items):
        lo, hi = batch.row_bounds[row], batch.row_bounds[row + 1]
        for start, length, page, offset in _DUMP.slot_write_runs(
            batch.slots[lo:hi].tolist(), B
        ):
            expected_runs.append(
                (row, page, offset, length, start, item.token_start + start)
            )
    assert runs == expected_runs
    assert record["write_run_digest"] == [
        _digest(key[batch.row_bounds[row] + start:
                    batch.row_bounds[row] + start + length])
        for row, _, _, length, start, _ in expected_runs
    ]
    assert record["write_run_read_digest"] == [
        _digest(pool.k[0][page][offset:offset + length])
        for _, page, offset, length, _, _ in expected_runs
    ]
    assert record["write_run_match"] == [True] * len(expected_runs)


def test_each_chunk_and_dumped_layer_gets_its_own_file(tmp_path, stub_kernel):
    """Four pool layers dump rows 0 and 2 only; chunk ids count up."""

    plan, pool = _wave(num_layers=4)
    recorder = _DUMP.PoolKernelDumpRecorder(
        rank=1, mode="nosync", directory=str(tmp_path), page_tokens=B,
        global_ids=GLOBAL_IDS,
    )
    pool.kernel_dump = recorder
    executor = _EXEC.PoolChunkExecutor(plan, PROMPTS, pool)
    chunk, = plan.chunks

    batch = executor.build(chunk)
    for layer in range(4):
        _attend(pool, batch, layer, seed=layer)

    assert recorder.layers == (0, 2)
    assert [Path(path).name for path in recorder.files] == [
        "rank1_chunk0000_layer00.npz", "rank1_chunk0000_layer02.npz",
    ]
    for path in recorder.files:
        record = _DUMP.read_dump(path)
        assert record["pool_layer"] in (0, 2)
        assert record["num_pool_layers"] == 4

    # A second chunk of the same wave advances the chunk id.
    recorder.begin_chunk(batch)
    assert recorder.chunk_index == 1
    _attend(pool, batch, 0, seed=9)
    assert Path(recorder.files[-1]).name == "rank1_chunk0001_layer00.npz"


def test_a_schema_bump_is_refused_rather_than_misread(tmp_path, stub_kernel):
    """A collector running against an older build must fail loudly."""

    import numpy as np

    plan, pool = _wave()
    pool.kernel_dump = _DUMP.PoolKernelDumpRecorder(
        rank=0, mode="sync", directory=str(tmp_path), page_tokens=B,
        global_ids=GLOBAL_IDS,
    )
    executor = _EXEC.PoolChunkExecutor(plan, PROMPTS, pool)
    chunk, = plan.chunks
    _attend(pool, executor.build(chunk), 0)
    path = pool.kernel_dump.files[0]

    with np.load(path, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}
    fields["schema"] = np.int32(_DUMP.SCHEMA_VERSION + 1)
    np.savez(path, **fields)

    with pytest.raises(ValueError, match="dump schema"):
        _DUMP.read_dump(path)


def test_the_recorder_refuses_an_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="dump mode"):
        _DUMP.PoolKernelDumpRecorder(
            rank=0, mode="eventually", directory=str(tmp_path), page_tokens=B,
        )


# --------------------------------------------------------------------------- #
#  (iv) Per-page write digests: the split, the clean case, the caught case
# --------------------------------------------------------------------------- #
def test_a_row_is_split_at_every_page_boundary_it_writes():
    """One page-aligned ascending block per page, each run its own page."""

    slots = list(range(8, 8 + 10))  # pages 2, 3, 4 at B == 4
    assert _DUMP.slot_write_runs(slots, B) == [
        (0, 4, 2, 0), (4, 4, 3, 0), (8, 2, 4, 0),
    ]
    # A row that starts mid page, and one that fills exactly one page.
    assert _DUMP.slot_write_runs([6, 7, 8], B) == [(0, 2, 1, 2), (2, 1, 2, 0)]
    assert _DUMP.slot_write_runs([12, 13, 14, 15], B) == [(0, 4, 3, 0)]
    assert _DUMP.slot_write_runs([], B) == []


def test_the_split_follows_the_slots_rather_than_an_assumed_layout():
    """A row whose slots are not one ascending block yields more runs.

    The point of deriving the split from ``batch.slots`` is that a defect in
    the slot list cannot hide inside one whole-row digest.
    """

    # Backwards inside a page, and a page revisited after leaving it.
    assert _DUMP.slot_write_runs([5, 4], B) == [(0, 1, 1, 1), (1, 1, 1, 0)]
    assert _DUMP.slot_write_runs([0, 1, 8, 2], B) == [
        (0, 2, 0, 0), (2, 1, 2, 0), (3, 1, 0, 2),
    ]
    # Consecutive slots that cross a page boundary still split by page.
    assert _DUMP.slot_write_runs([3, 4], B) == [(0, 1, 0, 3), (1, 1, 1, 0)]


@pytest.mark.parametrize("page_tokens", [0, -4])
def test_a_slot_split_refuses_a_non_positive_page(page_tokens):
    with pytest.raises(ValueError, match="page_tokens must be positive"):
        _DUMP.slot_write_runs([0, 1], page_tokens)


def test_every_token_of_every_row_is_covered_by_exactly_one_run(
    tmp_path, stub_kernel
):
    """The runs partition the chunk's tokens, and a partial page is covered.

    The live wave had exactly one row owning a whole page, which is why the
    row digest could not be compared anywhere else; a partial run is the case
    schema 2 adds.
    """

    plan, pool = _wave(prompts=PROMPTS_RAGGED)
    pool.kernel_dump = _DUMP.PoolKernelDumpRecorder(
        rank=2, mode="sync", directory=str(tmp_path), page_tokens=B,
        global_ids=GLOBAL_IDS,
    )
    executor = _EXEC.PoolChunkExecutor(plan, PROMPTS_RAGGED, pool)
    chunk, = plan.chunks
    batch = executor.build(chunk)
    _attend(pool, batch, 0)
    record = _DUMP.read_dump(pool.kernel_dump.files[0])

    # Each row's runs tile its own tokens from 0, in order.
    for row in range(len(batch.items)):
        own = batch.row_bounds[row + 1] - batch.row_bounds[row]
        spans = [
            (start, length) for run_row, start, length in zip(
                record["write_run_row"], record["write_run_source_start"],
                record["write_run_length"])
            if run_row == row
        ]
        assert spans, f"row {row} has no write run"
        at = 0
        for start, length in spans:
            assert start == at
            at += length
        assert at == own
    # Every run's slots are the batch's own slots, page and offset agreed.
    slots = batch.slots.tolist()
    for run_row, start, length, page, offset in zip(
        record["write_run_row"], record["write_run_source_start"],
        record["write_run_length"], record["write_run_page"],
        record["write_run_offset"],
    ):
        lo = batch.row_bounds[run_row]
        assert slots[lo + start:lo + start + length] == [
            page * B + offset + step for step in range(length)
        ]
    assert sum(record["write_run_length"]) == int(batch.input_ids.numel())
    assert any(length < B for length in record["write_run_length"]), (
        "the fixture must contain a partially filled page"
    )
    assert any(length == B for length in record["write_run_length"])
    assert record["write_run_match"] == [True] * len(record["write_run_row"])


def test_a_wrong_pool_write_is_caught_on_exactly_the_pages_it_damaged(tmp_path):
    """The defect class the per-page digests exist for.

    The recorder is driven directly so the write between its two sites can be
    the WRONG one: one row's source rotated by a token, which is the shape a
    misaligned prefix would have. Every page of that row must mismatch and no
    other page may.
    """

    plan, pool = _wave(prompts=PROMPTS_RAGGED)
    recorder = _DUMP.PoolKernelDumpRecorder(
        rank=5, mode="nosync", directory=str(tmp_path), page_tokens=B,
        global_ids=GLOBAL_IDS,
    )
    pool.kernel_dump = recorder
    executor = _EXEC.PoolChunkExecutor(plan, PROMPTS_RAGGED, pool)
    chunk, = plan.chunks
    batch = executor.build(chunk)

    tokens = int(batch.input_ids.numel())
    generator = torch.Generator().manual_seed(11)
    key = torch.randn(tokens, HEADS, K_DIM, generator=generator)
    recorder.record_write_sources(batch, key)

    damaged_row = len(batch.items) - 1
    lo, hi = batch.row_bounds[damaged_row], batch.row_bounds[damaged_row + 1]
    assert hi - lo > 1, "rotating a one-token row would be a no-op"
    wrong = key.clone()
    wrong[lo:hi] = key[lo:hi].roll(1, dims=0)
    pool.k[0].view(-1, HEADS, K_DIM).index_copy_(0, batch.slots, wrong)
    recorder.record_kernel_inputs(batch, pool.k[0], 0, 1)

    record = _DUMP.read_dump(recorder.files[0])
    mismatched = {
        row for row, match in zip(record["write_run_row"],
                                  record["write_run_match"])
        if not match
    }
    assert mismatched == {damaged_row}
    # ... and the whole-row digest of a clean row still agrees with the pool.
    assert record["write_run_match"].count(False) == sum(
        1 for row in record["write_run_row"] if row == damaged_row
    )


# --------------------------------------------------------------------------- #
#  (v) The worker plumbing, through the real batch-debug seam
# --------------------------------------------------------------------------- #
def test_the_worker_reads_both_flags_through_the_batch_debug_seam():
    source = Path(__file__).resolve().parents[2].joinpath(
        "batchgen", "batchgen_worker.py"
    ).read_text()

    assert "def _new_pool_kernel_dump_recorder(self, wave):" in source
    assert "pool.kernel_dump = self._new_pool_kernel_dump_recorder(wave)" in source
    seam = source.split("def _new_pool_kernel_dump_recorder")[1].split("\n\tdef ")[0]
    # One cached read per wave, shared with the other pooled probes, and the
    # flag decision precedes the import so the off path imports nothing.
    assert "self._pool_probe_debug(wave)" in seam
    assert seam.index("if debug is None:") < seam.index(
        "from batchgen.prefix_reuse.kernel_dump import"
    )
    assert "parse_dump_mode" in seam
    assert "dump_directory" in seam
    assert "return None" in seam
    # Setup is guarded: an undumpable directory costs the dump, not the wave.
    assert seam.count("run_probe(") == 3


def test_tensor_digest_handles_bfloat16_and_matches_raw_bytes():
    """bf16 broke the live run: numpy has no bfloat16, so the digest must hash
    the bit-exact uint8 view; for float32 the digest must equal hashing the
    plain numpy bytes (backward-compatible)."""
    import hashlib

    import torch

    from batchgen.prefix_reuse.kernel_dump import DIGEST_BYTES, _tensor_digest

    bf16 = torch.arange(64, dtype=torch.float32).to(torch.bfloat16).reshape(4, 16)
    expected_bf16 = hashlib.blake2b(
        bf16.contiguous().view(torch.uint8).numpy().tobytes(),
        digest_size=DIGEST_BYTES,
    ).hexdigest()
    assert _tensor_digest(bf16) == expected_bf16

    f32 = torch.arange(48, dtype=torch.float32).reshape(3, 16)
    expected_f32 = hashlib.blake2b(
        f32.numpy().tobytes(), digest_size=DIGEST_BYTES
    ).hexdigest()
    assert _tensor_digest(f32) == expected_f32

    # non-contiguous input must be handled too (the live call slices rows)
    sliced = bf16[:, ::2]
    assert _tensor_digest(sliced) == _tensor_digest(sliced.contiguous())


# --------------------------------------------------------------------------
# A dump failure costs the dump, never the wave
# --------------------------------------------------------------------------
def test_a_dump_it_cannot_write_logs_one_probe_error_and_returns_none(caplog):
    """The dump writes files from inside the prefill forward."""

    import logging as _logging

    import torch as _torch

    from batchgen.prefix_reuse.kernel_dump import PoolKernelDumpRecorder

    recorder = PoolKernelDumpRecorder(
        rank=6, mode="nosync", directory=tempfile.mkdtemp(), page_tokens=4,
    )
    with caplog.at_level(_logging.WARNING):
        # ``batch`` is not a ChunkBatch at all: whatever the dump reaches for
        # first is missing, which is exactly the shape of a probe bug.
        assert recorder.record_kernel_inputs(
            object(), _torch.zeros(2, 4, 1, 1), 0, 1
        ) is None

    line = next(
        record.getMessage() for record in caplog.records
        if "PROBE_ERROR" in record.getMessage()
    )
    assert "probe=kernel_dump.record_kernel_inputs" in line
    assert "rank=6" in line
    assert recorder.files == [] and recorder.strict is False


def test_a_layer_question_it_cannot_answer_reads_as_not_dumping(caplog):
    import logging as _logging

    from batchgen.prefix_reuse.kernel_dump import PoolKernelDumpRecorder

    recorder = PoolKernelDumpRecorder(
        rank=0, mode="nosync", directory=tempfile.mkdtemp(), page_tokens=4,
    )
    with caplog.at_level(_logging.WARNING):
        assert recorder.wants_layer(0, 0) is False

    assert any("PROBE_ERROR" in r.getMessage() for r in caplog.records)


def test_strict_mode_re_raises_for_the_tests_that_want_the_refusal():
    import torch as _torch

    from batchgen.prefix_reuse.kernel_dump import PoolKernelDumpRecorder

    recorder = PoolKernelDumpRecorder(
        rank=0, mode="nosync", directory=tempfile.mkdtemp(), page_tokens=4,
        strict=True,
    )
    with pytest.raises(Exception):
        recorder.record_kernel_inputs(object(), _torch.zeros(2, 4, 1, 1), 0, 1)
