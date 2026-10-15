"""``batchgen_debug.prefix_pool_dump_kernel_args``: the dump and its schema.

The probe records, at the one seam where the pooled prefill both writes the
pool and reads it back (``pool_prefill_attention``), the kernel arguments plus
a digest of every pool page the page table addresses and of every row's source
K. Pinned here: the mode parser, the dump directory rule, the two-layer limit,
the file schema round-tripped through ``read_dump`` with every digest
recomputed independently, and that nothing happens with the flag off.

CPU only: the pool and the executor are the real classes, loaded by file path,
and ``gqa_extend_fa`` is stubbed because the paged kernel is GPU work.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
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


def _wave(num_layers=1):
    plan = _PLAN.plan_wave_prefix_sharing(
        PROMPTS, block_tokens=B, pool_pages=64, chunk_tokens=4096,
        same_chunk_ancestors=True,
    )
    _PLAN.validate_wave_prefix_plan(plan, PROMPTS)
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
#  (iv) The worker plumbing, through the real batch-debug seam
# --------------------------------------------------------------------------- #
def test_the_worker_reads_both_flags_through_the_batch_debug_seam():
    source = Path(__file__).resolve().parents[2].joinpath(
        "batchgen", "batchgen_worker.py"
    ).read_text()

    assert "def _new_pool_kernel_dump_recorder(self, wave):" in source
    assert "pool.kernel_dump = self._new_pool_kernel_dump_recorder(wave)" in source
    seam = source.split("def _new_pool_kernel_dump_recorder")[1].split("\n\tdef ")[0]
    assert "_active_batchgen_debug_for_sequences" in seam
    assert "parse_dump_mode" in seam
    assert "dump_directory" in seam
    assert "return None" in seam


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
