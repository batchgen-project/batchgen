"""Unit tests for batchgen/token_store.py — CPU only (numpy + stdlib).

Covers the node-shared prompt arena (page chaining, capacity errors,
cross-process read-only attach, lifecycle) and the rank-private decoded store.

The module is loaded by path: importing ``batchgen`` runs ``__init__`` which
pulls in torch and the compiled batchgen_kernels, neither of which belongs in a
CPU-only unit test. POSIX shm names are capped at 31 characters on macOS, so
every test uses a SHORT unique prefix and unlinks it in teardown.
"""

import importlib.util
import mmap
import multiprocessing as mp
import os
import secrets
import sys
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[2] / "batchgen" / "token_store.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "batchgen_token_store_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


token_store = _load_module()

PAGE = token_store.DEFAULT_PAGE_SIZE_TOKENS


def _unique_prefix() -> str:
    """A short unique shm prefix (prefix + suffix must stay under 31 chars)."""
    return f"bgts{os.getpid() % 10000}{secrets.token_hex(3)}."


def _prefix_of(arena) -> str:
    return arena.name[: -len(token_store.PROMPT_ARENA_SHM_SUFFIX)]


def _tokens(length: int, seed: int = 0) -> np.ndarray:
    """Distinct int64 ids (as a real tokenizer hands them over)."""
    return np.arange(length, dtype=np.int64) * 7 + seed + 1


@pytest.fixture
def make_arena():
    """Build arenas with unique names and unlink every one in teardown."""
    created = []

    def make(capacity_pages=64, page_size_tokens=PAGE, **kwargs):
        arena = token_store.PromptTokenArena.create(
            _unique_prefix(),
            capacity_bytes=capacity_pages * page_size_tokens * 4,
            page_size_tokens=page_size_tokens,
            **kwargs,
        )
        created.append(arena)
        return arena

    yield make

    for arena in created:
        arena.unlink()  # idempotent once unlinked


# ------------------------------------------------------------- prompt arena


def test_geometry_from_capacity(make_arena):
    arena = make_arena(capacity_pages=8)
    assert arena.num_pages == 8
    assert arena.page_size_tokens == PAGE
    assert arena.capacity_tokens == 8 * PAGE
    assert arena.allocated_pages == 0
    assert arena.free_pages == 8
    assert arena.pages_for(0) == 0
    assert arena.pages_for(1) == 1
    assert arena.pages_for(PAGE) == 1
    assert arena.pages_for(PAGE + 1) == 2
    assert arena.name.endswith(token_store.PROMPT_ARENA_SHM_SUFFIX)


def test_capacity_must_hold_one_page():
    prefix = _unique_prefix()
    with pytest.raises(ValueError, match="does not hold one"):
        token_store.PromptTokenArena.create(prefix, capacity_bytes=4)


@pytest.mark.parametrize(
    "length", [1, 17, PAGE - 1, PAGE, PAGE + 1, 5 * PAGE + 7, 40 * PAGE]
)
def test_write_read_round_trip_across_page_boundaries(make_arena, length):
    arena = make_arena(capacity_pages=64)
    tokens = _tokens(length)

    handle = arena.write(tokens)
    first_page, handle_length = handle

    assert handle_length == length
    assert 0 <= first_page < arena.num_pages
    expected_pages = -(-length // PAGE)
    assert len(arena.page_chain(handle)) == expected_pages
    assert arena.allocated_pages == expected_pages

    out = arena.read(handle)
    assert out.dtype == np.int32
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, tokens.astype(np.int32))


def test_write_rejects_empty_and_non_1d(make_arena):
    arena = make_arena()
    with pytest.raises(ValueError, match="empty token sequence"):
        arena.write(np.empty(0, dtype=np.int32))
    with pytest.raises(ValueError, match="1-D"):
        arena.write(np.zeros((2, 3), dtype=np.int32))
    assert arena.allocated_pages == 0


def test_zero_length_handle_is_refused(make_arena):
    arena = make_arena()
    for handle in [(0, 0), (0, -1)]:
        with pytest.raises(ValueError, match="at least one token"):
            arena.read(handle)
        with pytest.raises(ValueError, match="at least one token"):
            arena.page_chain(handle)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_read_into_accepts_int32_and_int64(make_arena, dtype):
    arena = make_arena()
    length = 2 * PAGE + 123
    tokens = _tokens(length, seed=5)
    handle = arena.write(tokens)

    out = np.full(length + 16, -99, dtype=dtype)
    view = arena.read_into(handle, out)

    assert view.dtype == np.dtype(dtype)
    assert view.size == length
    np.testing.assert_array_equal(view, tokens.astype(dtype))
    # The tail past the handle is left alone.
    np.testing.assert_array_equal(out[length:], np.full(16, -99, dtype=dtype))
    # The returned view is into the caller's buffer, not a fresh allocation.
    assert view.base is out


def test_read_into_rejects_bad_destinations(make_arena):
    arena = make_arena()
    handle = arena.write(_tokens(10))

    with pytest.raises(ValueError, match="int32 or int64"):
        arena.read_into(handle, np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="1-D"):
        arena.read_into(handle, np.zeros((2, 10), dtype=np.int64))
    with pytest.raises(ValueError, match="holds 9 tokens"):
        arena.read_into(handle, np.zeros(9, dtype=np.int64))
    with pytest.raises(TypeError, match="numpy array"):
        arena.read_into(handle, [0] * 10)


def test_free_returns_pages_for_reuse(make_arena):
    arena = make_arena(capacity_pages=8)
    handles = [arena.write(_tokens(PAGE + 1, seed=i)) for i in range(3)]
    assert arena.allocated_pages == 6

    freed_pages = set(arena.page_chain(handles[1]))
    assert arena.free(handles[1]) == 2
    assert arena.allocated_pages == 4
    assert arena.free_pages == 4

    reused = arena.write(_tokens(PAGE + 1, seed=99))
    assert set(arena.page_chain(reused)) == freed_pages
    assert arena.allocated_pages == 6
    np.testing.assert_array_equal(
        arena.read(reused), _tokens(PAGE + 1, seed=99).astype(np.int32)
    )
    # The surviving chains are undisturbed.
    for i in (0, 2):
        np.testing.assert_array_equal(
            arena.read(handles[i]), _tokens(PAGE + 1, seed=i).astype(np.int32)
        )


@pytest.mark.parametrize("length", [PAGE, 3 * PAGE + 1])
def test_freed_handle_is_refused(make_arena, length):
    """A stale handle raises while its pages are still free (not after reuse)."""
    arena = make_arena(capacity_pages=8)
    handle = arena.write(_tokens(length))
    arena.free(handle)

    with pytest.raises(ValueError, match="already freed|on the free list"):
        arena.read(handle)
    with pytest.raises(ValueError, match="already freed|on the free list"):
        arena.free(handle)
    assert arena.allocated_pages == 0


def test_allocation_beyond_free_pages_raises(make_arena):
    arena = make_arena(capacity_pages=4)
    held = arena.write(_tokens(3 * PAGE))
    assert arena.free_pages == 1

    with pytest.raises(token_store.TokenStoreCapacityError) as excinfo:
        arena.write(_tokens(PAGE + 1))
    message = str(excinfo.value)
    assert "needs 2 pages" in message
    assert "1 of 4 capacity pages are free" in message

    # The failed write changed nothing.
    assert arena.allocated_pages == 3
    np.testing.assert_array_equal(
        arena.read(held), _tokens(3 * PAGE).astype(np.int32)
    )


def test_chain_integrity_after_interleaved_alloc_and_free(make_arena):
    arena = make_arena(capacity_pages=64)
    lengths = [1, PAGE, PAGE + 1, 3 * PAGE - 5, 2 * PAGE, 7, 4 * PAGE + 9]
    live = {}

    for round_index in range(4):
        for i, length in enumerate(lengths):
            seed = round_index * 100 + i
            live[seed] = (arena.write(_tokens(length, seed=seed)), length)
        # Drop every other chain, so the next round reuses a fragmented list.
        for seed in sorted(live)[::2]:
            handle, _ = live.pop(seed)
            arena.free(handle)

        used = []
        for seed, (handle, length) in live.items():
            pages = arena.page_chain(handle)
            assert len(pages) == -(-length // PAGE)
            used.extend(pages)
            np.testing.assert_array_equal(
                arena.read(handle), _tokens(length, seed=seed).astype(np.int32)
            )
        # No page is claimed by two live chains, and the counters agree.
        assert len(used) == len(set(used))
        assert arena.allocated_pages == len(used)
        assert arena.free_pages == arena.num_pages - len(used)


def test_release_free_memory_keeps_pages_reusable(make_arena):
    # release_free_fraction is low enough that free() triggers a pass itself.
    arena = make_arena(capacity_pages=16, release_free_fraction=0.25)
    handles = [arena.write(_tokens(PAGE, seed=i)) for i in range(12)]
    survivor = handles[0]
    for handle in handles[1:]:
        arena.free(handle)

    # Linux punches holes here and returns bytes; macOS has no MADV_REMOVE and
    # returns 0. Either way the arena stays correct.
    released = arena.release_free_memory()
    assert released >= 0
    assert released % 4096 == 0

    np.testing.assert_array_equal(
        arena.read(survivor), _tokens(PAGE, seed=0).astype(np.int32)
    )
    assert arena.allocated_pages == 1
    rewritten = arena.write(_tokens(3 * PAGE, seed=77))
    np.testing.assert_array_equal(
        arena.read(rewritten), _tokens(3 * PAGE, seed=77).astype(np.int32)
    )


def test_attached_arena_is_read_only(make_arena):
    arena = make_arena()
    tokens = _tokens(PAGE + 3)
    handle = arena.write(tokens)

    reader = token_store.PromptTokenArena.attach(_prefix_of(arena))
    try:
        assert not reader.is_creator
        assert reader.num_pages == arena.num_pages
        assert reader.page_size_tokens == arena.page_size_tokens
        np.testing.assert_array_equal(reader.read(handle), tokens.astype(np.int32))

        for call in (
            lambda: reader.write(_tokens(4)),
            lambda: reader.free(handle),
            lambda: reader.release_free_memory(),
            lambda: reader.allocated_pages,
            lambda: reader.free_pages,
        ):
            with pytest.raises(RuntimeError, match="attached read-only"):
                call()
        with pytest.raises(RuntimeError, match="only.*creator may unlink"):
            reader.unlink()
    finally:
        reader.close()
        reader.close()  # idempotent


def _reader_child(shm_prefix, handle, out_queue):
    """Attach in a spawned process and report what the handle holds."""
    arena = token_store.PromptTokenArena.attach(shm_prefix)
    try:
        out_queue.put(
            {
                "tokens": arena.read(handle).tolist(),
                "pages": arena.page_chain(handle),
                "page_size_tokens": arena.page_size_tokens,
                "num_pages": arena.num_pages,
            }
        )
    finally:
        arena.close()


def test_spawned_reader_sees_identical_tokens(make_arena):
    arena = make_arena(capacity_pages=16)
    tokens = _tokens(3 * PAGE + 11, seed=3)
    handle = arena.write(tokens)
    expected_pages = arena.page_chain(handle)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_reader_child, args=(_prefix_of(arena), handle, queue))
    proc.start()
    try:
        payload = queue.get(timeout=60)
    finally:
        proc.join(timeout=60)

    assert proc.exitcode == 0
    assert payload["page_size_tokens"] == PAGE
    assert payload["num_pages"] == 16
    assert payload["pages"] == expected_pages
    np.testing.assert_array_equal(
        np.asarray(payload["tokens"], dtype=np.int32), tokens.astype(np.int32)
    )


def test_create_refuses_an_existing_name():
    prefix = _unique_prefix()
    arena = token_store.PromptTokenArena.create(prefix, capacity_bytes=1 << 20)
    try:
        with pytest.raises(FileExistsError):
            token_store.PromptTokenArena.create(prefix, capacity_bytes=1 << 20)
    finally:
        arena.unlink()


def test_unlink_removes_the_segment():
    prefix = _unique_prefix()
    arena = token_store.PromptTokenArena.create(prefix, capacity_bytes=1 << 20)
    handle = arena.write(_tokens(8))
    np.testing.assert_array_equal(arena.read(handle), _tokens(8).astype(np.int32))

    arena.unlink()
    with pytest.raises(FileNotFoundError):
        token_store.PromptTokenArena.attach(prefix)


def test_attach_refuses_a_foreign_segment():
    from multiprocessing import shared_memory

    prefix = _unique_prefix()
    name = token_store.prompt_arena_shm_name(prefix)
    shm = shared_memory.SharedMemory(name=name, create=True, size=1 << 16)
    try:
        with pytest.raises(ValueError, match="not a prompt arena"):
            token_store.PromptTokenArena.attach(prefix)
    finally:
        shm.close()
        shm.unlink()


def test_large_capacity_creation_is_sparse():
    """A 1 GiB arena is ftruncate'd only — no zero-fill, no page touching."""
    prefix = _unique_prefix()
    try:
        arena = token_store.PromptTokenArena.create(prefix, capacity_bytes=1 << 30)
    except OSError as exc:  # pragma: no cover - platform shm limit
        pytest.skip(f"platform refused a 1 GiB shm segment: {exc}")
    try:
        assert arena.num_pages == (1 << 30) // (PAGE * 4)
        assert arena.allocated_pages == 0
        tokens = _tokens(PAGE + 1)
        handle = arena.write(tokens)
        np.testing.assert_array_equal(arena.read(handle), tokens.astype(np.int32))
    finally:
        arena.unlink()


# ------------------------------------------------------------ decoded store


def test_decoded_append_and_read_across_chunks():
    store = token_store.DecodedTokenStore(chunk_size_tokens=4)
    assert store.chunk_size_tokens == 4
    assert store.length(7) == 0
    assert store.read(7).size == 0

    store.append(7, [10, 11, 12])
    assert store.length(7) == 3
    store.append(7, np.array([13, 14, 15, 16, 17], dtype=np.int64))
    assert store.length(7) == 8

    expected = np.arange(10, 18, dtype=np.int32)
    out = store.read(7)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, expected)
    # Slices that start and end inside, on, and across chunk boundaries.
    np.testing.assert_array_equal(store.read(7, 0, 4), expected[:4])
    np.testing.assert_array_equal(store.read(7, 4), expected[4:])
    np.testing.assert_array_equal(store.read(7, 3, 6), expected[3:6])
    np.testing.assert_array_equal(store.read(7, 2, 2), expected[2:2])
    assert store.active_sequences == 1


def test_decoded_append_exact_chunk_boundary():
    store = token_store.DecodedTokenStore(chunk_size_tokens=4)
    store.append(1, np.arange(4, dtype=np.int32))
    assert store.length(1) == 4
    store.append(1, [99])
    assert store.length(1) == 5
    np.testing.assert_array_equal(
        store.read(1), np.array([0, 1, 2, 3, 99], dtype=np.int32)
    )
    store.append(1, np.empty(0, dtype=np.int32))
    assert store.length(1) == 5
    store.append(1, 100)  # a bare scalar step token
    np.testing.assert_array_equal(
        store.read(1), np.array([0, 1, 2, 3, 99, 100], dtype=np.int32)
    )


def test_decoded_append_rejects_2d():
    store = token_store.DecodedTokenStore(chunk_size_tokens=4)
    with pytest.raises(ValueError, match="1-D"):
        store.append(1, np.zeros((2, 2), dtype=np.int32))


def test_decoded_append_batch_one_token_per_sequence():
    store = token_store.DecodedTokenStore(chunk_size_tokens=2)
    seq_ids = np.array([5, 6, 7], dtype=np.int64)

    for step in range(5):
        store.append_batch(seq_ids, np.array([step, 100 + step, 200 + step]))

    assert store.active_sequences == 3
    for offset, seq_id in zip((0, 100, 200), seq_ids):
        assert store.length(int(seq_id)) == 5
        np.testing.assert_array_equal(
            store.read(int(seq_id)),
            np.arange(offset, offset + 5, dtype=np.int32),
        )


def test_decoded_append_batch_interleaves_with_append():
    store = token_store.DecodedTokenStore(chunk_size_tokens=3)
    store.append(4, [1, 2, 3, 4])
    store.append_batch(np.array([4, 5]), np.array([5, 50], dtype=np.int64))
    np.testing.assert_array_equal(
        store.read(4), np.array([1, 2, 3, 4, 5], dtype=np.int32)
    )
    np.testing.assert_array_equal(store.read(5), np.array([50], dtype=np.int32))

    with pytest.raises(ValueError, match="3 entries, tokens has 2"):
        store.append_batch(np.array([4, 5, 6]), np.array([1, 2]))


def test_decoded_extend_rows_flushes_a_step_log():
    store = token_store.DecodedTokenStore(chunk_size_tokens=8)
    slots = np.array([11, 12, 13, 14], dtype=np.int64)
    # Start the columns at different chunk offsets so flushes cross chunks.
    store.append(11, np.arange(6))
    store.append(12, np.arange(8))
    store.append(14, np.arange(3))
    # [steps, batch] step log, one column per slot.
    log = np.array(
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
            [4, 40, 400, 4000],
            [5, 50, 500, 5000],
        ],
        dtype=np.int64,
    )
    store.extend_rows(slots, log, counts=np.array([5, 3, 0, 1]))

    np.testing.assert_array_equal(
        store.read(11), np.r_[np.arange(6), 1, 2, 3, 4, 5].astype(np.int32)
    )
    np.testing.assert_array_equal(
        store.read(12), np.r_[np.arange(8), 10, 20, 30].astype(np.int32)
    )
    assert store.length(13) == 0
    np.testing.assert_array_equal(
        store.read(14), np.r_[np.arange(3), 1000].astype(np.int32)
    )

    # A scalar count applies to every column; None means all steps.
    store.extend_rows(slots, log, counts=1)
    assert store.length(13) == 1
    assert store.length(11) == 12
    store.extend_rows(np.array([21]), log[:, :1])
    np.testing.assert_array_equal(store.read(21), np.arange(1, 6, dtype=np.int32))


def test_decoded_extend_rows_validates_shapes():
    store = token_store.DecodedTokenStore(chunk_size_tokens=4)
    log = np.zeros((3, 2), dtype=np.int64)

    with pytest.raises(ValueError, match=r"\[steps, batch\]"):
        store.extend_rows(np.array([1]), np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError, match="3 entries, tokens has 2 columns"):
        store.extend_rows(np.array([1, 2, 3]), log)
    with pytest.raises(ValueError, match="counts has 3 entries"):
        store.extend_rows(np.array([1, 2]), log, counts=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match=r"outside \[0, 3\]"):
        store.extend_rows(np.array([1, 2]), log, counts=4)
    with pytest.raises(ValueError, match="exceeds the 4-token chunk"):
        store.extend_rows(np.array([1]), np.zeros((5, 1), dtype=np.int64))
    with pytest.raises(ValueError, match="slot must be >= 0"):
        store.append_batch(np.array([-1]), np.array([1]))


def test_decoded_store_matches_a_list_model_under_random_operations():
    """Append, batch append, partial flushes and frees against a reference."""
    rng = np.random.default_rng(1234)
    chunk = 8
    store = token_store.DecodedTokenStore(chunk_size_tokens=chunk)
    model = {}
    slots = np.arange(40)
    for _ in range(300):
        op = rng.integers(0, 4)
        if op == 0:
            slot = int(rng.integers(0, 40))
            tokens = rng.integers(0, 1000, size=int(rng.integers(0, 20)))
            store.append(slot, tokens)
            model.setdefault(slot, []).extend(tokens.tolist())
        elif op == 1:
            batch = rng.choice(slots, size=int(rng.integers(1, 40)), replace=False)
            tokens = rng.integers(0, 1000, size=batch.size)
            store.append_batch(batch, tokens)
            for s_, t_ in zip(batch.tolist(), tokens.tolist()):
                model.setdefault(s_, []).append(t_)
        elif op == 2:
            batch = rng.choice(slots, size=int(rng.integers(1, 40)), replace=False)
            steps = int(rng.integers(1, chunk + 1))
            log = rng.integers(0, 1000, size=(steps, batch.size))
            counts = rng.integers(0, steps + 1, size=batch.size)
            store.extend_rows(batch, log, counts)
            for j, s_ in enumerate(batch.tolist()):
                model.setdefault(s_, []).extend(log[: counts[j], j].tolist())
        else:
            slot = int(rng.integers(0, 40))
            assert store.free(slot) == len(model.pop(slot, []))
        for s_ in range(40):
            expected = np.array(model.get(s_, []), dtype=np.int32)
            assert store.length(s_) == expected.size
            np.testing.assert_array_equal(store.read(s_), expected)
    assert store.active_sequences == sum(1 for v in model.values() if v)


def test_decoded_read_bounds_and_free():
    store = token_store.DecodedTokenStore(chunk_size_tokens=4)
    store.append(9, np.arange(6, dtype=np.int32))

    with pytest.raises(ValueError, match="outside its 6 decoded tokens"):
        store.read(9, 0, 7)
    with pytest.raises(ValueError, match="outside its 6 decoded tokens"):
        store.read(9, 4, 2)
    with pytest.raises(ValueError, match="outside its 0 decoded tokens"):
        store.read(404, 1)

    assert store.free(9) == 6
    assert store.length(9) == 0
    assert store.read(9).size == 0
    assert store.active_sequences == 0
    assert store.free(9) == 0  # freeing twice is a no-op
    assert store.free(404) == 0

    # A freed id is reusable and starts empty again.
    store.append(9, [42])
    np.testing.assert_array_equal(store.read(9), np.array([42], dtype=np.int32))


def test_decoded_store_rejects_bad_chunk_size():
    with pytest.raises(ValueError, match="must be > 0"):
        token_store.DecodedTokenStore(chunk_size_tokens=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_close_then_unlink_removes_the_segment():
    prefix = _unique_prefix()
    arena = token_store.PromptTokenArena.create(prefix, capacity_bytes=4 * PAGE * 4)
    arena.close()
    arena.unlink()
    with pytest.raises(FileNotFoundError):
        token_store.PromptTokenArena.attach(prefix)
    arena.unlink()  # idempotent


def test_token_ids_must_be_int32_integers(make_arena):
    arena = make_arena(capacity_pages=4)
    with pytest.raises(ValueError, match="do not fit int32"):
        arena.write(np.array([1, 2**31], dtype=np.int64))
    with pytest.raises(TypeError, match="must be integers"):
        arena.write(np.array([1.5, 2.0]))
    assert arena.allocated_pages == 0

    store = token_store.DecodedTokenStore(chunk_size_tokens=4)
    with pytest.raises(ValueError, match="do not fit int32"):
        store.append(1, [-(2**31) - 1])
    with pytest.raises(ValueError, match="do not fit int32"):
        store.append_batch(np.array([7]), np.array([2**40]))
    # The failed batch append left nothing behind for the sequence.
    assert store.length(7) == 0
    assert store.active_sequences == 0


@pytest.mark.skipif(
    not hasattr(mmap, "MADV_REMOVE"), reason="MADV_REMOVE is Linux-only"
)
def test_release_punches_only_whole_system_pages_inside_free_runs(make_arena):
    """Unaligned token pages: live tokens beside a freed run stay intact."""
    page_tokens = 100  # 400 B token pages never align with system pages
    arena = make_arena(
        capacity_pages=256, page_size_tokens=page_tokens, release_free_fraction=1.0
    )
    before = arena.write(_tokens(10 * page_tokens, seed=1))
    freed = arena.write(_tokens(60 * page_tokens, seed=2))
    after = arena.write(_tokens(10 * page_tokens, seed=3))
    arena.free(freed)

    assert arena.release_free_memory() > 0
    for handle, seed in ((before, 1), (after, 3)):
        np.testing.assert_array_equal(
            arena.read(handle), _tokens(10 * page_tokens, seed=seed).astype(np.int32)
        )
    # Punched pages are reusable and read back what is written next.
    again = arena.write(_tokens(60 * page_tokens, seed=4))
    np.testing.assert_array_equal(
        arena.read(again), _tokens(60 * page_tokens, seed=4).astype(np.int32)
    )


def test_decoded_store_capacity_is_enforced_and_chunks_are_reused():
    store = token_store.DecodedTokenStore(chunk_size_tokens=4, capacity_tokens=8)
    assert store.capacity_chunks == 2
    store.append(0, np.arange(8))
    assert store.free_chunks == 0
    with pytest.raises(token_store.TokenStoreCapacityError, match="all 2 chunks"):
        store.append(1, [1])
    assert store.free(0) == 8
    assert store.free_chunks == 2
    store.append_batch(np.array([1, 2]), np.array([5, 6]))
    np.testing.assert_array_equal(store.read(1), np.array([5], dtype=np.int32))
    np.testing.assert_array_equal(store.read(2), np.array([6], dtype=np.int32))
    with pytest.raises(ValueError, match="does not hold one"):
        token_store.DecodedTokenStore(chunk_size_tokens=4, capacity_tokens=3)
