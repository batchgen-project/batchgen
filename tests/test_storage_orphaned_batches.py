"""A restarted server fails batches the previous process left active.

Left active, an orphaned batch can never finish and blocks resubmission of
the same content-deduplicated input file ("already has active batch").
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

# Import the two server modules without running batchgen/__init__.py, which
# loads the native core engine (unavailable without its build or a GPU).
_ROOT = Path(__file__).resolve().parents[1] / "batchgen"
for _name, _path in (("batchgen", _ROOT), ("batchgen.server", _ROOT / "server")):
    if _name not in sys.modules:
        _pkg = types.ModuleType(_name)
        _pkg.__path__ = [str(_path)]
        sys.modules[_name] = _pkg
_io = importlib.import_module("batchgen.server.io_struct")
StorageManager = importlib.import_module("batchgen.server.storage").StorageManager
BatchEndpoint = _io.BatchEndpoint
BatchObject = _io.BatchObject
BatchStatus = _io.BatchStatus
CompletionWindow = _io.CompletionWindow


def _save(storage, batch_id, status, file_id):
    storage.save_batch(BatchObject(
        id=batch_id,
        endpoint=BatchEndpoint.CHAT_COMPLETIONS,
        input_file_id=file_id,
        completion_window=CompletionWindow.ONE_DAY,
        status=status,
        created_at=1,
        expires_at=2,
    ))


def test_active_batches_are_failed_and_unblock_their_file(tmp_path):
    storage = StorageManager(tmp_path)
    _save(storage, "b-progress", BatchStatus.IN_PROGRESS, "file-a")
    _save(storage, "b-validating", BatchStatus.VALIDATING, "file-b")
    _save(storage, "b-cancelling", BatchStatus.CANCELLING, "file-c")
    _save(storage, "b-done", BatchStatus.COMPLETED, "file-d")
    _save(storage, "b-cancelled", BatchStatus.CANCELLED, "file-e")

    failed = storage.fail_orphaned_batches()

    assert sorted(failed) == ["b-cancelling", "b-progress", "b-validating"]
    for batch_id in failed:
        batch = storage.load_batch(batch_id)
        assert batch.status == BatchStatus.FAILED
        assert "server restarted" in batch.error
    assert storage.load_batch("b-done").status == BatchStatus.COMPLETED
    assert storage.load_batch("b-cancelled").status == BatchStatus.CANCELLED
    assert storage.get_active_batch_for_file("file-a") is None


def test_clean_storage_fails_nothing(tmp_path):
    assert StorageManager(tmp_path).fail_orphaned_batches() == []


def test_storage_ownership_is_exclusive_per_live_process(tmp_path):
    # One live server owns one storage path: a second claim on the same path
    # must fail at startup instead of failing the first server's batches.
    first = StorageManager(tmp_path)
    first.acquire_storage_ownership()
    second = StorageManager(tmp_path)
    try:
        second.acquire_storage_ownership()
    except RuntimeError as exc:
        assert "owned by a live server" in str(exc)
    else:
        raise AssertionError("second owner must be refused while the first lives")


def test_batch_writes_are_atomic_full_file_replace(tmp_path):
    # Status updates go through write-then-rename: no reader can observe a
    # truncated JSON file, and no .tmp residue outlives the write.
    storage = StorageManager(tmp_path)
    _save(storage, "b-progress", BatchStatus.IN_PROGRESS, "file-a")
    storage.update_batch_status(storage.load_batch("b-progress").id, BatchStatus.FAILED,
                                error="x")
    assert storage.load_batch("b-progress").status == BatchStatus.FAILED
    assert not list(Path(tmp_path).rglob("*.tmp"))
