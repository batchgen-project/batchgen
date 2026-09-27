from types import SimpleNamespace

from batchgen.server.batch_scheduler import BatchScheduler
from batchgen.server.io_struct import BatchObject, BatchStatus


class _Storage:
    def __init__(self):
        self.updates = []

    def update_batch_status(self, batch_id, status, **fields):
        # Mirror storage.update_batch_status: the error must be a valid BatchObject field.
        BatchObject(id=batch_id, endpoint="/v1/completions", input_file_id="f",
                    completion_window="24h", status=status, created_at=0,
                    expires_at=0, error=fields.get("error"))
        self.updates.append((batch_id, status, fields.get("error")))


def test_fail_all_active_batches_stores_string_errors():
    trackers = {
        "b1": SimpleNamespace(is_complete=False, is_failed=False, error=None),
        "b2": SimpleNamespace(is_complete=False, is_failed=False, error=None),
        "done": SimpleNamespace(is_complete=True, is_failed=False, error=None),
    }
    scheduler = object.__new__(BatchScheduler)
    scheduler._scheduling_pool = SimpleNamespace(_batch_trackers=trackers)
    scheduler.storage = _Storage()

    scheduler._fail_all_active_batches("rank 2 died")

    assert [u[0] for u in scheduler.storage.updates] == ["b1", "b2"]
    for _, status, error in scheduler.storage.updates:
        assert status == BatchStatus.FAILED
        assert error == "worker_fatal: rank 2 died"
