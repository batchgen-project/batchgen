from types import SimpleNamespace

from batchgen.sequence import SequenceStatus, StatusEdge, TransitionPlan
from batchgen.worker.admission.plan import update_batch_status


class _Pool:
    def __init__(self, missing=(), rejected=()):
        self.missing = list(missing)
        self.rejected = list(rejected)
        self.planned = None
        self.applied = None

    def plan_transitions(self, uuids, new_status):
        self.planned = (uuids, new_status)
        return ("transition",), self.missing

    def apply(self, plan):
        self.applied = plan
        return self.rejected


def test_update_batch_status_warns_for_missing_uuid(caplog):
    pool = _Pool(missing=["missing-uuid"])

    update_batch_status(pool, 3, ["missing-uuid"], SequenceStatus.IN_DECODE, StatusEdge.DECODE_ENTRY)

    assert "Rank 3: Sequence missing-uuid not found in global_batch" in caplog.text


def test_update_batch_status_warns_for_rejected_transition(caplog):
    rejection = SimpleNamespace(
        transition=SimpleNamespace(uuid="rejected-uuid"),
        reason="already complete",
    )
    pool = _Pool(rejected=[rejection])

    update_batch_status(pool, 4, ["rejected-uuid"], SequenceStatus.IN_DECODE, StatusEdge.DECODE_ENTRY)

    assert "Rank 4: Invalid status transition for rejected-uuid: already complete" in caplog.text


def test_update_batch_status_accepts_string_uuid():
    pool = _Pool()

    update_batch_status(pool, 0, "single-uuid", SequenceStatus.IN_DECODE, StatusEdge.DECODE_ENTRY)

    assert pool.planned == (["single-uuid"], SequenceStatus.IN_DECODE)
    assert isinstance(pool.applied, TransitionPlan)
