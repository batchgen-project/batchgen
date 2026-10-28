"""``batchgen.eventlog`` context fields: ``bind`` nesting, thread and asyncio
isolation, and stamping on the caller's thread.

CPU only; no CUDA, no torch.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()
current = ev.context.current


@pytest.fixture(autouse=True)
def _isolated():
    with tu.isolated_logging(ev):
        yield


def _configure(tmp_path: Path) -> None:
    ev.configure("rank", node=0, rank=0, run_id=tu.RUN_ID,
                 run_started_at=tu.RUN_STARTED_AT, log_dir=tmp_path,
                 console=False)


def _records_by_msg() -> dict[str, dict]:
    assert ev.flush()
    return {rec["msg"]: rec for rec in tu.read_jsonl(ev.jsonl_path())}


def test_nested_bind_restores_outer_values():
    assert current() == {}
    with ev.bind(batch_id="b1", wave=1):
        assert current() == {"batch_id": "b1", "wave": 1}
        with ev.bind(wave=2, step=7):
            assert current() == {"batch_id": "b1", "wave": 2, "step": 7}
        assert current() == {"batch_id": "b1", "wave": 1}
    assert current() == {}


def test_bind_restores_on_exception():
    with pytest.raises(RuntimeError):
        with ev.bind(batch_id="b1"):
            raise RuntimeError("boom")
    assert current() == {}


def test_set_and_clear_context():
    ev.set_context(batch_id="b2", wave=3, step=4)
    assert current() == {"batch_id": "b2", "wave": 3, "step": 4}
    ev.clear_context("step")
    assert current() == {"batch_id": "b2", "wave": 3}
    ev.clear_context()
    assert current() == {}


@pytest.mark.parametrize("fields,error", [
    ({"request": "x"}, ValueError),
    ({"wave": "2"}, TypeError),
    ({"batch_id": 5}, TypeError),
])
def test_bad_context_fields_are_rejected(fields, error):
    with pytest.raises(error):
        with ev.bind(**fields):
            pass
    with pytest.raises(error):
        ev.set_context(**fields)
    assert current() == {}


def test_unknown_clear_name_is_rejected():
    with pytest.raises(ValueError):
        ev.clear_context("request")


def test_fields_come_from_the_callers_context(tmp_path):
    _configure(tmp_path)
    logger = ev.get_logger("batchgen.t")
    with ev.bind(batch_id="b_caller", wave=1, step=2):
        logger.info("inside")
    # The listener writes after the bind has been left.
    logger.info("outside")
    recs = _records_by_msg()
    inside = recs["inside"]
    assert (inside["batch_id"], inside["wave"], inside["step"]) == (
        "b_caller", 1, 2)
    assert inside["thread"] == threading.current_thread().name
    assert "batch_id" not in recs["outside"]


def test_threads_have_isolated_context(tmp_path):
    _configure(tmp_path)
    logger = ev.get_logger("batchgen.t")
    barrier = threading.Barrier(2)
    errors = []

    def work(name: str) -> None:
        try:
            with ev.bind(batch_id=name):
                barrier.wait(5)  # both threads are inside their bind
                logger.info("from %s", name)
                barrier.wait(5)
            assert current() == {}
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    with ev.bind(batch_id="main"):
        threads = [threading.Thread(target=work, args=(n,), name=n)
                   for n in ("t_a", "t_b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert current() == {"batch_id": "main"}
    assert errors == []
    recs = _records_by_msg()
    for name in ("t_a", "t_b"):
        assert recs[f"from {name}"]["batch_id"] == name
        assert recs[f"from {name}"]["thread"] == name


def test_asyncio_tasks_have_isolated_context(tmp_path):
    _configure(tmp_path)
    logger = ev.get_logger("batchgen.t")

    async def task(name: str, wave: int) -> None:
        with ev.bind(batch_id=name, wave=wave):
            await asyncio.sleep(0)
            logger.info("task %s", name)
            await asyncio.sleep(0)
            assert current() == {"batch_id": name, "wave": wave, "step": 9}

    async def main() -> None:
        with ev.bind(step=9):
            await asyncio.gather(task("a", 1), task("b", 2))
            assert current() == {"step": 9}

    asyncio.run(main())
    recs = _records_by_msg()
    assert (recs["task a"]["batch_id"], recs["task a"]["wave"]) == ("a", 1)
    assert (recs["task b"]["batch_id"], recs["task b"]["wave"]) == ("b", 2)
    # Tasks inherit the context of their creator.
    assert recs["task a"]["step"] == recs["task b"]["step"] == 9
