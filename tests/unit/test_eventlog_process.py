"""``batchgen.eventlog`` across processes: a record flushed before
``os._exit`` survives, and two spawned ranks of one run write two separate,
line-parseable files.

CPU only; no CUDA, no torch.
"""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()


@pytest.fixture(autouse=True)
def _isolated():
    with tu.isolated_logging(ev):
        yield


_CRASH_SCRIPT = """
import os
import sys
sys.path.insert(0, {tests!r})
import eventlog_testutil as tu
ev = tu.import_eventlog()
ev.configure("server", node=0, run_id=tu.RUN_ID,
             run_started_at=tu.RUN_STARTED_AT, log_dir={log_dir!r},
             console=False)
ev.get_logger("batchgen.server").critical("fatal before hard exit")
if not ev.flush(10.0):
    sys.exit(1)
os._exit(3)
"""


def test_record_flushed_before_os_exit_is_on_disk(tmp_path):
    script = _CRASH_SCRIPT.format(tests=str(tu.TESTS_DIR),
                                  log_dir=str(tmp_path))
    done = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 3, done.stderr
    path = tmp_path / tu.RUN_DIR / "node0" / "server.jsonl"
    (rec,) = tu.read_jsonl(path)
    assert rec["level"] == "CRITICAL"
    assert rec["msg"] == "fatal before hard exit"


def test_spawned_ranks_write_separate_files(tmp_path):
    count = 50
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=tu.rank_child_main,
                    args=(str(tmp_path), rank, count))
        for rank in (0, 1)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(120)
    assert [proc.exitcode for proc in procs] == [0, 0]

    node_dir = tmp_path / tu.RUN_DIR / "node0"
    assert sorted(p.name for p in node_dir.iterdir()) == [
        "rank0.jsonl", "rank1.jsonl"]
    for rank in (0, 1):
        lines = (node_dir / f"rank{rank}.jsonl").read_text(
            encoding="utf-8").splitlines()
        assert len(lines) == count
        recs = [json.loads(line) for line in lines]
        assert {rec["rank"] for rec in recs} == {rank}
        assert {rec["local_rank"] for rec in recs} == {rank}
        assert {rec["proc"] for rec in recs} == {"rank"}
        assert {rec["run_id"] for rec in recs} == {tu.RUN_ID}
        assert len({rec["pid"] for rec in recs}) == 1
        assert [rec["data"]["index"] for rec in recs] == list(range(count))
        assert {rec["event"] for rec in recs} == {"worker.test.tick"}
