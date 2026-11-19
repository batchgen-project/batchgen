"""A finished batch's downloadable output file holds one item per request.

The pool path appends each request's output item to
``{incremental_output_dir}/{batch_id}.jsonl`` as its result is recorded, and
_finalize_batch_output publishes that file as the batch output: it is the
only record of the results. ``--no-incremental-save`` used to turn the append
off, which left every batch COMPLETED with request_counts but no
downloadable output, so the switch is gone and the directory is always set.
"""
import asyncio
import dataclasses
import json

import pytest

from tests.test_batch_request_counts import LONG, MODEL, _Server, io
from tests.test_server_args import server_args_module

ServerArgs = server_args_module.ServerArgs


def test_the_cli_has_no_switch_that_turns_batch_output_off(capsys):
    parser = server_args_module._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", MODEL, "--no-incremental-save"])
    assert "unrecognized arguments: --no-incremental-save" in capsys.readouterr().err
    assert "no_incremental_save" not in {
        field.name for field in dataclasses.fields(ServerArgs)}


def test_the_output_directory_is_always_set(tmp_path):
    derived = ServerArgs(model=MODEL, storage_path=tmp_path)
    assert derived.incremental_output_dir == str(tmp_path / "incremental")
    explicit = ServerArgs(model=MODEL, storage_path=tmp_path,
                          incremental_output_dir=str(tmp_path / "out"))
    assert explicit.incremental_output_dir == str(tmp_path / "out")


def test_the_downloaded_output_holds_one_item_per_request(tmp_path):
    """Launch-default ServerArgs: a completed, a worker-failed and an
    admission-rejected request each have their item in the output file
    GET /v1/files/{output_file_id}/content serves."""

    async def run():
        server = _Server(tmp_path)
        server.scheduler.server_args = ServerArgs(
            model=MODEL, storage_path=server.root, max_intake_capacity=64)
        await server.start()
        server.create("b", {"ok": "p1", "long": LONG, "err": "p3"})
        await server.scheduler._process_batch("b")
        await server.complete("b", "ok")
        await server.complete("b", "err", error="worker_error")
        await server.until_status("b", io.BatchStatus.COMPLETED)
        await server.stop()
        return server

    server = asyncio.run(run())
    batch = server.storage.load_batch("b")
    assert batch.request_counts.model_dump() == {
        "total": 3, "completed": 1, "failed": 2}
    served = server.storage.files_dir / batch.output_file_id
    items = [json.loads(line) for line in served.read_text().splitlines()]
    assert sorted(item["custom_id"] for item in items) == ["err", "long", "ok"]
    assert {item["custom_id"]: item["error"] is None for item in items} == {
        "ok": True, "long": False, "err": False}
    browsable = server.storage.output_dir / f"{batch.output_file_id}.jsonl"
    assert browsable.read_text() == served.read_text()
