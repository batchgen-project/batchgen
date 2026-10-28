"""POST /v1/turns and the session endpoints, end to end on CPU.

The real BatchScheduler, IntakePool, SchedulingPool, completion pump/listener
and the real turn routes run against a fake worker thread that reads the admit
messages off request_queue and answers on response_queue, the same channels
the GPU worker uses. Covered: request parsing, the batch-path prompt render,
identity in the admit message, event-driven completion (no batch tracker, no
output file, no 0.5 s poll), coexistence with a running batch, the 503/429
gates, abort through the cancel path, a pool fatal, and the timing legs.

The server package __init__ imports the GPU stack, so the server modules are
loaded by path with stubs for the engine-backed ones.
"""

import asyncio
import importlib.util
import json
import queue
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi import FastAPI  # noqa: E402
from pydantic import ValidationError  # noqa: E402

SERVER = Path(__file__).resolve().parents[1] / "batchgen" / "server"


def _load_server_modules():
    real = ("io_struct", "agent_sessions", "intake_pool", "scheduling_pool",
            "batch_scheduler", "turn_api")
    stubbed = (("server_args", "ServerArgs"), ("storage", "StorageManager"),
               ("worker_manager", "WorkerManager"))
    names = ["batchgen.server"] + [f"batchgen.server.{m}" for m in real] + [
        f"batchgen.server.{m}" for m, _ in stubbed]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        package = types.ModuleType("batchgen.server")
        package.__path__ = [str(SERVER)]
        sys.modules["batchgen.server"] = package
        for module, attr in stubbed:
            stub = types.ModuleType(f"batchgen.server.{module}")
            setattr(stub, attr, object)
            sys.modules[stub.__name__] = stub
        loaded = {}
        for module in real:
            spec = importlib.util.spec_from_file_location(
                f"batchgen.server.{module}", SERVER / f"{module}.py")
            loaded[module] = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded[module]
            spec.loader.exec_module(loaded[module])
        return loaded
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_M = _load_server_modules()
io_struct, bs, turn_api = _M["io_struct"], _M["batch_scheduler"], _M["turn_api"]
TurnError = _M["agent_sessions"].TurnError
MODEL = "gpt-oss-120b"


class _Tokenizer:
    """Chat-template stand-in: renders role/content pairs verbatim."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<{m['role']}>{m.get('content') or ''}" for m in messages) + "<assistant>"


class _FakeWorker:
    """Answers admit entries on the real request/response channels.

    An entry whose text contains HOLD is admitted but answered only after
    ``release`` is set; one containing TOO_LONG gets the worker's per-request
    context_length_exceeded error.
    """

    def __init__(self, capacity=8):
        self.state = "serving"
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.admitted = []
        self.admit_messages = []
        self.release = threading.Event()
        self.fatals = []
        self._capacity = capacity
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.response_queue.put({
            "type": "trajectory_pool_capacity",
            "capacity_semantics_version": 1,
            "total_capacity": self._capacity,
        })
        self._thread.start()

    def stop(self):
        self.release.set()
        self.request_queue.put(None)
        self._thread.join(timeout=5)

    def report_worker_fatal(self, reason, persisted=False):
        self.fatals.append(reason)

    def _run(self):
        held = []
        while True:
            try:
                msg = self.request_queue.get(timeout=0.01)
            except queue.Empty:
                msg = "idle"
            if msg is None:
                return
            if self.release.is_set():
                for entry, admitted_at in held:
                    self._answer(entry, admitted_at)
                held = []
            if not isinstance(msg, dict) or msg.get("type") != "admit":
                continue
            self.admit_messages.append(msg)
            for entry in msg["entries"]:
                admitted_at = time.time()
                self.admitted.append(entry)
                if "HOLD" in entry["text"] and not self.release.is_set():
                    held.append((entry, admitted_at))
                else:
                    self._answer(entry, admitted_at)

    def _answer(self, entry, admitted_at):
        item = {
            "type": "completion",
            "request_id": entry["request_id"],
            "batch_id": entry["batch_id"],
            "text": f"reply to {entry['request_id']}",
            "prompt_length": len(entry["text"]),
            "decoded_length": 3,
            "finish_reason": "stop",
            "cached_tokens": 7,
            "computed_tokens": len(entry["text"]) - 7,
            "admitted_at": admitted_at,
            "completed_at": time.time(),
        }
        if "TOO_LONG" in entry["text"]:
            item = {
                "type": "completion",
                "request_id": entry["request_id"],
                "batch_id": entry["batch_id"],
                "error": {"code": "context_length_exceeded",
                          "message": "Prompt length 9 exceeds model context 8"},
                "text": "",
            }
        self.response_queue.put(item)


class _Storage:
    def __init__(self, root: Path):
        self.files_dir = root / "files"
        self.output_dir = root / "out"
        self.files_dir.mkdir()
        self.output_dir.mkdir()
        self.statuses = []
        self.metadata = {}

    def update_batch_status(self, batch_id, status, **kwargs):
        self.statuses.append((batch_id, status, kwargs))

    def save_metadata(self, file_id, meta):
        self.metadata[file_id] = meta


def _scheduler(tmp_path, worker):
    args = SimpleNamespace(
        model=MODEL,
        max_intake_capacity=64,
        incremental_output_dir=str(tmp_path / "incr"),
        no_incremental_save=False,
        parse_thinking=False,
        parse_tool_call=False,
    )
    scheduler = bs.BatchScheduler(_Storage(tmp_path), worker, args)
    scheduler._get_tokenizer = lambda model: _Tokenizer()
    # The fake worker needs no init message (WorkerManager sends it for a
    # real generation before worker_generation_ready()).
    scheduler._pool_initialized = True
    return scheduler


def _turn_body(session_id, content="hi", traj_id="traj-1", parent=None, **extra):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
        "temperature": 0.5,
        "session": {"traj_id": traj_id, "session_id": session_id,
                    "parent_session_id": parent},
    }
    body.update(extra)
    return body


async def _serve(tmp_path, scenario, capacity=8):
    worker = _FakeWorker(capacity=capacity)
    worker.start()
    scheduler = _scheduler(tmp_path, worker)
    await scheduler.start()
    app = FastAPI()
    app.state.scheduler = scheduler
    app.state.worker = worker
    turn_api.register_turn_routes(app)
    try:
        # A real worker publishes its capacity long before the first request.
        await _until(lambda: scheduler._scheduling_pool.capacity > 0)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await scenario(client, scheduler, worker, app)
    finally:
        await scheduler.stop()
        worker.stop()


async def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


# ---------------------------------------------------------------- parsing


def test_turn_request_requires_session_identity():
    ok = io_struct.TurnRequest(**_turn_body("s1", parent="main"))
    assert ok.session.parent_session_id == "main"
    assert ok.max_tokens == 16 and ok.temperature == 0.5
    with pytest.raises(ValidationError):
        io_struct.TurnRequest(**{k: v for k, v in _turn_body("s1").items() if k != "session"})
    for bad in ({"traj_id": "t"}, {"traj_id": "", "session_id": "s"},
                {"traj_id": "t", "session_id": "s", "parent_session_id": ""}):
        with pytest.raises(ValidationError):
            io_struct.TurnRequest(**{**_turn_body("s1"), "session": bad})
    with pytest.raises(ValidationError):
        io_struct.TurnRequest(**_turn_body("s1", stream=True))


def test_close_reason_is_end_of_session_or_end_of_trajectory():
    assert io_struct.CloseSessionRequest(reason="end_of_trajectory").reason == "end_of_trajectory"
    with pytest.raises(ValidationError):
        io_struct.CloseSessionRequest(reason="done")
    assert io_struct.AbortSessionRequest().reason is None


def test_timing_summary_legs_and_overhead():
    t = {"http_received_at": 100.0, "intake_submitted_at": 100.002,
         "admit_sent_at": 100.003, "worker_admitted_at": 100.010,
         "worker_completed_at": 101.010, "completion_routed_at": 101.011,
         "http_responded_at": 101.013}
    s = bs.turn_timing_summary(t)
    assert s["receipt_to_submit_ms"] == 2.0
    assert s["submit_to_admit_sent_ms"] == 1.0
    assert s["admit_sent_to_worker_admit_ms"] == 7.0
    assert s["worker_admit_to_complete_ms"] == 1000.0
    assert s["worker_complete_to_response_ms"] == 3.0
    assert s["overhead_ms"] == 13.0
    assert bs.turn_timing_summary({"http_received_at": 1.0})["overhead_ms"] is None


# ------------------------------------------------- scheduler-level units


def test_admit_message_carries_identity_and_batches_carry_none(tmp_path):
    async def run():
        worker = SimpleNamespace(request_queue=queue.Queue(), response_queue=queue.Queue())
        scheduler = _scheduler(tmp_path, worker)
        scheduler._scheduling_pool.set_capacity(8)
        scheduler._loop = asyncio.get_running_loop()
        ok, error, requests = bs.parse_batch_file(json.dumps({
            "custom_id": "c0", "method": "POST", "url": "/v1/completions",
            "body": {"model": MODEL, "prompt": "p", "max_tokens": 4}}).encode())
        assert ok, error

        async def no_finalizer(*args):
            pass

        scheduler._wait_and_finalize_batch = no_finalizer
        await scheduler._process_batch_pool_mode(
            "batch_1", SimpleNamespace(batchgen_debug=None, max_context_length=None),
            requests, ["p"], [4], None)
        body = io_struct.TurnRequest(**_turn_body("sub", parent="main"))
        waiter = scheduler.submit_turn("turn_x", body, time.time())
        drain = asyncio.ensure_future(scheduler._drain_intake_to_worker())
        msg = await asyncio.get_running_loop().run_in_executor(
            None, lambda: worker.request_queue.get(timeout=5))
        scheduler._stopped.set()
        await drain
        return msg, waiter

    msg, waiter = asyncio.run(run())
    entries = {e["request_id"]: e for e in msg["entries"]}
    turn = entries["turn_x"]
    assert (turn["traj_id"], turn["session_id"], turn["parent_session_id"]) == (
        "traj-1", "sub", "main")
    assert turn["batch_id"] == "turn_x" and turn["max_tokens"] == 16
    # Rendered by the batch path: GPT-OSS gets its reasoning system message.
    assert turn["text"] == "<system>Reasoning: low<user>hi<assistant>"
    assert turn["sampling_params"]["temperature"] == 0.5
    batch = entries["c0@batch_1"]
    assert (batch["traj_id"], batch["session_id"], batch["parent_session_id"]) == (
        None, None, None)
    assert waiter.timing["admit_sent_at"] >= waiter.timing["intake_submitted_at"]


def test_listener_resolves_the_turn_future_without_a_tracker_or_file(tmp_path):
    async def run():
        worker = SimpleNamespace(request_queue=queue.Queue(), response_queue=queue.Queue())
        scheduler = _scheduler(tmp_path, worker)
        scheduler._loop = asyncio.get_running_loop()
        scheduler._completion_inbox = asyncio.Queue()

        async def must_not_poll(*args):
            raise AssertionError("a turn must not use the batch finalizer")

        scheduler._wait_and_finalize_batch = must_not_poll
        listener = asyncio.ensure_future(scheduler._pool_completion_listener())
        waiter = scheduler.submit_turn(
            "turn_y", io_struct.TurnRequest(**_turn_body("s1")), time.time())
        assert scheduler._scheduling_pool.get_batch_tracker("turn_y") is None
        completed_at = time.time()
        scheduler._offer_completion(scheduler._pump_generation, {
            "type": "completion", "request_id": "turn_y", "batch_id": "turn_y",
            "text": "ok", "prompt_length": 5, "decoded_length": 1,
            "admitted_at": completed_at - 0.5, "completed_at": completed_at,
            "cached_tokens": 0, "computed_tokens": 5})
        started = time.monotonic()
        result = await asyncio.wait_for(asyncio.shield(waiter.future), timeout=0.2)
        elapsed = time.monotonic() - started
        scheduler._stopped.set()
        await listener
        return scheduler, waiter, result, elapsed

    scheduler, waiter, result, elapsed = asyncio.run(run())
    assert result["text"] == "ok"
    assert elapsed < 0.2
    assert waiter.timing["completion_routed_at"] >= waiter.timing["worker_completed_at"]
    assert scheduler._turn_waiters == {}
    assert not (tmp_path / "incr").exists() or not list((tmp_path / "incr").iterdir())


def test_cancel_dequeues_a_turn_that_never_reached_a_worker(tmp_path):
    async def run():
        scheduler = _scheduler(tmp_path, SimpleNamespace(response_queue=queue.Queue()))
        waiter = scheduler.submit_turn(
            "turn_q", io_struct.TurnRequest(**_turn_body("s1")), time.time())
        assert scheduler.cancel_turn("turn_q", "test") == "dequeued"
        assert scheduler.cancel_turn("turn_q", "test") is None
        with pytest.raises(TurnError) as info:
            await waiter.future
        return scheduler, info.value

    scheduler, error = asyncio.run(run())
    assert (error.status_code, error.code) == (409, "turn_aborted")
    assert scheduler._intake_pool.size() == 0
    assert scheduler._detached_turns == set()


def test_pool_fatal_fails_pending_turns_from_another_thread(tmp_path):
    async def run():
        scheduler = _scheduler(tmp_path, SimpleNamespace(response_queue=queue.Queue()))
        scheduler._loop = asyncio.get_running_loop()
        waiter = scheduler.submit_turn(
            "turn_f", io_struct.TurnRequest(**_turn_body("s1")), time.time())
        # fail_in_flight runs on the restart thread, not the scheduler loop.
        await asyncio.get_running_loop().run_in_executor(
            None, scheduler.fail_in_flight, "rank 3 died")
        with pytest.raises(TurnError) as info:
            await asyncio.wait_for(waiter.future, timeout=1.0)
        return scheduler, info.value

    scheduler, error = asyncio.run(run())
    assert (error.status_code, error.code) == (503, "worker_fatal")
    assert "rank 3 died" in error.message
    assert scheduler._intake_pool.size() == 0 and scheduler._turn_waiters == {}


def test_submit_rejections(tmp_path):
    async def run():
        scheduler = _scheduler(tmp_path, SimpleNamespace(response_queue=queue.Queue()))
        cases = {
            "n": _turn_body("s1", n=2),
            "max": {k: v for k, v in _turn_body("s1").items() if k != "max_tokens"},
            "model": _turn_body("s1", model="DeepSeek-V3"),
        }
        out = {}
        for name, body in cases.items():
            with pytest.raises(TurnError) as info:
                scheduler.submit_turn(f"turn_{name}", io_struct.TurnRequest(**body), time.time())
            out[name] = (info.value.status_code, info.value.code)
        scheduler._intake_pool._max_capacity = 1
        scheduler.submit_turn("turn_ok", io_struct.TurnRequest(**_turn_body("s1")), time.time())
        with pytest.raises(TurnError) as info:
            scheduler.submit_turn("turn_full", io_struct.TurnRequest(**_turn_body("s2")), time.time())
        out["full"] = (info.value.status_code, info.value.code)
        return scheduler, out

    scheduler, out = asyncio.run(run())
    assert out == {
        "n": (400, "unsupported_n"),
        "max": (400, "max_tokens_required"),
        "model": (400, "model_mismatch"),
        "full": (429, "capacity_exceeded"),
    }
    assert list(scheduler._turn_waiters) == ["turn_ok"]


# ---------------------------------------------------------------- e2e


def test_e2e_two_turns_then_end_of_trajectory(tmp_path):
    async def scenario(client, scheduler, worker, app):
        r = await client.post("/v1/sessions", json={"traj_id": "traj-1", "session_id": "main"})
        assert r.status_code == 200 and r.json()["state"] == "open"
        r = await client.post("/v1/sessions", json={
            "traj_id": "traj-1", "session_id": "sub", "parent_session_id": "main"})
        assert r.status_code == 200
        first = await client.post("/v1/turns", json=_turn_body("main", "turn one"))
        assert app.state.session_registry.get("main").state.value == "idle"
        second = await client.post("/v1/turns", json=_turn_body(
            "sub", "turn two", parent="main"))
        closed = await client.post("/v1/sessions/main/close",
                                   json={"reason": "end_of_trajectory"})
        late = await client.post("/v1/turns", json=_turn_body("sub", "late", parent="main"))
        return first, second, closed, late, list(worker.admitted), scheduler

    first, second, closed, late, admitted, scheduler = asyncio.run(_serve(tmp_path, scenario))
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["object"] == "chat.completion"
    turn_id = body["batchgen"]["turn_id"]
    assert body["choices"][0]["message"] == {"role": "assistant", "content": f"reply to {turn_id}"}
    prompt = "<system>Reasoning: low<user>turn one<assistant>"
    assert body["usage"] == {
        "prompt_tokens": len(prompt), "completion_tokens": 3,
        "total_tokens": len(prompt) + 3, "prompt_tokens_details": {"cached_tokens": 7}}
    ext = body["batchgen"]
    assert ext["session"] == {"traj_id": "traj-1", "session_id": "main", "parent_session_id": None}
    assert (ext["cached_tokens"], ext["computed_tokens"]) == (7, len(prompt) - 7)
    timing = ext["timing"]
    for key in ("http_received_at", "intake_submitted_at", "admit_sent_at",
                "worker_admitted_at", "worker_completed_at", "completion_routed_at",
                "http_responded_at", "overhead_ms"):
        assert timing[key] is not None, key
    # Event-driven both ways: the drain is woken by the submit and the
    # completion is pushed to the waiter, so neither leg waits out a poll
    # period (drain 0.1 s, batch finalizer 0.5 s).
    assert timing["submit_to_admit_sent_ms"] < 50
    assert timing["worker_complete_to_response_ms"] < 250
    assert second.status_code == 200, second.text
    assert second.json()["batchgen"]["session"]["parent_session_id"] == "main"
    assert [(e["traj_id"], e["session_id"], e["parent_session_id"]) for e in admitted] == [
        ("traj-1", "main", None), ("traj-1", "sub", "main")]
    assert closed.status_code == 200
    assert sorted(s["session_id"] for s in closed.json()["closed"]) == ["main", "sub"]
    assert late.status_code == 409 and late.json()["detail"]["code"] == "session_closed"
    assert scheduler._scheduling_pool.num_active_slots() == 0
    assert scheduler._turn_waiters == {}


def test_e2e_rejections_and_gates(tmp_path):
    async def scenario(client, scheduler, worker, app):
        out = {}
        out["unknown"] = await client.post("/v1/turns", json=_turn_body("ghost"))
        await client.post("/v1/sessions", json={"traj_id": "traj-1", "session_id": "s1"})
        out["no_session"] = await client.post("/v1/turns", json={
            k: v for k, v in _turn_body("s1").items() if k != "session"})
        out["no_max"] = await client.post("/v1/turns", json={
            k: v for k, v in _turn_body("s1").items() if k != "max_tokens"})
        out["too_long"] = await client.post("/v1/turns", json=_turn_body("s1", "TOO_LONG"))
        out["state_after_error"] = app.state.session_registry.get("s1").state.value
        worker.state = "restarting"
        out["not_serving"] = await client.post("/v1/turns", json=_turn_body("s1"))
        worker.state = "serving"
        scheduler.intake_pool_usage_pct = lambda: 0.95
        out["pressure"] = await client.post("/v1/turns", json=_turn_body("s1"))
        out["bad_close"] = await client.post("/v1/sessions/s1/close", json={"reason": "done"})
        out["dup_open"] = await client.post(
            "/v1/sessions", json={"traj_id": "traj-1", "session_id": "s1"})
        return out

    out = asyncio.run(_serve(tmp_path, scenario))
    assert out["unknown"].status_code == 404
    assert out["unknown"].json()["detail"]["code"] == "session_not_found"
    assert out["no_session"].status_code == 422
    assert out["no_max"].status_code == 400
    assert out["no_max"].json()["detail"]["code"] == "max_tokens_required"
    assert out["too_long"].status_code == 400
    assert out["too_long"].json()["detail"] == {
        "code": "context_length_exceeded",
        "message": "Prompt length 9 exceeds model context 8"}
    assert out["state_after_error"] == "idle"
    assert out["not_serving"].status_code == 503
    assert out["not_serving"].headers["retry-after"] == "30"
    assert "state=restarting" in out["not_serving"].json()["detail"]
    assert out["pressure"].status_code == 429
    assert out["pressure"].headers["retry-after"] == "30"
    assert out["bad_close"].status_code == 422
    assert out["dup_open"].status_code == 409


def test_e2e_abort_cancels_the_in_flight_turn(tmp_path):
    async def scenario(client, scheduler, worker, app):
        await client.post("/v1/sessions", json={"traj_id": "traj-1", "session_id": "s1"})
        turn = asyncio.ensure_future(client.post("/v1/turns", json=_turn_body("s1", "HOLD")))
        await _until(lambda: worker.admitted)
        aborted = await client.post("/v1/sessions/s1/abort", json={"reason": "tool_crash"})
        answer = await turn
        detached = set(scheduler._detached_turns)
        worker.release.set()
        await _until(lambda: not scheduler._detached_turns)
        await _until(lambda: scheduler._scheduling_pool.num_active_slots() == 0)
        after = await client.post("/v1/turns", json=_turn_body("s1"))
        return aborted, answer, detached, after

    aborted, answer, detached, after = asyncio.run(_serve(tmp_path, scenario))
    assert aborted.status_code == 200, aborted.text
    payload = aborted.json()
    assert payload["state"] == "aborted" and payload["end_reason"] == "tool_crash"
    assert payload["cancelled_turn"] == "detached"
    assert detached == {payload["cancelled_turn_id"]}
    assert answer.status_code == 409
    assert answer.json()["detail"]["code"] == "turn_aborted"
    assert after.status_code == 409 and after.json()["detail"]["code"] == "session_aborted"


def test_e2e_turns_coexist_with_a_running_batch(tmp_path):
    async def scenario(client, scheduler, worker, app):
        ok, error, requests = bs.parse_batch_file("\n".join(json.dumps({
            "custom_id": f"c{i}", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": MODEL, "messages": [{"role": "user", "content": f"b{i}"}],
                     "max_tokens": 4}}) for i in range(3)).encode())
        assert ok, error
        batch = SimpleNamespace(batchgen_debug=None, max_context_length=None,
                                temperature=None, top_p=None, top_k=None)
        prompts, max_tokens, sampling = scheduler._convert_requests_to_worker_inputs(
            requests, batch)
        for sid in ("a", "b"):
            await client.post("/v1/sessions", json={"traj_id": f"traj-{sid}", "session_id": sid})
        await scheduler._process_batch_pool_mode(
            "batch_1", batch, requests, prompts, max_tokens, sampling)
        turns = await asyncio.gather(
            client.post("/v1/turns", json=_turn_body("a", traj_id="traj-a")),
            client.post("/v1/turns", json=_turn_body("b", traj_id="traj-b")),
        )
        storage = scheduler.storage
        await _until(lambda: any(s[1] == bs.BatchStatus.COMPLETED for s in storage.statuses))
        return turns, storage

    turns, storage = asyncio.run(_serve(tmp_path, scenario))
    assert [t.status_code for t in turns] == [200, 200]
    turn_ids = {t.json()["batchgen"]["turn_id"] for t in turns}
    (batch_id, status, kwargs), = [s for s in storage.statuses if s[0] == "batch_1"]
    assert status == bs.BatchStatus.COMPLETED
    rows = [json.loads(line) for line in
            (tmp_path / "incr" / "batch_1.jsonl").read_text().splitlines()]
    assert sorted(r["custom_id"] for r in rows) == ["c0", "c1", "c2"]
    # Turns write no output file of their own and none into the batch's.
    assert sorted(p.name for p in (tmp_path / "incr").iterdir()) == ["batch_1.jsonl"]
    assert not turn_ids & {r["custom_id"] for r in rows}
