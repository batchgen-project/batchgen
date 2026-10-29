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
CONTEXT = 4096  # the served model's context length in these tests


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
        self.control_messages = []
        # Every message the server put on request_queue, in order.
        self.messages = []
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
            if msg != "idle":
                self.messages.append(msg)
            if self.release.is_set():
                for entry, admitted_at in held:
                    self._answer(entry, admitted_at)
                held = []
            if isinstance(msg, dict) and msg.get("type") == "session_control":
                self.control_messages.append(msg)
                continue
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
        if "KVKEPT" in entry["text"]:
            # --enable-session-kv-hold: the worker reports what the turn
            # reused from its session's hold and what it holds now.
            item["session_resume"] = {"lcp_tokens": 70, "cached_tokens": 7,
                                      "dropped": None}
            item["session_hold"] = {"held": True, "kv_tokens": 12, "pages": 1}
            item["session_id"] = entry.get("session_id")
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


def _scheduler(tmp_path, worker, session_hold=False):
    args = SimpleNamespace(
        model=MODEL,
        enable_session_kv_hold=session_hold,
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
    mark_live = getattr(scheduler, "_mark_generation_live", None)
    if mark_live is not None and hasattr(worker, "request_queue"):
        # ... and this generation is live: session_control may reach it.
        mark_live()
    return scheduler


def _ns_worker(**extra):
    """A serving worker pool with only the channels a scheduler unit needs."""
    return SimpleNamespace(state="serving", response_queue=queue.Queue(), **extra)


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


async def _serve(tmp_path, scenario, capacity=8, session_hold=False):
    worker = _FakeWorker(capacity=capacity)
    worker.start()
    scheduler = _scheduler(tmp_path, worker, session_hold=session_hold)
    await scheduler.start()
    app = FastAPI()
    app.state.scheduler = scheduler
    app.state.worker = worker
    app.state.model_object = SimpleNamespace(max_context_length=CONTEXT)
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
        worker = _ns_worker(request_queue=queue.Queue())
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
        worker = _ns_worker(request_queue=queue.Queue())
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


def _drain_queue(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def test_session_control_never_precedes_a_restarted_generations_init(tmp_path):
    """A close or abort during a worker restart must not reach the new
    generation's queue ahead of its "init" (the worker dies on it)."""
    old_queue = queue.Queue()
    worker = _ns_worker(request_queue=old_queue)
    scheduler = _scheduler(tmp_path, worker, session_hold=True)
    scheduler.release_sessions(["s0"], "closed")
    assert [m["type"] for m in _drain_queue(old_queue)] == ["session_control"]

    # WorkerManager.restart_workers: quiesce, teardown, fresh channels, spawn.
    scheduler.quiesce()
    scheduler.release_sessions(["s1"], "closed")      # close mid-restart
    new_queue = queue.Queue()
    worker.request_queue = new_queue
    scheduler.release_sessions(["s2"], "aborted")     # abort mid-restart
    assert _drain_queue(old_queue) == []
    assert _drain_queue(new_queue) == []
    # The manager sends the generation's init, then calls the ready hook.
    new_queue.put({"type": "init"})
    scheduler.worker_generation_ready()
    scheduler.release_sessions(["s3"], "closed")
    sent = _drain_queue(new_queue)
    assert [m["type"] for m in sent] == ["init", "session_control"]
    assert sent[1]["release"] == [{"session_id": "s3", "reason": "closed"}]


def test_a_restart_clears_the_previous_generations_hold_metrics(tmp_path):
    """The restart frees the old generation's held KV; /v1/pool/status must
    not keep reporting its holds until the new generation publishes."""
    worker = _ns_worker(request_queue=queue.Queue())
    scheduler = _scheduler(tmp_path, worker, session_hold=True)
    scheduler._session_hold_info = {"held_sessions": 4, "held_kv_pages": 90}
    scheduler.quiesce()
    assert scheduler._session_hold_info is None


def test_session_control_is_dropped_after_a_fatal_until_the_next_generation(tmp_path):
    worker = _ns_worker(request_queue=queue.Queue())
    scheduler = _scheduler(tmp_path, worker, session_hold=True)
    # WorkerManager._handle_worker_failure: quiesce, teardown, reconcile.
    scheduler.fail_in_flight("rank 3 died")
    worker.request_queue = queue.Queue()
    scheduler.release_sessions(["s1"], "closed")
    assert _drain_queue(worker.request_queue) == []


def test_session_control_waits_for_the_first_generations_init(tmp_path):
    worker = _ns_worker(request_queue=queue.Queue())
    args = SimpleNamespace(model=MODEL, max_intake_capacity=64,
                           enable_session_kv_hold=True)
    scheduler = bs.BatchScheduler(_Storage(tmp_path), worker, args)
    scheduler.release_sessions(["s1"], "closed")
    assert _drain_queue(worker.request_queue) == []


def test_nothing_is_sent_to_the_worker_without_session_hold(tmp_path):
    worker = _ns_worker(request_queue=queue.Queue())
    scheduler = _scheduler(tmp_path, worker, session_hold=False)
    scheduler.release_sessions(["s1"], "closed")
    scheduler._send_session_control(cancel=["turn_x"], reason="abort")
    assert _drain_queue(worker.request_queue) == []


def test_worker_main_loop_skips_session_control_before_init():
    """Every rank gets the broadcast message and skips it with ``continue``,
    before the init branch and the unexpected-message raise."""
    import ast

    source = (SERVER.parent / "server_worker_main_loop.py").read_text()
    impl = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_server_worker_main_impl"
    )
    loop = next(
        node for node in ast.walk(impl)
        if isinstance(node, ast.While) and any(
            "Pool mode init received" in ast.unparse(stmt) for stmt in node.body)
    )
    texts = [ast.unparse(stmt) for stmt in loop.body]
    guard = next(i for i, text in enumerate(texts)
                 if "'session_control'" in text and text.startswith("if "))
    init = next(i for i, text in enumerate(texts) if "Pool mode init received" in text)
    assert guard < init
    assert isinstance(loop.body[guard].body[-1], ast.Continue)


class _ClosedQueue:
    def put(self, item):
        raise ValueError("Queue is closed")


def test_cancel_fails_the_waiter_even_when_the_worker_send_fails(tmp_path):
    """L1: the waiter leaves _turn_waiters first, so a send that raises must
    not leave its future pending forever."""
    async def run():
        worker = _ns_worker(request_queue=queue.Queue())
        scheduler = _scheduler(tmp_path, worker, session_hold=True)
        waiter = scheduler.submit_turn(
            "turn_a", io_struct.TurnRequest(**_turn_body("s1")), time.time())
        # Admitted: the drain took it out of the IntakePool.
        scheduler._intake_pool.remove_batch_entries("turn_a")
        # The generation's channel closed under the cancel (restart window).
        scheduler._control_queue = _ClosedQueue()
        outcome = scheduler.cancel_turn("turn_a", "test")
        with pytest.raises(TurnError) as info:
            await asyncio.wait_for(waiter.future, timeout=1.0)
        return outcome, info.value, scheduler

    outcome, error, scheduler = asyncio.run(run())
    assert outcome == "cancelled"
    assert (error.status_code, error.code) == (409, "turn_aborted")
    assert scheduler._detached_turns == {"turn_a"}


def test_a_generation_without_a_request_queue_is_not_live(tmp_path):
    worker = _ns_worker(request_queue=None)
    scheduler = _scheduler(tmp_path, worker, session_hold=True)
    scheduler._mark_generation_live()
    scheduler.release_sessions(["s1"], "closed")   # no AttributeError
    assert scheduler._control_queue is None


def test_cancel_dequeues_a_turn_that_never_reached_a_worker(tmp_path):
    async def run():
        scheduler = _scheduler(tmp_path, _ns_worker())
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
        scheduler = _scheduler(tmp_path, _ns_worker())
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


def test_fatal_fails_a_turn_submitted_after_the_restart_thread_called_it(tmp_path):
    # The restart thread calls fail_in_flight while the loop is busy; a turn
    # that the loop submits before the fail callback runs must be failed by
    # that callback, not left in _turn_waiters with no completion to come.
    async def run():
        scheduler = _scheduler(tmp_path, _ns_worker())
        scheduler._loop = asyncio.get_running_loop()
        restart = threading.Thread(target=scheduler.fail_in_flight, args=("rank 3 died",))
        restart.start()
        restart.join()  # blocks the loop: its callback cannot have run yet
        late = scheduler.submit_turn(
            "turn_late", io_struct.TurnRequest(**_turn_body("s1")), time.time())
        try:
            await asyncio.wait_for(asyncio.shield(late.future), timeout=1.0)
        except TurnError as exc:
            return scheduler, exc
        except asyncio.TimeoutError:
            return scheduler, None

    scheduler, error = asyncio.run(run())
    assert error is not None, "turn submitted after the fatal snapshot was never failed"
    assert (error.status_code, error.code) == (503, "worker_fatal")
    assert scheduler._turn_waiters == {} and scheduler._intake_pool.size() == 0


def test_turns_are_refused_until_the_restart_resumes(tmp_path):
    async def run():
        worker = _ns_worker()
        scheduler = _scheduler(tmp_path, worker)
        scheduler._loop = asyncio.get_running_loop()
        out = {}

        def submit(name):
            try:
                scheduler.submit_turn(
                    f"turn_{name}", io_struct.TurnRequest(**_turn_body("s1")), time.time())
                return "accepted"
            except TurnError as exc:
                return (exc.status_code, exc.code)

        await asyncio.get_running_loop().run_in_executor(
            None, scheduler.fail_in_flight, "rank 3 died")
        await asyncio.sleep(0)
        out["after_fatal"] = submit("a")
        worker.state = "restarting"
        out["restarting"] = submit("b")
        # Restart thread: resume() then the manager flips the state to serving.
        await asyncio.get_running_loop().run_in_executor(None, scheduler.resume)
        await asyncio.sleep(0)
        out["resumed_not_serving"] = submit("c")
        worker.state = "serving"
        out["serving"] = submit("d")
        return scheduler, out

    scheduler, out = asyncio.run(run())
    assert out == {
        "after_fatal": (503, "worker_fatal"),
        "restarting": (503, "worker_not_serving"),
        "resumed_not_serving": (503, "worker_not_serving"),
        "serving": "accepted",
    }
    assert list(scheduler._turn_waiters) == ["turn_d"]


def test_fatal_never_resolves_a_turn_future_off_the_loop(tmp_path):
    # With the scheduler loop stopped (not closed), the fail is queued on it,
    # not run on the calling thread against the loop's futures.
    scheduler = _scheduler(tmp_path, _ns_worker())
    loop = asyncio.new_event_loop()
    try:
        scheduler._loop = loop

        async def submit():
            return scheduler.submit_turn(
                "turn_s", io_struct.TurnRequest(**_turn_body("s1")), time.time())

        waiter = loop.run_until_complete(submit())
        scheduler.fail_in_flight("rank 3 died")
        done_off_loop = waiter.future.done()
        loop.run_until_complete(asyncio.sleep(0))
        error = waiter.future.exception()
    finally:
        loop.close()
    assert not done_off_loop
    assert (error.status_code, error.code) == (503, "worker_fatal")
    assert scheduler._turn_waiters == {}


def test_submit_rejections(tmp_path):
    async def run():
        scheduler = _scheduler(tmp_path, _ns_worker())
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

    holder = {}

    async def scenario_with_worker(client, scheduler, worker, app):
        holder["worker"] = worker
        return await scenario(client, scheduler, worker, app)

    aborted, answer, detached, after = asyncio.run(
        _serve(tmp_path, scenario_with_worker, session_hold=True)
    )
    assert aborted.status_code == 200, aborted.text
    payload = aborted.json()
    assert payload["state"] == "aborted" and payload["end_reason"] == "tool_crash"
    # The worker is told to cancel the admitted turn and to free the
    # session's held KV, in that order, through its admission queue.
    assert payload["cancelled_turn"] == "cancelled"
    assert detached == {payload["cancelled_turn_id"]}
    control = holder["worker"].control_messages
    assert [m["cancel"] for m in control] == [[payload["cancelled_turn_id"]], []]
    assert [m["release"] for m in control] == [
        [], [{"session_id": "s1", "reason": "aborted"}]]
    assert answer.status_code == 409
    assert answer.json()["detail"]["code"] == "turn_aborted"
    assert after.status_code == 409 and after.json()["detail"]["code"] == "session_aborted"


def test_e2e_close_releases_the_held_kv_of_every_closed_session(tmp_path):
    async def scenario(client, scheduler, worker, app):
        await client.post("/v1/sessions", json={"traj_id": "traj-1", "session_id": "main"})
        await client.post("/v1/sessions", json={
            "traj_id": "traj-1", "session_id": "sub", "parent_session_id": "main"})
        await client.post("/v1/turns", json=_turn_body("main"))
        single = await client.post("/v1/sessions/sub/close",
                                   json={"reason": "end_of_session"})
        await client.post("/v1/sessions", json={"traj_id": "traj-2", "session_id": "m2"})
        await client.post("/v1/sessions", json={
            "traj_id": "traj-2", "session_id": "s2", "parent_session_id": "m2"})
        await client.post("/v1/turns", json=_turn_body("m2", traj_id="traj-2"))
        whole = await client.post("/v1/sessions/s2/close",
                                  json={"reason": "end_of_trajectory"})
        await _until(lambda: len(worker.control_messages) == 2)
        return single, whole, list(worker.control_messages)

    single, whole, control = asyncio.run(
        _serve(tmp_path, scenario, session_hold=True))
    assert single.status_code == 200 and whole.status_code == 200
    assert control[0]["release"] == [{"session_id": "sub", "reason": "closed"}]
    assert sorted(item["session_id"] for item in control[1]["release"]) == ["m2", "s2"]
    assert all(m["cancel"] == [] for m in control)


def test_e2e_turn_reports_its_session_hold_and_pool_status_metrics(tmp_path):
    async def scenario(client, scheduler, worker, app):
        await client.post("/v1/sessions", json={"traj_id": "traj-1", "session_id": "s1"})
        answer = await client.post("/v1/turns", json=_turn_body("s1", "KVKEPT"))
        worker.response_queue.put({
            "type": "session_hold_metrics", "held_sessions": 1,
            "held_kv_tokens": 12, "held_kv_pages": 1})
        await _until(lambda: scheduler._session_hold_info is not None)
        return answer, scheduler._session_hold_info

    answer, metrics = asyncio.run(_serve(tmp_path, scenario, session_hold=True))
    assert answer.status_code == 200, answer.text
    meta = answer.json()["batchgen"]
    assert meta["session_resume"] == {"lcp_tokens": 70, "cached_tokens": 7,
                                      "dropped": None}
    assert meta["session_hold"] == {"held": True, "kv_tokens": 12, "pages": 1}
    assert metrics == {"held_sessions": 1, "held_kv_tokens": 12, "held_kv_pages": 1}


_TURN_ID = __import__("re").compile(r"turn_[0-9a-f]{24}")
_VOLATILE_KEYS = {"created", "opened_at", "updated_at", "timing", "id",
                  "admitted_at", "completed_at"}


def _normalize_flag_off_record(record):
    """Replace ids and wall-clock values so two runs compare equal."""
    names = {}

    def turn_name(match):
        return names.setdefault(match.group(0), f"turn_{len(names)}")

    def walk(value):
        if isinstance(value, dict):
            return {key: ("<volatile>" if key in _VOLATILE_KEYS else walk(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, str):
            return _TURN_ID.sub(turn_name, value)
        return value

    return walk(record)


def record_flag_off_turn_traffic(tmp_path):
    """Turns, an abort of an admitted turn, a close and the server -> worker
    request_queue traffic they cause, without --enable-session-kv-hold.

    Also run against the server modules of main (merge base 686d8e78) to
    produce FLAG_OFF_MAIN_RECORD below.
    """
    async def scenario(client, scheduler, worker, app):
        out = []

        async def call(label, method, path, body=None):
            response = await getattr(client, method)(path, json=body)
            out.append([label, response.status_code, response.json()])

        await call("open s1", "post", "/v1/sessions",
                   {"traj_id": "traj-1", "session_id": "s1"})
        await call("turn s1", "post", "/v1/turns", _turn_body("s1", "first"))
        held = asyncio.ensure_future(
            client.post("/v1/turns", json=_turn_body("s1", "HOLD")))
        await _until(lambda: len(worker.admitted) == 2)
        await call("abort s1", "post", "/v1/sessions/s1/abort",
                   {"reason": "tool_crash"})
        aborted = await held
        out.append(["aborted turn", aborted.status_code, aborted.json()])
        worker.release.set()
        await _until(lambda: not scheduler._detached_turns)
        await call("open s2", "post", "/v1/sessions",
                   {"traj_id": "traj-2", "session_id": "s2"})
        await call("turn s2", "post", "/v1/turns",
                   _turn_body("s2", "second", traj_id="traj-2"))
        await call("close s2", "post", "/v1/sessions/s2/close",
                   {"reason": "end_of_trajectory"})
        await asyncio.sleep(0.1)
        return {"http": out, "request_queue": list(worker.messages)}

    return _normalize_flag_off_record(asyncio.run(_serve(tmp_path, scenario)))


# record_flag_off_turn_traffic() run against the server modules of main at
# 686d8e78 (the merge base of this branch), normalized.
FLAG_OFF_MAIN_RECORD = json.loads(r"""
{
 "http": [
  [
   "open s1",
   200,
   {
    "end_reason": null,
    "in_flight_turn_id": null,
    "object": "session",
    "parent_session_id": null,
    "session_id": "s1",
    "state": "open",
    "traj_id": "traj-1",
    "turns": 0
   }
  ],
  [
   "turn s1",
   200,
   {
    "batchgen": {
     "cached_tokens": 7,
     "computed_tokens": 37,
     "session": {
      "parent_session_id": null,
      "session_id": "s1",
      "traj_id": "traj-1"
     },
     "timing": "<volatile>",
     "turn_id": "turn_0"
    },
    "choices": [
     {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
       "content": "reply to turn_0",
       "role": "assistant"
      }
     }
    ],
    "created": "<volatile>",
    "id": "<volatile>",
    "model": "gpt-oss-120b",
    "object": "chat.completion",
    "usage": {
     "completion_tokens": 3,
     "prompt_tokens": 44,
     "prompt_tokens_details": {
      "cached_tokens": 7
     },
     "total_tokens": 47
    }
   }
  ],
  [
   "abort s1",
   200,
   {
    "cancelled_turn": "detached",
    "cancelled_turn_id": "turn_1",
    "end_reason": "tool_crash",
    "in_flight_turn_id": null,
    "object": "session",
    "parent_session_id": null,
    "session_id": "s1",
    "state": "aborted",
    "traj_id": "traj-1",
    "turns": 2
   }
  ],
  [
   "aborted turn",
   409,
   {
    "detail": {
     "code": "turn_aborted",
     "message": "turn turn_1 was aborted: session s1 aborted (tool_crash)"
    }
   }
  ],
  [
   "open s2",
   200,
   {
    "end_reason": null,
    "in_flight_turn_id": null,
    "object": "session",
    "parent_session_id": null,
    "session_id": "s2",
    "state": "open",
    "traj_id": "traj-2",
    "turns": 0
   }
  ],
  [
   "turn s2",
   200,
   {
    "batchgen": {
     "cached_tokens": 7,
     "computed_tokens": 38,
     "session": {
      "parent_session_id": null,
      "session_id": "s2",
      "traj_id": "traj-2"
     },
     "timing": "<volatile>",
     "turn_id": "turn_2"
    },
    "choices": [
     {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
       "content": "reply to turn_2",
       "role": "assistant"
      }
     }
    ],
    "created": "<volatile>",
    "id": "<volatile>",
    "model": "gpt-oss-120b",
    "object": "chat.completion",
    "usage": {
     "completion_tokens": 3,
     "prompt_tokens": 45,
     "prompt_tokens_details": {
      "cached_tokens": 7
     },
     "total_tokens": 48
    }
   }
  ],
  [
   "close s2",
   200,
   {
    "closed": [
     {
      "end_reason": "end_of_trajectory",
      "in_flight_turn_id": null,
      "object": "session",
      "parent_session_id": null,
      "session_id": "s2",
      "state": "closed",
      "traj_id": "traj-2",
      "turns": 1
     }
    ],
    "object": "session.close",
    "reason": "end_of_trajectory"
   }
  ]
 ],
 "request_queue": [
  {
   "entries": [
    {
     "batch_id": "turn_0",
     "batchgen_debug": {},
     "ignore_eos": false,
     "max_tokens": 16,
     "parent_session_id": null,
     "priority": 0,
     "request_id": "turn_0",
     "sampling_params": {
      "temperature": 0.5,
      "top_k": null,
      "top_p": 1.0
     },
     "session_id": "s1",
     "text": "<system>Reasoning: low<user>first<assistant>",
     "traj_id": "traj-1"
    }
   ],
   "type": "admit"
  },
  {
   "entries": [
    {
     "batch_id": "turn_1",
     "batchgen_debug": {},
     "ignore_eos": false,
     "max_tokens": 16,
     "parent_session_id": null,
     "priority": 0,
     "request_id": "turn_1",
     "sampling_params": {
      "temperature": 0.5,
      "top_k": null,
      "top_p": 1.0
     },
     "session_id": "s1",
     "text": "<system>Reasoning: low<user>HOLD<assistant>",
     "traj_id": "traj-1"
    }
   ],
   "type": "admit"
  },
  {
   "entries": [
    {
     "batch_id": "turn_2",
     "batchgen_debug": {},
     "ignore_eos": false,
     "max_tokens": 16,
     "parent_session_id": null,
     "priority": 0,
     "request_id": "turn_2",
     "sampling_params": {
      "temperature": 0.5,
      "top_k": null,
      "top_p": 1.0
     },
     "session_id": "s2",
     "text": "<system>Reasoning: low<user>second<assistant>",
     "traj_id": "traj-2"
    }
   ],
   "type": "admit"
  }
 ]
}
""")


def test_flag_off_turn_responses_and_worker_traffic_match_main(tmp_path):
    """M2: without --enable-session-kv-hold, /v1/turns, abort and close
    answer exactly as on main and put exactly main's messages on the worker
    queue (no session_control, abort leaves the turn "detached")."""
    record = record_flag_off_turn_traffic(tmp_path)
    assert json.loads(json.dumps(record)) == FLAG_OFF_MAIN_RECORD


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


def test_e2e_session_ids_with_a_slash_reach_close_and_abort(tmp_path):
    # The documented id form is "<traj>/<name>" (e.g. "traj-17/main"). The
    # close and abort routes must accept it, not 404 on the extra path segment.
    async def scenario(client, scheduler, worker, app):
        out = {}
        for sid, parent in (("traj-17/main", None), ("traj-17/sub/1", "traj-17/main"),
                            ("traj-18/main", None)):
            r = await client.post("/v1/sessions", json={
                "traj_id": sid.split("/")[0], "session_id": sid, "parent_session_id": parent})
            assert r.status_code == 200, r.text
        out["turn"] = await client.post("/v1/turns", json=_turn_body(
            "traj-17/sub/1", "hi", traj_id="traj-17", parent="traj-17/main"))
        out["close"] = await client.post("/v1/sessions/traj-17/sub/1/close",
                                         json={"reason": "end_of_trajectory"})
        out["abort"] = await client.post("/v1/sessions/traj-18/main/abort",
                                         json={"reason": "tool_crash"})
        out["abort_unknown"] = await client.post("/v1/sessions/traj-19/main/abort")
        return out

    out = asyncio.run(_serve(tmp_path, scenario))
    assert out["turn"].status_code == 200, out["turn"].text
    assert out["close"].status_code == 200, out["close"].text
    assert sorted(s["session_id"] for s in out["close"].json()["closed"]) == [
        "traj-17/main", "traj-17/sub/1"]
    assert out["abort"].status_code == 200, out["abort"].text
    assert out["abort"].json()["session_id"] == "traj-18/main"
    assert out["abort"].json()["state"] == "aborted"
    assert out["abort_unknown"].status_code == 404
    assert out["abort_unknown"].json()["detail"]["code"] == "session_not_found"


def _route_endpoint(app, path):
    (route,) = [r for r in app.routes if getattr(r, "path", None) == path]
    return route.endpoint


def test_e2e_abort_between_completion_routing_and_response_is_already_completed(tmp_path):
    # The listener routes the completion and resolves the future; the turn's
    # handler has not resumed yet. An abort arriving in exactly that window
    # must not mark the session aborted while the turn returns 200.
    async def scenario(client, scheduler, worker, app):
        await client.post("/v1/sessions", json={"traj_id": "traj-1", "session_id": "s1"})
        abort = _route_endpoint(app, "/v1/sessions/{session_id:path}/abort")
        seen = {}
        original = scheduler._resolve_turn

        def resolve_then_abort(turn_id, result):
            original(turn_id, result)
            # The abort handler has no await: drive it to completion right
            # here, on the loop, before the turn handler can run again.
            coro = abort(request=SimpleNamespace(app=app), session_id="s1",
                         body=io_struct.AbortSessionRequest(reason="late"))
            try:
                coro.send(None)
            except StopIteration as stop:
                seen["abort"] = ("ok", stop.value)
            except Exception as exc:  # HTTPException from the handler
                seen["abort"] = ("error", exc.status_code, exc.detail)
            seen["state_after_abort"] = app.state.session_registry.get("s1").state.value

        scheduler._resolve_turn = resolve_then_abort
        turn = await client.post("/v1/turns", json=_turn_body("s1"))
        seen["state_after_turn"] = app.state.session_registry.get("s1").state.value
        return turn, seen

    turn, seen = asyncio.run(_serve(tmp_path, scenario))
    assert turn.status_code == 200, turn.text
    assert seen["abort"][0] == "error", seen["abort"]
    assert seen["abort"][1] == 409 and seen["abort"][2]["code"] == "already_completed"
    assert seen["state_after_abort"] == "turn_in_flight"
    assert seen["state_after_turn"] == "idle"


def test_e2e_turn_size_and_budget_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(turn_api, "MAX_TURN_BODY_BYTES", 4096)
    monkeypatch.setattr(turn_api, "MAX_TURN_MESSAGES", 4)
    monkeypatch.setattr(turn_api, "MAX_TURN_TOOLS", 2)
    monkeypatch.setattr(turn_api, "MAX_TURN_DEBUG_BYTES", 64)
    tool = {"type": "function", "function": {"name": "ls", "parameters": {"type": "object"}}}

    async def scenario(client, scheduler, worker, app):
        await client.post("/v1/sessions", json={"traj_id": "traj-1", "session_id": "s1"})
        out = {}
        out["max_tokens"] = await client.post(
            "/v1/turns", json=_turn_body("s1", max_tokens=CONTEXT + 1))
        out["max_completion_tokens"] = await client.post(
            "/v1/turns", json=_turn_body("s1", max_completion_tokens=10**12))
        out["body"] = await client.post(
            "/v1/turns", json=_turn_body("s1", "x" * 5000))
        undeclared = json.dumps(_turn_body("s1", "y" * 5000)).encode()

        async def chunks():
            for i in range(0, len(undeclared), 1000):
                yield undeclared[i:i + 1000]

        out["body_undeclared"] = await client.post(
            "/v1/turns", content=chunks(), headers={"content-type": "application/json"})
        many = _turn_body("s1")
        many["messages"] = [{"role": "user", "content": "m"}] * 5
        out["messages"] = await client.post("/v1/turns", json=many)
        out["tools"] = await client.post("/v1/turns", json=_turn_body("s1", tools=[tool] * 3))
        out["debug"] = await client.post(
            "/v1/turns", json=_turn_body("s1", batchgen_debug={"note": "z" * 100}))
        out["state"] = app.state.session_registry.get("s1").state.value
        out["turns"] = app.state.session_registry.get("s1").turns
        out["at_limit"] = await client.post("/v1/turns", json=_turn_body(
            "s1", max_tokens=CONTEXT, tools=[tool] * 2, batchgen_debug={"k": 1}))
        return out, list(worker.admitted)

    out, admitted = asyncio.run(_serve(tmp_path, scenario))
    expected = {
        "max_tokens": (400, "max_tokens_exceeds_context"),
        "max_completion_tokens": (400, "max_tokens_exceeds_context"),
        "body": (413, "request_body_too_large"),
        "body_undeclared": (413, "request_body_too_large"),
        "messages": (400, "too_many_messages"),
        "tools": (400, "too_many_tools"),
        "debug": (413, "batchgen_debug_too_large"),
    }
    got = {}
    for name in expected:
        response = out[name]
        detail = response.json().get("detail")
        got[name] = (response.status_code,
                     detail.get("code") if isinstance(detail, dict) else detail)
    assert got == expected
    # Rejected before the session was touched; nothing reached a worker.
    assert (out["state"], out["turns"]) == ("open", 0)
    assert out["at_limit"].status_code == 200, out["at_limit"].text
    assert len(admitted) == 1


# --------------------------------------------------------------------------- #
#  --enable-session-kv-hold: held sessions keep their scheduling slot
# --------------------------------------------------------------------------- #


def test_a_held_turn_keeps_its_slot_until_the_next_turn_or_release():
    pool = _M["scheduling_pool"].SchedulingPool(capacity=3)
    pool.allocate_slot("t1")
    assert pool.hold_slot("t1", "s1") is None
    assert pool.num_active_slots() == 0 and pool.num_held_slots() == 1
    assert pool.num_free_slots() == 2
    # The next turn needs its own slot while the hold still exists (the worker
    # binds it at admission and frees the held one only when its wave commits).
    pool.allocate_slot("t2")
    assert pool.num_free_slots() == 1
    # t2 is held again: the previous held slot is returned.
    assert pool.hold_slot("t2", "s1") is not None
    assert pool.num_held_slots() == 1 and pool.num_free_slots() == 2
    assert pool.release_held_slot("s1") is not None
    assert pool.release_held_slot("s1") is None
    assert pool.num_free_slots() == 3
    with pytest.raises(RuntimeError, match="held"):
        pool.allocate_slot("t3")
        pool.hold_slot("t3", "s2")
        pool.set_capacity(5)


def test_fatal_clear_returns_held_slots_too():
    pool = _M["scheduling_pool"].SchedulingPool(capacity=2)
    pool.allocate_slot("t1")
    pool.hold_slot("t1", "s1")
    pool.allocate_slot("t2")
    assert pool.clear_active_slots() == 2
    assert pool.num_free_slots() == 2 and pool.num_held_slots() == 0


def _scheduler_with_pool(pool):
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)
    scheduler._scheduling_pool = pool
    return scheduler


def test_completion_routes_slots_by_hold_and_resume():
    pool = _M["scheduling_pool"].SchedulingPool(capacity=4)
    scheduler = _scheduler_with_pool(pool)
    pool.allocate_slot("t1")
    scheduler._account_completion_slot(
        "t1", {"session_id": "s1", "session_hold": {"held": True}},
    )
    assert pool.num_held_slots() == 1 and pool.num_free_slots() == 3
    # A resumed turn that is not held again (cancelled) frees both slots.
    pool.allocate_slot("t2")
    scheduler._account_completion_slot(
        "t2", {"session_id": "s1", "session_hold": None,
               "session_resume": {"lcp_tokens": 10}},
    )
    assert pool.num_held_slots() == 0 and pool.num_free_slots() == 4
    # A batch request (no session) just frees its slot.
    pool.allocate_slot("b1")
    scheduler._account_completion_slot("b1", {"session_id": None})
    assert pool.num_free_slots() == 4

def test_e2e_held_turn_keeps_its_slot_until_close_and_status_reports_admission(tmp_path):
    async def scenario(client, scheduler, worker, app):
        await client.post("/v1/sessions", json={"traj_id": "traj-1", "session_id": "s1"})
        answer = await client.post("/v1/turns", json=_turn_body("s1", "KVKEPT"))
        held = scheduler._scheduling_pool.num_held_slots()
        worker.response_queue.put({
            "type": "kv_admission_metrics", "deferred": {"host_footprint": 2},
            "overcommit_active": False})
        await _until(lambda: scheduler._kv_admission_info is not None)
        free = scheduler._scheduling_pool.num_free_slots()
        await client.post("/v1/sessions/s1/close", json={"reason": "end_of_session"})
        return (answer, held, free, scheduler._kv_admission_info,
                scheduler._scheduling_pool.num_held_slots(),
                scheduler._scheduling_pool.num_free_slots())

    answer, held, free, info, after, free_after = asyncio.run(
        _serve(tmp_path, scenario, capacity=4))
    assert answer.status_code == 200, answer.text
    # The worker still binds the held turn's reservation: the server agrees.
    assert held == 1 and free == 3
    assert after == 0 and free_after == 4
    assert info == {"deferred": {"host_footprint": 2}, "overcommit_active": False}


def test_h2_held_slots_keep_the_scheduler_within_the_worker_trajectory_pool():
    """A2 audit H2: the worker keeps a held turn's full-context reservation.

    The server must count that slot, or it admits a turn the worker's
    trajectory pool cannot bind (QueryBookCapacityError, a worker fatal).
    """
    from batchgen.query_book import QueryBook, QueryBookCapacityError

    context = 256
    book = QueryBook(capacity_bytes=2 * context * QueryBook.TOKEN_BYTES, page_tokens=64)
    pool = _M["scheduling_pool"].SchedulingPool(
        capacity=book.reservation_total_capacity(context))
    scheduler = bs.BatchScheduler.__new__(bs.BatchScheduler)
    scheduler._scheduling_pool = pool
    assert pool.capacity == 2
    # Turn 1 of s1 is admitted, bound, completes and is held by the worker.
    pool.allocate_slot("t1")
    book.bind("t1", context)
    scheduler._account_completion_slot(
        "t1", {"session_id": "s1", "session_hold": {"held": True}})
    # The server now offers exactly what the worker can still bind.
    assert pool.num_free_slots() == book.reservation_free_count(context) == 1
    pool.allocate_slot("b1")
    book.bind("b1", context)
    assert pool.num_free_slots() == 0
    # The pre-fix accounting (free the slot at completion) offered one more
    # admission, which the worker's bind refuses.
    with pytest.raises(QueryBookCapacityError):
        book.bind("b2", context)

