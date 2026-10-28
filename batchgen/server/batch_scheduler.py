"""Batch scheduling and execution loop for OpenAI-compatible batch API."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from batchgen.server.agent_sessions import TurnError
from batchgen.server.io_struct import (
    BatchEndpoint,
    BatchRequestItem,
    BatchStatus,
    ChatCompletionRequest,
    CompletionRequest,
    FileObject,
    FilePurpose,
    FileStatus,
    ToolCall,
    ToolCallFunction,
)
from batchgen.server.intake_pool import IntakeEntry, IntakePool, Priority
from batchgen.server.scheduling_pool import SchedulingPool
from batchgen.server.server_args import ServerArgs
from batchgen.server.storage import StorageManager
from batchgen.server.worker_manager import WorkerManager

logger = logging.getLogger(__name__)


def completion_prompt_to_text(prompt: str | List[str]) -> str:
    if isinstance(prompt, list):
        return "\n".join(prompt)
    return prompt


def tokenizer_family_mismatch(
    request_model: str, served_model: str,
) -> Optional[str]:
    """Return an error when a request names another tokenizer family.

    The scheduler renders the chat template with the tokenizer resolved from
    the request's ``model``; the worker encodes the rendered text with the
    served checkpoint's tokenizer. A request from another tokenizer family
    would be rendered with that family's template and control markers, then
    encoded as plain text by the served tokenizer. Family is the tokenizer
    type the registry resolves, so aliases of the served model still pass.
    A request name that resolves to no family is left to the tokenizer load.
    """
    from batchgen.config.tokenizer_registry import resolve_tokenizer_type

    served_family = resolve_tokenizer_type(served_model)
    if served_family is None:
        return (
            f"served model {served_model!r} resolves to no registered "
            f"tokenizer family"
        )
    request_family = resolve_tokenizer_type(request_model)
    if request_family is None or request_family == served_family:
        return None
    return (
        f"model {request_model!r} uses tokenizer family {request_family!r}, "
        f"but this server serves {served_model!r} with tokenizer family "
        f"{served_family!r}; send a model name of the {served_family!r} family"
    )


def parse_batch_file(
    content: bytes,
    served_model: Optional[str] = None,
) -> Tuple[bool, Optional[str], List[BatchRequestItem]]:
    """Validate and parse a JSONL batch file.

    With ``served_model`` set, a request whose model is a different tokenizer
    family than the served checkpoint is rejected.
    """
    try:
        lines = content.decode("utf-8").strip().split("\n")
    except UnicodeDecodeError:
        return False, "File must be UTF-8 encoded", []

    requests: List[BatchRequestItem] = []
    model_name: Optional[str] = None
    custom_id_lines: Dict[str, int] = {}

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            request = BatchRequestItem(**payload)
        except Exception as exc:
            return False, f"Line {idx}: {exc}", []

        # custom_id names the output line and, in pool mode, the in-flight
        # request; an empty one gets a unique index-based fallback.
        if request.custom_id:
            first = custom_id_lines.setdefault(request.custom_id, idx)
            if first != idx:
                return False, (
                    f"Line {idx}: duplicate custom_id {request.custom_id!r} "
                    f"(first on line {first})"
                ), []

        if isinstance(request.body, ChatCompletionRequest):
            current_model = request.body.model
        elif isinstance(request.body, CompletionRequest):
            current_model = request.body.model
        else:
            return False, f"Line {idx}: Unsupported request body", []

        if model_name is None:
            model_name = current_model
            if served_model is not None:
                mismatch = tokenizer_family_mismatch(current_model, served_model)
                if mismatch is not None:
                    return False, f"Line {idx}: {mismatch}", []
        elif model_name != current_model:
            return False, f"Line {idx}: Inconsistent model value", []

        requests.append(request)

    if not requests:
        return False, "Batch file cannot be empty", []
    return True, None, requests


# Worker per-request error codes that describe the request itself; any other
# worker error on a turn is a server-side failure.
_TURN_REQUEST_ERROR_CODES = ("context_length_exceeded", "host_kv_capacity_exceeded")


def turn_timing_summary(timing: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Wall-clock stamps of one turn plus the per-leg durations in ms.

    Legs: HTTP receipt -> IntakePool submit -> admit message sent -> worker
    admit, worker admit -> worker completion (service), and worker completion
    -> completion routed -> HTTP response. ``overhead_ms`` is everything but
    the worker's service time. A leg whose stamp is missing is None.
    """
    def ms(start: str, end: str) -> Optional[float]:
        a, b = timing.get(start), timing.get(end)
        if a is None or b is None:
            return None
        return round((b - a) * 1000.0, 3)

    summary: Dict[str, Any] = dict(timing)
    summary["receipt_to_submit_ms"] = ms("http_received_at", "intake_submitted_at")
    summary["submit_to_admit_sent_ms"] = ms("intake_submitted_at", "admit_sent_at")
    summary["admit_sent_to_worker_admit_ms"] = ms("admit_sent_at", "worker_admitted_at")
    summary["submit_to_worker_admit_ms"] = ms("intake_submitted_at", "worker_admitted_at")
    summary["worker_admit_to_complete_ms"] = ms("worker_admitted_at", "worker_completed_at")
    summary["worker_complete_to_routed_ms"] = ms("worker_completed_at", "completion_routed_at")
    summary["routed_to_response_ms"] = ms("completion_routed_at", "http_responded_at")
    summary["worker_complete_to_response_ms"] = ms("worker_completed_at", "http_responded_at")
    before = ms("http_received_at", "worker_admitted_at")
    after = summary["worker_complete_to_response_ms"]
    summary["overhead_ms"] = (
        None if before is None or after is None else round(before + after, 3)
    )
    return summary


class _TurnWaiter:
    """One submitted turn waiting for its completion (scheduler-loop only)."""

    __slots__ = ("turn_id", "model", "session", "future", "timing")

    def __init__(self, turn_id, model, session, future, received_at) -> None:
        self.turn_id = turn_id
        self.model = model
        self.session = session
        self.future = future
        self.timing: Dict[str, Optional[float]] = {"http_received_at": received_at}


class BatchScheduler:
    """Async scheduler that executes batches sequentially."""

    def __init__(
        self,
        storage: StorageManager,
        worker: WorkerManager,
        server_args: ServerArgs,
    ):
        self.storage = storage
        self.worker = worker
        self.server_args = server_args
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        self._tokenizer = None
        self._tokenizer_model: Optional[str] = None
        # The worker publishes the fixed token-pool-derived capacity after the
        # model context is known. Keep no independent row cap here.
        self._pool_mode = True
        self._max_intake_capacity = getattr(server_args, 'max_intake_capacity', 1_000_000)
        self._batch_timeout = 86400  # 24h default, matches completion_window
        self._intake_pool = IntakePool(max_capacity=self._max_intake_capacity)
        self._scheduling_pool = SchedulingPool(capacity=0)
        self._trajectory_pool_info: Optional[Dict[str, Any]] = None
        self._pool_initialized = False  # First batch triggers worker init
        self._completion_listener_task: Optional[asyncio.Task] = None
        self._drain_task: Optional[asyncio.Task] = None
        # Completion delivery is generation-scoped: one daemon pump thread per
        # worker generation owns the blocking read of that generation's
        # response queue and forwards items into this asyncio queue, which the
        # listener task consumes.  See _completion_pump.
        self._completion_inbox: Optional[asyncio.Queue] = None
        self._pump_lock = threading.Lock()
        self._pump_generation = 0
        self._pump_thread: Optional[threading.Thread] = None
        self._pump_stop: Optional[threading.Event] = None
        # Per-request metadata for building output JSONL in pool mode
        # Structure: {batch_id: {request_id: {custom_id, url, model, prompt_text}}}
        # where request_id is f"{custom_id}@{batch_id}"
        self._pool_request_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Batches whose leftover intake entries the drain discarded; each is
        # logged once.
        self._stale_intake_batches: set = set()
        self._quiesced = threading.Event()
        self._quiesce_ack = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Per-turn path (POST /v1/turns). Each turn is a one-entry intake batch
        # keyed by its turn id; the completion listener resolves its waiter's
        # future directly, with no batch tracker and no output file.
        self._turn_waiters: Dict[str, _TurnWaiter] = {}
        # Aborted turns already sent to a worker. The worker still finishes
        # them (there is no worker-side cancel); their completion is dropped.
        self._detached_turns: set = set()
        # A turn submission wakes an idle drain at once instead of at its next
        # 0.1 s poll.
        self._intake_wakeup = asyncio.Event()

    async def start(self) -> None:
        if self._task:
            return
        self._stopped.clear()
        self._loop = asyncio.get_running_loop()
        self._completion_inbox = asyncio.Queue()
        self._task = asyncio.create_task(self._run())
        # Start pool mode background tasks.  The startup path uses the same
        # mechanism as a restart: the first worker generation gets pump #1.
        self._ensure_completion_listener()
        self._start_completion_pump()
        self._drain_task = asyncio.create_task(
            self._drain_intake_to_worker()
        )

    async def stop(self) -> None:
        if not self._task:
            return
        self._stopped.set()
        # Ask the live pump to leave its read loop.  It is never joined: a
        # pump blocked on a truncated frame can never return, and it is a
        # daemon thread precisely so that cannot hold up process exit.
        with self._pump_lock:
            if self._pump_stop is not None:
                self._pump_stop.set()
            self._pump_stop = None
            self._pump_thread = None
        # Cancel all tasks (background pool tasks + main task)
        for task in [self._completion_listener_task, self._drain_task, self._task]:
            if task:
                task.cancel()
        for task in [self._completion_listener_task, self._drain_task, self._task]:
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._completion_listener_task = None
        self._drain_task = None

    def intake_pool_usage_pct(self) -> float:
        """Return intake pool usage as a fraction (0.0-1.0)."""
        cap = self._intake_pool.max_capacity
        return self._intake_pool.size() / cap if cap > 0 else 0.0

    def intake_pool_size(self) -> int:
        return self._intake_pool.size()

    def intake_pool_capacity(self) -> int:
        return self._intake_pool.max_capacity

    async def enqueue(self, batch_id: str) -> None:
        await self._queue.put(batch_id)

    def quiesce(self) -> None:
        """Stop dispatching new work while retaining the upstream queues."""
        self._quiesced.set()
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop)
            self._quiesce_ack.wait(timeout=5)

    def resume(self) -> None:
        self._quiesced.clear()
        self._quiesce_ack.clear()

    def fail_in_flight(self, error_msg: str) -> int:
        return self._fail_all_active_batches(error_msg)

    def worker_generation_ready(self) -> None:
        # WorkerManager sends the generation's init message before this hook;
        # the drain task must not send a duplicate init after resume.
        self._pool_initialized = True
        # The new cohort writes to a new response queue, and the previous
        # generation's pump owns a queue that was abandoned with its cohort.
        # Completion delivery therefore has to be rebuilt here or no batch of
        # this generation can ever complete.  This runs on the restart's
        # thread, not the scheduler loop, so the task side is scheduled
        # through the loop.
        self._start_completion_pump()
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            # Already on the scheduler loop. Waiting for a scheduled callback
            # here would deadlock, because this thread IS the one that would
            # have to run it.
            self._ensure_completion_listener()
            return
        ensured = threading.Event()

        def ensure_on_loop() -> None:
            try:
                self._ensure_completion_listener()
            finally:
                ensured.set()

        loop.call_soon_threadsafe(ensure_on_loop)
        if not ensured.wait(timeout=5.0):
            raise RuntimeError(
                "completion listener could not be re-armed for the new worker "
                "generation: the scheduler event loop did not run the "
                "scheduling callback"
            )

    def _ensure_completion_listener(self) -> None:
        """Start the completion listener task unless one is already running.

        Runs on the scheduler loop.  The listener serves every worker
        generation and ends only when the scheduler stops, so this is crash
        recovery: it recreates a listener task that died, and is otherwise a
        no-op.
        """
        task = self._completion_listener_task
        if task is not None and not task.done():
            return
        self._completion_listener_task = asyncio.create_task(
            self._pool_completion_listener()
        )

    def _start_completion_pump(self) -> None:
        """Supersede the previous generation's pump and start this one's.

        The pump captures the response queue object at start, so it can never
        reach into a later generation's channel, and it is a daemon thread so
        a read that can never return cannot block process exit.
        """
        with self._pump_lock:
            previous = self._pump_stop
            if previous is not None:
                previous.set()
            self._pump_generation += 1
            generation = self._pump_generation
            stop_event = threading.Event()
            self._pump_stop = stop_event
            thread = threading.Thread(
                target=self._completion_pump,
                args=(generation, self.worker.response_queue, stop_event),
                name=f"pool-completion-pump-{generation}",
                daemon=True,
            )
            self._pump_thread = thread
            thread.start()

    def stop_completion_pump(self) -> None:
        """Retire the live pump ahead of its channel being abandoned.

        Called (on the restart thread) before WorkerManager closes the old
        cohort's queues, so the pump treats the resulting read failure as
        retirement instead of a channel error.  The next generation's pump
        is started separately by worker_generation_ready().
        """
        with self._pump_lock:
            if self._pump_stop is not None:
                self._pump_stop.set()
            # Retire the generation itself, not only its reader: anything the
            # retired cohort's pump still forwards, or that already waits in
            # the inbox, now fails the generation check and is dropped instead
            # of acting during the restart.  _start_completion_pump advances
            # the generation again for the next cohort.
            self._pump_generation += 1

    def _completion_pump(
        self,
        generation: int,
        response_queue: Any,
        stop_event: threading.Event,
    ) -> None:
        """Forward one generation's worker responses to the scheduler loop.

        Owns the only blocking read of ``response_queue``.  It exits when the
        scheduler stops, when it is superseded by a later generation, or once
        it has forwarded a terminal message.  A pump whose read never returns
        leaks one idle daemon thread and nothing else: it holds no lock, it
        cannot touch a later generation's queue, and whatever it forwards
        after supersession is dropped.
        """
        import queue as queue_mod

        logger.info("[POOL] Completion pump gen=%d started", generation)
        try:
            while not stop_event.is_set() and not self._stopped.is_set():
                try:
                    item = response_queue.get(timeout=1.0)
                except queue_mod.Empty:
                    continue
                except Exception as exc:
                    if stop_event.is_set() or self._stopped.is_set():
                        # The restart retired this channel before abandoning
                        # it; the failed read is the expected wake-up, not a
                        # live generation losing its channel.
                        logger.info(
                            "[POOL] Completion pump gen=%d exiting: "
                            "channel retired (%s)",
                            generation,
                            exc,
                        )
                        break
                    # The channel of a killed cohort is abandoned rather than
                    # drained, so a read on it can fail once it is closed.
                    # Report it as this generation's terminal item.
                    logger.error(
                        "[POOL] Completion pump gen=%d read failed: %s",
                        generation,
                        exc,
                        exc_info=True,
                    )
                    self._forward_completion(
                        generation,
                        {"type": "channel_error", "error": str(exc)},
                    )
                    break
                terminal = item is None or (
                    isinstance(item, dict)
                    and (item.get("type") == "pool_shutdown" or "error" in item)
                )
                self._forward_completion(generation, item)
                if terminal:
                    break
        finally:
            logger.info("[POOL] Completion pump gen=%d stopped", generation)

    def _forward_completion(self, generation: int, item: Any) -> None:
        """Hand one item to the scheduler loop. Called on the pump thread."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._offer_completion, generation, item)
        except RuntimeError:
            # The loop is closed; the scheduler is gone and so is the batch
            # state this item would have updated.
            logger.warning(
                "[POOL] Completion pump gen=%d dropped an item: "
                "the scheduler loop is closed",
                generation,
            )

    def _offer_completion(self, generation: int, item: Any) -> None:
        """Accept an item only from the live pump. Runs on the scheduler loop.

        The item is queued with its generation: an item accepted here can
        still be waiting in the inbox when a restart advances the generation,
        so the listener checks the tag again when it takes the item.
        """
        if generation != self._pump_generation:
            logger.warning(
                "[POOL] Dropping an item from superseded completion pump "
                "gen=%d (live generation is %d)",
                generation,
                self._pump_generation,
            )
            return
        inbox = self._completion_inbox
        if inbox is None:
            return
        inbox.put_nowait((generation, item))

    def pool_init_params(self) -> Dict[str, Any]:
        """Pool capacity for a worker generation's init message."""
        return {
            "max_output_len": getattr(self, "_pool_max_output_len", 4096),
            "max_context_length": getattr(self, "_pool_max_context_length", None),
        }

    async def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                batch_id = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self._process_batch(batch_id)
            except Exception:
                logger.exception("Batch %s failed", batch_id)
            finally:
                self._queue.task_done()

    async def _process_batch(self, batch_id: str) -> None:
        batch = self.storage.load_batch(batch_id)
        if not batch:
            logger.warning("Batch %s not found", batch_id)
            return
        if batch.status in {BatchStatus.CANCELLED, BatchStatus.CANCELLING}:
            logger.info(
                "Batch %s already cancelled, skipping execution", batch_id
            )
            return

        input_meta = self.storage.load_metadata(batch.input_file_id)
        if not input_meta:
            logger.error(
                "Input file %s not found for batch %s",
                batch.input_file_id,
                batch_id,
            )
            self.storage.update_batch_status(
                batch_id, BatchStatus.FAILED, error="Input file not found"
            )
            return

        input_path = self.storage.files_dir / batch.input_file_id
        if not input_path.exists():
            logger.error("Input file content missing at %s", input_path)
            self.storage.update_batch_status(
                batch_id, BatchStatus.FAILED, error="Input file content missing"
            )
            return

        started_at = int(time.time())
        self.storage.update_batch_status(
            batch_id, BatchStatus.IN_PROGRESS, started_at=started_at
        )

        with input_path.open("rb") as handle:
            content = handle.read()

        ok, error_message, requests = parse_batch_file(
            content, served_model=self.server_args.model
        )
        if not ok:
            self.storage.update_batch_status(
                batch_id, BatchStatus.FAILED, error=error_message
            )
            return

        try:
            prompts, per_request_max_tokens, sampling_params = self._convert_requests_to_worker_inputs(
                requests, batch
            )
        except Exception as exc:
            # Prompt construction can reject a request: an unknown chat role, a
            # conversation the tokenizer cannot render faithfully, a malformed
            # tool call. Without this the exception propagates to _run's bare
            # `except Exception: logger.exception(...)`, which never sets a
            # terminal status -- and the batch was already marked IN_PROGRESS
            # above, so every request in it is lost and the client polls
            # forever.
            #
            # NOTE this still fails the whole batch on one bad request.
            # Per-request rejection at admission is the proper fix and is left
            # as a follow-up; this only makes the failure visible and terminal.
            logger.exception("Batch %s: prompt construction failed", batch_id)
            self.storage.update_batch_status(
                batch_id,
                BatchStatus.FAILED,
                error=f"Prompt construction failed: {type(exc).__name__}: {exc}",
            )
            return
        # Apply batch-level max_decoding_length as fallback for requests without explicit value
        default_max = batch.max_decoding_length
        if default_max is None:
            missing_ids = [
                requests[i].get("custom_id", f"request-{i}")
                for i, mt in enumerate(per_request_max_tokens)
                if mt is None
            ]
            if missing_ids:
                error_message = (
                    f"Batch {batch_id}: {len(missing_ids)}/{len(per_request_max_tokens)} requests "
                    f"have no max_completion_tokens or max_tokens, and no batch-level "
                    f"max_decoding_length is set. Set one of these to proceed. "
                    f"First missing: {missing_ids[:5]}"
                )
                logger.error(error_message)
                self.storage.update_batch_status(
                    batch_id, BatchStatus.FAILED, error=error_message
                )
                return
        per_request_max_tokens = [
            mt if mt is not None else default_max
            for mt in per_request_max_tokens
        ]

        # Log batch-level sampling param defaults
        if batch.temperature is not None or batch.top_p is not None or batch.top_k is not None:
            logger.warning(
                f"Batch {batch_id}: batch-level sampling params "
                f"(temperature={batch.temperature}, top_p={batch.top_p}, top_k={batch.top_k}) "
                f"serve as defaults only; per-request values take priority"
            )

        await self._process_batch_pool_mode(
            batch_id, batch, requests, prompts,
            per_request_max_tokens, sampling_params,
        )

    def _convert_requests_to_worker_inputs(
        self, requests: List[BatchRequestItem], batch=None,
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        """Convert batch requests to worker inputs with per-request sampling params.

        Returns:
            (prompts, per_request_max_tokens, sampling_params) where
            per_request_max_tokens is a per-sequence list of max output token limits,
            and sampling_params is a list of dicts with keys: temperature, top_p, top_k.
            Per-request values take priority; batch-level values serve as defaults.
        """
        prompts: List[str] = []
        per_request_max_tokens: List[int] = []
        sampling_params: List[Dict[str, Any]] = []

        # Batch-level defaults (fallback when per-request is None)
        batch_temp = batch.temperature if batch else None
        batch_top_p = batch.top_p if batch else None
        batch_top_k = batch.top_k if batch else None

        for request in requests:
            body = request.body
            if isinstance(body, ChatCompletionRequest):
                model_lower = body.model.lower()
                if "glm-5.3" in model_lower:
                    if body.enable_thinking is False or body.thinking is False:
                        raise ValueError(
                            "GLM-5.3 does not support enable_thinking=false; "
                            "use reasoning_effort=low/high/max with thinking enabled."
                        )
                    if body.reasoning_effort == "medium":
                        raise ValueError(
                            "GLM-5.3 reasoning_effort must be low, high, or max; "
                            "medium is unsupported."
                        )
                # Inject reasoning_effort into system message for GPT-OSS models
                messages = self._inject_reasoning_effort(
                    [m.dict(exclude_none=True) for m in body.messages],
                    body.model,
                    body.reasoning_effort,
                )
                # Forward extra kwargs (thinking, tools) to chat template.
                # The GLM-5 / SGLang convention is `enable_thinking`; older
                # callers may send `thinking`. Prefer `enable_thinking` when
                # both are set, and forward BOTH names so templates written
                # against either convention work.
                template_kwargs = {}
                thinking_val = (
                    body.enable_thinking
                    if body.enable_thinking is not None
                    else body.thinking
                )
                if thinking_val is not None:
                    template_kwargs["enable_thinking"] = thinking_val
                    template_kwargs["thinking"] = thinking_val
                if body.tools is not None:
                    template_kwargs["tools"] = body.tools
                if body.reasoning_effort is not None and "glm-5.3" in model_lower:
                    template_kwargs["reasoning_effort"] = body.reasoning_effort
                if body.clear_thinking is not None:
                    template_kwargs["clear_thinking"] = body.clear_thinking
                if body.preserve_thinking is not None:
                    template_kwargs["preserve_thinking"] = body.preserve_thinking
                try:
                    prompt = self._format_chat_messages(
                        messages, body.model, **template_kwargs
                    )
                except Exception as exc:
                    # Attach the custom_id. The caller fails the batch; without
                    # the id there is no way to tell which of N requests did it.
                    custom_id = request.custom_id or "<no custom_id>"
                    raise ValueError(
                        f"request {custom_id!r}: the chat template rejected "
                        f"this conversation: {type(exc).__name__}: {exc}"
                    ) from exc
                # Priority: max_completion_tokens > max_tokens > None
                current_max_tokens = body.max_completion_tokens if body.max_completion_tokens is not None else body.max_tokens
            elif isinstance(body, CompletionRequest):
                prompt = completion_prompt_to_text(body.prompt)
                current_max_tokens = body.max_completion_tokens if body.max_completion_tokens is not None else body.max_tokens
            else:
                raise ValueError("Unsupported request body type")

            prompts.append(prompt)
            per_request_max_tokens.append(current_max_tokens)

            # Extract per-request sampling params with batch-level fallback
            req_temp = getattr(body, 'temperature', None)
            req_top_p = getattr(body, 'top_p', None)
            req_top_k = getattr(body, 'top_k', None)

            # Apply fallback: per-request → batch-level → None
            effective_temp = req_temp if req_temp is not None else batch_temp
            effective_top_p = req_top_p if req_top_p is not None else batch_top_p
            effective_top_k = req_top_k if req_top_k is not None else batch_top_k

            sampling_params.append({
                'temperature': effective_temp,
                'top_p': effective_top_p,
                'top_k': effective_top_k,
            })

        return prompts, per_request_max_tokens, sampling_params

    def _inject_reasoning_effort(
        self,
        messages: List[dict],
        model: str,
        reasoning_effort: Optional[str],
    ) -> List[dict]:
        """Inject reasoning_effort into system message for GPT-OSS models.

        GPT-OSS models use the Harmony response format where reasoning effort
        is specified in the system message as "Reasoning: {low|medium|high}".
        This follows the OpenAI reference implementation.
        """
        # Only apply to GPT-OSS models
        if "gpt-oss" not in model.lower():
            return messages
        # If no reasoning_effort specified, use default (low per OpenAI)
        if reasoning_effort is None:
            reasoning_effort = "low"

        # Find system message and prepend reasoning effort
        modified = []
        system_found = False
        for msg in messages:
            if msg.get("role") == "system" and not system_found:
                # Prepend reasoning effort to system content
                original_content = msg.get("content", "")
                new_content = f"Reasoning: {reasoning_effort}\n{original_content}"
                modified.append({**msg, "content": new_content})
                system_found = True
            else:
                modified.append(msg)

        # If no system message exists, insert one at the beginning
        if not system_found:
            modified.insert(0, {
                "role": "system",
                "content": f"Reasoning: {reasoning_effort}",
            })

        return modified

    def _format_chat_messages(self, messages: List[dict], model: str, **kwargs) -> str:
        tokenizer = self._get_tokenizer(model)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                f"Tokenizer for {model} does not support apply_chat_template"
            )
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def _parse_output(
        self,
        model: str,
        decoded_text: str,
    ) -> tuple[str, Optional[str], Optional[List[ToolCall]]]:
        """Apply thinking/tool-call parsing if flags are enabled.

        Returns:
            (content, reasoning_content, tool_calls)
        """
        content = decoded_text
        reasoning_content = None
        tool_calls = None

        tokenizer = self._get_tokenizer(model)
        if tokenizer is None:
            return content, reasoning_content, tool_calls

        if self.server_args.parse_thinking:
            try:
                reasoning_content, content = tokenizer.parse_thinking(content)
            except NotImplementedError:
                pass

        if self.server_args.parse_tool_call:
            try:
                raw_calls, content = tokenizer.parse_tool_calls(content)
                if raw_calls:
                    tool_calls = [
                        ToolCall(
                            id=c["id"],
                            type=c["type"],
                            function=ToolCallFunction(
                                name=c["function"]["name"],
                                arguments=c["function"]["arguments"],
                            ),
                        )
                        for c in raw_calls
                    ]
            except NotImplementedError:
                pass

        return content, reasoning_content, tool_calls

    def _get_tokenizer(self, model: str) -> Optional[Any]:
        """Load tokenizer for the given model.

        Uses BatchGen's tokenizer abstraction which removes the dependency
        on transformers.AutoTokenizer for supported models.

        The model name is used for pattern matching to select the appropriate
        tokenizer. Tokenizer files are loaded from the BatchGen package directory.
        """
        if self._tokenizer_model == model and self._tokenizer is not None:
            return self._tokenizer

        from batchgen.config.tokenizer_registry import load_tokenizer

        try:
            # Model name used for pattern matching; tokenizer loads from package dir
            tokenizer = load_tokenizer(model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer for {model}: {exc}"
            ) from exc

        self._tokenizer = tokenizer
        self._tokenizer_model = model
        return tokenizer

    # ============ Per-turn path (POST /v1/turns) ============

    def submit_turn(self, turn_id: str, body: Any, received_at: float) -> _TurnWaiter:
        """Render one turn as the batch path does and submit it to the IntakePool.

        ``body`` is a validated TurnRequest. The turn becomes a one-entry intake
        batch keyed by ``turn_id``; the returned waiter's future resolves with
        the worker's completion item. Raises TurnError for a request that
        cannot run (400) or a full intake pool (429). Runs on the scheduler
        loop.
        """
        if body.n not in (None, 1):
            raise TurnError(
                400, "unsupported_n",
                f"a turn returns exactly one choice; n={body.n} is not supported",
            )
        mismatch = tokenizer_family_mismatch(body.model, self.server_args.model)
        if mismatch is not None:
            raise TurnError(400, "model_mismatch", mismatch)
        item = BatchRequestItem.model_construct(
            custom_id=turn_id,
            method="POST",
            url=BatchEndpoint.CHAT_COMPLETIONS,
            body=body,
        )
        try:
            prompts, max_tokens, sampling_params = (
                self._convert_requests_to_worker_inputs([item])
            )
        except Exception as exc:
            raise TurnError(
                400, "invalid_request",
                f"prompt construction failed: {type(exc).__name__}: {exc}",
            ) from exc
        if max_tokens[0] is None:
            raise TurnError(
                400, "max_tokens_required",
                "a turn needs max_completion_tokens or max_tokens",
            )
        session = body.session
        entry = IntakeEntry(
            request_id=turn_id,
            batch_id=turn_id,
            raw_request={
                "text": prompts[0],
                "max_tokens": max_tokens[0],
                "priority": 0,
                "sampling_params": sampling_params[0],
                "ignore_eos": bool(body.ignore_eos),
                "batchgen_debug": body.batchgen_debug or {},
                "traj_id": session.traj_id,
                "session_id": session.session_id,
                "parent_session_id": session.parent_session_id,
            },
            priority=Priority.NORMAL,
        )
        waiter = _TurnWaiter(
            turn_id,
            body.model,
            session,
            asyncio.get_running_loop().create_future(),
            received_at,
        )
        # Registered before the submit: the drain discards an intake entry that
        # belongs to neither a live batch nor a live turn.
        self._turn_waiters[turn_id] = waiter
        if not self._intake_pool.submit_batch(turn_id, [entry], Priority.NORMAL):
            del self._turn_waiters[turn_id]
            raise TurnError(
                429, "capacity_exceeded",
                f"Server at capacity ({self._intake_pool.size()}/"
                f"{self._intake_pool.max_capacity} requests in queue). Retry later.",
            )
        waiter.timing["intake_submitted_at"] = time.time()
        self._pool_max_output_len = max(
            getattr(self, "_pool_max_output_len", 0), max_tokens[0]
        )
        self._intake_wakeup.set()
        return waiter

    def cancel_turn(self, turn_id: str, reason: str) -> Optional[str]:
        """Cancel a submitted turn; its waiter fails with 409 turn_aborted.

        Returns "dequeued" when the turn was still in the IntakePool (it never
        reaches a worker), "detached" when it was already admitted, or None
        for a turn that is not pending. A detached turn runs to completion on
        the worker -- the worker has no per-request cancel, the same as
        /v1/batches/{id}/cancel -- and its completion is discarded. Runs on the
        scheduler loop.
        """
        waiter = self._turn_waiters.pop(turn_id, None)
        if waiter is None:
            return None
        removed = self._intake_pool.remove_batch_entries(turn_id)
        self._intake_pool.remove_batch_info(turn_id)
        outcome = "dequeued" if removed else "detached"
        if not removed:
            self._detached_turns.add(turn_id)
        if not waiter.future.done():
            waiter.future.set_exception(TurnError(
                409, "turn_aborted", f"turn {turn_id} was aborted: {reason}",
            ))
        logger.info("[TURN] %s cancelled (%s): %s", turn_id, outcome, reason)
        return outcome

    def _resolve_turn(self, turn_id: str, result: dict) -> None:
        """Hand a turn's completion item to its waiter. Runs on the loop."""
        routed_at = time.time()
        self._intake_pool.remove_batch_info(turn_id)
        if turn_id in self._detached_turns:
            self._detached_turns.discard(turn_id)
            logger.info("[TURN] %s: discarding the completion of an aborted turn", turn_id)
            return
        waiter = self._turn_waiters.pop(turn_id)
        waiter.timing["worker_admitted_at"] = result.get("admitted_at")
        waiter.timing["worker_completed_at"] = result.get("completed_at")
        waiter.timing["completion_routed_at"] = routed_at
        if not waiter.future.done():
            waiter.future.set_result(result)

    def _fail_pending_turns(self, error_msg: str) -> None:
        """Fail every pending turn after a fatal worker error.

        Callable from any thread (a restart's fail_in_flight runs off the
        loop): queued entries are purged at once, futures are failed on the
        loop.
        """
        turn_ids = list(self._turn_waiters)
        for turn_id in turn_ids:
            self._intake_pool.remove_batch_entries(turn_id)

        def fail_on_loop() -> None:
            for turn_id in turn_ids:
                waiter = self._turn_waiters.pop(turn_id, None)
                if waiter is None:
                    continue
                self._intake_pool.remove_batch_info(turn_id)
                if not waiter.future.done():
                    waiter.future.set_exception(TurnError(
                        503, "worker_fatal", f"worker_fatal: {error_msg}",
                    ))
            # A detached turn's worker generation is gone with its completion.
            self._detached_turns.clear()

        if turn_ids:
            logger.error(
                "[TURN] Failing %d pending turn(s): %s", len(turn_ids), error_msg
            )
        loop = self._loop
        try:
            on_loop = loop is not None and asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop or loop is None or not loop.is_running():
            fail_on_loop()
        else:
            loop.call_soon_threadsafe(fail_on_loop)

    def build_turn_response(self, waiter: _TurnWaiter, result: dict) -> dict:
        """The chat.completion body for a resolved turn, with BatchGen metadata.

        A worker per-request error raises TurnError: 400 when it describes the
        request (e.g. context_length_exceeded), 500 otherwise.
        """
        error = result.get("error")
        if error:
            code = error.get("code", "worker_error")
            status = 400 if code in _TURN_REQUEST_ERROR_CODES else 500
            logger.warning(
                "[TURN] %s failed on the worker: session=%s code=%s message=%s",
                waiter.turn_id, waiter.session.session_id, code,
                error.get("message", ""),
            )
            raise TurnError(status, code, error.get("message", ""))
        body = self._chat_completion_body(waiter.model, result, int(time.time()))
        cached = result.get("cached_tokens")
        body["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
        waiter.timing["http_responded_at"] = time.time()
        timing = turn_timing_summary(waiter.timing)
        session = waiter.session
        body["batchgen"] = {
            "turn_id": waiter.turn_id,
            "session": {
                "traj_id": session.traj_id,
                "session_id": session.session_id,
                "parent_session_id": session.parent_session_id,
            },
            "cached_tokens": cached,
            "computed_tokens": result.get("computed_tokens"),
            "timing": timing,
        }
        logger.info(
            "[TURN] %s done traj=%s session=%s parent=%s prompt=%s cached=%s "
            "computed=%s decoded=%s finish=%s receipt_to_submit_ms=%s "
            "submit_to_worker_admit_ms=%s worker_ms=%s "
            "worker_complete_to_response_ms=%s overhead_ms=%s",
            waiter.turn_id, session.traj_id, session.session_id,
            session.parent_session_id, result.get("prompt_length"), cached,
            result.get("computed_tokens"), result.get("decoded_length"),
            result.get("finish_reason"), timing["receipt_to_submit_ms"],
            timing["submit_to_worker_admit_ms"],
            timing["worker_admit_to_complete_ms"],
            timing["worker_complete_to_response_ms"], timing["overhead_ms"],
        )
        return body

    # ============ Pool Mode ============

    async def _process_batch_pool_mode(
        self,
        batch_id: str,
        batch: Any,
        requests: List[BatchRequestItem],
        prompts: List[str],
        per_request_max_tokens: List[int],
        sampling_params: List[Dict[str, Any]],
    ) -> None:
        """Process a batch in pool mode: push to IntakePool for async processing.

        All batches (including the first) go through IntakePool → drain task → worker.
        The drain task sends an "init" message on first drain, then admission messages.
        """
        max_tokens = max(per_request_max_tokens)

        # Register batch for completion tracking
        self._scheduling_pool.register_batch(
            batch_id=batch_id,
            total_requests=len(requests),
            output_path=str(
                self.storage.output_dir / f"{batch_id}_output.jsonl"
            ) if hasattr(self.storage, 'output_dir') else None,
        )

        # custom_id is unique only within one Batch, but the scheduling pool,
        # the worker's sequence uuids and completion routing span all batches:
        # scope the internal request id by batch (custom_id first keeps the
        # worker's truncated uuid logs readable).
        custom_ids = [
            req.custom_id or f"{batch_id}_req_{idx}"
            for idx, req in enumerate(requests)
        ]
        request_ids = [f"{custom_id}@{batch_id}" for custom_id in custom_ids]

        # Build IntakeEntry objects and push to IntakePool
        entries = []
        for idx, req in enumerate(requests):
            entries.append(IntakeEntry(
                request_id=request_ids[idx],
                batch_id=batch_id,
                raw_request={
                    "text": prompts[idx],
                    "max_tokens": per_request_max_tokens[idx],
                    "priority": 0,  # TODO: support per-batch priority from API
                    "sampling_params": sampling_params[idx] if sampling_params else {},
                    # Per-request vendor extension, as in vLLM/SGLang.
                    "ignore_eos": bool(getattr(req.body, "ignore_eos", False)),
                    "batchgen_debug": batch.batchgen_debug or {},
                },
                priority=Priority.NORMAL,
            ))
        accepted = self._intake_pool.submit_batch(batch_id, entries, Priority.NORMAL)
        if not accepted:
            current = self._intake_pool.size()
            cap = self._intake_pool.max_capacity
            error_msg = (
                f"Server at capacity: intake pool has {current}/{cap} requests. "
                f"Batch with {len(entries)} requests rejected. Retry later."
            )
            logger.warning(f"[POOL] Batch {batch_id} rejected: {error_msg}")
            self.storage.update_batch_status(
                batch_id, BatchStatus.FAILED, error=f"capacity_exceeded: {error_msg}"
            )
            return
        # Store max_tokens for init message
        if not hasattr(self, '_pool_max_output_len'):
            self._pool_max_output_len = max_tokens
        else:
            self._pool_max_output_len = max(self._pool_max_output_len, max_tokens)
        if not hasattr(self, '_pool_max_context_length'):
            self._pool_max_context_length = batch.max_context_length

        # Store per-request metadata for output JSONL building
        self._pool_request_meta[batch_id] = {}
        for idx, req in enumerate(requests):
            self._pool_request_meta[batch_id][request_ids[idx]] = {
                "custom_id": custom_ids[idx],
                "url": req.url.value,
                "model": req.body.model,
                "prompt_text": prompts[idx],
            }

        # Ensure incremental output directory exists
        incr_dir = self.server_args.incremental_output_dir
        if incr_dir:
            from pathlib import Path
            Path(incr_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[POOL] Batch {batch_id}: {len(entries)} requests pushed to IntakePool "
            f"(total in pool: {self._intake_pool.size()})"
        )

        # Launch background task to wait for completion and finalize output.
        # Return immediately so the scheduler can process the next batch.
        asyncio.ensure_future(
            self._wait_and_finalize_batch(batch_id, requests, prompts)
        )

    async def _wait_and_finalize_batch(
        self,
        batch_id: str,
        requests: List[BatchRequestItem],
        prompts: List[str],
    ) -> None:
        """Background task: wait for a batch to complete, then finalize output."""
        import time as _time
        deadline = _time.time() + self._batch_timeout
        batch_failed = False

        while True:
            tracker = self._scheduling_pool.get_batch_tracker(batch_id)
            if tracker and tracker.is_complete:
                break
            if tracker and getattr(tracker, 'error', None):
                logger.error(f"[POOL] Batch {batch_id} failed: {tracker.error}")
                batch_failed = True
                break
            if _time.time() > deadline:
                logger.error(f"[POOL] Batch {batch_id} timed out after {self._batch_timeout}s")
                batch_failed = True
                break
            await asyncio.sleep(0.5)

        if batch_failed:
            error_msg = getattr(tracker, 'error', 'timeout') if tracker else 'timeout'
            self.storage.update_batch_status(
                batch_id,
                BatchStatus.FAILED,
                error=str(error_msg),
            )
            return

        self._finalize_batch_output(batch_id, requests, prompts)

    async def _drain_intake_to_worker(self) -> None:
        """Background task: run the intake drain; any failure is fatal.

        Without a drain every queued and later request stays in_progress
        forever, and entries already popped from the IntakePool or holding a
        slot cannot be recovered: fail the active batches, then stop the server.
        """
        try:
            await self._drain_intake_loop()
        except Exception as exc:
            reason = f"Server intake drain failed: {type(exc).__name__}: {exc}"
            logger.exception("[POOL] %s", reason)
            try:
                self._fail_all_active_batches(reason)
            finally:
                self.worker.report_worker_fatal(reason)

    async def _drain_intake_loop(self) -> None:
        """Drain IntakePool → send admission messages to worker.

        On first drain, sends an "init" message to trigger worker initialization.
        Subsequent drains send "admit" messages with sequences.
        """
        logger.info("[POOL] Intake drain task started")
        while not self._stopped.is_set():
            if self._quiesced.is_set():
                self._quiesce_ack.set()
                await asyncio.sleep(0.05)
                continue
            if self._intake_pool.is_empty():
                self._intake_wakeup.clear()
                try:
                    await asyncio.wait_for(self._intake_wakeup.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
                continue

            # First drain: send init message to worker
            if not self._pool_initialized:
                init_msg = {"type": "init", **self.pool_init_params()}
                self.worker.request_queue.put(init_msg)
                self._pool_initialized = True
                logger.info("[POOL] Init message sent to worker")
                # Brief wait for worker to initialize before sending sequences
                await asyncio.sleep(1.0)

            # Drain from IntakePool (up to scheduling pool free slots)
            free_slots = self._scheduling_pool.num_free_slots()
            if free_slots <= 0:
                # DIAG: Log when drain is blocked by full scheduling pool
                if not hasattr(self, '_drain_blocked_logged'):
                    self._drain_blocked_logged = False
                if not self._drain_blocked_logged:
                    logger.warning(
                        f"[POOL] Drain blocked: scheduling pool full "
                        f"(active={self._scheduling_pool.num_active_slots()}, "
                        f"capacity={self._scheduling_pool._capacity}, "
                        f"intake={self._intake_pool.size()})"
                    )
                    self._drain_blocked_logged = True
                await asyncio.sleep(0.2)
                continue
            self._drain_blocked_logged = False

            drained = self._scheduling_pool.select_from_intake(
                self._intake_pool, max_n=free_slots
            )
            if not drained:
                await asyncio.sleep(0.1)
                continue
            drained = self._drop_stale_intake_entries(drained)
            if not drained:
                await asyncio.sleep(0)
                continue

            # Build admission message from drained entries
            admit_entries = []
            for entry in drained:
                slot = self._scheduling_pool.allocate_slot(entry.request_id)
                admit_entries.append({
                    "request_id": entry.request_id,
                    "text": entry.raw_request.get("text", ""),
                    "max_tokens": entry.raw_request.get("max_tokens", 4096),
                    "batch_id": entry.batch_id,
                    "priority": entry.priority.value,
                    "sampling_params": entry.raw_request.get("sampling_params", {}),
                    "ignore_eos": entry.raw_request.get("ignore_eos", False),
                    "batchgen_debug": entry.raw_request.get("batchgen_debug", {}),
                    # Agent-session identity; None for batch requests.
                    "traj_id": entry.raw_request.get("traj_id"),
                    "session_id": entry.raw_request.get("session_id"),
                    "parent_session_id": entry.raw_request.get("parent_session_id"),
                })

            admission_msg = {
                "type": "admit",
                "entries": admit_entries,
            }
            if self._quiesced.is_set():
                # The entries were selected before quiesce. Put them back so
                # upstream work survives; no worker message is sent.
                for entry in reversed(drained):
                    self._intake_pool.requeue(entry)
                for entry in drained:
                    self._scheduling_pool.free_slot(entry.request_id)
                await asyncio.sleep(0)
                continue
            self.worker.request_queue.put(admission_msg)
            if self._turn_waiters:
                sent_at = time.time()
                for entry in drained:
                    waiter = self._turn_waiters.get(entry.batch_id)
                    if waiter is not None:
                        waiter.timing["admit_sent_at"] = sent_at
            logger.warning(
                f"[POOL] Drained {len(admit_entries)} entries to worker "
                f"(intake remaining: {self._intake_pool.size()}, "
                f"scheduling active: {self._scheduling_pool.num_active_slots()}, "
                f"free={self._scheduling_pool.num_free_slots()})"
            )

        logger.info("[POOL] Intake drain task stopped")

    def _drop_stale_intake_entries(
        self, drained: List[IntakeEntry]
    ) -> List[IntakeEntry]:
        """Keep only entries whose batch can still accept results.

        _fail_all_active_batches purges a failed batch's queued entries, but an
        entry selected before the failure can be requeued after it.  An entry
        whose batch tracker is gone, complete, or failed is discarded here,
        before any slot is allocated for it, and is never sent to a worker.
        """
        live: List[IntakeEntry] = []
        dropped: Dict[str, int] = {}
        for entry in drained:
            if entry.batch_id in self._turn_waiters:
                live.append(entry)
                continue
            tracker = self._scheduling_pool.get_batch_tracker(entry.batch_id)
            if (
                tracker is None
                or tracker.is_complete
                or getattr(tracker, "is_failed", False)
            ):
                dropped[entry.batch_id] = dropped.get(entry.batch_id, 0) + 1
                continue
            live.append(entry)
        for batch_id, count in dropped.items():
            if batch_id in self._stale_intake_batches:
                continue
            self._stale_intake_batches.add(batch_id)
            logger.warning(
                "[POOL] Discarding %d queued requests of batch %s: the batch "
                "is no longer active (failed, complete, or finalized)",
                count,
                batch_id,
            )
        return live

    def _fail_all_active_batches(self, error_msg: str) -> int:
        """Mark all in-progress batches as failed. Called on fatal listener error."""
        failed = 0
        for batch_id, tracker in list(self._scheduling_pool._batch_trackers.items()):
            if not tracker.is_complete and not getattr(tracker, 'is_failed', False):
                tracker.error = error_msg
                # A failed batch must send nothing more to any worker: its
                # entries still queued in the intake pool would otherwise be
                # admitted to the next worker generation.
                purged = self._intake_pool.remove_batch_entries(batch_id)
                self.storage.update_batch_status(
                    batch_id, BatchStatus.FAILED, error=f"worker_fatal: {error_msg}"
                )
                logger.error(
                    f"[POOL] Batch {batch_id} marked FAILED: {error_msg} "
                    f"({purged} queued requests discarded)"
                )
                failed += 1
        self._fail_pending_turns(error_msg)
        clear_slots = getattr(self._scheduling_pool, "clear_active_slots", None)
        if clear_slots is not None:
            clear_slots()
        return failed

    async def _pool_completion_listener(self) -> None:
        """Background task: route per-request completions from the pump inbox.

        Routes each completion to the correct batch tracker and writes
        incremental output.  The blocking read of the worker response queue
        belongs to the generation's pump thread, never to this task: a read
        that can never return must not sit in the default executor, whose
        threads are not daemons and would keep the process from exiting.

        One listener serves every worker generation and never exits on a
        terminal item: a listener that ended with its generation could exit
        after the next generation's re-arm check saw it still running, leaving
        that generation with no consumer.  It ends only when the scheduler
        stops.  Each inbox item carries the generation of the pump that read
        it; an item of a superseded generation is dropped before it acts, so a
        stale terminal item can never fail or report the live generation.
        """
        logger.info("[POOL] Completion listener started")
        inbox = self._completion_inbox
        if inbox is None:
            raise RuntimeError(
                "completion listener started before the completion inbox "
                "existed; BatchScheduler.start() creates it"
            )
        while not self._stopped.is_set():
            try:
                generation, result = await asyncio.wait_for(
                    inbox.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if generation != self._pump_generation:
                logger.info(
                    "[POOL] Dropping a queued item of superseded worker "
                    "generation %d (live generation is %d): %s",
                    generation,
                    self._pump_generation,
                    result.get("type") if isinstance(result, dict) else result,
                )
                continue

            if result is None:
                logger.info(
                    "[POOL] Worker generation %d sent its end-of-stream "
                    "sentinel",
                    generation,
                )
                continue

            # One malformed item must not end delivery for every later item:
            # routing failures are logged with their traceback and skipped.
            try:
                if isinstance(result, dict):
                    msg_type = result.get("type")
                    if msg_type == "completion":
                        request_id = result.get("request_id")
                        batch_id = result.get("batch_id")
                        if request_id:
                            try:
                                self._scheduling_pool.free_slot(request_id)
                            except KeyError:
                                pass
                        # DIAG: Periodic slot status after completions
                        if not hasattr(self, '_completion_count'):
                            self._completion_count = 0
                        self._completion_count += 1
                        if self._completion_count % 500 == 0:
                            logger.warning(
                                f"[POOL] Completion #{self._completion_count}: "
                                f"active={self._scheduling_pool.num_active_slots()}, "
                                f"free={self._scheduling_pool.num_free_slots()}, "
                                f"intake={self._intake_pool.size()}"
                            )
                        if (
                            request_id in self._turn_waiters
                            or request_id in self._detached_turns
                        ):
                            # A turn: resolve its HTTP waiter; no batch tracker
                            # and no output file.
                            self._resolve_turn(request_id, result)
                            continue
                        # Write output JSONL line
                        if batch_id and request_id:
                            self._write_pool_completion(batch_id, request_id, result)
                        if batch_id:
                            batch_done = self._scheduling_pool.mark_request_completed(
                                request_id, batch_id
                            )
                            if batch_done:
                                logger.info(f"[POOL] Batch {batch_id} completed")
                    elif msg_type == "channel_error":
                        # The generation's pump could not read its response queue.
                        # Same handling as the in-task read failure this replaced:
                        # the active batches are persisted as failed.  The pump
                        # has ended; the listener stays for the next generation.
                        error = result.get("error")
                        logger.error(
                            "[POOL] Completion listener fatal error: %s", error
                        )
                        self._fail_all_active_batches(f"Worker error: {error}")
                        continue
                    elif msg_type == "pool_shutdown":
                        error = result.get("error")
                        if error:
                            logger.error("[POOL] Worker fatal received: %s", error)
                            self._fail_all_active_batches(error)
                            # The scheduler has already persisted this fatal before
                            # notifying WorkerManager; persisted=True keeps monitor
                            # reconciliation from persisting it a second time.
                            self.worker.report_worker_fatal(error, persisted=True)
                        else:
                            logger.info("[POOL] Worker shutdown signal received")
                        continue
                    elif msg_type == "trajectory_pool_capacity":
                        # ``total_capacity`` is immutable token-pool capacity and
                        # sizes SchedulingPool.  ``free_reservations`` changes on
                        # bind/release and is status/admission telemetry only; it
                        # must never resize the scheduler's slot list.
                        if (
                            result.get("capacity_semantics_version") != 1
                            or "total_capacity" not in result
                        ):
                            reason = (
                                "Incompatible trajectory-pool capacity snapshot: "
                                "worker must publish capacity_semantics_version=1 "
                                "and total_capacity; refusing the legacy ambiguous "
                                "capacity field"
                            )
                            logger.error("[POOL] %s", reason)
                            self._fail_all_active_batches(reason)
                            self.worker.report_worker_fatal(reason)
                            continue
                        total_capacity = int(result["total_capacity"])
                        if total_capacity != self._scheduling_pool.capacity:
                            if self._scheduling_pool.num_active_slots():
                                logger.error(
                                    "[POOL] Ignoring total capacity change while requests are active: "
                                    "old=%s new=%s active=%s",
                                    self._scheduling_pool.capacity,
                                    total_capacity,
                                    self._scheduling_pool.num_active_slots(),
                                )
                            else:
                                self._scheduling_pool.set_capacity(total_capacity)
                        self._trajectory_pool_info = dict(result)
                        logger.info(
                            "[POOL] Token pool capacity published: total_sequences=%s "
                            "free_reservations=%s free_pages=%s page_tokens=%s",
                            total_capacity,
                            result.get("free_reservations"),
                            result.get("free_pages"),
                            result.get("page_tokens"),
                        )
                    elif "error" in result:
                        logger.error(f"[POOL] Worker error: {result}")
                        continue
                    else:
                        # Anything else is a protocol error from the worker
                        logger.warning(f"[POOL] Unexpected result: {type(result)}")
            except Exception:
                logger.exception(
                    "[POOL] Completion listener failed to route an item of "
                    "generation %d; skipping it: %r",
                    generation,
                    result.get("type") if isinstance(result, dict) else result,
                )

        logger.info("[POOL] Completion listener stopped")

    def _write_pool_completion(
        self, batch_id: str, request_id: str, result: dict
    ) -> None:
        """Write a single completion to the batch output JSONL file.

        Builds an OpenAI-compatible BatchResultItem and appends it to
        {incremental_output_dir}/{batch_id}.jsonl.

        A worker that failed ONE request publishes an ``error`` on the
        completion instead of text (an over-context prompt, a reservation no
        Host pool can ever hold). Such a request gets the Batch API's error
        item -- ``response: null`` plus the worker's code and message -- not a
        200 with empty text, which is indistinguishable from a generation that
        legitimately produced nothing.
        """
        incr_dir = self.server_args.incremental_output_dir
        if not incr_dir:
            return

        meta = self._pool_request_meta.get(batch_id, {}).get(request_id)
        if not meta:
            logger.warning(f"[POOL] No metadata for {request_id} in batch {batch_id}")
            return

        error = result.get("error")
        if error:
            logger.error(
                "[POOL] Request %s of batch %s failed: %s",
                request_id, batch_id, error,
            )
            self._append_pool_result_item(
                batch_id,
                request_id,
                {
                    "id": f"batch_req_{uuid.uuid4().hex[:24]}",
                    "custom_id": meta["custom_id"],
                    "response": None,
                    "error": {
                        "code": error.get("code", "worker_error"),
                        "message": error.get("message", ""),
                    },
                },
            )
            return

        decoded_text = result.get("text", "")
        prompt_length = result.get("prompt_length", 0)
        decoded_length = result.get("decoded_length", 0)
        finish_reason = result.get("finish_reason", "stop")
        model = meta["model"]
        custom_id = meta["custom_id"]
        url = meta["url"]
        created_at = int(time.time())

        # Build response body based on endpoint type
        if url == "/v1/chat/completions":
            body = self._chat_completion_body(model, result, created_at)
        else:
            body = {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": created_at,
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": decoded_text,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_length,
                    "completion_tokens": decoded_length,
                    "total_tokens": prompt_length + decoded_length,
                },
            }

        self._append_pool_result_item(batch_id, request_id, {
            "id": f"batch_req_{uuid.uuid4().hex[:24]}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": f"req_{uuid.uuid4().hex}",
                "body": body,
            },
            "error": None,
        })

    def _chat_completion_body(
        self, model: str, result: dict, created_at: int
    ) -> dict:
        """The chat.completion body for one worker completion item."""
        decoded_text = result.get("text", "")
        prompt_length = result.get("prompt_length", 0)
        decoded_length = result.get("decoded_length", 0)
        content, reasoning_content, tool_calls = self._parse_output(
            model, decoded_text
        )
        message = {
            "role": "assistant",
            "content": content,
        }
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = [
                call.dict(exclude_none=True) for call in tool_calls
            ]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created_at,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": result.get("finish_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": prompt_length,
                "completion_tokens": decoded_length,
                "total_tokens": prompt_length + decoded_length,
            },
        }

    def _append_pool_result_item(
        self, batch_id: str, request_id: str, result_item: dict
    ) -> None:
        """Append one BatchResultItem to the batch's incremental JSONL."""
        incr_dir = self.server_args.incremental_output_dir
        if not incr_dir:
            return
        from pathlib import Path
        output_path = Path(incr_dir) / f"{batch_id}.jsonl"
        try:
            with open(output_path, "a") as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            logger.error(f"[POOL] Failed to write completion for {request_id}: {e}")

    def _finalize_batch_output(
        self,
        batch_id: str,
        requests: List[BatchRequestItem],
        prompts: List[str],
    ) -> None:
        """Finalize batch output after all requests complete.

        Per-request output is written by _write_pool_completion as each request
        completes.
        This method writes the batch status and output file metadata.
        """
        output_file_id = f"file-{uuid.uuid4().hex}"

        # Check for incremental output
        incremental_path = None
        incremental_output_dir = (
            self.server_args.incremental_output_dir
            if not self.server_args.no_incremental_save
            else None
        )
        if incremental_output_dir:
            from pathlib import Path
            import shutil
            incremental_path = Path(incremental_output_dir) / f"{batch_id}.jsonl"

        if incremental_path and incremental_path.exists() and incremental_path.stat().st_size > 0:
            import shutil
            api_path = self.storage.files_dir / output_file_id
            shutil.copy2(incremental_path, api_path)
            output_path = self.storage.output_dir / f"{output_file_id}.jsonl"
            shutil.copy2(incremental_path, output_path)
        else:
            # No incremental output available — write empty placeholder
            output_path = self.storage.output_dir / f"{output_file_id}.jsonl"
            output_path.write_text("")
            logger.warning(
                f"[POOL] Batch {batch_id}: no incremental output found, "
                f"writing empty output"
            )

        output_meta = FileObject(
            id=output_file_id,
            bytes=output_path.stat().st_size,
            created_at=int(time.time()),
            filename=output_path.name,
            purpose=FilePurpose.BATCH_OUTPUT.value,
            status=FileStatus.PROCESSED.value,
            status_details=None,
            checksum=None,
        )
        self.storage.save_metadata(output_file_id, output_meta.dict())

        completed_at = int(time.time())
        self.storage.update_batch_status(
            batch_id,
            BatchStatus.COMPLETED,
            completed_at=completed_at,
            output_file_id=output_file_id,
        )

        # Clean up batch tracker, intake info, and request metadata
        self._scheduling_pool.remove_batch_tracker(batch_id)
        self._intake_pool.remove_batch_info(batch_id)
        self._pool_request_meta.pop(batch_id, None)
        logger.info(f"[POOL] Batch {batch_id} finalized")
