"""Batch scheduling and execution loop for OpenAI-compatible batch API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from batchgen.server.io_struct import (
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


def parse_batch_file(
    content: bytes,
) -> Tuple[bool, Optional[str], List[BatchRequestItem]]:
    """Validate and parse a JSONL batch file."""
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

        # custom_id is unique within one OpenAI Batch.  It is also used as
        # the public output key, so reject duplicates before they can alias
        # the scheduler's internal request slot.
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
        elif model_name != current_model:
            return False, f"Line {idx}: Inconsistent model value", []

        requests.append(request)

    if not requests:
        return False, "Batch file cannot be empty", []
    return True, None, requests


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
        # Per-request metadata for building output JSONL in pool mode
        # Structure: {batch_id: {request_id: {custom_id, url, model, prompt_text}}}
        # Internal request_id values are scoped by batch.
        self._pool_request_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def start(self) -> None:
        if self._task:
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())
        # Start pool mode background tasks
        self._completion_listener_task = asyncio.create_task(
            self._pool_completion_listener()
        )
        self._drain_task = asyncio.create_task(
            self._drain_intake_to_worker()
        )

    async def stop(self) -> None:
        if not self._task:
            return
        self._stopped.set()
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

        ok, error_message, requests = parse_batch_file(content)
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

        # custom_id is only unique inside its input Batch, while scheduling
        # slots and worker UUIDs span all concurrently queued Batches.  Keep
        # the caller's custom_id in metadata, but use a batch-scoped internal
        # request id for every in-flight lookup.
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
        """Run the intake drain and fail active Batches on a drain error.

        A drain exception can occur after entries leave IntakePool or slots are
        allocated.  Leaving this task dead strands those Batches in
        ``in_progress`` forever, so surface the exact error and stop through
        the worker-fatal path.  ``CancelledError`` intentionally propagates.
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
            if self._intake_pool.is_empty():
                await asyncio.sleep(0.1)
                continue

            # First drain: send init message to worker
            if not self._pool_initialized:
                init_msg = {
                    "type": "init",
                    "max_output_len": getattr(self, '_pool_max_output_len', 4096),
                    "max_context_length": getattr(self, '_pool_max_context_length', None),
                }
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
                })

            admission_msg = {
                "type": "admit",
                "entries": admit_entries,
            }
            self.worker.request_queue.put(admission_msg)
            logger.warning(
                f"[POOL] Drained {len(admit_entries)} entries to worker "
                f"(intake remaining: {self._intake_pool.size()}, "
                f"scheduling active: {self._scheduling_pool.num_active_slots()}, "
                f"free={self._scheduling_pool.num_free_slots()})"
            )

        logger.info("[POOL] Intake drain task stopped")

    def _fail_all_active_batches(self, error_msg: str) -> None:
        """Mark all in-progress batches as failed. Called on fatal listener error."""
        for batch_id, tracker in list(self._scheduling_pool._batch_trackers.items()):
            if not tracker.is_complete and not getattr(tracker, 'is_failed', False):
                tracker.error = error_msg
                self.storage.update_batch_status(
                    batch_id, BatchStatus.FAILED, error=f"worker_fatal: {error_msg}"
                )
                logger.error(f"[POOL] Batch {batch_id} marked FAILED: {error_msg}")

    async def _pool_completion_listener(self) -> None:
        """Background task: read per-request completions from worker response queue.

        Routes each completion to the correct batch tracker and writes
        incremental output.
        """
        import queue as queue_mod
        logger.info("[POOL] Completion listener started")
        while not self._stopped.is_set():
            try:
                result = await asyncio.to_thread(
                    self.worker.response_queue.get,
                    timeout=1.0,
                )
            except queue_mod.Empty:
                continue
            except Exception as e:
                logger.error(f"[POOL] Completion listener fatal error: {e}", exc_info=True)
                self._fail_all_active_batches(f"Worker error: {e}")
                break

            if result is None:
                break

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
                    # Write output JSONL line
                    if batch_id and request_id:
                        self._write_pool_completion(batch_id, request_id, result)
                    if batch_id:
                        batch_done = self._scheduling_pool.mark_request_completed(
                            request_id, batch_id
                        )
                        if batch_done:
                            logger.info(f"[POOL] Batch {batch_id} completed")
                elif msg_type == "pool_shutdown":
                    error = result.get("error")
                    if error:
                        logger.error("[POOL] Worker fatal received: %s", error)
                        self._fail_all_active_batches(error)
                        self.worker.report_worker_fatal(error)
                    else:
                        logger.info("[POOL] Worker shutdown signal received")
                    break
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
                        break
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
                    break
                else:
                    # Anything else is a protocol error from the worker
                    logger.warning(f"[POOL] Unexpected result: {type(result)}")

        logger.info("[POOL] Completion listener stopped")

    def _write_pool_completion(
        self, batch_id: str, request_id: str, result: dict
    ) -> None:
        """Write a single completion to the batch output JSONL file.

        Builds an OpenAI-compatible BatchResultItem and appends it to
        {incremental_output_dir}/{batch_id}.jsonl.
        """
        incr_dir = self.server_args.incremental_output_dir
        if not incr_dir:
            return

        meta = self._pool_request_meta.get(batch_id, {}).get(request_id)
        if not meta:
            logger.warning(f"[POOL] No metadata for {request_id} in batch {batch_id}")
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
            body = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": created_at,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_length,
                    "completion_tokens": decoded_length,
                    "total_tokens": prompt_length + decoded_length,
                },
            }
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

        result_item = {
            "id": f"batch_req_{uuid.uuid4().hex[:24]}",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": f"req_{uuid.uuid4().hex}",
                "body": body,
            },
            "error": None,
        }

        # Append to JSONL file
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
