"""Batch scheduling and execution loop for OpenAI-compatible batch API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from batchgen.server.io_struct import (
    BatchEndpoint,
    BatchError,
    BatchRequestItem,
    BatchResponse,
    BatchResponseBody,
    BatchResultItem,
    BatchStatus,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    FileObject,
    FilePurpose,
    FileStatus,
    ToolCall,
    ToolCallFunction,
    Usage,
)
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

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            request = BatchRequestItem(**payload)
        except Exception as exc:
            return False, f"Line {idx}: {exc}", []

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

    async def start(self) -> None:
        if self._task:
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if not self._task:
            return
        self._stopped.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

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

        prompts, per_request_max_tokens, sampling_params = self._convert_requests_to_worker_inputs(
            requests, batch
        )
        # Apply batch-level max_decoding_length as fallback for requests without explicit value
        default_max = batch.max_decoding_length
        if default_max is None:
            default_max = 128
            n_missing = sum(1 for mt in per_request_max_tokens if mt is None)
            if n_missing > 0:
                logger.warning(
                    f"Batch {batch_id}: {n_missing}/{len(per_request_max_tokens)} requests have no "
                    f"max_completion_tokens/max_tokens; using fallback={default_max}. "
                    f"Set batch-level max_decoding_length or per-request max_completion_tokens."
                )
        per_request_max_tokens = [
            mt if mt is not None else default_max
            for mt in per_request_max_tokens
        ]
        max_tokens = max(per_request_max_tokens)

        # Log batch-level sampling param defaults
        if batch.temperature is not None or batch.top_p is not None or batch.top_k is not None:
            logger.warning(
                f"Batch {batch_id}: batch-level sampling params "
                f"(temperature={batch.temperature}, top_p={batch.top_p}, top_k={batch.top_k}) "
                f"serve as defaults only; per-request values take priority"
            )

        # Build incremental writer metadata
        incremental_output_dir = (
            self.server_args.incremental_output_dir
            if not self.server_args.no_incremental_save
            else None
        )
        incremental_kwargs = {}
        if incremental_output_dir:
            incremental_kwargs = dict(
                custom_id_map={idx: req.custom_id for idx, req in enumerate(requests)},
                request_url_map={idx: req.url.value for idx, req in enumerate(requests)},
                prompt_text_map={idx: prompts[idx] for idx in range(len(prompts))},
                batch_id=batch_id,
                model_name=requests[0].body.model if requests else "unknown",
                incremental_output_dir=incremental_output_dir,
                parse_thinking=self.server_args.parse_thinking,
                parse_tool_call=self.server_args.parse_tool_call,
            )

        try:
            results = await asyncio.to_thread(
                self.worker.infer,
                prompts,
                None,  # max_input_len: dynamically determined from prompts
                max_tokens,
                False,  # ignore_eos
                None,  # temperature: handled via per-request sampling_params
                None,  # top_p: handled via per-request sampling_params
                max_context_length=batch.max_context_length,
                sampling_params=sampling_params,
                per_sequence_max_tokens=per_request_max_tokens,
                **incremental_kwargs,
            )
        except Exception as exc:
            self.storage.update_batch_status(
                batch_id, BatchStatus.FAILED, error=str(exc)
            )
            logger.exception("Batch %s inference failed", batch_id)
            return

        output_file_id = f"file-{uuid.uuid4().hex}"

        # If incremental save was active, use the incremental JSONL as the output
        incremental_path = None
        if incremental_output_dir:
            from pathlib import Path
            import shutil
            incremental_path = Path(incremental_output_dir) / f"{batch_id}.jsonl"

        if incremental_path and incremental_path.exists() and incremental_path.stat().st_size > 0:
            # Copy incremental file to storage for API access
            api_path = self.storage.files_dir / output_file_id
            shutil.copy2(incremental_path, api_path)
            output_path = self.storage.output_dir / f"{output_file_id}.jsonl"
            shutil.copy2(incremental_path, output_path)
            logger.info(
                f"Batch {batch_id}: using incremental output "
                f"({incremental_path} -> {output_path})"
            )
        else:
            # Fallback: build output the original way
            output_items = self._build_output_items(requests, results, prompts)
            output_path = self.storage.write_output_file(
                output_file_id, output_items
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
                # Inject reasoning_effort into system message for GPT-OSS models
                messages = self._inject_reasoning_effort(
                    [m.dict() for m in body.messages],
                    body.model,
                    body.reasoning_effort,
                )
                # Forward extra kwargs (thinking, tools) to chat template
                template_kwargs = {}
                if body.thinking is not None:
                    template_kwargs["thinking"] = body.thinking
                if body.tools is not None:
                    template_kwargs["tools"] = body.tools
                prompt = self._format_chat_messages(
                    messages, body.model, **template_kwargs
                )
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

    def _build_output_items(
        self,
        requests: List[BatchRequestItem],
        results: List[Any],
        prompts: List[str],
    ) -> List[BatchResultItem]:
        output: List[BatchResultItem] = []
        normalized_results = self._normalize_worker_results(
            results, len(requests)
        )
        for idx, request in enumerate(requests):
            result = (
                normalized_results[idx]
                if idx < len(normalized_results)
                else None
            )
            response = None
            error = None
            if result is None:
                error = BatchError(
                    code="missing_result",
                    message="No result returned for this request.",
                )
            else:
                prompt_text = prompts[idx] if idx < len(prompts) else ""
                # Handle both decoded strings and token IDs
                if isinstance(result, str):
                    # Worker returned decoded string (server-side detokenization)
                    body = self._build_response_body_from_text(
                        request, result, prompt_text
                    )
                    response = self._wrap_response(body)
                else:
                    # Worker returned token IDs (legacy behavior)
                    token_ids = self._coerce_token_ids(result)
                    if token_ids is None:
                        error = BatchError(
                            code="invalid_result",
                            message=(
                                "Unsupported result payload: " f"{type(result)}"
                            ),
                        )
                    else:
                        body = self._build_response_body(
                            request, token_ids, prompt_text
                        )
                        response = self._wrap_response(body)

            output.append(
                BatchResultItem(
                    id=f"batch_req_{uuid.uuid4().hex[:24]}",
                    custom_id=request.custom_id,
                    response=response,
                    error=error,
                )
            )

        if len(normalized_results) > len(requests):
            logger.warning(
                "Received more results (%d) than requests (%d)",
                len(normalized_results),
                len(requests),
            )
        return output

    def _wrap_response(self, body: BatchResponseBody) -> BatchResponse:
        return BatchResponse(
            status_code=200,
            request_id=f"req_{uuid.uuid4().hex}",
            body=body,
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

    def _build_response_body(
        self,
        request: BatchRequestItem,
        token_ids: List[int],
        prompt_text: str,
    ) -> BatchResponseBody:
        model = request.body.model
        created_at = int(time.time())
        decoded_text = self._decode_tokens(model, token_ids)
        usage = self._build_usage(model, prompt_text, token_ids)
        content, reasoning_content, tool_calls = self._parse_output(
            model, decoded_text
        )

        if request.url == BatchEndpoint.CHAT_COMPLETIONS:
            body: BatchResponseBody = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=created_at,
                model=model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            content=content,
                            reasoning_content=reasoning_content,
                            tool_calls=tool_calls,
                        ),
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
                usage=usage,
            )
        else:
            body = CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex}",
                created=created_at,
                model=model,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=decoded_text,
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
                usage=usage,
            )
        return body

    def _build_response_body_from_text(
        self,
        request: BatchRequestItem,
        decoded_text: str,
        prompt_text: str,
    ) -> BatchResponseBody:
        """Build response body from decoded text (server-side detokenization)."""
        model = request.body.model
        created_at = int(time.time())
        usage = self._build_usage_from_text(model, prompt_text, decoded_text)
        content, reasoning_content, tool_calls = self._parse_output(
            model, decoded_text
        )

        if request.url == BatchEndpoint.CHAT_COMPLETIONS:
            body: BatchResponseBody = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=created_at,
                model=model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            content=content,
                            reasoning_content=reasoning_content,
                            tool_calls=tool_calls,
                        ),
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
                usage=usage,
            )
        else:
            body = CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex}",
                created=created_at,
                model=model,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=decoded_text,
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
                usage=usage,
            )
        return body

    def _build_usage_from_text(
        self, model: str, prompt_text: str, completion_text: str
    ) -> Optional[Usage]:
        """Build usage stats from text (for server-side detokenization)."""
        tokenizer = self._get_tokenizer(model)
        if tokenizer is None:
            return None
        prompt_tokens = self._count_tokens(tokenizer, prompt_text)
        completion_tokens = self._count_tokens(tokenizer, completion_text)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _build_usage(
        self, model: str, prompt_text: str, token_ids: List[int]
    ) -> Optional[Usage]:
        tokenizer = self._get_tokenizer(model)
        if tokenizer is None:
            return None
        prompt_tokens = self._count_tokens(tokenizer, prompt_text)
        completion_tokens = len(token_ids)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _count_tokens(self, tokenizer: Any, text: str) -> int:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _decode_tokens(self, model: str, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        tokenizer = self._get_tokenizer(model)
        if tokenizer is None:
            return " ".join(str(token) for token in token_ids)
        trimmed = self._trim_tokens(token_ids, tokenizer)
        return tokenizer.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _trim_tokens(self, token_ids: List[int], tokenizer: Any) -> List[int]:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        trimmed = list(token_ids)
        if eos_token_id is not None and eos_token_id in trimmed:
            trimmed = trimmed[: trimmed.index(eos_token_id)]
        if pad_token_id is not None:
            while trimmed and trimmed[-1] == pad_token_id:
                trimmed.pop()
        return trimmed

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

    def _normalize_worker_results(
        self, results: List[Any], expected: int
    ) -> List[Any]:
        normalized = [self._normalize_result_value(item) for item in results]
        if len(normalized) == expected:
            return normalized
        if len(normalized) == 1:
            single = normalized[0]
            if (
                isinstance(single, list)
                and len(single) == expected
                and all(isinstance(item, (list, tuple)) for item in single)
            ):
                return [list(item) for item in single]
        return normalized

    def _coerce_token_ids(self, value: Any) -> Optional[List[int]]:
        normalized = self._normalize_result_value(value)
        if isinstance(normalized, list):
            if not normalized:
                return []
            if all(isinstance(item, (int, float)) for item in normalized):
                return [int(item) for item in normalized]
            if (
                len(normalized) == 1
                and isinstance(normalized[0], (list, tuple))
                and all(
                    isinstance(item, (int, float)) for item in normalized[0]
                )
            ):
                return [int(item) for item in normalized[0]]
        return None

    def _normalize_result_value(self, value: Any) -> Any:
        if (
            hasattr(value, "detach")
            and hasattr(value, "cpu")
            and hasattr(value, "tolist")
        ):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return [self._normalize_result_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._normalize_result_value(val)
                for key, val in value.items()
            }
        return value
