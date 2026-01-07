"""Batch scheduling and execution loop for OpenAI-compatible batch API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, List, Optional, Tuple

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
    max_tokens_value: Optional[int] = None

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
            current_max_tokens = request.body.max_tokens
        elif isinstance(request.body, CompletionRequest):
            current_model = request.body.model
            current_max_tokens = request.body.max_tokens
        else:
            return False, f"Line {idx}: Unsupported request body", []

        if not current_max_tokens:
            return False, f"Line {idx}: max_tokens is required", []

        if model_name is None:
            model_name = current_model
        elif model_name != current_model:
            return False, f"Line {idx}: Inconsistent model value", []

        if max_tokens_value is None:
            max_tokens_value = current_max_tokens
        elif max_tokens_value != current_max_tokens:
            return False, f"Line {idx}: Inconsistent max_tokens value", []

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

        prompts, max_tokens = self._convert_requests_to_worker_inputs(requests)
        try:
            results = await asyncio.to_thread(
                self.worker.infer,
                prompts,
                self.server_args.max_input_len,
                max_tokens,
                False,
            )
        except Exception as exc:
            self.storage.update_batch_status(
                batch_id, BatchStatus.FAILED, error=str(exc)
            )
            logger.exception("Batch %s inference failed", batch_id)
            return

        output_file_id = f"file-{uuid.uuid4().hex}"
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
        self, requests: List[BatchRequestItem]
    ) -> Tuple[List[str], int]:
        prompts: List[str] = []
        max_tokens: Optional[int] = None

        for request in requests:
            body = request.body
            if isinstance(body, ChatCompletionRequest):
                prompt = self._format_chat_messages(
                    [m.dict() for m in body.messages], body.model
                )
                current_max_tokens = (
                    body.max_tokens or self.server_args.max_output_len
                )
            elif isinstance(body, CompletionRequest):
                prompt = completion_prompt_to_text(body.prompt)
                current_max_tokens = (
                    body.max_tokens or self.server_args.max_output_len
                )
            else:
                raise ValueError("Unsupported request body type")

            prompts.append(prompt)
            if max_tokens is None:
                max_tokens = current_max_tokens

        if max_tokens is None:
            max_tokens = self.server_args.max_output_len
        return prompts, max_tokens

    def _format_chat_messages(self, messages: List[dict], model: str) -> str:
        tokenizer = self._get_tokenizer(model)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                f"Tokenizer for {model} does not support apply_chat_template"
            )
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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

        if request.url == BatchEndpoint.CHAT_COMPLETIONS:
            body: BatchResponseBody = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=created_at,
                model=model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            content=decoded_text
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

        if request.url == BatchEndpoint.CHAT_COMPLETIONS:
            body: BatchResponseBody = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=created_at,
                model=model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            content=decoded_text
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
        if self._tokenizer_model == model and self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "Transformers is required for batch decoding. "
                "Install transformers or configure it in the environment."
            )

        cache_dir = self.server_args.cache_dir or self.server_args.hf_cache_dir
        cache_dir_value = str(cache_dir) if cache_dir else None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                cache_dir=cache_dir_value,
                trust_remote_code=True,
                local_files_only=True,
            )
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
