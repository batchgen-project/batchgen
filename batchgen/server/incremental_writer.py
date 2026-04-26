"""Incremental result writer for crash-resilient batch output.

Writes completed sequence results to disk as they finish, rather than
waiting for the entire batch. Each completed sequence is detokenized in
a background thread and appended as a JSONL line with fsync for durability.

Runs on rank 0 in the worker process only.
"""

import json
import logging
import os
import queue
import threading
import time
import uuid as uuid_lib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from batchgen.server.io_struct import (
    BatchEndpoint,
    BatchError,
    BatchResponse,
    BatchResultItem,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionResponse,
    ToolCall,
    ToolCallFunction,
    Usage,
)

logger = logging.getLogger(__name__)


class IncrementalWriter:
    """Background writer that persists completed sequences to JSONL.

    Thread-safe: submit() can be called from the decode hot path with
    negligible overhead (just a queue.put). All detokenization and disk
    I/O happen in a daemon thread.
    """

    def __init__(
        self,
        output_dir: str,
        batch_id: str,
        model_name: str,
        custom_id_map: Dict[int, str],
        request_urls: Dict[int, str],
        prompt_texts: Dict[int, str],
        tokenizer: Any,
        eos_token_ids: Set[int],
        pad_token_id: int = 0,
        parse_thinking: bool = False,
        parse_tool_call: bool = False,
        batchgen_debug: Optional[Dict[str, Any]] = None,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self._output_dir / f"{batch_id}.jsonl"
        self._model_name = model_name
        self._custom_id_map = custom_id_map
        self._request_urls = request_urls
        self._prompt_texts = prompt_texts
        self._tokenizer = tokenizer
        self._eos_token_ids = set(eos_token_ids)
        self._pad_token_id = pad_token_id
        self._parse_thinking = parse_thinking
        self._parse_tool_call = parse_tool_call
        self._batchgen_debug = dict(batchgen_debug or {})

        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._count = 0

        # Create file eagerly so it exists before any writes
        self._output_path.touch(exist_ok=True)

        self._thread = threading.Thread(
            target=self._background_loop,
            name="incremental-writer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"IncrementalWriter started: batch_id={batch_id}, "
            f"output={self._output_path}, sequences={len(custom_id_map)}"
        )

    def submit(self, global_idx: int, decoded_tokens: torch.Tensor, finish_reason: str = "stop") -> None:
        """Enqueue a completed sequence for async writing. Thread-safe."""
        if self._closed:
            logger.warning("IncrementalWriter.submit() called after close()")
            return
        tokens_cpu = decoded_tokens.cpu() if decoded_tokens.is_cuda else decoded_tokens.clone()
        self._queue.put((global_idx, tokens_cpu, finish_reason))

    def submit_error(self, global_idx: int, error_code: str, error_message: str) -> None:
        """Enqueue an error result for a rejected sequence. Thread-safe."""
        if self._closed:
            logger.warning("IncrementalWriter.submit_error() called after close()")
            return
        self._queue.put(("error", global_idx, error_code, error_message))

    def close(self) -> None:
        """Flush remaining items and join the background thread."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)  # sentinel
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            logger.warning("IncrementalWriter thread did not join in 30s")
        logger.info(
            f"IncrementalWriter closed: {self._count} items written to {self._output_path}"
        )

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def items_written(self) -> int:
        return self._count

    # -------------------- Background thread --------------------

    def _background_loop(self) -> None:
        """Daemon thread: dequeue -> detokenize -> build JSON -> append + fsync."""
        try:
            with open(self._output_path, "a", encoding="utf-8") as fh:
                while True:
                    item = self._queue.get()
                    if item is None:
                        fh.flush()
                        os.fsync(fh.fileno())
                        break

                    try:
                        # Error items: ("error", global_idx, error_code, error_message)
                        if isinstance(item, tuple) and len(item) == 4 and item[0] == "error":
                            _, global_idx, error_code, error_message = item
                            line = self._build_error_line(global_idx, error_code, error_message)
                        else:
                            # Normal items: (global_idx, tokens, finish_reason)
                            global_idx, tokens, finish_reason = item
                            line = self._build_result_line(global_idx, tokens, finish_reason=finish_reason)
                        fh.write(line)
                        fh.write("\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                        self._count += 1
                    except Exception:
                        logger.exception(
                            f"IncrementalWriter: failed to write item"
                        )
        except Exception:
            logger.exception("IncrementalWriter background thread crashed")

    # -------------------- Result building --------------------

    def _build_result_line(self, global_idx: int, tokens: torch.Tensor, finish_reason: str = "stop") -> str:
        """Build a BatchResultItem-compatible JSON line."""
        custom_id = self._custom_id_map.get(global_idx, f"unknown_{global_idx}")
        endpoint_url = self._request_urls.get(global_idx, BatchEndpoint.CHAT_COMPLETIONS.value)
        prompt_text = self._prompt_texts.get(global_idx, "")

        # Detokenize
        decoded_text = self._detokenize(tokens)

        # Token counts
        prompt_tokens = len(prompt_text.split()) if prompt_text else 0
        # Approximate from decoded tokens tensor (actual non-padding count)
        completion_tokens = self._count_valid_tokens(tokens)

        # Parse thinking / tool calls
        content, reasoning_content, tool_calls = self._parse_output(decoded_text)

        created_at = int(time.time())
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        if endpoint_url == BatchEndpoint.CHAT_COMPLETIONS.value:
            body = ChatCompletionResponse(
                id=f"chatcmpl-{uuid_lib.uuid4().hex}",
                created=created_at,
                model=self._model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            content=content,
                            reasoning_content=reasoning_content,
                            tool_calls=tool_calls,
                        ),
                        logprobs=None,
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
            )
        else:
            body = CompletionResponse(
                id=f"cmpl-{uuid_lib.uuid4().hex}",
                created=created_at,
                model=self._model_name,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=decoded_text,
                        logprobs=None,
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
            )

        result_item = BatchResultItem(
            id=f"batch_req_{uuid_lib.uuid4().hex[:24]}",
            custom_id=custom_id,
            response=BatchResponse(
                status_code=200,
                request_id=f"req_{uuid_lib.uuid4().hex}",
                body=body,
            ),
            error=None,
            batchgen_debug=self._batchgen_debug or None,
        )

        return json.dumps(result_item.dict(), default=str, ensure_ascii=False)

    def _build_error_line(self, global_idx: int, error_code: str, error_message: str) -> str:
        """Build a BatchResultItem-compatible JSON line for a rejected sequence."""
        custom_id = self._custom_id_map.get(global_idx, f"unknown_{global_idx}")
        result_item = BatchResultItem(
            id=f"batch_req_{uuid_lib.uuid4().hex[:24]}",
            custom_id=custom_id,
            response=None,
            error=BatchError(code=error_code, message=error_message),
            batchgen_debug=self._batchgen_debug or None,
        )
        return json.dumps(result_item.dict(), default=str, ensure_ascii=False)

    # -------------------- Helpers --------------------

    def _detokenize(self, tokens: torch.Tensor) -> str:
        """Decode token tensor to string, stopping at first EOS.

        Mirrors BatchGenWorker._decode_tokens_to_string().
        """
        if tokens.dim() > 1:
            tokens = tokens.squeeze(0)
        tokens_list = tokens.tolist()

        # Find first EOS position (skip position 0 to avoid empty output)
        eos_positions = [
            i for i, t in enumerate(tokens_list)
            if t in self._eos_token_ids and i >= 1
        ]

        if eos_positions:
            end_pos = eos_positions[0]
        else:
            # No EOS: use all non-padding tokens
            non_pad = [i for i, t in enumerate(tokens_list) if t != self._pad_token_id]
            end_pos = non_pad[-1] + 1 if non_pad else len(tokens_list)

        return self._tokenizer.decode(tokens_list[:end_pos], skip_special_tokens=False)

    def _count_valid_tokens(self, tokens: torch.Tensor) -> int:
        """Count non-padding tokens up to first EOS.

        Both EOS and pad checks need the `i >= 1` guard. For models where
        pad_token_id is in eos_token_ids (e.g. GLM-5: pad=154820 is also the
        first EOS), the asymmetric guard caused ctok=0 when the model
        emitted token 154820 at position 0 (pad-check fires) vs ctok=1 when
        the model emitted 154827/154829 (eos-check skipped at i=0, pad found
        at i=1). The decoded behavior is identical — the inconsistency was
        purely in the reported completion_tokens.
        """
        if tokens.dim() > 1:
            tokens = tokens.squeeze(0)
        tokens_list = tokens.tolist()
        for i, t in enumerate(tokens_list):
            if i >= 1 and (t in self._eos_token_ids or t == self._pad_token_id):
                return i
        return len(tokens_list)

    def _parse_output(
        self, decoded_text: str
    ) -> tuple:
        """Apply thinking/tool-call parsing if flags are enabled.

        Returns (content, reasoning_content, tool_calls).
        Mirrors BatchScheduler._parse_output().
        """
        content = decoded_text
        reasoning_content = None
        tool_calls = None

        if self._parse_thinking:
            try:
                reasoning_content, content = self._tokenizer.parse_thinking(content)
            except (NotImplementedError, AttributeError):
                pass

        if self._parse_tool_call:
            try:
                raw_calls, content = self._tokenizer.parse_tool_calls(content)
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
            except (NotImplementedError, AttributeError):
                pass

        return content, reasoning_content, tool_calls
