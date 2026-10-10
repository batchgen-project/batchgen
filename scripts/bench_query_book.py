#!/usr/bin/env python3
"""Remote-only QueryBook microbenchmarks.

Reports fixed-store construction, prompt writes plus model-input copy, and
one-token writes for a decode batch whose rows have different positions.
Run this on the assigned GPU machine; do not use it as a dev-machine test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from batchgen.query_book import QueryBook


def median_ms(values_ns: list[int]) -> float:
    return statistics.median(values_ns) / 1_000_000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-gb", type=float, default=1.0)
    parser.add_argument("--page-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1_048_576)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    capacity_bytes = int(args.capacity_gb * (1024**3))
    if args.batch_size <= 0 or args.prompt_tokens <= 0:
        raise ValueError("batch-size and prompt-tokens must be positive")
    if args.prompt_tokens >= args.max_tokens:
        raise ValueError("prompt-tokens must be smaller than max-tokens")

    init_ns: list[int] = []
    for _ in range(args.warmup + args.repeats):
        start = time.perf_counter_ns()
        book = QueryBook(capacity_bytes, page_tokens=args.page_tokens)
        elapsed = time.perf_counter_ns() - start
        if _ >= args.warmup:
            init_ns.append(elapsed)
        del book

    book = QueryBook(capacity_bytes, page_tokens=args.page_tokens)
    slots = [
        book.bind(f"q-{idx}", max_tokens=args.max_tokens)
        for idx in range(args.batch_size)
    ]
    prompts = [
        torch.arange(args.prompt_tokens - idx % 257, dtype=torch.int64)
        for idx in range(args.batch_size)
    ]

    prefill_ns: list[int] = []
    for _ in range(args.warmup + args.repeats):
        start = time.perf_counter_ns()
        book.write_prompts(slots, prompts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter_ns() - start
        if _ >= args.warmup:
            prefill_ns.append(elapsed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpu_prefill_ns: list[int] = []
    gpu_outputs = [
        torch.empty(prompt.numel(), dtype=torch.int64, device=device)
        for prompt in prompts
    ]
    for _ in range(args.warmup + args.repeats):
        start = time.perf_counter_ns()
        for slot, prompt, output in zip(slots, prompts, gpu_outputs):
            book.copy_to(slot, device=device, dtype=torch.int64, length=prompt.numel(), out=output)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter_ns() - start
        if _ >= args.warmup:
            gpu_prefill_ns.append(elapsed)

    decode_ns: list[int] = []
    token_ids = list(range(args.batch_size))
    for _ in range(args.warmup + args.repeats):
        start = time.perf_counter_ns()
        book.append_tokens(slots, token_ids)
        elapsed = time.perf_counter_ns() - start
        if _ >= args.warmup:
            decode_ns.append(elapsed)
        # Restore the measured state by binding a fresh book when the rows hit
        # the configured limit; this stays outside the measured operation.
        if book.metadata(slots[0]).token_length >= args.max_tokens:
            book = QueryBook(capacity_bytes, page_tokens=args.page_tokens)
            slots = [
                book.bind(f"q-{idx}", max_tokens=args.max_tokens)
                for idx in range(args.batch_size)
            ]
            book.write_prompts(slots, prompts)

    result = {
        "capacity_bytes": capacity_bytes,
        "capacity_gib": capacity_bytes / (1024**3),
        "page_tokens": args.page_tokens,
        "page_count": book.page_count,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "prompt_tokens_min": min(prompt.numel() for prompt in prompts),
        "prompt_tokens_max": max(prompt.numel() for prompt in prompts),
        "device": str(device),
        "init_ms_median": median_ms(init_ns),
        "prefill_write_ms_median": median_ms(prefill_ns),
        "prefill_copy_to_model_ms_median": median_ms(gpu_prefill_ns),
        "decode_batch_append_ms_median": median_ms(decode_ns),
        "decode_batch_append_us_per_sequence": median_ms(decode_ns) * 1000 / args.batch_size,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
