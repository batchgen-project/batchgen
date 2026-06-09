#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("bench_serving")

_SANITY_PROMPTS = [
    "What is 2+2?",
    "Explain gravity in one sentence.",
    "Name three primary colors.",
    "What is the capital of France?",
    "Define photosynthesis briefly.",
    "How many continents are there?",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "Name the largest planet in our solar system.",
    "What does CPU stand for?",
    "Translate 'hello' to Spanish.",
    "What is the square root of 144?",
    "Name one noble gas.",
    "What year did World War II end?",
    "What is the chemical symbol for gold?",
    "How many legs does a spider have?",
    "What is the freezing point of water in Fahrenheit?",
    "Name the author of 1984.",
    "What is the powerhouse of the cell?",
]


def _repeat_sentence(sentence: str, target_tokens: int) -> str:
    words_per_token = 0.75
    target_words = int(target_tokens * words_per_token)
    words = sentence.split()
    reps = max(1, target_words // len(words))
    return " ".join(words * reps)


def _build_workload(name: str) -> tuple[list[str], int, float]:
    if name == "sanity":
        return _SANITY_PROMPTS[:20], 256, 0.0

    if name == "short":
        base = "The quick brown fox jumps over the lazy dog near the riverbank on a warm afternoon."
        prompts = [_repeat_sentence(base, 128) for _ in range(500)]
        return prompts, 128, 0.0

    if name == "long":
        base = (
            "Discuss the historical, economic, and cultural factors that contributed "
            "to the rise and fall of major civilizations throughout human history, "
            "including but not limited to the Roman Empire, the Mongol Empire, "
            "the Ottoman Empire, and the British Empire. Analyze how geography, "
            "technology, trade routes, and social structures influenced their trajectories."
        )
        prompts = [_repeat_sentence(base, 512) for _ in range(100)]
        return prompts, 4096, 0.6

    if name == "mixed":
        import random

        rng = random.Random(42)
        base_sentences = [
            "Explain the theory of relativity and its implications for modern physics.",
            "Describe the process of machine learning model training from data collection to deployment.",
            "Summarize the key events of the French Revolution and their lasting impact on Europe.",
            "Discuss the environmental challenges facing the world today and potential solutions.",
            "Analyze the role of technology in transforming education over the past two decades.",
        ]
        prompts = []
        for i in range(200):
            target = rng.choice([64, 128, 256, 512, 1024, 2048])
            sentence = base_sentences[i % len(base_sentences)]
            prompts.append(_repeat_sentence(sentence, target))
        return prompts, 2048, 0.7

    raise ValueError(f"Unknown workload: {name}")


@dataclass
class RequestResult:
    request_id: int
    start_time: float
    end_time: float
    first_token_time: Optional[float] = None
    output_tokens: int = 0
    status_code: int = 0
    error: Optional[str] = None
    inter_token_latencies: list[float] = field(default_factory=list)

    @property
    def latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is not None:
            return self.first_token_time - self.start_time
        return None

    @property
    def success(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300


def _bench_batchgen(
    base_url: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    concurrency: int,
    timeout: float,
) -> list[RequestResult]:
    import requests as req_lib

    url = f"{base_url.rstrip('/')}/v1/inference"
    results: list[RequestResult] = []
    request_id = 0

    for batch_start in range(0, len(prompts), concurrency):
        batch = prompts[batch_start : batch_start + concurrency]
        futures = {}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for i, prompt in enumerate(batch):
                rid = request_id + i
                payload = {
                    "prompts": [prompt],
                    "max_output_len": max_tokens,
                }
                if temperature > 0:
                    payload["temperature"] = temperature

                def _do_request(
                    _rid: int = rid,
                    _payload: dict = payload,
                ) -> RequestResult:
                    start = time.monotonic()
                    try:
                        resp = req_lib.post(
                            url,
                            json=_payload,
                            timeout=timeout,
                        )
                        end = time.monotonic()
                        output_tokens = 0
                        if resp.status_code == 200:
                            body = resp.json()
                            for text in body.get("results", []):
                                output_tokens += len(text.split())
                        return RequestResult(
                            request_id=_rid,
                            start_time=start,
                            end_time=end,
                            output_tokens=output_tokens,
                            status_code=resp.status_code,
                            error=None
                            if resp.status_code == 200
                            else resp.text[:200],
                        )
                    except Exception as exc:
                        return RequestResult(
                            request_id=_rid,
                            start_time=start,
                            end_time=time.monotonic(),
                            status_code=0,
                            error=str(exc)[:200],
                        )

                futures[pool.submit(_do_request)] = rid

            for future in as_completed(futures):
                results.append(future.result())

        request_id += len(batch)

    return results


async def _send_openai_streaming(
    session,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: int,
    timeout: float,
    model: str = "default",
) -> RequestResult:
    import aiohttp

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    start = time.monotonic()
    first_token_time = None
    output_tokens = 0
    last_chunk_time = None
    itl_list: list[float] = []

    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.post(
            url, json=payload, timeout=client_timeout
        ) as resp:
            status_code = resp.status
            if status_code != 200:
                body = await resp.text()
                return RequestResult(
                    request_id=request_id,
                    start_time=start,
                    end_time=time.monotonic(),
                    status_code=status_code,
                    error=body[:200],
                )

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if not content:
                    continue

                now = time.monotonic()
                output_tokens += 1

                if first_token_time is None:
                    first_token_time = now
                    last_chunk_time = now
                else:
                    itl_list.append(now - last_chunk_time)
                    last_chunk_time = now

        end = time.monotonic()
        return RequestResult(
            request_id=request_id,
            start_time=start,
            end_time=end,
            first_token_time=first_token_time,
            output_tokens=output_tokens,
            status_code=status_code,
            inter_token_latencies=itl_list,
        )

    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            start_time=start,
            end_time=time.monotonic(),
            status_code=0,
            error=str(exc)[:200],
        )


async def _bench_openai_streaming(
    base_url: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    concurrency: int,
    timeout: float,
    model: str = "default",
) -> list[RequestResult]:
    import aiohttp

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def _limited(idx: int, prompt: str) -> RequestResult:
        async with semaphore:
            return await _send_openai_streaming(
                session,
                url,
                prompt,
                max_tokens,
                temperature,
                idx,
                timeout,
                model,
            )

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_limited(i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

    return list(results)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _compute_aggregates(
    results: list[RequestResult],
    wall_start: float,
    wall_end: float,
    framework: str,
) -> dict:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    wall_time = wall_end - wall_start

    latencies = [r.latency for r in successful]
    total_output_tokens = sum(r.output_tokens for r in successful)

    agg = {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "success_rate": len(successful) / max(len(results), 1),
        "wall_clock_time_s": round(wall_time, 3),
        "throughput_tokens_per_s": round(
            total_output_tokens / max(wall_time, 1e-9), 2
        ),
        "total_output_tokens": total_output_tokens,
    }

    if latencies:
        agg["latency_mean_s"] = round(statistics.mean(latencies), 4)
        agg["latency_p50_s"] = round(_percentile(latencies, 50), 4)
        agg["latency_p95_s"] = round(_percentile(latencies, 95), 4)
        agg["latency_p99_s"] = round(_percentile(latencies, 99), 4)
    else:
        agg["latency_mean_s"] = 0
        agg["latency_p50_s"] = 0
        agg["latency_p95_s"] = 0
        agg["latency_p99_s"] = 0

    if framework in ("vllm", "sglang"):
        ttfts = [r.ttft for r in successful if r.ttft is not None]
        all_itls = []
        for r in successful:
            all_itls.extend(r.inter_token_latencies)

        agg["ttft_mean_s"] = round(statistics.mean(ttfts), 4) if ttfts else None
        agg["ttft_p50_s"] = round(_percentile(ttfts, 50), 4) if ttfts else None
        agg["ttft_p95_s"] = round(_percentile(ttfts, 95), 4) if ttfts else None
        agg["itl_mean_s"] = (
            round(statistics.mean(all_itls), 4) if all_itls else None
        )
        agg["itl_p50_s"] = (
            round(_percentile(all_itls, 50), 4) if all_itls else None
        )
        agg["itl_p95_s"] = (
            round(_percentile(all_itls, 95), 4) if all_itls else None
        )

    return agg


def _serialize_result(r: RequestResult) -> dict:
    d = {
        "request_id": r.request_id,
        "start_time": r.start_time,
        "end_time": r.end_time,
        "latency_s": round(r.latency, 4),
        "output_tokens": r.output_tokens,
        "status_code": r.status_code,
        "success": r.success,
    }
    if r.first_token_time is not None:
        d["first_token_time"] = r.first_token_time
        d["ttft_s"] = round(r.ttft, 4) if r.ttft is not None else None
    if r.inter_token_latencies:
        d["itl_mean_s"] = round(statistics.mean(r.inter_token_latencies), 4)
    if r.error:
        d["error"] = r.error
    return d


def main():
    parser = argparse.ArgumentParser(
        description="Framework-aware serving benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Server base URL (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--framework", required=True, choices=["batchgen", "vllm", "sglang"]
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=["sanity", "short", "long", "mixed"],
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max parallel requests (default: 16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: stdout)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model identifier for OpenAI-compatible APIs (vllm/sglang)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=0,
        help="Cap number of prompts from the workload (0 = use all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Override workload max_output_len (0 = workload default)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prompts, max_tokens, temperature = _build_workload(args.workload)
    if args.num_prompts > 0:
        prompts = prompts[: args.num_prompts]
    if args.max_tokens > 0:
        max_tokens = args.max_tokens
    logger.info(
        "Workload=%s  prompts=%d  max_tokens=%d  temp=%.1f  concurrency=%d  framework=%s",
        args.workload,
        len(prompts),
        max_tokens,
        temperature,
        args.concurrency,
        args.framework,
    )

    wall_start = time.monotonic()

    if args.framework == "batchgen":
        results = _bench_batchgen(
            args.base_url,
            prompts,
            max_tokens,
            temperature,
            args.concurrency,
            args.timeout,
        )
    else:
        results = asyncio.run(
            _bench_openai_streaming(
                args.base_url,
                prompts,
                max_tokens,
                temperature,
                args.concurrency,
                args.timeout,
                args.model,
            )
        )

    wall_end = time.monotonic()

    aggregates = _compute_aggregates(
        results, wall_start, wall_end, args.framework
    )

    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    logger.info(
        "Done: %d/%d succeeded, %.1f tok/s, wall=%.1fs",
        successful,
        len(results),
        aggregates["throughput_tokens_per_s"],
        aggregates["wall_clock_time_s"],
    )
    if failed > 0:
        errors = [r.error for r in results if not r.success and r.error]
        for err in errors[:5]:
            logger.warning("  Failed request: %s", err)

    output = {
        "framework": args.framework,
        "workload": args.workload,
        "params": {
            "concurrency": args.concurrency,
            "num_prompts": len(prompts),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout_s": args.timeout,
            "base_url": args.base_url,
        },
        "aggregates": aggregates,
        "per_request": [
            _serialize_result(r)
            for r in sorted(results, key=lambda r: r.request_id)
        ],
    }

    json_str = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        logger.info("Results written to %s", args.output)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
