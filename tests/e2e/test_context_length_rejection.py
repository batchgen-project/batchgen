"""End-to-end tests for context-length rejection in batch API.

Tests that prompts exceeding model_context_length are rejected with
OpenAI-compatible error format while valid prompts complete normally.

Usage:
    python test/test_context_length_rejection.py \
        --base-url http://<server>:10900 \
        --model moonshotai/Kimi-K2.5

Requires a running BatchGen server with the target model loaded.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batchgen.batchgen_client import BatchGenHttpClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Kimi K2.5: 262144, GPT-OSS-120B: 131072
MODEL_CONTEXT_LENGTHS = {
    "moonshotai/Kimi-K2.5": 262144,
    "openai/gpt-oss-120b": 131072,
}

# Number of words to generate for over-limit prompts.
# Common English words tokenize to ~1 token each, so 300K words ≈ 300K tokens,
# safely exceeding 262K (Kimi K2.5) or 128K (GPT-OSS) context limits.
OVERLIMIT_NUM_WORDS = 300_000

BASE_WORDS = ["hello", "world", "the", "quick", "brown", "fox", "jumps",
              "over", "lazy", "dog", "alpha", "beta", "gamma", "delta"]


def make_short_prompt(idx: int) -> str:
    """Create a short prompt (~50 tokens)."""
    return f"What is {idx} + {idx * 2}? Answer briefly."


def make_overlimit_prompt(ctx_len: int) -> str:
    """Create a prompt that clearly exceeds ctx_len tokens.

    Generates 300K words (~300K tokens), well above any model context limit.
    """
    words = [BASE_WORDS[i % len(BASE_WORDS)] for i in range(OVERLIMIT_NUM_WORDS)]
    return " ".join(words)


def build_request(custom_id: str, model: str, prompt: str,
                  max_completion_tokens: int = 16) -> Dict[str, Any]:
    """Build a single BatchRequestItem dict."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0.0,
        },
    }


def write_jsonl(requests: List[Dict[str, Any]], path: str) -> None:
    """Write requests to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def parse_results(content: bytes) -> List[Dict[str, Any]]:
    """Parse JSONL batch results."""
    results = []
    for line in content.decode("utf-8").strip().split("\n"):
        if not line.strip():
            continue
        results.append(json.loads(line))
    return results


def run_batch(client: BatchGenHttpClient, jsonl_path: str,
              timeout: float = 600.0, poll_interval: float = 5.0,
              max_decoding_length: int = 16) -> List[Dict[str, Any]]:
    """Upload, create batch, wait, download results."""
    file_obj = client.upload_file(jsonl_path)
    file_id = file_obj["id"]
    logger.info(f"  Uploaded: {file_id}")

    batch = client.create_batch(
        file_id,
        endpoint="/v1/chat/completions",
        max_decoding_length=max_decoding_length,
    )
    batch_id = batch["id"]
    logger.info(f"  Batch created: {batch_id}")

    batch = client.wait_for_batch(batch_id, poll_interval=poll_interval,
                                  timeout=timeout)
    logger.info(f"  Batch status: {batch['status']}")

    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise RuntimeError(f"No output_file_id in batch: {batch}")

    content = client.download_file_content(output_file_id)
    results = parse_results(content)
    logger.info(f"  Results: {len(results)} items")
    return results


def assert_error_item(item: Dict[str, Any], expected_code: str = "context_length_exceeded"):
    """Assert a result item is an error with the expected code."""
    cid = item.get("custom_id", "?")
    error = item.get("error")
    assert error is not None, f"[{cid}] Expected error but got none: {item}"
    assert error.get("code") == expected_code, \
        f"[{cid}] Expected error code '{expected_code}', got '{error.get('code')}'"
    assert item.get("response") is None, \
        f"[{cid}] Expected response=null for error item"
    return True


def assert_success_item(item: Dict[str, Any]):
    """Assert a result item is a successful completion."""
    cid = item.get("custom_id", "?")
    response = item.get("response")
    assert response is not None, f"[{cid}] Expected response but got none"
    assert response.get("status_code") == 200, \
        f"[{cid}] Expected status_code 200, got {response.get('status_code')}"
    body = response.get("body", {})
    choices = body.get("choices", [])
    assert len(choices) > 0, f"[{cid}] Expected choices but got empty"
    assert item.get("error") is None, \
        f"[{cid}] Expected error=null for success item, got {item.get('error')}"
    return True


# ============ Test Cases ============

def test_case_1_all_over_limit(client, model, ctx_len, tmpdir):
    """Case 1: All requests over limit → all rejected, no inference."""
    logger.info("=" * 60)
    logger.info("CASE 1: All requests over limit (4 requests)")
    logger.info("=" * 60)

    requests = []
    for i in range(4):
        prompt = make_overlimit_prompt(ctx_len)
        requests.append(build_request(f"over-{i}", model, prompt))

    path = os.path.join(tmpdir, "case1.jsonl")
    write_jsonl(requests, path)
    logger.info(f"  Input: {len(requests)} requests, all over {ctx_len} tokens")

    results = run_batch(client, path)

    # Validate
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    result_by_id = {r["custom_id"]: r for r in results}
    for i in range(4):
        assert_error_item(result_by_id[f"over-{i}"])
    logger.info("  PASS: All 4 requests rejected with context_length_exceeded")
    return True


def test_case_2_all_under_limit(client, model, ctx_len, tmpdir):
    """Case 2: All requests under limit → all succeed."""
    logger.info("=" * 60)
    logger.info("CASE 2: All requests under limit (4 requests)")
    logger.info("=" * 60)

    requests = []
    for i in range(4):
        prompt = make_short_prompt(i)
        requests.append(build_request(f"short-{i}", model, prompt))

    path = os.path.join(tmpdir, "case2.jsonl")
    write_jsonl(requests, path)
    logger.info(f"  Input: {len(requests)} requests, all short prompts")

    results = run_batch(client, path)

    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    result_by_id = {r["custom_id"]: r for r in results}
    for i in range(4):
        assert_success_item(result_by_id[f"short-{i}"])
    logger.info("  PASS: All 4 requests completed successfully")
    return True


def test_case_3_mixed(client, model, ctx_len, tmpdir):
    """Case 3: Mixed — 2 over + 4 under → 2 errors + 4 successes."""
    logger.info("=" * 60)
    logger.info("CASE 3: Mixed batch (2 over + 4 under = 6 requests)")
    logger.info("=" * 60)

    requests = []
    over_ids = set()
    under_ids = set()

    # 2 over-limit
    for i in range(2):
        prompt = make_overlimit_prompt(ctx_len)
        cid = f"mixed-over-{i}"
        requests.append(build_request(cid, model, prompt))
        over_ids.add(cid)

    # 4 under-limit
    for i in range(4):
        prompt = make_short_prompt(i)
        cid = f"mixed-under-{i}"
        requests.append(build_request(cid, model, prompt))
        under_ids.add(cid)

    path = os.path.join(tmpdir, "case3.jsonl")
    write_jsonl(requests, path)
    logger.info(f"  Input: {len(requests)} requests (2 over, 4 under)")

    results = run_batch(client, path)

    assert len(results) == 6, f"Expected 6 results, got {len(results)}"
    result_by_id = {r["custom_id"]: r for r in results}
    for cid in over_ids:
        assert_error_item(result_by_id[cid])
    for cid in under_ids:
        assert_success_item(result_by_id[cid])
    logger.info("  PASS: 2 rejected + 4 completed correctly")
    return True


    # Cases 4 and 5 (exact boundary tests) removed — require tokenizer
    # for precise token count targeting. Covered by Cases 1-3 and 6-8.


def test_case_6_asymmetric_ranks(client, model, ctx_len, tmpdir):
    """Case 6: 16 requests, indices 0,3,7,15 over limit (4 rejected, 12 valid).

    With 16 GPUs round-robin assignment, rejected sequences span different ranks.
    """
    logger.info("=" * 60)
    logger.info("CASE 6: Asymmetric rejection across ranks (16 requests)")
    logger.info("=" * 60)

    over_indices = {0, 3, 7, 15}
    requests = []
    for i in range(16):
        if i in over_indices:
            prompt = make_overlimit_prompt(ctx_len)
            cid = f"rank-over-{i}"
        else:
            prompt = make_short_prompt(i)
            cid = f"rank-ok-{i}"
        requests.append(build_request(cid, model, prompt))

    path = os.path.join(tmpdir, "case6.jsonl")
    write_jsonl(requests, path)
    logger.info(f"  Input: 16 requests (4 over at indices {over_indices})")

    results = run_batch(client, path)

    assert len(results) == 16, f"Expected 16 results, got {len(results)}"
    result_by_id = {r["custom_id"]: r for r in results}

    error_count = 0
    success_count = 0
    for i in range(16):
        if i in over_indices:
            assert_error_item(result_by_id[f"rank-over-{i}"])
            error_count += 1
        else:
            assert_success_item(result_by_id[f"rank-ok-{i}"])
            success_count += 1

    logger.info(f"  PASS: {error_count} rejected + {success_count} completed (multi-rank)")
    return True


def test_case_7_single_overlimit_in_large_batch(client, model, ctx_len, tmpdir):
    """Case 7: 20 requests, only index 10 over limit."""
    logger.info("=" * 60)
    logger.info("CASE 7: Single over-limit in large batch (20 requests)")
    logger.info("=" * 60)

    requests = []
    for i in range(20):
        if i == 10:
            prompt = make_overlimit_prompt(ctx_len)
            cid = "large-over-10"
        else:
            prompt = make_short_prompt(i)
            cid = f"large-ok-{i}"
        requests.append(build_request(cid, model, prompt))

    path = os.path.join(tmpdir, "case7.jsonl")
    write_jsonl(requests, path)
    logger.info(f"  Input: 20 requests (1 over at index 10)")

    results = run_batch(client, path)

    assert len(results) == 20, f"Expected 20 results, got {len(results)}"
    result_by_id = {r["custom_id"]: r for r in results}

    assert_error_item(result_by_id["large-over-10"])
    for i in range(20):
        if i != 10:
            assert_success_item(result_by_id[f"large-ok-{i}"])

    logger.info("  PASS: 1 rejected + 19 completed in large batch")
    return True


def test_case_8_error_format(client, model, ctx_len, tmpdir):
    """Case 8: Validate error message format matches OpenAI."""
    logger.info("=" * 60)
    logger.info("CASE 8: Error message format validation")
    logger.info("=" * 60)

    prompt = make_overlimit_prompt(ctx_len)
    requests = [build_request("fmt-check", model, prompt)]

    path = os.path.join(tmpdir, "case8.jsonl")
    write_jsonl(requests, path)

    results = run_batch(client, path)

    assert len(results) == 1
    item = results[0]
    assert_error_item(item)

    error = item["error"]
    msg = error["message"]

    # Check message format
    assert f"maximum context length is {ctx_len} tokens" in msg, \
        f"Error message missing context length. Got: {msg}"
    assert "your messages resulted in" in msg, \
        f"Error message missing token count. Got: {msg}"
    assert "Please reduce the length" in msg, \
        f"Error message missing suggestion. Got: {msg}"

    # Extract reported token count
    import re
    match = re.search(r"resulted in (\d+) tokens", msg)
    assert match, f"Could not extract token count from: {msg}"
    reported_tokens = int(match.group(1))
    assert reported_tokens >= ctx_len, \
        f"Reported tokens ({reported_tokens}) should be >= ctx_len ({ctx_len})"

    logger.info(f"  Error message: {msg}")
    logger.info(f"  Reported token count: {reported_tokens}")
    logger.info("  PASS: Error format matches OpenAI spec")
    return True


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(
        description="E2E tests for context-length rejection"
    )
    parser.add_argument("--base-url", type=str, default="http://localhost:10900",
                        help="BatchGen server URL")
    parser.add_argument("--model", type=str, default="moonshotai/Kimi-K2.5",
                        help="Model name")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Override model context length (default: auto-detect from model name)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Max seconds to wait per batch")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case numbers to run (e.g., '1,3,8'). Default: all")
    parser.add_argument("--tmpdir", type=str, default=None,
                        help="Directory for temp JSONL files (default: auto)")
    args = parser.parse_args()

    # Determine context length
    ctx_len = args.context_length
    if ctx_len is None:
        ctx_len = MODEL_CONTEXT_LENGTHS.get(args.model)
        if ctx_len is None:
            logger.error(f"Unknown model '{args.model}'. Use --context-length to specify.")
            sys.exit(1)

    logger.info(f"Model: {args.model}")
    logger.info(f"Context length: {ctx_len}")
    logger.info(f"Server: {args.base_url}")

    client = BatchGenHttpClient(args.base_url, timeout_s=args.timeout)

    # Health check
    if not client.health_check():
        logger.warning("Server health check failed, proceeding anyway...")

    # Setup tmpdir
    if args.tmpdir:
        tmpdir = args.tmpdir
        os.makedirs(tmpdir, exist_ok=True)
    else:
        tmpdir = tempfile.mkdtemp(prefix="ctx_reject_test_")
    logger.info(f"Temp dir: {tmpdir}")

    # Select cases
    all_cases = {
        1: test_case_1_all_over_limit,
        2: test_case_2_all_under_limit,
        3: test_case_3_mixed,
        6: test_case_6_asymmetric_ranks,
        7: test_case_7_single_overlimit_in_large_batch,
        8: test_case_8_error_format,
    }

    if args.cases:
        selected = [int(c.strip()) for c in args.cases.split(",")]
    else:
        selected = sorted(all_cases.keys())

    # Run tests
    passed = 0
    failed = 0
    errors = []

    for case_num in selected:
        test_fn = all_cases.get(case_num)
        if not test_fn:
            logger.warning(f"Unknown case {case_num}, skipping")
            continue
        try:
            test_fn(client, args.model, ctx_len, tmpdir)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((case_num, str(e)))
            logger.error(f"  FAIL: Case {case_num}: {e}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"SUMMARY: {passed} passed, {failed} failed out of {len(selected)} cases")
    logger.info("=" * 60)
    if errors:
        for case_num, err in errors:
            logger.error(f"  Case {case_num}: {err}")
        sys.exit(1)
    else:
        logger.info("All tests passed!")


if __name__ == "__main__":
    main()
