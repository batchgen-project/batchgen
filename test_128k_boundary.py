#!/usr/bin/env python3
"""Test 128K context boundary crash.

Creates prompts of ~131069 tokens (128K - 3) so decode crosses 131072 after
just a few tokens. If the server crashes at token 131072, confirms the 128K limit.

Usage:
    python test_128k_boundary.py --base-url http://localhost:10900
"""

import argparse
import json
import os
import sys
import tempfile
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_TOKENS = 131072 - 3  # 131069 tokens in prompt


def build_long_prompt(target_tokens: int) -> str:
    """Build a chat prompt that tokenizes to approximately target_tokens."""
    # K2.5 uses a BPE tokenizer. A simple repeating sentence averages
    # ~1.3 tokens per word. Use a fixed sentence and repeat.
    # "The quick brown fox jumps over the lazy dog." = ~10 tokens
    # K2.5 BPE: ~0.89 tokens per word. Overshoot then trim with tokenizer.
    sentence = "The quick brown fox jumps over the lazy dog. "
    num_repeats = int(target_tokens * 0.15) + 1000  # generous overshoot
    long_text = sentence * num_repeats
    return long_text


def build_batch_jsonl(prompt: str, max_tokens: int = 256) -> str:
    """Build a JSONL batch file with one request."""
    request = {
        "custom_id": "test-128k-boundary",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "moonshotai/Kimi-K2.5",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    }

    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="test_128k_")
    with open(path, "w") as f:
        f.write(json.dumps(request) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:10900")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--target-prompt-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    logger.info(f"Target prompt tokens: {args.target_prompt_tokens}")
    logger.info(f"Max decode tokens: {args.max_tokens}")
    logger.info(f"Expected total at boundary: {args.target_prompt_tokens + args.max_tokens}")

    # Build long prompt
    logger.info("Building long prompt...")
    prompt = build_long_prompt(args.target_prompt_tokens)
    logger.info(f"Prompt length: {len(prompt)} chars")

    # Try to measure exact token count
    try:
        from transformers import AutoTokenizer
        cache_dir = os.environ.get(
            "TOKENIZER_CACHE",
            "/data2/tairan/modelscope/hub/models/moonshotai/Kimi-K2.5"
        )
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, trust_remote_code=True)
        tokens = tokenizer.encode(prompt)
        logger.info(f"Exact prompt tokens: {len(tokens)}")

        # Trim to exact target
        if len(tokens) > args.target_prompt_tokens:
            tokens = tokens[:args.target_prompt_tokens]
            prompt = tokenizer.decode(tokens)
            verify = tokenizer.encode(prompt)
            logger.info(f"Trimmed to {len(verify)} tokens ({len(prompt)} chars)")
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}. Using char-based estimate.")

    # Build JSONL
    jsonl_path = build_batch_jsonl(prompt, args.max_tokens)
    logger.info(f"Batch JSONL: {jsonl_path}")

    # Submit via BatchGen client
    from batchgen.batchgen_client import BatchGenHttpClient
    client = BatchGenHttpClient(base_url=args.base_url)

    logger.info("Uploading batch...")
    file_resp = client.upload_file(jsonl_path)
    file_id = file_resp["id"]
    logger.info(f"File ID: {file_id}")

    logger.info("Creating batch...")
    batch = client.create_batch(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        max_decoding_length=args.max_tokens,
    )
    batch_id = batch["id"]
    logger.info(f"Batch ID: {batch_id}")

    # Poll
    logger.info("Waiting for batch to complete...")
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > args.timeout:
            logger.error(f"Timeout after {elapsed:.0f}s")
            break

        batch = client.get_batch(batch_id)
        status = batch["status"]

        if status == "completed":
            logger.info(f"Batch completed in {elapsed:.1f}s")
            # Download results
            output_file_id = batch.get("output_file_id")
            if output_file_id:
                results = client.download_file(output_file_id)
                for line in results.strip().split("\n"):
                    result = json.loads(line)
                    resp = result.get("response", {}).get("body", {})
                    usage = resp.get("usage", {})
                    logger.info(f"Usage: {usage}")
                    choices = resp.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        logger.info(f"Output ({len(content)} chars): {content[:200]}...")
            logger.info("SUCCESS: No crash at 128K boundary")
            break
        elif status in ("failed", "cancelled"):
            logger.error(f"Batch {status} after {elapsed:.1f}s")
            logger.error(f"Error: {batch.get('error')}")
            break
        else:
            if int(elapsed) % 30 == 0:
                logger.info(f"Status: {status}, elapsed: {elapsed:.0f}s")
            time.sleep(5)

    os.unlink(jsonl_path)


if __name__ == "__main__":
    main()
