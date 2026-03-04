"""Single-scenario test runner for dynamic host KV testing.

Submits a JSONL batch to a running BatchGen server and collects results.

Usage:
    python run_test.py \
        --input data/scenarioA.jsonl \
        --output results/dynamic_scenarioA.jsonl \
        --server-url http://localhost:10900 \
        --timeout 7200
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add BatchGen to path
BATCHGEN_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BATCHGEN_ROOT))

from batchgen.batchgen_client import BatchGenHttpClient


def count_requests(input_path: str) -> int:
    """Count number of requests in input JSONL."""
    count = 0
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Submit batch test to BatchGen server")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    parser.add_argument("--server-url", type=str, default="http://localhost:10900", help="Server base URL")
    parser.add_argument("--timeout", type=float, default=7200, help="Max seconds to wait")
    parser.add_argument("--poll-interval", type=float, default=10, help="Seconds between status checks")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    n_requests = count_requests(str(input_path))
    logger.info(f"Input: {input_path} ({n_requests} requests)")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "results" / f"{input_path.stem}_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to server
    client = BatchGenHttpClient(args.server_url, timeout_s=args.timeout)
    logger.info(f"Connecting to {args.server_url}...")

    if not client.health_check():
        logger.error("Server health check FAILED")
        sys.exit(1)
    logger.info("Server health check OK")

    # Submit batch
    t_start = time.time()
    logger.info(f"Submitting batch ({n_requests} requests)...")

    try:
        batch = client.submit_batch(
            input_file_path=str(input_path),
            output_file_path=str(output_path),
            endpoint="/v1/chat/completions",
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            temperature=0.0,
        )
    except Exception as e:
        logger.error(f"Batch submission failed: {e}")
        sys.exit(1)

    elapsed = time.time() - t_start
    status = batch.get("status", "unknown")
    output_file_id = batch.get("output_file_id")
    total_requests = batch.get("request_counts", {}).get("total", n_requests)
    completed = batch.get("request_counts", {}).get("completed", 0)
    failed = batch.get("request_counts", {}).get("failed", 0)

    logger.info(f"Batch completed in {elapsed:.1f}s")
    logger.info(f"  Status: {status}")
    logger.info(f"  Completed: {completed}/{total_requests}")
    logger.info(f"  Failed: {failed}")

    # Download results if not already saved
    if output_file_id and not output_path.exists():
        content = client.download_file_content(output_file_id)
        with open(output_path, "wb") as f:
            f.write(content)
        logger.info(f"Results saved to {output_path}")
    elif output_path.exists():
        logger.info(f"Results already at {output_path}")

    # Quick summary of results
    if output_path.exists():
        n_results = 0
        n_empty = 0
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    n_results += 1
                    result = json.loads(line)
                    response = result.get("response", {})
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if not choices or not choices[0].get("message", {}).get("content"):
                        n_empty += 1
        logger.info(f"Results: {n_results} responses ({n_empty} empty)")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Run Summary")
    print(f"{'='*60}")
    print(f"Input:     {input_path}")
    print(f"Output:    {output_path}")
    print(f"Requests:  {n_requests}")
    print(f"Completed: {completed}")
    print(f"Failed:    {failed}")
    print(f"Time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Throughput: {completed/elapsed:.1f} seq/s" if elapsed > 0 else "")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
