#!/usr/bin/env python3
"""Staggered MMLU-Pro batch submission test.

Submits the full MMLU-Pro dataset in multiple batches with a configurable
interval between submissions. Each batch is submitted while previous batches
are still running, testing the pool's concurrent batch handling.

Usage:
    python tests/staggered_mmlu_test.py \
        --base-url http://localhost:10900 \
        --model moonshotai/Kimi-K2.5 \
        --batch-size 2000 \
        --interval 1200 \
        --max-decoding-length 262144
"""
import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.environ.get("BATCHGEN_ROOT", "/data2/tairan/workspace/BatchGen"))
from batchgen.batchgen_client import BatchGenHttpClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------- Dataset helpers ----------

def form_options(options: List[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "".join(f"({letters[i]}): {opt}\n" for i, opt in enumerate(options))


def build_few_shot_prefix(val_df: pd.DataFrame, category: str) -> str:
    cat_examples = val_df[val_df["category"] == category].head(5)
    parts = []
    for _, row in cat_examples.iterrows():
        cot = row.get("cot_content", "")
        if isinstance(cot, str) and cot.startswith("A: "):
            cot = cot[3:]
        q = row["question"]
        opts = form_options(row["options"])
        ans = row["answer"]
        parts.append(f"Q: {q}\n{opts}A: {cot}\nThe answer is ({ans}).\n\n")
    return "".join(parts)


SYSTEM_PROMPT = (
    "You are a knowledge expert, you are supposed to answer the "
    "multi-choice question to derive your final answer as "
    "`The answer is ...`. Please follow the following examples and "
    "strictly give the answer with format "
    "'the answer is (A/B/C/D/E/F/G/H/I/J)'."
)

QUERY_TEMPLATE = (
    "Please read the following 5 examples: \n{prefix}"
    "Please answer the following question: \n"
    "Q: {question}\n{options}\n"
)


def extract_answer(text: str) -> Optional[str]:
    patterns = [
        r"(?i)\b(?:the\s+)?answer\s+is\s*\(?([ABCDEFGHIJ])\)?",
        r"(?i)Answer[s]?\s*[:\-]?\s*\(?([ABCDEFGHIJ])\)?",
        r"\\boxed\{[^}]*?([ABCDEFGHIJ])[^}]*\}",
        r"(?<![A-Za-z0-9])[\(\[]\s*([ABCDEFGHIJ])\s*[\)\]](?![A-Za-z0-9])",
        r"^\s*([ABCDEFGHIJ])\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


# ---------- Batch creation ----------

def create_batch_jsonl(
    samples: pd.DataFrame,
    val_df: pd.DataFrame,
    model: str,
    max_tokens: int,
    batch_idx: int,
) -> Tuple[str, dict]:
    """Create JSONL file for a batch, return (path, {custom_id: answer})."""
    answers = {}
    lines = []
    for i, (_, row) in enumerate(samples.iterrows()):
        prefix = build_few_shot_prefix(val_df, row["category"])
        query = QUERY_TEMPLATE.format(
            prefix=prefix,
            question=row["question"],
            options=form_options(row["options"]),
        )
        cid = f"b{batch_idx}-{i}"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "max_completion_tokens": max_tokens,
            "temperature": 0,
        }
        lines.append(json.dumps({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))
        answers[cid] = row["answer"]

    path = f"/tmp/staggered_batch_{batch_idx}.jsonl"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path, answers


# ---------- Batch polling thread ----------

class BatchPoller(threading.Thread):
    """Poll a batch for completion in background."""
    def __init__(self, client, batch_id, batch_idx, answers):
        super().__init__(daemon=True)
        self.client = client
        self.batch_id = batch_id
        self.batch_idx = batch_idx
        self.answers = answers
        self.correct = 0
        self.total = 0
        self.extraction_failures = 0
        self.elapsed = 0
        self.done = False
        self.error = None

    def run(self):
        try:
            t0 = time.time()
            batch = self.client.wait_for_batch(self.batch_id, poll_interval=10.0, timeout=86400)
            self.elapsed = time.time() - t0

            out_id = batch.get("output_file_id")
            if not out_id:
                self.error = "No output_file_id"
                self.done = True
                return

            content = self.client.download_file_content(out_id)
            for line in content.decode().strip().split("\n"):
                if not line.strip():
                    continue
                o = json.loads(line)
                cid = o.get("custom_id", "")
                choices = o.get("response", {}).get("body", {}).get("choices", [])
                text = choices[0].get("message", {}).get("content", "") if choices else ""
                pred = extract_answer(text)
                expected = self.answers.get(cid)
                self.total += 1
                if pred is None:
                    self.extraction_failures += 1
                elif pred == expected:
                    self.correct += 1

            self.done = True
        except Exception as e:
            self.error = str(e)
            self.done = True


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Staggered MMLU-Pro batch test")
    parser.add_argument("--base-url", default="http://localhost:10900")
    parser.add_argument("--model", default="moonshotai/Kimi-K2.5")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--interval", type=int, default=1200, help="Seconds between batch submissions (default 1200 = 20 min)")
    parser.add_argument("--interval-min", type=int, default=None, help="Min interval for random range (overrides --interval)")
    parser.add_argument("--interval-max", type=int, default=None, help="Max interval for random range")
    parser.add_argument("--max-decoding-length", type=int, default=262144)
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit total prompts (default: full dataset)")
    args = parser.parse_args()

    # Load dataset
    batchgen_root = os.environ.get("BATCHGEN_ROOT", "/data2/tairan/workspace/BatchGen")
    test_parquet = os.path.join(batchgen_root, "test/r1_mmlu_pro_test/mmlu_pro_test.parquet")
    val_parquet = os.path.join(batchgen_root, "test/r1_mmlu_pro_test/mmlu_pro_validation.parquet")
    dataset = pd.read_parquet(test_parquet)
    val_df = pd.read_parquet(val_parquet)

    if args.max_prompts:
        dataset = dataset.head(args.max_prompts)

    total_samples = len(dataset)
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    logger.info(f"Total samples: {total_samples}, batch_size: {args.batch_size}, num_batches: {num_batches}")
    logger.info(f"Interval: {args.interval}s ({args.interval/60:.0f} min) between submissions")

    client = BatchGenHttpClient(base_url=args.base_url)
    pollers = []
    t_start = time.time()

    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, total_samples)
        batch_samples = dataset.iloc[start:end]

        logger.info(f"=== Submitting batch {batch_idx}/{num_batches-1} ({len(batch_samples)} samples, rows {start}-{end-1}) ===")

        # Check pool status
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"{args.base_url}/v1/pool/status")
            status = json.loads(resp.read())
            logger.info(f"Pool status: intake={status.get('intake_pool_size',0)}, "
                       f"scheduling_active={status.get('scheduling_pool_active',0)}, "
                       f"scheduling_free={status.get('scheduling_pool_free',0)}")
        except Exception:
            pass

        # Create batch (non-blocking — returns immediately, poller tracks completion)
        path, answers = create_batch_jsonl(batch_samples, val_df, args.model, args.max_decoding_length, batch_idx)
        # Upload file
        file_info = client.upload_file(path)
        file_id = file_info.get("id")
        # Create batch (returns immediately, does NOT wait for completion)
        batch = client.create_batch(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            max_decoding_length=args.max_decoding_length,
            temperature=0,
        )
        batch_id = batch.get("id", "unknown")
        logger.info(f"Batch {batch_idx} submitted: {batch_id}")

        # Start background poller
        poller = BatchPoller(client, batch_id, batch_idx, answers)
        poller.start()
        pollers.append(poller)

        # Report completed batches so far
        for p in pollers:
            if p.done and not hasattr(p, '_reported'):
                acc = p.correct / p.total * 100 if p.total > 0 else 0
                logger.info(f"  >> Batch {p.batch_idx} COMPLETED: {p.correct}/{p.total} ({acc:.1f}%), "
                           f"extraction_failures={p.extraction_failures}, elapsed={p.elapsed:.0f}s"
                           + (f", error={p.error}" if p.error else ""))
                p._reported = True

        # Wait interval before next batch (skip for last batch)
        if batch_idx < num_batches - 1:
            import random
            if args.interval_min is not None and args.interval_max is not None:
                interval = random.randint(args.interval_min, args.interval_max)
            else:
                interval = args.interval
            logger.info(f"Waiting {interval}s ({interval/60:.1f} min) before next batch submission...")
            # Sleep in 30s chunks so we can report completions
            waited = 0
            while waited < interval:
                time.sleep(min(30, interval - waited))
                waited += 30
                for p in pollers:
                    if p.done and not hasattr(p, '_reported'):
                        acc = p.correct / p.total * 100 if p.total > 0 else 0
                        logger.info(f"  >> Batch {p.batch_idx} COMPLETED: {p.correct}/{p.total} ({acc:.1f}%), "
                                   f"extraction_failures={p.extraction_failures}, elapsed={p.elapsed:.0f}s"
                                   + (f", error={p.error}" if p.error else ""))
                        p._reported = True

    # Wait for all batches to complete
    logger.info("All batches submitted. Waiting for remaining to complete...")
    for p in pollers:
        p.join()

    # Final report
    total_correct = sum(p.correct for p in pollers)
    total_tested = sum(p.total for p in pollers)
    total_failures = sum(p.extraction_failures for p in pollers)
    total_elapsed = time.time() - t_start
    overall_acc = total_correct / total_tested * 100 if total_tested > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGGERED MMLU-PRO TEST RESULTS")
    logger.info("=" * 60)
    for p in pollers:
        acc = p.correct / p.total * 100 if p.total > 0 else 0
        logger.info(f"  Batch {p.batch_idx}: {p.correct}/{p.total} ({acc:.1f}%), "
                   f"failures={p.extraction_failures}, time={p.elapsed:.0f}s"
                   + (f", ERROR={p.error}" if p.error else ""))
    logger.info("-" * 60)
    logger.info(f"  TOTAL: {total_correct}/{total_tested} ({overall_acc:.1f}%)")
    logger.info(f"  Extraction failures: {total_failures}")
    logger.info(f"  Wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
