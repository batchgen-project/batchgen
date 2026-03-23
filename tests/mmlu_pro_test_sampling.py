#!/usr/bin/env python3
"""MMLU-Pro test with per-request sampling params and high max_tokens.

Mimics production settings: each request has different temperature/top_p.
Tests eviction re-entry correctness under realistic conditions.
"""
import argparse
import json
import logging
import os
import random
import tempfile
import time

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_BATCHGEN_ROOT = os.environ.get("BATCHGEN_ROOT", "/data2/tairan/workspace/BatchGen")
_TEST_PARQUET = os.path.join(_BATCHGEN_ROOT, "test/r1_mmlu_pro_test/mmlu_pro_test.parquet")
_VAL_PARQUET = os.path.join(_BATCHGEN_ROOT, "test/r1_mmlu_pro_test/mmlu_pro_validation.parquet")


def load_dataset(max_prompts=None):
    test_df = pd.read_parquet(_TEST_PARQUET)
    val_df = pd.read_parquet(_VAL_PARQUET)
    if max_prompts and max_prompts < len(test_df):
        test_df = test_df.head(max_prompts)
    return test_df, val_df


def form_options(options):
    return "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))


def build_few_shot_prefix(val_df, category, n_shot=5):
    cat_rows = val_df[val_df["category"] == category].head(n_shot)
    lines = []
    for _, row in cat_rows.iterrows():
        opts = form_options(row["options"])
        answer_letter = chr(65 + row["answer_index"])
        lines.append(
            f"Question: {row['question']}\n{opts}\nAnswer: ({answer_letter})"
        )
    return "\n\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-prompts", type=int, default=3000)
    parser.add_argument("--max-decoding-length", type=int, default=256000)
    parser.add_argument("--base-url", default="http://localhost:10900")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--timeout", type=float, default=72000)
    args = parser.parse_args()

    test_df, val_df = load_dataset(args.max_prompts)
    logger.info(f"Loaded {len(test_df)} test prompts")

    # Build JSONL with per-request sampling
    random.seed(42)
    few_shot_cache = {}
    lines = []
    sampling_stats = {"temps": [], "top_ps": []}

    for idx, row in test_df.iterrows():
        cat = row["category"]
        if cat not in few_shot_cache:
            few_shot_cache[cat] = build_few_shot_prefix(val_df, cat)

        opts = form_options(row["options"])
        query = (
            f"{few_shot_cache[cat]}\n\n"
            f"Question: {row['question']}\n{opts}\n"
            f"Answer: Let me think step by step."
        )

        # Per-request sampling params
        temp = round(random.uniform(0.3, 1.0), 2)
        top_p = round(random.uniform(0.8, 1.0), 2)
        sampling_stats["temps"].append(temp)
        sampling_stats["top_ps"].append(top_p)

        body = {
            "model": "moonshotai/Kimi-K2.5",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that thinks step by step."},
                {"role": "user", "content": query},
            ],
            "max_tokens": args.max_decoding_length,
            "temperature": temp,
            "top_p": top_p,
        }
        lines.append(json.dumps({
            "custom_id": f"mmlu-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))

    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="mmlu_sampling_")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Built batch JSONL: {path}")
    logger.info(
        f"Sampling stats: temp=[{min(sampling_stats['temps']):.2f}, {max(sampling_stats['temps']):.2f}], "
        f"top_p=[{min(sampling_stats['top_ps']):.2f}, {max(sampling_stats['top_ps']):.2f}]"
    )

    # Upload and create batch
    from batchgen.batchgen_client import BatchGenClient
    client = BatchGenClient(base_url=args.base_url)
    file_id = client.upload_batch_file(path)
    logger.info(f"Uploaded file: {file_id}")

    batch_id = client.create_batch(
        file_id, args.timeout, args.max_decoding_length,
        temperature=None, top_p=None, top_k=None,
    )
    logger.info(f"Created batch: {batch_id}")

    # Poll
    start = time.time()
    while True:
        status = client.get_batch_status(batch_id)
        if status["status"] in ("completed", "failed", "cancelled"):
            break
        elapsed = time.time() - start
        if elapsed > args.timeout:
            logger.error(f"Timeout after {elapsed:.0f}s")
            break
        logger.info(f"Batch {batch_id} status: {status['status']}, waiting...")
        time.sleep(5)

    logger.info(f"Batch completed: {status}")

    # Download results
    if status.get("output_file_id"):
        result_path = client.download_batch_results(status["output_file_id"])
        logger.info(f"Results: {result_path}")

        # Analyze
        tokens = []
        no_think = []
        short_seqs = []
        long_seqs = []
        with open(result_path) as f:
            for line in f:
                d = json.loads(line)
                ct = d["response"]["body"]["usage"]["completion_tokens"]
                c = d["response"]["body"]["choices"][0]["message"]["content"]
                cid = d["custom_id"]
                tokens.append(ct)
                if "</think>" not in c and ct < args.max_decoding_length:
                    no_think.append((cid, ct))
                if ct < 200:
                    short_seqs.append((cid, ct, c[-60:].replace("\n", " ")))
                if ct > 50000:
                    long_seqs.append((cid, ct, c[-60:].replace("\n", " ")))

        tokens.sort()
        n = len(tokens)
        print(f"\n=== RESULTS ===")
        print(f"Total: {n}, Mean: {sum(tokens)/n:.1f}, Median: {tokens[n//2]}")
        print(f"Min: {tokens[0]}, Max: {tokens[-1]}")
        print(f"P5: {tokens[int(n*0.05)]}, P95: {tokens[int(n*0.95)]}, P99: {tokens[int(n*0.99)]}")
        print(f"Short (<200 tokens): {len(short_seqs)}")
        for s in short_seqs:
            print(f"  {s[0]}: {s[1]} tok — [{s[2]}]")
        print(f"No </think> (non-maxlen): {len(no_think)}")
        for t in no_think[:5]:
            print(f"  {t[0]}: {t[1]} tok")
        print(f"Very long (>50K tokens): {len(long_seqs)}")
        for s in long_seqs[:5]:
            print(f"  {s[0]}: {s[1]} tok — [{s[2]}]")


if __name__ == "__main__":
    main()
