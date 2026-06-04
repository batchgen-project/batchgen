#!/usr/bin/env python3
"""Cross-framework MMLU-Pro 5-shot accuracy test.

Runs MMLU-Pro evaluation against batchgen, vllm, or sglang and reports
overall accuracy, per-category accuracy, no-think counts, and token stats.

Usage:
    python mmlu_pro_cross_framework.py --base-url http://localhost:8000 --framework batchgen
    python mmlu_pro_cross_framework.py --base-url http://localhost:8000 --framework vllm
    python mmlu_pro_cross_framework.py --base-url http://localhost:8000 --framework sglang
"""

import argparse
import asyncio
import json
import logging
import re
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_TEST_PARQUET = _SCRIPT_DIR / "r1_mmlu_pro_test" / "mmlu_pro_test.parquet"
_VAL_PARQUET = _SCRIPT_DIR / "r1_mmlu_pro_test" / "mmlu_pro_validation.parquet"


# ---------------------------------------------------------------------------
# Helpers (copied inline from r1_mmlu_pro_test.py and mmlu_pro_test_sampling.py)
# ---------------------------------------------------------------------------


def form_options(options: List[str]) -> str:
    option_str = "Options are:\n"
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, letter in zip(options, letters):
        option_str += f"({letter}): {opt}\n"
    return option_str


def _parse_think_output(text: str) -> Tuple[str, str]:
    """Split optional <think>...</think> from final answer content."""
    if "<think>" not in text:
        return "", text
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end() :].strip()
    start = text.find("<think>")
    return text[start + len("<think>") :].strip(), ""


_EXTRACT_PATTERNS = [
    r"(?i)\b(?:the\s+)?answer\s+is\s*\(?([ABCDEFGHIJ])\)?",
    r"(?i)(?:\*{1,2}|_{1,2})?Answer[s]?\s*[:\-–]?(?:\*{1,2}|_{1,2})?\s*\(?([ABCDEFGHIJ])\)?",
    r"(?i)correct answer is \(?([ABCDEFGHIJ])\)?",
    r"(?i)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCDEFGHIJ])\b",
    r"\\boxed\{[^}]*?([ABCDEFGHIJ])[^}]*\}",
    r"(?<![A-Za-z0-9])[\(\[]\s*([ABCDEFGHIJ])\s*[\)\]](?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCDEFGHIJ])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])",
    r"(?:^|\s)\(?([ABCDEFGHIJ])\)?\s*$",
    r"(?:^|\s)([ABCDEFGHIJ])[\.\:]",
    r"^\s*([ABCDEFGHIJ])\s*$",
]


def extract_prediction(model_output: str) -> Tuple[Optional[str], bool]:
    """Extract the predicted letter answer from a model output string.

    Matches the GLM5 standard (test/glm5_mmlu_pro_test/glm5_mmlu_pro_batch_test.py).
    Strips <think>...</think> blocks first, then tries an ordered list of regex
    patterns covering "the answer is (X)", boxed answers, bare letters, etc.
    Returns (predicted_letter or None, whether_think_tag_was_present).
    """
    _, answer_content = _parse_think_output(model_output)
    think_tag_found = bool(answer_content) or "<think>" in model_output
    search_text = answer_content if answer_content else model_output

    for pattern in _EXTRACT_PATTERNS:
        m = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), think_tag_found

    return None, think_tag_found


def build_few_shot_prefix(
    val_df: pd.DataFrame, category: str, n_shot: int = 5
) -> str:
    cat_rows = val_df[val_df["category"] == category].head(n_shot)
    parts = []
    for _, row in cat_rows.iterrows():
        opts = form_options(row["options"])
        answer_letter = row["answer"]
        cot = row["cot_content"].strip()
        if cot.startswith("A:"):
            cot = cot[2:].strip()
        parts.append(
            f"Q: {row['question']}\n{opts}A: {cot}\nThe answer is ({answer_letter}).\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(max_prompts: Optional[int] = None):
    test_df = pd.read_parquet(_TEST_PARQUET)
    val_df = pd.read_parquet(_VAL_PARQUET)
    if max_prompts and max_prompts < len(test_df):
        test_df = test_df.head(max_prompts)
    return test_df, val_df


def build_prompts(test_df: pd.DataFrame, val_df: pd.DataFrame) -> List[Dict]:
    prompts = []
    for idx, row in test_df.iterrows():
        prefix = build_few_shot_prefix(val_df, row["category"])
        opts = form_options(row["options"])
        question_block = f"Q: {row['question']}\n{opts}A:"
        full_prompt = prefix + question_block
        correct_letter = row["answer"]
        prompts.append(
            {
                "custom_id": f"mmlu_pro_{idx}",
                "prompt": full_prompt,
                "correct_answer": correct_letter,
                "category": row["category"],
            }
        )
    return prompts


# ---------------------------------------------------------------------------
# Framework adapters
# ---------------------------------------------------------------------------


def call_batchgen(
    base_url: str,
    prompts: List[Dict],
    max_decoding_length: int,
    temperature: float,
) -> List[Dict]:
    url = f"{base_url.rstrip('/')}/v1/inference"
    results = []
    for i, p in enumerate(prompts):
        try:
            resp = requests.post(
                url,
                json={
                    "prompts": [p["prompt"]],
                    "max_output_len": max_decoding_length,
                    "temperature": temperature,
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["results"][0] if data.get("results") else ""
            results.append(
                {
                    "custom_id": p["custom_id"],
                    "text": text,
                    "completion_tokens": len(text.split()),
                    "error": None,
                }
            )
        except Exception as e:
            logger.error("Prompt %s failed: %s", p["custom_id"], e)
            results.append(
                {
                    "custom_id": p["custom_id"],
                    "text": "",
                    "completion_tokens": 0,
                    "error": str(e),
                }
            )
        if (i + 1) % 100 == 0:
            logger.info(
                "Progress: %d / %d prompts completed", i + 1, len(prompts)
            )
    return results


_MMLU_SYSTEM_PROMPT = (
    "You are an expert at answering multiple-choice questions. Follow the examples "
    "provided, reason step by step, then give your final answer in the format: "
    "The answer is (X)."
)


async def _call_openai_single(
    session: aiohttp.ClientSession,
    url: str,
    prompt: Dict,
    max_decoding_length: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    model: str = "default",
) -> Dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _MMLU_SYSTEM_PROMPT},
            {"role": "user", "content": prompt["prompt"]},
        ],
        "max_tokens": max_decoding_length,
        "temperature": temperature,
    }
    async with semaphore:
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "custom_id": prompt["custom_id"],
                    "text": text,
                    "completion_tokens": usage.get(
                        "completion_tokens", len(text.split())
                    ),
                    "error": None,
                }
        except Exception as e:
            logger.error("Prompt %s failed: %s", prompt["custom_id"], e)
            return {
                "custom_id": prompt["custom_id"],
                "text": "",
                "completion_tokens": 0,
                "error": str(e),
            }


async def call_openai_compat(
    base_url: str,
    prompts: List[Dict],
    max_decoding_length: int,
    temperature: float,
    concurrency: int,
    model: str = "default",
) -> List[Dict]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    results: List[Dict] = []
    completed = 0

    async with aiohttp.ClientSession() as session:
        tasks = []
        for p in prompts:
            tasks.append(
                _call_openai_single(
                    session,
                    url,
                    p,
                    max_decoding_length,
                    temperature,
                    semaphore,
                    model,
                )
            )

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 100 == 0:
                logger.info(
                    "Progress: %d / %d prompts completed",
                    completed,
                    len(prompts),
                )

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    prompts: List[Dict],
    results: List[Dict],
) -> Dict:
    result_map = {r["custom_id"]: r for r in results}

    per_category: Dict[str, Dict] = {}
    total_correct = 0
    total_count = 0
    num_no_think = 0
    token_counts: List[int] = []

    for p in prompts:
        cid = p["custom_id"]
        r = result_map.get(cid)
        if r is None or r.get("error"):
            cat = p["category"]
            per_category.setdefault(cat, {"correct": 0, "total": 0})
            per_category[cat]["total"] += 1
            total_count += 1
            continue

        predicted, think_found = extract_prediction(r["text"])
        is_correct = predicted == p["correct_answer"]
        if not think_found:
            num_no_think += 1

        r["predicted"] = predicted
        r["correct_answer"] = p["correct_answer"]
        r["is_correct"] = is_correct

        cat = p["category"]
        per_category.setdefault(cat, {"correct": 0, "total": 0})
        per_category[cat]["total"] += 1
        if is_correct:
            per_category[cat]["correct"] += 1
            total_correct += 1
        total_count += 1
        token_counts.append(r["completion_tokens"])

    for cat in per_category:
        t = per_category[cat]["total"]
        c = per_category[cat]["correct"]
        per_category[cat]["accuracy"] = round(c / t, 4) if t > 0 else 0.0

    overall_accuracy = (
        round(total_correct / total_count, 4) if total_count > 0 else 0.0
    )

    token_stats = {}
    if token_counts:
        sorted_tokens = sorted(token_counts)
        p95_idx = int(len(sorted_tokens) * 0.95)
        token_stats = {
            "mean": round(statistics.mean(token_counts), 1),
            "median": round(statistics.median(token_counts), 1),
            "p95": sorted_tokens[min(p95_idx, len(sorted_tokens) - 1)],
        }

    detail_results = []
    for p in prompts:
        cid = p["custom_id"]
        r = result_map.get(cid, {})
        detail_results.append(
            {
                "custom_id": cid,
                "category": p.get("category"),
                "predicted": r.get("predicted"),
                "correct_answer": p["correct_answer"],
                "is_correct": r.get("is_correct", False),
                "completion_tokens": r.get("completion_tokens", 0),
                "response_text": (r.get("text") or "")[:2000],
            }
        )

    return {
        "overall_accuracy": overall_accuracy,
        "per_category": dict(sorted(per_category.items())),
        "num_no_think": num_no_think,
        "token_stats": token_stats,
        "results": detail_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-framework MMLU-Pro accuracy test"
    )
    parser.add_argument("--base-url", required=True, help="Server base URL")
    parser.add_argument(
        "--framework",
        required=True,
        choices=["batchgen", "vllm", "sglang"],
        help="Inference framework to test",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=3000, help="Max test prompts"
    )
    parser.add_argument(
        "--max-decoding-length",
        type=int,
        default=8192,
        help="Max output tokens",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel requests (vllm/sglang)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model identifier for OpenAI-compatible APIs (vllm/sglang)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(
        "Starting MMLU-Pro eval: framework=%s, base_url=%s, max_prompts=%d",
        args.framework,
        args.base_url,
        args.max_prompts,
    )

    test_df, val_df = load_dataset(args.max_prompts)
    logger.info(
        "Loaded %d test, %d validation examples", len(test_df), len(val_df)
    )

    prompts = build_prompts(test_df, val_df)
    logger.info("Built %d prompts", len(prompts))

    t0 = time.time()
    if args.framework == "batchgen":
        results = call_batchgen(
            args.base_url, prompts, args.max_decoding_length, args.temperature
        )
    else:
        results = asyncio.run(
            call_openai_compat(
                args.base_url,
                prompts,
                args.max_decoding_length,
                args.temperature,
                args.concurrency,
                args.model,
            )
        )
    elapsed = time.time() - t0
    logger.info("Inference completed in %.1f seconds", elapsed)

    report = evaluate(prompts, results)
    report["framework"] = args.framework

    logger.info(
        "Overall accuracy: %.2f%% (%s)",
        report["overall_accuracy"] * 100,
        args.framework,
    )
    logger.info("No-think responses: %d", report["num_no_think"])
    if report["token_stats"]:
        logger.info(
            "Token stats — mean: %.1f, median: %.1f, p95: %d",
            report["token_stats"]["mean"],
            report["token_stats"]["median"],
            report["token_stats"]["p95"],
        )
    for cat, stats in report["per_category"].items():
        logger.info(
            "  %s: %.1f%% (%d/%d)",
            cat,
            stats["accuracy"] * 100,
            stats["correct"],
            stats["total"],
        )

    output_path = args.output or f"mmlu_pro_{args.framework}_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
