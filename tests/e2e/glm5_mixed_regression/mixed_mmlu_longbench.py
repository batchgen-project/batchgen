"""Mixed MMLU-Pro + LongBench single-batch regression for GLM-5-FP8.

Loads N MMLU-Pro prompts (with category-based few-shot prefix) and M LongBench
prompts (sampled across all tasks), shuffles them together with a deterministic
seed, and submits as one batch via `BatchGenHttpClient.submit_inference`. The
server is expected to be already running; the script is a pure client.

Scores the MMLU subset by extracting the answer letter from the model output
(stripping GLM-5 `<think>...</think>` first). LongBench is treated as a
completion-only check (no in-tree scorer for this corpus).

Exit code: 0 if every prompt returned non-empty text AND (if --mmlu-threshold
is > 0) MMLU accuracy meets the floor; 1 otherwise.

Used by `.github/workflows/pr-gpu-smoke.yml` (PR GPU regression CI).
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer

from batchgen.batchgen_client import BatchGenHttpClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MMLU_TEST_PARQUET = REPO_ROOT / "tests/e2e/r1_mmlu_pro_test/mmlu_pro_test.parquet"
MMLU_VAL_PARQUET = REPO_ROOT / "tests/e2e/r1_mmlu_pro_test/mmlu_pro_validation.parquet"
LONGBENCH_DIR = REPO_ROOT / "tests/e2e/r1_longbench_test/LongBench"

GLM5_SYSTEM = (
    "You are an expert at answering multiple-choice questions. "
    "Follow the examples provided, reason step by step, then give "
    "your final answer in the format: The answer is (X)."
)
LONGBENCH_SYSTEM = "You are a helpful assistant."

# Answer-letter patterns lifted from batchgen_benchmark/mmlu_pro_test.py
# (kept in sync intentionally — same extraction surface).
_ANSWER_PATTERNS = [
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


def _form_options(options: List[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "Options are:\n" + "".join(
        f"({l}): {o}\n" for o, l in zip(options, letters)
    )


def _build_few_shot_prefix(val_df: pd.DataFrame, category: str) -> str:
    """5-shot CoT examples for a category, from the validation set."""
    cat_examples = val_df[val_df["category"] == category].head(5)
    parts = []
    for _, row in cat_examples.iterrows():
        cot = row.get("cot_content", "")
        if isinstance(cot, str) and cot.startswith("A: "):
            cot = cot[3:]
        parts.append(
            f"Q: {row['question']}\n"
            + _form_options(row["options"])
            + f"{cot}\n"
            + f"The answer is ({row['answer']}).\n\n"
        )
    return "".join(parts)


def _extract_letter(text: str) -> Optional[str]:
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def _parse_glm5(text: str) -> Optional[str]:
    """Strip <think>...</think> if present, then extract answer letter."""
    search = text
    if "<think>" in text:
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            search = text[m.end():].strip()
    if not search:
        return _extract_letter(text)
    pred = _extract_letter(search)
    return pred if pred is not None else _extract_letter(text)


def build_mmlu_prompts(
    n: int,
    tokenizer,
    seed: int,
    enable_thinking: bool,
) -> List[Tuple[str, str]]:
    """Return list of (prompt_text, gold_letter) for n MMLU-Pro items."""
    test_df = pd.read_parquet(MMLU_TEST_PARQUET)
    val_df = pd.read_parquet(MMLU_VAL_PARQUET)
    test_df = test_df.sample(n=min(n, len(test_df)), random_state=seed).reset_index(drop=True)

    few_shot_cache = {}
    out: List[Tuple[str, str]] = []
    for _, row in test_df.iterrows():
        cat = row["category"]
        if cat not in few_shot_cache:
            few_shot_cache[cat] = _build_few_shot_prefix(val_df, cat)
        query = (
            f"{few_shot_cache[cat]}Q: {row['question']}\n"
            f"{_form_options(row['options'])}A:"
        )
        messages = [
            {"role": "system", "content": GLM5_SYSTEM},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        out.append((text, row["answer"]))
    return out


def build_longbench_prompts(n: int, tokenizer, seed: int, enable_thinking: bool) -> List[str]:
    """Return list of n chat-templated LongBench prompts, sampled across tasks."""
    if not LONGBENCH_DIR.exists():
        raise FileNotFoundError(f"LongBench dir missing: {LONGBENCH_DIR}")

    dfs = []
    for sub in sorted(LONGBENCH_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for pq in sub.glob("*.parquet"):
            try:
                dfs.append(pd.read_parquet(pq))
            except Exception as e:
                logger.warning(f"skipping {pq}: {e}")
    if not dfs:
        raise RuntimeError(f"No LongBench parquets loaded from {LONGBENCH_DIR}")

    combined = pd.concat(dfs, ignore_index=True)
    sampled = combined.sample(n=min(n, len(combined)), random_state=seed).reset_index(drop=True)

    out: List[str] = []
    for _, row in sampled.iterrows():
        query = f"{row['context']}\n\n{row['question']}"
        messages = [
            {"role": "system", "content": LONGBENCH_SYSTEM},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        out.append(text)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mixed MMLU-Pro + LongBench single-batch regression for GLM-5-FP8."
    )
    parser.add_argument("--base-url", default="http://localhost:10900")
    parser.add_argument("--cache-dir", required=True,
                        help="Local tokenizer / model cache (for chat template).")
    parser.add_argument("--mmlu-prompts", type=int, default=512)
    parser.add_argument("--longbench-prompts", type=int, default=512)
    parser.add_argument("--max-decoding-length", type=int, default=65536)
    parser.add_argument("--mmlu-threshold", type=float, default=0.0,
                        help="MMLU accuracy floor in percent; 0 disables.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=18000.0)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Pass enable_thinking=True to the chat template.")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info(f"Loading GLM-5 tokenizer from {args.cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.cache_dir, trust_remote_code=True, local_files_only=True,
    )

    logger.info(f"Building {args.mmlu_prompts} MMLU-Pro prompts")
    mmlu_items = build_mmlu_prompts(
        args.mmlu_prompts, tokenizer, args.seed, args.enable_thinking,
    )
    logger.info(f"Building {args.longbench_prompts} LongBench prompts")
    longbench = build_longbench_prompts(
        args.longbench_prompts, tokenizer, args.seed, args.enable_thinking,
    )

    pool = [{"prompt": p, "origin": "mmlu", "gold": g} for p, g in mmlu_items]
    pool += [{"prompt": p, "origin": "longbench", "gold": None} for p in longbench]
    random.shuffle(pool)

    n_mmlu = sum(1 for x in pool if x["origin"] == "mmlu")
    n_long = len(pool) - n_mmlu
    logger.info(f"Submitting {len(pool)} prompts ({n_mmlu} MMLU + {n_long} LongBench) to {args.base_url}")

    client = BatchGenHttpClient(args.base_url, timeout_s=args.timeout)
    if not client.health_check():
        logger.error("Server health check failed")
        return 1

    prompts = [x["prompt"] for x in pool]
    start = pd.Timestamp.now()
    results = client.submit_inference(
        prompts=prompts,
        max_output_len=args.max_decoding_length,
    )
    latency = (pd.Timestamp.now() - start).total_seconds()
    logger.info(f"Inference completed in {latency:.1f}s, {len(results)} responses")

    if len(results) != len(prompts):
        logger.error(f"Response count mismatch: got {len(results)}, expected {len(prompts)}")
        return 1

    n_empty = sum(1 for r in results if not r or not r.strip())
    if n_empty:
        logger.warning(f"{n_empty}/{len(results)} responses were empty")

    correct = 0
    mmlu_total = 0
    extract_fail = 0
    for item, result in zip(pool, results):
        if item["origin"] != "mmlu":
            continue
        mmlu_total += 1
        pred = _parse_glm5(result or "")
        if pred is None:
            extract_fail += 1
            continue
        if pred == item["gold"]:
            correct += 1

    acc_pct = (correct / mmlu_total * 100.0) if mmlu_total else 0.0
    print(f"MMLU:      {correct}/{mmlu_total} correct, {extract_fail} extraction failures, accuracy {acc_pct:.2f}%")
    print(f"LongBench: {n_long} completed (no scoring)")

    ok = n_empty == 0
    if args.mmlu_threshold > 0:
        if acc_pct < args.mmlu_threshold:
            logger.error(f"MMLU accuracy {acc_pct:.2f}% < threshold {args.mmlu_threshold:.2f}%")
            ok = False
        else:
            logger.info(f"MMLU accuracy {acc_pct:.2f}% >= threshold {args.mmlu_threshold:.2f}%")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
