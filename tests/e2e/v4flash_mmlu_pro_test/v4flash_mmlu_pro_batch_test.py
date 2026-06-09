"""MMLU-Pro accuracy test for DeepSeek-V4-Flash via the OpenAI Batch API.

Adapted from test/glm5_mmlu_pro_test/glm5_mmlu_pro_batch_test.py (the multi-model
reference). Same 5-shot Batch-API workflow and <think>-aware answer extraction;
V4-Flash is a DeepSeek model so the GLM-5 thinking parser applies directly. The
GLM-specific enable_thinking body field is omitted.
"""

import argparse
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from batchgen.batchgen_client import BatchGenHttpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_think_output(text: str) -> Tuple[str, str]:
    if "<think>" not in text:
        return "", text
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end() :].strip()
    start = text.find("<think>")
    return text[start + len("<think>") :].strip(), ""


def form_options(options: List[str]) -> str:
    option_str = "Options are:\n"
    opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, letter in zip(options, opts):
        option_str += f"({letter}): {opt}\n"
    return option_str


def extract_prediction(model_output: str) -> Optional[str]:
    _, answer_content = parse_think_output(model_output)
    search_text = answer_content if answer_content else model_output
    patterns = [
        r"(?i)\b(?:the\s+)?answer\s+is\s*\(?([ABCDEFGHIJ])\)?",
        r"(?i)(?:\*{1,2}|_{1,2})?Answer[s]?\s*[:\-–]?(?:\*{1,2}|_{1,2})?\s*\(?([ABCDEFGHIJ])\)?",
        r"(?i)correct answer is \(?([ABCDEFGHIJ])\)?",
        r"(?:^|\s)([ABCDEFGHIJ])[\.\:]",
        r"^\s*([ABCDEFGHIJ])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def create_batch_input_file(
    queries: List[str], model_name: str, max_tokens: int, output_path: Path
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for idx, query in enumerate(queries):
            body = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at answering multiple-choice questions. Follow the examples provided, reason step by step, then give your final answer in the format: The answer is (X).",
                    },
                    {"role": "user", "content": query},
                ],
                "max_tokens": max_tokens,
            }
            f.write(
                json.dumps(
                    {
                        "custom_id": f"mmlu-{idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info(
        f"Created batch input with {len(queries)} requests: {output_path}"
    )


def parse_batch_results(content: bytes) -> List[Dict[str, Any]]:
    results = []
    for line in content.decode("utf-8").strip().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse result line: {e}")
    return results


def run_batch_workflow(
    input_file_path: str,
    base_url: str,
    poll_interval: float,
    timeout: Optional[float],
    temperature: Optional[float],
    top_p: Optional[float],
) -> List[Dict[str, Any]]:
    client = BatchGenHttpClient(base_url, timeout_s=timeout)
    if not client.health_check():
        logger.warning("Server health check failed, proceeding anyway...")
    batch = client.submit_batch(
        input_file_path=input_file_path,
        output_file_path=None,
        endpoint="/v1/chat/completions",
        poll_interval=poll_interval,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
    )
    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch completed but no output_file_id returned")
    return parse_batch_results(client.download_file_content(output_file_id))


def main():
    parser = argparse.ArgumentParser(
        description="MMLU-Pro Batch-API accuracy test for DeepSeek-V4-Flash"
    )
    parser.add_argument("--hugging_face_checkpoint", type=str, required=True)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_decoding_length", type=int, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--poll_interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--no_few_shot", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    r1_test_dir = Path(__file__).parent.parent / "r1_mmlu_pro_test"
    dataset = pd.read_parquet(r1_test_dir / "mmlu_pro_test.parquet")
    if args.max_prompts and args.max_prompts > 0:
        dataset = dataset.head(args.max_prompts)

    categories = [
        "computer science",
        "math",
        "chemistry",
        "engineering",
        "law",
        "biology",
        "health",
        "physics",
        "business",
        "philosophy",
        "economics",
        "other",
        "psychology",
        "history",
    ]
    prompts = {c: "" for c in categories}
    if not args.no_few_shot:
        val = pd.read_parquet(r1_test_dir / "mmlu_pro_validation.parquet")
        counts = {c: 0 for c in categories}
        for _, row in val.iterrows():
            cat = row["category"]
            if counts[cat] < 5:
                cot = row["cot_content"].strip()
                if cot.startswith("A:"):
                    cot = cot[2:].strip()
                prompts[cat] += (
                    f"Q: {row['question']}\n"
                    + form_options(row["options"])
                    + f"A: {cot}\n"
                    + f"The answer is ({row['answer']}).\n\n"
                )
                counts[cat] += 1

    queries: List[str] = []
    for _, entry in dataset.iterrows():
        queries.append(
            prompts[entry["category"]]
            + f"Q: {entry['question']}\n"
            + form_options(entry["options"])
            + "A:"
        )
    logger.info(f"Loaded {len(queries)} MMLU-Pro samples")

    input_file = (
        Path(tempfile.gettempdir()) / "v4flash_mmlu_pro_batch_input.jsonl"
    )
    create_batch_input_file(
        queries,
        args.hugging_face_checkpoint,
        args.max_decoding_length,
        input_file,
    )

    results = run_batch_workflow(
        str(input_file),
        args.base_url,
        args.poll_interval,
        args.timeout,
        args.temperature,
        args.top_p,
    )
    results.sort(key=lambda x: int(x.get("custom_id", "mmlu-0").split("-")[1]))

    answer_set: List[str] = []
    for result in results:
        choices = result.get("response", {}).get("body", {}).get("choices", [])
        answer_set.append(
            choices[0].get("message", {}).get("content", "") if choices else ""
        )

    ground_truths = dataset["answer"].tolist()
    success = 0
    extraction_failures = 0
    incorrect: List[Dict[str, Any]] = []
    for i in range(len(answer_set)):
        extracted = extract_prediction(answer_set[i])
        prediction = extracted if extracted else "Z"
        if extracted is None:
            extraction_failures += 1
        if prediction == ground_truths[i]:
            success += 1
        else:
            incorrect.append(
                {
                    "id": i,
                    "extracted": prediction,
                    "gt": ground_truths[i],
                    "extraction_failed": extracted is None,
                }
            )

    total = len(answer_set)
    accuracy = success / total if total else 0.0
    print("\n--- MMLU-Pro Evaluation (DeepSeek-V4-Flash) ---")
    print(f"Total: {total}  Correct: {success}  Accuracy: {accuracy:.2%}")
    print(
        f"Extraction failures: {extraction_failures} ({extraction_failures / total:.2%})"
        if total
        else ""
    )
    for s in incorrect[:20]:
        tag = "(extract failed)" if s["extraction_failed"] else ""
        print(
            f"  Q{s['id']:4d}: chose {s['extracted']}, correct {s['gt']} {tag}"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "total": total,
                    "correct": success,
                    "accuracy": accuracy,
                    "extraction_failures": extraction_failures,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
