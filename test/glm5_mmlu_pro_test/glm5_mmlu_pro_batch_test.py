"""MMLU Pro test for GLM-5 using OpenAI-compatible Batch API.

Same workflow as gpt_oss_mmlu_pro_batch_test.py but adapted for GLM-5:
- Uses enable_thinking field instead of reasoning_effort
- Parses <think>...</think> tags instead of Harmony format
- No reasoning_effort parameter
"""

import argparse
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer

from batchgen.batchgen_client import BatchGenHttpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_glm5_output(text: str) -> Tuple[str, str]:
    """Parse GLM-5 output with optional <think>...</think> tags.

    Returns:
        Tuple of (thinking_content, answer_content)
    """
    if "<think>" not in text:
        return "", text

    # Extract thinking content
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        # Everything after </think> is the answer
        answer = text[think_match.end():].strip()
        return thinking, answer

    # Malformed: <think> without </think> — treat entire text as thinking
    think_start = text.find("<think>")
    return text[think_start + len("<think>"):].strip(), ""


def form_options(options: List[str]) -> str:
    """Format multiple choice options."""
    option_str = "Options are:\n"
    opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, letter in zip(options, opts):
        option_str += f"({letter}): {opt}\n"
    return option_str


def extract_prediction(model_output: str) -> Optional[str]:
    """Extract the predicted letter answer from model output.

    For GLM-5:
    1. Parse <think>...</think> to get answer portion
    2. Apply answer extraction patterns
    """
    _, answer_content = parse_glm5_output(model_output)
    search_text = answer_content if answer_content else model_output

    patterns = [
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

    for pattern in patterns:
        match = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    return None


def create_batch_input_file(
    queries: List[str],
    model_name: str,
    max_tokens: int,
    output_path: Path,
    enable_thinking: Optional[bool] = None,
) -> None:
    """Create JSONL file in OpenAI batch format for GLM-5."""
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
            if enable_thinking is not None:
                body["enable_thinking"] = enable_thinking
            request = {
                "custom_id": f"mmlu-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    logger.info(f"Created batch input file with {len(queries)} requests: {output_path}")
    if enable_thinking is not None:
        logger.info(f"enable_thinking: {enable_thinking}")


def parse_batch_results(content: bytes) -> List[Dict[str, Any]]:
    """Parse JSONL batch results file."""
    results = []
    for line in content.decode("utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            results.append(item)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse result line: {e}")
    return results


def run_batch_workflow(
    input_file_path: str,
    output_file_path: Optional[str],
    base_url: str,
    poll_interval: float = 5.0,
    timeout: Optional[float] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run the full OpenAI Batch API workflow."""
    client = BatchGenHttpClient(base_url, timeout_s=timeout)

    if not client.health_check():
        logger.warning("Server health check failed, proceeding anyway...")

    logger.info("Starting batch workflow...")
    batch = client.submit_batch(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        endpoint="/v1/chat/completions",
        poll_interval=poll_interval,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
    )

    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch completed but no output_file_id returned")

    content = client.download_file_content(output_file_id)
    results = parse_batch_results(content)

    logger.info(f"Batch completed with {len(results)} results")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MMLU Pro test for GLM-5 using OpenAI-compatible Batch API"
    )
    parser.add_argument("--hugging_face_checkpoint", type=str, required=True)
    parser.add_argument(
        "--max_prompts", type=int, default=None,
        help="Max number of prompts to process. If not set, run the whole dataset.",
    )
    parser.add_argument(
        "--max_decoding_length", type=int, required=True, help="Max tokens to decode"
    )
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="Server base URL (e.g., http://localhost:10900)",
    )
    parser.add_argument(
        "--server_host", type=str, default="localhost", help="Server hostname"
    )
    parser.add_argument("--server_port", type=int, default=10900, help="Server port")
    parser.add_argument(
        "--poll_interval", type=float, default=5.0,
        help="Seconds between batch status checks",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="Maximum seconds to wait for batch"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: None = greedy decoding)",
    )
    parser.add_argument(
        "--top_p", type=float, default=None,
        help="Nucleus sampling threshold (default: None = disabled)",
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", default=False,
        help="Enable GLM-5 thinking/reasoning mode (<think>...</think>)",
    )
    parser.add_argument(
        "--no_thinking", action="store_true", default=False,
        help="Explicitly disable thinking mode (send enable_thinking=false)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output for every sample",
    )
    args = parser.parse_args()

    # Construct base URL
    if args.base_url:
        base_url = args.base_url
    else:
        base_url = f"http://{args.server_host}:{args.server_port}"

    hugging_face_checkpoint = args.hugging_face_checkpoint
    benchmark_name = "TIGER-Lab/MMLU-Pro"

    # Load dataset
    r1_test_dir = Path(__file__).parent.parent / "r1_mmlu_pro_test"
    logger.info(f"Loading dataset {benchmark_name}")
    dataset = pd.read_parquet(r1_test_dir / "mmlu_pro_test.parquet")
    if args.max_prompts is not None and args.max_prompts > 0:
        dataset = dataset.head(args.max_prompts)
        logger.info(f"Using top {args.max_prompts} sequences from dataset")

    # Load validation set for few-shot examples
    validation_set = pd.read_parquet(r1_test_dir / "mmlu_pro_validation.parquet")
    categories = [
        "computer science", "math", "chemistry", "engineering", "law",
        "biology", "health", "physics", "business", "philosophy",
        "economics", "other", "psychology", "history",
    ]
    prompts = {c: "" for c in categories}
    example_counts = {c: 0 for c in categories}
    MAX_EXAMPLES = 5

    for _, row in validation_set.iterrows():
        cat = row["category"]
        if example_counts[cat] < MAX_EXAMPLES:
            answer_letter = row["answer"]
            cot = row["cot_content"].strip()
            prompts[cat] += (
                f"Q: {row['question']}\n"
                + form_options(row["options"])
                + f"A: {cot}\n"
                + f"The answer is ({answer_letter}).\n\n"
            )
            example_counts[cat] += 1

    # Build queries
    queries: List[str] = []
    for _, entry in dataset.iterrows():
        prefix = prompts[entry["category"]]
        prompt = (
            prefix
            + f"Q: {entry['question']}\n"
            + form_options(entry["options"])
            + "A:"
        )
        queries.append(prompt)

    logger.info(f"Loaded {len(queries)} samples from the dataset.")

    # Determine enable_thinking setting
    enable_thinking = None
    if args.enable_thinking:
        enable_thinking = True
    elif args.no_thinking:
        enable_thinking = False

    # Create temp file for batch input
    temp_dir = Path(tempfile.gettempdir())
    input_file = temp_dir / "glm5_mmlu_pro_batch_input.jsonl"

    create_batch_input_file(
        queries=queries,
        model_name=hugging_face_checkpoint,
        max_tokens=args.max_decoding_length,
        output_path=input_file,
        enable_thinking=enable_thinking,
    )

    # Run batch workflow
    logger.info(f"Connecting to server at {base_url}")
    if args.temperature is not None or args.top_p is not None:
        logger.info(f"Sampling params: temperature={args.temperature}, top_p={args.top_p}")
    results = run_batch_workflow(
        input_file_path=str(input_file),
        output_file_path=None,
        base_url=base_url,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Sort results by custom_id
    results.sort(key=lambda x: int(x.get("custom_id", "mmlu-0").split("-")[1]))

    # Extract answers
    answer_set: List[str] = []
    for result in results:
        response = result.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            answer_set.append(content)
        else:
            answer_set.append("")

    # Load tokenizer for re-encoding (to show raw token IDs)
    logger.info(f"Loading tokenizer from {hugging_face_checkpoint} for token-level debug...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            hugging_face_checkpoint,
            cache_dir=args.hf_cache_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}. Token IDs will not be shown.")
        tokenizer = None

    # Print results
    ground_truths = dataset["answer"].tolist()
    if args.verbose:
        for idx in range(len(answer_set)):
            print("=" * 70)
            print(f"[Sample {idx}]")
            print(f"Query:\n{queries[idx]}")
            print(f"\n--- De-tokenized Output (full) ---")
            print(answer_set[idx])
            if tokenizer and answer_set[idx]:
                token_ids = tokenizer.encode(answer_set[idx], add_special_tokens=False)
                print(f"\n--- Re-encoded Token IDs ({len(token_ids)} tokens) ---")
                print(token_ids[:200])
                if len(token_ids) > 200:
                    print(f"  ... ({len(token_ids) - 200} more tokens)")
                # Show per-token decode for first 50 tokens
                print(f"\n--- Per-token Decode (first 50) ---")
                for i, tid in enumerate(token_ids[:50]):
                    decoded = tokenizer.decode([tid])
                    print(f"  [{i:3d}] id={tid:6d} -> {repr(decoded)}")
            thinking, answer = parse_glm5_output(answer_set[idx])
            if thinking:
                print(f"\n--- GLM-5 Thinking ---")
                print(f"Thinking: {thinking[:500]}..." if len(thinking) > 500 else f"Thinking: {thinking}")
                print(f"Answer: {answer[:500]}..." if len(answer) > 500 else f"Answer: {answer}")
            extracted = extract_prediction(answer_set[idx])
            print(f"\n--- Extracted vs Ground Truth ---")
            print(f"Extracted Choice: {extracted if extracted else '(FAILED)'}")
            print(f"Ground Truth: {ground_truths[idx]}")
            print(f"Result: {'CORRECT' if extracted == ground_truths[idx] else 'WRONG'}")
            print("=" * 70 + "\n")
    else:
        for idx in range(min(5, len(answer_set))):
            print("=" * 70)
            print(f"Query {idx}:\n{queries[idx]}")
            print(f"\n--- De-tokenized Output (full) ---")
            print(answer_set[idx])
            if tokenizer and answer_set[idx]:
                token_ids = tokenizer.encode(answer_set[idx], add_special_tokens=False)
                print(f"\n--- Re-encoded Token IDs ({len(token_ids)} tokens) ---")
                print(token_ids[:200])
                # Per-token decode for first 30 tokens
                print(f"\n--- Per-token Decode (first 30) ---")
                for i, tid in enumerate(token_ids[:30]):
                    decoded = tokenizer.decode([tid])
                    print(f"  [{i:3d}] id={tid:6d} -> {repr(decoded)}")
            thinking, answer = parse_glm5_output(answer_set[idx])
            if thinking:
                print(f"\n--- GLM-5 Thinking ---")
                print(f"Thinking: {thinking[:300]}...")
                print(f"Answer: {answer[:300]}...")
            print("=" * 70)

    # Evaluate accuracy
    success = 0
    extraction_failures = 0
    predictions: List[str] = []
    total_samples = len(answer_set)
    incorrect_samples: List[Dict[str, Any]] = []

    for i in range(total_samples):
        model_output = answer_set[i]
        extracted_answer = extract_prediction(model_output)
        if extracted_answer:
            prediction = extracted_answer
        else:
            extraction_failures += 1
            prediction = "Z"
        predictions.append(prediction)
        if prediction == ground_truths[i]:
            success += 1
        else:
            incorrect_samples.append({
                "id": i,
                "extracted": prediction,
                "ground_truth": ground_truths[i],
                "extraction_failed": extracted_answer is None,
            })

    accuracy = success / total_samples if total_samples > 0 else 0
    print("\n--- Evaluation Summary ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct: {success}")
    print(f"Incorrect: {total_samples - success}")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 30)
    print(f"Extraction Failures: {extraction_failures} ({extraction_failures / total_samples:.2%})")

    print("\n--- Incorrect Answers Summary ---")
    print(f"Total Wrong: {len(incorrect_samples)}")
    print("-" * 50)
    for sample in incorrect_samples:
        status = "(extraction failed)" if sample["extraction_failed"] else ""
        print(f"Q{sample['id']:4d}: Chose {sample['extracted']}, Correct {sample['ground_truth']} {status}")
    print("-" * 50)
