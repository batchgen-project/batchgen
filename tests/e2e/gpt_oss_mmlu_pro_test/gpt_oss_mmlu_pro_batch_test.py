"""MMLU Pro test for GPT-OSS-120B using OpenAI-compatible Batch API.

This script demonstrates the full OpenAI Batch API workflow:
1. Read parquet dataset
2. Create JSONL input file in OpenAI batch format
3. Upload file via /v1/files API
4. Create batch via /v1/batches API
5. Poll for completion
6. Download results via /v1/files/{id}/content
"""

import argparse
import json
import logging
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from batchgen.batchgen_client import BatchGenHttpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_harmony_output(text: str) -> Tuple[str, str]:
    """Parse GPT-OSS Harmony format output.

    GPT-OSS models use the Harmony response format with special tokens to
    separate chain-of-thought reasoning (analysis channel) from user-facing
    output (final channel).

    Format: <|channel|>channel_name<|message|>content...

    Args:
        text: Raw model output (may contain Harmony special tokens)

    Returns:
        Tuple of (analysis_content, final_content)
    """
    # If no Harmony tokens, return text as final content (plain text output)
    if "<|channel|>" not in text:
        return "", text

    # Parse channel sections
    # Format: <|channel|>channel_name<|message|>content...
    analysis_parts = []
    final_parts = []

    # Split by <|channel|> and process each section
    # Match: <|channel|>channelname<|message|>content (until next <|channel|> or end tokens)
    pattern = r"<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|channel\|>|<\|end\|>|<\|return\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    for channel, content in matches:
        if channel == "analysis":
            analysis_parts.append(content.strip())
        elif channel == "final":
            final_parts.append(content.strip())

    # Handle malformed cases where <|message|> comes after multiple <|channel|> tags
    # e.g., <|channel|>analysis<|channel|>final<|message|>content
    if not matches:
        # Fallback: find the last <|message|> and extract content after it
        message_idx = text.rfind("<|message|>")
        if message_idx != -1:
            content = text[message_idx + len("<|message|>") :]
            # Clean up any trailing special tokens
            for token in ["<|end|>", "<|return|>", "<|channel|>"]:
                if token in content:
                    content = content[: content.find(token)]
            final_parts.append(content.strip())

    return "\n".join(analysis_parts), "\n".join(final_parts)


def form_options(options: List[str]) -> str:
    """Format multiple choice options."""
    option_str = "Options are:\n"
    opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, letter in zip(options, opts):
        option_str += f"({letter}): {opt}\n"
    return option_str


def extract_prediction(model_output: str) -> Optional[str]:
    """Extract the predicted letter answer from model output.

    Handles multiple output formats:
    - Harmony format (GPT-OSS): extracts from 'final' channel
    - Thinking format (K2.5, R1): extracts from post-</think> section
    - Plain text: extracts from full output

    Uses LAST match (not first) to prefer the final answer over
    intermediate reasoning in thinking models.

    Args:
        model_output: Raw model output string

    Returns:
        Extracted answer letter (A-J) or None if not found
    """
    # Step 1: Determine search text based on output format
    # Check Harmony format first (GPT-OSS)
    analysis_content, final_content = parse_harmony_output(model_output)
    if "<|channel|>" in model_output and final_content:
        search_text = final_content
    else:
        # Check thinking format (K2.5, R1): extract post-</think> section
        think_end = model_output.find("</think>")
        if think_end != -1:
            search_text = model_output[think_end + len("</think>"):]
        else:
            search_text = model_output

    # Answer extraction patterns (adapted from OpenAI's abcd_grader.py)
    # Extended to support A-J for MMLU-Pro's 10-choice questions
    patterns = [
        # "The answer is (A)" or "the answer is A" - matches system prompt format
        r"(?i)\b(?:the\s+)?answer\s+is\s*\(?([ABCDEFGHIJ])\)?",
        # "Answer: A" or "Answers: B" with optional markdown wrappers
        r"(?i)(?:\*{1,2}|_{1,2})?Answer[s]?\s*[:\-–]?(?:\*{1,2}|_{1,2})?\s*\(?([ABCDEFGHIJ])\)?",
        # "correct answer is (A)"
        r"(?i)correct answer is \(?([ABCDEFGHIJ])\)?",
        # "Option B" or "Choice: C"
        r"(?i)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCDEFGHIJ])\b",
        # LaTeX \boxed{...A...}
        r"\\boxed\{[^}]*?([ABCDEFGHIJ])[^}]*\}",
        # Standalone letter in parens/brackets: "(A)" or "[B]"
        r"(?<![A-Za-z0-9])[\(\[]\s*([ABCDEFGHIJ])\s*[\)\]](?![A-Za-z0-9])",
        # Markdown-wrapped: *A* or **B** or _C_ or __D__
        r"(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCDEFGHIJ])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])",
        # Letter at end of line (common pattern)
        r"(?:^|\s)\(?([ABCDEFGHIJ])\)?\s*$",
        # Letter followed by period or colon
        r"(?:^|\s)([ABCDEFGHIJ])[\.\:]",
        # Final fallback: bare letter on its own line
        r"^\s*([ABCDEFGHIJ])\s*$",
    ]

    # Use findall + last match to prefer final answer over intermediate reasoning
    for pattern in patterns:
        matches = re.findall(pattern, search_text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()

    return None


def create_batch_input_file(
    queries: List[str],
    model_name: str,
    max_tokens: int,
    output_path: Path,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    random_sampling_params: bool = False,
    random_max_completion_tokens: bool = False,
    min_completion_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
) -> Optional[List[int]]:
    """Create JSONL file in OpenAI batch format.

    Each line is a JSON object with:
    - custom_id: Unique identifier for the request
    - method: HTTP method (POST)
    - url: API endpoint (/v1/chat/completions)
    - body: Request body with model, messages, max_tokens/max_completion_tokens, reasoning_effort, sampling params

    Returns:
        List of per-request max_completion_tokens if random mode is enabled, None otherwise.
    """
    per_seq_limits: Optional[List[int]] = None
    if random_max_completion_tokens:
        lo = min_completion_tokens or 16
        hi = max_completion_tokens or max_tokens or 128
        per_seq_limits = [random.randint(lo, hi) for _ in range(len(queries))]

    with output_path.open("w", encoding="utf-8") as f:
        for idx, query in enumerate(queries):
            body = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a knowledge expert. Answer the multi-choice question and provide your final answer in the format 'The answer is (X)' where X is A, B, C, D, E, F, G, H, I, or J.",
                    },
                    {"role": "user", "content": query},
                ],
            }
            # Per-request max_completion_tokens or uniform max_tokens
            if per_seq_limits is not None:
                body["max_completion_tokens"] = per_seq_limits[idx]
            else:
                body["max_tokens"] = max_tokens

            # Add reasoning_effort for GPT-OSS models
            if reasoning_effort is not None:
                body["reasoning_effort"] = reasoning_effort
            # Per-request sampling params
            if random_sampling_params:
                body["temperature"] = random.choice([0.0, 0.3, 0.5, 0.7, 1.0])
                body["top_p"] = random.choice([0.5, 0.8, 0.9, 0.95, 1.0])
                body["top_k"] = random.choice([0, 10, 20, 40, 50])
            else:
                if temperature is not None:
                    body["temperature"] = temperature
                if top_p is not None:
                    body["top_p"] = top_p
                if top_k is not None:
                    body["top_k"] = top_k

            request = {
                "custom_id": f"mmlu-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    info_parts = [f"{len(queries)} requests"]
    if random_sampling_params:
        info_parts.append("random per-request sampling params")
    if per_seq_limits is not None:
        lo = min(per_seq_limits)
        hi = max(per_seq_limits)
        info_parts.append(f"random max_completion_tokens [{lo}, {hi}]")
    logger.info(f"Created batch input file with {', '.join(info_parts)}: {output_path}")
    if reasoning_effort:
        logger.info(f"Reasoning effort: {reasoning_effort}")
    return per_seq_limits


def parse_batch_results(content: bytes) -> List[Dict[str, Any]]:
    """Parse JSONL batch results file.

    Returns list of result dictionaries with custom_id and response content.
    """
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
    max_context_length: int = 131072,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run the full OpenAI Batch API workflow.

    Args:
        input_file_path: Path to JSONL input file
        output_file_path: Path to save output JSONL (optional)
        base_url: Server base URL
        poll_interval: Seconds between status checks
        timeout: Maximum seconds to wait
        max_context_length: Max total context (prompt + decode). Default 128K.
        temperature: Sampling temperature (None = greedy decoding)
        top_p: Nucleus sampling threshold (None = disabled)

    Returns:
        List of result dictionaries
    """
    client = BatchGenHttpClient(base_url, timeout_s=timeout)

    # Health check
    if not client.health_check():
        logger.warning("Server health check failed, proceeding anyway...")

    # Run the batch workflow
    logger.info("Starting batch workflow...")
    batch = client.submit_batch(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        endpoint="/v1/chat/completions",
        poll_interval=poll_interval,
        timeout=timeout,
        max_context_length=max_context_length,
        temperature=temperature,
        top_p=top_p,
    )

    # Download and parse results
    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise RuntimeError("Batch completed but no output_file_id returned")

    content = client.download_file_content(output_file_id)
    results = parse_batch_results(content)

    logger.info(f"Batch completed with {len(results)} results")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MMLU Pro test for GPT-OSS-120B using OpenAI-compatible Batch API"
    )
    parser.add_argument("--hugging_face_checkpoint", type=str, required=True)
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Max number of prompts to process. If not set, run the whole dataset.",
    )
    parser.add_argument(
        "--max_decoding_length", type=int, default=None,
        help="Max tokens to decode (required unless --random_max_completion_tokens is set)",
    )
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Server base URL (e.g., http://localhost:10900)",
    )
    parser.add_argument(
        "--server_host", type=str, default="localhost", help="Server hostname"
    )
    parser.add_argument("--server_port", type=int, default=10900, help="Server port")
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=5.0,
        help="Seconds between batch status checks",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="Maximum seconds to wait for batch"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=131072,
        help="Max total context length (prompt + decode). Default 128K.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: None = greedy decoding)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling threshold (default: None = disabled)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k filtering (default: None = disabled)",
    )
    parser.add_argument(
        "--random_sampling_params",
        action="store_true",
        help="Generate random per-request sampling params (temperature, top_p, top_k) for each request",
    )
    parser.add_argument(
        "--random_max_completion_tokens",
        action="store_true",
        help="Generate random per-request max_completion_tokens for each request",
    )
    parser.add_argument(
        "--min_completion_tokens",
        type=int,
        default=None,
        help="Lower bound for random max_completion_tokens (default: 16)",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
        help="Upper bound for random max_completion_tokens (default: --max_decoding_length)",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="GPT-OSS reasoning effort level (default: low per OpenAI standard)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for every sample (query, raw answer, parsed choice, ground truth)",
    )
    args = parser.parse_args()

    # Validate: max_decoding_length required unless random_max_completion_tokens is set
    if not args.random_max_completion_tokens and args.max_decoding_length is None:
        parser.error("--max_decoding_length is required unless --random_max_completion_tokens is set")

    # Construct base URL
    if args.base_url:
        base_url = args.base_url
    else:
        base_url = f"http://{args.server_host}:{args.server_port}"

    hugging_face_checkpoint = args.hugging_face_checkpoint
    benchmark_name = "TIGER-Lab/MMLU-Pro"

    # Load dataset - use the same parquet files from r1_mmlu_pro_test
    r1_test_dir = Path(__file__).parent.parent / "r1_mmlu_pro_test"
    logger.info(f"Loading dataset {benchmark_name}")
    dataset = pd.read_parquet(r1_test_dir / "mmlu_pro_test.parquet")
    if args.max_prompts is not None and args.max_prompts > 0:
        dataset = dataset.head(args.max_prompts)
        logger.info(f"Using top {args.max_prompts} sequences from dataset")

    # Load validation set for few-shot examples
    validation_set = pd.read_parquet(r1_test_dir / "mmlu_pro_validation.parquet")
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
    # Build few-shot examples (limit to 5 per category, append answer format)
    prompts = {c: "" for c in categories}
    example_counts = {c: 0 for c in categories}
    MAX_EXAMPLES = 5

    for _, row in validation_set.iterrows():
        cat = row["category"]
        if example_counts[cat] < MAX_EXAMPLES:
            # Format: Q + Options + CoT + Final Answer (demonstrating expected format)
            answer_letter = row["answer"]  # Ground truth letter
            prompts[cat] += (
                "Q: " + row["question"] + "\n"
                + form_options(row["options"])
                + row["cot_content"] + "\n"
                + f"The answer is ({answer_letter}).\n\n"
            )
            example_counts[cat] += 1

    # Build queries (raw prompts without chat template - batch API applies it)
    queries: List[str] = []
    for _, entry in dataset.iterrows():
        prefix = prompts[entry["category"]]
        prompt = (
            "The following are example questions with step-by-step solutions:\n\n"
            + prefix
            + "Now answer the following question:\n"
            + "Q: " + entry["question"] + "\n"
            + form_options(entry["options"])
        )
        queries.append(prompt)

    logger.info(f"Loaded {len(queries)} samples from the dataset.")

    # Create temp file for batch input (will be uploaded to server)
    temp_dir = Path(tempfile.gettempdir())
    input_file = temp_dir / "gpt_oss_mmlu_pro_batch_input.jsonl"

    # Create batch input file (with per-request sampling params in JSONL body)
    per_seq_limits = create_batch_input_file(
        queries=queries,
        model_name=hugging_face_checkpoint,
        max_tokens=args.max_decoding_length or 128,
        output_path=input_file,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        random_sampling_params=args.random_sampling_params,
        random_max_completion_tokens=args.random_max_completion_tokens,
        min_completion_tokens=args.min_completion_tokens,
        max_completion_tokens=args.max_completion_tokens,
    )

    # Run batch workflow
    # Sampling params are now per-request in the JSONL body.
    # Batch-level params serve as defaults only when per-request values are None.
    logger.info(f"Connecting to server at {base_url}")
    if args.random_max_completion_tokens:
        logger.info("Using random per-request max_completion_tokens (set in JSONL body)")
    if args.random_sampling_params:
        logger.info("Using random per-request sampling params (set in JSONL body)")
    elif args.temperature is not None or args.top_p is not None or args.top_k is not None:
        logger.info(f"Sampling params: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    results = run_batch_workflow(
        input_file_path=str(input_file),
        output_file_path=None,  # Results downloaded from server
        base_url=base_url,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        max_context_length=args.max_context_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Sort results by custom_id (mmlu-{idx}) to match original dataset order
    results.sort(key=lambda x: int(x.get("custom_id", "mmlu-0").split("-")[1]))

    # Build custom_id → dataset index map for correct scoring
    # custom_id is "mmlu-{idx}" where idx is the row index in the dataset
    result_dataset_indices = []
    for result in results:
        custom_id = result.get("custom_id", "mmlu-0")
        dataset_idx = int(custom_id.split("-")[1])
        result_dataset_indices.append(dataset_idx)

    # Extract answers from results
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

    # Verify per-sequence max_completion_tokens compliance
    if per_seq_limits is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_checkpoint)
        print("\n--- Per-Sequence max_completion_tokens Verification ---")
        violations = 0
        for idx, result in enumerate(results):
            response = result.get("response", {})
            body = response.get("body", {})
            usage = body.get("usage", {})
            server_completion_tokens = usage.get("completion_tokens", -1)

            choices = body.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            retokenized_count = len(tokenizer.encode(content, add_special_tokens=False)) if content else 0

            limit = per_seq_limits[idx]
            server_ok = server_completion_tokens <= limit if server_completion_tokens >= 0 else True
            retok_ok = retokenized_count <= limit

            if not server_ok or not retok_ok:
                violations += 1
                print(
                    f"  VIOLATION seq {idx}: limit={limit}, "
                    f"server={server_completion_tokens}, retokenized={retokenized_count}"
                )

        if violations == 0:
            print(f"  ALL {len(results)} sequences respect their max_completion_tokens limits")
        else:
            print(f"  {violations}/{len(results)} sequences VIOLATED their limits")
        print("-" * 50)

    # Print results with Harmony format parsing info
    # Use dataset index from custom_id for ground truth lookup
    all_ground_truths = dataset["answer"].tolist()
    if args.verbose:
        # Verbose mode: print all samples with extracted vs ground truth
        for idx in range(len(answer_set)):
            ds_idx = result_dataset_indices[idx]
            gt = all_ground_truths[ds_idx]
            print("=" * 70)
            print(f"[Sample mmlu-{ds_idx}]")
            print(f"Query: {queries[idx][:500]}...")
            print(f"\nRaw Answer: {answer_set[idx][:1000]}")
            # Show parsed Harmony channels
            analysis, final = parse_harmony_output(answer_set[idx])
            if analysis or "<|channel|>" in answer_set[idx]:
                print(f"\n--- Harmony Format Parsing ---")
                print(f"Analysis channel: {analysis[:300]}..." if analysis else "Analysis: (none)")
                print(f"Final channel: {final[:300]}..." if final else "Final: (none)")
            # Show parsed choice vs ground truth
            extracted = extract_prediction(answer_set[idx])
            print(f"\n--- Extracted vs Ground Truth ---")
            print(f"Extracted Choice: {extracted if extracted else '(FAILED)'}")
            print(f"Ground Truth: {gt}")
            print(f"Result: {'CORRECT' if extracted == gt else 'WRONG'}")
            print("=" * 70 + "\n")
    else:
        # Print first 5 samples for brevity (default behavior)
        for idx in range(min(5, len(answer_set))):
            print("==================================================================")
            print(f"Query {idx}: {queries[idx][:500]}...")
            print("\n")
            print(f"Raw Answer {idx}: {answer_set[idx][:1000]}")
            # Show parsed Harmony channels
            analysis, final = parse_harmony_output(answer_set[idx])
            if analysis or "<|channel|>" in answer_set[idx]:
                print(f"\n--- Harmony Format Parsing ---")
                print(f"Analysis channel: {analysis[:300]}..." if analysis else "Analysis: (none)")
                print(f"Final channel: {final[:300]}..." if final else "Final: (none)")
            print("==================================================================")

    # Evaluate accuracy — match by custom_id (mmlu-{dataset_idx}), not position
    success = 0
    extraction_failures = 0
    predictions: List[str] = []
    total_samples = len(answer_set)
    incorrect_samples: List[Dict[str, Any]] = []

    for i in range(total_samples):
        ds_idx = result_dataset_indices[i]
        gt = all_ground_truths[ds_idx]
        model_output = answer_set[i]
        extracted_answer = extract_prediction(model_output)
        if extracted_answer:
            prediction = extracted_answer
        else:
            extraction_failures += 1
            prediction = "Z"
        predictions.append(prediction)
        if prediction == gt:
            success += 1
        else:
            incorrect_samples.append(
                {
                    "id": ds_idx,
                    "extracted": prediction,
                    "ground_truth": gt,
                    "extraction_failed": extracted_answer is None,
                }
            )

    accuracy = success / total_samples if total_samples > 0 else 0
    print("\n--- Evaluation Summary ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct: {success}")
    print(f"Incorrect: {total_samples - success}")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 30)
    print(
        f"Extraction Failures: {extraction_failures} ({extraction_failures / total_samples:.2%})"
    )

    # Print all incorrect samples
    print("\n--- Incorrect Answers Summary ---")
    print(f"Total Wrong: {len(incorrect_samples)}")
    print("-" * 50)
    for sample in incorrect_samples:
        status = "(extraction failed)" if sample["extraction_failed"] else ""
        print(f"Q{sample['id']:4d}: Chose {sample['extracted']}, Correct {sample['ground_truth']} {status}")
    print("-" * 50)
