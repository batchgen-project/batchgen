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
import re
import tempfile
import time
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

    For GPT-OSS models using Harmony format:
    1. Parse the output to extract 'final' channel content
    2. Apply answer extraction patterns to final content only
    3. Fall back to full output if parsing fails

    Args:
        model_output: Raw model output string (may contain Harmony tokens)

    Returns:
        Extracted answer letter (A-J) or None if not found
    """
    # Parse Harmony format to get final channel content
    analysis_content, final_content = parse_harmony_output(model_output)

    # Prefer final channel content, fall back to full output
    search_text = final_content if final_content else model_output

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
    reasoning_effort: Optional[str] = None,
) -> None:
    """Create JSONL file in OpenAI batch format.

    Each line is a JSON object with:
    - custom_id: Unique identifier for the request
    - method: HTTP method (POST)
    - url: API endpoint (/v1/chat/completions)
    - body: Request body with model, messages, max_tokens, reasoning_effort
    """
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
                "max_tokens": max_tokens,
            }
            # Add reasoning_effort for GPT-OSS models
            if reasoning_effort is not None:
                body["reasoning_effort"] = reasoning_effort
            request = {
                "custom_id": f"mmlu-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    logger.info(f"Created batch input file with {len(queries)} requests: {output_path}")
    if reasoning_effort:
        logger.info(f"Reasoning effort: {reasoning_effort}")


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
        "--max_decoding_length", type=int, required=True, help="Max tokens to decode"
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

    # Create batch input file
    create_batch_input_file(
        queries=queries,
        model_name=hugging_face_checkpoint,
        max_tokens=args.max_decoding_length,
        output_path=input_file,
        reasoning_effort=args.reasoning_effort,
    )

    # Run batch workflow
    logger.info(f"Connecting to server at {base_url}")
    if args.temperature is not None or args.top_p is not None:
        logger.info(f"Sampling params: temperature={args.temperature}, top_p={args.top_p}")
    results = run_batch_workflow(
        input_file_path=str(input_file),
        output_file_path=None,  # Results downloaded from server
        base_url=base_url,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Save raw batch results to JSONL for post-analysis
    results_dir = Path("./gpt-oss-120b-bench")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with results_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Batch results saved to {results_file}")

    # Sort results by custom_id to match original order
    results.sort(key=lambda x: int(x.get("custom_id", "mmlu-0").split("-")[1]))

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

    # Print results with Harmony format parsing info
    ground_truths = dataset["answer"].tolist()
    if args.verbose:
        # Verbose mode: print all samples with extracted vs ground truth
        for idx in range(len(answer_set)):
            print("=" * 70)
            print(f"[Sample {idx}]")
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
            print(f"Ground Truth: {ground_truths[idx]}")
            print(f"Result: {'CORRECT' if extracted == ground_truths[idx] else 'WRONG'}")
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
            incorrect_samples.append(
                {
                    "id": i,
                    "extracted": prediction,
                    "ground_truth": ground_truths[i],
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
