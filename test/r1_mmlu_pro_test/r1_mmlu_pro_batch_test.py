"""MMLU Pro test using OpenAI-compatible Batch API.

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer

from batchgen.batchgen_client import BatchGenHttpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def form_options(options: List[str]) -> str:
    """Format multiple choice options."""
    option_str = "Options are:\n"
    opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, letter in zip(options, opts):
        option_str += f"({letter}): {opt}\n"
    return option_str


def extract_prediction(model_output: str) -> Tuple[Optional[str], bool]:
    """Extract the predicted letter answer from a model output string."""
    think_end_pos = model_output.find("</think>")
    think_tag_found = think_end_pos != -1
    search_text = (
        model_output[think_end_pos + len("</think>") :]
        if think_tag_found
        else model_output
    )
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, search_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), think_tag_found
    return None, think_tag_found


def create_batch_input_file(
    queries: List[str],
    model_name: str,
    max_tokens: int,
    output_path: Path,
) -> None:
    """Create JSONL file in OpenAI batch format.

    Each line is a JSON object with:
    - custom_id: Unique identifier for the request
    - method: HTTP method (POST)
    - url: API endpoint (/v1/chat/completions)
    - body: Request body with model, messages, max_tokens
    """
    with output_path.open("w", encoding="utf-8") as f:
        for idx, query in enumerate(queries):
            request = {
                "custom_id": f"mmlu-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`. Please follow the following examples and strictly give the answer with format 'the answer is (A/B/C/D/E/F/G/H/I/J)'.",
                        },
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": max_tokens,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    logger.info(f"Created batch input file with {len(queries)} requests: {output_path}")


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
        description="MMLU Pro test using OpenAI-compatible Batch API"
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
    args = parser.parse_args()

    # Construct base URL
    if args.base_url:
        base_url = args.base_url
    else:
        base_url = f"http://{args.server_host}:{args.server_port}"

    hugging_face_checkpoint = args.hugging_face_checkpoint
    benchmark_name = "TIGER-Lab/MMLU-Pro"

    # Load dataset
    logger.info(f"Loading dataset {benchmark_name}")
    dataset = pd.read_parquet(
        os.path.join(os.path.dirname(__file__), "mmlu_pro_test.parquet")
    )
    if args.max_prompts is not None and args.max_prompts > 0:
        dataset = dataset.head(args.max_prompts)
        logger.info(f"Using top {args.max_prompts} sequences from dataset")

    # Load validation set for few-shot examples
    validation_set = pd.read_parquet(
        os.path.join(os.path.dirname(__file__), "mmlu_pro_validation.parquet")
    )
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
    for _, row in validation_set.iterrows():
        prompts[row["category"]] += (
            "Q:"
            + " "
            + row["question"]
            + "\n"
            + form_options(row["options"])
            + "\n"
            + row["cot_content"]
            + "\n\n"
        )

    # Build queries (raw prompts without chat template - batch API applies it)
    queries: List[str] = []
    for _, entry in dataset.iterrows():
        prefix = prompts[entry["category"]]
        prompt = (
            "Please read the following 5 examples: \n"
            + prefix
            + "Please answer the following question: \n"
            + "Q: "
            + entry["question"]
            + "\n"
            + form_options(entry["options"])
            + "\n"
        )
        queries.append(prompt)

    logger.info(f"Loaded {len(queries)} samples from the dataset.")

    # Create temp file for batch input (will be uploaded to server)
    temp_dir = Path(tempfile.gettempdir())
    input_file = temp_dir / "mmlu_pro_batch_input.jsonl"

    # Create batch input file
    create_batch_input_file(
        queries=queries,
        model_name=hugging_face_checkpoint,
        max_tokens=args.max_decoding_length,
        output_path=input_file,
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

    # Print results
    print_result = True
    if print_result:
        for idx in range(min(5, len(answer_set))):  # Print first 5 for brevity
            print("==================================================================")
            print(f"Query {idx}: {queries[idx][:500]}...")
            print("\n")
            print(f"Answer {idx}: {answer_set[idx][:1000]}")
            print("==================================================================\n")

    # Evaluate accuracy
    success = 0
    extraction_failures = 0
    no_think_tag_count = 0
    predictions: List[str] = []
    ground_truths = dataset["answer"].tolist()
    total_samples = len(answer_set)
    incorrect_samples: List[Dict[str, Any]] = []

    for i in range(total_samples):
        model_output = answer_set[i]
        extracted_answer, think_tag_found = extract_prediction(model_output)
        if not think_tag_found:
            no_think_tag_count += 1
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
                    "no_think_tag": not think_tag_found,
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
        f"Outputs missing '</think>' tag: {no_think_tag_count} ({no_think_tag_count / total_samples:.2%})"
    )
    print(
        f"Extraction Failures: {extraction_failures} ({extraction_failures / total_samples:.2%})"
    )
