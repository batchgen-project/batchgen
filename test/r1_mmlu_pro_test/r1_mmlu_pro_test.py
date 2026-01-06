import argparse
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "This script requires the 'requests' package. "
        "Install it with: pip install requests"
    ) from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchGenHttpClient:
    """HTTP client for the new OpenAI-compatible BatchGen API."""

    def __init__(self, base_url: str, timeout_s: float = 6000.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._session = requests.Session()

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        response = self._session.post(url, json=payload, timeout=self._timeout_s)
        self._raise_for_status(response, "POST", url)
        if not response.content:
            return {}
        return response.json()

    def health_check(self) -> bool:
        try:
            url = f"{self._base_url}/health"
            response = self._session.get(url, timeout=10.0)
            return response.status_code == 200
        except Exception:
            return False

    def _raise_for_status(
        self, response: requests.Response, method: str, url: str
    ) -> None:
        if response.status_code < 400:
            return
        detail = None
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"{method} {url} failed ({response.status_code}): {detail}"
        )


def form_options(options: List[str]) -> str:
    option_str = "Options are:\n"
    opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, letter in zip(options, opts):
        option_str += f"({letter}): {opt}\n"
    return option_str


def decode_to_eos(
    tokenizer: AutoTokenizer, tokens: List[int], min_tokens: int = 5
) -> str:
    tokens_array = np.array(tokens)
    eos_positions = np.where(tokens_array == tokenizer.eos_token_id)[0]
    valid_eos_positions = eos_positions[eos_positions >= min_tokens]
    end_pos = (
        valid_eos_positions[0]
        if len(valid_eos_positions) > 0
        else len(tokens_array)
    )
    return tokenizer.decode(tokens[:end_pos], skip_special_tokens=True)


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


def _tensorize_sequence(token_ids: Any) -> torch.Tensor:
    if torch.is_tensor(token_ids):
        return token_ids.detach().cpu()
    if hasattr(token_ids, "tolist"):
        return torch.tensor(token_ids.tolist(), dtype=torch.int64)
    if isinstance(token_ids, np.ndarray):
        return torch.tensor(token_ids.tolist(), dtype=torch.int64)
    return torch.tensor(list(token_ids), dtype=torch.int64)


def normalize_token_sequences(results: Any) -> Optional[List[List[int]]]:
    """Normalize various result formats to List[List[int]]."""
    if torch.is_tensor(results):
        results = results.detach().cpu().tolist()
    elif isinstance(results, np.ndarray):
        results = results.tolist()
    elif isinstance(results, list) and results and torch.is_tensor(results[0]):
        results = [item.detach().cpu().tolist() for item in results]

    # Handle nested single-element lists
    if isinstance(results, list) and len(results) == 1:
        inner = results[0]
        if torch.is_tensor(inner):
            results = inner.detach().cpu().tolist()
        elif isinstance(inner, np.ndarray):
            results = inner.tolist()
        else:
            results = inner

    if isinstance(results, list):
        if results and all(isinstance(x, int) for x in results):
            return [list(results)]
        if results and all(isinstance(x, (list, tuple)) for x in results):
            return [list(seq) for seq in results]
    return None


def run_inference_via_http_api(
    queries: List[str],
    max_input_length: Optional[int],
    max_decoding_length: int,
    base_url: str,
    ignore_eos: bool = False,
    timeout_s: float = 6000.0,
) -> List[torch.Tensor]:
    """Run inference via BatchGen HTTP API.

    Args:
        queries: List of prompt strings
        max_input_length: Max input length hint. None = auto-detect from longest prompt.
        max_decoding_length: Max tokens to decode
        base_url: Server base URL (e.g., http://localhost:10900)
        ignore_eos: Whether to ignore EOS tokens during generation
        timeout_s: Request timeout in seconds

    Returns:
        List of output token tensors
    """
    client = BatchGenHttpClient(base_url, timeout_s)

    # Health check
    if not client.health_check():
        logger.warning("Server health check failed, proceeding anyway...")

    payload = {
        "prompts": queries,
        "max_input_len": max_input_length,
        "max_output_len": max_decoding_length,
        "ignore_eos": ignore_eos,
    }

    start_time = pd.Timestamp.now()
    response = client.post_json("/v1/inference", payload)
    latency = (pd.Timestamp.now() - start_time).total_seconds()

    server_latency_ms = response.get("latency_ms", "N/A")
    logger.info(f"Inference completed in {latency:.2f}s (server latency: {server_latency_ms}ms)")

    if response.get("status") != "success":
        raise RuntimeError(f"Inference failed: {response}")

    results = response.get("results")
    if not results:
        raise RuntimeError("Server returned empty results.")

    # Normalize results to List[List[int]]
    sequences = normalize_token_sequences(results)
    if not sequences:
        raise RuntimeError(f"Unexpected result format: {type(results)}")

    return [_tensorize_sequence(tokens) for tokens in sequences]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMLU Pro test client for BatchGen HTTP API")
    parser.add_argument("--hugging_face_checkpoint", type=str, required=True)
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Max number of prompts to process. If not set, run the whole dataset.")
    parser.add_argument("--max_input_length", type=int, default=None,
                        help="Max input length hint. If not set, determined dynamically from longest prompt.")
    parser.add_argument("--max_decoding_length", type=int, required=True)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    # New HTTP API arguments
    parser.add_argument("--base_url", type=str, default=None,
                        help="Server base URL (e.g., http://localhost:10900). Takes precedence over host/port.")
    parser.add_argument("--server_host", type=str, default="localhost",
                        help="Server hostname (legacy, use --base_url instead)")
    parser.add_argument("--server_port", type=int, default=10900,
                        help="Server port (legacy, use --base_url instead)")
    parser.add_argument("--ignore_eos", action="store_true",
                        help="Ignore EOS tokens and decode to max output length")
    parser.add_argument("--timeout", type=float, default=6000.0,
                        help="Request timeout in seconds")
    args = parser.parse_args()

    # Construct base URL from host/port if not provided directly
    if args.base_url:
        base_url = args.base_url
    else:
        base_url = f"http://{args.server_host}:{args.server_port}"

    hugging_face_checkpoint = args.hugging_face_checkpoint
    benchmark_name = "TIGER-Lab/MMLU-Pro"
    logger.info(f"Loading dataset {benchmark_name}")
    dataset = pd.read_parquet(
        os.path.join(os.path.dirname(__file__), "mmlu_pro_test.parquet")
    )
    # Use top max_prompts sequences if specified, otherwise run whole dataset
    if args.max_prompts is not None and args.max_prompts > 0:
        dataset = dataset.head(args.max_prompts)
        logger.info(f"Using top {args.max_prompts} sequences from dataset")

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

    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.hf_cache_dir:
        tokenizer_kwargs["cache_dir"] = args.hf_cache_dir
    tokenizer = AutoTokenizer.from_pretrained(
        hugging_face_checkpoint, **tokenizer_kwargs
    )
    for prompt_idx in range(len(queries)):
        messages = [
            {
                "role": "system",
                "content": "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`. Please follow the following examples and strictly give the answer with format 'the answer is (A/B/C/D/E/F/G/H/I/J)'.",
            },
            {"role": "user", "content": queries[prompt_idx]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        queries[prompt_idx] = text

    logger.info(f"Loaded {len(queries)} samples from the dataset.")
    tokenized = tokenizer(
        queries,
        add_special_tokens=True,
        padding=False,
        truncation=False,  # No truncation - full prompts sent to server
    )
    prompt_lengths = [len(t) for t in tokenized['input_ids']]
    logger.info(
        f"Prompt lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)}, "
        f"mean={sum(prompt_lengths)/len(prompt_lengths):.1f} tokens"
    )

    if torch.cuda.is_available():
        gpu0_memory = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        logger.info(f"GPU 0 memory usage before API call: {gpu0_memory} GB")
    else:
        logger.info("CUDA not available; skipping GPU memory log.")

    logger.info(f"Connecting to server at {base_url}")

    answer_set = run_inference_via_http_api(
        queries=queries,
        max_input_length=args.max_input_length,
        max_decoding_length=args.max_decoding_length,
        base_url=base_url,
        ignore_eos=args.ignore_eos,
        timeout_s=args.timeout,
    )

    print_result = True
    if print_result:
        for idx in range(len(answer_set)):
            tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist())
            print(
                "=================================================================="
            )
            print(f"Query {idx}: {queries[idx]}")
            print("\n\n")
            print(f"Answer {idx}: {tmp_answer}")
            print(
                "=================================================================="
            )
            print("\n\n")

    success = 0
    extraction_failures = 0
    no_think_tag_count = 0
    predictions: List[str] = []
    ground_truths = dataset["answer"].tolist()
    total_samples = len(answer_set)
    incorrect_samples: List[Dict[str, Any]] = []

    for i in range(total_samples):
        model_output = decode_to_eos(tokenizer, answer_set[i].tolist())
        extracted_answer, think_tag_found = extract_prediction(model_output)
        if not think_tag_found:
            no_think_tag_count += 1
            logger.warning(f"Sample {i}: No </think> tag found.")
        if extracted_answer:
            prediction = extracted_answer
        else:
            extraction_failures += 1
            logger.warning(
                f"Sample {i}: Could not extract answer. Marking as incorrect."
            )
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
        f"Extraction Failures (random guess used): {extraction_failures} ({extraction_failures / total_samples:.2%})"
    )
    print("---------------------------------\n")

    if incorrect_samples:
        print("\n" + "=" * 80)
        print("DETAILED ERROR ANALYSIS - Incorrect Predictions")
        print("=" * 80)
        print(f"\nTotal Incorrect: {len(incorrect_samples)}\n")
        for idx, error in enumerate(incorrect_samples, 1):
            print(f"Error #{idx}:")
            print(f"  Question ID: {error['id']}")
            print(f"  Extracted Answer: {error['extracted']}")
            print(f"  Ground Truth: {error['ground_truth']}")
            flags = []
            if error["extraction_failed"]:
                flags.append("EXTRACTION FAILED")
            if error["no_think_tag"]:
                flags.append("NO </think> TAG")
            if flags:
                print(f"  Flags: {' | '.join(flags)}")
            print("-" * 40)
        print("=" * 80 + "\n")
    else:
        print("\nPerfect Score! No incorrect predictions.\n")

    assert accuracy >= 0.835, (
        f"Test Failed: Accuracy of {accuracy:.2%} is below the 83.5% threshold."
    )
