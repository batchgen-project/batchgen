# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) FlashMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""LongBench test using OpenAI-compatible Batch API.

This script demonstrates the full OpenAI Batch API workflow:
1. Read parquet datasets from LongBench
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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer

from batchgen.batchgen_client import BatchGenHttpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_longbench_datasets(base_dir: str = "longben") -> pd.DataFrame:
    """Load all parquet files from LongBench dataset directories.

    Args:
        base_dir: Base directory containing the dataset subdirectories

    Returns:
        Concatenated DataFrame with all datasets
    """
    dataset_names = [
        "2wikimqa", "2wikimqa_e", "dureader", "gov_report", "gov_report_e",
        "hotpotqa", "hotpotqa_e", "lcc", "lcc_e", "lsht",
        "multi_news", "multi_news_e", "multifieldqa_en", "multifieldqa_en_e",
        "multifieldqa_zh", "musique", "narrativeqa", "passage_count",
        "passage_count_e", "passage_retrieval_en", "passage_retrieval_en_e",
        "passage_retrieval_zh", "qasper", "qasper_e", "qmsum",
        "repobench-p", "repobench-p_e", "samsum", "samsum_e",
        "trec", "trec_e", "triviaqa", "triviaqa_e", "vcsum"
    ]

    all_dataframes: List[pd.DataFrame] = []
    load_stats: Dict[str, int] = {}
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory '{base_dir}' not found")

    logger.info(f"Loading datasets from: {base_path.absolute()}")

    for dataset_name in dataset_names:
        dataset_path = base_path / dataset_name

        if not dataset_path.exists():
            continue

        parquet_files = list(dataset_path.glob("*.parquet"))
        if not parquet_files:
            continue

        dataset_dfs = []
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                df['dataset_name'] = dataset_name
                df['source_file'] = parquet_file.name
                dataset_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {parquet_file}: {e}")
                continue

        if dataset_dfs:
            combined_dataset = pd.concat(dataset_dfs, ignore_index=True)
            all_dataframes.append(combined_dataset)
            load_stats[dataset_name] = len(combined_dataset)
            logger.info(f"Loaded {dataset_name}: {len(combined_dataset)} rows")

    if not all_dataframes:
        logger.warning("No data loaded from any dataset!")
        return pd.DataFrame()

    final_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Total loaded: {len(final_df)} rows from {len(load_stats)} datasets")
    return final_df


def create_batch_input_file(
    queries: List[str],
    model_name: str,
    max_tokens: int,
    output_path: Path,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    random_sampling_params: bool = False,
    random_max_completion_tokens: bool = False,
    min_completion_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    ignore_eos: bool = False,
) -> Optional[List[int]]:
    """Create JSONL file in OpenAI batch format.

    Each line is a JSON object with:
    - custom_id: Unique identifier for the request
    - method: HTTP method (POST)
    - url: API endpoint (/v1/chat/completions)
    - body: Request body with model, messages, max_tokens/max_completion_tokens, sampling params

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
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                ],
            }
            # Per-request max_completion_tokens or uniform max_tokens
            if per_seq_limits is not None:
                body["max_completion_tokens"] = per_seq_limits[idx]
            else:
                body["max_tokens"] = max_tokens

            # extra_body: force full-length decode (fixed-output-length BCT benchmark)
            if ignore_eos:
                body["ignore_eos"] = True

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
                "custom_id": f"longbench-{idx}",
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
        description="LongBench test using OpenAI-compatible Batch API"
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
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to LongBench dataset directory",
    )
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
        "--ignore_eos", action="store_true",
        help="Set ignore_eos=true in each request body (extra_body) so every "
             "sequence decodes its full output length (fixed-length BCT benchmark).",
    )
    parser.add_argument(
        "--max_input_length", type=int, default=None,
        help="Head-truncate each prompt to this many input tokens (client-side, "
             "via the model tokenizer) for fixed-input BCT workloads (e.g. 8000).",
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

    # Load dataset
    if args.dataset_dir:
        longbench_path = args.dataset_dir
    else:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        longbench_path = os.path.join(current_file_dir, "LongBench")

    logger.info(f"Loading LongBench dataset from {longbench_path}")
    query_df = load_longbench_datasets(longbench_path)

    # Extract queries: combine context + question fields
    queries: List[str] = []
    for _, row in query_df.iterrows():
        prompt = row['context'] + "\n\n" + row['question']
        queries.append(prompt)
        if args.max_prompts is not None and len(queries) >= args.max_prompts:
            break

    if args.max_prompts is not None:
        logger.info(f"Using top {args.max_prompts} prompts from dataset")
    else:
        logger.info(f"Running whole dataset ({len(queries)} prompts)")

    # Client-side head-truncation to a fixed input length (peak-performance BCT
    # workloads need exactly ~N input tokens; output length is fixed via --ignore_eos).
    if args.max_input_length is not None:
        # Use BatchGen's tokenizer registry (GLM-5/Kimi/etc. use custom tokenizers
        # that AutoTokenizer cannot load); matches what the server tokenizes with.
        from batchgen.config import load_tokenizer
        logger.info(
            f"Head-truncating {len(queries)} prompts to {args.max_input_length} "
            f"tokens using BatchGen tokenizer for {hugging_face_checkpoint}"
        )
        tok = load_tokenizer(hugging_face_checkpoint)
        n_trunc = 0
        for i, q in enumerate(queries):
            ids = tok.encode(q, add_special_tokens=False)
            if len(ids) > args.max_input_length:
                queries[i] = tok.decode(ids[:args.max_input_length])
                n_trunc += 1
        logger.info(
            f"Truncated {n_trunc}/{len(queries)} prompts to <= "
            f"{args.max_input_length} input tokens"
        )

    # Create temp file for batch input (will be uploaded to server)
    temp_dir = Path(tempfile.gettempdir())
    input_file = temp_dir / "longbench_batch_input.jsonl"

    # Create batch input file (with per-request sampling params in JSONL body)
    per_seq_limits = create_batch_input_file(
        queries=queries,
        model_name=hugging_face_checkpoint,
        max_tokens=args.max_decoding_length,
        output_path=input_file,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        random_sampling_params=args.random_sampling_params,
        random_max_completion_tokens=args.random_max_completion_tokens,
        min_completion_tokens=args.min_completion_tokens,
        max_completion_tokens=args.max_completion_tokens,
        ignore_eos=args.ignore_eos,
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
    )

    # Sort results by custom_id to match original order
    results.sort(key=lambda x: int(x.get("custom_id", "longbench-0").split("-")[1]))

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

    # Print sample results
    print_result = True
    if print_result:
        for idx in range(min(5, len(answer_set))):  # Print first 5 for brevity
            print("==================================================================")
            print(f"Query {idx}: {queries[idx][:500]}...")
            print("\n")
            print(f"Answer {idx}: {answer_set[idx][:1000]}")
            print("==================================================================\n")

    # Summary
    print("\n--- Batch Processing Summary ---")
    print(f"Total Prompts: {len(queries)}")
    print(f"Results Received: {len(answer_set)}")
