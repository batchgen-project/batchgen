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

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def load_longbench_datasets(base_dir: str = "longben") -> pd.DataFrame:
    """
    Load all parquet files from LongBench dataset directories and concatenate into a single DataFrame.

    Args:
        base_dir: Base directory containing the dataset subdirectories

    Returns:
        Concatenated DataFrame containing all datasets with an additional 'dataset_name' column
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

    # Check if base directory exists
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory '{base_dir}' not found")

    print(f"Starting to load datasets from: {base_path.absolute()}")
    print(f"{'='*70}")

    for dataset_name in dataset_names:
        dataset_path = base_path / dataset_name

        # Check if directory exists
        if not dataset_path.exists():
            print(f"  Directory not found: {dataset_name}")
            continue

        # Find all parquet files in the directory
        parquet_files = list(dataset_path.glob("*.parquet"))

        if not parquet_files:
            print(f"  No parquet files in: {dataset_name}")
            continue

        # Load and concatenate all parquet files for this dataset
        dataset_dfs = []
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                df['dataset_name'] = dataset_name  # Add source identifier
                df['source_file'] = parquet_file.name  # Track which file it came from
                dataset_dfs.append(df)
            except Exception as e:
                print(f"  Error reading {parquet_file.name} from {dataset_name}: {str(e)}")
                continue

        if dataset_dfs:
            combined_dataset = pd.concat(dataset_dfs, ignore_index=True)
            all_dataframes.append(combined_dataset)
            load_stats[dataset_name] = len(combined_dataset)
            print(f"  Loaded {dataset_name}: {len(combined_dataset):,} rows from {len(parquet_files)} file(s)")
        else:
            print(f"  Failed to load any data from: {dataset_name}")

    print(f"{'='*70}")

    # Concatenate all datasets
    if not all_dataframes:
        print("  No data loaded from any dataset!")
        return pd.DataFrame()

    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Print summary statistics
    print(f"\nLOADING COMPLETE: {len(final_df):,} total rows")

    return final_df


def _tensorize_sequence(token_ids: Any) -> torch.Tensor:
    """Convert various token ID formats to a PyTorch tensor."""
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


def decode_to_eos(tokenizer, tokens, min_tokens=5):
    """Decode tokens up to the first EOS token."""
    tokens_array = np.array(tokens)
    eos_positions = np.where(tokens_array == tokenizer.eos_token_id)[0]

    # Filter out EOS positions that are too early
    valid_eos_positions = eos_positions[eos_positions >= min_tokens]

    end_pos = valid_eos_positions[0] if len(valid_eos_positions) > 0 else len(tokens_array)
    return tokenizer.decode(tokens[:end_pos], skip_special_tokens=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongBench test client for BatchGen HTTP API")
    parser.add_argument("--hugging_face_checkpoint", type=str, required=True)
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Max number of prompts to process. If not set, run the whole dataset.")
    parser.add_argument("--max_input_length", type=int, default=None,
                        help="Max input length hint. If not set, determined dynamically from longest prompt.")
    parser.add_argument("--max_decoding_length", type=int, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to LongBench dataset directory. If not set, uses ./LongBench relative to script.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Path to model cache directory.")
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

    """
        Step 1: Select Model.
    """
    hugging_face_checkpoint = args.hugging_face_checkpoint

    """
        Step 2: Load dataset and apply chat template (prompt engineering).
    """
    benchmark_name = "Xnhyacinth/LongBench"
    max_prompts = args.max_prompts

    # Use provided dataset_dir or default to ./LongBench relative to script
    if args.dataset_dir:
        longbench_path = args.dataset_dir
    else:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        longbench_path = os.path.join(current_file_dir, "LongBench")

    query_df = load_longbench_datasets(longbench_path)
    queries = []
    # Load queries - limit to max_prompts if specified, otherwise load all
    for q in query_df['context']:
        queries.append(q)
        if max_prompts is not None and len(queries) >= max_prompts:
            break

    if max_prompts is not None:
        logger.info(f"Using top {max_prompts} prompts from dataset")
    else:
        logger.info(f"Running whole dataset ({len(queries)} prompts)")

    tokenizer = AutoTokenizer.from_pretrained(
        args.cache_dir,
        trust_remote_code=True,
        local_files_only=True
    )
    for prompt_idx in range(len(queries)):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": queries[prompt_idx]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        queries[prompt_idx] = text
    logger.info(f"Number of prompts: {len(queries)}")

    """
        Step 3: Run inference via BatchGen HTTP API
    """
    logger.info(f"Connecting to server at {base_url}")

    answer_set = run_inference_via_http_api(
        queries=queries,
        max_input_length=args.max_input_length,
        max_decoding_length=args.max_decoding_length,
        base_url=base_url,
        ignore_eos=args.ignore_eos,
        timeout_s=args.timeout,
    )

    """
        Step 4: Print responses to the prompts.
    """
    print_result = True
    if print_result:
        for idx in range(len(answer_set)):
            tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist())
            print("==================================================================")
            print(f"Query {idx}: {queries[idx][:1000]}")  # Truncate long queries for display
            print("\n\n")
            print(f"Answer {idx}: {tmp_answer}")
            print("\n\n")
