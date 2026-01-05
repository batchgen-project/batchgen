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

from batchgen import BatchGenClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            print(f"⚠️  Directory not found: {dataset_name}")
            continue
        
        # Find all parquet files in the directory
        parquet_files = list(dataset_path.glob("*.parquet"))
        
        if not parquet_files:
            print(f"⚠️  No parquet files in: {dataset_name}")
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
                print(f"❌ Error reading {parquet_file.name} from {dataset_name}: {str(e)}")
                continue
        
        if dataset_dfs:
            combined_dataset = pd.concat(dataset_dfs, ignore_index=True)
            all_dataframes.append(combined_dataset)
            load_stats[dataset_name] = len(combined_dataset)
            print(f"✓ Loaded {dataset_name}: {len(combined_dataset):,} rows from {len(parquet_files)} file(s)")
        else:
            print(f"⚠️  Failed to load any data from: {dataset_name}")
    
    print(f"{'='*70}")
    
    # Concatenate all datasets
    if not all_dataframes:
        print("❌ No data loaded from any dataset!")
        return pd.DataFrame()
    
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Print summary statistics
    print(f"\n📊 LOADING COMPLETE")
    
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


def run_inference_via_api(
    queries: List[str],
    max_input_length: Optional[int],
    max_decoding_length: int,
    server_host: str,
    server_port: int,
) -> List[torch.Tensor]:
    """Run inference via BatchGen API.
    
    Args:
        queries: List of prompt strings
        max_input_length: Max input length hint. None = auto-detect from longest prompt.
        max_decoding_length: Max tokens to decode
        server_host: Server hostname
        server_port: Server port
        
    Returns:
        List of output token tensors
    """
    client = BatchGenClient(server_host, server_port)
    client.connect()
    
    start_time = pd.Timestamp.now()
    response = client.submit_inference(
        queries=queries,
        max_input_len=max_input_length,
        max_output_len=max_decoding_length,
    )
    latency = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"Inference round-trip completed in {latency:.2f}s")
    client.close()
    
    if not isinstance(response, dict) or "results" not in response:
        raise RuntimeError(f"Invalid response from server: {response}")
    sequences = response["results"]
    if not sequences:
        raise RuntimeError("Server returned empty results.")
    return [_tensorize_sequence(tokens) for tokens in sequences[0]]


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--hugging_face_checkpoint", type=str)
	parser.add_argument("--max_prompts", type=int, default=1024)
	parser.add_argument("--max_input_length", type=int, default=None, help="Max input length hint. If not set, determined dynamically from longest prompt.")
	parser.add_argument("--max_decoding_length", type=int)
	parser.add_argument("--dataset_cache_dir", type=str, default="~/.cache/huggingface/datasets/Xnhyacinth___long_bench")
	parser.add_argument("--cache_dir", type=str, default=None)
	parser.add_argument("--server_host", type=str, default="localhost")
	parser.add_argument("--server_port", type=int, default=10900)
	
	args = parser.parse_args()

	"""
		Step 1: Select Model.
	"""
	hugging_face_checkpoint = args.hugging_face_checkpoint

	"""
		Step 2: Load dataset and apply chat template(prompt engineering).
	"""
	benchmark_name = "Xnhyacinth/LongBench"
	max_prompts = args.max_prompts

	current_file_dir = os.path.dirname(os.path.abspath(__file__))
	longbench_path = os.path.join(current_file_dir, "LongBench")
	query_df = load_longbench_datasets(longbench_path)
	queries = []
	# Load all queries without length filtering
	for q in query_df['context']:
		queries.append(q)
		if len(queries) == max_prompts:
			break

	# If number of queries is less than max_prompts, fill the rest by duplicating
	if len(queries) < max_prompts:
		queries = queries * (max_prompts // len(queries)) + queries[: max_prompts % len(queries)]
	
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
	if len(queries) != args.max_prompts:
		logger.warning(f"Number of prompts {len(queries)} is not equal to max_prompts {args.max_prompts}.")

	"""
		Step 3: Run inference via BatchGen client API
	"""
	logger.info(f"Connecting to server at {args.server_host}:{args.server_port}")
	
	answer_set = run_inference_via_api(
		queries=queries,
		max_input_length=args.max_input_length,
		max_decoding_length=args.max_decoding_length,
		server_host=args.server_host,
		server_port=args.server_port,
	)

	"""
		Step 4: Print responses to the prompts.
	"""
	def decode_to_eos(tokenizer, tokens, min_tokens=5):
		tokens_array = np.array(tokens)
		eos_positions = np.where(tokens_array == tokenizer.eos_token_id)[0]
		
		# Filter out EOS positions that are too early
		valid_eos_positions = eos_positions[eos_positions >= min_tokens]
		
		end_pos = valid_eos_positions[0] if len(valid_eos_positions) > 0 else len(tokens_array)
		return tokenizer.decode(tokens[:end_pos], skip_special_tokens=False)

	print_result = True
	if print_result:
		for idx in range(len(answer_set)):
			tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist())
			print("==================================================================")
			print(f"Query {idx}: {queries[idx][:1000]}")  # Truncate long queries for display
			print("\n\n")
			print(f"Answer {idx}: {tmp_answer}")
			print("\n\n")