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

import logging
import os
from transformers import AutoTokenizer
# from batchgen.engine import batchgen
from batchgen.entrypoint import BatchGen
import numpy as np
import datasets
import argparse
import torch

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--hugging_face_checkpoint", type=str)
	parser.add_argument("--host_kv_cache_size", type=int)
	parser.add_argument("--max_prompts", type=int)
	parser.add_argument("--ATTN_MODE", type=str)
	parser.add_argument("--SPLIT_RATIO_W", type=str)
	parser.add_argument("--max_input_length", type=int)
	parser.add_argument("--max_decoding_length", type=int)
	parser.add_argument("--dataset_cache_dir", type=str, default="/root/.cache/huggingface/datasets/Xnhyacinth___long_bench")
	parser.add_argument("--cache_dir", type=str, default=None)
	# Add parameter server connection options
	parser.add_argument("--server_host", type=str, default="localhost")
	parser.add_argument("--server_port", type=int, default=10900)
	parser.add_argument("--dist_init_addr", type=str)
	parser.add_argument("--nnodes", type=int, default=2)
	parser.add_argument("--node_rank", type=int, default=0)



	args = parser.parse_args()

	"""
		Step 1: Select Model.
	"""
	# hugging_face_checkpoint = "mistralai/Mixtral-8x7B-Instruct-v0.1"
	hugging_face_checkpoint = args.hugging_face_checkpoint

	"""
		Step 2: Load dataset and apply chat template(prompt engineering).
	"""
	max_prompts = args.max_prompts
	task_names = [
		"2wikimqa",
		"2wikimqa_e",
		"dureader",
		"gov_report",
		"gov_report_e",
		"hotpotqa",
		"hotpotqa_e",
		"lcc",
		"lcc_e",
		"lsht",
		"multi_news",
		"multi_news_e",
		"multifieldqa_en",
		"multifieldqa_en_e",
		"multifieldqa_zh",
		"musique",
		"narrativeqa",
		"passage_count",
		"passage_count_e",
		"passage_retrieval_en",
		"passage_retrieval_en_e",
		"passage_retrieval_zh",
		"qasper",
		"qasper_e",
		"qmsum",
		"repobench-p",
		"repobench-p_e",
		"samsum",
		"samsum_e",
		"trec",
		"trec_e",
		"triviaqa",
		"triviaqa_e",
		"vcsum"
	]
	queries = []
	for task_name in task_names:
		dataset = datasets.load_dataset(
			f"{args.dataset_cache_dir}/{task_name}", 
			split="test",
			trust_remote_code=True
		)
		for q in dataset["context"]:
			if len(q.split(" ")) >= args.max_input_length:
				queries.append(q)
				if len(queries) == max_prompts:
					break
		if len(queries) == max_prompts:
			break

	# If number of queries is less than max_prompts, fill the rest by duplicating
	if len(queries) < max_prompts:
		queries = queries * (max_prompts // len(queries)) + queries[: max_prompts % len(queries)]
	

	tokenizer = AutoTokenizer.from_pretrained(
		hugging_face_checkpoint, trust_remote_code=True
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
	logging.info(f"Number of prompts: {len(queries)}")
	if len(queries) != args.max_prompts:
		logging.warning(f"Number of prompts {len(queries)} is not equal to max_prompts {args.max_prompts}.")

	"""
		Step 3: Launch BatchGen using the standalone parameter server
	"""
	logging.info(f"Connecting to parameter server at {args.server_host}:{args.server_port}")
	logging.info(f"Using model {hugging_face_checkpoint}")
	
	# Log device 0 gpu memory usage
	gpu0_memory = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
	logging.info(f"GPU 0 memory usage before BatchGen init: {gpu0_memory} GB")
	# Run inference with our standalone parameter server
	answer_set = BatchGen(
		huggingface_ckpt_name=hugging_face_checkpoint,
		queries=queries,
		max_input_length=args.max_input_length,
		max_decoding_length=args.max_decoding_length,
		device=[0,1,2,3,4,5,6,7],		
		host_kv_cache_size=args.host_kv_cache_size,
		cache_dir=args.cache_dir,
		# Connect to our standalone parameter server
		parameter_server_host=args.server_host,
		parameter_server_port=args.server_port,
		dist_init_addr = args.dist_init_addr,
		nnodes = args.nnodes,
		node_rank = args.node_rank,
		device_per_node= 8,
	).run()

	"""
		Step 4: Print responses to the prompts.
	"""
	def decode_to_eos(tokenizer, tokens):
		tokens_array = np.array(tokens)
		eos_positions = np.where(tokens_array == tokenizer.eos_token_id)[0]
		end_pos = eos_positions[0] if len(eos_positions) > 0 else len(tokens_array)
		return tokenizer.decode(tokens[:end_pos], skip_special_tokens=True)

	print_result = True
	if print_result:
		for idx in range(len(answer_set)):
			tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist()[0])
			print(f"Answer {idx}: {tmp_answer}")
			print("\n\n")