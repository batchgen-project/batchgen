"""
	This script is for testing the accuracy of DeepSeek-R1 inference on the MMLU-Pro dataset.
	The accurateness is defined as exact match accuracy.
"""

import logging
import os
from transformers import AutoTokenizer
from batchgen.engine import batchgen
import numpy as np
import datasets
import argparse
import torch
import pandas as pd
import re
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
	# Accept hugging_face_checkpoint,host_kv_cache_size and max_prompts
	# as command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--hugging_face_checkpoint", type=str)
	parser.add_argument("--host_kv_cache_size", type=int)
	parser.add_argument("--max_prompts", type=int)
	parser.add_argument("--ATTN_MODE", type=str)
	parser.add_argument("--SPLIT_RATIO_W", type=str)
	parser.add_argument("--max_input_length", type=int)
	parser.add_argument("--max_decoding_length", type=int)
	parser.add_argument("--hf_cache_dir", type=str, default=None)
	parser.add_argument("--cache_dir", type=str, default=None)
	# Add parameter server connection options
	parser.add_argument("--server_host", type=str, default="localhost")
	parser.add_argument("--server_port", type=int, default=9090)
	parser.add_argument("--dist_init_addr", type=str)
	parser.add_argument("--nnodes", type=int, default=2)
	parser.add_argument("--node_rank", type=int, default=0)
	args = parser.parse_args()

	"""
		Step 1: Select Model.
	"""
	hugging_face_checkpoint = args.hugging_face_checkpoint

	"""
		Step 2: Load dataset and apply chat template(prompt engineering).
	"""
	benchmark_name = "TIGER-Lab/MMLU-Pro"
	logging.info(f"Loading dataset {benchmark_name}")
	dataset = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'mmlu_pro_test.parquet'))	

	## to be removed
	dataset = dataset.head(args.max_prompts)
	# print all the questions:
	# print(dataset['question'].tolist())
	# exit()

	def form_options(options: list):
		option_str = 'Options are:\n'
		opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
		for opt, o in zip(options, opts):
			option_str += f'({o}): {opt}' + '\n'
		return option_str
	

	
	validation_set = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'mmlu_pro_validation.parquet'))
	categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
				  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
				  'psychology', 'history']

	# load 5-shot prompts for each category
	prompts = {c: '' for c in categories}
	# for d in validation_set:
	for idx, d in validation_set.iterrows():
		prompts[d['category']] += 'Q:' + ' ' + d['question'] + '\n' + form_options(d['options']) + '\n' + d['cot_content'] + '\n\n'


	queries = []
	for row_idx, entry in dataset.iterrows():
		prefix = prompts[entry['category']]
		prompt = prefix + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + '\n' 
		queries.append(prompt)

	tokenizer = AutoTokenizer.from_pretrained(
		hugging_face_checkpoint, trust_remote_code=True
	)
	for prompt_idx in range(len(queries)):
		messages = [
			{"role": "system", "content": "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`. Please follow the following examples and strictly give the answer with format 'the answer is (A/B/C/D/E/F/G/H/I/J)'."},
			{"role": "user", "content": queries[prompt_idx]},
		]
		text = tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True
		)
		queries[prompt_idx] = text
	
	# print(queries)
	# exit()
	
	logging.info(f"Loaded {len(queries)} samples from the dataset.")
	# Print first 5 questions:
	# for i in range(min(5, len(queries))):
	# 	logging.info(f"Prompt {i}: {queries[i][:args.max_input_length]}")
	# exit()
	"""
		Step 3: Launch BatchGen using the standalone parameter server
	"""
	logging.info(f"Connecting to parameter server at {args.server_host}:{args.server_port}")
	logging.info(f"Using model {hugging_face_checkpoint}")
	
	# Log device 0 gpu memory usage
	gpu0_memory = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
	logging.info(f"GPU 0 memory usage before moe-gen init: {gpu0_memory} GB")
	# Run inference with our standalone parameter server
	answer_set = batchgen(
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
		device_per_node= 8
	)

	"""
		Step 4: Print responses to the prompts.
	"""
	def decode_to_eos(tokenizer, tokens):
		tokens_array = np.array(tokens)
		eos_positions = np.where(tokens_array == tokenizer.eos_token_id)[0]
		end_pos = eos_positions[0] if len(eos_positions) > 0 else len(tokens_array)
		# return tokenizer.decode(tokens[:end_pos], skip_special_tokens=True)
		return tokenizer.decode(tokens[:end_pos], skip_special_tokens=False)

	print_result = True
	if print_result:
		for idx in range(len(answer_set)):
		# for idx in range(4):
			tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist()[0])
			# print(f"Prompt {idx}: {queries[idx][:args.max_input_length]}")
			print(f"Answer {idx}: {tmp_answer}")
			print("\n\n")	
	

	# """
	# 	Step 5: Calculate the accuracy.
	# """
	# def get_prediction(output):
	# 	pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
	# 	match = re.search(pattern, output)
	# 	if match:
	# 		return match.group(1)
	# 	else:
	# 		print("extraction failed, do a random guess")
	# 		return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
		
	# success, fail = 0, 0
	# answers = []
	# ground_truths = dataset['answer'].tolist()
	# for answer_idx in range(len(answer_set)):
	# 	pred_answer = decode_to_eos(tokenizer, answer_set[answer_idx].tolist()[0])
	# 	prediction = get_prediction(pred_answer)
	# 	answers.append(prediction)
	
	# 	if prediction == ground_truths[answer_idx]:
	# 		success += 1
	# 	else:
	# 		fail += 1
	
	# print(f"Total samples: {success + fail}, Success: {success}, Fail: {fail}")
	# print(f"Accuracy: {success / (success + fail)}")

	"""
		Step 5: Calculate the accuracy.
	"""
	def get_prediction(output,idx):
		# Find the end of thinking section
		think_end = output.find("</think>")
		
		if think_end != -1:
			# Search for answer pattern only after </think>
			search_text = output[think_end + len("</think>"):]
		else:
			# If no </think> tag found, search the entire output
			search_text = output
			# print("Warning: No </think> tag found, searching entire output")
			logging.warning(f"No </think> tag found in output for sample {idx}, searching entire output")
		
		pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
		match = re.search(pattern, search_text)
		if match:
			return match.group(1)
		else:
			print("extraction failed, do a random guess")
			return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

	success, fail = 0, 0
	answers = []
	ground_truths = dataset['answer'].tolist()
	for answer_idx in range(len(answer_set)):
		pred_answer = decode_to_eos(tokenizer, answer_set[answer_idx].tolist()[0])
		prediction = get_prediction(pred_answer, answer_idx)
		answers.append(prediction)

		if prediction == ground_truths[answer_idx]:
			success += 1
		else:
			fail += 1

	print(f"Total samples: {success + fail}, Success: {success}, Fail: {fail}")
	print(f"Accuracy: {success / (success + fail)}")
		