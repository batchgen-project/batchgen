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
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
	# Accept hugging_face_checkpoint,host_kv_cache_size and max_prompts
	# as command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--hugging_face_checkpoint", type=str)
	parser.add_argument("--host_kv_cache_size", type=int)
	parser.add_argument("--max_prompts", type=int, default=0)
	parser.add_argument("--ATTN_MODE", type=str)
	parser.add_argument("--SPLIT_RATIO_W", type=str)
	parser.add_argument("--max_input_length", type=int)
	parser.add_argument("--max_decoding_length", type=int)
	parser.add_argument("--hf_cache_dir", type=str, default=None)
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
	hugging_face_checkpoint = args.hugging_face_checkpoint

	"""
		Step 2: Load dataset and apply chat template(prompt engineering).
	"""
	benchmark_name = "TIGER-Lab/MMLU-Pro"
	logging.info(f"Loading dataset {benchmark_name}")
	dataset = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'mmlu_pro_test.parquet'))	
	if args.max_prompts != 0:
		dataset = dataset.head(args.max_prompts)

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
		prompt = 'Please read the following 5 examples: \n' + prefix + 'Please answer the following question: \n' + 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + '\n' 
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
	
	
	logging.info(f"Loaded {len(queries)} samples from the dataset.")
	# Log the longest query length
	# logging.info(f"Longest query length: {max([len(q.split(' ')) for q in queries])} tokens")
	tokenized = tokenizer(
		queries, 
		add_special_tokens=True,
		padding=True,          # Pads sentences to the same length in the batch
		truncation=True,       # Truncates sentences that are too long
		max_length=args.max_input_length         # Specify a max length for truncation
	)

	# Note: The output 'tokenized' is now a dictionary containing 'input_ids', 'attention_mask', etc.
	# To get the lengths, you access the 'input_ids' key.
	logging.info(f"Longest query length: {max([len(t) for t in tokenized['input_ids']])} tokens")


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
		# for idx in range(4):
			tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist())
			# tmp_answer = tokenizer.decode(answer_set[idx].tolist(), skip_special_tokens=False)
			# print(f"Prompt {idx}: {queries[idx][:args.max_input_length]}")
			print("==================================================================")
			print(f"Query {idx}: {queries[idx][:args.max_input_length]}")
			print("\n\n")
			print(f"Answer {idx}: {tmp_answer}")
			print("==================================================================")
			print("\n\n")
	

	# """
	# 	Step 5: Calculate the accuracy.
	# """
	# def get_prediction(output,idx):
	# 	# Find the end of thinking section
	# 	think_end = output.find("</think>")
		
	# 	if think_end != -1:
	# 		# Search for answer pattern only after </think>
	# 		search_text = output[think_end + len("</think>"):]
	# 	else:
	# 		# If no </think> tag found, search the entire output
	# 		search_text = output
	# 		# print("Warning: No </think> tag found, searching entire output")
	# 		logging.warning(f"No </think> tag found in output for sample {idx}, searching entire output")
		
	# 	pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
	# 	match = re.search(pattern, search_text)
	# 	if match:
	# 		return match.group(1)
	# 	else:
	# 		print("extraction failed, do a random guess")
	# 		return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

	# success, fail = 0, 0
	# answers = []
	# ground_truths = dataset['answer'].tolist()
	# for answer_idx in range(len(answer_set)):
	# 	pred_answer = decode_to_eos(tokenizer, answer_set[answer_idx].tolist())
	# 	# pred_answer = tokenizer.decode(answer_set[answer_idx].tolist(), skip_special_tokens=False)
	# 	prediction = get_prediction(pred_answer, answer_idx)
	# 	answers.append(prediction)

	# 	if prediction == ground_truths[answer_idx]:
	# 		success += 1
	# 	else:
	# 		fail += 1

	# print(f"Total samples: {success + fail}, Success: {success}, Fail: {fail}")
	# accuracy = success / (success + fail)
	# print(f"Accuracy: {accuracy}")
	# assert accuracy >= 0.80, "Accuracy is below 80%, test failed"
	
	
	# --- Step 1: Refined Prediction Extraction Function ---
	def extract_prediction(model_output: str) -> Tuple[Optional[str], bool]:
		"""
		Extracts the predicted letter answer from a model's output string.

		This function first checks for a '</think>' tag to determine where to search
		for the answer. It then uses a regular expression to find the answer.

		Args:
			model_output: The complete string generated by the model.

		Returns:
			A tuple containing:
			- The predicted answer ('A' through 'J') or None if not found.
			- A boolean indicating if the '</think>' tag was present.
		"""
		# Check for the presence of the '</think>' tag
		think_end_pos = model_output.find("</think>")
		think_tag_found = think_end_pos != -1

		# Determine the part of the output to search for the answer
		if think_tag_found:
			search_text = model_output[think_end_pos + len("</think>"):]
		else:
			search_text = model_output # Search the entire output if the tag is missing

		# A more flexible regex to capture "answer is A", "answer is: (B)", etc.
		pattern = r"answer is\s*:?\s*\(?([A-J])\)?"
		match = re.search(pattern, search_text, re.IGNORECASE)

		if match:
			# Return the captured group (the letter), capitalized, and tag status
			return match.group(1).upper(), think_tag_found
		else:
			# Return None for the answer if no match is found
			return None, think_tag_found

	# --- Step 2: Main Evaluation Logic ---
	# These variables are assumed to be defined elsewhere in your script
	# tokenizer, answer_set, dataset, decode_to_eos

	# Initialize counters for a detailed summary
	success = 0
	extraction_failures = 0
	no_think_tag_count = 0
	predictions = []
	ground_truths = dataset['answer'].tolist()
	total_samples = len(answer_set)

	# Process each sample
	for i in range(total_samples):
		# Decode the model's raw output to a string
		model_output = decode_to_eos(tokenizer, answer_set[i].tolist())

		# Use the refined function to extract the answer
		extracted_answer, was_think_tag_found = extract_prediction(model_output)

		# Log if the <think> tag was missing
		if not was_think_tag_found:
			no_think_tag_count += 1
			logging.warning(f"Sample {i}: No </think> tag found.")

		# Determine the final prediction, guessing if extraction failed
		if extracted_answer:
			prediction = extracted_answer
		else:
			extraction_failures += 1
			logging.warning(f"Sample {i}: Could not extract answer. Making a random guess.")
			# Fallback to a random guess if the pattern isn't found
			prediction = random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

		predictions.append(prediction)

		# Check if the prediction was correct
		if prediction == ground_truths[i]:
			success += 1

	# --- Step 3: Calculate and Display Results ---
	accuracy = success / total_samples if total_samples > 0 else 0

	# Print a comprehensive summary
	print("\n--- ✅ Evaluation Summary ---")
	print(f"Total Samples: {total_samples}")
	print(f"✅ Correct: {success}")
	print(f"❌ Incorrect: {total_samples - success}")
	print("-" * 30)
	print(f"🎯 Accuracy: {accuracy:.2%}")
	print("-" * 30)
	# This is the log you requested, plus additional helpful metrics
	print(f"⚠️ Outputs missing '</think>' tag: {no_think_tag_count} ({no_think_tag_count / total_samples:.2%})")
	print(f"❓ Extraction Failures (random guess used): {extraction_failures} ({extraction_failures / total_samples:.2%})")
	print("---------------------------------\n")

	# Assert the final accuracy against the required threshold
	assert accuracy >= 0.80, f"Test Failed: Accuracy of {accuracy:.2%} is below the 80% threshold."
	
		