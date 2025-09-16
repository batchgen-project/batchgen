"""
	We modularize prefill and decoding.
	This module defines prefill. 
	The scheduler would call prefill and decoding.
"""

import torch
import math
import logging
import tqdm 
from batchgen.models.Wrapper import Attn_Wrapper
from batchgen.utils import create_position_ids_from_attention_mask
from batchgen.sampling import sample_with_temperature_top_p, greedy_decode



class Prefill():
	"""
		When prefill(), we first configure the parallelism strategy and engine status for prefill.
		Then we call the model to generate the hidden states.

		The default behavior as follows:
		1. Prefill the input batch of sequences and offload the KV to host memory.

		We rely on the scheduler to check the host is full.
	"""
	def __init__(self, model_config, engine_config, core_engine, parallel_manager, comm):
		self.model_config = model_config
		self.engine_config = engine_config
		self.parallel_manager = parallel_manager
		self.core_engine = core_engine
		self.comm = comm
	
	def config_prefill(self):
		self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
		self.set_phase("prefill")
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_prefill_buffer()
		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
		self.core_engine.clear_kv_storage()
		self.core_engine.start_h2d_worker()
	
	def cleanup_prefill(self):
		pass
		
	def run(self, batch):
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = False

		input_ids = torch.cat(
			[
				self.query_book[query_idx].encoded["input_ids"][
					:, : self.max_input_length
				]
				for query_idx in batch
			],
			dim=0,
		)
		attention_masks = torch.cat(
			[
				self.query_book[query_idx].encoded["attention_mask"][
					:, : self.max_input_length
				]
				for query_idx in batch
			],
			dim=0,
		)

		num_prefill_micro_batches = math.ceil(
			len(batch)
			/ self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size
		)
		prefill_micro_batch_input_ids = torch.split(
			input_ids,
			self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
		)
		Prefill_micro_batch_attention_masks = torch.split(
			attention_masks,
			self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
		)
		logging.info(
			f"Number of prefill micro batches: {num_prefill_micro_batches}"
		)
		cur_batch_start = 0
		output_tokens = []
		for micro_batch_idx in tqdm(
			range(num_prefill_micro_batches), desc="Prefill Micro Batch"
		):
			with torch.inference_mode():
				Attn_Wrapper.attention_mask = (
					Prefill_micro_batch_attention_masks[micro_batch_idx]
				)
				# if "deepseek" in self.model_config.model_type:
				# 	Attn_Wrapper.position_ids = (
				# 		create_position_ids_from_attention_mask(
				# 			Prefill_micro_batch_attention_masks[micro_batch_idx]
				# 		)
				# 	)
				# else:
				# 	Attn_Wrapper.position_ids = (
				# 		create_position_ids_from_attention_mask(
				# 			Prefill_micro_batch_attention_masks[micro_batch_idx]
				# 		)
				# 	)
				Attn_Wrapper.position_ids = (
					create_position_ids_from_attention_mask(
						Prefill_micro_batch_attention_masks[micro_batch_idx]
					)
				)

				cur_batch_size = prefill_micro_batch_input_ids[
					micro_batch_idx
				].shape[0]
				cur_batch = batch[
					cur_batch_start : cur_batch_start + cur_batch_size
				]
				Attn_Wrapper.cur_batch = cur_batch
				cur_batch_start += cur_batch_size
				assert len(cur_batch) == cur_batch_size

				outputs = self.model(
					prefill_micro_batch_input_ids[micro_batch_idx].to(
						self.torch_device
					),
					attention_mask=Prefill_micro_batch_attention_masks[
						micro_batch_idx
					].to(self.torch_device),
					# position_ids=micro_batch_position_ids[micro_batch_idx].to(self.torch_device),
					use_cache=False,
				)
				# Greedy
				new_tokens = torch.argmax(
					outputs.logits[:, -1, :], dim=-1
				).view(-1, 1)
				# new_tokens = sample_with_temperature_top_p(
				# 	outputs.logits[:, -1, :],
				# 	temperature=0.6,
				# 	top_p=0.95,
				# ).view(-1, 1)
				output_tokens.append(new_tokens)

		new_tokens = torch.cat(output_tokens, dim=0)
		self.update_new_token(new_tokens, batch, 0)
		return new_tokens



