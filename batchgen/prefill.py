"""
	We modularize prefill and decoding.
	This module defines prefill. 
	The scheduler would call prefill and decoding.
"""

import torch
import math
import logging
import tqdm
from dataclasses import dataclass
import torch.distributed as dist
from typing import Optional
from batchgen.models.Wrapper import Attn_Wrapper
from batchgen.utils import create_position_ids_from_attention_mask
from batchgen.sequence import SequenceBatch, SequenceStatus
from batchgen.utils import deep_free_model_memory

@dataclass
class PrefillRequest:
	"""
		PrefillRequest holds the input data for for model forward pass in prefill stage.
	"""
	sequence_uuids: list[int]
	input_ids: torch.Tensor # (batch_size, seq_len)
	attention_mask: torch.Tensor # (batch_size, seq_len)
	position_ids: torch.Tensor # (batch_size, seq_len)
	prompt_lengths: list[int]

	@property
	def batch_size(self):
		return len(self.sequence_uuids)

@dataclass
class PrefillResult:
	"""
		PrefillResult encapsulates the output from the prefill stage.
	"""
	sequence_uuids: list[int]
	first_tokens: torch.Tensor # (batch_size, )
	first_token_logits: Optional[torch.Tensor] = None # (batch_size, vocab_size) has value when doing logit sampling


class PrefillExecutor():
	"""
		PrefillExecutor manages the prefill process, including configuration, execution, and cleanup.
		It interacts with the model, engine, and parallel manager to perform prefill operations.
	"""
	def __init__(self, model_config, engine_config, core_engine, parallel_manager, comm):
		self.model_config = model_config
		self.engine_config = engine_config
		self.parallel_manager = parallel_manager
		self.core_engine = core_engine
		self.comm = comm
		self.model = None

		# For convenience
		self.rank = dist.get_rank()
	
	def _configure_prefill(self):
		# TODO: Revise configuration steps
		self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_prefill_buffer()
		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
		self.core_engine.clear_kv_storage()
		self.core_engine.start_h2d_worker()

		# Model-specific adjustments
		self._configure_model_specifics(self)

	def _configure_model_specifics(self):
		# TODO: will be removed after fully refactoring of model forward pass
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = False
		# Add other model-specific configurations as needed

	def _run_prefill(self, requests: PrefillRequest) -> PrefillResult:
		prefill_global_batch_size = self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size
		num_prefill_micro_batches = math.ceil(requests.batch_size / prefill_global_batch_size)
		prefill_micro_batch_input_ids = torch.split(
			requests.input_ids, prefill_global_batch_size
		)
		prefill_micro_batch_attention_masks = torch.split(
			requests.attention_mask, prefill_global_batch_size
		)
		prefill_micro_batch_position_ids = torch.split(
			requests.position_ids, prefill_global_batch_size
		)
		uuid_batches = [requests.sequence_uuids[i * prefill_global_batch_size:(i + 1) * prefill_global_batch_size] for i in range(num_prefill_micro_batches)]
		if self.rank == 0:
			logging.info(f"[PREFILL] Micro batches: {num_prefill_micro_batches}")
		all_tokens = []
		# all_logits = [] if self.engine_config.Prefill_Config.return_logits else None
		with torch.inference_mode():
			for micro_batch_idx in tqdm.tqdm(range(num_prefill_micro_batches), desc="Prefill Micro Batch"):
				Attn_Wrapper.attention_mask = prefill_micro_batch_attention_masks[micro_batch_idx]
				Attn_Wrapper.position_ids = create_position_ids_from_attention_mask(
					prefill_micro_batch_attention_masks[micro_batch_idx]
				)
				Attn_Wrapper.cur_batch = uuid_batches[micro_batch_idx]

				outputs = self.model(
					prefill_micro_batch_input_ids[micro_batch_idx].to(self.model_config.device),
					attention_mask=prefill_micro_batch_attention_masks[micro_batch_idx].to(self.model_config.device),
					use_cache=True,
					position_ids=prefill_micro_batch_position_ids[micro_batch_idx].to(self.model_config.device),
				)
				# Greedy decoding for the first token
				new_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1).view(-1, 1)
				all_tokens.append(new_tokens)
				# if self.engine_config.Prefill_Config.return_logits:
				# 	all_logits.append(outputs.logits[:, -1, :].cpu())
		return PrefillResult(
			sequence_uuids=requests.sequence_uuids,
			first_tokens=torch.cat(all_tokens, dim=0),
			# first_token_logits=torch.cat(all_logits, dim=0) if self.engine_config.Prefill_Config.return_logits else None
		)

	def _cleanup_prefill(self):
		# TODO: Revise cleanup steps
		self.model = deep_free_model_memory(self.model)

	def _prepare_requests(self, sequence_batch: SequenceBatch) -> PrefillRequest:
		"""
			Prepare PrefillRequest from SequenceBatch.
			Args:
				sequence_batch (SequenceBatch): The batch of sequences being processed.
			Returns:
				PrefillRequest: The prepared request for prefill.
		"""
		sequence_uuids = []
		input_id_list = []
		prompt_lengths = []

		for seq in sequence_batch.sequences.values():
			if seq.status != 1: # IN_PREFILL
				raise ValueError(f"Sequence {seq.uuid} not in IN_PREFILL status.")
			if seq.input_ids is None:
				raise ValueError(f"Input IDs for sequence {seq.uuid} not set.")
			sequence_uuids.append(seq.uuid)
			input_id_list.append(seq.input_ids)
			prompt_lengths.append(seq.prompt_length)
		
		input_ids = torch.nn.utils.rnn.pad_sequence(input_id_list, batch_first=True, padding_value=0)
		attention_mask = (input_ids != 0)
		position_ids = create_position_ids_from_attention_mask(attention_mask)

		return PrefillRequest(
			sequence_uuids=sequence_uuids,
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			prompt_lengths=prompt_lengths
		)

	def _update_sequence_batch(self, sequence_batch: SequenceBatch, result: PrefillResult) -> SequenceBatch:
		"""
			Update SequenceBatch based on PrefillResult.
			Args:
				sequence_batch (SequenceBatch): The batch of sequences being processed.
				result (PrefillResult): The result from the prefill stage.
			Returns:
				SequenceBatch: The updated sequence batch.
		"""
		for uuid, new_token in zip(result.sequence_uuids, result.first_tokens):
			seq = sequence_batch.get_sequence(uuid)
			if seq is None:
				logging.error(f"Sequence with UUID {uuid} not found in batch.")
				raise KeyError(f"Sequence with UUID {uuid} not found in batch.")
			# Update sequence with new token
			if seq.input_ids is None:
				seq.input_ids = new_token.unsqueeze(0)  # Initialize input_ids if None
			else:
				seq.input_ids = torch.cat([seq.input_ids, new_token.unsqueeze(0)], dim=1)
			seq.decoded_length += 1
			seq.current_context_length += 1
			# Transition status to PREFILLED
			seq.status_transition(SequenceStatus.PREFILLED)
		return sequence_batch
		
	def execute(self, sequence_batch: SequenceBatch):
		"""
			Execute the prefill process for a batch of requests.
			Args:
				sequence_batch (SequenceBatch): The batch of sequences being processed.
			Returns:
				torch.Tensor: The new tokens generated during prefill.

		"""
		try:
			requests = self._prepare_requests(sequence_batch)
			self._configure_prefill()
			result = self._run_prefill(requests) #result: PrefillResult
			sequence_batch = self._update_sequence_batch(sequence_batch, result)
			return sequence_batch
		finally:
			self._cleanup_prefill()

	

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



