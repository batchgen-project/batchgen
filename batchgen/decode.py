import torch
import math
import logging
import tqdm
from typing import Optional, List, Dict
import torch.distributed as dist
from dataclasses import dataclass

from batchgen.models.Wrapper import Attn_Wrapper
from batchgen.utils import create_position_ids_from_attention_mask
from batchgen.sampling import sample_with_temperature_top_p, greedy_decode
from batchgen.sequence import SequenceBatch

@dataclass
class DecodeInput:
	"""
	DecodeRequest holds the input data for model forward pass in decode stage.
	"""
	sequence_uuids: List[int]
	input_tokens: torch.Tensor  # (batch_size, 1) - current tokens to decode
	attention_mask: torch.Tensor # 
	
	@property
	def batch_size(self):
		return len(self.sequence_uuids)

@dataclass
class DecodeOutput:
    """
    DecodeStepResult encapsulates the output from a single decode step.
    """
    sequence_uuids: List[int]
    new_tokens: torch.Tensor  # (batch_size, 1)
    # finished_uuids: List[int]  # Sequences that reached EOS or max length



class DecodeExecutor:
	"""
	DecodeExecutor manages the decoding process with a preset number of steps(e.g. 1 or page size). 
	It only handles the core logic of decoding. I.e. decoding the input batch for a few steps and update the status of each sequence.

	"""

	"""
	Deprecated:
	Core logic:
	1. Configure the engine for decoding phase.
	2. After each forward pass:
		- Update newly generated tokens.
		- Check EOS, max_decode_tokens, and context window limits.
		- Mark completed sequences as COMPLETED.
		- If continuous_batching=True: backfill with PREFILLED sequences.
		- If continuous_batching=False: decode batch to completion, then load next batch from PREFILLED.
	3. Stop signal: decode batch is empty (no active sequences).
		- For continuous_batching=True, this means no enough PREFILLED sequences so BatchGen shift back to prefill phase or all the sequences are completed.
		- For continuous_batching=False, this means all PREFILLED sequences are processed. 
			BatchGen shift back to prefill phase or terminate.
	"""
	def __init__(self, model_config, engine_config, core_engine, parallel_manager, comm, 
			  decode_batch: SequenceBatch, decode_steps=1):
		self.model_config = model_config
		self.engine_config = engine_config
		self.parallel_manager = parallel_manager
		self.core_engine = core_engine
		self.comm = comm
		self.decode_batch = decode_batch # A view from global batch.
		self.decode_step = decode_steps # Number of decode steps in one execute() call.

		self.rank = self.engine_config.Basic_Config.rank
		self.world_size = self.engine_config.Basic_Config.world_size
		self.torch_device = self.engine_config.Basic_Config.device_torch
	
	def _prepare_forward_input(self) -> DecodeInput:
		"""
		Prepare the input dictionary for model forward pass.
		1. Gather current tokens from decode_batch.
		2. Create attention masks and position ids if needed.
		3. Return a dictionary with all necessary inputs for the model.
		"""
		pass

	def _decode_one_step(self) -> DecodeOutput:
		"""
		Perform a single decode step for the given DecodeRequest.
		1. Run model forward pass.
		2. Process the output to get new tokens.
		3. Identify finished sequences.
		4. Return DecodeStepResult with new tokens and finished sequence UUIDs.
		"""
		pass

	def _update_sequences(self, decode_result:dict):
		"""
		Update the sequences in decode_batch based on the DecodeStepResult.
		1. Append new tokens to each sequence.
		2. Update sequence status (ACTIVE, COMPLETED).
		3. Handle context window overflow if necessary.
		"""
		pass

	def execute(self)		
		"""
		Decode the current decode_batch for a preset number of steps.
		1. Prepare the input DecodeRequest.
		2. Run model forward.
		3. Process the output DecodeStepResult.
		4. Return the finished sequences and the number of active sequences remaining.
		"""
		cur_step = 0
		while cur_step < self.decode_step:
			cur_decode_result = self._decode_one_step(self.decode_batch)
			self._update_sequences(cur_decode_result)
			cur_step += 1




class Decode():
	def __init__(self, model_config, engine_config, core_engine, parallel_manager, comm):
		self.model_config = model_config
		self.engine_config = engine_config
		self.parallel_manager = parallel_manager
		self.core_engine = core_engine
		self.comm = comm

		self.rank = self.engine_config.Basic_Config.rank
		self.world_size = self.engine_config.Basic_Config.world_size
		self.torch_device = self.engine_config.Basic_Config.device_torch

	def config_decode(self, num_seq, comm=None):
		logging.info(f"Start Config Decoding")
		self.deep_free_model_memory()


		# Get number of sequences for each rank 
		num_seq_per_rank = torch.zeros(self.world_size, dtype=torch.int32, device=self.torch_device)
		num_seq_per_rank[self.rank] = num_seq
		dist.all_reduce(num_seq_per_rank, op=dist.ReduceOp.SUM)
		# Get the maximum number of sequences across all ranks
		max_num_seq = int(num_seq_per_rank.max().item())


		# TODO:
		if self.world_size <= 8:
			self.model, self.weight_copy_task = self.parallel_manager.configure_decoding()
			self.set_phase("decoding")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			self.core_engine.clear_kv_buffer()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			self.core_engine.start_h2d_worker()
		else:
			self.model, self.weight_copy_task = self.parallel_manager.pure_gpu_decoding(max_num_seq, comm)

			self.set_phase("decoding")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			self.core_engine.clear_kv_buffer()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()

		logging.info(f"{self.rank} End Config Decoding")

	def cleanup_decode(self):
		pass


	def run(
		self, 
		new_tokens: torch.Tensor, 
		batch: list[int],
		past_key_states: Optional[torch.Tensor] = None,
		past_value_states: Optional[torch.Tensor] = None,
		scale_dict: Optional[dict] = None,
	):
		"""
		Handle the decoding for a full model batch.
		All the queries reach <EOS> or the max decoding length.

		return
				- answer_set: dict[query_idx, decoded_tokens]
		"""
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = True
		new_token_idx = 1

		RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode
		# Log device memory usage
		logging.info(f"{self.rank} Device memory usage: {torch.cuda.memory_allocated(self.torch_device) / (1024**3)} GB")

		if RUNTIME_ATTN_MODE == 3:
			"""
				KV ACCUMULATION IN GPU.`
			"""
			Attn_Wrapper.scale = scale_dict
			Attn_Wrapper.past_key_states = past_key_states
			Attn_Wrapper.past_value_states = past_value_states
			while new_token_idx < self.max_decoding_length and len(batch) > 0:
				# Log for every 50 tokens.
				if self.rank == 0 and new_token_idx % 50 == 0:
					logging.info(f"Decoding new token idx: {new_token_idx}")
				with torch.inference_mode():
					attention_mask = torch.cat(
						[
							self.query_book[query_idx].encoded[
								"attention_mask"
							][:, : self.max_input_length + new_token_idx]
							for query_idx in batch
						],
						dim=0,
					).to(self.torch_device)
					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = (attention_mask.sum(-1) - 1).unsqueeze(-1)
					Attn_Wrapper.cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
					Attn_Wrapper.max_seqlen = Attn_Wrapper.cache_seqlens.max().item()

					new_tokens = self.model(
						new_tokens.to(self.torch_device),
						attention_mask=attention_mask.to(self.torch_device),
						# position_ids=position_ids.to(self.torch_device),
						use_cache=False,
					)
					new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
						-1, 1
					)
					self.update_new_token(new_tokens, batch, new_token_idx)
				new_token_idx += 1
			Attn_Wrapper.scale = None
			Attn_Wrapper.past_key_states = None
			Attn_Wrapper.past_value_states = None
		
		
		else:
			while new_token_idx < self.max_decoding_length and len(batch) > 0:
				if self.rank == 0:
					logging.info(f"Decoding new token idx: {new_token_idx}")
				# Step 1: Before each round of decoding, review the attention mode and batching plan.
				# TODO: review attention mode. Current fixing attention mode.
				RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode
				# logging.info(f"RUNTIME_ATTN_MODE: {RUNTIME_ATTN_MODE}")

				if RUNTIME_ATTN_MODE == 0:
					"""
						CPU ATTN MODE
							- NO ATTN MICRO BATCH
					"""
					# self.set_attn_mode(0)
					# self.core_engine.set_attn_mode(0)
					with torch.inference_mode():
						Attn_Wrapper.cur_batch = [batch]
						attention_mask = torch.cat(
							[
								self.query_book[query_idx].encoded[
									"attention_mask"
								][:, : self.max_input_length + new_token_idx]
								for query_idx in batch
							],
							dim=0,
						)
						if "deepseek" not in self.model_config.model_type:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)[:, -1].unsqueeze(-1)
						else:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)
						# DeepSeek use flash-attn by default
						
						# if attention_mask.dim() == 2 and (
						#     self.model_config.model_type not in ["Qwen2", "deepseek"]
						# ):
						#     attention_mask = attention_mask.unsqueeze(1).unsqueeze(
						#         2
						#     )
						#     attention_mask = torch.where(
						#         attention_mask == 0,
						#         torch.finfo(torch.bfloat16).min,
						#         torch.tensor(
						#             0.0,
						#             dtype=torch.bfloat16,
						#             device=attention_mask.device,
						#         ),
						#     )

						Attn_Wrapper.attention_mask = attention_mask
						Attn_Wrapper.position_ids = position_ids
						new_tokens = self.model(
							new_tokens.to(self.torch_device),
							attention_mask=attention_mask.to(self.torch_device),
							# position_ids=position_ids.to(self.torch_device),
							use_cache=False,
						)
						new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
							-1, 1
						)
						# logging.info(f"New tokens: {new_tokens}")
						# start = time.perf_counter()
						self.update_new_token(new_tokens, batch, new_token_idx)
						# logging.info(
						#     f"Update new token time is ms: {(time.perf_counter() - start) * 1000} ms"
						# )

					# TODO: Temporally remove.
					# Check <EOS>, if <EOS>, remove from batch.
					# for idx, query_idx in enumerate(batch):
					# 	if new_tokens[idx] == self.tokenizer.eos_token_id:
					# 		batch.remove(query_idx)
					new_token_idx += 1

				elif RUNTIME_ATTN_MODE == 1:
					"""
						GPU ATTN MODE
							- ATTN MICRO BATCH
					"""
					# Submit KV copy task to the core engine.
					micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
					num_micro_batches = math.ceil(len(batch) / micro_batch_size)
					# logging.info(f"num_micro_batches: {num_micro_batches}")
					micro_batches = [
						batch[
							micro_batch_idx * micro_batch_size : (
								micro_batch_idx + 1
							)
							* micro_batch_size
						]
						for micro_batch_idx in range(num_micro_batches)
					]
					Attn_Wrapper.cur_batch = micro_batches
					# TODO: init ModelConfig in the initializer.
					# Resub every 32 new tokens.
					if (new_token_idx - 1) % 32 == 0:
						for idx in range(new_token_idx - 1, new_token_idx + 31):
							# Note: DeepSeek use fp8 kv.
							if "deepseek" in self.model_config.model_type:
								# past_kv_byte_size = (
								#     (self.max_input_length + idx)
								#     * self.model_config.compressed_kv_dim
								#     * 2
								# )
								# past_kv_byte_size = (
								#     (self.max_input_length + idx)
								#     * self.model_config.compressed_kv_dim
								# )

								# Copy one more token to avoid torch::cat in attention forward.
								past_kv_byte_size = (
									(self.max_input_length + idx + 1)
									* self.model_config.compressed_kv_dim
								)

							elif "mixtral" in self.model_config.model_type:
								past_kv_byte_size = (
									(self.max_input_length + idx)
									* self.model_config.num_key_value_heads
									* self.model_config.head_dim
									* 2
								)
							else:
								raise ValueError(
									f"Model architecture {self.model_config.model_type} not supported yet."
								)

							for layer_idx in range(
								self.model_config.num_hidden_layers
							):
								for micro_batch_idx in range(num_micro_batches):
									cur_batch = micro_batches[micro_batch_idx]
									# logging.info(f"token idx: {idx}, layer idx: {layer_idx}, micro_batch_idx: {micro_batch_idx} current batch: {cur_batch}")
									self.core_engine.submit_to_KV_queue(
										cur_batch,
										micro_batch_idx,
										layer_idx,
										past_kv_byte_size,
									)

					with torch.inference_mode():
						attention_mask = torch.cat(
							[
								self.query_book[query_idx].encoded[
									"attention_mask"
								][:, : self.max_input_length + new_token_idx]
								for query_idx in batch
							],
							dim=0,
						).to(self.torch_device)
						if "deepseek" in self.model_config.model_type:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)
						else:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)[:, -1].unsqueeze(-1)

						# if attention_mask.dim() == 2 and (
						#     self.model_config.model_type not in ["Qwen2", "deepseek"]
						# ):
						#     attention_mask = attention_mask.unsqueeze(1).unsqueeze(
						#         2
						#     )
						#     attention_mask = torch.where(
						#         attention_mask == 0,
						#         torch.finfo(torch.bfloat16).min,
						#         torch.tensor(
						#             0.0,
						#             dtype=torch.bfloat16,
						#             device=attention_mask.device,
						#         ),
						#     )

						Attn_Wrapper.attention_mask = attention_mask
						Attn_Wrapper.position_ids = position_ids
						# logging.info(f"rank: {self.rank} attention_mask: {attention_mask}")
						# logging.info(f"rank: {self.rank} position_ids: {position_ids}")
						new_tokens = self.model(
							new_tokens.to(self.torch_device),
							attention_mask=attention_mask.to(self.torch_device),
							# position_ids=position_ids.to(self.torch_device),
							use_cache=False,
						)
						# torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
						# torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
						new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
							-1, 1
						)
						self.update_new_token(new_tokens, batch, new_token_idx)
						# torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
						# torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
						# logging.info(f"New tokens: {new_tokens}")
					new_token_idx += 1

					# Step 1.1 Config new micro_batch size. Magic Number change every 32 new tokens.
					# seq_len = self.query_book[batch[0]].encoded["input_ids"].shape[1] + self.query_book[batch[0]].num_decoded_tokens
					# ATTN_DECODING_MICRO_BATCH_SIZE = self.engine_config.GPU_Buffer_Config.k_buffer_num_tokens // seq_len
				elif RUNTIME_ATTN_MODE == 2:
					"""
						CPU-GPU Parallel ATTN.
						Deprecated.
					"""
					w = float(os.getenv("SPLIT_RATIO_W", None))
					if w is None:
						logging.info(
							f"CPU compute ratio not set. Default setting applied."
						)
						w = 0.6
					logging.info(f"Split ratio: {w}")
					# TODO: wordload partitioning.
					CPU_batch = batch[: math.ceil(len(batch) * w)]
					GPU_batch = batch[math.ceil(len(batch) * w) :]
					logging.info(
						f"CPU batch size: {len(CPU_batch)}, GPU batch size: {len(GPU_batch)}"
					)

					GPU_micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
					num_GPU_micro_batches = math.ceil(
						len(GPU_batch) / GPU_micro_batch_size
					)
					GPU_micro_batches = [
						GPU_batch[
							micro_batch_idx * GPU_micro_batch_size : (
								micro_batch_idx + 1
							)
							* GPU_micro_batch_size
						]
						for micro_batch_idx in range(num_GPU_micro_batches)
					]
					Attn_Wrapper.cur_batch = [CPU_batch] + GPU_micro_batches
					# TODO:
					if (new_token_idx - 1) % 32 == 0:
						for idx in range(new_token_idx - 1, new_token_idx + 31):
							if "deepseek" in self.model_config.model_type:
								past_kv_byte_size = (
									(self.max_input_length + idx)
									* self.model_config.compressed_kv_dim
									* self.engine_config.Basic_Config.torch_dtype.itemsize
								)
							else:
								past_kv_byte_size = (
									(self.max_input_length + idx)
									* self.model_config.num_key_value_heads
									* self.model_config.head_dim
									* self.engine_config.Basic_Config.torch_dtype.itemsize
								)

							if "deepseek" in self.model_config.model_type:
								for layer_idx in range(
									self.model_config.num_hidden_layers
								):
									self.core_engine.submit_to_KV_queue(
										cur_batch, 0, layer_idx, past_kv_byte_size
									)

							else:
								for layer_idx in range(
									self.model_config.num_hidden_layers
								):
									for micro_batch_idx in range(
										num_GPU_micro_batches
									):
										cur_batch = GPU_micro_batches[
											micro_batch_idx
										]
										self.core_engine.submit_to_KV_queue(
											cur_batch,
											micro_batch_idx,
											layer_idx,
											past_kv_byte_size,
										)

					with torch.inference_mode():
						attention_mask = torch.cat(
							[
								self.query_book[query_idx].encoded[
									"attention_mask"
								][:, : self.max_input_length + new_token_idx]
								for query_idx in batch
							],
							dim=0,
						)
						if "deepseek" not in self.model_config.model_type:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)[:, -1].unsqueeze(-1)
						else:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)
						if attention_mask.dim() == 2 and (
							self.model_config.model_type not in ["Qwen2"]
						):
							attention_mask = attention_mask.unsqueeze(1).unsqueeze(
								2
							)
							attention_mask = torch.where(
								attention_mask == 0,
								torch.finfo(torch.bfloat16).min,
								torch.tensor(
									0.0,
									dtype=torch.bfloat16,
									device=attention_mask.device,
								),
							)

						Attn_Wrapper.attention_mask = attention_mask
						Attn_Wrapper.position_ids = position_ids
						new_tokens = self.model(
							new_tokens.to(self.torch_device),
							attention_mask=attention_mask.to(self.torch_device),
							# position_ids=position_ids,
							use_cache=False,
						)
						new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
							-1, 1
						)
						self.update_new_token(new_tokens, batch, new_token_idx)
						print(f"New tokens: {new_tokens}")
					new_token_idx += 1
