"""Decode phase management for batch inference.

This module defines the decode phase execution, including token-by-token
generation with continuous batching support. The scheduler calls decode
after prefill to generate output tokens.

Classes:
    DecodeInput: Input data for decode model forward pass
    DecodeOutput: Output from a single decode step
    DecodeConfig: Configuration parameters for decode phase
    DecodeTimingStats: Timing statistics for decode operations
    DecodeExecutor: Manages single decode steps
    Decode: Legacy decode implementation (to be deprecated)
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import tqdm

from batchgen.config.config import EngineConfig, ModelConfig
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.models.engine_loader import core_engine as bg
# Use new wrapper system - Attn_Wrapper is alias for backward compatibility
from batchgen.models.wrappers import AttnWrapperBase
Attn_Wrapper = AttnWrapperBase
from batchgen.sampling import greedy_decode, sample_with_temperature_top_p
from batchgen.sequence import SequenceBatch
from batchgen.utils import create_position_ids_from_attention_mask

logger = logging.getLogger(__name__)

@dataclass
class DecodeInput:
	"""
	DecodeRequest holds the input data for model forward pass in decode stage.
	"""
	sequence_uuids: List[int]
	input_tokens: torch.Tensor  # (batch_size, 1) - current tokens to decode
	attention_mask: torch.Tensor # 
	position_ids: torch.Tensor  # (batch_size, seq_len)
	
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
	It only handles the core logic of decoding. 
	I.e. decoding the input batch for a few steps and update the status of each sequence.
	"""
	def __init__(self, model_config: ModelConfig, engine_config: EngineConfig, inference_runtime, 
			  		decode_batch: SequenceBatch, decode_steps=1):
		self.model_config = model_config
		self.engine_config = engine_config
		self.inference_runtime = inference_runtime
		self.decode_batch = decode_batch # A view from global batch.
		self.decode_step = decode_steps
		# Number of decode steps in one execute() call. This determines the granularity of pd scheduling.

		self.rank = self.engine_config.Basic_Config.rank
		self.world_size = self.engine_config.Basic_Config.world_size
		self.torch_device = self.engine_config.Basic_Config.device_torch

		self.decode_input = self._prepare_forward_input(decode_batch)

	def _prepare_forward_input(self, decode_batch: SequenceBatch) -> DecodeInput:
		"""
		Prepare the input dictionary for model forward pass.
		1. Gather current tokens from decode_batch.
		2. Create attention masks and position ids if needed.
		3. Return a dictionary with all necessary inputs for the model.
		"""
		attention_mask = decode_batch.get_attention_mask()
		return DecodeInput(
			sequence_uuids=decode_batch.get_sequence_uuids(),
			input_tokens=decode_batch.get_last_tokens(),
			attention_mask=attention_mask,
			position_ids=create_position_ids_from_attention_mask(attention_mask)
		)

	def _decode_one_step(self) -> DecodeOutput:
		"""
		Perform a single decode step for the given DecodeRequest.
		1. Run model forward pass.
		2. Process the output to get new tokens.
		3. Identify finished sequences.
		4. Return DecodeStepResult with new tokens and finished sequence UUIDs.
		"""
		return self.inference_runtime.decode(self.decode_input)
		

	def _update_sequences(self, decode_result:DecodeOutput):
		"""
		Update the sequences in decode_batch based on the DecodeStepResult.
		1. Append new tokens to each sequence.
		2. Update sequence status (ACTIVE, COMPLETED).
		3. Handle context window overflow if necessary.
		"""
		self.decode_batch.update_result(decode_result.sequence_uuids, decode_result.new_tokens)

	def execute(self):	
		"""
		Decode the current decode_batch for a preset number of steps.
		1. Prepare the input DecodeRequest.
		2. Run model forward.
		3. Process the output DecodeStepResult.
		4. Return the finished sequences and the number of active sequences remaining.
		"""
		assert self.inference_runtime.get_current_phase() == "decode", "DecodeExecutor can only run in decode phase."
		cur_step = 0
		while cur_step < self.decode_step:
			cur_decode_result = self._decode_one_step(self.decode_batch)
			self._update_sequences(cur_decode_result)
			cur_step += 1




class Decode():
	def __init__(self, model_config: ModelConfig, engine_config: EngineConfig, core_engine, parallel_manager, comm):
		self.model_config = model_config
		self.engine_config = engine_config
		self.parallel_manager = parallel_manager
		self.core_engine = core_engine
		self.comm = comm

		self.rank = self.engine_config.Basic_Config.rank
		self.world_size = self.engine_config.Basic_Config.world_size
		self.torch_device = self.engine_config.Basic_Config.device_torch

		# Sampling parameters (None = greedy decoding)
		self._temperature: Optional[float] = None
		self._top_p: Optional[float] = None

		# self.gpu_paged_kv_manager = GPUPagedKVCacheManager(self.engine_config)

	def set_sampling_params(self, temperature: Optional[float] = None, top_p: Optional[float] = None) -> None:
		"""Set sampling parameters for token generation."""
		self._temperature = temperature
		self._top_p = top_p

	def _select_tokens(self, logits: torch.Tensor) -> torch.Tensor:
		"""
		Select next tokens from logits using greedy or sampling strategy.

		Args:
			logits: [batch_size, vocab_size] logits from model

		Returns:
			[batch_size, 1] selected token indices
		"""
		# Fast path: greedy decoding (default)
		if self._temperature is None or self._temperature <= 0:
			return torch.argmax(logits, dim=-1, keepdim=True)

		# Sampling with temperature/top_p
		from batchgen.sampling import sample_tokens
		return sample_tokens(logits, temperature=self._temperature, top_p=self._top_p)

	def config_decode(self, num_seq, comm=None):
		if self.rank == 0:
			logging.info("Start Config Decoding")
		self.deep_free_model_memory()

		# self.gpu_paged_kv_manager.initialize()

		# Get number of sequences for each rank 
		num_seq_per_rank = torch.zeros(self.world_size, dtype=torch.int32, device=self.torch_device)
		num_seq_per_rank[self.rank] = num_seq
		dist.all_reduce(num_seq_per_rank, op=dist.ReduceOp.SUM)
		# Get the maximum number of sequences across all ranks
		max_num_seq = int(num_seq_per_rank.max().item())


		# Unified method handles all deployment scenarios (multi-node, single-node with/without offloading)
		self.model, self.weight_copy_task = self.parallel_manager.configure_decoding(
			padding_bsz=max_num_seq, comm=comm
		)
		self.set_phase("decode")
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_kv_copy_queue()
		self.core_engine.clear_kv_buffer()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_decoding_buffer()
		# Only start H2D worker if there are experts to offload
		if self.weight_copy_task.get("routed_expert"):
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			self.core_engine.start_h2d_worker()

		if self.rank == 0:
			logging.info("End Config Decoding")

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
					new_tokens = self._select_tokens(new_tokens.logits)
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
						new_tokens = self._select_tokens(new_tokens.logits)
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
							elif "gpt_oss" in self.model_config.model_type:
								# GPT-OSS GQA: 8 KV heads, head_dim=64, separate K and V
								past_kv_byte_size = (
									(self.max_input_length + idx + 1)
									* self.model_config.num_key_value_heads
									* self.model_config.head_dim
									* 2  # K + V
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
						new_tokens = self._select_tokens(new_tokens.logits)
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
						new_tokens = self._select_tokens(new_tokens.logits)
						self.update_new_token(new_tokens, batch, new_token_idx)
						print(f"New tokens: {new_tokens}")
					new_token_idx += 1


# ============ Additional Decode Utilities ============


@dataclass
class DecodeConfig:
    """Configuration parameters for decode phase.

    This dataclass holds all configuration needed to set up and execute
    a decode phase with continuous batching support.
    """

    # Batch configuration
    max_batch_size: int = 256  # Maximum sequences per decode batch
    page_size: int = 64  # Tokens per page for KV cache

    # Generation configuration
    max_new_tokens: int = 128  # Maximum tokens to generate
    ignore_eos: bool = False  # Whether to ignore EOS tokens

    # Sampling configuration
    temperature: Optional[float] = None  # None = greedy
    top_p: Optional[float] = None  # None = disabled
    top_k: int = 0  # 0 = disabled

    # Continuous batching configuration
    use_continuous_batching: bool = True
    two_page_buffer: bool = True  # Allocate 2 pages initially

    # Performance configuration
    boundary_check_interval: int = 64  # Check page boundary every N tokens


@dataclass
class DecodeTimingStats:
    """Timing statistics for decode phase."""

    total_ms: float = 0.0
    config_ms: float = 0.0
    forward_ms: float = 0.0
    sampling_ms: float = 0.0
    kv_operations_ms: float = 0.0
    page_boundary_ms: float = 0.0

    # Counts
    num_steps: int = 0
    num_tokens_generated: int = 0
    num_sequences_completed: int = 0
    num_page_boundaries: int = 0

    def summary(self) -> str:
        """Return a one-line summary of timing."""
        tokens_per_sec = (
            self.num_tokens_generated / (self.total_ms / 1000)
            if self.total_ms > 0
            else 0
        )
        return (
            f"total={self.total_ms:.1f}ms, forward={self.forward_ms:.1f}ms, "
            f"tokens={self.num_tokens_generated}, tps={tokens_per_sec:.1f}"
        )


@dataclass
class DecodeBatchState:
    """State tracking for a decode batch during continuous batching.

    This class tracks the state of all sequences in a decode batch,
    enabling efficient updates and status queries.
    """

    # Sequence tracking
    active_uuids: List[str] = field(default_factory=list)
    completed_uuids: List[str] = field(default_factory=list)
    on_hold_uuids: List[str] = field(default_factory=list)

    # Token tracking
    current_step: int = 0
    tokens_in_page: int = 0  # Tokens generated since last page boundary

    # GPU state
    gpu_batch_indices: List[int] = field(default_factory=list)

    @property
    def num_active(self) -> int:
        """Number of actively decoding sequences."""
        return len(self.active_uuids)

    @property
    def num_completed(self) -> int:
        """Number of completed sequences."""
        return len(self.completed_uuids)

    def is_at_page_boundary(self, page_size: int = 64) -> bool:
        """Check if we're at a page boundary."""
        return self.tokens_in_page >= page_size

    def advance_step(self) -> None:
        """Advance to next decode step."""
        self.current_step += 1
        self.tokens_in_page += 1


def check_eos_batch(
    new_tokens: torch.Tensor,
    eos_token_id,
    sequence_uuids: List[str],
) -> Tuple[List[str], List[str]]:
    """Check for EOS tokens in a batch of new tokens.

    Args:
        new_tokens: [batch_size, 1] tensor of new token IDs
        eos_token_id: EOS token ID (int) or set/list of EOS token IDs
        sequence_uuids: List of sequence UUIDs corresponding to tokens

    Returns:
        (continuing_uuids, completed_uuids)
    """
    # Normalize to set for O(1) lookup
    if isinstance(eos_token_id, int):
        eos_token_id = {eos_token_id}
    eos_set = set(eos_token_id)

    # Flatten tokens for comparison
    tokens_flat = new_tokens.view(-1).cpu().tolist()

    continuing = []
    completed = []

    for uuid, token in zip(sequence_uuids, tokens_flat):
        if token in eos_set:
            completed.append(uuid)
        else:
            continuing.append(uuid)

    return continuing, completed


def check_max_length_batch(
    sequence_lengths: Dict[str, int],
    max_length: int,
    sequence_uuids: List[str],
) -> Tuple[List[str], List[str]]:
    """Check for sequences that have reached max length.

    Args:
        sequence_lengths: Dict mapping uuid -> current decoded length
        max_length: Maximum allowed length
        sequence_uuids: List of sequence UUIDs to check

    Returns:
        (continuing_uuids, completed_uuids)
    """
    continuing = []
    completed = []

    for uuid in sequence_uuids:
        length = sequence_lengths.get(uuid, 0)
        if length >= max_length:
            completed.append(uuid)
        else:
            continuing.append(uuid)

    return continuing, completed


def select_tokens_with_config(
    logits: torch.Tensor,
    config: DecodeConfig,
) -> torch.Tensor:
    """Select tokens using configuration.

    Args:
        logits: [batch_size, vocab_size] logits from model
        config: Decode configuration with sampling parameters

    Returns:
        [batch_size, 1] selected token indices
    """
    # Fast path: greedy decoding
    if config.temperature is None or config.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Import here to avoid circular import
    from batchgen.sampling import sample_tokens

    return sample_tokens(
        logits,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )


def create_decode_attention_mask(
    prompt_lengths: List[int],
    decoded_lengths: List[int],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Create attention mask for decode step.

    Args:
        prompt_lengths: List of prompt lengths per sequence
        decoded_lengths: List of decoded token counts per sequence
        max_length: Maximum sequence length for mask
        device: Device to create mask on

    Returns:
        [batch_size, max_length] attention mask
    """
    batch_size = len(prompt_lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.int64, device=device)

    for i, (prompt_len, decoded_len) in enumerate(zip(prompt_lengths, decoded_lengths)):
        total_len = prompt_len + decoded_len
        mask[i, :total_len] = 1

    return mask
