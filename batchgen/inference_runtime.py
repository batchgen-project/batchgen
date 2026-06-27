import torch
import math
import logging
import tqdm
from typing import Optional, List, Dict
import torch.distributed as dist
from dataclasses import dataclass

from batchgen.get_parallel_strategy_manager import get_parallel_strategy_manager
# Use new wrapper system - Attn_Wrapper/Expert_Wrapper are aliases for backward compatibility
from batchgen.models.wrappers import BaseModuleWrapper, AttnWrapperBase, ExpertWrapperBase
Attn_Wrapper = AttnWrapperBase
Expert_Wrapper = ExpertWrapperBase
from batchgen.model_instance import ModelForwardInput, ModelForwardOutput
from batchgen.utils import deep_free_model_memory

"""
	This class encapsulates the inference runtime. It provides following APIs:
	- config_prefill: configure the prefill phase
	- prefill: pass the input to self.model
	- config_decode: configure the decode phase
	- decode: pass the input to self.model and do one forward pass.
	- get_current_phase: get the current phase, prefill or decode
"""
class InferenceRuntime:
	def __init__(self, model_name: str, engine_config, model_config, loaded_model_config, core_engine, skeleton_state_dict, local_rank:int, global_rank:int, world_size:int):
		""" Init variables """
		self.model_name = model_name
		self.engine_config = engine_config
		self.model_config = model_config
		self.loaded_model_config = loaded_model_config
		self.core_engine = core_engine
		self.skeleton_state_dict = skeleton_state_dict
		self.local_rank = local_rank
		self.global_rank = global_rank
		self.world_size = world_size

		""" Init model initializer, currently named parallel manager """
		self.parallel_manager = get_parallel_strategy_manager(self.model_name)
		self.parallel_manager = self.parallel_manager(
			self.loaded_model_config,
			self.engine_config,
			self.model_config,
			self.core_engine,
			self.skeleton_state_dict,
			self.local_rank,
			self.global_rank,
			self.world_size
		) 

		self.current_phase = None
		self.model_instance = None
		self.weight_copy_task = None

	def config_prefill(self, **kwargs):
		"""
		Configure the prefill phase with necessary parameters.
		"""
		phase = "prefill"
		self.model_instance, self.weight_copy_task = self.parallel_manager.configure_prefill()
		self.core_engine.set_phase(phase)
		Attn_Wrapper.phase = phase
		Expert_Wrapper.phase = phase
		BaseModuleWrapper.phase = phase
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_prefill_buffer()
		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
		self.core_engine.clear_kv_storage()
		self.core_engine.start_h2d_worker()
		self.current_phase = phase
		if self.global_rank == 0:
			logging.info("End Config Prefill")

	def prefill(self, input_batch: ModelForwardInput) -> ModelForwardOutput:
		"""
		Perform the prefill phase with the given input batch.
		"""
		assert self.model_instance.phase == "prefill", "Model instance is not in prefill phase"
		return self.model_instance.forward(input_batch)

	def cleanup_prefill(self):
		deep_free_model_memory(self.model_instance)
		self.model_instance = None
		self.weight_copy_task = None
		self.current_phase = None

	def config_decode(self, **kwargs):
		"""
		Configure the decode phase with necessary parameters.
		"""
		num_seq = kwargs.get("num_seq", None)
		comm = kwargs.get("comm", None)
		assert num_seq is not None, "num_seq must be provided for decode configuration"
		assert comm is not None, "comm must be provided for decode configuration"
		# Get number of sequences for each rank 
		num_seq_per_rank = torch.zeros(self.world_size, dtype=torch.int32, device=self.torch_device)
		num_seq_per_rank[self.rank] = kwargs.get("num_seq", None)
		dist.all_reduce(num_seq_per_rank, op=dist.ReduceOp.SUM)
		# Get the maximum number of sequences across all ranks
		max_num_seq = int(num_seq_per_rank.max().item())

		# Unified method handles all deployment scenarios (multi-node, single-node with/without offloading)
		phase = "decode"
		self.model, self.weight_copy_task = self.parallel_manager.configure_decoding(
			padding_bsz=max_num_seq, comm=comm
		)
		# Quiesce the H2D producer BEFORE set_phase("decode"): set_phase ->
		# resize_buffer -> reset_slot_events destroys/recreates the per-slot CUDA
		# fence events that the still-live prefill producer reads lock-free
		# (cudaStreamWaitEvent). Stopping first makes all buffer/event mutation
		# happen with the producer dead (mirrors config_prefill stop-then-reset).
		self.core_engine.stop_h2d_worker()
		self.core_engine.set_phase(phase)
		Attn_Wrapper.phase = phase
		Expert_Wrapper.phase = phase
		BaseModuleWrapper.phase = phase
		self.core_engine.clear_kv_copy_queue()
		self.core_engine.clear_kv_buffer()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_decoding_buffer()
		# Only start H2D worker if there are experts to offload
		if self.weight_copy_task.get("routed_expert"):
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			self.core_engine.start_h2d_worker()

		self.current_phase = phase
		if self.rank == 0:
			logging.info("End Config Decoding")

	def decode(self, decode_batch):
		"""
		Perform one step of decoding with the given decode batch.
		"""
		assert self.model_instance.phase == "decode", "Model instance is not in decode phase"
		return self.model_instance.forward(decode_batch)

	def cleanup_decode(self):
		deep_free_model_memory(self.model_instance)
		self.model_instance = None
		self.weight_copy_task = None
		self.current_phase = None

	def get_current_phase(self) -> str:
		"""
		Return the current phase: 'prefill' or 'decode'.
		"""
		return self.current_phase
