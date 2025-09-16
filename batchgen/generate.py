import torch
import math
import logging
import tqdm
import os
import time
import gc
import torch.distributed as dist

from batchgen.prefill import Prefill
from batchgen.decode import Decode
from batchgen.distributed.utils import StatelessProcessGroup
from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
from batchgen.task_scheduler import QueryScheduler



class Generate():
	"""
		Give a batch of sequences, schedule the prefill and decode stages.
	"""
	def __init__(self, model_config, engine_config, core_engine, parallel_manager, task_scheduler):
		self.model_config = model_config
		self.engine_config = engine_config
		self.core_engine = core_engine
		self.parallel_manager = parallel_manager
		self.task_scheduler = task_scheduler
		
		self.comm = self._init_communicator()
		self.prefill = Prefill(model_config, engine_config, core_engine, parallel_manager, self.comm)
		self.decode = Decode(model_config, engine_config, core_engine, parallel_manager, self.comm)


	
	def run(self):
		while self.task_scheduler.has_pending_queries():
			task = self.task_scheduler.get_next_task()
			if task.type == "prefill":
				self.prefill.config_prefill()
				new_token = self.prefill(task.batch)
				self.task_scheduler.update_new_token(new_token, task.batch)
				self.prefill.cleanup_prefill()
			elif task.type == "decode":
				self.decode.config_decode(len(task.batch), self.comm)
				self.decode(task.new_token, task.batch)
				self.task_scheduler.update_new_token(task.new_token, task.batch)
				self.decode.cleanup_decode()
			else:
				logging.error(f"Unknown task type: {task.type}")
				raise ValueError(f"Unknown task type: {task.type}")

			self.task_scheduler.update_query_status(task.batch)

	def run_bak(self):
		prefill_time = 0
		decoding_time = 0
		if len(self.model_batches) > 0:
			for model_batch_idx in tqdm(
				range(len(self.model_batches)), desc="Model Batch"
			):
				dist.barrier()
				logging.info(f"Rank: {self.rank} pre-prefill barrier done.")
				self._config_prefill()
				logging.info(f"Rank: {self.rank} prefill config done.")
				prefill_start_time = time.perf_counter()
				with torch.inference_mode():
					new_token = self.prefill(self.model_batches[model_batch_idx])
				prefill_time += time.perf_counter() - prefill_start_time
				self._unregister_fp8_weights()
				dist.barrier()

				# log new_tokens from prefill:
				if self.rank == 0:
					logging.info(
						f"Model batch {model_batch_idx} prefill new tokens: {new_token.squeeze().tolist()}"
					)


				# Random create new token.
				# new_token = torch.randint(
				#     0,
				#     1000,
				#     # 129280, # self.model_config.vocab_size,
				#     (len(self.model_batches[model_batch_idx]), 1),
				#     device=self.torch_device,
				# )
				# self.update_new_token(new_token, self.model_batches[model_batch_idx], 0)
				# logging.info("Entering kv_storage creation...")
				# self.core_engine.create_fake_kv_storage()
				# self.core_engine.start_h2d_worker()
				# time.sleep(2)
					
				self._config_decoding(len(new_token), self.comm)
				# self.core_engine.copy_kv_to_worker(self.model_batches[model_batch_idx], self.max_input_length + self.max_decoding_length)
				if self.engine_config.Basic_Config.attn_mode == 3:
					# FULL GPU DECODING MODE.
					if self.model_config.model_type == "deepseek_v3":
						past_key_states= self.core_engine.get_past_key_states(self.model_batches[model_batch_idx], self.max_input_length + self.max_decoding_length)
						past_value_states = None
						# scale_dict = self.core_engine.get_kv_scale(self.model_batches[model_batch_idx], self.max_input_length)
						scale_dict = self.core_engine.get_kv_scale(self.model_batches[model_batch_idx], self.max_input_length + self.max_decoding_length)
						
				
					else:
						# TODO:
						pass
				
				
				dist.barrier()
				torch.cuda.empty_cache()
				decoding_start_time = time.perf_counter()
				with torch.inference_mode():
					logging.info(
						f"decoding batch size: {len(self.model_batches[model_batch_idx])}"
					)
					self.decoding(new_token, self.model_batches[model_batch_idx], past_key_states, past_value_states, scale_dict)
				decoding_time += time.perf_counter() - decoding_start_time
				self.core_engine.clear_kv_storage()
				self._unregister_fp8_weights()
				self.deep_free_model_memory()
				del past_key_states
				del past_value_states
				del scale_dict
				gc.collect()
				torch.cuda.empty_cache()
		else:
			# For small input batch, some worker might do not have any input.
			# In this case, it only participate in the decoding phase.
			# Todo: 
			self._config_decoding(0)

			# Log used memory before decoding
			if self.rank == 0:
				free_memory, total_memory = torch.cuda.mem_get_info()
				free_memory = free_memory / 1024 / 1024 / 1024
				total_memory = total_memory / 1024 / 1024 / 1024
				logging.info(
					f"Rank: {self.rank} Device torch memory usage before decoding: {torch.cuda.memory_allocated(self.torch_device) / (1024**3)} GB / {total_memory} GB"
				)
				logging.info(
					f"Rank: {self.rank} Device torch free memory before decoding: {free_memory} GB / {total_memory} GB"
				)
			dist.barrier()
			torch.cuda.empty_cache()
			decoding_start_time = time.perf_counter()
			with torch.inference_mode():
				self.decoding(None, None)
			decoding_time += time.perf_counter() - decoding_start_time
			self.core_engine.clear_kv_storage()


		
		dist.barrier()
		self.model = None 
		# torch.cuda.empty_cache()

		logging.info(
			f"Rank {self.rank} Prefill total time: {prefill_time:.1f} seconds,\n"
			f"Decoding total time: {decoding_time:.1f} seconds,\n"
			f"Waiting for process clean up..."
		)

		res = [
			self.query_book[query_idx].decoded_tokens
			for query_idx in range(self.num_queries)
		]

		# Print first 5 sequences
		for query_idx in range(5):
			logging.info(
				f"Decoded tokens: {res[query_idx].squeeze().tolist()[:20]}"
			)

		# Gather results from all rank to rank 0
		# logging.info(f"Rank {self.rank} res: {res}")
		all_results = [None] * self.world_size
		dist.all_gather_object(all_results, res)
		dist.destroy_process_group()
		all_results = [item for sublist in all_results for item in sublist]
		# logging.info(f"Size of all_results: {len(all_results)}")
		# Concat to a single tensor and copy to cpu
		res_tensor = torch.cat(all_results, dim=0).cpu()
		# logging.info(f"res_tensor shape {res_tensor.shape}")
		if self.rank == 0:
			return [res_tensor]
		else:
			return []


	def _init_communicator(self):
		self.rank = dist.get_rank()
		self.world_size = dist.get_world_size()
		device = torch.device("cuda", self.rank % torch.cuda.device_count())
		comm_master_addr = os.getenv("COMM_MASTER_ADDR")
		try:
			group = StatelessProcessGroup.create(
				host=comm_master_addr,
				port=20001,
				rank=self.rank,
				world_size=self.world_size,
				data_expiration_seconds=6000,
			)
			comm = PyNcclCommunicator(
				group=group,
				device=device
			)	
			return comm	
		except Exception as e:
			logging.error(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")
			raise RuntimeError(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")