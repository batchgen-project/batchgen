import torch
import math
import logging
import tqdm
from typing import Optional, List, Dict, Union
import torch.distributed as dist

from batchgen.query_manager import QueryManager

"""
	Support streaming query pool with query status tracking.
"""
class task():
	"""
		Attributes TBD.
	"""
	def __init__(self):
		self.type = None  # "prefill" or "decode"
		self.query_ids = []  # list of query IDs
		self.input_ids = None # Torch tensor of input_ids. Shape: (batch_size, seq_len)
		self.attention_mask = None # Torch tensor of attention_mask. Shape: (batch_size, seq_len)
		

class TaskScheduler():
	def __init__(self, QueryManager: QueryManager):
		self.query_manager = QueryManager
		self.scheduler_policy = self._scheduler_v1
		self.current_task = 'idle' # 'idle', 'prefill', 'decode'
		

	def has_pending_queries(self):
		return self.query_manager.has_pending_queries()
	
	def update_new_token(self, new_token, batch):
		return self.query_manager.update_new_token(new_token, batch)

	def get_next_task(self):
		"""
			Decide the next task to run: prefill or decode based current query status.
			Return a task object with type and batch of sequences.
			Task type: "prefill" or "decode"
			Task batch: a list of query IDs to process in this task.
		"""
		return self.scheduler_policy()

	def _scheduler_v1(self):
		"""
			Default scheduling policy:
			1. Prioritize prefill tasks until host prefill KV pool is full.
			2. Do decode tasks with continuous batching untill all the queries prefilled are completed. 
		"""
		next_task = task()
		# Start running
		if self.current_task == 'idle' and self.query_manager.has_pending_prefill_queries():
			self.current_task = 'prefill'
			next_task.type = 'prefill'
			next_task.query_ids, next_task.input_ids, next_task.attention_mask = self.query_manager.get_prefill_batch()
			return next_task

		# All queries are completed. System remain idle.
		if self.current_task == 'idle' and not self.query_manager.has_pending_prefill_queries() and self.query_manager.has_pending_decode_queries():
			next_task.type = 'completed'
			return next_task

		# Switch to decode task when host prefill KV pool is full.
		if self.current_task == 'prefill' and self.query_manager.host_kv_pool_full():
			self.current_task = 'decode'
			next_task.type = 'decode'
			next_task.query_ids, next_task.input_ids, next_task.attention_mask = self.query_manager.get_decode_batch()
			return next_task

		# Keep running decode tasks until all the queries prefilled are completed.
		if self.current_task == 'decode' and self.query_manager.has_pending_decode_queries():
			next_task.type = 'decode'
			next_task.query_ids, next_task.input_ids, next_task.attention_mask = self.query_manager.get_decode_batch()
			return next_task
		
		# If decode completed, switch to prefill
		if self.current_task == 'decode' and not self.query_manager.has_pending_decode_queries() and self.query_manager.has_pending_prefill_queries():
			self.current_task = 'prefill'
			next_task.type = 'prefill'
			next_task.query_ids, next_task.input_ids, next_task.attention_mask = self.query_manager.get_prefill_batch()
			return next_task


			





