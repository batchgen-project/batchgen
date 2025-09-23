import torch
import math
import logging
import tqdm
from typing import Optional, List, Dict, Union, Tuple
import torch.distributed as dist


"""
	Data structure to manage queries including:
	1. The streaming query pool keep receiving incoming queries.
	2. For each query, we keep records of its status, e.g., prefill done or not, decode done or not, decoding step, etc.
"""

class QueryStatus:
	"""
		TBD
	"""
	def __init__(self, query_id: int, string: str):
		self.query_id = query_id
		self.string = string
		self.tokenized_input = None  # Tokenized input ids
		self.attention_mask = None  # Attention mask for input ids
		self.generated_tokens = None  # List of generated token ids

		self.status = 'pending'  # pending, prefill, decode, completed
		self.prefill_done = False
		self.decode_done = False
		self.decode_step = 0
		self.max_decode_step = 0

class QueryManager:
	def __init__(self, engine_config):
		self.pending_query_pool = {}
		self.prefilled_query_pool = {}
		self.in_decoding_query_pool = {}
		self.completed_query_pool = {} # Flush with some delay.
		self._next_id = 1
		self._free_ids = set()

		self.engine_config = engine_config
		self.cpu_page_size = engine_config.cpu_page_size
		
		self.prefill_kv_pool = None
		self.decode_kv_pool = None
	
	def add_query(self, query):
		"""Add query with ID recycling."""
		if isinstance(query, str):
			# Get ID from free pool or generate new
			if self._free_ids:
				query_id = min(self._free_ids)
				self._free_ids.remove(query_id)
			else:
				query_id = self._next_id
				self._next_id += 1
			
			self.query_pool[query_id] = QueryStatus(query_id, query)
			return query_id
			
		elif isinstance(query, list):
			query_ids = []
			for q in query:
				if self._free_ids:
					query_id = min(self._free_ids)
					self._free_ids.remove(query_id)
				else:
					query_id = self._next_id
					self._next_id += 1
				
				self.query_pool[query_id] = QueryStatus(query_id, q)
				query_ids.append(query_id)
			return query_ids
		
		return None
	
	def remove_query(self, query_id: int) -> bool:
		"""Remove query and free its ID."""
		if query_id in self.query_pool:
			del self.query_pool[query_id]
			self._free_ids.add(query_id)
			return True
		return False

	def has_pending_prefill_queries(self) -> bool:
		"""
			Check if there are queries not prefilled.
		"""
		for q in self.query_pool.values():
			if not q.prefill_done:
				return True
		return False

	def has_pending_decode_queries(self) -> bool:
		"""
			Check if there are queries in decoding.
		"""
		for q in self.query_pool.values():
			if not q.decode_done:
				return True
		return False

	def get_prefill_batch(self) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
		"""
			Get a batch of queries from pool to prefill.
			Find a group of queries that smaller than remaining pages in host prefill KV pool.
		"""
		remaining_pages = self.prefill_kv_pool.get_remaining_page_num()
		if remaining_pages == 0:
			return [], None, None

		selected_queries = []
		total_pages = 0
		for q in self.pending_query_pool.values():
			num_tokens = len(q.tokenized_input)
			num_pages = math.ceil(num_tokens / self.cpu_page_size)
			if total_pages + num_pages <= remaining_pages:
				selected_queries.append(q)
				total_pages += num_pages
			if total_pages >= remaining_pages:
				break
		if not selected_queries:
			return [], None, None
		# Prepare batch tensors
		batch_size = len(selected_queries)
		max_seq_len = max(len(q.tokenized_input) for q in selected_queries)
		input_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
		attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
		query_ids = []
		for i, q in enumerate(selected_queries):
			seq_len = len(q.tokenized_input)
			input_ids[i, :seq_len] = torch.tensor(q.tokenized_input, dtype=torch.long)
			attention_mask[i, :seq_len] = torch.tensor(q.attention_mask, dtype=torch.long)
			query_ids.append(q.query_id)
		return query_ids, input_ids, attention_mask

	
	def get_decode_batch(self, decode_batch_size):
		"""
			Get a batch from prefilled pool to start decode.
			Output:
				1. query_ids: List[int], list of query ids in the batch
				2. input_ids: torch.Tensor, shape (batch_size, 1), the last generated token for each query
				3. attention_mask: torch.Tensor, shape (batch_size, max_seq_len), the attention mask for each query
		"""
		if not self.prefilled_query_pool:
			return [], None, None
		selected_queries = []
		for q in self.prefilled_query_pool.values():
			selected_queries.append(q)
			if len(selected_queries) >= decode_batch_size:
				break
		if not selected_queries:
			return [], None, None
		# Prepare batch tensors
		batch_size = len(selected_queries)
		# Input ids as the last generated token
		input_ids = torch.zeros((batch_size, 1), dtype=torch.long)			
		# The default padding side is right.
		max_seq_len = max(len(q.tokenized_input) + q.decode_step for q in selected_queries)
		attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
		query_ids = []
		for i, q in enumerate(selected_queries):
			input_ids[i, 0] = q.generated_tokens[-1] 
			seq_len = len(q.tokenized_input) + q.decode_step
			attention_mask[i, :seq_len] = 1
			query_ids.append(q.query_id)
		return query_ids, input_ids, attention_mask


	def prefill_done(self, query_ids: List[int], generated_tokens: torch.Tensor):
		"""
			1. Update the queries as prefill done and move them to prefilled pool.
			2. Add the first generated token (e.g., BOS) to generated_tokens.
			3. Increment decode step to 1.
		"""
		for i, qid in enumerate(query_ids):
			if qid in self.pending_query_pool:
				q = self.pending_query_pool[qid]
				q.prefill_done = True
				q.generated_tokens = [generated_tokens[i].item()]  # Initialize with the first generated token
				q.decode_step = 1
				self.prefilled_query_pool[qid] = q
				del self.pending_query_pool[qid]
			else:
				logging.error(f"Query ID {qid} not found in pending pool during prefill_done.")
				raise ValueError(f"Query ID {qid} not found in pending pool during prefill_done.")


	def decode_step_done(self, query_ids: List[int], generated_tokens: torch.Tensor):
		"""
			Decode step done. 
			1. Update the generated tokens for each query.
			2. Add one step.
			3. If we reach <EOS> or max decode step, mark as decode done and move to completed pool.
		"""
		for i, qid in enumerate(query_ids):
			if qid in self.prefilled_query_pool:
				q = self.prefilled_query_pool[qid]
				q.generated_tokens.append(generated_tokens[i].item())
				q.decode_step += 1
				if (generated_tokens[i].item() == self.engine_config.eos_token_id) or (q.decode_step >= q.max_decode_step):
					q.decode_done = True
					q.status = 'completed'
					self.completed_query_pool[qid] = q
					del self.prefilled_query_pool[qid]
				else:
					q.status = 'decode'
					self.in_decoding_query_pool[qid] = q
					del self.prefilled_query_pool[qid]
			else:
				logging.warning(f"Query ID {qid} not found in prefilled pool during decode_step_done.")


	def get_prefilled_query(self):
		"""
			Get one(if any) prefilled query for continuous batching in decoding. 
		"""
		# If prefilled query pool is empty, return None
		if len(self.prefilled_query_pool) == 0:
			return None
		# Just return the first one in the prefilled pool. Vanilla strategy.
		# More advanced strategies can be implemented later.
		for q in self.prefilled_query_pool.values():
			return q
		return None




				
		


