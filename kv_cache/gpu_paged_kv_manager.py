import torch
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class GPUPagedKVStats:
	num_total_pages: int
	num_free_pages: int
	num_used_pages: int
	num_total_pages_allocated: int

class GPUKVCacheManager:
	"""
	Manages a GPU-paged key-value cache.
	It executes low-level page allocation and copy commands.
	Not include any high-level scheduling logic.
	This class supports GQA, MQA, and MLA-style layouts throught its configuration.
	"""
	def __init__(self, engine_config):
		self.config = engine_config.gpu_kv_config
		self.device = torch.device(engine_config.basic_config.gpu_device_id)

		self.k_cache_gpu: Optional[torch.Tensor] = None
		self.v_cache_gpu: Optional[torch.Tensor] = None

		self.num_total_pages: int = 0
		self.free_pages: List[int] = []
		self.page_table: Dict[int, List[int]] = {} # sequence_id -> List[page_indices]
		self.sequence_lengths: Dict[int, int] = {}  # sequence_id -> current sequence length

		self.logger = logging.getLogger(self.__class__.__name__)

	"""Public APIs"""
	def init(self):
		"""Instantiate GPU KV tensors and allocator states"""
		# --- 1. Instantiate K-Cache ---
		# Shape: [num_layers, num_gpu_pages, page_size, num_k_heads, k_head_dim]
		# For MLA of DeepSeek-V3, this would be num_k_heads=1, k_head_dim=576.
		self.k_cache_gpu = torch.empty(
			(
				self.config.num_layers,
				self.config.num_gpu_pages,
				self.config.page_size_tokens,
				self.config.num_k_heads,
				self.config.k_head_dim
			),
			dtype=self.config.kv_dtype_torch,
			device=self.device
		)
		self.logger.info(f"Initialized K-Cache with shape {self.k_cache_gpu.shape}")
		# --- 2. Instantiate V-Cache (Optional) ---

		if self.config.num_v_heads is not None and self.config.num_v_heads > 0 and \
			self.config.v_head_dim is not None and self.config.v_head_dim > 0:
			self.v_cache_gpu = torch.empty(
				(
					self.config.num_layers,
					self.config.num_gpu_pages,
					self.config.page_size_tokens,
					self.config.num_v_heads,
					self.config.v_head_dim
				),
				dtype=self.config.kv_dtype_torch,
				device=self.device
			)
			self.logger.info(f"Initialized V-Cache with shape {self.v_cache_gpu.shape}")
		else:
			self.logger.info("Initialized with no V-Cache (K-Cache only mode).")

		# --- 3. GPU Page Allocator State ---
		self.num_total_pages = self.config.num_gpu_pages
		self.free_pages: List[int] = list(range(self.num_total_pages))
		self.page_table: Dict[int, List[int]] = {}  # sequence_id -> List of allocated page IDs
	
	# --- 1. Page Management Primitives ---
	# def allocate_pages(self, sequence_id: int, num_pages: int) -> List[int]:
	# 	"""
	# 	Allocates 'num_pages' from the free list for a sequence.
		
	# 	This method asssumes the caller (upper-level scheduler) has already ensured
	# 	there is enough free pages.

	# 	Args:
	# 		sequence_id (int): Unique identifier for the sequence.
	# 		num_pages (int): Number of pages to allocate.
		
	# 	Returns:
	# 		List[int]: List of allocated page indices.

	# 	Raises:
	# 		RuntimeError: If not enough free pages are available indicating a scheduler logic error.

	# 	"""
	# 	if len(self.free_pages) < num_pages:
	# 		self.logger.error(
	# 			f"Not enough free pages to allocate {num_pages} pages for sequence {sequence_id}."
	# 			f" Available: {len(self.free_pages)}. This indicates a scheduler logic error."
	# 		)
	# 		raise RuntimeError(f"{self.__class__.__name__}: Insufficient free pages for allocation.")

	# 	# Pop Pages from the free list(LIFO).
	# 	new_pages = [self.free_pages.pop() for _ in range(num_pages)]

	# 	# Add to the page table
	# 	if sequence_id not in self.page_table:
	# 		self.page_table[sequence_id] = []
	# 	self.page_table[sequence_id].extend(new_pages)
		
	# 	self.logger.debug(f"Allocated {num_pages} new pages for seq {sequence_id}: {new_pages}")
	# 	return new_pages

	# --- 1. Page Management Primitives ---
	def allocate_pages(self, sequence_id: int, num_tokens: int) -> List[int]:
		"""
		Allocates pages from the free list for a sequence based on the number of tokens.
		
		This method automatically calculates the number of pages needed based on
		the page size and number of tokens. It assumes the caller (upper-level scheduler)
		has already ensured there are enough free pages.

		Args:
			sequence_id (int): Unique identifier for the sequence.
			num_tokens (int): Number of tokens that need to be stored for this sequence.
		
		Returns:
			List[int]: List of allocated page indices.

		Raises:
			ValueError: If num_tokens is invalid (≤ 0).
			RuntimeError: If not enough free pages are available, indicating a scheduler logic error.

		Example:
			>>> # Allocate pages for a sequence with 150 tokens
			>>> # If page_size_tokens=64, this will allocate 3 pages (ceil(150/64)=3)
			>>> allocated = manager.allocate_pages(sequence_id=42, num_tokens=150)
			>>> # allocated = [5, 7, 2]  # Example page indices
		"""
		# Validate input
		if num_tokens <= 0:
			raise ValueError(
				f"{self.__class__.__name__}: num_tokens must be positive, got {num_tokens}"
			)
		
		# Calculate number of pages needed
		num_pages = math.ceil(num_tokens / self.config.page_size_tokens)
		
		self.logger.debug(
			f"Sequence {sequence_id} requires {num_pages} pages "
			f"for {num_tokens} tokens (page_size={self.config.page_size_tokens})"
		)
		
		# Check if enough free pages available
		if len(self.free_pages) < num_pages:
			self.logger.error(
				f"Not enough free pages to allocate {num_pages} pages for sequence {sequence_id}. "
				f"Required: {num_pages}, Available: {len(self.free_pages)}. "
				f"This indicates a scheduler logic error."
			)
			raise RuntimeError(
				f"{self.__class__.__name__}: Insufficient free pages for allocation. "
				f"Need {num_pages} pages for {num_tokens} tokens, only {len(self.free_pages)} available."
			)

		# Pop pages from the free list (LIFO)
		new_pages = [self.free_pages.pop() for _ in range(num_pages)]

		# Add to the page table
		if sequence_id not in self.page_table:
			self.page_table[sequence_id] = []
		self.page_table[sequence_id].extend(new_pages)
		
		# Track sequence length
		if hasattr(self, 'sequence_lengths'):
			self.sequence_lengths[sequence_id] = num_tokens
		
		self.logger.debug(
			f"Allocated {num_pages} pages for seq {sequence_id} "
			f"({num_tokens} tokens): {new_pages}"
		)
		
		return new_pages

	def free_sequence(self, sequence_id: int):
		"""
		Frees all GPU pages associated with a sequence and
		returns them to the free pool.
		"""
		# Atomically remove the sequence and get its pages
		freed_pages = self.page_table.pop(sequence_id, None)

		if freed_pages:
			self.free_pages.extend(freed_pages)
			self.logger.debug(f"Freed pages for seq {sequence_id}: {freed_pages}")
		else:
			self.logger.error(f"Attempted to free pages for unknown sequence {sequence_id}.")
			raise RuntimeError(f"{self.__class__.__name__}: Sequence ID {sequence_id} not found.")
	
	# --- 2. Kernel Support Primitives ---

	def _build_page_table(self, sequence_ids: List[int]) -> torch.Tensor:
		"""
		Builds the 2D page table(aka. block table) for attention kernel launch.
		The shape is [batch_size, max_pages_in_batch]

		Args:
			sequence_ids: The list of sequence IDs, in batch order.

		Returns:
			A 2D torch.Tensor on the device, padded with -1.
		"""
		if not sequence_ids:
			return torch.empty((0, 0), dtype=torch.int32, device=self.device)

		# Find the max number of pages allocated among sequences in this batch.
		max_pages_in_batch = 0
		for seq_id in sequence_ids:
			if seq_id in self.page_table:
				max_pages_in_batch = max(max_pages_in_batch, len(self.page_table[seq_id]))	

		num_sequences = len(sequence_ids)

		# Initialize with -1 (a common padding value for kernels)
		block_table = torch.full(
			(num_sequences, max_pages_in_batch),
			-1,
			dtype=torch.int32,
			device=self.device
		)

		# Fill the table
		for i, seq_id in enumerate(sequence_ids):
			if seq_id not in self.page_table:
				# This sequence is in the batch but has no GPU pages
				self.logger.warning(f"Sequence ID {seq_id} is not in GPU, this should not hapen in normal cases.")
				continue
			
			pages = self.page_table[seq_id]
			block_table[i, :len(pages)] = torch.tensor(
				pages, dtype=torch.int32, device=self.device
			)
		return block_table

	def get_layer_kv_with_page_table(self, layer_idx: int, sequence_ids: List[int]) \
			-> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
		"""
		Return KV cache tensor for a specific layer along with the page table.

		Args: 
			layer_idx: which layer to get KV-Cache for (0-indexed).
			sequence_ids: List of sequence IDs(UUID) in the current batch order.

		Returns:
			Tuple of (K-Cache Tensor, V-Cache Tensor or None, page table for current batch.
		"""
		if self.k_cache_gpu is None:
			raise RuntimeError("GPUKVCacheManager not initialized.")
		if layer_idx < 0 or layer_idx >= self.config.num_layers:
			raise RuntimeError(f"Invalid layer index {layer_idx} for GPU KV-Cache.")
		
		k_layer = self.k_cache_gpu[layer_idx]  # Shape: [num_gpu_pages, page_size, num_k_heads, k_head_dim]
		v_layer = self.v_cache_gpu[layer_idx] if self.v_cache_gpu is not None else None
		page_table = self._build_page_table(sequence_ids)  # Shape: [batch_size, max_pages_in_batch]

		return k_layer, v_layer, page_table

	def get_sequence_layer_page_pointers(
		self, 
		sequence_id: int, 
		layer_idx: int
	) -> Tuple[List[int], Optional[List[int]]]:
		"""
		Returns the page pointers (memory addresses) for a specific sequence and layer.
		
		This method is used to get raw memory pointers for copy operations between
		CPU and GPU KV caches. The returned pointers can be passed directly to the
		copy engine for efficient batched transfers.
		
		Args:
			sequence_id: Unique identifier for the sequence.
			layer_idx: Layer index to get page pointers for (0-indexed).
		
		Returns:
			Tuple of (k_page_ptrs, v_page_ptrs) where:
			- k_page_ptrs: List of memory addresses (as integers) for K cache pages
			- v_page_ptrs: List of memory addresses (as integers) for V cache pages,
						or None if V cache is not allocated (e.g., MLA mode)
		
		Raises:
			ValueError: If sequence_id is not found or layer_idx is invalid.
			RuntimeError: If GPU KV cache is not initialized.
		
		Example:
			>>> k_ptrs, v_ptrs = manager.get_sequence_layer_page_pointers(seq_id=42, layer_idx=0)
			>>> # k_ptrs = [140234567890, 140234668900, ...]  # Memory addresses
			>>> copy_engine.blocking_page_copy(cpu_ptrs, k_ptrs, page_sizes)
		"""
		# Validate initialization
		if self.k_cache_gpu is None:
			raise RuntimeError(
				f"{self.__class__.__name__}: GPU KV cache not initialized. "
				"Call init() first."
			)
		
		# Validate sequence exists
		if sequence_id not in self.page_table:
			raise ValueError(
				f"{self.__class__.__name__}: Sequence {sequence_id} not found in page table. "
				f"Available sequences: {list(self.page_table.keys())}"
			)
		
		# Validate layer index
		if layer_idx < 0 or layer_idx >= self.config.num_layers:
			raise ValueError(
				f"{self.__class__.__name__}: Invalid layer_idx {layer_idx}. "
				f"Must be in range [0, {self.config.num_layers})"
			)
		
		# Get the page indices allocated to this sequence
		page_indices = self.page_table[sequence_id]
		
		if not page_indices:
			self.logger.warning(
				f"Sequence {sequence_id} has no pages allocated. "
				"Returning empty pointer lists."
			)
			return ([], None if self.v_cache_gpu is None else [])
		
		# Extract K cache page pointers
		k_page_ptrs = []
		for page_idx in page_indices:
			# Get the page tensor slice for this layer and page
			# Shape: [page_size_tokens, num_k_heads, k_head_dim]
			k_page = self.k_cache_gpu[layer_idx, page_idx]
			
			# Get the raw memory address
			k_page_ptrs.append(k_page.data_ptr())
		
		# Extract V cache page pointers (if V cache exists)
		v_page_ptrs = None
		if self.v_cache_gpu is not None:
			v_page_ptrs = []
			for page_idx in page_indices:
				# Shape: [page_size_tokens, num_v_heads, v_head_dim]
				v_page = self.v_cache_gpu[layer_idx, page_idx]
				v_page_ptrs.append(v_page.data_ptr())
		
		self.logger.debug(
			f"Retrieved {len(k_page_ptrs)} page pointers for sequence {sequence_id}, "
			f"layer {layer_idx}"
		)
		
		return (k_page_ptrs, v_page_ptrs)

	def update_new_token(
		self, 
		k_tensor: torch.Tensor, 
		v_tensor: Optional[torch.Tensor], 
		sequence_ids: List[int], 
		layer_idx: int
	):
		"""
		Appends new KV cache data for newly generated tokens to their correct page(s).
		
		This function is called during the decode phase to write new KV pairs to the cache.
		It handles both single-token (typical autoregressive) and multi-token (speculative
		decoding, Medusa) generation scenarios.
		
		Expected k_tensor shape: [batch_size, seq_len, num_k_heads, k_head_dim]
		Expected v_tensor shape: [batch_size, seq_len, num_v_heads, v_head_dim] or None
		
		Note: seq_len normally equals 1 for autoregressive generation but could be >1
		in multi-token prediction scenarios.
		
		Args:
			k_tensor: K cache tensor for new tokens [batch_size, seq_len, num_k_heads, k_head_dim]
			v_tensor: V cache tensor for new tokens (optional, for non-MLA models)
			sequence_ids: List of sequence IDs, length must match batch_size
			layer_idx: Which layer to update (0-indexed)
		
		Raises:
			ValueError: If input validation fails
			RuntimeError: If cache not initialized or insufficient pages allocated
		
		Example:
			>>> # Single token generation (batch_size=2, seq_len=1)
			>>> k_new = torch.randn(2, 1, 8, 128, device='cuda')
			>>> v_new = torch.randn(2, 1, 8, 128, device='cuda')
			>>> manager.update_new_token(k_new, v_new, sequence_ids=[42, 43], layer_idx=0)
			
			>>> # Multi-token generation (batch_size=1, seq_len=3)
			>>> k_new = torch.randn(1, 3, 8, 128, device='cuda')
			>>> manager.update_new_token(k_new, None, sequence_ids=[42], layer_idx=0)
		"""
		# --- 1. Validate Inputs ---
		if self.k_cache_gpu is None:
			raise RuntimeError(
				f"{self.__class__.__name__}: GPU KV cache not initialized. Call init() first."
			)
		
		if layer_idx < 0 or layer_idx >= self.config.num_layers:
			raise ValueError(
				f"{self.__class__.__name__}: Invalid layer_idx {layer_idx}. "
				f"Must be in range [0, {self.config.num_layers})"
			)
		
		# Validate tensor dimensions
		if k_tensor.dim() != 4:
			raise ValueError(
				f"{self.__class__.__name__}: Expected k_tensor to have 4 dimensions "
				f"[batch_size, seq_len, num_k_heads, k_head_dim], got shape {k_tensor.shape}"
			)
		
		batch_size, seq_len, num_k_heads, k_head_dim = k_tensor.shape
		
		# Validate batch size matches sequence_ids
		if len(sequence_ids) != batch_size:
			raise ValueError(
				f"{self.__class__.__name__}: Batch size mismatch. "
				f"k_tensor batch_size={batch_size}, len(sequence_ids)={len(sequence_ids)}"
			)
		
		# Validate K tensor dimensions match config
		if num_k_heads != self.config.num_k_heads or k_head_dim != self.config.k_head_dim:
			raise ValueError(
				f"{self.__class__.__name__}: K tensor dimension mismatch. "
				f"Expected (num_k_heads={self.config.num_k_heads}, k_head_dim={self.config.k_head_dim}), "
				f"got ({num_k_heads}, {k_head_dim})"
			)
		
		# Validate V tensor if provided
		if v_tensor is not None:
			if self.v_cache_gpu is None:
				raise ValueError(
					f"{self.__class__.__name__}: V tensor provided but V cache not initialized "
					"(running in K-only mode, e.g., MLA)"
				)
			
			if v_tensor.dim() != 4:
				raise ValueError(
					f"{self.__class__.__name__}: Expected v_tensor to have 4 dimensions, "
					f"got shape {v_tensor.shape}"
				)
			
			v_batch_size, v_seq_len, num_v_heads, v_head_dim = v_tensor.shape
			
			if v_batch_size != batch_size or v_seq_len != seq_len:
				raise ValueError(
					f"{self.__class__.__name__}: V tensor shape mismatch. "
					f"Expected batch_size={batch_size}, seq_len={seq_len}, "
					f"got batch_size={v_batch_size}, seq_len={v_seq_len}"
				)
			
			if num_v_heads != self.config.num_v_heads or v_head_dim != self.config.v_head_dim:
				raise ValueError(
					f"{self.__class__.__name__}: V tensor dimension mismatch. "
					f"Expected (num_v_heads={self.config.num_v_heads}, v_head_dim={self.config.v_head_dim}), "
					f"got ({num_v_heads}, {v_head_dim})"
				)
		elif self.v_cache_gpu is not None:
			# V cache exists but no tensor provided - this might be intentional (MLA mode)
			# Just log a debug message
			self.logger.debug(
				f"V cache exists but v_tensor is None for layer {layer_idx}. "
				"Assuming K-only update."
			)
		
		# Validate device
		if k_tensor.device != self.device:
			raise ValueError(
				f"{self.__class__.__name__}: K tensor on wrong device. "
				f"Expected {self.device}, got {k_tensor.device}"
			)
		
		if v_tensor is not None and v_tensor.device != self.device:
			raise ValueError(
				f"{self.__class__.__name__}: V tensor on wrong device. "
				f"Expected {self.device}, got {v_tensor.device}"
			)
		
		# Validate all sequences exist and have pages allocated
		for seq_id in sequence_ids:
			if seq_id not in self.page_table:
				raise ValueError(
					f"{self.__class__.__name__}: Sequence {seq_id} not found in page table"
				)
			
			if not self.page_table[seq_id]:
				raise ValueError(
					f"{self.__class__.__name__}: Sequence {seq_id} has no pages allocated"
				)
		
		# --- 2. Update Cache for Each Sequence in Batch ---
		for batch_idx, seq_id in enumerate(sequence_ids):
			# Get current token count for this sequence
			current_tokens = self.sequence_lengths.get(seq_id, 0)
			
			# Extract K and V tensors for this sequence
			# Shape: [seq_len, num_k_heads, k_head_dim]
			k_seq = k_tensor[batch_idx]
			v_seq = v_tensor[batch_idx] if v_tensor is not None else None
			
			# Get allocated pages for this sequence
			page_indices = self.page_table[seq_id]
			
			# Calculate current capacity
			current_capacity = len(page_indices) * self.config.page_size_tokens
			
			# Check if we need more pages
			new_total_tokens = current_tokens + seq_len
			required_pages = math.ceil(new_total_tokens / self.config.page_size_tokens)
			
			if required_pages > len(page_indices):
				raise RuntimeError(
					f"{self.__class__.__name__}: Insufficient pages for sequence {seq_id}. "
					f"Current pages: {len(page_indices)}, Required: {required_pages}. "
					f"Current tokens: {current_tokens}, New tokens: {seq_len}. "
					f"Call append_tokens() or allocate_pages() before update_new_token()."
				)
			
			# --- 3. Write Tokens to Pages ---
			tokens_written = 0
			
			for token_idx in range(seq_len):
				# Calculate absolute token position in the sequence
				absolute_token_pos = current_tokens + token_idx
				
				# Determine which page and offset within that page
				page_idx_in_seq = absolute_token_pos // self.config.page_size_tokens
				offset_in_page = absolute_token_pos % self.config.page_size_tokens
				
				# Get the actual GPU page index
				gpu_page_idx = page_indices[page_idx_in_seq]
				
				# Write K cache
				# Source: k_seq[token_idx] has shape [num_k_heads, k_head_dim]
				# Destination: self.k_cache_gpu[layer_idx, gpu_page_idx, offset_in_page]
				self.k_cache_gpu[layer_idx, gpu_page_idx, offset_in_page].copy_(
					k_seq[token_idx]
				)
				
				# Write V cache if applicable
				if v_seq is not None and self.v_cache_gpu is not None:
					self.v_cache_gpu[layer_idx, gpu_page_idx, offset_in_page].copy_(
						v_seq[token_idx]
					)
				
				tokens_written += 1
			
			# --- 4. Update Bookkeeping ---
			# Update sequence length tracking
			self.sequence_lengths[seq_id] = new_total_tokens
			
			self.logger.debug(
				f"Updated {tokens_written} tokens for seq {seq_id} at layer {layer_idx}. "
				f"Token range: [{current_tokens}, {new_total_tokens}), "
				f"Pages used: {page_idx_in_seq + 1}/{len(page_indices)}"
			)
		
		self.logger.debug(
			f"Batch update complete for layer {layer_idx}: "
			f"{batch_size} sequences, {seq_len} tokens each"
		)


	def get_kv_tensors(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
			"""
			Returns the handles to the full K and V cache tensors.
			The V-cache may be None (e.g., in MLA mode).
			"""
			if self.k_cache_gpu is None:
				raise RuntimeError("GPUKVCacheManager not initialized. Call init() first.")
				
			return self.k_cache_gpu, self.v_cache_gpu

	def get_stats(self) -> GPUPagedKVStats:
			"""Returns current GPU KV-Cache statistics."""
			num_free = len(self.free_pages)
			num_used = self.num_total_pages - num_free
			
			# num_total_pages_allocated is just num_used, but we
			# can calculate it from the page_table for a sanity check.
			# allocated_check = sum(len(pages) for pages in self.page_table.values())
			# assert num_used == allocated_check, "GPU page accounting mismatch!"

			return GPUPagedKVStats(
				num_total_pages=self.num_total_pages,
				num_free_pages=num_free,
				num_used_pages=num_used,
				num_total_pages_allocated=num_used
			)

		

