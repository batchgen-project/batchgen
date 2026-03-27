"""
WorkerState: Central shared state for all sub-managers.

All sub-managers receive a WorkerState reference. Single source of truth.
Sub-managers read/write WorkerState but never hold their own copy of batch state.
This prevents the "two copies of truth" problem that caused metadata staleness.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus


@dataclass
class WorkerState:
    """Central shared state accessed by all scheduler sub-managers.

    Every field here was previously a self.* attribute on BatchGenWorker.
    Moving them to a shared dataclass makes the dependency graph explicit.
    """

    # --- Identity ---
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    torch_device: torch.device = None

    # --- Core batch state ---
    global_batch: Optional[SequenceBatch] = None
    query_book: Optional[Dict] = None
    model_batch_book: Dict = field(default_factory=dict)

    # --- Index maps (owned by IndexManager, stored here for shared access) ---
    local_to_uuid_map: Dict[int, str] = field(default_factory=dict)
    uuid_to_local_map: Dict[str, int] = field(default_factory=dict)
    free_local_indices: Set[int] = field(default_factory=set)
    next_local_idx: int = 0

    # --- KV cache managers ---
    gpu_kv_manager: object = None  # GPUPagedKVCacheManager
    host_kv_view: object = None  # core_engine.MLAHostPagedKVWorkerView or DefaultHostPagedKVWorkerView
    sequences_with_gpu_kv: Set[str] = field(default_factory=set)

    # --- Config ---
    engine_config: object = None  # EngineConfig
    model_config: object = None
    loaded_model_config: object = None

    # --- Model ---
    model: object = None  # nn.Module
    tokenizer: object = None

    # --- Async KV state ---
    deferred_kv_entries: List = field(default_factory=list)
    pending_kv_tasks: List = field(default_factory=list)
    pending_kv_tensors: List = field(default_factory=list)
    kv_offload_event: object = None  # torch.cuda.Event

    # --- Runtime config ---
    max_input_length: int = 0
    max_decoding_length: int = 0
    max_context_length: Optional[int] = None
    model_context_length: Optional[int] = None
    eos_token_id: Optional[int] = None
    num_global_queries: int = 0
    num_local_queries: int = 0

    # --- Sampling ---
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    per_sequence_sampling_params: Optional[list] = None
    ignore_eos: bool = False

    # --- Adaptive chunk sizer ---
    adaptive_chunk_sizer: object = None  # AdaptiveChunkSizer

    # --- Host KV config ---
    host_kv_chunk_size: int = 0
    host_kv_eviction_watermark: int = 70
    enable_host_kv_eviction: bool = True
    enable_decode_preemption: bool = True
    host_kv_watermark: int = 70

    # --- Page buffer config ---
    enable_prepack: bool = True
    gpu_memory_frac: float = 0.9
    gpu_kv_cache_size_gb: Optional[float] = None

    # --- Distributed ---
    comm: object = None  # PyNcclCommunicator
    nccl_group: object = None
