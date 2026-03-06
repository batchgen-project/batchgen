import torch
import logging
import math
import json
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# Get a logger for this module
logger = logging.getLogger(__name__)

# Placeholder for ModelForwardInput dataclass
@dataclass
class ModelForwardInput:
	input_ids: torch.Tensor
	attention_mask: torch.Tensor
	position_ids: torch.Tensor
	cache_seqlen: Optional[torch.Tensor] = None
# --- Configuration (Example - Define elsewhere) ---
# @dataclass
# class ModelForwardInput:
#     input_ids: torch.Tensor
#     attention_mask: torch.Tensor
#     position_ids: torch.Tensor
#     cache_seqlen: Optional[torch.Tensor] = None # For decode

# --- Sequence Status ---
class SequenceStatus(IntEnum):
    """ Defines the possible states of a sequence during its lifecycle. """
    WAITING_IN_QUEUE = 0 # Initial state after loading
    PREFILL_PENDING = 1  # Selected for prefill, waiting resources
    IN_PREFILL = 2       # Actively being processed by the prefill stage
    DECODE_READY = 3     # Prefill complete, ready for decoding
    IN_DECODE = 4        # Actively being processed by the decode stage
    DECODE_PAUSED = 5 # Temporarily swapped out (e.g., host kv)
    COMPLETED = 6        # Generation finished successfully (<EOS> or max_length)
    FAILED = 7           # An error occurred during processing

# --- Sequence Metadata ---
class SequenceEntry:
    """
    Holds all *metadata* for a single sequence. Lightweight object.
    """
    __slots__ = (
        'uuid',               # User-provided unique string ID (e.g., "request-1")
        'input_ids',          # Prompt tokens (torch.Tensor, CPU, int64)
        'prompt_length',      # Number of tokens in the prompt
        'max_output_length',  # Max number of tokens to generate (from request)
        'status',             # Current SequenceStatus
        'output_tokens',      # Generated tokens (List[int])
        'current_length',     # prompt_length + len(output_tokens)
        'rank_owner',         # Global rank currently responsible for this sequence
        'host_page_ids',      # List of physical page IDs on the Host KV Cache server
        # Per-request sampling parameters (set once at creation, immutable)
        'temperature',        # None = use default (greedy). Float > 0 = sampling.
        'top_p',              # None = disabled. Float in (0, 1] = nucleus sampling.
        'top_k',              # None or 0 = disabled. Int > 0 = top-k filtering.
    )

    # Define valid state transitions (optional but good practice)
    VALID_TRANSITIONS = {
        SequenceStatus.WAITING_IN_QUEUE: {SequenceStatus.PREFILL_PENDING},
        SequenceStatus.PREFILL_PENDING: {SequenceStatus.IN_PREFILL, SequenceStatus.FAILED},
        SequenceStatus.IN_PREFILL: {SequenceStatus.DECODE_READY, SequenceStatus.FAILED},
        SequenceStatus.DECODE_READY: {SequenceStatus.IN_DECODE, SequenceStatus.DECODE_PAUSED},
        SequenceStatus.IN_DECODE: {SequenceStatus.IN_DECODE, SequenceStatus.DECODE_PAUSED, SequenceStatus.COMPLETED, SequenceStatus.FAILED},
        SequenceStatus.DECODE_PAUSED: {SequenceStatus.IN_DECODE}, # When loaded back
        SequenceStatus.COMPLETED: set(),
        SequenceStatus.FAILED: set(),
    }

    def __init__(self,
                 uuid: str,
                 input_ids: List[int],
                 max_output_length: int,
                 rank_owner: int = 0,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None):

        if not isinstance(uuid, str) or not uuid:
            raise ValueError("SequenceEntry requires a non-empty string UUID.")
        if not input_ids:
            raise ValueError(f"SequenceEntry {uuid} received empty input_ids.")

        self.uuid = uuid
        # Store as a CPU tensor for efficiency
        self.input_ids = torch.tensor(input_ids, dtype=torch.int64, device='cpu')
        self.prompt_length = len(input_ids)
        self.max_output_length = max_output_length
        self.status = SequenceStatus.WAITING_IN_QUEUE
        self.output_tokens = []
        self.current_length = self.prompt_length
        self.rank_owner = rank_owner
        self.host_page_ids = []
        # Per-request sampling parameters (immutable after creation)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def update_status(self, new_status: SequenceStatus):
        """ Safely transition the sequence to a new status. """
        # Enforce transition rules
        if new_status not in self.VALID_TRANSITIONS.get(self.status, set()):
            raise ValueError(f"Invalid status transition for {self.uuid}: from {self.status.name} to {new_status.name}")
        self.status = new_status

    def append_output_token(self, token_id: int):
        """ Appends a generated token and updates length. """
        self.output_tokens.append(token_id)
        self.current_length += 1

    def __repr__(self):
        return (f"SequenceEntry(uuid='{self.uuid}', status={self.status.name}, "
                f"len={self.current_length}/{self.prompt_length + self.max_output_length}, "
                f"owner={self.rank_owner})")

# --- Global Registry ---
class GlobalSequenceRegistry:
    """
    The master list ("phonebook") holding metadata for all sequences
    in the current batch job. (Level 3 Object).
    Does NOT store bulk token data in a single tensor.
    """
    __slots__ = ('sequences')

    def __init__(self, sequences: List[SequenceEntry]):
        self.sequences: Dict[str, SequenceEntry] = {seq.uuid: seq for seq in sequences}
        logger.info(f"GlobalSequenceRegistry initialized with {len(self.sequences)} sequences.")

    def get_sequence(self, uuid: str) -> Optional[SequenceEntry]:
        """ Retrieves a sequence entry by its UUID. """
        return self.sequences.get(uuid)

    def __len__(self) -> int:
        return len(self.sequences)

# --- Factory Function for Global Registry ---
def create_global_registry_from_jsonl(
    jsonl_path: str,
    tokenizer: Any, # Use your specific tokenizer type hint
    model_max_context_length: int,
    user_max_input_length: Optional[int] = None # Optional user-defined limit
) -> GlobalSequenceRegistry:
    """
    Loads sequences from a JSONL file, tokenizes, performs checks,
    and creates the GlobalSequenceRegistry.

    Args:
        jsonl_path: Path to the input JSONL file.
        tokenizer: The tokenizer instance.
        model_max_context_length: The absolute max length the model supports.
        user_max_input_length: Optional limit provided by the user to truncate prompts.

    Returns:
        A GlobalSequenceRegistry instance.

    Raises:
        ValueError: On duplicate UUIDs or invalid sequence lengths.
        KeyError: If required fields are missing in the JSONL.
    """
    entries: List[SequenceEntry] = []
    seen_uuids = set()
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0) # Handle tokenizers without pad_token

    logger.info(f"Loading sequences from {jsonl_path}...")
    line_num = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            line_num += 1
            try:
                data = json.loads(line)

                # --- 1. Get and Validate UUID ---
                uuid = data["custom_id"]
                if not isinstance(uuid, str) or not uuid:
                     raise ValueError("Missing or invalid 'custom_id'")
                if uuid in seen_uuids:
                    raise ValueError(f"Duplicate custom_id '{uuid}' found")
                seen_uuids.add(uuid)

                # --- 2. Get and Validate Prompt ---
                # Adapt this logic based on your exact JSON structure
                if "body" in data and "messages" in data["body"] and data["body"]["messages"]:
                    prompt_str = data["body"]["messages"][-1]["content"] # Assuming last message is prompt
                elif "prompt" in data: # Handle simpler format
                     prompt_str = data["prompt"]
                else:
                    raise KeyError("Could not find prompt content ('messages' or 'prompt')")

                # --- 3. Get and Validate Max Output Length ---
                if "body" in data and "max_tokens" in data["body"]:
                     max_output_length = int(data["body"]["max_tokens"])
                elif "max_output_tokens" in data: # Handle simpler format
                     max_output_length = int(data["max_output_tokens"])
                else:
                    raise KeyError("Could not find max output tokens ('max_tokens' or 'max_output_tokens')")

                if max_output_length <= 0:
                     raise ValueError(f"max_output_length must be positive, got {max_output_length}")

                # --- 4. Tokenize and Truncate (if needed) ---
                input_ids = tokenizer(prompt_str).input_ids
                prompt_length = len(input_ids)

                # Determine effective max prompt length
                effective_max_input = model_max_context_length
                if user_max_input_length is not None:
                    if user_max_input_length <= 0:
                        logger.warning(f"User max_input_length ({user_max_input_length}) is invalid, ignoring.")
                    else:
                        effective_max_input = min(effective_max_input, user_max_input_length)
                        logger.debug(f"Applying user max_input_length: {user_max_input_length}")

                # Truncate if necessary (keep the end, common practice)
                if prompt_length > effective_max_input:
                    logger.warning(
                        f"Sequence {uuid}: Prompt length ({prompt_length}) exceeds effective limit "
                        f"({effective_max_input}). Truncating from the left."
                    )
                    input_ids = input_ids[-effective_max_input:]
                    prompt_length = len(input_ids) # Update length after truncation

                # --- 5. Final Validation ---
                if prompt_length == 0:
                     raise ValueError("Prompt resulted in zero tokens after tokenization/truncation")
                if prompt_length + max_output_length > model_max_context_length:
                     raise ValueError(
                         f"Sequence {uuid} length after truncation ({prompt_length}) + "
                         f"max_output_length ({max_output_length}) exceeds model context limit "
                         f"({model_max_context_length}). Please reduce max_output_length or shorten the prompt."
                     )

                # --- 6. Create Entry ---
                entry = SequenceEntry(
                    uuid=uuid,
                    input_ids=input_ids,
                    max_output_length=max_output_length,
                    rank_owner=0 # All sequences start owned by Rank 0
                )
                entries.append(entry)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error processing line {line_num} in {jsonl_path}: {e}")
                # Decide: raise immediately, or collect errors and raise later?
                # Raising immediately is safer for now.
                raise ValueError(f"Failed to load batch file at line {line_num}: {e}") from e

    if not entries:
         raise ValueError(f"Input file {jsonl_path} resulted in zero valid sequences.")

    logger.info(f"Successfully loaded {len(entries)} sequences from {jsonl_path}.")
    return GlobalSequenceRegistry(entries)


# --- Active Batch View ---
class ActiveBatch:
    """
    A "view" into the GlobalSequenceRegistry representing the current
    working set for a specific task (Prefill or Decode). (Level 2 Object).

    Contains high-performance, vectorized methods for generating kernel inputs.
    """
    __slots__ = ('global_registry', 'active_uuids', 'device', 'pad_token_id',
                 'active_seqs', 'batch_size',
                 '_sampling_temps', '_sampling_top_ps', '_sampling_top_ks')

    def __init__(self,
                 global_registry: GlobalSequenceRegistry,
                 active_uuids: List[str],
                 device: torch.device,
                 pad_token_id: int):

        if not active_uuids:
             logger.warning("Initializing ActiveBatch with zero sequences.")

        self.global_registry = global_registry
        self.active_uuids = list(active_uuids) # Ensure it's a list copy
        self.device = device
        self.pad_token_id = pad_token_id

        # Create the metadata view (list of references)
        self.active_seqs: List[SequenceEntry] = []
        missing_uuids = []
        for uuid in self.active_uuids:
             seq = self.global_registry.get_sequence(uuid)
             if seq:
                  self.active_seqs.append(seq)
             else:
                  missing_uuids.append(uuid)
                  logger.error(f"UUID '{uuid}' requested for ActiveBatch not found in GlobalRegistry!")
        if missing_uuids:
            # This indicates a logic error upstream
             raise KeyError(f"Critical error: The following UUIDs were not found in the GlobalRegistry: {missing_uuids}")

        self.batch_size = len(self.active_seqs)

        # Build and cache per-sequence sampling param tensors
        self._sampling_temps = None
        self._sampling_top_ps = None
        self._sampling_top_ks = None
        if self.active_seqs:
            self._build_sampling_param_tensors()

    def _build_sampling_param_tensors(self):
        """Build [B] tensors for per-sequence sampling parameters. Cached until rebuild."""
        # Temperature: None or <=0 means greedy. Use 0.0 as sentinel for greedy.
        temps = [seq.temperature if seq.temperature is not None else 0.0
                 for seq in self.active_seqs]
        top_ps = [seq.top_p if seq.top_p is not None else 1.0
                  for seq in self.active_seqs]
        top_ks = [seq.top_k if seq.top_k is not None else 0
                  for seq in self.active_seqs]

        self._sampling_temps = torch.tensor(temps, dtype=torch.float32, device=self.device)
        self._sampling_top_ps = torch.tensor(top_ps, dtype=torch.float32, device=self.device)
        self._sampling_top_ks = torch.tensor(top_ks, dtype=torch.int64, device=self.device)

    @property
    def sampling_temps(self) -> Optional[torch.Tensor]:
        return self._sampling_temps

    @property
    def sampling_top_ps(self) -> Optional[torch.Tensor]:
        return self._sampling_top_ps

    @property
    def sampling_top_ks(self) -> Optional[torch.Tensor]:
        return self._sampling_top_ks

    def build_prefill_inputs(self) -> Any: # Use ModelForwardInput type hint
        """
        Generates all tensors for a prefill step using JIT padding. Vectorized.
        """
        if not self.active_seqs:
            # Handle empty batch case gracefully
            return ModelForwardInput(
                input_ids=torch.empty((0,0), dtype=torch.int64, device=self.device),
                attention_mask=torch.empty((0,0), dtype=torch.int64, device=self.device),
                position_ids=torch.empty((0,0), dtype=torch.int64, device=self.device)
            )

        prompt_lengths = [seq.prompt_length for seq in self.active_seqs]
        max_batch_len = max(prompt_lengths) if prompt_lengths else 0

        # Create empty padded tensor on GPU
        input_ids = torch.full(
            (self.batch_size, max_batch_len),
            fill_value=self.pad_token_id,
            dtype=torch.int64,
            device=self.device
        )

        # JIT Padding Loop (copying small CPU tensors to GPU slices)
        for i, seq in enumerate(self.active_seqs):
            length = seq.prompt_length
            # Ensure input_ids tensor is correctly transferred
            input_ids[i, :length] = seq.input_ids.to(device=self.device, non_blocking=True)

        # Build attention_mask and position_ids (vectorized)
        seq_range = torch.arange(max_batch_len, device=self.device)
        lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.int64, device=self.device).unsqueeze(1)

        attention_mask = (seq_range < lengths_tensor).int()
        position_ids = seq_range.unsqueeze(0).expand(self.batch_size, -1)

        return ModelForwardInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_seqlen=None # Cache lengths are not needed for prefill kernels usually
        )

    def build_decode_inputs(self) -> Any: # Use ModelForwardInput type hint
        """
        Generates all tensors for a *single* decode step. Vectorized.
        """
        if not self.active_seqs:
             return ModelForwardInput(...) # Handle empty case similar to prefill

        # Get the *last token* for input_ids
        # (This is fast even with a Python loop for small batch sizes)
        input_ids_list = []
        for seq in self.active_seqs:
            if not seq.output_tokens:
                last_token = seq.input_ids[-1].item() # Last prompt token
            else:
                last_token = seq.output_tokens[-1] # Last generated token
            input_ids_list.append(last_token)

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=self.device).unsqueeze(-1)

        # Get current lengths
        current_lengths = torch.tensor(
            [seq.current_length for seq in self.active_seqs],
            dtype=torch.int64,
            device=self.device
        )

        # Build position_ids and attention_mask
        position_ids = (current_lengths - 1).unsqueeze(-1)
        attention_mask = torch.ones_like(input_ids) # Just [B, 1] filled with 1s

        # Build cache_seqlen (or equivalent needed by your kernel)
        # This tensor directly tells the kernel the valid length of each sequence in the KV cache
        cache_seqlen = current_lengths

        return ModelForwardInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_seqlen=cache_seqlen
        )

    def update_after_decode_step(self, new_token_ids: torch.Tensor):
        """
        Updates the metadata in the referenced SequenceEntry objects
        after a decode step.

        Args:
            new_token_ids: Tensor of shape [batch_size] with the newly generated token ID for each sequence.
        """
        if new_token_ids.shape[0] != self.batch_size:
            raise ValueError(f"Shape mismatch: Expected {self.batch_size} new tokens, got {new_token_ids.shape[0]}")

        # Move to CPU for faster Python list appends if needed, or iterate directly
        new_token_ids_cpu = new_token_ids.cpu().tolist() # Or iterate tensor directly if preferred

        for i, seq in enumerate(self.active_seqs):
            seq.append_output_token(new_token_ids_cpu[i])

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self) -> str:
        return f"ActiveBatch(size={self.batch_size}, uuids={self.active_uuids})"
