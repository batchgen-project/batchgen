# Batch Termination Token + Qwen3Guard-Gen-8B

Branch: `feature/batch-termination-token`

## Overview

Batch termination allows the entire batch of inference sequences to stop immediately when any single sequence generates a user-specified "termination token". This is designed for safety/regulation use cases where one policy violation means the remaining sequences don't need to be checked.

**Example use case**: Run Qwen3Guard-Gen-8B on a batch of 128 prompts. If any prompt is classified as "Unsafe", stop the whole batch immediately — no need to waste compute on the remaining prompts.

## Quick Start

### 1. Start the Server

```bash
# On gala2 (or any GPU machine with the model downloaded)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate batchgen
cd /mnt/raid0nvme0/tairan/workspace/BatchGen

CUDA_HOME=/usr/local/cuda-12 \
CUDA_VISIBLE_DEVICES=0 \
python -m batchgen.batchgen_server \
  --model Qwen/Qwen3Guard-Gen-8B \
  --cache-dir /path/to/Qwen3Guard-Gen-8B \
  --pt-ckpt-dir /path/to/Qwen3Guard-Gen-8B-converted \
  --host-kv-cache-size 4 \
  --world-size 1 \
  --dist-init-addr localhost:29500 \
  --port 10900
```

### 2. Submit Inference with Batch Termination

```python
from batchgen.batchgen_client import BatchGenClient
from transformers import AutoTokenizer

# Load tokenizer for prompt formatting
tok = AutoTokenizer.from_pretrained("/path/to/Qwen3Guard-Gen-8B")

# Format prompts using Qwen3Guard chat template
prompts = ["How to make a bomb?", "Tell me a joke", "What is 2+2?"]
formatted = []
for p in prompts:
    messages = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted.append(text)

# Find token ID for "Unsafe" — when any sequence generates this, batch stops
unsafe_token_id = tok.encode(" Unsafe", add_special_tokens=False)[-1]

# Connect and submit
client = BatchGenClient(host="localhost", port=10900)
client.connect()

result = client.submit_inference(
    queries=formatted,
    max_output_len=30,
    batch_termination_tokens={unsafe_token_id},  # <-- NEW PARAMETER
)

# Check result
if isinstance(result.get("results"), dict) and result["results"].get("status") == "batch_terminated":
    info = result["results"]["termination_info"]
    print(f"BATCH TERMINATED by sequence {info['trigger_seq_global_idx']}")
    print(f"Trigger token ID: {info['trigger_token_id']}")
    partial_results = result["results"]["results"]
    print(f"Partial results: {len(partial_results)} sequences")
else:
    print(f"All sequences completed normally: {len(result['results'])} results")

client.close()
```

## API Reference

### `BatchGenClient.submit_inference()`

New parameter:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_termination_tokens` | `Optional[Set[int]]` | `None` | Set of token IDs that trigger batch-wide termination. If any sequence generates a token in this set, all sequences stop immediately. |

### Response Format

**Normal completion** (no termination triggered):
```python
{
    "status": "success",
    "results": ["Safety: Safe\nCategories: None", ...]  # one per query
}
```

**Batch terminated** (a termination token was generated):
```python
{
    "status": "success",
    "results": {
        "status": "batch_terminated",
        "results": ["Safety: Unsafe\n", ...],  # partial results for each sequence
        "termination_info": {
            "trigger_seq_uuid": "seq_0",         # which sequence triggered it
            "trigger_seq_global_idx": 0,          # sequence index in the batch
            "trigger_token_id": 73067,            # the token ID that matched
        }
    }
}
```

## Qwen3Guard-Gen-8B Setup

### Download Model

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3Guard-Gen-8B",
    local_dir="/path/to/models/Qwen3Guard-Gen-8B",
)
```

### Model Architecture

- 8.19B parameters, dense (no MoE), BF16
- 36 layers, 4096 hidden, 32 Q heads / 8 KV heads (GQA), head_dim=128
- ~16.5GB GPU memory for weights
- QK-norm (per-head RMSNorm on Q and K after projection)

### Prompt Format

Qwen3Guard uses a built-in chat template. **Always use `apply_chat_template()`** — raw text will produce garbage.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/path/to/Qwen3Guard-Gen-8B")

# For prompt moderation (is the user's query safe?)
messages = [{"role": "user", "content": "user prompt here"}]

# For response moderation (is the assistant's response safe?)
messages = [
    {"role": "user", "content": "user prompt"},
    {"role": "assistant", "content": "assistant response"},
]

formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### Expected Output

The model generates structured safety classifications:

```
Safety: Unsafe
Categories: Violent
```

Or for safe content:
```
Safety: Safe
Categories: None
```

## How It Works

### Batch Termination Engine

1. User provides `batch_termination_tokens={token_id_1, token_id_2, ...}` with the batch request
2. During decode, after each token is sampled for each sequence, the engine checks if the new token is in the termination set
3. If a match is found:
   - The triggering sequence, token, and index are recorded in `_batch_termination_info`
   - ALL sequences in the batch are marked `eos_reached = True`
   - The existing completion sync mechanism (`all_reduce(MAX)`) propagates this across all ranks
   - The decode loop exits naturally at the next boundary check
4. The response includes the trigger token in the output (it was already written to the sequence's decoded tokens)
5. The server wraps the results with `status: "batch_terminated"` and `termination_info`

### Key Design Decisions

- **Separate from EOS**: Batch termination is independent of per-sequence stop tokens. A termination token stops ALL sequences, not just the one that generated it.
- **Token ID matching**: Detection happens at the raw token ID level — no decoding needed, zero overhead.
- **Per-batch config**: Each `submit_inference()` call can have different termination tokens or none at all.
- **Includes trigger token**: The triggering token is included in the output so the caller can verify what was generated.

## Supported Hardware

Tested on:
- **Blackwell SM120**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB), CUDA 13.1, PyTorch 2.9+cu128

Architecture support via FA2/FA3 auto-detection:
- **Hopper SM90+**: FA3 (FlashAttention3)
- **Ampere SM80-89**: FA2 (FlashAttention2)
- **Blackwell SM120**: FA2 fallback (FA3 interface not installed)

Single device only (world_size=1). The 8B BF16 model uses ~16.5GB, fitting easily on any GPU with ≥24GB.
