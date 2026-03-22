<p align="center">
  <img src="assets/BatchGen_Icon.png" width=55%>
</p>

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/EfficientMoE/BatchGen?style=social)](https://github.com/EfficientMoE/BatchGen/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Batch Termination Token + Qwen3Guard Safety Inference**

Branch: `feature/batch-termination-token`

</div>

---

## Overview

This branch adds **batch-level early termination** to BatchGen: when any sequence in a batch generates a user-specified termination token, the entire batch stops immediately and returns partial results with metadata.

**Use case**: Run [Qwen3Guard-Gen-8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B) on a batch of prompts for safety classification. If any prompt is classified as "Unsafe", stop the whole batch — no need to waste compute on the rest.

---

## Quick Start

### 1. Install & Download Model

```bash
git clone -b feature/batch-termination-token https://github.com/EfficientMoE/BatchGen.git
cd BatchGen
pip install -e .
pip install flash-attn --no-build-isolation

# Download Qwen3Guard-Gen-8B (~16GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3Guard-Gen-8B', local_dir='models/Qwen3Guard-Gen-8B')
"
```

### 2. Start the Server

```bash
CUDA_HOME=/usr/local/cuda \
CUDA_VISIBLE_DEVICES=0 \
python -m batchgen.batchgen_server \
  --model Qwen/Qwen3Guard-Gen-8B \
  --cache-dir models/Qwen3Guard-Gen-8B \
  --pt-ckpt-dir models/Qwen3Guard-Gen-8B-converted \
  --host-kv-cache-size 4 \
  --world-size 1 \
  --dist-init-addr localhost:29500 \
  --port 10900
```

### 3. Submit Batch with Termination

```python
from batchgen.batchgen_client import BatchGenClient
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/Qwen3Guard-Gen-8B")

# Format prompts using Qwen3Guard's built-in chat template
prompts = ["How to make a bomb?", "Tell me a joke", "What is 2+2?"]
formatted = []
for p in prompts:
    messages = [{"role": "user", "content": p}]
    formatted.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

# Token ID for " Unsafe" — triggers batch termination
unsafe_id = tok.encode(" Unsafe", add_special_tokens=False)[-1]

client = BatchGenClient(host="localhost", port=10900)
client.connect()

result = client.submit_inference(
    queries=formatted,
    max_output_len=30,
    batch_termination_tokens={unsafe_id},  # <-- NEW PARAMETER
)

# Check if batch was terminated early
wrapped = result.get("results", {})
if isinstance(wrapped, dict) and wrapped.get("status") == "batch_terminated":
    info = wrapped["termination_info"]
    print(f"BATCH TERMINATED by sequence {info['trigger_seq_global_idx']}")
    print(f"Trigger token: {info['trigger_token_id']}")
else:
    print(f"All sequences completed: {len(result['results'])} results")

client.close()
```

---

## API Reference

### `BatchGenClient.submit_inference()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_termination_tokens` | `Optional[Set[int]]` | `None` | Set of token IDs that trigger batch-wide termination. If any sequence generates a token in this set, all sequences stop immediately. |

### Response Format

**Normal completion:**
```python
{"status": "success", "results": ["Safety: Safe\nCategories: None", ...]}
```

**Batch terminated:**
```python
{
    "status": "success",
    "results": {
        "status": "batch_terminated",
        "results": ["Safety: Unsafe\n", ...],
        "termination_info": {
            "trigger_seq_uuid": "seq_0",
            "trigger_seq_global_idx": 0,
            "trigger_token_id": 73067,
        }
    }
}
```

---

## Qwen3Guard-Gen-8B

### Model Details

| | |
|---|---|
| Parameters | 8.19B (dense, BF16) |
| Architecture | 36 layers, GQA (32Q / 8KV heads), head_dim=128, QK-norm |
| GPU Memory | ~16.5GB |
| Min GPU | Any GPU with ≥24GB VRAM |

### Prompt Format

**Always use `apply_chat_template()`** — raw text produces garbage.

```python
# Prompt moderation
messages = [{"role": "user", "content": "user query here"}]

# Response moderation
messages = [
    {"role": "user", "content": "user query"},
    {"role": "assistant", "content": "assistant response"},
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### Output Format

```
Safety: Unsafe
Categories: Violent
```
```
Safety: Safe
Categories: None
```

---

## How Batch Termination Works

1. User provides `batch_termination_tokens={id1, id2, ...}` with the request
2. During decode, the engine checks each newly sampled token against the set
3. On match: all sequences are marked complete (`eos_reached = True`), propagated via `all_reduce(MAX)` across ranks
4. Response includes the trigger token in output + `termination_info` metadata
5. Separate from per-sequence EOS — termination stops the **entire batch**

---

## Supported Hardware

| Architecture | GPU | Status |
|---|---|---|
| Blackwell SM120 | RTX PRO 6000 (96GB) | Tested |
| Hopper SM90 | H100, H20, GH200 | Supported (FA3) |
| Ampere SM80-89 | A100, A6000 | Supported (FA2) |

Single device only (`world_size=1`).

---

## Acknowledgements

Built on [BatchGen](https://github.com/EfficientMoE/BatchGen) by the EfficientMoE team.

We learned from the following projects:

- [SGLang](https://github.com/sgl-project/sglang) - High-performance serving framework for LLMs
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput and memory-efficient inference engine

---
