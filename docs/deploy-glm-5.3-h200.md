# Deploy GLM-5.3-FP8 on 8 H200 GPUs

BatchGen serves the official `zai-org/GLM-5.3` FP8 checkpoint on one node with
**8 H200 GPUs**. GLM-5.3 shares the GLM-5.2 DSA execution geometry, but it has a
separate chat template and request contract. Use the GLM-5.3 identifier so the
runtime selects its dedicated tokenizer and template.

The model context window is 1,048,576 tokens. The current qualification uses
pure DP prefill, dual KV caches, and the host KV cache. The guide covers
checkpoint staging, conversion, installation, host preparation, launch, and
request semantics. The full MMLU-Pro and long-context qualification remains a
separate release gate until its recorded run is complete.

## 1. Download and stage the checkpoint

Download the complete checkpoint to persistent storage, then copy it to local
XFS/NVMe storage on the assigned H200 node before starting the server. Local
staging avoids repeatedly reading the 141-shard checkpoint over the persistent
filesystem during startup.

```bash
huggingface-cli download zai-org/GLM-5.3 \
    --local-dir /persistent/models/GLM-5.3-FP8
```

The downloaded directory must contain `config.json`, tokenizer files,
`model.safetensors.index.json`, and all referenced shards. Do not substitute a
GLM-5.3-Flash or BF16 checkpoint for this guide. The local destination must
have enough space for the source and converted checkpoint; keep any `.partial`
directory private to the current staging run.

## 2. Convert to BatchGen format

```bash
python -m batchgen.tools.convert_checkpoint \
    --input-dir /persistent/models/GLM-5.3-FP8

python -m batchgen.tools.convert_checkpoint \
    --input-dir /persistent/models/GLM-5.3-FP8 \
    --validate-only
```

The default output is
`/persistent/models/GLM-5.3-FP8/converted_ckpt`. Keep the converted directory
beside the staged source so the two paths can be checked together.

## 3. Install BatchGen

Use a release containing the GLM-5.3 model and request support. A full Hopper
installation is:

```bash
git clone https://github.com/batchgen-project/batchgen.git
cd batchgen
./scripts/install_deps.sh --all
```

Run the server from the supported `batchgen` environment. The model-specific
and `batchgen_kernels` worktrees must be from the same release revision; a
shared checkout that another model session moves can select incompatible
kernels.

## 4. Prepare the host

Use an otherwise idle 8×H200 host with sufficient RAM for the FP8 weights and
host KV cache. The standard H200 pre-launch clean step now remounts `/dev/shm`
to the host's `MemTotal` and verifies the resulting capacity before allowing a
server launch. Run the family-specific identity gate, clean step, and verifier
for the exact assigned host before starting the server.

Transparent huge pages should be enabled for shared-memory initialization:

```bash
cat /sys/kernel/mm/transparent_hugepage/shmem_enabled
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled
```

The exact host KV size depends on available RAM and the request mix. Start with
a conservative value, then increase it only after checking `MemAvailable` and
the verifier's host-memory gate.

## 5. Start the server

Replace the paths and choose unused HTTP and rendezvous ports:

```bash
source /path/to/env/activate_batchgen.sh

numactl --interleave=all python -m batchgen.launch_http_server \
    --model zai-org/GLM-5.3-FP8 \
    --cache-dir /local/models/GLM-5.3-FP8 \
    --converted-ckpt-dir /local/models/GLM-5.3-FP8/converted_ckpt \
    --listen-port 10904 \
    --world-size 8 \
    --nnodes 1 \
    --node-rank 0 \
    --dist-init-addr 127.0.0.1:29504 \
    --host-kv-cache-size 280 \
    --kv-dtype bfloat16 \
    --gpu-memory-frac 0.96 \
    --parse-thinking \
    --startup-timeout 3600
```

Use the production H200 defaults above for performance measurements. A lower
`--gpu-memory-frac` can leave no positive GPU-KV budget after the GLM-5.3
weights are resident; BatchGen then falls back to a 1-GiB minimum pool, which
is a configuration failure for throughput testing. Keep
`--disable-cuda-graphs` only for an explicitly isolated compatibility
diagnostic, not for a release performance run.

`--parse-thinking` places the generated reasoning before the template-primed
`</think>` boundary in `message.reasoning_content` and leaves the answer in
`message.content`. GLM-5.3 supports `reasoning_effort=low`, `high`, and `max`;
`medium` and `enable_thinking=false` are rejected by the request path. There is
no CUDA-graph context-length flag.

Wait for readiness before submitting a batch:

```bash
curl --fail http://127.0.0.1:10904/health
```

## 6. Submit requests

A request may select the reasoning effort explicitly:

```json
{"custom_id":"glm53-1","method":"POST","url":"/v1/chat/completions","body":{"model":"zai-org/GLM-5.3-FP8","messages":[{"role":"user","content":"Explain sparse attention."}],"reasoning_effort":"high","max_completion_tokens":256,"temperature":0.0}}
```

The batch API accepts the same JSONL upload and polling flow as other models:

```python
from batchgen.batchgen_client import BatchGenHttpClient

client = BatchGenHttpClient("http://127.0.0.1:10904")
batch = client.submit_batch(
    input_file_path="input.jsonl",
    output_file_path="output.jsonl",
    max_context_length=1_048_576,
)
print(batch["status"])
```

When `--parse-thinking` is enabled, inspect both `message.reasoning_content` and
`message.content`; do not search for a literal closing marker in the visible
answer. GLM-5.3 is a thinking-only model, so disabling thinking is unsupported.

## 7. Qualification status and troubleshooting

The current H200 qualification has verified complete weights, contiguous
checkpoint conversion, local-SSD startup, the dedicated 2,048-token prompt
rendering, deterministic short answers, and structured reasoning output on the
assigned host. The 2,048-sequence MMLU-Pro run is the remaining long-running
correctness/lifecycle gate; its final accuracy and postflight record must be
consulted before making a release performance claim.

If a worker exits, inspect the first Python traceback from the failing rank.
When no Python exception exists, use `CUDA_LAUNCH_BLOCKING=1` and per-operation
logging to localize a CUDA-side failure. Do not attribute a later collective
hang to the supervisor's exit code alone.
