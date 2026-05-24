# Deploy GLM-5.1-FP8 + Run Terminal-Bench 2.0 on Daytona

End-to-end guide:

1. Deploy `zai-org/GLM-5.1-FP8` on BatchGen (multi-node, OpenAI-compatible).
2. Configure server-side tool-call extraction (`--tool-call-parser glm`).
3. Point Terminal-Bench 2.0 (Harbor harness) at the BatchGen endpoint.
4. Run the full TB2.0 suite using Daytona cloud sandboxes — no Docker on the agent driver host.

Topology:

```
┌────────────────────────────┐        ┌───────────────────────────────┐
│  GPU cluster (H20 / H100)  │        │  Daytona cloud sandboxes      │
│                            │        │  (per-task, ephemeral)        │
│  batchgen-server           │◄───────┤   harbor run --env daytona    │
│  GLM-5.1-FP8               │ OpenAI │   - bash, git, task fixtures  │
│  --tool-call-parser glm    │  HTTP  │   - tests/test_outputs.py     │
└────────────────────────────┘        └───────────────────────────────┘
        ▲
        │ http://<gpu-host>:10900/v1
        │
   driver machine (laptop / CI runner)
   - harbor CLI
   - no Docker needed
```

---

## Hardware

GLM-5.1-FP8 is architecturally identical to GLM-5; the weights are FP8 (~760 GB). Supported configurations on H20:

| Config | TP × PP | Notes |
|---|---|---|
| 2 × H20 nodes (16 × H20) | TP=16 | Recommended; matches the GLM-5 reference setup |
| 4 × H100 nodes (32 × H100) | TP=8, PP=4 | Tested in CI |

For H100/H200 with different memory budgets, see `docs/support-glm-5.1.md` and `batchgen/server/process_utils.py` `MODEL_BYTE_SIZES`.

---

## Prerequisites

On the GPU host(s):
- BatchGen installed (`docs/INSTALL.md` or `docker/Dockerfile`)
- `huggingface_hub` for downloads
- Shared filesystem visible from all nodes (NFS / Lustre / shared SSD)

On the driver host (where you run `harbor`):
- Python 3.10+
- `uv` (or `pip`)
- **No Docker required** when using `--env daytona`
- A [Daytona](https://app.daytona.io) account + API key (free tier works for smoke testing; Tier 3 recommended for the full 200-task suite)

---

## 1. Download GLM-5.1-FP8 weights

On a node with access to the shared filesystem:

```bash
pip install huggingface_hub
huggingface-cli login

huggingface-cli download zai-org/GLM-5.1-FP8 \
    --local-dir /shared/models/GLM-5.1-FP8 \
    --local-dir-use-symlinks False
```

Verify:

```bash
ls /shared/models/GLM-5.1-FP8/
# config.json, tokenizer.json, model-00001-of-NNN.safetensors, …
```

---

## 2. Convert checkpoints to BatchGen format

BatchGen uses a contiguous `.bin` + `.json` layout for fast sequential SSD reads:

```bash
python -m batchgen.tools.convert_checkpoint \
    --input-dir /shared/models/GLM-5.1-FP8

# Produces /shared/models/GLM-5.1-FP8/converted_ckpt/
```

Validate:

```bash
python -m batchgen.tools.convert_checkpoint \
    --input-dir /shared/models/GLM-5.1-FP8 \
    --validate-only
```

---

## 3. Build the BatchGen container (optional but recommended)

```bash
# From the BatchGen project root
docker buildx build --progress=plain -f docker/Dockerfile -t batchgen:latest .
```

If you install BatchGen natively instead, follow `docs/manual-installation.md`.

---

## 4. Launch the BatchGen server

Mount `/dev/shm` to host memory size and start one server process per node.

```bash
# Both nodes — adjust paths and addresses
sudo mount -o remount,size=1500G /dev/shm
```

**Node 0 (master):**

```bash
python -m batchgen.batchgen_server \
    --model zai-org/GLM-5.1-FP8 \
    --cache-dir /shared/models/GLM-5.1-FP8 \
    --converted-ckpt-dir /shared/models/GLM-5.1-FP8/converted_ckpt \
    --world-size 16 \
    --nnodes 2 \
    --node-rank 0 \
    --dist-init-addr <node0-ip>:33001 \
    --listen-ip 0.0.0.0 \
    --listen-port 10900 \
    --kv-dtype bf16 \
    --gpu-memory-frac 0.96 \
    --host-kv-cache-size 1000 \
    --enable-hugetlbfs \
    --storage-path /shared/storage \
    --save-result \
    --parse-thinking \
    --tool-call-parser glm
```

**Node 1:**

Same command, replace `--node-rank 0` with `--node-rank 1`.

Key flags:
- `--tool-call-parser glm` — extracts `<tool_call>…</tool_call>` blocks into the OpenAI-compatible `tool_calls` array. TB2.0 agents (Terminus-2 / Claude-Code) depend on this.
- `--parse-thinking` — splits `<think>…</think>` reasoning into `reasoning_content`. Keeps `content` clean.
- Both flags only affect `/v1/chat/completions`; raw `/v1/completions` responses are untouched.

Verify the server is up from the driver host:

```bash
curl http://<node0-ip>:10900/v1/models
# {"object":"list","data":[{"id":"zai-org/GLM-5.1-FP8",…}]}
```

---

## 5. Install Harbor (TB2.0 harness)

TB2.0 ships as Harbor (`harbor-framework/terminal-bench-2`). Install on the driver host:

```bash
uv tool install harbor
# or: pip install harbor-eval
harbor --help
```

---

## 6. Configure Daytona

Get an API key at https://app.daytona.io and export it:

```bash
export DAYTONA_API_KEY="dtn_xxxxxxxxxxxxxxxxxxxx"
```

Account sizing:
- Free tier: ~3 concurrent sandboxes. OK for a few tasks; full suite is slow.
- **Tier 3** (250 vCPU / 500 GB RAM pool): can run `-n 32` to `-n 48` concurrent — full suite in ~10–15 minutes.

Harbor auto-requests per-task resources (most tasks: 1 vCPU / 2 GB; a few: 4 vCPU / 8 GB).

---

## 7. Point Harbor at the BatchGen endpoint

Harbor uses LiteLLM under the hood. The `openai/` prefix routes through OpenAI-compatible HTTP.

```bash
export OPENAI_BASE_URL="http://<node0-ip>:10900/v1"
export OPENAI_API_KEY="sk-dummy"   # batchgen does not validate
```

---

## 8. Run Terminal-Bench 2.0

Smoke test on a single task first:

```bash
harbor run \
    -d terminal-bench/terminal-bench-2 \
    -m openai/zai-org/GLM-5.1-FP8 \
    -a terminus-2 \
    --env daytona \
    --task-id hello-world
```

If that resolves, run the full suite:

```bash
harbor run \
    -d terminal-bench/terminal-bench-2 \
    -m openai/zai-org/GLM-5.1-FP8 \
    -a terminus-2 \
    --env daytona \
    -n 32
```

Notes:
- `-m openai/<model>` — `<model>` must match BatchGen's `--model` argument exactly.
- `-a terminus-2` — Harbor's default agent. `claude-code` and custom `BaseInstalledAgent` subclasses also work.
- `--env daytona` — Harbor allocates one Daytona sandbox per task, no local Docker needed.
- `-n 32` — concurrent tasks. Bound by Daytona account tier and BatchGen server throughput.

Expected wall-clock (reference numbers from Harbor docs, agent-dependent):

| Environment | Concurrency | Full suite (~200 tasks) |
|---|---|---|
| Local Docker | 4 | ~90 min |
| **Daytona Cloud** | **48** | **~10–15 min** |

---

## 9. Inspect and submit results

Harbor writes per-trial logs and the aggregated scoreboard to `./runs/<run-id>/`:

```bash
ls runs/
cat runs/<run-id>/results.json
```

For leaderboard submission, follow the README at the [Terminal-Bench HuggingFace repo](https://huggingface.co/datasets/terminal-bench).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `tool_calls: null` in BatchGen responses | `--tool-call-parser` not set, or per-request `tools` field missing | Confirm `--tool-call-parser glm` is in the launch command; Harbor's agents inject `tools` automatically |
| Empty `content`, only `<tool_call>` tokens | `--tool-call-parser glm` missing | Restart server with the flag |
| `<think>…</think>` leaking into agent input | `--parse-thinking` not set | Restart server with the flag |
| Daytona "out of quota" | Concurrency exceeds account tier | Lower `-n` or upgrade tier |
| `litellm.BadRequestError: model not found` | `-m openai/<name>` doesn't match `--model` | Match strings byte-for-byte, including the `zai-org/` prefix |
| Harbor hangs on first task | DNS or network to BatchGen blocked from Daytona sandbox | Ensure BatchGen's `--listen-ip 0.0.0.0` is publicly reachable from Daytona; consider a Cloudflare Tunnel or ngrok |

---

## See also

- [`docs/support-glm-5.1.md`](support-glm-5.1.md) — GLM-5.1 model registration internals
- [`docs/server-flags.md`](server-flags.md) — full `batchgen-server` CLI reference, including `--tool-call-parser`
- [`docs/output-format.md`](output-format.md) — response shape with `--parse-thinking` and `--tool-call-parser`
- [`batchgen/function_call/`](../batchgen/function_call/) — tool-call detector registry (`glm`, `glm45`, `glm47`, `deepseekv3`, `kimi_k2`, …)
- [Harbor docs](https://harborframework.com/docs/tutorials/running-terminal-bench) — agent options, leaderboard submission
- [Daytona Python SDK](https://www.daytona.io/docs/en/python-sdk) — if you want to drive sandboxes programmatically instead of via Harbor
