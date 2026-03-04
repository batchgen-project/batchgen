# Watchdog & Health Monitoring

BatchGen includes three watchdog mechanisms and a health endpoint for detecting stuck inference, failed startup, and reporting server status to external orchestrators.

---

## Overview

| Mechanism | Flag | Monitors | Default |
|-----------|------|----------|---------|
| General watchdog | `--watchdog-timeout` | Per-step/micro-batch progress (prefill + decode) | Disabled |
| Decode watchdog | `--decode-step-timeout` | Individual decode iterations in the hot loop | Disabled |
| Startup watchdog | `--startup-timeout` | Time from process launch to server ready | Disabled |

- **General/decode watchdog**: On timeout, the watchdog sends `SIGQUIT` to the parent process, triggering graceful shutdown. The `/health` endpoint returns `503 Unhealthy` if the server is still reachable during shutdown.
- **Startup watchdog**: On timeout, the process exits immediately with code 1 and error logs. Since `/health` isn't serving yet during startup (uvicorn blocks on worker initialization), the caller detects failure via the process exit code.

---

## Flag Semantics

### `--decode-step-timeout <seconds>`

Maximum wall-clock time allowed for a **single decode iteration** in the continuous decoding loop. The decode watchdog is a per-worker daemon thread that monitors the decode hot path only — it is **disabled** during worker initialization, weight loading, CUDA graph capture, NCCL warmup, and idle waiting between requests. It is enabled when the worker enters `decoding_continuous()` and disabled when it exits.

Each decode iteration includes one forward pass plus scheduling overhead (page boundary checks, KV management). On healthy hardware, a single decode step typically completes in milliseconds to low seconds depending on batch size and model. A timeout here indicates a stuck NCCL collective, GPU hang, or deadlock.

On timeout, the watchdog dumps a py-spy stack trace for diagnostics, then sends `SIGQUIT` to the parent process.

**Recommended value**: 300s (5 min) for production. Set higher for very large batch sizes or slow interconnects.

### `--startup-timeout <seconds>`

Maximum wall-clock time from process launch to server ready. "Server ready" means all worker processes have completed initialization (weight loading to VRAM, NCCL process group setup, host KV cache allocation, CUDA graph capture) and the HTTP server is accepting requests.

The timer runs as a daemon thread in the main process, started before `uvicorn.run()`. It polls `health_state.is_startup_complete()` every second. If the deadline expires before startup completes, the process calls `os._exit(1)` — an immediate exit that bypasses Python cleanup. This is necessary because during a stuck startup (e.g., NCCL init deadlock), the main thread is blocked in the lifespan and cannot respond to signals.

The caller (shell, systemd, K8s) detects failure via the non-zero exit code.

**Recommended value**: 1800s (30 min) for large models like DeepSeek-R1 on H20. Normal startup takes ~5-16 minutes depending on model size and storage speed. Set higher for cold starts with model download.

### `--watchdog-timeout <seconds>`

General per-step/micro-batch timeout covering both prefill and decode phases. The watchdog counter is incremented (`feed()`) after each prefill micro-batch and each decode step. If no progress is made within the timeout period, the watchdog fires.

This is the broadest timeout — it catches any situation where the worker stops making progress. The decode watchdog (`--decode-step-timeout`) is a more targeted version that only monitors the decode loop.

**Recommended value**: 600s (10 min) for production. Needs to be generous enough to accommodate large prefill batches.

---

## Health Endpoint

### Querying Health Status

**curl:**

```bash
curl -s http://localhost:10900/health | python3 -m json.tool
```

Healthy response (HTTP 200):
```json
{"status": "healthy"}
```

Unhealthy response (HTTP 503):
```json
{"status": "unhealthy", "reason": "Worker process exited."}
```

**Shell polling loop (for spot instance scripts):**

```bash
while true; do
  status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:10900/health)
  if [ "$status" != "200" ]; then
    echo "Server unhealthy! Status code: $status"
    curl -s http://localhost:10900/health
    break
  fi
  sleep 30
done
```

**Python client:**

```python
import requests

resp = requests.get("http://localhost:10900/health")
if resp.status_code == 200:
    print("Server is healthy")
else:
    print(f"Server unhealthy: {resp.json()['reason']}")
```

**Kubernetes liveness probe:**

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 10900
  initialDelaySeconds: 1800  # Allow time for model loading
  periodSeconds: 30
  failureThreshold: 3
```

### What Triggers Unhealthy Status

| Condition | HTTP Response | Reason String |
|-----------|--------------|---------------|
| Worker process crashed | 503 | `"Worker process exited."` |
| Decode watchdog timeout | 503 (if server still reachable) | via SIGQUIT shutdown |
| General watchdog timeout | 503 (if server still reachable) | via SIGQUIT shutdown |
| Startup timeout | N/A (process exits with code 1) | Error in server logs |

---

## Production Configuration

Recommended flags for production / spot instance deployments:

```bash
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --watchdog-timeout 600 \
    --decode-step-timeout 300 \
    --startup-timeout 1800
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  launch_http_server.py (main process)               │
│                                                     │
│  ┌──────────────┐   ┌────────────────────────────┐  │
│  │ Startup Timer │   │ uvicorn (FastAPI)           │  │
│  │ (daemon thrd) │   │                            │  │
│  │               │   │  /health ──► ServerHealth   │  │
│  │ On timeout:   │   │             State           │  │
│  │ os._exit(1)   │   │             + WorkerExit    │  │
│  │               │   │             State           │  │
│  └──────────────┘   └────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ Worker Processes (per GPU rank)              │   │
│  │                                              │   │
│  │  General Watchdog ◄── feed() on each step    │   │
│  │  Decode Watchdog  ◄── feed() on decode iter  │   │
│  │  (disabled during init, enabled in decode)   │   │
│  │                                              │   │
│  │  On timeout: SIGQUIT → parent                │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

`ServerHealthState` is a thread-safe object shared across:
- Uvicorn async handler (`/health` endpoint reads it)
- Worker exit callback (sets unhealthy on worker crash)

The general and decode watchdogs run in worker processes. On timeout, they send `SIGQUIT` to the parent (main) process, which triggers graceful shutdown via the signal handler.

The startup timer runs as a daemon thread in the main process. On timeout, it calls `os._exit(1)` to force-terminate.

---

## Troubleshooting

### Common Timeout Scenarios

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Startup timeout on first launch | Weight loading from slow NFS/S3 | Increase `--startup-timeout` or use local `--cache-dir` |
| Startup timeout on multi-node | NCCL init waiting for all nodes | Ensure all nodes start within the timeout window |
| Decode watchdog triggers on first batch | First decode step includes CUDA warmup overhead | Increase `--decode-step-timeout` (300s recommended) |
| Decode watchdog triggers consistently | Stuck NCCL collective or GPU hang | Check `nvidia-smi`, NCCL debug logs (`NCCL_DEBUG=INFO`) |
| General watchdog triggers during prefill | Very long prompts with large batch | Increase `--watchdog-timeout` |

### Diagnosing Stuck Processes

When a watchdog triggers, it dumps a py-spy stack trace before killing the process. Check the server logs for the dump.

To manually check if a process is stuck:

```bash
# Check GPU utilization (should be >0% during inference)
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1

# Check if the server is responding
curl -s http://localhost:10900/health

# Check NCCL for deadlocks (enable debug logging)
NCCL_DEBUG=INFO python -m batchgen.launch_http_server ...
```

---

## See Also

- [Server Flags Reference](server-flags.md) — Complete CLI flag reference
- [Deployment Guide](deploy-deepseek-r1-h20.md) — Multi-node deployment
