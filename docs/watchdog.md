# Watchdog & Health Monitoring

BatchGen includes three watchdog mechanisms and a health endpoint for detecting stuck inference, failed startup, and reporting server status to external orchestrators.

---

## Overview

| Mechanism | Flag | Monitors | Default |
|-----------|------|----------|---------|
| General watchdog | `--watchdog-timeout` | Per-step/micro-batch progress (prefill + decode) | Disabled |
| Decode watchdog | `--decode-step-timeout` | Individual decode iterations in the hot loop | Disabled |
| Startup watchdog | `--startup-timeout` | Time from process launch to server ready | Disabled |

The general and decode watchdogs integrate with the `/health` endpoint — when triggered, `/health` returns `503 Unhealthy`. The startup watchdog exits the process with code 1 (since `/health` isn't serving yet during startup).

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
{"status": "unhealthy", "reason": "Decode step timeout (300s) exceeded"}
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

| Condition | Reason String |
|-----------|--------------|
| Worker process crashed | `"Worker process exited."` |
| Decode step exceeded timeout | `"Decode step timeout (Xs) exceeded"` |
| Startup exceeded timeout | `"Startup timeout (Xs) exceeded"` |

- **General/decode watchdog**: The server continues running with `/health` returning 503, allowing external orchestrators to detect the failure.
- **Startup watchdog**: The process exits with code 1 and error logs (since `/health` isn't serving during startup).

---

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--watchdog-timeout` | float | None (disabled) | General per-step/micro-batch timeout in seconds |
| `--decode-step-timeout` | float | None (disabled) | Max seconds for a single decode iteration |
| `--startup-timeout` | float | None (disabled) | Max seconds from process launch to server ready |
| `--no-watchdog` | flag | - | Explicitly disable watchdog (default behavior) |

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

- **`--watchdog-timeout 600`** (10 min): General stuck detection. Covers prefill micro-batches and decode steps. Use a generous value to avoid false triggers during large prefills.
- **`--decode-step-timeout 300`** (5 min): Per-decode-step timeout. Catches stuck NCCL collectives or GPU hangs during decode. Shorter than the general watchdog since individual decode steps should complete quickly.
- **`--startup-timeout 1800`** (30 min): Startup timeout. Covers weight loading, NCCL initialization, hugepage allocation, and CUDA graph capture. Increase for very large models or slow storage.

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
│  │ set_unhealthy │   │             + WorkerExit    │  │
│  │ + SIGQUIT     │   │             State           │  │
│  └──────────────┘   └────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ Worker Processes (per GPU rank)              │   │
│  │                                              │   │
│  │  General Watchdog ◄── feed() on each step    │   │
│  │  Decode Watchdog  ◄── feed() on decode iter  │   │
│  │                                              │   │
│  │  On timeout: SIGQUIT → parent                │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

`ServerHealthState` is a thread-safe object shared across:
- Startup timer thread (sets unhealthy on timeout)
- Uvicorn async handler (`/health` endpoint reads it)
- Worker exit callback (sets unhealthy on worker crash)

The general and decode watchdogs run in worker processes. On timeout, they send `SIGQUIT` to the parent (main) process, which triggers graceful shutdown via the signal handler.

---

## Troubleshooting

### Common Timeout Scenarios

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Startup timeout on first launch | Weight loading from slow NFS/S3 | Increase `--startup-timeout` or use local `--cache-dir` |
| Startup timeout on multi-node | NCCL init waiting for all nodes | Ensure all nodes start within the timeout window |
| Decode watchdog triggers randomly | Insufficient `--decode-step-timeout` for large batches | Increase timeout or reduce batch size |
| Decode watchdog triggers consistently | Stuck NCCL collective or GPU hang | Check `nvidia-smi`, NCCL debug logs (`NCCL_DEBUG=INFO`) |
| General watchdog triggers during prefill | Very long prompts with large batch | Increase `--watchdog-timeout` |

### Diagnosing Stuck Processes

When a watchdog triggers, it dumps a py-spy flame graph before killing the process. Check the server logs for the dump location.

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
