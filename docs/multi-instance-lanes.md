# Multi-instance lane operations

`scripts/lane_runtime.py` is the fail-closed operational boundary for running
qualified BatchGen servers on disjoint GPU subsets of one host. It complements
the core runtime isolation in the preceding stacked PRs; it is not a general
scheduler.

The initial qualification is deliberately narrow:

- model `openai/gpt-oss-120b`;
- one node;
- 1, 2, 4, or 8 physical GPU UUIDs per lane;
- explicit host-memory and `/dev/shm` reservations; and
- explicit, non-overlapping HTTP, distributed-init, and bounded PyNccl ports.

Before any remote `start`, `stop`, `verify`, or `exclusive-run`, operators and
agents must run the workspace's machine identity gate for their exact,
authorized alias. The public repository intentionally contains no machine
endpoint or credential data.

## Start a lane

The worktree must contain both `batchgen/` and `batchgen_kernels/`. The tool
creates a lane-local `pyroot` whose two package links point to that same
worktree, plus isolated storage, conversion, temporary, compiler-cache, and log
directories. The lane root must be disjoint from the worktree and every active
lane root, including their logs and `pyroot` directories. Checkpoints may be
shared read-only; runtime writable paths are lane-isolated.

```bash
python scripts/lane_runtime.py start \
  --instance-id lane-0 \
  --gpu-uuid GPU-... \
  --gpu-uuid GPU-... \
  --listen-port 11000 \
  --dist-port 12000 \
  --pynccl-port-base 21000 \
  --pynccl-port-span 8 \
  --host-kv-cache-gb 16 \
  --host-memory-gb 192 \
  --shm-gb 128 \
  --checkpoint /read-only/checkpoints/gpt-oss-120b \
  --o200k-base-file /read-only/tokenizers/o200k_base.tiktoken \
  --converted-ckpt-dir /runtime/lanes/lane-0/converted \
  --worktree /runtime/worktrees/BatchGen-lane-0 \
  --lane-root /runtime/lanes/lane-0 \
  --python /runtime/env/bin/python
```

Admission is serialized. The launcher rejects ambiguous stale manifests,
overlapping GPU UUIDs, ports, PyNccl ranges, or writable paths, foreign compute
processes on assigned GPUs, and unsafe memory/SHM budgets. It transfers live
resource-lock descriptors into the server and waits until the server holds its
logical-instance lock before returning.

For the qualified GPT-OSS model, each SHM reservation must cover the model
weight region, Host-KV allocation, and transient SHM headroom. The host-memory
reservation must additionally cover private runtime headroom. Underreported
lane budgets are rejected before launch.

The GPT-OSS tokenizer uses `tiktoken`'s `o200k_base` encoding. Supply the
encoding asset via `--o200k-base-file`; startup verifies its SHA-256
(`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`)
and seeds a private cache in that lane's temporary directory. The child server
uses that cache even if the parent shell has a different tiktoken cache set.
This asset is distinct from Hugging Face's model-root `tokenizer.json`.

## Inspect, verify, and stop

```bash
python scripts/lane_runtime.py status --instance-id lane-0
python scripts/lane_runtime.py verify --instance-id lane-0
python scripts/lane_runtime.py stop --instance-id lane-0 --timeout 30
```

`verify` revalidates boot ID, PID start time, process group, GPU ownership, and
every resource lock. Linux PIDFD support is required to start and stop live
lanes. `stop` opens a PIDFD, revalidates the owner after opening it, and sends
TERM only through that descriptor. If graceful stop times out, it sends KILL
only to that same owner descriptor while its identity still matches; it never
signals a numeric PID or process group. If the owner exits while its process
group remains, or run-owned SHM/runtime directories remain, stop fails closed
and leaves evidence in place. There is no TTL or stale-state takeover.

## Exclusive whole-node operations

Whole-node cleanup must hold the exclusive host lock. For the qualified H200
family, the workspace's cleanup wrapper acquires that lock on the exact
requested machine before it checks or removes anything. Set
`EXACT_ASSIGNED_ALIAS` from the approved private registry first:

```bash
bash scripts/remote/server_clean_h200.sh \
  --machine "$EXACT_ASSIGNED_ALIAS" \
  --purpose multi-instance-preflight
bash scripts/remote/verify_clean_h200.sh \
  --machine "$EXACT_ASSIGNED_ALIAS" \
  --purpose multi-instance-preflight
```

The clean command fails immediately while any shared lane (including an orphaned
worker that inherited the host lock) is alive. The strict zero-GPU, clean-SHM,
and host-memory verifier must pass before a new fleet starts.

## Qualification boundary

Do not call the feature complete from unit tests alone. On the exact assigned
machine, run Q1, Q2, Q4, and Q8 as separate correctness regimes with
an exclusive clean/verify between regimes. In each co-resident regime, keep
the unaffected lanes under health/correctness probes while stopping,
restarting, and injecting a controlled startup failure into one lane.
