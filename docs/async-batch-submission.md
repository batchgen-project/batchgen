# Async Batch Submission — Tutorial

When a single caller submits **multiple batches that should run concurrently**
— for example a stress test that simulates production traffic arriving at
staggered intervals — the blocking `submit_batch()` convenience method gets in
the way. This guide shows the async pattern that uses the same public
`BatchGenHttpClient` API, without any new dependencies.

For the one-shot blocking path (upload + create + wait + download in a single
call), see [`batch-api-guide.md`](batch-api-guide.md) instead.

## Two levels of client API

`BatchGenHttpClient` (`batchgen/batchgen_client.py`) exposes both low-level and
high-level methods. Both are public API — pick the level that matches your
control-flow needs.

| Level | Method | Blocks? | Use case |
|---|---|---|---|
| low  | `upload_file(path, purpose="batch")` | no | Upload input JSONL, get `file_id`. |
| low  | `create_batch(file_id, endpoint, …)` | no | Enqueue the batch into the server's IntakePool; returns immediately with `batch_id`. |
| low  | `get_batch(batch_id)` | no | Poll status of a single batch. |
| low  | `wait_for_batch(batch_id, poll_interval, timeout)` | **yes** | Poll until terminal status. |
| low  | `download_file_content(file_id)` | no | Fetch an output JSONL by `output_file_id`. |
| high | `submit_batch(path, …)` | **yes** | Convenience wrapper = `upload_file` → `create_batch` → `wait_for_batch` → (optional) download. |

> **Key fact.** `create_batch` is non-blocking on both sides. As soon as it
> returns, the server has enqueued the requests into the IntakePool, and the
> scheduler is free to start admitting them into the per-GPU scheduling pool
> **concurrently with any other active batch**. Multi-batch interleaving is a
> first-class server feature since v1.0.7.

## The pattern: upload + create, poll asynchronously

This is the canonical async flow. It submits `N` batches at configured
intervals, tracks them in an `active` list, polls via `get_batch`, and exits
when all reach terminal status.

```python
from batchgen.batchgen_client import BatchGenHttpClient
import time, random

client = BatchGenHttpClient("http://localhost:10900")

batch_paths = ["batch_0.jsonl", "batch_1.jsonl", "batch_2.jsonl", "batch_3.jsonl"]
active = []   # [(batch_id, submit_time), ...]

for i, path in enumerate(batch_paths):
    # Stagger submissions (skip delay for the first).
    if i > 0:
        time.sleep(random.randint(300, 1200))   # 5-20 min

    # Async submit: upload + create return immediately; we do NOT block on
    # completion here. The server begins admitting this batch's sequences
    # to the GPU scheduling pool right away.
    file_obj = client.upload_file(path, purpose="batch")
    batch    = client.create_batch(
        file_obj["id"],
        endpoint="/v1/chat/completions",
    )
    active.append((batch["id"], time.time()))
    print(f"Submitted {batch['id']} (batch {i+1}/{len(batch_paths)})")

# All batches submitted. Wait for every in-flight batch to terminate.
while active:
    still_active = []
    for bid, submitted_at in active:
        b = client.get_batch(bid)
        if b["status"] in ("completed", "failed", "cancelled"):
            elapsed = time.time() - submitted_at
            print(
                f"{bid}: {b['status']} | "
                f"{elapsed:.0f}s | output_file_id={b.get('output_file_id')}"
            )
        else:
            still_active.append((bid, submitted_at))
    active = still_active
    if active:
        time.sleep(10)
```

### Polling while waiting for the next interval

In a stress harness, the interval between batch submissions is often long
(minutes). Use that time to keep the `active` list's status fresh — you can
detect completed batches early instead of only finding out in the drain loop:

```python
def poll_all(client, active):
    """Return the subset of `active` that is still non-terminal."""
    still_active = []
    for bid, submitted_at in active:
        b = client.get_batch(bid)
        if b["status"] not in ("completed", "failed", "cancelled"):
            still_active.append((bid, submitted_at))
    return still_active

# Inside the inter-batch wait:
wait_end = time.time() + random.randint(300, 1200)
while time.time() < wait_end:
    if active:
        active = poll_all(client, active)
    time.sleep(min(10, max(0, wait_end - time.time())))
```

## What happens when a batch completes

You generally **don't need to call `download_file_content`** unless you want
the result JSONL on the client machine. The server does the work for you:

1. **During the run**, each completed sequence is appended to
   `{storage_path}/incremental/{batch_id}.jsonl`. This is crash-resilient — if
   the server restarts mid-run, the partial output survives.
2. **On batch completion** (inside `_finalize_batch_output` in
   `batchgen/server/batch_scheduler.py`), the server generates an
   `output_file_id = "file-<uuid>"` and `shutil.copy2`'s the incremental file to
   both:
   - `{storage_path}/outputs/{output_file_id}.jsonl` — user-visible,
     filesystem-browsable.
   - `{storage_path}/files/{output_file_id}` — served by
     `GET /v1/files/{output_file_id}/content`.
3. The batch record's `output_file_id` field is set. `get_batch(batch_id)`
   returns it; you can then use it either via the API or by reading the file
   directly if you have filesystem access to the server's storage volume.

In short: **the result is already in the result dir at the moment
`get_batch` reports `status="completed"`**. Downloading via the API is
optional.

## Backpressure

`create_batch` returns **HTTP 429** if the server's IntakePool is full. The
whole batch is rejected atomically (not partially admitted). `BatchGenHttpClient`
already implements exponential-backoff retry for 429
(`1s → 2s → 4s → 8s → 16s`, capped at 60 s, up to 5 attempts; honors
`Retry-After` headers). You do not need to write retry logic yourself.

Raw HTTP / `curl` callers must handle 429 themselves.

## Observing concurrency from the server side

Use the pool-status endpoint to confirm batches are interleaving:

```bash
curl -s http://localhost:10900/v1/pool/status
```

```json
{
  "intake_pool_size":        0,
  "intake_pool_capacity":    1000000,
  "scheduling_pool_active":  3420,
  "scheduling_pool_free":    2580,
  "scheduling_pool_capacity": 6000,
  "active_batches":          4,
  "pool_mode":               true
}
```

Healthy async stress-test indicators:

- `active_batches` grows above 1 after the first inter-batch interval elapses.
- `scheduling_pool_active` rises well beyond any single batch's admitted-seq
  count.
- `intake_pool_size` normally drains near-instantly — requests spend very
  little time there. Persistently non-zero `intake_pool_size` means either
  (a) a batch is larger than the scheduling pool's free capacity, or (b) the
  scheduler has stopped admitting (likely a backpressure condition worth
  investigating, e.g. host KV pressure).

## Common pitfalls

1. **Don't call `submit_batch()` in a loop expecting concurrency.** It blocks
   on `wait_for_batch` internally, so each iteration finishes before the next
   submission starts — the opposite of async.
2. **`intake_pool_size` is *not* a queue of pending batches.** It's the
   per-request queue inside the server; it drains near-instantly under normal
   operation. Batches pending in the client's own `for` loop have no server
   visibility at all.
3. **Retry on timeouts, not on 429.** The client retries 429 for you. If
   `get_batch` or `create_batch` times out (network), retry at the application
   layer.
4. **`completion_window`** is accepted by `create_batch` for OpenAI API
   compatibility but BatchGen does not enforce it (batches run until they
   complete or a higher-level cancellation arrives).

## See also

- [`batch-api-guide.md`](batch-api-guide.md) — end-to-end guide including the
  blocking `submit_batch` quickstart.
- [`batch-api.md`](batch-api.md) — REST endpoint reference.
- [`client-api.md`](client-api.md) — full `BatchGenHttpClient` method reference.
