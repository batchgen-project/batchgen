# Deployment sequence length

`--max-model-len` sets the deployment's prompt-plus-completion limit, `L`.
The default is the registered model context length. An explicit value must be
between 2 and that context length. `--max-model-len=-1` opts into startup
auto-fit; other values never silently shrink to fit memory.

Before worker readiness, the worker loads decode residency, measures the GPU
pool budget, and checks every rank and every node's host KV regions. The
node-shared input-ID pool must also hold one sequence. GPU regions use the
same equal-page-count budget split as allocation. For a region with `P` tokens
per page, `B` bytes per page across its physical layers, and replication factor
`G`, one sequence requires `ceil(L / P) * G` pages and that count times `B`
bytes. GPU checks run per rank with `G=1`; host checks conservatively reserve
the attention group size on every node. Input IDs use one copy.

`MaxModelLenCapacityError` names the resource, region, node, rank, required and
available pages/bytes, and estimated feasible length. Each resource permits
`floor(available_pages / G) * P` tokens. Auto-fit chooses the smallest of these
resource bounds and the model context. A capacity below 2 tokens still fails.

The GPU budget input is `kv_plan.provisional_pool_gb(total_gb, gpu_memory_frac,
used_gb, floor_gb=None) = total_gb * gpu_memory_frac - used_gb`, measured with decode
weights resident. The tightest rank sets the common pool size. Startup freezes
this budget in every mode, before any graph capture. The late post-capture
resizing hook is therefore inactive for these deployments. The provisional
pool retained by persistent phase instances must also support `L`. Phase switches retain the existing destruction or
fixed-address unmapping lifecycle; persistent phase instances retain buffers.

The worker publishes the verified `max_model_len` before signalling readiness.
The scheduler consumes that publication before HTTP readiness. `GET /v1/models`
and `GET /v1/models/{model_id}` read the current scheduler value on each call,
including after worker restart. Both `max_model_len` and the compatibility
field `max_context_length` report `L`; an unpublished generation returns 503.

After tokenization, a prompt with at least `L` tokens fails on its own. Batch
output uses `response: null` and an `invalid_prompt` error with
`status_code: 400`; other requests continue. Turns return HTTP 400. Budget
precedence is `max_completion_tokens`, `max_tokens`, batch `max_decoding_length`,
then `L - prompt_length`. Explicit budgets are bounded by the remaining context
before building an admission entry. Every entry carries `input_ids` as a flat
int32 array and a required `max_tokens`.

This change does not replace mode-dependent decode padding or the existing
fraction-based slack with a measured eager transient reservation. That requires
a startup forward protocol at the per-rank cap for every model, including lazy
graph capture and allocations outside the tensor allocator. Until then, the
startup check proves the KV length contract against the measured pool budget;
it is not a measurement of every later eager allocation.

The same publication carries `startup_transient`: per flavor (`eager`,
`capture`), the maximum over ranks of the device memory the startup decode
configure costs above steady state, measured right after the pool is sized
(`batchgen/worker/startup_transient.py`). The scheduler requires it with
`max_model_len`, and `/v1/models` reports it. It is report-only: the pool
budget does not deduct it. A flavor the deployment cannot measure is published
with an `unavailable` reason instead of a value: eager needs a model-provided
cap-sized synthetic decode step, and capture is measured only where the
lifecycle captures at startup.

The same publication carries the cohort's `code_identity` and `cohort` (the
restart epoch that spawned it). Local rank 0 of each node computes the
identity at the top of its start-up, before the model build
(`batchgen/worker/code_identity.py`), because a worker restart re-imports
Python from disk and the server process cannot see what the new cohort
imported. `source=git`: the commit of the work tree holding the `batchgen`
package and whether that tree differs from it (one `git status` per cohort
start); `source=none`: no git metadata beside the package (an installed
package records no revision). Anything git cannot answer is revision
`unknown` with a fixed reason. `WorkerManager.record_code_identity` logs one
line per generation and node before the scheduler marks the deployment ready:

    [CODE_IDENTITY] generation=<G> node=<N> source=<git|none> revision=<commit|unknown> dirty=<true|false|unknown> reason=<reason|->

`GET /v1/workers/status` reports `code_identity` (`source`, `revision`,
`dirty`, `reason`) for the current generation. While a restart is in flight
the outgoing generation is still current, so its identity stays reported next
to state `restarting`; after a failed restart it is null. The identity names a
revision, never a directory.

The publication also carries the cohort's `server_config`
(`batchgen/worker/server_config.py`): the parallel layout the model's
strategy manager declares per phase (`batchgen/parallel_declaration.py`;
an undeclared manager reports `undeclared`, never a guessed default), the
CUDA-graph flag with the owning model's installed or captured bucket ladder, the host- and device-KV
pool sizes and page counts, the GPU memory fraction and the derived
scheduling capacity. Local rank 0 of each node snapshots it while the decode
pool and captured graphs are live (`_snapshot_server_config`, before the
startup switch to prefill), and `WorkerManager.record_server_config` logs one
line per generation and node before the scheduler marks the deployment ready:

    [SERVER_CONFIG] generation=<G> node=<N> prefill_attn=<slot> prefill_moe=<slot> decode_attn=<slot> decode_moe=<slot> cuda_graphs=<on|off> graph_buckets=<b1,b2,...|none> host_kv_gib=<G.GGG> host_kv_pages=<P> device_kv_gib=<G.GGG> device_kv_pages=<P> gpu_memory_frac=<F> max_pool=<N>

where `<slot>` is `undeclared`, `none`, or a `scheme:residency` token whose
sharded schemes carry their group width: `dp:streamed`, `tp8:resident`,
`ep8:partial`.

# Batch request counts

`BatchObject.request_counts` (`total`, `completed`, `failed`) is decided per
request in one place, `BatchScheduler._record_pool_result`, which every batch
request result passes through (worker completions and admission rejections). It counts the result
on the `SchedulingPool` tracker (`record_request_result`, once per request;
an `error` result is failed) and only then writes the output item, so the
output file and the counts cannot diverge. `total` is persisted at
registration. While the tracker exists, `with_live_request_counts` serves its
live counts on `GET /v1/batches`, `GET /v1/batches/{id}` and cancel; the
completed and failed transitions persist the tracker's final counts. Startup
recovery (`StorageManager.fail_orphaned_batches`) rebuilds completed/failed of
an orphaned batch from its incremental output items. Records written before
the field existed load with zero counts.
