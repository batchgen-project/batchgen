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
