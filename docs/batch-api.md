# Batch REST API Reference

OpenAI-compatible REST API for batch inference. Base URL: `http://<host>:<port>` (default port `10900`).

## Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{model_id}` | Get model metadata |
| POST | `/v1/files` | Upload input file |
| GET | `/v1/files` | List files |
| GET | `/v1/files/{file_id}` | Get file metadata |
| DELETE | `/v1/files/{file_id}` | Delete file |
| GET | `/v1/files/{file_id}/content` | Download file content |
| POST | `/v1/batches` | Create batch job |
| GET | `/v1/batches` | List batches |
| GET | `/v1/batches/{batch_id}` | Get batch status |
| POST | `/v1/batches/{batch_id}/cancel` | Cancel batch |
| POST | `/v1/sessions` | Open an agent session |
| POST | `/v1/sessions/{session_id}/close` | Close a session (end of session / trajectory) |
| POST | `/v1/sessions/{session_id}/abort` | Abort a session and its in-flight turn |
| POST | `/v1/turns` | Run one chat-completions turn of a session |
| GET | `/health` | Health check |

---

## Model Endpoints

### GET /v1/models

List the model currently loaded on the server.

```bash
curl http://localhost:10900/v1/models
```

**Response:** `ListModelsResponse`

```json
{
  "object": "list",
  "data": [
    {
      "id": "Kimi-K2.5",
      "object": "model",
      "created": 1711234567,
      "owned_by": "batchgen",
      "max_context_length": 262144
    }
  ]
}
```

### GET /v1/models/{model_id}

Retrieve metadata for a specific model.

```bash
curl http://localhost:10900/v1/models/Kimi-K2.5
```

**Response:** `ModelObject`

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Model name (last component of model path) |
| `object` | string | Always `"model"` |
| `created` | integer | Unix timestamp when server started |
| `owned_by` | string | Always `"batchgen"` |
| `max_context_length` | integer | Maximum context length in tokens (prompt + completion) |

**Error:** Returns `404` if `model_id` does not match the loaded model.

---

## File Endpoints

### POST /v1/files

Upload a JSONL input file for batch processing.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | The JSONL file to upload |
| `purpose` | string | Yes | Must be `"batch"` |

**Response:** `FileObject`

```json
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 1024,
  "created_at": 1710000000,
  "filename": "requests.jsonl",
  "purpose": "batch",
  "status": "uploaded",
  "checksum": "sha256:..."
}
```

### GET /v1/files

List uploaded files.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `purpose` | string | None | Filter by purpose |
| `limit` | int | 10000 | Max results (1-10000) |
| `order` | string | `desc` | Sort order (`asc` or `desc`) |
| `after` | string | None | Cursor for pagination |

**Response:** `ListFilesResponse`

```json
{
  "data": [ ...FileObjects... ],
  "has_more": false
}
```

### GET /v1/files/{file_id}

Get metadata for a specific file.

**Response:** `FileObject` (same schema as upload response)

### DELETE /v1/files/{file_id}

Delete a file.

**Response:**

```json
{
  "id": "file-abc123",
  "deleted": true,
  "object": "file"
}
```

### GET /v1/files/{file_id}/content

Download file content (input or output JSONL).

**Response:** Raw file bytes with `Content-Disposition: attachment` header.

---

## Batch Endpoints

### POST /v1/batches

Create a batch job from an uploaded input file.

**Request:** `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input_file_id` | string | Yes | - | ID of uploaded JSONL file |
| `endpoint` | string | No | `/v1/chat/completions` | Target endpoint (`/v1/chat/completions` or `/v1/completions`) |
| `completion_window` | string | No | `24h` | Completion time window |
| `metadata` | object | No | None | Arbitrary metadata |
| `max_decoding_length` | int | No | None | Batch-level fallback max output tokens (see [Input Format](input-format.md#output-length-priority)) |
| `max_context_length` | int | No | None | Max total context length (prompt + decode). None = model maximum. |
| `temperature` | float | No | None | Default sampling temperature. Per-request values override. |
| `top_p` | float | No | None | Default nucleus sampling threshold. Per-request values override. |
| `top_k` | int | No | None | Default top-k filtering. Per-request values override. |

**Response:** `BatchObject`

```json
{
  "id": "batch_abc123",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "input_file_id": "file-abc123",
  "output_file_id": null,
  "completion_window": "24h",
  "status": "validating",
  "created_at": 1710000000,
  "expires_at": 1710086400,
  "started_at": null,
  "completed_at": null,
  "cancelled_at": null,
  "cancelling_at": null,
  "error": null,
  "metadata": null,
  "max_decoding_length": 1024,
  "max_context_length": null,
  "temperature": null,
  "top_p": null,
  "top_k": null
}
```

### GET /v1/batches

List batch jobs.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Max results (1-100) |
| `after` | string | None | Cursor for pagination |

**Response:**

```json
{
  "data": [ ...BatchObjects... ],
  "has_more": false
}
```

### GET /v1/batches/{batch_id}

Get current status of a batch job. Use this endpoint to poll for completion.

**Response:** `BatchObject` (same schema as create response, with updated status and timestamps)

### POST /v1/batches/{batch_id}/cancel

Cancel a batch in `validating` or `in_progress` state (other states return an error). The batch is marked `cancelled` immediately; `cancelling_at` and `cancelled_at` are set to the same timestamp. A cancelled batch that has not started yet is skipped by the scheduler; sequences already executing are not interrupted.

**Response:** `BatchObject` with `status: "cancelled"`

---

## Batch Status Flow

```
validating → in_progress → completed
                         → failed
                         → cancelled
```

| Status | Description |
|--------|-------------|
| `validating` | Input file is being parsed and validated |
| `in_progress` | Sequences are being processed |
| `completed` | All sequences finished. `output_file_id` is set. |
| `failed` | Processing failed. `error` field has details. |
| `cancelling` | Reserved. Defined in the API schema but currently never emitted — cancel sets `cancelled` directly. |
| `cancelled` | Batch was cancelled. `cancelled_at` is set. |

**Timestamps:** `started_at`, `completed_at`, `cancelled_at`, `cancelling_at` are set as the batch transitions through states.

---

## Agent Turn and Session Endpoints

For an offline agent harness that sends one model turn at a time: a turn is
answered in its own HTTP response, with no file upload and no batch polling.
Each turn carries its trajectory and session identity in-band; the session
lifecycle is signalled out of band. Turns share the IntakePool and the workers
with running batches. Like the batch routes, these are served by node 0 only.

| Term | Meaning |
|------|---------|
| `traj_id` | One agent task (item); shared by its sub-agents |
| `session_id` | One model context inside a trajectory; not reused (see below) |
| `parent_session_id` | The session that spawned this one (optional; same trajectory) |

Session states: `open` → (turn) `turn_in_flight` → (answered or failed) `idle`
→ (turn) … ; `open`/`idle` → `closed`; `open`/`idle`/`turn_in_flight` →
`aborted`. `closed` and `aborted` are terminal. A session must be opened
explicitly: a turn for an unknown session gets `404 session_not_found`, and
there is no implicit open. A session has at most one turn in flight.

A `closed` or `aborted` session is kept as a tombstone that blocks reuse of
its id until `terminal_ttl_s` (default 3600) seconds after it ended, or until
`max_terminal` (default 65536) later sessions have ended, whichever comes
first. After that the server forgets it: a turn for that id
gets `404 session_not_found` and the id can be opened again. Clients should
still use ids that are unique per run (e.g. `<traj_id>/<name>`).

### POST /v1/sessions

```json
{"traj_id": "traj-17", "session_id": "traj-17/main", "parent_session_id": null}
```

**Response:** the session (`state: "open"`). `409 session_exists` if the id is
live or still held as a tombstone; `404 parent_session_not_found` / `400 parent_session_other_trajectory`
for a bad parent.

### POST /v1/sessions/{session_id}/close

```json
{"reason": "end_of_session"}
```

The id is the rest of the path and may contain `/`
(`POST /v1/sessions/traj-17/main/close`); the same holds for `/abort`.
`reason` is `end_of_session` (this session) or `end_of_trajectory` (every live
session of the same `traj_id`). All-or-nothing: `409 turn_in_flight` if any of
them has a turn running. **Response:** `{"object": "session.close", "reason": …, "closed": [sessions]}`.

### POST /v1/sessions/{session_id}/abort

Optional body `{"reason": "tool_crash"}`. Marks the session `aborted` and
cancels its in-flight turn, whose caller gets `409 turn_aborted`. A turn still
queued is removed before it reaches a worker (`cancelled_turn: "dequeued"`); a
turn already running on a worker runs to completion there and its result is
discarded (`cancelled_turn: "detached"`) — the same limit as batch cancel.

A turn is finished once the server routes its worker completion, even before
its HTTP response is written. An abort that arrives in that window gets
`409 already_completed` and changes nothing: the turn returns its answer and
the session goes `idle`. Abort again after the turn has returned to end the
session.

### POST /v1/turns

An OpenAI chat-completions body (`model`, `messages`, `tools`, `max_tokens` or
`max_completion_tokens` (required), sampling parameters, `reasoning_effort`, …)
plus a `session` object and an optional per-turn `batchgen_debug`. The prompt is
rendered exactly as for a `/v1/batches` chat request. `n` must be 1.

```json
{
  "model": "gpt-oss-120b",
  "messages": [{"role": "user", "content": "List the files."}],
  "tools": [{"type": "function", "function": {"name": "ls", "parameters": {"type": "object"}}}],
  "max_tokens": 1024,
  "temperature": 0.6,
  "session": {"traj_id": "traj-17", "session_id": "traj-17/main"}
}
```

**Response:** a `chat.completion` object with
`usage.prompt_tokens_details.cached_tokens` and a `batchgen` extension
(values below are illustrative):

```json
{
  "id": "chatcmpl-…", "object": "chat.completion", "model": "gpt-oss-120b",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "…"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 812, "completion_tokens": 95, "total_tokens": 907,
            "prompt_tokens_details": {"cached_tokens": 0}},
  "batchgen": {
    "turn_id": "turn_3f…",
    "session": {"traj_id": "traj-17", "session_id": "traj-17/main", "parent_session_id": null},
    "cached_tokens": 0,
    "computed_tokens": 812,
    "timing": {
      "http_received_at": 1761650000.101, "intake_submitted_at": 1761650000.104,
      "admit_sent_at": 1761650000.104, "worker_admitted_at": 1761650000.121,
      "worker_completed_at": 1761650003.470, "completion_routed_at": 1761650003.471,
      "http_responded_at": 1761650003.472,
      "receipt_to_submit_ms": 3.0, "submit_to_admit_sent_ms": 0.2,
      "admit_sent_to_worker_admit_ms": 17.0, "submit_to_worker_admit_ms": 17.2,
      "worker_admit_to_complete_ms": 3349.0, "worker_complete_to_routed_ms": 1.0,
      "routed_to_response_ms": 1.0, "worker_complete_to_response_ms": 2.0,
      "overhead_ms": 22.0
    }
  }
}
```

`cached_tokens` / `computed_tokens` are the prompt tokens the sequence's last
prefill took from the prefix cache / computed. Timestamps are wall-clock seconds
(server and worker rank 0 share node 0's clock); `overhead_ms` is everything but
the worker's admit-to-complete time. Each turn also logs one `[TURN] … done`
line on the server and `[SESSION] admit` / `[SESSION] complete` lines on the
worker rank that owns the sequence.

**Limits** (checked before the session is touched; `detail` is
`{"code", "message"}`):

| Limit | Value | Error |
|-------|-------|-------|
| Request body | 32 MiB (declared `Content-Length` and bytes received) | `413 request_body_too_large` |
| `max_tokens`, `max_completion_tokens` | the model context length (`max_context_length` of `GET /v1/models`) | `400 max_tokens_exceeds_context` |
| `messages` | 8192 entries | `400 too_many_messages` |
| `tools` | 1024 entries | `400 too_many_tools` |
| `batchgen_debug` | 64 KiB as JSON | `413 batchgen_debug_too_large` |

The batch path has no request-size limit of its own; prompt plus output
beyond the context is still rejected per request by the worker
(`400 context_length_exceeded`).

**Errors:** `503` when the worker pool is not serving and `429` under intake
pressure (same gates and `Retry-After` as `POST /v1/batches`); `404` / `409` /
`400` from the session checks; `422` for a malformed body; `400`/`413` for the
limits above; `400` with the worker's code for a request the worker rejects
(`context_length_exceeded`, `host_kv_capacity_exceeded`); `409 turn_aborted`;
`503 worker_fatal` when the worker pool fails while the turn is pending, and
for every new turn after that until the pool has restarted.

---

## Health Check

### GET /health

Returns server health status.

**Response (healthy):** `200 OK`

```json
{
  "status": "healthy"
}
```

**Response (unhealthy):** `503 Service Unavailable`

---

## Removed: /v1/inference

`POST /v1/inference` is **disabled**. It always returns `410 Gone` with
`detail.code = "legacy_inference_deprecated"` and `detail.use_instead = "/v1/batches"`;
the route is kept only so callers get that explanation instead of a 404. The worker-side
path behind it has been deleted. Use the batch API (`/v1/files` + `/v1/batches`) for every
workload.

---

## Python Client

The `BatchGenHttpClient` class wraps these REST endpoints. See [Client API Reference](client-api.md) for full documentation.

| Client Method | REST Endpoint |
|---------------|---------------|
| `upload_file()` | `POST /v1/files` |
| `get_file()` | `GET /v1/files/{file_id}` |
| `download_file_content()` | `GET /v1/files/{file_id}/content` |
| `create_batch()` | `POST /v1/batches` |
| `get_batch()` | `GET /v1/batches/{batch_id}` |
| `wait_for_batch()` | Polls `GET /v1/batches/{batch_id}` |
| `submit_batch()` | Upload → Create → Wait → Download (convenience) |
| `health_check()` | `GET /health` |

Quick example:

```python
from batchgen.batchgen_client import BatchGenHttpClient

client = BatchGenHttpClient(base_url="http://localhost:10900")

# One-liner: upload, run, wait, download
batch = client.submit_batch(
    input_file_path="requests.jsonl",
    output_file_path="results.jsonl",
    max_decoding_length=1024,
)
print(f"Done: {batch['status']}, output: {batch['output_file_id']}")
client.close()
```

---

## See Also

- [Client API Reference](client-api.md) — Python client methods and parameters
- [Input Format](input-format.md) — JSONL input file structure and sampling parameters
- [Output Format](output-format.md) — Result JSONL structure
- [Server Flags](server-flags.md) — Server configuration options
