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

Cancel a running batch. The batch transitions to `cancelling`, then `cancelled` once the worker stops processing.

**Response:** `BatchObject` with `status: "cancelling"`

---

## Batch Status Flow

```
validating → in_progress → completed
                         → failed
                         → cancelling → cancelled
```

| Status | Description |
|--------|-------------|
| `validating` | Input file is being parsed and validated |
| `in_progress` | Sequences are being processed |
| `completed` | All sequences finished. `output_file_id` is set. |
| `failed` | Processing failed. `error` field has details. |
| `cancelling` | Cancel requested, waiting for worker to stop |
| `cancelled` | Batch was cancelled. `cancelled_at` is set. |

**Timestamps:** `started_at`, `completed_at`, `cancelled_at`, `cancelling_at` are set as the batch transitions through states.

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

## Deprecated: /v1/inference

`POST /v1/inference` exists for legacy direct inference but is **no longer maintained**. Use the batch API (`/v1/files` + `/v1/batches`) for all production workloads.

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
