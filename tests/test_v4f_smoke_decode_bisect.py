"""Submit a 3-prompt smoke batch to BatchGen V4-Flash:
  - prompt A: very short ("What is 2+2?")     — minimal prefill, should hit non-compressed path
  - prompt B: medium ("Solve 17*23 step by step") — still short
  - prompt C: chat mode (no thinking)         — different suffix, simpler decode

If A+B+C all produce coherent output, prefill works for short seqs and the L1
garbage is in the compressed-prefill path. If they're all garbage too, the
bug is fundamental forward.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.parse

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ["NO_PROXY"] = "*"

BASE = "http://localhost:10900"


def _post(path, body, files=None, timeout=120):
    if files:
        # multipart upload
        boundary = "----v4fsmoke" + str(int(time.time()))
        parts = []
        for fname, fcontent, ftype in files:
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(
                f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'.encode()
            )
            parts.append(f"Content-Type: {ftype}\r\n\r\n".encode())
            parts.append(fcontent if isinstance(fcontent, bytes) else fcontent.encode())
            parts.append(b"\r\n")
        for k, v in (body or {}).items():
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
            parts.append(str(v).encode())
            parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(parts)
        req = urllib.request.Request(
            BASE + path, data=data, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
    else:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            BASE + path, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path, timeout=60):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


PROMPTS = [
    {
        "custom_id": "smoke-A-short-thinking",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50,
            "temperature": 0.0,
            "enable_thinking": True,
        },
    },
    {
        "custom_id": "smoke-B-medium-thinking",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [
                {"role": "user", "content": "Solve 17 times 23 step by step. Show your work."}
            ],
            "max_tokens": 80,
            "temperature": 0.0,
            "enable_thinking": True,
        },
    },
    {
        "custom_id": "smoke-C-short-chat-mode",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "Capital of France?"}],
            "max_tokens": 30,
            "temperature": 0.0,
            "enable_thinking": False,
        },
    },
]

# Build the input JSONL
jsonl_path = "/tmp/v4f/smoke.jsonl"
with open(jsonl_path, "w") as f:
    for p in PROMPTS:
        f.write(json.dumps(p) + "\n")

print(f"=== uploading {jsonl_path} ({sum(len(p['body']['messages'][0]['content']) for p in PROMPTS)} chars total) ===")

# Upload as a file
upload = _post("/v1/files", {"purpose": "batch"},
               files=[("smoke.jsonl", open(jsonl_path, "rb").read(), "application/jsonl")])
print("upload response:", json.dumps(upload, indent=2))
file_id = upload["id"]

# Create the batch
batch = _post("/v1/batches", {
    "input_file_id": file_id,
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h",
})
print("batch response:", json.dumps(batch, indent=2))
batch_id = batch["id"]

# Poll for completion
print(f"polling batch {batch_id}...")
deadline = time.time() + 900  # 15 min
last = None
while time.time() < deadline:
    cur = json.loads(_get(f"/v1/batches/{batch_id}"))
    s = cur.get("status")
    if s != last:
        print(f"  status={s} req_counts={cur.get('request_counts')}")
        last = s
    if s in ("completed", "failed", "cancelled", "expired"):
        break
    time.sleep(5)
else:
    print("TIMEOUT")
    sys.exit(1)

if s != "completed":
    print(f"NON-OK terminal status: {s}")
    print(json.dumps(cur, indent=2))
    sys.exit(2)

out_id = cur["output_file_id"]
print(f"\n=== fetching outputs (file_id={out_id}) ===\n")
out_bytes = _get(f"/v1/files/{out_id}/content")
for line in out_bytes.decode().splitlines():
    rec = json.loads(line)
    cid = rec.get("custom_id")
    body = rec.get("response", {})
    if isinstance(body, dict) and "body" in body:
        body = body["body"]
    content = (body.get("choices") or [{}])[0].get("message", {}).get("content", "") if isinstance(body, dict) else ""
    print(f"--- {cid} ---")
    print(repr(content)[:600])
    print()
