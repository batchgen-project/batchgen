#!/usr/bin/env python3
"""Submit one decode workload to a running inference server.

usage:
  python3 scripts/decode_batch_driver.py --name RUN --input WORKLOAD.jsonl \
      --workload-manifest WORKLOAD.manifest.json --out-dir DIR \
      [--base-url http://127.0.0.1:11710] [--debug '{"k3_decode_profile_steps": 5}'] \
      [--mode batch-api|concurrent-chat] [--prefill-peak]

Modes:
  batch-api (default)  upload the file to /v1/files and run it through
                       /v1/batches (BatchGen's Batch API).
  concurrent-chat      post every row's body to /v1/chat/completions, all rows
                       in flight from t0 (one thread and connection per row),
                       for engines without the Batch API (SGLang).  Requests
                       are written to the server strictly in arrival order --
                       file order, or a seeded permutation with
                       --arrival-order-seed -- each one sent as soon as the
                       previous one is on the wire, so an order-dependent
                       dispatcher (SGLang round_robin) sees a reproducible
                       order.  The output JSONL has the Batch API output
                       shape (custom_id, response {status_code, body}), in
                       file order, so the recorder reads both alike.

Row bodies are sent as written in the request file, for every engine.
--prefill-peak is the single exception, applied identically in both modes and
for every engine: each row's completion cap (max_completion_tokens, and
max_tokens when present) is set to 1 for the pure-prefill side channel.

The request file must hash to the manifest's request_jsonl_sha256; a mismatch
is refused before anything is sent.  The client log line
"Batch completed in <s>s, got <n> results" is the wall-time evidence that
scripts/decode_run.py records.  Standard library plus curl only, so it runs on
GPU nodes as-is.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import resource
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

try:
    from scripts import workload_policy
except ImportError:  # executed as scripts/decode_batch_driver.py
    import workload_policy

TERMINAL = ("completed", "failed", "expired", "cancelled")
MODES = ("batch-api", "concurrent-chat")
CHAT_URL = "/v1/chat/completions"
COMPLETIONS_URL = "/v1/completions"
ARRIVAL_ORDER_LINE = "Arrival order: seed={seed} sha256={sha256} rows={rows}"


def curl(args: list[str]) -> str:
    result = subprocess.run(["curl", "-s", "-m", "120"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({result.returncode}): {result.stderr[:200]}")
    return result.stdout


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("body"), dict) or not row.get("custom_id"):
                raise ValueError(f"{path}:{line_no}: row needs custom_id and an object body")
            rows.append(row)
    return rows


def prefill_peak_body(body: dict[str, Any]) -> dict[str, Any]:
    """The row body with its completion cap set to one token."""
    if "max_completion_tokens" not in body and "max_tokens" not in body:
        raise ValueError("--prefill-peak needs rows that state max_completion_tokens or max_tokens")
    capped = dict(body)
    for key in ("max_completion_tokens", "max_tokens"):
        if key in capped:
            capped[key] = 1
    return capped


def arrival_order(rows: list[dict[str, Any]], seed: int | None) -> list[int]:
    """Row indices in submission order: file order, or a permutation drawn
    with random.Random(seed).shuffle."""
    order = list(range(len(rows)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    return order


def arrival_order_sha256(custom_ids: list[str]) -> str:
    """SHA-256 of the custom_id order as compact JSON."""
    return hashlib.sha256(json.dumps(custom_ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_concurrent_chat(
    rows: list[dict[str, Any]], base_url: str, timeout: float, order: list[int] | None = None,
    endpoint: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Post every row, in ``order``, each on its own connection and thread;
    row k+1 is sent once row k's request is written, and no row waits for a
    response before the next is sent.  Returns (output rows in file order,
    wall seconds from the first send to the last response)."""
    url = urllib.parse.urlsplit(base_url.rstrip("/"))
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError(f"--base-url must be http(s)://host[:port], got {base_url!r}")
    connection_class = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    path = url.path + (endpoint or CHAT_URL)
    order = list(range(len(rows))) if order is None else order
    results: list[dict[str, Any] | None] = [None] * len(rows)
    go = [threading.Event() for _ in rows]
    sent = [threading.Event() for _ in rows]

    def worker(index: int, row: dict[str, Any]) -> None:
        go[index].wait()
        status, body, error = None, None, None
        connection = connection_class(url.hostname, url.port, timeout=timeout)
        try:
            data = json.dumps(row["body"], ensure_ascii=False).encode("utf-8")
            try:
                connection.request("POST", path, body=data, headers={"Content-Type": "application/json"})
            finally:
                sent[index].set()
            response = connection.getresponse()
            status, payload = response.status, response.read()
            try:
                body = json.loads(payload.decode("utf-8")) if payload else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = {"raw": payload.decode("utf-8", errors="replace")[:2000]}
        except Exception as exc:  # noqa: BLE001 -- every failure becomes an error row
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
        finally:
            connection.close()
        results[index] = {
            "id": f"chat-{index:06d}",
            "custom_id": row["custom_id"],
            "response": {"status_code": status, "request_id": None, "body": body},
            "error": error,
        }

    threads = [threading.Thread(target=worker, args=(i, row), daemon=True) for i, row in enumerate(rows)]
    for thread in threads:
        thread.start()
    start = time.perf_counter()
    for index in order:
        go[index].set()
        sent[index].wait()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - start
    return [row for row in results if row is not None], wall


def raise_open_file_limit(rows: int) -> tuple[int, int, int]:
    """Raise the RLIMIT_NOFILE soft limit to min(hard, max(4096, 4 * rows))
    for one connection per row.  Returns (previous soft, new soft, hard)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = max(4096, 4 * rows)
    target = want if hard == resource.RLIM_INFINITY else min(hard, want)
    if soft != resource.RLIM_INFINITY and soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            pass  # the check below reports the limit actually in force
    new_soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft, new_soft, hard


def _row_ok(row: dict[str, Any]) -> bool:
    status = row["response"]["status_code"]
    return row["error"] is None and status is not None and 200 <= status < 300


def main_concurrent_chat(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    rows = read_rows(args.input)
    # Rows name their own OpenAI-compatible endpoint: /v1/chat/completions
    # (chat-format workloads) or /v1/completions (pre-rendered prompt text).
    allowed = {CHAT_URL, COMPLETIONS_URL}
    endpoints = {row.get("url", CHAT_URL) for row in rows}
    bad = [row["custom_id"] for row in rows
           if row.get("url", CHAT_URL) not in allowed or row.get("method", "POST") != "POST"]
    if bad or len(endpoints) != 1:
        log(f"REFUSED {args.name}: rows must all POST to one of {sorted(allowed)}; "
            f"endpoints={sorted(endpoints)} bad rows {bad[:5]}")
        return 2
    previous, soft, hard = raise_open_file_limit(len(rows))
    log(f"{args.name} RLIMIT_NOFILE soft={soft} (was {previous}) hard={hard}")
    if soft != resource.RLIM_INFINITY and soft < 2 * len(rows):
        log(f"REFUSED {args.name}: open-file soft limit {soft} (hard {hard}) is below 2 x {len(rows)} rows; "
            "one connection per row needs it (raise the hard limit, e.g. ulimit -Hn)")
        return 2
    if args.prefill_peak:
        rows = [dict(row, body=prefill_peak_body(row["body"])) for row in rows]
    order = arrival_order(rows, args.arrival_order_seed)
    custom_ids = [rows[index]["custom_id"] for index in order]
    digest = arrival_order_sha256(custom_ids)
    (args.out_dir / f"{args.name}-arrival-order.json").write_text(json.dumps(
        {"seed": args.arrival_order_seed, "sha256": digest, "custom_ids": custom_ids}, indent=1
    ) + "\n", encoding="utf-8")
    log(f"{args.name} mode=concurrent-chat profile={manifest.get('profile')} rows={len(rows)} "
        f"in_flight={len(rows)} prefill_peak={args.prefill_peak}")
    print(ARRIVAL_ORDER_LINE.format(
        seed="file" if args.arrival_order_seed is None else args.arrival_order_seed, sha256=digest, rows=len(rows)
    ), flush=True)
    output, wall = run_concurrent_chat(
        rows, args.base_url, args.timeout_seconds, order, endpoint=next(iter(endpoints))
    )
    path = args.out_dir / f"{args.name}-output.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    failed = [row["custom_id"] for row in output if not _row_ok(row)]
    if failed:
        log(f"FAILED {args.name}: {len(failed)} of {len(output)} rows did not return 2xx after {wall:.2f}s "
            f"(first: {failed[:5]})")
        return 1
    print(f"Batch completed in {wall:.2f}s, got {len(output)} results", flush=True)
    log(f"{args.name} DONE output_lines={len(output)}")
    return 0


def main_batch_api(args: argparse.Namespace, manifest: dict[str, Any], debug: dict[str, Any] | None) -> int:
    upload = args.input
    temp_path = None
    if args.prefill_peak:
        rows = [dict(row, body=prefill_peak_body(row["body"])) for row in read_rows(args.input)]
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".jsonl", prefix=f"{args.name}-prefill-peak-", dir=args.out_dir, delete=False
        )
        with handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        upload = temp_path = Path(handle.name)
    try:
        uploaded = json.loads(curl([
            "-F", f"file=@{upload};type=application/jsonl",
            "-F", "purpose=batch",
            args.base_url + "/v1/files",
        ]))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    body = {"input_file_id": uploaded["id"], "endpoint": CHAT_URL, "completion_window": "24h"}
    if debug:
        body["batchgen_debug"] = debug
    start = time.perf_counter()
    created = json.loads(curl([
        "-X", "POST", args.base_url + "/v1/batches",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(body),
    ]))
    batch_id = created.get("id")
    if not batch_id:
        log(f"FAILED {args.name}: batch create returned {json.dumps(created)[:400]}")
        return 1
    log(f"{args.name} batch={batch_id} profile={manifest.get('profile')} rows={manifest.get('request_count')} "
        f"debug={json.dumps(debug or {})} prefill_peak={args.prefill_peak}")
    while True:
        time.sleep(args.poll_seconds)
        state = json.loads(curl([args.base_url + "/v1/batches/" + batch_id]))
        status = state.get("status")
        log(f"{args.name} status={status}")
        if status in TERMINAL:
            break
        if time.perf_counter() - start > args.timeout_seconds:
            log(f"FAILED {args.name}: no terminal status after {args.timeout_seconds}s")
            return 1
    wall = time.perf_counter() - start
    (args.out_dir / f"{args.name}-batch.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    lines = 0
    if state.get("output_file_id"):
        output = curl([args.base_url + "/v1/files/" + state["output_file_id"] + "/content"])
        (args.out_dir / f"{args.name}-output.jsonl").write_text(output, encoding="utf-8")
        lines = len(output.splitlines())
    if status != "completed":
        log(f"FAILED {args.name}: batch ended {status} after {wall:.2f}s")
        return 1
    print(f"Batch completed in {wall:.2f}s, got {lines} results", flush=True)
    log(f"{args.name} DONE output_lines={lines}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11710")
    parser.add_argument("--mode", choices=MODES, default="batch-api")
    parser.add_argument("--debug", default="", help="batch-level batchgen_debug JSON object (batch-api only)")
    parser.add_argument(
        "--prefill-peak", action="store_true",
        help="Cap every row at 1 completion token (pure-prefill side channel), identically for every engine.",
    )
    # Wall time is measured at poll resolution; keep it small against minute-scale runs.
    parser.add_argument(
        "--arrival-order-seed", type=int,
        help="concurrent-chat: submit rows in random.Random(seed).shuffle order (bodies unchanged); "
             "default is file order.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    args = parser.parse_args(argv)

    manifest = json.loads(args.workload_manifest.read_text(encoding="utf-8"))
    actual = workload_policy.sha256_file(args.input)
    if manifest.get("request_jsonl_sha256") != actual:
        log(f"REFUSED {args.name}: {args.input} sha256 {actual} != manifest {manifest.get('request_jsonl_sha256')}")
        return 2
    debug = json.loads(args.debug) if args.debug else None
    if args.mode == "batch-api" and args.arrival_order_seed is not None:
        log(f"REFUSED {args.name}: --arrival-order-seed applies to concurrent-chat; the Batch API takes the file as is")
        return 2
    if args.mode == "concurrent-chat" and debug:
        log(f"REFUSED {args.name}: --debug is a batch-level Batch API flag; concurrent-chat has no batch")
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "concurrent-chat":
        return main_concurrent_chat(args, manifest)
    return main_batch_api(args, manifest, debug)


if __name__ == "__main__":
    sys.exit(main())
