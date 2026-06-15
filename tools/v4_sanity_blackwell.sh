#!/usr/bin/env bash
# Engine sanity check: simple factual prompts, greedy decode, inspect coherence.
# Isolates "engine correct" from "MMLU output-quality" issues.
set -uo pipefail

REPO="${REPO:-/work}"
VENV="${VENV:-python}"
CKPT="${CKPT:-/mnt/raid0nvme0/leyang/v4flash_converted}"
SNAP="${SNAP:-/mnt/raid0nvme0/public/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}"
ART="${ART:-/work/.sisyphus/blackwell/sanity}"
PORT="${PORT:-10940}"
DIST_PORT="${DIST_PORT:-12465}"
MAXOUT="${MAXOUT:-64}"

SERVER_LOG="$ART/sanity_server.log"
OUT="$ART/sanity_out.json"
DONE="$ART/sanity.DONE"

mkdir -p "$ART"
rm -f "$DONE" "$OUT"
pkill -9 -f '[l]aunch_http_server' 2>/dev/null || true
pkill -9 -f '[s]erver_worker' 2>/dev/null || true
sleep 3
rm -f /dev/shm/batchgen_host_kv_cache /dev/shm/shm_* 2>/dev/null || true

cd "$REPO" || { echo norepo >"$DONE"; exit 2; }

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 PYTHONPATH="$REPO:$REPO/tools" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  BATCHGEN_NCCL_TIMEOUT_SEC=86400 \
  "$VENV" -m batchgen.launch_http_server \
  --model deepseek-ai/DeepSeek-V4-Flash --converted-ckpt-dir "$CKPT" --cache-dir "$SNAP" \
  --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch blackwell --gpu-memory-frac 0.65 \
  --dist-init-addr "localhost:$DIST_PORT" --world-size 4 --listen-port "$PORT" \
  --watchdog-timeout 86400 > "$SERVER_LOG" 2>&1 &
SRV=$!

READY=0
for i in $(seq 1 150); do
  grep -q 'Uvicorn running' "$SERVER_LOG" 2>/dev/null && { READY=1; break; }
  kill -0 "$SRV" 2>/dev/null || { echo SERVER_DIED >"$DONE"; exit 1; }
  sleep 5
done
[ "$READY" -ne 1 ] && { echo READY_TIMEOUT >"$DONE"; exit 124; }

curl -s -m 1700 -X POST "http://127.0.0.1:$PORT/v1/inference" \
  -H 'Content-Type: application/json' \
  -d "{\"prompts\":[\"The capital of France is\",\"The opposite of hot is\",\"2 + 2 =\",\"The first president of the United States was\"],\"max_output_len\":$MAXOUT,\"temperature\":0}" \
  > "$OUT" 2>&1
echo "curl_rc=$?" >> "$OUT"

sleep 3
pkill -9 -f '[l]aunch_http_server' 2>/dev/null || true
pkill -9 -f '[s]erver_worker' 2>/dev/null || true
echo done >"$DONE"
