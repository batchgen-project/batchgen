#!/usr/bin/env bash
set -uo pipefail

# DeepSeek-V4-Flash DP-collective (1EB OOM) reproduction harness.
# Run INSIDE the container (docker exec ... bash /data3/leyangxue/batchgen-dpfix/tools/v4_repro_launch.sh).
# Idempotent: cleans stale state, launches on 4 GPUs with the collective tracer,
# waits for ready, fires a few-prompt request to drive the empty/padded-rank decode path,
# then prints per-rank collective trace tails + any 1EB/crash.

REPO=/data3/leyangxue/batchgen-dpfix
VENV=/root/moegen/.venv/bin/python
CKPT=/data2/models/deepseek-ai/DeepSeek-V4-Flash/v4flash_mp4_fp8/converted_ckpt/converted_ckpt
GPUS=${GPUS:-0,1,2,3}
DIST_PORT=${DIST_PORT:-12399}
HTTP_PORT=${HTTP_PORT:-10902}
TRACE_DIR=/tmp/v4trace
LOG=/tmp/v4_launch.log

echo "=== [1/6] pre-clean stale state ==="
pkill -9 -f launch_http_server 2>/dev/null || true
sleep 3
find /root/.cache/torch_extensions -name "*lock*" -delete 2>/dev/null || true
rm -f /dev/shm/batchgen_host_kv_cache /dev/shm/shm_* 2>/dev/null || true
rm -rf "$TRACE_DIR"; mkdir -p "$TRACE_DIR"

echo "=== [2/6] launch (GPUS=$GPUS dist=$DIST_PORT http=$HTTP_PORT) ==="
cd "$REPO"
nohup env \
  CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_OFFLINE=1 \
  PYTHONPATH="$REPO:$REPO/tools" \
  V4_COLL_TRACE=1 V4_COLL_TRACE_DIR="$TRACE_DIR" \
  "$VENV" -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --converted-ckpt-dir "$CKPT" --cache-dir "$CKPT" \
    --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch hopper \
    --dist-init-addr "localhost:$DIST_PORT" \
    --world-size 4 --listen-port "$HTTP_PORT" --watchdog-timeout 300 \
  > "$LOG" 2>&1 &
echo "LAUNCH_PID=$!"

echo "=== [3/6] wait for ready (max 360s) ==="
for i in $(seq 1 72); do
  if grep -q "Uvicorn running" "$LOG" 2>/dev/null; then echo "READY after ~$((i*5))s"; break; fi
  if grep -qiE "worker process exit|Application startup failed|EADDRINUSE" "$LOG" 2>/dev/null; then
    echo "LAUNCH FAILED:"; grep -iE "error|EADDRINUSE|worker process exit" "$LOG" | tail -10; exit 1
  fi
  if ! pgrep -f launch_http_server >/dev/null; then echo "PROCESS DIED:"; tail -15 "$LOG"; exit 1; fi
  sleep 5
done

echo "=== [4/6] fire few-prompt request (2 prompts, world_size=4 => empty/padded ranks) ==="
curl -s -m 180 -X POST "http://127.0.0.1:$HTTP_PORT/v1/inference" \
  -H "Content-Type: application/json" \
  -d '{"prompts":["The capital of France is","Two plus two equals"],"max_output_len":32,"temperature":0}' \
  2>&1 | head -c 1500
echo; echo "CURL_RC=$?"

echo "=== [5/6] crash / 1EB scan ==="
grep -iE "1EB|Tried to allocate|out of memory|all_gather_object|RuntimeError|Detected worker process exit" "$LOG" | tail -15 || echo "(no crash markers)"

echo "=== [6/6] per-rank collective trace tails ==="
for f in "$TRACE_DIR"/v4_coll_trace_rank*.log; do
  echo "--- $f (lines: $(wc -l < "$f")) ---"
  tail -8 "$f"
done
echo "DONE. Full log: $LOG ; traces: $TRACE_DIR"
