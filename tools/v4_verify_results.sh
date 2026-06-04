#!/usr/bin/env bash
set -uo pipefail
# Self-contained V4 decode result-gather verification.
# Runs INSIDE the container. Persists log to /data3 so it survives container death.
REPO=/data3/leyangxue/batchgen-dpfix
VENV=/root/moegen/.venv/bin/python
CKPT=/data2/models/deepseek-ai/DeepSeek-V4-Flash/v4flash_mp4_fp8/converted_ckpt/converted_ckpt
LOG=/data3/leyangxue/v4-repro-artifacts/verify_results.log
PORT=${PORT:-10917}

rm -f /dev/shm/batchgen_host_kv_cache /dev/shm/shm_* 2>/dev/null
mkdir -p /data3/leyangxue/v4-repro-artifacts
cd "$REPO"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 V4_RESULT_DEBUG=1 \
  PYTHONPATH="$REPO:$REPO/tools" \
  "$VENV" -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --converted-ckpt-dir "$CKPT" --cache-dir "$CKPT" \
    --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch hopper \
    --dist-init-addr localhost:12439 --world-size 4 --listen-port "$PORT" --watchdog-timeout 1200 \
  > "$LOG" 2>&1 &
SRV=$!
echo "server pid=$SRV log=$LOG"

for i in $(seq 1 90); do
  grep -q "Uvicorn running" "$LOG" 2>/dev/null && { echo "READY ~$((i*5))s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED during boot"; tail -5 "$LOG"; exit 1; }
  sleep 5
done

curl -s -m 1700 -X POST "http://127.0.0.1:$PORT/v1/inference" \
  -H "Content-Type: application/json" \
  -d '{"prompts":["The capital of France is","Two plus two equals"],"max_output_len":8,"temperature":0}' \
  > /data3/leyangxue/v4-repro-artifacts/verify_curl.txt 2>&1
echo "curl rc=$?"
echo "=== RESULT ==="
cat /data3/leyangxue/v4-repro-artifacts/verify_curl.txt
echo
echo "=== gather log ==="
grep -iE "V4_RESULT_DEBUG|Detokenization complete|Results are unexpect|no decoded tokens" "$LOG" | tail -8
