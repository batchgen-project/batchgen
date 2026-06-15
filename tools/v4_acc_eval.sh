#!/usr/bin/env bash
set -uo pipefail

REPO=/data3/leyangxue/batchgen
VENV=/root/moegen/.venv/bin/python
CKPT=/data2/tairan/models/deepseek-ai/DeepSeek-V4-Flash/v4flash_mp4_fp8/converted_ckpt/converted_ckpt
ART=/data3/leyangxue/v4-e2e-artifacts
PORT="${PORT:-10920}"
DIST_PORT="${DIST_PORT:-12420}"
MAX_DEC="${MAX_DEC:-1024}"
MAX_PROMPTS="${MAX_PROMPTS:-40}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.65}"
SERVER_LOG="$ART/acc_server.log"
E2E_LOG="$ART/acc_e2e.log"
RESULT_JSON="$ART/acc_result.json"
DONE="$ART/acc.DONE"

mkdir -p "$ART"
rm -f "$DONE" "$RESULT_JSON"
pkill -9 -f '[l]aunch_http_server' 2>/dev/null || true
pkill -9 -f '[s]erver_worker' 2>/dev/null || true
pkill -9 -f '[v]4flash_mmlu_pro_batch_test.py' 2>/dev/null || true
sleep 3
find /root/.cache/torch_extensions -name '*lock*' -delete 2>/dev/null || true
rm -f /dev/shm/batchgen_host_kv_cache /dev/shm/shm_* 2>/dev/null || true
rm -f "$REPO"/batchgen/storage/files/* "$REPO"/batchgen/storage/files_meta/*.json "$REPO"/batchgen/storage/batches/*.json 2>/dev/null || true

cd "$REPO" || { echo "no repo" > "$DONE"; exit 2; }

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 PYTHONPATH="$REPO:$REPO/tools" V4_RESULT_DEBUG=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VENV" -m batchgen.launch_http_server \
  --model deepseek-ai/DeepSeek-V4-Flash --converted-ckpt-dir "$CKPT" --cache-dir "$CKPT" \
  --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch hopper --gpu-memory-frac "$GPU_MEM_FRAC" \
  --dist-init-addr "localhost:$DIST_PORT" --world-size 4 --listen-port "$PORT" \
  --watchdog-timeout 3600 > "$SERVER_LOG" 2>&1 &
SRV=$!

READY=0
for i in $(seq 1 150); do
  if grep -q 'Uvicorn running' "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then echo "SERVER_DIED" >> "$E2E_LOG"; echo "dead" > "$DONE"; exit 1; fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then echo "READY_TIMEOUT" >> "$E2E_LOG"; echo "timeout" > "$DONE"; exit 124; fi

timeout 240m env PYTHONPATH="$REPO:$REPO/tools" "$VENV" \
  tests/e2e/v4flash_mmlu_pro_test/v4flash_mmlu_pro_batch_test.py \
  --hugging_face_checkpoint deepseek-ai/DeepSeek-V4-Flash \
  --max_decoding_length "$MAX_DEC" --base_url "http://127.0.0.1:$PORT" \
  --max_prompts "$MAX_PROMPTS" --poll_interval 10 --timeout 14400 \
  --output "$RESULT_JSON" > "$E2E_LOG" 2>&1
echo "e2e_rc=$?" >> "$E2E_LOG"

pkill -9 -f '[l]aunch_http_server' 2>/dev/null || true
pkill -9 -f '[s]erver_worker' 2>/dev/null || true
echo "done" > "$DONE"
