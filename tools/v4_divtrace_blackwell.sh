#!/usr/bin/env bash
# Paired A/B divergence trace for DeepSeek-V4-Flash decode on Blackwell (sm120).
# Localizes WHERE two prompts (len 6 vs 16) collapse to identical hidden states.
# Run INSIDE batchgen:v4-kernels with --ipc=host. See .sisyphus/HANDOFF.md decision tree.
set -uo pipefail

REPO="${REPO:-/work}"
VENV="${VENV:-python}"
CKPT="${CKPT:-/mnt/raid0nvme0/leyang/v4flash_converted}"
SNAP="${SNAP:-/mnt/raid0nvme0/public/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136}"
ART="${ART:-/work/.sisyphus/blackwell/divtrace}"
PORT="${PORT:-10933}"
DIST_PORT="${DIST_PORT:-12458}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.65}"

SERVER_LOG="$ART/divtrace_server.log"
CURL_OUT="$ART/divtrace_curl.txt"
DONE="$ART/divtrace.DONE"

mkdir -p "$ART"
rm -f "$DONE" "$ART"/divtrace_rank*.pt
pkill -9 -f '[l]aunch_http_server' 2>/dev/null || true
pkill -9 -f '[s]erver_worker' 2>/dev/null || true
sleep 3
rm -f /dev/shm/batchgen_host_kv_cache /dev/shm/shm_* 2>/dev/null || true
rm -f "$REPO"/batchgen/storage/files/* "$REPO"/batchgen/storage/files_meta/*.json "$REPO"/batchgen/storage/batches/*.json 2>/dev/null || true

cd "$REPO" || { echo norepo >"$DONE"; exit 2; }

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 PYTHONPATH="$REPO:$REPO/tools" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  BATCHGEN_V4_DIVTRACE=1 BATCHGEN_V4_DIVTRACE_DUMP_PATH="$ART" \
  "$VENV" -m batchgen.launch_http_server \
  --model deepseek-ai/DeepSeek-V4-Flash --converted-ckpt-dir "$CKPT" --cache-dir "$SNAP" \
  --kv-dtype fp8 --host-kv-cache-size 100 --gpu-arch blackwell --gpu-memory-frac "$GPU_MEM_FRAC" \
  --dist-init-addr "localhost:$DIST_PORT" --world-size 4 --listen-port "$PORT" \
  --watchdog-timeout 3600 > "$SERVER_LOG" 2>&1 &
SRV=$!

READY=0
for i in $(seq 1 150); do
  grep -q 'Uvicorn running' "$SERVER_LOG" 2>/dev/null && { READY=1; break; }
  kill -0 "$SRV" 2>/dev/null || { echo SERVER_DIED >"$DONE"; exit 1; }
  sleep 5
done
[ "$READY" -ne 1 ] && { echo READY_TIMEOUT >"$DONE"; exit 124; }

# Prompt A = 6 tokens, Prompt B = 16 tokens (matches analyze_divtrace.py PROMPT_A_SEQLEN=6, B=16).
# Need >=world_size(4) prompts so no rank gets 0 sequences (empty torch.cat crash in prefill_prepacked).
curl -s -m 1700 -X POST "http://127.0.0.1:$PORT/v1/inference" \
  -H 'Content-Type: application/json' \
  -d '{"prompts":["The capital of France is","A B C D E F G H I J K L M N O","Once upon a time there","Hello world this is a test of"],"max_output_len":2,"temperature":0}' \
  > "$CURL_OUT" 2>&1
echo "curl_rc=$?" >> "$CURL_OUT"

sleep 5
pkill -9 -f '[l]aunch_http_server' 2>/dev/null || true
pkill -9 -f '[s]erver_worker' 2>/dev/null || true
ls -l "$ART"/divtrace_rank*.pt >> "$CURL_OUT" 2>&1 || echo "NO_TRACE_FILES" >> "$CURL_OUT"
echo done >"$DONE"
