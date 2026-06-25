#!/usr/bin/env bash
# ---------------------------------------------------------------------------- #
#  Rebuild the Hopper/H20 batchgen image from CURRENT source and launch a full
#  DeepSeek-V4-Flash run on 4x H20 (sm90).
#
#  WHY THIS EXISTS
#  ---------------
#  The pre-built batchgen:v4flash-tp-fix image on TencentNode0 is weeks stale and
#  cannot run current code end-to-end. Mounting current source over it surfaces
#  three image-vs-code drift blockers (all FIXED by a rebuild from current HEAD):
#    1. FlashMLA API: adapter needs the zero-arg get_mla_metadata (hopper ref
#       c741387, see docker/Dockerfile:54-66). Stale image has the old signature.
#    2. tilelang import: needs apache-tvm-ffi==0.1.5 (Dockerfile:80,107). Stale
#       image has a wrong pin -> "AttributeError: attribute '__dict__' ...".
#    3. shm sizing: 100GB host-KV + 320GB weight region needs --shm-size 512g.
#
#  The sm-aware kernel switch (MXFP4 sm120 / FP8 sm90) is already validated:
#  server loads + becomes ready on H20 with ZERO PTXASError / cvt.e2m1 errors.
#  This script finishes the loop by rebuilding so inference actually runs.
#
#  PREREQUISITES (already true on TencentNode0 as of this writing)
#  --------------------------------------------------------------
#    - Build context (current repo) synced to $REPO_ON_NODE (rsync, see below).
#    - Weights at $CKPT_DIR (pre-converted MP4 FP8) and $CACHE_DIR (HF config).
#    - 4 idle H20s. Docker root on /data1 (1.2T free).
#
#  USAGE
#  -----
#    # 1. (from laptop/dev box) sync current source to the node:
#    #    rsync -az --delete --exclude='.git/' --exclude='.venv/' \
#    #      --exclude='**/__pycache__/' --exclude='batchgen/storage/' \
#    #      ./ TencentNode0:/data3/leyangxue/batchgen/
#    # 2. (on the node, or via ssh) run this script:
#    #    bash v4_h20_rebuild_and_launch.sh build     # rebuild image (~30-60min)
#    #    bash v4_h20_rebuild_and_launch.sh launch     # start server on 4 H20s
#    #    bash v4_h20_rebuild_and_launch.sh smoke       # one inference request
#    #    bash v4_h20_rebuild_and_launch.sh logs        # tail server log
#    #    bash v4_h20_rebuild_and_launch.sh stop        # stop + free GPUs
# ---------------------------------------------------------------------------- #
set -uo pipefail

# ---- config (override via env) --------------------------------------------- #
REPO_ON_NODE="${REPO_ON_NODE:-/data3/leyangxue/batchgen}"
IMAGE="${IMAGE:-batchgen:v4flash-hopper-current}"
# CN-mirror build args (see docker/README.md); defaults are CN-fast.
TORCH_FIND_LINKS="${TORCH_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu129}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
CONTAINER="${CONTAINER:-leyang-v4-fullrun}"
DEVICES="${DEVICES:-0,1,2,3}"
WORLD_SIZE="${WORLD_SIZE:-4}"
PORT="${PORT:-10944}"
SHM_SIZE="${SHM_SIZE:-512g}"

CKPT_DIR="${CKPT_DIR:-/data2/models/deepseek-ai/DeepSeek-V4-Flash/v4flash_mp4_fp8/converted_ckpt/converted_ckpt}"
CACHE_DIR="${CACHE_DIR:-/data2/models/deepseek-ai/DeepSeek-V4-Flash}"
HOST_KV_GB="${HOST_KV_GB:-100}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.90}"
DIST_PORT="${DIST_PORT:-12464}"

# Persistent host dir for torch JIT extensions (core_engine + the runtime
# load()/load_inline kernels). Without this the cache lives at the container's
# /root/.cache/torch_extensions and is recompiled (~15-20min, 4 extensions) on
# every fresh container. Mounting a host dir makes the SECOND launch reuse the
# compiled .so (cache key = source hash), so startup->decode is immediate.
TORCH_EXT_CACHE="${TORCH_EXT_CACHE:-/data3/leyangxue/torch_ext_cache}"

LOG="/tmp/v4_h20_server.log"

# ---- sm-aware + runtime env flags ------------------------------------------ #
#  GROUPED_MOE=0 : my B1 gate also auto-disables grouped MXFP4 MoE on sm90,
#                  this is belt-and-suspenders.
#  INDEXER_QUANT=auto : my A3 dispatch -> FP8 indexer quant on sm90.
#  SPARSE_PREFILL : keep =1 once tilelang works (rebuild fixes it); =0 forces the
#                   tilelang-free dense prefill fallback if you must skip it.
RUN_ENV=(
  CUDA_VISIBLE_DEVICES="$DEVICES"
  HF_HUB_OFFLINE=1
  PYTHONPATH=/workspace/repo:/workspace/repo/tools
  BATCHGEN_V4_GROUPED_MOE=0
  BATCHGEN_V4_INDEXER_QUANT=auto
  BATCHGEN_V4_PYNCCL_COMM=1
  BATCHGEN_V4_SPARSE_PREFILL="${BATCHGEN_V4_SPARSE_PREFILL:-1}"
  TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
)
# Extra "KEY=VALUE" env entries appended to the server launch (space-separated).
# Used by v4_h20_validate_decode_fix.sh to inject BATCHGEN_DECODE_DEADLOCK_TRACE=1.
RUN_ENV_EXTRA="${RUN_ENV_EXTRA:-}"

cmd_build() {
  echo ">>> Building $IMAGE for GPU_ARCH=hopper from $REPO_ON_NODE (~30-60min)"
  cd "$REPO_ON_NODE" || { echo "repo not found at $REPO_ON_NODE"; exit 1; }
  DOCKER_BUILDKIT=1 docker build \
    --build-arg GPU_ARCH=hopper \
    --build-arg TORCH_FIND_LINKS="$TORCH_FIND_LINKS" \
    --build-arg UV_DEFAULT_INDEX="$UV_DEFAULT_INDEX" \
    -f docker/Dockerfile \
    -t "$IMAGE" . 2>&1 | tail -40
  echo ">>> Build exit: ${PIPESTATUS[0]}"
  docker images | grep -E "${IMAGE%%:*}.*${IMAGE##*:}" || true
}

cmd_launch() {
  echo ">>> Launching $CONTAINER on GPUs $DEVICES (shm=$SHM_SIZE)"
  echo ">>> JIT extension cache (persistent): $TORCH_EXT_CACHE"
  mkdir -p "$TORCH_EXT_CACHE"
  docker rm -f "$CONTAINER" 2>/dev/null || true
  docker run -d --name "$CONTAINER" \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$DEVICES" \
    --shm-size="$SHM_SIZE" \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO_ON_NODE":/workspace/repo \
    -v /data2/models:/data2/models:ro \
    -v "$TORCH_EXT_CACHE":/root/.cache/torch_extensions \
    -w /workspace/repo \
    -e PYTHONPATH=/workspace/repo:/workspace/repo/tools \
    "$IMAGE" sleep infinity
  # clear any stale shm regions from prior crashed launches (critical!)
  docker exec "$CONTAINER" bash -lc 'rm -rf /dev/shm/* 2>/dev/null; df -h /dev/shm | tail -1'
  docker exec -d "$CONTAINER" bash -lc "cd /workspace/repo && \
    $(printf '%s ' "${RUN_ENV[@]}") $RUN_ENV_EXTRA \
    python -m batchgen.launch_http_server \
      --model deepseek-ai/DeepSeek-V4-Flash \
      --converted-ckpt-dir '$CKPT_DIR' \
      --cache-dir '$CACHE_DIR' \
      --kv-dtype fp8 --host-kv-cache-size $HOST_KV_GB \
      --gpu-arch hopper --gpu-memory-frac $GPU_MEM_FRAC \
      --dist-init-addr localhost:$DIST_PORT \
      --world-size $WORLD_SIZE --listen-port $PORT \
      --watchdog-timeout 6000 > $LOG 2>&1"
  echo ">>> Launched. Server ready in ~225s. Watch: $0 logs"
}

cmd_logs() {
  docker exec "$CONTAINER" bash -lc "tail -f $LOG"
}

cmd_wait() {
  echo ">>> Waiting for server ready (timeout 360s)..."
  for i in $(seq 1 72); do
    if docker exec "$CONTAINER" bash -lc "grep -q 'Uvicorn running' $LOG 2>/dev/null"; then
      echo ">>> READY"; docker exec "$CONTAINER" bash -lc "grep -E 'server ready|Uvicorn running' $LOG | tail -2"
      return 0
    fi
    if ! docker exec "$CONTAINER" bash -lc "pgrep -f launch_http_server >/dev/null"; then
      echo ">>> SERVER DIED. Last errors:"
      docker exec "$CONTAINER" bash -lc "grep -iE 'Error|not enough|tilelang|ptxas|Traceback' $LOG | grep -ivE 'resource_tracker|throwOnCuda|cudaMemcpy' | tail -8"
      return 1
    fi
    sleep 5
  done
  echo ">>> TIMEOUT waiting for ready"; return 1
}

cmd_smoke() {
  echo ">>> Smoke inference on port $PORT"
  docker exec "$CONTAINER" bash -lc \
    "curl -s -m 120 -X POST http://127.0.0.1:$PORT/v1/inference \
      -H 'Content-Type: application/json' \
      -d '{\"prompts\":[\"The capital of France is\"],\"max_output_len\":24,\"temperature\":0.0}'"
  echo
}

cmd_mmlu() {
  echo ">>> MMLU-Pro batch test (20 prompts)"
  docker exec "$CONTAINER" bash -lc \
    "cd /workspace/repo && python tests/e2e/v4flash_mmlu_pro_test/v4flash_mmlu_pro_batch_test.py \
      --hugging_face_checkpoint '$CACHE_DIR' \
      --base_url http://127.0.0.1:$PORT \
      --max_prompts 20 --max_decoding_length 512 --timeout 6000"
}

# First request triggers the one-time torch JIT compile (4 extensions, ~15-20min)
# into the persistent cache. Run this ONCE per image build so all later launches
# (reusing the same $TORCH_EXT_CACHE volume) skip straight to fast decode.
cmd_warmup() {
  echo ">>> JIT warmup (first compile populates $TORCH_EXT_CACHE; allow ~25min)"
  docker exec "$CONTAINER" bash -lc \
    "curl -s -m 1800 -X POST http://127.0.0.1:$PORT/v1/inference \
      -H 'Content-Type: application/json' \
      -d '{\"prompts\":[\"Hello\"],\"max_output_len\":4,\"temperature\":0.0}'"
  echo
  cmd_cache_status
}

cmd_cache_status() {
  echo ">>> Persistent JIT cache contents ($TORCH_EXT_CACHE):"
  ls -1 "$TORCH_EXT_CACHE"/*/ 2>/dev/null | grep -vE '/$|^$' | sort -u || true
  find "$TORCH_EXT_CACHE" -name '*.so' 2>/dev/null | sed "s|$TORCH_EXT_CACHE/||" | head
}

cmd_stop() {
  echo ">>> Stopping $CONTAINER and freeing GPUs"
  docker exec "$CONTAINER" bash -lc 'pkill -9 -f launch_http_server 2>/dev/null; rm -rf /dev/shm/* 2>/dev/null' || true
  sleep 3
  docker rm -f "$CONTAINER" 2>/dev/null || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -8
}

case "${1:-help}" in
  build)        cmd_build ;;
  launch)       cmd_launch ;;
  wait)         cmd_wait ;;
  logs)         cmd_logs ;;
  smoke)        cmd_smoke ;;
  warmup)       cmd_warmup ;;
  cache-status) cmd_cache_status ;;
  mmlu)         cmd_mmlu ;;
  stop)         cmd_stop ;;
  full)         cmd_launch && cmd_wait && cmd_smoke ;;
  *) echo "usage: $0 {build|launch|wait|logs|smoke|warmup|cache-status|mmlu|stop|full}"; exit 1 ;;
esac
