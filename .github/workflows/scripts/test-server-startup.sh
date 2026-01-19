#!/bin/bash
set -euo pipefail

echo "=== test-0: Docker Build + Server Startup Test ==="

# Parse arguments
CACHE_DIR="/root/.cache/huggingface/hub/models/deepseek-ai/DeepSeek-R1"
STORAGE_PATH="/data2/tairan/workspace/H20_bench"
DIST_INIT_ADDR="29.194.13.138:33001"
TIMEOUT_SECONDS=600  # 10 minutes for server to start

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        --storage-path) STORAGE_PATH="$2"; shift 2 ;;
        --dist-init-addr) DIST_INIT_ADDR="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

datetime=$(date +%Y%m%d_%H%M%S)
mkdir -p ./test-logs

echo "[1/4] Resizing /dev/shm..."
TARGET_SHM_GB=2048
CURRENT_SHM_GB=$(df -BG --output=size /dev/shm | tail -n1 | tr -dc '0-9')
if (( CURRENT_SHM_GB < TARGET_SHM_GB )); then
  mount -o remount,size=${TARGET_SHM_GB}G /dev/shm || echo "Warning: failed to remount /dev/shm"
fi
echo "   /dev/shm size: $(df -h /dev/shm | tail -1 | awk '{print $2}')"

echo "[2/4] Starting BatchGen HTTP server..."
# Start server in background (single-node, 8 GPUs with EP offloading)
python -m batchgen.launch_http_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir "$CACHE_DIR" \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --nnodes 1 \
    --node-rank 0 \
    --kv-dtype "bf16" \
    --world-size 8 \
    --host-kv-cache-size 128 \
    --enable-hugetlbfs \
    --storage-path "$STORAGE_PATH" \
    --save-result \
    --gpu-memory-frac 0.94 \
    --enable-ep-with-offloading \
    --ep-offloading-ratio 0.3 \
    > ./test-logs/server_${datetime}.log 2>&1 &
SERVER_PID=$!

# Ensure server is killed on script exit
trap "echo 'Cleaning up...'; kill $SERVER_PID 2>/dev/null || true" EXIT

echo "   Server PID: $SERVER_PID"
echo "   Log file: ./test-logs/server_${datetime}.log"

echo "[3/4] Waiting for server to become healthy..."
start_time=$(date +%s)
attempt=0
while true; do
    attempt=$((attempt + 1))
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died! Check logs:"
        tail -50 ./test-logs/server_${datetime}.log
        exit 1
    fi

    # Check health endpoint
    if curl -s "http://localhost:10900/health" > /dev/null 2>&1; then
        echo "   Server is healthy after ${elapsed}s (attempt $attempt)"
        break
    fi

    # Check timeout
    if (( elapsed >= TIMEOUT_SECONDS )); then
        echo "ERROR: Server failed to start within ${TIMEOUT_SECONDS}s"
        echo "Last 50 lines of server log:"
        tail -50 ./test-logs/server_${datetime}.log
        exit 1
    fi

    echo "   Waiting... (${elapsed}s elapsed, attempt $attempt)"
    sleep 30
done

echo "[4/4] Server startup test PASSED!"
echo ""
echo "=== test-0 COMPLETED SUCCESSFULLY ==="

# Keep server running briefly to verify stability
echo "Verifying server stability for 60 seconds..."
sleep 60

# Final health check
if curl -s "http://localhost:10900/health" > /dev/null 2>&1; then
    echo "Server still healthy after stability check."
    echo "TEST PASSED"
    exit 0
else
    echo "ERROR: Server became unhealthy during stability check"
    exit 1
fi
