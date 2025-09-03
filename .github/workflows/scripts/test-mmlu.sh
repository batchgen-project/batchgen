#!/bin/bash
set -euo pipefail

# parse arguments
NODE_RANK=""
DIST_INIT_ADDR=""
CACHE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --dist-init-addr)
            DIST_INIT_ADDR="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# check arguments
if [[ -z "$NODE_RANK" || -z "$DIST_INIT_ADDR" || -z "$CACHE_DIR" ]]; then
    echo "Error: --node-rank, --dist-init-addr, and --cache-dir are required."
    exit 1
fi

TARGET_SHM_GB=2048
CURRENT_SHM_GB=$(df -BG --output=size /dev/shm | tail -n1 | tr -dc '0-9')

if (( CURRENT_SHM_GB < TARGET_SHM_GB )); then
  echo "Resizing /dev/shm from ${CURRENT_SHM_GB}G to ${TARGET_SHM_GB}G..."
  mount -o remount,size=${TARGET_SHM_GB}G /dev/shm || \
    echo "Warning: failed to remount /dev/shm"
else
  echo "/dev/shm already ${CURRENT_SHM_GB}G, no remount needed."
fi

# start parameter server
HF_ENDPOINT=https://hf-mirror.com \
python -m batchgen.parameter_server \
        --model deepseek-ai/DeepSeek-R1 \
        --cache-dir "$CACHE_DIR" &
PARAM_SERVER_PID=$!

# ensure the parameter server is killed on script exit
trap "kill $PARAM_SERVER_PID 2>/dev/null" EXIT

# wait for the parameter server to be ready
function wait_for_port() {
    local host="localhost"
    local port=10900
    while ! (echo > "/dev/tcp/$host/$port") >/dev/null 2>&1; do
        echo "Waiting for port $port to become available..."
        sleep 10
    done
    echo "Port $port is available."
}

wait_for_port

# run test
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python test/r1_mmlu_pro_test/r1_mmlu_pro_test.py \
        --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
        --host_kv_cache_size 256 \
        --max_input_length 8192 \
        --max_decoding_length 8192 \
        --ATTN_MODE 3 \
        --cache_dir "$CACHE_DIR" \
        --server_host "localhost" \
        --server_port 10900 \
        --dist_init_addr "$DIST_INIT_ADDR" \
        --nnodes 2 \
        --node_rank "$NODE_RANK"