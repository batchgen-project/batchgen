#!/bin/bash
# GPT-OSS-120B MMLU Pro batch test script
#
# Usage:
#   ./gpt_oss_mmlu_pro_test.sh [--server_host HOST] [--server_port PORT] [--max_prompts N]
#
# Prerequisites:
#   1. Start BatchGen server with GPT-OSS-120B:
#      python -m batchgen.launch_http_server \
#          --model openai/gpt-oss-120b \
#          --world-size 1 \
#          --listen-port 10900 \
#          --hugetlbfs
#
#   2. Run this test script:
#      ./gpt_oss_mmlu_pro_test.sh --server_host localhost --server_port 10900

set -e

# Default values
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_PORT="${SERVER_PORT:-10900}"
MAX_PROMPTS="${MAX_PROMPTS:-100}"
MAX_DECODING_LENGTH="${MAX_DECODING_LENGTH:-512}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server_host)
            SERVER_HOST="$2"
            shift 2
            ;;
        --server_port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --max_prompts)
            MAX_PROMPTS="$2"
            shift 2
            ;;
        --max_decoding_length)
            MAX_DECODING_LENGTH="$2"
            shift 2
            ;;
        --all)
            MAX_PROMPTS=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "GPT-OSS-120B MMLU Pro Batch Test"
echo "=========================================="
echo "Server: ${SERVER_HOST}:${SERVER_PORT}"
echo "Max Prompts: ${MAX_PROMPTS:-all}"
echo "Max Decoding Length: ${MAX_DECODING_LENGTH}"
echo "=========================================="

# Build command
CMD="python ${SCRIPT_DIR}/gpt_oss_mmlu_pro_batch_test.py \
    --hugging_face_checkpoint openai/gpt-oss-120b \
    --server_host ${SERVER_HOST} \
    --server_port ${SERVER_PORT} \
    --max_decoding_length ${MAX_DECODING_LENGTH}"

if [ -n "${MAX_PROMPTS}" ]; then
    CMD="${CMD} --max_prompts ${MAX_PROMPTS}"
fi

echo "Running: ${CMD}"
echo ""

eval ${CMD}
