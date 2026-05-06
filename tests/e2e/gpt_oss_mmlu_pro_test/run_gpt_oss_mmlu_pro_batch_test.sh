#!/bin/bash
# GPT-OSS-120B MMLU Pro Batch Test Script
#
# Usage:
#   ./run_gpt_oss_mmlu_pro_batch_test.sh
#
# Prerequisites:
#   1. BatchGen server running with GPT-OSS-120B loaded
#   2. Model checkpoint downloaded to specified path
#
# Server startup command (run separately):
#   python -m batchgen.launch_http_server \
#       --model openai/gpt-oss-120b \
#       --cache-dir /path/to/models/gpt-oss-120b \
#       --listen-port 10900 \
#       --world-size 1 \
#       --host-kv-cache-size 128

set -e

# Configuration
MODEL_PATH="${GPT_OSS_MODEL_PATH:?set GPT_OSS_MODEL_PATH to the GPT-OSS-120B checkpoint dir}"
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_PORT="${SERVER_PORT:-10900}"
MAX_PROMPTS="${MAX_PROMPTS:-1024}"
MAX_DECODING_LENGTH="${MAX_DECODING_LENGTH:-512}"
TEMPERATURE="${TEMPERATURE:-0}"
# Reasoning effort: low (default per OpenAI), medium, or high
REASONING_EFFORT="${REASONING_EFFORT:-low}"

# Create log directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Timestamp for log file
datetime=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/gpt_oss_mmlu_pro_${datetime}.log"

echo "=================================================="
echo "GPT-OSS-120B MMLU Pro Batch Test"
echo "=================================================="
echo "Model Path: ${MODEL_PATH}"
echo "Server: ${SERVER_HOST}:${SERVER_PORT}"
echo "Max Prompts: ${MAX_PROMPTS}"
echo "Max Decoding Length: ${MAX_DECODING_LENGTH}"
echo "Temperature: ${TEMPERATURE}"
echo "Reasoning Effort: ${REASONING_EFFORT}"
echo "Log File: ${LOG_FILE}"
echo "=================================================="

# Run the test
python "${SCRIPT_DIR}/gpt_oss_mmlu_pro_batch_test.py" \
    --hugging_face_checkpoint "openai/gpt-oss-120b" \
    --max_decoding_length "${MAX_DECODING_LENGTH}" \
    --cache_dir "${MODEL_PATH}" \
    --server_host "${SERVER_HOST}" \
    --server_port "${SERVER_PORT}" \
    --max_prompts "${MAX_PROMPTS}" \
    --temperature "${TEMPERATURE}" \
    --reasoning_effort "${REASONING_EFFORT}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Test completed. Log saved to: ${LOG_FILE}"
