#!/bin/bash
# Release MMLU test runner. Executes on the self-hosted runner; sources
# runner-local config from $BATCHGEN_CI_ENV_FILE so cluster networking/paths
# stay out of the public repo.
set -euo pipefail

NODE_RANK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$NODE_RANK" ]]; then
    echo "ERROR: --node-rank is required" >&2
    exit 1
fi

ENV_FILE="${BATCHGEN_CI_ENV_FILE:-/opt/batchgen-ci/env.sh}"
if [[ ! -r "$ENV_FILE" ]]; then
    echo "ERROR: runner env file not readable: $ENV_FILE" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

: "${IMAGE:?IMAGE must be set in workflow env}"
: "${DIST_INIT_ADDR:?DIST_INIT_ADDR must be set}"
: "${MODEL_CACHE_DIR:?MODEL_CACHE_DIR must be set}"
: "${MAX_PROMPTS:=0}"
: "${MAX_INPUT_LENGTH:=8192}"
: "${MAX_DECODING_LENGTH:=8192}"
: "${KV_CACHE_SIZE:=256}"
: "${BATCHGEN_CI_SHM_SIZE:=512g}"

# NCCL/GLOO/IB vars are expected to be defined by $ENV_FILE. Pass them
# through to the container via -e <name> (no value) so they pick up the
# current process env without being echoed into the workflow log.
NCCL_VARS=(
    GLOO_SOCKET_IFNAME
    NCCL_SOCKET_IFNAME
    NCCL_IB_DISABLE
    NCCL_NET
    NCCL_OOB_NET_IFNAME
    NCCL_OOB_NET_ENABLE
    NCCL_IB_HCA
)

docker_env_args=()
for var in "${NCCL_VARS[@]}"; do
    if [[ -n "${!var:-}" ]]; then
        docker_env_args+=("-e" "$var")
    fi
done

docker run --rm --gpus all \
    "${docker_env_args[@]}" \
    -v "${MODEL_CACHE_DIR}":/root/.cache/huggingface/hub/models/deepseek-ai/DeepSeek-R1/ \
    --network host \
    --privileged \
    --shm-size "$BATCHGEN_CI_SHM_SIZE" \
    --ipc=host \
    "$IMAGE" \
    bash .github/workflows/scripts/test-mmlu.sh \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "${DIST_INIT_ADDR}" \
    --cache-dir "/root/.cache/huggingface/hub/models/deepseek-ai/DeepSeek-R1/" \
    --max-prompts "${MAX_PROMPTS}" \
    --max-input-length "${MAX_INPUT_LENGTH}" \
    --max-decoding-length "${MAX_DECODING_LENGTH}" \
    --kv-cache-size "${KV_CACHE_SIZE}"
