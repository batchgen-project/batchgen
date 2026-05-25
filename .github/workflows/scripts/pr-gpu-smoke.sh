#!/bin/bash
# PR GPU regression launcher (host side, runs on the self-hosted runner).
#
# Responsibilities:
#   1. Source runner-local config from $BATCHGEN_CI_ENV_FILE so cluster
#      networking and paths stay out of the public repo.
#   2. docker run the BatchGen image with the right mounts and env, and
#      hand off to pr-gpu-smoke-incontainer.sh inside the container.
#
# Assumes the runner has (configure via $BATCHGEN_CI_ENV_FILE):
#   BATCHGEN_CI_IMAGE             # docker image (matches release image)
#   BATCHGEN_CI_SHM_SIZE          # e.g. 512g
#   BATCHGEN_CI_GLM5_MODEL_CACHE  # host path with GLM-5-FP8 weights
#   BATCHGEN_CI_MMLU_THRESHOLD    # accuracy floor in percent, e.g. 65.0; 0 disables
#   GLOO_SOCKET_IFNAME / NCCL_*   # network config for distributed init
set -euo pipefail

NODE_RANK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
    echo "ERROR: --node-rank must be 0 or 1" >&2
    exit 1
fi

ENV_FILE="${BATCHGEN_CI_ENV_FILE:-/opt/batchgen-ci/env.sh}"
if [[ ! -r "$ENV_FILE" ]]; then
    echo "ERROR: runner env file not readable: $ENV_FILE" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

: "${BATCHGEN_CI_IMAGE:?must be set in $ENV_FILE}"
: "${BATCHGEN_CI_SHM_SIZE:=512g}"
: "${BATCHGEN_CI_GLM5_MODEL_CACHE:?must be set in $ENV_FILE}"
: "${BATCHGEN_CI_MMLU_THRESHOLD:=0}"
: "${DIST_INIT_ADDR:?must be passed via workflow secret}"

REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

# NCCL/GLOO/IB vars expected from $ENV_FILE — pass through by name only so
# values never appear in workflow logs.
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

echo "=== PR GPU Regression ==="
echo "Node rank:        $NODE_RANK"
echo "Image:            $BATCHGEN_CI_IMAGE"
echo "Repo:             $REPO_DIR"
echo "MMLU threshold:   $BATCHGEN_CI_MMLU_THRESHOLD%"

exec docker run --rm \
    --gpus all \
    --shm-size "$BATCHGEN_CI_SHM_SIZE" \
    --ipc=host \
    --network host \
    --privileged \
    "${docker_env_args[@]}" \
    -e NODE_RANK="$NODE_RANK" \
    -e DIST_INIT_ADDR="$DIST_INIT_ADDR" \
    -e BATCHGEN_CI_MMLU_THRESHOLD="$BATCHGEN_CI_MMLU_THRESHOLD" \
    -v "$REPO_DIR":/workspace:rw \
    -v "$BATCHGEN_CI_GLM5_MODEL_CACHE":/models/glm5-fp8:ro \
    -w /workspace \
    "$BATCHGEN_CI_IMAGE" \
    bash /workspace/.github/workflows/scripts/pr-gpu-smoke-incontainer.sh
