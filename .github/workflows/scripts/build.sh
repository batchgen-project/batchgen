#!/bin/bash
# Build Test runner (host side, runs on the self-hosted H20 runner).
#
# Builds a wheel from the PR checkout inside the BatchGen image. The image
# already has torch, ninja, numa, cufile, and the CUDA toolkit, so we just
# need to run pip wheel --no-build-isolation against the mounted source.
#
# Sources $BATCHGEN_CI_ENV_FILE to pick up BATCHGEN_CI_IMAGE without leaking
# the image tag into the public workflow YAML.
set -euo pipefail

ENV_FILE="${BATCHGEN_CI_ENV_FILE:-/opt/batchgen-ci/env.sh}"
if [[ ! -r "$ENV_FILE" ]]; then
    echo "ERROR: runner env file not readable: $ENV_FILE" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${BATCHGEN_CI_IMAGE:?must be set in $ENV_FILE}"

REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

echo "=== Build Test ==="
echo "Image: $BATCHGEN_CI_IMAGE"
echo "Repo:  $REPO_DIR"

exec docker run --rm \
    -v "$REPO_DIR":/workspace:rw \
    -w /workspace \
    -e BUILD_OPS=1 \
    "$BATCHGEN_CI_IMAGE" \
    bash -lc '
        set -euo pipefail
        # Match pr-gpu-smoke-incontainer.sh: prefer conda env if the image
        # ships one (node1-style), otherwise rely on the venv already on PATH
        # (node0-style).
        if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
            source /root/miniconda3/etc/profile.d/conda.sh
            conda activate batchgen
        fi
        echo "Using python: $(which python)"
        python --version
        BUILD_OPS=1 pip wheel . --no-deps --no-build-isolation -v -w dist/
        ls -la dist/
    '
