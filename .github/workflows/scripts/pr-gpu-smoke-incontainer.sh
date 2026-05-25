#!/bin/bash
# PR GPU regression in-container runner.
#
# Launches GLM-5-FP8 server on each node, waits for both ranks to handshake,
# and (on rank 0 only) runs the in-repo mixed MMLU-Pro 512 + LongBench 512
# regression as a single batch via
# tests/e2e/glm5_mixed_regression/mixed_mmlu_longbench.py.
#
# Env required: NODE_RANK, DIST_INIT_ADDR, BATCHGEN_CI_MMLU_THRESHOLD.
# Mounts assumed: /workspace (PR checkout), /models/glm5-fp8 (weights).
set -euo pipefail

: "${NODE_RANK:?}"
: "${DIST_INIT_ADDR:?}"
: "${BATCHGEN_CI_MMLU_THRESHOLD:=0}"

SERVER_LOG="/workspace/server-rank${NODE_RANK}.log"
HEALTH_TIMEOUT=1800   # 30 min for distributed init + model load
HEALTH_INTERVAL=10
PORT=10900

echo "[in-container] node_rank=$NODE_RANK dist=$DIST_INIT_ADDR"
cd /workspace

# Activate the conda env if the image ships one — node1's image uses conda;
# node0's relies on a venv already on PATH. Either path provides batchgen.
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate batchgen
fi

echo "Using python: $(which python)"

# Install package fresh from PR source so we test the PR build, not a
# pre-baked image.
echo "--- pip install -e . ---"
pip install -q -e .

echo "--- launching batchgen.launch_http_server (rank $NODE_RANK) ---"
python -m batchgen.launch_http_server \
    --model zai-org/GLM-5-FP8 \
    --cache-dir /models/glm5-fp8 \
    --host-kv-cache-size 600 \
    --enable-cuda-graph \
    --cuda-graph-num-buckets 7 \
    --cuda-graph-max-bucket-size 64 \
    --fast-init \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --nnodes 2 \
    --node-rank "$NODE_RANK" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "--- waiting for /health (up to ${HEALTH_TIMEOUT}s) ---"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server (rank $NODE_RANK) died before becoming healthy"
        echo "--- last 200 lines of $SERVER_LOG ---"
        tail -200 "$SERVER_LOG" || true
        exit 1
    fi
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "Server healthy (rank $NODE_RANK)"
        break
    fi
    if [[ $(date +%s) -ge $deadline ]]; then
        echo "ERROR: server (rank $NODE_RANK) failed to become healthy within ${HEALTH_TIMEOUT}s"
        tail -200 "$SERVER_LOG" || true
        exit 1
    fi
    sleep "$HEALTH_INTERVAL"
done

if [[ "$NODE_RANK" != "0" ]]; then
    # Rank 1: keep the server alive while rank 0 drives the benchmark.
    # When rank 0's worker exits, NCCL propagates the disconnect and the
    # local server process exits too. The EXIT trap is the safety net.
    echo "Rank 1: holding server until peer disconnect..."
    wait "$SERVER_PID" || true
    echo "Rank 1: server exited, done."
    exit 0
fi

# Rank 0 only from here on.
echo "--- mixed MMLU-Pro 512 + LongBench 512 (GLM-5-FP8, max_decoding=65536) ---"
MIXED_LOG="/workspace/mixed-mmlu-longbench.log"
set +e
python tests/e2e/glm5_mixed_regression/mixed_mmlu_longbench.py \
    --base-url "http://localhost:${PORT}" \
    --cache-dir /models/glm5-fp8 \
    --mmlu-prompts 512 \
    --longbench-prompts 512 \
    --max-decoding-length 65536 \
    --mmlu-threshold "$BATCHGEN_CI_MMLU_THRESHOLD" \
    --enable-thinking \
    --seed 0 2>&1 | tee "$MIXED_LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ $rc -ne 0 ]]; then
    echo "FAIL: mixed_mmlu_longbench exited $rc"
    exit "$rc"
fi

echo "=== PR GPU Regression PASSED ==="
