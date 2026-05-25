#!/bin/bash
# PR GPU regression in-container runner.
#
# Launches GLM-5-FP8 server with the regression config, waits for both nodes
# to handshake, and (on rank 0 only) runs MMLU Pro 512 via
# batchgen_benchmark.mmlu_pro_test with an optional accuracy floor.
#
# Env required: NODE_RANK, DIST_INIT_ADDR, BATCHGEN_CI_MMLU_THRESHOLD.
# Mounts assumed: /workspace (PR checkout), /models/glm5-fp8 (weights),
# /batchgen_benchmark (host batchgen_benchmark/ checkout).
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

# batchgen_benchmark/ lives outside the repo; expose it on PYTHONPATH.
export PYTHONPATH="/batchgen_benchmark:${PYTHONPATH:-}"

# batchgen_benchmark/mmlu_pro_test.py loads test/r1_mmlu_pro_test/*.parquet
# from $BATCHGEN_ROOT. The PR's checkout has the parquets under
# tests/e2e/r1_mmlu_pro_test/. Bridge that with a symlink so the default
# lookup path resolves.
mkdir -p test
ln -sfn /workspace/tests/e2e/r1_mmlu_pro_test test/r1_mmlu_pro_test

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
echo "--- MMLU Pro 512 (GLM-5-FP8, --enable-thinking, max_decoding=65536) ---"
MMLU_LOG="/workspace/mmlu-pro-512.log"
set +e
python -m batchgen_benchmark.mmlu_pro_test \
    --model-type glm5 \
    --max-prompts 512 \
    --max-decoding-length 65536 \
    --enable-thinking \
    --base-url "http://localhost:${PORT}" \
    --batchgen-root /workspace 2>&1 | tee "$MMLU_LOG"
mmlu_rc=${PIPESTATUS[0]}
set -e

if [[ $mmlu_rc -ne 0 ]]; then
    echo "FAIL: mmlu_pro_test exited $mmlu_rc"
    exit "$mmlu_rc"
fi

# Optional accuracy gate. batchgen_benchmark prints "Accuracy:  XX.XX%".
if [[ "$BATCHGEN_CI_MMLU_THRESHOLD" != "0" ]]; then
    acc_line=$(grep -E '^Accuracy:' "$MMLU_LOG" | tail -1 || true)
    if [[ -z "$acc_line" ]]; then
        echo "FAIL: could not parse 'Accuracy:' line from $MMLU_LOG"
        exit 1
    fi
    acc_pct=$(echo "$acc_line" | sed -E 's/[^0-9.]//g' | head -c 8)
    awk -v a="$acc_pct" -v t="$BATCHGEN_CI_MMLU_THRESHOLD" \
        'BEGIN { if (a+0 < t+0) { printf "FAIL: accuracy %s%% < threshold %s%%\n", a, t; exit 1 } else { printf "PASS: accuracy %s%% >= threshold %s%%\n", a, t } }'
fi

echo "--- LongBench 512 (GLM-5-FP8, max_decoding=65536) ---"
LONGBENCH_LOG="/workspace/longbench-512.log"
set +e
python tests/e2e/r1_longbench_test/longbench_dual_node.py \
    --hugging_face_checkpoint zai-org/GLM-5-FP8 \
    --max_prompts 512 \
    --max_decoding_length 65536 \
    --dataset_dir tests/e2e/r1_longbench_test/LongBench \
    --cache_dir /models/glm5-fp8 \
    --base_url "http://localhost:${PORT}" 2>&1 | tee "$LONGBENCH_LOG"
longbench_rc=${PIPESTATUS[0]}
set -e
if [[ $longbench_rc -ne 0 ]]; then
    echo "FAIL: longbench_dual_node exited $longbench_rc"
    exit "$longbench_rc"
fi

echo "=== PR GPU Regression PASSED ==="
