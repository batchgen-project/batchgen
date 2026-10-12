#!/bin/bash
# PR GPU regression in-container runner.
#
# Assumes:
#   - Running inside tairan's long-running container (tairan-batchgen on
#     node0, batchgen on node1), entered via docker exec with conda env
#     `batchgen` already activated. The PR is tested by creating a per-
#     worktree venv with --system-site-packages (inheriting torch + CUDA
#     deps from the conda env) and `pip install -e . --no-deps
#     --no-build-isolation` into it. PYTHONPATH overrides do NOT work
#     here: batchgen ships compiled C++/CUDA extensions, so any drift
#     between the conda env's installed copy and the PR source produces
#     silent worker crashes on module import. The venv approach forces
#     extensions to be rebuilt against the PR source while keeping the
#     conda env (POIS's interactive shell) untouched.
#   - cwd is the worktree (set by pr-gpu-smoke.sh after `git worktree add`).
#
# Env (set by the outer host script via docker exec -e):
#   NODE_RANK, DIST_INIT_ADDR, WORKTREE_PATH, BATCHGEN_CI_GLM5_MODEL_DIR,
#   BATCHGEN_LONGBENCH_DIR, BATCHGEN_MMLU_PRO_DIR, BATCHGEN_CI_MMLU_THRESHOLD,
#   and the NCCL/GLOO/IB networking vars.
set -euo pipefail

: "${NODE_RANK:?}"
: "${DIST_INIT_ADDR:?}"
: "${WORKTREE_PATH:?}"
: "${BATCHGEN_CI_GLM5_MODEL_DIR:?}"
: "${BATCHGEN_CI_MMLU_THRESHOLD:=0}"

SERVER_LOG="$WORKTREE_PATH/server-rank${NODE_RANK}.log"
HEALTH_TIMEOUT=1800   # 30 min for distributed init + model load
HEALTH_INTERVAL=10
PORT=10900

# Conda env `batchgen` is expected to be active when we get here (the
# outer host script's docker exec runs `conda activate batchgen` first).
echo "[in-container] node_rank=$NODE_RANK dist=$DIST_INIT_ADDR conda_python=$(which python)"
cd "$WORKTREE_PATH"

# Create a per-worktree venv inheriting torch + CUDA deps from the
# conda env, then install the PR's batchgen into it. This rebuilds
# the C++/CUDA extensions against the PR source while keeping the
# conda env's batchgen install (POIS's interactive shell) untouched.
VENV_DIR="$WORKTREE_PATH/.venv-ci"
PIP_LOG="$WORKTREE_PATH/pip-install-rank${NODE_RANK}.log"
echo "--- creating venv at $VENV_DIR (system-site-packages) ---"
python -m venv "$VENV_DIR" --system-site-packages
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[in-container] venv_python=$(which python)"

echo "--- pip install -e . --no-deps --no-build-isolation (logs: $PIP_LOG) ---"
BUILD_OPS=1 pip install -e . --no-deps --no-build-isolation -v > "$PIP_LOG" 2>&1
echo "pip install complete; tail of log:"
tail -8 "$PIP_LOG"

echo "--- launching batchgen.launch_http_server (rank $NODE_RANK) ---"
python -m batchgen.launch_http_server \
    --model zai-org/GLM-5-FP8 \
    --cache-dir "$BATCHGEN_CI_GLM5_MODEL_DIR" \
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
    echo "Rank 1: holding server until peer disconnect..."
    wait "$SERVER_PID" || true
    echo "Rank 1: server exited, done."
    exit 0
fi

# Rank 0 only from here on.
echo "--- mixed MMLU-Pro 512 + LongBench 512 (GLM-5-FP8, max_decoding=65536) ---"
MIXED_LOG="$WORKTREE_PATH/mixed-mmlu-longbench.log"
set +e
python tests/e2e/glm5_mixed_regression/mixed_mmlu_longbench.py \
    --base-url "http://localhost:${PORT}" \
    --cache-dir "$BATCHGEN_CI_GLM5_MODEL_DIR" \
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
