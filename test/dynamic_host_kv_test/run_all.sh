#!/bin/bash
# ============================================================================
# Dynamic Host KV — Fully Automated Integration Test Suite
# ============================================================================
#
# Runs on dev machine. SSHes to H20 Node0+Node1, handles:
#   cleanup → git sync → data generation → server launch → batch submission →
#   log collection → verification
#
# Usage:
#   ./run_all.sh                    # Run full test matrix
#   ./run_all.sh --run 1            # Run only test matrix entry 1
#   ./run_all.sh --run 1,2          # Run entries 1 and 2
#   ./run_all.sh --skip-sync        # Skip git push/pull
#   ./run_all.sh --skip-generate    # Skip data generation
#
# Prerequisites:
#   - SSH access to wechat_87 and wechat_96
#   - GPT-OSS-120B model at /data2/tairan/models/gpt-oss-120b on Node0
#   - Branch tairan/dynamic-host-kv checked out on both nodes
# ============================================================================

set -euo pipefail

# ===== Configuration =====
NODE0_SSH="wechat_87"
NODE0_DOCKER="tairan-batchgen"
NODE0_PATH="/data2/tairan/workspace/BatchGen"
NODE1_SSH="wechat_96"
NODE1_DOCKER="batchgen"
NODE1_PATH="/data3/tairan/workspace/BatchGen"
MODEL_PATH="/data2/tairan/models/gpt-oss-120b"
MASTER_IP="29.194.13.138"
MASTER_PORT=33001
SERVER_PORT=10900
CONDA="/root/miniconda3/envs/batchgen/bin"

NCCL_ENV="NCCL_SOCKET_IFNAME=bond0 UCX_NET_DEVICES=bond0"
NCCL_ENV+=" NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_4,mlx5_bond_2,mlx5_bond_6,mlx5_bond_3,mlx5_bond_7,mlx5_bond_1,mlx5_bond_5"

# Test directory (on remote)
TEST_DIR="test/dynamic_host_kv_test"
DATA_DIR="${TEST_DIR}/data"
RESULTS_DIR="${TEST_DIR}/results"
LOGS_DIR="${TEST_DIR}/logs"

# Local directories
LOCAL_RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
LOCAL_LOGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse args
SKIP_SYNC=false
SKIP_GENERATE=false
RUN_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-sync) SKIP_SYNC=true; shift ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --run) RUN_FILTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ===== Helper Functions =====

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

# Execute command on remote node inside docker
remote_exec() {
    local ssh_host=$1
    local docker_name=$2
    shift 2
    local cmd="$*"
    ssh -o ConnectTimeout=10 "$ssh_host" \
        "docker exec $docker_name bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate batchgen && $cmd'"
}

# Execute command on remote node host (not inside docker)
remote_host_exec() {
    local ssh_host=$1
    shift
    ssh -o ConnectTimeout=10 "$ssh_host" "$@"
}

clean_node() {
    local ssh_host=$1
    local docker_name=$2
    local node_name=$3
    log "Cleaning $node_name ($ssh_host)..."

    # Step 1: Kill GPU processes (from host)
    remote_host_exec "$ssh_host" "pkill -9 python 2>/dev/null || true"
    sleep 3

    # Step 2: Verify GPU clean (retry up to 5 times)
    local count=99
    for i in 1 2 3 4 5; do
        count=$(remote_host_exec "$ssh_host" \
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l" || echo "99")
        count=$(echo "$count" | tr -d '[:space:]')
        if [ "$count" -eq 0 ]; then
            break
        fi
        log "  GPU not clean yet ($count processes), waiting... (attempt $i/5)"
        sleep 5
    done
    if [ "$count" -ne 0 ]; then
        log "ERROR: GPU not clean on $node_name after 5 attempts"
        return 1
    fi
    log "  GPU clean on $node_name"

    # Step 3: Clean SHM/hugepages (inside docker)
    remote_exec "$ssh_host" "$docker_name" \
        "rm -rf /dev/shm/shm_* /dev/shm/batchgen_* /dev/shm/nccl-* /dev/hugepages/* && \
         echo 0 > /proc/sys/vm/nr_hugepages && \
         rm -f /tmp/batchgen_skel_*.pt"
    log "  SHM/hugepages cleaned on $node_name"

    # Step 4: Verify memory
    local mem_info
    mem_info=$(remote_exec "$ssh_host" "$docker_name" "free -h | head -2")
    log "  Memory on $node_name: $mem_info"
}

git_sync() {
    log "Syncing git to both nodes..."

    # Push locally
    log "  Pushing local changes..."
    cd "$(dirname "$SCRIPT_DIR")/.."
    git push 2>&1 | head -5
    cd "$SCRIPT_DIR"

    # Pull on Node0
    log "  Pulling on Node0..."
    remote_exec "$NODE0_SSH" "$NODE0_DOCKER" \
        "cd $NODE0_PATH && git pull --ff-only" 2>&1 | head -5

    # Pull on Node1
    log "  Pulling on Node1..."
    remote_exec "$NODE1_SSH" "$NODE1_DOCKER" \
        "cd $NODE1_PATH && git pull --ff-only" 2>&1 | head -5

    log "Git sync complete"
}

generate_data() {
    log "Generating test data on Node0..."
    remote_exec "$NODE0_SSH" "$NODE0_DOCKER" \
        "cd $NODE0_PATH && python $TEST_DIR/generate_test_data.py --scenario all --output-dir $DATA_DIR/"
    log "Test data generated"
}

launch_server() {
    local config_name=$1
    local extra_args=$2
    local run_log_dir="${LOGS_DIR}/${TIMESTAMP}_${config_name}"

    log "Launching server with config: $config_name"

    # Create log dirs
    remote_exec "$NODE0_SSH" "$NODE0_DOCKER" "mkdir -p ${NODE0_PATH}/${run_log_dir}"
    remote_exec "$NODE1_SSH" "$NODE1_DOCKER" "mkdir -p ${NODE1_PATH}/${run_log_dir}"

    # Launch Node1 (worker) first
    log "  Starting Node1 (worker)..."
    remote_exec "$NODE1_SSH" "$NODE1_DOCKER" \
        "cd $NODE1_PATH && \
         $NCCL_ENV BATCHGEN_CB_LOG=DEBUG \
         nohup ${CONDA}/python -m batchgen.launch_http_server \
            --model openai/gpt-oss-120b \
            --cache-dir $MODEL_PATH \
            --dist-init-addr ${MASTER_IP}:${MASTER_PORT} \
            --nnodes 2 --node-rank 1 --world-size 16 \
            --kv-dtype bf16 --enable-hugetlbfs --gpu-memory-frac 0.96 \
            $extra_args \
            > ${NODE1_PATH}/${run_log_dir}/server_node1.log 2>&1 &" || true

    sleep 2

    # Launch Node0 (master)
    log "  Starting Node0 (master)..."
    remote_exec "$NODE0_SSH" "$NODE0_DOCKER" \
        "cd $NODE0_PATH && \
         $NCCL_ENV BATCHGEN_CB_LOG=DEBUG \
         nohup ${CONDA}/python -m batchgen.launch_http_server \
            --model openai/gpt-oss-120b \
            --cache-dir $MODEL_PATH \
            --dist-init-addr ${MASTER_IP}:${MASTER_PORT} \
            --nnodes 2 --node-rank 0 --world-size 16 \
            --listen-port $SERVER_PORT \
            --kv-dtype bf16 --enable-hugetlbfs --gpu-memory-frac 0.96 \
            $extra_args \
            > ${NODE0_PATH}/${run_log_dir}/server_node0.log 2>&1 &" || true

    log "  Server processes launched"
    echo "$run_log_dir"  # Return log dir for later collection
}

wait_server_ready() {
    log "Waiting for server to be ready..."
    local max_wait=600  # 10 minutes
    local elapsed=0
    local interval=15

    while [ $elapsed -lt $max_wait ]; do
        # Check health endpoint via Node0
        local status
        status=$(remote_exec "$NODE0_SSH" "$NODE0_DOCKER" \
            "curl -s -o /dev/null -w '%{http_code}' http://localhost:${SERVER_PORT}/health 2>/dev/null" || echo "000")
        status=$(echo "$status" | tr -d '[:space:]')

        if [ "$status" = "200" ]; then
            log "Server ready! (${elapsed}s)"
            return 0
        fi

        log "  Not ready yet (status=$status), waiting... (${elapsed}/${max_wait}s)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log "ERROR: Server failed to start within ${max_wait}s"
    return 1
}

submit_scenario() {
    local scenario=$1
    local config_name=$2
    local input_file="${DATA_DIR}/scenario${scenario}.jsonl"
    local output_file="${RESULTS_DIR}/${config_name}_scenario${scenario}.jsonl"

    log "Submitting scenario $scenario (config: $config_name)..."
    remote_exec "$NODE0_SSH" "$NODE0_DOCKER" \
        "cd $NODE0_PATH && ${CONDA}/python $TEST_DIR/run_test.py \
            --input $input_file \
            --output $output_file \
            --server-url http://localhost:${SERVER_PORT} \
            --timeout 7200 \
            --poll-interval 10"
    log "Scenario $scenario complete"
}

stop_server() {
    log "Stopping server on both nodes..."
    remote_host_exec "$NODE0_SSH" "pkill -9 python 2>/dev/null || true"
    remote_host_exec "$NODE1_SSH" "pkill -9 python 2>/dev/null || true"
    sleep 5
    log "Server stopped"
}

collect_logs() {
    local config_name=$1
    local run_log_dir=$2
    local local_dir="${LOCAL_LOGS_DIR}/${TIMESTAMP}_${config_name}"

    log "Collecting logs for $config_name..."
    mkdir -p "$local_dir"

    # Copy logs from Node0
    ssh "$NODE0_SSH" "docker cp ${NODE0_DOCKER}:${NODE0_PATH}/${run_log_dir}/server_node0.log /tmp/" 2>/dev/null || true
    scp -q "${NODE0_SSH}:/tmp/server_node0.log" "$local_dir/" 2>/dev/null || true

    # Copy logs from Node1
    ssh "$NODE1_SSH" "docker cp ${NODE1_DOCKER}:${NODE1_PATH}/${run_log_dir}/server_node1.log /tmp/" 2>/dev/null || true
    scp -q "${NODE1_SSH}:/tmp/server_node1.log" "$local_dir/" 2>/dev/null || true

    # Copy results from Node0
    mkdir -p "${LOCAL_RESULTS_DIR}"
    ssh "$NODE0_SSH" "docker cp ${NODE0_DOCKER}:${NODE0_PATH}/${RESULTS_DIR}/. /tmp/dyn_kv_results/" 2>/dev/null || true
    scp -q -r "${NODE0_SSH}:/tmp/dyn_kv_results/*" "${LOCAL_RESULTS_DIR}/" 2>/dev/null || true

    log "Logs collected to $local_dir"
    echo "$local_dir"
}

run_verification() {
    local config_name=$1
    local scenario=$2
    local baseline_config=$3  # empty string if no comparison

    local dynamic_file="${LOCAL_RESULTS_DIR}/${config_name}_scenario${scenario}.jsonl"

    if [ ! -f "$dynamic_file" ]; then
        log "WARNING: Results file not found: $dynamic_file"
        return 1
    fi

    local verify_args="--dynamic $dynamic_file"

    # Add baseline comparison if provided
    if [ -n "$baseline_config" ]; then
        local baseline_file="${LOCAL_RESULTS_DIR}/${baseline_config}_scenario${scenario}.jsonl"
        if [ -f "$baseline_file" ]; then
            verify_args+=" --baseline $baseline_file"
        else
            log "WARNING: Baseline file not found: $baseline_file"
        fi
    fi

    # Add server log if available
    local log_dir
    log_dir=$(ls -d "${LOCAL_LOGS_DIR}/${TIMESTAMP}_${config_name}" 2>/dev/null | head -1 || true)
    if [ -n "$log_dir" ] && [ -f "$log_dir/server_node0.log" ]; then
        verify_args+=" --server-log $log_dir/server_node0.log"
    fi

    log "Verifying ${config_name} scenario ${scenario}..."
    python "${SCRIPT_DIR}/verify_results.py" $verify_args || true
}

# ===== Test Matrix =====
# Format: "run_id:config_name:extra_args:scenarios"
declare -a TEST_MATRIX=(
    "1:baseline:--host-kv-chunk-size 131072 --no-adaptive-chunk --host-kv-cache-size 128:A,B,C,F"
    "2:dynamic:--host-kv-chunk-size 8192 --adaptive-chunk --host-kv-cache-size 128:A,B,C,F"
    "3:dynamic_longctx:--host-kv-chunk-size 8192 --adaptive-chunk --host-kv-cache-size 128:D"
    "4:eviction:--host-kv-chunk-size 8192 --adaptive-chunk --host-kv-cache-size 32 --enable-host-kv-eviction --host-kv-eviction-watermark 10:E"
    "5:aggressive:--host-kv-chunk-size 512 --no-adaptive-chunk --host-kv-cache-size 128:C"
)

# ===== Main =====
log "============================================"
log "Dynamic Host KV Integration Test Suite"
log "Timestamp: $TIMESTAMP"
log "============================================"

# Git sync
if [ "$SKIP_SYNC" = false ]; then
    git_sync
fi

# Generate data
if [ "$SKIP_GENERATE" = false ]; then
    generate_data
fi

# Create result dirs on remote
remote_exec "$NODE0_SSH" "$NODE0_DOCKER" "mkdir -p ${NODE0_PATH}/${RESULTS_DIR}"

# Run test matrix
for entry in "${TEST_MATRIX[@]}"; do
    IFS=':' read -r run_id config_name extra_args scenarios <<< "$entry"

    # Filter check
    if [ -n "$RUN_FILTER" ]; then
        if ! echo ",$RUN_FILTER," | grep -q ",$run_id,"; then
            log "Skipping run $run_id ($config_name) — not in filter"
            continue
        fi
    fi

    log ""
    log "============================================"
    log "Run $run_id: $config_name"
    log "  Config: $extra_args"
    log "  Scenarios: $scenarios"
    log "============================================"

    # Clean both nodes
    clean_node "$NODE0_SSH" "$NODE0_DOCKER" "Node0"
    clean_node "$NODE1_SSH" "$NODE1_DOCKER" "Node1"

    # Launch server
    run_log_dir=$(launch_server "$config_name" "$extra_args")

    # Wait for server
    if ! wait_server_ready; then
        log "Server failed to start for $config_name, skipping..."
        stop_server
        continue
    fi

    # Submit each scenario
    IFS=',' read -ra SCENARIO_ARRAY <<< "$scenarios"
    for scenario in "${SCENARIO_ARRAY[@]}"; do
        submit_scenario "$scenario" "$config_name" || {
            log "WARNING: Scenario $scenario failed for $config_name"
        }
    done

    # Stop server and collect logs
    stop_server
    local_log_dir=$(collect_logs "$config_name" "$run_log_dir")

    log "Run $run_id ($config_name) complete"
done

# ===== Verification =====
log ""
log "============================================"
log "Running Verification"
log "============================================"

# Compare runs 1 (baseline) vs 2 (dynamic) for scenarios A,B,C,F
for scenario in A B C F; do
    run_verification "dynamic" "$scenario" "baseline"
done

# Run 3: dynamic_longctx scenario D (completeness only, no baseline)
run_verification "dynamic_longctx" "D" ""

# Run 4: eviction scenario E (completeness + eviction metrics)
run_verification "eviction" "E" ""

# Run 5: aggressive scenario C (compare to dynamic run 2)
run_verification "aggressive" "C" "dynamic"

log ""
log "============================================"
log "All tests complete!"
log "Results: ${LOCAL_RESULTS_DIR}"
log "Logs:    ${LOCAL_LOGS_DIR}"
log "============================================"
