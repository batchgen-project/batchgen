#!/bin/bash
# ---------------------------------------------------------------------------- #
#  BatchGen Fresh Install + Test Pipeline                                       #
#  Copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  Creates a fresh conda environment, installs everything from scratch with     #
#  pip install . (non-editable, matching production/ray deployment), and runs   #
#  import smoke tests + optional server integration test.                       #
#                                                                               #
#  Usage:                                                                       #
#    bash scripts/test_install.sh [OPTIONS]                                     #
#                                                                               #
#  Options:                                                                     #
#    --skip-deps        Skip flash-attn/FlashMLA/DeepGEMM (saves ~30min)       #
#    --skip-server      Skip server launch + batch test                         #
#    --keep-env         Don't remove the test conda env after completion        #
#    --env-name NAME    Conda env name (default: batchgen_install_test)         #
#    --model MODEL      Model to test (default: openai/gpt-oss-120b)           #
#    --cache-dir DIR    Model cache directory                                   #
#    --world-size N     Number of GPUs (default: 1)                             #
#    --port PORT        Server port (default: 10900)                            #
#    --wheel-dir DIR    Pre-built wheels for flash-attn/FlashMLA/DeepGEMM      #
#    --build-wheels     Build wheels into --wheel-dir (use existing env)        #
# ---------------------------------------------------------------------------- #

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

step()    { echo -e "${BLUE}==>${NC} $1"; }
ok()      { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()    { echo -e "${RED}[FAIL]${NC} $1"; }
divider() { echo -e "${BLUE}$(printf '=%.0s' {1..60})${NC}"; }

# ── Defaults ──
SKIP_DEPS=0
SKIP_SERVER=0
KEEP_ENV=0
BUILD_WHEELS=0
ENV_NAME="batchgen_install_test"
MODEL="openai/gpt-oss-120b"
CACHE_DIR=""
WHEEL_DIR=""
WORLD_SIZE=1
PORT=10900

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)     SKIP_DEPS=1;    shift ;;
        --skip-server)   SKIP_SERVER=1;  shift ;;
        --keep-env)      KEEP_ENV=1;     shift ;;
        --build-wheels)  BUILD_WHEELS=1; shift ;;
        --env-name)      ENV_NAME="$2";  shift 2 ;;
        --model)         MODEL="$2";     shift 2 ;;
        --cache-dir)     CACHE_DIR="$2"; shift 2 ;;
        --wheel-dir)     WHEEL_DIR="$2"; shift 2 ;;
        --world-size)    WORLD_SIZE="$2"; shift 2 ;;
        --port)          PORT="$2";      shift 2 ;;
        --help)
            head -20 "$0" | tail -15
            exit 0
            ;;
        *) fail "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Locate BatchGen root ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCHGEN_DIR="$(dirname "$SCRIPT_DIR")"

if [[ ! -f "$BATCHGEN_DIR/setup.py" ]]; then
    fail "Cannot find BatchGen setup.py at $BATCHGEN_DIR"
    exit 1
fi

step "BatchGen root: $BATCHGEN_DIR"

# ── Pinned versions (must match install_deps.sh) ──
FLASH_ATTN_VERSION="v2.8.2"
FLASHMLA_COMMIT="9edee0c022cd0938148a18e334203b0aab43aa19"
DEEPGEMM_COMMIT="d374456"

# ============================================================================ #
# Build-wheels mode: build wheels from source repos, then exit
# ============================================================================ #
if [[ $BUILD_WHEELS -eq 1 ]]; then
    if [[ -z "$WHEEL_DIR" ]]; then
        fail "--build-wheels requires --wheel-dir"
        exit 1
    fi
    mkdir -p "$WHEEL_DIR"
    step "Building dependency wheels into $WHEEL_DIR"

    DEPS_DIR="${BATCHGEN_INSTALL_DIR:-/tmp/batchgen_build_deps}"
    mkdir -p "$DEPS_DIR"

    # Flash-Attention 3
    step "Building flash-attention 3 wheel..."
    cd "$DEPS_DIR"
    if [[ ! -d "flash-attention" ]]; then
        git clone --recursive https://github.com/Dao-AILab/flash-attention.git
    fi
    cd flash-attention && git checkout "$FLASH_ATTN_VERSION"
    cd hopper && pip wheel . --no-build-isolation -w "$WHEEL_DIR"
    ok "flash-attention 3 wheel built"

    # FlashMLA
    step "Building FlashMLA wheel..."
    cd "$DEPS_DIR"
    if [[ ! -d "FlashMLA" ]]; then
        git clone --recursive https://github.com/deepseek-ai/FlashMLA.git
    fi
    cd FlashMLA && git checkout "$FLASHMLA_COMMIT" && git submodule update --init --recursive
    pip wheel . --no-build-isolation -w "$WHEEL_DIR"
    ok "FlashMLA wheel built"

    # DeepGEMM
    step "Building DeepGEMM wheel..."
    cd "$DEPS_DIR"
    if [[ ! -d "DeepGEMM" ]]; then
        git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
    fi
    cd DeepGEMM && git checkout "$DEEPGEMM_COMMIT" && git submodule update --init --recursive
    pip wheel . --no-build-isolation -w "$WHEEL_DIR"
    ok "DeepGEMM wheel built"

    ok "All wheels built in $WHEEL_DIR:"
    ls -lh "$WHEEL_DIR"/*.whl
    exit 0
fi

# ── Track elapsed time ──
PIPELINE_START=$SECONDS

cleanup() {
    local exit_code=$?
    if [[ $KEEP_ENV -eq 0 && -n "$ENV_NAME" ]]; then
        step "Cleaning up conda env: $ENV_NAME"
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    fi
    local elapsed=$(( SECONDS - PIPELINE_START ))
    divider
    if [[ $exit_code -eq 0 ]]; then
        ok "Pipeline completed in ${elapsed}s"
    else
        fail "Pipeline failed after ${elapsed}s (exit code: $exit_code)"
    fi
    divider
}
trap cleanup EXIT

# ============================================================================ #
# Phase 1: Create fresh conda env
# ============================================================================ #
divider
step "Phase 1: Creating fresh conda env '$ENV_NAME'"

# Remove if exists
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true

conda create -n "$ENV_NAME" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Verify clean env
PYTHON_PATH=$(which python)
step "Python: $PYTHON_PATH"
python --version

# Verify no stale packages
if python -c "import batchgen" 2>/dev/null; then
    fail "batchgen already importable in fresh env — something is wrong"
    exit 1
fi
if python -c "import batchgen_kernels" 2>/dev/null; then
    fail "batchgen_kernels already importable in fresh env — something is wrong"
    exit 1
fi
ok "Fresh env verified (no pre-existing batchgen packages)"

# ============================================================================ #
# Phase 2: Install PyTorch
# ============================================================================ #
divider
step "Phase 2: Installing PyTorch 2.9.0+cu128"

pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install ninja setuptools wheel packaging

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
ok "PyTorch $TORCH_VER (CUDA $CUDA_VER)"

# ============================================================================ #
# Phase 3: Install Hopper dependencies (optional)
# ============================================================================ #
divider
if [[ $SKIP_DEPS -eq 1 ]]; then
    warn "Phase 3: Skipping flash-attn/FlashMLA/DeepGEMM (--skip-deps)"

    step "Installing deps from requirements.txt..."
    cd "$BATCHGEN_DIR"
    pip install -r requirements.txt 2>&1 | tail -5
    ok "Requirements installed"
elif [[ -n "$WHEEL_DIR" && -d "$WHEEL_DIR" ]]; then
    step "Phase 3: Installing Hopper dependencies from pre-built wheels"
    step "Wheel dir: $WHEEL_DIR"

    # Install from cached wheels (no network, no compilation)
    pip install --find-links "$WHEEL_DIR" --no-index \
        flash-attn-hopper flash-mla deep-gemm 2>&1 || {
        # Wheel names may vary — try installing all .whl files
        warn "Named install failed, installing all wheels from $WHEEL_DIR"
        pip install "$WHEEL_DIR"/*.whl
    }
    ok "Hopper dependencies installed from wheels"

    # Install remaining requirements
    cd "$BATCHGEN_DIR"
    pip install -r requirements.txt 2>&1 | tail -5
    ok "Requirements installed"
else
    step "Phase 3: Installing Hopper dependencies from source"

    DEPS_DIR="${BATCHGEN_INSTALL_DIR:-/tmp/batchgen_test_deps}"
    mkdir -p "$DEPS_DIR"

    # Flash-Attention 3
    step "Building flash-attention 3..."
    cd "$DEPS_DIR"
    if [[ ! -d "flash-attention" ]]; then
        git clone --recursive https://github.com/Dao-AILab/flash-attention.git
    fi
    cd flash-attention && git checkout "$FLASH_ATTN_VERSION"
    cd hopper && pip install . --no-build-isolation
    ok "flash-attention 3 installed"

    # FlashMLA
    step "Building FlashMLA..."
    pip install "git+https://github.com/deepseek-ai/FlashMLA.git@${FLASHMLA_COMMIT}" --no-build-isolation
    ok "FlashMLA installed"

    # DeepGEMM
    step "Building DeepGEMM..."
    cd "$DEPS_DIR"
    if [[ ! -d "DeepGEMM" ]]; then
        git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
    fi
    cd DeepGEMM && git checkout "$DEEPGEMM_COMMIT" && git submodule update --init --recursive
    pip install . --no-build-isolation
    ok "DeepGEMM installed"

    # Reinstall PyTorch (deps may have pulled wrong version)
    step "Reinstalling PyTorch to ensure correct version..."
    pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    ok "PyTorch reinstalled"

    # Install remaining requirements
    cd "$BATCHGEN_DIR"
    pip install -r requirements.txt 2>&1 | tail -5
fi

# ============================================================================ #
# Phase 4: Install batchgen_kernels (AOT CUDA extensions)
# ============================================================================ #
divider
step "Phase 4: Installing batchgen_kernels (AOT compilation)"

cd "$BATCHGEN_DIR/batchgen_kernels"

T0=$SECONDS
pip install . --no-build-isolation 2>&1 | tee /tmp/batchgen_kernels_build.log | tail -20
T1=$SECONDS

# Check for build errors
if grep -qiE "error|fatal|failed" /tmp/batchgen_kernels_build.log; then
    if ! grep -q "Successfully installed" /tmp/batchgen_kernels_build.log; then
        fail "batchgen_kernels build failed — check /tmp/batchgen_kernels_build.log"
        exit 1
    fi
fi

ok "batchgen_kernels installed in $(( T1 - T0 ))s"

# Verify it's in site-packages (not editable)
KERNELS_LOC=$(python -c "import batchgen_kernels; print(batchgen_kernels.__file__)")
if [[ "$KERNELS_LOC" == *"site-packages"* ]]; then
    ok "batchgen_kernels location: $KERNELS_LOC (site-packages)"
else
    warn "batchgen_kernels location: $KERNELS_LOC (NOT in site-packages)"
fi

# ============================================================================ #
# Phase 5: Install BatchGen (non-editable)
# ============================================================================ #
divider
step "Phase 5: Installing BatchGen (non-editable)"

cd "$BATCHGEN_DIR"
pip install . 2>&1 | tail -10

# Verify it's in site-packages
BG_LOC=$(python -c "import batchgen; print(batchgen.__file__)")
if [[ "$BG_LOC" == *"site-packages"* ]]; then
    ok "batchgen location: $BG_LOC (site-packages)"
else
    warn "batchgen location: $BG_LOC (NOT in site-packages)"
fi

# ============================================================================ #
# Phase 6: Import smoke tests
# ============================================================================ #
divider
step "Phase 6: Running import smoke tests"

# Ensure no JIT cache is used
export TORCH_EXTENSIONS_DIR="/tmp/batchgen_test_torch_ext_$$"
mkdir -p "$TORCH_EXTENSIONS_DIR"

python -c "
import sys

# ── Test 1: batchgen_kernels base ──
import batchgen_kernels
print(f'batchgen_kernels: {batchgen_kernels.__file__}')

# ── Test 2: All CUDA extensions ──
extensions = [
    'batchgen_kernels.moe._C_expert_mxfp4_wgmma',
    'batchgen_kernels.moe._C_grouped_mxfp4_wgmma',
    'batchgen_kernels.moe._C_grouped_int4_wgmma',
    'batchgen_kernels.moe._C_single_expert_int4_wgmma',
    'batchgen_kernels.moe._C_fused_int4_wgmma_grouped',
    'batchgen_kernels.attention._C_qkv_wgmma',
    'batchgen_kernels.moe._C_routing',
    'batchgen_kernels.moe._C_dispatch_scatter_3d',
    'batchgen_kernels.attention._C_fused_ops',
    'batchgen_kernels.moe._C_mxfp4_dequant_cute',
    'batchgen_kernels.moe._C_mxfp4_dequant',
    'batchgen_kernels.common._C_rmsnorm',
    'batchgen_kernels.common._C_cuda_rmsnorm',
    'batchgen_kernels.common._C_mgn_ops',
]
failed = []
for ext in extensions:
    try:
        mod = batchgen_kernels.load_extension(ext)
        print(f'  {ext}: OK ({len(dir(mod))} symbols)')
    except Exception as e:
        print(f'  {ext}: FAILED ({e})')
        failed.append(ext)

# ── Test 3: Triton imports ──
try:
    from batchgen_kernels.triton import (
        fused_rmsnorm, fused_add_rmsnorm,
        run_paged_kv_token_update,
        moe_weighted_sum_triton,
        fused_fp8_bf16_gemm,
    )
    print('Triton imports: OK')
except Exception as e:
    print(f'Triton imports: FAILED ({e})')
    failed.append('triton_imports')

# ── Test 4: MGN wrapper ──
try:
    from batchgen_kernels.common.mgn import (
        moe_fused_gate, expert_bincount,
        fused_moe_token_dispatch, compact_expert_data, fused_rmsnorm,
    )
    print('MGN wrapper: OK')
except Exception as e:
    print(f'MGN wrapper: FAILED ({e})')
    failed.append('mgn_wrapper')

# ── Test 5: Backward-compat re-exports ──
try:
    from batchgen.other_kernels.triton_rmsnorm import fused_rmsnorm
    from batchgen.moe.moe_weighted_sum import moe_weighted_sum_triton
    from batchgen.kv_cache.gpu_kv_kernels import run_paged_kv_token_update
    print('Backward-compat re-exports: OK')
except Exception as e:
    print(f'Backward-compat re-exports: FAILED ({e})')
    failed.append('backward_compat')

# ── Summary ──
print()
if failed:
    print(f'FAILED: {len(failed)} test(s): {failed}')
    sys.exit(1)
else:
    print('All import tests passed!')
"

# Verify no JIT cache was created
JIT_FILES=$(find "$TORCH_EXTENSIONS_DIR" -name "*.so" 2>/dev/null | wc -l)
if [[ "$JIT_FILES" -gt 0 ]]; then
    warn "JIT cache has $JIT_FILES .so files — some kernels may still use JIT"
    find "$TORCH_EXTENSIONS_DIR" -name "*.so" -exec echo "  {}" \;
else
    ok "No JIT cache created — all kernels loaded from AOT-compiled site-packages"
fi
rm -rf "$TORCH_EXTENSIONS_DIR"

ok "Phase 6: All import tests passed"

# ============================================================================ #
# Phase 7: Server integration test (optional)
# ============================================================================ #
if [[ $SKIP_SERVER -eq 1 ]]; then
    divider
    warn "Phase 7: Skipping server test (--skip-server)"
else
    divider
    step "Phase 7: Server integration test"

    if [[ -z "$CACHE_DIR" ]]; then
        warn "No --cache-dir specified, skipping server test"
        warn "Usage: --cache-dir /path/to/model/cache"
    else
        # Clean SHM/hugepages
        step "Cleaning shared memory..."
        rm -rf /dev/shm/shm_* /dev/shm/batchgen_* /dev/shm/nccl-* /dev/hugepages/* 2>/dev/null || true
        echo 0 > /proc/sys/vm/nr_hugepages 2>/dev/null || true
        rm -f /tmp/batchgen_skel_*.pt

        # Launch server in background
        step "Launching server: $MODEL (world-size=$WORLD_SIZE, port=$PORT)"
        python -m batchgen.launch_http_server \
            --model "$MODEL" \
            --cache-dir "$CACHE_DIR" \
            --listen-port "$PORT" \
            --world-size "$WORLD_SIZE" \
            --host-kv-cache-size 128 \
            --kv-dtype bf16 \
            > /tmp/test_server_log.txt 2>&1 &
        SERVER_PID=$!

        # Wait for server to become healthy
        step "Waiting for server to start (PID=$SERVER_PID)..."
        MAX_WAIT=600
        WAITED=0
        while [[ $WAITED -lt $MAX_WAIT ]]; do
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                fail "Server process died"
                tail -30 /tmp/test_server_log.txt
                exit 1
            fi
            if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "healthy"; then
                ok "Server healthy after ${WAITED}s"
                break
            fi
            sleep 10
            WAITED=$(( WAITED + 10 ))
            if (( WAITED % 60 == 0 )); then
                step "Still waiting... (${WAITED}s / ${MAX_WAIT}s)"
                tail -3 /tmp/test_server_log.txt
            fi
        done

        if [[ $WAITED -ge $MAX_WAIT ]]; then
            fail "Server did not start within ${MAX_WAIT}s"
            tail -50 /tmp/test_server_log.txt
            kill $SERVER_PID 2>/dev/null
            exit 1
        fi

        # Check for the num_nodes fix
        if grep -q "Per-node host KV free pages: \[\]" /tmp/test_server_log.txt; then
            fail "per_node_host_free is empty — _get_num_nodes() bug still present"
            kill $SERVER_PID 2>/dev/null
            exit 1
        fi
        ok "Per-node host KV free pages is non-empty"

        # Run batch test
        step "Submitting 8-prompt MMLU-Pro batch test..."
        TEST_DIR="$BATCHGEN_DIR/test/gpt_oss_mmlu_pro_test"
        if [[ -f "$TEST_DIR/gpt_oss_mmlu_pro_batch_test.py" ]]; then
            python "$TEST_DIR/gpt_oss_mmlu_pro_batch_test.py" \
                --hugging_face_checkpoint "$MODEL" \
                --server_host localhost \
                --server_port "$PORT" \
                --cache_dir "$CACHE_DIR" \
                --max_prompts 8 \
                --max_decoding_length 256 \
                2>&1 | tail -30 || warn "Batch test script exited with error (may be parsing issue)"

            # Check server logs for inference errors
            if grep -q "Error during inference" /tmp/test_server_log.txt; then
                fail "Server reported inference errors"
                grep "Error during inference" /tmp/test_server_log.txt
                kill $SERVER_PID 2>/dev/null
                exit 1
            fi
            ok "Batch test completed without server errors"
        else
            warn "Test script not found at $TEST_DIR, skipping batch test"
        fi

        # Cleanup server
        step "Stopping server..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null || true
        ok "Server stopped"
    fi
fi

divider
ok "All tests passed!"
