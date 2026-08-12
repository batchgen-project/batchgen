#!/usr/bin/env bash
# ---------------------------------------------------------------------------- #
#  Kimi-K3 M2 — staged GPU validation (single CUDA GPU)                        #
#                                                                              #
#  Run from the repo root on the GPU machine, inside the project's conda      #
#  env, AFTER syncing the branch (git push local + git pull on remote —       #
#  NEVER scp/rsync into the repo):                                             #
#                                                                              #
#    bash tests/gpu/run_kimi_k3_kda.sh                                         #
#                                                                              #
#  Requires: fla-core >= 0.5.0 — the test probes chunk_kda's SIGNATURE for     #
#  use_beta_sigmoid_in_kernel and hard-fails without it (older fla swallows    #
#  the kwarg and consumes beta RAW).  transformers >= 4.56, einops.  GPU 0.    #
#  Set PYTHONPATH=<repo>:<fla-src> — a stale batchgen_kernels in site-packages #
#  shadows the in-tree build when the repo root is not on sys.path.            #
# ---------------------------------------------------------------------------- #
set -euo pipefail

LOG_DIR="${K3_LOG_DIR:-/tmp}"
LOG="$LOG_DIR/k3_m2_kda_gpu_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

echo "== Kimi-K3 M2 GPU stage =="
echo "log: $LOG"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | tee -a "$LOG"

K3_GPU_STAGE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest \
    tests/gpu/test_kimi_k3_kda_fla_parity.py -x -q -rA -p no:cacheprovider \
    2>&1 | tee -a "$LOG"

echo "log: $LOG"
