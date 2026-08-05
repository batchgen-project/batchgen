#!/usr/bin/env bash
# ---------------------------------------------------------------------------- #
#  Kimi-K3 M2 — staged GPU validation, h20-instance-1, GPU 0 (reserved for K3) #
#                                                                              #
#  Run ON instance-1 from the BatchGen-k3-kimi-linear repo root, inside the    #
#  `batchgen` conda env, AFTER syncing the branch (git push local +            #
#  git pull on remote — NEVER scp/rsync into the repo):                        #
#                                                                              #
#    ssh h20-instance-1 'source /root/miniconda3/etc/profile.d/conda.sh && \
#      conda activate batchgen && \
#      cd <REMOTE_CHECKOUT>/BatchGen-k3-kimi-linear && \
#      bash tests/gpu/run_kimi_k3_kda_instance1.sh'                            #
#                                                                              #
#  Requires: fla-core >= 0.5.0 — the test probes chunk_kda's SIGNATURE for     #
#  use_beta_sigmoid_in_kernel and hard-fails without it (older fla swallows    #
#  the kwarg and consumes beta RAW).  transformers >= 4.56, einops.  GPU 0.    #
#  Set PYTHONPATH=<repo>:<fla-src> — a stale batchgen_kernels in site-packages #
#  shadows the in-tree build when the repo root is not on sys.path.            #
#  ALWAYS report the log path printed below back to POIS.                      #
# ---------------------------------------------------------------------------- #
set -euo pipefail

LOG_DIR=/taijifs_zw35/share_304153846/hunyuan/tairanxu/logs
LOG="$LOG_DIR/k3_m2_kda_gpu_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

echo "== Kimi-K3 M2 GPU stage =="
echo "log: $LOG"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | tee -a "$LOG"

K3_GPU_STAGE=1 CUDA_VISIBLE_DEVICES=0 python -m pytest \
    tests/gpu/test_kimi_k3_kda_fla_parity.py -x -q -rA -p no:cacheprovider \
    2>&1 | tee -a "$LOG"

echo "log: $LOG"
