ARG CUDA_VERSION=12.9.0
ARG PYTHON_VERSION=3.11.13
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ARG PYTHON_VERSION

# Target GPU architecture: "blackwell" (sm120) or "hopper" (sm90).
# Build hopper with:  docker buildx build --build-arg GPU_ARCH=hopper ...
ARG GPU_ARCH=blackwell

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_ENDPOINT=https://hf-mirror.com \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

# RDMA Python UV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git fish libnuma-dev \
    build-essential cmake \
    libibverbs1 libibverbs-dev ibverbs-providers rdma-core \
    vim tmux htop tree silversearcher-ag lsof unzip \
    && rm -rf /var/lib/apt/lists/*

# Install uv (with retry and proper error handling)
RUN set -o pipefail && \
    for i in 1 2 3 4 5; do \
      curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh && break || \
      { echo "Attempt $i failed, retrying in 10s..."; sleep 10; }; \
    done && \
    uv --version

    
WORKDIR /root/moegen
RUN uv venv --python ${PYTHON_VERSION} --seed

ENV PATH="/root/moegen/.venv/bin:$PATH"

# Make sure to install ninja to enable fast builds
RUN uv pip install torch==2.9.0+cu129 --extra-index-url https://download.pytorch.org/whl/cu129 \
    setuptools wheel packaging ninja \
    && uv cache clean
    
# Install Flash Attention
RUN MAX_JOBS=16 git clone --recursive https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && git checkout v2.8.2 \
    && cd hopper \
    && MAX_JOBS=16 FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install . --no-build-isolation \
    && uv cache clean

# Install FlashMLA (built from public upstream source, arch-pinned commit).
# Hopper (sm90) needs the deferred-scheduling API (zero-arg get_mla_metadata + attn_sink,
# DeepSeek-V3.2 sparse) that batchgen/attention/dsa/v4_flashmla_adapter.py depends on; that
# API landed upstream at c741387 (the "1.0.0+c741387" build that was previously vendored is
# simply upstream at this commit). Blackwell (sm120) auto-routes V4 attention to the
# torch/sm120-triton path and never calls FlashMLA, so the older 1408756a is sufficient.
RUN git clone --recursive https://github.com/deepseek-ai/FlashMLA.git \
    && cd FlashMLA \
    && if [ "$GPU_ARCH" = "hopper" ]; then FLASHMLA_REF=c741387bcb1f17fe86d7367201505a6c4dcfaf42; \
       else FLASHMLA_REF=1408756a88e52a25196b759eaf8db89d2b51b5a1; fi \
    && git checkout "$FLASHMLA_REF" \
    && git submodule update --init --recursive \
    && FLASH_MLA_DISABLE_SM100=1 uv pip install -v . --no-build-isolation \
    && uv cache clean

# Install DeepGEMM
RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git \
    && cd DeepGEMM \
    && git checkout v2.1.1.post3 \
    && git submodule update --init --recursive \
    && bash ./install.sh \
    && uv cache clean

# DeepSeek-V4-Flash sparse-prefill kernels (assets/inference/kernel.py) require tilelang,
# and the indexer requires fast_hadamard_transform. Needed by BOTH archs (prefill path is
# arch-independent). tilelang 0.1.8 needs apache-tvm-ffi pinned to 0.1.5 for torch 2.9.
# fast_hadamard_transform is built from GitHub source: its PyPI sdist is broken (missing csrc).
RUN uv pip install tilelang==0.1.8 apache-tvm-ffi==0.1.5 \
    && python -c 'import tilelang, tilelang.language as T; print("tilelang OK", tilelang.__version__)' \
    && git clone https://github.com/Dao-AILab/fast-hadamard-transform.git \
    && cd fast-hadamard-transform \
    && MAX_JOBS=16 uv pip install . --no-build-isolation \
    && python -c 'import fast_hadamard_transform; from fast_hadamard_transform import hadamard_transform; print("fht OK")' \
    && uv cache clean

# Install from moe gen source
COPY . /root/moegen

# Install batchgen_kernels (AOT-compiled CUDA extensions).
# Arch-specific BUILD_ARCH (see batchgen_kernels/setup.py): Hopper sm90a (WGMMA), Blackwell sm120.
RUN cd /root/moegen/batchgen_kernels \
    && if [ "$GPU_ARCH" = "hopper" ]; then BUILD_ARCH=sm90a; TORCH_CUDA_ARCH_LIST="9.0a"; \
       else BUILD_ARCH=sm120; TORCH_CUDA_ARCH_LIST="12.0"; fi \
    && BUILD_ARCH=$BUILD_ARCH TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS=16 uv pip install . --no-build-isolation \
    && uv cache clean

# Install BatchGen (filter out torch/nvidia/triton — already installed with CUDA variant in step 6).
# Re-pin apache-tvm-ffi==0.1.5 LAST: flashinfer-python/requirements pull in 0.1.12, which breaks
# tilelang import on torch 2.9 ("attribute '__dict__' of 'type' objects is not writable").
RUN grep -vE '^(torch==|triton==|nvidia-)' requirements.txt > /tmp/reqs-filtered.txt \
    && uv pip install -r /tmp/reqs-filtered.txt \
    && uv pip install . -v --no-deps \
    && uv pip install flashinfer-python==0.6.12 \
    && uv pip install pytest \
    && uv pip install apache-tvm-ffi==0.1.5 \
    && python -c 'import tilelang, tilelang.language; print("tilelang reimport OK", tilelang.__version__)' \
    && uv cache clean

ENV NCCL_BUFFSIZE=16777216
 
CMD ["sleep", "infinity"]
