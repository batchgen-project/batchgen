ARG CUDA_VERSION=12.8.0
ARG PYTHON_VERSION=3.11.13
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_ENDPOINT=https://hf-mirror.com

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
RUN uv pip install torch==2.9.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 \
    setuptools wheel packaging ninja \
    && uv cache clean
    
# Install Flash Attention
RUN git clone --recursive https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && git checkout v2.8.3 \
    && cd hopper \
    && uv pip install . --no-build-isolation \
    && uv cache clean

# Install FlashMLA
RUN git clone --recursive https://github.com/deepseek-ai/FlashMLA.git \
    && cd FlashMLA \
    && git checkout 1408756a88e52a25196b759eaf8db89d2b51b5a1 \
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

# Install from moe gen source
COPY . /root/moegen

# Install batchgen_kernels (AOT-compiled CUDA extensions)
RUN cd /root/moegen/batchgen_kernels \
    && uv pip install . --no-build-isolation \
    && uv cache clean

# Install BatchGen
RUN uv pip install -r requirements.txt && uv pip install . -v \
    && uv cache clean

ENV NCCL_BUFFSIZE=16777216
 
CMD ["sleep", "infinity"]
