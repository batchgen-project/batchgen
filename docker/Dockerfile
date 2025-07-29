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

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh

    
WORKDIR /root/moegen
RUN uv venv --python ${PYTHON_VERSION}

ENV PATH="/root/moegen/.venv/bin:$PATH"

# Make sure to install ninja to enable fast builds
RUN uv pip install torch==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 wheel packaging ninja \
    && uv cache clean
    
# Install Flash Attention, DeepGEMM and FlashMLA
RUN git clone --recursive https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && git checkout v2.8.2 \
    && cd hopper \
    && uv pip install . --no-build-isolation \
    && uv pip install git+https://github.com/deepseek-ai/FlashMLA.git@9edee0c022cd0938148a18e334203b0aab43aa19 --no-build-isolation \
    && git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git \
    && cd DeepGEMM \
    && git checkout d374456 \
    && git submodule update --init --recursive \
    && uv pip install . --no-build-isolation \
    && uv cache clean


# Install from moe gen source
COPY . /root/moegen

RUN uv pip install -e . -v \
    && uv cache clean

RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


CMD ["sleep", "infinity"]
