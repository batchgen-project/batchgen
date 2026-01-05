#!bin/bash
# This script is for running the r1_mmlu_pro_test.py on dual nodes with specific configurations.
# It tests the first 1024 samples of the MMLU Pro test set using the DeepSeek-R1 model.
# Input prompt length is truncated to 2048 tokens, and the model generates up to 8192 tokens.
# The script is configured to use fp8 for key-value cache storage and runs on 2 nodes
datetime=$(date '+%Y-%m-%d-%H-%M-%S')
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_BUFFSIZE=16777216
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python <dir-to-r1_mmlu_pro_test.py> \
    --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
    --max_prompts 1024 \
	--host_kv_cache_size 256 \
    --max_input_length 2048 \
    --max_decoding_length 8192 \
    --ATTN_MODE 3 \
    --cache_dir <dir to model checkpoint> \
    --server_host "localhost" \
    --server_port 10900 \
	--dist_init_addr <dist-init-addr> \
	--nnodes 2 \
	--node_rank <node-rank> \
    --kv_dtype "fp8" \
    > ./deepseek-r1-bench/${datetime}.log 2>&1

