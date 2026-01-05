#!/bin/bash
# Launch LongBench client test
export NCCL_BUFFSIZE=16777216
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python longbench_dual_node.py \
    --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
    --max_prompts 368 \
    --max_decoding_length 1024 \
    --cache_dir <dir to model checkpoint> \
    --server_host "localhost" \
    --server_port 10900 \
    > ./deepseek-r1-bench/${datetime}.log 2>&1


