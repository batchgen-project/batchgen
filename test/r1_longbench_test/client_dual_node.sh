#!/bin/bash
# Launch LongBench client test using the new HTTP API
#
# Server must be running first:
#   python -m batchgen.launch_http_server --model deepseek-ai/DeepSeek-R1 \
#       --cache-dir <dir to model checkpoint> --listen-port 10900
#
export NCCL_BUFFSIZE=16777216

datetime=$(date +%Y%m%d_%H%M%S)
mkdir -p ./deepseek-r1-bench

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python longbench_dual_node.py \
    --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
    --max_prompts 368 \
    --max_decoding_length 1024 \
    --cache_dir <dir to model checkpoint> \
    --base_url "http://localhost:10900" \
    --timeout 6000 \
    > ./deepseek-r1-bench/${datetime}.log 2>&1
