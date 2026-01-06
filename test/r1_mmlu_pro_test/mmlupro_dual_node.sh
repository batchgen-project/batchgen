#!/bin/bash
# This script is for running the r1_mmlu_pro_test.py using the new HTTP API.
# It tests the first 1024 samples of the MMLU Pro test set using the DeepSeek-R1 model.
# Input prompt length is determined dynamically from the longest prompt in the batch.
# The model generates up to 10240 tokens.
#
# Server must be running first:
#   python -m batchgen.launch_http_server --model deepseek-ai/DeepSeek-R1 \
#       --cache-dir <dir to model checkpoint> --listen-port 10900

datetime=$(date +%Y%m%d_%H%M%S)
mkdir -p ./deepseek-r1-bench

export HF_ENDPOINT=https://hf-mirror.com

python test/r1_mmlu_pro_test/r1_mmlu_pro_test.py \
    --hugging_face_checkpoint "/root/.cache/huggingface/hub/models/deepseek-ai/DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad" \
    --max_prompts 1024 \
    --max_decoding_length 10240 \
    --cache_dir "/root/.cache/huggingface/hub/models/deepseek-ai/DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad" \
    --base_url "http://localhost:10900" \
    --timeout 6000 \
    > ./deepseek-r1-bench/${datetime}.log 2>&1
