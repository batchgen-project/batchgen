#!bin/bash
# This script is for running the r1_mmlu_pro_test.py on dual nodes with specific configurations.
# It tests the first 1024 samples of the MMLU Pro test set using the DeepSeek-R1 model.
# Input prompt length is determined dynamically from the longest prompt in the batch.
# The model generates up to 10240 tokens.
# The script is configured to use fp8 for key-value cache storage and runs on 2 nodes
datetime=$(date +%Y%m%d_%H%M%S)
export HF_ENDPOINT=https://hf-mirror.com
python test/r1_mmlu_pro_test/r1_mmlu_pro_test.py \
    --hugging_face_checkpoint "/root/.cache/huggingface/hub/models/deepseek-ai/DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad" \
    --max_prompts 1024 \
    --max_decoding_length 10240 \
    --cache_dir "/root/.cache/huggingface/hub/models/deepseek-ai/DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad" \
    --server_host "localhost" \
    --server_port 10900 \
    > ./deepseek-r1-bench/${datetime}.log 2>&1

