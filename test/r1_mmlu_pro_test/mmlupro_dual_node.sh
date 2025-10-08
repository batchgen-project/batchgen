#!bin/bash
datetime=$(date '+%Y-%m-%d-%H-%M-%S')
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python <dir-to-r1_mmlu_pro_test.py> \
    --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
	--host_kv_cache_size 256 \
    --max_input_length 2048 \
    --max_decoding_length 4096 \
    --ATTN_MODE 3 \
    --cache_dir <dir to model checkpoint> \
    --server_host "localhost" \
    --server_port 10900 \
	--dist_init_addr <dist-init-addr> \
	--nnodes 2 \
	--node_rank <node-rank> \
    --kv_dtype "fp8" \
    > ./deepseek-r1-bench/${datetime}.log 2>&1

