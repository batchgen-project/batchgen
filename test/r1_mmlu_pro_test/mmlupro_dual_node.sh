#!bin/bash
datetime=$(date '+%Y-%m-%d-%H-%M-%S')
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
<<<<<<< HEAD
python /data2/tairan/workspace/BatchGen/test/r1_mmlu_pro_test/r1_mmlu_pro_test.py \
    --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
	--host_kv_cache_size 256 \
    --max_prompts 768 \
    --max_input_length 4096 \
    --max_decoding_length 4096 \
    --ATTN_MODE 3 \
    --cache_dir "/data2/tairan/modelscope/hub/models/deepseek-ai/DeepSeek-R1" \
=======
python <dir-to-r1_mmlu_pro_test.py> \
    --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
	--host_kv_cache_size 256 \
    --max_input_length 4096 \
    --max_decoding_length 2048 \
    --ATTN_MODE 3 \
    --cache_dir <dir to model checkpoint> \
>>>>>>> 46949f3339712055f2986fbddcf25deb0a2adb93
    --server_host "localhost" \
    --server_port 9090 \
	--dist_init_addr <dist-init-addr> \
	--nnodes 2 \
	--node_rank <node-rank> \
    > ./deepseek-r1-bench/${datetime}.log 2>&1