# Example: Serving with two H20*8 nodes
## Start server on two nodes respectively:
```python
#Add the following line if you have problem in accessing Huggingface.co
export HF_ENDPOINT=https://hf-mirror.com
python -m batchgen.parameter_server --model deepseek-ai/DeepSeek-R1 --cache-dir "<dir-to-your-model-checkpoint>"
```

## Run tasks on two nodes
Please first copy the ```two_nodes_H20_benchmark.py``` to your working dir (outside /BatchGen directory) on two nodes respectively.

Then run following commands.
Cache-dir can be something like: ```**/modelscope/hub/models/deepseek-ai/DeepSeek-R1```
```python
echo "🚀 Launch Node 0"
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python "<dir-to-two_nodes_H20_benchmark.py>" \
    --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
	--host_kv_cache_size 192 \
    --max_prompts 768 \
    --max_input_length 13000 \
    --max_decoding_length 1000 \
    --ATTN_MODE 3 \
    --cache_dir "<dir-to-model-checkpoint>" \
    --server_host "localhost" \
    --server_port 10900 \
	--dist_init_addr "10.0.0.8:12335" \
	--nnodes 2 \
	--node_rank 0
```

```python
echo "🚀 Launch Node 1"
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python "<dir-to-two_nodes_H20_benchmark.py>" \
    --hugging_face_checkpoint "deepseek-ai/DeepSeek-R1" \
	--host_kv_cache_size 192 \
    --max_prompts 768 \
    --max_input_length 13000 \
    --max_decoding_length 1000 \
    --ATTN_MODE 3 \
    --cache_dir "<dir-to-model-checkpoint>" \
    --server_host "localhost" \
    --server_port 10900 \
	--dist_init_addr "10.0.0.8:12335" \
	--nnodes 2 \
	--node_rank 1
```

# Our example is based on Longbench. If you would like to switch dataset please modify the example code.
```
## Clean-up(Optional)
If the program terminated or killed without proper clean-up, you may need to manually clean the occupied pages before next start BatchGen server.
```bash
rm -f /dev/hugepages/*
```
By ```sudo sysctl -w vm.nr_hugepages=0```, we can revert to default page configurations.
