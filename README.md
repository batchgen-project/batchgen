<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/8e70c5d1-6c3a-4507-b3b2-821ebf127989">
    <img src="https://github.com/user-attachments/assets/5587e43e-a2ef-4dde-a84c-365c31f284f8" width=55%>
  </picture>
</p>


<div align="center">
 <h3> High-throughput Offline Inference for MoE Models with Limited GPU Memory</h3>
  <strong><a href="#Performance">Performance</a> | <a href="#Installation"> Installation</a> | <a href="#Quick-Start">Quick Start </a> </strong>
</div>


# About
BatchGen is an efficient serving engine optimized specifically for **Mixture-of-Expert(MoE)** based large language models. It is tailored for bulk **offline inference** tasks and **limited GPU resources**. It enables low cost serving for latency-insensitive applications.

**Core Features**

- **Module-Based Batching**: A fine-grained batching strategy ensures consistently high GPU utilization throughout every forward pass.
- **Efficient Data Swapping Engine**: Supports inference of large-scale models (e.g., DeepSeek-R1) on constrained hardware setups such as single NVIDIA A5000 or RTX 4090 GPUs, aggressively maximizing overlap between computation and memory transfers to achieve optimal efficiency.
- **Tailored Offloading and Parallel Strategy**: Different parallel strategies, model weights offloaidng and KV-Cache offloading are applied to different models and hardware settings. 


# Application Scenarios
- MoE model evaluation.
- Company deployed LLM workflow for raw data formation.
- Latency-insensitive bulk inference tasks. Such as large batch inference launched in valley period.
- Deep-research applications. Deliver high-quality results overnight.



# Supported Models
- **DeepSeek-R1/V3-671B. FULL Precision.**

# Supported Hardware
Hooper and Ampere archtecture are supported. 

Recommended configurations for 8xH20, 8xA100 and 8xA5000 node are included in ./batchgen/configurations/


## Installation

We recommend installing BatchGen in a virtual environment. To install BatchGen, you can either install it from PyPI or build it from source.

### Hardware Requirements
```bash
Host Memory > Model Size
Disk Space >= 2 * Model Size
```
For example,  to efficiently serve DeepSeek-R1-6871B-FP8, 1TB Host memory is recommended. 1.5TB disk space is needed.


### Create conda environment.

```bash
conda create --name batchgen python=3.11
conda activate batchgen
```

### Dependencies installation
```bash
pip install flash-attn --no-build-isolation
```
For Hooper user, please install flash-attention 3 beta release refer to https://github.com/Dao-AILab/flash-attention
```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd ./flash-attention/hopper/
python setup.py install
```
For Hooper user, please install FlashMLA refer to https://github.com/deepseek-ai/FlashMLA/tree/main
```bash
git clone git@github.com:deepseek-ai/FlashMLA.git
cd FlashMLA
python setup.py install
```

For Hooper user, please install DeepGEMM refer to https://github.com/deepseek-ai/DeepGEMM
```bash
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM
cat develop.sh
./develop.sh
cat install.sh
./install.sh
```

Currently, BatchGen depends on torch==2.70+cu128
```bash
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### Install BatchGen from codebase
```bash
git clone git@github.com:EfficientMoE/BatchGen.git
cd BatchGen
pip install -e .
```

## Quick Start
For 2-8*H20 Serving, please refer to ```./docs/Serving_with_two_nodes.```

Following is an example for one node or within one node.
BatchGen seamlessly integrates with Huggingface environment. Start inference with Huggingface checkpoint name.
### Example usage of serving DeepSeek-R1 on 8*H20 node
#### Start Server
```bash
python -m batchgen.parameter_server \
    --model deepseek-ai/DeepSeek-R1 \
    --cache-dir "..." 
```
Please provide --cache-dir as your model checkpoint if you are not using hugging face default model checkpoint directory.

#### Submit Batch
Please prepare the batch in python and call the client.
```python
import logging
import os
from transformers import AutoTokenizer
from batchgen.engine import batchgen
import numpy as np
import datasets

if __name__ == "__main__":
    """
        Step 1: Task Configs
    """
    hugging_face_checkpoint_name = "deepseek-ai/DeepSeek-R1"
    
    # The host memory reserved for KV-Cache. Adjust based on your hardware platform.
    host_kv_cache_size = 100 
    
    max_input_length = 13000
    max_decoding_length = 100
    engine_config_json_dir = "/BatchGen/configurations/DeepSeek-R1/engine_config_H20_8.json"
    # Change if there is port conflict.
    server_host = "localhost"
    server_port = "9090" 


    """
        Step 2: Prepare the input requests(list of string).
                Here we use huggingface dataset as an example.
    """
    benchmark_name = "THUDM/LongBench"
    num_requests = 2400

    task_names = [
        "2wikimqa", "2wikimqa_e", "dureader", "gov_report", "gov_report_e",
        "hotpotqa", "hotpotqa_e", "lcc", "lcc_e", "lsht",
        "multi_news", "multi_news_e", "multifieldqa_en", "multifieldqa_en_e", "multifieldqa_zh",
        "musique", "narrativeqa", "passage_count", "passage_count_e", "passage_retrieval_en",
        "passage_retrieval_en_e", "passage_retrieval_zh", "qasper", "qasper_e", "qmsum",
        "repobench-p", "repobench-p_e", "samsum", "samsum_e", "trec",
        "trec_e", "triviaqa", "triviaqa_e", "vcsum"
    ]
    queries = []
    for task_name in task_names:
        dataset = datasets.load_dataset(benchmark_name, task_name, split="test")
        for q in dataset["context"]:
            if len(q.split(" ")) >= max_input_length:
            # if len(q.split(" ")) >= 9000:
                queries.append(q)
                if len(queries) == num_requests:
                    break
        if len(queries) == num_requests:
            break


    # If number of queries is less than max_prompts, fill the rest by duplicating
    if len(queries) < num_requests:
        queries = queries * (num_requests // len(queries)) + queries[: num_requests % len(queries)]
    

    tokenizer = AutoTokenizer.from_pretrained(
        hugging_face_checkpoint, trust_remote_code=True
    )
    for prompt_idx in range(len(queries)):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": queries[prompt_idx]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        queries[prompt_idx] = text

    """
        Step 3: Launch BatchGen
    """    
    logging.info(f"Connecting to parameter server at {server_host}:{server_port}")
    logging.info(f"Using model {hugging_face_checkpoint}")
    
    # Run inference with our standalone parameter server
    answer_set = batchgen(
        huggingface_ckpt_name=hugging_face_checkpoint,
        queries=queries,
        max_input_length=max_input_length,
        max_decoding_length=max_decoding_length,
        device=[0,1,2,3,4,5,6,7],
        engine_config_json_dir = engine_config_json_dir,
        host_kv_cache_size=host_kv_cache_size,

        # Connect to our standalone parameter server
        parameter_server_host=server_host,
        parameter_server_port=server_port,
    )

    """
        Step 4: Print responses to the prompts.
    """
    def decode_to_eos(tokenizer, tokens):
        tokens_array = np.array(tokens)
        eos_positions = np.where(tokens_array == tokenizer.eos_token_id)[0]
        end_pos = eos_positions[0] if len(eos_positions) > 0 else len(tokens_array)
        return tokenizer.decode(tokens[:end_pos], skip_special_tokens=True)

    print_result = True
    if print_result:
        for idx in range(len(answer_set)):
            tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist()[0])
            print(f"Answer {idx}: {tmp_answer}")
            print("\n\n")

```
#### Running Inference

This command runs the script on selected GPUs.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \ 
python example.py
```



## Citation
```
@misc{xu2025moegenhighthroughputmoeinference,
      title={BatchGen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching},
      author={Tairan Xu and Leyang Xue and Zhan Lu and Adrian Jackson and Luo Mai},
      year={2025},
      eprint={2503.09716},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2503.09716},
}
```
