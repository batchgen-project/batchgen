# ---------------------------------------------------------------------------- #
#  MoE-Gen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

import logging
import os

import datasets
import numpy as np
from transformers import AutoTokenizer

from moe_gen.engine import moe_gen

if __name__ == "__main__":
    """
        Supported Models:
        1. DeepSeek-V2 Architecture:
            * deepseek-ai/DeepSeek-V2-Lite-Chat
            * deepseek-ai/DeepSeek-V2
            * deepseek-ai/DeepSeek-V2.5
            * deepseek-ai/DeepSeek-V2.5-1210
            * deepseek-ai/DeepSeek-Coder-V2-Instruct
        2. DeepSeek-V3 Architecture:
            * deepseek-ai/DeepSeek-R1
            * deepseek-ai/DeepSeek-V3
        3. Mixtral:
            * mistralai/Mixtral-8x7B-Instruct-v0.1
            * mistralai/Mixtral-8x22B-Instruct-v0.1
    """

    """
        Step 1: Select Model.
        We recommend using the DeepSeek-V2-Lite-Chat model to get started.
        Feel free to experiment with other models by changing the hugging_face_checkpoint variable with above supported models.
    """
    # hugging_face_checkpoint = "/root/.cache/huggingface/hub/models/deepseek-ai/DeepSeek-R1"
    # hugging_face_checkpoint = "deepseek-ai/DeepSeek-R1"
    hugging_face_checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"


    """
        Step 2: Load dataset and apply chat template(prompt engineering).
    """

    benchmark_name = "Muennighoff/flan"
    dataset = datasets.load_dataset(benchmark_name, split="train")

    max_prompts = 1
    queries = dataset["inputs"][:max_prompts]

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
    logging.info(f"Number of prompts: {len(queries)}")

    """
        Step 3: Launch MoE-Gen
    """
    answer_set = moe_gen(
        huggingface_ckpt_name=hugging_face_checkpoint,
        queries=queries,
        max_input_length=512,
        max_decoding_length=10,
        device=[0],
        host_kv_cache_size=10,
    )

    """
        Step 4: Print responses to the prompts.
    """

    def decode_to_eos(tokenizer, tokens):
        tokens_array = np.array(tokens)
        eos_positions = np.where(tokens_array == tokenizer.eos_token_id)[0]
        end_pos = (
            eos_positions[0] if len(eos_positions) > 0 else len(tokens_array)
        )
        return tokenizer.decode(tokens[:end_pos], skip_special_tokens=True)

    print_result = True
    if print_result:
        for idx in range(len(answer_set)):
            tmp_answer = decode_to_eos(tokenizer, answer_set[idx].tolist()[0])
            print(f"Prompt {idx}: {queries[idx]}")
            print(f"Answer {idx}: {tmp_answer}")
            print("\n\n")
