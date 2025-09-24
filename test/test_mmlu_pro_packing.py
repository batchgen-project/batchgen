import logging
import os
import time
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer

from batchgen.utils.packing import (
    PackedBatch,
    PrefillPacker,
    SequenceMapping,
    SequenceSegment,
)

logging.basicConfig(level=logging.INFO)


def form_options(options: list):
    option_str = "Options are:\n"
    opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, o in zip(options, opts):
        option_str += f"({o}): {opt}" + "\n"
    return option_str


def load_mmlu_pro_dataset():
    dataset = pd.read_parquet(
        os.path.join(
            os.path.dirname(__file__), "r1_mmlu_pro_test/mmlu_pro_test.parquet"
        )
    )
    validation_set = pd.read_parquet(
        os.path.join(
            os.path.dirname(__file__),
            "r1_mmlu_pro_test/mmlu_pro_validation.parquet",
        )
    )
    categories = [
        "computer science",
        "math",
        "chemistry",
        "engineering",
        "law",
        "biology",
        "health",
        "physics",
        "business",
        "philosophy",
        "economics",
        "other",
        "psychology",
        "history",
    ]

    # load 5-shot prompts for each category
    prompts = {c: "" for c in categories}
    # for d in validation_set:
    for idx, d in validation_set.iterrows():
        prompts[d["category"]] += (
            "Q:"
            + " "
            + d["question"]
            + "\n"
            + form_options(d["options"])
            + "\n"
            + d["cot_content"]
            + "\n\n"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1", trust_remote_code=True
    )

    queries = []
    for row_idx, entry in dataset.iterrows():
        prefix = prompts[entry["category"]]
        prompt = (
            "Please read the following 5 examples: \n"
            + prefix
            + "Please answer the following question: \n"
            + "Q: "
            + entry["question"]
            + "\n"
            + form_options(entry["options"])
            + "\n"
        )
        queries.append(prompt)

    for prompt_idx in range(len(queries)):
        messages = [
            {
                "role": "system",
                "content": "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`. Please follow the following examples and strictly give the answer with format 'the answer is (A/B/C/D/E/F/G/H/I/J)'.",
            },
            {"role": "user", "content": queries[prompt_idx]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        queries[prompt_idx] = text

    return queries, tokenizer


def pack_queries(
    queries: List[str], tokenizer, max_length: int, pad_token_id: int
):
    input_ids_list = [
        tokenizer.encode(q, add_special_tokens=False) for q in queries
    ]
    total_tokens = sum(len(ids) for ids in input_ids_list)
    logging.info(
        f"Total tokens before packing: {total_tokens}, batch size: {len(queries)}"
    )

    start = time.time()
    packed_batch = PrefillPacker.pack(
        input_ids_list, max_length, pad_token_id, include_sequence_mappings=True
    )
    end = time.time()
    logging.info(
        f"Packing took {end - start:.2f} seconds. Total tokens after packing: {packed_batch.attention_mask.sum().item()} / {len(packed_batch.input_ids) * max_length}"
    )
    return packed_batch


if __name__ == "__main__":
    queries, tokenizer = load_mmlu_pro_dataset()
    packed_batch = pack_queries(
        queries, tokenizer, max_length=8192, pad_token_id=tokenizer.pad_token_id
    )
    logging.info(
        f"Packed {len(queries)} queries into {len(packed_batch.input_ids)} batches. Efficiency: {PrefillPacker.compute_packing_efficiency(packed_batch):.2f}"
    )
