"""Comparison script: Run GPT-OSS-120B through OpenAI reference implementation.

This script tests the same MMLU Pro prompts through the original OpenAI
reference implementation to verify correctness.

Usage:
    python gpt_oss_reference_comparison.py --checkpoint /path/to/model --max_prompts 3
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch

# Add gpt-oss to path
GPT_OSS_ROOT = Path(__file__).parent.parent.parent.parent.parent / "gpt-oss"
sys.path.insert(0, str(GPT_OSS_ROOT))

from gpt_oss.torch.model import Transformer, TokenGenerator, ModelConfig
from gpt_oss.tokenizer import get_tokenizer

# GPT-OSS special tokens
GPT_OSS_SPECIAL_TOKENS = {
    "<|startoftext|>": 199998,
    "<|endoftext|>": 199999,
    "<|return|>": 200002,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
}


def apply_chat_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Apply GPT-OSS Harmony chat template.

    Template format:
    <|start|>system<|message|>{system_content}

    # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
    <|start|>user<|message|>{user_content}<|end|>
    <|start|>assistant  (if add_generation_prompt)
    """
    result = ""
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            result += f"<|start|>system<|message|>{content}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        elif role == "user":
            result += f"<|start|>user<|message|>{content}<|end|>"
        elif role == "assistant":
            result += f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"

    if add_generation_prompt:
        result += "<|start|>assistant"

    return result


def form_options(options: List[str]) -> str:
    """Format multiple choice options."""
    option_str = "Options are:\n"
    opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, letter in zip(options, opts):
        option_str += f"({letter}): {opt}\n"
    return option_str


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Reference Implementation Comparison")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--max_prompts", type=int, default=3, help="Number of prompts to test")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = get_tokenizer()

    # Load MMLU Pro dataset
    r1_test_dir = Path(__file__).parent.parent / "r1_mmlu_pro_test"
    print(f"Loading dataset from {r1_test_dir}")
    dataset = pd.read_parquet(r1_test_dir / "mmlu_pro_test.parquet")
    validation_set = pd.read_parquet(r1_test_dir / "mmlu_pro_validation.parquet")

    if args.max_prompts > 0:
        dataset = dataset.head(args.max_prompts)

    # Build few-shot prompts per category
    categories = [
        "computer science", "math", "chemistry", "engineering", "law",
        "biology", "health", "physics", "business", "philosophy",
        "economics", "other", "psychology", "history",
    ]
    prompts = {c: "" for c in categories}
    for _, row in validation_set.iterrows():
        prompts[row["category"]] += (
            "Q:" + " " + row["question"] + "\n"
            + form_options(row["options"]) + "\n"
            + row["cot_content"] + "\n\n"
        )

    # Build queries
    queries: List[str] = []
    for _, entry in dataset.iterrows():
        prefix = prompts[entry["category"]]
        prompt = (
            "Please read the following 5 examples: \n"
            + prefix
            + "Please answer the following question: \n"
            + "Q: " + entry["question"] + "\n"
            + form_options(entry["options"]) + "\n"
        )
        queries.append(prompt)

    print(f"Loaded {len(queries)} queries")

    # Build full prompts with chat template
    system_message = (
        "You are a knowledge expert. Answer the multi-choice question and provide "
        "your final answer in the format 'The answer is (X)' where X is A, B, C, D, E, F, G, H, I, or J."
    )

    full_prompts = []
    for query in queries:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
        full_prompt = apply_chat_template(messages, add_generation_prompt=True)
        full_prompts.append(full_prompt)

    # Print first prompt for verification
    print("\n" + "="*80)
    print("FIRST PROMPT (with chat template):")
    print("="*80)
    print(full_prompts[0][:2000] + "..." if len(full_prompts[0]) > 2000 else full_prompts[0])
    print("="*80 + "\n")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    print("This may take a while for the 120B model...")

    generator = TokenGenerator(args.checkpoint, device)
    print("Model loaded successfully!\n")

    # Stop tokens
    stop_tokens = [
        GPT_OSS_SPECIAL_TOKENS["<|return|>"],
        GPT_OSS_SPECIAL_TOKENS["<|end|>"],
        GPT_OSS_SPECIAL_TOKENS["<|endoftext|>"],
    ]

    # Generate for each prompt
    for idx, (query, full_prompt) in enumerate(zip(queries, full_prompts)):
        print(f"\n{'='*80}")
        print(f"QUERY {idx}:")
        print("="*80)
        print(query[:500] + "..." if len(query) > 500 else query)

        # Tokenize
        prompt_tokens = tokenizer.encode(full_prompt, allowed_special="all")
        print(f"\nPrompt tokens: {len(prompt_tokens)}")

        # Generate
        print(f"\nGenerating (max {args.max_tokens} tokens, temperature=0)...")
        generated_tokens = []
        for token in generator.generate(
            prompt_tokens,
            stop_tokens=stop_tokens,
            temperature=0.0,
            max_tokens=args.max_tokens,
        ):
            generated_tokens.append(token)
            # Print token as it's generated
            decoded = tokenizer.decode([token])
            print(decoded, end="", flush=True)

        print("\n")

        # Full decoded output
        full_output = tokenizer.decode(generated_tokens)
        print(f"Generated tokens: {generated_tokens}")
        print(f"Full output: {full_output}")

        # Ground truth
        ground_truth = dataset.iloc[idx]["answer"]
        print(f"\nGround truth: ({ground_truth})")
        print("="*80)


if __name__ == "__main__":
    main()
