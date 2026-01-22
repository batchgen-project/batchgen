"""Simple completion test for GPT-OSS-120B.

Tests basic model functionality with simple prompts.
"""

import sys
from pathlib import Path

GPT_OSS_PATH = "/data2/tairan/workspace/gpt-oss"
sys.path.insert(0, GPT_OSS_PATH)

import torch
from dataclasses import fields
import json
import os

from gpt_oss.torch.model import TokenGenerator, Transformer, ModelConfig
from gpt_oss.torch.weights import Checkpoint
from gpt_oss.tokenizer import get_tokenizer


def patched_from_checkpoint(path: str, device: str = "cuda"):
    """Load model, filtering unknown config fields."""
    if not isinstance(device, torch.device):
        device = torch.device(device)

    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as f:
        json_config = json.load(f)

    valid_fields = {f.name for f in fields(ModelConfig)}
    filtered_config = {k: v for k, v in json_config.items() if k in valid_fields}
    config = ModelConfig(**filtered_config)

    print(f"Creating model on device: {device}")
    model = Transformer(config=config, device=device)
    model.eval()

    print(f"Loading checkpoint from: {path}")
    checkpoint = Checkpoint(path, device)

    param_list = list(model.named_parameters())
    total_params = len(param_list)
    for idx, (name, param) in enumerate(param_list):
        if idx % 100 == 0:
            print(f"  Loading parameter {idx}/{total_params}...")
        loaded_tensor = checkpoint.get(name)
        param.data.copy_(loaded_tensor)

    print(f"Model loaded!")
    return model


def main():
    checkpoint_path = "/data2/tairan/modelscope/hub/models/openai/gpt-oss-120b/original"
    device = torch.device("cpu")

    # Patch the from_checkpoint method
    Transformer.from_checkpoint = staticmethod(patched_from_checkpoint)

    tokenizer = get_tokenizer()

    print("Loading model...")
    generator = TokenGenerator(checkpoint_path, device)

    # Simple test prompts - both raw and with chat template
    test_prompts = [
        # Raw completion (no chat template)
        ("Raw: Capital", "The capital of France is"),
        ("Raw: Math", "1 + 1 ="),
        ("Raw: Continue", "Once upon a time, there was a"),

        # With minimal chat template
        ("Chat: Simple Q", "<|start|>user<|message|>What is 2+2?<|end|><|start|>assistant"),

        # With system message
        ("Chat: With System",
         "<|start|>system<|message|>You are a helpful assistant.<|end|>"
         "<|start|>user<|message|>What is the capital of France?<|end|>"
         "<|start|>assistant"),
    ]

    stop_tokens = [199999, 200002, 200007]  # endoftext, return, end

    for name, prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")

        tokens = tokenizer.encode(prompt, allowed_special="all")
        print(f"Tokens: {len(tokens)}")

        print("Generating (max 30 tokens, temp=0)...")
        generated = []
        for token in generator.generate(tokens, stop_tokens, temperature=0.0, max_tokens=30):
            generated.append(token)
            print(tokenizer.decode([token]), end="", flush=True)

        print(f"\n\nGenerated tokens: {generated}")
        print(f"Full output: {tokenizer.decode(generated)}")


if __name__ == "__main__":
    main()
