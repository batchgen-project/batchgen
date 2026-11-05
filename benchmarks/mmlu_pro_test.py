#!/usr/bin/env python3
"""Unified MMLU-Pro benchmark for all BatchGen models.

Supports: glm5, kimi-k25, kimi-k3, gpt-oss, minimax, deepseek-v4-flash
Uses BatchGen Batch API (upload JSONL → poll → download results → score).

Usage:
    python mmlu_pro_test.py --model-type glm5 --max-prompts 32 --max-decoding-length 512
    python mmlu_pro_test.py --model-type kimi-k25 --max-prompts 128 --max-decoding-length 4096
    python mmlu_pro_test.py --model-type deepseek-v4-flash --level L1
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid as _uuid_mod
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Resolve dataset path relative to BatchGen repo
_BATCHGEN_ROOT = os.environ.get(
    "BATCHGEN_ROOT",
    "/data2/tairan/workspace/BatchGen",  # H20 default
)
_TEST_PARQUET = os.path.join(_BATCHGEN_ROOT, "test/r1_mmlu_pro_test/mmlu_pro_test.parquet")
_VAL_PARQUET = os.path.join(_BATCHGEN_ROOT, "test/r1_mmlu_pro_test/mmlu_pro_validation.parquet")


def load_dataset(max_prompts: Optional[int] = None,
                 prompt_start: int = 0,
                 prompt_end: Optional[int] = None):
    """Load MMLU-Pro test + validation parquets.

    Prompt selection precedence: if prompt_start/prompt_end are set, they
    slice the test_df to [prompt_start:prompt_end]. Otherwise max_prompts
    takes the first N. The row IDs after slicing are preserved via the
    original pandas index, so `test_df.index[i]` is the absolute prompt
    id usable as a cross-run pairing key.
    """
    test_df = pd.read_parquet(_TEST_PARQUET)
    val_df = pd.read_parquet(_VAL_PARQUET)

    if prompt_start > 0 or prompt_end is not None:
        end = prompt_end if prompt_end is not None else len(test_df)
        test_df = test_df.iloc[prompt_start:end]
    elif max_prompts and max_prompts < len(test_df):
        test_df = test_df.head(max_prompts)

    return test_df, val_df


def form_options(options: list) -> str:
    letters = "ABCDEFGHIJ"
    lines = ["Options are:\n"]
    for opt, letter in zip(options, letters):
        lines.append(f"({letter}): {opt}\n")
    return "".join(lines)


def build_few_shot_prefix(val_df: pd.DataFrame, category: str, shots: int = 5) -> str:
    """Build few-shot examples for a category from validation set."""
    cat_examples = val_df[val_df["category"] == category].head(shots)
    parts = []
    for _, row in cat_examples.iterrows():
        cot = row.get("cot_content", "")
        if isinstance(cot, str) and cot.startswith("A: "):
            cot = cot[3:]
        parts.append(
            f"Q: {row['question']}\n"
            + form_options(row["options"])
            + f"{cot}\n"
            + f"The answer is ({row['answer']}).\n\n"
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Model-specific configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "glm5": {
        "system_prompt": (
            "You are an expert at answering multiple-choice questions. "
            "Follow the examples provided, reason step by step, then give "
            "your final answer in the format: The answer is (X)."
        ),
        "query_template": "{prefix}Q: {question}\n{options}A:",
        "default_thinking": True,
        "thinking_field": "enable_thinking",
    },
    "kimi-k25": {
        "system_prompt": (
            "You are a knowledge expert, you are supposed to answer the "
            "multi-choice question to derive your final answer as "
            "`The answer is ...`. Please follow the following examples and "
            "strictly give the answer with format "
            "'the answer is (A/B/C/D/E/F/G/H/I/J)'."
        ),
        "query_template": (
            "Please read the following 5 examples: \n{prefix}"
            "Please answer the following question: \n"
            "Q: {question}\n{options}\n"
        ),
        "default_thinking": False,
        "thinking_field": None,
    },
    "kimi-k3": {
        "system_prompt": (
            "You are a knowledge expert, you are supposed to answer the "
            "multi-choice question to derive your final answer as "
            "`The answer is ...`. Please follow the following examples and "
            "strictly give the answer with format "
            "'the answer is (A/B/C/D/E/F/G/H/I/J)'."
        ),
        # No example count in the text: decode workloads use 2-, 3- and 5-shot
        # prefixes with this template.
        "query_template": (
            "Please read the following examples: \n{prefix}"
            "Please answer the following question: \n"
            "Q: {question}\n{options}\n"
        ),
        # K3 thinks by default (HuggingFace parity); BatchGen's OpenAI seam
        # forwards enable_thinking to the XTML renderer.
        "default_thinking": True,
        "thinking_field": "enable_thinking",
    },
    "gpt-oss": {
        "system_prompt": (
            "You are a knowledge expert. Answer the multi-choice question "
            "and provide your final answer in the format 'The answer is (X)' "
            "where X is A, B, C, D, E, F, G, H, I, or J."
        ),
        "query_template": (
            "The following are example questions with step-by-step solutions:\n\n"
            "{prefix}Now answer the following question:\n"
            "Q: {question}\n{options}"
        ),
        "default_thinking": False,
        "thinking_field": None,
        "extra_body": {"reasoning_effort": "low"},
    },
    "minimax": {
        "system_prompt": (
            "You are a knowledge expert, you are supposed to answer the "
            "multi-choice question to derive your final answer as "
            "`The answer is ...`. Please follow the following examples and "
            "strictly give the answer with format "
            "'the answer is (A/B/C/D/E/F/G/H/I/J)'."
        ),
        "query_template": (
            "Please read the following 5 examples: \n{prefix}"
            "Please answer the following question: \n"
            "Q: {question}\n{options}\n"
        ),
        "default_thinking": False,
        "thinking_field": None,
    },
    "deepseek-v4-flash": {
        "system_prompt": (
            "You are a knowledge expert. Solve each multiple-choice question "
            "carefully, then provide the final answer in exactly this format: "
            "The answer is (X)."
        ),
        "query_template": (
            "Please read the following 5 examples:\n{prefix}"
            "Please answer the following question:\n"
            "Q: {question}\n{options}\n"
        ),
        "default_thinking": True,
        "thinking_field": "enable_thinking",
    },
}


# ---------------------------------------------------------------------------
# Answer extraction (model-specific parsing + shared regex)
# ---------------------------------------------------------------------------

# Shared answer extraction patterns (10 patterns, ordered by specificity)
_ANSWER_PATTERNS = [
    r"(?i)\b(?:the\s+)?answer\s+is\s*\(?([ABCDEFGHIJ])\)?",
    r"(?i)(?:\*{1,2}|_{1,2})?Answer[s]?\s*[:\-\u2013]?(?:\*{1,2}|_{1,2})?\s*\(?([ABCDEFGHIJ])\)?",
    r"(?i)correct answer is \(?([ABCDEFGHIJ])\)?",
    r"(?i)\b(?:Option|Choice)\b\s*[:\-\u2013]?\s*([ABCDEFGHIJ])\b",
    r"\\boxed\{[^}]*?([ABCDEFGHIJ])[^}]*\}",
    r"(?<![A-Za-z0-9])[\(\[]\s*([ABCDEFGHIJ])\s*[\)\]](?![A-Za-z0-9])",
    r"(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCDEFGHIJ])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])",
    r"(?:^|\s)\(?([ABCDEFGHIJ])\)?\s*$",
    r"(?:^|\s)([ABCDEFGHIJ])[\.\:]",
    r"^\s*([ABCDEFGHIJ])\s*$",
]


def _extract_letter(text: str) -> Optional[str]:
    """Apply shared regex patterns to extract answer letter."""
    for pattern in _ANSWER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def parse_glm5(text: str) -> Tuple[str, Optional[str]]:
    """Parse GLM-5 output: extract thinking, then answer letter."""
    answer_text = text
    if "<think>" in text:
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            answer_text = text[m.end():].strip()
        else:
            answer_text = ""
    return answer_text, _extract_letter(answer_text if answer_text else text)


def parse_r1(text: str) -> Tuple[str, Optional[str]]:
    """Parse R1/Kimi-K2.5 output: search after </think> tag."""
    pos = text.find("</think>")
    search = text[pos + len("</think>"):] if pos != -1 else text
    return search, _extract_letter(search)


_K3_RESPONSE_OPEN = "<|open|>response<|sep|>"


def parse_k3(text: str) -> Tuple[str, Optional[str]]:
    """Parse Kimi-K3 output.  With --parse-thinking the server returns only the
    response channel; without it the content is raw XTML, so search after the
    response channel's opening tag."""
    pos = text.find(_K3_RESPONSE_OPEN)
    search = text[pos + len(_K3_RESPONSE_OPEN):] if pos != -1 else text
    return search, _extract_letter(search)


def parse_gptoss(text: str) -> Tuple[str, Optional[str]]:
    """Parse GPT-OSS Harmony format: extract from 'final' channel."""
    if "<|channel|>" not in text:
        return text, _extract_letter(text)

    final_parts = []
    pattern = r"<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|channel\|>|<\|end\|>|<\|return\|>|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    for channel, content in matches:
        if channel == "final":
            final_parts.append(content.strip())

    if not final_parts and not matches:
        idx = text.rfind("<|message|>")
        if idx != -1:
            content = text[idx + len("<|message|>"):]
            for token in ["<|end|>", "<|return|>", "<|channel|>"]:
                if token in content:
                    content = content[:content.find(token)]
            final_parts.append(content.strip())

    search = "\n".join(final_parts) if final_parts else text
    return search, _extract_letter(search)


PARSERS = {
    "glm5": parse_glm5,
    "kimi-k25": parse_r1,
    "kimi-k3": parse_k3,
    "gpt-oss": parse_gptoss,
    "minimax": parse_r1,
    "deepseek-v4-flash": parse_r1,
}


# ---------------------------------------------------------------------------
# Coherence heuristic
# ---------------------------------------------------------------------------

def check_coherence(outputs: List[str]) -> Dict[str, Any]:
    """Check outputs for repetition, short responses, garbled text."""
    flags = []
    for i, text in enumerate(outputs):
        issues = []
        tokens = text.split()

        # Short response
        if len(tokens) < 10:
            issues.append("short")

        # Repeated 4-grams
        if len(tokens) >= 20:
            ngrams = [" ".join(tokens[j:j+4]) for j in range(len(tokens) - 3)]
            counts = Counter(ngrams)
            max_repeat = max(counts.values()) if counts else 0
            if max_repeat >= 5:
                issues.append("repeat")

        # High non-ASCII ratio (garbled)
        if len(text) > 20:
            non_ascii = sum(1 for c in text if ord(c) > 127)
            if non_ascii / len(text) > 0.5:
                issues.append("garbled")

        if issues:
            flags.append({"idx": i, "issues": issues, "preview": text[:200]})

    n_ok = len(outputs) - len(flags)
    issue_counts = Counter(issue for f in flags for issue in f["issues"])
    summary_parts = [f"{k}: {v}" for k, v in issue_counts.items()]
    summary = f"COHERENCE: {n_ok}/{len(outputs)} OK"
    if flags:
        summary += f", {len(flags)} flagged ({', '.join(summary_parts)})"

    return {"summary": summary, "flags": flags, "n_ok": n_ok}


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def build_batch_jsonl(
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_type: str,
    model_name: str,
    max_decoding_length: int,
    enable_thinking: Optional[bool],
    temperature: Optional[float],
    per_request_sampling: bool = False,
    max_completion_tokens: Optional[int] = None,
    ignore_eos: bool = False,
    prompt_tokens: Optional[int] = None,
    tokenizer: Any = None,
    request_manifest: Optional[str] = None,
    workload_scope: str = "normal_client",
    wave_size: Optional[int] = None,
    workload_profile: Optional[str] = None,
) -> str:
    """Build JSONL input for batch API. Returns path to temp file."""
    cfg = MODEL_CONFIGS[model_type]
    few_shot_cache = {}

    # Fixed-shape records must be reproducible between baseline and candidate.
    # Ordinary clients retain a random prefix because they may submit multiple
    # batches to one server lifetime.
    run_id = "fixed-workload" if prompt_tokens is not None else _uuid_mod.uuid4().hex[:8]

    lines = []
    actual_prompt_tokens: list[int] = []
    for idx, row in test_df.iterrows():
        cat = row["category"]
        if cat not in few_shot_cache:
            few_shot_cache[cat] = build_few_shot_prefix(val_df, cat)

        query = cfg["query_template"].format(
            prefix=few_shot_cache[cat],
            question=row["question"],
            options=form_options(row["options"]),
        )

        messages = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": query},
        ]
        template_options = {
            "model_type": model_type,
            "enable_thinking": enable_thinking,
            "extra_body": cfg.get("extra_body", {}),
        }
        if prompt_tokens is not None:
            if tokenizer is None:
                raise ValueError("--prompt-tokens requires --tokenizer")
            messages = _pad_messages_to_exact_length(
                messages, prompt_tokens, tokenizer, template_options
            )
            actual_prompt_tokens.append(
                _count_rendered_tokens(messages, tokenizer, template_options)
            )

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens or max_decoding_length,
        }
        if ignore_eos:
            body["ignore_eos"] = True

        # Model-specific body fields
        if enable_thinking is not None and cfg.get("thinking_field"):
            body[cfg["thinking_field"]] = enable_thinking
        if workload_profile:
            # A registered profile fixes sampling explicitly in every request;
            # absent fields would fall back to batch or server defaults.
            from scripts import workload_policy

            body.update(workload_policy.profile_definition(
                workload_policy.load_registry(), workload_profile
            ).get("sampling", {}))
        elif per_request_sampling:
            import random
            body["temperature"] = round(random.uniform(0.3, 1.0), 2)
            body["top_p"] = round(random.uniform(0.8, 1.0), 2)
        elif temperature is not None:
            body["temperature"] = temperature
        if "extra_body" in cfg:
            body.update(cfg["extra_body"])

        lines.append(json.dumps({
            "custom_id": f"mmlu-{run_id}-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))

    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="mmlu_batch_")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    if request_manifest:
        manifest = {
            "schema_version": "batchgen/mmlu-fixed-workload/v1",
            "model": model_name,
            "request_count": len(lines),
            "workload_scope": workload_scope,
            "prompt_tokens_target": prompt_tokens,
            "prompt_tokens_actual": actual_prompt_tokens,
            "prompt_tokens_all_equal": bool(actual_prompt_tokens) and len(set(actual_prompt_tokens)) == 1,
            "max_completion_tokens": max_completion_tokens or max_decoding_length,
            "ignore_eos": ignore_eos,
            "request_jsonl_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        if workload_profile:
            # Declares the registered workload profile this request file
            # claims; scripts/decode_run.py verifies the claim at record time.
            from scripts import workload_policy

            definition = workload_policy.profile_definition(
                workload_policy.load_registry(), workload_profile
            )
            if definition is None:
                raise ValueError(f"unknown workload profile {workload_profile!r}")
            manifest["profile"] = workload_profile
            manifest["profile_sha256"] = workload_policy.profile_digest(definition)
            manifest["sampling"] = definition.get("sampling", {})
        if workload_scope == "wave_matrix" and wave_size:
            manifest["wave_size"] = wave_size
            manifest["wave_count"] = len(lines) // wave_size
        Path(request_manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _render_messages(
    messages: list[dict[str, str]], template_options: Optional[dict[str, Any]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    options = template_options or {}
    rendered_messages = [dict(message) for message in messages]
    model_type = options.get("model_type")
    extra_body = options.get("extra_body", {})
    if model_type == "gpt-oss":
        reasoning_effort = extra_body.get("reasoning_effort", "low")
        system_index = next(
            (index for index, message in enumerate(rendered_messages)
             if message.get("role") == "system"),
            None,
        )
        prefix = f"Reasoning: {reasoning_effort}\n"
        if system_index is None:
            rendered_messages.insert(0, {"role": "system", "content": prefix})
        else:
            rendered_messages[system_index] = {
                **rendered_messages[system_index],
                "content": prefix + rendered_messages[system_index].get("content", ""),
            }
    kwargs: dict[str, Any] = {}
    thinking = options.get("enable_thinking")
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
        kwargs["thinking"] = thinking
    return rendered_messages, kwargs


def _count_rendered_tokens(
    messages: list[dict[str, str]],
    tokenizer: Any,
    template_options: Optional[dict[str, Any]] = None,
) -> int:
    """Count exactly what the chat template sends to the model."""
    messages, kwargs = _render_messages(messages, template_options)
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **kwargs
        )
        return len(ids)
    return len(tokenizer(messages[-1]["content"]).input_ids)


def _pad_messages_to_exact_length(
    messages: list[dict[str, str]],
    target_tokens: int,
    tokenizer: Any,
    template_options: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    """Append deterministic filler until the server-rendered prompt is exact.

    BatchGen uses the same tokenizer/chat-template path for the request.  We
    therefore measure the rendered message list, rather than guessing from
    the final user string.  A bounded search keeps workload generation out of
    the request/decode hot path and fails closed if a tokenizer cannot produce
    an exact length.
    """
    if target_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    base = [dict(message) for message in messages]
    current = _count_rendered_tokens(base, tokenizer, template_options)
    if current > target_tokens:
        raise ValueError(f"prompt has {current} tokens, target is {target_tokens}")
    if current == target_tokens:
        return base

    original = base[-1]["content"]
    marker = "\n[benchmark padding]"

    def candidate(repeats: int) -> list[dict[str, str]]:
        value = [dict(message) for message in base]
        value[-1]["content"] = original + marker + (" x" * repeats)
        return value

    lo, hi = 0, target_tokens - current + 128
    while lo < hi:
        mid = (lo + hi) // 2
        if _count_rendered_tokens(candidate(mid), tokenizer, template_options) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    for repeats in range(max(0, lo - 128), lo + 129):
        value = candidate(repeats)
        if _count_rendered_tokens(value, tokenizer, template_options) == target_tokens:
            return value
    raise ValueError(
        f"could not construct exact rendered prompt length {target_tokens} "
        f"from base length {current}"
    )


def run_batch(
    base_url: str,
    input_file: str,
    poll_interval: float,
    timeout: Optional[float],
    max_decoding_length: int,
    temperature: Optional[float],
) -> List[Dict[str, Any]]:
    """Submit batch, wait, download results."""
    # Import BatchGen client
    sys.path.insert(0, os.path.join(_BATCHGEN_ROOT))
    from batchgen.batchgen_client import BatchGenHttpClient

    client = BatchGenHttpClient(base_url=base_url)

    batch = client.submit_batch(
        input_file_path=input_file,
        endpoint="/v1/chat/completions",
        poll_interval=poll_interval,
        timeout=timeout,
        max_decoding_length=max_decoding_length,
        temperature=temperature,
    )

    # Download results
    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        raise RuntimeError(f"Batch failed: {batch}")

    content = client.download_file_content(output_file_id)
    results = []
    for line in content.decode("utf-8").strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))

    # Sort by custom_id to match test_df order.
    # custom_id format: "mmlu-{run_id}-{idx}" — idx is the last component.
    results.sort(key=lambda r: int(r.get("custom_id", "mmlu-0-0").split("-")[-1]))
    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(
    results: List[Dict[str, Any]],
    test_df: pd.DataFrame,
    model_type: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Score batch results against ground truth."""
    parser = PARSERS[model_type]
    correct = 0
    total = 0
    extraction_failures = 0
    outputs = []
    incorrect = []

    for i, result in enumerate(results):
        response = result.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])
        content = ""
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
        outputs.append(content)

        if i >= len(test_df):
            break

        gt = test_df.iloc[i]["answer"]
        _, prediction = parser(content)

        if prediction is None:
            extraction_failures += 1
            prediction = "Z"

        total += 1
        if prediction == gt:
            correct += 1
        else:
            incorrect.append({
                "idx": i,
                "predicted": prediction,
                "ground_truth": gt,
                "failed_extract": prediction == "Z",
            })

        if verbose:
            status = "OK" if prediction == gt else "WRONG"
            logger.info(f"[{i}] {status} pred={prediction} gt={gt} | {content[:120]}...")

    accuracy = correct / total if total > 0 else 0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "extraction_failures": extraction_failures,
        "incorrect": incorrect,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified MMLU-Pro benchmark")
    parser.add_argument("--model-type", required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model type for parsing and prompt construction")
    parser.add_argument("--model-name", default=None,
                        help="Model name for API body (default: inferred from model-type)")
    parser.add_argument("--base-url", default="http://localhost:10900",
                        help="BatchGen server URL")
    parser.add_argument("--prompt-start", type=int, default=0,
                        help="First prompt index to include (inclusive). "
                             "Overrides --max-prompts when --prompt-end is set.")
    parser.add_argument("--prompt-end", type=int, default=None,
                        help="Last prompt index to include (exclusive). "
                             "Slices the dataset to [prompt-start:prompt-end].")
    parser.add_argument("--max-prompts", type=int, default=32,
                        help="Number of test samples")
    parser.add_argument("--max-decoding-length", type=int, default=512,
                        help="Max output tokens per request")
    parser.add_argument("--max-completion-tokens", type=int, default=None,
                        help="Explicit per-request max_completion_tokens cap")
    parser.add_argument("--offered-batch-size", type=int, default=None,
                        help="Decode wave size; each submitted wave has this many requests")
    parser.add_argument("--dataset-size", type=int, default=None,
                        help="Total fixed ordered MMLU rows; split into offered-batch-size waves")
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="Render every prompt to this exact token length")
    parser.add_argument("--tokenizer", default=None,
                        help="Explicit HF tokenizer fallback if the BatchGen registry is unavailable")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow custom tokenizer code when loading --tokenizer")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Keep decoding through EOS until max_completion_tokens")
    parser.add_argument("--enable-thinking", action="store_true", default=None,
                        help="Enable thinking mode (model-specific)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Explicitly disable thinking")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (None = server default)")
    parser.add_argument("--per-request-sampling", action="store_true",
                        help="Assign random temp (0.3-1.0) and top_p (0.8-1.0) per request")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between batch status checks")
    parser.add_argument("--timeout", type=float, default=6000.0,
                        help="Max seconds to wait for batch")
    parser.add_argument("--output-file", default=None,
                        help="Save raw outputs as JSONL")
    parser.add_argument("--request-manifest", default=None,
                        help="Save fixed-workload token counts and request hash")
    parser.add_argument("--request-jsonl", default=None,
                        help="Preserve the generated request JSONL at this path")
    parser.add_argument("--workload-profile", default=None,
                        help="Registered decode workload profile (configs/decode_workloads.json) "
                             "recorded in the request manifest")
    parser.add_argument("--coherence-check", action="store_true",
                        help="Enable coherence heuristic (repetition, garbled, short)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample results")
    parser.add_argument("--batchgen-root", default=None,
                        help="Override BATCHGEN_ROOT (path to BatchGen repo on server)")
    parser.add_argument("--level", default=None,
                        choices=["L1", "L2", "L3", "full"],
                        help="Test level (auto-sets max-prompts and max-decoding-length from configs/models.json)")
    parser.add_argument("--config", default=None,
                        help="Path to models.json config (default: configs/models.json next to this script)")

    args = parser.parse_args()

    # The central remote launcher supplies this path so every standard MMLU
    # experiment feeds the decode ledger without requiring model-specific
    # client edits.  An explicit CLI path remains the normal local behavior.
    if os.environ.get("BATCHGEN_OUTPUT_FILE"):
        args.output_file = os.environ["BATCHGEN_OUTPUT_FILE"]

    # Load test level from config if --level is specified
    if args.level:
        config_path = args.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "models.json")
        if not os.path.exists(config_path):
            parser.error(f"Config file not found: {config_path}")
        with open(config_path) as f:
            model_configs_json = json.load(f)
        if args.model_type not in model_configs_json:
            parser.error(f"Model {args.model_type} not found in {config_path}")
        level_config = model_configs_json[args.model_type].get("test_levels", {}).get(args.level)
        if level_config is None:
            parser.error(f"Level {args.level} not defined for {args.model_type} in {config_path}")
        # Override max_prompts and max_decoding_length from level config
        level_prompts = level_config.get("max_prompts")
        level_decode = level_config.get("max_decoding_length")
        if level_prompts is not None:
            args.max_prompts = level_prompts
        else:
            args.max_prompts = None  # None = all prompts
        if level_decode is not None:
            args.max_decoding_length = level_decode
        logger.info(f"Level {args.level}: max_prompts={args.max_prompts}, max_decoding_length={args.max_decoding_length}")

    if args.max_completion_tokens is not None:
        args.max_decoding_length = args.max_completion_tokens
    if args.offered_batch_size is not None:
        if args.offered_batch_size <= 0:
            parser.error("--offered-batch-size must be positive")
        args.max_prompts = args.dataset_size or args.offered_batch_size
    elif args.dataset_size is not None:
        if args.dataset_size <= 0:
            parser.error("--dataset-size must be positive")
        args.max_prompts = args.dataset_size

    # Override globals
    global _BATCHGEN_ROOT, _TEST_PARQUET, _VAL_PARQUET
    if args.batchgen_root:
        _BATCHGEN_ROOT = args.batchgen_root
        _TEST_PARQUET = os.path.join(_BATCHGEN_ROOT, "test/r1_mmlu_pro_test/mmlu_pro_test.parquet")
        _VAL_PARQUET = os.path.join(_BATCHGEN_ROOT, "test/r1_mmlu_pro_test/mmlu_pro_validation.parquet")

    # Resolve thinking
    enable_thinking = None
    if args.no_thinking:
        enable_thinking = False
    elif args.enable_thinking:
        enable_thinking = True
    elif MODEL_CONFIGS[args.model_type]["default_thinking"]:
        enable_thinking = True

    # Default model name
    model_name = args.model_name or {
        "glm5": "zai-org/GLM-5-FP8",
        "kimi-k25": "moonshotai/Kimi-K2.5",
        "kimi-k3": "moonshotai/Kimi-K3",
        "gpt-oss": "openai/gpt-oss-120b",
        "minimax": "MiniMaxAI/MiniMax-M2.5",
        "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
    }[args.model_type]

    tokenizer = None
    if args.prompt_tokens is not None:
        tokenizer_name = args.tokenizer or model_name
        try:
            # Prefer BatchGen's bundled registry: it is the same tokenizer and
            # chat-template implementation used by the server. HF loading is
            # an explicit fallback for models outside the registry.
            sys.path.insert(0, _BATCHGEN_ROOT)
            from batchgen.config.tokenizer_registry import load_tokenizer
            tokenizer = load_tokenizer(model_name)
        except Exception as exc:
            if not args.tokenizer:
                parser.error(f"unable to load BatchGen tokenizer {model_name!r}: {exc}")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, trust_remote_code=args.trust_remote_code
                )
            except Exception as fallback_exc:
                parser.error(
                    f"unable to load tokenizer {tokenizer_name!r}: {fallback_exc}"
                )

    logger.info(f"Model: {args.model_type} ({model_name})")
    if args.prompt_end is not None or args.prompt_start > 0:
        logger.info(f"Prompts: slice [{args.prompt_start}:{args.prompt_end}], "
                    f"MaxTokens: {args.max_decoding_length}")
    else:
        prompts_str = str(args.max_prompts) if args.max_prompts else "all"
        logger.info(f"Prompts: {prompts_str}, MaxTokens: {args.max_decoding_length}")
    logger.info(f"Thinking: {enable_thinking}, Temperature: {args.temperature}")

    # Load dataset
    test_df, val_df = load_dataset(
        args.max_prompts,
        prompt_start=args.prompt_start,
        prompt_end=args.prompt_end,
    )
    if args.offered_batch_size is not None:
        if len(test_df) < args.offered_batch_size:
            parser.error(
                f"selected {len(test_df)} prompts, fewer than wave size "
                f"{args.offered_batch_size}"
            )
        if len(test_df) % args.offered_batch_size != 0:
            parser.error(
                f"dataset size {len(test_df)} must be divisible by wave size "
                f"{args.offered_batch_size}"
            )
    logger.info(f"Loaded {len(test_df)} test samples, {len(val_df)} validation examples")

    # Build batch input
    t0 = time.time()
    if args.per_request_sampling:
        import random
        random.seed(42)
    input_file = build_batch_jsonl(
        test_df, val_df, args.model_type, model_name,
        args.max_decoding_length, enable_thinking, args.temperature,
        per_request_sampling=args.per_request_sampling,
        max_completion_tokens=args.max_completion_tokens,
        ignore_eos=args.ignore_eos,
        prompt_tokens=args.prompt_tokens,
        tokenizer=tokenizer,
        request_manifest=args.request_manifest,
        workload_scope=("wave_matrix" if args.offered_batch_size is not None else "normal_client"),
        wave_size=args.offered_batch_size,
        workload_profile=args.workload_profile,
    )
    logger.info(f"Built batch JSONL: {input_file}")
    if args.request_jsonl:
        Path(args.request_jsonl).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_file, args.request_jsonl)

    # Submit and wait. A fixed matrix uses the same ordered dataset at every
    # point, split into sequential waves so the scheduler sees exactly B
    # concurrent requests without requiring a new server-side batch cap.
    t0 = time.time()
    if args.offered_batch_size is None:
        results = run_batch(
            args.base_url, input_file, args.poll_interval,
            args.timeout, args.max_decoding_length, args.temperature,
        )
    else:
        lines = Path(input_file).read_text(encoding="utf-8").splitlines()
        results = []
        for start in range(0, len(lines), args.offered_batch_size):
            fd, wave_file = tempfile.mkstemp(suffix=".jsonl", prefix="mmlu_wave_")
            with os.fdopen(fd, "w", encoding="utf-8") as wave_handle:
                wave_handle.write("\n".join(lines[start:start + args.offered_batch_size]) + "\n")
            try:
                results.extend(run_batch(
                    args.base_url, wave_file, args.poll_interval,
                    args.timeout, args.max_decoding_length, args.temperature,
                ))
            finally:
                os.unlink(wave_file)
    wall_time = time.time() - t0
    logger.info(f"Batch completed in {wall_time:.1f}s, got {len(results)} results")

    # Clean up temp file
    os.unlink(input_file)

    # Score
    scores = score_results(results, test_df, args.model_type, args.verbose)

    # Print results
    print("\n" + "=" * 60)
    print(f"MMLU-Pro Results — {args.model_type}")
    print("=" * 60)
    print(f"Samples:    {scores['total']}")
    print(f"Correct:    {scores['correct']}")
    print(f"Accuracy:   {scores['accuracy']:.2%}")
    print(f"Extract fail: {scores['extraction_failures']}")
    print(f"Wall time:  {wall_time:.1f}s")

    print("Decode throughput: recorded from the server decode evidence")
    print("=" * 60)

    # Coherence check
    if args.coherence_check:
        coherence = check_coherence(scores["outputs"])
        print(f"\n{coherence['summary']}")
        if coherence["flags"]:
            print("\nFlagged samples:")
            for f in coherence["flags"][:5]:
                print(f"  [{f['idx']}] {f['issues']}: {f['preview'][:150]}...")

        # Always show 3-5 sample outputs for POIS to eyeball
        print("\n--- Sample outputs (first 5) ---")
        for i, out in enumerate(scores["outputs"][:5]):
            gt = test_df.iloc[i]["answer"] if i < len(test_df) else "?"
            _, pred = PARSERS[args.model_type](out)
            status = "OK" if pred == gt else "WRONG"
            print(f"\n[{i}] {status} (pred={pred}, gt={gt}):")
            # Show last 300 chars (answer portion)
            print(f"  ...{out[-300:]}" if len(out) > 300 else f"  {out}")

    # Save outputs
    if args.output_file:
        with open(args.output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Saved outputs to {args.output_file}")


if __name__ == "__main__":
    main()
