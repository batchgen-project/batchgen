#!/usr/bin/env python3
"""Response Validation Checklist for BatchGen batch outputs.

Validates every sequence in a batch output JSONL against 4 checks:
  1. MMLU Accuracy — extract answer, compare to ground truth
  2. Coherency — unique tokens, dominant token ratio, minimum length
  3. Special Token Leakage — model-specific special tokens in visible output
  4. Gibberish & Repeating — repeated n-grams, EOS-only, control chars

Usage:
    python tests/response_checklist.py output.jsonl \
        --model openai/gpt-oss-120b \
        [--ground-truth mmlu_pro_test.parquet] \
        [--accuracy-threshold 50]

Can also be imported:
    from tests.response_checklist import run_checklist
    report = run_checklist("output.jsonl", model="openai/gpt-oss-120b")
"""
import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check 1: MMLU Accuracy (reuses pattern from r1_mmlu_pro_test.py)
# ---------------------------------------------------------------------------

def extract_prediction(model_output: str) -> Optional[str]:
    """Extract predicted letter answer (A-J) from model output."""
    # Strip thinking section if present
    think_end = model_output.find("</think>")
    text = model_output[think_end + len("</think>"):] if think_end != -1 else model_output

    # Also strip Harmony analysis channel if present
    harmony_final = re.search(
        r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|channel\|>|<\|return\|>|$)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if harmony_final:
        text = harmony_final.group(1)

    # Pattern: "the answer is (X)" or "Answer: X"
    for pattern in [
        r"[Tt]he\s+answer\s+is\s*\(?([A-J])\)?",
        r"[Aa]nswer:\s*\(?([A-J])\)?",
        r"\b([A-J])\)\s*$",  # Trailing "(X)" at end
        r"\\boxed\{([A-J])\}",
        r"\*\*([A-J])\*\*",
    ]:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()

    # Bare single letter at end of text
    stripped = text.strip()
    if stripped and stripped[-1] in "ABCDEFGHIJ" and (len(stripped) < 5 or stripped[-2] in " (\n"):
        return stripped[-1].upper()

    return None


# ---------------------------------------------------------------------------
# Check 2: Coherency
# ---------------------------------------------------------------------------

def check_coherency(text: str) -> Tuple[bool, str]:
    """Check output coherency: sufficient unique content, no dominant repetition."""
    if not text or not text.strip():
        return False, "empty output"

    words = text.split()
    if len(words) < 3:
        return False, f"only {len(words)} words"

    unique_words = set(words)
    if len(unique_words) < 5 and len(words) > 20:
        return False, f"only {len(unique_words)} unique words in {len(words)} total"

    # Dominant word ratio (excluding common words)
    counts = Counter(words)
    most_common_word, most_common_count = counts.most_common(1)[0]
    if len(words) > 20 and most_common_count / len(words) > 0.4:
        return False, f"word '{most_common_word}' is {most_common_count}/{len(words)} ({100*most_common_count/len(words):.0f}%)"

    return True, "ok"


# ---------------------------------------------------------------------------
# Check 3: Special Token Leakage
# ---------------------------------------------------------------------------

# Tokens that should NEVER appear in parsed output (post-parsing)
UNIVERSAL_LEAKED_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|pad|>", "<|eos|>",
]

# Model-specific tokens that are expected in RAW output but not in parsed content
MODEL_SPECIFIC_LEAKED = {
    "openai/gpt-oss-120b": [],  # Harmony tokens are in raw output, OK
    "moonshotai/Kimi-K2.5": ["<|im_start|>", "<|im_end|>"],
    "deepseek": ["<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>"],
}


def check_special_token_leakage(text: str, model: str = "") -> Tuple[bool, str]:
    """Check for leaked special tokens in visible output."""
    found = []
    for token in UNIVERSAL_LEAKED_TOKENS:
        if token in text:
            found.append(token)

    model_tokens = MODEL_SPECIFIC_LEAKED.get(model, [])
    for token in model_tokens:
        if token in text:
            found.append(token)

    if found:
        return False, f"leaked: {found}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Check 4: Gibberish & Repeating Detection
# ---------------------------------------------------------------------------

def check_repeating(text: str) -> Tuple[bool, str]:
    """Detect repeated n-gram patterns (3-10 words repeated 5+ times consecutively)."""
    words = text.split()
    if len(words) < 15:
        return True, "ok"  # Too short to detect patterns

    for n in range(3, min(11, len(words) // 5 + 1)):
        i = 0
        while i + n * 5 <= len(words):
            pattern = tuple(words[i:i + n])
            repeats = 0
            j = i
            while j + n <= len(words) and tuple(words[j:j + n]) == pattern:
                repeats += 1
                j += n
            if repeats >= 5:
                snippet = " ".join(pattern)
                if len(snippet) > 60:
                    snippet = snippet[:60] + "..."
                return False, f"'{snippet}' repeated {repeats}x at word {i}"
            i += 1
    return True, "ok"


def check_gibberish(text: str) -> Tuple[bool, str]:
    """Detect gibberish: empty, EOS-only, control chars, etc."""
    stripped = text.strip()
    if not stripped:
        return False, "empty output"

    # EOS-only output
    if stripped.replace("<|endoftext|>", "").strip() == "":
        return False, "EOS-only output"

    # Control character ratio
    control = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r")
    if len(text) > 0 and control / len(text) > 0.1:
        return False, f"{control} control chars in {len(text)} chars ({100*control/len(text):.0f}%)"

    return True, "ok"


# ---------------------------------------------------------------------------
# Per-sequence result
# ---------------------------------------------------------------------------

@dataclass
class SeqResult:
    custom_id: str
    text: str
    completion_tokens: int = 0
    # Check results: (passed, detail)
    accuracy: Tuple[bool, str] = (True, "no ground truth")
    coherency: Tuple[bool, str] = (True, "ok")
    special_tokens: Tuple[bool, str] = (True, "ok")
    gibberish: Tuple[bool, str] = (True, "ok")
    repeating: Tuple[bool, str] = (True, "ok")

    @property
    def all_passed(self) -> bool:
        return (self.coherency[0] and self.special_tokens[0]
                and self.gibberish[0] and self.repeating[0])


@dataclass
class ChecklistReport:
    model: str
    total: int = 0
    accuracy_correct: int = 0
    accuracy_total: int = 0  # sequences with ground truth
    coherency_pass: int = 0
    special_tokens_pass: int = 0
    gibberish_pass: int = 0
    repeating_pass: int = 0
    sequences: List[SeqResult] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)

    @property
    def accuracy_pct(self) -> float:
        return 100.0 * self.accuracy_correct / self.accuracy_total if self.accuracy_total > 0 else 0.0

    def print_report(self, accuracy_threshold: float = 0.0):
        n = self.total
        print(f"\n{'='*60}")
        print(f"RESPONSE CHECKLIST | Model: {self.model} | Sequences: {n}")
        print(f"{'='*60}")

        # Check 1
        if self.accuracy_total > 0:
            acc_ok = self.accuracy_pct >= accuracy_threshold
            print(f"CHECK 1 — MMLU Accuracy:     {self.accuracy_correct}/{self.accuracy_total} "
                  f"({self.accuracy_pct:.1f}%) {'PASS' if acc_ok else 'FAIL'} "
                  f"[threshold: >{accuracy_threshold:.0f}%]")
        else:
            print(f"CHECK 1 — MMLU Accuracy:     SKIPPED (no ground truth)")

        # Check 2-4
        for name, count in [
            ("Coherency", self.coherency_pass),
            ("Special Tokens", self.special_tokens_pass),
            ("Gibberish/Repeat", min(self.gibberish_pass, self.repeating_pass)),
        ]:
            status = "PASS" if count == n else "FAIL"
            print(f"CHECK {'234'['Coherency Special Gibberish'.split().index(name.split()[0])]} — {name:20s} {count}/{n} {status}")

        # Overall
        checks_passed = sum([
            self.accuracy_total == 0 or self.accuracy_pct >= accuracy_threshold,
            self.coherency_pass == n,
            self.special_tokens_pass == n,
            self.gibberish_pass == n and self.repeating_pass == n,
        ])
        overall = "PASS" if checks_passed == 4 else "FAIL"
        print(f"\nOVERALL: {overall} ({checks_passed}/4 checks passed)")

        # Flagged sequences
        flagged = [s for s in self.sequences if not s.all_passed or not s.accuracy[0]]
        if flagged:
            print(f"\nFlagged sequences ({len(flagged)}):")
            for s in flagged[:20]:
                reasons = []
                if not s.accuracy[0]:
                    reasons.append(f"WRONG({s.accuracy[1]})")
                if not s.coherency[0]:
                    reasons.append(f"INCOHERENT({s.coherency[1]})")
                if not s.special_tokens[0]:
                    reasons.append(f"SPECIAL_TOKEN({s.special_tokens[1]})")
                if not s.gibberish[0]:
                    reasons.append(f"GIBBERISH({s.gibberish[1]})")
                if not s.repeating[0]:
                    reasons.append(f"REPEATING({s.repeating[1]})")
                print(f"  {s.custom_id}: {', '.join(reasons)}")
            if len(flagged) > 20:
                print(f"  ... and {len(flagged) - 20} more")

        # Token stats
        if self.token_counts:
            tc = sorted(self.token_counts)
            n_tc = len(tc)
            print(f"\nToken stats:")
            print(f"  Mean: {sum(tc)/n_tc:.1f} | Median: {tc[n_tc//2]} | "
                  f"Min: {tc[0]} | Max: {tc[-1]}")
            print(f"  P5: {tc[int(n_tc*0.05)]} | P95: {tc[int(n_tc*0.95)]} | "
                  f"P99: {tc[min(int(n_tc*0.99), n_tc-1)]}")
            short = sum(1 for t in tc if t < 200)
            long = sum(1 for t in tc if t > 50000)
            print(f"  Short (<200): {short} | Long (>50K): {long}")

        return overall == "PASS"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def load_ground_truth(parquet_path: str) -> Dict[int, str]:
    """Load MMLU Pro ground truth: index → answer letter."""
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        gt = {}
        for idx, row in df.iterrows():
            gt[idx] = chr(65 + row["answer_index"])  # 0→A, 1→B, ...
        return gt
    except Exception as e:
        logger.warning(f"Could not load ground truth: {e}")
        return {}


def run_checklist(
    jsonl_path: str,
    model: str = "",
    ground_truth: Optional[Dict[int, str]] = None,
    accuracy_threshold: float = 0.0,
) -> ChecklistReport:
    """Run all checks on a batch output JSONL file."""
    report = ChecklistReport(model=model)

    with open(jsonl_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        d = json.loads(line)
        custom_id = d.get("custom_id", "?")
        resp = d.get("response", {}).get("body", {})
        choices = resp.get("choices", [])
        usage = resp.get("usage", {})

        # Handle error responses
        error = d.get("response", {}).get("error")
        if error:
            text = ""
            tokens = 0
        elif choices:
            text = choices[0].get("message", {}).get("content", "")
            tokens = usage.get("completion_tokens", 0)
        else:
            text = ""
            tokens = 0

        seq = SeqResult(custom_id=custom_id, text=text, completion_tokens=tokens)

        # Check 1: Accuracy (if ground truth available)
        if ground_truth:
            # Extract index from custom_id (e.g., "mmlu-5" → 5)
            idx_match = re.search(r"(\d+)", custom_id)
            if idx_match:
                idx = int(idx_match.group(1))
                expected = ground_truth.get(idx)
                if expected:
                    predicted = extract_prediction(text)
                    if predicted == expected:
                        seq.accuracy = (True, f"correct ({expected})")
                        report.accuracy_correct += 1
                    else:
                        seq.accuracy = (False, f"predicted={predicted}, expected={expected}")
                    report.accuracy_total += 1

        # Check 2: Coherency
        if text:
            seq.coherency = check_coherency(text)
        else:
            seq.coherency = (False, "no output text") if not error else (True, "error response")

        # Check 3: Special tokens
        if text:
            seq.special_tokens = check_special_token_leakage(text, model)
        else:
            seq.special_tokens = (True, "no text to check")

        # Check 4a: Gibberish
        if text:
            seq.gibberish = check_gibberish(text)
        else:
            seq.gibberish = (False, "no output") if not error else (True, "error response")

        # Check 4b: Repeating
        if text:
            seq.repeating = check_repeating(text)
        else:
            seq.repeating = (True, "no text to check")

        # Aggregate
        report.total += 1
        if seq.coherency[0]:
            report.coherency_pass += 1
        if seq.special_tokens[0]:
            report.special_tokens_pass += 1
        if seq.gibberish[0]:
            report.gibberish_pass += 1
        if seq.repeating[0]:
            report.repeating_pass += 1
        if tokens > 0:
            report.token_counts.append(tokens)
        report.sequences.append(seq)

    return report


def main():
    parser = argparse.ArgumentParser(description="Response Validation Checklist")
    parser.add_argument("jsonl", help="Batch output JSONL file")
    parser.add_argument("--model", default="", help="Model name for model-specific checks")
    parser.add_argument("--ground-truth", default=None, help="MMLU Pro parquet for accuracy scoring")
    parser.add_argument("--accuracy-threshold", type=float, default=0.0,
                        help="Minimum accuracy %% to pass (default: 0)")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth) if args.ground_truth else None
    report = run_checklist(args.jsonl, model=args.model, ground_truth=gt,
                           accuracy_threshold=args.accuracy_threshold)
    passed = report.print_report(accuracy_threshold=args.accuracy_threshold)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
