#!/usr/bin/env python3
"""Quick pool mode validation: 32 simple prompts, check output is not garbage."""
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("BATCHGEN_ROOT", str(Path(__file__).resolve().parents[1])))
from batchgen.batchgen_client import BatchGenHttpClient

BASE_URL = os.environ.get("BATCHGEN_URL", "http://localhost:10900")
MODEL = os.environ.get("BATCHGEN_MODEL", "openai/gpt-oss-120b")

QUESTIONS = [
    ("What is the capital of France?", "(A) Berlin (B) Madrid (C) Paris (D) Rome", "C"),
    ("Which element has atomic number 1?", "(A) Helium (B) Hydrogen (C) Lithium (D) Oxygen", "B"),
    ("What year did World War II end?", "(A) 1943 (B) 1944 (C) 1945 (D) 1946", "C"),
    ("What is 2 + 2?", "(A) 3 (B) 4 (C) 5 (D) 6", "B"),
    ("Which planet is closest to the Sun?", "(A) Venus (B) Earth (C) Mercury (D) Mars", "C"),
    ("What is the chemical symbol for water?", "(A) CO2 (B) H2O (C) NaCl (D) O2", "B"),
    ("Who wrote Romeo and Juliet?", "(A) Dickens (B) Shakespeare (C) Twain (D) Austen", "B"),
    ("What is the speed of light in km/s?", "(A) 100000 (B) 200000 (C) 300000 (D) 400000", "C"),
]

def main():
    lines = []
    answers = {}
    for i in range(32):
        q, opts, answer = QUESTIONS[i % len(QUESTIONS)]
        prompt = f"Answer with just the letter in parentheses. {q}\n{opts}"
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0,
        }
        cid = f"test-{i}"
        lines.append(json.dumps({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))
        answers[cid] = answer

    path = "/tmp/pool_quick_test.jsonl"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Created {path} with {len(lines)} prompts")

    client = BatchGenHttpClient(base_url=BASE_URL)
    t0 = time.time()
    batch = client.submit_batch(
        input_file_path=path,
        endpoint="/v1/chat/completions",
        poll_interval=5.0,
        timeout=600,
        max_decoding_length=4096,
        temperature=0,
    )
    elapsed = time.time() - t0
    print(f"Batch completed in {elapsed:.1f}s: {batch.get('status', 'unknown')}")

    output_id = batch.get("output_file_id")
    if not output_id:
        print(f"FAIL: No output. Batch: {batch}")
        return

    content = client.download_file_content(output_id)
    outputs = [json.loads(l) for l in content.decode("utf-8").strip().split("\n") if l.strip()]
    print(f"Got {len(outputs)} outputs")

    correct = 0
    garbage = 0
    for o in outputs:
        cid = o.get("custom_id", "?")
        resp = o.get("response", {}).get("body", {})
        choices = resp.get("choices", [])
        text = choices[0]["message"]["content"][:500] if choices else ""

        # Check for garbage: Chinese chars, csharp, README, etc.
        is_garbage = any(k in text.lower() for k in ["csharp", "readme", "\u4e2d\u6587"])
        if is_garbage:
            garbage += 1
            print(f"  GARBAGE {cid}: {text[:100]}")
        else:
            expected = answers.get(cid, "?")
            is_correct = f"({expected})" in text or text.strip().startswith(expected)
            if is_correct:
                correct += 1
            print(f"  {'OK' if is_correct else 'WRONG'} {cid}: {text[:100]}")

    total = len(outputs)
    acc = correct / total * 100 if total > 0 else 0
    print(f"\n=== RESULTS ===")
    print(f"Total: {total}, Correct: {correct}, Garbage: {garbage}")
    print(f"Accuracy: {acc:.1f}%")
    print(f"PASS" if garbage == 0 and acc > 30 else "FAIL")

if __name__ == "__main__":
    main()
