"""Prepare stories.jsonl for the nqa64k-distinct-v1 workload.

Pools the parquet conversion of deepmind/narrativeqa train+test splits,
groups rows by document, keeps every book with >= 1 question and enough
text to survive a 64k-token truncation (>= 260k chars as a cheap proxy,
final token check happens in the workload builder), and writes the longest
books first as {doc_id, text, questions} rows.  Target: >= 560 books so the
builder can discard tokenizer-level shortfalls and still fill 512 rows.
"""
import json
import os
import sys

OUT_DIR = sys.argv[1]
MIN_CHARS = 260_000
TARGET_BOOKS = 600
os.makedirs(OUT_DIR, exist_ok=True)

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
files = [
    f for f in api.list_repo_files(
        "deepmind/narrativeqa", repo_type="dataset", revision="refs/convert/parquet")
    if f.endswith(".parquet") and ("/train/" in f or "/test/" in f or "train-" in f or "test-" in f)
]
print(f"parquet files: {len(files)}")

import pyarrow.parquet as pq

books = {}
for rf in sorted(files):
    lp = hf_hub_download("deepmind/narrativeqa", rf, repo_type="dataset",
                         revision="refs/convert/parquet")
    t = pq.read_table(lp)
    for row in t.to_pylist():
        doc = row["document"]
        doc_id = doc["id"]
        b = books.setdefault(doc_id, {"doc_id": doc_id, "text": doc["text"], "questions": []})
        q = row["question"]["text"].strip()
        if q and q not in b["questions"]:
            b["questions"].append(q)

eligible = [b for b in books.values() if len(b["text"]) >= MIN_CHARS and b["questions"]]
eligible.sort(key=lambda b: len(b["text"]), reverse=True)
keep = eligible[:TARGET_BOOKS]
print(f"books total={len(books)} eligible>=260k chars={len(eligible)} kept={len(keep)}")
with open(os.path.join(OUT_DIR, "stories_distinct.jsonl"), "w") as f:
    for b in keep:
        f.write(json.dumps(b, ensure_ascii=False) + "\n")
print("written", os.path.join(OUT_DIR, "stories_distinct.jsonl"))
