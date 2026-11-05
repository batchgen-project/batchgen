"""Prepare stories.jsonl for the nqa64k-grouped-v1 workload.

Downloads the parquet conversion of deepmind/narrativeqa (test split only),
groups rows by document, keeps books with >= 8 distinct questions and enough
text to survive a 64k-token truncation, and writes the 80 longest as
{doc_id, text, questions} rows.
"""
import json
import os
import sys

OUT_DIR = sys.argv[1]
os.makedirs(OUT_DIR, exist_ok=True)

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
files = [
    f for f in api.list_repo_files(
        "deepmind/narrativeqa", repo_type="dataset", revision="refs/convert/parquet"
    )
    if "test/" in f and f.endswith(".parquet")
]
if not files:
    raise SystemExit("no test parquet files found on refs/convert/parquet")
print("test parquet files:", files, flush=True)

import pandas as pd

docs = {}
for name in files:
    path = hf_hub_download(
        "deepmind/narrativeqa", name, repo_type="dataset",
        revision="refs/convert/parquet", local_dir=os.path.join(OUT_DIR, "raw"),
    )
    df = pd.read_parquet(path)
    print(name, "rows", len(df), flush=True)
    for _, row in df.iterrows():
        doc = row["document"]
        q = row["question"]
        doc_id = doc["id"]
        entry = docs.setdefault(doc_id, {"doc_id": doc_id, "text": None, "questions": []})
        if entry["text"] is None:
            entry["text"] = doc["text"]
        qt = (q["text"] if isinstance(q, dict) else str(q)).strip()
        if qt and qt not in entry["questions"]:
            entry["questions"].append(qt)

MIN_CHARS = 280_000  # ~>=70k tokens of book text, survives a 64k truncation
candidates = [
    d for d in docs.values()
    if len(d["questions"]) >= 8 and d["text"] and len(d["text"]) >= MIN_CHARS
]
candidates.sort(key=lambda d: len(d["text"]), reverse=True)
selected = candidates[:80]
print(
    f"docs total {len(docs)}, candidates {len(candidates)}, selected {len(selected)}",
    flush=True,
)
if len(selected) < 64:
    raise SystemExit("fewer than 64 qualifying books; lower MIN_CHARS")

with open(os.path.join(OUT_DIR, "stories.jsonl"), "w", encoding="utf-8") as out:
    for d in selected:
        out.write(json.dumps(
            {"doc_id": d["doc_id"], "text": d["text"], "questions": d["questions"][:16]},
            ensure_ascii=False) + "\n")
print("wrote", os.path.join(OUT_DIR, "stories.jsonl"), flush=True)
