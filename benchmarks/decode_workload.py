#!/usr/bin/env python3
"""Build a canonical decode workload: request JSONL + decode-workload/v1 manifest.

usage:
  python3 scripts/decode_workload.py build \
      --profile glm-dsa-diverse-v1 --rows 512 \
      --model zai-org/GLM-5.2-FP8 \
      --batchgen-root /path/to/BatchGen \
      --mmlu-dir /path/to/datasets/mmlu_pro \
      --longbench-dir /path/to/datasets/longbench \
      --gpu-pool-pages 2982 --world-size 8 \
      --out-dir /path/to/workloads

The profile (configs/decode_workloads.json) fixes everything that affects the
token stream: distinct prompts, sampling, completion cap, and for DSA models
the S/X/L length strata around index_topk.  Rows are generated sequentially
from seeded orders with strata interleaved by smooth weighted round robin, so
the first N rows of a larger build equal a build of N rows.  Prompt lengths
are counted with BatchGen's own tokenizer and chat template (the server's
count; the recorder re-checks every row against server usage anyway).

The build fails, writing nothing, if the rows cannot all be admitted at once:
it replays the engine's admission (least-sum-of-squared-prompt-length rank
assignment, longest first; each sequence reserves ceil((prompt+completion)/64)
pages; a rank may fill admission_fraction of its pool).

--allow-oversubscribed writes such a workload anyway, for diagnostics of
oversubscribed behavior (e.g. G1024 against a pool sized for G512): the rows
are selected exactly as without the flag (so the nested-prefix property
holds), and the manifest records admission_check.fits = false plus a
top-level "oversubscribed": true.  A run of an oversubscribed workload is
never a canonical fixed-shape performance row (scripts/decode_run.py).  When
the rows fit, the flag changes nothing: the manifest is byte-identical to a
build without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts import workload_policy
except ImportError:  # executed as scripts/decode_workload.py
    import workload_policy

PAGE_TOKENS = 64
# Admission reserves min(prompt + INITIAL_GPU_PAGE_BUFFER pages, kv budget);
# with completion caps below 32 pages this is prompt + completion.
INITIAL_BUFFER_TOKENS = 32 * PAGE_TOKENS


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(workload_policy.sha256_file(path).encode())
    return digest.hexdigest()


def _smooth_wrr(weights: dict[str, float], count: int) -> list[str]:
    """Deterministic interleaving: every prefix holds each stratum within one
    row of its weighted share."""
    names = sorted(weights)
    current = {name: 0.0 for name in names}
    total = sum(weights.values())
    order = []
    for _ in range(count):
        for name in names:
            current[name] += weights[name]
        best = max(names, key=lambda name: (current[name], -names.index(name)))
        current[best] -= total
        order.append(best)
    return order


class Counter:
    def __init__(self, tokenizer: Any, thinking_field: str | None, enable_thinking: bool):
        self.tokenizer = tokenizer
        self.kwargs = {thinking_field: enable_thinking} if thinking_field else {}

    def count(self, messages: list[dict[str, str]]) -> int:
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **self.kwargs
        )
        return len(ids)

    def render(self, messages: list[dict[str, str]]) -> str:
        """The same template rendered to text (for completions-format rows)."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.kwargs
        )


class MmluSource:
    def __init__(self, mmlu_dir: Path, template: str, seed: int):
        import mmlu_pro_test

        import pandas as pd

        self.m = mmlu_pro_test
        self.files = [mmlu_dir / "mmlu_pro_test.parquet", mmlu_dir / "mmlu_pro_validation.parquet"]
        self.test = pd.read_parquet(self.files[0])
        self.val = pd.read_parquet(self.files[1])
        self.cfg = mmlu_pro_test.MODEL_CONFIGS[template]
        self.order = list(range(len(self.test)))
        random.Random(seed).shuffle(self.order)
        self.cursor = 0
        self.prefix_cache: dict[tuple[str, int], str] = {}

    def messages(self, row: Any, shots: int) -> list[dict[str, str]]:
        key = (row["category"], shots)
        if key not in self.prefix_cache:
            self.prefix_cache[key] = self.m.build_few_shot_prefix(self.val, row["category"], shots)
        query = self.cfg["query_template"].format(
            prefix=self.prefix_cache[key],
            question=row["question"],
            options=self.m.form_options(row["options"]),
        )
        return [
            {"role": "system", "content": self.cfg["system_prompt"]},
            {"role": "user", "content": query},
        ]

    def next_row(self, shots: list[int], band: list[int], counter: Counter, rng: random.Random,
                 used: set[str]) -> tuple[list[dict[str, str]], int, str] | None:
        while self.cursor < len(self.order):
            row = self.test.iloc[self.order[self.cursor]]
            self.cursor += 1
            source_id = f"mmlu_pro/{row['question_id']}"
            if source_id in used:
                continue
            k = shots[rng.randrange(len(shots))]
            messages = self.messages(row, k)
            tokens = counter.count(messages)
            if band[0] <= tokens <= band[1]:
                return messages, tokens, f"{source_id}/{k}shot"
        return None


class LongBenchSource:
    def __init__(self, longbench_dir: Path, tokenizer: Any, seed: int):
        import pandas as pd

        self.tokenizer = tokenizer
        self.files = sorted(longbench_dir.glob("*/*.parquet"))
        rows = []
        seen = set()
        for path in self.files:
            frame = pd.read_parquet(path)
            for _, item in frame.iterrows():
                context = str(item.get("context") or "")
                question = str(item.get("question") or "")
                key = hashlib.sha256((context + "\x00" + question).encode()).hexdigest()
                if not context or key in seen:
                    continue  # LongBench-E repeats rows of the base subsets
                seen.add(key)
                rows.append((f"longbench/{path.parent.name}/{item.get('_id')}", context, question))
        self.rows = rows
        self.order = list(range(len(rows)))
        random.Random(seed).shuffle(self.order)
        self.cursor = 0

    @staticmethod
    def _messages(context: str, question: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"{context}\n\n{question}"}]

    def _truncate(self, ids: list[int], budget: int) -> str:
        """Middle truncation, as in the LongBench reference harness."""
        if budget >= len(ids):
            return self.tokenizer.decode(ids)
        head = budget // 2
        return self.tokenizer.decode(ids[:head]) + self.tokenizer.decode(ids[len(ids) - (budget - head):])

    def next_row(self, band: list[int], counter: Counter, rng: random.Random,
                 used: set[str]) -> tuple[list[dict[str, str]], int, str] | None:
        while self.cursor < len(self.order):
            source_id, context, question = self.rows[self.order[self.cursor]]
            self.cursor += 1
            if source_id in used:
                continue
            target = rng.randint(band[0], band[1])
            overhead = counter.count(self._messages("", question))
            ids = self.tokenizer.encode(context, add_special_tokens=False)
            if overhead + len(ids) < band[0]:
                continue
            budget = target - overhead
            for _ in range(4):
                messages = self._messages(self._truncate(ids, max(budget, 0)), question)
                tokens = counter.count(messages)
                if band[0] <= tokens <= band[1]:
                    return messages, tokens, source_id
                budget += target - tokens
        return None


class LiveCodeBenchSource:
    """Competitive-programming prompts from livecodebench/code_generation_lite.

    Reads the lightweight prompts.jsonl extracted from the release files
    (question_id/title/difficulty/platform/question_content/starter_code);
    prompts are used at their natural length, the stratum's prompt_tokens
    band only filters, and an optional per-stratum "difficulties" list
    restricts the pool (hard problems keep forced-length decodes on real
    reasoning content instead of post-EOS gibberish)."""

    def __init__(self, lcb_dir: Path, seed: int):
        rows = []
        self.files = [lcb_dir / "prompts.jsonl"]
        with open(self.files[0], encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                rows.append((f"livecodebench/{item['question_id']}", item))
        self.rows = rows
        self.order = list(range(len(rows)))
        random.Random(seed).shuffle(self.order)
        self.cursor = 0

    @staticmethod
    def _messages(item: dict[str, Any]) -> list[dict[str, str]]:
        parts = [
            "Solve the following competitive programming problem. Reason carefully "
            "about the algorithm and its complexity before writing the final "
            "solution in Python.",
            "",
            f"### {item.get('title', '')}",
            "",
            item["question_content"].strip(),
        ]
        starter = (item.get("starter_code") or "").strip()
        if starter:
            parts += ["", "Complete the following starter code:", "```python",
                      starter, "```"]
        else:
            parts += ["", "The program reads from standard input and writes to "
                          "standard output."]
        return [{"role": "user", "content": "\n".join(parts)}]

    def next_row(self, band: list[int], counter: Counter, rng: random.Random,
                 used: set[str], difficulties: list[str] | None = None
                 ) -> tuple[list[dict[str, str]], int, str] | None:
        while self.cursor < len(self.order):
            source_id, item = self.rows[self.order[self.cursor]]
            self.cursor += 1
            if source_id in used:
                continue
            if difficulties and item["difficulty"] not in difficulties:
                continue
            messages = self._messages(item)
            tokens = counter.count(messages)
            if band[0] <= tokens <= band[1]:
                return messages, tokens, source_id
        return None


class NarrativeQAGroupedSource:
    """Document-grouped long-prefix prompts from the original NarrativeQA.

    Reads stories.jsonl ({doc_id, text, questions: [...]}) prepared from the
    HF test split.  Each selected book becomes one GROUP: its text is
    truncated to the stratum's ``prefix_tokens_target`` tokens and shared
    verbatim by ``rows_per_group`` rows whose suffix is one question each, so
    distinct groups share distinct prefixes (the RAG / doc-QA batch shape).
    Rows are emitted group-contiguous; use a single stratum so the WRR order
    cannot interleave groups."""

    def __init__(self, nqa_dir: Path, tokenizer: Any, seed: int):
        self.files = [nqa_dir / "stories.jsonl"]
        docs = []
        with open(self.files[0], encoding="utf-8") as handle:
            for line in handle:
                docs.append(json.loads(line))
        random.Random(seed).shuffle(docs)
        self.docs = docs
        self.tokenizer = tokenizer
        self.cursor = 0  # index into the flattened (group, question) stream
        self.group_idx = -1
        self.pending: list[tuple[list[dict[str, str]], str]] = []
        self.truncated_cache: str | None = None

    _INSTRUCTION = (
        "You are given the full text of a story. Read it carefully, then "
        "answer the question at the end. Think the question through against "
        "the text before giving a complete, well-justified answer."
    )

    def _next_group(self, groups: int, rows_per_group: int,
                    prefix_tokens_target: int) -> None:
        while True:
            self.group_idx += 1
            if self.group_idx >= len(self.docs) or self.group_idx >= groups:
                raise SystemExit(
                    f"narrativeqa_grouped: ran out of groups at {self.group_idx} "
                    f"(need {groups} with >= {rows_per_group} questions)"
                )
            doc = self.docs[self.group_idx]
            questions = list(dict.fromkeys(doc["questions"]))
            if len(questions) < rows_per_group:
                continue
            ids = self.tokenizer.encode(doc["text"], add_special_tokens=False)
            if len(ids) < prefix_tokens_target:
                continue
            book = self.tokenizer.decode(ids[:prefix_tokens_target])
            prefix = f"{self._INSTRUCTION}\n\n<story>\n{book}\n</story>\n\n"
            self.pending = [
                ([{"role": "user", "content": f"{prefix}Question: {q.strip()}"}],
                 f"narrativeqa/{doc['doc_id']}#q{i}")
                for i, q in enumerate(questions[:rows_per_group])
            ]
            return

    def next_row(self, counter: Counter, groups: int, rows_per_group: int,
                 prefix_tokens_target: int
                 ) -> tuple[list[dict[str, str]], int, str] | None:
        if not self.pending:
            self._next_group(groups, rows_per_group, prefix_tokens_target)
        messages, source_id = self.pending.pop(0)
        return messages, counter.count(messages), source_id


def admission_check(prompt_tokens: list[int], max_completion: int, world_size: int,
                    pool_pages: int, fraction: float, max_rank_rows: int) -> dict[str, Any]:
    """Replay decode admission for one batch holding every row at once."""
    load = [0.0] * world_size
    rank_rows = [0] * world_size
    rank_pages = [0] * world_size
    for tokens in sorted(prompt_tokens, reverse=True):
        rank = min(range(world_size), key=lambda r: load[r])
        load[rank] += float(tokens) * float(tokens)
        rank_rows[rank] += 1
        reserve = min(tokens + INITIAL_BUFFER_TOKENS, tokens + max_completion)
        rank_pages[rank] += math.ceil(reserve / PAGE_TOKENS)
    capacity = int(pool_pages * fraction)
    return {
        "world_size": world_size,
        "gpu_pool_pages_per_rank": pool_pages,
        "admission_fraction": fraction,
        "capacity_pages_per_rank": capacity,
        "rank_pages": rank_pages,
        "rank_rows": rank_rows,
        "max_rank_rows_allowed": max_rank_rows,
        "max_page_utilization": max(rank_pages) / capacity,
        "fits": max(rank_pages) <= capacity and max(rank_rows) <= max_rank_rows,
    }


def build(args: argparse.Namespace) -> int:
    registry = workload_policy.load_registry()
    profile = workload_policy.profile_definition(registry, args.profile)
    if profile is None or profile.get("lane") != "single_batch":
        raise SystemExit(f"{args.profile} is not a single-batch profile in the registry")
    policy = workload_policy.model_policy(registry, args.model)
    if policy is None or args.profile not in policy.get("profiles", {}).get("single_batch", []):
        raise SystemExit(f"{args.profile} is not registered for model {args.model}")

    sys.path.insert(0, str(args.batchgen_root))
    from batchgen.config.tokenizer_registry import load_tokenizer

    import mmlu_pro_test

    tokenizer = load_tokenizer(args.model)
    # The request field that carries the thinking switch, and the same switch
    # handed to the chat template when counting (as the server does).
    thinking_field = mmlu_pro_test.MODEL_CONFIGS[policy["mmlu_template"]].get("thinking_field")
    counter = Counter(tokenizer, thinking_field, bool(profile.get("enable_thinking")))
    seed = int(profile["selection_seed"])
    strata = profile["strata"]
    sources_needed = {spec["source"] for spec in strata.values()}
    mmlu = MmluSource(args.mmlu_dir, policy["mmlu_template"], seed) if "mmlu_pro" in sources_needed else None
    longbench = LongBenchSource(args.longbench_dir, tokenizer, seed + 1) if "longbench" in sources_needed else None
    lcb = LiveCodeBenchSource(args.livecodebench_dir, seed + 2) if "livecodebench" in sources_needed else None
    nqa = (NarrativeQAGroupedSource(args.narrativeqa_dir, tokenizer, seed + 3)
           if "narrativeqa_grouped" in sources_needed else None)
    rngs = {name: random.Random(f"{seed}:{name}") for name in strata}

    order = _smooth_wrr({name: float(spec["weight"]) for name, spec in strata.items()}, args.rows)
    used: set[str] = set()
    prompts: set[str] = set()
    lines: list[str] = []
    rows_meta: list[dict[str, Any]] = []
    sampling = profile["sampling"]
    for index, name in enumerate(order):
        spec = strata[name]
        while True:
            if spec["source"] == "mmlu_pro":
                found = mmlu.next_row(spec["shots"], spec["prompt_tokens"], counter, rngs[name], used)
            elif spec["source"] == "livecodebench":
                found = lcb.next_row(spec["prompt_tokens"], counter, rngs[name], used,
                                     spec.get("difficulties"))
            elif spec["source"] == "narrativeqa_grouped":
                found = nqa.next_row(counter, int(spec["groups"]),
                                     int(spec["rows_per_group"]),
                                     int(spec["prefix_tokens_target"]))
            else:
                found = longbench.next_row(spec["prompt_tokens"], counter, rngs[name], used)
            if found is None:
                raise SystemExit(f"source {spec['source']} ran out of rows for stratum {name} at row {index}")
            messages, tokens, source_id = found
            digest = workload_policy.sha256_bytes(workload_policy.canonical_json(messages))
            if digest not in prompts:
                break
        used.add(source_id.rsplit("/", 1)[0] if source_id.endswith("shot") else source_id)
        prompts.add(digest)
        custom_id = f"{args.profile}-{index:05d}"
        body = {
            "model": args.model,
            "messages": messages,
            "max_completion_tokens": int(profile["max_completion_tokens"]),
            "ignore_eos": bool(profile["ignore_eos"]),
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
        }
        # v1 profiles state top_k 0 (disabled); v2 profiles omit it, which
        # both engines read as disabled (SGLang rejects an explicit 0).
        if "top_k" in sampling:
            body["top_k"] = int(sampling["top_k"])
        if thinking_field and "enable_thinking" in profile:
            body[thinking_field] = bool(profile["enable_thinking"])
        if "chat_template_kwargs" in profile:
            body["chat_template_kwargs"] = dict(profile["chat_template_kwargs"])
        url = "/v1/chat/completions"
        if args.request_format == "completions":
            # Pre-render the SAME template to text so a foreign engine
            # tokenizes exactly the predicted count instead of applying its
            # own chat rendering (verified: encode(rendered, no specials) ==
            # apply_chat_template(tokenize=True) for the K3 tokenizer).
            prompt_text = counter.render(messages)
            body = {
                "model": body["model"],
                "prompt": prompt_text,
                "max_tokens": int(profile["max_completion_tokens"]),
                "ignore_eos": bool(profile["ignore_eos"]),
                "temperature": float(sampling["temperature"]),
                "top_p": float(sampling["top_p"]),
            }
            if "top_k" in sampling:
                body["top_k"] = int(sampling["top_k"])
            url = "/v1/completions"
        lines.append(json.dumps(
            {"custom_id": custom_id, "method": "POST", "url": url, "body": body},
            sort_keys=True, ensure_ascii=False,
        ))
        rows_meta.append({
            "custom_id": custom_id,
            "stratum": name,
            "source": source_id,
            "predicted_prompt_tokens": tokens,
            "messages_sha256": digest,
        })

    tokens_all = [row["predicted_prompt_tokens"] for row in rows_meta]
    check = admission_check(tokens_all, int(profile["max_completion_tokens"]), args.world_size,
                            args.gpu_pool_pages, args.admission_fraction, args.max_rank_rows)
    oversubscribed = not check["fits"]
    if oversubscribed:
        print(json.dumps(check, indent=2), file=sys.stderr)
        if not args.allow_oversubscribed:
            raise SystemExit("rows cannot all be admitted at once; adjust the profile bands or the row count")
        print(
            f"WARNING: writing an OVERSUBSCRIBED workload (max_page_utilization "
            f"{check['max_page_utilization']:.4f}); its runs are diagnostics, never canonical rows",
            file=sys.stderr,
        )

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    dataset_files = [f for src in (mmlu, longbench, lcb, nqa) if src for f in src.files]
    batchgen_commit = os.popen(f"git -C {args.batchgen_root} rev-parse HEAD 2>/dev/null").read().strip() or "unknown"
    per_stratum = {}
    for name in strata:
        values = [row["predicted_prompt_tokens"] for row in rows_meta if row["stratum"] == name]
        per_stratum[name] = {
            "rows": len(values),
            "prompt_tokens": {"min": min(values), "max": max(values), "mean": statistics.mean(values)} if values else None,
        }
    manifest = {
        "schema_version": workload_policy.WORKLOAD_SCHEMA,
        "profile": args.profile,
        "profile_sha256": workload_policy.profile_digest(profile),
        "lane": "single_batch",
        "model": args.model,
        "model_policy": policy["name"],
        "request_count": args.rows,
        "target_global_batch": args.rows,
        "request_jsonl_sha256": workload_policy.sha256_bytes(payload),
        "dataset_label": "+".join(sorted(sources_needed)),
        "dataset_revision": "sha256:" + _sha256_files(dataset_files),
        "selection_seed": seed,
        "sampling": sampling,
        "enable_thinking": profile.get("enable_thinking"),
        "ignore_eos": profile["ignore_eos"],
        "max_completion_tokens": profile["max_completion_tokens"],
        "distinct_prompts": len(prompts),
        "prompt_tokens": {"min": min(tokens_all), "max": max(tokens_all), "mean": statistics.mean(tokens_all)},
        "strata": per_stratum,
        "tokenizer": {"batchgen_commit": batchgen_commit, "class": type(tokenizer).__name__},
        "admission_check": check,
        "rows": rows_meta,
    }
    if oversubscribed:
        manifest["oversubscribed"] = True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.profile}_n{args.rows}"
    if args.request_format == "completions":
        stem += "_completions"
    (args.out_dir / f"{stem}.jsonl").write_bytes(payload)
    (args.out_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "jsonl": str(args.out_dir / f"{stem}.jsonl"),
        "manifest": str(args.out_dir / f"{stem}.manifest.json"),
        "request_jsonl_sha256": manifest["request_jsonl_sha256"],
        "strata": {name: value["rows"] for name, value in per_stratum.items()},
        "max_page_utilization": round(check["max_page_utilization"], 4),
        "rank_rows": check["rank_rows"],
        "oversubscribed": oversubscribed,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--profile", required=True)
    b.add_argument("--rows", type=int, required=True)
    b.add_argument("--model", required=True, help="server model id; selects the workload policy and tokenizer")
    b.add_argument("--batchgen-root", type=Path, required=True, help="BatchGen checkout providing the tokenizer")
    b.add_argument("--mmlu-dir", type=Path, required=True)
    b.add_argument("--longbench-dir", type=Path)
    b.add_argument("--narrativeqa-dir", type=Path,
                   help="Directory holding stories.jsonl prepared from the "
                        "original NarrativeQA test split (doc_id/text/questions).")
    b.add_argument("--livecodebench-dir", type=Path,
                   help="directory holding the extracted prompts.jsonl of livecodebench/code_generation_lite")
    b.add_argument("--gpu-pool-pages", type=int, required=True, help="per-rank GPU KV pages of the target engine")
    b.add_argument("--admission-fraction", type=float, default=0.9)
    b.add_argument("--world-size", type=int, default=8)
    b.add_argument("--max-rank-rows", type=int, default=128)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument(
        "--request-format", choices=["chat", "completions"], default="chat",
        help="completions: pre-render the chat template to prompt text so a "
        "foreign engine tokenizes exactly the predicted prompt tokens",
    )
    b.add_argument(
        "--allow-oversubscribed", action="store_true",
        help="Write a workload whose rows cannot all be admitted at once (diagnostic only; "
             "the manifest says oversubscribed: true).",
    )
    args = parser.parse_args(argv)
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
