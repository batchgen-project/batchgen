#!/usr/bin/env python3
"""Canonical decode workload policy shared by the recorder, comparator, and builder.

MoE routing follows the token stream, so a workload of replicated prompts
decoded greedily routes identical rows to identical experts and misstates both
expert-parallel balance and expert weight traffic.  The registry in
configs/decode_workloads.json names the workloads that count as MoE
performance evidence; this module decides, from the uploaded request file and
the server's per-row usage, whether a run actually is one of them.

Standard library only: the recorder runs on GPU nodes without pandas or a
tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REGISTRY_PATH = Path(__file__).resolve().parent / "configs" / "decode_workloads.json"
WORKLOAD_SCHEMA = "batchgen/decode-workload/v1"
# Registry v2: every profile states its kind.  The canonical-MoE rules in
# evaluate() describe decode workloads only; a profile without the field is a
# decode profile, so registries written before v2 behave as before.
WORKLOAD_KINDS = ("decode", "prefix")
# Labels that classify a profile without changing the workload it builds.
# They stay out of profile_digest(), so workload manifests built before the
# labels existed still match their profile.
CLASSIFICATION_FIELDS = ("workload_kind", "workload_class")


def load_registry(path: Path | None = None) -> dict[str, Any]:
    with (path or REGISTRY_PATH).open(encoding="utf-8") as f:
        registry = json.load(f)
    if registry.get("schema_version") != "batchgen/decode-workloads/v1":
        raise ValueError("decode workload registry has an unexpected schema_version")
    for name, profile in registry.get("profiles", {}).items():
        try:
            workload_kind(profile)
        except ValueError as exc:
            raise ValueError(f"decode workload registry profile {name!r}: {exc}") from None
    return registry


def workload_kind(profile: dict[str, Any]) -> str:
    kind = profile.get("workload_kind", "decode")
    if kind not in WORKLOAD_KINDS:
        raise ValueError(f"unknown workload_kind {kind!r}; expected one of {WORKLOAD_KINDS}")
    return kind


def moe_enforced(evidence: dict[str, Any]) -> bool:
    """Whether the canonical-MoE rules bind a run with this evidence block:
    the model is MoE and its profile (if any) is a decode profile."""
    return bool(evidence.get("moe")) and evidence.get("moe_enforced", True) is not False


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_digest(profile: dict[str, Any]) -> str:
    """Identity of a profile definition; a manifest built from an older
    definition is not evidence for the current one.  CLASSIFICATION_FIELDS
    are labels, not workload definition, and are excluded."""
    identity = {key: value for key, value in profile.items() if key not in CLASSIFICATION_FIELDS}
    return sha256_bytes(canonical_json(identity))


def model_policy(registry: dict[str, Any], model_id: str | None) -> dict[str, Any] | None:
    if not model_id:
        return None
    """Match a model id, launcher alias, or checkpoint path.  Patterns that
    start with "~" are regexes searched in the id (so "GLM-5.2-FP8",
    "zai-org/GLM-5.2-FP8" and ".../models/GLM-5.2-FP8" all match); others are
    exact aliases."""
    for entry in registry.get("models", []):
        for pattern in entry.get("match", []):
            if pattern.startswith("~"):
                if re.search(pattern[1:], model_id, re.I):
                    return entry
            elif pattern.lower() == model_id.lower():
                return entry
    return None


def profile_definition(registry: dict[str, Any], name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    return registry.get("profiles", {}).get(name)


def request_facts(row: dict[str, Any]) -> dict[str, Any]:
    body = row.get("body") if isinstance(row.get("body"), dict) else {}
    messages = body.get("messages")
    # Completions-format rows carry the pre-rendered prompt text instead of
    # messages; the distinct-prompt identity hashes whichever form the row has.
    prompt_content = messages if messages is not None else body.get("prompt")
    return {
        "custom_id": row.get("custom_id"),
        "messages_sha256": sha256_bytes(canonical_json(prompt_content)) if prompt_content is not None else None,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "ignore_eos": body.get("ignore_eos"),
        "max_completion_tokens": body.get("max_completion_tokens", body.get("max_tokens")),
        "chat_template_kwargs": body.get("chat_template_kwargs"),
    }


def output_facts(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    body = response.get("body") if isinstance(response.get("body"), dict) else row
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else row.get("usage", {})
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    finish = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else row.get("finish_reason")
    return {
        "custom_id": row.get("custom_id"),
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
        "finish_reason": finish,
    }


def classify_stratum(prompt_tokens: int, completion_tokens: int, index_topk: int) -> str:
    """S: stays <= index_topk for the whole run; X: crosses it during decode;
    L: the prompt already exceeds it."""
    if prompt_tokens > index_topk:
        return "L"
    if prompt_tokens + completion_tokens > index_topk:
        return "X"
    return "S"


def _sampling_equal(key: str, value: Any, expected: Any) -> bool:
    """A sampling value as both engines read it: an omitted top_k means
    disabled, the same as top_k 0."""
    if key == "top_k" and value is None:
        value = 0
    return _float_equal(value, expected)


def _float_equal(value: Any, expected: float) -> bool:
    try:
        return abs(float(value) - float(expected)) < 1e-9
    except (TypeError, ValueError):
        return False


def evaluate(
    *,
    registry: dict[str, Any],
    model_id: str | None,
    profile_name: str | None,
    lane: str | None,
    performance_claim: bool,
    workload_manifest: dict[str, Any] | None,
    request_rows: Iterable[dict[str, Any]] | None,
    request_jsonl_sha256: str | None,
    output_rows: Iterable[dict[str, Any]] | None,
    target_global_batch: int | None,
    max_active: int | None,
) -> dict[str, Any]:
    """Return the record's ``performance_evidence`` block.

    ``canonical`` is true only when the run declares a profile registered for
    its model and lane and every rule of that profile holds on the actual
    request file and server usage.  ``declared_profile_failed`` separates a
    run that claimed a profile and missed it (a broken run) from a run that
    never claimed one (a diagnostic run).
    """
    policy = model_policy(registry, model_id)
    moe = bool(policy and policy.get("moe"))
    evidence: dict[str, Any] = {
        "profile": profile_name,
        "lane": lane,
        "model_policy": policy.get("name") if policy else None,
        "moe": moe,
        "performance_claim": performance_claim,
        "canonical": False,
        "declared_profile_failed": False,
        "reasons": [],
    }
    reasons: list[str] = evidence["reasons"]
    profile = profile_definition(registry, profile_name)
    if profile is None:
        if profile_name:
            reasons.append(f"unknown_profile:{profile_name}")
            evidence["declared_profile_failed"] = True
        else:
            reasons.append("no_profile_declared")
        return evidence
    kind = workload_kind(profile)
    if kind != "decode":
        # The canonical-MoE rules below describe decode workloads.  A run of
        # another kind is never canonical decode evidence, and the MoE
        # diagnostic downgrade does not apply to it (moe_enforced).
        evidence["workload_kind"] = kind
        evidence["moe_enforced"] = False
        reasons.append(f"workload_kind_not_decode:{kind}")
        return evidence
    if lane and profile.get("lane") != lane:
        reasons.append(f"profile_lane_mismatch:{profile.get('lane')}!={lane}")
    if policy is None:
        reasons.append("model_has_no_workload_policy")
    elif profile_name not in policy.get("profiles", {}).get(profile.get("lane"), []):
        reasons.append(f"profile_not_registered_for_model:{policy.get('name')}")

    if workload_manifest is None:
        reasons.append("workload_manifest_missing")
    else:
        if workload_manifest.get("profile") != profile_name:
            reasons.append(f"workload_manifest_profile_mismatch:{workload_manifest.get('profile')}")
        if workload_manifest.get("profile_sha256") != profile_digest(profile):
            reasons.append("workload_manifest_built_from_a_different_profile_definition")
        expected_sha = workload_manifest.get("request_jsonl_sha256")
        if not request_jsonl_sha256:
            reasons.append("request_jsonl_missing")
        elif expected_sha != request_jsonl_sha256:
            reasons.append("request_jsonl_sha256_mismatch")

    requests = [request_facts(row) for row in request_rows or []]
    if not requests:
        if "request_jsonl_missing" not in reasons:
            reasons.append("request_jsonl_missing")
        evidence["declared_profile_failed"] = True
        return evidence
    evidence["request_rows"] = len(requests)
    ids = [row["custom_id"] for row in requests]
    if len(set(ids)) != len(ids) or None in ids:
        reasons.append("request_custom_ids_not_unique")
    if profile.get("distinct_prompts"):
        distinct = len({row["messages_sha256"] for row in requests})
        evidence["distinct_prompts"] = distinct
        if distinct != len(requests):
            reasons.append(f"duplicate_prompts:{len(requests) - distinct}")
    # Every declared sampling parameter must be explicit in every request: a
    # missing request value falls back to batch-level or server defaults,
    # which the request file cannot prove.
    sampling = profile.get("sampling", {})
    for key, expected in sampling.items():
        bad = sum(not _sampling_equal(key, row[key], expected) for row in requests)
        if bad:
            reasons.append(f"{key}_not_{expected}:{bad}_rows")
    if "top_k" not in sampling:
        # A profile that omits top_k declares it disabled: rows may omit it
        # or state 0, nothing else.
        bad = sum(not _sampling_equal("top_k", row["top_k"], 0) for row in requests)
        if bad:
            reasons.append(f"top_k_not_disabled:{bad}_rows")
    if "chat_template_kwargs" in profile:
        expected_kwargs = profile["chat_template_kwargs"]
        bad = sum(row["chat_template_kwargs"] != expected_kwargs for row in requests)
        if bad:
            reasons.append(f"chat_template_kwargs_not_{canonical_json(expected_kwargs).decode()}:{bad}_rows")
    if profile.get("ignore_eos"):
        bad = sum(row["ignore_eos"] is not True for row in requests)
        if bad:
            reasons.append(f"ignore_eos_not_set:{bad}_rows")
    max_completion = profile.get("max_completion_tokens")
    if max_completion is not None:
        bad = sum(row["max_completion_tokens"] != max_completion for row in requests)
        if bad:
            reasons.append(f"max_completion_tokens_not_{max_completion}:{bad}_rows")
    elif len({row["max_completion_tokens"] for row in requests}) != 1:
        reasons.append("max_completion_tokens_not_uniform")

    outputs = {row["custom_id"]: row for row in (output_facts(r) for r in output_rows or [])}
    joined = []
    missing = 0
    for row in requests:
        out = outputs.get(row["custom_id"])
        if out is None or out["prompt_tokens"] is None or out["completion_tokens"] is None:
            missing += 1
            continue
        joined.append((row, out))
    if missing:
        reasons.append(f"output_usage_missing:{missing}_rows")
    if profile.get("ignore_eos"):
        short = sum(out["completion_tokens"] != row["max_completion_tokens"] for row, out in joined)
        if short:
            reasons.append(f"completion_not_at_cap:{short}_rows")
    # Ties the server's execution to this request file: the builder's
    # prompt-token count (BatchGen's tokenizer and chat template) must equal
    # the server's usage for every row.
    predicted = {
        row.get("custom_id"): row.get("predicted_prompt_tokens")
        for row in (workload_manifest or {}).get("rows", [])
        if isinstance(row, dict)
    }
    if predicted:
        bad = sum(
            predicted.get(row["custom_id"]) != out["prompt_tokens"] for row, out in joined
        )
        if bad:
            reasons.append(f"prompt_tokens_differ_from_manifest:{bad}_rows")
    elif profile.get("lane") == "single_batch":
        reasons.append("workload_manifest_has_no_per_row_prompt_tokens")

    index_topk = policy.get("dsa_index_topk") if policy else None
    required = profile.get("required_strata", [])
    if index_topk and profile.get("lane") == "single_batch" and joined:
        counts: dict[str, int] = {"S": 0, "X": 0, "L": 0}
        for _, out in joined:
            counts[classify_stratum(int(out["prompt_tokens"]), int(out["completion_tokens"]), int(index_topk))] += 1
        evidence["dsa_strata"] = {"index_topk": index_topk, "counts": counts}
        minimum = float(profile.get("min_stratum_fraction", 0.0))
        for name in required:
            if name not in counts:
                continue
            if counts[name] < minimum * len(joined):
                reasons.append(f"dsa_stratum_{name}_below_{minimum:g}:{counts[name]}/{len(joined)}")
        extra = [name for name, count in counts.items() if count and name not in required]
        if extra:
            reasons.append(f"dsa_unexpected_strata:{','.join(extra)}")
        if set(required) == {"S", "X", "L"} and not (counts["S"] and counts["L"]):
            reasons.append("dsa_rows_not_on_both_sides_of_index_topk")

    if target_global_batch is not None:
        if max_active is None:
            reasons.append("observed_global_decode_batch_missing")
        elif max_active < target_global_batch:
            reasons.append(f"observed_global_decode_batch_{max_active}_below_target_{target_global_batch}")

    evidence["declared_profile_failed"] = bool(reasons)
    evidence["canonical"] = not reasons
    return evidence


def matrix_context_violation(policy: dict[str, Any] | None, contexts: list[int]) -> str | None:
    """A DSA model's exact-length matrix must hold cells on both sides of
    index_topk; a single cell at the boundary exercises neither side."""
    index_topk = policy.get("dsa_index_topk") if policy else None
    if not index_topk:
        return None
    if not any(c < index_topk for c in contexts) or not any(c > index_topk for c in contexts):
        return (
            f"{policy.get('name')} uses DSA index_topk={index_topk}: matrix contexts {contexts} "
            f"need at least one cell below and one above {index_topk}"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("policy", help="print the workload policy of a model id or alias")
    show.add_argument("model_id")
    digest = sub.add_parser("profile-digest", help="print the digest of a profile definition")
    digest.add_argument("profile")
    contexts = sub.add_parser("check-contexts", help="validate matrix context cells for a model")
    contexts.add_argument("model_id")
    contexts.add_argument("contexts", help="comma-separated prompt lengths")
    args = parser.parse_args()
    registry = load_registry()
    if args.command == "check-contexts":
        violation = matrix_context_violation(
            model_policy(registry, args.model_id), [int(c) for c in args.contexts.split(",") if c]
        )
        if violation:
            print(violation, file=sys.stderr)
            return 2
        return 0
    if args.command == "policy":
        policy = model_policy(registry, args.model_id)
        print(json.dumps(policy or {}, sort_keys=True))
        return 0
    profile = profile_definition(registry, args.profile)
    if profile is None:
        print(f"unknown profile {args.profile}", file=sys.stderr)
        return 2
    print(profile_digest(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
