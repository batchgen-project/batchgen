#!/usr/bin/env bash
# PR File Hygiene check — enforces the high-precision subset of PR_MERGE_POLICY.md §1/§4
# plus the §2.4–§2.6 change-type boundary (a model/kernel PR may not touch scaffolding).
#
# §1/§4 scan ONLY the files/lines a PR adds, so they never trip on pre-existing (legacy)
# violations. The §2 boundary inspects all changed files vs the declared PR type. See
# PR_MERGE_POLICY.md §8. Set PR_TYPE=model|kernel|core|fix|infra|docs to test §2 locally.
#
# Usage:   bash .github/workflows/scripts/check-pr-hygiene.sh [base_ref]
#   base_ref defaults to origin/main.
#
# Exit status:
#   - Blocking violations found AND HYGIENE_ENFORCE=1  -> exit 1
#   - Otherwise (report-only, or no blocking violations) -> exit 0
# Advisory findings are printed but never change the exit status.

set -uo pipefail

BASE_REF="${1:-origin/main}"
ENFORCE="${HYGIENE_ENFORCE:-0}"

# Importable / shipped code. A scratch script or stray test here is a violation.
PROD_RE='^(batchgen|batchgen_kernels|core|server|op_builder)/'
# Files exempt from the env-flag advisory (sanctioned env-flag modules).
ALLOWLIST=("batchgen/timing.py")

# §2.6 scaffolding layer — only a `core` PR may modify these.
SCAFFOLD_FILE_RE='^batchgen/(batchgen_worker|server_worker_main_loop|continuous_batching|scheduler|task_scheduler|decode|decode_task|prefill|prefill_task|sampling|sequence|migration|pd_orchestrator|query_book|query_manager|parameter_server|parameter_server_client|node_manager|inference_runtime|batch_inference|batch_order|batchgen_server|batchgen_server_dev|batchgen_client|client_optimized|launch_server|launch_http_server|entrypoint|lifespan|generate|model_instance)\.py$'
SCAFFOLD_DIR_RE='^batchgen/(scheduler|server|kv_cache|worker|sequence_manager|distributed|planner|cuda_graph|core)/'
# §2.5 per-type allowlists — a `model`/`kernel` PR may touch ONLY these paths.
MODEL_ALLOW_RE='^(batchgen/models/|batchgen/get_initializer\.py$|batchgen/get_parallel_strategy_manager\.py$|batchgen_kernels/|configurations/|tests/|docs/)'
KERNEL_ALLOW_RE='^(batchgen_kernels/|batchgen/(moe|attention|gemm|quantization|triton_kernels|other_kernels)/|op_builder/|batchgen/op_builder/|tests/|docs/)'

merge_base="$(git merge-base "$BASE_REF" HEAD 2>/dev/null || echo "$BASE_REF")"

# New files this PR introduces (added only). Read-loop instead of mapfile so this
# runs on macOS bash 3.2 (local pre-check), not just the ubuntu CI runner.
ADDED=()
while IFS= read -r _f; do [[ -n "$_f" ]] && ADDED+=("$_f"); done \
  < <(git diff --name-only --diff-filter=A "$merge_base"...HEAD)

blocking=0
advisory=0
block() { echo "  ✗ BLOCK  $1"; blocking=$((blocking + 1)); }
warn()  { echo "  ⚠ ADVISE $1"; advisory=$((advisory + 1)); }

in_allowlist() { local f="$1"; for a in "${ALLOWLIST[@]}"; do [[ "$f" == "$a" ]] && return 0; done; return 1; }

# §2.4 declared change type — from $PR_TYPE if set (local use), else parsed from the
# Type-of-Change section of $PR_BODY (the PR description, passed by CI). One type per line.
resolve_declared_type() {
  if [[ -n "${PR_TYPE:-}" ]]; then echo "$PR_TYPE"; return; fi
  [[ -z "${PR_BODY:-}" ]] && return
  printf '%s\n' "$PR_BODY" | tr -d '\r' \
    | awk '/^## Type of Change/{f=1;next} /^## /{f=0} f' \
    | grep -oE '\[[xX]\][^`]*`(model|kernel|core|fix|infra|docs)`' \
    | grep -oE '(model|kernel|core|fix|infra|docs)' | sort -u
}

echo "== PR File Hygiene =="
echo "base: $BASE_REF (merge-base ${merge_base:0:12})   enforce: $ENFORCE"
echo "new files in PR: ${#ADDED[@]}"
echo

# ---- File-name checks on newly-added files (H1, H2, H6, H7) ----
for f in ${ADDED[@]+"${ADDED[@]}"}; do
  base="$(basename "$f")"
  in_prod=0; [[ "$f" =~ $PROD_RE || "$f" != */* ]] && in_prod=1

  # H1: throwaway scratch / debug / check scripts in a production package or repo root.
  # bench_* is intentionally NOT here — benchmarks may live alongside the kernels (§1.1).
  if [[ $in_prod -eq 1 && "$base" =~ ^(debug_|check_|scratch_|tmp_).*\.py$ || "$base" =~ _scratch\.py$ && $in_prod -eq 1 ]]; then
    block "§1.1 throwaway scratch/debug script in production tree: $f"
    continue
  fi
  # H2: test_*.py interleaved in the runtime package (tests belong in tests/).
  if [[ $in_prod -eq 1 && "$base" =~ ^test_.*\.py$ ]]; then
    block "§1.2 test file inside runtime package (move to tests/): $f"
    continue
  fi
  # H7: loose .py at repo root other than setup.py.
  if [[ "$f" != */* && "$f" =~ \.py$ && "$f" != "setup.py" ]]; then
    block "§1.7 loose utility at repo root (move to scripts/): $f"
    continue
  fi
  # H6: generated artifacts / weights / wheels.
  if [[ "$base" =~ \.(log|nsys-rep|ncu-rep|sqlite|whl|pt|pth|safetensors|bin)$ ]]; then
    block "§1.6 generated artifact committed: $f"
    continue
  fi
  # H6: benchmark CSVs (allow genuine data dirs).
  if [[ "$base" =~ \.csv$ && ! "$f" =~ ^(configurations|assets|tests|docs)/ ]]; then
    block "§1.6 CSV artifact committed outside a data dir: $f"
    continue
  fi
done

# ---- Commit-message check (§4.1): no Co-Authored-By in the PR range ----
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  sha="${line%% *}"; subj="${line#* }"
  if git log -1 --format='%B' "$sha" | grep -qi '^co-authored-by:'; then
    block "§4.1 Co-Authored-By trailer in commit ${sha:0:12} ($subj)"
  fi
done < <(git log --format='%H %s' "$merge_base"..HEAD)

# ---- Advisory line checks on added lines in production files (§1.3, §1.4) ----
CHANGED_PROD=()
while IFS= read -r _f; do [[ -n "$_f" ]] && CHANGED_PROD+=("$_f"); done \
  < <(git diff --name-only "$merge_base"...HEAD | grep -E "$PROD_RE" || true)
for f in ${CHANGED_PROD[@]+"${CHANGED_PROD[@]}"}; do
  in_allowlist "$f" && continue
  # Added content lines only (single leading '+', not the '+++' header); drop explicit
  # waivers. awk (not BSD grep on '\+') keeps this portable to macOS.
  added="$(git diff "$merge_base"...HEAD -- "$f" \
    | awk '/^\+\+\+/{next} /^\+/{print}' | grep -vF 'noqa: hygiene' || true)"
  [[ -z "$added" ]] && continue
  if printf '%s\n' "$added" | grep -Eq '(environ\.get|getenv)\([^)]*BATCHGEN_'; then
    warn "§1.3 new BATCHGEN_* env guard in production code — use a batchgen_debug batch flag: $f"
  fi
  if printf '%s\n' "$added" | grep -Eq '^\+[[:space:]]*print\('; then
    warn "§1.4 new print() in production path — use a one-shot guarded logger: $f"
  fi
done

# ---- §2 change-type boundary (path-scope, §2.4–§2.6) ----
# A `model`/`kernel` PR may touch only its allowlist; the scaffolding layer is core-only.
declared=()
while IFS= read -r _t; do [[ -n "$_t" ]] && declared+=("$_t"); done < <(resolve_declared_type)
echo
if [[ ${#declared[@]} -eq 0 ]]; then
  echo "== §2 boundary: PR type undeclared (set PR_TYPE, or tick one Type of Change) — skipped =="
elif [[ ${#declared[@]} -gt 1 ]]; then
  echo "== §2 boundary: multiple types ticked =="
  warn "§2.4 more than one Type of Change ticked (${declared[*]}) — declare exactly one"
else
  ptype="${declared[0]}"
  echo "== §2 boundary: declared type = $ptype =="
  if [[ "$ptype" == "model" || "$ptype" == "kernel" ]]; then
    [[ "$ptype" == "model" ]] && allow="$MODEL_ALLOW_RE" || allow="$KERNEL_ALLOW_RE"
    CHANGED_ALL=()
    while IFS= read -r _f; do [[ -n "$_f" ]] && CHANGED_ALL+=("$_f"); done \
      < <(git diff --name-only "$merge_base"...HEAD)
    for f in ${CHANGED_ALL[@]+"${CHANGED_ALL[@]}"}; do
      [[ "$f" =~ $allow ]] && continue
      if [[ "$f" =~ $SCAFFOLD_FILE_RE || "$f" =~ $SCAFFOLD_DIR_RE ]]; then
        block "§2.6 scaffolding edit in a $ptype PR (only a core PR may touch it): $f"
      else
        block "§2.5 file outside the $ptype allowlist (split it into its own PR): $f"
      fi
    done
  else
    echo "  (type=$ptype — no model/kernel path restriction; bound by §1/§3)"
  fi
fi

echo
echo "== summary: $blocking blocking, $advisory advisory =="
if [[ $blocking -gt 0 && "$ENFORCE" == "1" ]]; then
  echo "FAIL: blocking hygiene violations (see PR_MERGE_POLICY.md §1). Override per §9 if intended."
  exit 1
fi
if [[ $blocking -gt 0 ]]; then
  echo "REPORT-ONLY: blocking violations present but gate not yet enforced (HYGIENE_ENFORCE!=1)."
fi
exit 0
