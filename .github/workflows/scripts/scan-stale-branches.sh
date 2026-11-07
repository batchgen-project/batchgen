#!/usr/bin/env bash
#
# Audit remote branches against the default branch and help enforce
# PR_MERGE_POLICY.md §4 ("Delete the PR branch once merged").
#
# Every remote branch is classified as:
#   linear   — tip is an ancestor of origin/<default>      (full linear history)
#   squash   — net diff already in <default> (git cherry)  (rewritten / squash-merged)
#   unmerged — has unique work not in <default>            (kept; never deleted)
# and, for unmerged branches, flagged dead/stale by last-commit age.
#
# A Markdown report is written to $GITHUB_STEP_SUMMARY. Deletion is OPT-IN:
# it runs only on a manual workflow_dispatch with MODE=delete-merged and
# CONFIRM=DELETE, and only ever removes branches that are provably merged
# (linear, or single-commit squash). Scheduled runs are always report-only.
# Unmerged / dead / stale / protected / open-PR branches are never deleted.
#
# Detection of squash-merges uses a synthetic full-tree commit on the
# merge-base plus `git cherry` patch-id matching — equivalent to
# `git-delete-merged-branches --effort=3`. See docs/branch-maintenance.md.
set -euo pipefail

DEF="${DEFAULT_BRANCH:-main}"
MAIN="origin/${DEF}"
STALE_DAYS="${STALE_DAYS:-90}"
DEAD_DAYS="${DEAD_DAYS:-180}"
MODE="${MODE:-report}"
CONFIRM="${CONFIRM:-}"
EVENT="${EVENT_NAME:-workflow_dispatch}"
PROTECT_RE="${PROTECT_REGEX:-^(main|master|release/.*|release-.*|hotfix/.*)$}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
NOW="$(date +%s)"

# Safety backstop: scheduled runs never delete, regardless of inputs.
if [ "$EVENT" = "schedule" ]; then MODE="report"; fi

echo "Fetching remote branches (prune)..."
git fetch --no-tags --prune origin '+refs/heads/*:refs/remotes/origin/*' >/dev/null 2>&1 || true

# Branches with an open PR must never be touched.
OPEN_PRS="$(gh pr list --state open --limit 1000 --json headRefName --jq '.[].headRefName' 2>/dev/null | sort -u || true)"
is_open_pr() { [ -n "$OPEN_PRS" ] && printf '%s\n' "$OPEN_PRS" | grep -qxF "$1"; }

linear=()   # "br|age"        ancestor of default (safe to delete)
squash1=()  # "br|age"        single-commit squash (safe to delete)
squashN=()  # "br|age|ahead"  multi-commit squash (report; manual review)
dead=()     # "br|age"        unmerged, idle >= DEAD_DAYS
stale=()    # "br|age"        unmerged, STALE_DAYS..DEAD_DAYS
active=()   # "br|age"        unmerged, idle < STALE_DAYS

while read -r ref; do
  br="${ref#origin/}"
  [ -z "$br" ] && continue
  [ "$br" = "$DEF" ] && continue
  if printf '%s\n' "$br" | grep -qE "$PROTECT_RE"; then continue; fi
  if is_open_pr "$br"; then continue; fi

  ts="$(git log -1 --format=%ct "$ref" 2>/dev/null || echo "$NOW")"
  age=$(( (NOW - ts) / 86400 ))
  ahead="$(git rev-list --count "${MAIN}..${ref}" 2>/dev/null || echo 0)"

  if [ "$ahead" -eq 0 ]; then
    linear+=("${br}|${age}")
    continue
  fi

  # Not a linear ancestor: test for a squash/rebase-merge via patch-id.
  mb="$(git merge-base "$MAIN" "$ref" 2>/dev/null || true)"
  sig=""
  if [ -n "$mb" ]; then
    tree="$(git rev-parse "${ref}^{tree}")"
    synth="$(git commit-tree "$tree" -p "$mb" -m _)"
    sig="$(git cherry "$MAIN" "$synth" 2>/dev/null | head -1 | cut -c1 || true)"
  fi

  if [ "$sig" = "-" ]; then
    if [ "$ahead" -eq 1 ]; then squash1+=("${br}|${age}"); else squashN+=("${br}|${age}|${ahead}"); fi
  elif [ "$age" -ge "$DEAD_DAYS" ]; then dead+=("${br}|${age}")
  elif [ "$age" -ge "$STALE_DAYS" ]; then stale+=("${br}|${age}")
  else active+=("${br}|${age}")
  fi
done < <(git for-each-ref --format='%(refname:short)' refs/remotes/origin/ | grep -vE '(^origin$|/HEAD$)')

n_linear=${#linear[@]}; n_sq1=${#squash1[@]}; n_sqN=${#squashN[@]}
n_dead=${#dead[@]}; n_stale=${#stale[@]}; n_active=${#active[@]}

bullets() {
  local -n _arr="$1"
  if [ "${#_arr[@]}" -eq 0 ]; then echo "_none_"; return; fi
  local e
  for e in "${_arr[@]}"; do echo "- \`${e%%|*}\` (${e##*|}d idle)"; done
}

{
  echo "# Stale / merged branch audit"
  echo
  echo "Default branch \`${DEF}\` · thresholds: stale ≥ ${STALE_DAYS}d, dead ≥ ${DEAD_DAYS}d · mode \`${MODE}\`."
  echo "Helps enforce **PR_MERGE_POLICY.md §4** — *delete the PR branch once merged*. Unmerged branches are never auto-deleted."
  echo
  echo "| Category | Count | Action |"
  echo "|----------|------:|--------|"
  echo "| Linear-merged | ${n_linear} | safe to delete |"
  echo "| Squash-merged (single commit) | ${n_sq1} | safe to delete |"
  echo "| Squash-merged (multi-commit) | ${n_sqN} | manual review |"
  echo "| Unmerged — dead (≥ ${DEAD_DAYS}d) | ${n_dead} | ping owner |"
  echo "| Unmerged — stale (${STALE_DAYS}–${DEAD_DAYS}d) | ${n_stale} | ping owner |"
  echo "| Unmerged — active (< ${STALE_DAYS}d) | ${n_active} | keep |"
  echo
  echo "## Merged — safe to delete · linear (${n_linear})"
  bullets linear
  echo
  echo "## Merged — safe to delete · single-commit squash (${n_sq1})"
  bullets squash1
  echo
  echo "## Squash-merged, multi-commit — review before deleting (${n_sqN})"
  if [ "$n_sqN" -eq 0 ]; then
    echo "_none_"
  else
    for e in "${squashN[@]}"; do
      rest="${e#*|}"
      echo "- \`${e%%|*}\` (${rest%%|*}d idle, ${rest##*|} commits)"
    done
  fi
  echo
  echo "## Unmerged — dead, never auto-deleted (${n_dead})"
  bullets dead
  echo
  echo "## Unmerged — stale, never auto-deleted (${n_stale})"
  bullets stale
} >> "$SUMMARY"

# ---- Opt-in deletion (manual dispatch only; merged branches only) ----
if [ "$MODE" = "delete-merged" ] && [ "$EVENT" = "workflow_dispatch" ] && [ "$CONFIRM" = "DELETE" ]; then
  to_delete=()
  for e in "${linear[@]:-}";  do [ -n "$e" ] && to_delete+=("${e%%|*}"); done
  for e in "${squash1[@]:-}"; do [ -n "$e" ] && to_delete+=("${e%%|*}"); done
  if [ "${#to_delete[@]}" -gt 0 ]; then
    echo "Deleting ${#to_delete[@]} merged branches (linear + single-commit squash)..."
    git push origin --delete "${to_delete[@]}"
    {
      echo
      echo "## Deleted ${#to_delete[@]} merged branches"
      printf -- '- `%s`\n' "${to_delete[@]}"
    } >> "$SUMMARY"
  else
    { echo; echo "_delete-merged requested, but no merged branches matched._"; } >> "$SUMMARY"
  fi
elif [ "$MODE" = "delete-merged" ]; then
  {
    echo
    echo "> **delete-merged not executed.** It requires a manual run with \`confirm=DELETE\`; scheduled runs are always report-only."
  } >> "$SUMMARY"
fi

echo "Done. linear=${n_linear} squash1=${n_sq1} squashN=${n_sqN} dead=${n_dead} stale=${n_stale} active=${n_active}"
