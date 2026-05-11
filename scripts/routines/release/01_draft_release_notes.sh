#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

RELEASE_TAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --release-tag) RELEASE_TAG="$2"; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n "$RELEASE_TAG" ]] || die "--release-tag is required"
require_cmd git
require_cmd python3
require_manifest_state "$RELEASE_TAG" "input_confirmed"

BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
if [[ "$PREVIOUS_TAG" == "None" || -z "$PREVIOUS_TAG" ]]; then
    PREVIOUS_TAG="$(git -C "$LOCAL_CANONICAL_REPO" describe --tags --abbrev=0 --match 'v*' "$BASE_COMMIT^" 2>/dev/null || true)"
fi

NOTES_FILE="$(notes_for_tag "$RELEASE_TAG")"
mkdir -p "$(dirname "$NOTES_FILE")"

LOG_RANGE="$BASE_COMMIT"
if [[ -n "$PREVIOUS_TAG" && "$PREVIOUS_TAG" != "None" ]]; then
    LOG_RANGE="$PREVIOUS_TAG..$BASE_COMMIT"
fi

COMMITS_FILE="$(state_dir_for_tag "$RELEASE_TAG")/release_commits.txt"
git -C "$LOCAL_CANONICAL_REPO" log --reverse --pretty=format:'%h %s' "$LOG_RANGE" > "$COMMITS_FILE"

python3 - "$NOTES_FILE" "$RELEASE_TAG" "$VERSION" "$COMMITS_FILE" "$PREVIOUS_TAG" <<'PY'
import re
import sys
from pathlib import Path

notes_path, tag, version, commits_path, previous_tag = sys.argv[1:]
commits = Path(commits_path).read_text(encoding="utf-8").splitlines()

items = []
missing_pr = []
for line in commits:
    match = re.search(r"#(\d+)|\(#(\d+)\)", line)
    pr = None
    if match:
        pr = "#" + next(g for g in match.groups() if g)
    summary = re.sub(r"\s*\(#\d+\)\s*$", "", line)
    summary = re.sub(r"^[0-9a-f]+\s+", "", summary)
    if pr:
        items.append(f"- {summary}. See {pr}.")
    else:
        items.append(f"- TODO: add public PR reference for `{line}`.")
        missing_pr.append(line)

if not items:
    items = ["- TODO: add public change summary and PR reference."]
    missing_pr.append("no commits found")

missing_block = ""
if missing_pr:
    missing_block = "\n<!-- RELEASE-NOTES-TODO: PR references are missing; publish validation must fail until resolved. -->\n"

content = f"""# BatchGen {tag}

## What's New

{chr(10).join(items)}
{missing_block}
## Compatibility and Installation

- Python: unchanged unless noted by linked PRs.
- CUDA/PyTorch: unchanged unless noted by linked PRs.
- Wheels: release assets attached to GitHub release `{tag}`.
- Install: use the wheel assets from the GitHub release page.

## Notes

- This is a formal BatchGen release.
"""

Path(notes_path).write_text(content, encoding="utf-8")
PY

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$(manifest_value "$RELEASE_TAG" package_scope)" \
    "$(manifest_value "$RELEASE_TAG" build_arch)" "$PREVIOUS_TAG" \
    "release_notes_drafted" "01_draft_release_notes" "true" "false" "$KERNEL_VERSION"

cat <<EOF
RELEASE NOTES DRAFTED
- notes: $NOTES_FILE
- commits: $COMMITS_FILE
- previous_release_tag: ${PREVIOUS_TAG:-auto-not-found}
EOF
