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
require_cmd gh
require_cmd python3
require_manifest_state "$RELEASE_TAG" "wheels_verified"

WORKTREE="$(release_worktree_for_tag "$RELEASE_TAG")"
BRANCH="tairan/release-$RELEASE_TAG"
NOTES_FILE="$(notes_for_tag "$RELEASE_TAG")"
VALIDATION_FILE="$(validation_for_tag "$RELEASE_TAG")"
PLAN_FILE="$(wheel_plan_for_tag "$RELEASE_TAG")"

[[ -f "$NOTES_FILE" ]] || die "missing release notes: $NOTES_FILE"
[[ -f "$VALIDATION_FILE" ]] || die "missing release notes validation: $VALIDATION_FILE"

python3 - "$VALIDATION_FILE" "$(manifest_for_tag "$RELEASE_TAG")" <<'PY'
import json
import sys
from pathlib import Path

validation = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if validation.get("status") != "passed":
    raise SystemExit("release notes validation did not pass")
if manifest.get("state") != "wheels_verified":
    raise SystemExit("manifest must be in wheels_verified state")
if not manifest.get("release_notes_confirmed_by_pois"):
    raise SystemExit("release notes are not confirmed by POIS")
PY

if gh release view "$RELEASE_TAG" --repo "$GITHUB_REPO" >/dev/null 2>&1; then
    die "GitHub release already exists: $RELEASE_TAG"
fi

mapfile -t LOCAL_WHEELS < <(python3 - "$PLAN_FILE" <<'PY'
import glob
import json
import os
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
files = []
for entry in plan["required_wheels"]:
    for pattern in entry["expected_patterns"]:
        hits = sorted(glob.glob(os.path.join(plan["local_wheel_dir"], pattern)))
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one local wheel for {pattern}, got {hits}")
        files.extend(hits)
for path in files:
    print(path)
PY
)

[[ "${#LOCAL_WHEELS[@]}" -gt 0 ]] || die "no verified local wheels found for upload"

git -C "$WORKTREE" push origin "HEAD:refs/heads/$BRANCH"
git -C "$WORKTREE" push origin "refs/tags/$RELEASE_TAG"

gh release create "$RELEASE_TAG" \
    --repo "$GITHUB_REPO" \
    --title "BatchGen $RELEASE_TAG" \
    --notes-file "$NOTES_FILE" \
    --latest

gh release upload "$RELEASE_TAG" "${LOCAL_WHEELS[@]}" --repo "$GITHUB_REPO"

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "published" "12_publish_formal_release" "true" "true" "$KERNEL_VERSION"

echo "FORMAL RELEASE PUBLISHED: $RELEASE_TAG"
