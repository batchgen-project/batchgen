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
require_cmd gh
require_cmd python3
require_manifest_state "$RELEASE_TAG" "published"

PLAN_FILE="$(wheel_plan_for_tag "$RELEASE_TAG")"
VIEW_JSON="$(state_dir_for_tag "$RELEASE_TAG")/gh_release_view.json"
gh release view "$RELEASE_TAG" --repo "$GITHUB_REPO" --json isDraft,isPrerelease,assets,body > "$VIEW_JSON"

python3 - "$PLAN_FILE" "$VIEW_JSON" <<'PY'
import glob
import json
import os
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
view = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
errors = []

if view.get("isDraft"):
    errors.append("release is draft")
if view.get("isPrerelease"):
    errors.append("release is prerelease")
asset_names = {asset["name"] for asset in view.get("assets", [])}
expected = set()
for entry in plan["required_wheels"]:
    for pattern in entry["expected_patterns"]:
        hits = sorted(glob.glob(os.path.join(plan["local_wheel_dir"], pattern)))
        if len(hits) != 1:
            errors.append(f"cannot resolve expected local wheel {pattern}: {hits}")
        else:
            expected.add(os.path.basename(hits[0]))

missing = sorted(expected - asset_names)
if missing:
    errors.append(f"missing release assets: {missing}")

if errors:
    for error in errors:
        print(f"FORMAL RELEASE VERIFICATION FAILED: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "release_verified" "13_verify_published_release" "true" "true" "$KERNEL_VERSION"

echo "FORMAL RELEASE VERIFICATION PASSED"
