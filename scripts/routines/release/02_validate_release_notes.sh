#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

RELEASE_TAG=""
CONFIRM_POIS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --release-tag) RELEASE_TAG="$2"; shift 2 ;;
        --confirm-pois) CONFIRM_POIS=1; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n "$RELEASE_TAG" ]] || die "--release-tag is required"
require_cmd python3

if [[ "$CONFIRM_POIS" -eq 1 ]]; then
    require_manifest_state "$RELEASE_TAG" "release_notes_validated"
else
    require_manifest_state "$RELEASE_TAG" "release_notes_drafted"
fi

NOTES_FILE="$(notes_for_tag "$RELEASE_TAG")"
VALIDATION_FILE="$(validation_for_tag "$RELEASE_TAG")"
[[ -f "$NOTES_FILE" ]] || die "release notes file not found: $NOTES_FILE"

python3 - "$NOTES_FILE" "$VALIDATION_FILE" "$RELEASE_TAG" <<'PY'
import json
import re
import sys
from pathlib import Path

notes_path, validation_path, tag = sys.argv[1:]
text = Path(notes_path).read_text(encoding="utf-8")

errors = []

required = [
    f"# BatchGen {tag}",
    "## What's New",
    "## Compatibility and Installation",
]
for marker in required:
    if marker not in text:
        errors.append(f"missing required section/header: {marker}")

if "RELEASE-NOTES-TODO" in text or "TODO:" in text:
    errors.append("release notes contain TODO placeholders")

if not re.search(r"(?:^|\s)#\d+\b|https://github\.com/[^/]+/[^/]+/pull/\d+", text):
    errors.append("release notes do not reference any PR")

forbidden_patterns = [
    (r"/Users/", "private local path"),
    (r"/data[0-9]+/", "private remote data path"),
    (r"/root/", "private root path"),
    (r"~/(?:\.claude|\.copilot)", "agent memory/session path"),
    (r"\bGH02\b|\bwechat_[0-9]+\b|\bnode[01]\b", "internal machine name"),
    (r"\brank\s+[0-9]+\b", "internal rank detail"),
    (r"\bbatch_[a-f0-9]+\b", "internal batch id"),
    (r"\bfile-[a-f0-9]+\b", "internal file id"),
    (r"\b(?:server|client).*\.log\b", "raw log path/name"),
    (r"\bretry[0-9]+\b", "debug retry chronology"),
    (r"\bPOIS\b", "internal user/context marker"),
    (r"\bCopilot\b|\bagent session\b", "agent/session detail"),
    (r"\b\d+(?:\.\d+)?\s*(?:x|×)\s+faster\b", "unapproved performance number"),
]

for pattern, label in forbidden_patterns:
    if re.search(pattern, text, flags=re.IGNORECASE):
        errors.append(f"forbidden content detected: {label}")

status = "failed" if errors else "passed"
data = {
    "status": status,
    "release_tag": tag,
    "notes": notes_path,
    "errors": errors,
}
Path(validation_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if errors:
    print("RELEASE NOTES VALIDATION FAILED", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    sys.exit(1)
PY

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"

STATE="release_notes_validated"
NOTES_CONFIRMED="false"
if [[ "$CONFIRM_POIS" -eq 1 ]]; then
    STATE="release_notes_confirmed"
    NOTES_CONFIRMED="true"
fi

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "$STATE" "02_validate_release_notes" "true" "$NOTES_CONFIRMED" "$KERNEL_VERSION"

cat <<EOF
RELEASE NOTES VALIDATION PASSED
- notes: $NOTES_FILE
- validation: $VALIDATION_FILE
EOF

if [[ "$CONFIRM_POIS" -eq 0 ]]; then
    cat <<EOF

POIS confirmation required before automatic release steps:
confirm release notes $RELEASE_TAG
EOF
fi
