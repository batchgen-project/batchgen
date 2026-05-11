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
require_manifest_state "$RELEASE_TAG" "release_verified"

SUMMARY="$(state_dir_for_tag "$RELEASE_TAG")/summary.md"
MANIFEST="$(manifest_for_tag "$RELEASE_TAG")"
PLAN_FILE="$(wheel_plan_for_tag "$RELEASE_TAG")"

cat > "$SUMMARY" <<EOF
# Internal BatchGen release summary: $RELEASE_TAG

This file is internal routine state. Do not reuse it as public release notes.

## Manifest

- Manifest: $MANIFEST
- Wheel plan: $PLAN_FILE
- Release notes: $(notes_for_tag "$RELEASE_TAG")
- GitHub release view: $(state_dir_for_tag "$RELEASE_TAG")/gh_release_view.json

## Locations

- Local release worktree: $(release_worktree_for_tag "$RELEASE_TAG")
- Local wheel dir: $(local_wheel_dir_for_tag "$RELEASE_TAG")
- Remote release worktree: $(remote_worktree_for_tag "$RELEASE_TAG")
- Remote artifact dir: $(remote_artifact_dir_for_tag "$RELEASE_TAG")
- Remote wheel dir: $(remote_wheel_dir_for_tag "$RELEASE_TAG")
- Remote log dir: $(remote_log_dir_for_tag "$RELEASE_TAG")
EOF

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"

cat >> "$SUMMARY" <<EOF

## Versions

- BatchGen: $VERSION
- batchgen_kernels: ${KERNEL_VERSION:-not applicable}
- Build arch: $ARCH
- AoT wheel verification: $(local_artifact_dir_for_tag "$RELEASE_TAG")/wheel_aot_verification.json
EOF

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "complete" "14_write_release_summary" "true" "true" "$KERNEL_VERSION"

echo "RELEASE SUMMARY WRITTEN: $SUMMARY"
