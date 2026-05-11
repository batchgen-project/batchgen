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
require_manifest_state "$RELEASE_TAG" "versions_verified"

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
WORKTREE="$(release_worktree_for_tag "$RELEASE_TAG")"

ALLOWED='^(setup\.py|batchgen_kernels/_version\.py|batchgen/kernel_compat\.py|docs/INSTALL\.md|docker/README\.md|CMakeLists\.txt|pyproject\.toml|setup\.cfg)$'
CHANGED="$(git -C "$WORKTREE" diff --name-only)"
[[ -n "$CHANGED" ]] || die "no version changes to commit"
while IFS= read -r file; do
    [[ "$file" =~ $ALLOWED ]] || die "unexpected file in version commit: $file"
done <<< "$CHANGED"

git -C "$WORKTREE" add $CHANGED
git -C "$WORKTREE" commit -m "Release BatchGen $RELEASE_TAG"
COMMIT_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"
echo "$COMMIT_SHA" > "$(state_dir_for_tag "$RELEASE_TAG")/version_commit.txt"

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "version_committed" "06_commit_version_bump" "true" "true" "$KERNEL_VERSION"

echo "VERSION COMMIT CREATED: $COMMIT_SHA"
