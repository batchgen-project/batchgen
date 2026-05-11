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
require_manifest_state "$RELEASE_TAG" "version_committed"

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
WORKTREE="$(release_worktree_for_tag "$RELEASE_TAG")"
HEAD_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"

if git -C "$WORKTREE" rev-parse --verify --quiet "refs/tags/$RELEASE_TAG" >/dev/null; then
    die "tag already exists locally: $RELEASE_TAG"
fi

git -C "$WORKTREE" tag -a "$RELEASE_TAG" -m "BatchGen $RELEASE_TAG"
TAG_SHA="$(git -C "$WORKTREE" rev-list -n 1 "$RELEASE_TAG")"
[[ "$TAG_SHA" == "$HEAD_SHA" ]] || die "tag points to $TAG_SHA, expected $HEAD_SHA"

echo "$TAG_SHA" > "$(state_dir_for_tag "$RELEASE_TAG")/tag_commit.txt"

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "tagged" "07_tag_release" "true" "true" "$KERNEL_VERSION"

echo "TAG CREATED: $RELEASE_TAG -> $TAG_SHA"
