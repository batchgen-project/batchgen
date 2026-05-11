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
require_manifest_state "$RELEASE_TAG" "release_notes_confirmed"

BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
WORKTREE="$(release_worktree_for_tag "$RELEASE_TAG")"
BRANCH="tairan/release-$RELEASE_TAG"

print_locations "$RELEASE_TAG"

if [[ -e "$WORKTREE" ]]; then
    [[ -d "$WORKTREE/.git" || -f "$WORKTREE/.git" ]] || die "release worktree path exists but is not a git worktree: $WORKTREE"
    CURRENT="$(git -C "$WORKTREE" rev-parse HEAD)"
    [[ "$CURRENT" == "$BASE_COMMIT" ]] || die "existing release worktree is at $CURRENT, expected $BASE_COMMIT"
else
    git -C "$LOCAL_CANONICAL_REPO" worktree add "$WORKTREE" -b "$BRANCH" "$BASE_COMMIT"
fi

[[ -z "$(git -C "$WORKTREE" status --porcelain)" ]] || die "release worktree is dirty after creation: $WORKTREE"

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "worktree_created" "03_create_release_worktree" "true" "true" "$KERNEL_VERSION"

echo "RELEASE WORKTREE READY: $WORKTREE"
