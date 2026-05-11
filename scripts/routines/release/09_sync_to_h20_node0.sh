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
require_cmd ssh
require_manifest_state "$RELEASE_TAG" "wheel_plan_created"

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
WORKTREE="$(release_worktree_for_tag "$RELEASE_TAG")"
REMOTE_WORKTREE="$(remote_worktree_for_tag "$RELEASE_TAG")"
BRANCH="tairan/release-$RELEASE_TAG"
LOCAL_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"

print_locations "$RELEASE_TAG"

git -C "$WORKTREE" push origin "HEAD:refs/heads/$BRANCH"

ssh "$REMOTE_BUILD_MACHINE" "docker exec -i $REMOTE_DOCKER_CONTAINER bash -s" <<EOF
set -euo pipefail
REMOTE_WORKTREE="$REMOTE_WORKTREE"
REMOTE_ROOT="$REMOTE_RELEASE_WORKTREE_ROOT"
BRANCH="$BRANCH"
GITHUB_REPO="$GITHUB_REPO"
EXPECTED_SHA="$LOCAL_SHA"

mkdir -p "\$REMOTE_ROOT"
if [ -e "\$REMOTE_WORKTREE" ]; then
    if [ ! -d "\$REMOTE_WORKTREE/.git" ] && [ ! -f "\$REMOTE_WORKTREE/.git" ]; then
        echo "remote path exists but is not a git worktree: \$REMOTE_WORKTREE" >&2
        exit 1
    fi
    if [ -n "\$(git -C "\$REMOTE_WORKTREE" status --porcelain)" ]; then
        echo "remote release worktree is dirty: \$REMOTE_WORKTREE" >&2
        exit 1
    fi
    git -C "\$REMOTE_WORKTREE" fetch origin "\$BRANCH" --quiet
    git -C "\$REMOTE_WORKTREE" checkout "\$BRANCH"
    git -C "\$REMOTE_WORKTREE" reset --hard "origin/\$BRANCH"
else
    git clone --branch "\$BRANCH" "git@github.com:\$GITHUB_REPO.git" "\$REMOTE_WORKTREE"
fi

ACTUAL_SHA="\$(git -C "\$REMOTE_WORKTREE" rev-parse HEAD)"
if [ "\$ACTUAL_SHA" != "\$EXPECTED_SHA" ]; then
    echo "remote SHA mismatch: got \$ACTUAL_SHA expected \$EXPECTED_SHA" >&2
    exit 1
fi
EOF

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "synced_to_remote" "09_sync_to_h20_node0" "true" "true" "$KERNEL_VERSION"

echo "REMOTE SYNC PASSED: $REMOTE_BUILD_MACHINE:$REMOTE_WORKTREE at $LOCAL_SHA"
