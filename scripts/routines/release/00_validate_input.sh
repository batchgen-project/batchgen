#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

BASE_COMMIT=""
RELEASE_VERSION=""
RELEASE_TAG=""
PACKAGE_SCOPE=""
BUILD_ARCH="sm90a"
PREVIOUS_RELEASE_TAG=""
KERNEL_RELEASE_VERSION=""
CONFIRM_POIS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-commit) BASE_COMMIT="$2"; shift 2 ;;
        --release-version) RELEASE_VERSION="$2"; shift 2 ;;
        --release-tag) RELEASE_TAG="$2"; shift 2 ;;
        --package-scope) PACKAGE_SCOPE="$2"; shift 2 ;;
        --build-arch) BUILD_ARCH="$2"; shift 2 ;;
        --previous-release-tag) PREVIOUS_RELEASE_TAG="$2"; shift 2 ;;
        --kernel-release-version) KERNEL_RELEASE_VERSION="$2"; shift 2 ;;
        --confirm-pois) CONFIRM_POIS=1; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n "$BASE_COMMIT" ]] || die "--base-commit is required"
[[ -n "$RELEASE_VERSION" ]] || die "--release-version is required"
[[ -n "$PACKAGE_SCOPE" ]] || die "--package-scope is required"
[[ -n "$RELEASE_TAG" ]] || RELEASE_TAG="v$RELEASE_VERSION"

require_scope "$PACKAGE_SCOPE"
require_arch "$BUILD_ARCH"
if [[ -n "$KERNEL_RELEASE_VERSION" ]]; then
    require_kernel_version "$KERNEL_RELEASE_VERSION"
fi
require_cmd git
require_cmd python3

[[ "$RELEASE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.post[0-9]+$ ]] \
    || die "invalid release version: $RELEASE_VERSION (expected X.X.X.postX with no suffix)"
[[ "$RELEASE_TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+\.post[0-9]+$ ]] \
    || die "invalid release tag: $RELEASE_TAG (expected vX.X.X.postX with no suffix)"
[[ "$RELEASE_TAG" == "v$RELEASE_VERSION" ]] \
    || die "release tag must equal v{release_version}; got $RELEASE_TAG for $RELEASE_VERSION"

print_locations "$RELEASE_TAG"

[[ -d "$LOCAL_CANONICAL_REPO/.git" || -f "$LOCAL_CANONICAL_REPO/.git" ]] \
    || die "local canonical repo not found: $LOCAL_CANONICAL_REPO"

REPO_ROOT="$(git -C "$LOCAL_CANONICAL_REPO" rev-parse --show-toplevel)"
[[ "$REPO_ROOT" == "$LOCAL_CANONICAL_REPO" ]] \
    || die "canonical repo mismatch: expected $LOCAL_CANONICAL_REPO, got $REPO_ROOT"

git -C "$LOCAL_CANONICAL_REPO" fetch origin "$MAIN_BRANCH" --tags --quiet

git -C "$LOCAL_CANONICAL_REPO" cat-file -e "$BASE_COMMIT^{commit}" \
    || die "base_commit does not exist: $BASE_COMMIT"

BASE_COMMIT_FULL="$(git -C "$LOCAL_CANONICAL_REPO" rev-parse "$BASE_COMMIT^{commit}")"

git -C "$LOCAL_CANONICAL_REPO" merge-base --is-ancestor "$BASE_COMMIT_FULL" "origin/$MAIN_BRANCH" \
    || die "base_commit is not reachable from origin/$MAIN_BRANCH: $BASE_COMMIT_FULL"

if git -C "$LOCAL_CANONICAL_REPO" rev-parse --verify --quiet "refs/tags/$RELEASE_TAG" >/dev/null; then
    die "release tag already exists locally: $RELEASE_TAG"
fi

if git -C "$LOCAL_CANONICAL_REPO" ls-remote --exit-code --tags origin "refs/tags/$RELEASE_TAG" >/dev/null 2>&1; then
    die "release tag already exists on origin: $RELEASE_TAG"
fi

if [[ -n "$(git -C "$LOCAL_CANONICAL_REPO" status --porcelain)" ]]; then
    die "canonical repo has dirty or untracked files; clean it before release: $LOCAL_CANONICAL_REPO"
fi

if [[ -z "$PREVIOUS_RELEASE_TAG" ]]; then
    PREVIOUS_RELEASE_TAG="$(git -C "$LOCAL_CANONICAL_REPO" describe --tags --abbrev=0 --match 'v*' "$BASE_COMMIT_FULL^" 2>/dev/null || true)"
fi

KERNEL_VERSION_REQUIRED=0
if scope_requires_kernel_version "$PACKAGE_SCOPE"; then
    KERNEL_VERSION_REQUIRED=1
elif [[ "$PACKAGE_SCOPE" == "auto" ]]; then
    if [[ -n "$PREVIOUS_RELEASE_TAG" ]]; then
        if git -C "$LOCAL_CANONICAL_REPO" diff --name-only "$PREVIOUS_RELEASE_TAG..$BASE_COMMIT_FULL" | grep -q '^batchgen_kernels/'; then
            KERNEL_VERSION_REQUIRED=1
        fi
    elif git -C "$LOCAL_CANONICAL_REPO" diff-tree --no-commit-id --name-only -r "$BASE_COMMIT_FULL" | grep -q '^batchgen_kernels/'; then
        KERNEL_VERSION_REQUIRED=1
    fi
fi

if [[ "$KERNEL_VERSION_REQUIRED" -eq 1 && -z "$KERNEL_RELEASE_VERSION" ]]; then
    die "--kernel-release-version is required when package_scope includes batchgen_kernels"
fi

if [[ "$PACKAGE_SCOPE" == "batchgen-only" && -n "$KERNEL_RELEASE_VERSION" ]]; then
    die "--kernel-release-version is invalid with package_scope=batchgen-only"
fi

STATE="input_validated"
CONFIRMED="false"
if [[ "$CONFIRM_POIS" -eq 1 ]]; then
    STATE="input_confirmed"
    CONFIRMED="true"
fi

write_manifest "$RELEASE_TAG" "$RELEASE_VERSION" "$BASE_COMMIT_FULL" "$PACKAGE_SCOPE" \
    "$BUILD_ARCH" "$PREVIOUS_RELEASE_TAG" "$STATE" "00_validate_input" "$CONFIRMED" "false" \
    "$KERNEL_RELEASE_VERSION"

VALIDATION_FILE="$(state_dir_for_tag "$RELEASE_TAG")/input_validation.json"
python3 - "$VALIDATION_FILE" "$BASE_COMMIT_FULL" "$RELEASE_VERSION" "$RELEASE_TAG" \
    "$PACKAGE_SCOPE" "$BUILD_ARCH" "$PREVIOUS_RELEASE_TAG" "$KERNEL_RELEASE_VERSION" \
    "$KERNEL_VERSION_REQUIRED" "$CONFIRMED" <<'PY'
import json
import sys

(
    path,
    base_commit,
    version,
    tag,
    scope,
    arch,
    previous,
    kernel_version,
    kernel_version_required,
    confirmed,
) = sys.argv[1:]

data = {
    "status": "passed",
    "base_commit": base_commit,
    "release_version": version,
    "release_tag": tag,
    "package_scope": scope,
    "build_arch": arch,
    "previous_release_tag": previous or None,
    "kernel_release_version": kernel_version or None,
    "confirmed_by_pois": confirmed == "true",
    "checks": {
        "base_commit_exists": True,
        "base_commit_reachable_from_origin_main": True,
        "tag_absent_local": True,
        "tag_absent_origin": True,
        "worktree_clean": True,
        "formal_release_only": True,
        "kernel_release_version_required": kernel_version_required == "1",
        "kernel_release_version_valid": bool(kernel_version) if kernel_version_required == "1" else True,
    },
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY

cat <<EOF
VALIDATION PASSED

Normalized /batchgen-release input:
- base_commit: $BASE_COMMIT_FULL
- release_version: $RELEASE_VERSION
- release_tag: $RELEASE_TAG
- package_scope: $PACKAGE_SCOPE
- build_arch: $BUILD_ARCH
- kernel_release_version: ${KERNEL_RELEASE_VERSION:-not-applicable}
- previous_release_tag: ${PREVIOUS_RELEASE_TAG:-auto-not-found}
- release_mode: formal GitHub release, not draft
- state_dir: $(state_dir_for_tag "$RELEASE_TAG")
- local_release_worktree: $(release_worktree_for_tag "$RELEASE_TAG")
- remote_release_worktree: $(remote_worktree_for_tag "$RELEASE_TAG")
- remote_artifact_dir: $(remote_artifact_dir_for_tag "$RELEASE_TAG")
EOF

if [[ "$CONFIRM_POIS" -eq 0 ]]; then
    cat <<EOF

POIS confirmation required before any mutation:
confirm release input $RELEASE_TAG
EOF
fi
