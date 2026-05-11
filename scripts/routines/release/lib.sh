#!/bin/bash
set -euo pipefail

RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$RELEASE_SCRIPT_DIR/release_locations.env"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

log() {
    echo "==> $*"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_scope() {
    case "$1" in
        auto|batchgen-only|kernels-only|batchgen-and-kernels|full-dependency-wheels) ;;
        *) die "invalid package_scope: $1" ;;
    esac
}

require_arch() {
    case "$1" in
        sm90a|sm100|all) ;;
        *) die "invalid build_arch: $1" ;;
    esac
}

require_kernel_version() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
        || die "invalid kernel_release_version: $1 (expected X.Y.Z without local arch suffix)"
}

scope_requires_kernel_version() {
    case "$1" in
        kernels-only|batchgen-and-kernels|full-dependency-wheels) return 0 ;;
        *) return 1 ;;
    esac
}

state_dir_for_tag() {
    local tag="$1"
    echo "$LOCAL_ROUTINE_STATE_ROOT/$tag"
}

manifest_for_tag() {
    local tag="$1"
    echo "$(state_dir_for_tag "$tag")/state.json"
}

notes_for_tag() {
    local tag="$1"
    echo "$(state_dir_for_tag "$tag")/release_notes.md"
}

validation_for_tag() {
    local tag="$1"
    echo "$(state_dir_for_tag "$tag")/release_notes_validation.json"
}

wheel_plan_for_tag() {
    local tag="$1"
    echo "$(state_dir_for_tag "$tag")/wheel_plan.json"
}

release_worktree_for_tag() {
    local tag="$1"
    echo "$LOCAL_RELEASE_WORKTREE_PARENT/BatchGen-release-$tag"
}

remote_worktree_for_tag() {
    local tag="$1"
    echo "$REMOTE_RELEASE_WORKTREE_ROOT/BatchGen-release-$tag"
}

remote_artifact_dir_for_tag() {
    local tag="$1"
    echo "$REMOTE_ARTIFACT_ROOT/$tag"
}

remote_wheel_dir_for_tag() {
    local tag="$1"
    echo "$(remote_artifact_dir_for_tag "$tag")/wheels"
}

remote_log_dir_for_tag() {
    local tag="$1"
    echo "$(remote_artifact_dir_for_tag "$tag")/logs"
}

local_artifact_dir_for_tag() {
    local tag="$1"
    echo "$LOCAL_ARTIFACT_ROOT/$tag"
}

local_wheel_dir_for_tag() {
    local tag="$1"
    echo "$(local_artifact_dir_for_tag "$tag")/wheels"
}

write_manifest() {
    local tag="$1"
    local version="$2"
    local base_commit="$3"
    local scope="$4"
    local build_arch="$5"
    local previous_tag="$6"
    local state="$7"
    local current_step="$8"
    local confirmed="$9"
    local notes_confirmed="${10:-false}"
    local kernel_release_version="${11:-}"

    local state_dir
    state_dir="$(state_dir_for_tag "$tag")"
    mkdir -p "$state_dir"

    if [[ -z "$kernel_release_version" && -f "$(manifest_for_tag "$tag")" ]]; then
        kernel_release_version="$(python3 - "$(manifest_for_tag "$tag")" <<'PY' || true
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    data = {}

value = data.get("kernel_release_version")
if value is not None:
    print(value)
PY
)"
    fi

    python3 - "$state_dir/state.json" \
        "$tag" "$version" "$base_commit" "$scope" "$build_arch" "$previous_tag" \
        "$kernel_release_version" "$state" "$current_step" "$confirmed" "$notes_confirmed" \
        "$state_dir" "$LOCAL_CANONICAL_REPO" "$(release_worktree_for_tag "$tag")" \
        "$(remote_worktree_for_tag "$tag")" "$(remote_artifact_dir_for_tag "$tag")" \
        "$(remote_wheel_dir_for_tag "$tag")" "$(remote_log_dir_for_tag "$tag")" \
        "$(local_artifact_dir_for_tag "$tag")" "$(local_wheel_dir_for_tag "$tag")" <<'PY'
import json
import sys

(
    path,
    tag,
    version,
    base_commit,
    scope,
    build_arch,
    previous_tag,
    kernel_release_version,
    state,
    current_step,
    confirmed,
    notes_confirmed,
    state_dir,
    local_repo,
    local_worktree,
    remote_worktree,
    remote_artifact_dir,
    remote_wheel_dir,
    remote_log_dir,
    local_artifact_dir,
    local_wheel_dir,
) = sys.argv[1:]

data = {
    "routine_id": "batchgen-release",
    "release_tag": tag,
    "release_version": version,
    "base_commit": base_commit,
    "package_scope": scope,
    "build_arch": build_arch,
    "previous_release_tag": None if previous_tag == "" else previous_tag,
    "kernel_release_version": None if kernel_release_version == "" else kernel_release_version,
    "state": state,
    "current_step": current_step,
    "confirmed_by_pois": confirmed == "true",
    "release_notes_confirmed_by_pois": notes_confirmed == "true",
    "paths": {
        "state_dir": state_dir,
        "local_canonical_repo": local_repo,
        "local_release_worktree": local_worktree,
        "remote_release_worktree": remote_worktree,
        "remote_artifact_dir": remote_artifact_dir,
        "remote_wheel_dir": remote_wheel_dir,
        "remote_log_dir": remote_log_dir,
        "local_artifact_dir": local_artifact_dir,
        "local_wheel_dir": local_wheel_dir,
    },
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

manifest_optional_value() {
    local tag="$1"
    local key="$2"
    local manifest
    manifest="$(manifest_for_tag "$tag")"
    [[ -f "$manifest" ]] || die "missing release manifest: $manifest"
    python3 - "$manifest" "$key" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

cur = data
for part in sys.argv[2].split("."):
    if not isinstance(cur, dict) or part not in cur:
        print("")
        raise SystemExit(0)
    cur = cur[part]

if cur is None:
    print("")
else:
    print(cur)
PY
}

manifest_value() {
    local tag="$1"
    local key="$2"
    local manifest
    manifest="$(manifest_for_tag "$tag")"
    [[ -f "$manifest" ]] || die "missing release manifest: $manifest"
    python3 - "$manifest" "$key" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

cur = data
for part in sys.argv[2].split("."):
    cur = cur[part]
print(cur)
PY
}

require_manifest_state() {
    local tag="$1"
    local expected="$2"
    local actual
    actual="$(manifest_value "$tag" state)"
    [[ "$actual" == "$expected" ]] || die "release manifest for $tag must be state=$expected, found state=$actual"
}

print_locations() {
    local tag="$1"
    log "Local canonical repo: $LOCAL_CANONICAL_REPO"
    log "Local release worktree: $(release_worktree_for_tag "$tag")"
    log "State dir: $(state_dir_for_tag "$tag")"
    log "Remote build machine: $REMOTE_BUILD_MACHINE"
    log "Remote node: $REMOTE_NODE_NAME"
    log "Remote Docker container: $REMOTE_DOCKER_CONTAINER"
    log "Remote release worktree: $(remote_worktree_for_tag "$tag")"
    log "Remote artifact dir: $(remote_artifact_dir_for_tag "$tag")"
    log "Local artifact dir: $(local_artifact_dir_for_tag "$tag")"
}
