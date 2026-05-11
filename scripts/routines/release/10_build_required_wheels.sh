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
require_cmd python3
require_cmd ssh
require_manifest_state "$RELEASE_TAG" "synced_to_remote"

PLAN_FILE="$(wheel_plan_for_tag "$RELEASE_TAG")"
[[ -f "$PLAN_FILE" ]] || die "missing wheel plan: $PLAN_FILE"

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"

REMOTE_SCRIPT="$(python3 - "$PLAN_FILE" <<'PY'
import json
import shlex
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
remote_worktree = shlex.quote(plan["remote_worktree"])
wheel_dir = shlex.quote(plan["remote_wheel_dir"])
log_dir = shlex.quote(plan["remote_log_dir"])
version = shlex.quote(plan["release_version"])
arch = shlex.quote(plan["build_arch"])

lines = [
    "set -euo pipefail",
    "source /root/miniconda3/etc/profile.d/conda.sh",
    "conda activate batchgen",
    f"mkdir -p {wheel_dir} {log_dir}",
    f"cd {remote_worktree}",
]

for entry in plan["required_wheels"]:
    cmd_id = entry["build_command_id"]
    log = f'{plan["remote_log_dir"]}/{cmd_id}.log'
    if cmd_id == "build_batchgen_only":
        cmd = f"BATCHGEN_VERSION={version} pip wheel . --no-deps -w {wheel_dir}"
    elif cmd_id == "build_kernels_only":
        cmd = f"BUILD_ARCH={arch} pip wheel batchgen_kernels/ --no-build-isolation --no-deps -w {wheel_dir}"
    elif cmd_id == "build_full_dependency_wheels":
        cmd = f"BATCHGEN_VERSION={version} BUILD_ARCH={arch} bash scripts/build_wheels.sh --output-dir {wheel_dir}"
    else:
        raise SystemExit(f"unknown build_command_id: {cmd_id}")
    lines.append(f"({cmd}) 2>&1 | tee {shlex.quote(log)}")

print("\n".join(lines))
PY
)"

ssh "$REMOTE_BUILD_MACHINE" "docker exec -i $REMOTE_DOCKER_CONTAINER bash -s" <<< "$REMOTE_SCRIPT"

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "wheels_built" "10_build_required_wheels" "true" "true" "$KERNEL_VERSION"

echo "WHEEL BUILD COMMANDS COMPLETED"
echo "- remote wheel dir: $(remote_wheel_dir_for_tag "$RELEASE_TAG")"
echo "- remote log dir: $(remote_log_dir_for_tag "$RELEASE_TAG")"
