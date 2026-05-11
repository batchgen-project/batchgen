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
require_cmd python3
require_manifest_state "$RELEASE_TAG" "tagged"

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
WORKTREE="$(release_worktree_for_tag "$RELEASE_TAG")"
PLAN_FILE="$(wheel_plan_for_tag "$RELEASE_TAG")"

DIFF_RANGE="$BASE_COMMIT..$RELEASE_TAG"
if [[ -n "$PREVIOUS_TAG" ]]; then
    DIFF_RANGE="$PREVIOUS_TAG..$RELEASE_TAG"
fi

CHANGED_FILE="$(state_dir_for_tag "$RELEASE_TAG")/changed_files_for_wheels.txt"
git -C "$WORKTREE" diff --name-only "$DIFF_RANGE" > "$CHANGED_FILE"

if [[ "$SCOPE" == "auto" ]]; then
    if grep -Eq '^(scripts/build_wheels\.sh|scripts/install_deps\.sh|requirements.*\.txt|docker/)' "$CHANGED_FILE"; then
        die "auto wheel scope detected dependency/build changes; POIS must rerun with explicit full-dependency-wheels or narrower scope"
    fi
fi

python3 - "$WORKTREE" "$PLAN_FILE" "$RELEASE_TAG" "$VERSION" "$SCOPE" "$ARCH" \
    "$KERNEL_VERSION" \
    "$(remote_worktree_for_tag "$RELEASE_TAG")" "$(remote_wheel_dir_for_tag "$RELEASE_TAG")" \
    "$(remote_log_dir_for_tag "$RELEASE_TAG")" "$(local_wheel_dir_for_tag "$RELEASE_TAG")" "$CHANGED_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

(
    worktree,
    plan_file,
    tag,
    version,
    scope,
    arch,
    kernel_release_version,
    remote_worktree,
    remote_wheel_dir,
    remote_log_dir,
    local_wheel_dir,
    changed_file,
) = sys.argv[1:]

changed = Path(changed_file).read_text(encoding="utf-8").splitlines()

def kernel_version(root: Path) -> str:
    text = (root / "batchgen_kernels" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("cannot read batchgen_kernels __version__")
    return match.group(1)

packages = []
if scope == "batchgen-only":
    packages = ["batchgen"]
elif scope == "kernels-only":
    packages = ["batchgen_kernels"]
elif scope == "batchgen-and-kernels":
    packages = ["batchgen_kernels", "batchgen"]
elif scope == "full-dependency-wheels":
    packages = ["full-dependency-wheels"]
elif scope == "auto":
    kernels_changed = any(p.startswith("batchgen_kernels/") for p in changed)
    batchgen_changed = any(
        p.startswith("batchgen/")
        or p in {"setup.py", "requirements.txt", "MANIFEST.in", "docs/INSTALL.md"}
        for p in changed
    )
    if kernels_changed:
        packages.append("batchgen_kernels")
    if batchgen_changed or not packages:
        packages.append("batchgen")
else:
    raise SystemExit(f"invalid package scope: {scope}")

entries = []
if "full-dependency-wheels" in packages:
    entries.append({
        "package": "full-dependency-wheels",
        "build_command_id": "build_full_dependency_wheels",
        "expected_patterns": ["*.whl"],
    })
else:
    for package in packages:
        if package == "batchgen":
            entries.append({
                "package": "batchgen",
                "version": version,
                "build_command_id": "build_batchgen_only",
                "expected_patterns": [f"batchgen-{version}-py3-none-any.whl"],
            })
        elif package == "batchgen_kernels":
            kv = kernel_version(Path(worktree))
            suffix = "" if arch == "all" else f"+{arch}"
            entries.append({
                "package": "batchgen_kernels",
                "version": kv,
                "build_command_id": "build_kernels_only",
                "expected_patterns": [f"batchgen_kernels-{kv}{suffix}-cp311-cp311-linux_x86_64.whl"],
            })

if not entries:
    raise SystemExit("wheel detection produced no required wheels")

plan = {
    "release_tag": tag,
    "release_version": version,
    "kernel_release_version": kernel_release_version or None,
    "package_scope": scope,
    "build_arch": arch,
    "remote_worktree": remote_worktree,
    "remote_wheel_dir": remote_wheel_dir,
    "remote_log_dir": remote_log_dir,
    "local_wheel_dir": local_wheel_dir,
    "changed_files": changed,
    "required_wheels": entries,
}

Path(plan_file).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "wheel_plan_created" "08_detect_changed_wheels" "true" "true" "$KERNEL_VERSION"

echo "WHEEL PLAN CREATED: $PLAN_FILE"
