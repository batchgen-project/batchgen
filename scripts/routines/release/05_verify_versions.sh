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
require_manifest_state "$RELEASE_TAG" "versions_updated"

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
WORKTREE="$(release_worktree_for_tag "$RELEASE_TAG")"

python3 - "$WORKTREE" "$VERSION" "$SCOPE" "$KERNEL_VERSION" <<'PY'
import ast
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
version = sys.argv[2]
scope = sys.argv[3]
kernel_version = sys.argv[4]
errors = []

setup_text = (root / "setup.py").read_text(encoding="utf-8")
if f'version=os.getenv("BATCHGEN_VERSION", "{version}")' not in setup_text:
    errors.append("setup.py default BATCHGEN_VERSION does not match release version")

version_py = root / "batchgen_kernels" / "_version.py"
kernel_file_changed = False
try:
    import subprocess
    changed = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only"],
        text=True,
    ).splitlines()
    kernel_file_changed = "batchgen_kernels/_version.py" in changed
except Exception as exc:
    errors.append(f"failed to inspect changed files: {exc}")

if scope in {"kernels-only", "batchgen-and-kernels", "full-dependency-wheels"} or kernel_file_changed:
    if not kernel_version:
        errors.append("kernel_release_version is required when release scope includes kernels")
    elif not re.fullmatch(r"\d+\.\d+\.\d+", kernel_version):
        errors.append(f"invalid kernel_release_version: {kernel_version}")

    text = version_py.read_text(encoding="utf-8")
    m_ver = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    m_info = re.search(r"version_info\s*=\s*(\([^)]+\))", text)
    if not m_ver or not m_info:
        errors.append("batchgen_kernels/_version.py missing version fields")
    elif kernel_version and m_ver.group(1) != kernel_version:
        errors.append("batchgen_kernels __version__ does not match kernel_release_version")
    else:
        parsed = ast.literal_eval(m_info.group(1))
        if not isinstance(parsed, tuple):
            errors.append("batchgen_kernels version_info is not a tuple")
        elif kernel_version:
            expected = tuple(int(x) for x in kernel_version.split("."))
            if parsed != expected:
                errors.append(f"batchgen_kernels version_info {parsed!r} does not match {expected!r}")

    compat_text = (root / "batchgen" / "kernel_compat.py").read_text(encoding="utf-8")
    m_min = re.search(r"MIN_KERNELS_VERSION\s*=\s*(\([^)]+\))", compat_text)
    if not m_min:
        errors.append("batchgen/kernel_compat.py missing MIN_KERNELS_VERSION")
    elif kernel_version:
        parsed_min = ast.literal_eval(m_min.group(1))
        expected = tuple(int(x) for x in kernel_version.split("."))
        if not isinstance(parsed_min, tuple):
            errors.append("MIN_KERNELS_VERSION is not a tuple")
        elif parsed_min < expected:
            errors.append(f"MIN_KERNELS_VERSION {parsed_min!r} is lower than kernel_release_version {expected!r}")
        elif parsed_min != expected:
            errors.append(f"MIN_KERNELS_VERSION {parsed_min!r} does not exactly match kernel_release_version {expected!r}")

if "TODO" in setup_text:
    errors.append("setup.py contains TODO after version update")

if errors:
    for error in errors:
        print(f"VERSION VERIFICATION FAILED: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "versions_verified" "05_verify_versions" "true" "true" "$KERNEL_VERSION"

echo "VERSION VERIFICATION PASSED"
