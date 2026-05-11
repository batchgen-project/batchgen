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
require_manifest_state "$RELEASE_TAG" "worktree_created"

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
WORKTREE="$(release_worktree_for_tag "$RELEASE_TAG")"

[[ -d "$WORKTREE" ]] || die "release worktree missing: $WORKTREE"
[[ -z "$(git -C "$WORKTREE" status --porcelain)" ]] || die "release worktree dirty before version update"

DIFF_BASE="$BASE_COMMIT"
if [[ -n "$PREVIOUS_TAG" ]]; then
    DIFF_BASE="$PREVIOUS_TAG..$BASE_COMMIT"
fi

KERNELS_CHANGED=0
if git -C "$WORKTREE" diff --name-only "$DIFF_BASE" | grep -q '^batchgen_kernels/'; then
    KERNELS_CHANGED=1
fi

UPDATE_KERNELS=0
case "$SCOPE" in
    kernels-only|batchgen-and-kernels|full-dependency-wheels) UPDATE_KERNELS=1 ;;
    auto) UPDATE_KERNELS="$KERNELS_CHANGED" ;;
    batchgen-only) UPDATE_KERNELS=0 ;;
esac

if [[ "$UPDATE_KERNELS" -eq 1 ]]; then
    [[ -n "$KERNEL_VERSION" ]] || die "kernel_release_version is required to update batchgen_kernels"
    require_kernel_version "$KERNEL_VERSION"
fi

python3 - "$WORKTREE" "$VERSION" "$UPDATE_KERNELS" "$KERNEL_VERSION" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
version = sys.argv[2]
update_kernels = sys.argv[3] == "1"
kernel_version = sys.argv[4]
changed = []

setup_py = root / "setup.py"
text = setup_py.read_text(encoding="utf-8")
new_text, count = re.subn(
    r'version=os\.getenv\("BATCHGEN_VERSION",\s*"[^"]+"\)',
    f'version=os.getenv("BATCHGEN_VERSION", "{version}")',
    text,
)
if count != 1:
    raise SystemExit("setup.py version expression not found exactly once")
if new_text != text:
    setup_py.write_text(new_text, encoding="utf-8")
    changed.append("setup.py")

if update_kernels:
    if not kernel_version:
        raise SystemExit("kernel_release_version is required when updating kernels")
    version_py = root / "batchgen_kernels" / "_version.py"
    text = version_py.read_text(encoding="utf-8")
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", kernel_version)
    if not match:
        raise SystemExit(f"invalid kernel_release_version: {kernel_version}")
    version_info = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    tuple_text = "(" + ", ".join(repr(x) for x in version_info) + ")"
    text2, c1 = re.subn(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{kernel_version}"', text)
    text2, c2 = re.subn(r"version_info\s*=\s*\([^)]+\)", f"version_info = {tuple_text}", text2)
    if c1 != 1 or c2 != 1:
        raise SystemExit("batchgen_kernels/_version.py fields not found exactly once")
    if text2 != text:
        version_py.write_text(text2, encoding="utf-8")
        changed.append("batchgen_kernels/_version.py")

    compat_py = root / "batchgen" / "kernel_compat.py"
    text = compat_py.read_text(encoding="utf-8")
    text2, count = re.subn(
        r"MIN_KERNELS_VERSION\s*=\s*\([^)]+\)",
        f"MIN_KERNELS_VERSION = {tuple_text}",
        text,
    )
    if count != 1:
        raise SystemExit("batchgen/kernel_compat.py MIN_KERNELS_VERSION not found exactly once")
    if text2 != text:
        compat_py.write_text(text2, encoding="utf-8")
        changed.append("batchgen/kernel_compat.py")

install_md = root / "docs" / "INSTALL.md"
if install_md.exists():
    text = install_md.read_text(encoding="utf-8")
    text2 = re.sub(r"v[0-9]+\.[0-9]+\.[0-9]+(?:\.post[0-9]+)?", f"v{version}", text)
    text2 = re.sub(r"batchgen-[0-9]+\.[0-9]+\.[0-9]+(?:\.post[0-9]+)?-", f"batchgen-{version}-", text2)
    if text2 != text:
        install_md.write_text(text2, encoding="utf-8")
        changed.append("docs/INSTALL.md")

print("\n".join(changed))
PY

CHANGES_FILE="$(state_dir_for_tag "$RELEASE_TAG")/version_changes.txt"
git -C "$WORKTREE" diff --name-only > "$CHANGES_FILE"

[[ -s "$CHANGES_FILE" ]] || die "version update produced no file changes"

write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "versions_updated" "04_update_versions" "true" "true" "$KERNEL_VERSION"

echo "VERSIONS UPDATED"
cat "$CHANGES_FILE"
