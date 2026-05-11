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
require_cmd scp
require_manifest_state "$RELEASE_TAG" "wheels_built"

PLAN_FILE="$(wheel_plan_for_tag "$RELEASE_TAG")"
LOCAL_WHEEL_DIR="$(local_wheel_dir_for_tag "$RELEASE_TAG")"
LOCAL_VERIFY_FILE="$(local_artifact_dir_for_tag "$RELEASE_TAG")/wheel_verification.json"
LOCAL_AOT_VERIFY_FILE="$(local_artifact_dir_for_tag "$RELEASE_TAG")/wheel_aot_verification.json"
mkdir -p "$LOCAL_WHEEL_DIR"

REMOTE_VERIFY_SCRIPT="$(python3 - "$PLAN_FILE" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
plan_json = json.dumps(plan)

print("set -euo pipefail")
print("source /root/miniconda3/etc/profile.d/conda.sh")
print("conda activate batchgen")
print("python3 - <<'PY2'")
print("import glob, json, os, site, subprocess, sys")
print("from pathlib import Path")
print("import torch")
print(f"plan = json.loads({plan_json!r})")
print("wheel_dir = plan['remote_wheel_dir']")
print("log_dir = Path(plan['remote_log_dir'])")
print("log_dir.mkdir(parents=True, exist_ok=True)")
print("matched = []")
print("matched_by_package = []")
print("errors = []")
print("for entry in plan['required_wheels']:")
print("    for pattern in entry['expected_patterns']:")
print("        hits = sorted(glob.glob(os.path.join(wheel_dir, pattern)))")
print("        if len(hits) != 1:")
print("            errors.append(f'pattern {pattern} matched {len(hits)} files: {hits}')")
print("        else:")
print("            matched.append(hits[0])")
print("            matched_by_package.append({'package': entry['package'], 'path': hits[0]})")
print("verification_path = log_dir / 'wheel_verification.json'")
print("verification_data = {'status': 'failed' if errors else 'passed', 'matched_wheels': matched, 'errors': errors}")
print("verification_path.write_text(json.dumps(verification_data, indent=2, sort_keys=True) + '\\n', encoding='utf-8')")
print("if errors:")
print("    for error in errors: print(error, file=sys.stderr)")
print("    sys.exit(1)")
print("aot_path = log_dir / 'wheel_aot_verification.json'")
print("kernel_wheels = [m['path'] for m in matched_by_package if m['package'] == 'batchgen_kernels']")
print("batchgen_wheels = [m['path'] for m in matched_by_package if m['package'] == 'batchgen']")
print("aot_data = {'status': 'skipped', 'reason': 'no batchgen_kernels wheel in plan', 'installed_wheels': []}")
print("if kernel_wheels:")
print("    install_wheels = kernel_wheels + batchgen_wheels")
print("    install_log = log_dir / 'wheel_aot_pip_install.log'")
print("    with install_log.open('w', encoding='utf-8') as f:")
print("        subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', *install_wheels], stdout=f, stderr=subprocess.STDOUT, check=True)")
print("    os.environ.pop('BATCHGEN_KERNELS_DEV', None)")
print("    import batchgen_kernels")
print("    ext = batchgen_kernels.load_extension('batchgen_kernels.moe._C_routing')")
print("    if not hasattr(ext, 'glm5_router_gemm'):")
print("        raise RuntimeError('batchgen_kernels.moe._C_routing missing glm5_router_gemm')")
print("    ext_path = Path(ext.__file__).resolve()")
print("    site_roots = []")
print("    for item in site.getsitepackages() + [site.getusersitepackages()]:")
print("        if item:")
print("            site_roots.append(Path(item).resolve())")
print("    if not any(ext_path == root or root in ext_path.parents for root in site_roots):")
print("        raise RuntimeError(f'AOT extension did not import from site-packages: {ext_path}')")
print("    if 'torch_extensions' in ext_path.parts or '.cache' in ext_path.parts or 'build' in ext_path.parts:")
print("        raise RuntimeError(f'AOT extension path looks like JIT/build output: {ext_path}')")
print("    if not torch.cuda.is_available():")
print("        raise RuntimeError('CUDA is required for glm5_router_gemm_cuda smoke')")
print("    from batchgen.moe.routing import glm5_router_gemm_cuda")
print("    torch.manual_seed(20260511)")
print("    device = torch.device('cuda')")
print("    world_size = 2")
print("    bucket = 4")
print("    hidden = (torch.randn(world_size * bucket, 64, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)")
print("    weight = (torch.randn(32, 64, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)")
print("    rank_counts = torch.tensor([1, 3], device=device, dtype=torch.int64)")
print("    rows = torch.arange(world_size * bucket, device=device)")
print("    valid = (rows % bucket) < rank_counts[rows // bucket]")
print("    hidden_poisoned = hidden.clone()")
print("    hidden_poisoned[~valid] = torch.tensor(float('nan'), device=device, dtype=torch.bfloat16)")
print("    actual = glm5_router_gemm_cuda(hidden_poisoned, weight, rank_token_counts=rank_counts, bucket_size=bucket, world_size=world_size)")
print("    torch.cuda.synchronize()")
print("    if actual.dtype != torch.float32:")
print("        raise RuntimeError(f'router logits dtype mismatch: {actual.dtype}')")
print("    torch.testing.assert_close(actual[~valid], torch.zeros_like(actual[~valid]), atol=0, rtol=0)")
print("    old_tf32 = torch.backends.cuda.matmul.allow_tf32")
print("    torch.backends.cuda.matmul.allow_tf32 = False")
print("    try:")
print("        ref = hidden[valid].contiguous().matmul(weight.t()).float()")
print("    finally:")
print("        torch.backends.cuda.matmul.allow_tf32 = old_tf32")
print("    torch.testing.assert_close(actual[valid], ref, atol=1e-5, rtol=1.6e-2)")
print("    aot_data = {")
print("        'status': 'passed',")
print("        'installed_wheels': install_wheels,")
print("        'extension_file': str(ext_path),")
print("        'has_glm5_router_gemm': True,")
print("        'batchgen_kernels_version': batchgen_kernels.__version__,")
print("        'batchgen_kernels_version_info': list(batchgen_kernels.version_info),")
print("        'smoke': 'glm5_router_gemm_cuda bf16 rank-count invalid-row zeroing',")
print("    }")
print("aot_path.write_text(json.dumps(aot_data, indent=2, sort_keys=True) + '\\n', encoding='utf-8')")
print("print('\\n'.join(matched))")
print("PY2")
PY
)"

REMOTE_MATCHES="$(ssh "$REMOTE_BUILD_MACHINE" "docker exec -i $REMOTE_DOCKER_CONTAINER bash -s" <<< "$REMOTE_VERIFY_SCRIPT")"
[[ -n "$REMOTE_MATCHES" ]] || die "remote wheel verification returned no wheels"

while IFS= read -r remote_file; do
    [[ -n "$remote_file" ]] || continue
    scp "$REMOTE_BUILD_MACHINE:$remote_file" "$LOCAL_WHEEL_DIR/"
done <<< "$REMOTE_MATCHES"

scp "$REMOTE_BUILD_MACHINE:$(remote_log_dir_for_tag "$RELEASE_TAG")/wheel_aot_verification.json" "$LOCAL_AOT_VERIFY_FILE"

python3 - "$PLAN_FILE" "$LOCAL_WHEEL_DIR" "$LOCAL_VERIFY_FILE" "$LOCAL_AOT_VERIFY_FILE" <<'PY'
import glob
import json
import os
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
local_wheel_dir = sys.argv[2]
verify_path = sys.argv[3]
aot_verify_path = sys.argv[4]
errors = []
matched = []

for entry in plan["required_wheels"]:
    for pattern in entry["expected_patterns"]:
        hits = sorted(glob.glob(os.path.join(local_wheel_dir, pattern)))
        if len(hits) != 1:
            errors.append(f"local pattern {pattern} matched {len(hits)} files: {hits}")
        else:
            matched.append(hits[0])

data = {
    "status": "failed" if errors else "passed",
    "local_wheel_dir": local_wheel_dir,
    "matched_wheels": matched,
    "aot_verification": aot_verify_path,
    "errors": errors,
}
Path(verify_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)
PY

VERSION="$(manifest_value "$RELEASE_TAG" release_version)"
BASE_COMMIT="$(manifest_value "$RELEASE_TAG" base_commit)"
SCOPE="$(manifest_value "$RELEASE_TAG" package_scope)"
ARCH="$(manifest_value "$RELEASE_TAG" build_arch)"
PREVIOUS_TAG="$(manifest_value "$RELEASE_TAG" previous_release_tag)"
[[ "$PREVIOUS_TAG" == "None" ]] && PREVIOUS_TAG=""
KERNEL_VERSION="$(manifest_optional_value "$RELEASE_TAG" kernel_release_version)"
write_manifest "$RELEASE_TAG" "$VERSION" "$BASE_COMMIT" "$SCOPE" "$ARCH" "$PREVIOUS_TAG" \
    "wheels_verified" "11_verify_wheels" "true" "true" "$KERNEL_VERSION"

echo "WHEEL VERIFICATION PASSED"
echo "- local wheel dir: $LOCAL_WHEEL_DIR"
echo "- verification: $LOCAL_VERIFY_FILE"
echo "- aot verification: $LOCAL_AOT_VERIFY_FILE"
