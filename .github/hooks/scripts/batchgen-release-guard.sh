#!/bin/bash
# PreToolUse(Bash): block unsafe BatchGen release commands outside the
# /batchgen-release routine scripts.
# Exit 0 = allow. Exit 2 = block with stderr message.
set -euo pipefail

INPUT="$(cat)"

python3 - "$INPUT" <<'PY'
import json
import re
import sys

try:
    payload = json.loads(sys.argv[1])
except json.JSONDecodeError:
    sys.exit(0)

tool_name = payload.get("tool_name") or payload.get("toolName") or ""
tool_input = payload.get("tool_input") or payload.get("toolArgs") or {}
if isinstance(tool_input, str):
    try:
        tool_input = json.loads(tool_input)
    except json.JSONDecodeError:
        tool_input = {"command": tool_input}

command = tool_input.get("command", "")
if tool_name not in {"Bash", "bash", "shell"} or not command:
    sys.exit(0)

approved_scripts = (
    "scripts/routines/release/07_tag_release.sh",
    "scripts/routines/release/10_build_required_wheels.sh",
    "scripts/routines/release/12_publish_formal_release.sh",
)

def approved_release_script(cmd: str) -> bool:
    return any(script in cmd for script in approved_scripts)

blocked = []

if re.search(r"(?is)\bgh\s+release\s+create\b", command):
    if "--draft" in command:
        blocked.append("Draft GitHub releases are forbidden for /batchgen-release.")
    if "--prerelease" in command:
        blocked.append("Prereleases are forbidden for /batchgen-release.")
    if not approved_release_script(command):
        blocked.append("Direct gh release create is forbidden; use 12_publish_formal_release.sh.")

if re.search(r"(?is)\bgh\s+release\s+upload\b", command) and not approved_release_script(command):
    blocked.append("Direct gh release upload is forbidden; use verified wheel plan through 12_publish_formal_release.sh.")

if re.search(r"(?is)\bgit\s+push\b[^\n]*--tags\b", command):
    blocked.append("Broad git push --tags is forbidden; push the exact manifest tag only.")

if re.search(r"(?is)\bgit\s+tag\s+-a\b", command) and "scripts/routines/release/07_tag_release.sh" not in command:
    blocked.append("Direct annotated tag creation is forbidden; use 07_tag_release.sh.")

if re.search(r"(?is)\bscripts/build_wheels\.sh\b", command) and "scripts/routines/release/10_build_required_wheels.sh" not in command:
    blocked.append("Direct scripts/build_wheels.sh is forbidden in release routine unless called by 10_build_required_wheels.sh.")

if re.search(r"(?is)\b(?:scp|rsync)\b", command):
    if re.search(r"(?is)(?:/data[0-9]+/[^ ]*/BatchGen|BatchGen-release-|:.*BatchGen)", command):
        blocked.append("scp/rsync into BatchGen git repos or release worktrees is forbidden; use git push/pull.")

if re.search(r"(?is)(^|[;&|]\s*)cp\s+", command):
    if re.search(r"(?is)(?:/data[0-9]+/[^ ]*/BatchGen|BatchGen-release-|/Users/andrew/BatchGen_Project/BatchGen(?:-|/|$))", command):
        blocked.append("cp into BatchGen git repos or release worktrees is forbidden.")

if blocked:
    print("BLOCKED: BatchGen release guard.", file=sys.stderr)
    for reason in blocked:
        print(f"  - {reason}", file=sys.stderr)
    print("Use /batchgen-release routine scripts or stop and ask POIS.", file=sys.stderr)
    sys.exit(2)

sys.exit(0)
PY

