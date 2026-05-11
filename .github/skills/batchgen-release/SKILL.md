---
name: batchgen-release
description: BatchGen formal version release routine. Use when POIS asks to release, tag, build wheels, draft release notes, or publish a BatchGen version.
---

# BatchGen Release Skill

This skill is the only approved routine for a BatchGen formal version release.

The routine is fail-closed:

- Do not edit files, commit, tag, build wheels, push, or publish until Step 0 input validation has passed and POIS has explicitly confirmed the normalized input.
- Immediately after input confirmation, draft and validate public release notes. Do not proceed until POIS explicitly confirms the release notes.
- After POIS confirms release notes, the remaining steps are automatic. Do not ask for new decisions unless a step fails.
- If any step fails, stop and alarm POIS. Do not bypass, retry with an alternate workflow, rebuild extra packages, or infer a substitute command.
- This routine creates a formal GitHub release only. Do not create a draft release or prerelease.
- Temporarily disabled: Docker image publication is not part of this routine. If a release tag starts `.github/workflows/docker-release.yml`, cancel that workflow immediately and record Docker as intentionally skipped.

## Required input

Collect these inputs before any tool use that mutates state:

- `base_commit`: required git commit SHA/ref. It must exist and already be reachable from `origin/main`.
- `release_version`: required exact post-release version, e.g. `1.0.9.post6`. It must match `X.X.X.postX`; do not append topical suffixes.
- `release_tag`: optional; defaults to `v{release_version}` and must match that default. It must match `vX.X.X.postX`; do not append topical suffixes such as `-glm5-stability`.
- `package_scope`: required. One of `auto`, `batchgen-only`, `kernels-only`, `batchgen-and-kernels`, `full-dependency-wheels`.
- `build_arch`: optional; defaults to `sm90a`. One of `sm90a`, `sm100`, `all`.
- `kernel_release_version`: required when `package_scope` includes `batchgen_kernels`. Independent `batchgen_kernels` semver, e.g. `0.3.3`; do not include the `+sm90a` wheel local-version suffix.
- `previous_release_tag`: optional; if omitted, scripts detect the latest prior `v*` tag.

There is no `publish_mode` input. This routine always publishes a formal release.

## Fixed locations

The release scripts must use `scripts/routines/release/release_locations.env`.
Default paths are:

- Local canonical repo: `/Users/andrew/BatchGen_Project/BatchGen`
- Local release worktree parent: `/Users/andrew/BatchGen_Project`
- Local release worktree: `/Users/andrew/BatchGen_Project/BatchGen-release-<release_tag>`
- Local routine state: `/Users/andrew/BatchGen_Project/BatchGen/.routine_state/batchgen-release/<release_tag>`
- H20 node0 release worktree: `/data2/tairan/workspace/BatchGen-release-<release_tag>`
- H20 node0 release artifacts: `/data2/tairan/workspace/batchgen-release-artifacts/<release_tag>`
- H20 node0 SSH/container: `wechat_87` / `tairan-batchgen`

Do not infer alternate locations from chat context.

## Step 0: validate and confirm input

Run `scripts/routines/release/00_validate_input.sh` from the canonical repo. This script validates input and writes the routine manifest in `input_validated` state.

Show POIS the normalized input summary and ask for this exact confirmation:

```text
confirm release input <release_tag>
```

After POIS confirms, rerun the validation script with `--confirm-pois` to freeze the input manifest.

## Step 1: draft and confirm public release notes

Run:

1. `scripts/routines/release/01_draft_release_notes.sh`
2. `scripts/routines/release/02_validate_release_notes.sh`

The notes must follow `release_notes_contract.md`. They may include only public information: what's new, PR references, compatibility/install notes, optional highlights, breaking changes, fixes, and public known issues.

Release notes must not include internal machine names, usernames, private paths, raw logs, batch/file IDs, agent/session details, POIS references, debugging chronology, or unapproved benchmark numbers.

Show POIS the validated release notes and ask for this exact confirmation:

```text
confirm release notes <release_tag>
```

After POIS confirms, rerun `02_validate_release_notes.sh` with `--confirm-pois`. After this point, continue automatically.

## Automatic release steps after release-note confirmation

Run the numbered scripts in order:

```text
03_create_release_worktree.sh
04_update_versions.sh
05_verify_versions.sh
06_commit_version_bump.sh
07_tag_release.sh
08_detect_changed_wheels.sh
09_sync_to_h20_node0.sh
10_build_required_wheels.sh
11_verify_wheels.sh
12_publish_formal_release.sh
13_verify_published_release.sh
14_write_release_summary.sh
```

Every script must read the same frozen manifest and fail if the prior step has not completed.

## Forbidden behavior

- Do not run `gh release create` with `--draft` or `--prerelease`.
- Do not run `git push --tags`.
- Do not publish without validated and POIS-confirmed release notes.
- Do not append topical suffixes to release versions, release tags, release names, or release titles. Use exactly `X.X.X.postX` for versions and `vX.X.X.postX` for tags/names; describe release content in release notes only.
- Do not upload wheel globs unless they expand from the verified wheel plan.
- Do not call `scripts/build_wheels.sh` unless `package_scope=full-dependency-wheels`.
- Do not rebuild all wheels because an expected wheel was not found.
- Do not use `scp`, `rsync`, or `cp` to move files into a git repo. Use git push/pull only.
- Do not publish Docker images or wait for DockerHub release workflows.
- Do not use Docker start/restart as a recovery path.

## Temporarily disabled: Docker release workflow

Until POIS explicitly re-enables Docker releases, treat Docker image publication as out of scope for every BatchGen release.

After creating or pushing a `v*` release tag, immediately check for tag-triggered DockerHub workflow runs and cancel them if they are queued or running:

```bash
GH_PAGER=cat gh run list \
    --repo batchgen-project/batchgen \
    --workflow docker-release.yml \
    --branch "v<version>" \
    --limit 5
```

If any matching `Build and Push to DockerHub` run is queued or in progress, cancel it:

```bash
gh run cancel <run-id> \
    --repo batchgen-project/batchgen
```

Record Docker as `intentionally skipped` in the release summary. Do not retry Docker login, add DockerHub credentials, rerun the failed job, or treat Docker workflow failure as blocking the wheel/GitHub release unless POIS explicitly adds Docker back to scope.

## Failure report format

If a step fails, stop and report:

- failed step ID and script
- exact command
- local or remote location
- reason from script output
- state directory
- relevant artifact/log path
- next decision needed from POIS
