# BatchGen Release Notes Contract

Release notes are public-facing GitHub release notes. They are not an internal
debug summary, benchmark log, or agent handoff.

## Required sections

1. `# BatchGen <release_tag>`
2. `## What's New`
3. `## Compatibility and Installation`

## Optional sections

- `## Breaking Changes`
- `## Fixes`
- `## Known Issues`
- `## Notes`

## Required content

- User-facing changes must reference PRs when available, using `#<number>` or a
  GitHub PR URL.
- The install section must reference public release assets or a public install
  command.
- Compatibility notes should mention public platform constraints only, such as
  Python, CUDA, PyTorch, or public GPU architecture names.

## Allowed content

- What's new for users.
- Bug fixes and user impact.
- Breaking changes and public migration notes.
- Compatibility and installation notes.
- Public known issues and public workarounds.
- Optional highlights worth knowing in this release.

## Forbidden content

Do not include:

- Internal machine names, SSH aliases, runner names, or private hostnames.
- Usernames, private paths, workspace paths, cache paths, or local paths.
- Raw server/client logs, tracebacks, batch IDs, file IDs, or run IDs.
- Agent/session details, memory paths, Copilot session state, or POIS references.
- Debugging chronology such as retry numbers, node/rank details, or incident
  narrative.
- Unapproved benchmark numbers, accuracy numbers, speedups, or latency claims.
- Secrets, tokens, private URLs, internal ports, or unreleased private details.

## Failure policy

If release notes validation fails, stop and alarm POIS. Do not publish the
release and do not bypass the validator.

