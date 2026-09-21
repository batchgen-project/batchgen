# Contributing to BatchGen

Thank you for helping improve BatchGen. We welcome model integrations, kernels, scheduler changes, evaluation tooling, deployment documentation, and bug fixes from both users and researchers.

Please discuss a substantial change in an issue before implementing it. For a small documentation or bug-fix change, a focused pull request is usually enough.

## Before you start

1. Search the [issue tracker](https://github.com/batchgen-project/batchgen/issues) and existing pull requests.
2. Explain the problem, the intended behavior, and the affected model or hardware in an issue when the change is non-trivial.
3. Fork the repository, create a feature branch, and keep unrelated formatting or refactoring out of the branch.
4. Read the [PR Merge Policy Contract](PR_MERGE_POLICY.md). It is the binding source for file-scope allowlists, hygiene, review, CI, and merge authority.

## Development setup

Use the same dependency family as the target deployment. For a complete development installation:

```bash
git clone https://github.com/batchgen-project/batchgen.git
cd batchgen
./scripts/install_deps.sh --all
pip install -r requirements-lint.txt
pre-commit install --install-hooks
```

Run formatting and static hooks before opening a pull request:

```bash
pre-commit run -a
```

Do not run performance-sensitive or GPU benchmarks on a development laptop. Run hardware-dependent validation on the matching registered remote machine and record the exact model, commit, GPU topology, workload, baseline version, and timing boundary.

## What to include in a change

### Bug fixes

- Add a focused regression test when possible.
- Describe the symptom, root cause, and validation in the pull request.
- Update the relevant design or troubleshooting documentation when behavior changes.

### Features, models, and kernels

- Include tests and an example or deployment note where appropriate.
- Keep model and kernel changes within the allowlists in `PR_MERGE_POLICY.md`; split a required core/scaffolding change into a separate pull request.
- For a new model, document checkpoint format, precision, expected topology, known limitations, and accuracy status.
- For a performance change, provide before/after measurements from the same workload. Report speedups as “1.5× faster”, not as a slowdown fraction, and do not generalize a single long-context or decode result into a universal ranking.

### Documentation

- Prefer short, runnable examples and link to the [support matrix](docs/support-matrix.md) instead of duplicating model status.
- Check relative links and code blocks locally.
- If a README or guide reports a number, identify whether it is a published paper result or a repository engineering measurement, and include hardware, workload, baseline, and date/commit when available.

## Pull request workflow

1. Rebase or merge the current `main` into your branch as appropriate and keep the diff focused.
2. Run the applicable pre-commit hooks and tests. For docs-only changes, at minimum run the repository hygiene check and `git diff --check`.
3. Fill out the [pull request template](.github/PULL_REQUEST_TEMPLATE.md), select exactly one change type, and explain validation and any known limitations.
4. Request review from the relevant code owners. Respond to review comments with new commits or a clearly explained resolution.
5. The project owner merges after the required approval and green CI. Contributors, including users with write access, should not press the Merge button.

The merge contract is authoritative if this guide and the contract differ.

## Commit messages

Use the Angular-style format:

```text
<type>: <short summary>

<optional body>
```

Use one of `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `test`, or `chore`. Keep the summary concise and explain the motivation in the body for non-trivial changes. A `docs` commit may omit the body; other commit types should include enough context for a reviewer to understand the change.

Examples:

```text
docs: clarify long-context support matrix
```

```text
fix: preserve host KV pages across rank growth

Add a regression test for the cross-rank release and growth boundary.
```

## Questions and discussion

Open an issue for design questions, use the pull request for implementation discussion, and include reproducible commands or logs when reporting a failure. For model or performance reports, start from the exact deployment guide and support matrix entry so that others can reproduce the same topology.
