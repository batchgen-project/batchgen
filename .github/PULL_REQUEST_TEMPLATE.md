## Description
Briefly describe your changes.

## Motivation
Explain why this change is needed and what problem it solves.
If it fixes an issue, link it (e.g., `close #123`).

## Milestone _(optional)_
<!-- The milestone this targets, if any. -->

## Type of Change
<!-- Pick EXACTLY ONE. It fixes the files this PR may touch — see PR_MERGE_POLICY.md §2.5. -->
- [ ] `model` — add/extend model support (`models/**` + registration seam + model kernels only)
- [ ] `kernel` — add/optimize a compute kernel (`batchgen_kernels/**` + in-tree kernel dirs)
- [ ] `core` — change scheduling/serving/runtime scaffolding (the **only** type that may)
- [ ] `fix` — narrow bug fix (+ a regression test)
- [ ] `infra` — build / CI / packaging / scripts / Docker
- [ ] `docs` — documentation only

## File changes
<!-- One row per file. At draft, list the files you intend to touch; keep it current as you push.
     Lets scope be audited against the declared type's allowlist (§2.5) before the diff is reviewed.
     Δ = add / mod / del / mv. -->
| File | Δ | Note |
|------|---|------|
| `path/to/file` | add | why this file |

## Checklist
- [ ] I have read the [CONTRIBUTING](https://github.com/batchgen-project/batchgen/blob/main/CONTRIBUTING.md) guide and the [PR Merge Policy Contract](https://github.com/batchgen-project/batchgen/blob/main/PR_MERGE_POLICY.md).
- [ ] I have updated the tests (if applicable).
- [ ] I have updated the documentation (if applicable).

<!--
Before marking "Ready for review", self-check your change against the author
pre-merge checklist in PR_MERGE_POLICY.md §5 (file hygiene, diff scope, commit
rules, tests). That checklist is an author-local gate — do NOT paste it into this
PR body; the "PR Merge Policy Contract" link above is the reference.
-->
