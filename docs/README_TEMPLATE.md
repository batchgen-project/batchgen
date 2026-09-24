# README template

Use the repository `README.md` as the rendered project page. This file is the
stable editing template and defines the order of its sections.

```markdown
# BatchGen

## News

- **YYYY/MM — Newest release or result.** One or two sentences with a link to
  the canonical guide or support-matrix row.
- **YYYY/MM — Previous entry.** Preserve every existing entry below the new one.

## What is BatchGen?

One paragraph describing the product and its primary users.

## Selected results

Explain the measurement boundary, then keep directly comparable rows in the
performance matrix. Every number needs a workload, hardware, baseline, and
provenance link or commit.

## Supported models and hardware

Link to `docs/support-matrix.md` and summarize maturity without duplicating the
full matrix.

## Quick start

Installation, deployment, and one minimal batch request.

## Documentation

Link to the canonical guides instead of copying their procedures.

## Roadmap

Keep this short and current.
```

News is an append-only history. Add the newest dated bullet immediately below
`## News`; never replace or reorder older bullets. Run the repository checker
before opening the PR.
