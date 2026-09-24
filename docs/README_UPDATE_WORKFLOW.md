# README update workflow

Use this process whenever `README.md` changes.

1. Read [README_TEMPLATE.md](README_TEMPLATE.md) and fetch the current base
   branch before editing.
2. Add the new News bullet directly below `## News`. Keep all previous bullets
   unchanged and in their existing order. Dates use `YYYY/MM` and News remains
   newest-first.
3. Update the performance matrix only when the result has a fixed workload,
   hardware, baseline, timing boundary, and provenance. Use speedup wording
   such as `1.5× faster`.
4. Link deployment details and installation instructions to their canonical
   documents. Avoid copying private paths, machine aliases, credentials, or
   raw runtime logs into README or PR text.
5. Run the preservation check from the repository root:

   ```bash
   python3 scripts/check_readme_update.py --base-ref origin/main
   ```

6. Review the rendered diff. Confirm that the new item is at the top of News,
   the old history is still present, links resolve, and the performance matrix
   and support matrix agree.

The `README Update` GitHub check runs this same command for every pull request
that changes `README.md`. A PR that drops or rewrites an old News bullet fails
the check until the history is restored.
