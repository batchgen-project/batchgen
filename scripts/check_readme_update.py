#!/usr/bin/env python3
"""Check that README News updates preserve history and newest-first order."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


DATE_RE = re.compile(r"(?<!\d)(20\d{2}/(?:0[1-9]|1[0-2]))(?!\d)")


def news_lines(text: str) -> list[str]:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "## News")
    except StopIteration as exc:
        raise ValueError("README is missing a '## News' section") from exc

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    entries = [line.rstrip() for line in lines[start + 1 : end] if line.strip()]
    if not entries:
        raise ValueError("README News section is empty")
    if not any(line.lstrip().startswith("-") for line in entries):
        raise ValueError("README News section has no bullet entries")
    return entries


def entry_dates(lines: list[str]) -> list[str]:
    dates = []
    for line in lines:
        if not line.lstrip().startswith("-"):
            continue
        match = DATE_RE.search(line)
        if not match:
            raise ValueError(f"News bullet has no YYYY/MM date: {line}")
        dates.append(match.group(1))
    return dates


def git_text(ref: str, path: str) -> str:
    try:
        return subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot read {path} at git ref {ref}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--base-ref", help="Git ref whose News history must be retained")
    args = parser.parse_args()

    missing: list[str] = []
    current = Path(args.readme).read_text()
    current_lines = news_lines(current)
    dates = entry_dates(current_lines)
    if dates != sorted(dates, reverse=True):
        raise SystemExit(f"News dates must be newest-first: {dates}")

    if args.base_ref:
        previous_lines = news_lines(git_text(args.base_ref, args.readme))
        missing = []
        position = 0
        for line in previous_lines:
            try:
                position = current_lines.index(line, position) + 1
            except ValueError:
                missing.append(line)
        if missing:
            # A deliberate rewrite (for example consolidating entries) must
            # leave a fresh acknowledgement comment in the News section, so
            # the rewrite is visible in the diff and cannot happen silently.
            ack = re.compile(r"<!--\s*news-rewrite-ack\s+\d{4}/\d{2}:.+-->")
            new_acks = [
                line for line in current_lines
                if ack.search(line) and line not in previous_lines
            ]
            if not new_acks:
                raise SystemExit(
                    "README News dropped or changed a historical line; "
                    "prepend the new item and preserve existing bullets, or "
                    "record a deliberate rewrite with a NEW comment line "
                    "'<!-- news-rewrite-ack YYYY/MM: reason -->' in the News "
                    "section. Dropped or changed lines:\n  "
                    + "\n  ".join(missing)
                )
            print(
                f"README News rewrite acknowledged by {new_acks[0].strip()}; "
                f"{len(missing)} historical line(s) rewritten"
            )

    print(f"README News OK: {len(dates)} entries, newest-first")
    if args.base_ref and not missing:
        print(f"README News history preserved from {args.base_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
