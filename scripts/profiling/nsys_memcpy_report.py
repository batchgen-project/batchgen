#!/usr/bin/env python3
"""Per-memcpy table and pinned-vs-pageable verdict from an Nsight Systems capture.

Input is a ``.nsys-rep`` (exported to SQLite with ``nsys export``) or an
already exported ``.sqlite``. Output:

* one row per CUDA memcpy: bytes, duration, GB/s (1e9 bytes/s), and the
  copy/source/destination kinds resolved to names;
* a summary: counts and bandwidth percentiles per (copyKind, srcKind, dstKind)
  group, and a host-endpoint verdict (ALL_PINNED, ALL_PAGEABLE, MIXED or
  NO_HOST_ENDPOINT).

Enum names are read from the report's own ``ENUM_CUDA_MEM_KIND`` and
``ENUM_CUDA_MEMCPY_OPER`` tables. Nsight Systems renumbers the CUPTI enums
(its table has 0=Pageable, 1=Pinned, 2=Device, while the raw CUPTI enum starts
at 0=Unknown), so a hardcoded id mapping mislabels every row. A report without
those tables, without a memcpy table, or without any selected memcpy row is an
error, never an empty table.

Standard library only. Exit status: 0 success, 2 bad input or report,
3 ``--expect`` not met.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

SUMMARY_SCHEMA = "nsys-memcpy-summary/v1"
MEMCPY_TABLE = "CUPTI_ACTIVITY_KIND_MEMCPY"
MEM_KIND_TABLE = "ENUM_CUDA_MEM_KIND"
MEMCPY_OPER_TABLE = "ENUM_CUDA_MEMCPY_OPER"
REQUIRED_MEMCPY_COLUMNS = (
    "start",
    "end",
    "bytes",
    "copyKind",
    "srcKind",
    "dstKind",
)
OPTIONAL_MEMCPY_COLUMNS = ("deviceId", "streamId", "globalPid")
REQUIRED_ENUM_COLUMNS = ("id", "name", "label")
# Host memory kinds are identified by the enum NAME suffix from the report,
# never by numeric id.
PINNED_SUFFIX = "_PINNED"
PAGEABLE_SUFFIX = "_PAGEABLE"
EXIT_REPORT_ERROR = 2
EXIT_EXPECT_FAILED = 3


class NsysReportError(RuntimeError):
    """A capture or report that cannot produce a trustworthy memcpy table."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass(frozen=True)
class EnumEntry:
    id: int
    name: str
    label: str


@dataclass(frozen=True)
class MemcpyRow:
    index: int
    start_ns: int
    end_ns: int
    bytes: int
    duration_ns: int
    gbps: float | None
    copy_kind: str
    src_kind: str
    dst_kind: str
    copy_kind_id: int
    src_kind_id: int
    dst_kind_id: int
    device_id: int | None
    stream_id: int | None
    global_pid: int | None


def export_sqlite(report: Path, nsys: str) -> Path:
    """Export ``report`` next to itself as ``<stem>.sqlite`` with ``nsys export``."""
    exe = shutil.which(nsys)
    if exe is None:
        raise NsysReportError(
            "NsysNotFound", f"nsys executable not found: {nsys!r} (pass --nsys)"
        )
    target = report.with_suffix(".sqlite")
    cmd = [
        exe,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "-o",
        str(target),
        str(report),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not target.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise NsysReportError(
            "NsysExportFailed",
            f"`{' '.join(cmd)}` rc={proc.returncode}; output tail: {' | '.join(tail)}",
        )
    return target


def resolve_sqlite(path: Path, nsys: str) -> Path:
    if not path.is_file():
        raise NsysReportError("InputMissing", f"no such file: {path}")
    if path.suffix == ".sqlite":
        return path
    if path.suffix == ".nsys-rep":
        return export_sqlite(path, nsys)
    raise NsysReportError(
        "UnsupportedInput",
        f"expected a .nsys-rep or .sqlite file, got {path.name}",
    )


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def load_enum(conn: sqlite3.Connection, table: str) -> dict[int, EnumEntry]:
    if table not in _tables(conn):
        raise NsysReportError(
            "MissingEnumTable",
            f"{table} is absent; kind names cannot be resolved from this report "
            "(re-export with a current nsys; a hardcoded id mapping is not a substitute)",
        )
    cols = _columns(conn, table)
    missing = [c for c in REQUIRED_ENUM_COLUMNS if c not in cols]
    if missing:
        raise NsysReportError(
            "EnumSchemaMismatch", f"{table} columns {cols} lack {missing}"
        )
    entries = {
        int(i): EnumEntry(int(i), str(n), str(lbl))
        for i, n, lbl in conn.execute(f"SELECT id, name, label FROM {table}")
    }
    if not entries:
        raise NsysReportError("EmptyEnumTable", f"{table} has no rows")
    return entries


def _label(
    enum: dict[int, EnumEntry], value: int, table: str, column: str, index: int
) -> str:
    entry = enum.get(int(value))
    if entry is None:
        raise NsysReportError(
            "UnknownEnumValue",
            f"memcpy row {index} {column}={value} is not in {table} ids {sorted(enum)}",
        )
    return entry.label


def load_rows(
    sqlite_path: Path, min_bytes: int = 0
) -> tuple[list[MemcpyRow], dict]:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = _tables(conn)
        if MEMCPY_TABLE not in tables:
            raise NsysReportError(
                "NoMemcpyTable",
                f"{MEMCPY_TABLE} is absent: the capture recorded no CUDA memcpy "
                "(was --trace=cuda on, and did the capture range cover the copies?)",
            )
        mem_kind = load_enum(conn, MEM_KIND_TABLE)
        oper = load_enum(conn, MEMCPY_OPER_TABLE)
        cols = _columns(conn, MEMCPY_TABLE)
        missing = [c for c in REQUIRED_MEMCPY_COLUMNS if c not in cols]
        if missing:
            raise NsysReportError(
                "MemcpySchemaMismatch",
                f"{MEMCPY_TABLE} columns {cols} lack {missing}",
            )
        optional = [c for c in OPTIONAL_MEMCPY_COLUMNS if c in cols]
        select = ", ".join(
            f'"{c}"' for c in (*REQUIRED_MEMCPY_COLUMNS, *optional)
        )
        raw = list(
            conn.execute(
                f'SELECT {select} FROM {MEMCPY_TABLE} ORDER BY "start"'
            )
        )
    finally:
        conn.close()

    pinned_labels = sorted(
        e.label for e in mem_kind.values() if e.name.endswith(PINNED_SUFFIX)
    )
    pageable_labels = sorted(
        e.label for e in mem_kind.values() if e.name.endswith(PAGEABLE_SUFFIX)
    )
    if not pinned_labels or not pageable_labels:
        raise NsysReportError(
            "EnumSchemaMismatch",
            f"{MEM_KIND_TABLE} has no *{PINNED_SUFFIX} / *{PAGEABLE_SUFFIX} entry: "
            f"{[e.name for e in mem_kind.values()]}",
        )
    total_in_table = len(raw)
    if total_in_table == 0:
        raise NsysReportError(
            "NoMemcpyRows", f"{MEMCPY_TABLE} exists but has zero rows"
        )
    rows: list[MemcpyRow] = []
    for index, record in enumerate(raw, 1):
        start, end, nbytes, ck, sk, dk = record[:6]
        extra = dict(zip(optional, record[6:]))
        if int(nbytes) < min_bytes:
            continue
        duration = int(end) - int(start)
        rows.append(
            MemcpyRow(
                index=index,
                start_ns=int(start),
                end_ns=int(end),
                bytes=int(nbytes),
                duration_ns=duration,
                # bytes per nanosecond == GB/s with GB = 1e9 bytes
                gbps=(int(nbytes) / duration) if duration > 0 else None,
                copy_kind=_label(
                    oper, ck, MEMCPY_OPER_TABLE, "copyKind", index
                ),
                src_kind=_label(mem_kind, sk, MEM_KIND_TABLE, "srcKind", index),
                dst_kind=_label(mem_kind, dk, MEM_KIND_TABLE, "dstKind", index),
                copy_kind_id=int(ck),
                src_kind_id=int(sk),
                dst_kind_id=int(dk),
                device_id=extra.get("deviceId"),
                stream_id=extra.get("streamId"),
                global_pid=extra.get("globalPid"),
            )
        )
    if not rows:
        raise NsysReportError(
            "NoMemcpyRows",
            f"{total_in_table} memcpy rows in the report, none with bytes >= {min_bytes}",
        )
    meta = {
        "rows_in_table": total_in_table,
        "mem_kind_enum": {
            e.id: f"{e.name} ({e.label})" for e in mem_kind.values()
        },
        "host_kind_labels": {
            "pinned": pinned_labels,
            "pageable": pageable_labels,
        },
    }
    return rows, meta


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an ascending, non-empty sequence."""
    rank = max(1, math.ceil(q / 100.0 * len(sorted_values)))
    return sorted_values[rank - 1]


def summarize(
    rows: Iterable[MemcpyRow], host_kind_labels: dict[str, list[str]]
) -> dict:
    rows = list(rows)
    groups: dict[tuple[str, str, str], list[MemcpyRow]] = {}
    for row in rows:
        groups.setdefault(
            (row.copy_kind, row.src_kind, row.dst_kind), []
        ).append(row)
    group_out = []
    for (ck, sk, dk), members in sorted(
        groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        rates = sorted(r.gbps for r in members if r.gbps is not None)
        group_out.append(
            {
                "copy_kind": ck,
                "src_kind": sk,
                "dst_kind": dk,
                "count": len(members),
                "total_bytes": sum(r.bytes for r in members),
                "zero_duration": len(members) - len(rates),
                "gbps_min": rates[0] if rates else None,
                "gbps_p50": percentile(rates, 50) if rates else None,
                "gbps_p90": percentile(rates, 90) if rates else None,
                "gbps_max": rates[-1] if rates else None,
            }
        )
    pinned_labels = set(host_kind_labels["pinned"])
    pageable_labels = set(host_kind_labels["pageable"])
    pinned = pageable = host_rows = 0
    for row in rows:
        ends = (row.src_kind, row.dst_kind)
        p = sum(e in pinned_labels for e in ends)
        q = sum(e in pageable_labels for e in ends)
        pinned += p
        pageable += q
        host_rows += 1 if (p or q) else 0
    if host_rows == 0:
        verdict = "NO_HOST_ENDPOINT"
    elif pageable == 0:
        verdict = "ALL_PINNED"
    elif pinned == 0:
        verdict = "ALL_PAGEABLE"
    else:
        verdict = "MIXED"
    return {
        "rows_selected": len(rows),
        "groups": group_out,
        "host_endpoints": {
            "rows_with_host_endpoint": host_rows,
            "pinned": pinned,
            "pageable": pageable,
        },
        "verdict": verdict,
    }


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def render(rows: list[MemcpyRow], summary: dict, max_rows: int) -> str:
    out = [f"NSYS_MEMCPY_ROWS selected={len(rows)}"]
    out.append(
        "idx\tbytes\tdur_ms\tgbps\tcopyKind\tsrcKind\tdstKind\tdeviceId\tglobalPid"
    )
    for row in rows[:max_rows]:
        out.append(
            f"{row.index}\t{row.bytes}\t{row.duration_ns / 1e6:.3f}\t{_fmt(row.gbps)}\t"
            f"{row.copy_kind}({row.copy_kind_id})\t{row.src_kind}({row.src_kind_id})\t"
            f"{row.dst_kind}({row.dst_kind_id})\t{_fmt(row.device_id, 'd')}\t{_fmt(row.global_pid, 'd')}"
        )
    if len(rows) > max_rows:
        out.append(
            f"... {len(rows) - max_rows} more rows (all rows go to --rows-out)"
        )
    out.append("NSYS_MEMCPY_GROUPS")
    out.append(
        "copyKind\tsrcKind\tdstKind\tcount\ttotal_bytes\tgbps_min\tgbps_p50\tgbps_p90\tgbps_max"
    )
    for g in summary["groups"]:
        out.append(
            f"{g['copy_kind']}\t{g['src_kind']}\t{g['dst_kind']}\t{g['count']}\t{g['total_bytes']}\t"
            f"{_fmt(g['gbps_min'])}\t{_fmt(g['gbps_p50'])}\t{_fmt(g['gbps_p90'])}\t{_fmt(g['gbps_max'])}"
        )
    h = summary["host_endpoints"]
    out.append(
        f"NSYS_MEMCPY_VERDICT verdict={summary['verdict']} rows_in_table={summary['rows_in_table']} "
        f"rows_selected={summary['rows_selected']} min_bytes={summary['min_bytes']} "
        f"host_rows={h['rows_with_host_endpoint']} pinned_endpoints={h['pinned']} "
        f"pageable_endpoints={h['pageable']}"
    )
    return "\n".join(out)


def write_rows_tsv(rows: list[MemcpyRow], path: Path) -> None:
    fields = list(asdict(rows[0]).keys())
    with path.open("w") as fh:
        fh.write("\t".join(fields) + "\n")
        for row in rows:
            fh.write(
                "\t".join(
                    "" if v is None else str(v) for v in asdict(row).values()
                )
                + "\n"
            )


def build_report(
    source: Path, nsys: str = "nsys", min_bytes: int = 0
) -> tuple[list[MemcpyRow], dict]:
    sqlite_path = resolve_sqlite(source, nsys)
    rows, meta = load_rows(sqlite_path, min_bytes=min_bytes)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "source": str(source),
        "sqlite": str(sqlite_path),
        "min_bytes": min_bytes,
        "rows_in_table": meta["rows_in_table"],
        "mem_kind_enum": {str(k): v for k, v in meta["mem_kind_enum"].items()},
        **summarize(rows, meta["host_kind_labels"]),
    }
    return rows, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "report", type=Path, help=".nsys-rep or exported .sqlite"
    )
    parser.add_argument(
        "--nsys",
        default="nsys",
        help="nsys executable used to export a .nsys-rep",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=0,
        help="ignore copies smaller than this",
    )
    parser.add_argument(
        "--max-rows", type=int, default=50, help="rows printed to stdout"
    )
    parser.add_argument(
        "--json-out", type=Path, help="write the summary JSON here"
    )
    parser.add_argument(
        "--rows-out", type=Path, help="write every selected row as TSV here"
    )
    parser.add_argument(
        "--expect",
        choices=("pinned", "pageable"),
        help="exit 3 unless every host endpoint of a selected copy has this kind",
    )
    args = parser.parse_args(argv)
    if args.min_bytes < 0 or args.max_rows < 0:
        parser.error("--min-bytes and --max-rows must be non-negative")
    try:
        rows, summary = build_report(args.report, args.nsys, args.min_bytes)
    except NsysReportError as exc:
        print(f"NSYS_MEMCPY_ERROR {exc}", file=sys.stderr)
        return EXIT_REPORT_ERROR
    print(render(rows, summary, args.max_rows))
    if args.rows_out:
        write_rows_tsv(rows, args.rows_out)
        print(f"NSYS_MEMCPY_ROWS_TSV {args.rows_out}")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(f"NSYS_MEMCPY_SUMMARY_JSON {args.json_out}")
    if args.expect:
        wanted = "ALL_PINNED" if args.expect == "pinned" else "ALL_PAGEABLE"
        if summary["verdict"] != wanted:
            print(
                f"NSYS_MEMCPY_EXPECT_FAILED expected={wanted} got={summary['verdict']}",
                file=sys.stderr,
            )
            return EXIT_EXPECT_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
