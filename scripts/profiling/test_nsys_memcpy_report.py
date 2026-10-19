"""Unit tests for nsys_memcpy_report.py over small synthetic SQLite exports.

The fixture mirrors the shape of a real ``nsys export --type sqlite`` report:
the ``CUPTI_ACTIVITY_KIND_MEMCPY`` columns the parser reads and the
``ENUM_CUDA_MEM_KIND`` rows exactly as Nsight Systems emits them (0=Pageable,
1=Pinned, 2=Device, ...). Byte counts and durations are synthetic.

Run: python -m pytest scripts/profiling/test_nsys_memcpy_report.py
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("nsys_memcpy_report.py")
_SPEC = importlib.util.spec_from_file_location(
    "nsys_memcpy_report", _MODULE_PATH
)
report = importlib.util.module_from_spec(_SPEC)
sys.modules["nsys_memcpy_report"] = report
_SPEC.loader.exec_module(report)

GIB = 1 << 30
MIB = 1 << 20

MEM_KIND_ROWS = [
    (0, "CUDA_MEMOPR_MEMORY_KIND_PAGEABLE", "Pageable"),
    (1, "CUDA_MEMOPR_MEMORY_KIND_PINNED", "Pinned"),
    (2, "CUDA_MEMOPR_MEMORY_KIND_DEVICE", "Device"),
    (3, "CUDA_MEMOPR_MEMORY_KIND_ARRAY", "Array"),
    (4, "CUDA_MEMOPR_MEMORY_KIND_MANAGED", "Managed"),
    (5, "CUDA_MEMOPR_MEMORY_KIND_DEVICE_STATIC", "Device Static"),
    (6, "CUDA_MEMOPR_MEMORY_KIND_MANAGED_STATIC", "Managed Static"),
    (7, "CUDA_MEMOPR_MEMORY_KIND_UNKNOWN", "Unknown"),
]
MEMCPY_OPER_ROWS = [
    (0, "CUDA_MEMCPY_OPER_UNKNOWN", "Unknown"),
    (1, "CUDA_MEMCPY_OPER_HTOD", "Host-to-Device"),
    (2, "CUDA_MEMCPY_OPER_DTOH", "Device-to-Host"),
    (8, "CUDA_MEMCPY_OPER_DTOD", "Device-to-Device"),
]
HTOD, DTOH, DTOD = 1, 2, 8
PAGEABLE, PINNED, DEVICE = 0, 1, 2


def make_sqlite(
    path: Path, memcpy_rows, *, mem_kind=True, oper=True, memcpy_table=True
) -> Path:
    """memcpy_rows: (start_ns, end_ns, bytes, copyKind, srcKind, dstKind)."""
    conn = sqlite3.connect(path)
    if mem_kind:
        conn.execute(
            "CREATE TABLE ENUM_CUDA_MEM_KIND (id INTEGER PRIMARY KEY, name TEXT, label TEXT)"
        )
        conn.executemany(
            "INSERT INTO ENUM_CUDA_MEM_KIND VALUES (?, ?, ?)", MEM_KIND_ROWS
        )
    if oper:
        conn.execute(
            "CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER PRIMARY KEY, name TEXT, label TEXT)"
        )
        conn.executemany(
            "INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (?, ?, ?)",
            MEMCPY_OPER_ROWS,
        )
    if memcpy_table:
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER NOT NULL, end INTEGER NOT NULL, "
            "deviceId INTEGER, contextId INTEGER, streamId INTEGER, correlationId INTEGER, "
            "globalPid INTEGER, bytes INTEGER NOT NULL, copyKind INTEGER NOT NULL, "
            "srcKind INTEGER, dstKind INTEGER)"
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY "
            "(start, end, deviceId, contextId, streamId, correlationId, globalPid, bytes, copyKind, srcKind, dstKind) "
            "VALUES (?, ?, 0, 1, 7, ?, 4242, ?, ?, ?, ?)",
            [
                (s, e, i, b, ck, sk, dk)
                for i, (s, e, b, ck, sk, dk) in enumerate(memcpy_rows)
            ],
        )
    conn.commit()
    conn.close()
    return path


# Three-arm control capture: malloc (pageable), cudaMallocHost (pinned), and a
# host mapping without device access (pageable), 1 GiB each.
CONTROL_ROWS = [
    (1_000_000, 41_000_000, GIB, HTOD, PAGEABLE, DEVICE),
    (100_000_000, 110_000_000, GIB, HTOD, PINNED, DEVICE),
    (200_000_000, 240_000_000, GIB, HTOD, PAGEABLE, DEVICE),
]


def importer_rows():
    """Ten 64 MiB pinned H2D chunks plus small sub-threshold copies."""
    rows = [
        (
            i * 10_000_000,
            i * 10_000_000 + 1_000_000 + i * 1_000,
            64 * MIB,
            HTOD,
            PINNED,
            DEVICE,
        )
        for i in range(10)
    ]
    rows.append((500_000_000, 500_001_000, 4096, HTOD, PAGEABLE, DEVICE))
    rows.append((600_000_000, 600_002_000, 8192, DTOH, DEVICE, PAGEABLE))
    return rows


@pytest.fixture
def control_db(tmp_path):
    return make_sqlite(tmp_path / "control.sqlite", CONTROL_ROWS)


@pytest.fixture
def importer_db(tmp_path):
    return make_sqlite(tmp_path / "importer.sqlite", importer_rows())


def test_kind_names_come_from_report_enum_not_cupti_ids(control_db):
    rows, summary = report.build_report(control_db)
    # srcKind id 0 is Pageable and id 1 is Pinned in the nsys enum; a mapping
    # copied from the raw CUPTI enum (1=PAGEABLE, 2=PINNED) would invert this.
    assert [(r.src_kind_id, r.src_kind) for r in rows] == [
        (0, "Pageable"),
        (1, "Pinned"),
        (0, "Pageable"),
    ]
    assert {r.dst_kind for r in rows} == {"Device"}
    assert {r.copy_kind for r in rows} == {"Host-to-Device"}
    assert summary["verdict"] == "MIXED"
    assert summary["host_endpoints"] == {
        "rows_with_host_endpoint": 3,
        "pinned": 1,
        "pageable": 2,
    }
    groups = {(g["src_kind"], g["dst_kind"]): g for g in summary["groups"]}
    assert groups[("Pageable", "Device")]["count"] == 2
    assert groups[("Pinned", "Device")]["count"] == 1
    assert (
        summary["mem_kind_enum"]["1"]
        == "CUDA_MEMOPR_MEMORY_KIND_PINNED (Pinned)"
    )


def test_bandwidth_is_bytes_per_ns(control_db):
    rows, _ = report.build_report(control_db)
    assert rows[1].duration_ns == 10_000_000
    assert rows[1].gbps == pytest.approx(GIB / 10_000_000)


def test_min_bytes_filter_and_all_pinned_verdict(importer_db):
    rows, summary = report.build_report(importer_db, min_bytes=16 * MIB)
    assert len(rows) == 10 and summary["rows_in_table"] == 12
    assert summary["verdict"] == "ALL_PINNED"
    (group,) = summary["groups"]
    assert (group["copy_kind"], group["src_kind"], group["dst_kind"]) == (
        "Host-to-Device",
        "Pinned",
        "Device",
    )
    rates = sorted(r.gbps for r in rows)
    assert group["gbps_min"] == rates[0] and group["gbps_max"] == rates[-1]
    assert group["gbps_p50"] == rates[4] and group["gbps_p90"] == rates[8]
    assert group["total_bytes"] == 10 * 64 * MIB


def test_unfiltered_importer_capture_is_mixed(importer_db):
    _, summary = report.build_report(importer_db)
    assert summary["rows_selected"] == 12
    assert summary["verdict"] == "MIXED"
    assert summary["host_endpoints"]["pageable"] == 2


def test_device_only_copies_have_no_host_endpoint(tmp_path):
    db = make_sqlite(
        tmp_path / "d2d.sqlite", [(0, 1000, MIB, DTOD, DEVICE, DEVICE)]
    )
    _, summary = report.build_report(db)
    assert summary["verdict"] == "NO_HOST_ENDPOINT"


def test_zero_duration_row_is_counted_not_divided(tmp_path):
    db = make_sqlite(
        tmp_path / "z.sqlite",
        [
            (5, 5, MIB, HTOD, PINNED, DEVICE),
            (10, 1010, MIB, HTOD, PINNED, DEVICE),
        ],
    )
    rows, summary = report.build_report(db)
    assert rows[0].gbps is None
    (group,) = summary["groups"]
    assert group["zero_duration"] == 1 and group["gbps_min"] == pytest.approx(
        MIB / 1000
    )


@pytest.mark.parametrize(
    "kwargs, kind",
    [
        ({"mem_kind": False}, "MissingEnumTable"),
        ({"oper": False}, "MissingEnumTable"),
        ({"memcpy_table": False}, "NoMemcpyTable"),
    ],
)
def test_missing_tables_fail_loudly(tmp_path, kwargs, kind):
    db = make_sqlite(tmp_path / "bad.sqlite", CONTROL_ROWS, **kwargs)
    with pytest.raises(report.NsysReportError) as err:
        report.build_report(db)
    assert err.value.kind == kind


def test_empty_capture_fails_loudly(tmp_path):
    db = make_sqlite(tmp_path / "empty.sqlite", [])
    with pytest.raises(report.NsysReportError) as err:
        report.build_report(db)
    assert err.value.kind == "NoMemcpyRows"


def test_filter_that_selects_nothing_fails_loudly(importer_db):
    with pytest.raises(report.NsysReportError) as err:
        report.build_report(importer_db, min_bytes=GIB)
    assert err.value.kind == "NoMemcpyRows"


def test_kind_id_absent_from_enum_fails_loudly(tmp_path):
    db = make_sqlite(tmp_path / "u.sqlite", [(0, 1000, MIB, HTOD, 99, DEVICE)])
    with pytest.raises(report.NsysReportError) as err:
        report.build_report(db)
    assert err.value.kind == "UnknownEnumValue"


def test_enum_without_host_kinds_fails_loudly(tmp_path):
    db = make_sqlite(tmp_path / "e.sqlite", CONTROL_ROWS)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM ENUM_CUDA_MEM_KIND WHERE name LIKE '%_PINNED'")
    conn.commit()
    conn.close()
    with pytest.raises(report.NsysReportError) as err:
        report.build_report(db)
    assert err.value.kind == "EnumSchemaMismatch"


def test_input_validation(tmp_path):
    with pytest.raises(report.NsysReportError) as err:
        report.build_report(tmp_path / "absent.sqlite")
    assert err.value.kind == "InputMissing"
    other = tmp_path / "trace.json"
    other.write_text("{}")
    with pytest.raises(report.NsysReportError) as err:
        report.build_report(other)
    assert err.value.kind == "UnsupportedInput"
    rep = tmp_path / "capture.nsys-rep"
    rep.write_bytes(b"")
    with pytest.raises(report.NsysReportError) as err:
        report.build_report(rep, nsys=str(tmp_path / "no-such-nsys"))
    assert err.value.kind == "NsysNotFound"


def test_cli_writes_summary_and_rows(importer_db, tmp_path, capsys):
    out_json, out_tsv = tmp_path / "summary.json", tmp_path / "rows.tsv"
    rc = report.main(
        [
            str(importer_db),
            "--min-bytes",
            str(16 * MIB),
            "--expect",
            "pinned",
            "--json-out",
            str(out_json),
            "--rows-out",
            str(out_tsv),
            "--max-rows",
            "3",
        ]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "NSYS_MEMCPY_VERDICT verdict=ALL_PINNED" in stdout
    assert "... 7 more rows" in stdout
    summary = json.loads(out_json.read_text())
    assert (
        summary["schema"] == "nsys-memcpy-summary/v1"
        and summary["rows_selected"] == 10
    )
    lines = out_tsv.read_text().splitlines()
    assert len(lines) == 11 and lines[0].startswith("index\tstart_ns")


def test_cli_expect_mismatch_and_error_exit_codes(control_db, tmp_path, capsys):
    assert (
        report.main([str(control_db), "--expect", "pinned"])
        == report.EXIT_EXPECT_FAILED
    )
    assert (
        "NSYS_MEMCPY_EXPECT_FAILED expected=ALL_PINNED got=MIXED"
        in capsys.readouterr().err
    )
    empty = make_sqlite(tmp_path / "empty.sqlite", [])
    assert report.main([str(empty)]) == report.EXIT_REPORT_ERROR
    assert "NSYS_MEMCPY_ERROR NoMemcpyRows" in capsys.readouterr().err


def test_percentile_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0]
    assert report.percentile(values, 50) == 2.0
    assert report.percentile(values, 90) == 4.0
    assert report.percentile([7.0], 50) == 7.0
