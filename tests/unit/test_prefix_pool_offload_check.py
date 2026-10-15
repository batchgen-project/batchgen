"""The pooled-wave offload byte-equality check.

Flag `batchgen_debug.prefix_pool_offload_check` arms a recorder for the wave;
every pooled row the rank offloads has its SOURCE bytes digested at
offload-issue time, and after the wave's D2H retires the worker re-reads the
DESTINATION Host pages and compares. The check has to be exact about
destination addressing -- that is the defect class it exists to localize -- so
the Host page read is exercised against real memory through a fake worker view
whose pointers are genuine addresses.

batchgen_worker imports the whole engine, so the flag-plumbing and
verification methods are compiled in isolation against a fake worker, and the
retire-point placement is asserted on the AST (the precedent is
test_prefix_drain_metric.py).
"""

import ast
import copy
import ctypes
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pytest
import torch

from batchgen.prefix_reuse.offload_check import (
    DEBUG_FLAG,
    LOG_TAG,
    PoolOffloadCheckRecorder,
    PoolOffloadRow,
    compare_row_bytes,
    digest_bytes,
    page_digests,
    probe_destination_pages,
    read_host_row_bytes,
    verify_pool_offload_rows,
)

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "batchgen" / "batchgen_worker.py"
KIMI_WRAPPERS = (
    REPO / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "wrappers.py"
)
WORKER_TREE = ast.parse(WORKER.read_text())


# --------------------------------------------------------------------------
# Synthetic destination Host pages: real memory, addressed as the view does.
# --------------------------------------------------------------------------
class _FakeHostView:
    """Host pages behind the worker view's two page-table APIs.

    ``build_page_table`` returns the LOGICAL page ids (an attached prefix chain
    first, then the sequence's private pages), ``k_page_ptr`` maps a page id to
    its real address, and ``get_sequence_layer_page_pointers`` deliberately
    lists only the OWNED pages -- the split the probe must not trip over.

    Page ids are not 0..n, so code that mistook a page ORDINAL for a page id
    would fail instead of silently reading the wrong page.
    """

    def __init__(self, *, page_bytes, attached=0, owned=1, fill=0):
        self.page_bytes = page_bytes
        self.pointer_calls = []
        self.table_calls = []
        self.attached_ids = [700 + index for index in range(attached)]
        self.owned_ids = [900 + index for index in range(owned)]
        self.logical_ids = self.attached_ids + self.owned_ids
        self._pages = {}
        for page_id in self.logical_ids:
            page = ctypes.create_string_buffer(page_bytes)
            ctypes.memset(ctypes.addressof(page), fill, page_bytes)
            self._pages[page_id] = page

    def write_at(self, byte_offset, data):
        """Write across LOGICAL page boundaries, as the range offload does."""
        pos = 0
        while pos < len(data):
            slot, inner = divmod(byte_offset + pos, self.page_bytes)
            take = min(self.page_bytes - inner, len(data) - pos)
            page = self._pages[self.logical_ids[slot]]
            ctypes.memmove(
                ctypes.addressof(page) + inner, data[pos : pos + take], take
            )
            pos += take

    def build_page_table(self, sequence_ids):
        self.table_calls.append(list(sequence_ids))
        return [list(self.logical_ids) for _ in sequence_ids]

    def k_page_ptr(self, layer_idx, page_idx):
        return ctypes.addressof(self._pages[int(page_idx)])

    def get_sequence_layer_page_pointers(self, sequence_id, layer_idx, max_tokens):
        self.pointer_calls.append((sequence_id, layer_idx, max_tokens))
        return (
            [ctypes.addressof(self._pages[page_id]) for page_id in self.owned_ids],
            None,
        )


class _Logger:
    def __init__(self):
        self.errors = []
        self.infos = []

    def error(self, message, *args):
        self.errors.append(message % args)

    def info(self, message, *args):
        self.infos.append(message % args)


def _record(**overrides):
    fields = dict(
        rank=3,
        kind="tail",
        row=2,
        layer_idx=7,
        host_sid=41,
        dest_start_tokens=8,
        token_count=12,
        token_bytes=4,
        page_tokens=4,
        row_digest="",
        page_digests=(),
        view=None,
    )
    fields.update(overrides)
    return PoolOffloadRow(**fields)


def _recorded(source_bytes, **overrides):
    """A record whose digests describe ``source_bytes`` exactly."""
    record = _record(**overrides)
    assert len(source_bytes) == record.num_bytes
    return _record(
        row_digest=digest_bytes(source_bytes),
        page_digests=page_digests(
            source_bytes,
            page_bytes=record.page_bytes,
            start_offset=record.dest_start_tokens * record.token_bytes,
        ),
        **overrides,
    )


# --------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------
def test_digest_is_blake2b_over_the_raw_bytes():
    assert digest_bytes(b"kv") == hashlib.blake2b(b"kv", digest_size=16).hexdigest()


def test_page_digests_use_absolute_destination_page_indices():
    data = bytes(range(32))
    # Aligned start: page 2 of a 16-byte page is the first chunk.
    assert [index for index, _ in page_digests(data, page_bytes=16, start_offset=32)] == [
        2,
        3,
    ]
    # Unaligned start: the first chunk is the remainder of page 0, so the
    # chunk boundaries follow the DESTINATION, not the source length.
    pages = page_digests(data, page_bytes=16, start_offset=4)
    assert [index for index, _ in pages] == [0, 1, 2]
    assert pages[0][1] == digest_bytes(data[:12])
    assert pages[1][1] == digest_bytes(data[12:28])
    assert pages[2][1] == digest_bytes(data[28:])


def test_page_digests_reject_a_non_positive_page():
    with pytest.raises(ValueError, match="page_bytes must be positive"):
        page_digests(b"x", page_bytes=0, start_offset=0)


# --------------------------------------------------------------------------
# Comparison helper
# --------------------------------------------------------------------------
def test_comparison_matches_identical_bytes():
    source = bytes(range(48))
    comparison = compare_row_bytes(_recorded(source), source)
    assert comparison.matched
    assert comparison.first_differing_page is None
    assert comparison.source_digest == comparison.dest_digest == digest_bytes(source)


def test_comparison_names_the_first_differing_destination_page():
    source = bytes(range(48))  # 12 tokens x 4 B, dest starts at token 8
    record = _recorded(source)
    # Pages of 4 tokens: dest tokens 8-19 are pages 2, 3, 4. Corrupt a byte in
    # the middle page so the first differing page is 3, not 2.
    dest = bytearray(source)
    dest[20] ^= 0xFF
    comparison = compare_row_bytes(record, bytes(dest))
    assert not comparison.matched
    assert comparison.first_differing_page == 3
    assert comparison.source_digest == digest_bytes(source)
    assert comparison.dest_digest == digest_bytes(bytes(dest))


def test_comparison_page_index_follows_an_unaligned_destination_start():
    source = bytes(range(48))
    record = _recorded(source, dest_start_tokens=2, page_tokens=8)
    # Pages of 8 tokens from token 2: page 0 holds tokens 2-7 (24 B), page 1
    # holds 8-13 of this row. A byte at row offset 30 is in page 1.
    dest = bytearray(source)
    dest[30] ^= 0xFF
    assert compare_row_bytes(record, bytes(dest)).first_differing_page == 1
    other = bytearray(source)
    other[0] ^= 0xFF
    assert compare_row_bytes(record, bytes(other)).first_differing_page == 0


def test_comparison_reports_a_short_destination_without_hashing_pages():
    source = bytes(range(48))
    comparison = compare_row_bytes(_recorded(source), source[:40])
    assert not comparison.matched
    assert "dest_bytes=40 src_bytes=48" in comparison.detail
    assert comparison.first_differing_page == 2


# --------------------------------------------------------------------------
# Host page read: the LOGICAL table, which is the table the offload walks
# --------------------------------------------------------------------------
def test_host_read_walks_the_logical_table_from_the_destination_start():
    page_bytes = 4 * 4  # page_tokens=4, token_bytes=4
    view = _FakeHostView(page_bytes=page_bytes, owned=8, fill=0xAA)
    payload = bytes(range(48))
    view.write_at(8 * 4, payload)  # destination token 8
    record = _record(view=view)
    assert read_host_row_bytes(record) == payload
    # The logical table is what it asked for, by sequence id.
    assert view.table_calls == [[41]]


def test_host_read_handles_a_row_that_starts_inside_a_page():
    page_bytes = 8 * 4
    view = _FakeHostView(page_bytes=page_bytes, owned=4, fill=0x5A)
    payload = bytes(range(48))
    view.write_at(2 * 4, payload)
    record = _record(view=view, dest_start_tokens=2, page_tokens=8)
    assert read_host_row_bytes(record) == payload


def test_host_read_reaches_pages_the_owned_only_api_cannot_see():
    """A member holding an attached chain: 12 attached + 2 private pages.

    The owned-pages API lists 2, so a probe reading it would stop at page 2;
    the destination lives at logical pages 12-13 and the read must reach it.
    """
    page_bytes = 4 * 4
    view = _FakeHostView(page_bytes=page_bytes, attached=12, owned=2)
    payload = bytes(range(32))
    view.write_at(48 * 4, payload)  # destination token 48 == logical page 12
    record = _record(view=view, dest_start_tokens=48, token_count=8)
    assert len(view.get_sequence_layer_page_pointers(41, 7, None)[0]) == 2
    assert read_host_row_bytes(record) == payload


def test_host_read_hard_fails_only_when_the_logical_table_is_short():
    view = _FakeHostView(page_bytes=16, owned=2)
    record = _record(view=view)
    with pytest.raises(RuntimeError, match="TRUNCATED.*logical page slot 2"):
        read_host_row_bytes(record)


def test_host_read_hard_fails_without_a_logical_page_table_api():
    record = _record(view=object())
    with pytest.raises(RuntimeError, match="exposes no build_page_table"):
        read_host_row_bytes(record)


def test_host_read_hard_fails_without_a_page_pointer_api():
    class _TableOnly:
        def build_page_table(self, sequence_ids):
            return [[700] for _ in sequence_ids]

    record = _record(view=_TableOnly(), dest_start_tokens=0, token_count=4)
    with pytest.raises(RuntimeError, match="exposes no k_page_ptr"):
        read_host_row_bytes(record)


# --------------------------------------------------------------------------
# The api-split discriminator: observation, never a fault
# --------------------------------------------------------------------------
def test_api_split_is_classified_and_the_bytes_still_verify():
    page_bytes = 4 * 4
    view = _FakeHostView(page_bytes=page_bytes, attached=12, owned=2)
    payload = bytes(range(32))
    view.write_at(48 * 4, payload)
    record = _recorded(
        payload, view=view, dest_start_tokens=48, token_count=8, kind="tail",
    )
    probe, page_ids = probe_destination_pages(record)
    assert (probe.logical_pages, probe.backend_pages) == (14, 2)
    assert probe.dest_pages_needed == 14 and probe.classification == "api-split"
    assert page_ids == view.logical_ids
    assert read_host_row_bytes(record, page_ids) == payload

    logger = _Logger()
    result = verify_pool_offload_rows([record], logger=logger, wave_tag="m1s1")
    assert result.matched and result.api_split_rows == 1
    assert logger.errors == []
    split_line = next(line for line in logger.infos if "api-split" in line)
    for fragment in (
        "logical_pages=14", "backend_pages=2", "dest_ordinal_needed=14",
        "host_sid=41", "layer=7", "kind=tail",
    ):
        assert fragment in split_line, split_line
    assert "api_split_rows=1" in logger.infos[-1]


def test_an_owned_only_destination_is_not_flagged_as_split():
    view = _FakeHostView(page_bytes=16, owned=8)
    payload = bytes(range(48))
    view.write_at(8 * 4, payload)
    record = _recorded(payload, view=view)
    probe, _ = probe_destination_pages(record)
    assert probe.classification == "aligned"
    assert (probe.logical_pages, probe.backend_pages, probe.dest_pages_needed) == (
        8, 8, 5,
    )
    logger = _Logger()
    assert verify_pool_offload_rows([record], logger=logger).api_split_rows == 0
    assert [line for line in logger.infos if "api-split" in line] == []


def test_a_logical_table_short_of_the_destination_raises():
    view = _FakeHostView(page_bytes=16, attached=1, owned=1)
    record = _record(view=view, dest_start_tokens=48, token_count=8)
    with pytest.raises(RuntimeError) as excinfo:
        probe_destination_pages(record)
    message = str(excinfo.value)
    for fragment in (
        "TRUNCATED", "sid=41", "layer=7", "logical_pages=2", "backend_pages=1",
        "dest_ordinal_needed=14", "dest_tokens=[48,56)",
    ):
        assert fragment in message, message


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------
def test_recorder_digests_the_source_and_keeps_its_destination():
    recorder = PoolOffloadCheckRecorder(rank=5, page_tokens=4)
    source = torch.arange(24, dtype=torch.int16).reshape(1, 12, 1, 2)
    record = recorder.record_row(
        view="view",
        layer_idx=9,
        row=1,
        kind="segment",
        host_sid=-17,
        dest_start_tokens=0,
        token_count=12,
        source=source,
    )
    raw = ctypes.string_at(source.contiguous().reshape(-1).data_ptr(), 48)
    assert recorder.rows == [record]
    assert record.rank == 5 and record.kind == "segment" and record.host_sid == -17
    assert record.token_bytes == 4 and record.num_bytes == 48
    assert record.page_tokens == 4 and record.page_bytes == 16
    assert record.row_digest == digest_bytes(raw)
    assert [index for index, _ in record.page_digests] == [0, 1, 2]
    assert compare_row_bytes(record, raw).matched


def test_recorder_rejects_metadata_it_cannot_digest():
    recorder = PoolOffloadCheckRecorder(rank=0, page_tokens=4)
    with pytest.raises(ValueError, match="token_count 0"):
        recorder.record_row(
            view=None, layer_idx=0, row=0, kind="tail", host_sid=1,
            dest_start_tokens=0, token_count=0,
            source=torch.zeros(4, dtype=torch.int16),
        )
    with pytest.raises(RuntimeError, match="does not divide into 5 tokens"):
        recorder.record_row(
            view=None, layer_idx=0, row=0, kind="tail", host_sid=1,
            dest_start_tokens=0, token_count=5,
            source=torch.zeros(12, dtype=torch.int16),
        )
    with pytest.raises(ValueError, match="page_tokens must be positive"):
        PoolOffloadCheckRecorder(rank=0, page_tokens=0)


# --------------------------------------------------------------------------
# Verification driver and its log lines
# --------------------------------------------------------------------------
def _two_row_wave():
    """A segment row at token 0 and a tail row at token 8, both in Host pages."""
    segment_pages = _FakeHostView(page_bytes=16, owned=4)
    tail_pages = _FakeHostView(page_bytes=16, owned=8)
    segment_bytes = bytes(range(32))
    tail_bytes = bytes(range(100, 148))
    segment_pages.write_at(0, segment_bytes)
    tail_pages.write_at(8 * 4, tail_bytes)
    records = [
        _recorded(
            segment_bytes, kind="segment", row=0, host_sid=-9, layer_idx=7,
            dest_start_tokens=0, token_count=8, view=segment_pages,
        ),
        _recorded(
            tail_bytes, kind="tail", row=1, host_sid=41, layer_idx=7,
            dest_start_tokens=8, token_count=12, view=tail_pages,
        ),
    ]
    return records, segment_pages, tail_pages


def test_matching_wave_logs_one_summary_line_and_no_error():
    records, _, _ = _two_row_wave()
    logger = _Logger()
    result = verify_pool_offload_rows(records, logger=logger, wave_tag="m2s1")
    assert result.matched and result.mismatches == ()
    assert (result.rows_checked, result.segment_rows, result.tail_rows) == (2, 1, 1)
    assert result.bytes_hashed == 32 + 48
    assert result.layers == (7,)
    assert logger.errors == []
    assert len(logger.infos) == 1
    summary = logger.infos[0]
    assert summary.startswith(LOG_TAG)
    for fragment in (
        "rank=3", "wave=m2s1", "rows_checked=2", "bytes_hashed=80",
        "segment_rows=1", "tail_rows=1", "layers=1", "api_split_rows=0",
        "mismatches=0",
    ):
        assert fragment in summary, summary


def test_mismatching_row_logs_one_localizing_line():
    records, _, tail_pages = _two_row_wave()
    # Corrupt destination page 4 of the tail row (dest tokens 16-19).
    tail_pages.write_at(16 * 4, b"\xff" * 4)
    logger = _Logger()
    result = verify_pool_offload_rows(records, logger=logger, wave_tag="m2s1")
    assert not result.matched
    assert len(result.mismatches) == 1
    record, comparison = result.mismatches[0]
    assert record.row == 1 and comparison.first_differing_page == 4
    assert len(logger.errors) == 1
    line = logger.errors[0]
    for fragment in (
        f"{LOG_TAG} MISMATCH", "rank=3", "kind=tail", "row=1", "host_sid=41",
        "layer=7", "dest_start_tokens=8", "dest_tokens=[8,20)", "token_bytes=4",
        "page_tokens=4", "logical_pages=8", "backend_pages=8",
        "dest_ordinal_needed=5", "page_table=aligned",
        f"src_digest={record.row_digest}",
        f"dst_digest={comparison.dest_digest}", "first_bad_page=4", "wave=m2s1",
    ):
        assert fragment in line, line
    # The summary still lands, so a gate can tell "ran and failed" from
    # "never ran".
    assert "mismatches=1" in logger.infos[0]


def test_empty_wave_still_reports_that_the_check_ran():
    logger = _Logger()
    result = verify_pool_offload_rows([], logger=logger)
    assert result.rows_checked == 0 and result.matched
    assert "rows_checked=0" in logger.infos[0]


# --------------------------------------------------------------------------
# Flag plumbing, compiled out of the worker against a fake
# --------------------------------------------------------------------------
def _worker_method(name):
    return next(
        node for node in ast.walk(WORKER_TREE)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


_NAMESPACE = {"logging": logging, "Optional": Optional}
exec(compile(ast.fix_missing_locations(ast.Module(
    body=[
        copy.deepcopy(_worker_method(name)) for name in (
            "_active_batchgen_debug_for_sequences",
            "_debug_flag_enabled",
            "_new_pool_offload_check_recorder",
            "_verify_prefill_pool_offload_bytes",
        )
    ],
    type_ignores=[],
)), str(WORKER), "exec"), _NAMESPACE)


class _Sequence:
    def __init__(self, debug):
        self.batchgen_debug = debug


class _Plan:
    segments = (object(),)


class _Wave:
    page_tokens = 64
    plan = _Plan()

    def __init__(self, debug):
        self.uuids = ["a", "b"]
        self._debug = debug


class _Worker:
    rank = 2
    _batchgen_debug = None

    _active_batchgen_debug_for_sequences = _NAMESPACE[
        "_active_batchgen_debug_for_sequences"
    ]
    _debug_flag_enabled = _NAMESPACE["_debug_flag_enabled"]
    _new_pool_offload_check_recorder = _NAMESPACE["_new_pool_offload_check_recorder"]
    _verify_prefill_pool_offload_bytes = _NAMESPACE[
        "_verify_prefill_pool_offload_bytes"
    ]

    def __init__(self, debug):
        self._prefill_pool_wave = _Wave(debug)
        sequences = {uuid: _Sequence(debug) for uuid in self._prefill_pool_wave.uuids}
        self.global_batch = type(
            "_Batch", (), {"get_sequence": staticmethod(sequences.get)}
        )()


@pytest.mark.parametrize("value", [True, 1, "1", "true", "on", "YES"])
def test_flag_on_arms_a_recorder_for_the_wave(value):
    worker = _Worker({DEBUG_FLAG: value})
    recorder = worker._new_pool_offload_check_recorder(worker._prefill_pool_wave)
    assert isinstance(recorder, PoolOffloadCheckRecorder)
    assert recorder.rank == 2 and recorder.page_tokens == 64


@pytest.mark.parametrize(
    "debug",
    [None, {}, {"prefix_rank_assignment_mode": "packing_v2"}, {DEBUG_FLAG: False},
     {DEBUG_FLAG: 0}, {DEBUG_FLAG: "off"}],
)
def test_flag_off_arms_nothing(debug):
    worker = _Worker(debug)
    assert worker._new_pool_offload_check_recorder(worker._prefill_pool_wave) is None


def test_verification_is_a_no_op_without_a_recorder(monkeypatch):
    from batchgen.models.wrappers.attention import AttnWrapperBase

    monkeypatch.setattr(AttnWrapperBase, "prefill_pool_offload_recorder", None)
    calls = []
    monkeypatch.setattr(
        "batchgen.prefix_reuse.offload_check.verify_pool_offload_rows",
        lambda *a, **k: calls.append((a, k)),
    )
    _Worker({})._verify_prefill_pool_offload_bytes()
    assert calls == []


def test_verification_consumes_the_recorder_and_tags_the_wave(monkeypatch):
    from batchgen.models.wrappers.attention import AttnWrapperBase

    records, _, _ = _two_row_wave()
    recorder = PoolOffloadCheckRecorder(rank=2, page_tokens=4)
    recorder.rows.extend(records)
    monkeypatch.setattr(
        AttnWrapperBase, "prefill_pool_offload_recorder", recorder, raising=False
    )
    seen = {}

    def _verify(rows, *, logger, wave_tag="-"):
        seen["rows"] = list(rows)
        seen["wave_tag"] = wave_tag
        return None

    monkeypatch.setattr(
        "batchgen.prefix_reuse.offload_check.verify_pool_offload_rows", _verify
    )
    _Worker({})._verify_prefill_pool_offload_bytes()
    assert seen["rows"] == records
    assert seen["wave_tag"] == "m2s1"
    # Consumed: a second wave cannot inherit this wave's rows.
    assert AttnWrapperBase.prefill_pool_offload_recorder is None


# --------------------------------------------------------------------------
# Placement: the check runs at the retire point, and the flag gates the hash
# --------------------------------------------------------------------------
def _call_lines(node, attr):
    return sorted(
        call.lineno for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attr
    )


def test_check_runs_after_the_d2h_retires_and_before_the_wave_is_finished():
    generate = _worker_method("generate")
    retire = _call_lines(generate, "retire_pending_prefill_offloads")
    verify = _call_lines(generate, "_verify_prefill_pool_offload_bytes")
    finish = _call_lines(generate, "_finish_prefill_pool_wave")
    commit = _call_lines(generate, "_commit_prefix_cache_for_prefill")
    assert len(retire) == len(verify) == len(finish) == len(commit) == 1
    assert retire[0] < verify[0] < finish[0] < commit[0]


def test_recorder_install_precedes_the_pool_chunk_loop():
    pool_prefill = _worker_method("_prefill_prepacked_pool")
    install = _call_lines(pool_prefill, "_new_pool_offload_check_recorder")
    builds = _call_lines(pool_prefill, "build")
    assert len(install) == 1 and builds
    assert install[0] < builds[0]


def test_offload_issue_site_hashes_only_behind_the_flag():
    tree = ast.parse(KIMI_WRAPPERS.read_text())
    offload = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_offload_prefill_pool_rows"
    )
    reads = [
        node.lineno for node in ast.walk(offload)
        if isinstance(node, ast.Attribute)
        and node.attr == "prefill_pool_offload_recorder"
    ]
    records = _call_lines(offload, "record_row")
    assert len(reads) == 1 and len(records) == 1
    assert reads[0] < records[0]
    # The record call is reached only when the recorder exists, and the offload
    # itself is outside that branch.
    guards = [
        node for node in ast.walk(offload)
        if isinstance(node, ast.If)
        and _call_lines(node, "record_row") == records
    ]
    assert len(guards) == 1
    assert _call_lines(guards[0], "async_offload_layer_kv_range_to_host") == []
    assert _call_lines(offload, "async_offload_layer_kv_range_to_host")
