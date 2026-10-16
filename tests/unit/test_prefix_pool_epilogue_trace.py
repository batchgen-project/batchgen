"""``batchgen_debug.prefix_pool_epilogue_trace``: the flag, the shard
arithmetic and the formatted line.

The pooled prefill's epilogue - the tail-row gather, the streamed-SP8 shard
reassembly and the row -> wave index -> uuid -> sequence pairing - is the one
stretch of the path no oracle or GPU replay reached, and first-token parity is
the argmax of exactly what comes out of it. Pinned here: the flag parser, the
shard split against the production ``balanced_row_split``, the owner lookup
(including the out-of-range case the production all-zero canary cannot see),
the line format, the per-chunk reset, and that both arming seams exist in the
worker and in the gather.

CPU only; no CUDA, no model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, *relative):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = _ROOT.joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_TRACE = _load("test_epilogue_trace_mod", "batchgen", "prefix_reuse",
               "epilogue_trace.py")
# The production split the trace must agree with.
_RESHARD = _load("test_epilogue_reshard_mod", "batchgen", "models",
                 "moonshotai", "kimi_linear", "moe_tp_reshard.py")


# --------------------------------------------------------------------------- #
#  (i) The flag
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    (True, True), (1, True), ("1", True), ("on", True), ("TRUE", True),
    (" yes ", True),
    (None, False), (False, False), ("", False), ("0", False), ("off", False),
    ("no", False), (0, False),
])
def test_the_flag_parser_accepts_exactly_the_documented_values(value, expected):
    assert _TRACE.parse_trace_flag(value) is expected


@pytest.mark.parametrize("value", ["maybe", "trace", [], {}, "2x"])
def test_an_unsupported_flag_value_is_refused_by_name(value):
    """A mistyped flag must not read as a clean, untraced run."""

    with pytest.raises(ValueError, match="prefix_pool_epilogue_trace"):
        _TRACE.parse_trace_flag(value)


# --------------------------------------------------------------------------- #
#  (ii) The shard arithmetic, against the production split
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,size", [
    (22480, 8), (22096, 8), (1, 8), (7, 8), (8, 8), (9, 8), (63, 4), (100, 1),
])
def test_the_shard_matches_the_production_row_split(rows, size):
    """``balanced_shard`` is a transcription; it has to agree exactly."""

    expected = _RESHARD.balanced_row_split(rows, size)
    got = [_TRACE.balanced_shard(rows, size, rank) for rank in range(size)]
    assert got == expected


@pytest.mark.parametrize("rows,size", [(22480, 8), (22096, 8), (9, 8), (63, 4)])
def test_every_row_has_exactly_one_owner(rows, size):
    """The split is a partition, so a row in range cannot be double-owned.

    This is the refutation of the "two ranks both own a tail row" hypothesis:
    the all-reduce adds one contribution per row, never two.
    """

    shards = [_TRACE.balanced_shard(rows, size, rank) for rank in range(size)]
    for row in range(rows):
        owners = [rank for rank, (start, end) in enumerate(shards)
                  if start <= row < end]
        assert owners == [_TRACE.owner_of_row(row, rows, size)]
        assert len(owners) == 1


@pytest.mark.parametrize("row", [-1, 22480, 99999])
def test_an_out_of_range_row_has_no_owner(row):
    """The silent case: no rank owns it, so no rank's canary inspects it."""

    assert _TRACE.owner_of_row(row, 22480, 8) == -1


def test_the_live_victims_are_shard_interior():
    """Measured off the rank0/rank8 schema-2 dumps.

    The five corrupted members' last-token rows sit far from every shard
    boundary, while the rows that ARE boundary-adjacent are clean - which is
    why a shard-offset or zero-fill defect does not predict this victim set.
    """

    # (last token row, expected owner, distance to the nearest shard edge)
    group0 = {"ch-001": (837, 0, 837), "ch-004": (1909, 0, 900),
              "ch-055": (19741, 7, 71), "ch-063": (22479, 7, 0)}
    for label, (row, owner, distance) in group0.items():
        start, end = _TRACE.balanced_shard(22480, 8, owner)
        assert _TRACE.owner_of_row(row, 22480, 8) == owner, label
        assert min(row - start, end - 1 - row) == distance, label
    group1 = {"ch-003": (1540, 0, 1221), "ch-011": (4326, 1, 1197),
              "ch-013": (5016, 1, 507), "ch-054": (19293, 6, 40)}
    for label, (row, owner, distance) in group1.items():
        start, end = _TRACE.balanced_shard(22096, 8, owner)
        assert _TRACE.owner_of_row(row, 22096, 8) == owner, label
        assert min(row - start, end - 1 - row) == distance, label


# --------------------------------------------------------------------------- #
#  (iii) The formatted line and the recorder
# --------------------------------------------------------------------------- #
def test_the_line_carries_every_field_the_discrimination_needs():
    line = _TRACE.format_row(
        rank=3, attn_tp_rank=3, attn_tp_size=8, chunk=0, table_row=1,
        tail_row=837, wave_index=0, global_idx=65, uuid="abc-123",
        num_global_rows=22480, shard_start=8430, shard_end=11240, owner=0,
        owned_here=False, owned_count=4, pre_norm=0.0, post_norm=12.5,
        canary_zero_rows=0, token=4242,
    )
    assert line.startswith("[EPILOGUE_TRACE] ")
    for fragment in ("rank=3", "tp=3/8", "chunk=0", "row=1", "tail_row=837",
                     "wave=0", "global_idx=65", "uuid=abc-123",
                     "rows=22480", "shard=[8430,11240)", "owner=0",
                     "owned_here=0", "owned_count=4", "canary_zero=0",
                     "token=4242"):
        assert fragment in line, fragment
    assert "pre_l2=0.000000e+00" in line and "post_l2=1.250000e+01" in line


def test_a_missing_number_reads_as_na_rather_than_zero():
    """An untraced gather must not look like a zero norm."""

    line = _TRACE.format_row(
        rank=0, attn_tp_rank=0, attn_tp_size=1, chunk=0, table_row=0,
        tail_row=5, wave_index=0, global_idx=64, uuid="u", num_global_rows=0,
        shard_start=0, shard_end=0, owner=-1, owned_here=False,
        owned_count=0, pre_norm=None, post_norm=None, canary_zero_rows=None,
        token=None,
    )
    assert "pre_l2=na" in line and "post_l2=na" in line
    assert "token=na" in line and "canary_zero=None" in line


def test_the_tracer_pairs_rows_with_their_sequence_and_owner():
    """The whole point: a row's global id and its shard owner on one line."""

    tracer = _TRACE.EpilogueTracer(rank=0)
    tracer.begin_chunk(0)
    tracer.record_gather(
        attn_tp_rank=0, attn_tp_size=8, num_global_rows=22480,
        shard_start=0, shard_end=2810,
        owned=[True, True, False],
        pre_norms=[11.0, 12.0, 0.0], post_norms=[11.0, 12.0, 9.5],
        canary_zero_rows=0,
    )
    lines = tracer.emit(
        table_rows=[1, 5, 62],
        tail_rows=[837, 1909, 22479],
        wave_indices=[0, 2, 31],
        global_ids=list(range(64, 96)),
        uuids=[f"u{index}" for index in range(32)],
        tokens=[7, 8, 9],
        log=lambda line: None,
    )

    assert len(lines) == 3
    assert "global_idx=64" in lines[0] and "owner=0" in lines[0]
    assert "owned_here=1" in lines[0] and "token=7" in lines[0]
    # wave index 2 -> global id 66, and the row is this rank's
    assert "wave=2" in lines[1] and "global_idx=66" in lines[1]
    # the deepest tail belongs to rank 7's shard, not this one
    assert "owner=7" in lines[2] and "owned_here=0" in lines[2]
    assert "pre_l2=0.000000e+00" in lines[2] and "post_l2=9.500000e+00" in lines[2]
    assert "owned_count=2" in lines[0]


def test_a_new_chunk_drops_the_previous_gather():
    """Stale norms from the previous chunk must not be reported as this one's."""

    tracer = _TRACE.EpilogueTracer(rank=0)
    tracer.begin_chunk(0)
    tracer.record_gather(
        attn_tp_rank=0, attn_tp_size=1, num_global_rows=10,
        shard_start=0, shard_end=10, owned=[True], pre_norms=[1.0],
        post_norms=[1.0], canary_zero_rows=0,
    )
    tracer.begin_chunk(1)
    lines = tracer.emit(
        table_rows=[0], tail_rows=[3], wave_indices=[0], global_ids=[64],
        uuids=["u"], log=lambda line: None,
    )

    assert tracer.chunk == 1
    assert "chunk=1" in lines[0]
    assert "pre_l2=na" in lines[0] and "post_l2=na" in lines[0]
    assert "canary_zero=None" in lines[0]


# --------------------------------------------------------------------------- #
#  (iv) The per-chunk timing stamps
# --------------------------------------------------------------------------- #
def test_a_phase_line_carries_both_clocks():
    """The wall clock is the only cross-NODE comparable field, so it is there.

    ``perf_counter`` has a per-process origin: comparing two nodes with it
    would be meaningless, which is why the line carries ``wall`` as well.
    """

    line = _TRACE.format_phase(
        rank=3, phase="chunk_begin", chunk=0, wall=1760000000.5,
        mono_ms=12.5, elapsed_ms=None, rows=63, tokens=22480, chunks=1,
    )
    assert line.startswith("[EPILOGUE_TRACE] kind=phase ")
    for fragment in ("rank=3", "phase=chunk_begin", "chunk=0",
                     "wall=1760000000.500000", "mono_ms=12.500",
                     "elapsed_ms=na", "rows=63", "tokens=22480", "chunks=1"):
        assert fragment in line, fragment


def test_a_row_line_and_a_phase_line_are_greppable_apart():
    """One prefix for both, a ``kind`` field to separate them."""

    phase = _TRACE.format_phase(
        rank=0, phase="wave_begin", chunk=-1, wall=1.0, mono_ms=0.0,
        elapsed_ms=None, rows=None, tokens=None, chunks=None)
    row = _TRACE.format_row(
        rank=0, attn_tp_rank=0, attn_tp_size=1, chunk=0, table_row=0,
        tail_row=5, wave_index=0, global_idx=64, uuid="u",
        num_global_rows=10, shard_start=0, shard_end=10, owner=0,
        owned_here=True, owned_count=1, pre_norm=1.0, post_norm=1.0,
        canary_zero_rows=0, token=7)
    assert "kind=phase" in phase and "kind=row" not in phase
    assert "kind=row" in row and "kind=phase" not in row
    assert phase.split()[0] == row.split()[0] == "[EPILOGUE_TRACE]"
    # Missing counts read as na, never as 0.
    assert "rows=na" in phase and "tokens=na" in phase and "chunks=na" in phase


def test_a_chunk_is_stamped_at_both_ends_with_a_duration():
    lines = []
    tracer = _TRACE.EpilogueTracer(rank=1)
    tracer.chunk_begin(0, rows=63, tokens=22480, log=lines.append)
    tracer.chunk_end(rows=63, tokens=22480, log=lines.append)

    assert len(lines) == 2
    assert "phase=chunk_begin" in lines[0] and "elapsed_ms=na" in lines[0]
    assert "phase=chunk_end" in lines[1]
    assert "chunk=0" in lines[0] and "chunk=0" in lines[1]
    elapsed = float(lines[1].split("elapsed_ms=")[1].split()[0])
    assert elapsed >= 0.0
    # Both clocks advance, and the wall clock is absolute.
    walls = [float(line.split("wall=")[1].split()[0]) for line in lines]
    assert walls[1] >= walls[0] > 1_600_000_000.0
    monos = [float(line.split("mono_ms=")[1].split()[0]) for line in lines]
    assert monos[1] >= monos[0] >= 0.0


def test_chunk_begin_also_resets_the_gather_record():
    """A new chunk must not report the previous chunk's norms."""

    tracer = _TRACE.EpilogueTracer(rank=0)
    tracer.chunk_begin(0, log=lambda line: None)
    tracer.record_gather(
        attn_tp_rank=0, attn_tp_size=1, num_global_rows=10, shard_start=0,
        shard_end=10, owned=[True], pre_norms=[1.0], post_norms=[1.0],
        canary_zero_rows=0)
    tracer.chunk_begin(1, log=lambda line: None)

    assert tracer.chunk == 1
    assert tracer.pre_norms is None and tracer.post_norms is None
    row = tracer.emit(
        table_rows=[0], tail_rows=[3], wave_indices=[0], global_ids=[64],
        uuids=["u"], log=lambda line: None)[0]
    assert "pre_l2=na" in row and "chunk=1" in row


def test_an_unclosed_chunk_reports_no_duration_rather_than_a_wrong_one():
    tracer = _TRACE.EpilogueTracer(rank=0)
    line = tracer.chunk_end(log=lambda _: None)  # never begun

    assert "phase=chunk_end" in line and "elapsed_ms=na" in line


def test_the_worker_stamps_the_wave_and_every_chunk():
    source = _ROOT.joinpath("batchgen", "batchgen_worker.py").read_text()

    assert 'epilogue_trace.phase(\n\t\t\t\t"wave_begin"' in source
    assert 'epilogue_trace.phase(\n\t\t\t\t\t"wave_end"' in source
    assert "epilogue_trace.chunk_begin(" in source
    assert "epilogue_trace.chunk_end(" in source
    # wave_end must carry the prefill duration the rank measured.
    tail = source.split('"wave_end"')[1].split(")")[0]
    assert "elapsed_ms=prefill_s * 1e3" in tail


# --------------------------------------------------------------------------- #
#  (v) Both arming seams, through the real sources
# --------------------------------------------------------------------------- #
def test_the_worker_arms_the_trace_through_the_batch_debug_seam():
    source = _ROOT.joinpath("batchgen", "batchgen_worker.py").read_text()

    assert "def _new_pool_epilogue_tracer(self, wave):" in source
    assert "epilogue_trace = self._new_pool_epilogue_tracer(wave)" in source
    seam = source.split("def _new_pool_epilogue_tracer")[1].split("\n\tdef ")[0]
    assert "_active_batchgen_debug_for_sequences" in seam
    assert "parse_trace_flag" in seam
    assert "return None" in seam
    # The pairing side is emitted beside the token selection.
    assert "epilogue_trace.emit(" in source
    assert "tail_rows=rows.tail_rows.tolist()" in source
    assert "global_ids=list(wave.global_ids)" in source


def test_the_gather_takes_the_trace_and_defaults_to_none():
    """``trace=None`` must leave the production gather byte-identical."""

    source = _ROOT.joinpath(
        "batchgen", "models", "moonshotai", "kimi_linear",
        "Parallel_Strategy_Manager.py"
    ).read_text()

    assert ("def gather_prefill_last_token_hidden(\n"
            "        self, hidden_states, last_token_indices, num_global_rows, "
            "trace=None\n    ):") in source
    body = source.split("def gather_prefill_last_token_hidden")[1]
    body = body.split("\n    def ")[0]
    # Norms are taken on BOTH sides of the all-reduce, and only when armed.
    assert "if trace is not None else None" in body
    assert body.index("pre_norms =") < body.index("dist.all_reduce(selected")
    assert body.index("dist.all_reduce(selected") < body.index(
        "post_norms=selected")
    assert "canary_zero_rows=(owned_all_zero" in body


def test_the_kda_state_manager_reports_its_layer_map():
    """A non-injective KDA layer map is logged as an ERROR at construction."""

    source = _ROOT.joinpath(
        "batchgen", "kv_cache", "kda_state_gpu_manager.py"
    ).read_text()

    assert "def _log_layer_mapping_isolation(self)" in source
    assert "self._log_layer_mapping_isolation()" in source
    body = source.split("def _log_layer_mapping_isolation")[1]
    body = body.split("\n    def ")[0]
    assert "is NOT injective" in body
    assert "physical.count(row) > 1" in body
    assert "logging.error" in body
