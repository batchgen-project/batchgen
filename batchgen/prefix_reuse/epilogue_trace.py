"""Flag-gated trace of the pooled prefill EPILOGUE, row by row and rank by rank.

Everything between the pooled kernels and the first generated token is now the
only untested stretch on the metric path. The kernels themselves are cleared
for the live gapped two-group wave - planner, executor layout, kernel
arguments, per-page pool bytes, the FA3 paged read, fla's chunk_kda, the KDA
levels and forks, and the conv kernel with its state plumbing - yet the same
five members corrupt deterministically, and first-token parity is the argmax
of the PREFILL's final logits. So the defect is either in the last hidden
state or in the row -> sequence pairing that carries it to sampling.

Three mappings sit there, and none of the local oracles or GPU replays reached
any of them, because all of them were single-rank and compared tensors rather
than the pairing (``batchgen_worker.py`` numbers are the pooled-wave epilogue):

1. ``hidden_states[:, rows.tail_rows, :]`` - the tail-row gather, where
   ``tail_rows[i] = len(input_ids) + item.tokens - 1`` was fixed at pack time
   (``prefix_reuse/executor.py``).
2. ``gather_prefill_last_token_hidden`` (Parallel_Strategy_Manager) - under
   streamed-SP8 the final hidden state is only THIS rank's contiguous token
   shard ``balanced_row_split(num_global_rows, attn_tp)[attn_tp_rank]``, so a
   zero-filled all-reduce reassembles the requested rows. A row whose owner
   gathers the wrong offset contributes a NON-zero wrong vector, which the
   existing all-zero canary cannot see.
3. ``_select_tokens(logits, [...])`` and ``tokens_by_index[index] = ...`` - the
   row -> wave index -> uuid -> sequence pairing.

What the trace records, per tail row and per rank, when armed: the row's flat
index, its wave index, the sequence's global id, the shard that owns it, how
many of this rank's owned rows that row accounts for, the L2 norm of the
hidden vector BEFORE the all-reduce (zero on a rank that does not own it) and
AFTER it, and the token finally selected for that row. One line per row per
rank, which is 32 lines per rank on the live wave.

That is enough to decide, from one armed run, between:

* a wrong pairing      - the global id on a row is not the member whose text
  came out wrong;
* a dropped row        - post-all-reduce norm 0, or no rank reporting owner;
* a wrong-offset gather - the owner's pre-all-reduce norm is non-zero but the
  post-all-reduce vector disagrees with the owner's;
* an innocent epilogue - every row pairs correctly and carries a healthy
  norm, which moves the defect back into the stack above it.

Off by default and nothing in this module runs.

Arming (batch level, no env var, no cold restart)::

    {"batchgen_debug": {"prefix_pool_epilogue_trace": true}}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Batch-level ``batchgen_debug`` keys.
DEBUG_FLAG = "prefix_pool_epilogue_trace"
# Optional companion: a token id whose first-token logit every row reports, so
# a run can ask "how far was the reference's first token from winning here".
PROBE_FLAG = "prefix_pool_epilogue_probe_token"
PREFIX = "EPILOGUE_TRACE"


def parse_trace_flag(value) -> bool:
    """``True`` only when the batch asked; an unsupported value is refused.

    A mistyped flag must not read as off, for the same reason the kernel dump
    refuses one: a silent no-op looks exactly like a clean run.
    """

    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "no", "off"}:
            return False
        if text in {"1", "true", "yes", "on"}:
            return True
    raise ValueError(
        f"{DEBUG_FLAG} must be a boolean (or 0/1/on/off), got {value!r}"
    )


def parse_probe_token(value) -> Optional[int]:
    """``None`` when not asked; a non-negative token id otherwise.

    A value that is not an integer id is refused by name rather than read as
    "no probe" - a mistyped probe must not quietly print ``na`` everywhere.
    """

    if value is None or value is False:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{PROBE_FLAG} must be a token id, got {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{PROBE_FLAG} must be >= 0, got {value!r}")
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"{PROBE_FLAG} must be a token id, got {value!r}")


def logit_diagnostics(logits, probe_token: Optional[int] = None) -> list:
    """Per row: top-2 ids and logits, their margin, the probe's logit, vocab.

    ``logits`` is the ``[rows, vocab]`` tensor ``_select_tokens`` consumes.
    It must be the FULL vocabulary: the vocab-parallel head all-gathers and
    reassembles ``[..., V]`` before returning (vocab_parallel.py, forward and
    assemble_gathered_logits), so a top-k here is global. ``vocab`` is reported
    on every line anyway, so a shard-local tensor could never pass as global.

    The margin is what separates the two readings of a wrong first token: a
    near-tie that pooled numerics tipped (small margin) from a hidden state
    that is substantively elsewhere (large margin for the wrong token).
    """

    import torch

    if logits.dim() != 2:
        raise ValueError(f"expected [rows, vocab] logits, got {tuple(logits.shape)}")
    rows, vocab = logits.shape
    work = logits.detach().float()
    k = min(2, vocab)
    values, indices = torch.topk(work, k, dim=-1)
    values = values.cpu().tolist()
    indices = indices.cpu().tolist()
    probe_values = None
    if probe_token is not None and 0 <= probe_token < vocab:
        probe_values = work[:, probe_token].cpu().tolist()
    out = []
    for row in range(rows):
        top1 = (int(indices[row][0]), float(values[row][0]))
        top2 = ((int(indices[row][1]), float(values[row][1]))
                if k > 1 else None)
        if probe_token is None:
            probe = None
        elif probe_values is None:
            probe = (int(probe_token), "out_of_range")
        else:
            probe = (int(probe_token), float(probe_values[row]))
        out.append({
            "top1": top1, "top2": top2,
            "margin": (top1[1] - top2[1]) if top2 is not None else None,
            "probe": probe, "vocab": int(vocab),
        })
    return out


def format_diag(diag: Optional[dict]) -> str:
    """`` top1=<id>:<logit> top2=<id>:<logit> margin=<f> probe=<id>:<v|na> vocab=<V>``."""

    if diag is None:
        return ""

    def pair(item):
        if item is None:
            return "na"
        ident, value = item
        return (f"{ident}:{value}" if isinstance(value, str)
                else f"{ident}:{value:.6f}")

    margin = "na" if diag["margin"] is None else f"{diag['margin']:.6f}"
    probe = "na" if diag["probe"] is None else pair(diag["probe"])
    return (f" top1={pair(diag['top1'])} top2={pair(diag['top2'])} "
            f"margin={margin} probe={probe} vocab={diag['vocab']}")


def balanced_shard(num_rows: int, group_size: int, group_rank: int):
    """``[start, end)`` of one rank, mirroring ``balanced_row_split``.

    Transcribed rather than imported so the trace states the shard it believes
    in even if the production split changes under it; the two are compared by
    ``tests/unit/test_prefix_pool_epilogue_trace.py``.
    """

    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if not 0 <= group_rank < group_size:
        raise ValueError(
            f"group_rank {group_rank} outside [0, {group_size})"
        )
    base, rem = divmod(int(num_rows), group_size)
    start = base * group_rank + min(group_rank, rem)
    end = start + base + (1 if group_rank < rem else 0)
    return start, end


def owner_of_row(row: int, num_rows: int, group_size: int) -> int:
    """Which shard owns a flat row index, or -1 when it is out of range.

    ``balanced_row_split`` is a contiguous partition, so a row in range has
    exactly one owner; -1 is the case the production all-zero canary cannot
    see, because then NO rank owns the row and no rank inspects it.
    """

    if not 0 <= row < int(num_rows):
        return -1
    base, rem = divmod(int(num_rows), group_size)
    boundary = (base + 1) * rem
    if row < boundary:
        return row // (base + 1)
    return rem + (row - boundary) // base


PATHS = ("pool", "unpooled")


def tracer_from_debug(debug, *, rank: int, path: str):
    """The armed tracer for one prefill, or None. Shared by both prefill paths.

    The pooled wave and the unpooled prefill read the SAME batch-level keys,
    so one armed batch traces whichever path actually runs; ``path`` tags
    every line it emits.
    """

    if path not in PATHS:
        raise ValueError(f"epilogue trace path must be one of {PATHS}, got {path!r}")
    debug = debug or {}
    if not parse_trace_flag(debug.get(DEBUG_FLAG)):
        return None
    return EpilogueTracer(
        rank=rank, path=path, probe_token=parse_probe_token(debug.get(PROBE_FLAG))
    )


def format_phase(
    *,
    rank: int,
    phase: str,
    chunk: int,
    wall: float,
    mono_ms: float,
    elapsed_ms: Optional[float],
    rows: Optional[int],
    tokens: Optional[int],
    chunks: Optional[int],
    path: str = "pool",
) -> str:
    """One timing line. ``wall`` is the only field comparable across NODES.

    ``mono_ms`` is ``perf_counter`` since this tracer was built, which is
    meaningful only within one process: perf_counter has a per-process origin,
    so comparing two nodes with it would be meaningless. The wall clock
    (CLOCK_REALTIME) is comparable to whatever the two hosts' clocks agree on,
    which is what answers "did group 1 leave prefill while group 0 was still
    inside its pooled chunk".
    """

    def number(value, digits=3):
        return "na" if value is None else f"{value:.{digits}f}"

    def count(value):
        return "na" if value is None else str(int(value))

    return (
        f"[{PREFIX}] kind=phase path={path} rank={rank} phase={phase} "
        f"chunk={chunk} "
        f"wall={wall:.6f} mono_ms={mono_ms:.3f} "
        f"elapsed_ms={number(elapsed_ms)} rows={count(rows)} "
        f"tokens={count(tokens)} chunks={count(chunks)}"
    )


def format_row(
    *,
    rank: int,
    attn_tp_rank: int,
    attn_tp_size: int,
    chunk: int,
    table_row: int,
    tail_row: int,
    wave_index: int,
    global_idx: int,
    uuid: str,
    num_global_rows: int,
    shard_start: int,
    shard_end: int,
    owner: int,
    owned_here: bool,
    owned_count: int,
    pre_norm: Optional[float],
    post_norm: Optional[float],
    canary_zero_rows: Optional[int],
    token: Optional[int],
    diag: Optional[dict] = None,
    path: str = "pool",
) -> str:
    """One compact line. Pure, so the formatter is testable on its own."""

    def number(value):
        return "na" if value is None else f"{value:.6e}"

    return (
        f"[{PREFIX}] kind=row path={path} rank={rank} "
        f"tp={attn_tp_rank}/{attn_tp_size} "
        f"chunk={chunk} row={table_row} tail_row={tail_row} "
        f"wave={wave_index} global_idx={global_idx} uuid={uuid} "
        f"rows={num_global_rows} shard=[{shard_start},{shard_end}) "
        f"owner={owner} owned_here={int(bool(owned_here))} "
        f"owned_count={owned_count} pre_l2={number(pre_norm)} "
        f"post_l2={number(post_norm)} canary_zero={canary_zero_rows} "
        f"token={'na' if token is None else int(token)}"
        f"{format_diag(diag)}"
    )


@dataclass
class EpilogueTracer:
    """Collects the gather-side numbers, then logs one line per tail row.

    The worker builds one per armed wave and hands it to
    ``gather_prefill_last_token_hidden`` so the shard-side facts (which are
    only visible inside that function) reach the same line as the pairing.
    """

    rank: int
    attn_tp_rank: int = 0
    attn_tp_size: int = 1
    chunk: int = -1
    num_global_rows: int = 0
    shard: tuple = (0, 0)
    pre_norms: Optional[list] = None
    post_norms: Optional[list] = None
    owned: Optional[list] = None
    owned_count: int = 0
    canary_zero_rows: Optional[int] = None
    lines: list = field(default_factory=list)
    # Timing. ``_origin`` anchors the per-process monotonic clock; the wall
    # clock is emitted raw because it is the only cross-NODE comparable field.
    _origin: float = field(default_factory=time.perf_counter)
    _chunk_started: Optional[float] = None
    # ``prefix_pool_epilogue_probe_token``: the id whose logit every row reports.
    probe_token: Optional[int] = None
    # Which prefill emitted the line: the pooled wave or the unpooled path.
    path: str = "pool"

    # -- timing ------------------------------------------------------------ #
    def phase(
        self,
        name: str,
        *,
        elapsed_ms: Optional[float] = None,
        rows: Optional[int] = None,
        tokens: Optional[int] = None,
        chunks: Optional[int] = None,
        log=None,
    ) -> str:
        """Stamp one boundary; returns the line so the tests can read it."""

        line = format_phase(
            rank=self.rank, phase=name, chunk=self.chunk, wall=time.time(),
            mono_ms=(time.perf_counter() - self._origin) * 1e3,
            elapsed_ms=elapsed_ms, rows=rows, tokens=tokens, chunks=chunks,
            path=self.path,
        )
        self.lines.append(line)
        (logging.info if log is None else log)(line)
        return line

    def chunk_begin(self, chunk_index: int, *, rows=None, tokens=None,
                    log=None) -> str:
        """Open a chunk: resets the gather-side record and stamps the start."""

        self.begin_chunk(chunk_index)
        self._chunk_started = time.perf_counter()
        return self.phase("chunk_begin", rows=rows, tokens=tokens, log=log)

    def chunk_end(self, *, rows=None, tokens=None, log=None) -> str:
        """Close a chunk, with its wall duration."""

        elapsed = (None if self._chunk_started is None
                   else (time.perf_counter() - self._chunk_started) * 1e3)
        self._chunk_started = None
        return self.phase("chunk_end", elapsed_ms=elapsed, rows=rows,
                          tokens=tokens, log=log)

    def begin_chunk(self, chunk_index: int) -> None:
        self.chunk = int(chunk_index)
        self.pre_norms = None
        self.post_norms = None
        self.owned = None
        self.owned_count = 0
        self.canary_zero_rows = None

    # -- the gather side --------------------------------------------------- #
    def record_gather(
        self,
        *,
        attn_tp_rank: int,
        attn_tp_size: int,
        num_global_rows: int,
        shard_start: int,
        shard_end: int,
        owned,
        pre_norms,
        post_norms,
        canary_zero_rows: Optional[int],
    ) -> None:
        """Called from inside the gather, before and after the all-reduce."""

        self.attn_tp_rank = int(attn_tp_rank)
        self.attn_tp_size = int(attn_tp_size)
        self.num_global_rows = int(num_global_rows)
        self.shard = (int(shard_start), int(shard_end))
        self.owned = [bool(x) for x in owned]
        self.pre_norms = [float(x) for x in pre_norms]
        self.post_norms = [float(x) for x in post_norms]
        self.owned_count = sum(1 for x in self.owned if x)
        self.canary_zero_rows = (None if canary_zero_rows is None
                                 else int(canary_zero_rows))

    # -- the pairing side -------------------------------------------------- #
    def emit(
        self,
        *,
        table_rows: Sequence[int],
        tail_rows: Sequence[int],
        wave_indices: Sequence[int],
        global_ids: Sequence[int],
        uuids: Sequence[str],
        tokens: Sequence[int] = (),
        log=None,
        logits=None,
    ) -> list:
        """Log one line per tail row; returns the lines for the tests.

        ``logits`` is the ``[rows, vocab]`` tensor ``_select_tokens`` consumed;
        when given, each line also carries the first-token diagnostics.
        """

        emit_log = logging.info if log is None else log
        diags = (logit_diagnostics(logits, self.probe_token)
                 if logits is not None else None)
        if diags is not None and len(diags) != len(tail_rows):
            raise ValueError(
                f"{len(diags)} logit rows for {len(tail_rows)} tail rows: the "
                f"diagnostics would pair a row with another row's logits"
            )
        self.lines = []
        for index, tail_row in enumerate(tail_rows):
            wave_index = int(wave_indices[index])
            owner = owner_of_row(int(tail_row), self.num_global_rows,
                                 self.attn_tp_size) \
                if self.num_global_rows else -1
            line = format_row(
                rank=self.rank,
                attn_tp_rank=self.attn_tp_rank,
                attn_tp_size=self.attn_tp_size,
                chunk=self.chunk,
                table_row=int(table_rows[index]) if index < len(table_rows)
                else -1,
                tail_row=int(tail_row),
                wave_index=wave_index,
                global_idx=int(global_ids[wave_index])
                if wave_index < len(global_ids) else -1,
                uuid=str(uuids[wave_index]) if wave_index < len(uuids)
                else "na",
                num_global_rows=self.num_global_rows,
                shard_start=self.shard[0],
                shard_end=self.shard[1],
                owner=owner,
                owned_here=(self.owned[index]
                            if self.owned and index < len(self.owned)
                            else False),
                owned_count=self.owned_count,
                pre_norm=(self.pre_norms[index]
                          if self.pre_norms and index < len(self.pre_norms)
                          else None),
                post_norm=(self.post_norms[index]
                           if self.post_norms and index < len(self.post_norms)
                           else None),
                canary_zero_rows=self.canary_zero_rows,
                token=(tokens[index] if index < len(tokens) else None),
                diag=(diags[index] if diags is not None else None),
                path=self.path,
            )
            self.lines.append(line)
            emit_log(line)
        return self.lines
