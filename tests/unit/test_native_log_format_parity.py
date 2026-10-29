"""Format parity between the native (spdlog) console lines and Python's.

The native loggers in ``core/`` and the Python loggers write to the SAME
console stream (stderr), so one capture interleaves both. That is only
readable if both stacks emit the same line shape, and only machine-parseable
if one regex covers both. Nothing in the build enforces that: the spdlog
pattern is a string literal in ``core/utils.cpp`` and the Python format is a
``logging`` format string, and they can drift apart silently.

This suite pins the pattern literal, renders the line spdlog would produce
from that very literal, and asserts that one regex parses both that rendered
line and a line produced by a real ``logging.Formatter``. A change to either
side that breaks the shared shape fails here.

Nothing is built or imported from the native extension: the C++ side is read
as source text (the idiom of ``tests/contracts/``), so this runs on a machine
with no GPU toolchain and costs milliseconds. The one thing source text
cannot prove is that the compiler accepts the edit -- that is the GPU-node
build check, not this test.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
UTILS_CPP = REPO / "core" / "utils.cpp"
EVENTLOG_INIT = REPO / "batchgen" / "eventlog" / "__init__.py"

# ---------------------------------------------------------------------------
# The pinned native pattern.
#
# Frozen copy of the literal passed to ``logger->set_pattern()`` in
# ``core/utils.cpp``'s ``init_logger``. Changing the native line shape is a
# deliberate act: update this pin in the same change and say why.
#
#   %Y-%m-%d %H:%M:%S   date and time, matching logging's default_time_format
#   ,%e                 zero-padded 3-digit milliseconds (spdlog pad3),
#                       matching logging's default_msec_format "%s,%03d"
#   [%n]                logger name in brackets
#   %^%l%$              level name, wrapped in the sink's color range; %l is
#                       lowercase ("info", "warning"), where Python's
#                       %(levelname)s is uppercase -- see the level test
#   %v                  the message
# ---------------------------------------------------------------------------
PINNED_SPDLOG_PATTERN = "%Y-%m-%d %H:%M:%S,%e - [%n] - %^%l%$ - %v"

# The Python-side format this mirrors. In-repo instance:
# batchgen/tools/convert_checkpoint.py's logging.basicConfig(format=...).
PYTHON_LOG_FORMAT = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"

# ---------------------------------------------------------------------------
# The parity regex.
#
# PROVENANCE: defined here, not imported. As of this commit the repo has no
# reusable named regex for this line shape -- batchgen/eventlog/ parses only
# its own JSONL records and renders its own distinct console format, and no
# in-repo script parses a native console line. This regex therefore mirrors
# PYTHON_LOG_FORMAT above (asctime - [name] - LEVEL - message) and is
# asserted against BOTH stacks' output below, so it cannot quietly describe
# only one of them.
#
# Level matching is case-insensitive on purpose: spdlog's %l is lowercase and
# Python's %(levelname)s is uppercase, and the same regex must take both.
# ---------------------------------------------------------------------------
PARITY_RE = re.compile(
    r"^(?P<asctime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r" - \[(?P<name>[^\]]+)\]"
    r" - (?P<level>trace|debug|info|warn|warning|err|error|critical)"
    r" - (?P<message>.*)$",
    re.IGNORECASE,
)

# One fixed instant, so the rendered example is deterministic.
_EXAMPLE_FIELDS = {
    "Y": "2026",
    "m": "10",
    "d": "29",
    "H": "12",
    "M": "34",
    "S": "56",
    "e": "789",
}
_EXAMPLE_LOGGER_NAME = "Parameter_Server"
_EXAMPLE_LEVEL = "info"
_EXAMPLE_MESSAGE = "mapped 4096 bytes"


def render_spdlog_example(pattern: str) -> str:
    """Render the line spdlog emits for ``pattern``, per its flag semantics.

    Color-range markers (``%^``/``%$``) render as nothing: the formatter only
    records the span, and the ansicolor sink inserts escapes around it ONLY
    when its target stream is a terminal (``color_mode::automatic`` ->
    ``in_terminal()``). A redirected capture -- which is what any parser
    reads -- is therefore plain text.

    An unmodelled flag is a hard error, never a passthrough: a silent
    passthrough would let a new flag render as a literal ``%x`` and still
    satisfy a loose regex.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char != "%":
            out.append(char)
            i += 1
            continue
        if i + 1 >= len(pattern):
            raise ValueError(f"pattern ends in a bare '%': {pattern!r}")
        flag = pattern[i + 1]
        i += 2
        if flag in _EXAMPLE_FIELDS:
            out.append(_EXAMPLE_FIELDS[flag])
        elif flag == "n":
            out.append(_EXAMPLE_LOGGER_NAME)
        elif flag == "l":
            out.append(_EXAMPLE_LEVEL)
        elif flag == "v":
            out.append(_EXAMPLE_MESSAGE)
        elif flag in ("^", "$"):
            pass  # color range marker: no text of its own
        elif flag == "%":
            out.append("%")
        else:
            raise ValueError(
                f"render_spdlog_example: unmodelled spdlog flag '%{flag}' in "
                f"{pattern!r}; model it here and re-check the parity regex"
            )
    return "".join(out)


def python_logging_example(
    *, name: str, levelno: int, message: str
) -> str:
    """One line from a real ``logging.Formatter``, not a hand-written string."""
    record = logging.LogRecord(
        name=name,
        level=levelno,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    return logging.Formatter(PYTHON_LOG_FORMAT).format(record)


def _set_patterns() -> list[str]:
    source = UTILS_CPP.read_text()
    patterns = re.findall(r'set_pattern\("((?:[^"\\]|\\.)*)"\)', source)
    assert patterns, "no logger->set_pattern(\"...\") call found in core/utils.cpp"
    return patterns


def test_spdlog_pattern_literal_is_pinned():
    # Every set_pattern call in init_logger must use the one pinned shape:
    # two loggers that disagree would interleave two formats in one capture.
    patterns = _set_patterns()
    assert set(patterns) == {PINNED_SPDLOG_PATTERN}, patterns


def test_native_console_sink_is_stderr_like_the_python_console():
    # The two halves of the parity claim, both source-text:
    # native sink == stderr, and the Python console handler defaults to
    # sys.stderr. If either moves, the "one stream, one shape" story breaks.
    source = UTILS_CPP.read_text()
    assert "spdlog::stderr_color_mt(logger_name)" in source
    # The color setup casts to the sink type the factory produced; the stdout
    # sink type is a different type, so a stale cast would answer null and
    # silently drop the colors instead of failing.
    assert "dynamic_cast<spdlog::sinks::stderr_color_sink_mt*>" in source
    assert "stdout_color" not in source

    eventlog = EVENTLOG_INIT.read_text()
    assert "sys.stderr if console_stream is None else console_stream" in eventlog


def test_rendered_spdlog_line_parses():
    line = render_spdlog_example(PINNED_SPDLOG_PATTERN)
    # The pattern renders to exactly the documented example shape.
    assert line == "2026-10-29 12:34:56,789 - [Parameter_Server] - info - mapped 4096 bytes"
    match = PARITY_RE.match(line)
    assert match is not None, line
    assert match.group("asctime") == "2026-10-29 12:34:56,789"
    assert match.group("name") == _EXAMPLE_LOGGER_NAME
    assert match.group("level") == _EXAMPLE_LEVEL
    assert match.group("message") == _EXAMPLE_MESSAGE


def test_python_logging_line_parses():
    line = python_logging_example(
        name=_EXAMPLE_LOGGER_NAME,
        levelno=logging.INFO,
        message=_EXAMPLE_MESSAGE,
    )
    match = PARITY_RE.match(line)
    assert match is not None, line
    assert match.group("name") == _EXAMPLE_LOGGER_NAME
    assert match.group("level") == "INFO"
    assert match.group("message") == _EXAMPLE_MESSAGE


def test_both_stacks_agree_field_by_field():
    # Same logger name and message through both stacks: everything except the
    # timestamp and the level's case must come out identical.
    native = PARITY_RE.match(render_spdlog_example(PINNED_SPDLOG_PATTERN))
    python = PARITY_RE.match(
        python_logging_example(
            name=_EXAMPLE_LOGGER_NAME,
            levelno=logging.INFO,
            message=_EXAMPLE_MESSAGE,
        )
    )
    assert native is not None and python is not None
    assert native.group("name") == python.group("name")
    assert native.group("message") == python.group("message")
    assert native.group("level").upper() == python.group("level").upper()
    # Both timestamps carry 3-digit milliseconds after a comma: spdlog's %e is
    # pad3, logging's default_msec_format is "%s,%03d".
    for group in (native.group("asctime"), python.group("asctime")):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", group)


@pytest.mark.parametrize(
    "spdlog_name,python_levelno",
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("critical", logging.CRITICAL),
    ],
)
def test_level_spellings_agree_ignoring_case(spdlog_name, python_levelno):
    # spdlog's SPDLOG_LEVEL_NAMES are lowercase; logging's levelnames are
    # uppercase. The spellings themselves must agree, so one case-insensitive
    # level alternation covers both stacks.
    assert logging.getLevelName(python_levelno) == spdlog_name.upper()
    for line in (
        f"2026-10-29 12:34:56,789 - [X] - {spdlog_name} - m",
        python_logging_example(name="X", levelno=python_levelno, message="m"),
    ):
        match = PARITY_RE.match(line)
        assert match is not None, line
        assert match.group("level").upper() == spdlog_name.upper()


def test_unmodelled_flag_is_a_hard_error():
    # The renderer must not pass an unknown flag through as literal text.
    with pytest.raises(ValueError, match=r"unmodelled spdlog flag '%q'"):
        render_spdlog_example("%Y %q")
    with pytest.raises(ValueError, match="bare '%'"):
        render_spdlog_example("%Y %")
