# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The K3 Triton prewarm's planning, recorder and guard, on CPU.

Under test: ``batchgen/models/moonshotai/kimi_linear/triton_prewarm.py``.

  * the shape table witnesses every ``(T, N, NT, varlen)`` integer class that
    the serving sweep (``_kda_segment_plan`` as ``_kda_chunk_segments`` uses
    it) can hand to ``chunk_kda``, checked against random micro-batches at
    both serving segment caps;
  * the rank partition covers every entry exactly once;
  * the late-compile guard's states, the recorder's parsing of a Triton 3.5
    hook payload, its compile/load split and its hook chaining;
  * the phase runner's order: own share, barrier, everything;
  * the module imports with neither Triton nor fla installed.

Loaded by file path (no ``import batchgen``); ``_kda_segment_plan`` is lifted
out of ``serving_modules.py`` by AST so its heavy imports never run.

Run: python -m pytest tests/unit/test_k3_triton_prewarm.py -q
"""

import ast
import importlib.util
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_KL_DIR = (Path(__file__).resolve().parents[2] / "batchgen" / "models"
           / "moonshotai" / "kimi_linear")
_SERVING_CAPS = (4096, 16384)


def _load_prewarm(name="_k3_triton_prewarm_under_test"):
    spec = importlib.util.spec_from_file_location(
        name, _KL_DIR / "triton_prewarm.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _serving_namespace():
    """``_kda_segment_plan`` and its constants, without the module imports."""
    path = _KL_DIR / "serving_modules.py"
    tree = ast.parse(path.read_text())
    wanted = {"_KDA_CHUNK_SIZE", "KDA_PREFILL_SEGMENT_TOKENS"}
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef)
            and node.name == "_kda_segment_plan")
        or (isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted for t in node.targets))
    ]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace


TP = _load_prewarm()
SERVING = _serving_namespace()
PLAN = SERVING["_kda_segment_plan"]
CHUNK = SERVING["_KDA_CHUNK_SIZE"]


# --------------------------------------------------------------------------- #
#  Shape classes
# --------------------------------------------------------------------------- #
def test_int_class_follows_triton_integer_specialisation():
    assert TP.int_class(1) == "1"
    assert [TP.int_class(v) for v in (16, 32, 4096)] == ["d16"] * 3
    assert [TP.int_class(v) for v in (2, 15, 17, 1000)] == ["any"] * 4


@pytest.mark.parametrize("cap", _SERVING_CAPS)
def test_table_resolves_and_names_are_unique(cap):
    names = [shape.name for shape in TP.KDA_PREFILL_SHAPES]
    assert len(names) == len(set(names))
    for shape in TP.KDA_PREFILL_SHAPES:
        assert all(n > 0 for n in shape.resolve(cap))


def test_serving_default_cap_is_a_tested_cap():
    assert SERVING["KDA_PREFILL_SEGMENT_TOKENS"] in _SERVING_CAPS


@pytest.mark.parametrize("cap", _SERVING_CAPS)
def test_call_classes_mirror_the_segmented_sweep(cap):
    def classes(lengths):
        return TP.kda_call_classes(lengths, cap, PLAN, CHUNK)

    # Single call: cu_seqlens is always passed, even for one sequence.
    assert classes([1]) == [("1", "1", "1", True)]
    assert classes([cap]) == [("d16", "1", "d16", True)]
    # Over the cap: a cap-sized batch-1 call, then a batch-1 tail.
    assert classes([cap + 1]) == [("d16", "-", "d16", False),
                                  ("1", "-", "1", False)]
    # The cut snaps back to the second sequence's chunk grid (cap - 63), so
    # the first segment is varlen over both sequences with cap / 64 chunks,
    # and the 65-token remainder is a batch-1 call.
    half = cap // 2 + 1
    assert classes([half, half]) == [("any", "any", "d16", True),
                                      ("any", "-", "any", False)]


@pytest.mark.parametrize("cap", _SERVING_CAPS)
def test_table_witnesses_every_reachable_class(cap):
    planned = TP.planned_kda_call_classes(TP.KDA_PREFILL_SHAPES, cap, PLAN,
                                          CHUNK)
    # 15 varlen classes plus 7 batch-1 classes are reachable in total.
    assert len(planned) == 22
    rng = random.Random(20261101 + cap)
    ranges = ((1, 4), (1, 200), (1, 3000), (cap - 100, cap + 100),
              (1, 2 * cap))
    draws = [[1], [cap], [cap + 1], [1] * 17, [cap - 17, 40],
             [cap // 2 + 1] * 2, [1] * 15 + [cap + 66]]
    for _ in range(3000):
        count = rng.choice((1, 1, 2, 3, 5, 15, 16, 17, 33))
        draws.append([rng.randint(*rng.choice(ranges)) for _ in range(count)])
    for lengths in draws:
        missing = set(TP.kda_call_classes(lengths, cap, PLAN, CHUNK)) - planned
        assert not missing, (lengths, missing)


@pytest.mark.parametrize("heads,cap", [(12, 131_072), (12, 16_384), (8, 4_096)])
def test_o_norm_counts_reach_every_block_count_in_both_row_classes(heads, cap):
    counts = TP.o_norm_token_counts(heads, cap)
    per_block = TP.O_NORM_ROWS_PER_BLOCK
    max_blocks = -(-cap * heads // per_block)
    seen = {}
    for tokens in counts:
        assert 1 <= tokens <= cap
        blocks = -(-tokens * heads // per_block)
        seen.setdefault(blocks, set()).add(TP.int_class(tokens * heads) == "d16")
    assert set(seen) == set(range(1, max_blocks + 1))
    # The last block count may be reachable only by the cap itself.
    for blocks, classes in seen.items():
        if blocks < max_blocks:
            assert classes == {True, False}


def test_partition_covers_each_entry_once_and_is_deterministic():
    items = list(range(23))
    for count in (1, 2, 3, 8):
        shares = [TP.partition(items, index, count) for index in range(count)]
        assert sorted(x for share in shares for x in share) == items
        assert shares == [TP.partition(items, index, count)
                          for index in range(count)]
    with pytest.raises(ValueError):
        TP.partition(items, 8, 8)


# --------------------------------------------------------------------------- #
#  Guard and recorder
# --------------------------------------------------------------------------- #
def _payload(name="_fake_kernel"):
    """The keyword arguments Triton 3.5 passes to both compile hooks."""
    jit = SimpleNamespace(arg_names=["x_ptr", "n", "BLOCK"])
    return dict(
        key=f"key-{name}",
        repr=f"{name}[num_warps=4](x_ptr: *bf16, n: i32, BLOCK: constexpr)",
        fn=SimpleNamespace(module="fake_mod", name=name, jit_function=jit),
        compile={
            "key": f"key-{name}",
            "signature": {"x_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"},
            "constants": {(2,): 1024},
            "configs": [{(0,): [["tt.divisibility", 16]]}],
            "num_warps": 4, "num_stages": 3, "num_ctas": 1,
            "device": 0, "specialization_data": "{}", "is_warmup": False,
        },
        is_manual_warmup=False,
        already_compiled=False,
    )


class _Clock:
    def __init__(self, *values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


def _recorder(debug=None, clock=None):
    guard = TP.LateCompileGuard(debug_source=lambda: debug or {})
    recorder = TP.TritonCompileRecorder(guard=guard,
                                        clock=clock or _Clock(*[0.0] * 64))
    runtime = SimpleNamespace(jit_cache_hook=None, jit_post_compile_hook=None)
    assert recorder.install(runtime)
    return recorder, runtime


def _compile(runtime, payload):
    assert runtime.jit_cache_hook(**payload) is None
    runtime.jit_post_compile_hook(**payload)


def test_guard_states():
    debug = {}
    guard = TP.LateCompileGuard(debug_source=lambda: debug)
    assert guard.state() == TP.DISARMED
    guard.arm()
    assert guard.state() == TP.LOG
    for value in (True, 1, "1", "true", "on"):
        debug[TP.STRICT_DEBUG_KEY] = value
        assert guard.state() == TP.STRICT
    for value in (False, 0, "false", None):
        debug[TP.STRICT_DEBUG_KEY] = value
        assert guard.state() == TP.LOG
    guard.arm(strict=True)
    assert guard.state() == TP.STRICT
    guard.disarm()
    debug[TP.STRICT_DEBUG_KEY] = True
    assert guard.state() == TP.DISARMED


def test_payload_parsing_names_arguments():
    event = TP.parse_hook_payload(_payload())
    assert event.kernel == "fake_mod._fake_kernel"
    assert event.constexprs == "{BLOCK=1024}"
    assert event.signature == "{x_ptr:*bf16,n:i32,BLOCK:constexpr}"
    assert event.attrs == "{x_ptr:tt.divisibility=16}"
    assert event.options == "num_warps=4,num_stages=3,num_ctas=1"


def test_payload_parsing_tolerates_missing_fields():
    event = TP.parse_hook_payload({"repr": "kern[num_warps=4](a: i32)"})
    assert event.kernel == "kern"
    assert event.constexprs == "{}"


def test_compile_and_load_are_split_by_wall_time():
    recorder, runtime = _recorder(clock=_Clock(0.0, 2.0, 10.0, 10.01))
    _compile(runtime, _payload("slow"))
    _compile(runtime, _payload("fast"))
    assert recorder.counts() == {"compiled": 1, "loaded": 1}
    assert recorder.per_kernel() == {
        "fake_mod.slow": (1, 0), "fake_mod.fast": (0, 1),
    }


def test_disarmed_compiles_are_not_late(caplog):
    recorder, runtime = _recorder()
    with caplog.at_level(logging.DEBUG):
        _compile(runtime, _payload())
    assert not recorder.events[0].late
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_armed_compile_logs_the_full_key(caplog):
    recorder, runtime = _recorder()
    recorder.guard.arm()
    with caplog.at_level(logging.WARNING):
        _compile(runtime, _payload())
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    message = record.getMessage()
    assert message.startswith("[K3_TRITON_WARMUP] late compile kernel=")
    for part in ("fake_mod._fake_kernel", "BLOCK=1024", "x_ptr:*bf16",
                 "x_ptr:tt.divisibility=16", "num_warps=4"):
        assert part in message
    assert recorder.events[0].late


@pytest.mark.parametrize("strict_by", ["batch_flag", "arm"])
def test_strict_raises_before_compiling(caplog, strict_by):
    debug = {TP.STRICT_DEBUG_KEY: True} if strict_by == "batch_flag" else {}
    recorder, runtime = _recorder(debug=debug)
    recorder.guard.arm(strict=strict_by == "arm")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(TP.LateTritonCompileError,
                           match="fake_mod._fake_kernel"):
            runtime.jit_cache_hook(**_payload())
    assert "late compile (strict)" in caplog.text
    assert recorder.events == []


def test_previous_hooks_are_chained():
    calls = []

    def previous_cache(**payload):
        calls.append("cache")
        return None

    def previous_post(**payload):
        calls.append("post")
        return "post-result"

    runtime = SimpleNamespace(jit_cache_hook=previous_cache,
                              jit_post_compile_hook=previous_post)
    recorder = TP.TritonCompileRecorder(
        guard=TP.LateCompileGuard(debug_source=dict),
        clock=_Clock(0.0, 0.0),
    )
    assert recorder.install(runtime)
    assert runtime.jit_cache_hook(**_payload()) is None
    assert runtime.jit_post_compile_hook(**_payload()) == "post-result"
    assert calls == ["cache", "post"]
    recorder.uninstall()
    assert runtime.jit_cache_hook is previous_cache
    assert runtime.jit_post_compile_hook is previous_post


def test_a_previous_hook_that_skips_the_compile_is_respected():
    runtime = SimpleNamespace(jit_cache_hook=lambda **payload: True,
                              jit_post_compile_hook=None)
    recorder = TP.TritonCompileRecorder(
        guard=TP.LateCompileGuard(debug_source=dict))
    assert recorder.install(runtime)
    assert runtime.jit_cache_hook(**_payload()) is True
    assert recorder._pending == {}


def test_install_without_hooks_reports_unavailable():
    recorder = TP.TritonCompileRecorder(
        guard=TP.LateCompileGuard(debug_source=dict))
    assert not recorder.install(SimpleNamespace())
    assert not recorder.installed


# --------------------------------------------------------------------------- #
#  Phases
# --------------------------------------------------------------------------- #
def _entries(trace, count, fail=()):
    def run(index):
        trace.append(index)
        if index in fail:
            raise RuntimeError(f"entry {index} broke")

    return [TP.PrewarmEntry(f"e{index}", lambda index=index: run(index))
            for index in range(count)]


def test_phases_run_own_share_then_barrier_then_everything():
    trace = []
    entries = _entries(trace, 7, fail={4})
    reports = TP.run_prewarm(
        entries, part_index=1, part_count=3,
        barrier=lambda: trace.append("barrier"), sync=lambda: None,
    )
    assert trace == [1, 4, "barrier", 0, 1, 2, 3, 4, 5, 6]
    assert [(r.name, r.entries, r.failed) for r in reports] == [
        ("phase1", 2, 1), ("phase2", 7, 1),
    ]


def test_single_participant_skips_phase_two():
    trace = []
    reports = TP.run_prewarm(_entries(trace, 3), sync=lambda: None,
                             barrier=lambda: trace.append("barrier"))
    assert trace == [0, 1, 2]
    assert [r.name for r in reports] == ["phase1"]


# --------------------------------------------------------------------------- #
#  Import hygiene
# --------------------------------------------------------------------------- #
def test_imports_without_triton_or_fla(monkeypatch):
    for name in ("triton", "triton.language", "fla", "fla.ops", "fla.modules"):
        monkeypatch.setitem(sys.modules, name, None)
    module = _load_prewarm("_k3_triton_prewarm_no_gpu_deps")
    assert callable(module.prewarm_k3_triton_kernels)
    recorder = module.TritonCompileRecorder(
        guard=module.LateCompileGuard(debug_source=dict))
    assert not recorder.install()
