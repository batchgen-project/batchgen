# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The model-neutral Triton prewarm core, on CPU.

Under test: ``batchgen_kernels/triton_prewarm.py``.

  * the integer classes and the rank partition;
  * the late-compile guard's states for configurable strict keys, the
    recorder's parsing of a Triton 3.5 hook payload, its compile/load split,
    its log tag and its hook chaining;
  * the phase runner's order: own share, barrier, everything;
  * the module imports with neither Triton, fla nor torch installed;
  * the DSA indexer-score and head-gate kernels take the batch size ``B`` as
    a runtime int, so a new batch size does not compile a new kernel.

Loaded by file path (no ``import batchgen_kernels``, whose ``__init__``
imports torch).

Run: python -m pytest tests/unit/test_triton_prewarm_core.py -q
"""

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_KERNELS = _ROOT / "batchgen_kernels"
_MODULE_PATH = _KERNELS / "triton_prewarm.py"


def _load_prewarm(name="_triton_prewarm_core_under_test"):
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TP = _load_prewarm()

_LOG_TAGS = (TP.DEFAULT_LOG_TAG, "[CUSTOM_PREWARM]")
_STRICT_KEYS = (TP.DEFAULT_STRICT_KEYS, ("custom_triton_strict",),
                ("first_strict", "second_strict"))


# --------------------------------------------------------------------------- #
#  Integer classes and partitioning
# --------------------------------------------------------------------------- #
def test_int_class_follows_triton_integer_specialisation():
    assert TP.int_class(1) == "1"
    assert [TP.int_class(v) for v in (16, 32, 4096)] == ["d16"] * 3
    assert [TP.int_class(v) for v in (2, 15, 17, 1000)] == ["any"] * 4


def test_class_representatives_pick_one_value_per_int_class():
    buckets = [1, 2, 4, 8, 16, 24, 32, 64, 96, 128]
    assert TP.class_representatives(buckets) == [1, 2, 16]
    assert TP.class_representatives([0, 32, 64]) == [32]


def test_partition_covers_each_entry_once_and_is_deterministic():
    items = list(range(23))
    for count in (1, 2, 3, 8):
        shares = [TP.partition(items, index, count) for index in range(count)]
        assert sorted(x for share in shares for x in share) == items
        assert shares == [TP.partition(items, index, count)
                          for index in range(count)]
    with pytest.raises(ValueError):
        TP.partition(items, 8, 8)
    with pytest.raises(ValueError):
        TP.partition(items, 0, 0)


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


def _recorder(debug=None, clock=None, log_tag=TP.DEFAULT_LOG_TAG,
              strict_keys=TP.DEFAULT_STRICT_KEYS):
    guard = TP.LateCompileGuard(debug_source=lambda: debug or {},
                                strict_keys=strict_keys)
    recorder = TP.TritonCompileRecorder(guard=guard,
                                        clock=clock or _Clock(*[0.0] * 64),
                                        log_tag=log_tag)
    runtime = SimpleNamespace(jit_cache_hook=None, jit_post_compile_hook=None)
    assert recorder.install(runtime)
    return recorder, runtime


def _compile(runtime, payload):
    assert runtime.jit_cache_hook(**payload) is None
    runtime.jit_post_compile_hook(**payload)


@pytest.mark.parametrize("keys", _STRICT_KEYS)
def test_guard_states(keys):
    debug = {}
    guard = TP.LateCompileGuard(debug_source=lambda: debug, strict_keys=keys)
    assert guard.state() == TP.DISARMED
    guard.arm()
    assert guard.state() == TP.LOG
    for key in keys:
        debug.clear()
        for value in (True, 1, "1", "true", "on"):
            debug[key] = value
            assert guard.state() == TP.STRICT
            assert guard.strict_reason() == f"batchgen_debug.{key}"
        for value in (False, 0, "false", None):
            debug[key] = value
            assert guard.state() == TP.LOG
    debug.clear()
    # A key the guard was not configured with does not make it strict.
    debug["unrelated_strict"] = True
    assert guard.state() == TP.LOG
    guard.arm(strict=True)
    assert guard.state() == TP.STRICT
    assert guard.strict_reason() == "a guard armed strict"
    guard.disarm()
    debug.update({key: True for key in keys})
    assert guard.state() == TP.DISARMED


def test_guard_accepts_a_single_key_string():
    guard = TP.LateCompileGuard(debug_source=lambda: {"only_key": "yes"},
                                strict_keys="only_key")
    assert guard.strict_keys == ("only_key",)
    guard.arm()
    assert guard.state() == TP.STRICT


def test_default_guard_has_no_batch_debug_source():
    guard = TP.LateCompileGuard()
    assert guard.strict_keys == TP.DEFAULT_STRICT_KEYS
    guard.arm()
    assert guard.state() == TP.LOG


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
    mark = recorder.mark()
    assert mark == 2
    assert recorder.counts(mark) == {"compiled": 0, "loaded": 0}


def test_disarmed_compiles_are_not_late(caplog):
    recorder, runtime = _recorder()
    with caplog.at_level(logging.DEBUG):
        _compile(runtime, _payload())
    assert not recorder.events[0].late
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("tag", _LOG_TAGS)
def test_armed_compile_logs_the_full_key(caplog, tag):
    recorder, runtime = _recorder(log_tag=tag)
    recorder.guard.arm()
    with caplog.at_level(logging.WARNING):
        _compile(runtime, _payload())
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    message = record.getMessage()
    assert message.startswith(f"{tag} late compile kernel=")
    for part in ("fake_mod._fake_kernel", "BLOCK=1024", "x_ptr:*bf16",
                 "x_ptr:tt.divisibility=16", "num_warps=4"):
        assert part in message
    assert recorder.events[0].late


@pytest.mark.parametrize("keys", _STRICT_KEYS)
@pytest.mark.parametrize("tag", _LOG_TAGS)
@pytest.mark.parametrize("strict_by", ["batch_flag", "arm"])
def test_strict_raises_before_compiling(caplog, strict_by, tag, keys):
    key = keys[-1]
    debug = {key: True} if strict_by == "batch_flag" else {}
    recorder, runtime = _recorder(debug=debug, log_tag=tag, strict_keys=keys)
    recorder.guard.arm(strict=strict_by == "arm")
    reason = (f"batchgen_debug.{key}" if strict_by == "batch_flag"
              else "a guard armed strict")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(TP.LateTritonCompileError) as info:
            runtime.jit_cache_hook(**_payload())
    assert "fake_mod._fake_kernel" in str(info.value)
    assert reason in str(info.value)
    assert f"{tag} late compile (strict)" in caplog.text
    assert recorder.events == []
    assert recorder._pending == {}


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


@pytest.mark.parametrize("tag", _LOG_TAGS)
def test_install_without_hooks_reports_unavailable(caplog, tag):
    recorder = TP.TritonCompileRecorder(
        guard=TP.LateCompileGuard(debug_source=dict), log_tag=tag)
    with caplog.at_level(logging.WARNING):
        assert not recorder.install(SimpleNamespace())
    assert not recorder.installed
    assert f"{tag} this Triton has no jit_cache_hook" in caplog.text


def test_get_recorder_keeps_the_first_callers_tag(monkeypatch):
    monkeypatch.setattr(TP, "_RECORDER", None)
    first = TP.get_recorder(log_tag="[FIRST]")
    second = TP.get_recorder(log_tag="[SECOND]")
    assert first is second
    assert first.log_tag == "[FIRST]"
    monkeypatch.setattr(TP, "_RECORDER", None)
    assert TP.get_recorder().log_tag == TP.DEFAULT_LOG_TAG


def test_get_recorder_wires_the_first_callers_debug_source(monkeypatch):
    monkeypatch.setattr(TP, "_RECORDER", None)
    debug = {}
    recorder = TP.get_recorder(
        log_tag="[MODEL]", debug_source=lambda: debug,
        strict_keys=("model_triton_strict",),
    )
    recorder.guard.arm()
    assert recorder.guard.state() == TP.LOG
    debug["model_triton_strict"] = True
    assert recorder.guard.state() == TP.STRICT
    assert recorder.guard.strict_reason() == "batchgen_debug.model_triton_strict"
    # A later caller's wiring is ignored, like its tag.
    assert TP.get_recorder(strict_keys=("other",)) is recorder


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
    # No recorder: the counts are reported as unavailable.
    assert {(r.compiled, r.loaded) for r in reports} == {(-1, -1)}


def test_single_participant_skips_phase_two():
    trace = []
    reports = TP.run_prewarm(_entries(trace, 3), sync=lambda: None,
                             barrier=lambda: trace.append("barrier"))
    assert trace == [0, 1, 2]
    assert [r.name for r in reports] == ["phase1"]


@pytest.mark.parametrize("tag", _LOG_TAGS)
def test_phase_reports_count_per_phase_under_the_recorder_tag(caplog, tag):
    recorder, runtime = _recorder(log_tag=tag)
    entries = [TP.PrewarmEntry(
        f"e{index}", lambda index=index: _compile(runtime, _payload(f"k{index}"))
    ) for index in range(3)]
    with caplog.at_level(logging.INFO):
        reports = TP.run_prewarm(entries, recorder=recorder, part_index=0,
                                 part_count=2, sync=lambda: None)
    # The fake clock reads 0.0 throughout, so every event counts as a load.
    assert [(r.name, r.compiled, r.loaded) for r in reports] == [
        ("phase1", 0, 2), ("phase2", 0, 3),
    ]
    phase_lines = [r.getMessage() for r in caplog.records
                   if " phase" in r.getMessage()]
    assert len(phase_lines) == 2
    assert all(line.startswith(f"{tag} rank=0 phase") for line in phase_lines)


def test_explicit_log_tag_overrides_the_recorder_tag(caplog):
    with caplog.at_level(logging.INFO):
        TP.run_prewarm([], sync=lambda: None, log_tag="[EXPLICIT]")
    assert "[EXPLICIT] rank=0 phase1" in caplog.text


# --------------------------------------------------------------------------- #
#  Import hygiene and the kernels layer boundary
# --------------------------------------------------------------------------- #
def test_imports_without_triton_fla_or_torch(monkeypatch):
    for name in ("triton", "triton.language", "fla", "fla.ops", "fla.modules",
                 "torch", "torch.cuda"):
        monkeypatch.setitem(sys.modules, name, None)
    module = _load_prewarm("_triton_prewarm_core_no_gpu_deps")
    assert callable(module.run_prewarm)
    recorder = module.TritonCompileRecorder(
        guard=module.LateCompileGuard(debug_source=dict))
    assert not recorder.install()


def test_module_does_not_import_the_batchgen_package():
    tree = ast.parse(_MODULE_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [name for name in imported
                if name == "batchgen" or name.startswith("batchgen.")]


# --------------------------------------------------------------------------- #
#  Batch size as a runtime int in the DSA kernels
# --------------------------------------------------------------------------- #
_RUNTIME_B_KERNELS = (
    ("attention/dsa/fused_indexer_score.py", "_fused_score_kernel"),
    ("attention/dsa/fused_indexer_score.py", "_fused_paged_score_kernel"),
    ("attention/dsa/fused_indexer_score.py",
     "_fused_paged_score_with_slots_kernel"),
    ("attention/dsa/head_gates.py", "_head_gates_out_kernel"),
)


def _is_constexpr(annotation) -> bool:
    return annotation is not None and ast.unparse(annotation).endswith(
        "constexpr")


@pytest.mark.parametrize("relpath,kernel", _RUNTIME_B_KERNELS)
def test_dsa_kernels_take_batch_size_as_a_runtime_int(relpath, kernel):
    tree = ast.parse((_KERNELS / relpath).read_text())
    (func,) = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == kernel]
    args = {arg.arg: arg for arg in func.args.args}
    assert "B" in args, f"{kernel} lost its B parameter"
    assert not _is_constexpr(args["B"].annotation), (
        f"{kernel} declares B: tl.constexpr; every batch size compiles anew")
    # B only sizes the launch grid; a body use would need it as a constexpr.
    body = func.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)):
        body = body[1:]
    names = {node.id for stmt in body for node in ast.walk(stmt)
             if isinstance(node, ast.Name)}
    assert "B" not in names
