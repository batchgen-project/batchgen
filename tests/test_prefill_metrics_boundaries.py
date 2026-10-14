"""Regression: prefill [METRICS] must report host, GPU-complete and KV-ready time.

`prefill_s` used to be a host `perf_counter` delta taken right after the
prepacked forward loop. Kernel launches are asynchronous and the last layers'
KV offloads are still in flight there, so the record could end before the GPU
or host KV had caught up. prefill_prepacked now takes three boundaries in
order -- host return, current-stream synchronize, final KV offload retirement --
and the metrics line carries all three plus the legacy `prefill_s` alias.

batchgen_worker imports the whole engine, so this is a static check on the
method body. The boundaries must be direct statements of the method body (not
nested under any branch) so every rank that emits the metrics line took them.
"""
import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"

CONTRACT = "forward-host+gpu-complete+kv-ready-v1"
TIMING_FIELDS = {
    "prefill_forward_host_s": "_prefill_forward_host_s",
    "prefill_forward_gpu_complete_s": "_prefill_forward_gpu_complete_s",
    "prefill_kv_ready_s": "_prefill_kv_ready_s",
}


def _method(name):
    tree = ast.parse(WORKER.read_text())
    worker = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "BatchGenWorker")
    return next(n for n in worker.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _assign_index(body, target):
    hits = [i for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == target
                    for t in stmt.targets)]
    assert len(hits) == 1, (
        f"{target} must be assigned exactly once at method level")
    return hits[0]


def _is_elapsed_since_t0(value):
    # time.perf_counter() - _prefill_forward_t0
    return (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub)
            and isinstance(value.left, ast.Call)
            and isinstance(value.left.func, ast.Attribute)
            and value.left.func.attr == "perf_counter"
            and isinstance(value.right, ast.Name)
            and value.right.id == "_prefill_forward_t0")


def _is_self_attr(node, attr):
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Name) and node.value.id == "self")


def _metrics_call(body):
    hits = []
    for i, stmt in enumerate(body):
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        call = stmt.value
        if (isinstance(call.func, ast.Attribute) and call.func.attr == "info"
                and call.args and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "[METRICS] %s"):
            hits.append((i, call))
    assert len(hits) == 1, "[METRICS] must be logged exactly once at method level"
    return hits[0]


def _metrics_dict(call):
    dumps = call.args[1]
    assert isinstance(dumps, ast.Call) and dumps.func.attr == "dumps"
    record = dumps.args[0]
    assert isinstance(record, ast.Dict)
    return {k.value: v for k, v in zip(record.keys, record.values)}


def test_boundaries_are_ordered_and_adjacent_after_forward_loop():
    body = _method("prefill_prepacked").body
    host = _assign_index(body, "_prefill_forward_host_s")
    gpu = _assign_index(body, "_prefill_forward_gpu_complete_s")
    retire = _assign_index(body, "_num_final_kv_tasks")
    ready = _assign_index(body, "_prefill_kv_ready_s")

    # The host boundary directly closes the forward loop that samples tokens.
    loop = body[host - 1]
    assert isinstance(loop, (ast.For, ast.With)), ast.dump(loop)
    assert any(isinstance(c, ast.Call) and _is_self_attr(c.func, "_select_tokens")
               for c in ast.walk(loop))

    sync = host + 1
    assert (gpu, retire, ready) == (sync + 1, sync + 2, sync + 3)
    for idx in (host, gpu, ready):
        assert _is_elapsed_since_t0(body[idx].value), ast.dump(body[idx])

    # torch.cuda.current_stream(self.torch_device).synchronize()
    stmt = body[sync]
    assert isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
    func = stmt.value.func
    assert isinstance(func, ast.Attribute) and func.attr == "synchronize"
    stream = func.value
    assert isinstance(stream, ast.Call) and stream.func.attr == "current_stream"
    assert stream.func.value.attr == "cuda"
    assert len(stream.args) == 1 and _is_self_attr(stream.args[0], "torch_device")

    # AttnWrapperBase.retire_pending_prefill_offloads(device=..., reason=...)
    call = body[retire].value
    assert isinstance(call, ast.Call)
    assert call.func.attr == "retire_pending_prefill_offloads"
    assert isinstance(call.func.value, ast.Name)
    assert call.func.value.id == "AttnWrapperBase"
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert set(kwargs) == {"device", "reason"}
    assert _is_self_attr(kwargs["device"], "torch_device")
    assert kwargs["reason"].value == "prefill metrics KV-ready boundary"

    metrics_idx, _ = _metrics_call(body)
    assert ready < metrics_idx


def test_single_synchronize_between_host_boundary_and_metrics():
    body = _method("prefill_prepacked").body
    host = _assign_index(body, "_prefill_forward_host_s")
    metrics_idx, _ = _metrics_call(body)
    syncs = [c for stmt in body[host:metrics_idx] for c in ast.walk(stmt)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "synchronize"]
    assert len(syncs) == 1


def test_metrics_record_carries_contract_timings_and_legacy_alias():
    _, call = _metrics_call(_method("prefill_prepacked").body)
    record = _metrics_dict(call)

    assert record["prefill_metrics_contract"].value == CONTRACT
    for field, var in TIMING_FIELDS.items():
        assert isinstance(record[field], ast.Name) and record[field].id == var
    # The legacy alias stays the host-only value.
    assert isinstance(record["prefill_s"], ast.Name)
    assert record["prefill_s"].id == "_prefill_forward_host_s"
    assert isinstance(record["final_kv_offload_tasks_retired"], ast.Name)
    assert record["final_kv_offload_tasks_retired"].id == "_num_final_kv_tasks"


def test_generate_level_retirement_is_kept():
    fn = _method("generate")
    reasons = [kw.value.value for c in ast.walk(fn)
               if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and c.func.attr == "retire_pending_prefill_offloads"
               for kw in c.keywords if kw.arg == "reason"]
    assert "end of prefill" in reasons
