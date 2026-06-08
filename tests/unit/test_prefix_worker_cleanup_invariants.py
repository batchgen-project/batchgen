from pathlib import Path


_WORKER_SOURCE = (
    Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"
)


def _source() -> str:
    return _WORKER_SOURCE.read_text()


def _method_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"\tdef {name}(")
    end = source.index(f"\n\tdef {next_name}(", start)
    return source[start:end]


def test_prefill_prepack_scope_cleans_global_state_in_finally():
    source = _source()
    scope = _method_body(
        source,
        "_prefill_prepack_runtime_scope",
        "prefill_prepacked",
    )

    assert "\n\t\tfinally:\n" in scope
    assert "self._reset_prefill_prepack_runtime_state()" in scope
    assert "prefix_materialization.close(empty_cuda_cache=False)" in scope
    assert "self._destroy_gpu_paged_kv_cache(empty_cuda_cache=True)" in scope


def test_prefill_prepacked_uses_cleanup_scope_around_inference_loop():
    source = _source()
    body = _method_body(
        source,
        "prefill_prepacked",
        "_compute_boundary_decisions",
    )
    scope_call = (
        "self._prefill_prepack_runtime_scope(batch_prefix_materialization)"
    )
    inference_call = "torch.inference_mode()"

    assert scope_call in body
    assert inference_call in body
    assert body.index(scope_call) < body.index(
        "Prepacked Prefill",
    )
