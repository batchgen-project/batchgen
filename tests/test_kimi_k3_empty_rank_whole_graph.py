"""An empty DP rank joins the K3 whole-model graph replay (zero valid rows).

r27 (D512, 2x8 H200): once one node had drained, the other node's last ~330
decode steps ran eager at 150-230 ms instead of ~80 ms graph replays, because
the whole-model path required every EP rank to hold rows.  The empty rank now
arms the peers' synced bucket with ``bsz=0`` (== the capture arming: scratch
KDA slots, page 0, no KV write) and replays it, so the captured EP collectives
stay matched on every rank.  These contracts pin the pieces.
"""
import ast
from pathlib import Path

_MODEL_ROOT = Path(__file__).parents[1] / "batchgen/models/moonshotai/kimi_linear"
_SRC = (_MODEL_ROOT / "cuda_graph_segments.py").read_text()


def _method(name: str) -> str:
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_SRC, node)
    raise AssertionError(name)


def test_begin_step_routes_an_empty_rank_into_the_whole_graph_in_graph_mode():
    src = _method("_begin_step")
    head = src[: src.index("bucket = self.bucketing.get_padded_size(bsz)")]
    assert "if bsz == 0:" in head
    assert 'if mode == "graph":' in head
    assert "self._begin_empty_whole_step()" in head


def test_whole_model_step_does_not_wait_for_every_rank_to_be_non_empty():
    src = _method("_begin_step")
    whole = src[src.index('if mode == "graph":'):]
    whole = whole[: whole.index("self._capture_bucket(bucket)")]
    assert "_moe_all_ranks_nonempty" not in whole
    assert "self._moe_group_bucket_for_step() is not None" in whole


def test_empty_whole_step_arms_zero_rows_on_the_synced_bucket():
    # The collective half (build, capture, zero-row refresh) is shared with
    # decode prewarm so an empty rank issues the same EP collectives at both.
    collective = _method("_prepare_empty_whole_bucket")
    assert "self._whole_model_bucket_for_step()" in collective
    assert "self._capture_whole_bucket(whole_bucket)" in collective
    assert "self._refresh_statics(whole_bucket, 0, [], None)" in collective
    # the KV-storage signature check must not be skipped on the empty rank
    assert "self._signature(kv_manager)" in collective
    step = _method("_begin_empty_whole_step")
    assert "self._prepare_empty_whole_bucket()" in step
    assert "self._bsz = 0" in step
    assert "self._whole_step_active = True" in step


def test_prewarm_runs_the_empty_rank_collective_half():
    src = _method("prewarm_decode_graphs")
    empty = src[src.index("if bsz == 0:"):]
    empty = empty[: empty.index("kv_manager = AttnWrapperBase.gpu_paged_kv_manager")]
    assert "self._prepare_empty_whole_bucket()" in empty


def test_refresh_statics_handles_zero_rows_without_wrapper_batch_tensors():
    src = _method("_refresh_statics")
    zero = src[src.index("if bsz == 0:"):src.index("cache_seqlens = AttnWrapperBase")]
    assert "statics.refresh(0, empty, empty, statics.page_table[:0])" in zero
    assert "return" in zero
