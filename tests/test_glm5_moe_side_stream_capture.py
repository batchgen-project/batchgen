"""Capture legality and parity of the shared-expert side-stream pattern.

The GLM-5 MoE layers overlap the shared expert on a side stream with the
routed pipeline on the main stream (fork via wait_stream, join before the
final add). These tests pin the exact pattern used by
`Glm5MoE._forward_decode_3d` and the MoE graph segment:

- eager: side-stream result consumed on the main stream after a join, with
  `record_stream` protecting the allocator block;
- capture: the fork/join is captured into one CUDA graph and replays with
  parity against a single-stream reference.
"""
import pytest
import torch

from batchgen.models.glm.glm5.model import Glm5MoE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _overlapped(x, shared_w, routed_w, side):
    main = torch.cuda.current_stream(x.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        shared_out = x @ shared_w
    routed_out = x @ routed_w
    main.wait_stream(side)
    if not torch.cuda.is_current_stream_capturing():
        shared_out.record_stream(main)
    return routed_out + shared_out


def test_shared_expert_stream_is_a_process_singleton():
    dev = torch.device("cuda")
    s1 = Glm5MoE._get_shared_expert_stream(dev)
    s2 = Glm5MoE._get_shared_expert_stream(dev)
    assert s1 is s2
    assert isinstance(s1, torch.cuda.Stream)


def test_side_stream_overlap_matches_single_stream_eager():
    torch.manual_seed(20261007)
    dev = torch.device("cuda")
    x = torch.randn(96, 256, device=dev, dtype=torch.bfloat16)
    shared_w = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
    routed_w = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
    side = Glm5MoE._get_shared_expert_stream(dev)

    ref = x @ routed_w + x @ shared_w
    out = _overlapped(x, shared_w, routed_w, side)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_side_stream_overlap_captures_and_replays_with_parity():
    torch.manual_seed(20261007)
    dev = torch.device("cuda")
    static_x = torch.zeros(96, 256, device=dev, dtype=torch.bfloat16)
    shared_w = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
    routed_w = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
    side = Glm5MoE._get_shared_expert_stream(dev)

    # Warmup on a non-default stream (required before capture).
    warm = torch.cuda.Stream(device=dev)
    warm.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(warm):
        for _ in range(3):
            _overlapped(static_x, shared_w, routed_w, side)
    torch.cuda.current_stream(dev).wait_stream(warm)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = _overlapped(static_x, shared_w, routed_w, side)

    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        new_x = torch.randn_like(static_x)
        static_x.copy_(new_x)
        graph.replay()
        torch.cuda.synchronize()
        ref = new_x @ routed_w + new_x @ shared_w
        assert torch.equal(static_out, ref), f"replay mismatch at seed {seed}"
