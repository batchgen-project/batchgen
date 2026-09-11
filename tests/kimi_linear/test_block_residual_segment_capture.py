"""K3 Block Attention Residual state under per-layer CUDA-graph replay.

This is deliberately a segment-level gate: attention is a deterministic BF16
linear map so the test isolates the state that the K3 adapter adds around it.
Five layers cross boundaries 0, 2 and 4. The graph path uses one fixed-address
full buffer and captured in-place boundary writes; the independent eager oracle
uses progressively growing ``torch.cat`` tensors, matching model.py.

Run on a remote GPU machine only:

    python batchgen_kernels/tests/kimi_linear/test_block_residual_segment_capture.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.cuda_graph.graph_manager import BatchSizeBucketing, CUDAGraphManager
from batchgen.models.moonshotai.kimi_linear import cuda_graph_segments as graph_mod
from batchgen.models.moonshotai.kimi_linear.block_residual import apply_attn_res


DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HIDDEN = 64
KV_DIM = 8
LAYERS = 5
BLOCK_SIZE = 2
BOUNDARIES = 3
BUCKET = 4
BSZ = 3
TOL = 1e-2


class RMSNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        x32 = x.float()
        out = x32 * torch.rsqrt(
            x32.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        return (out * self.weight.float()).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, hidden))
        self.kv_lora_rank = KV_DIM
        self.qk_rope_head_dim = 0


class Layer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = HIDDEN
        self.is_linear_attn = False
        self.use_attn_residuals = True
        self.attn_res_block_size = BLOCK_SIZE
        self.self_attn = Attention(HIDDEN)
        self.input_layernorm = RMSNorm(HIDDEN)
        self.post_attention_layernorm = RMSNorm(HIDDEN)
        self.self_attention_res_norm = RMSNorm(HIDDEN)
        self.mlp_res_norm = RMSNorm(HIDDEN)
        self.self_attention_res_proj = nn.Linear(HIDDEN, 1, bias=False)
        self.mlp_res_proj = nn.Linear(HIDDEN, 1, bias=False)
        if layer_idx == 0:
            self.mlp = nn.Sequential(
                nn.Linear(HIDDEN, 2 * HIDDEN, bias=False),
                nn.SiLU(),
                nn.Linear(2 * HIDDEN, HIDDEN, bias=False),
            )
        else:
            self.block_sparse_moe = nn.Identity()
            self.ffn = nn.Sequential(
                nn.Linear(HIDDEN, 2 * HIDDEN, bias=False),
                nn.SiLU(),
                nn.Linear(2 * HIDDEN, HIDDEN, bias=False),
            )

    def _run_ffn(self, hidden_states):
        return self.ffn(hidden_states)


def fake_mla(attn, hidden_states, **_kwargs):
    out = F.linear(hidden_states.squeeze(1), attn.weight).unsqueeze(1)
    k_tensor = hidden_states[..., :KV_DIM].reshape(
        hidden_states.shape[0], 1, 1, KV_DIM
    )
    return out, k_tensor


def init_module(module, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    module.to(device=DEVICE, dtype=DTYPE)
    for name, param in module.named_parameters():
        if "norm" in name or name == "weight" and isinstance(module, RMSNorm):
            value = torch.randn(
                param.shape, generator=gen, device=DEVICE, dtype=torch.float32
            ) * 0.02 + 1.0
        else:
            value = torch.randn(
                param.shape, generator=gen, device=DEVICE, dtype=torch.float32
            ) * 0.04
        param.data.copy_(value.to(param.dtype))
        param.requires_grad_(False)
    return module


def reference_layer(layer, hidden_states, block_residual):
    prefix_sum = hidden_states
    flat_prefix = prefix_sum.reshape(-1, HIDDEN)
    if block_residual.shape[1]:
        hidden_states = apply_attn_res(
            flat_prefix,
            block_residual,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
        ).view_as(hidden_states)

    if layer.layer_idx % BLOCK_SIZE == 0:
        block_residual = torch.cat(
            (block_residual, flat_prefix.unsqueeze(1)), dim=1
        )
        prefix_sum = None

    attn_out, _ = fake_mla(layer.self_attn, layer.input_layernorm(hidden_states))
    prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out
    ffn_input = apply_attn_res(
        prefix_sum.reshape(-1, HIDDEN),
        block_residual,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
    ).view_as(hidden_states)
    normed = layer.post_attention_layernorm(ffn_input)
    ffn_out = layer.mlp(normed) if layer.layer_idx == 0 else layer._run_ffn(normed)
    return prefix_sum + ffn_out, block_residual


def run_reference(layers, hidden_states, output_proj, output_norm):
    block_residual = torch.zeros(
        BSZ, 0, HIDDEN, dtype=DTYPE, device=DEVICE
    )
    trace = []
    for layer in layers:
        hidden_states, block_residual = reference_layer(
            layer, hidden_states, block_residual
        )
        trace.append((hidden_states.clone(), block_residual.clone()))
    output = apply_attn_res(
        hidden_states.reshape(-1, HIDDEN),
        block_residual,
        output_proj,
        output_norm,
    ).view_as(hidden_states)
    return output, trace


def run_graph(manager, segments, statics, layers, hidden_states,
              output_proj, output_norm):
    # Poison every column. Correct execution overwrites each boundary before
    # that column can be read; an early full-buffer read propagates NaNs.
    statics.block_residual.fill_(float("nan"))
    trace = []
    for idx, layer in enumerate(layers):
        outputs = manager.replay(
            "layer_{}".format(idx), BSZ, hidden_states=hidden_states
        )
        hidden_states = (
            outputs["hidden"] if segments[idx].fold_ffn
            else outputs["residual"] + layer._run_ffn(outputs["normed"])
        )
        columns = segments[idx].num_blocks_after
        trace.append((
            hidden_states.clone(),
            statics.block_residual[:BSZ, :columns].clone(),
        ))
    output = apply_attn_res(
        hidden_states.reshape(-1, HIDDEN),
        statics.block_residual[:BSZ],
        output_proj,
        output_norm,
    ).view_as(hidden_states)
    return output, trace


def check_close(name, got, ref):
    finite = bool(torch.isfinite(got.float()).all())
    delta = (got.float() - ref.float()).abs().max().item()
    bitwise = torch.equal(got, ref)
    ok = finite and (bitwise or delta <= TOL)
    print("[{}] {} bitwise={} max|delta|={:.3e}".format(
        "PASS" if ok else "FAIL", name, bitwise, delta
    ))
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    torch.set_grad_enabled(False)
    torch.cuda.set_device(DEVICE)
    torch.manual_seed(1)

    layers = [init_module(Layer(i), 100 + i) for i in range(LAYERS)]
    output_proj = init_module(nn.Linear(HIDDEN, 1, bias=False), 201)
    output_norm = init_module(RMSNorm(HIDDEN), 202)

    slots = torch.zeros(BUCKET, dtype=torch.int32, device=DEVICE)
    statics = graph_mod._BucketStatics(
        BUCKET,
        1,
        DEVICE,
        slots,
        block_residual_columns=BOUNDARIES,
        hidden_size=HIDDEN,
        dtype=DTYPE,
    )
    statics.arm_for_capture()
    shared = {BUCKET: statics}

    original_mla = graph_mod._mla_decode_graph_safe
    graph_mod._mla_decode_graph_safe = fake_mla
    manager = CUDAGraphManager(BatchSizeBucketing([BUCKET]), device=DEVICE)
    manager.WARMUP_ITERATIONS = 2
    manager.WARMUP_ITERATIONS_SUBSEQUENT = 1
    segments = {}
    try:
        for idx, layer in enumerate(layers):
            segment = graph_mod.KimiLinearSpanSegment(
                layer, idx, shared, page_size_tokens=1, dtype=DTYPE
            )
            segments[idx] = segment
            manager.register_segment("layer_{}".format(idx), segment)
        manager.warmup_and_capture_all()

        expected_counts = [(0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]
        got_counts = [
            (segments[i].num_blocks_before, segments[i].num_blocks_after)
            for i in range(LAYERS)
        ]
        ok = got_counts == expected_counts
        print("[{}] progressive columns {}".format(
            "PASS" if ok else "FAIL", got_counts
        ))

        ptr = statics.block_residual.data_ptr()
        stride = statics.block_residual.stride()
        gen = torch.Generator(device=DEVICE).manual_seed(303)
        for pass_idx in range(2):
            hidden = torch.randn(
                (BSZ, 1, HIDDEN), generator=gen, device=DEVICE,
                dtype=torch.float32,
            ).to(DTYPE)
            ref_out, ref_trace = run_reference(
                layers, hidden, output_proj, output_norm
            )
            graph_out, graph_trace = run_graph(
                manager, segments, statics, layers, hidden,
                output_proj, output_norm,
            )
            for layer_idx, ((got_h, got_b), (ref_h, ref_b)) in enumerate(
                zip(graph_trace, ref_trace)
            ):
                ok &= check_close(
                    "pass {} layer {} prefix".format(pass_idx, layer_idx),
                    got_h,
                    ref_h,
                )
                ok &= check_close(
                    "pass {} layer {} block state".format(pass_idx, layer_idx),
                    got_b,
                    ref_b,
                )
            ok &= check_close(
                "pass {} final output mix".format(pass_idx), graph_out, ref_out
            )

        stable = (
            statics.block_residual.data_ptr() == ptr
            and statics.block_residual.stride() == stride
            and statics.block_residual.shape
            == (BUCKET, BOUNDARIES, HIDDEN)
        )
        print("[{}] block-residual storage stayed fixed shape={} stride={}".format(
            "PASS" if stable else "FAIL",
            tuple(statics.block_residual.shape),
            tuple(statics.block_residual.stride()),
        ))
        ok &= stable
        torch.cuda.synchronize()
    finally:
        graph_mod._mla_decode_graph_safe = original_mla

    print("\n{}".format("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
